"""Phone link: the WebSocket server the iPhone pointer connects to (docs/phone-protocol.md).

Two ways to use it
------------------
``PhoneLink`` is an asyncio object implementing ``interfaces.PhoneLinkAPI``. ``await start()`` binds the
server (``port=0`` picks a free port, readable from ``.port``). Use it directly when the caller owns an
asyncio loop (the tests do).

``PhoneLinkThread`` is for the main app, which runs a Qt event loop on the main thread. It owns a
``PhoneLink`` on a private asyncio loop in a daemon thread. ``start()`` blocks until the socket is bound
(or raises the bind error), ``stop()`` shuts the server down and joins the thread. ``latest()``,
``history()``, ``pop_buttons()``, ``connected()``, ``stalled()`` and ``stats()`` are safe to call from any
thread (all shared state sits behind one lock; each returns immutable snapshots or copies).
``send_nowait(msg)`` validates ``msg`` on the caller's thread (raising ``ValueError`` if it is invalid) and
schedules the send on the link's loop; it never blocks and is a no-op without a phone.

Clock mapping
-------------
The phone sends ``t_ms`` (its ``Date.now()``). The two clocks are unrelated, so per message we compute
``d = received_at - t_ms/1000`` and keep ``offset = min(d)`` over the last 2 s of arrivals; then
``PhoneState.t = t_ms/1000 + offset``. The sample with the least network delay defines the offset, so
``t <= received_at`` up to float rounding, and ``received_at - t`` is the *excess* delay over the best recent sample.
Absolute one-way latency cannot be known without a shared clock; ``LinkStats.excess_latency_ms`` reports
that excess for the newest sample. The filter restarts on every new connection.

Stalls
------
``stalled()`` is true while connected and ``clock() - last sample arrival > stall_ms``. ``LinkStats.stalls``
counts gaps between consecutive samples (within one connection) that exceeded ``stall_ms``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import threading
from collections import deque
from dataclasses import dataclass

import jsonschema
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from pointview import schemas
from pointview.interfaces import (
    EXTRA_TOGGLES,
    LAYER_KINDS,
    ButtonEvent,
    ButtonName,
    Clock,
    LayerName,
    Look,
    MarkerShape,
    PhoneState,
    Quat,
    now,
)
from pointview.phone_tracking import TrackingTarget

_OFFSET_WINDOW_S = 2.0
_RATE_WINDOW_S = 1.0
_MAX_PENDING_BUTTONS = 256
_MAX_MESSAGE_BYTES = 64 * 1024

_phone_validator = jsonschema.Draft202012Validator(schemas.load("phone_message"))
_laptop_validator = jsonschema.Draft202012Validator(schemas.load("laptop_message"))


def validate_outgoing(message: dict) -> None:
    """Raise ``ValueError`` unless ``message`` is a valid laptop -> phone message."""
    try:
        _laptop_validator.validate(message)
    except jsonschema.ValidationError as e:
        raise ValueError(f"invalid laptop message: {e.message}") from e


def _reject_constant(name: str):
    raise ValueError(f"non-finite number {name}")


def _parse(raw: str | bytes) -> dict | None:
    """Parse and schema-validate one frame; None if it is not a valid phone message."""
    if not isinstance(raw, str):
        return None
    try:
        msg = json.loads(raw, parse_constant=_reject_constant)
    except ValueError:
        return None
    if not isinstance(msg, dict) or not _phone_validator.is_valid(msg):
        return None
    return msg


def _normalize(q: list[float]) -> Quat | None:
    n = math.sqrt(sum(c * c for c in q))
    if n < 1e-6:
        return None
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


@dataclass(frozen=True, slots=True)
class LinkStats:
    """Snapshot of link health (see the module docstring)."""

    connected: bool
    connections: int
    messages: int
    invalid: int
    stalls: int
    rate_hz: float
    longest_gap_ms: float
    #: Delay of the newest sample above the best sample of the last 2 s (not absolute latency).
    excess_latency_ms: float | None


class PhoneLink:
    """WebSocket server for the phone; implements ``interfaces.PhoneLinkAPI``."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        clock: Clock = now,
        history_s: float = 2.0,
        stall_ms: float = 50,
    ) -> None:
        self.host = host
        self._requested_port = port
        self._clock = clock
        self._history_s = history_s
        self._stall_s = stall_ms / 1000.0
        self._lock = threading.RLock()
        self._tracking_target = TrackingTarget.create()
        self._server: Server | None = None
        self._ws: ServerConnection | None = None
        self._latest: PhoneState | None = None
        self._history: deque[PhoneState] = deque()
        self._buttons: deque[ButtonEvent] = deque(maxlen=_MAX_PENDING_BUTTONS)
        self._layers: dict[LayerName, bool] = dict.fromkeys((*LAYER_KINDS, *EXTRA_TOGGLES), True)
        self._marker: MarkerShape = "square"
        self._look: Look = "normal"
        self._passthrough = False
        self._offsets: deque[tuple[float, float]] = deque()
        self._rx_times: deque[float] = deque()
        self._last_rx: float | None = None
        self._excess_s: float | None = None
        self._connections = 0
        self._messages = 0
        self._invalid = 0
        self._stalls = 0
        self._longest_gap_s = 0.0

    # ------------------------------------------------------------------ lifecycle

    @property
    def port(self) -> int:
        """The bound port (useful with ``port=0``); the requested port before ``start()``."""
        if self._server is not None and self._server.sockets:
            return self._server.sockets[0].getsockname()[1]
        return self._requested_port

    async def start(self) -> None:
        if self._server is not None:
            return
        self._server = await serve(
            self._handler,
            self.host,
            self._requested_port,
            max_size=_MAX_MESSAGE_BYTES,
            compression=None,
            ping_interval=5,
            ping_timeout=5,
        )

    async def stop(self) -> None:
        server, self._server = self._server, None
        with self._lock:
            ws, self._ws = self._ws, None
        if ws is not None:
            await ws.close()
        if server is not None:
            server.close()
            await server.wait_closed()

    # ------------------------------------------------------------------ connection handling

    async def _handler(self, ws: ServerConnection) -> None:
        with self._lock:
            old, self._ws = self._ws, ws
            self._connections += 1
            self._tracking_target = TrackingTarget.create()
            self._latest = None
            self._history.clear()
            self._buttons.clear()
            # New phone, possibly a different clock: restart timing state and keep only display preferences.
            self._offsets.clear()
            self._rx_times.clear()
            self._last_rx = self._clock()
            self._excess_s = None
        if old is not None:
            with contextlib.suppress(Exception):
                await old.close(code=1000, reason="replaced by a new connection")
        try:
            async for raw in ws:
                if self._ws is not ws:
                    break
                self._on_frame(raw)
        except ConnectionClosed:
            pass
        finally:
            with self._lock:
                if self._ws is ws:
                    self._ws = None

    def _on_frame(self, raw: str | bytes) -> None:
        received_at = self._clock()
        msg = _parse(raw)
        quat = _normalize(msg["quat"]) if msg is not None else None
        if msg is None or quat is None:
            with self._lock:
                self._invalid += 1
            return
        with self._lock:
            t_phone = msg["t_ms"] / 1000.0
            self._offsets.append((received_at, received_at - t_phone))
            while self._offsets and self._offsets[0][0] < received_at - _OFFSET_WINDOW_S:
                self._offsets.popleft()
            offset = min(d for _, d in self._offsets)
            t = t_phone + offset
            self._excess_s = received_at - t

            if self._last_rx is not None and self._rx_times:
                gap = received_at - self._last_rx
                self._longest_gap_s = max(self._longest_gap_s, gap)
                if gap > self._stall_s:
                    self._stalls += 1
            self._last_rx = received_at
            self._messages += 1
            self._rx_times.append(received_at)
            while self._rx_times and self._rx_times[0] < received_at - _RATE_WINDOW_S:
                self._rx_times.popleft()

            if "layers" in msg:
                self._layers.update(msg["layers"])
            self._look = msg.get("look", self._look)
            self._passthrough = msg.get("passthrough", self._passthrough)
            self._marker = msg.get("marker", self._marker)
            tracking = msg.get("tracking", {})
            tracking_current = tracking.get("session") == self._tracking_target.session
            buttons: tuple[ButtonName, ...] = tuple(msg["buttons"])
            state = PhoneState(
                t_ms=float(msg["t_ms"]),
                t=t,
                received_at=received_at,
                quat=quat,
                omega=(float(msg["omega"][0]), float(msg["omega"][1]), float(msg["omega"][2])),
                heading_deg=msg["heading_deg"],
                heading_acc=msg["heading_acc"],
                lat=msg["lat"],
                lon=msg["lon"],
                gps_acc=msg["gps_acc"],
                buttons=buttons,
                layers=dict(self._layers),
                look=self._look,
                passthrough=self._passthrough,
                marker=self._marker,
                tracking_session=tracking.get("session") if tracking_current else None,
                tracking_enabled=bool(tracking_current and tracking.get("enabled", False)),
                marker_size_m=tracking.get("marker_size_m") if tracking_current else None,
            )
            for name in buttons:
                self._buttons.append(ButtonEvent(name=name, t_ms=state.t_ms, t=t))
            self._latest = state
            self._history.append(state)
            while self._history and self._history[0].t < t - self._history_s:
                self._history.popleft()

    def tracking_target(self) -> TrackingTarget:
        with self._lock:
            return self._tracking_target

    # ------------------------------------------------------------------ PhoneLinkAPI

    def latest(self) -> PhoneState | None:
        with self._lock:
            return self._latest

    def history(self, seconds: float = 1.0) -> list[PhoneState]:
        """Samples from the last ``seconds`` (at most ``history_s``) of laptop time, oldest first."""
        cutoff = self._clock() - seconds
        with self._lock:
            return [s for s in self._history if s.t >= cutoff]

    def pop_buttons(self) -> list[ButtonEvent]:
        with self._lock:
            out = list(self._buttons)
            self._buttons.clear()
            return out

    def connected(self) -> bool:
        with self._lock:
            return self._ws is not None

    def stalled(self) -> bool:
        with self._lock:
            if self._ws is None or self._last_rx is None:
                return False
            return self._clock() - self._last_rx > self._stall_s

    async def send(self, message: dict) -> None:
        validate_outgoing(message)
        with self._lock:
            ws = self._ws
        if ws is None:
            return
        with contextlib.suppress(ConnectionClosed):
            await ws.send(json.dumps(message))

    # ------------------------------------------------------------------ stats

    def stats(self) -> LinkStats:
        with self._lock:
            t = self._clock()
            recent = sum(1 for x in self._rx_times if x >= t - _RATE_WINDOW_S)
            return LinkStats(
                connected=self._ws is not None,
                connections=self._connections,
                messages=self._messages,
                invalid=self._invalid,
                stalls=self._stalls,
                rate_hz=recent / _RATE_WINDOW_S,
                longest_gap_ms=self._longest_gap_s * 1000.0,
                excess_latency_ms=None if self._excess_s is None else self._excess_s * 1000.0,
            )


class PhoneLinkThread:
    """A ``PhoneLink`` on its own asyncio loop in a daemon thread, with a synchronous interface."""

    def __init__(self, *args, **kwargs) -> None:
        self.link = PhoneLink(*args, **kwargs)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self.link.port

    def start(self, timeout: float = 5.0) -> None:
        """Start the thread and wait until the socket is bound. Raises the bind error if any."""
        if self._thread is not None:
            return
        ready = threading.Event()
        error: list[BaseException] = []

        def run() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.link.start())
            except BaseException as e:
                error.append(e)
                ready.set()
                loop.close()
                return
            ready.set()
            try:
                loop.run_forever()
            finally:
                loop.run_until_complete(self.link.stop())
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        self._thread = threading.Thread(target=run, name="pointview-phone-link", daemon=True)
        self._thread.start()
        if not ready.wait(timeout):
            raise TimeoutError("phone link did not start")
        if error:
            self._thread.join()
            self._thread = None
            self._loop = None
            raise error[0]

    def stop(self, timeout: float = 5.0) -> None:
        thread, loop = self._thread, self._loop
        if thread is None or loop is None:
            return
        self._thread = None
        self._loop = None
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout)

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def send_nowait(self, message: dict) -> None:
        """Validate on this thread, then schedule the send. No-op if not running or no phone."""
        validate_outgoing(message)
        loop = self._loop
        if loop is None or not self.link.connected():
            return
        asyncio.run_coroutine_threadsafe(self.link.send(message), loop)

    def tracking_target(self) -> TrackingTarget:
        return self.link.tracking_target()

    # Thread-safe pass-throughs (PhoneLink guards its state with a lock).
    def latest(self) -> PhoneState | None:
        return self.link.latest()

    def history(self, seconds: float = 1.0) -> list[PhoneState]:
        return self.link.history(seconds)

    def pop_buttons(self) -> list[ButtonEvent]:
        return self.link.pop_buttons()

    def connected(self) -> bool:
        return self.link.connected()

    def stalled(self) -> bool:
        return self.link.stalled()

    def stats(self) -> LinkStats:
        return self.link.stats()

    def __enter__(self) -> PhoneLinkThread:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
