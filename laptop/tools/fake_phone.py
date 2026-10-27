"""Scriptable phone simulator for PointView (no hardware needed).

CLI::

    python tools/fake_phone.py --url ws://127.0.0.1:8765 --script align_then_point --rate 60 --duration 10 \
        [--lat 37.8 --lon -122.4 --north-offset 0] [--stall 3:200] [--jitter-ms 5]
    python tools/fake_phone.py --list

Library::

    fake_phone = load_module("laptop/tools/fake_phone.py")       # importlib, see below
    stats = await fake_phone.run_fake_phone(url, fake_phone.SCRIPTS["sweep"](), rate_hz=60, duration_s=5)

It depends only on ``websockets`` (it deliberately does not import ``pointview``). To import it from the
app or tests without ``tools`` being a package::

    spec = importlib.util.spec_from_file_location("fake_phone", path)
    mod = importlib.util.module_from_spec(spec); sys.modules["fake_phone"] = mod; spec.loader.exec_module(mod)

What it sends (docs/phone-protocol.md): schema-valid messages at ``rate_hz``.

* ``quat`` rotates phone-frame vectors into a gravity-aligned reference frame R (x east, y north, z up) so
  that the phone's top edge (P +y) points at the scripted yaw (clockwise from +y about +z) and pitch (up is
  positive): ``q = qz(-yaw) * qx(pitch)``. The screen faces up at pitch 0.
* ``omega`` is the matching angular velocity in the phone frame, rad/s:
  ``(pitch_rate, -yaw_rate * sin(pitch), -yaw_rate * cos(pitch))`` (rates in rad/s).
* ``heading_deg`` = (scripted yaw + north_offset_deg) mod 360, so a correct align-and-point pipeline
  recovers ``north_offset`` as the difference between heading and the pointer's tracking yaw.
* ``t_ms`` = this process's wall clock at the sample, plus ``clock_skew_ms`` (to test clock mapping).
"""

import argparse
import asyncio
import itertools
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from websockets.asyncio.client import connect

Quat = tuple[float, float, float, float]


# ----------------------------------------------------------------------------- quaternion helper


def qmul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def qaxis(axis: str, angle_rad: float) -> Quat:
    h = angle_rad / 2.0
    c, s = math.cos(h), math.sin(h)
    return {"x": (c, s, 0.0, 0.0), "y": (c, 0.0, s, 0.0), "z": (c, 0.0, 0.0, s)}[axis]


def qrotate(q: Quat, v: tuple[float, float, float]) -> tuple[float, float, float]:
    """v' = q v q* (Hamilton)."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * (q_vec x v); v' = v + w t + q_vec x t
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def pointer_quat(yaw_deg: float, pitch_deg: float) -> Quat:
    """Phone attitude whose top edge (P +y) points at (yaw clockwise from +y, pitch up)."""
    return qmul(qaxis("z", -math.radians(yaw_deg)), qaxis("x", math.radians(pitch_deg)))


def pointer_direction(yaw_deg: float, pitch_deg: float) -> tuple[float, float, float]:
    """World-frame unit vector (x east, y north, z up) for the given yaw / pitch."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return (math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p))


def phone_omega(
    yaw_rate_dps: float, pitch_rate_dps: float, pitch_deg: float
) -> tuple[float, float, float]:
    """Angular velocity in the phone frame (rad/s) for ``pointer_quat`` moving at the given rates."""
    yr, pr = math.radians(yaw_rate_dps), math.radians(pitch_rate_dps)
    p = math.radians(pitch_deg)
    return (pr, -yr * math.sin(p), -yr * math.cos(p))


# ----------------------------------------------------------------------------- scripts


@dataclass(frozen=True)
class Keyframe:
    t_s: float
    yaw_deg: float
    pitch_deg: float = 0.0
    #: Button presses fired once when the script time reaches ``t_s``.
    buttons: tuple[str, ...] = ()
    #: Layer toggles to report when the script reaches ``t_s`` (sent in that message only).
    layers: dict[str, bool] | None = None


@dataclass
class Script:
    """Keyframes sorted by time; yaw and pitch are linearly interpolated, held after the last one."""

    keyframes: list[Keyframe]
    name: str = "custom"

    def __post_init__(self) -> None:
        if not self.keyframes:
            raise ValueError("a script needs at least one keyframe")
        self.keyframes = sorted(self.keyframes, key=lambda k: k.t_s)

    @property
    def duration_s(self) -> float:
        return self.keyframes[-1].t_s

    def pose_at(self, t_s: float) -> tuple[float, float, float, float]:
        """(yaw_deg, pitch_deg, yaw_rate_dps, pitch_rate_dps) at script time ``t_s``."""
        ks = self.keyframes
        if t_s <= ks[0].t_s:
            return ks[0].yaw_deg, ks[0].pitch_deg, 0.0, 0.0
        for a, b in itertools.pairwise(ks):
            if t_s <= b.t_s:
                dt = b.t_s - a.t_s
                if dt <= 0:
                    return b.yaw_deg, b.pitch_deg, 0.0, 0.0
                f = (t_s - a.t_s) / dt
                return (
                    a.yaw_deg + f * (b.yaw_deg - a.yaw_deg),
                    a.pitch_deg + f * (b.pitch_deg - a.pitch_deg),
                    (b.yaw_deg - a.yaw_deg) / dt,
                    (b.pitch_deg - a.pitch_deg) / dt,
                )
        return ks[-1].yaw_deg, ks[-1].pitch_deg, 0.0, 0.0

    def events_between(self, t0_s: float, t1_s: float, first: bool = False) -> list[Keyframe]:
        """Keyframes with buttons/layers whose time is in (t0, t1] (or [t0, t1] when ``first``)."""
        return [
            k
            for k in self.keyframes
            if (k.buttons or k.layers) and (t0_s < k.t_s <= t1_s or (first and k.t_s == t0_s))
        ]


def script_still() -> Script:
    return Script([Keyframe(0.0, 0.0), Keyframe(10.0, 0.0)], "still")


def script_sweep() -> Script:
    """Yaw 0 -> +30 -> -30 -> 0 over 12 s (peak about 20 deg/s)."""
    return Script(
        [Keyframe(0.0, 0.0), Keyframe(3.0, 30.0), Keyframe(9.0, -30.0), Keyframe(12.0, 0.0)],
        "sweep",
    )


def script_align_then_point() -> Script:
    """Point at yaw 0, tap Align at 1 s, sweep slowly +-20 deg, tap Info a few times."""
    return Script(
        [
            Keyframe(0.0, 0.0),
            Keyframe(1.0, 0.0, buttons=("align",)),
            Keyframe(3.0, 20.0, buttons=("info",)),
            Keyframe(5.0, 20.0, buttons=("info",)),
            Keyframe(9.0, -20.0),
            Keyframe(10.0, -20.0, buttons=("info",)),
            Keyframe(12.0, 0.0),
        ],
        "align_then_point",
    )


SCRIPTS: dict[str, Callable[[], Script]] = {
    "still": script_still,
    "sweep": script_sweep,
    "align_then_point": script_align_then_point,
}


# ----------------------------------------------------------------------------- message building


def build_message(
    script: Script,
    t_s: float,
    t_ms: float,
    *,
    lat: float | None = 37.8078,
    lon: float | None = -122.475,
    north_offset_deg: float = 0.0,
    heading_acc: float | None = 5.0,
    gps_acc: float | None = 5.0,
    buttons: tuple[str, ...] = (),
    layers: dict[str, bool] | None = None,
) -> dict:
    """One schema-valid phone message for script time ``t_s``."""
    yaw, pitch, yaw_rate, pitch_rate = script.pose_at(t_s)
    heading = round((yaw + north_offset_deg) % 360.0, 3) % 360.0
    msg = {
        "v": 1,
        "t_ms": t_ms,
        "quat": list(pointer_quat(yaw, pitch)),
        "omega": list(phone_omega(yaw_rate, pitch_rate, pitch)),
        "heading_deg": heading,
        "heading_acc": heading_acc,
        "lat": lat,
        "lon": lon,
        "gps_acc": gps_acc,
        "buttons": list(buttons),
    }
    if layers:
        msg["layers"] = dict(layers)
    return msg


@dataclass
class FakePhoneStats:
    sent: int = 0
    stalls_injected: int = 0
    #: Laptop -> phone messages received (parsed JSON).
    received: list[dict] = field(default_factory=list)


async def run_fake_phone(
    url: str,
    script: Script,
    rate_hz: float = 60.0,
    duration_s: float | None = None,
    *,
    lat: float | None = 37.8078,
    lon: float | None = -122.475,
    north_offset_deg: float = 0.0,
    stalls: list[tuple[float, float]] | None = None,
    jitter_ms: float = 0.0,
    clock_skew_ms: float = 0.0,
    seed: int | None = 0,
    on_message: Callable[[dict], None] | None = None,
    settings: dict | None = None,
) -> FakePhoneStats:
    """Connect to ``url`` and send the scripted samples in real time.

    ``duration_s`` defaults to the script length. ``stalls`` is a list of ``(at_s, duration_ms)``: no
    samples are sent for that long (button presses due meanwhile go out with the first sample after).
    ``jitter_ms`` delays each send by a random 0..jitter_ms (the sample time stays the true sample time).
    ``clock_skew_ms`` is added to ``t_ms`` to emulate a phone clock that disagrees with the laptop's.
    Laptop messages received are stored in ``stats.received`` and passed to ``on_message``.
    ``settings`` (e.g. ``{"look": "nvg", "passthrough": True, "marker": "point"}``) is sent in the
    first message and then every second, like the real app's settings.
    """
    rng = random.Random(seed)
    stats = FakePhoneStats()
    duration = script.duration_s if duration_s is None else duration_s
    period = 1.0 / rate_hz
    pending_stalls = sorted(stalls or [])
    async with connect(url, compression=None, max_size=None) as ws:

        async def reader() -> None:
            try:
                async for raw in ws:
                    try:
                        m = json.loads(raw)
                    except ValueError:
                        continue
                    stats.received.append(m)
                    if on_message is not None:
                        on_message(m)
            except Exception:
                return

        reader_task = asyncio.create_task(reader())
        loop = asyncio.get_running_loop()
        start = loop.time()
        wall0_ms = time.time() * 1000.0
        last_t = 0.0
        i = 0
        try:
            while True:
                t = i * period
                if t > duration:
                    break
                while pending_stalls and pending_stalls[0][0] <= t:
                    _, dur_ms = pending_stalls.pop(0)
                    stats.stalls_injected += 1
                    # Skip the ticks that fall inside the stall.
                    skip = math.ceil(dur_ms / 1000.0 / period)
                    i += skip
                    t = i * period
                    await asyncio.sleep(max(0.0, start + t - loop.time()))
                if t > duration:
                    break
                delay = start + t - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                events = script.events_between(last_t, t, first=(i == 0))
                buttons = tuple(b for k in events for b in k.buttons)
                layers: dict[str, bool] = {}
                for k in events:
                    layers.update(k.layers or {})
                msg = build_message(
                    script,
                    t,
                    wall0_ms + t * 1000.0 + clock_skew_ms,
                    lat=lat,
                    lon=lon,
                    north_offset_deg=north_offset_deg,
                    buttons=buttons,
                    layers=layers or None,
                )
                if settings and (i == 0 or int(t) != int(last_t)):
                    msg.update(settings)
                if jitter_ms > 0:
                    await asyncio.sleep(rng.uniform(0.0, jitter_ms) / 1000.0)
                await ws.send(json.dumps(msg))
                stats.sent += 1
                last_t = t
                i += 1
        finally:
            await ws.close()
            reader_task.cancel()
    return stats


# ----------------------------------------------------------------------------- CLI


def _parse_stall(text: str) -> tuple[float, float]:
    at, ms = text.split(":")
    return float(at), float(ms)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PointView fake phone")
    ap.add_argument("--url", default="ws://127.0.0.1:8765")
    ap.add_argument("--script", default="align_then_point", choices=sorted(SCRIPTS))
    ap.add_argument("--list", action="store_true", help="list built-in scripts and exit")
    ap.add_argument("--rate", type=float, default=60.0, help="messages per second")
    ap.add_argument("--duration", type=float, default=None, help="seconds (default: script length)")
    ap.add_argument("--lat", type=float, default=37.8078)
    ap.add_argument("--lon", type=float, default=-122.475)
    ap.add_argument(
        "--heading",
        "--north-offset",
        type=float,
        default=0.0,
        dest="north_offset",
        help="degrees added to the scripted yaw to form the compass heading (default 0)",
    )
    ap.add_argument("--stall", action="append", type=_parse_stall, default=[], metavar="AT_S:MS")
    ap.add_argument("--jitter-ms", type=float, default=0.0)
    ap.add_argument("--clock-skew-ms", type=float, default=0.0)
    args = ap.parse_args(argv)
    if args.list:
        for name in sorted(SCRIPTS):
            print(f"{name}: {SCRIPTS[name]().duration_s:g} s")
        return 0
    stats = asyncio.run(
        run_fake_phone(
            args.url,
            SCRIPTS[args.script](),
            args.rate,
            args.duration,
            lat=args.lat,
            lon=args.lon,
            north_offset_deg=args.north_offset,
            stalls=args.stall,
            jitter_ms=args.jitter_ms,
            clock_skew_ms=args.clock_skew_ms,
        )
    )
    print(
        f"sent {stats.sent} messages, {stats.stalls_injected} stalls, {len(stats.received)} replies"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
