"""Aircraft layer: live ADS-B traffic around the phone, as ``aircraft`` entities (contracts v2 section 7).

* ``AircraftLayer`` implements ``interfaces.Layer``. ``update()`` never does I/O. A background thread
  polls adsb.lol ``/v2/point/{lat}/{lon}/{radius_nm}`` around the phone and falls back to OpenSky
  ``states/all`` with a bounding box. Between polls every aircraft is dead-reckoned from its own
  report time, ground speed, track and vertical rate; aircraft not heard from for 60 s are dropped.
* Each frame the direction from the phone to each aircraft comes from ``world.look_angles`` and only
  aircraft passing ``world.in_view(..., margin_deg=5)`` are returned (no view or no GPS fix: ``[]``).
  A cheap vectorised cone test first rejects aircraft far outside the view so 300 aircraft stay
  well inside the 5 ms budget.
* Per-item enrichment (the route "JFK -> LAX") runs only for ``ctx.focus_id``, on a worker thread,
  and is cached per callsign (memory + SQLite).
* The last poll is stored in SQLite so a restart shows traffic within seconds.

Data terms: adsb.lol data is ODbL 1.0 (``ADSBLOL_ATTRIBUTION``); OpenSky data is for research and
non-commercial use (``OPENSKY_ATTRIBUTION``). Routes come from the adsb.lol "vrs-standing-data" set
(schedule-based, so a callsign flown on another leg can show the wrong route).

Ported from God's Eye View (https://github.com/bilawalsidhu/gods-eye-view), MIT License,
Copyright (c) 2026 Bilawal Sidhu: the adsb.lol field mapping in ``src/data/adsbLolFallback.js``
(ft/kt/fpm unit factors, ``alt_baro == "ground"``, ``seen_pos``-based observation time) and the
"sticky enrichment" idea of ``src/data/aircraftMeta.js`` (a later snapshot that misses callsign,
registration or type does not erase what we already knew).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import queue
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from pointview import world
from pointview.entities import Entity, GeoPos
from pointview.interfaces import LayerContext

log = logging.getLogger(__name__)

ADSBLOL_ATTRIBUTION = "adsb.lol · ODbL 1.0"
OPENSKY_ATTRIBUTION = "OpenSky Network"
USER_AGENT = (
    "PointView/0.1 (open-source AR labels for VITURE glasses; https://github.com/pointview)"
)
ADSBLOL_POINT_URL = "https://api.adsb.lol/v2/point/{lat:.4f}/{lon:.4f}/{radius_nm}"
OPENSKY_URL = "https://opensky-network.org/api/states/all"
ROUTE_URL = "https://vrs-standing-data.adsb.lol/routes/{prefix}/{callsign}.json"

FT_M = 0.3048
KT_MPS = 0.514444
FPM_MPS = 0.00508
EARTH_R = 6_371_000.0
_WGS84_A = 6_378_137.0
_WGS84_E2 = 6.69437999014e-3

#: Aircraft not heard from for longer than this are dropped (seconds).
STALE_S = 60.0
#: Dead reckoning never extrapolates further than this past the last position report (seconds).
MAX_EXTRAPOLATE_S = 30.0
#: adsb.lol caps the point query radius at 250 nm.
MAX_RADIUS_NM = 250
VIEW_MARGIN_DEG = 5.0
#: Extra cone padding for the cheap prefilter, so it never rejects something in_view would accept.
PREFILTER_PAD_DEG = 8.0
MAX_BACKOFF_S = 300.0
CACHE_MAX_AGE_S = 3600.0
ROUTE_TTL_S = 7 * 86400.0
ROUTE_MISS_TTL_S = 86400.0
ROUTE_RETRY_S = 60.0

EMERGENCY_SQUAWKS = {"7500": "hijack", "7600": "radio failure", "7700": "general emergency"}
_CALLSIGN_RE = re.compile(r"^[A-Z0-9]{3,8}$")


# ------------------------------------------------------------------------------------- records


@dataclass(frozen=True, slots=True)
class _Ac:
    """One aircraft as reported by a source (SI units, report times as Unix seconds)."""

    hex: str
    callsign: str
    reg: str
    type: str
    lat: float
    lon: float
    alt_m: float
    gs_mps: float
    track_deg: float
    vrate_mps: float
    #: When the position was observed / when the aircraft was last heard (Unix seconds).
    t_pos: float
    t_seen: float
    squawk: str
    military: bool
    ground: bool


class _Snapshot:
    """An immutable poll result: static info per aircraft plus arrays for vectorised math."""

    __slots__ = (
        "ac",
        "alt",
        "cos_trk",
        "gs",
        "lat",
        "lon",
        "n",
        "sin_trk",
        "source",
        "t_pos",
        "t_seen",
        "vrate",
    )

    def __init__(self, ac: list[_Ac], source: str):
        self.ac = ac
        self.source = source
        self.n = len(ac)
        f = np.float64
        self.lat = np.array([a.lat for a in ac], dtype=f)
        self.lon = np.array([a.lon for a in ac], dtype=f)
        self.alt = np.array([a.alt_m for a in ac], dtype=f)
        self.gs = np.array([a.gs_mps for a in ac], dtype=f)
        trk = np.radians(np.array([a.track_deg for a in ac], dtype=f))
        self.sin_trk = np.sin(trk)
        self.cos_trk = np.cos(trk)
        self.vrate = np.array([a.vrate_mps for a in ac], dtype=f)
        self.t_pos = np.array([a.t_pos for a in ac], dtype=f)
        self.t_seen = np.array([a.t_seen for a in ac], dtype=f)


# -------------------------------------------------------------------------------------- parsing


def _num(v) -> float | None:
    if isinstance(v, bool) or v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _sticky(new: str, old: str) -> str:
    """God's Eye View ``stickyText``: keep what we knew when a snapshot misses a field."""
    return new or old


def parse_adsblol(
    payload: dict, prev: dict[str, _Ac] | None = None, fallback_now: float | None = None
) -> list[_Ac]:
    """adsb.lol / readsb ``{"now": ms, "ac": [...]}`` -> aircraft with a position.

    Observation times come from the payload: ``now - seen_pos`` (position) and ``now - seen``.
    ``dbFlags`` bit 0 is the database's military flag.
    """
    now = _num(payload.get("now"))
    if now is None:
        now = (fallback_now if fallback_now is not None else time.time()) * 1000.0
    now_s = now / 1000.0 if now > 1e11 else now
    rows = payload["ac"]
    if not isinstance(rows, list):
        raise ValueError("adsb.lol 'ac' is not a list")
    out: list[_Ac] = []
    seen_hex: set[str] = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        hexid = str(r.get("hex") or "").strip().lower().lstrip("~")
        lat, lon = _num(r.get("lat")), _num(r.get("lon"))
        if not hexid or hexid in seen_hex or lat is None or lon is None:
            continue
        if abs(lat) > 90 or abs(lon) > 180:
            continue
        seen_hex.add(hexid)
        ground = r.get("alt_baro") == "ground"
        alt_geom, alt_baro = _num(r.get("alt_geom")), _num(r.get("alt_baro"))
        alt_ft = 0.0 if ground else (alt_geom if alt_geom is not None else alt_baro)
        if alt_ft is None:
            continue  # no altitude at all: cannot place it in the sky
        seen = max(0.0, _num(r.get("seen")) or 0.0)
        seen_pos = max(
            0.0, _num(r.get("seen_pos")) if _num(r.get("seen_pos")) is not None else seen
        )
        vr = _num(r.get("baro_rate"))
        if vr is None:
            vr = _num(r.get("geom_rate"))
        old = (prev or {}).get(hexid)
        out.append(
            _Ac(
                hex=hexid,
                callsign=_sticky(str(r.get("flight") or "").strip(), old.callsign if old else ""),
                reg=_sticky(str(r.get("r") or "").strip(), old.reg if old else ""),
                type=_sticky(str(r.get("t") or "").strip(), old.type if old else ""),
                lat=lat,
                lon=lon,
                alt_m=alt_ft * FT_M,
                gs_mps=(_num(r.get("gs")) or 0.0) * KT_MPS,
                track_deg=_num(r.get("track")) or 0.0,
                vrate_mps=0.0 if ground or vr is None else vr * FPM_MPS,
                t_pos=now_s - seen_pos,
                t_seen=now_s - seen,
                squawk=str(r.get("squawk") or "").strip(),
                military=bool(int(_num(r.get("dbFlags")) or 0) & 1),
                ground=ground,
            )
        )
    return out


def parse_opensky(payload: dict, prev: dict[str, _Ac] | None = None) -> list[_Ac]:
    """OpenSky ``states/all`` -> aircraft with a position. Has no type or registration."""
    states = payload["states"] or []
    if not isinstance(states, list):
        raise ValueError("OpenSky 'states' is not a list")
    out: list[_Ac] = []
    seen_hex: set[str] = set()
    for s in states:
        if not isinstance(s, list) or len(s) < 15:
            continue
        hexid = str(s[0] or "").strip().lower()
        lon, lat = _num(s[5]), _num(s[6])
        if not hexid or hexid in seen_hex or lat is None or lon is None:
            continue
        seen_hex.add(hexid)
        ground = bool(s[8])
        alt_m = _num(s[13])
        if alt_m is None:
            alt_m = _num(s[7])
        if ground:
            alt_m = 0.0
        if alt_m is None:
            continue
        t_last = _num(s[4]) or _num(payload.get("time")) or time.time()
        t_pos = _num(s[3]) or t_last
        old = (prev or {}).get(hexid)
        out.append(
            _Ac(
                hex=hexid,
                callsign=_sticky(str(s[1] or "").strip(), old.callsign if old else ""),
                reg=old.reg if old else "",
                type=old.type if old else "",
                lat=lat,
                lon=lon,
                alt_m=alt_m,
                gs_mps=_num(s[9]) or 0.0,
                track_deg=_num(s[10]) or 0.0,
                vrate_mps=0.0 if ground else (_num(s[11]) or 0.0),
                t_pos=t_pos,
                t_seen=max(t_last, t_pos),
                squawk=str(s[14] or "").strip(),
                military=False,  # OpenSky does not say
                ground=ground,
            )
        )
    return out


def opensky_bbox(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """(lamin, lomin, lamax, lomax) around a point."""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(0.05, math.cos(math.radians(lat))))
    return (
        max(-90.0, lat - dlat),
        max(-180.0, lon - dlon),
        min(90.0, lat + dlat),
        min(180.0, lon + dlon),
    )


def route_from_standing_data(data: dict) -> str | None:
    """``"LAS -> EWR -> IAD"`` from a vrs-standing-data route file (IATA codes, else ICAO)."""
    codes = str(data.get("_airport_codes_iata") or data.get("airport_codes") or "").strip()
    parts = [p for p in codes.split("-") if p]
    if len(parts) < 2:
        return None
    return " → ".join(parts)


# ---------------------------------------------------------------------------------- vectorised


def _ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sl * sl)
    return np.stack(
        [
            (n + alt_m) * cl * np.cos(lon),
            (n + alt_m) * cl * np.sin(lon),
            (n * (1.0 - _WGS84_E2) + alt_m) * sl,
        ],
        axis=-1,
    )


def _enu_dirs(obs_lat: float, obs_lon: float, obs_alt: float, lat, lon, alt) -> np.ndarray:
    """Unit east/north/up directions from the observer to each point (n x 3)."""
    o = _ecef(np.array([obs_lat]), np.array([obs_lon]), np.array([obs_alt]))[0]
    d = _ecef(lat, lon, alt) - o
    la, lo = math.radians(obs_lat), math.radians(obs_lon)
    sl, cl, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    e = -so * d[:, 0] + co * d[:, 1]
    n = -sl * co * d[:, 0] - sl * so * d[:, 1] + cl * d[:, 2]
    u = cl * co * d[:, 0] + cl * so * d[:, 1] + sl * d[:, 2]
    enu = np.stack([e, n, u], axis=-1)
    norm = np.linalg.norm(enu, axis=1)
    return enu / np.maximum(norm, 1.0)[:, None]


# -------------------------------------------------------------------------------------- cache


class _Cache:
    """SQLite cache: the last poll (raw body) and per-callsign routes. Failure-tolerant."""

    def __init__(self, path: Path | None):
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        if path is None:
            return
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(path), check_same_thread=False)
            db.execute(
                "CREATE TABLE IF NOT EXISTS aircraft_poll (id INTEGER PRIMARY KEY CHECK (id = 1), "
                "source TEXT, lat REAL, lon REAL, fetched REAL, body TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS aircraft_route "
                "(callsign TEXT PRIMARY KEY, fetched REAL, route TEXT)"
            )
            db.commit()
            self._db = db
        except (sqlite3.Error, OSError) as e:
            log.warning("aircraft cache disabled (%s)", e)

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    def _run(self, sql: str, args: tuple = (), *, write: bool = False):
        with self._lock:
            if self._db is None:
                return []
            try:
                rows = self._db.execute(sql, args).fetchall()
                if write:
                    self._db.commit()
                return rows
            except sqlite3.Error as e:
                log.warning("aircraft cache error: %s", e)
                return []

    def put_poll(self, source: str, lat: float, lon: float, fetched: float, body: str) -> None:
        self._run(
            "INSERT OR REPLACE INTO aircraft_poll VALUES (1,?,?,?,?,?)",
            (source, lat, lon, fetched, body),
            write=True,
        )

    def get_poll(self) -> tuple[str, float, float, float, str] | None:
        rows = self._run("SELECT source, lat, lon, fetched, body FROM aircraft_poll WHERE id = 1")
        return rows[0] if rows else None

    def put_route(self, callsign: str, route: str | None) -> None:
        self._run(
            "INSERT OR REPLACE INTO aircraft_route VALUES (?,?,?)",
            (callsign, time.time(), route or ""),
            write=True,
        )

    def get_route(self, callsign: str) -> tuple[float, str] | None:
        rows = self._run(
            "SELECT fetched, route FROM aircraft_route WHERE callsign = ?", (callsign,)
        )
        return (rows[0][0], rows[0][1]) if rows else None


# ----------------------------------------------------------------------------------------- layer


class _Source:
    """Per-source backoff state."""

    def __init__(self, name: str, base_s: float):
        self.name = name
        self.base_s = base_s
        self.failures = 0
        self.next_at = 0.0  # on the layer's monotonic clock

    def ok(self, now: float, interval: float) -> None:
        self.failures = 0
        self.next_at = now + interval

    def fail(self, now: float, retry_after: float = 0.0) -> None:
        self.failures += 1
        delay = min(self.base_s * 2 ** (self.failures - 1), MAX_BACKOFF_S)
        self.next_at = now + max(delay, min(retry_after, MAX_BACKOFF_S))


class _PollError(Exception):
    def __init__(self, msg: str, retry_after: float = 0.0):
        super().__init__(msg)
        self.retry_after = retry_after


class AircraftLayer:
    """Live aircraft in the glasses' view, as ``aircraft`` entities."""

    name = "aircraft"

    def __init__(
        self,
        cache_path: Path | None,
        client: httpx.Client | None = None,
        radius_km: float = 40,
        poll_s: float = 8.0,
        *,
        fallback_poll_s: float = 30.0,
        include_ground: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.radius_km = float(radius_km)
        self.poll_s = float(poll_s)
        self.fallback_poll_s = max(float(fallback_poll_s), self.poll_s)
        self.include_ground = include_ground
        self._clock = clock
        self._own_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(10.0), headers={"User-Agent": USER_AGENT}, follow_redirects=True
        )
        self._cache = _Cache(cache_path)

        self._lock = threading.Lock()
        self._poll_lock = threading.Lock()
        self._pos: tuple[float, float] | None = None
        self._snap: _Snapshot | None = None
        self._by_hex: dict[str, _Ac] = {}
        self._primary = _Source("adsb.lol", self.poll_s)
        self._fallback = _Source("opensky", self.fallback_poll_s)
        self._next_cycle = 0.0
        self._cache_loaded = False
        self.last_error: str | None = None
        self.source: str | None = None
        self.requests_sent = {"adsb.lol": 0, "opensky": 0, "route": 0}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._data_event = threading.Event()

        self._routes: dict[str, tuple[float, str | None]] = {}  # callsign -> (expires, route)
        self._route_pending: set[str] = set()
        self._route_q: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        self._route_thread: threading.Thread | None = None

    # --- Layer API

    def update(self, ctx: LayerContext) -> list[Entity]:
        p = ctx.phone
        if "aircraft" not in ctx.enabled or p is None or p.lat is None or p.lon is None:
            return []
        lat, lon = p.lat, p.lon
        with self._lock:
            self._pos = (lat, lon)
            snap = self._snap
        self._ensure_thread()
        self._wake.set()
        view = ctx.view
        if view is None or snap is None or snap.n == 0:
            return []

        now = ctx.unix_time if ctx.unix_time > 0 else time.time()
        fresh = (now - snap.t_seen) <= STALE_S
        if not self.include_ground:
            fresh &= snap.alt > 0.0
        idx = np.nonzero(fresh)[0]
        if idx.size == 0:
            return []
        dt = np.clip(now - snap.t_pos[idx], 0.0, MAX_EXTRAPOLATE_S)
        dist = snap.gs[idx] * dt
        c_lat = np.cos(np.radians(snap.lat[idx]))
        alat = snap.lat[idx] + np.degrees(dist * snap.cos_trk[idx] / EARTH_R)
        alon = snap.lon[idx] + np.degrees(
            dist * snap.sin_trk[idx] / (EARTH_R * np.maximum(c_lat, 0.01))
        )
        aalt = np.maximum(snap.alt[idx] + snap.vrate[idx] * dt, 0.0)

        # Cheap cone test (observer altitude is 0 until the phone reports one).
        dirs = _enu_dirs(lat, lon, 0.0, alat, alon, aalt)
        vy, vp = math.radians(view.yaw_deg), math.radians(view.pitch_deg)
        center = np.array([math.sin(vy) * math.cos(vp), math.cos(vy) * math.cos(vp), math.sin(vp)])
        fov_h, fov_v = view.display_fov_deg
        half = min(180.0, math.hypot(fov_h, fov_v) / 2 + VIEW_MARGIN_DEG + PREFILTER_PAD_DEG)
        cand = np.nonzero(dirs @ center >= math.cos(math.radians(half)))[0]

        focus = ctx.focus_id
        out: list[Entity] = []
        for k in cand:
            a = snap.ac[idx[k]]
            la, lo, al = float(alat[k]), float(alon[k]), float(aalt[k])
            yaw, pitch, rng = world.look_angles(lat, lon, 0.0, la, lo, al)
            if not world.in_view(
                yaw,
                pitch,
                view.head_quat,
                view.north_offset_deg,
                view.display_fov_deg,
                margin_deg=VIEW_MARGIN_DEG,
            ):
                continue
            eid = f"ac:{a.hex}"
            meta = {
                "altitude_ft": str(round(al / FT_M)),
                "speed_kt": str(round(a.gs_mps / KT_MPS)),
                "track": str(round(a.track_deg) % 360),
            }
            if a.type:
                meta["type"] = a.type
            if a.military:
                meta["military"] = "1"
            if a.squawk in EMERGENCY_SQUAWKS:
                meta["emergency"] = EMERGENCY_SQUAWKS[a.squawk]
                meta["squawk"] = a.squawk
            if focus == eid:
                route = self._route_for(a.callsign)
                if route:
                    meta["route"] = route
            out.append(
                Entity(
                    id=eid,
                    kind="aircraft",
                    label=a.callsign or a.reg or a.hex.upper(),
                    world_yaw_deg=yaw % 360.0,
                    world_pitch_deg=pitch,
                    updated_at=ctx.t,
                    distance_m=rng,
                    meta=meta,
                    geo=GeoPos(la, lo, al),
                )
            )
        return out

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._route_q.put(None)
        for t in (self._thread, self._route_thread):
            if t is not None and t is not threading.current_thread():
                t.join(timeout=5)
        if self._own_client:
            self._client.close()
        self._cache.close()

    # --- extras for the app / tests

    @property
    def attribution(self) -> str:
        return OPENSKY_ATTRIBUTION if self.source == "opensky" else ADSBLOL_ATTRIBUTION

    @property
    def has_data(self) -> bool:
        return self._snap is not None

    def wait_for_data(self, timeout: float = 5.0) -> bool:
        return self._data_event.wait(timeout)

    # --- routes (focus-only enrichment)

    def _route_for(self, callsign: str) -> str | None:
        """Cached route, or None while a background fetch for this callsign is pending."""
        cs = callsign.strip().upper()
        if not _CALLSIGN_RE.match(cs):
            return None
        with self._lock:
            hit = self._routes.get(cs)
            if hit is not None and self._clock() < hit[0]:
                return hit[1]
            if cs in self._route_pending:
                return None
            self._route_pending.add(cs)
        if self._route_thread is None and not self._stop.is_set():
            self._route_thread = threading.Thread(
                target=self._route_worker, name="aircraft-route", daemon=True
            )
            self._route_thread.start()
        self._route_q.put(cs)
        return None

    def _route_worker(self) -> None:
        while not self._stop.is_set():
            cs = self._route_q.get()
            if cs is None:
                break
            try:
                self.resolve_route(cs)
            except Exception:
                log.exception("route lookup failed")
                with self._lock:
                    self._route_pending.discard(cs)

    def resolve_route(self, callsign: str) -> str | None:
        """Blocking: disk cache, then the network. Stores the result for ``_route_for``."""
        cs = callsign.strip().upper()
        route: str | None = None
        cached = self._cache.get_route(cs)
        ttl = ROUTE_TTL_S
        if cached is not None:
            ttl = ROUTE_TTL_S if cached[1] else ROUTE_MISS_TTL_S
        if cached is not None and time.time() - cached[0] < ttl:
            route = cached[1] or None
        else:
            try:
                self.requests_sent["route"] += 1
                r = self._client.get(
                    ROUTE_URL.format(prefix=cs[:2], callsign=cs), headers={"User-Agent": USER_AGENT}
                )
            except httpx.HTTPError:
                r = None
            if r is not None and r.status_code == 200:
                try:
                    route = route_from_standing_data(r.json())
                except ValueError:
                    route = None
                self._cache.put_route(cs, route)
            elif r is not None and r.status_code == 404:
                self._cache.put_route(cs, None)
            else:
                # transient failure: do not ask again for a while
                with self._lock:
                    self._routes[cs] = (self._clock() + ROUTE_RETRY_S, None)
                    self._route_pending.discard(cs)
                return None
        with self._lock:
            self._routes[cs] = (self._clock() + (ROUTE_TTL_S if route else ROUTE_MISS_TTL_S), route)
            self._route_pending.discard(cs)
        return route

    # --- background polling

    def _ensure_thread(self) -> None:
        if self._thread is None and not self._stop.is_set():
            self._thread = threading.Thread(target=self._run, name="aircraft", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self.poll_once()
            except Exception:  # never let the layer thread die
                log.exception("aircraft poll failed")
                self._primary.fail(self._clock())
                self._next_cycle = self._clock() + self.poll_s

    def poll_once(self, force: bool = False) -> bool:
        with self._poll_lock:
            return self._poll_once(force)

    def _poll_once(self, force: bool) -> bool:
        """One scheduling step (the thread calls this): load the cache once, then poll if a source
        is due. Returns True when a network request was made. Public so tests can drive it."""
        with self._lock:
            pos = self._pos
        if pos is None:
            return False
        if not self._cache_loaded:
            self._cache_loaded = True
            self._load_cache(pos)
        now = self._clock()
        if not force and now < self._next_cycle:
            return False
        lat, lon = pos
        tried = False
        if force or now >= self._primary.next_at:
            tried = True
            if self._try(self._primary, lambda: self._fetch_adsblol(lat, lon), lat, lon):
                self._next_cycle = self._primary.next_at
                return True
        # OpenSky is only a fallback: never poll it while adsb.lol is healthy.
        if self._primary.failures > 0 and (force or now >= self._fallback.next_at):
            tried = True
            if self._try(self._fallback, lambda: self._fetch_opensky(lat, lon), lat, lon):
                self._next_cycle = min(self._primary.next_at, self._fallback.next_at)
                return True
        if self._primary.failures == 0:
            self._next_cycle = self._primary.next_at
        else:
            self._next_cycle = min(self._primary.next_at, self._fallback.next_at)
        return tried

    def _try(self, src: _Source, fetch, lat: float, lon: float) -> bool:
        try:
            fetched = fetch()
        except _PollError as e:
            self.last_error = f"{src.name}: {e}"
            src.fail(self._clock(), e.retry_after)
            return False
        text, ac = fetched
        with self._lock:
            self._by_hex = {a.hex: a for a in ac}
            self._snap = _Snapshot(ac, src.name)
        self.source = src.name
        self.last_error = None
        src.ok(self._clock(), self.poll_s if src is self._primary else self.fallback_poll_s)
        self._cache.put_poll(src.name, lat, lon, time.time(), text)
        self._data_event.set()
        return True

    def _get(self, kind: str, url: str, params: dict | None = None) -> httpx.Response:
        self.requests_sent[kind] += 1
        try:
            r = self._client.get(url, params=params, headers={"User-Agent": USER_AGENT})
        except httpx.HTTPError as e:
            raise _PollError(type(e).__name__) from e
        if r.status_code != 200:
            ra = 0.0
            for h in ("Retry-After", "X-Rate-Limit-Retry-After-Seconds"):
                with contextlib.suppress(ValueError):
                    ra = max(ra, float(r.headers.get(h, 0)))
            raise _PollError(f"HTTP {r.status_code}", ra)
        return r

    def _fetch_adsblol(self, lat: float, lon: float) -> tuple[str, list[_Ac]]:
        nm = max(1, min(MAX_RADIUS_NM, round(self.radius_km / 1.852)))
        r = self._get("adsb.lol", ADSBLOL_POINT_URL.format(lat=lat, lon=lon, radius_nm=nm))
        try:
            data = r.json()
            ac = parse_adsblol(data, self._by_hex)
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            raise _PollError("bad adsb.lol response") from e
        return r.text, ac

    def _fetch_opensky(self, lat: float, lon: float) -> tuple[str, list[_Ac]]:
        lamin, lomin, lamax, lomax = opensky_bbox(lat, lon, self.radius_km)
        r = self._get(
            "opensky",
            OPENSKY_URL,
            {
                "lamin": f"{lamin:.4f}",
                "lomin": f"{lomin:.4f}",
                "lamax": f"{lamax:.4f}",
                "lomax": f"{lomax:.4f}",
            },
        )
        try:
            ac = parse_opensky(r.json(), self._by_hex)
        except (ValueError, KeyError, AttributeError, TypeError) as e:
            raise _PollError("bad OpenSky response") from e
        return r.text, ac

    def _load_cache(self, pos: tuple[float, float]) -> None:
        row = self._cache.get_poll()
        if row is None:
            return
        source, clat, clon, fetched, body = row
        if time.time() - fetched > CACHE_MAX_AGE_S:
            return
        if world.haversine_m(pos[0], pos[1], clat, clon) > self.radius_km * 500.0:
            return
        try:
            data = json.loads(body)
            ac = (
                parse_opensky(data)
                if source == "opensky"
                else parse_adsblol(data, fallback_now=fetched)
            )
        except (ValueError, KeyError, AttributeError, TypeError):
            return
        with self._lock:
            if self._snap is None:
                self._by_hex = {a.hex: a for a in ac}
                self._snap = _Snapshot(ac, source)
                self.source = source
        self._data_event.set()
