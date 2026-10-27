"""Satellites layer: CelesTrak GP data propagated with SGP4, as ``satellite`` entities.

* ``SatelliteLayer`` implements ``interfaces.Layer`` (contracts §7, the focus rule).
* Data: CelesTrak GP in TLE form, per group (``stations``, ``visual`` by default), cached in SQLite
  and refreshed every ``refresh_h`` hours (12 by default; CelesTrak publishes new elements about
  every 2 h and asks clients not to poll more often). Offline, the cache is used however old it is.
* Propagation: a coarse pass (vectorised SGP4 over the whole catalogue, every 2 s of wall time, on
  the layer's thread) finds the satellites above the horizon; each frame ``update()`` propagates
  only those, turns TEME into ECEF (GMST rotation) and ECEF into azimuth / elevation / range at the
  phone's position, and keeps what passes ``world.in_view(..., margin_deg=5)`` with elevation > 0.
* Visibility: sunlit (cylindrical Earth shadow) and the observer in darkness (Sun below -6 degrees,
  from ``world.sun_direction``). ``meta["visible"] == "yes"`` when both hold.
* Focus enrichment: only for ``ctx.focus_id``, the next pass over 24 h (>= 10 degrees elevation) is
  computed on the layer's thread, cached for a few minutes, and put in ``meta["next_pass"]``.

Ported from God's Eye View (MIT License, Copyright (c) 2026 Bilawal Sidhu): the group-to-class table
(``src/data/satelliteClass.js``), the cylindrical shadow test, the low-precision Sun, the -6 degree
darkness rule and the 10 degree / 24 h next-pass definition (``src/data/satellitePass.js``). The
Python implementation, the vectorised propagation and all frame conversions are new.

Data: CelesTrak (https://celestrak.org/NORAD/documentation/gp-data-formats.php), public, free to
use; please keep the request rate low (we fetch each group at most every 12 h).
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, tzinfo
from pathlib import Path

import httpx
import numpy as np
from sgp4.api import Satrec, SatrecArray

from pointview import world
from pointview.entities import Entity, GeoPos
from pointview.interfaces import LayerContext

log = logging.getLogger(__name__)

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"
CELESTRAK_ATTRIBUTION = "Orbital elements: CelesTrak"
USER_AGENT = "PointView/0.1 (open-source AR labels for VITURE glasses; https://github.com/kylephernandez/pointview)"

#: CelesTrak group -> entity class (God's Eye View ``GROUP_CLASS``). Unknown groups -> "visual".
GROUP_CLASS: dict[str, str] = {
    "stations": "station",
    "visual": "visual",
    "gps-ops": "nav",
    "glonass-ops": "nav",
    "galileo": "nav",
    "geo": "geo",
    "starlink": "comms",
}
ISS_NORAD = 25544

# WGS84 (km)
_A = 6378.137
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
#: Mean Earth radius for the cylindrical shadow (as in God's Eye View).
_SHADOW_R = 6371.0

COARSE_PERIOD_S = 2.0
#: Satellites below this elevation at the coarse pass are not propagated per frame. At most a few
#: degrees of motion happen between coarse passes, so -10 leaves a wide safety margin.
COARSE_MIN_EL = -10.0
#: The coarse list is also recomputed when the observer moves this far (km).
COARSE_MOVE_KM = 5.0
VIEW_MARGIN_DEG = 5.0
DARK_SUN_PITCH_DEG = -6.0
PASS_HORIZON_S = 24 * 3600
PASS_STEP_S = 30
PASS_MIN_EL = 10.0
PASS_CACHE_S = 300.0


# ------------------------------------------------------------------------------ astronomy


def _jd_split(unix: float) -> tuple[float, float]:
    """(jd, fr) for sgp4: jd is a half-integer day number, fr the fraction of day (UTC)."""
    days = math.floor(unix / 86400.0)
    return 2440587.5 + days, (unix - days * 86400.0) / 86400.0


def gmst_rad(jd: np.ndarray | float, fr: np.ndarray | float) -> np.ndarray:
    """Greenwich mean sidereal time, radians (IAU-82, Vallado ``gstime``); UT1 taken as UTC."""
    t = ((np.asarray(jd) - 2451545.0) + np.asarray(fr)) / 36525.0
    sec = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t + 0.093104 * t * t - 6.2e-6 * t**3
    return np.mod(sec % 86400.0, 86400.0) * (2 * math.pi / 86400.0)


def teme_to_ecef(r_teme: np.ndarray, gmst: np.ndarray | float) -> np.ndarray:
    """Rotate TEME positions (..., 3) into ECEF by GMST about the z axis (polar motion ignored,
    < 0.01 degree). ``gmst`` broadcasts against the leading dimensions."""
    g = np.asarray(gmst, dtype=float)
    c, s = np.cos(g), np.sin(g)
    x, y, z = r_teme[..., 0], r_teme[..., 1], r_teme[..., 2]
    return np.stack([c * x + s * y, -s * x + c * y, z + 0 * c], axis=-1)


def observer_ecef_km(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = _A / math.sqrt(1 - _E2 * math.sin(lat) ** 2)
    return np.array(
        [
            (n + alt_km) * math.cos(lat) * math.cos(lon),
            (n + alt_km) * math.cos(lat) * math.sin(lon),
            (n * (1 - _E2) + alt_km) * math.sin(lat),
        ]
    )


def look_angles_ecef(
    r_ecef: np.ndarray, lat_deg: float, lon_deg: float, alt_km: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Topocentric (azimuth deg clockwise from north in [0, 360), elevation deg, range km) of ECEF
    positions (..., 3) seen from a geodetic observer."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    d = r_ecef - observer_ecef_km(lat_deg, lon_deg, alt_km)
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    east = -so * d[..., 0] + co * d[..., 1]
    north = -sl * co * d[..., 0] - sl * so * d[..., 1] + cl * d[..., 2]
    up = cl * co * d[..., 0] + cl * so * d[..., 1] + sl * d[..., 2]
    rng = np.sqrt(east**2 + north**2 + up**2)
    el = np.degrees(np.arcsin(np.clip(up / np.maximum(rng, 1e-9), -1, 1)))
    az = np.mod(np.degrees(np.arctan2(east, north)), 360.0)
    return az, el, rng


def ecef_to_geodetic(r_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lat deg, lon deg, height km) on WGS84 by fixed-point iteration (converges to < 1 mm)."""
    x, y, z = r_ecef[..., 0], r_ecef[..., 1], r_ecef[..., 2]
    p = np.hypot(x, y)
    lon = np.degrees(np.arctan2(y, x))
    lat = np.arctan2(z, p * (1 - _E2))
    h = np.zeros_like(p)
    for _ in range(6):
        n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
        h = p / np.maximum(np.cos(lat), 1e-12) - n
        lat = np.arctan2(z, p * (1 - _E2 * n / (n + h)))
    return np.degrees(lat), lon, h


def sun_unit_teme(jd: float, fr: float) -> np.ndarray:
    """Unit vector to the Sun in an equatorial inertial frame (low-precision almanac, ~0.01 deg;
    the TEME/mean-equinox difference is ignored). From God's Eye View ``solarPositionECI``."""
    d = (jd - 2451545.0) + fr
    g = math.radians((357.529 + 0.98560028 * d) % 360.0)
    q = (280.459 + 0.98564736 * d) % 360.0
    lam = math.radians((q + 1.915 * math.sin(g) + 0.02 * math.sin(2 * g)) % 360.0)
    eps = math.radians(23.439 - 0.00000036 * d)
    return np.array([math.cos(lam), math.cos(eps) * math.sin(lam), math.sin(eps) * math.sin(lam)])


def sunlit(r_teme: np.ndarray, sun_unit: np.ndarray) -> np.ndarray:
    """Cylindrical Earth-shadow test for positions (..., 3) km: True when outside the shadow."""
    proj = r_teme @ sun_unit
    perp = r_teme - proj[..., None] * sun_unit
    return (proj > 0) | (np.linalg.norm(perp, axis=-1) > _SHADOW_R)


# ----------------------------------------------------------------------------------- TLE data


@dataclass(frozen=True, slots=True)
class _Sat:
    norad: int
    name: str
    klass: str
    group: str
    satrec: Satrec


@dataclass(frozen=True, slots=True)
class NextPass:
    """The next pass at or above PASS_MIN_EL. Times are Unix seconds (UTC)."""

    rise: float
    peak: float
    set: float
    max_el_deg: float
    visible: bool
    #: True when the satellite is already above PASS_MIN_EL at the start (``rise`` = start).
    up_now: bool = False


def parse_tle(text: str, group: str) -> list[_Sat]:
    """Parse 3-line (or bare 2-line) TLE text. Bad records are skipped."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out: list[_Sat] = []
    i = 0
    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            l1, l2 = lines[i], lines[i + 1]
            name = (
                lines[i - 1].strip() if i > 0 and not lines[i - 1].startswith(("1 ", "2 ")) else ""
            )
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name, l1, l2 = lines[i].strip(), lines[i + 1], lines[i + 2]
            i += 3
        else:
            i += 1
            continue
        if name.startswith("0 "):
            name = name[2:]
        try:
            sat = Satrec.twoline2rv(l1[:69], l2[:69])
            norad = int(sat.satnum)
        except (ValueError, TypeError, RuntimeError):
            continue
        if not np.isfinite(sat.no_kozai) or sat.no_kozai <= 0:
            continue
        klass = "station" if norad == ISS_NORAD else GROUP_CLASS.get(group, "visual")
        out.append(_Sat(norad, name or f"NORAD {norad}", klass, group, sat))
    return out


class _Cache:
    """SQLite cache of raw TLE text per group. Thread-safe, failure-tolerant."""

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
                "CREATE TABLE IF NOT EXISTS satellite_gp "
                "(grp TEXT PRIMARY KEY, fetched REAL, body TEXT)"
            )
            db.commit()
            self._db = db
        except (sqlite3.Error, OSError) as e:
            log.warning("satellite cache disabled (%s)", e)

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    def get(self, group: str) -> tuple[float, str] | None:
        with self._lock:
            if self._db is None:
                return None
            try:
                row = self._db.execute(
                    "SELECT fetched, body FROM satellite_gp WHERE grp = ?", (group,)
                ).fetchone()
            except sqlite3.Error as e:
                log.warning("satellite cache error: %s", e)
                return None
        return None if row is None else (float(row[0]), str(row[1]))

    def put(self, group: str, body: str) -> None:
        with self._lock:
            if self._db is None:
                return
            try:
                self._db.execute(
                    "INSERT OR REPLACE INTO satellite_gp VALUES (?,?,?)",
                    (group, time.time(), body),
                )
                self._db.commit()
            except sqlite3.Error as e:
                log.warning("satellite cache error: %s", e)


# -------------------------------------------------------------------------------------- layer


@dataclass(frozen=True, slots=True)
class _Coarse:
    unix: float
    lat: float
    lon: float
    sats: tuple[_Sat, ...]
    arr: SatrecArray
    generation: int


class SatelliteLayer:
    """Satellites above the horizon and inside the glasses' view, as ``satellite`` entities."""

    name = "satellites"

    def __init__(
        self,
        cache_path: Path | None,
        client: httpx.Client | None = None,
        groups: tuple[str, ...] = ("stations", "visual"),
        refresh_h: float = 12,
        *,
        backoff_s: float = 60.0,
        tz: tzinfo | None = None,
    ):
        self.groups = tuple(groups)
        self.refresh_s = float(refresh_h) * 3600.0
        self._backoff_s = backoff_s
        self._tz = tz  # None = the laptop's local time zone
        self._own_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(10.0, read=40.0), headers={"User-Agent": USER_AGENT}
        )
        self._cache = _Cache(cache_path)

        self._lock = threading.Lock()
        self._sats: tuple[_Sat, ...] = ()
        self._by_norad: dict[int, _Sat] = {}
        self._all_arr: SatrecArray | None = None
        self._generation = 0
        self._coarse: _Coarse | None = None
        self._want: tuple[float, float, float] | None = None  # (unix, lat, lon)
        self._pass_cache: dict[int, tuple[float, float, NextPass | None]] = {}
        self._pass_pending: set[tuple[int, float, float, float]] = set()

        self._failures = 0
        self._next_data_check = 0.0
        self.last_error: str | None = None
        self.fetches = 0
        self.pass_computations = 0
        self.coarse_computations = 0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._data_event = threading.Event()

    # --- Layer API

    def update(self, ctx: LayerContext) -> list[Entity]:
        self._ensure_thread()
        p = ctx.phone
        if (
            "satellite" not in ctx.enabled
            or ctx.view is None
            or p is None
            or p.lat is None
            or p.lon is None
        ):
            return []
        lat, lon, unix = p.lat, p.lon, ctx.unix_time
        co = self._coarse
        if (
            co is None
            or co.generation != self._generation
            or self._coarse_stale(co, unix, lat, lon)
        ):
            self._want = (unix, lat, lon)
            self._wake.set()
        if co is None or co.generation != self._generation or not co.sats:
            return []
        return self._frame(ctx, co, lat, lon, unix)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=5)
        if self._own_client:
            self._client.close()
        self._cache.close()

    # --- extras for the app / tests

    @property
    def has_data(self) -> bool:
        return bool(self._sats)

    def wait_for_data(self, timeout: float = 5.0) -> bool:
        """Block until a catalogue (network or cache) has been loaded. For tests/startup."""
        return self._data_event.wait(timeout)

    def compute_now(self, ctx: LayerContext) -> None:
        """Run the coarse pass synchronously for this context (tests; the thread does it live)."""
        p = ctx.phone
        if p is not None and p.lat is not None and p.lon is not None:
            self._compute_coarse(ctx.unix_time, p.lat, p.lon)

    def drain(self, timeout: float = 5.0) -> bool:
        """Wait for queued next-pass computations to finish. For tests."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if not self._pass_pending:
                return True
            time.sleep(0.005)
        return not self._pass_pending

    def next_pass(self, norad: int, unix: float, lat: float, lon: float) -> NextPass | None:
        """Blocking next-pass search (what the thread runs for the focused satellite)."""
        sat = self._by_norad.get(norad)
        if sat is None:
            return None
        self.pass_computations += 1
        return compute_next_pass(sat.satrec, unix, lat, lon)

    # --- per frame

    @staticmethod
    def _coarse_stale(co: _Coarse, unix: float, lat: float, lon: float) -> bool:
        if abs(unix - co.unix) >= COARSE_PERIOD_S:
            return True
        dlat = (lat - co.lat) * 111.0
        dlon = (lon - co.lon) * 111.0 * math.cos(math.radians(lat))
        return math.hypot(dlat, dlon) > COARSE_MOVE_KM

    def _frame(self, ctx: LayerContext, co: _Coarse, lat: float, lon: float, unix: float):
        jd, fr = _jd_split(unix)
        e, r, _ = co.arr.sgp4(np.array([jd]), np.array([fr]))
        r_teme = r[:, 0, :]
        ecef = teme_to_ecef(r_teme, gmst_rad(jd, fr))
        az, el, rng = look_angles_ecef(ecef, lat, lon)
        up = np.flatnonzero((e[:, 0] == 0) & (el > 0.0) & np.isfinite(el))
        if up.size == 0:
            return []
        view = ctx.view
        keep = [
            int(i)
            for i in up
            if world.in_view(
                float(az[i]),
                float(el[i]),
                view.head_quat,
                view.north_offset_deg,
                view.display_fov_deg,
                margin_deg=VIEW_MARGIN_DEG,
            )
        ]
        if not keep:
            return []
        k = np.array(keep)
        lit = sunlit(r_teme[k], sun_unit_teme(jd, fr))
        sun_pitch = world.sun_direction(unix, lat, lon)[1]
        dark = sun_pitch < DARK_SUN_PITCH_DEG
        glat, glon, galt = ecef_to_geodetic(ecef[k])
        out: list[Entity] = []
        for j, i in enumerate(keep):
            sat = co.sats[i]
            eid = f"sat:{sat.norad}"
            meta = {
                "altitude_km": f"{galt[j]:.0f}",
                "class": sat.klass,
                "norad": str(sat.norad),
                "group": sat.group,
                "sunlit": "yes" if lit[j] else "no",
                "visible": "yes" if (lit[j] and dark) else "no",
            }
            if ctx.focus_id == eid:
                np_str = self._next_pass_text(sat, unix, lat, lon)
                if np_str:
                    meta["next_pass"] = np_str
            out.append(
                Entity(
                    id=eid,
                    kind="satellite",
                    label=sat.name,
                    world_yaw_deg=float(az[i]),
                    world_pitch_deg=float(el[i]),
                    updated_at=ctx.t,
                    distance_m=float(rng[i]) * 1000.0,
                    meta=meta,
                    geo=GeoPos(float(glat[j]), float(glon[j]), float(galt[j]) * 1000.0),
                )
            )
        return out

    # --- next pass (focus only)

    def _next_pass_text(self, sat: _Sat, unix: float, lat: float, lon: float) -> str | None:
        now = time.monotonic()
        with self._lock:
            hit = self._pass_cache.get(sat.norad)
            fresh = (
                hit is not None
                and now - hit[0] < PASS_CACHE_S
                and abs(unix - hit[1]) < PASS_CACHE_S
            )
            if not fresh:
                key = (sat.norad, round(unix), lat, lon)
                if not any(k[0] == sat.norad for k in self._pass_pending):
                    self._pass_pending.add(key)
                    self._wake.set()
        if hit is None:
            return None
        return format_pass(hit[2], self._tz)

    # --- background thread

    def _ensure_thread(self) -> None:
        if self._thread is None and not self._stop.is_set():
            self._thread = threading.Thread(target=self._run, name="satellites", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                if time.monotonic() >= self._next_data_check:
                    self._refresh_data()
                want = self._want
                if want is not None and self._all_arr is not None:
                    co = self._coarse
                    if (
                        co is None
                        or co.generation != self._generation
                        or self._coarse_stale(co, *want)
                    ):
                        self._compute_coarse(*want)
                self._run_passes()
            except Exception:  # keep the thread alive; a layer must never crash the app
                log.exception("satellite worker failed")

    def _run_passes(self) -> None:
        while True:
            with self._lock:
                if not self._pass_pending:
                    return
                key = next(iter(self._pass_pending))
            norad, unix, lat, lon = key
            try:
                result = self.next_pass(norad, float(unix), lat, lon)
            except Exception:
                log.exception("next-pass search failed")
                result = None
            with self._lock:
                self._pass_cache[norad] = (time.monotonic(), float(unix), result)
                self._pass_pending.discard(key)

    def _compute_coarse(self, unix: float, lat: float, lon: float) -> None:
        with self._lock:
            sats, arr, gen = self._sats, self._all_arr, self._generation
        if arr is None or not sats:
            return
        jd, fr = _jd_split(unix)
        e, r, _ = arr.sgp4(np.array([jd]), np.array([fr]))
        ecef = teme_to_ecef(r[:, 0, :], gmst_rad(jd, fr))
        _, el, _ = look_angles_ecef(ecef, lat, lon)
        idx = np.flatnonzero((e[:, 0] == 0) & (el > COARSE_MIN_EL) & np.isfinite(el))
        chosen = tuple(sats[int(i)] for i in idx)
        sub = SatrecArray([s.satrec for s in chosen]) if chosen else None
        self.coarse_computations += 1
        self._coarse = _Coarse(unix, lat, lon, chosen, sub, gen)  # type: ignore[arg-type]

    # --- data

    def _refresh_data(self) -> None:
        """Load every group (cache first), fetching those older than ``refresh_s``."""
        now = time.time()
        bodies: dict[str, str] = {}
        stale = False
        failed = False
        for g in self.groups:
            cached = self._cache.get(g)
            if cached is not None:
                bodies[g] = cached[1]
            if cached is None or now - cached[0] >= self.refresh_s:
                stale = True
        if bodies and not self._sats:
            self._install(bodies)  # show the cache straight away, even offline
        if stale:
            for g in self.groups:
                cached = self._cache.get(g)
                if cached is not None and now - cached[0] < self.refresh_s:
                    continue
                body = self._fetch(g)
                if body is None:
                    failed = True
                    continue
                self._cache.put(g, body)
                bodies[g] = body
            if bodies:
                self._install(bodies)
        if failed:
            self._failures += 1
            delay = min(self._backoff_s * 2 ** (self._failures - 1), 3600.0)
        else:
            self._failures = 0
            delay = 600.0
        self._next_data_check = time.monotonic() + delay

    def _fetch(self, group: str) -> str | None:
        self.fetches += 1
        try:
            r = self._client.get(CELESTRAK_URL, params={"GROUP": group, "FORMAT": "tle"})
            if r.status_code != 200:
                self.last_error = f"CelesTrak {group}: HTTP {r.status_code}"
                return None
            text = r.text
            if not parse_tle(text, group):
                self.last_error = f"CelesTrak {group}: no usable TLEs"
                return None
            return text
        except httpx.HTTPError as e:
            self.last_error = f"CelesTrak {group}: {e}"
            return None

    def _install(self, bodies: dict[str, str]) -> None:
        seen: dict[int, _Sat] = {}
        for g in self.groups:  # the order of ``groups`` decides which duplicate wins
            for s in parse_tle(bodies.get(g, ""), g):
                seen.setdefault(s.norad, s)
        sats = tuple(seen.values())
        arr = SatrecArray([s.satrec for s in sats]) if sats else None
        with self._lock:
            self._sats = sats
            self._by_norad = dict(seen)
            self._all_arr = arr
            self._generation += 1
        if sats:
            self._data_event.set()


# --------------------------------------------------------------------------------- next pass


def compute_next_pass(
    satrec: Satrec,
    unix: float,
    lat: float,
    lon: float,
    horizon_s: float = PASS_HORIZON_S,
    step_s: float = PASS_STEP_S,
    min_el: float = PASS_MIN_EL,
) -> NextPass | None:
    """First pass at or above ``min_el`` within ``horizon_s``; None if there is none.

    Samples every ``step_s`` seconds (vectorised), interpolates the crossings linearly and the peak
    with a parabola. ``visible`` = sunlit and Sun below -6 degrees at some sample above ``min_el``.
    """
    ts = unix + np.arange(0.0, horizon_s + step_s, step_s)
    days = np.floor(ts / 86400.0)
    jd = 2440587.5 + days
    fr = (ts - days * 86400.0) / 86400.0
    e, r, _ = satrec.sgp4_array(jd, fr)
    ecef = teme_to_ecef(r, gmst_rad(jd, fr))
    _, el, _ = look_angles_ecef(ecef, lat, lon)
    el = np.where(e != 0, -90.0, el)
    above = el >= min_el
    if not above.any():
        return None
    i0 = int(np.argmax(above))
    up_now = i0 == 0
    if up_now:
        rise = float(ts[0])
    else:
        rise = float(ts[i0 - 1] + (min_el - el[i0 - 1]) / (el[i0] - el[i0 - 1]) * step_s)
    below_after = np.flatnonzero(~above[i0:])
    if below_after.size:
        i1 = i0 + int(below_after[0])  # first sample below
        set_t = float(ts[i1 - 1] + (el[i1 - 1] - min_el) / (el[i1 - 1] - el[i1]) * step_s)
    else:
        i1 = len(ts)
        set_t = float(ts[-1])
    seg = slice(i0, i1)
    k = i0 + int(np.argmax(el[seg]))
    peak_t, max_el = float(ts[k]), float(el[k])
    if 0 < k < len(ts) - 1:
        y0, y1, y2 = el[k - 1], el[k], el[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            off = 0.5 * (y0 - y2) / den
            peak_t = float(ts[k] + off * step_s)
            max_el = float(y1 - 0.25 * (y0 - y2) * off)
    visible = False
    for i in range(i0, i1):
        jd_i, fr_i = float(jd[i]), float(fr[i])
        lit = bool(sunlit(r[i], sun_unit_teme(jd_i, fr_i)))
        if lit and world.sun_direction(float(ts[i]), lat, lon)[1] < DARK_SUN_PITCH_DEG:
            visible = True
            break
    return NextPass(rise, peak_t, set_t, max_el, visible, up_now)


def format_pass(p: NextPass | None, tz: tzinfo | None = None) -> str:
    """Short text like ``rises 21:04 · max 63° · visible`` (24 h clock, ``tz`` or local time)."""
    if p is None:
        return "no pass in 24 h"

    def hhmm(t: float) -> str:
        return datetime.fromtimestamp(t, tz).strftime("%H:%M")

    head = f"up now, sets {hhmm(p.set)}" if p.up_now else f"rises {hhmm(p.rise)}"
    return f"{head} · max {p.max_el_deg:.0f}° · {'visible' if p.visible else 'not visible'}"
