"""Event and vessel layers: USGS earthquakes (keyless) and AISStream ships (needs an API key).

Concept follows God's Eye View (https://github.com/bilawalsidhu/gods-eye-view, MIT License,
Copyright (c) 2026 Bilawal Sidhu): earthquakes from the USGS GeoJSON summary feed and live vessels
from the AISStream WebSocket. The code here is a fresh Python implementation written against the
public API documentation (USGS GeoJSON summary format; https://aisstream.io/documentation), not a
line-by-line port. The ideas shared with God's Eye View: treat AIS sentinel values (SOG 102.3,
COG 360, lat 91 / lon 181) as "not available", dead-reckon ships between reports, drop stale ships.

Contract (docs/contracts.md section 7):

* ``update()`` never does I/O. Network work runs on each layer's own thread.
* Only entities inside the glasses' view (margin 5 deg) are returned; no view or no GPS fix -> [].
* Per-item enrichment (quake detail, vessel static data) only for ``ctx.focus_id``.

Terms: USGS data is public domain (https://www.usgs.gov/information-policies-and-instructions/
copyrights-and-credits); AISStream is a free beta service that needs an API key and registration
(https://aisstream.io/documentation), with no accuracy guarantee and not for navigation.
Attribution strings: ``USGS_ATTRIBUTION``, ``AISSTREAM_ATTRIBUTION``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import httpx

from pointview import world
from pointview.entities import Entity, GeoPos
from pointview.interfaces import LayerContext

log = logging.getLogger(__name__)

USGS_ATTRIBUTION = "Earthquakes: USGS"
AISSTREAM_ATTRIBUTION = "Vessels: AISStream.io (not for navigation)"
USER_AGENT = "PointView/0.1 (open-source AR labels for VITURE glasses)"

USGS_FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
USGS_DETAIL_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/{id}.geojson"
AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"

POLL_INTERVAL_S = 300.0
#: A cached quake detail younger than this is reused without a request.
DETAIL_FRESH_S = 3600.0
VIEW_MARGIN_DEG = 5.0
MAX_QUAKES = 400

KNOT_MS = 1852.0 / 3600.0
VESSEL_MAX_AGE_S = 600.0
RESUBSCRIBE_KM = 5.0
#: Bounding box padding (km) beyond ``radius_km`` so a move below the resubscribe threshold keeps
#: full coverage.
BBOX_PAD_KM = RESUBSCRIBE_KM
MAX_VESSELS = 2000


# ------------------------------------------------------------------------------------- helpers


def _now(ctx: LayerContext) -> float:
    return ctx.unix_time if ctx.unix_time > 0 else time.time()


def _in_view(ctx: LayerContext, yaw: float, pitch: float) -> bool:
    v = ctx.view
    if v is None:
        return False
    return world.in_view(
        yaw,
        pitch,
        v.head_quat,
        v.north_offset_deg,
        v.display_fov_deg,
        margin_deg=VIEW_MARGIN_DEG,
    )


def _ago(seconds: float) -> str:
    s = max(0.0, seconds)
    if s < 90:
        return "just now"
    if s < 3600:
        return f"{round(s / 60)} min ago"
    if s < 48 * 3600:
        return f"{round(s / 3600)} h ago"
    return f"{round(s / 86400)} d ago"


# ------------------------------------------------------------------------------------- cache


class _Cache:
    """SQLite cache for the USGS feed and quake details. Thread-safe, failure-tolerant."""

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
                "CREATE TABLE IF NOT EXISTS quake_feed (k TEXT PRIMARY KEY, fetched REAL, body TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS quake_detail "
                "(id TEXT PRIMARY KEY, fetched REAL, body TEXT)"
            )
            db.commit()
            self._db = db
        except (sqlite3.Error, OSError) as e:
            log.warning("event cache disabled (%s)", e)

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    def _run(self, sql: str, args: tuple = (), *, write: bool = False) -> list:
        with self._lock:
            if self._db is None:
                return []
            try:
                rows = self._db.execute(sql, args).fetchall()
                if write:
                    self._db.commit()
                return rows
            except sqlite3.Error as e:
                log.warning("event cache error (%s)", e)
                return []

    def get_feed(self) -> tuple[float, str] | None:
        rows = self._run("SELECT fetched, body FROM quake_feed WHERE k='feed'")
        return (rows[0][0], rows[0][1]) if rows else None

    def put_feed(self, fetched: float, body: str) -> None:
        self._run(
            "INSERT OR REPLACE INTO quake_feed VALUES ('feed', ?, ?)", (fetched, body), write=True
        )

    def get_detail(self, quake_id: str) -> tuple[float, str] | None:
        rows = self._run("SELECT fetched, body FROM quake_detail WHERE id=?", (quake_id,))
        return (rows[0][0], rows[0][1]) if rows else None

    def put_detail(self, quake_id: str, fetched: float, body: str) -> None:
        self._run(
            "INSERT OR REPLACE INTO quake_detail VALUES (?, ?, ?)",
            (quake_id, fetched, body),
            write=True,
        )


# ------------------------------------------------------------------------------------- quakes


@dataclass(frozen=True, slots=True)
class _Quake:
    id: str  # USGS id
    mag: float
    place: str
    time_s: float  # Unix seconds
    lat: float
    lon: float
    depth_km: float
    url: str


def parse_usgs_feed(body: dict | str | bytes) -> list[_Quake]:
    """Parse a USGS GeoJSON summary feed. Malformed features are skipped."""
    try:
        data = json.loads(body) if isinstance(body, (str, bytes)) else body
        features = data.get("features", [])
    except (ValueError, AttributeError):
        return []
    out: list[_Quake] = []
    for f in features:
        try:
            p = f["properties"]
            coords = f["geometry"]["coordinates"]
            lon, lat = coords[:2]
            depth = coords[2] if len(coords) > 2 else 0.0
            mag = p.get("mag")
            t = p.get("time")
            if mag is None or t is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                continue
            out.append(
                _Quake(
                    id=str(f["id"]),
                    mag=float(mag),
                    place=str(p.get("place") or "unknown location"),
                    time_s=float(t) / 1000.0,
                    lat=float(lat),
                    lon=float(lon),
                    depth_km=float(depth or 0.0),
                    url=str(p.get("url") or ""),
                )
            )
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out


def parse_usgs_detail(body: dict | str | bytes) -> dict[str, str]:
    """Felt reports and tsunami flag (plus CDI / PAGER alert when present) as entity meta."""
    try:
        data = json.loads(body) if isinstance(body, (str, bytes)) else body
        p = data["properties"]
    except (ValueError, KeyError, TypeError):
        return {}
    meta: dict[str, str] = {}
    if p.get("felt") is not None:
        meta["felt_reports"] = str(int(p["felt"]))
    if p.get("tsunami") is not None:
        meta["tsunami"] = "yes" if p["tsunami"] else "no"
    if p.get("cdi") is not None:
        meta["felt_intensity_cdi"] = f"{float(p['cdi']):.1f}"
    if p.get("alert"):
        meta["pager_alert"] = str(p["alert"])
    return meta


class EventLayer:
    """Recent earthquakes near the phone, as ``event`` entities."""

    name = "events"

    def __init__(
        self,
        cache_path: Path | None,
        client: httpx.Client | None = None,
        radius_km: float = 500.0,
        min_mag: float = 2.5,
        *,
        poll_interval_s: float = POLL_INTERVAL_S,
        feed_url: str = USGS_FEED_URL,
    ):
        self.radius_km = float(radius_km)
        self.min_mag = float(min_mag)
        self._poll_interval_s = poll_interval_s
        self._feed_url = feed_url
        self._own_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(10.0, read=30.0), headers={"User-Agent": USER_AGENT}
        )
        self._cache = _Cache(cache_path)
        self._lock = threading.Lock()
        self._quakes: tuple[_Quake, ...] = ()
        self._details: dict[str, dict[str, str]] = {}
        self._detail_pending: set[str] = set()
        self._detail_want: str | None = None
        self.feed_requests = 0
        self.detail_requests = 0
        self.last_error: str | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._data_event = threading.Event()
        self._thread: threading.Thread | None = None

    # --- Layer API

    def update(self, ctx: LayerContext) -> list[Entity]:
        p = ctx.phone
        if "event" not in ctx.enabled or p is None or p.lat is None or p.lon is None:
            return []
        self._ensure_thread()
        if ctx.view is None:
            return []
        lat, lon = p.lat, p.lon
        now = _now(ctx)
        radius_m = self.radius_km * 1000.0
        with self._lock:
            quakes = self._quakes
            details = self._details
        out: list[Entity] = []
        for q in quakes:
            if q.mag < self.min_mag:
                continue
            if world.haversine_m(lat, lon, q.lat, q.lon) > radius_m:
                continue
            depth_m = q.depth_km * 1000.0
            yaw, pitch, rng = world.look_angles(lat, lon, 0.0, q.lat, q.lon, -depth_m)
            # Far quakes sit below the horizon (curvature, depth): draw them on the horizon.
            pitch = max(0.0, pitch)
            if not _in_view(ctx, yaw, pitch):
                continue
            eid = f"quake:{q.id}"
            meta = {
                "mag": f"{q.mag:.1f}",
                "place": q.place,
                "time": datetime.fromtimestamp(q.time_s, UTC).strftime("%Y-%m-%d %H:%M:%SZ"),
                "depth_km": f"{q.depth_km:.1f}",
            }
            if q.url:
                meta["url"] = q.url
            if eid == ctx.focus_id:
                self._want_detail(q.id)
                meta.update(details.get(q.id, {}))
            out.append(
                Entity(
                    id=eid,
                    kind="event",
                    label=f"M{q.mag:.1f} quake · {_ago(now - q.time_s)}",
                    world_yaw_deg=yaw,
                    world_pitch_deg=pitch,
                    updated_at=ctx.t,
                    distance_m=rng,
                    meta=meta,
                    geo=GeoPos(q.lat, q.lon, -depth_m),
                )
            )
        return out

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

    def wait_for_data(self, timeout: float = 5.0) -> bool:
        return self._data_event.wait(timeout)

    def wait_for_detail(self, quake_id: str, timeout: float = 5.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._lock:
                if quake_id in self._details:
                    return True
            time.sleep(0.005)
        return False

    # --- background

    def _want_detail(self, quake_id: str) -> None:
        with self._lock:
            if quake_id in self._details or quake_id in self._detail_pending:
                return
            self._detail_want = quake_id
        self._wake.set()

    def _ensure_thread(self) -> None:
        if self._thread is not None or self._stop.is_set():
            return
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="events", daemon=True)
            self._thread.start()

    def _set_quakes(self, quakes: list[_Quake]) -> None:
        quakes = sorted(quakes, key=lambda q: -q.time_s)[:MAX_QUAKES]
        with self._lock:
            self._quakes = tuple(quakes)
        self._data_event.set()

    def _run(self) -> None:
        cached = self._cache.get_feed()
        if cached:
            self._set_quakes(parse_usgs_feed(cached[1]))
        next_poll = 0.0
        while not self._stop.is_set():
            if time.monotonic() >= next_poll:
                self._poll_feed()
                next_poll = time.monotonic() + self._poll_interval_s
            self._fetch_wanted_detail()
            self._wake.wait(timeout=max(0.05, min(1.0, next_poll - time.monotonic())))
            self._wake.clear()

    def _poll_feed(self) -> None:
        self.feed_requests += 1
        try:
            r = self._client.get(self._feed_url)
            r.raise_for_status()
            if '"features"' not in r.text:
                raise ValueError("not a USGS feed")
            self._cache.put_feed(time.time(), r.text)
            self._set_quakes(parse_usgs_feed(r.text))
            self.last_error = None
        except (httpx.HTTPError, ValueError) as e:
            self.last_error = f"feed: {e}"
            log.warning("USGS feed failed (%s)", e)

    def _fetch_wanted_detail(self) -> None:
        with self._lock:
            qid = self._detail_want
            self._detail_want = None
            if qid is None or qid in self._details or qid in self._detail_pending:
                return
            self._detail_pending.add(qid)
        cached = self._cache.get_detail(qid)
        try:
            if cached and time.time() - cached[0] < DETAIL_FRESH_S:
                meta = parse_usgs_detail(cached[1])
            else:
                self.detail_requests += 1
                r = self._client.get(USGS_DETAIL_URL.format(id=qid))
                r.raise_for_status()
                self._cache.put_detail(qid, time.time(), r.text)
                meta = parse_usgs_detail(r.text)
            with self._lock:
                self._details[qid] = meta
        except (httpx.HTTPError, ValueError) as e:
            self.last_error = f"detail: {e}"
            log.warning("USGS detail %s failed (%s)", qid, e)
            if cached:  # stale beats nothing
                with self._lock:
                    self._details[qid] = parse_usgs_detail(cached[1])
        finally:
            with self._lock:
                self._detail_pending.discard(qid)


# ------------------------------------------------------------------------------------- vessels


def ship_type_name(code: int | None) -> str | None:
    """AIS "ship and cargo type" code (ITU-R M.1371) -> short English name."""
    if code is None or code <= 0:
        return None
    named = {
        30: "fishing",
        31: "towing",
        32: "towing",
        33: "dredger",
        34: "diving ops",
        35: "military",
        36: "sailing",
        37: "pleasure craft",
        50: "pilot",
        51: "search & rescue",
        52: "tug",
        53: "port tender",
        55: "law enforcement",
    }
    if code in named:
        return named[code]
    for lo, hi, name in (
        (40, 49, "high-speed craft"),
        (60, 69, "passenger"),
        (70, 79, "cargo"),
        (80, 89, "tanker"),
    ):
        if lo <= code <= hi:
            return name
    return "vessel"


@dataclass(frozen=True, slots=True)
class _Report:
    mmsi: int
    lat: float
    lon: float
    sog_kn: float | None
    cog_deg: float | None
    heading_deg: float | None
    t: float  # Unix seconds of the report
    name: str | None


@dataclass(frozen=True, slots=True)
class _Static:
    name: str | None = None
    ship_type: int | None = None
    destination: str | None = None
    eta: str | None = None
    imo: int | None = None
    call_sign: str | None = None


def _parse_time_utc(s: object) -> float | None:
    """AISStream MetaData.time_utc, e.g. ``"2022-12-29 18:22:32.318353 +0000 UTC"``."""
    if not isinstance(s, str):
        return None
    try:
        head = s.split(" +")[0].strip()
        if "." in head:
            base, frac = head.split(".", 1)
            dt = datetime.strptime(f"{base}.{frac[:6]}", "%Y-%m-%d %H:%M:%S.%f")
        else:
            dt = datetime.strptime(head, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None


def _clean(s: object) -> str | None:
    if not isinstance(s, str):
        return None
    s = s.replace("@", " ").strip()
    return s or None


def _num(x: object) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


_POSITION_TYPES = ("PositionReport", "StandardClassBPositionReport", "ExtendedClassBPositionReport")


def parse_ais_message(
    raw: str | bytes | dict, received_at: float
) -> _Report | tuple[int, _Static] | None:
    """Parse one AISStream message (documented format).

    Returns a ``_Report`` for position messages, ``(mmsi, _Static)`` for static data, ``None`` for
    anything else (error envelopes, unknown types, invalid positions).
    """
    try:
        msg = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        mtype = msg["MessageType"]
        meta = msg.get("MetaData") or {}
        body = (msg.get("Message") or {}).get(mtype) or {}
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    mmsi = body.get("UserID", meta.get("MMSI"))
    if not isinstance(mmsi, int) or mmsi <= 0:
        return None
    name = _clean(meta.get("ShipName"))

    if mtype in _POSITION_TYPES:
        lat = _num(body.get("Latitude", meta.get("latitude")))
        lon = _num(body.get("Longitude", meta.get("longitude")))
        if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return None  # 91 / 181 are AIS "not available"
        sog = _num(body.get("Sog"))
        cog = _num(body.get("Cog"))
        hdg = _num(body.get("TrueHeading"))
        return _Report(
            mmsi=mmsi,
            lat=lat,
            lon=lon,
            sog_kn=sog if sog is not None and 0 <= sog < 102.3 else None,
            cog_deg=cog if cog is not None and 0 <= cog < 360 else None,
            heading_deg=hdg if hdg is not None and 0 <= hdg < 360 else None,
            t=_parse_time_utc(meta.get("time_utc")) or received_at,
            name=name or _clean(body.get("Name")),
        )
    if mtype == "ShipStaticData":
        eta = body.get("Eta") or {}
        eta_s = None
        try:
            mo, d, h, mi = (int(eta[k]) for k in ("Month", "Day", "Hour", "Minute"))
            if 1 <= mo <= 12 and 1 <= d <= 31 and h < 24 and mi < 60:  # 0 / 24 / 60: unavailable
                eta_s = f"{mo:02d}-{d:02d} {h:02d}:{mi:02d} UTC"
        except (KeyError, TypeError, ValueError):
            pass
        imo = body.get("ImoNumber")
        stype = body.get("Type")
        return mmsi, _Static(
            name=_clean(body.get("Name")) or name,
            ship_type=stype if isinstance(stype, int) else None,
            destination=_clean(body.get("Destination")),
            eta=eta_s,
            imo=imo if isinstance(imo, int) and imo > 0 else None,
            call_sign=_clean(body.get("CallSign")),
        )
    return None


def dead_reckon(
    lat: float, lon: float, sog_kn: float | None, cog_deg: float | None, dt_s: float
) -> tuple[float, float]:
    """Advance a position by SOG (knots) along COG (true course, degrees) for ``dt_s`` seconds."""
    if not sog_kn or cog_deg is None or dt_s <= 0:
        return lat, lon
    d = sog_kn * KNOT_MS * dt_s / 6_371_000.0
    th = math.radians(cog_deg)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(th))
    l2 = l1 + math.atan2(
        math.sin(th) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2)
    )
    return math.degrees(p2), (math.degrees(l2) + 540.0) % 360.0 - 180.0


def bbox_around(lat: float, lon: float, half_km: float) -> list[list[list[float]]]:
    """AISStream ``BoundingBoxes`` value (two [lat, lon] corners) around a point."""
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * max(0.05, math.cos(math.radians(lat))))
    return [
        [
            [max(-90.0, lat - dlat), max(-180.0, lon - dlon)],
            [min(90.0, lat + dlat), min(180.0, lon + dlon)],
        ]
    ]


class VesselLayer:
    """Ships near the phone from AISStream. Inactive (returns []) without an API key."""

    name = "vessels"

    def __init__(
        self,
        api_key: str | None = None,
        radius_km: float = 30.0,
        *,
        connect: Callable | None = None,
        url: str = AISSTREAM_URL,
        reconnect_s: float = 5.0,
        check_s: float = 1.0,
    ):
        key = api_key if api_key is not None else os.environ.get("AISSTREAM_API_KEY")
        self._key = key.strip() if key and key.strip() else None
        self.radius_km = float(radius_km)
        self._connect = connect
        self._url = url
        self._reconnect_s = reconnect_s
        self._check_s = check_s
        self._lock = threading.Lock()
        self._reports: dict[int, _Report] = {}
        self._static: dict[int, _Static] = {}
        self._pos: tuple[float, float] | None = None
        self._subscribed: tuple[float, float] | None = None
        self.subscriptions_sent = 0
        self.last_error: str | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_evt: asyncio.Event | None = None
        self._stopped = threading.Event()
        if self._key is None:
            log.info("vessel layer inactive: no AISStream API key (set AISSTREAM_API_KEY)")

    @property
    def active(self) -> bool:
        return self._key is not None

    # --- Layer API

    def update(self, ctx: LayerContext) -> list[Entity]:
        if self._key is None:
            return []
        p = ctx.phone
        if "vessel" not in ctx.enabled or p is None or p.lat is None or p.lon is None:
            return []
        lat, lon = p.lat, p.lon
        with self._lock:
            self._pos = (lat, lon)
            reports = tuple(self._reports.values())
            static = self._static
        self._ensure_thread()
        if ctx.view is None:
            return []
        now = _now(ctx)
        radius_m = self.radius_km * 1000.0
        out: list[Entity] = []
        for r in reports:
            age = now - r.t
            if age > VESSEL_MAX_AGE_S:
                continue
            vlat, vlon = dead_reckon(r.lat, r.lon, r.sog_kn, r.cog_deg, age)
            if world.haversine_m(lat, lon, vlat, vlon) > radius_m:
                continue
            yaw, pitch, rng = world.look_angles(lat, lon, 0.0, vlat, vlon, 0.0)
            if not _in_view(ctx, yaw, pitch):
                continue
            st = static.get(r.mmsi)
            kind = ship_type_name(st.ship_type) if st else None
            label = r.name or (st.name if st else None) or f"MMSI {r.mmsi}"
            if kind:
                label += f" · {kind}"
            eid = f"ais:{r.mmsi}"
            meta = {"mmsi": str(r.mmsi)}
            if r.sog_kn is not None:
                meta["sog_kn"] = f"{r.sog_kn:.1f}"
            if r.cog_deg is not None:
                meta["cog_deg"] = f"{r.cog_deg:.0f}"
            if eid == ctx.focus_id and st:
                if st.destination:
                    meta["destination"] = st.destination
                if st.eta:
                    meta["eta"] = st.eta
                if kind:
                    meta["ship_type"] = kind
                if st.imo:
                    meta["imo"] = str(st.imo)
                if st.call_sign:
                    meta["call_sign"] = st.call_sign
            out.append(
                Entity(
                    id=eid,
                    kind="vessel",
                    label=label,
                    world_yaw_deg=yaw,
                    world_pitch_deg=pitch,
                    updated_at=ctx.t,
                    distance_m=rng,
                    meta=meta,
                    geo=GeoPos(vlat, vlon, 0.0),
                )
            )
        return out

    def close(self) -> None:
        self._stopped.set()
        loop, ev = self._loop, self._stop_evt
        if loop is not None and ev is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(ev.set)
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=5)

    # --- ingestion (also used directly by tests)

    def ingest(self, raw: str | bytes | dict, received_at: float | None = None) -> None:
        now = received_at if received_at is not None else time.time()
        parsed = parse_ais_message(raw, now)
        if parsed is None:
            return
        with self._lock:
            if isinstance(parsed, _Report):
                old = self._reports.get(parsed.mmsi)
                if old is not None and old.t > parsed.t:
                    return
                if parsed.name is None and old is not None:
                    parsed = replace(parsed, name=old.name)
                self._reports[parsed.mmsi] = parsed
                if len(self._reports) > MAX_VESSELS:
                    self._prune_locked(now)
            else:
                mmsi, st = parsed
                self._static[mmsi] = st

    def _prune_locked(self, now: float) -> None:
        for mmsi in [m for m, r in self._reports.items() if now - r.t > VESSEL_MAX_AGE_S]:
            del self._reports[mmsi]
            self._static.pop(mmsi, None)

    # --- websocket thread

    def subscription_message(self, lat: float, lon: float) -> str:
        return json.dumps(
            {
                "APIKey": self._key,
                "BoundingBoxes": bbox_around(lat, lon, self.radius_km + BBOX_PAD_KM),
                "FilterMessageTypes": [
                    "PositionReport",
                    "StandardClassBPositionReport",
                    "ShipStaticData",
                ],
            }
        )

    def _ensure_thread(self) -> None:
        if self._thread is not None or self._stopped.is_set():
            return
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._thread_main, name="vessels", daemon=True)
            self._thread.start()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception:  # never take the app down
            log.exception("vessel stream crashed")

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_evt = asyncio.Event()
        if self._stopped.is_set():
            return
        connect = self._connect
        if connect is None:
            import websockets

            connect = websockets.connect
        while not self._stop_evt.is_set():
            try:
                async with connect(self._url) as ws:
                    await self._session(ws)
            except Exception as e:
                self.last_error = str(e)
                log.warning("AISStream connection failed (%s)", e)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_evt.wait(), self._reconnect_s)

    async def _session(self, ws) -> None:
        assert self._stop_evt is not None
        self._subscribed = None
        reader = asyncio.ensure_future(self._read(ws))
        stop = asyncio.ensure_future(self._stop_evt.wait())
        try:
            while not stop.done() and not reader.done():
                with self._lock:
                    pos = self._pos
                if pos is not None and (
                    self._subscribed is None
                    or world.haversine_m(*self._subscribed, *pos) > RESUBSCRIBE_KM * 1000.0
                ):
                    await ws.send(self.subscription_message(*pos))
                    self._subscribed = pos
                    self.subscriptions_sent += 1
                await asyncio.wait(
                    {stop, reader}, timeout=self._check_s, return_when=asyncio.FIRST_COMPLETED
                )
        finally:
            reader.cancel()
            stop.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader

    async def _read(self, ws) -> None:
        async for raw in ws:
            self.ingest(raw)
