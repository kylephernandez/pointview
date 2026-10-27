"""OSM building footprints from Overpass (contracts §11).

Query: ``way["building"]`` and ``relation["building"]`` around a point with ``out geom``. Only the
outer rings are used. Heights come from ``height``, else ``building:levels`` x 3 m, else 6 m.

Production uses Overpass only. (The fixtures were recorded through the OSM 0.6 API and converted
to the Overpass JSON shape, see tests/fixtures/mesh/README.md.)

Data © OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

OSM_ATTRIBUTION = "© OpenStreetMap contributors"
USER_AGENT = "PointView/0.1 (https://github.com/kylephernandez/pointview)"
OVERPASS_ENDPOINTS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)

DEFAULT_HEIGHT_M = 6.0
LEVEL_HEIGHT_M = 3.0
#: Taller than anything real is a typo ("1200") or a unit mistake; fall back to the default.
MAX_HEIGHT_M = 900.0
_FT_M = 0.3048

_NUM = r"[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)"
_FT_IN = re.compile(
    rf"^\s*({_NUM})\s*(?:'|ft|feet|foot)\s*(?:({_NUM})\s*(?:\"|in|inch|inches)?)?\s*$"
)
_VALUE = re.compile(rf"^\s*({_NUM})\s*(m|meter|meters|metre|metres)?\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Footprint:
    """One building outline: closed outer ring (first point not repeated) and vertical extent."""

    id: str  # "way/123" or "relation/45" ("relation/45#1" for a second outer ring)
    ring: tuple[tuple[float, float], ...]  # (lat, lon) degrees
    height_m: float  # top above the ground
    min_height_m: float = 0.0  # bottom above the ground


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def parse_length_m(value: object) -> float | None:
    """A tag value like "12", "12.5 m", "12,5", "40 ft", "40'6\"" -> meters, or None if junk."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or len(s) > 40:
        return None
    try:
        m = _VALUE.match(s)
        if m:
            v = _num(m.group(1))
        else:
            m = _FT_IN.match(s)
            if not m:
                return None
            v = (_num(m.group(1)) + (_num(m.group(2)) / 12.0 if m.group(2) else 0.0)) * _FT_M
    except ValueError:
        return None
    if not math.isfinite(v) or v < 0 or v > MAX_HEIGHT_M:
        return None
    return v


def parse_levels(value: object) -> float | None:
    """Level counts are plain numbers ("3", "2.5"); anything else is junk."""
    if value is None:
        return None
    m = re.match(rf"^\s*({_NUM})\s*$", str(value))
    if not m:
        return None
    try:
        v = _num(m.group(1))
    except ValueError:
        return None
    return v if math.isfinite(v) and 0 <= v <= 300 else None


def building_extent(tags: dict) -> tuple[float, float]:
    """(min_height_m, height_m) above ground. height is always > min_height."""
    lo = parse_length_m(tags.get("min_height"))
    if lo is None:
        lv = parse_levels(tags.get("building:min_level"))
        lo = lv * LEVEL_HEIGHT_M if lv is not None else 0.0
    hi = parse_length_m(tags.get("height"))
    if hi is None:
        lv = parse_levels(tags.get("building:levels"))
        if lv is not None and lv > 0:
            hi = lv * LEVEL_HEIGHT_M
    if hi is None:
        hi = DEFAULT_HEIGHT_M
    if hi <= lo:
        # inconsistent tags (min above top, or a 0 m building): keep a visible block
        hi = lo + LEVEL_HEIGHT_M if lo > 0 else DEFAULT_HEIGHT_M
    return lo, hi


# --------------------------------------------------------------------------------------- rings


def _pt(p: object) -> tuple[float, float] | None:
    try:
        lat, lon = float(p["lat"]), float(p["lon"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)) or abs(lat) > 90 or abs(lon) > 180:
        return None
    return lat, lon


def _geometry(el: dict) -> list[tuple[float, float]]:
    g = el.get("geometry")
    if not isinstance(g, list):
        return []
    pts = [_pt(p) for p in g]
    return [p for p in pts if p is not None]


def _close_ring(pts: list[tuple[float, float]]) -> tuple[tuple[float, float], ...] | None:
    """Drop the repeated closing point and consecutive duplicates; None if not a polygon."""
    out: list[tuple[float, float]] = []
    for p in pts:
        if not out or p != out[-1]:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return tuple(out) if len(out) >= 3 else None


def _stitch(segments: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Join open way pieces end to end into rings (multipolygon outers split over several ways).
    Rings that cannot be closed are dropped."""
    rings: list[list[tuple[float, float]]] = []
    pool = [s for s in segments if len(s) >= 2]
    while pool:
        cur = list(pool.pop(0))
        progress = True
        while cur[0] != cur[-1] and progress:
            progress = False
            for i, s in enumerate(pool):
                if s[0] == cur[-1]:
                    cur += s[1:]
                elif s[-1] == cur[-1]:
                    cur += s[-2::-1]
                elif s[-1] == cur[0]:
                    cur = s[:-1] + cur
                elif s[0] == cur[0]:
                    cur = s[:0:-1] + cur
                else:
                    continue
                pool.pop(i)
                progress = True
                break
        if cur[0] == cur[-1]:
            rings.append(cur)
    return rings


def parse_buildings(body: dict | str | bytes) -> list[Footprint]:
    """Overpass ``out geom`` JSON -> footprints (outer rings only). Junk elements are skipped."""
    try:
        data = json.loads(body) if isinstance(body, (str, bytes)) else body
    except ValueError:
        return []
    if not isinstance(data, dict):
        return []
    out: list[Footprint] = []
    for el in data.get("elements") or []:
        if not isinstance(el, dict):
            continue
        tags = el.get("tags")
        if not isinstance(tags, dict) or "building" not in tags or tags.get("building") == "no":
            continue
        typ, eid = el.get("type"), el.get("id")
        lo, hi = building_extent(tags)
        if typ == "way":
            ring = _close_ring(_geometry(el))
            if ring is not None:
                out.append(Footprint(f"way/{eid}", ring, hi, lo))
        elif typ == "relation":
            segs = [
                _geometry(m)
                for m in el.get("members") or []
                if isinstance(m, dict) and m.get("type") == "way" and m.get("role") == "outer"
            ]
            for k, pts in enumerate(_stitch([s for s in segs if s])):
                ring = _close_ring(pts)
                if ring is not None:
                    out.append(Footprint(f"relation/{eid}" + (f"#{k}" if k else ""), ring, hi, lo))
    return out


# -------------------------------------------------------------------------------------- network


def build_query(lat: float, lon: float, radius_m: float) -> str:
    a = f"(around:{int(radius_m)},{lat:.6f},{lon:.6f})"
    return f'[out:json][timeout:25];\n(\n  way["building"]{a};\n  relation["building"]{a};\n);\nout geom;\n'


def fetch_buildings_raw(
    client: httpx.Client,
    lat: float,
    lon: float,
    radius_m: float,
    endpoints: Sequence[str] = OVERPASS_ENDPOINTS,
) -> str:
    """POST the query to the first Overpass endpoint that answers. Raises RuntimeError if none do."""
    query = build_query(lat, lon, radius_m)
    last: Exception | None = None
    for url in endpoints:
        try:
            r = client.post(
                url,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=httpx.Timeout(10.0, read=45.0),
            )
            r.raise_for_status()
            text = r.text
            if not isinstance(json.loads(text), dict):
                raise ValueError("not a JSON object")
            return text
        except (httpx.HTTPError, ValueError) as e:
            last = e
            log.info("overpass %s failed: %s", url, e)
    raise RuntimeError(f"all Overpass endpoints failed ({last})")
