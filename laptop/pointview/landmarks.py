"""Landmarks layer: OpenStreetMap (Overpass) features around the phone's GPS fix, as entities.

* ``LandmarkLayer`` implements ``interfaces.Layer``. ``update()`` never does I/O: HTTP runs on a
  background thread, results are cached in SQLite so a restart offline still shows landmarks, and
  ``update()`` only turns the newest feature list into world-direction entities (bearing and
  distance from the *current* position, recomputed every call).
* ``fetch_summary()`` fetches a Wikipedia REST summary (blocking, cached) for the info card.
  ``LandmarkLayer.summary_for()`` is the non-blocking variant for the render loop.

Privacy: the only data sent is an approximate position (Overpass) or an article title (Wikipedia).
Attribution is required by the licenses: ``OSM_ATTRIBUTION`` (ODbL) and ``WIKIPEDIA_ATTRIBUTION``
(CC BY-SA).
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import sqlite3
import threading
import time
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from pointview import world
from pointview.entities import Entity
from pointview.interfaces import LayerContext

log = logging.getLogger(__name__)

OSM_ATTRIBUTION = "© OpenStreetMap contributors"
WIKIPEDIA_ATTRIBUTION = "Wikipedia · CC BY-SA"
USER_AGENT = "PointView/0.1 (open-source AR labels for VITURE glasses; https://github.com/kylephernandez/pointview)"

OVERPASS_ENDPOINTS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)

#: ``tourism`` values worth labelling (hotels, info boards, picnic sites etc. are noise).
TOURISM_VALUES = (
    "attraction|museum|viewpoint|artwork|gallery|zoo|theme_park|aquarium|monument|memorial"
)
MAX_ENTITIES = 40
#: Cached Overpass results older than this are shown but also refreshed.
CACHE_FRESH_S = 7 * 86400
#: Extra pull for features with a Wikipedia article when picking the 40 to show (meters).
NOTABLE_BONUS_M = 600.0
#: Parsed features kept per query (nearest first); bounds the per-update() work.
MAX_FEATURES = 400


@dataclass(frozen=True, slots=True)
class WikiSummary:
    title: str
    extract: str
    url: str


@dataclass(frozen=True, slots=True)
class _Feature:
    id: str  # "osm:way/123"
    name: str
    lat: float
    lon: float
    osm: str  # "way/123"
    wikipedia: str | None
    category: str | None


# ------------------------------------------------------------------------------------- cache


class _Cache:
    """SQLite cache for Overpass results and Wikipedia summaries. Thread-safe, failure-tolerant."""

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
                "CREATE TABLE IF NOT EXISTS overpass (key TEXT PRIMARY KEY, lat REAL, lon REAL, "
                "radius REAL, fetched REAL, body TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS wiki (tag TEXT PRIMARY KEY, fetched REAL, body TEXT)"
            )
            db.commit()
            self._db = db
        except (sqlite3.Error, OSError) as e:
            log.warning("landmark cache disabled (%s)", e)

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
                log.warning("landmark cache error: %s", e)
                return []

    @staticmethod
    def overpass_key(lat: float, lon: float, radius_m: float) -> str:
        return f"{lat:.3f},{lon:.3f},{int(radius_m)}"

    def put_overpass(self, lat: float, lon: float, radius_m: float, body: str) -> None:
        self._run(
            "INSERT OR REPLACE INTO overpass VALUES (?,?,?,?,?,?)",
            (self.overpass_key(lat, lon, radius_m), lat, lon, radius_m, time.time(), body),
            write=True,
        )

    def nearest_overpass(
        self, lat: float, lon: float, radius_m: float, max_dist_m: float
    ) -> tuple[float, float, float, str] | None:
        """Cached result with this radius whose query point is nearest to (lat, lon) and within
        ``max_dist_m``: (center_lat, center_lon, fetched_epoch, body)."""
        best = None
        for clat, clon, fetched, body in self._run(
            "SELECT lat, lon, fetched, body FROM overpass WHERE radius = ?", (radius_m,)
        ):
            d = world.haversine_m(lat, lon, clat, clon)
            if d <= max_dist_m and (best is None or d < best[0]):
                best = (d, clat, clon, fetched, body)
        return None if best is None else best[1:]

    def get_wiki(self, tag: str) -> WikiSummary | None:
        rows = self._run("SELECT body FROM wiki WHERE tag = ?", (tag,))
        if not rows:
            return None
        try:
            d = json.loads(rows[0][0])
            return WikiSummary(d["title"], d["extract"], d["url"])
        except (ValueError, KeyError, TypeError):
            return None

    def put_wiki(self, tag: str, s: WikiSummary) -> None:
        body = json.dumps({"title": s.title, "extract": s.extract, "url": s.url})
        self._run(
            "INSERT OR REPLACE INTO wiki VALUES (?,?,?)", (tag, time.time(), body), write=True
        )


# ------------------------------------------------------------------------------------ Overpass


def build_query(lat: float, lon: float, radius_m: float) -> str:
    a = f"(around:{int(radius_m)},{lat:.6f},{lon:.6f})"
    return (
        "[out:json][timeout:25];\n(\n"
        f'  nwr{a}["name"]["tourism"~"^({TOURISM_VALUES})$"];\n'
        f'  nwr{a}["name"]["historic"];\n'
        f'  nwr{a}["name"]["amenity"="place_of_worship"];\n'
        f'  nwr{a}["name"]["natural"="peak"];\n'
        f'  nwr{a}["name"]["building"];\n'
        ");\nout center tags;\n"
    )


def _category(tags: dict) -> str | None:
    for k in ("tourism", "historic", "natural"):
        if k in tags:
            return f"{k}={tags[k]}"
    if tags.get("amenity") == "place_of_worship":
        return "amenity=place_of_worship"
    if "building" in tags:
        return "building"
    return None


def parse_overpass(body: dict | str, lat: float, lon: float) -> list[_Feature]:
    """Overpass JSON -> named features, nearest to (lat, lon) first, at most MAX_FEATURES."""
    data = json.loads(body) if isinstance(body, str) else body
    out: list[tuple[float, _Feature]] = []
    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        name = tags.get("name")
        pos = el if "lat" in el and "lon" in el else el.get("center")
        if not name or not pos or el.get("type") not in ("node", "way", "relation"):
            continue
        osm = f"{el['type']}/{el['id']}"
        f = _Feature(
            id=f"osm:{osm}",
            name=name,
            lat=float(pos["lat"]),
            lon=float(pos["lon"]),
            osm=osm,
            wikipedia=tags.get("wikipedia") or None,
            category=_category(tags),
        )
        out.append((world.haversine_m(lat, lon, f.lat, f.lon), f))
    out.sort(key=lambda p: p[0])
    return [f for _, f in out[:MAX_FEATURES]]


# ------------------------------------------------------------------------------------- layer


class LandmarkLayer:
    """Named OSM features around the phone, as ``landmark`` entities (contracts §3.2)."""

    name = "landmarks"

    def __init__(
        self,
        cache_path: Path | None,
        client: httpx.Client | None = None,
        radius_m: float = 2000,
        requery_m: float = 250,
        *,
        endpoints: Sequence[str] = OVERPASS_ENDPOINTS,
        backoff_s: float = 5.0,
        max_entities: int = MAX_ENTITIES,
    ):
        self.radius_m = float(radius_m)
        self.requery_m = float(requery_m)
        self.max_entities = max_entities
        self._endpoints = tuple(endpoints)
        self._backoff_s = backoff_s
        self._own_client = client is None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(10.0, read=40.0), headers={"User-Agent": USER_AGENT}
        )
        self._cache = _Cache(cache_path)
        _set_default_cache(self._cache)

        self._lock = threading.Lock()
        self._pos: tuple[float, float] | None = None
        self._features: tuple[_Feature, ...] = ()
        self._qpoint: tuple[float, float] | None = None
        self._failures = 0
        self._next_attempt = 0.0
        self.last_error: str | None = None
        self.queries_sent = 0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._data_event = threading.Event()
        self._summaries: dict[str, WikiSummary | None] = {}
        self._summary_pending: set[str] = set()

    # --- Layer API

    def update(self, ctx: LayerContext) -> list[Entity]:
        p = ctx.phone
        if "landmark" not in ctx.enabled or p is None or p.lat is None or p.lon is None:
            return []
        lat, lon = p.lat, p.lon
        with self._lock:
            self._pos = (lat, lon)
            feats = self._features
        self._ensure_thread()
        self._wake.set()

        scored: list[tuple[float, float, _Feature]] = []
        radius = self.radius_m * 1.25
        for f in feats:
            d = world.haversine_m(lat, lon, f.lat, f.lon)
            if d > radius:
                continue
            scored.append((d - (NOTABLE_BONUS_M if f.wikipedia else 0.0), d, f))
        scored.sort(key=lambda s: (s[0], s[2].id))
        out: list[Entity] = []
        for _, d, f in scored[: self.max_entities]:
            meta = {"osm": f.osm}
            if f.wikipedia:
                meta["wikipedia"] = f.wikipedia
            if f.category:
                meta["category"] = f.category
            out.append(
                Entity(
                    id=f.id,
                    kind="landmark",
                    label=f.name,
                    world_yaw_deg=world.bearing_deg(lat, lon, f.lat, f.lon),
                    world_pitch_deg=0.0,
                    updated_at=ctx.t,
                    distance_m=d,
                    meta=meta,
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

    @property
    def has_data(self) -> bool:
        return bool(self._features)

    def wait_for_data(self, timeout: float = 5.0) -> bool:
        """Block until a feature list (network or cache) has been loaded. For tests/startup."""
        return self._data_event.wait(timeout)

    def fetch_summary(self, wikipedia_tag: str) -> WikiSummary | None:
        """Blocking Wikipedia summary through this layer's cache and client."""
        return fetch_summary(wikipedia_tag, self._client, cache=self._cache)

    def summary_for(self, wikipedia_tag: str) -> WikiSummary | None:
        """Non-blocking: the summary if already known, else None and a background fetch starts."""
        with self._lock:
            if wikipedia_tag in self._summaries:
                return self._summaries[wikipedia_tag]
            if wikipedia_tag in self._summary_pending:
                return None
            self._summary_pending.add(wikipedia_tag)
        cached = self._cache.get_wiki(wikipedia_tag)
        if cached is not None:
            self._finish_summary(wikipedia_tag, cached)
            return cached
        threading.Thread(
            target=lambda: self._finish_summary(wikipedia_tag, self.fetch_summary(wikipedia_tag)),
            name="landmarks-wiki",
            daemon=True,
        ).start()
        return None

    def _finish_summary(self, tag: str, s: WikiSummary | None) -> None:
        with self._lock:
            self._summary_pending.discard(tag)
            if s is not None:
                self._summaries[tag] = s

    # --- background thread

    def _ensure_thread(self) -> None:
        if self._thread is None and not self._stop.is_set():
            self._thread = threading.Thread(target=self._run, name="landmarks", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                with self._lock:
                    pos, q = self._pos, self._qpoint
                if pos is None or time.monotonic() < self._next_attempt:
                    continue
                if q is not None and world.haversine_m(*pos, *q) < self.requery_m:
                    continue
                self._refresh(pos)
            except Exception:  # keep the thread alive; the layer must never crash the app
                log.exception("landmark refresh failed")
                self._fail("internal error")

    def _set_features(self, feats: list[_Feature]) -> None:
        with self._lock:
            self._features = tuple(feats)
        self._data_event.set()

    def _fail(self, msg: str) -> None:
        self.last_error = msg
        self._failures += 1
        delay = min(self._backoff_s * 2 ** (self._failures - 1), 300.0)
        self._next_attempt = time.monotonic() + delay

    def _refresh(self, pos: tuple[float, float]) -> None:
        lat, lon = pos
        # 1. disk cache first, so a restart offline shows landmarks immediately
        hit = self._cache.nearest_overpass(lat, lon, self.radius_m, self.radius_m / 2)
        if hit is not None:
            clat, clon, fetched, body = hit
            try:
                self._set_features(parse_overpass(body, lat, lon))
            except (ValueError, KeyError, TypeError):
                hit = None
            else:
                if time.time() - fetched < CACHE_FRESH_S and (
                    world.haversine_m(lat, lon, clat, clon) <= self.requery_m
                ):
                    with self._lock:
                        self._qpoint = (clat, clon)
                    return
        # 2. network
        body = self._query_overpass(lat, lon)
        if body is None:
            return  # _fail() already scheduled the retry; keep showing what we have
        try:
            feats = parse_overpass(body, lat, lon)
        except (ValueError, KeyError, TypeError) as e:
            self._fail(f"bad Overpass response: {e}")
            return
        self._cache.put_overpass(lat, lon, self.radius_m, body)
        self._set_features(feats)
        self._failures = 0
        self.last_error = None
        with self._lock:
            self._qpoint = (lat, lon)

    def _query_overpass(self, lat: float, lon: float) -> str | None:
        query = build_query(lat, lon, self.radius_m)
        retry_after = 0.0
        err = "no endpoint"
        for url in self._endpoints:
            if self._stop.is_set():
                return None
            try:
                self.queries_sent += 1
                r = self._client.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT})
            except httpx.HTTPError as e:
                err = f"{type(e).__name__} from {urllib.parse.urlparse(url).hostname}"
                continue
            if r.status_code == 200:
                try:
                    data = r.json()
                except ValueError:
                    err = "non-JSON response"
                    continue
                if not isinstance(data.get("elements"), list) or (
                    not data["elements"] and "runtime error" in str(data.get("remark", ""))
                ):
                    err = "Overpass runtime error"
                    continue
                return r.text
            err = f"HTTP {r.status_code}"
            if r.status_code in (429, 503, 504):
                with contextlib.suppress(ValueError):
                    retry_after = max(retry_after, min(float(r.headers.get("Retry-After", 0)), 120))
        self._fail(err)
        self._next_attempt = max(self._next_attempt, time.monotonic() + retry_after)
        return None


# ---------------------------------------------------------------------------------- Wikipedia

_default_cache: _Cache | None = None
_memory_cache: dict[str, WikiSummary | None] = {}


def _set_default_cache(cache: _Cache) -> None:
    global _default_cache
    _default_cache = cache


def parse_wikipedia_tag(tag: str) -> tuple[str, str] | None:
    """ "en:Golden Gate Bridge" -> ("en", "Golden Gate Bridge"). A bare title is taken as English."""
    tag = tag.strip()
    if not tag:
        return None
    if ":" in tag:
        lang, title = tag.split(":", 1)
        if re.fullmatch(r"[a-z]{2,3}(-[a-z0-9]+)*", lang.strip()) and title.strip():
            return lang.strip(), title.strip()
    return "en", tag


def summary_url(lang: str, title: str) -> str:
    return f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(
        title.replace(" ", "_"), safe=",()"
    )


def fetch_summary(
    wikipedia_tag: str,
    client: httpx.Client | None = None,
    *,
    cache: _Cache | None = None,
    retries: int = 2,
    backoff_s: float = 1.0,
) -> WikiSummary | None:
    """Wikipedia REST summary for an OSM ``wikipedia`` tag. Blocking; call off the render thread
    (or use ``LandmarkLayer.summary_for``). Cached on disk (when a LandmarkLayer exists or ``cache``
    is given) and in memory. Returns None when the article is missing or the lookup fails."""
    parsed = parse_wikipedia_tag(wikipedia_tag)
    if parsed is None:
        return None
    key = f"{parsed[0]}:{parsed[1]}"
    cache = cache or _default_cache
    if key in _memory_cache and _memory_cache[key] is not None:
        return _memory_cache[key]
    if cache is not None:
        hit = cache.get_wiki(key)
        if hit is not None:
            _memory_cache[key] = hit
            return hit

    own = client is None
    c = client or httpx.Client(timeout=10.0, headers={"User-Agent": USER_AGENT})
    try:
        for attempt in range(retries + 1):
            try:
                r = c.get(
                    summary_url(*parsed),
                    headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                    follow_redirects=True,
                )
            except httpx.HTTPError as e:
                log.info("wikipedia lookup failed: %s", type(e).__name__)
                r = None
            if r is not None and r.status_code == 200:
                try:
                    d = r.json()
                    s = WikiSummary(
                        title=str(d.get("title") or parsed[1]),
                        extract=str(d.get("extract") or ""),
                        url=str(
                            d.get("content_urls", {}).get("desktop", {}).get("page")
                            or f"https://{parsed[0]}.wikipedia.org/wiki/"
                            + urllib.parse.quote(parsed[1].replace(" ", "_"))
                        ),
                    )
                except (ValueError, AttributeError):
                    return None
                if not s.extract:
                    return None
                _memory_cache[key] = s
                if cache is not None:
                    cache.put_wiki(key, s)
                return s
            if r is not None and r.status_code == 404:
                return None
            if attempt < retries and (r is None or r.status_code in (403, 429, 500, 502, 503, 504)):
                time.sleep(backoff_s * 2**attempt)
                continue
            return None
        return None
    finally:
        if own:
            c.close()


_PAREN = re.compile(r"\s*\([^()]*\)")


def summary_lines(summary: WikiSummary, max_chars: int, max_lines: int = 2) -> tuple[str, ...]:
    """Trim a summary to at most ``max_lines`` lines of ``max_chars`` for the info card.

    Parenthetical asides (pronunciations, dates) are dropped, words are wrapped, and the last line
    ends with an ellipsis when text was cut."""
    text = _PAREN.sub("", summary.extract)
    words = text.split()
    lines: list[str] = []
    cur = ""
    i = 0
    while i < len(words) and len(lines) < max_lines:
        w = words[i]
        cand = f"{cur} {w}".strip()
        if len(cand) <= max_chars:
            cur = cand
            i += 1
        elif cur:
            lines.append(cur)
            cur = ""
        else:  # a single word longer than a line
            lines.append(w[: max_chars - 1] + "…")
            i += 1
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if i < len(words) and lines:  # truncated
        last = lines[-1]
        if not last.endswith("…"):
            if len(last) + 1 > max_chars:
                last = last[: max_chars - 1].rstrip()
            lines[-1] = last.rstrip(" .,;:") + "…"
    return tuple(lines)
