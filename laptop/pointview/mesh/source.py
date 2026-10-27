"""OsmTerrainMeshSource: OSM buildings (Overpass) + AWS Terrarium terrain as one WorldMesh.

* ``mesh_near()`` never does I/O or heavy work: it records the position, wakes the worker thread
  and returns the most recent mesh (loaded from the disk cache at construction, so offline
  startup still shows the last model).
* A worker thread fetches buildings and terrain tiles, builds the mesh and swaps it in. It
  refetches after the user moves more than ``refetch_m`` from the last origin, with exponential
  backoff on failures.
* Disk cache in ``cache_dir``: ``mesh_cache.sqlite`` (raw Overpass responses and terrain PNGs),
  ``last_mesh.npz`` + ``last_mesh.json`` (the last built mesh).

Data: © OpenStreetMap contributors (ODbL); terrain from Mapzen/AWS Terrain Tiles.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import httpx
import numpy as np

from pointview import world
from pointview.entities import GeoPos
from pointview.interfaces import WorldMesh
from pointview.mesh import osm_buildings, terrain
from pointview.mesh.assemble import build_world_mesh
from pointview.mesh.osm_buildings import USER_AGENT, Footprint
from pointview.mesh.terrain import TerrainModel, TileKey

log = logging.getLogger(__name__)

#: Cached mesh/Overpass data older than this is still used offline but refreshed when possible.
CACHE_FRESH_S = 7 * 86400
#: A new query within this distance of a fresh cached Overpass response reuses it (politeness).
REUSE_CACHED_M = 50.0
CACHE_FILE = "mesh_cache.sqlite"
MESH_NPZ = "last_mesh.npz"
MESH_JSON = "last_mesh.json"


class _Cache:
    """SQLite cache for raw responses. Thread-safe and failure-tolerant (errors disable it)."""

    def __init__(self, path: Path):
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(path), check_same_thread=False)
            db.execute(
                "CREATE TABLE IF NOT EXISTS overpass (key TEXT PRIMARY KEY, lat REAL, lon REAL, "
                "radius REAL, fetched REAL, body TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS tiles (z INT, x INT, y INT, fetched REAL, png BLOB, "
                "PRIMARY KEY (z, x, y))"
            )
            db.commit()
            self._db = db
        except (sqlite3.Error, OSError) as e:
            log.warning("mesh cache disabled (%s)", e)

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
                log.warning("mesh cache error: %s", e)
                return []

    def put_overpass(self, lat: float, lon: float, radius_m: float, body: str) -> None:
        self._run(
            "INSERT OR REPLACE INTO overpass VALUES (?,?,?,?,?,?)",
            (f"{lat:.5f},{lon:.5f},{int(radius_m)}", lat, lon, radius_m, time.time(), body),
            write=True,
        )

    def nearest_overpass(
        self, lat: float, lon: float, radius_m: float, max_dist_m: float
    ) -> tuple[float, float, float, str] | None:
        """Nearest cached response with this radius within ``max_dist_m``:
        (query_lat, query_lon, fetched_epoch, body)."""
        best = None
        for clat, clon, fetched, body in self._run(
            "SELECT lat, lon, fetched, body FROM overpass WHERE radius = ?", (radius_m,)
        ):
            d = world.haversine_m(lat, lon, clat, clon)
            if d <= max_dist_m and (best is None or d < best[0]):
                best = (d, clat, clon, fetched, body)
        return None if best is None else best[1:]

    def get_tile(self, key: TileKey) -> bytes | None:
        rows = self._run("SELECT png FROM tiles WHERE z=? AND x=? AND y=?", key)
        return bytes(rows[0][0]) if rows else None

    def put_tile(self, key: TileKey, png: bytes) -> None:
        self._run(
            "INSERT OR REPLACE INTO tiles VALUES (?,?,?,?,?)", (*key, time.time(), png), write=True
        )


class OsmTerrainMeshSource:
    """``interfaces.MeshSource`` combining OSM buildings and Terrarium terrain (contracts §11)."""

    def __init__(
        self,
        cache_dir: Path,
        client: httpx.Client | None = None,
        radius_m: float = 800,
        refetch_m: float = 200,
        *,
        endpoints: Sequence[str] = osm_buildings.OVERPASS_ENDPOINTS,
        backoff_s: float = 5.0,
        zoom: int = terrain.DEFAULT_ZOOM,
    ):
        self.radius_m = float(radius_m)
        self.refetch_m = float(refetch_m)
        self.zoom = zoom
        self._endpoints = tuple(endpoints)
        self._backoff_s = backoff_s
        self._dir = Path(cache_dir)
        self._own_client = client is None
        self._client = client or httpx.Client(headers={"User-Agent": USER_AGENT})
        self._cache = _Cache(self._dir / CACHE_FILE)

        self._lock = threading.Lock()
        self._pos: tuple[float, float] | None = None
        self._mesh: WorldMesh | None = None
        self._terrain: TerrainModel | None = None
        self._complete = False  # last installed mesh had buildings and terrain
        self._failures = 0
        self._next_attempt = 0.0
        self.last_error: str | None = None
        self.builds = 0
        self.last_build_s = 0.0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mesh_event = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._load_disk()

    # --- MeshSource API

    def mesh_near(self, lat: float, lon: float) -> WorldMesh | None:
        with self._lock:
            self._pos = (lat, lon)
            mesh = self._mesh
        if self._needs_fetch((lat, lon)):
            self._idle.clear()
        self._ensure_thread()
        self._wake.set()
        if mesh is None:
            return None
        # A model of another place (stale cache, far from here) must not be drawn.
        if world.haversine_m(lat, lon, mesh.origin.lat, mesh.origin.lon) > self.radius_m:
            return None
        return mesh

    def ground_alt_m(self, lat: float, lon: float) -> float | None:
        t = self._terrain
        return None if t is None else t.elevation_at(lat, lon)

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

    def wait_for_mesh(self, timeout: float = 5.0) -> bool:
        """Block until some mesh (cache or network) is available. For tests/startup."""
        return self._mesh_event.wait(timeout)

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Block until the worker has nothing pending for the last reported position."""
        return self._idle.wait(timeout)

    # --- disk persistence

    def _load_disk(self) -> None:
        try:
            meta = json.loads((self._dir / MESH_JSON).read_text())
            with np.load(self._dir / MESH_NPZ) as z:
                verts, edges = z["vertices"], z["edges"]
            o = meta["origin"]
            mesh = WorldMesh(
                origin=GeoPos(float(o["lat"]), float(o["lon"]), float(o["alt_m"])),
                vertices=np.ascontiguousarray(verts, np.float32).reshape(-1, 3),
                edges=np.ascontiguousarray(edges, np.int32).reshape(-1, 2),
                source=str(meta["source"]),
                fetched_at=float(meta["fetched_at"]),
                attribution=str(meta.get("attribution", "")),
            )
            keys = [tuple(int(v) for v in k) for k in meta.get("tiles", [])]
            pngs = {k: self._cache.get_tile(k) for k in keys}
            tiles = {}
            for k, png in pngs.items():
                if png is not None:
                    with contextlib.suppress(ValueError):
                        tiles[k] = terrain.decode_terrarium(png)
            if mesh.edges.size and int(mesh.edges.max()) >= len(mesh.vertices):
                raise ValueError("edge index out of range")
        except (OSError, ValueError, KeyError, TypeError):
            return  # no usable cached mesh
        self._terrain = TerrainModel(tiles, self.zoom) if tiles else None
        self._complete = bool(meta.get("complete", True))
        self._mesh = mesh
        self._mesh_event.set()

    def _save_disk(self, mesh: WorldMesh, tiles: list[TileKey], complete: bool) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "origin": {
                    "lat": mesh.origin.lat,
                    "lon": mesh.origin.lon,
                    "alt_m": mesh.origin.alt_m,
                },
                "source": mesh.source,
                "fetched_at": mesh.fetched_at,
                "attribution": mesh.attribution,
                "radius_m": self.radius_m,
                "tiles": [list(k) for k in tiles],
                "complete": complete,
            }
            tmp_npz = self._dir / (MESH_NPZ + ".tmp")
            with open(tmp_npz, "wb") as f:
                np.savez_compressed(f, vertices=mesh.vertices, edges=mesh.edges)
            tmp_json = self._dir / (MESH_JSON + ".tmp")
            tmp_json.write_text(json.dumps(meta))
            os.replace(tmp_npz, self._dir / MESH_NPZ)
            os.replace(tmp_json, self._dir / MESH_JSON)
        except OSError as e:
            log.warning("could not save mesh cache: %s", e)

    # --- worker thread

    def _ensure_thread(self) -> None:
        if self._thread is None and not self._stop.is_set():
            self._thread = threading.Thread(target=self._run, name="mesh", daemon=True)
            self._thread.start()

    def _needs_fetch(self, pos: tuple[float, float]) -> bool:
        mesh = self._mesh
        if mesh is None or not self._complete:
            return True
        if time.time() - mesh.fetched_at > CACHE_FRESH_S:
            return True
        return world.haversine_m(*pos, mesh.origin.lat, mesh.origin.lon) > self.refetch_m

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                with self._lock:
                    pos = self._pos
                if pos is None or not self._needs_fetch(pos):
                    self._idle.set()
                    continue
                if time.monotonic() < self._next_attempt:
                    continue
                self._idle.clear()
                self._refresh(pos)
            except Exception:  # the thread must survive anything
                log.exception("mesh refresh failed")
                self._fail("internal error")

    def _fail(self, msg: str) -> None:
        self.last_error = msg
        self._failures += 1
        self._next_attempt = time.monotonic() + min(
            self._backoff_s * 2 ** (self._failures - 1), 300.0
        )
        self._idle.set()

    def _get_buildings(
        self, lat: float, lon: float
    ) -> tuple[float, float, float, list[Footprint]] | None:
        """(query_lat, query_lon, fetched_epoch, footprints) from the cache or Overpass; None if
        neither works."""
        hit = self._cache.nearest_overpass(lat, lon, self.radius_m, REUSE_CACHED_M)
        if hit is not None and time.time() - hit[2] < CACHE_FRESH_S:
            with contextlib.suppress(ValueError, TypeError):
                return (*hit[:3], osm_buildings.parse_buildings(hit[3]))
        try:
            body = osm_buildings.fetch_buildings_raw(
                self._client, lat, lon, self.radius_m, self._endpoints
            )
            fps = osm_buildings.parse_buildings(body)
            self._cache.put_overpass(lat, lon, self.radius_m, body)
            return lat, lon, time.time(), fps
        except RuntimeError as e:
            self.last_error = str(e)
        hit = self._cache.nearest_overpass(lat, lon, self.radius_m, self.radius_m / 2)
        if hit is not None:
            with contextlib.suppress(ValueError, TypeError):
                return (*hit[:3], osm_buildings.parse_buildings(hit[3]))
        return None

    def _get_terrain(self, lat: float, lon: float) -> tuple[TerrainModel, list[TileKey]] | None:
        keys = terrain.tiles_covering(lat, lon, self.radius_m + 30.0, self.zoom)
        arrays = {}
        for key in keys:
            png = self._cache.get_tile(key)
            if png is None:
                try:
                    png = terrain.fetch_tile(self._client, key, USER_AGENT)
                except (httpx.HTTPError, ValueError) as e:
                    self.last_error = f"terrain tile {key}: {e}"
                    return None
                self._cache.put_tile(key, png)
            try:
                arrays[key] = terrain.decode_terrarium(png)
            except ValueError:
                self.last_error = f"bad cached terrain tile {key}"
                return None
        return TerrainModel(arrays, self.zoom), keys

    def _refresh(self, pos: tuple[float, float]) -> None:
        t0 = time.monotonic()
        got = self._get_buildings(*pos)
        qlat, qlon = (pos[0], pos[1]) if got is None else (got[0], got[1])
        tr = self._get_terrain(qlat, qlon)
        if tr is None:
            self._fail(self.last_error or "terrain unavailable")
            return
        model, keys = tr
        fps = None if got is None else got[3]
        mesh = build_world_mesh(
            fps, model, qlat, qlon, self.radius_m, fetched_at=None if got is None else got[2]
        )
        complete = fps is not None
        cur = self._mesh
        if (
            not complete
            and cur is not None
            and world.haversine_m(*pos, cur.origin.lat, cur.origin.lon) <= self.radius_m
        ):
            # keep the existing mesh (it has buildings); retry the buildings later
            self._fail(self.last_error or "buildings unavailable")
            return
        self._save_disk(mesh, keys, complete)
        self._terrain = model
        self._complete = complete
        self._mesh = mesh  # atomic swap; readers hold the old object until they call again
        self.last_build_s = time.monotonic() - t0
        self.builds += 1
        self._mesh_event.set()
        if complete:
            self._failures = 0
            self.last_error = None
            self._idle.set()
        else:
            self._fail(self.last_error or "buildings unavailable")
