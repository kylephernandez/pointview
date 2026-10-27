"""Terrain from AWS Terrain Tiles (Terrarium PNG) and a sparse draped grid mesh (contracts §11).

Terrarium encoding: elevation_m = (R * 256 + G + B / 256) - 32768. Tiles are 256 x 256 slippy-map
``z/x/y`` PNGs at ``s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`` (no key).

Attribution: Mapzen / AWS Terrain Tiles; the data are 3DEP (NED), SRTM, GMTED2010 (courtesy of the
U.S. Geological Survey) and ETOPO1 (NOAA) and national datasets in places. See
https://github.com/tilezen/joerd/blob/master/docs/attribution.md
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Iterable, Mapping

import cv2
import httpx
import numpy as np

log = logging.getLogger(__name__)

TERRAIN_ATTRIBUTION = "Terrain: Mapzen/AWS Terrain Tiles (USGS 3DEP, SRTM, GMTED2010; NOAA ETOPO1)"
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_SIZE = 256
DEFAULT_ZOOM = 14
M_PER_DEG_LAT = 111_320.0  # local flat-earth scale; fine for the ~1 km grid

TileKey = tuple[int, int, int]  # z, x, y


# ------------------------------------------------------------------------------------ tile math


def tile_float(lat: float, lon: float, z: int) -> tuple[float, float]:
    """Fractional slippy-map tile coordinates (x, y) of a point."""
    n = 2.0**z
    lat = max(min(lat, 85.0511), -85.0511)
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile_of(lat: float, lon: float, z: int) -> tuple[int, int]:
    x, y = tile_float(lat, lon, z)
    n = 2**z
    return min(max(int(x), 0), n - 1), min(max(int(y), 0), n - 1)


def tiles_covering(lat: float, lon: float, radius_m: float, z: int = DEFAULT_ZOOM) -> list[TileKey]:
    """Every tile touched by the square of half-width ``radius_m`` (+1 pixel margin) around a point."""
    dlat = radius_m / M_PER_DEG_LAT
    dlon = radius_m / (M_PER_DEG_LAT * max(math.cos(math.radians(lat)), 0.01))
    x0, y1 = tile_of(lat - dlat, lon - dlon, z)  # south-west corner has the larger y
    x1, y0 = tile_of(lat + dlat, lon + dlon, z)
    return [(z, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def decode_terrarium(png: bytes) -> np.ndarray:
    """Terrarium PNG bytes -> float32 (256, 256) elevations in meters. Raises ValueError."""
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.shape[:2] != (TILE_SIZE, TILE_SIZE):
        raise ValueError("not a 256x256 Terrarium PNG")
    b = img[..., 0].astype(np.float32)
    g = img[..., 1].astype(np.float32)
    r = img[..., 2].astype(np.float32)
    return (r * 256.0 + g + b / 256.0 - 32768.0).astype(np.float32)


def fetch_tile(client: httpx.Client, key: TileKey, user_agent: str) -> bytes:
    z, x, y = key
    r = client.get(
        TILE_URL.format(z=z, x=x, y=y),
        headers={"User-Agent": user_agent},
        timeout=httpx.Timeout(10.0, read=30.0),
    )
    r.raise_for_status()
    decode_terrarium(r.content)  # validate before the caller caches it
    return r.content


# ------------------------------------------------------------------------------------- terrain


class TerrainModel:
    """Elevation lookup over a set of decoded tiles of one zoom level. Immutable after creation,
    so it is safe to read from any thread."""

    def __init__(self, tiles: Mapping[TileKey, np.ndarray], zoom: int = DEFAULT_ZOOM):
        self.zoom = zoom
        self._tiles = {(x, y): a for (z, x, y), a in tiles.items() if z == zoom}

    def __bool__(self) -> bool:
        return bool(self._tiles)

    @property
    def keys(self) -> list[TileKey]:
        return sorted((self.zoom, x, y) for x, y in self._tiles)

    def _pixels(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        """Elevations at global integer pixel coordinates; NaN where the tile is not loaded."""
        out = np.full(ix.shape, np.nan, np.float64)
        tx, px = np.divmod(ix, TILE_SIZE)
        ty, py = np.divmod(iy, TILE_SIZE)
        pairs = np.unique(np.stack([tx.ravel(), ty.ravel()], axis=1), axis=0)
        for a, b in pairs:
            tile = self._tiles.get((int(a), int(b)))
            if tile is None:
                continue
            m = (tx == a) & (ty == b)
            out[m] = tile[py[m], px[m]]
        return out

    def elevation_many(self, lat, lon) -> np.ndarray:
        """Bilinear elevations (m) for arrays/scalars of lat/lon; NaN outside the loaded tiles."""
        lat = np.atleast_1d(np.asarray(lat, np.float64))
        lon = np.atleast_1d(np.asarray(lon, np.float64))
        lat, lon = np.broadcast_arrays(lat, lon)
        n = 2.0**self.zoom
        latc = np.clip(lat, -85.0511, 85.0511)
        fx = (lon + 180.0) / 360.0 * n
        fy = (1.0 - np.arcsinh(np.tan(np.radians(latc))) / math.pi) / 2.0 * n
        gx = fx * TILE_SIZE - 0.5  # pixel centres sit at +0.5
        gy = fy * TILE_SIZE - 0.5
        x0 = np.floor(gx).astype(np.int64)
        y0 = np.floor(gy).astype(np.int64)
        tx = gx - x0
        ty = gy - y0
        v00 = self._pixels(x0, y0)
        v10 = self._pixels(x0 + 1, y0)
        v01 = self._pixels(x0, y0 + 1)
        v11 = self._pixels(x0 + 1, y0 + 1)
        # A missing neighbour (edge of the loaded area) falls back to the available corners.
        stack = np.stack([v00, v10, v01, v11])
        w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty])
        ok = ~np.isnan(stack)
        wsum = (w * ok).sum(axis=0)
        num = (np.where(ok, stack, 0.0) * w).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            res = np.where(wsum > 1e-9, num / wsum, np.nan)
        return res

    def elevation_at(self, lat: float, lon: float) -> float | None:
        v = float(self.elevation_many(lat, lon)[0])
        return None if math.isnan(v) else v


# ---------------------------------------------------------------------------------- grid mesh


def _axis_samples(
    fixed: float, extent: float, near_m: float, near_sp: float, far_sp: float
) -> np.ndarray:
    """Coordinates along a grid line crossing the other axis at ``fixed``: ``near_sp`` apart while
    within ``near_m`` of the user (and the line itself is that close), ``far_sp`` apart beyond."""
    far_k = np.arange(-int(extent // far_sp), int(extent // far_sp) + 1) * far_sp
    if abs(fixed) > near_m:
        return far_k
    kn = int(min(near_m, extent) // near_sp)
    near_k = np.arange(-kn, kn + 1) * near_sp
    far_k = far_k[np.abs(far_k) > near_m]
    return np.unique(np.round(np.concatenate([near_k, far_k]), 3))


def _line_coords(radius_m: float, near_m: float, near_sp: float, far_sp: float) -> np.ndarray:
    kn = int(min(near_m, radius_m) // near_sp)
    kf = int(radius_m // far_sp)
    near = np.arange(-kn, kn + 1) * near_sp
    far = np.arange(-kf, kf + 1) * far_sp
    far = far[np.abs(far) > near_m]
    return np.unique(np.round(np.concatenate([near, far]), 3))


def grid_layout(
    radius_m: float, near_m: float, near_spacing: float, far_spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Flat (E, N) vertices (float64 (N, 2)) and edges (int32 (M, 2)) of the grid, before draping.
    Lines run east-west and north-south and stop at the circle of ``radius_m``."""
    index: dict[tuple[int, int], int] = {}
    pts: list[tuple[float, float]] = []
    edges: list[tuple[int, int]] = []

    def vid(x: float, y: float) -> int:
        k = (round(x * 1000), round(y * 1000))
        i = index.get(k)
        if i is None:
            i = index[k] = len(pts)
            pts.append((x, y))
        return i

    for c in _line_coords(radius_m, near_m, near_spacing, far_spacing):
        c = float(c)
        # A dense line that is not also a sparse line stops at the near disc, so the far field
        # really is sparse.
        on_far = abs(c) > near_m or abs(c / far_spacing - round(c / far_spacing)) < 1e-6
        reach = radius_m if on_far else min(near_m, radius_m)
        extent = math.sqrt(max(reach**2 - c**2, 0.0))
        s = _axis_samples(c, extent, near_m, near_spacing, far_spacing)
        s = s[np.abs(s) <= extent + 1e-6]
        ids_ew = [vid(float(x), c) for x in s]  # east-west line at north = c
        ids_ns = [vid(c, float(y)) for y in s]  # north-south line at east = c
        edges += list(itertools.pairwise(ids_ew))
        edges += list(itertools.pairwise(ids_ns))
    if not pts:
        return np.zeros((0, 2)), np.zeros((0, 2), np.int32)
    e = np.unique(np.sort(np.asarray(edges, np.int64).reshape(-1, 2), axis=1), axis=0)
    return np.asarray(pts, np.float64), e.astype(np.int32)


def grid_mesh(
    terrain: TerrainModel,
    origin_lat: float,
    origin_lon: float,
    origin_alt_m: float,
    radius_m: float = 800.0,
    near_m: float = 300.0,
    near_spacing_m: float = 40.0,
    far_spacing_m: float = 120.0,
    max_edges: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """A sparse terrain grid draped on ``terrain``: float32 ENU vertices (N, 3) relative to the
    origin (up = elevation - origin_alt_m) and int32 edges (M, 2).

    Lines are ``near_spacing_m`` apart within ``near_m`` of the origin and ``far_spacing_m`` apart
    out to ``radius_m``. If that would exceed ``max_edges`` (large radius), spacings are widened
    until it fits, so the edge count is always bounded."""
    ns, fs = near_spacing_m, far_spacing_m
    for _ in range(20):
        flat, edges = grid_layout(radius_m, near_m, ns, fs)
        if len(edges) <= max_edges:
            break
        ns, fs = ns * 1.25, fs * 1.25
    if len(flat) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int32)
    m_per_deg_lon = M_PER_DEG_LAT * math.cos(math.radians(origin_lat))
    lat = origin_lat + flat[:, 1] / M_PER_DEG_LAT
    lon = origin_lon + flat[:, 0] / m_per_deg_lon
    elev = terrain.elevation_many(lat, lon)
    ok = ~np.isnan(elev)
    fill = float(elev[ok].mean()) if ok.any() else origin_alt_m
    elev = np.where(ok, elev, fill)
    verts = np.column_stack([flat[:, 0], flat[:, 1], elev - origin_alt_m]).astype(np.float32)
    return verts, edges


def tiles_from_png(pngs: Mapping[TileKey, bytes], zoom: int = DEFAULT_ZOOM) -> TerrainModel:
    return TerrainModel({k: decode_terrarium(v) for k, v in pngs.items()}, zoom)


def missing_tiles(model: TerrainModel | None, wanted: Iterable[TileKey]) -> list[TileKey]:
    have = set(model.keys) if model else set()
    return [k for k in wanted if k not in have]
