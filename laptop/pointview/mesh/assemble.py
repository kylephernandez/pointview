"""Assemble a WorldMesh from building footprints and a terrain model (pure, no I/O)."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import numpy as np

from pointview import world
from pointview.entities import GeoPos
from pointview.interfaces import WorldMesh
from pointview.mesh.osm_buildings import OSM_ATTRIBUTION, Footprint
from pointview.mesh.terrain import TERRAIN_ATTRIBUTION, TerrainModel, grid_mesh

SIMPLIFY_TOL_M = 0.5
#: Safety caps so a dense downtown cannot produce an unbounded mesh (nearest buildings win).
MAX_BUILDING_EDGES = 60_000
MAX_BUILDINGS = 6_000


def douglas_peucker(pts: np.ndarray, tol: float) -> np.ndarray:
    """Indices (sorted) of the points kept by Douglas-Peucker on an open polyline (N, 2)."""
    n = len(pts)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        rel = pts[a + 1 : b] - pts[a]
        norm = math.hypot(seg[0], seg[1])
        if norm < 1e-9:
            d = np.hypot(rel[:, 0], rel[:, 1])
        else:
            d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm
        i = int(np.argmax(d))
        if d[i] > tol:
            k = a + 1 + i
            keep[k] = True
            stack += [(a, k), (k, b)]
    return np.flatnonzero(keep)


def simplify_ring(xy: np.ndarray, tol: float = SIMPLIFY_TOL_M) -> np.ndarray:
    """Indices of the corners kept from a closed ring (first point not repeated). Never returns
    fewer than 3 points."""
    n = len(xy)
    if n <= 3:
        return np.arange(n)
    far = int(np.argmax(np.hypot(*(xy - xy[0]).T)))
    if far == 0:
        return np.arange(n)
    first = douglas_peucker(xy[: far + 1], tol)
    rest_pts = np.vstack([xy[far:], xy[:1]])
    second = douglas_peucker(rest_pts, tol)[:-1] + far  # drop the repeated start
    idx = np.unique(np.concatenate([first, second]))
    return idx if len(idx) >= 3 else np.arange(n)


def _centroid(lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    """Area centroid (planar, fine at building scale); vertex mean for degenerate rings."""
    lat0, lon0 = float(lat[0]), float(lon[0])
    k = math.cos(math.radians(lat0))
    x = (lon - lon0) * k  # translate first: absolute degrees lose precision in the cross terms
    y = lat - lat0
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-18:
        return float(lat.mean()), float(lon.mean())
    cx = ((x + x1) * cross).sum() / (6 * a)
    cy = ((y + y1) * cross).sum() / (6 * a)
    return lat0 + float(cy), lon0 + float(cx / k)


def building_edges(
    footprints: Sequence[Footprint],
    terrain: TerrainModel,
    origin: GeoPos,
    radius_m: float,
    max_edges: int = MAX_BUILDING_EDGES,
    max_buildings: int = MAX_BUILDINGS,
) -> tuple[np.ndarray, np.ndarray]:
    """Extruded building wireframes: float32 ENU vertices (N, 3) and int32 edges (M, 2).

    Each building sits on the terrain elevation at its footprint centroid and contributes bottom
    footprint edges, roof edges and vertical edges at its corners. Shared vertices are deduped."""
    if not footprints:
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int32)
    cents = np.array([_centroid(*np.asarray(f.ring, np.float64).T) for f in footprints])
    ground = terrain.elevation_many(cents[:, 0], cents[:, 1])
    ground = np.where(np.isnan(ground), origin.alt_m, ground)

    lens = np.array([len(f.ring) for f in footprints])
    lat = np.concatenate([np.asarray(f.ring, np.float64)[:, 0] for f in footprints])
    lon = np.concatenate([np.asarray(f.ring, np.float64)[:, 1] for f in footprints])
    alt = np.repeat(ground, lens)
    enu = np.asarray(world.enu_from_geodetic(origin.lat, origin.lon, origin.alt_m, lat, lon, alt))
    enu = enu.reshape(-1, 3)
    offs = np.concatenate([[0], np.cumsum(lens)])

    dist = np.array(
        [
            float(np.hypot(*enu[offs[i] : offs[i + 1], :2].mean(axis=0)))
            for i in range(len(footprints))
        ]
    )
    order = [i for i in np.argsort(dist, kind="stable") if dist[i] <= radius_m][:max_buildings]

    index: dict[tuple[int, int, int], int] = {}
    verts: list[tuple[float, float, float]] = []
    edges: list[tuple[int, int]] = []

    def vid(p: np.ndarray) -> int:
        k = (round(float(p[0]) * 1000), round(float(p[1]) * 1000), round(float(p[2]) * 1000))
        i = index.get(k)
        if i is None:
            i = index[k] = len(verts)
            verts.append((float(p[0]), float(p[1]), float(p[2])))
        return i

    for i in order:
        f = footprints[i]
        pts = enu[offs[i] : offs[i + 1]]
        keep = simplify_ring(pts[:, :2])
        pts = pts[keep]
        n = len(pts)
        if len(edges) + 3 * n > max_edges:
            break
        bot = pts.copy()
        bot[:, 2] += f.min_height_m
        top = pts.copy()
        top[:, 2] += f.height_m
        b = [vid(p) for p in bot]
        t = [vid(p) for p in top]
        for k in range(n):
            k2 = (k + 1) % n
            edges.append((b[k], b[k2]))
            edges.append((t[k], t[k2]))
            edges.append((b[k], t[k]))
    if not edges:
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int32)
    e = np.asarray(edges, np.int64)
    e = e[e[:, 0] != e[:, 1]]
    e = np.unique(np.sort(e, axis=1), axis=0)
    return np.asarray(verts, np.float32), e.astype(np.int32)


def build_world_mesh(
    footprints: Sequence[Footprint] | None,
    terrain: TerrainModel,
    lat: float,
    lon: float,
    radius_m: float = 800.0,
    fetched_at: float | None = None,
) -> WorldMesh:
    """Terrain grid + extruded buildings around (lat, lon). ``footprints=None`` builds terrain only.
    The origin is the point itself at terrain height."""
    ground = terrain.elevation_at(lat, lon)
    origin = GeoPos(lat, lon, 0.0 if ground is None else float(ground))
    gv, ge = grid_mesh(terrain, lat, lon, origin.alt_m, radius_m)
    parts_v = [gv]
    parts_e = [ge]
    names = ["terrarium"]
    if footprints is not None:
        bv, be = building_edges(footprints, terrain, origin, radius_m)
        parts_e.append(be + len(gv))
        parts_v.append(bv)
        names.insert(0, "osm-buildings")
    return WorldMesh(
        origin=origin,
        vertices=np.vstack(parts_v).astype(np.float32),
        edges=np.vstack(parts_e).astype(np.int32),
        source="+".join(names),
        fetched_at=time.time() if fetched_at is None else fetched_at,
        attribution=f"{OSM_ATTRIBUTION} · {TERRAIN_ATTRIBUTION}"
        if footprints is not None
        else TERRAIN_ATTRIBUTION,
    )
