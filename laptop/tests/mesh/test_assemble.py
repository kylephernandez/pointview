from __future__ import annotations

import math
import time

import numpy as np
import pytest

from pointview.entities import GeoPos
from pointview.mesh import assemble as A
from pointview.mesh import osm_buildings as ob
from pointview.mesh import terrain as T
from tests.mesh._helpers import LAT, LON, encode_terrarium

M_LAT = 111_320.0
M_LON = M_LAT * math.cos(math.radians(LAT))


def flat_terrain(h: float = 10.0) -> T.TerrainModel:
    return T.TerrainModel({(14, 2618, 6330): np.full((256, 256), h, np.float32)}, 14)


def square(i: int, east_m: float, north_m: float, side_m: float = 20.0, **tags) -> ob.Footprint:
    lat0, lon0 = LAT + north_m / M_LAT, LON + east_m / M_LON
    dlat, dlon = side_m / M_LAT, side_m / M_LON
    ring = ((lat0, lon0), (lat0, lon0 + dlon), (lat0 + dlat, lon0 + dlon), (lat0 + dlat, lon0))
    lo, hi = ob.building_extent({"building": "yes", **tags})
    return ob.Footprint(f"way/{i}", ring, hi, lo)


def test_encode_decode_roundtrip():
    e = np.linspace(-30, 900, 256 * 256, dtype=np.float64).reshape(256, 256)
    got = T.decode_terrarium(encode_terrarium(e))
    assert np.abs(got - e).max() <= 1 / 256 + 1e-4


def test_single_building_vertical_edges_and_base_on_terrain():
    origin = GeoPos(LAT, LON, 10.0)
    v, e = A.building_edges([square(1, 50, 30, height="9")], flat_terrain(10.0), origin, 800)
    assert v.dtype == np.float32 and e.dtype == np.int32
    assert len(v) == 8 and len(e) == 12
    zs = sorted(set(np.round(v[:, 2], 2)))
    assert zs == [pytest.approx(0.0, abs=0.01), pytest.approx(9.0, abs=0.01)]  # base == terrain
    a, b = v[e[:, 0]], v[e[:, 1]]
    vertical = (np.abs(a[:, 0] - b[:, 0]) < 1e-3) & (np.abs(a[:, 1] - b[:, 1]) < 1e-3)
    assert vertical.sum() == 4
    assert np.allclose(np.abs(a[vertical, 2] - b[vertical, 2]), 9.0, atol=0.01)
    assert np.allclose(v[:, 0].min(), 50, atol=0.5) and np.allclose(v[:, 1].min(), 30, atol=0.5)


def test_building_base_follows_terrain_at_centroid():
    # ramp rising 1 m per pixel (pixel ~ 4.7 m east-west here): base takes the centroid height
    ramp = np.tile(np.arange(256, dtype=np.float32), (256, 1))
    tm = T.TerrainModel({(14, 2618, 6330): ramp}, 14)
    origin = GeoPos(LAT, LON, float(tm.elevation_at(LAT, LON)))
    fp = square(1, 60, 0, side_m=20, height="5")
    v, _ = A.building_edges([fp], tm, origin, 800)
    clat = np.mean([p[0] for p in fp.ring])
    clon = np.mean([p[1] for p in fp.ring])
    expect = tm.elevation_at(clat, clon) - origin.alt_m
    assert expect > 5  # really uphill of the origin
    bottoms = v[np.isclose(v[:, 2], v[:, 2].min(), atol=1e-3)]
    assert len(bottoms) == 4
    assert bottoms[0, 2] == pytest.approx(expect, abs=0.05)
    assert v[:, 2].max() == pytest.approx(expect + 5, abs=0.05)


def test_min_height_lifts_the_bottom():
    origin = GeoPos(LAT, LON, 10.0)
    v, _ = A.building_edges(
        [square(1, 40, 40, min_height="12", height="20")], flat_terrain(), origin, 800
    )
    assert sorted(set(np.round(v[:, 2], 1))) == [12.0, 20.0]


def test_shared_vertices_are_deduped():
    origin = GeoPos(LAT, LON, 10.0)
    a, b = square(1, 40, 40, height="6"), square(2, 60, 40, height="6")  # share an edge
    v, e = A.building_edges([a, b], flat_terrain(), origin, 800)
    assert len(v) == 12  # 16 minus the 4 shared corners (2 on the ground, 2 on the roof)
    assert len(e) == 20  # 24 minus the shared bottom, top and two vertical edges
    assert len(np.unique(e, axis=0)) == len(e)
    assert e.min() >= 0 and e.max() < len(v)


def test_radius_and_edge_cap():
    origin = GeoPos(LAT, LON, 10.0)
    near, far = square(1, 30, 30), square(2, 1500, 0)
    v, e = A.building_edges([near, far], flat_terrain(), origin, 800)
    assert len(v) == 8
    many = [square(i, 20 + 25 * (i % 20), 20 + 25 * (i // 20)) for i in range(100)]
    _, e = A.building_edges(many, flat_terrain(), origin, 800, max_edges=120)
    assert 0 < len(e) <= 120
    v0, e0 = A.building_edges([], flat_terrain(), origin, 800)
    assert len(v0) == 0 and len(e0) == 0


def test_simplify_ring():
    # square with a midpoint on every side -> 4 corners
    xy = np.array([[0, 0], [5, 0], [10, 0], [10, 5], [10, 10], [5, 10], [0, 10], [0, 5]], float)
    assert list(A.simplify_ring(xy)) == [0, 2, 4, 6]
    # 0.2 m wobble is below tolerance
    xy = np.array([[0, 0], [5, 0.2], [10, 0], [10, 10], [0, 10]], float)
    assert list(A.simplify_ring(xy)) == [0, 2, 3, 4]
    # a real 2 m bump is kept
    xy = np.array([[0, 0], [5, -2], [10, 0], [10, 10], [0, 10]], float)
    assert list(A.simplify_ring(xy)) == [0, 1, 2, 3, 4]
    # dense circle: far fewer points, still a decent polygon
    t = np.linspace(0, 2 * math.pi, 400, endpoint=False)
    circle = np.column_stack([15 * np.cos(t), 15 * np.sin(t)])
    kept = A.simplify_ring(circle)
    assert 8 <= len(kept) <= 40
    assert len(A.simplify_ring(np.array([[0, 0], [1, 0], [0, 1]], float))) == 3


def test_dense_circle_building_is_simplified_end_to_end():
    r = 15
    t = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    ring = tuple((LAT + (60 + r * math.sin(a)) / M_LAT, LON + r * math.cos(a) / M_LON) for a in t)
    fp = ob.Footprint("way/1", ring, 8.0)
    v, e = A.building_edges([fp], flat_terrain(), GeoPos(LAT, LON, 10.0), 800)
    assert len(v) < 2 * 40 and len(e) < 3 * 40


def test_real_fixture_mesh(buildings_json, tile_pngs):
    fps = ob.parse_buildings(buildings_json)
    tm = T.tiles_from_png(tile_pngs)
    t0 = time.perf_counter()
    mesh = A.build_world_mesh(fps, tm, LAT, LON, 800, fetched_at=123.0)
    dt = time.perf_counter() - t0
    assert dt < 2.0
    assert mesh.source == "osm-buildings+terrarium" and mesh.fetched_at == 123.0
    assert "OpenStreetMap" in mesh.attribution and "Terrain" in mesh.attribution
    assert mesh.origin.lat == LAT and mesh.origin.lon == LON
    assert mesh.origin.alt_m == pytest.approx(tm.elevation_at(LAT, LON))
    assert mesh.vertices.dtype == np.float32 and mesh.vertices.shape[1] == 3
    assert mesh.edges.dtype == np.int32 and mesh.edges.shape[1] == 2
    assert np.isfinite(mesh.vertices).all()
    assert mesh.edges.min() >= 0 and mesh.edges.max() < len(mesh.vertices)
    assert np.all(mesh.edges[:, 0] != mesh.edges[:, 1])
    assert len(np.unique(mesh.edges, axis=0)) == len(mesh.edges)
    assert 1000 < len(mesh.edges) < 5000
    # the 49 buildings are built on the ground near the origin (coastal flats, a few m up)
    grid_v, _grid_e = T.grid_mesh(tm, LAT, LON, mesh.origin.alt_m, 800)
    bv = mesh.vertices[len(grid_v) :]
    assert len(bv) > 200
    assert bv[:, 2].min() > -8 and bv[:, 2].min() < 12
    assert bv[:, 2].max() < 100  # the south end of the box climbs the Presidio hill
    assert np.hypot(bv[:, 0], bv[:, 1]).max() < 600  # box is about 350 x 670 m


def test_terrain_only_mesh():
    mesh = A.build_world_mesh(None, flat_terrain(), LAT, LON, 800)
    assert mesh.source == "terrarium" and "OpenStreetMap" not in mesh.attribution
    assert len(mesh.edges) > 0
