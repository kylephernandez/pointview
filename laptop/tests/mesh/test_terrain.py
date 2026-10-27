from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from pointview.mesh import terrain as T
from tests.mesh._helpers import LAT, LON


def test_decode_formula():
    img = np.zeros((256, 256, 3), np.uint8)
    img[0, 0] = (0, 0, 128)  # BGR: R=128 -> 0 m
    img[0, 1] = (0, 5, 128)  # +5 m
    img[0, 2] = (128, 0, 128)  # +0.5 m
    img[0, 3] = (0, 0, 127)  # -256 m
    img[0, 4] = (64, 250, 129)  # 256 + 250 + 0.25 m
    _ok, buf = cv2.imencode(".png", img)
    e = T.decode_terrarium(buf.tobytes())
    assert e.dtype == np.float32 and e.shape == (256, 256)
    assert list(e[0, :5]) == [0.0, 5.0, 0.5, -256.0, 506.25]
    assert e[1, 0] == -32768.0


def test_decode_rejects_junk():
    with pytest.raises(ValueError):
        T.decode_terrarium(b"not a png")
    _ok, buf = cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError):
        T.decode_terrarium(buf.tobytes())


def test_tile_math():
    assert T.tile_of(LAT, LON, 14) == (2618, 6330)
    assert T.tile_of(LAT, LON, 13) == (1309, 3165)
    assert T.tile_of(0.0, 0.0, 1) == (1, 1)
    fx, fy = T.tile_float(LAT, LON, 14)
    assert (int(fx), int(fy)) == (2618, 6330) and 0.34 < fx % 1 < 0.36
    keys = T.tiles_covering(LAT, LON, 800, 14)
    assert set(keys) == {(14, 2617, 6330), (14, 2617, 6331), (14, 2618, 6330), (14, 2618, 6331)}
    assert len(T.tiles_covering(LAT, LON, 50, 14)) <= 2


def test_real_tiles_known_points(tile_pngs):
    m = T.tiles_from_png(tile_pngs)
    # The mock location is on Crissy Field, a few meters above sea level. Recorded value 3.67 m
    # (USGS NED via Terrarium z14), consistent with the ~3 m mapped elevation there.
    h = m.elevation_at(LAT, LON)
    assert h is not None and 2.0 < h < 6.0
    assert h == pytest.approx(3.665, abs=0.05)
    # San Francisco Bay: Terrarium carries bathymetry (water is below 0); by default the model
    # clamps to the water surface, which is what you see through the glasses.
    assert m.elevation_at(37.8100, -122.4600) == 0.0
    raw = T.TerrainModel(
        m._tiles and {(m.zoom, x, y): a for (x, y), a in m._tiles.items()}, m.zoom, floor_m=None
    )
    assert raw.elevation_at(37.8100, -122.4600) < -5
    # Presidio hills to the south: tens of meters up.
    assert 50 < m.elevation_at(37.7985, -122.4640) < 100
    # Outside the four tiles: unknown.
    assert m.elevation_at(37.9, -122.468) is None
    assert m.elevation_at(0.0, 0.0) is None


def test_bilinear_on_a_known_ramp():
    # elevation = 2 * global pixel x (m), so bilinear must be exactly linear in longitude
    z, x, y = 14, 2618, 6330
    cols = np.arange(256, dtype=np.float32)
    tile = np.tile(cols * 2.0, (256, 1))
    m = T.TerrainModel({(z, x, y): tile}, z)
    _fx, fy = T.tile_float(LAT, LON, z)
    lon_at = lambda px: (px / 256.0 + x) / 2**z * 360.0 - 180.0  # noqa: E731
    # pixel centre 100 holds 200 m; halfway to 101 is 201 m
    lat_mid = 37.8060  # inside tile row range, value independent of latitude
    assert m.elevation_at(lat_mid, lon_at(100.5)) == pytest.approx(200.0, abs=1e-3)
    assert m.elevation_at(lat_mid, lon_at(101.0)) == pytest.approx(201.0, abs=1e-3)
    assert m.elevation_at(lat_mid, lon_at(101.25)) == pytest.approx(201.5, abs=1e-3)
    assert 0 < fy % 1 < 1


def test_bilinear_is_continuous_across_tiles():
    z = 14
    left = np.full((256, 256), 10.0, np.float32)
    right = np.full((256, 256), 20.0, np.float32)
    m = T.TerrainModel({(z, 2617, 6330): left, (z, 2618, 6330): right}, z)
    lon_edge = 2618 / 2**z * 360.0 - 180.0  # border between the tiles
    lat = 37.806
    vals = [m.elevation_at(lat, lon_edge + d) for d in (-3e-5, -1e-6, 0.0, 1e-6, 3e-5)]
    assert vals == sorted(vals)  # monotone, no jump
    assert vals[2] == pytest.approx(15.0, abs=0.1)
    assert vals[0] < 15 < vals[-1]


def test_elevation_many_vectorised_nan_outside():
    z = 14
    m = T.TerrainModel({(z, 2618, 6330): np.full((256, 256), 7.0, np.float32)}, z)
    out = m.elevation_many(np.array([LAT, 10.0]), np.array([LON, 10.0]))
    assert out[0] == pytest.approx(7.0) and math.isnan(out[1])


# ------------------------------------------------------------------------------------- grid


def test_grid_layout_spacing_and_bounds():
    verts, edges = T.grid_layout(800, 300, 40, 120)
    assert len(edges) < 2000
    assert edges.dtype == np.int32 and edges.min() >= 0 and edges.max() < len(verts)
    assert len(np.unique(edges, axis=0)) == len(edges)  # no duplicate edges
    r = np.hypot(verts[:, 0], verts[:, 1])
    assert r.max() <= 800 + 1e-6
    d = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    assert d.max() <= 120 + 1e-6 and d.min() >= 40 - 1e-6
    # all edges axis-aligned
    delta = np.abs(verts[edges[:, 0]] - verts[edges[:, 1]])
    assert np.all((delta[:, 0] < 1e-9) | (delta[:, 1] < 1e-9))
    # near field is the dense 40 m grid, far field is sparser: edge density per area
    mid = np.hypot(*((verts[edges[:, 0]] + verts[edges[:, 1]]) / 2).T)
    near_len = d[mid < 300].sum() / (math.pi * 300**2)
    far_len = d[mid >= 300].sum() / (math.pi * (800**2 - 300**2))
    assert near_len > 2 * far_len
    # lines near the user are ~40 m apart
    ys = np.unique(np.round(verts[np.abs(verts[:, 0]) < 1, 1]))
    near_ys = ys[np.abs(ys) <= 280]
    assert np.all(np.diff(near_ys) == 40)


def test_grid_mesh_drapes_on_terrain_and_is_bounded(tile_pngs):
    m = T.tiles_from_png(tile_pngs)
    o = m.elevation_at(LAT, LON)
    v, e = T.grid_mesh(m, LAT, LON, o, 800)
    assert v.dtype == np.float32 and e.dtype == np.int32
    assert v.shape[1] == 3 and e.shape[1] == 2 and e.max() < len(v)
    assert 300 < len(e) < 2000
    # origin vertex sits at terrain height (up == 0)
    at_origin = np.flatnonzero((v[:, 0] == 0) & (v[:, 1] == 0))
    assert len(at_origin) == 1 and abs(v[at_origin[0], 2]) < 0.01
    # every vertex matches the terrain model at its own ENU position
    mlon = 111_320.0 * math.cos(math.radians(LAT))
    lat = LAT + v[:, 1].astype(np.float64) / 111_320.0
    lon = LON + v[:, 0].astype(np.float64) / mlon
    exp = m.elevation_many(lat, lon) - o
    ok = ~np.isnan(exp)
    assert np.allclose(v[ok, 2], exp[ok], atol=1e-3)
    assert v[:, 2].max() - v[:, 2].min() > 20  # the area has real relief (hill and bay)


def test_grid_edge_cap_widens_spacing():
    m = T.TerrainModel({}, 14)
    _v, e = T.grid_mesh(m, LAT, LON, 0.0, radius_m=6000, max_edges=1500)
    assert 0 < len(e) <= 1500
    v2, e2 = T.grid_mesh(m, LAT, LON, 0.0, radius_m=800)
    assert len(e2) > 0 and float(v2[:, 2].max()) == 0.0  # no tiles: flat at the origin height


def test_zero_radius_grid():
    _v, e = T.grid_mesh(T.TerrainModel({}, 14), LAT, LON, 0.0, radius_m=0)
    assert len(e) == 0
