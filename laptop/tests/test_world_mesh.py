"""Tests for world.enu_from_geodetic and world.project_mesh (contract v2.1)."""

from __future__ import annotations

import math
import time

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from pointview import world as w
from pointview.entities import GeoPos
from pointview.interfaces import WorldMesh

settings.register_profile("pointview-mesh", deadline=None, max_examples=200)
settings.load_profile("pointview-mesh")

IDENT = (1.0, 0.0, 0.0, 0.0)
DISPLAY = (640, 480)
FOV = (40.0, 30.0)
OBS = (47.6, -122.3, 50.0)


def _mesh(verts, edges, origin=None) -> WorldMesh:
    o = origin or GeoPos(*OBS)
    return WorldMesh(o, np.asarray(verts, np.float32), np.asarray(edges, np.int32), "test", 0.0)


def _proj(mesh, quat=IDENT, off=0.0, **kw):
    return w.project_mesh(mesh, *OBS, quat, off, DISPLAY, FOV, **kw)


def _target_geodetic(enu):
    """Geodetic (lat, lon, alt) whose ENU at OBS is ``enu`` (via ECEF, Bowring iteration)."""
    e = w._enu_matrix(*OBS[:2])
    x, y, z = np.array(w.ecef_from_geodetic(*OBS)) + e.T @ np.asarray(enu, dtype=float)
    e2 = w.WGS84_F * (2 - w.WGS84_F)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - e2))
    alt = 0.0
    for _ in range(10):
        n = w.WGS84_A / math.sqrt(1 - e2 * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1 - e2 * n / (n + alt)))
    return math.degrees(lat), math.degrees(lon), alt


# ----------------------------------------------------------------------------- enu


def test_enu_origin_and_axes():
    assert np.allclose(w.enu_from_geodetic(*OBS, *OBS), 0.0, atol=1e-6)
    n = w.enu_from_geodetic(*OBS, OBS[0] + 0.001, OBS[1], OBS[2])
    assert abs(n[0]) < 1e-3
    assert 110.0 < n[1] < 112.0 and abs(n[2]) < 0.01
    e = w.enu_from_geodetic(*OBS, OBS[0], OBS[1] + 0.001, OBS[2])
    assert 74.0 < e[0] < 76.0 and abs(e[1]) < 0.01
    u = w.enu_from_geodetic(*OBS, OBS[0], OBS[1], OBS[2] + 10.0)
    assert np.allclose(u, [0, 0, 10], atol=1e-6)


def test_enu_vectorized_matches_look_angles():
    rng = np.random.default_rng(1)
    lat = OBS[0] + rng.uniform(-0.05, 0.05, 20)
    lon = OBS[1] + rng.uniform(-0.05, 0.05, 20)
    alt = rng.uniform(0, 500, 20)
    enu = w.enu_from_geodetic(*OBS, lat, lon, alt)
    assert enu.shape == (20, 3)
    for i in range(20):
        yaw, pitch, rng_m = w.look_angles(*OBS, lat[i], lon[i], alt[i])
        v = w.vector_from_yaw_pitch(yaw, pitch)
        assert np.allclose(enu[i], np.array(v) * rng_m, atol=1e-4)
    # Scalars give shape (3,), 2-D input gives (a, b, 3).
    assert w.enu_from_geodetic(*OBS, lat[0], lon[0], alt[0]).shape == (3,)
    assert w.enu_from_geodetic(
        *OBS, lat.reshape(4, 5), lon.reshape(4, 5), alt.reshape(4, 5)
    ).shape == (4, 5, 3)


# ----------------------------------------------------------------------- consistency


@st.composite
def _pose_and_point(draw):
    """A head pose plus a point placed (in head-frame angles) inside the display, 10-400 m away."""
    pose = (
        draw(st.floats(0, 360)),
        draw(st.floats(-60, 60)),
        draw(st.floats(-30, 30)),
        draw(st.floats(-180, 180)),
    )
    alpha = math.radians(draw(st.floats(-17, 17)))
    beta = math.radians(draw(st.floats(-11, 11)))
    rng_m = draw(st.floats(10, 400))
    return pose, (alpha, beta, rng_m)


@given(_pose_and_point())
def test_vertex_matches_project_to_display(case):
    (yaw, pitch, roll, off), (alpha, beta, rng_m) = case
    quat = w.quat_from_yaw_pitch_roll(yaw, pitch, roll)
    # Head-frame ray (x right, y forward, z up) -> observer ENU via the inverse of world_to_head.
    hv = np.array([math.tan(alpha), 1.0, math.tan(beta)])
    hv *= rng_m / np.linalg.norm(hv)
    target = np.array(w.head_to_world(tuple(hv), quat, off))
    # A mesh origin ~1.5 km away exercises the frame-rotation correction.
    origin = GeoPos(OBS[0] + 0.012, OBS[1] - 0.015, OBS[2] + 5.0)
    p = w.enu_from_geodetic(*OBS, origin.lat, origin.lon, origin.alt_m)
    m = w._enu_matrix(*OBS[:2]) @ w._enu_matrix(origin.lat, origin.lon).T
    v = m.T @ (target - p)  # mesh-ENU vertex that sits at `target` in observer ENU
    v2 = m.T @ (target * 1.001 - p)  # tiny edge along the same ray, to survive clipping
    mesh = _mesh([v, v2], [[0, 1]], origin)
    # Reference: geodetic round trip through look_angles + project_to_display.
    ref_yaw, ref_pitch, ref_rng = w.look_angles(*OBS, *_target_geodetic(target))
    dp = w.project_to_display(ref_yaw, ref_pitch, quat, off, DISPLAY, FOV)
    assume(dp.in_front and 2.0 < dp.x < DISPLAY[0] - 2 and 2.0 < dp.y < DISPLAY[1] - 2)
    seg, dist = w.project_mesh(mesh, *OBS, quat, off, DISPLAY, FOV, max_range_m=1e5)
    assert len(seg) == 1
    # Vertex 0 is the point under test (float32 vertices -> sub-pixel tolerance).
    d0 = min(
        math.hypot(seg[0][0] - dp.x, seg[0][1] - dp.y),
        math.hypot(seg[0][2] - dp.x, seg[0][3] - dp.y),
    )
    assert d0 < 0.5
    assert abs(dist[0] - ref_rng) < 0.002 * ref_rng + 0.5


# ------------------------------------------------------------------------ behaviours


def test_behind_viewer_dropped():
    # Looking north (identity quat, offset 0): vertices to the south are behind.
    mesh = _mesh([[0, -50, 0], [10, -60, 0], [0, 50, 0], [10, 50, 5]], [[0, 1], [2, 3]])
    seg, dist = _proj(mesh)
    assert len(seg) == 1 and len(dist) == 1
    assert abs(dist[0] - math.sqrt(5**2 + 50**2 + 2.5**2)) < 0.5


def test_near_plane_clip_no_wild_coords():
    f = w.display_focal_px(DISPLAY, FOV)
    # Crosses the near plane at lateral 0.01 m: clipped end lands at x = W/2 + f*0.01/0.5.
    mesh = _mesh([[0.01, 1.0, 0.0], [0.01, -5.0, 0.0]], [[0, 1]])
    seg, _ = _proj(mesh, near_m=0.5)
    assert len(seg) == 1 and np.all(np.isfinite(seg))
    x_expect = DISPLAY[0] / 2 + f * 0.01 / 0.5
    ends_x = [seg[0, 0], seg[0, 2]]
    assert any(abs(x - x_expect) < 0.05 for x in ends_x)
    # A wide crossing edge stays finite and inside the display.
    mesh = _mesh([[3, 20, 0], [3, -20, 0]], [[0, 1]])
    seg, _ = _proj(mesh, near_m=0.5)
    assert np.all(np.isfinite(seg))
    assert (
        seg.min() >= 0 and seg[:, [0, 2]].max() <= DISPLAY[0] and seg[:, [1, 3]].max() <= DISPLAY[1]
    )


def test_edge_entirely_before_near_plane_dropped():
    mesh = _mesh([[0, 0.2, 0], [1, 0.3, 0]], [[0, 1]])
    assert len(_proj(mesh, near_m=0.5)[0]) == 0


def test_clipped_to_screen_border():
    f = w.display_focal_px(DISPLAY, FOV)
    xl = -(DISPLAY[0] / 2 + 200) / f * 10.0  # 200 px beyond the left border, at 10 m
    mesh = _mesh([[xl, 10, 0], [0, 10, 0]], [[0, 1]])
    seg, _ = _proj(mesh)
    assert len(seg) == 1
    s = seg[0]
    assert min(s[0], s[2]) == 0.0
    assert np.isclose(max(s[0], s[2]), DISPLAY[0] / 2, atol=1e-3)
    assert np.isclose(s[1], DISPLAY[1] / 2, atol=1e-3) and np.isclose(
        s[3], DISPLAY[1] / 2, atol=1e-3
    )
    # Wholly off screen: dropped. Spanning the whole screen: clipped at both borders.
    assert len(_proj(_mesh([[100, 10, 0], [120, 10, 0]], [[0, 1]]))[0]) == 0
    s = _proj(_mesh([[-100, 10, 0], [100, 10, 0]], [[0, 1]]))[0][0]
    assert sorted([s[0], s[2]]) == [0.0, float(DISPLAY[0])]
    # Diagonal through the corner regions stays inside the rectangle.
    s = _proj(_mesh([[-100, 10, -100], [100, 10, 100]], [[0, 1]]))[0]
    assert (
        (s >= 0).all() and (s[:, [0, 2]] <= DISPLAY[0]).all() and (s[:, [1, 3]] <= DISPLAY[1]).all()
    )


def test_range_cut():
    mesh = _mesh(
        [[0, 100, 0], [1, 100, 0], [0, 900, 0], [1, 900, 0], [0, 700, 0], [1, 900, 0]],
        [[0, 1], [2, 3], [4, 5]],
    )
    seg, dist = _proj(mesh, max_range_m=800.0)
    # edge 0 near, edge 1 wholly beyond (dropped), edge 2 has one end within range (kept)
    assert len(seg) == 2
    assert sorted(np.round(dist).tolist()) == [100, 800]
    assert len(_proj(mesh, max_range_m=50.0)[0]) == 0


def test_north_offset_rotates():
    # East vertex, 20 m. world_to_head(v, q, off) is the reference for which sign puts it ahead.
    mesh = _mesh([[20, 0, 0], [20, 0, 1]], [[0, 1]])
    assert len(_proj(mesh, off=0.0)[0]) == 0  # east is 90 deg to the side: not in front
    ahead_off = next(
        o for o in (90.0, -90.0) if w.world_to_head((20.0, 0.0, 0.0), IDENT, o)[1] > 19.9
    )
    other_off = -ahead_off
    assert len(_proj(mesh, off=other_off)[0]) == 0  # the opposite sign puts it behind
    seg, _ = _proj(mesh, off=ahead_off)
    assert len(seg) == 1
    f = w.display_focal_px(DISPLAY, FOV)
    s = seg[0]
    xs, ys = sorted([(s[0], s[1]), (s[2], s[3])], key=lambda p: -p[1])
    assert np.isclose(xs[0], DISPLAY[0] / 2, atol=0.01)  # bottom end dead ahead
    assert np.isclose(xs[1], DISPLAY[1] / 2, atol=0.01)
    assert np.isclose(ys[1], DISPLAY[1] / 2 - f / 20.0, atol=0.01)  # 1 m up at 20 m


def test_empty_mesh():
    seg, dist = _proj(_mesh(np.zeros((0, 3)), np.zeros((0, 2))))
    assert seg.shape == (0, 4) and dist.shape == (0,)
    assert seg.dtype == np.float32 and dist.dtype == np.float32


def test_output_dtypes():
    seg, dist = _proj(_mesh([[0, 10, 0], [1, 10, 1]], [[0, 1]]))
    assert seg.dtype == np.float32 and seg.shape == (1, 4)
    assert dist.dtype == np.float32 and dist.shape == (1,)


# ------------------------------------------------------------------------ performance


def test_performance_50k_edges():
    rng = np.random.default_rng(0)
    n = 30_000
    verts = np.column_stack(
        [rng.uniform(-800, 800, n), rng.uniform(-800, 800, n), rng.uniform(0, 30, n)]
    )
    ia = rng.integers(0, n, 50_000)
    ib = np.clip(ia + rng.integers(1, 4, 50_000), 0, n - 1)
    # Short, building-like edges.
    verts[ib] = verts[ia] + rng.normal(0, 15, (50_000, 3))
    mesh = _mesh(verts, np.column_stack([ia, ib]))
    quat = w.quat_from_yaw_pitch_roll(30.0, 5.0, 0.0)
    times = []
    seg = None
    for _ in range(30):
        t = time.perf_counter()
        seg, _ = w.project_mesh(mesh, *OBS, quat, 12.0, (1920, 1080), (46.0, 27.0))
        times.append(time.perf_counter() - t)
    med = float(np.median(times)) * 1000
    print(f"\nproject_mesh 50k edges: median {med:.2f} ms, {len(seg)} segments out")
    assert med <= 10.0
