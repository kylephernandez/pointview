"""Tests for pointview.world: hand-computed values, hypothesis properties, fixture cross-check."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from pointview import world as w
from pointview.entities import Entity
from pointview.interfaces import Alignment, CameraIntrinsics

# The fixture generator has its own, independent projection code. Cross-check against it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import make_fixtures as gen

settings.register_profile("pointview", deadline=None, max_examples=300)
settings.load_profile("pointview")

IDENT = (1.0, 0.0, 0.0, 0.0)
DISPLAY = (1920, 1080)
FOV = (46.0, 27.0)

# ------------------------------------------------------------------------------ strategies

finite = dict(allow_nan=False, allow_infinity=False)
yaws = st.floats(-720, 720, **finite)
yaws360 = st.floats(0, 359.999, **finite)
pitches = st.floats(-89.0, 89.0, **finite)
rolls = st.floats(-180, 180, **finite)
offsets = st.floats(-400, 400, **finite)
unit_s = st.floats(0, 1, **finite)


@st.composite
def quats(draw):
    q = tuple(draw(st.floats(-1, 1, **finite)) for _ in range(4))
    assume(sum(c * c for c in q) > 1e-3)
    return w.quat_normalize(q)


@st.composite
def vecs(draw, lo=-10.0, hi=10.0):
    return tuple(draw(st.floats(lo, hi, **finite)) for _ in range(3))


@st.composite
def unit_vecs(draw):
    v = draw(vecs(-1, 1))
    n = math.sqrt(sum(c * c for c in v))
    assume(n > 1e-2)
    return tuple(c / n for c in v)


def approx_vec(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


def same_rotation(a, b, tol=1e-9):
    return approx_vec(a, b, tol) or approx_vec(a, tuple(-c for c in b), tol)


def angle_eq(a, b, tol=1e-6):
    return abs(w.wrap_deg_signed(a - b)) <= tol


def ent(i, yaw, pitch=0.0):
    return Entity(
        id=i, kind="vehicle", label="car", world_yaw_deg=yaw, world_pitch_deg=pitch, updated_at=0.0
    )


# --------------------------------------------------------------------------------- quaternions


def test_quat_basic_values():
    assert w.quat_normalize((2, 0, 0, 0)) == IDENT
    assert w.quat_normalize((0, 0, 0, 0)) == IDENT
    assert w.quat_conj((1, 2, 3, 4)) == (1, -2, -3, -4)
    # i * j = k
    assert w.quat_mul((0, 1, 0, 0), (0, 0, 1, 0)) == (0, 0, 0, 1)
    q = w.quat_from_axis_angle((0, 0, 1), math.pi / 2)
    assert approx_vec(q, (math.sqrt(0.5), 0, 0, math.sqrt(0.5)))
    assert w.quat_from_axis_angle((0, 0, 0), 1.0) == IDENT
    # Right-handed 90 deg about +z: x -> y
    assert approx_vec(w.quat_rotate(q, (1, 0, 0)), (0, 1, 0))
    # axis need not be unit
    assert approx_vec(w.quat_from_axis_angle((0, 0, 5), math.pi / 2), q)


def test_quat_rotate_matches_matrix():
    q = w.quat_normalize((0.3, -0.5, 0.2, 0.7))
    v = (1.0, -2.0, 0.5)
    expect = gen.rotmat(q) @ np.array(v)
    assert approx_vec(w.quat_rotate(q, v), expect)


@given(quats(), vecs())
def test_rotate_preserves_length(q, v):
    r = w.quat_rotate(q, v)
    assert math.isclose(math.hypot(*r), math.hypot(*v), rel_tol=1e-9, abs_tol=1e-9)


@given(quats(), quats(), vecs())
def test_mul_is_composition(a, b, v):
    assert approx_vec(
        w.quat_rotate(w.quat_mul(a, b), v), w.quat_rotate(a, w.quat_rotate(b, v)), 1e-8
    )


@given(quats(), quats(), quats())
def test_mul_associative(a, b, c):
    assert approx_vec(w.quat_mul(w.quat_mul(a, b), c), w.quat_mul(a, w.quat_mul(b, c)), 1e-12)


@given(quats())
def test_mul_inverse(q):
    assert approx_vec(w.quat_mul(q, w.quat_conj(q)), IDENT, 1e-12)
    assert approx_vec(w.quat_mul(w.quat_conj(q), q), IDENT, 1e-12)


@given(quats(), vecs())
def test_conj_inverts_rotation(q, v):
    assert approx_vec(w.quat_rotate(w.quat_conj(q), w.quat_rotate(q, v)), v, 1e-8)


def test_slerp_hand_values():
    a = IDENT
    b = w.quat_from_axis_angle((0, 0, 1), math.radians(90))
    mid = w.quat_slerp(a, b, 0.5)
    assert approx_vec(mid, w.quat_from_axis_angle((0, 0, 1), math.radians(45)))
    # Shortest path: -b is the same rotation, result must not take the long way round.
    mid_neg = w.quat_slerp(a, tuple(-c for c in b), 0.5)
    assert same_rotation(mid_neg, mid)
    # Identical inputs: no 0/0
    assert approx_vec(w.quat_slerp(a, a, 0.3), a)
    # Extrapolation continues along the arc
    assert approx_vec(w.quat_slerp(a, b, 2.0), w.quat_from_axis_angle((0, 0, 1), math.pi))


@given(quats(), quats(), unit_s)
@example((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0), 0.5)  # 180° apart: ambiguous midpoint
def test_slerp_properties(a, b, s):
    r = w.quat_slerp(a, b, s)
    assert math.isclose(math.sqrt(sum(c * c for c in r)), 1.0, abs_tol=1e-9)
    assert same_rotation(w.quat_slerp(a, b, 0.0), a, 1e-9)
    assert same_rotation(w.quat_slerp(a, b, 1.0), b, 1e-9)
    # Shortest path: neither endpoint sign matters, except for rotations 180° apart (a·b = 0),
    # where the two great-circle paths are equally short and the midpoint is ambiguous.
    nb = tuple(-c for c in b)
    if abs(sum(p * q for p, q in zip(a, b, strict=True))) > 1e-6:
        assert same_rotation(w.quat_slerp(a, nb, s), r, 1e-6)

    # The rotation angle from a to r never exceeds the angle from a to b.
    def ang(x, y):
        d = abs(sum(p * q for p, q in zip(x, y, strict=True)))
        return 2 * math.acos(min(1.0, d))

    assert ang(a, r) <= ang(a, b) + 1e-6


def test_yaw_pitch_roll_hand_values():
    # yaw 90 (east): forward +y -> +x ; clockwise from above
    q = w.quat_from_yaw_pitch_roll(90, 0)
    assert approx_vec(w.quat_rotate(q, (0, 1, 0)), (1, 0, 0))
    assert approx_vec(w.quat_rotate(q, (1, 0, 0)), (0, -1, 0))  # right now points south
    # pitch up 30
    q = w.quat_from_yaw_pitch_roll(0, 30)
    assert approx_vec(w.quat_rotate(q, (0, 1, 0)), (0, math.cos(math.radians(30)), 0.5))
    # pitch is about the NEW x: yaw 90 then pitch 90 -> straight up, right axis stays -y
    q = w.quat_from_yaw_pitch_roll(90, 90)
    assert approx_vec(w.quat_rotate(q, (0, 1, 0)), (0, 0, 1))
    assert approx_vec(w.quat_rotate(q, (1, 0, 0)), (0, -1, 0))
    # roll clockwise as seen by the wearer: head's up (+z) tilts toward the right (+x)
    q = w.quat_from_yaw_pitch_roll(0, 0, 90)
    assert approx_vec(w.quat_rotate(q, (0, 0, 1)), (1, 0, 0))
    assert approx_vec(w.quat_rotate(q, (0, 1, 0)), (0, 1, 0))  # forward unchanged by roll
    assert w.quat_from_yaw_pitch_roll(0, 0) == IDENT


@given(yaws, pitches, rolls)
def test_yaw_pitch_roll_forward_points_at_yaw_pitch(yaw, pitch, roll):
    q = w.quat_from_yaw_pitch_roll(yaw, pitch, roll)
    assert math.isclose(sum(c * c for c in q), 1.0, abs_tol=1e-12)
    fwd = w.quat_rotate(q, (0, 1, 0))
    assert approx_vec(fwd, w.vector_from_yaw_pitch(yaw, pitch), 1e-9)


# ---------------------------------------------------------------------------------- directions


@pytest.mark.parametrize(
    "a,expect",
    [(0, 0), (360, 0), (-1, 359), (725, 5), (359.5, 359.5), (-720, 0), (-1e-20, 0.0)],
)
def test_wrap_deg(a, expect):
    r = w.wrap_deg(a)
    assert 0.0 <= r < 360.0
    assert math.isclose(r, expect, abs_tol=1e-9)


@pytest.mark.parametrize("a,expect", [(0, 0), (180, -180), (-180, -180), (190, -170), (-190, 170)])
def test_wrap_deg_signed(a, expect):
    assert w.wrap_deg_signed(a) == pytest.approx(expect)


@given(st.floats(-1e6, 1e6, **finite))
def test_wrap_ranges(a):
    assert 0.0 <= w.wrap_deg(a) < 360.0
    assert -180.0 <= w.wrap_deg_signed(a) < 180.0
    assert angle_eq(w.wrap_deg(a), a, 1e-6)
    assert angle_eq(w.wrap_deg_signed(a), a, 1e-6)


def test_vector_from_yaw_pitch_hand_values():
    assert approx_vec(w.vector_from_yaw_pitch(0, 0), (0, 1, 0))
    assert approx_vec(w.vector_from_yaw_pitch(90, 0), (1, 0, 0))
    assert approx_vec(w.vector_from_yaw_pitch(180, 0), (0, -1, 0))
    assert approx_vec(w.vector_from_yaw_pitch(270, 0), (-1, 0, 0))
    assert approx_vec(w.vector_from_yaw_pitch(123, 90), (0, 0, 1))
    assert approx_vec(w.vector_from_yaw_pitch(0, -90), (0, 0, -1))


def test_yaw_pitch_from_vector_hand_values():
    assert w.yaw_pitch_from_vector((0, 1, 0)) == (0.0, 0.0)
    assert w.yaw_pitch_from_vector((1, 0, 0)) == pytest.approx((90.0, 0.0))
    assert w.yaw_pitch_from_vector((-1, 0, 0)) == pytest.approx((270.0, 0.0))
    assert w.yaw_pitch_from_vector((0, -1, 0))[0] == pytest.approx(180.0)
    assert w.yaw_pitch_from_vector((0, 0, 1)) == (0.0, 90.0)
    assert w.yaw_pitch_from_vector((0, 0, -3)) == (0.0, -90.0)
    # unnormalized input
    assert w.yaw_pitch_from_vector((0, 5, 5)) == pytest.approx((0.0, 45.0))
    # tiny negative x just west of north wraps to ~360, never 360.0 itself
    yaw, _ = w.yaw_pitch_from_vector((-1e-18, 1, 0))
    assert 0.0 <= yaw < 360.0


@given(yaws360, st.floats(-90, 90, **finite))
def test_vector_roundtrip(yaw, pitch):
    v = w.vector_from_yaw_pitch(yaw, pitch)
    assert math.isclose(math.hypot(*v), 1.0, abs_tol=1e-12)
    y2, p2 = w.yaw_pitch_from_vector(v)
    assert 0.0 <= y2 < 360.0
    assert p2 == pytest.approx(pitch, abs=1e-6)
    if abs(pitch) < 89.9:
        assert angle_eq(y2, yaw, 1e-6)


@given(unit_vecs())
def test_vector_roundtrip_from_vector(v):
    y, p = w.yaw_pitch_from_vector(v)
    assert approx_vec(w.vector_from_yaw_pitch(y, p), v, 1e-9)


def test_angular_distance_hand_values():
    assert w.angular_distance_deg(0, 0, 0, 0) == 0.0
    assert w.angular_distance_deg(0, 0, 90, 0) == pytest.approx(90)
    assert w.angular_distance_deg(350, 0, 10, 0) == pytest.approx(20)  # wrap-around
    assert w.angular_distance_deg(0, 0, 180, 0) == pytest.approx(180)
    assert w.angular_distance_deg(0, 90, 123, 90) == pytest.approx(0, abs=1e-9)  # same zenith
    assert w.angular_distance_deg(0, 45, 180, 45) == pytest.approx(90)  # over the pole
    assert w.angular_distance_deg(10, 0, 10, 30) == pytest.approx(30)
    assert w.angular_distance_deg(0, 0, 0.001, 0) == pytest.approx(0.001, rel=1e-6)


@given(yaws, pitches, yaws, pitches)
def test_angular_distance_metric(y1, p1, y2, p2):
    d = w.angular_distance_deg(y1, p1, y2, p2)
    assert 0.0 <= d <= 180.0
    assert d == pytest.approx(w.angular_distance_deg(y2, p2, y1, p1), abs=1e-9)
    assert w.angular_distance_deg(y1 + 360, p1, y1, p1) == pytest.approx(0, abs=1e-6)


# ------------------------------------------------------------------------------------- geo


def test_bearing_cardinal():
    assert w.bearing_deg(0, 0, 10, 0) == pytest.approx(0)
    assert w.bearing_deg(0, 0, 0, 10) == pytest.approx(90)
    assert w.bearing_deg(0, 0, -10, 0) == pytest.approx(180)
    assert w.bearing_deg(0, 0, 0, -10) == pytest.approx(270)
    assert w.bearing_deg(10, 20, 10, 20) == 0.0  # same point: defined, in range
    assert w.bearing_deg(89, 0, 90, 0) == pytest.approx(0)


def test_bearing_known_cities():
    london = (51.5074, -0.1278)
    paris = (48.8566, 2.3522)
    nyc = (40.7128, -74.0060)
    la = (34.0522, -118.2437)
    sydney = (-33.8688, 151.2093)
    assert w.bearing_deg(*london, *paris) == pytest.approx(148.1, abs=0.2)
    assert w.bearing_deg(*paris, *london) == pytest.approx(330.0, abs=0.2)
    assert w.bearing_deg(*nyc, *la) == pytest.approx(273.7, abs=0.3)
    # across the antimeridian: Auckland -> Honolulu goes east-north-east
    assert w.bearing_deg(-36.85, 174.76, 21.31, -157.86) == pytest.approx(28.57, abs=0.05)
    assert 0 <= w.bearing_deg(*sydney, *london) < 360


def test_haversine_known():
    one_deg = 2 * math.pi * 6_371_000 / 360
    assert w.haversine_m(0, 0, 0, 1) == pytest.approx(one_deg, rel=1e-9)
    assert w.haversine_m(0, 0, 1, 0) == pytest.approx(one_deg, rel=1e-9)
    assert w.haversine_m(10, 10, 10, 10) == 0.0
    assert w.haversine_m(0, 0, 0, 180) == pytest.approx(math.pi * 6_371_000, rel=1e-9)
    assert w.haversine_m(51.5074, -0.1278, 48.8566, 2.3522) == pytest.approx(343_550, rel=2e-3)
    assert w.haversine_m(40.7128, -74.0060, 34.0522, -118.2437) == pytest.approx(
        3_936_000, rel=3e-3
    )


@given(
    st.floats(-89, 89, **finite),
    st.floats(-180, 180, **finite),
    st.floats(-89, 89, **finite),
    st.floats(-180, 180, **finite),
)
def test_geo_properties(la1, lo1, la2, lo2):
    d = w.haversine_m(la1, lo1, la2, lo2)
    assert d == pytest.approx(w.haversine_m(la2, lo2, la1, lo1), abs=1e-6)
    assert 0 <= d <= math.pi * 6_371_000 + 1
    assert 0.0 <= w.bearing_deg(la1, lo1, la2, lo2) < 360.0


@given(
    st.floats(-60, 60, **finite),
    st.floats(-170, 170, **finite),
    st.floats(0, 359, **finite),
    st.floats(1_000, 100_000, **finite),
)
def test_bearing_matches_destination_point(lat, lon, brg, dist):
    """Walk dist along a bearing with the spherical destination formula; bearing_deg recovers it."""
    r = 6_371_000.0
    dd = dist / r
    p1, l1, th = math.radians(lat), math.radians(lon), math.radians(brg)
    p2 = math.asin(math.sin(p1) * math.cos(dd) + math.cos(p1) * math.sin(dd) * math.cos(th))
    l2 = l1 + math.atan2(
        math.sin(th) * math.sin(dd) * math.cos(p1), math.cos(dd) - math.sin(p1) * math.sin(p2)
    )
    lat2, lon2 = math.degrees(p2), math.degrees(l2)
    assert w.haversine_m(lat, lon, lat2, lon2) == pytest.approx(dist, rel=1e-6)
    assert angle_eq(w.bearing_deg(lat, lon, lat2, lon2), brg, 1e-4)


# ------------------------------------------------------------------------------------ camera


@pytest.fixture
def intr():
    return CameraIntrinsics(800.0, 700.0, 480.0, 270.0, 960, 540)


def test_pixel_principal_point_is_forward(intr):
    assert w.pixel_to_camera_ray(480, 270, intr) == (0.0, 0.0, 1.0)
    assert w.camera_angles_deg(480, 270, intr) == (0.0, 0.0)
    assert w.pixel_to_world(480, 270, intr, IDENT, 0.0, IDENT) == (0.0, 0.0)


def test_camera_angles_hand_values(intr):
    a, b = w.camera_angles_deg(480 + 800, 270 + 700, intr)
    assert a == pytest.approx(45.0)
    assert b == pytest.approx(45.0)  # down-positive
    a, b = w.camera_angles_deg(480 - 800 * math.tan(math.radians(30)), 270 - 700, intr)
    assert a == pytest.approx(-30.0)
    assert b == pytest.approx(-45.0)


def test_pixel_to_camera_ray_hand_values(intr):
    x, y, z = w.pixel_to_camera_ray(480 + 800, 270, intr)
    assert (x, y, z) == pytest.approx((math.sqrt(0.5), 0.0, math.sqrt(0.5)))
    x, y, z = w.pixel_to_camera_ray(480, 270 + 700, intr)
    assert (x, y, z) == pytest.approx((0.0, math.sqrt(0.5), math.sqrt(0.5)))  # image down = +y


@given(st.floats(0, 960, **finite), st.floats(0, 540, **finite))
def test_camera_ray_unit_and_consistent(u, v):
    i = CameraIntrinsics(800.0, 700.0, 480.0, 270.0, 960, 540)
    r = w.pixel_to_camera_ray(u, v, i)
    assert math.isclose(math.hypot(*r), 1.0, abs_tol=1e-12)
    assert r[2] > 0
    a, b = w.camera_angles_deg(u, v, i)
    assert math.tan(math.radians(a)) == pytest.approx(r[0] / r[2], abs=1e-9)
    assert math.tan(math.radians(b)) == pytest.approx(r[1] / r[2], abs=1e-9)


def test_distortion_is_applied():
    import cv2

    dist = (-0.3, 0.1, 0.001, -0.0005, -0.02)
    base = CameraIntrinsics(800.0, 800.0, 480.0, 270.0, 960, 540)
    d = CameraIntrinsics(800.0, 800.0, 480.0, 270.0, 960, 540, dist)
    # Distort a known ray with OpenCV, then undistort through world.py: must recover the ray.
    ray = np.array([[0.4, -0.25, 1.0]])
    k = np.array([[800.0, 0, 480], [0, 800.0, 270], [0, 0, 1]])
    px, _ = cv2.projectPoints(ray, np.zeros(3), np.zeros(3), k, np.array(dist))
    u, v = px[0, 0]
    got = w.pixel_to_camera_ray(u, v, d)
    want = ray[0] / np.linalg.norm(ray[0])
    assert approx_vec(got, want, 1e-6)
    # And it differs from the plain pinhole result.
    assert not approx_vec(w.pixel_to_camera_ray(u, v, base), want, 1e-3)
    a, b = w.camera_angles_deg(u, v, d)
    assert math.tan(math.radians(a)) == pytest.approx(0.4, abs=1e-6)
    assert math.tan(math.radians(b)) == pytest.approx(-0.25, abs=1e-6)
    # principal point is a fixed point
    assert w.pixel_to_camera_ray(480, 270, d) == pytest.approx((0, 0, 1))


def test_camera_to_head_axes():
    # camera: x right, y down, z forward  ->  head: x right, y forward, z up
    assert w.camera_to_head((1, 0, 0), IDENT) == (1, 0, 0)
    assert w.camera_to_head((0, 0, 1), IDENT) == (0, 1, 0)
    assert w.camera_to_head((0, 1, 0), IDENT) == (0, 0, -1)
    # a camera mounted 10 deg to the left of the display axis: camera-forward is head yaw -10
    q = w.quat_from_yaw_pitch_roll(-10, 0)
    h = w.camera_to_head((0, 0, 1), q)
    assert w.yaw_pitch_from_vector(h) == pytest.approx((350.0, 0.0))


def test_head_to_world_hand_values():
    # head looking east (yaw 90 in T), no north offset: forward is +x
    q = w.quat_from_yaw_pitch_roll(90, 0)
    assert approx_vec(w.head_to_world((0, 1, 0), q, 0.0), (1, 0, 0))
    # north offset +30 shifts every yaw by +30
    assert w.yaw_pitch_from_vector(w.head_to_world((0, 1, 0), IDENT, 30.0))[0] == pytest.approx(30)
    assert w.yaw_pitch_from_vector(w.head_to_world((0, 1, 0), q, 30.0))[0] == pytest.approx(120)
    assert w.yaw_pitch_from_vector(w.head_to_world((0, 1, 0), q, -100.0))[0] == pytest.approx(350)
    # offset does not touch pitch or length
    v = w.head_to_world((0.2, 0.9, 0.3), IDENT, 77.0)
    assert v[2] == pytest.approx(0.3)


@given(unit_vecs(), quats(), offsets)
def test_head_world_exact_inverse(v, q, north):
    back = w.world_to_head(w.head_to_world(v, q, north), q, north)
    assert approx_vec(back, v, 1e-9)
    fwd = w.head_to_world(w.world_to_head(v, q, north), q, north)
    assert approx_vec(fwd, v, 1e-9)


@given(unit_vecs(), quats(), offsets)
def test_head_to_world_preserves_length(v, q, north):
    assert math.isclose(math.hypot(*w.head_to_world(v, q, north)), 1.0, abs_tol=1e-9)


@given(yaws360, pitches, pitches, offsets)
def test_world_yaw_is_tracking_yaw_plus_offset(head_yaw, head_pitch, pitch, north):
    """Contract §2: world_yaw = tracking_yaw + north_offset, for any head attitude."""
    t = w.quat_from_yaw_pitch_roll(head_yaw, head_pitch)
    v_t = w.vector_from_yaw_pitch(40.0, pitch)
    v_head = w.world_to_head(v_t, t, 0.0)
    yaw_w, pitch_w = w.yaw_pitch_from_vector(w.head_to_world(v_head, t, north))
    assert angle_eq(yaw_w, 40.0 + north, 1e-6)
    assert pitch_w == pytest.approx(pitch, abs=1e-6)


def test_pixel_to_world_edges(intr):
    # Right edge of the image looking north at level: yaw = atan(480/800)
    yaw, pitch = w.pixel_to_world(960, 270, intr, IDENT, 0.0, IDENT)
    assert yaw == pytest.approx(math.degrees(math.atan(480 / 800)))
    assert pitch == pytest.approx(0.0, abs=1e-9)
    # Top edge: image up => world up
    yaw, pitch = w.pixel_to_world(480, 0, intr, IDENT, 0.0, IDENT)
    assert pitch > 0
    # head yaw 90 + offset 10 + centre pixel => world yaw 100
    q = w.quat_from_yaw_pitch_roll(90, 0)
    assert w.pixel_to_world(480, 270, intr, q, 10.0, IDENT)[0] == pytest.approx(100)


# ----------------------------------------------------------------------------------- display


def test_display_focal_and_edge():
    f = w.display_focal_px(DISPLAY, FOV)
    assert f == pytest.approx(960 / math.tan(math.radians(23)))
    assert w.display_focal_px((1000, 500), (90.0, 50.0)) == pytest.approx(500.0)


def test_project_center_and_fov_edges():
    p = w.project_to_display(0, 0, IDENT, 0.0, DISPLAY, FOV)
    assert (p.x, p.y, p.on_screen, p.in_front) == (960.0, 540.0, True, True)
    right = w.project_to_display(23.0, 0, IDENT, 0.0, DISPLAY, FOV)
    left = w.project_to_display(-23.0, 0, IDENT, 0.0, DISPLAY, FOV)
    assert right.x == pytest.approx(1920.0)
    assert left.x == pytest.approx(0.0, abs=1e-9)
    assert w.project_to_display(24.0, 0, IDENT, 0.0, DISPLAY, FOV).on_screen is False
    assert w.project_to_display(22.0, 0, IDENT, 0.0, DISPLAY, FOV).on_screen is True
    # Up on the sky is up on the screen (y smaller)
    up = w.project_to_display(0, 10, IDENT, 0.0, DISPLAY, FOV)
    assert up.y < 540.0 and up.x == pytest.approx(960.0)
    f = w.display_focal_px(DISPLAY, FOV)
    assert up.y == pytest.approx(540.0 - f * math.tan(math.radians(10)))
    # Clockwise yaw => to the right
    assert w.project_to_display(10, 0, IDENT, 0.0, DISPLAY, FOV).x > 960.0


def test_project_with_head_pose_and_offset():
    q = w.quat_from_yaw_pitch_roll(50, 0)  # head looks at tracking yaw 50
    # world yaw 70 with north offset 20 == tracking yaw 50 => centre
    p = w.project_to_display(70, 0, q, 20.0, DISPLAY, FOV)
    assert p.x == pytest.approx(960.0) and p.y == pytest.approx(540.0)
    # head pitched up: a level target drops below centre
    qu = w.quat_from_yaw_pitch_roll(0, 10)
    assert w.project_to_display(0, 0, qu, 0.0, DISPLAY, FOV).y > 540.0
    # roll: head rolled clockwise by 90 puts a target above on the left of the view...
    qr = w.quat_from_yaw_pitch_roll(0, 0, 90)
    p = w.project_to_display(0, 5, qr, 0.0, DISPLAY, FOV)
    assert p.x < 960.0 and p.y == pytest.approx(540.0)  # head-up is east; world-up appears left


def test_project_behind_viewer():
    for yaw, pitch in [(180, 0), (170, 10), (90.0001, 0), (0, 0)]:
        q = IDENT if yaw != 0 else w.quat_from_yaw_pitch_roll(180, 0)
        p = w.project_to_display(yaw, pitch, q, 0.0, DISPLAY, FOV)
        assert p.in_front is False
        assert p.on_screen is False
        assert math.isfinite(p.x) and math.isfinite(p.y)
    # exactly 90 deg to the side is not in front
    assert w.project_to_display(90, 0, IDENT, 0.0, DISPLAY, FOV).in_front is False
    assert w.project_to_display(89, 0, IDENT, 0.0, DISPLAY, FOV).in_front is True


def test_head_angles_hand_values():
    a, b, front = w.head_angles_deg(10, 5, IDENT, 0.0)
    # tan(alpha') = x/y = tan(10 deg); tan(beta') = z/y = tan(5 deg)/cos(10 deg)
    assert a == pytest.approx(10.0)
    assert math.tan(math.radians(b)) == pytest.approx(
        math.tan(math.radians(5)) / math.cos(math.radians(10))
    )
    assert front is True
    a, b, front = w.head_angles_deg(0, 0, IDENT, 0.0)
    assert (a, b, front) == (0.0, 0.0, True)
    a, b, front = w.head_angles_deg(30, 0, IDENT, 0.0)
    assert (a, b) == pytest.approx((30.0, 0.0))
    a, b, front = w.head_angles_deg(0, 20, IDENT, 0.0)
    assert (a, b) == pytest.approx((0.0, 20.0))
    # north offset: world 40 with offset 10 is tracking 30
    assert w.head_angles_deg(40, 0, IDENT, 10.0)[0] == pytest.approx(30.0)
    # behind
    a, b, front = w.head_angles_deg(180, 0, IDENT, 0.0)
    assert front is False
    assert abs(a) == pytest.approx(180.0)


@given(yaws360, pitches, quats(), offsets)
def test_head_angles_match_projection(yaw, pitch, q, north):
    a, b, front = w.head_angles_deg(yaw, pitch, q, north)
    p = w.project_to_display(yaw, pitch, q, north, DISPLAY, FOV)
    assert p.in_front == front
    if front and abs(a) < 80 and abs(b) < 80:
        f = w.display_focal_px(DISPLAY, FOV)
        assert p.x == pytest.approx(960 + f * math.tan(math.radians(a)), abs=1e-5, rel=1e-6)
        assert p.y == pytest.approx(540 - f * math.tan(math.radians(b)), abs=1e-5, rel=1e-6)


def test_angular_size_to_px():
    f = w.display_focal_px(DISPLAY, FOV)
    assert w.angular_size_to_px(0.0, DISPLAY, FOV) == 0.0
    # The full horizontal FOV spans the full display width.
    assert w.angular_size_to_px(46.0, DISPLAY, FOV) == pytest.approx(1920.0)
    # small angles: ~ f * radians
    assert w.angular_size_to_px(1.0, DISPLAY, FOV) == pytest.approx(f * math.radians(1), rel=1e-3)
    assert math.isfinite(w.angular_size_to_px(500.0, DISPLAY, FOV))


@given(
    st.floats(0.5, 22, **finite),
    st.floats(-12, 12, **finite).filter(lambda x: abs(x) < 12),
)
def test_pixel_to_world_to_display_consistency(du, dv):
    """A camera pixel -> world direction -> display: where a world point appears on the display
    is the same direction the camera saw it in (identity calibration, display axis == camera axis).
    """
    intr = CameraIntrinsics(700.0, 700.0, 480.0, 270.0, 960, 540)
    head = w.quat_from_yaw_pitch_roll(33.0, 7.0, 3.0)
    north = 12.0
    u, v = 480 + du * 20, 270 + dv * 10
    yaw, pitch = w.pixel_to_world(u, v, intr, head, north, IDENT)
    # Same head pose now: display direction (alpha', beta') must equal camera ray direction.
    a, b, front = w.head_angles_deg(yaw, pitch, head, north)
    assert front
    x, y = (u - 480) / 700, (v - 270) / 700  # tan(alpha), tan(beta_down)
    assert math.tan(math.radians(a)) == pytest.approx(x, abs=1e-9)
    assert math.tan(math.radians(b)) == pytest.approx(-y, abs=1e-9)
    # And via the display: horizontal screen offset is f_g * x.
    p = w.project_to_display(yaw, pitch, head, north, DISPLAY, FOV)
    f = w.display_focal_px(DISPLAY, FOV)
    assert p.x == pytest.approx(960 + f * x, abs=1e-6)
    assert p.y == pytest.approx(540 + f * y, abs=1e-6)  # down in camera == down on screen
    assert p.on_screen == (0 <= p.x <= 1920 and 0 <= p.y <= 1080)


@given(yaws360, pitches, quats(), offsets)
def test_display_world_roundtrip(yaw, pitch, q, north):
    """Project a world direction, then un-project the display pixel with the same head pose."""
    p = w.project_to_display(yaw, pitch, q, north, DISPLAY, FOV)
    assume(p.in_front)
    f = w.display_focal_px(DISPLAY, FOV)
    assume(abs(p.x - 960) < 5e4 and abs(p.y - 540) < 5e4)
    h = (p.x - 960, f, -(p.y - 540))  # head-frame vector up to scale
    n = math.sqrt(sum(c * c for c in h))
    h = tuple(c / n for c in h)
    y2, p2 = w.yaw_pitch_from_vector(w.head_to_world(h, q, north))
    assert p2 == pytest.approx(pitch, abs=1e-5)
    if abs(pitch) < 89:
        assert angle_eq(y2, yaw, 1e-4)


# ----------------------------------------------------------------------------- phone & alignment


def test_phone_pointer_vector_and_head_yaw():
    assert w.phone_pointer_vector(IDENT) == (0.0, 1.0, 0.0)
    q = w.quat_from_yaw_pitch_roll(90, 20)
    assert approx_vec(w.phone_pointer_vector(q), w.vector_from_yaw_pitch(90, 20))
    assert w.head_forward_yaw_deg(IDENT) == 0.0
    assert w.head_forward_yaw_deg(w.quat_from_yaw_pitch_roll(-30, 10)) == pytest.approx(330)
    assert w.head_forward_yaw_deg(w.quat_from_yaw_pitch_roll(200, -45, 25)) == pytest.approx(200)
    # Phone lying flat, screen up, top edge pointing north is identity; tilting it up 90 about x
    # (top edge to the sky) gives pitch 90.
    up = w.quat_from_axis_angle((1, 0, 0), math.pi / 2)
    assert approx_vec(w.phone_pointer_vector(up), (0, 0, 1))


def test_align_hand_values():
    head = w.quat_from_yaw_pitch_roll(100, 0)
    phone = w.quat_from_yaw_pitch_roll(30, 0)
    a = w.align(head, phone, 250.0, 8.0, 12.5)
    assert a.phone_to_head_deg == pytest.approx(70.0)
    assert a.north_offset_deg == pytest.approx(150.0)
    assert a.aligned_at == 12.5
    assert a.north_source == "compass"
    assert a.heading_acc_deg == 8.0
    # No compass
    a = w.align(head, phone, None, None, 1.0)
    assert a.north_offset_deg == 0.0 and a.north_source == "none" and a.heading_acc_deg is None
    assert a.phone_to_head_deg == pytest.approx(70.0)
    # Wrap: head 10, phone 350 => +20 (not -340)
    a = w.align(w.quat_from_yaw_pitch_roll(10, 0), w.quat_from_yaw_pitch_roll(350, 0), 5.0, None, 0)
    assert a.phone_to_head_deg == pytest.approx(20.0)
    # heading 5, head yaw 350 => +15 offset
    a = w.align(w.quat_from_yaw_pitch_roll(350, 0), IDENT, 5.0, None, 0)
    assert a.north_offset_deg == pytest.approx(15.0)


def test_pointer_ray_hand_values():
    al = Alignment(phone_to_head_deg=70.0, north_offset_deg=150.0, aligned_at=0.0)
    q = w.quat_from_yaw_pitch_roll(30, 12)
    y, p = w.pointer_ray(q, al)
    assert y == pytest.approx(250.0)
    assert p == pytest.approx(12.0)
    # wraps into [0, 360)
    y, _ = w.pointer_ray(w.quat_from_yaw_pitch_roll(300, 0), al)
    assert y == pytest.approx(160.0)
    assert w.pointer_ray(IDENT, Alignment()) == (0.0, 0.0)


@given(
    yaws360, pitches, yaws360, offsets, st.one_of(st.none(), yaws360), st.floats(0, 100, **finite)
)
def test_align_invariants(head_yaw, head_pitch, phone_yaw, phone_roll_unused, heading, t):
    head = w.quat_from_yaw_pitch_roll(head_yaw, head_pitch)
    phone = w.quat_from_yaw_pitch_roll(phone_yaw, head_pitch, phone_roll_unused % 360)
    a = w.align(head, phone, heading, 7.0, t)
    assert a.aligned_at == t
    assert -180 <= a.phone_to_head_deg < 180 and -180 <= a.north_offset_deg < 180
    yaw, pitch = w.pointer_ray(phone, a)
    # Pointer lands in T on the head's forward yaw...
    assert angle_eq(yaw - a.north_offset_deg, head_yaw, 1e-5)
    # ...pitch passes through unchanged (phone and head pitch agree here)
    assert pitch == pytest.approx(head_pitch, abs=1e-6)
    head_fwd_world = w.yaw_pitch_from_vector(w.head_to_world((0, 1, 0), head, a.north_offset_deg))
    if abs(head_pitch) < 88:
        assert angle_eq(yaw, head_fwd_world[0], 1e-5)
        assert pitch == pytest.approx(head_fwd_world[1], abs=1e-6)
    if heading is None:
        assert a.north_source == "none" and a.north_offset_deg == 0.0
    else:
        assert a.north_source == "compass" and a.heading_acc_deg == 7.0
        # north offset makes the head yaw equal the compass heading
        assert angle_eq(head_yaw + a.north_offset_deg, heading, 1e-5)
        assert angle_eq(yaw, heading, 1e-5)


@given(yaws360, yaws360, st.floats(-100, 100, **finite), pitches, yaws360, yaws360)
def test_align_north_to_landmark(p2h, north, drift, pitch, phone_yaw, landmark):
    al = Alignment(p2h, north, 3.0, "compass", 12.0)
    phone = w.quat_from_yaw_pitch_roll(phone_yaw, pitch)
    track_yaw = w.wrap_deg(phone_yaw + al.phone_to_head_deg)
    new = w.align_north_to_landmark(al, track_yaw, landmark)
    assert new.phone_to_head_deg == al.phone_to_head_deg
    assert new.aligned_at == al.aligned_at
    assert new.north_source == "landmark" and new.heading_acc_deg is None
    assert -180 <= new.north_offset_deg < 180
    y, p = w.pointer_ray(phone, new)
    assert angle_eq(y, landmark, 1e-5)
    assert p == pytest.approx(pitch, abs=1e-6)
    assert al.north_source == "compass"  # input unchanged (frozen)


def test_long_press_example():
    al = w.align(IDENT, IDENT, 20.0, 15.0, 0.0)  # compass says north is 20 off
    # Pointer at tracking yaw 0 is really at landmark bearing 33
    new = w.align_north_to_landmark(al, 0.0, 33.0)
    assert new.north_offset_deg == pytest.approx(33.0)
    assert w.pointer_ray(IDENT, new)[0] == pytest.approx(33.0)


# ----------------------------------------------------------------------------------- predict


def test_predict_hand_values():
    q = w.quat_from_yaw_pitch_roll(0, 0)
    # Phone yaw rate: clockwise yaw is rotation about +z by -yaw, so omega_z = -10 deg/s => +yaw
    rate = math.radians(-100.0)
    out = w.predict_quat(q, (0, 0, rate), 0.1)
    assert w.yaw_pitch_from_vector(w.phone_pointer_vector(out))[0] == pytest.approx(10.0)
    # dt clamped to 0.15 s
    out = w.predict_quat(q, (0, 0, rate), 5.0)
    assert w.yaw_pitch_from_vector(w.phone_pointer_vector(out))[0] == pytest.approx(15.0)
    # negative dt clamped to 0 => unchanged
    assert w.predict_quat(q, (0, 0, rate), -1.0) == q
    # body-frame omega: phone pitched 90 up (top edge to sky) rotating about its own x
    up = w.quat_from_axis_angle((1, 0, 0), math.pi / 2)
    out = w.predict_quat(up, (math.radians(100.0), 0, 0), 0.1)  # tilts the top edge further back
    assert w.yaw_pitch_from_vector(w.phone_pointer_vector(out))[1] == pytest.approx(
        90.0 + 10.0 - 20.0
    )
    # a body-z spin of the phone lying flat is a yaw about world z in this frame
    flat = IDENT
    out = w.predict_quat(flat, (0, 0, rate), 0.1)
    assert same_rotation(out, w.quat_from_axis_angle((0, 0, 1), rate * 0.1))
    # rolling about the pointing axis (body y) leaves the pointer where it was
    out = w.predict_quat(q, (0, 3.0, 0), 0.1)
    assert approx_vec(w.phone_pointer_vector(out), (0, 1, 0))


@given(quats(), unit_vecs(), st.floats(0, 5, **finite))
def test_predict_properties(q, axis, rate):
    assert w.predict_quat(q, (0.0, 0.0, 0.0), 0.1) == q  # identity
    omega = tuple(c * rate for c in axis)
    out = w.predict_quat(q, omega, 0.05)
    assert math.isclose(sum(c * c for c in out), 1.0, abs_tol=1e-9)
    # dt clamp: anything >= MAX_PREDICT_S gives the same result
    far = w.predict_quat(q, omega, 10.0)
    cap = w.predict_quat(q, omega, w.MAX_PREDICT_S)
    assert approx_vec(far, cap, 1e-12)

    # rotation angle is rate*dt
    def ang(a, b):
        d = abs(sum(x * y for x, y in zip(a, b, strict=True)))
        return 2 * math.acos(min(1.0, d))

    assert (
        ang(q, cap) == pytest.approx(rate * w.MAX_PREDICT_S, abs=1e-6)
        or rate * w.MAX_PREDICT_S > math.pi
    )
    # dt <= 0 : unchanged
    assert w.predict_quat(q, omega, -3.0) == q
    assert w.predict_quat(q, omega, 0.0) == q


@given(quats(), unit_vecs(), st.floats(0.1, 5, **finite), st.floats(0.01, 0.07, **finite))
def test_predict_composes(q, axis, rate, dt):
    omega = tuple(c * rate for c in axis)
    two = w.predict_quat(w.predict_quat(q, omega, dt), omega, dt)
    one = w.predict_quat(q, omega, 2 * dt)
    assert same_rotation(two, one, 1e-9)


# ------------------------------------------------------------------------------------ snapping


def test_snap_hand_values():
    es = [ent("a", 10.0), ent("b", 14.0), ent("far", 40.0)]
    assert w.choose_snap(0, 0, es, None) is None  # nearest is 10 deg away
    assert w.choose_snap(8, 0, es, None) == "a"
    assert w.choose_snap(12.5, 0, es, None) == "b"
    assert w.choose_snap(0, 0, [], None) is None
    # exactly on the radius is inside
    assert w.choose_snap(5.0, 0, [ent("e", 10.0)], None) == "e"
    # custom radius
    assert w.choose_snap(8, 0, es, None, radius_deg=1.0) is None
    # wrap-around at north
    assert w.choose_snap(359, 0, [ent("n", 2.0)], None) == "n"
    # pitch counts
    assert w.choose_snap(10, 4, [ent("hi", 10.0, 8.5)], None) == "hi"
    assert w.choose_snap(10, 0, [ent("hi", 10.0, 8.5)], None) is None


def test_snap_hysteresis():
    es = [ent("a", 10.0), ent("b", 14.0)]
    # pointer at 11.0: a is 1.0 away, b 3.0. Currently on b: a is 2.0 closer -> switch.
    assert w.choose_snap(11.0, 0, es, "b") == "a"
    # pointer at 11.5: a 1.5, b 2.5: only 1.0 closer -> stay on b
    assert w.choose_snap(11.5, 0, es, "b") == "b"
    # pointer at 12: a 2, b 2 (exact tie) -> stay with current
    assert w.choose_snap(12.0, 0, es, "a") == "a"
    assert w.choose_snap(12.0, 0, es, "b") == "b"
    # current falls outside the radius -> pick best fresh candidate
    assert w.choose_snap(8.0, 0, [ent("a", 10.0), ent("b", 14.0), ent("c", 20.0)], "c") == "a"
    # current id vanished from the list
    assert w.choose_snap(8.0, 0, es, "gone") == "a"
    # custom hysteresis
    assert w.choose_snap(11.5, 0, es, "b", hysteresis_deg=0.5) == "a"


def test_snap_tie_break_by_id():
    tie = [ent("zz", 7.0), ent("aa", 7.0), ent("mm", 7.0), ent("00far", 9.0)]
    assert w.choose_snap(5.0, 0, tie, None) == "aa"
    assert w.choose_snap(5.0, 0, list(reversed(tie)), None) == "aa"


@given(
    st.lists(
        st.tuples(yaws360, st.floats(-30, 30, **finite)), min_size=0, max_size=12, unique=True
    ),
    yaws360,
    st.floats(-30, 30, **finite),
    st.data(),
)
def test_snap_invariants(pts, py, pp, data):
    es = [ent(f"e{i:02d}", y, p) for i, (y, p) in enumerate(pts)]
    dist = {e.id: w.angular_distance_deg(py, pp, e.world_yaw_deg, e.world_pitch_deg) for e in es}
    current = data.draw(st.sampled_from([None, "nope", *dist.keys()]))
    got = w.choose_snap(py, pp, es, current)
    inside = {k: d for k, d in dist.items() if d <= w.SNAP_RADIUS_DEG}
    if not inside:
        assert got is None
        return
    # never outside the radius
    assert got in inside
    best = min(inside.values())
    if current not in inside:
        assert inside[got] == pytest.approx(best, abs=1e-9)
    else:
        if got != current:
            # switched: must be at least hysteresis closer
            assert inside[got] <= inside[current] - w.SNAP_HYSTERESIS_DEG + 1e-9
            assert inside[got] == pytest.approx(best, abs=1e-9)
        else:
            # stayed: nobody is hysteresis-or-more closer (allow float fuzz at the boundary)
            assert best > inside[current] - w.SNAP_HYSTERESIS_DEG - 1e-9
    # result independent of input order
    shuffled = data.draw(st.permutations(es))
    assert w.choose_snap(py, pp, shuffled, current) == got
    # idempotent when fed back as current
    assert w.choose_snap(py, pp, es, got) == got


# ------------------------------------------------------------------ fixture generator cross-check


def test_quat_matches_generator_pose_track(pose_track):
    for pose in pose_track[::7]:
        yaw, pitch = gen.head_yaw_pitch(pose.t)
        mine = w.quat_from_yaw_pitch_roll(yaw, pitch)
        assert same_rotation(mine, pose.quat, 2e-8), pose.t
        assert same_rotation(mine, gen.quat_yaw_pitch(yaw, pitch), 1e-12)
        assert w.head_forward_yaw_deg(pose.quat) == pytest.approx(w.wrap_deg(yaw), abs=1e-5)


def test_pixel_to_world_recovers_markers(pose_track, scene, fixture_intrinsics):
    ident = IDENT
    seen = 0
    for pose in pose_track[::5]:
        for m in scene["markers_tracking_frame"]:
            d = gen.direction(m["yaw_deg"], m["pitch_deg"])
            px = gen.project(pose.quat, d)
            if px is None:
                continue
            seen += 1
            yaw, pitch = w.pixel_to_world(px[0], px[1], fixture_intrinsics, pose.quat, 0.0, ident)
            assert angle_eq(yaw, m["yaw_deg"], 1e-6), (pose.t, m["id"])
            assert pitch == pytest.approx(m["pitch_deg"], abs=1e-6)
    assert seen > 100


def test_markers_visible_in_generator_frame_project_on_display(pose_track, scene):
    """If the generator says a marker is at pixel (u, v) then it is at the same angle on the
    display (same head pose), and non-visible ones behind the head are reported as such."""
    for pose in pose_track[::50]:
        for m in scene["markers_tracking_frame"]:
            px = gen.project(pose.quat, gen.direction(m["yaw_deg"], m["pitch_deg"]))
            p = w.project_to_display(m["yaw_deg"], m["pitch_deg"], pose.quat, 0.0, DISPLAY, FOV)
            assert p.in_front == (px is not None)
            if px is not None:
                fd = w.display_focal_px(DISPLAY, FOV)
                assert (p.x - 960) / fd == pytest.approx((px[0] - 480) / gen.FX, abs=1e-9)
                assert (p.y - 540) / fd == pytest.approx((px[1] - 270) / gen.FX, abs=1e-9)


def test_generator_pose_with_north_offset(pose_track, scene, fixture_intrinsics):
    """With a north offset the world yaw is tracking yaw + offset (contract §2)."""
    pose = pose_track[600]
    m = scene["markers_tracking_frame"][1]
    px = gen.project(pose.quat, gen.direction(m["yaw_deg"], m["pitch_deg"]))
    yaw, pitch = w.pixel_to_world(*px, fixture_intrinsics, pose.quat, 77.0, IDENT)
    assert angle_eq(yaw, m["yaw_deg"] + 77.0, 1e-6)
    assert pitch == pytest.approx(m["pitch_deg"], abs=1e-6)


# --------------------------------------------------------------------------------- performance


def test_fast_enough_for_a_frame():
    q = w.quat_from_yaw_pitch_roll(10, 5)
    es = [ent(f"e{i}", i * 1.1 % 360, (i % 20) - 10) for i in range(300)]
    t0 = time.perf_counter()
    for _ in range(10):
        for e in es:
            w.project_to_display(e.world_yaw_deg, e.world_pitch_deg, q, 3.0, DISPLAY, FOV)
        w.choose_snap(12, 0, es, None)
    per_frame = (time.perf_counter() - t0) / 10
    assert per_frame < 0.016
