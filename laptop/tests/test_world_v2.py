"""Tests for world.py v2: geodesy, view test, backdrop homography, Sun position."""

from __future__ import annotations

import itertools
import math
import random
from datetime import UTC, datetime

import pytest

from pointview import world as w
from pointview.interfaces import CameraIntrinsics

FOV = (46.0, 27.0)
DISP = (1920, 1080)
INTR = CameraIntrinsics(800.0, 800.0, 480.0, 270.0, 960, 540)
Q0 = w.quat_from_yaw_pitch_roll(0, 0)
C2H = w.quat_from_yaw_pitch_roll(1.5, -2.0, 0.7)


def _unix(y, mo, d, h=0, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=UTC).timestamp()


# ------------------------------------------------------------------------------- geodesy


def test_ecef_known_points():
    assert w.ecef_from_geodetic(0, 0, 0) == pytest.approx((w.WGS84_A, 0, 0), abs=1e-6)
    b = w.WGS84_A * (1 - w.WGS84_F)
    assert w.ecef_from_geodetic(90, 0, 0) == pytest.approx((0, 0, b), abs=1e-6)
    assert w.ecef_from_geodetic(0, 90, 100)[1] == pytest.approx(w.WGS84_A + 100, abs=1e-6)


def test_look_straight_up():
    _, pitch, rng = w.look_angles(37.0, -122.0, 0.0, 37.0, -122.0, 1000.0)
    assert pitch == pytest.approx(90.0, abs=1e-6)
    assert rng == pytest.approx(1000.0, abs=1e-6)


def test_look_due_east_equator():
    yaw, pitch, _ = w.look_angles(0.0, 0.0, 0.0, 0.0, 0.5, 0.0)
    assert yaw == pytest.approx(90.0, abs=1e-6)
    # Chord of 0.5 deg of equator dips by half the central angle.
    assert pitch == pytest.approx(-0.25, abs=0.01)


def test_look_cardinal_yaws():
    north = w.look_angles(10, 20, 0, 11, 20, 0)[0]
    assert min(north, 360 - north) < 1e-6
    assert w.look_angles(10, 20, 0, 9, 20, 0)[0] == pytest.approx(180.0, abs=1e-6)
    assert w.look_angles(10, 20, 0, 10, 19, 0)[0] == pytest.approx(270.0, abs=0.1)


def test_look_100km_sea_level_below_horizon():
    # Central angle 100 km / 6371 km = 0.899 deg; pitch ~ -theta/2 = -0.45 deg (independent).
    lat2 = math.degrees(100_000.0 / 6_371_000.0)
    _, pitch, rng = w.look_angles(0.0, 0.0, 0.0, lat2, 0.0, 0.0)
    assert pitch == pytest.approx(-0.45, abs=0.02)
    assert rng == pytest.approx(100_000.0, rel=0.01)  # meridian degree is 110.6 km at the equator


def test_look_plane_10km_up_50km_away():
    # Spherical hand calc: tan(pitch) = ((R+h) cos t - R) / ((R+h) sin t), t = 50 km / R.
    r, h = 6_371_000.0, 10_000.0
    t = 50_000.0 / r
    expect = math.degrees(math.atan2((r + h) * math.cos(t) - r, (r + h) * math.sin(t)))
    _, pitch, _ = w.look_angles(0.0, 0.0, 0.0, math.degrees(t), 0.0, h)
    assert expect == pytest.approx(11.1, abs=0.1)
    assert pitch == pytest.approx(expect, abs=0.1)


@pytest.mark.parametrize("seed", range(20))
def test_look_matches_bearing_and_haversine_short(seed):
    rnd = random.Random(seed)
    lat, lon = rnd.uniform(-70, 70), rnd.uniform(-180, 180)
    lat2 = lat + rnd.uniform(-0.05, 0.05)
    lon2 = lon + rnd.uniform(-0.05, 0.05)
    yaw, _, rng = w.look_angles(lat, lon, 0.0, lat2, lon2, 0.0)
    d = w.haversine_m(lat, lon, lat2, lon2)
    assert abs(w.wrap_deg_signed(yaw - w.bearing_deg(lat, lon, lat2, lon2))) < 0.3
    assert abs(rng - d) / d < 0.01  # sphere vs ellipsoid


def test_look_coincident():
    assert w.look_angles(1, 2, 3, 1, 2, 3) == (0.0, 0.0, 0.0)


# ------------------------------------------------------------------------------- in_view


def test_in_view_basic_and_margin():
    assert w.in_view(0, 0, Q0, 0, FOV)
    assert w.in_view(22.9, 0, Q0, 0, FOV)
    assert not w.in_view(23.5, 0, Q0, 0, FOV)
    assert w.in_view(23.5, 0, Q0, 0, FOV, margin_deg=1.0)
    assert not w.in_view(0, 14.5, Q0, 0, FOV)
    assert w.in_view(0, 14.5, Q0, 0, FOV, margin_deg=1.5)


def test_in_view_behind_false():
    assert not w.in_view(180, 0, Q0, 0, FOV)
    assert not w.in_view(180, 0, Q0, 0, FOV, margin_deg=30)
    assert not w.in_view(90, 0, Q0, 0, FOV, margin_deg=60)


def test_in_view_north_offset_and_head():
    q = w.quat_from_yaw_pitch_roll(30, 10)
    assert w.in_view(40, 10, q, 10, FOV)  # tracking yaw 30 + offset 10 = world 40
    assert not w.in_view(40, 10, q, -30, FOV)


@pytest.mark.parametrize("seed", range(300))
def test_in_view_consistent_with_head_angles(seed):
    rnd = random.Random(seed)
    q = w.quat_from_yaw_pitch_roll(rnd.uniform(0, 360), rnd.uniform(-60, 60), rnd.uniform(-5, 5))
    off = rnd.uniform(-180, 180)
    yaw, pitch = rnd.uniform(0, 360), rnd.uniform(-90, 90)
    margin = rnd.choice([0.0, 2.0, 5.0])
    a, b, front = w.head_angles_deg(yaw, pitch, q, off)
    got = w.in_view(yaw, pitch, q, off, FOV, margin)
    assert got == (front and abs(a) <= 23 + margin and abs(b) <= 13.5 + margin)


@pytest.mark.parametrize("seed", range(300))
def test_in_view_zero_margin_matches_on_screen(seed):
    rnd = random.Random(1000 + seed)
    q = w.quat_from_yaw_pitch_roll(rnd.uniform(0, 360), rnd.uniform(-60, 60))
    off = rnd.uniform(-180, 180)
    yaw, pitch = rnd.uniform(0, 360), rnd.uniform(-90, 90)
    # Vertical FOV that matches the 16:9 display exactly.
    fov_v = math.degrees(2 * math.atan(math.tan(math.radians(23)) * 1080 / 1920))
    p = w.project_to_display(yaw, pitch, q, off, DISP, FOV)
    a, b, _ = w.head_angles_deg(yaw, pitch, q, off)
    if abs(abs(a) - 23) < 1e-6 or abs(abs(b) - fov_v / 2) < 1e-6:
        return  # exactly on the edge
    assert w.in_view(yaw, pitch, q, off, (46.0, fov_v)) == p.on_screen


# -------------------------------------------------------------------------- homography


def _apply(hm, u, v):
    x = hm[0] * u + hm[1] * v + hm[2]
    y = hm[3] * u + hm[4] * v + hm[5]
    d = hm[6] * u + hm[7] * v + hm[8]
    return x / d, y / d


def test_homography_normalized_and_centered():
    hm = w.backdrop_homography(INTR, Q0, Q0, Q0, DISP, FOV)
    assert len(hm) == 9 and hm[8] == pytest.approx(1.0)
    f = w.display_focal_px(DISP, FOV)
    assert _apply(hm, 480, 270) == pytest.approx((960, 540), abs=1e-6)
    assert _apply(hm, 560, 270) == pytest.approx((960 + 0.1 * f, 540), abs=1e-6)


@pytest.mark.parametrize("seed", range(100))
def test_homography_equals_project_to_display(seed):
    rnd = random.Random(seed)
    cap = w.quat_from_yaw_pitch_roll(rnd.uniform(0, 360), rnd.uniform(-40, 40), rnd.uniform(-5, 5))
    now = w.quat_mul(cap, w.quat_from_yaw_pitch_roll(rnd.uniform(-4, 4), rnd.uniform(-3, 3)))
    off = rnd.uniform(-180, 180)
    u, v = rnd.uniform(0, 960), rnd.uniform(0, 540)
    hm = w.backdrop_homography(INTR, C2H, cap, now, DISP, FOV)
    x, y = _apply(hm, u, v)
    yaw, pitch = w.pixel_to_world(u, v, INTR, cap, off, C2H)
    p = w.project_to_display(yaw, pitch, now, off, DISP, FOV)
    assert p.in_front
    assert (x, y) == pytest.approx((p.x, p.y), abs=1e-6)


def test_homography_head_turn_right_shifts_content_left():
    now = w.quat_from_yaw_pitch_roll(5.0, 0.0)  # head turned right by 5 deg
    hm = w.backdrop_homography(INTR, Q0, Q0, now, DISP, FOV)
    f = w.display_focal_px(DISP, FOV)
    x, y = _apply(hm, 480, 270)
    assert x == pytest.approx(960 - f * math.tan(math.radians(5)), abs=1e-6)
    assert y == pytest.approx(540, abs=1e-6)


# ---------------------------------------------------------------------------------- sun


def _scan(lat, lon, t0, step, n):
    return [(t0 + i * step, *w.sun_direction(t0 + i * step, lat, lon)) for i in range(n)]


def test_sun_j2000_near_zenith():
    # Almanac: at 2000-01-01 12:00 UT the Sun's declination is -23.0 deg and the equation of
    # time about -3.2 min, so at lat -23.0, lon 0 the Sun is within ~1 deg of the zenith.
    assert w.sun_direction(_unix(2000, 1, 1, 12), -23.0, 0.0)[1] > 88.5


@pytest.mark.parametrize(
    ("date", "lat", "expect"),
    [
        # Noon elevation = 90 - lat + declination (+23.44 June solstice, -23.44 December
        # solstice, ~0 at equinoxes); matches the NOAA solar calculator to ~0.1 deg.
        ((2024, 6, 20), 40.0, 90 - 40 + 23.44),
        ((2024, 12, 21), 51.5, 90 - 51.5 - 23.44),
        ((2024, 3, 20), 0.0, 90.0),
        ((2024, 9, 22), 35.0, 55.0),
    ],
)
def test_sun_noon_elevation(date, lat, expect):
    best = max(_scan(lat, 0.0, _unix(*date), 120, 720), key=lambda r: r[2])
    assert best[2] == pytest.approx(expect, abs=0.6)
    if lat > 0.5:  # northern observer: Sun due south at noon
        assert best[1] == pytest.approx(180.0, abs=2.0)


def test_sun_sunrise_azimuth_london_solstice():
    # Sunrise azimuth: cos A = sin(dec)/cos(lat) (standard spherical astronomy). dec 23.44,
    # lat 51.5 gives A = 50.3 deg at geometric h = 0, about 49.9 deg at the usual -0.833 deg
    # (refraction + solar radius). NOAA/timeanddate list roughly 49-50 deg for London.
    rows = _scan(51.5, 0.0, _unix(2024, 6, 20), 30, 2880)
    rise = next(r for a, r in itertools.pairwise(rows) if a[2] < -0.833 <= r[2])
    assert rise[1] == pytest.approx(49.9, abs=1.0)


def test_sun_equinox_sunrise_due_east():
    rows = _scan(40.0, 0.0, _unix(2024, 3, 20), 30, 2880)
    rise = next(r for a, r in itertools.pairwise(rows) if a[2] < 0 <= r[2])
    assert rise[1] == pytest.approx(90.0, abs=1.5)


def test_sun_dark_at_local_midnight():
    assert w.sun_direction(_unix(2024, 1, 15, 0), 51.5, 0.0)[1] < -20
