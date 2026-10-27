from __future__ import annotations

import math

import pytest

from pointview.detection.convert import (
    REAL_HEIGHT_M,
    estimate_distance_m,
    scale_intrinsics,
    track_to_entity,
)
from pointview.detection.tracker import Track
from pointview.interfaces import IDENTITY_QUAT, CameraIntrinsics

from . import _world_ref as ref

INTR = CameraIntrinsics(fx=960.0, fy=960.0, cx=960.0, cy=540.0, width=1920, height=1080)


def track(box, label="car", kind="vehicle", tid=7, t=5.0, score=0.8) -> Track:
    return Track(tid, kind, label, box, score, 3, t, confirmed=True)


def test_distance_formula_person_and_car():
    # d = fy * H / h: a 1.7 m person 170 px tall at fy = 1000 is 10 m away.
    assert estimate_distance_m("person", (0, 100, 50, 270), 1000.0) == pytest.approx(10.0)
    assert estimate_distance_m("car", (0, 100, 100, 250), 1000.0) == pytest.approx(10.0)
    assert REAL_HEIGHT_M["person"] == 1.7 and REAL_HEIGHT_M["car"] == 1.5
    assert set(REAL_HEIGHT_M) == {"person", "bicycle", "car", "motorcycle", "bus", "truck"}


def test_distance_halves_when_box_doubles():
    d1 = estimate_distance_m("bus", (0, 0 + 10, 10, 110), 900.0)
    d2 = estimate_distance_m("bus", (0, 10, 10, 210), 900.0)
    assert d1 == pytest.approx(2 * d2)


def test_distance_none_for_truncated_or_degenerate_boxes():
    assert estimate_distance_m("person", (0, 0, 50, 300), 1000.0, frame_h=1080) is None
    assert estimate_distance_m("person", (0, 700, 50, 1080), 1000.0, frame_h=1080) is None
    assert estimate_distance_m("person", (0, 100, 50, 100), 1000.0) is None
    assert estimate_distance_m("zebra", (0, 100, 50, 200), 1000.0) is None


def test_scale_intrinsics():
    s = scale_intrinsics(INTR, 960, 540)
    assert (s.fx, s.fy, s.cx, s.cy, s.width, s.height) == (480.0, 480.0, 480.0, 270.0, 960, 540)
    assert scale_intrinsics(INTR, 1920, 1080) is INTR


def test_box_at_image_center_maps_to_head_yaw_at_capture_time():
    head = ref.quat_from_yaw_pitch(40.0, 5.0)
    e = track_to_entity(track((910, 490, 1010, 590)), INTR, head, 0.0, IDENTITY_QUAT, (1920, 1080))
    assert e.id == "det:7" and e.kind == "vehicle" and e.label == "car"
    assert e.world_yaw_deg == pytest.approx(40.0, abs=1e-6)
    assert e.world_pitch_deg == pytest.approx(5.0, abs=1e-6)
    assert e.updated_at == 5.0 and e.confidence == 0.8


def test_north_offset_and_off_center_box():
    head = ref.quat_from_yaw_pitch(10.0, 0.0)
    # Box center 200 px right of center and 100 px above it.
    box = (1100, 400, 1220, 500)  # center (1160, 450)
    e = track_to_entity(track(box), INTR, head, 30.0, IDENTITY_QUAT, (1920, 1080))
    alpha = math.degrees(math.atan(200 / 960))
    beta_up = math.degrees(math.atan(90 / 960))
    assert e.world_yaw_deg == pytest.approx(10.0 + 30.0 + alpha, abs=0.2)
    assert 0 < e.world_pitch_deg == pytest.approx(beta_up, abs=0.2)


def test_size_deg_from_box_corners_and_distance_use_fy():
    box = (860, 440, 1060, 640)  # 200 x 200 px around the center
    e = track_to_entity(
        track(box, "person", "person"), INTR, IDENTITY_QUAT, 0.0, IDENTITY_QUAT, (1920, 1080)
    )
    expect = 2 * math.degrees(math.atan(100 / 960))
    assert e.size_deg.w == pytest.approx(expect, abs=1e-6)
    assert e.size_deg.h == pytest.approx(expect, abs=1e-6)
    assert e.distance_m == pytest.approx(960 * 1.7 / 200)
    assert e.world_yaw_deg == pytest.approx(0.0, abs=1e-6)


def test_frame_size_differs_from_intrinsics_size():
    # Same scene delivered at 960x540: a centered box still maps to the center.
    e = track_to_entity(
        track((430, 220, 530, 320)),
        INTR,
        ref.quat_from_yaw_pitch(90.0),
        0.0,
        IDENTITY_QUAT,
        (960, 540),
    )
    assert e.world_yaw_deg == pytest.approx(90.0, abs=1e-6)
    assert e.distance_m == pytest.approx(480 * 1.5 / 100)
