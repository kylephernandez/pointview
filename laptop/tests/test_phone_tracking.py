"""Synthetic camera images plus independent camera/IMU paths; no real device required."""

import math
from dataclasses import replace

import cv2
import numpy as np
import pytest

from pointview import world
from pointview.entities import Entity
from pointview.interfaces import Alignment, Calibration, CameraIntrinsics, Frame, PhoneState
from pointview.phone_tracking import (
    DICTIONARY,
    MarkerDetector,
    Observation,
    PhoneTracker,
    TrackingTarget,
    gyro_delta,
    rotation_error,
    rotation_quat,
    sample_at,
)

INTR = CameraIntrinsics(650, 650, 320, 240, 640, 480)
TARGET = TrackingTarget("a" * 32, 42)
IDENTITY = (1.0, 0.0, 0.0, 0.0)


def sample(t, quat=IDENTITY, omega=(0.0, 0.0, 0.0), **kw):
    return PhoneState(
        t_ms=t * 1000,
        t=t,
        received_at=t,
        quat=quat,
        omega=omega,
        heading_deg=None,
        heading_acc=None,
        lat=None,
        lon=None,
        gps_acc=None,
        **kw,
    )


def motion_samples(rate=0.5, axis=(0.0, 0.0, 1.0), reference=IDENTITY):
    return [
        sample(
            i / 100,
            world.quat_mul(reference, world.quat_from_axis_angle(axis, rate * i / 100)),
            tuple(rate * a for a in axis),
        )
        for i in range(101)
    ]


def acquire(tracker, samples, reference=IDENTITY, start=0.1, end=0.6):
    results = []
    for t in np.arange(start, end + 0.001, 0.05):
        phone = sample_at(samples, float(t))
        results.append(
            tracker.observe(
                Observation(float(t), world.quat_mul(reference, phone.quat), (0.0, 0.4, -0.4), 0.1),
                samples,
            )
        )
    return results


def synthetic_frame(marker_id=42, rotation=None, position=(0.03, 0.10, 0.6), t=0.1):
    # A phone tilted toward the camera; image corner order is top-left, top-right, bottom-right, bottom-left.
    rot = cv2.Rodrigues(np.array([2.65, 0.15, 0.1]))[0] if rotation is None else rotation
    pos = np.array(position, dtype=float)
    s = 0.08 / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=float)
    k = np.array([[INTR.fx, 0, INTR.cx], [0, INTR.fy, INTR.cy], [0, 0, 1]], dtype=float)
    corners = cv2.projectPoints(obj, cv2.Rodrigues(rot)[0], pos, k, None)[0].reshape(4, 2)
    tile = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(DICTIONARY), marker_id, 240
    )
    transform = cv2.getPerspectiveTransform(
        np.array([[0, 0], [239, 0], [239, 239], [0, 239]], dtype=np.float32),
        corners.astype(np.float32),
    )
    gray = cv2.warpPerspective(
        tile, transform, (INTR.width, INTR.height), flags=cv2.INTER_NEAREST, borderValue=255
    )
    return Frame(t, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)), rot, pos


def test_wire_pattern_is_exact_dictionary_marker():
    message = TARGET.message()
    image = np.array([[int(p) * 255 for p in row] for row in message["cells"]], np.uint8)
    expected = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICTIONARY), 42, 8)
    np.testing.assert_array_equal(image, expected)
    assert TrackingTarget.create() != TrackingTarget.create()


def test_perspective_pose_camera_calibration_head_motion_and_physical_scale():
    frame, rot, pos = synthetic_frame()
    head = world.quat_from_yaw_pitch_roll(47, 12, 7)
    calibration = Calibration(world.quat_from_yaw_pitch_roll(4, -3, 2))
    obs = MarkerDetector().detect(frame, INTR, TARGET, 0.08, head, calibration)
    assert obs is not None
    basis = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
    expected = world.quat_mul(
        head, world.quat_mul(calibration.camera_to_head_quat, rotation_quat(basis @ rot))
    )
    assert rotation_error(obs.quat, expected) < 2
    np.testing.assert_allclose(
        obs.position,
        world.quat_rotate(head, world.camera_to_head(tuple(pos), calibration.camera_to_head_quat)),
        atol=0.01,
    )
    assert MarkerDetector().detect(frame, INTR, TARGET, None, head, calibration).position is None


def test_wrong_id_blank_duplicate_and_wrong_image_size_are_rejected():
    detector = MarkerDetector()
    wrong, _, _ = synthetic_frame(marker_id=17)
    assert detector.detect(wrong, INTR, TARGET, 0.08, IDENTITY, Calibration()) is None
    assert (
        detector.detect(
            Frame(0.1, np.full((480, 640, 3), 255, np.uint8)),
            INTR,
            TARGET,
            0.08,
            IDENTITY,
            Calibration(),
        )
        is None
    )
    frame, _, _ = synthetic_frame()
    assert (
        detector.detect(frame, replace(INTR, width=1280), TARGET, 0.08, IDENTITY, Calibration())
        is None
    )
    a, _, _ = synthetic_frame(position=(-0.12, 0.0, 0.6))
    b, _, _ = synthetic_frame(position=(0.12, 0.0, 0.6))
    duplicate = Frame(0.1, np.minimum(a.bgr, b.bgr))
    assert detector.detect(duplicate, INTR, TARGET, 0.08, IDENTITY, Calibration()) is None


def test_full_rotation_is_acquired_without_compass_and_carried_out_of_view():
    tracker = PhoneTracker(TARGET)
    samples = motion_samples()
    reference = world.quat_from_yaw_pitch_roll(45, 8, -5)
    results = acquire(tracker, samples, reference)
    assert not results[0] and any(results)
    assert rotation_error(tracker.reference_quat, reference) < 1e-5
    assert tracker.state(0.61) == "tracked"
    alignment = Alignment(aligned_at=0.6, phone_to_tracking_quat=tracker.reference_quat)
    expected = world.yaw_pitch_from_vector(
        world.quat_rotate(world.quat_mul(reference, samples[90].quat), (0, 1, 0))
    )
    assert world.pointer_ray(samples[90].quat, alignment) == pytest.approx(expected)
    assert tracker.state(1.0) == "coasting"
    assert tracker.origin_world(1.0, 0) is None
    assert tracker.origin_world(0.61, 90) == pytest.approx((0.4, 0.0, -0.4))


def test_stationary_target_does_not_acquire_even_with_same_id():
    tracker = PhoneTracker(TARGET)
    assert not any(acquire(tracker, motion_samples(rate=0)))
    assert tracker.reference_quat is None


def test_independent_gyro_and_vision_mismatch_cannot_refresh_alignment():
    tracker = PhoneTracker(TARGET)
    samples = motion_samples()
    acquire(tracker, samples)
    accepted = tracker.accepted
    # Same marker held still while the actual phone rotates.
    for t in np.arange(0.65, 0.96, 0.05):
        tracker.observe(Observation(float(t), accepted.quat, (0.0, 0.0, 0.0), 0.1), samples)
    assert tracker.accepted.t < 0.8
    assert not tracker.verified
    assert tracker.reference_quat is not None  # inertial fallback is preserved
    assert tracker.origin_world(0.96, 0) is None
    # Quat stream alone matching vision is insufficient when the separate gyro disagrees.
    tracker = PhoneTracker(TARGET)
    silent_gyro = [replace(s, omega=(0.0, 0.0, 0.0)) for s in samples]
    assert not any(acquire(tracker, silent_gyro))


def test_reverse_and_wrong_axis_rotations_do_not_match():
    for axis in [(0, 0, -1), (1, 0, 0)]:
        tracker = PhoneTracker(TARGET)
        phone = motion_samples()
        visual = motion_samples(axis=axis)
        for t in np.arange(0.1, 0.61, 0.05):
            obs = Observation(float(t), sample_at(visual, float(t)).quat, None, 0)
            assert not tracker.observe(obs, phone)
        assert tracker.reference_quat is None


def test_missing_stale_out_of_order_and_gapped_samples_fail_closed():
    samples = motion_samples()
    assert sample_at(samples, -0.01) is None
    assert sample_at(samples, 1.1) is None
    assert gyro_delta([samples[0], samples[50]], 0, 0.5) is None
    tracker = PhoneTracker(TARGET)
    acquire(tracker, samples)
    assert not tracker.observe(Observation(0.2, IDENTITY, None, 0), samples)
    assert not tracker.verified
    assert not tracker.observe(Observation(2.0, IDENTITY, None, 0), samples)


def test_fresh_translation_corrects_close_selection_but_unknown_distance_does_not():
    # Hand 40 cm below the eyes, aiming at an object 10 m away at eye height.
    target = Entity("a", "vehicle", "car", 0.0, 0.0, 1.0, distance_m=10)
    pitch = math.degrees(math.atan2(0.4, 10))
    assert world.choose_snap(0, pitch, [target], None, radius_deg=0.1) is None
    assert (
        world.choose_snap(0, pitch, [target], None, radius_deg=0.1, origin_world_m=(0, 0, -0.4))
        == "a"
    )
    assert (
        world.choose_snap(
            0,
            pitch,
            [replace(target, distance_m=None)],
            None,
            radius_deg=0.1,
            origin_world_m=(0, 0, -0.4),
        )
        is None
    )


def test_visual_alignment_preserves_landmark_north():
    a = Alignment(
        north_offset_deg=63,
        north_source="landmark",
        aligned_at=1,
        phone_to_tracking_quat=world.quat_from_yaw_pitch_roll(10, 0),
    )
    assert world.pointer_ray(IDENTITY, a) == pytest.approx((73, 0))
    b = world.align_north_to_landmark(a, 10, 90)
    assert b.phone_to_tracking_quat == a.phone_to_tracking_quat
    assert world.pointer_ray(IDENTITY, b) == pytest.approx((90, 0))


def test_app_camera_to_gyro_alignment_and_loss_reacquisition_with_moving_head():
    from test_app import Clock, FakeGlasses, FakePhone, StaticLayer

    from pointview.app import PointViewApp
    from pointview.interfaces import Pose

    def matrix(q):
        return np.array([world.quat_rotate(q, v) for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]).T

    basis = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
    base_rot = cv2.Rodrigues(np.array([2.65, 0.15, 0.1]))[0]
    base = rotation_quat(basis @ base_rot)
    samples = [
        replace(s, tracking_enabled=True, tracking_session=TARGET.session, marker_size_m=0.08)
        for s in motion_samples(rate=0.6, axis=(0, 1, 0), reference=base)
    ]
    reference = world.quat_from_yaw_pitch_roll(8, 0)

    class Glasses(FakeGlasses):
        frame = None
        camera_intrinsics = INTR

        def pose_at(self, t):
            return Pose(t, world.quat_from_yaw_pitch_roll(15 * t, 3 * t))

        def latest_frame(self):
            return self.frame

    clock = Clock()
    phone = FakePhone(clock)
    phone.tracking_target = lambda: TARGET
    glasses = Glasses(0)
    app = PointViewApp(glasses, [StaticLayer([])], phone, None, clock=clock)
    for t in np.arange(0.10, 0.81, 0.05):
        clock.t = float(t)
        phone.state = samples[round(t * 100)]
        phone.past = [s for s in samples if s.t < phone.state.t]
        exposed = float(t - 0.04)
        q = world.quat_mul(reference, sample_at(samples, exposed).quat)
        camera_rot = basis.T @ matrix(glasses.pose_at(exposed).quat).T @ matrix(q)
        glasses.frame, _, _ = synthetic_frame(rotation=camera_rot, t=exposed)
        app.tick()
    assert app.alignment.aligned_at is not None
    assert app.alignment.north_source == "none"
    assert rotation_error(app.alignment.phone_to_tracking_quat, reference) < 2
    reference_before = app.alignment.phone_to_tracking_quat
    assert any("tracking" in m for m in phone.sent)
    # Camera disappears: pointer still uses current phone rotation, no translation is reused.
    clock.t = 0.9
    phone.state = samples[90]
    glasses.frame = None
    app.tick()
    assert app.tracker.state(clock.t) == "coasting"
    assert app.tracker.origin_world(clock.t, 0) is None
    assert app.alignment.phone_to_tracking_quat == reference_before
    # A new connection cannot use the old visual reference or old marker acknowledgment.
    phone.tracking_target = lambda: TrackingTarget("b" * 32, 17)
    app.tick()
    assert app.alignment.phone_to_tracking_quat is None
    assert app.alignment.aligned_at is None


def test_update_rejects_stale_frames_and_only_processes_each_exposure_once():
    from test_app import FakeGlasses

    glasses = FakeGlasses(0)
    frame, _, _ = synthetic_frame(t=0.2)
    glasses.latest_frame = lambda: frame
    tracker = PhoneTracker(TARGET)
    samples = motion_samples()
    tracker.update(glasses, samples, Calibration(), 0.08, 0.22)
    count = len(tracker.observations)
    assert not tracker.update(glasses, samples, Calibration(), 0.08, 0.23)
    assert len(tracker.observations) == count
    assert not tracker.update(glasses, samples, Calibration(), 0.08, 0.6)
    assert not tracker.verified
    assert not tracker.observations
