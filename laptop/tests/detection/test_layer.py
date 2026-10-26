from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from pointview.detection import DetectionLayer
from pointview.detection.model import Detection
from pointview.interfaces import (
    IDENTITY_QUAT,
    Alignment,
    Calibration,
    CameraIntrinsics,
    Frame,
    Layer,
    LayerContext,
    Pose,
)

from . import _world_ref as ref

INTR = CameraIntrinsics(fx=480.0, fy=480.0, cx=480.0, cy=270.0, width=960, height=540)
CENTER_BOX = (430.0, 220.0, 530.0, 320.0)


class FakeGlasses:
    camera_intrinsics = INTR
    display_size = (1920, 1080)
    display_fov_deg = (46.0, 27.0)

    def __init__(self):
        self.frame: Frame | None = None
        self.pose_requests: list[float] = []
        self.yaw_of_t = lambda t: 0.0

    def start(self): ...
    def stop(self): ...

    def latest_frame(self):
        return self.frame

    def pose_at(self, t):
        self.pose_requests.append(t)
        return Pose(t, ref.quat_from_yaw_pitch(self.yaw_of_t(t)))


class FakeDetector:
    def __init__(self):
        self.result: list[Detection] = []
        self.calls = 0
        self.delay = 0.0
        self.threads: set[str] = set()

    def detect(self, bgr):
        self.calls += 1
        self.threads.add(threading.current_thread().name)
        time.sleep(self.delay)
        return list(self.result)


def car(box=CENTER_BOX) -> Detection:
    return Detection("car", "vehicle", 0.9, box)


def person(box) -> Detection:
    return Detection("person", "person", 0.8, box)


class Rig:
    def __init__(self, **kw):
        self.glasses = FakeGlasses()
        self.detector = FakeDetector()
        self.layer = DetectionLayer(
            Path("unused.onnx"), max_fps=1000.0, detector=self.detector, **kw
        )
        self.alignment = Alignment()
        self.enabled = frozenset({"person", "vehicle", "landmark"})
        self.n = 0

    def ctx(self, t: float) -> LayerContext:
        return LayerContext(
            t, self.glasses, self.alignment, Calibration(), None, enabled=self.enabled
        )

    def step(self, t: float):
        """Deliver a frame captured at t and wait until the worker has processed it."""
        self.n += 1
        self.glasses.frame = Frame(t, np.zeros((540, 960, 3), np.uint8))
        deadline = time.monotonic() + 5
        while self.detector.calls < self.n:
            self.layer.update(self.ctx(t))
            assert time.monotonic() < deadline, "worker did not process the frame"
            time.sleep(0.002)
        while self.layer.frames_processed < self.n:
            assert time.monotonic() < deadline
            time.sleep(0.002)
        return self.layer.update(self.ctx(t))


@pytest.fixture
def rig():
    r = Rig()
    yield r
    r.layer.close()


def test_is_a_layer():
    layer = DetectionLayer(Path("x"), detector=FakeDetector())
    assert isinstance(layer, Layer) and layer.name == "detection"
    layer.close()


def test_three_hits_then_entity_with_world_direction(rig):
    rig.detector.result = [car()]
    rig.glasses.yaw_of_t = lambda t: 40.0
    assert rig.step(1.00) == []
    assert rig.step(1.07) == []
    (e,) = rig.step(1.14)
    assert e.id.startswith("det:") and e.kind == "vehicle" and e.label == "car"
    assert e.world_yaw_deg == pytest.approx(40.0, abs=1e-6)
    assert e.world_pitch_deg == pytest.approx(0.0, abs=1e-6)
    assert e.updated_at == 1.14 and e.confidence == 0.9
    assert e.distance_m == pytest.approx(480 * 1.5 / 100)
    assert e.size_deg is not None and e.size_deg.w > 0


def test_uses_capture_time_pose_and_alignment(rig):
    rig.detector.result = [car()]
    rig.glasses.yaw_of_t = lambda t: 100.0 * t  # head turns fast: pose depends on exact t
    rig.alignment = Alignment(north_offset_deg=15.0)
    for t in (0.10, 0.17, 0.24):
        out = rig.step(t)
    assert 0.24 in rig.glasses.pose_requests
    assert out[0].world_yaw_deg == pytest.approx(100.0 * 0.24 + 15.0, abs=1e-6)


def test_hold_300ms_keeps_last_world_direction_then_expires(rig):
    rig.detector.result = [car()]
    rig.glasses.yaw_of_t = lambda t: 20.0 if t < 1.2 else 80.0  # the head turns after the loss
    for t in (1.0, 1.07, 1.14):
        out = rig.step(t)
    (e0,) = out
    rig.detector.result = []
    (held,) = rig.step(1.25)  # lost, head now elsewhere
    assert held.id == e0.id
    assert held.world_yaw_deg == pytest.approx(20.0, abs=1e-6)  # last world direction kept
    assert held.updated_at == 1.14
    # update() alone (no new frame) expires it by laptop time.
    assert rig.layer.update(rig.ctx(1.14 + 0.29)) != []
    assert rig.layer.update(rig.ctx(1.14 + 0.31)) == []
    assert rig.step(1.50) == []


def test_ids_stable_across_frames(rig):
    rig.detector.result = [car()]
    for t in (0.0, 0.07, 0.14):
        out = rig.step(t)
    (e,) = out
    rig.detector.result = [car((435.0, 222.0, 535.0, 322.0))]
    (e2,) = rig.step(0.21)
    assert e2.id == e.id


def test_respects_enabled_kinds(rig):
    rig.detector.result = [car(), person((100.0, 100.0, 140.0, 220.0))]
    for t in (0.0, 0.07, 0.14):
        out = rig.step(t)
    assert {e.kind for e in out} == {"vehicle", "person"}
    rig.enabled = frozenset({"person"})
    assert {e.kind for e in rig.layer.update(rig.ctx(0.14))} == {"person"}
    rig.enabled = frozenset()
    assert rig.layer.update(rig.ctx(0.14)) == []


def test_update_is_fast_and_inference_runs_on_its_own_thread(rig):
    rig.detector.delay = 0.2  # slow "model"
    rig.detector.result = [car()]
    rig.glasses.frame = Frame(0.0, np.zeros((540, 960, 3), np.uint8))
    worst = 0.0
    t_end = time.monotonic() + 0.5
    while time.monotonic() < t_end:
        t0 = time.perf_counter()
        rig.layer.update(rig.ctx(0.0))
        worst = max(worst, time.perf_counter() - t0)
        time.sleep(0.005)
    assert worst < 0.005
    assert rig.detector.calls >= 1
    assert threading.current_thread().name not in rig.detector.threads


def test_same_frame_is_not_processed_twice_and_no_frame_means_no_work(rig):
    rig.layer.update(rig.ctx(0.0))
    time.sleep(0.05)
    assert rig.detector.calls == 0  # latest_frame() is None
    rig.step(0.5)
    time.sleep(0.05)
    assert rig.detector.calls == 1


def test_max_fps_throttle():
    glasses, det = FakeGlasses(), FakeDetector()
    layer = DetectionLayer(Path("x"), max_fps=20.0, detector=det)
    try:
        n = 0
        t_end = time.monotonic() + 0.5
        while time.monotonic() < t_end:
            n += 1
            glasses.frame = Frame(n * 0.001, np.zeros((540, 960, 3), np.uint8))  # always new
            layer.update(LayerContext(0.0, glasses, Alignment(), Calibration(), None))
            time.sleep(0.002)
        assert 5 <= det.calls <= 13  # about 10 in 0.5 s at 20 fps
    finally:
        layer.close()


def test_worker_survives_detector_errors(rig):
    class Boom(FakeDetector):
        def detect(self, bgr):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return [car()]

    rig.layer.close()
    rig = Rig()
    rig.detector = Boom()
    rig.layer = DetectionLayer(Path("x"), max_fps=1000.0, detector=rig.detector)
    try:
        rig.glasses.frame = Frame(0.0, np.zeros((540, 960, 3), np.uint8))
        rig.layer.update(rig.ctx(0.0))
        deadline = time.monotonic() + 3
        while rig.layer.error is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert isinstance(rig.layer.error, RuntimeError)
        assert rig.detector.calls >= 1
        rig.glasses.frame = Frame(0.07, np.zeros((540, 960, 3), np.uint8))
        deadline = time.monotonic() + 3
        while rig.detector.calls < 2 and time.monotonic() < deadline:
            rig.layer.update(rig.ctx(0.07))
            time.sleep(0.005)
        assert rig.detector.calls >= 2
    finally:
        rig.layer.close()


def test_close_stops_thread_and_clears(rig):
    rig.detector.result = [car()]
    for t in (0.0, 0.07, 0.14):
        rig.step(t)
    rig.layer.close()
    assert rig.layer._thread is not None and not rig.layer._thread.is_alive()
    assert rig.layer.update(rig.ctx(0.14)) == []


def test_identity_calibration_default_is_identity():
    assert Calibration().camera_to_head_quat == IDENTITY_QUAT
