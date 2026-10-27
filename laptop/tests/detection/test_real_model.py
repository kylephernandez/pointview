"""Runs the real ONNX model on a real photo. Skipped when the model file is not downloaded."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from pointview.detection import DetectionLayer, Detector
from pointview.interfaces import Alignment, Calibration, CameraIntrinsics, Frame, LayerContext, Pose

from . import _world_ref as ref

pytestmark = pytest.mark.model

PHOTO = Path(__file__).resolve().parents[1] / "fixtures" / "detection" / "street_porlamar.jpg"


@pytest.fixture(scope="module")
def photo() -> np.ndarray:
    img = cv2.imread(str(PHOTO))
    assert img is not None and img.shape[:2] == (557, 960)
    return img


@pytest.fixture(scope="module")
def detector(model_path) -> Detector:
    return Detector(model_path, providers=["CPUExecutionProvider"])


def test_finds_people_cars_and_a_motorcycle(detector, photo):
    dets = detector.detect(photo)
    counts = Counter(d.label for d in dets)
    assert counts["person"] >= 4
    assert counts["car"] + counts["truck"] >= 2  # the SUV may be called car or truck
    assert counts["motorcycle"] >= 1
    assert set(counts) <= {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
    # The big blue SUV on the right (about x 530-770, y 325-510 in the 960x557 photo).
    suv = [d for d in dets if d.kind == "vehicle" and d.box[2] - d.box[0] > 180]
    assert suv, "expected the large SUV"
    x1, y1, x2, y2 = suv[0].box
    assert abs(x1 - 530) < 30 and abs(x2 - 768) < 30 and abs(y1 - 326) < 30 and abs(y2 - 508) < 30
    # The man in the red/blue striped shirt in the middle-left.
    mid = [d for d in dets if d.label == "person" and 280 < d.box[0] < 320 and d.box[3] > 480]
    assert mid, "expected the man in the striped shirt"


def test_inference_latency_cpu_report(detector, photo, capsys):
    detector.detect(photo)  # warm up
    times = []
    for _ in range(8):
        t0 = time.perf_counter()
        detector.detect(photo)
        times.append(time.perf_counter() - t0)
    median_ms = sorted(times)[len(times) // 2] * 1000
    with capsys.disabled():
        print(f"\n[detection] CPU inference median: {median_ms:.0f} ms")
    assert median_ms < 2000  # sanity only; real numbers are in the report


def test_detection_layer_end_to_end_with_real_model(model_path, photo):
    intr = CameraIntrinsics(fx=700.0, fy=700.0, cx=480.0, cy=278.5, width=960, height=557)

    class Glasses:
        camera_intrinsics = intr
        display_size = (1920, 1080)
        display_fov_deg = (46.0, 27.0)
        frame = None

        def start(self): ...
        def stop(self): ...
        def latest_frame(self):
            return self.frame

        def pose_at(self, t):
            return Pose(t, ref.quat_from_yaw_pitch(90.0))

    g = Glasses()
    layer = DetectionLayer(model_path, providers=["CPUExecutionProvider"], max_fps=1000.0)
    try:
        ents = []
        deadline = time.monotonic() + 60
        t = 0.0
        while time.monotonic() < deadline and not ents:
            t += 0.07
            g.frame = Frame(t, photo)
            ctx = LayerContext(t, g, Alignment(), Calibration(), None)
            # Let the worker take this frame, then look at the snapshot.
            for _ in range(200):
                ents = layer.update(ctx)
                if layer.frames_processed >= round(t / 0.07):
                    break
                time.sleep(0.01)
            ents = layer.update(ctx)
        assert {"person", "vehicle"} <= {e.kind for e in ents}
        for e in ents:
            assert 0 <= e.world_yaw_deg < 360 and e.id.startswith("det:")
            # Camera looks east (yaw 90): everything in a 960 px / ~69 degree wide frame is near it.
            assert abs(e.world_yaw_deg - 90.0) < 40
    finally:
        layer.close()
