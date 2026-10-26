"""End-to-end: ``python -m pointview --mock`` for 10 s, as a real subprocess.

MockGlasses + the real DetectionLayer, LandmarkLayer (fixture HTTP), PhoneLink + fake phone over
localhost, world math, scene and the Qt renderer (offscreen). Screenshots go to test-output/e2e*/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

LAPTOP = Path(__file__).resolve().parents[1]
OUT = LAPTOP / "test-output"


def run_mock(tmp_path: Path, shots: Path, *extra: str, duration: float = 10.0) -> dict:
    stats_path = tmp_path / "stats.json"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    cmd = [sys.executable, "-m", "pointview", "--mock", "--duration", str(duration)]
    cmd += ["--screenshot-dir", str(shots), "--stats-json", str(stats_path), *extra]
    proc = subprocess.run(
        cmd, cwd=LAPTOP, env=env, capture_output=True, text=True, timeout=duration + 60
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "fake phone failed" not in proc.stderr, proc.stderr[-3000:]
    return json.loads(stats_path.read_text())


def test_mock_run_10s(tmp_path):
    shots = OUT / "e2e"
    for old in shots.glob("*.png"):
        old.unlink()
    stats = run_mock(tmp_path, shots)

    assert stats["frames"] > 300, stats  # ~60 fps target; generous for slow CI
    assert stats["aligns"] == 1
    assert "landmark" in stats["kinds_seen"] and stats["max_entities"] >= 3
    assert stats["snaps"] >= 1
    assert stats["info_frames"] > 0 and "Golden Gate Bridge" in stats["info_titles"]
    assert stats["buttons"].count("info") >= 2

    card = shots / "info_card.png"
    assert card.exists() and (shots / "last.png").exists()
    img = cv2.imread(str(card))
    lit = np.count_nonzero(img.max(axis=2) > 40)
    assert lit > 2000  # a card, labels and a ring were actually drawn on the black background
    assert len(list(shots.glob("frame_*.png"))) >= 4


@pytest.mark.model
def test_mock_detection_pipeline(tmp_path, fixtures_dir):
    """Real model on a street photo played as the glasses' video → person/vehicle entities."""
    model = LAPTOP / "models" / "rtdetr_v2_r18vd.onnx"
    if not model.exists():
        pytest.skip("model not downloaded: python tools/download_model.py")
    photo = cv2.imread(str(fixtures_dir / "detection" / "street_porlamar.jpg"))
    frame = cv2.resize(photo, (960, 540))
    video = tmp_path / "street.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 540))
    for _ in range(90):
        vw.write(frame)
    vw.release()

    stats = run_mock(
        tmp_path, OUT / "e2e_detection", "--video", str(video), "--no-landmarks", duration=8
    )
    assert {"person", "vehicle"} <= set(stats["kinds_seen"]), stats
