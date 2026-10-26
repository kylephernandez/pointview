"""Detection test setup: patch the stub pointview.world functions with the reference math."""

from __future__ import annotations

from pathlib import Path

import pytest

from pointview import world

from . import _world_ref as ref

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "rtdetr_v2_r18vd.onnx"


@pytest.fixture(autouse=True)
def patched_world(monkeypatch):
    """The lead drops this once W3's world.py is merged."""
    monkeypatch.setattr(world, "pixel_to_world", ref.pixel_to_world)
    monkeypatch.setattr(world, "camera_angles_deg", ref.camera_angles_deg)


@pytest.fixture(scope="session")
def model_path() -> Path:
    if not MODEL_PATH.exists():
        pytest.skip("model not downloaded: python tools/download_model.py")
    return MODEL_PATH
