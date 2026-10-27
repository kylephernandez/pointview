"""Detection test setup."""

from __future__ import annotations

from pathlib import Path

import pytest

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "rtdetr_v2_r18vd.onnx"


@pytest.fixture(scope="session")
def model_path() -> Path:
    if not MODEL_PATH.exists():
        pytest.skip("model not downloaded: python tools/download_model.py")
    return MODEL_PATH
