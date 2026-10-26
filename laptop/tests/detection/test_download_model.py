from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "download_model.py"


@pytest.fixture(scope="module")
def dl():
    spec = importlib.util.spec_from_file_location("download_model", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pins(dl):
    assert dl.MODEL_URL.startswith("https://huggingface.co/onnx-community/")
    assert "/resolve/" in dl.MODEL_URL and "/main/" not in dl.MODEL_URL  # pinned revision
    assert len(dl.MODEL_SHA256) == 64
    assert (
        dl.DEFAULT_DEST.name == "rtdetr_v2_r18vd.onnx" and dl.DEFAULT_DEST.parent.name == "models"
    )


def test_download_verifies_hash_and_is_idempotent(dl, tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"pretend model")
    sha = hashlib.sha256(b"pretend model").hexdigest()
    dest = tmp_path / "out" / "m.onnx"
    assert dl.download(dest, src.as_uri(), sha) == dest
    assert dest.read_bytes() == b"pretend model"
    src.unlink()  # a second call must not need the network/source at all
    assert dl.download(dest, src.as_uri(), sha) == dest


def test_hash_mismatch_raises_and_leaves_nothing(dl, tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"tampered")
    dest = tmp_path / "m.onnx"
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        dl.download(dest, src.as_uri(), "0" * 64)
    assert not dest.exists() and not list(tmp_path.glob("*.part"))


def test_corrupt_existing_file_is_replaced(dl, tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"good")
    dest = tmp_path / "m.onnx"
    dest.write_bytes(b"corrupt")
    dl.download(dest, src.as_uri(), hashlib.sha256(b"good").hexdigest())
    assert dest.read_bytes() == b"good"
