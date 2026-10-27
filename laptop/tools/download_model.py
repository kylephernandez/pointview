"""Download the pinned COCO detector (RT-DETR v2 R18, ONNX) to laptop/models/.

    python tools/download_model.py            # downloads if missing or the hash differs
    python tools/download_model.py --force    # re-download

The URL is pinned to an exact Hugging Face commit and the sha256 is verified, so the file is
reproducible. Idempotent: an existing file with the right hash is left alone. Licenses: see
laptop/models/README.md (Apache-2.0 weights and code).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

MODEL_NAME = "rtdetr_v2_r18vd"
MODEL_URL = (
    "https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX/resolve/"
    "936f90b6a476c6da4dfe053fc521af55285976ba/onnx/model.onnx"
)
MODEL_SHA256 = "583a236ac21c95a7fd94f284fc21485e42355bfef82c27011ba78fbc09ee87e2"
MODEL_SIZE = 81_057_510

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_DEST = MODELS_DIR / f"{MODEL_NAME}.onnx"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(
    dest: Path = DEFAULT_DEST,
    url: str = MODEL_URL,
    sha256: str = MODEL_SHA256,
    force: bool = False,
) -> Path:
    """Fetch ``url`` to ``dest`` unless it is already there with the right hash. Raises on mismatch."""
    dest = Path(dest)
    if dest.exists() and not force and sha256_of(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        got = sha256_of(tmp)
        if got != sha256:
            raise RuntimeError(f"sha256 mismatch for {url}: expected {sha256}, got {got}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--force", action="store_true", help="download again even if present")
    args = ap.parse_args(argv)
    path = download(args.dest, force=args.force)
    print(f"ok: {path} ({path.stat().st_size} bytes, sha256 {MODEL_SHA256})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
