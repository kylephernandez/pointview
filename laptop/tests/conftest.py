"""Shared fixtures (FROZEN — see docs/contracts.md). Module tests add their own conftest in a subdir."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from pointview.interfaces import CameraIntrinsics, Pose

FIXTURES = Path(__file__).parent / "fixtures"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("POINTVIEW_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="live network test; set POINTVIEW_NETWORK_TESTS=1")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def scene() -> dict:
    """tests/fixtures/scene.json: intrinsics, video timing, marker directions."""
    return json.loads((FIXTURES / "scene.json").read_text())


@pytest.fixture(scope="session")
def sample_video_path() -> Path:
    """960x540 @ 30 fps, 6 s. Frame i shows the synthetic world at pose_track time i/30."""
    return FIXTURES / "sample.mp4"


@pytest.fixture(scope="session")
def fixture_intrinsics(scene) -> CameraIntrinsics:
    i = scene["intrinsics"]
    return CameraIntrinsics(
        i["fx"], i["fy"], i["cx"], i["cy"], i["width"], i["height"], tuple(i["dist"])
    )


@pytest.fixture(scope="session")
def pose_track() -> list[Pose]:
    """200 Hz synthetic head poses for 10 s; t starts at 0.0 (seconds)."""
    with (FIXTURES / "pose_track.csv").open() as f:
        return [
            Pose(float(r["t"]), (float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])))
            for r in csv.DictReader(f)
        ]


@pytest.fixture(scope="session")
def phone_messages() -> list[dict]:
    """Valid phone→laptop messages (docs/phone-protocol.md), one per line."""
    lines = (FIXTURES / "phone_messages.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines if x.strip()]
