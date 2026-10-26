"""``--mock`` mode: everything runs on this machine with no hardware and no network.

* Glasses: MockGlasses playing tests/fixtures/sample.mp4 with the synthetic pose track (or ``--video``).
* Phone: tools/fake_phone.py runs in-process and connects to the real PhoneLink over localhost.
* Landmarks: the real LandmarkLayer, served by the landmark test fixtures (Crissy Field, San Francisco)
  unless ``live_landmarks`` is set.

The fake phone runs a scripted tour: Align at 1 s, then point at the Golden Gate Bridge, open its info
card, sweep away and come back.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType

from websockets.exceptions import ConnectionClosedOK

from pointview import world
from pointview.interfaces import CameraIntrinsics

LAPTOP_DIR = Path(__file__).resolve().parents[1]
FIXTURES = LAPTOP_DIR / "tests" / "fixtures"
FAKE_PHONE_PATH = LAPTOP_DIR / "tools" / "fake_phone.py"

#: Landmark fixture position (tests/fixtures/landmarks/README.md) and the tour's target.
MOCK_LAT, MOCK_LON = 37.8060, -122.4680
TARGET_LAT, TARGET_LON = 37.81972222, -122.47861111  # Golden Gate Bridge
#: The phone yaw at which the pointer hits the target. Non-zero so Align is really exercised.
TARGET_PHONE_YAW = -12.0


def load_fake_phone() -> ModuleType:
    if "fake_phone" in sys.modules:
        return sys.modules["fake_phone"]
    spec = importlib.util.spec_from_file_location("fake_phone", FAKE_PHONE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fake_phone"] = mod
    spec.loader.exec_module(mod)
    return mod


def fixture_intrinsics() -> CameraIntrinsics:
    i = json.loads((FIXTURES / "scene.json").read_text())["intrinsics"]
    return CameraIntrinsics(
        i["fx"], i["fy"], i["cx"], i["cy"], i["width"], i["height"], tuple(i["dist"])
    )


def make_glasses(video: Path | None = None, realtime: bool = True):
    from pointview.glasses import MockGlasses

    return MockGlasses(
        video or FIXTURES / "sample.mp4",
        FIXTURES / "pose_track.csv",
        fixture_intrinsics(),
        realtime=realtime,
    )


def landmark_client():
    """httpx client answering Overpass/Wikipedia from the landmark fixtures (no network)."""
    if str(LAPTOP_DIR) not in sys.path:
        sys.path.insert(0, str(LAPTOP_DIR))
    from tests.fixtures.landmarks.mock_transport import FixtureServer

    return FixtureServer().client()


def tour_north_offset() -> float:
    """Fake compass offset so the pointer lands on the target at TARGET_PHONE_YAW.

    After Align, pointer world yaw = phone yaw + this offset (see docs/contracts.md §2).
    """
    return world.wrap_deg(
        world.bearing_deg(MOCK_LAT, MOCK_LON, TARGET_LAT, TARGET_LON) - TARGET_PHONE_YAW
    )


def tour_script():
    fp = load_fake_phone()
    k, y = fp.Keyframe, TARGET_PHONE_YAW
    return fp.Script(
        [
            k(0.0, 0.0),
            k(1.0, 0.0, buttons=("align",)),
            k(2.5, y),
            k(3.5, y, buttons=("info",)),
            k(6.0, y),
            k(6.5, y, buttons=("info",)),
            k(8.0, y + 25.0),
            k(9.5, y),
            k(10.0, y, buttons=("info",)),
            k(30.0, y),
        ],
        "mock_tour",
    )


class FakePhoneThread:
    """Runs the fake phone on its own asyncio loop in a daemon thread."""

    def __init__(self, url: str, script=None, duration_s: float = 3600.0) -> None:
        self.url = url
        self.script = script or tour_script()
        self.duration_s = duration_s
        self.stats = None
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="fake-phone", daemon=True)

    def _run(self) -> None:
        fp = load_fake_phone()
        try:
            self.stats = asyncio.run(
                fp.run_fake_phone(
                    self.url,
                    self.script,
                    duration_s=self.duration_s,
                    lat=MOCK_LAT,
                    lon=MOCK_LON,
                    north_offset_deg=tour_north_offset(),
                )
            )
        except ConnectionClosedOK:
            pass  # the laptop closed the link cleanly (app shutdown): a normal end of the run
        except BaseException as e:  # reported by the caller; the thread must not die silently
            self.error = e

    def start(self) -> None:
        self._thread.start()
