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
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import httpx
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


#: Real adsb.lol response around MOCK_LAT/LON, recorded 2026-10-27 (tests/fixtures/live/mock/).
LIVE_FIXTURES = FIXTURES / "live"
MOCK_AIRCRAFT = LIVE_FIXTURES / "mock" / "adsblol_sf.json"


@dataclass(frozen=True)
class Tour:
    """Where the fake phone points after Align: a world direction (yaw, pitch)."""

    name: str
    yaw_deg: float
    pitch_deg: float = 0.0


def landmark_tour() -> Tour:
    return Tour("landmarks", world.bearing_deg(MOCK_LAT, MOCK_LON, TARGET_LAT, TARGET_LON))


def recording_unix_time() -> float:
    """The moment the mock aircraft were recorded; the mock's wall clock starts here."""
    return json.loads(MOCK_AIRCRAFT.read_text())["now"] / 1000.0


def aircraft_tour() -> Tour:
    """Point at the nearest airborne aircraft in the recording that sits 3–12° up."""
    best = None
    for a in json.loads(MOCK_AIRCRAFT.read_text())["ac"]:
        alt = a.get("alt_geom", a.get("alt_baro"))
        if not isinstance(alt, int | float) or alt < 500 or "lat" not in a:
            continue
        yaw, pitch, rng = world.look_angles(
            MOCK_LAT, MOCK_LON, 0.0, a["lat"], a["lon"], alt * 0.3048
        )
        if 3.0 <= pitch <= 12.0 and (best is None or rng < best[2]):
            best = (yaw, pitch, rng)
    if best is None:
        raise RuntimeError(f"no suitable aircraft in {MOCK_AIRCRAFT}")
    return Tour("aircraft", best[0], best[1])


def tour_north_offset(tour: Tour) -> float:
    """Fake compass offset so the pointer lands on the target at TARGET_PHONE_YAW.

    After Align, pointer world yaw = phone yaw + this offset (see docs/contracts.md §2).
    """
    return world.wrap_deg(tour.yaw_deg - TARGET_PHONE_YAW)


def tour_script(tour: Tour):
    fp = load_fake_phone()
    k, y, p = fp.Keyframe, TARGET_PHONE_YAW, tour.pitch_deg
    return fp.Script(
        [
            k(0.0, 0.0),
            k(1.0, 0.0, buttons=("align",)),
            k(2.5, y, p),
            k(3.5, y, p, buttons=("info",)),
            k(6.0, y, p),
            k(6.5, y, p, buttons=("info",)),
            k(8.0, y + 25.0, p),
            k(9.5, y, p),
            k(10.0, y, p, buttons=("info",)),
            k(30.0, y, p),
        ],
        f"mock_tour_{tour.name}",
    )


class _FixtureFeeds:
    """Serves the live layers' HTTP requests from recordings (no network in --mock)."""

    def __init__(self) -> None:
        self.aircraft = MOCK_AIRCRAFT.read_bytes()
        sats = LIVE_FIXTURES / "satellites"
        self.tle = {g: (sats / f"tle_{g}.txt").read_bytes() for g in ("stations", "visual")}
        self.quakes = (LIVE_FIXTURES / "events" / "summary_2.5_day.geojson").read_bytes()

    def handle(self, request: httpx.Request) -> httpx.Response:
        host, url = request.url.host, str(request.url)
        if host == "api.adsb.lol" and "/v2/point/" in url:
            return httpx.Response(200, content=self.aircraft)
        if host == "celestrak.org":
            group = request.url.params.get("GROUP", "")
            if group in self.tle:
                return httpx.Response(200, content=self.tle[group])
        if host == "earthquake.usgs.gov" and url.endswith("summary/2.5_day.geojson"):
            return httpx.Response(200, content=self.quakes)
        return httpx.Response(404)


def live_client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(_FixtureFeeds().handle))


class MockWallClock:
    """Unix time that starts at the recording time and runs in real time."""

    def __init__(self, start_unix: float | None = None) -> None:
        self.start_unix = recording_unix_time() if start_unix is None else start_unix
        self._t0 = time.monotonic()

    def __call__(self) -> float:
        return self.start_unix + (time.monotonic() - self._t0)


class FakePhoneThread:
    """Runs the fake phone on its own asyncio loop in a daemon thread."""

    def __init__(self, url: str, tour: Tour | None = None, duration_s: float = 3600.0) -> None:
        self.url = url
        self.tour = tour or landmark_tour()
        self.script = tour_script(self.tour)
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
                    north_offset_deg=tour_north_offset(self.tour),
                )
            )
        except ConnectionClosedOK:
            pass  # the laptop closed the link cleanly (app shutdown): a normal end of the run
        except BaseException as e:  # reported by the caller; the thread must not die silently
            self.error = e

    def start(self) -> None:
        self._thread.start()
