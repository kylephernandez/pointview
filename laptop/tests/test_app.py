"""PointViewApp.tick() with fake glasses, phone and layer (no Qt, no threads)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pointview import world
from pointview.app import PointViewApp
from pointview.entities import Entity
from pointview.interfaces import (
    LAYER_KINDS,
    ButtonEvent,
    CameraIntrinsics,
    Crosshair,
    InfoCard,
    PhoneState,
)
from pointview.interfaces import Pose as _Pose


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class FakeGlasses:
    camera_intrinsics = CameraIntrinsics(500, 500, 320, 240, 640, 480)
    display_size = (1920, 1080)
    display_fov_deg = (46.0, 27.0)

    def __init__(self, head_yaw: float) -> None:
        self.quat = world.quat_from_yaw_pitch_roll(head_yaw, 0.0)

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest_frame(self):
        return None

    def pose_at(self, t):
        return _Pose(t, self.quat)


class FakePhone:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.state: PhoneState | None = None
        self.queue: list[ButtonEvent] = []
        self.sent: list[dict] = []

    def point(self, yaw: float, heading: float | None = None, **layers) -> None:
        t = self.clock()
        self.state = PhoneState(
            t_ms=t * 1000,
            t=t,
            received_at=t,
            quat=world.quat_from_yaw_pitch_roll(yaw, 0.0),
            omega=(0.0, 0.0, 0.0),
            heading_deg=heading,
            heading_acc=5.0,
            lat=37.8,
            lon=-122.4,
            gps_acc=5.0,
            layers={"people": True, "vehicles": True, "landmarks": True, **layers},
        )

    def press(self, name: str) -> None:
        self.queue.append(ButtonEvent(name, self.clock() * 1000, self.clock()))

    def latest(self):
        return self.state

    def pop_buttons(self):
        q, self.queue = self.queue, []
        return q

    def connected(self) -> bool:
        return self.state is not None

    def stalled(self) -> bool:
        return False

    def send_nowait(self, message: dict) -> None:
        self.sent.append(message)

    def haptics(self) -> list[str]:
        return [m["style"] for m in self.sent if m["type"] == "haptic"]


class StaticLayer:
    name = "static"

    def __init__(self, entities: list[Entity]) -> None:
        self.entities = entities
        self.last_ctx = None

    def update(self, ctx):
        self.last_ctx = ctx
        return [e for e in self.entities if e.kind in ctx.enabled]

    def close(self) -> None: ...


HEAD_YAW = 30.0  # tracking frame
TOWER = Entity("osm:node/1", "landmark", "Tower", 100.0, 0.0, 0.0, distance_m=800)
CAR = Entity("det:1", "vehicle", "car", 95.0, 0.0, 0.0, distance_m=12)


@pytest.fixture
def rig():
    clock = Clock()
    phone = FakePhone(clock)
    layer = StaticLayer([TOWER, CAR])
    app = PointViewApp(FakeGlasses(HEAD_YAW), [layer], phone, None, clock=clock)
    return app, phone, layer, clock


def test_unaligned_shows_crosshair_and_no_pointer(rig):
    app, phone, _, _ = rig
    phone.point(0.0, heading=90.0)
    items = app.tick()
    assert any(isinstance(i, Crosshair) for i in items)
    assert app.snapped_id is None


def test_align_then_point_snaps_and_opens_info(rig):
    app, phone, _, clock = rig
    # Phone yaw 0 in its own frame while the head looks at tracking yaw 30; compass says 90°.
    phone.point(0.0, heading=90.0)
    phone.press("align")
    app.tick()
    assert app.alignment.aligned_at is not None and app.stats.aligns == 1
    assert phone.haptics() == ["align_ok"]
    # World yaw of the pointer is phone yaw + 90 now, so phone yaw 10 points at the tower (100°).
    clock.t += 0.1
    phone.point(10.0, heading=100.0)
    app.tick()
    assert app.snapped_id == TOWER.id and phone.haptics()[-1] == "snap"
    phone.press("info")
    clock.t += 0.1
    items = app.tick()
    (card,) = [i for i in items if isinstance(i, InfoCard)]
    assert card.title == "Tower" and phone.haptics()[-1] == "select"
    phone.press("info")
    clock.t += 0.1
    assert not [i for i in app.tick() if isinstance(i, InfoCard)]
    assert app.stats.snaps == 1


def test_hysteresis_keeps_selection_between_neighbours(rig):
    app, phone, _, clock = rig
    phone.point(0.0, heading=90.0)
    phone.press("align")
    app.tick()
    phone.point(10.0)  # tower at 100°
    app.tick()
    assert app.snapped_id == TOWER.id
    phone.point(7.2)  # 97.2°: car (95) is 2.2 closer? no: tower 2.8, car 2.2 → only 0.6 closer
    clock.t += 0.05
    app.tick()
    assert app.snapped_id == TOWER.id
    phone.point(5.0)  # 95°: car 0, tower 5 → switch
    clock.t += 0.05
    app.tick()
    assert app.snapped_id == CAR.id


def test_long_press_on_landmark_corrects_north(rig):
    app, phone, _, clock = rig
    phone.point(0.0, heading=85.0)  # compass is 5° off
    phone.press("align")
    app.tick()
    phone.point(14.0)  # world 99° with the bad north → nearest is the tower at 100°
    clock.t += 0.05
    app.tick()
    assert app.snapped_id == TOWER.id
    phone.press("align_long")
    clock.t += 0.05
    app.tick()
    assert app.alignment.north_source == "landmark"
    yaw, _ = world.pointer_ray(phone.state.quat, app.alignment)
    assert abs(world.wrap_deg_signed(yaw - TOWER.world_yaw_deg)) < 1e-6


def test_long_press_without_landmark_is_an_error(rig):
    app, phone, _, _ = rig
    phone.point(0.0, heading=90.0)
    phone.press("align")
    app.tick()
    phone.point(-90.0)  # pointing at nothing
    phone.press("align_long")
    app.tick()
    assert phone.haptics()[-1] == "error" and app.alignment.north_source == "compass"


def test_layer_toggles_reach_layers_and_status_is_sent(rig):
    app, phone, layer, _ = rig
    phone.point(0.0, heading=0.0, vehicles=False)
    app.tick()
    assert layer.last_ctx.enabled == frozenset(LAYER_KINDS.values()) - {"vehicle"}
    assert layer.last_ctx.unix_time > 1.7e9 and layer.last_ctx.view is not None
    assert [e.id for e in app.entities] == [TOWER.id]
    status = [m for m in phone.sent if m["type"] == "status"]
    assert status and status[0]["aligned"] is False


def test_card_closes_when_entity_disappears(rig):
    app, phone, layer, clock = rig
    phone.point(0.0, heading=90.0)
    phone.press("align")
    app.tick()
    phone.point(5.0)  # the car
    app.tick()
    phone.press("info")
    app.tick()
    assert app.info_id == CAR.id
    layer.entities = [TOWER]
    clock.t += 0.1
    app.tick()
    assert app.info_id is None


def test_stale_phone_hides_pointer(rig):
    app, phone, _, clock = rig
    phone.point(0.0, heading=90.0)
    phone.press("align")
    app.tick()
    phone.point(10.0)
    app.tick()
    assert app.snapped_id == TOWER.id
    phone.state = replace(phone.state, received_at=clock.t - 5)
    app.tick()
    assert app.snapped_id is None
