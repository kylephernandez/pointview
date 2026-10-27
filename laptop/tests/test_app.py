"""PointViewApp.tick() with fake glasses, phone and layer (no Qt, no threads)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pointview import world
from pointview.app import PointViewApp
from pointview.entities import Entity
from pointview.interfaces import (
    LAYER_KINDS,
    Backdrop,
    ButtonEvent,
    CameraIntrinsics,
    Crosshair,
    Frame,
    InfoCard,
    PhoneState,
    WireMesh,
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


# ---------------------------------------------------------------------------- v2: live + looks


def test_live_entities_only_inside_glasses_view(rig):
    app, phone, layer, _ = rig
    near = Entity("ac:1", "aircraft", "UAL1", HEAD_YAW + 10.0, 2.0, 0.0)  # in view (north 0)
    behind = Entity("ac:2", "aircraft", "UAL2", HEAD_YAW + 180.0, 2.0, 0.0)
    layer.entities = [near, behind, TOWER]  # the tower is outside view but not live: kept
    phone.point(0.0)
    app.tick()
    assert {e.id for e in app.entities} == {"ac:1", TOWER.id}
    assert layer.last_ctx.view is not None and abs(layer.last_ctx.view.yaw_deg - HEAD_YAW) < 1e-6


def test_focus_id_and_pointer_reach_layers(rig):
    app, phone, layer, clock = rig
    phone.point(0.0, heading=90.0)
    phone.press("align")
    app.tick()
    phone.point(10.0)  # tower
    clock.t += 0.05
    app.tick()
    clock.t += 0.05
    app.tick()
    assert layer.last_ctx.focus_id == TOWER.id
    assert layer.last_ctx.pointer is not None


class FramedGlasses(FakeGlasses):
    """FakeGlasses with a front camera frame and grayscale tracking frames."""

    tracking_camera_intrinsics = CameraIntrinsics(300, 300, 160, 120, 320, 240)
    tracking_camera_to_head_quat = (1.0, 0.0, 0.0, 0.0)

    def latest_frame(self):
        return Frame(99.0, np.full((480, 640, 3), 80, np.uint8))

    def latest_tracking_frame(self):
        return Frame(99.0, np.full((240, 320, 3), 200, np.uint8))

    def set_tracking_exposure(self, exposure_ms, gain=None) -> None: ...


@pytest.mark.parametrize(("look", "height"), [("normal", 480), ("nvg", 240)])
def test_passthrough_backdrop_uses_the_right_viture_camera(look, height):
    clock = Clock()
    phone = FakePhone(clock)
    app = PointViewApp(FramedGlasses(0.0), [], phone, None, clock=clock)
    phone.point(0.0)
    phone.state = replace(phone.state, look=look, passthrough=True)
    items = app.tick()
    assert isinstance(items[0], Backdrop) and items[0].bgr.shape[0] == height
    # Same pose at capture and now: the camera centre lands on the display centre.
    h = np.array(items[0].homography).reshape(3, 3)
    cx, cy = (
        (
            FramedGlasses.camera_intrinsics
            if look == "normal"
            else FramedGlasses.tracking_camera_intrinsics
        ).cx,
        (
            FramedGlasses.camera_intrinsics
            if look == "normal"
            else FramedGlasses.tracking_camera_intrinsics
        ).cy,
    )
    p = h @ np.array([cx, cy, 1.0])
    assert np.allclose(p[:2] / p[2], (960, 540), atol=1e-6)
    status = [m for m in phone.sent if m["type"] == "status"]
    assert status[-1]["look"] == look and status[-1]["passthrough"] is True


def test_no_backdrop_without_passthrough():
    clock = Clock()
    phone = FakePhone(clock)
    app = PointViewApp(FramedGlasses(0.0), [], phone, None, clock=clock)
    phone.point(0.0)
    assert not any(isinstance(i, Backdrop) for i in app.tick())


# ------------------------------------------------------------------------------ v2.1: world mesh


class GridMeshSource:
    """A flat 20 m grid of lines around the origin, standing in for the OSM/terrain source."""

    def __init__(self) -> None:
        from pointview.entities import GeoPos
        from pointview.interfaces import WorldMesh

        xs = np.arange(-200, 201, 20, dtype=np.float32)
        verts, edges = [], []
        for x in xs:
            for a, b in (((x, -200), (x, 200)), ((-200, x), (200, x))):
                i = len(verts)
                verts += [(*a, 0.0), (*b, 0.0)]
                edges.append((i, i + 1))
        self.mesh = WorldMesh(
            GeoPos(37.8, -122.4, 0.0),
            np.array(verts, np.float32),
            np.array(edges, np.int32),
            "test-grid",
            0.0,
        )
        self.calls = 0

    def mesh_near(self, lat, lon):
        self.calls += 1
        return self.mesh

    def ground_alt_m(self, lat, lon):
        return 0.0

    def close(self) -> None: ...


def test_world_mesh_after_align_with_markers_on_top(rig):
    app, phone, _, clock = rig
    app.mesh_source = GridMeshSource()
    level = FakeGlasses(HEAD_YAW)
    level.quat = world.quat_from_yaw_pitch_roll(HEAD_YAW, -5.0)  # ground grid out to the horizon
    app.glasses = level
    phone.point(0.0, heading=90.0)
    assert not any(isinstance(i, WireMesh) for i in app.tick())  # not aligned yet: no mesh
    phone.press("align")
    clock.t += 0.05
    items = app.tick()
    mesh = next(i for i in items if isinstance(i, WireMesh))
    assert len(mesh.segments) > 10 and mesh.alpha is not None
    assert items.index(mesh) == 0  # under everything
    assert app.stats.mesh_segments_max == len(mesh.segments)
    phone.state = replace(phone.state, layers={**phone.state.layers, "mesh": False})
    assert not any(isinstance(i, WireMesh) for i in app.tick())  # toggled off on the phone
