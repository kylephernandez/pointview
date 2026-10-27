"""The per-frame loop: glasses → layers → world → scene → renderer, plus the phone's Align/Info actions.

``PointViewApp.tick()`` is called at about 60 Hz by ``__main__`` (from a Qt timer). It has no Qt code of its
own, so tests can drive it directly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from pointview import landmarks, world
from pointview.entities import LIVE_KINDS, Entity
from pointview.interfaces import (
    IDENTITY_QUAT,
    LAYER_KINDS,
    Alignment,
    Backdrop,
    ButtonEvent,
    Calibration,
    Clock,
    DrawItem,
    GlassesSource,
    InfoCard,
    Layer,
    LayerContext,
    Look,
    MeshSource,
    PhoneState,
    Renderer,
    TrackingCameraSource,
    View,
    WireMesh,
    now,
)
from pointview.scene import InfoContent, SceneInput, Status, build_scene, info_for

log = logging.getLogger(__name__)

#: World mesh: how far it is drawn, how many edges at most (nearest kept), its look.
MESH_RANGE_M = 800.0
MESH_MAX_SEGMENTS = 8000
MESH_COLOR = (0, 190, 220)
MESH_ALPHA = 0.55
#: Live entities this far outside the display still count as "in view" (labels don't pop).
VIEW_MARGIN_DEG = 5.0
STATUS_PERIOD_S = 0.5
#: Hide the pointer when the newest phone sample is older than this.
POINTER_TIMEOUT_S = 1.0


class PhoneLinkLike(Protocol):
    """The parts of phone_link.PhoneLinkThread the loop uses."""

    def latest(self) -> PhoneState | None: ...
    def pop_buttons(self) -> list[ButtonEvent]: ...
    def connected(self) -> bool: ...
    def stalled(self) -> bool: ...
    def send_nowait(self, message: dict) -> None: ...


@dataclass
class RunStats:
    """Counters for the end-to-end test and the exit summary."""

    frames: int = 0
    max_entities: int = 0
    kinds_seen: set[str] = field(default_factory=set)
    snaps: int = 0
    aligns: int = 0
    info_frames: int = 0
    info_titles: set[str] = field(default_factory=set)
    buttons: list[str] = field(default_factory=list)
    fps: float = 0.0
    detections_per_s: float = 0.0
    mesh_segments_max: int = 0


class PointViewApp:
    def __init__(
        self,
        glasses: GlassesSource,
        layers: Sequence[Layer],
        phone: PhoneLinkLike,
        renderer: Renderer | None,
        *,
        calibration: Calibration | None = None,
        landmark_layer: landmarks.LandmarkLayer | None = None,
        clock: Clock = now,
        wall_clock: Clock = time.time,
        render_latency_s: float = 0.0,
        look: Look = "normal",
        passthrough: bool = False,
        mesh_source: MeshSource | None = None,
    ) -> None:
        self.glasses = glasses
        self.layers = list(layers)
        self.phone = phone
        self.renderer = renderer
        self.calibration = calibration or Calibration()
        self.landmark_layer = landmark_layer
        self.clock = clock
        self.wall_clock = wall_clock
        self.render_latency_s = render_latency_s
        #: Used until the phone reports its own choice.
        self.default_look: Look = look
        self.default_passthrough = passthrough
        self.mesh_source = mesh_source
        self.look: Look | None = None
        self.alignment = Alignment()
        self.snapped_id: str | None = None
        self.info_id: str | None = None
        self.entities: list[Entity] = []
        self.items: list[DrawItem] = []
        self.stats = RunStats()
        self._last_status_t = -1e9
        self._fps_t0: float | None = None
        self._fps_frames = 0

    # ------------------------------------------------------------------------------------ actions

    def _haptic(self, style: str) -> None:
        try:
            self.phone.send_nowait({"type": "haptic", "style": style})
        except Exception:  # a haptic is never worth crashing the frame
            log.debug("haptic send failed", exc_info=True)

    def _pointer_world(self, phone: PhoneState, t: float) -> tuple[float, float]:
        q = world.predict_quat(phone.quat, phone.omega, t - phone.t)
        return world.pointer_ray(q, self.alignment)

    def _align(self, ev: ButtonEvent, phone: PhoneState) -> None:
        pose = self.glasses.pose_at(ev.t)
        if pose is None:
            self._haptic("error")
            return
        self.alignment = world.align(
            pose.quat, phone.quat, phone.heading_deg, phone.heading_acc, self.clock()
        )
        self.stats.aligns += 1
        self._haptic("align_ok")
        log.info("aligned: %s", self.alignment)

    def _align_north(self, phone: PhoneState, t: float) -> None:
        target = next((e for e in self.entities if e.id == self.snapped_id), None)
        if self.alignment.aligned_at is None or target is None or target.kind != "landmark":
            self._haptic("error")
            return
        ptr_yaw, _ = self._pointer_world(phone, t)
        tracking_yaw = ptr_yaw - self.alignment.north_offset_deg
        self.alignment = world.align_north_to_landmark(
            self.alignment, tracking_yaw, target.world_yaw_deg
        )
        self._haptic("align_ok")
        log.info("north set from landmark %s: %s", target.label, self.alignment)

    def _info(self) -> None:
        if self.info_id is not None:
            self.info_id = None
        elif self.snapped_id is not None:
            self.info_id = self.snapped_id
            self._haptic("select")
        else:
            self._haptic("error")

    def _handle_buttons(self, phone: PhoneState | None, t: float) -> None:
        for ev in self.phone.pop_buttons():
            self.stats.buttons.append(ev.name)
            if phone is None:
                continue
            if ev.name == "align" or (
                ev.name == "align_long" and self.alignment.aligned_at is None
            ):
                self._align(ev, phone)
            elif ev.name == "align_long":
                self._align_north(phone, t)
            elif ev.name == "info":
                self._info()

    # -------------------------------------------------------------------------------------- frame

    def _info_content(self) -> InfoContent | None:
        if self.info_id is None:
            return None
        e = next((x for x in self.entities if x.id == self.info_id), None)
        if e is None:  # the object is gone (lost track, layer off): close the card
            self.info_id = None
            return None
        if e.kind != "landmark":
            return info_for(e)
        tag = e.meta.get("wikipedia")
        summary = self.landmark_layer.summary_for(tag) if (tag and self.landmark_layer) else None
        if summary is None:
            return info_for(e)
        return info_for(e, landmarks.summary_lines(summary, 40), landmarks.WIKIPEDIA_ATTRIBUTION)

    def _in_view(self, e: Entity, view: View | None) -> bool:
        if view is None:
            return False
        return world.in_view(
            e.world_yaw_deg,
            e.world_pitch_deg,
            view.head_quat,
            view.north_offset_deg,
            view.display_fov_deg,
            VIEW_MARGIN_DEG,
        )

    def _backdrop(self, look: Look, head_quat_now) -> Backdrop | None:
        """Passthrough from a VITURE camera, warped to line up with the world (contracts §8).
        NVG prefers the grayscale tracking cameras; everything else uses the front camera."""
        g = self.glasses
        frame, intr, cam_q = None, None, self.calibration.camera_to_head_quat
        if look == "nvg" and isinstance(g, TrackingCameraSource):
            frame = g.latest_tracking_frame()
            if frame is not None:
                intr, cam_q = g.tracking_camera_intrinsics, g.tracking_camera_to_head_quat
        if frame is None:
            frame, intr = g.latest_frame(), g.camera_intrinsics
        if frame is None or intr is None:
            return None
        pose_then = g.pose_at(frame.t)
        if pose_then is None:
            return None
        h = world.backdrop_homography(
            intr, cam_q, pose_then.quat, head_quat_now, g.display_size, g.display_fov_deg
        )
        return Backdrop(frame.bgr, h)

    def _mesh(self, phone: PhoneState | None, head_quat) -> WireMesh | None:
        """The sparse world mesh for this frame (contracts §11). Needs a GPS fix and an Align
        (without true north the wireframe would be rotated)."""
        if self.mesh_source is None or phone is None or phone.lat is None or phone.lon is None:
            return None
        if not phone.layers.get("mesh", True) or self.alignment.aligned_at is None:
            return None
        mesh = self.mesh_source.mesh_near(phone.lat, phone.lon)
        if mesh is None:
            return None
        ground = self.mesh_source.ground_alt_m(phone.lat, phone.lon)
        eye_alt = (ground if ground is not None else mesh.origin.alt_m) + world.EYE_HEIGHT_M
        seg, dist = world.project_mesh(
            mesh,
            phone.lat,
            phone.lon,
            eye_alt,
            head_quat,
            self.alignment.north_offset_deg,
            self.glasses.display_size,
            self.glasses.display_fov_deg,
            max_range_m=MESH_RANGE_M,
        )
        if len(seg) > MESH_MAX_SEGMENTS:
            keep = np.argpartition(dist, MESH_MAX_SEGMENTS)[:MESH_MAX_SEGMENTS]
            seg, dist = seg[keep], dist[keep]
        alpha = np.clip(1.0 - dist / MESH_RANGE_M, 0.15, 1.0) * MESH_ALPHA
        self.stats.mesh_segments_max = max(self.stats.mesh_segments_max, len(seg))
        return WireMesh(seg, MESH_COLOR, alpha.astype(np.float32))

    def tick(self) -> list[DrawItem]:
        t = self.clock()
        pose = self.glasses.pose_at(t + self.render_latency_s)
        head_quat = pose.quat if pose is not None else IDENTITY_QUAT
        phone = self.phone.latest()
        connected = self.phone.connected()

        enabled = frozenset(LAYER_KINDS.values())
        if phone is not None:
            enabled = frozenset(
                k for name, k in LAYER_KINDS.items() if phone.layers.get(name, True)
            )
        fresh = phone is not None and connected and t - phone.received_at < POINTER_TIMEOUT_S
        pointer = None
        if fresh and self.alignment.aligned_at is not None:
            pointer = self._pointer_world(phone, t)
        view = None
        if pose is not None:
            fwd = world.quat_rotate(head_quat, (0.0, 1.0, 0.0))
            yaw, pitch = world.yaw_pitch_from_vector(fwd)
            north = self.alignment.north_offset_deg
            view = View(
                head_quat, north, self.glasses.display_fov_deg, world.wrap_deg(yaw + north), pitch
            )
        ctx = LayerContext(
            t,
            self.glasses,
            self.alignment,
            self.calibration,
            phone,
            enabled,
            unix_time=self.wall_clock(),
            view=view,
            pointer=pointer,
            focus_id=self.snapped_id,
        )
        entities: list[Entity] = []
        for layer in self.layers:
            try:
                entities.extend(layer.update(ctx))
            except Exception:
                log.exception("layer %s failed", getattr(layer, "name", layer))
        # Live data only for what the glasses are looking at (contracts §7).
        entities = [e for e in entities if e.kind not in LIVE_KINDS or self._in_view(e, view)]
        self.entities = entities

        self._handle_buttons(phone, t)

        if pointer is not None:
            new = world.choose_snap(pointer[0], pointer[1], entities, self.snapped_id)
            if new is not None and new != self.snapped_id:
                self.stats.snaps += 1
                self._haptic("snap")
            self.snapped_id = new
        else:
            self.snapped_id = None

        info = self._info_content()
        status = Status(
            phone_connected=connected,
            phone_stalled=self.phone.stalled(),
            gps_acc_m=phone.gps_acc if phone else None,
            alignment=self.alignment,
            since_align_s=(t - self.alignment.aligned_at) if self.alignment.aligned_at else None,
            heading_acc_deg=phone.heading_acc if phone else None,
        )
        self.items = build_scene(
            SceneInput(
                entities=entities,
                head_quat=head_quat,
                alignment=self.alignment,
                display_size=self.glasses.display_size,
                display_fov_deg=self.glasses.display_fov_deg,
                status=status,
                pointer=pointer,
                snapped_id=self.snapped_id,
                info=info,
                show_crosshair=self.alignment.aligned_at is None,
                marker_shape=phone.marker if phone is not None else "square",
                mesh=self._mesh(phone, head_quat) if pose is not None else None,
                attribution=landmarks.OSM_ATTRIBUTION,
            )
        )
        look = phone.look if phone is not None else self.default_look
        passthrough = phone.passthrough if phone is not None else self.default_passthrough
        if passthrough and pose is not None:
            backdrop = self._backdrop(look, head_quat)
            if backdrop is not None:
                self.items.insert(0, backdrop)
        if self.renderer is not None:
            if look != self.look and hasattr(self.renderer, "set_look"):
                self.renderer.set_look(look)
            self.look = look
            self.renderer.show(self.items)

        if connected and t - self._last_status_t >= STATUS_PERIOD_S:
            self._last_status_t = t
            selected = next((e.label for e in entities if e.id == self.snapped_id), None)
            try:
                self.phone.send_nowait(
                    {
                        "type": "status",
                        "aligned": self.alignment.aligned_at is not None,
                        "selected": selected,
                        "fps": round(self.stats.fps, 1),
                        "look": look,
                        "passthrough": passthrough,
                    }
                )
            except Exception:
                log.debug("status send failed", exc_info=True)

        self._count(t, entities)
        return self.items

    def _count(self, t: float, entities: list[Entity]) -> None:
        s = self.stats
        s.frames += 1
        s.max_entities = max(s.max_entities, len(entities))
        s.kinds_seen.update(e.kind for e in entities)
        cards = [i for i in self.items if isinstance(i, InfoCard)]
        if cards:
            s.info_frames += 1
            s.info_titles.update(c.title for c in cards)
        if self._fps_t0 is None:
            self._fps_t0 = t
        self._fps_frames += 1
        if t - self._fps_t0 >= 1.0:
            s.fps = self._fps_frames / (t - self._fps_t0)
            self._fps_t0, self._fps_frames = t, 0

    def close(self) -> None:
        for layer in self.layers:
            try:
                layer.close()
            except Exception:
                log.exception("closing layer failed")
