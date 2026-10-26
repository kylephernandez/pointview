"""Pixel boxes -> world-direction Entities, and the monocular distance estimate.

All frame math goes through ``pointview.world`` (contracts section 3.5); nothing here re-implements it.
"""

from __future__ import annotations

import dataclasses

from pointview import world
from pointview.detection.tracker import Box, Track
from pointview.entities import Entity, SizeDeg
from pointview.interfaces import CameraIntrinsics, Quat

#: Typical real-world heights (m) used for d ~= f * H / h. Rough on purpose (+-30 %, design doc).
#: person 1.7 and car 1.5 are fixed by the design doc; the rest are assumptions:
#: bicycle ~1.0 m (bike alone, rider's torso adds more but boxes usually include the rider ->
#: 1.1), motorcycle ~1.2 m (box often includes the rider -> would be ~1.6; 1.2 is a compromise),
#: bus ~3.0 m, truck ~2.8 m (vans/pickups ~1.9, box trucks ~3.3, semis ~4).
REAL_HEIGHT_M: dict[str, float] = {
    "person": 1.7,
    "bicycle": 1.1,
    "car": 1.5,
    "motorcycle": 1.2,
    "bus": 3.0,
    "truck": 2.8,
}

#: A box this close (px) to the top/bottom frame edge is probably truncated -> no distance.
EDGE_MARGIN_PX = 2.0


def scale_intrinsics(intr: CameraIntrinsics, width: int, height: int) -> CameraIntrinsics:
    """Intrinsics for a frame of a different size than the calibration (uniform per-axis scale)."""
    if (width, height) == (intr.width, intr.height):
        return intr
    sx, sy = width / intr.width, height / intr.height
    return dataclasses.replace(
        intr, fx=intr.fx * sx, fy=intr.fy * sy, cx=intr.cx * sx, cy=intr.cy * sy,
        width=width, height=height,
    )  # fmt: skip


def estimate_distance_m(
    label: str, box: Box, fy: float, frame_h: float | None = None
) -> float | None:
    """d ~= fy * H / h_px. None when the label has no height, the box is empty, or (given
    ``frame_h``) the box touches the top or bottom edge, where its height is truncated."""
    height_m = REAL_HEIGHT_M.get(label)
    h_px = box[3] - box[1]
    if height_m is None or h_px <= 0:
        return None
    if frame_h is not None and (box[1] <= EDGE_MARGIN_PX or box[3] >= frame_h - EDGE_MARGIN_PX):
        return None
    return fy * height_m / h_px


def track_to_entity(
    track: Track,
    intr: CameraIntrinsics,
    head_quat: Quat,
    north_offset_deg: float,
    camera_to_head_quat: Quat,
    frame_size: tuple[int, int],
) -> Entity:
    """Convert a track's box (pixels of a ``frame_size`` = (w, h) frame) into a world Entity."""
    intr = scale_intrinsics(intr, *frame_size)
    x1, y1, x2, y2 = track.box
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    yaw, pitch = world.pixel_to_world(u, v, intr, head_quat, north_offset_deg, camera_to_head_quat)
    a_left, _ = world.camera_angles_deg(x1, v, intr)
    a_right, _ = world.camera_angles_deg(x2, v, intr)
    _, b_top = world.camera_angles_deg(u, y1, intr)
    _, b_bottom = world.camera_angles_deg(u, y2, intr)
    return Entity(
        id=f"det:{track.id}",
        kind=track.kind,
        label=track.label,
        world_yaw_deg=yaw,
        world_pitch_deg=pitch,
        updated_at=track.last_seen_t,
        distance_m=estimate_distance_m(track.label, track.box, intr.fy, frame_size[1]),
        size_deg=SizeDeg(w=abs(a_right - a_left), h=abs(b_bottom - b_top)),
        confidence=track.score,
    )
