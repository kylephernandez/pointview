"""World math: frames, alignment, pointer ray, snapping, prediction. Pure functions, no I/O.

The SIGNATURES in this file are part of the frozen contract (docs/contracts.md, "world.py API").
The bodies are implemented by the world-math module owner. Frames and conventions are defined in
pointview/interfaces.py. All angles in and out are DEGREES unless the name says ``_rad``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pointview.entities import Entity
from pointview.interfaces import Alignment, CameraIntrinsics, Quat, Vec3

SNAP_RADIUS_DEG = 5.0
SNAP_HYSTERESIS_DEG = 2.0
#: Do not extrapolate phone motion further than this (stall smoothing, design doc).
MAX_PREDICT_S = 0.15

# ------------------------------------------------------------------------------- quaternions


def quat_normalize(q: Quat) -> Quat:
    raise NotImplementedError


def quat_mul(a: Quat, b: Quat) -> Quat:
    """Hamilton product a ⊗ b (apply b first, then a)."""
    raise NotImplementedError


def quat_conj(q: Quat) -> Quat:
    raise NotImplementedError


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    """Rotate vector v by unit quaternion q."""
    raise NotImplementedError


def quat_from_axis_angle(axis: Vec3, angle_rad: float) -> Quat:
    raise NotImplementedError


def quat_slerp(a: Quat, b: Quat, s: float) -> Quat:
    """Shortest-path spherical interpolation, s in [0, 1]."""
    raise NotImplementedError


def quat_from_yaw_pitch_roll(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> Quat:
    """Head/phone-style orientation in a z-up frame whose forward axis is +y.

    Yaw is clockwise about +z seen from above (compass sense), then pitch up about the new +x,
    then roll clockwise about the new +y. quat_rotate(q, (0, 1, 0)) points at (yaw, pitch).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------------- directions


def wrap_deg(a: float) -> float:
    """Normalize to [0, 360)."""
    raise NotImplementedError


def wrap_deg_signed(a: float) -> float:
    """Normalize to [-180, 180)."""
    raise NotImplementedError


def vector_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> Vec3:
    """Unit vector in a z-up frame, yaw clockwise from +y."""
    raise NotImplementedError


def yaw_pitch_from_vector(v: Vec3) -> tuple[float, float]:
    """Inverse of vector_from_yaw_pitch. Yaw in [0, 360); yaw is 0 when v is straight up/down."""
    raise NotImplementedError


def angular_distance_deg(yaw1: float, pitch1: float, yaw2: float, pitch2: float) -> float:
    """Great-circle angle between two directions, in [0, 180]."""
    raise NotImplementedError


# -------------------------------------------------------------------------------- geo (landmarks)


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, clockwise from true north, [0, 360)."""
    raise NotImplementedError


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters (Earth radius 6_371_000 m)."""
    raise NotImplementedError


# ------------------------------------------------------------------------------------- camera


def pixel_to_camera_ray(u: float, v: float, intr: CameraIntrinsics) -> Vec3:
    """Unit ray in the camera frame (x right, y down, z forward) through pixel (u, v).
    Applies intr.dist undistortion when it is non-empty."""
    raise NotImplementedError


def camera_angles_deg(u: float, v: float, intr: CameraIntrinsics) -> tuple[float, float]:
    """(alpha, beta) of design doc step 1: alpha = atan((u - cx)/fx) right-positive,
    beta = atan((v - cy)/fy) DOWN-positive."""
    raise NotImplementedError


def camera_to_head(v_cam: Vec3, camera_to_head_quat: Quat) -> Vec3:
    """Camera-frame vector → head frame: (x, z, -y), then rotated by camera_to_head_quat."""
    raise NotImplementedError


def head_to_world(v_head: Vec3, head_quat: Quat, north_offset_deg: float) -> Vec3:
    """Head frame → tracking frame (head_quat) → world frame (rotate yaw by north_offset_deg)."""
    raise NotImplementedError


def world_to_head(v_world: Vec3, head_quat: Quat, north_offset_deg: float) -> Vec3:
    """Exact inverse of head_to_world."""
    raise NotImplementedError


def pixel_to_world(
    u: float,
    v: float,
    intr: CameraIntrinsics,
    head_quat: Quat,
    north_offset_deg: float,
    camera_to_head_quat: Quat,
) -> tuple[float, float]:
    """Camera pixel at capture-time head pose → (world_yaw_deg, world_pitch_deg)."""
    raise NotImplementedError


# ------------------------------------------------------------------------------------ display


def head_angles_deg(
    world_yaw_deg: float, world_pitch_deg: float, head_quat: Quat, north_offset_deg: float
) -> tuple[float, float, bool]:
    """World direction → (alpha', beta', in_front) relative to where the head points now.

    alpha' = atan2(h.x, h.y) right-positive, beta' = atan2(h.z, h.y) UP-positive (h = head-frame
    vector). These are the tangent-plane angles the display projection uses. in_front = h.y > 0.
    """
    raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DisplayPoint:
    x: float
    y: float
    #: Within the display rectangle.
    on_screen: bool
    #: In front of the viewer (h.y > 0). When False, x/y are not meaningful.
    in_front: bool


def display_focal_px(display_size: tuple[int, int], display_fov_deg: tuple[float, float]) -> float:
    """f_g = (W_g / 2) / tan(FOV_h / 2)  (design doc step 4). Square pixels assumed."""
    raise NotImplementedError


def project_to_display(
    world_yaw_deg: float,
    world_pitch_deg: float,
    head_quat: Quat,
    north_offset_deg: float,
    display_size: tuple[int, int],
    display_fov_deg: tuple[float, float],
) -> DisplayPoint:
    """World direction → display pixel. x = W/2 + f*h.x/h.y, y = H/2 - f*h.z/h.y."""
    raise NotImplementedError


def angular_size_to_px(
    size_deg: float, display_size: tuple[int, int], display_fov_deg: tuple[float, float]
) -> float:
    """Approximate on-display size (px) of an object subtending size_deg near the center."""
    raise NotImplementedError


# ----------------------------------------------------------------------------- phone & alignment


def phone_pointer_vector(phone_quat: Quat) -> Vec3:
    """Direction of the phone's top edge (P +y) in the phone reference frame R."""
    raise NotImplementedError


def head_forward_yaw_deg(head_quat: Quat) -> float:
    """Tracking-frame yaw of the head's forward axis (H +y)."""
    raise NotImplementedError


def align(
    head_quat: Quat,
    phone_quat: Quat,
    phone_heading_deg: float | None,
    heading_acc_deg: float | None,
    t: float,
) -> Alignment:
    """The one-tap Align (design doc step 5), done while looking and pointing at the same thing.

    phone_to_head = head yaw − phone pointer yaw (both from their quaternions).
    north_offset = phone compass heading − head yaw; if phone_heading_deg is None, north_offset = 0
    and north_source = "none".
    """
    raise NotImplementedError


def align_north_to_landmark(
    alignment: Alignment, pointer_tracking_yaw_deg: float, landmark_world_yaw_deg: float
) -> Alignment:
    """Long-press Align on a known landmark: set north_offset so the pointer's tracking yaw maps
    exactly onto the landmark's true bearing. Keeps phone_to_head; north_source = "landmark"."""
    raise NotImplementedError


def pointer_ray(phone_quat: Quat, alignment: Alignment) -> tuple[float, float]:
    """(world_yaw_deg, world_pitch_deg) of the pointer: the phone's top edge rotated by
    phone_to_head_deg (into T) and north_offset_deg (into W). Pitch is unchanged by alignment."""
    raise NotImplementedError


def predict_quat(phone_quat: Quat, omega_rad_s: Vec3, dt_s: float) -> Quat:
    """Extrapolate the phone's attitude by its body-frame rotation rate ω over dt
    (ψ_now = ψ_sample + ω·Δt). dt is clamped to [0, MAX_PREDICT_S]."""
    raise NotImplementedError


# ------------------------------------------------------------------------------------- snapping


def choose_snap(
    pointer_yaw_deg: float,
    pointer_pitch_deg: float,
    entities: Sequence[Entity],
    current_id: str | None,
    radius_deg: float = SNAP_RADIUS_DEG,
    hysteresis_deg: float = SNAP_HYSTERESIS_DEG,
) -> str | None:
    """Which entity the pointer is snapped to (design doc step 6).

    Candidates are entities within radius_deg of the pointer. If current_id is still a candidate,
    keep it unless another candidate is at least hysteresis_deg closer. Otherwise pick the closest
    candidate. None if there are no candidates. Ties break by entity id (deterministic).
    """
    raise NotImplementedError
