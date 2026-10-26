"""World math: frames, alignment, pointer ray, snapping, prediction. Pure functions, no I/O.

The SIGNATURES in this file are part of the frozen contract (docs/contracts.md, "world.py API").
Frames and conventions are defined in pointview/interfaces.py and docs/contracts.md. All angles in
and out are DEGREES unless the name says ``_rad``.

Conventions used throughout: quaternions are ``(w, x, y, z)``; yaw is clockwise seen from above
(a rotation about +z by ``-yaw``); pitch is up-positive; direction vectors are ``(x east/right,
y north/forward, z up)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np

from pointview.entities import Entity
from pointview.interfaces import Alignment, CameraIntrinsics, Quat, Vec3

SNAP_RADIUS_DEG = 5.0
SNAP_HYSTERESIS_DEG = 2.0
#: Do not extrapolate phone motion further than this (stall smoothing, design doc).
MAX_PREDICT_S = 0.15

_EARTH_RADIUS_M = 6_371_000.0
_EPS = 1e-12
#: h.y at or below this counts as "not in front" (absorbs float noise at exactly 90 deg).
_FRONT_EPS = 1e-9
_RAD = math.pi / 180.0
_DEG = 180.0 / math.pi
_IDENTITY: Quat = (1.0, 0.0, 0.0, 0.0)

# ---------------------------------------------------------------------------- private helpers


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rot_z_clockwise(v: Vec3, deg: float) -> Vec3:
    """Rotate v about +z so that its yaw (clockwise from +y) increases by ``deg``."""
    r = deg * _RAD
    c, s = math.cos(r), math.sin(r)
    return (v[0] * c + v[1] * s, -v[0] * s + v[1] * c, v[2])


def _angle_between(a: Vec3, b: Vec3) -> float:
    """Angle in degrees between two (not necessarily unit) vectors; stable for tiny angles."""
    cx, cy, cz = _cross(a, b)
    d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), d) * _DEG


def _normalized_xy(u: float, v: float, intr: CameraIntrinsics) -> tuple[float, float]:
    """Pixel -> undistorted normalized image coordinates (x/z, y/z)."""
    if not intr.dist:
        return (u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy
    k = np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]])
    pts = np.array([[[u, v]]], dtype=np.float64)
    out = cv2.undistortPoints(pts, k, np.asarray(intr.dist, dtype=np.float64))
    return float(out[0, 0, 0]), float(out[0, 0, 1])


# ------------------------------------------------------------------------------- quaternions


def quat_normalize(q: Quat) -> Quat:
    """Scale to unit norm. A (near) zero quaternion becomes the identity."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < _EPS:
        return _IDENTITY
    return (w / n, x / n, y / n, z / n)


def quat_mul(a: Quat, b: Quat) -> Quat:
    """Hamilton product a ⊗ b (apply b first, then a)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conj(q: Quat) -> Quat:
    """Conjugate; the inverse for a unit quaternion."""
    return (q[0], -q[1], -q[2], -q[3])


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    """Rotate vector v by unit quaternion q."""
    w, x, y, z = q
    # v' = v + w*t + qv x t with t = 2 (qv x v); exact for a unit q.
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    )


def quat_from_axis_angle(axis: Vec3, angle_rad: float) -> Quat:
    """Right-handed rotation of angle_rad about axis (need not be unit; zero axis = identity)."""
    n = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    if n < _EPS:
        return _IDENTITY
    h = angle_rad / 2.0
    k = math.sin(h) / n
    return (math.cos(h), axis[0] * k, axis[1] * k, axis[2] * k)


def quat_slerp(a: Quat, b: Quat, s: float) -> Quat:
    """Shortest-path spherical interpolation, s in [0, 1].

    q and -q are the same rotation, so the interpolation negates b when needed. ``s`` outside
    [0, 1] extrapolates along the same great arc. The result is unit length.
    """
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    dot = aw * bw + ax * bx + ay * by + az * bz
    if dot < 0.0:
        bw, bx, by, bz = -bw, -bx, -by, -bz
        dot = -dot
    if dot > 0.9995:  # nearly parallel: normalized lerp is accurate and avoids 0/0
        k0, k1 = 1.0 - s, s
    else:
        theta = math.acos(_clamp(dot, -1.0, 1.0))
        sin_t = math.sin(theta)
        k0 = math.sin((1.0 - s) * theta) / sin_t
        k1 = math.sin(s * theta) / sin_t
    return quat_normalize(
        (k0 * aw + k1 * bw, k0 * ax + k1 * bx, k0 * ay + k1 * by, k0 * az + k1 * bz)
    )


def quat_from_yaw_pitch_roll(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> Quat:
    """Head/phone-style orientation in a z-up frame whose forward axis is +y.

    Yaw is clockwise about +z seen from above (compass sense), then pitch up about the new +x,
    then roll clockwise about the new +y. quat_rotate(q, (0, 1, 0)) points at (yaw, pitch).

    "Clockwise" roll is as seen by the wearer looking along +y (the head tilts to the right),
    which is a right-handed rotation about +y by roll_deg.
    """
    qz = quat_from_axis_angle((0.0, 0.0, 1.0), -yaw_deg * _RAD)
    qx = quat_from_axis_angle((1.0, 0.0, 0.0), pitch_deg * _RAD)
    qy = quat_from_axis_angle((0.0, 1.0, 0.0), roll_deg * _RAD)
    return quat_mul(quat_mul(qz, qx), qy)


# ---------------------------------------------------------------------------------- directions


def wrap_deg(a: float) -> float:
    """Normalize to [0, 360)."""
    r = a % 360.0
    return 0.0 if r >= 360.0 else r


def wrap_deg_signed(a: float) -> float:
    """Normalize to [-180, 180)."""
    r = (a + 180.0) % 360.0 - 180.0
    return -180.0 if r >= 180.0 else r


def vector_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> Vec3:
    """Unit vector in a z-up frame, yaw clockwise from +y."""
    y, p = yaw_deg * _RAD, pitch_deg * _RAD
    cp = math.cos(p)
    return (math.sin(y) * cp, math.cos(y) * cp, math.sin(p))


def yaw_pitch_from_vector(v: Vec3) -> tuple[float, float]:
    """Inverse of vector_from_yaw_pitch. Yaw in [0, 360); yaw is 0 when v is straight up/down.

    The input need not be unit length.
    """
    x, y, z = v
    horiz = math.hypot(x, y)
    pitch = math.atan2(z, horiz) * _DEG
    if horiz < _EPS * max(1.0, abs(z)):
        return 0.0, pitch
    return wrap_deg(math.atan2(x, y) * _DEG), pitch


def angular_distance_deg(yaw1: float, pitch1: float, yaw2: float, pitch2: float) -> float:
    """Great-circle angle between two directions, in [0, 180]."""
    return _angle_between(vector_from_yaw_pitch(yaw1, pitch1), vector_from_yaw_pitch(yaw2, pitch2))


# -------------------------------------------------------------------------------- geo (landmarks)


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, clockwise from true north, [0, 360)."""
    p1, p2 = lat1 * _RAD, lat2 * _RAD
    dl = (lon2 - lon1) * _RAD
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return wrap_deg(math.atan2(y, x) * _DEG)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters (Earth radius 6_371_000 m)."""
    p1, p2 = lat1 * _RAD, lat2 * _RAD
    dp = p2 - p1
    dl = (lon2 - lon1) * _RAD
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


# ------------------------------------------------------------------------------------- camera


def pixel_to_camera_ray(u: float, v: float, intr: CameraIntrinsics) -> Vec3:
    """Unit ray in the camera frame (x right, y down, z forward) through pixel (u, v).
    Applies intr.dist undistortion when it is non-empty."""
    x, y = _normalized_xy(u, v, intr)
    n = math.sqrt(x * x + y * y + 1.0)
    return (x / n, y / n, 1.0 / n)


def camera_angles_deg(u: float, v: float, intr: CameraIntrinsics) -> tuple[float, float]:
    """(alpha, beta) of design doc step 1: alpha = atan((u - cx)/fx) right-positive,
    beta = atan((v - cy)/fy) DOWN-positive.

    When intr.dist is non-empty the pixel is undistorted first (same as pixel_to_camera_ray).
    """
    x, y = _normalized_xy(u, v, intr)
    return math.atan(x) * _DEG, math.atan(y) * _DEG


def camera_to_head(v_cam: Vec3, camera_to_head_quat: Quat) -> Vec3:
    """Camera-frame vector → head frame: (x, z, -y), then rotated by camera_to_head_quat."""
    return quat_rotate(camera_to_head_quat, (v_cam[0], v_cam[2], -v_cam[1]))


def head_to_world(v_head: Vec3, head_quat: Quat, north_offset_deg: float) -> Vec3:
    """Head frame → tracking frame (head_quat) → world frame (rotate yaw by north_offset_deg)."""
    return _rot_z_clockwise(quat_rotate(head_quat, v_head), north_offset_deg)


def world_to_head(v_world: Vec3, head_quat: Quat, north_offset_deg: float) -> Vec3:
    """Exact inverse of head_to_world."""
    return quat_rotate(quat_conj(head_quat), _rot_z_clockwise(v_world, -north_offset_deg))


def pixel_to_world(
    u: float,
    v: float,
    intr: CameraIntrinsics,
    head_quat: Quat,
    north_offset_deg: float,
    camera_to_head_quat: Quat,
) -> tuple[float, float]:
    """Camera pixel at capture-time head pose → (world_yaw_deg, world_pitch_deg)."""
    ray = pixel_to_camera_ray(u, v, intr)
    v_head = camera_to_head(ray, camera_to_head_quat)
    return yaw_pitch_from_vector(head_to_world(v_head, head_quat, north_offset_deg))


# ------------------------------------------------------------------------------------ display


def head_angles_deg(
    world_yaw_deg: float, world_pitch_deg: float, head_quat: Quat, north_offset_deg: float
) -> tuple[float, float, bool]:
    """World direction → (alpha', beta', in_front) relative to where the head points now.

    alpha' = atan2(h.x, h.y) right-positive, beta' = atan2(h.z, h.y) UP-positive (h = head-frame
    vector). These are the tangent-plane angles the display projection uses. in_front = h.y > 0
    (in practice h.y > 1e-9, so a direction exactly 90 deg to the side is not in front).
    """
    h = world_to_head(
        vector_from_yaw_pitch(world_yaw_deg, world_pitch_deg), head_quat, north_offset_deg
    )
    return math.atan2(h[0], h[1]) * _DEG, math.atan2(h[2], h[1]) * _DEG, h[1] > _FRONT_EPS


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
    return (display_size[0] / 2.0) / math.tan(display_fov_deg[0] * _RAD / 2.0)


def project_to_display(
    world_yaw_deg: float,
    world_pitch_deg: float,
    head_quat: Quat,
    north_offset_deg: float,
    display_size: tuple[int, int],
    display_fov_deg: tuple[float, float],
) -> DisplayPoint:
    """World direction → display pixel. x = W/2 + f*h.x/h.y, y = H/2 - f*h.z/h.y.

    Behind the viewer (h.y <= 0) the point is (W/2, H/2) with in_front=False, on_screen=False.
    """
    w, h_px = display_size
    h = world_to_head(
        vector_from_yaw_pitch(world_yaw_deg, world_pitch_deg), head_quat, north_offset_deg
    )
    if h[1] <= _FRONT_EPS:
        return DisplayPoint(w / 2.0, h_px / 2.0, False, False)
    f = display_focal_px(display_size, display_fov_deg)
    x = w / 2.0 + f * h[0] / h[1]
    y = h_px / 2.0 - f * h[2] / h[1]
    return DisplayPoint(x, y, 0.0 <= x <= w and 0.0 <= y <= h_px, True)


def angular_size_to_px(
    size_deg: float, display_size: tuple[int, int], display_fov_deg: tuple[float, float]
) -> float:
    """Approximate on-display size (px) of an object subtending size_deg near the center.

    Exactly the pixel extent of an object centered on the optical axis: 2 f tan(size/2).
    size_deg is limited to [0, 179] so the result stays finite.
    """
    f = display_focal_px(display_size, display_fov_deg)
    return 2.0 * f * math.tan(_clamp(size_deg, 0.0, 179.0) * _RAD / 2.0)


# ----------------------------------------------------------------------------- phone & alignment


def phone_pointer_vector(phone_quat: Quat) -> Vec3:
    """Direction of the phone's top edge (P +y) in the phone reference frame R."""
    return quat_rotate(phone_quat, (0.0, 1.0, 0.0))


def head_forward_yaw_deg(head_quat: Quat) -> float:
    """Tracking-frame yaw of the head's forward axis (H +y)."""
    return yaw_pitch_from_vector(quat_rotate(head_quat, (0.0, 1.0, 0.0)))[0]


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

    Both offsets are returned wrapped to [-180, 180); adding them to a yaw is equivalent modulo 360.
    """
    head_yaw = head_forward_yaw_deg(head_quat)
    phone_yaw = yaw_pitch_from_vector(phone_pointer_vector(phone_quat))[0]
    phone_to_head = wrap_deg_signed(head_yaw - phone_yaw)
    if phone_heading_deg is None:
        return Alignment(
            phone_to_head_deg=phone_to_head,
            north_offset_deg=0.0,
            aligned_at=t,
            north_source="none",
            heading_acc_deg=None,
        )
    return Alignment(
        phone_to_head_deg=phone_to_head,
        north_offset_deg=wrap_deg_signed(phone_heading_deg - head_yaw),
        aligned_at=t,
        north_source="compass",
        heading_acc_deg=heading_acc_deg,
    )


def align_north_to_landmark(
    alignment: Alignment, pointer_tracking_yaw_deg: float, landmark_world_yaw_deg: float
) -> Alignment:
    """Long-press Align on a known landmark: set north_offset so the pointer's tracking yaw maps
    exactly onto the landmark's true bearing. Keeps phone_to_head; north_source = "landmark".
    The compass accuracy no longer applies, so heading_acc_deg is cleared."""
    return replace(
        alignment,
        north_offset_deg=wrap_deg_signed(landmark_world_yaw_deg - pointer_tracking_yaw_deg),
        north_source="landmark",
        heading_acc_deg=None,
    )


def pointer_ray(phone_quat: Quat, alignment: Alignment) -> tuple[float, float]:
    """(world_yaw_deg, world_pitch_deg) of the pointer: the phone's top edge rotated by
    phone_to_head_deg (into T) and north_offset_deg (into W). Pitch is unchanged by alignment."""
    yaw, pitch = yaw_pitch_from_vector(phone_pointer_vector(phone_quat))
    return wrap_deg(yaw + alignment.phone_to_head_deg + alignment.north_offset_deg), pitch


def predict_quat(phone_quat: Quat, omega_rad_s: Vec3, dt_s: float) -> Quat:
    """Extrapolate the phone's attitude by its body-frame rotation rate ω over dt
    (ψ_now = ψ_sample + ω·Δt). dt is clamped to [0, MAX_PREDICT_S].

    ω is expressed in the phone body frame, so the increment is applied on the right:
    q_now = q ⊗ exp(ω·dt). With ω = 0 (or dt <= 0) the input is returned unchanged.
    """
    dt = _clamp(dt_s, 0.0, MAX_PREDICT_S)
    wx, wy, wz = omega_rad_s
    rate = math.sqrt(wx * wx + wy * wy + wz * wz)
    if dt == 0.0 or rate < _EPS:
        return phone_quat
    dq = quat_from_axis_angle(omega_rad_s, rate * dt)
    return quat_normalize(quat_mul(phone_quat, dq))


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
    pv = vector_from_yaw_pitch(pointer_yaw_deg, pointer_pitch_deg)
    best_id: str | None = None
    best_d = 0.0
    current_d: float | None = None
    for e in entities:
        d = _angle_between(pv, vector_from_yaw_pitch(e.world_yaw_deg, e.world_pitch_deg))
        if d > radius_deg:
            continue
        if e.id == current_id:
            current_d = d
        if best_id is None or d < best_d or (d == best_d and e.id < best_id):
            best_id, best_d = e.id, d
    if best_id is None:
        return None
    if current_d is not None and best_id != current_id and best_d > current_d - hysteresis_deg:
        return current_id
    return best_id
