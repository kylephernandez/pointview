"""Small reference implementation of the pointview.world functions W2 uses (contracts section 2).

world.py bodies are written by another worker; until they are merged, the detection tests patch
``pointview.world`` with these. Conventions (docs/contracts.md): quaternions (w, x, y, z), Hamilton;
World/Tracking x east, y north, z up, yaw clockwise from north; Head x right, y forward, z up;
Camera x right, y down, z forward; ``head = (c.x, c.z, -c.y)`` then rotated by camera_to_head_quat;
``Pose.quat`` rotates head into tracking; ``world_yaw = tracking_yaw + north_offset_deg``.
"""

from __future__ import annotations

import math

from pointview.interfaces import CameraIntrinsics, Quat, Vec3


def quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    _, x, y, z = quat_mul(quat_mul(q, (0.0, *v)), quat_conj(q))
    return (x, y, z)


def quat_axis_angle(axis: Vec3, angle_rad: float) -> Quat:
    s = math.sin(angle_rad / 2)
    return (math.cos(angle_rad / 2), axis[0] * s, axis[1] * s, axis[2] * s)


def quat_from_yaw_pitch(yaw_deg: float, pitch_deg: float = 0.0) -> Quat:
    """Head orientation looking at (yaw, pitch): yaw clockwise about +z, then pitch up."""
    qyaw = quat_axis_angle((0.0, 0.0, 1.0), -math.radians(yaw_deg))
    qpitch = quat_axis_angle((1.0, 0.0, 0.0), math.radians(pitch_deg))
    return quat_mul(qyaw, qpitch)


def camera_angles_deg(u: float, v: float, intr: CameraIntrinsics) -> tuple[float, float]:
    alpha = math.degrees(math.atan((u - intr.cx) / intr.fx))
    beta = math.degrees(math.atan((v - intr.cy) / intr.fy))
    return alpha, beta


def pixel_to_world(
    u: float,
    v: float,
    intr: CameraIntrinsics,
    head_quat: Quat,
    north_offset_deg: float,
    camera_to_head_quat: Quat,
) -> tuple[float, float]:
    cx, cy, cz = (u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0
    n = math.sqrt(cx * cx + cy * cy + cz * cz)
    c = (cx / n, cy / n, cz / n)
    head = quat_rotate(camera_to_head_quat, (c[0], c[2], -c[1]))
    t = quat_rotate(head_quat, head)
    phi = math.radians(north_offset_deg)  # rotate yaw clockwise by the offset
    wx = t[0] * math.cos(phi) + t[1] * math.sin(phi)
    wy = -t[0] * math.sin(phi) + t[1] * math.cos(phi)
    yaw = math.degrees(math.atan2(wx, wy)) % 360.0
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, t[2]))))
    return yaw, pitch
