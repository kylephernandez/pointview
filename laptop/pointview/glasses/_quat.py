"""Small private quaternion helpers (w, x, y, z; Hamilton). Pure Python, no numpy needed."""

from __future__ import annotations

import math

from pointview.interfaces import Quat


def mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def conj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def normalize(q: Quat) -> Quat:
    n = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    if n < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def dot(a: Quat, b: Quat) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def slerp(a: Quat, b: Quat, u: float) -> Quat:
    """Shortest-path spherical interpolation, ``u`` in [0, 1] (values outside extrapolate)."""
    d = dot(a, b)
    if d < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
        d = -d
    if d > 0.9995:  # nearly parallel: normalized lerp
        return normalize(
            (
                a[0] + u * (b[0] - a[0]),
                a[1] + u * (b[1] - a[1]),
                a[2] + u * (b[2] - a[2]),
                a[3] + u * (b[3] - a[3]),
            )
        )
    theta = math.acos(min(1.0, d))
    s = math.sin(theta)
    wa = math.sin((1.0 - u) * theta) / s
    wb = math.sin(u * theta) / s
    return normalize(
        (
            wa * a[0] + wb * b[0],
            wa * a[1] + wb * b[1],
            wa * a[2] + wb * b[2],
            wa * a[3] + wb * b[3],
        )
    )


def rotvec(q: Quat) -> tuple[float, float, float]:
    """Rotation vector (axis * angle, rad) of the shortest rotation represented by ``q``."""
    w, x, y, z = q
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-12:
        return (0.0, 0.0, 0.0)
    k = 2.0 * math.atan2(s, w) / s
    return (x * k, y * k, z * k)


def from_rotvec(v: tuple[float, float, float]) -> Quat:
    ang = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if ang < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(ang / 2) / ang
    return (math.cos(ang / 2), v[0] * s, v[1] * s, v[2] * s)
