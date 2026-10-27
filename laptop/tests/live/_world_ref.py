"""Reference implementations of world.py v2 geodesy, for tests until W7's bodies are merged.

WGS84 ECEF -> local ENU (contracts section 7/10). ``install(monkeypatch)`` patches
``pointview.world.ecef_from_geodetic`` / ``look_angles`` / ``in_view``; the lead drops the call after
merging W7. ``in_view`` uses the real (v1) ``world.head_angles_deg``.
"""

from __future__ import annotations

import math

A = 6_378_137.0
F = 1 / 298.257223563
E2 = F * (2 - F)


def ecef_from_geodetic(lat: float, lon: float, alt_m: float = 0.0):
    la, lo = math.radians(lat), math.radians(lon)
    n = A / math.sqrt(1 - E2 * math.sin(la) ** 2)
    return (
        (n + alt_m) * math.cos(la) * math.cos(lo),
        (n + alt_m) * math.cos(la) * math.sin(lo),
        (n * (1 - E2) + alt_m) * math.sin(la),
    )


def look_angles(obs_lat, obs_lon, obs_alt_m, lat, lon, alt_m):
    o = ecef_from_geodetic(obs_lat, obs_lon, obs_alt_m)
    p = ecef_from_geodetic(lat, lon, alt_m)
    dx, dy, dz = p[0] - o[0], p[1] - o[1], p[2] - o[2]
    la, lo = math.radians(obs_lat), math.radians(obs_lon)
    sl, cl, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    e = -so * dx + co * dy
    n = -sl * co * dx - sl * so * dy + cl * dz
    u = cl * co * dx + cl * so * dy + sl * dz
    rng = math.sqrt(e * e + n * n + u * u)
    return (
        math.degrees(math.atan2(e, n)) % 360.0,
        math.degrees(math.atan2(u, math.hypot(e, n))),
        rng,
    )


def in_view(yaw, pitch, head_quat, north_offset_deg, display_fov_deg, margin_deg=0.0):
    from pointview import world

    alpha, beta, front = world.head_angles_deg(yaw, pitch, head_quat, north_offset_deg)
    return (
        front
        and abs(alpha) <= display_fov_deg[0] / 2 + margin_deg
        and abs(beta) <= display_fov_deg[1] / 2 + margin_deg
    )


def install(monkeypatch) -> None:
    from pointview import world

    monkeypatch.setattr(world, "ecef_from_geodetic", ecef_from_geodetic)
    monkeypatch.setattr(world, "look_angles", look_angles)
    monkeypatch.setattr(world, "in_view", in_view)
