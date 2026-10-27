"""Reference geo math for the events/vessels tests, until world.py v2 has real bodies (W7).

Spherical Earth (R = 6_371_000 m) instead of WGS84; accurate to well under 0.5 deg at the ranges
these tests use. ``install(monkeypatch)`` patches ``look_angles`` and ``in_view`` on
``pointview.world``; drop the call after W7 merges.
"""

from __future__ import annotations

import math

from pointview import world

R = 6_371_000.0


def _ecef(lat: float, lon: float, alt: float) -> tuple[float, float, float]:
    p, l_ = math.radians(lat), math.radians(lon)
    r = R + alt
    return (r * math.cos(p) * math.cos(l_), r * math.cos(p) * math.sin(l_), r * math.sin(p))


def look_angles(
    obs_lat: float, obs_lon: float, obs_alt_m: float, lat: float, lon: float, alt_m: float
) -> tuple[float, float, float]:
    o = _ecef(obs_lat, obs_lon, obs_alt_m)
    t = _ecef(lat, lon, alt_m)
    d = (t[0] - o[0], t[1] - o[1], t[2] - o[2])
    p, l_ = math.radians(obs_lat), math.radians(obs_lon)
    east = (-math.sin(l_), math.cos(l_), 0.0)
    north = (-math.sin(p) * math.cos(l_), -math.sin(p) * math.sin(l_), math.cos(p))
    up = (math.cos(p) * math.cos(l_), math.cos(p) * math.sin(l_), math.sin(p))
    e = sum(a * b for a, b in zip(d, east, strict=True))
    n = sum(a * b for a, b in zip(d, north, strict=True))
    u = sum(a * b for a, b in zip(d, up, strict=True))
    rng = math.sqrt(e * e + n * n + u * u)
    return (math.degrees(math.atan2(e, n)) % 360.0, math.degrees(math.asin(u / rng)), rng)


def in_view(
    world_yaw_deg: float,
    world_pitch_deg: float,
    head_quat,
    north_offset_deg: float,
    display_fov_deg: tuple[float, float],
    margin_deg: float = 0.0,
) -> bool:
    a, b, front = world.head_angles_deg(world_yaw_deg, world_pitch_deg, head_quat, north_offset_deg)
    return (
        front
        and abs(a) <= display_fov_deg[0] / 2 + margin_deg
        and abs(b) <= display_fov_deg[1] / 2 + margin_deg
    )


def install(monkeypatch) -> None:
    monkeypatch.setattr(world, "look_angles", look_angles)
    monkeypatch.setattr(world, "in_view", in_view)
