"""Reference geo math for tests, until world.py's bodies exist (W3 implements them in parallel).

Spherical Earth, radius 6_371_000 m (same as the contract). ``install(monkeypatch)`` patches
``pointview.world.haversine_m`` / ``bearing_deg``; the lead drops the call after merging W3.
"""

from __future__ import annotations

import math

R = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360.0


def install(monkeypatch) -> None:
    from pointview import world

    monkeypatch.setattr(world, "haversine_m", haversine_m)
    monkeypatch.setattr(world, "bearing_deg", bearing_deg)
