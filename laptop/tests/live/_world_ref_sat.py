"""Reference implementations of world.in_view and world.sun_direction for the satellite tests,
until W7's bodies exist. ``install(monkeypatch)`` patches ``pointview.world``.

* ``in_view``: yaw/pitch -> world vector -> tracking frame (subtract the north offset) -> head frame
  (inverse head quaternion); inside when in front and within the FOV widened by the margin.
* ``sun_direction``: NOAA "General Solar Position Calculations" (fractional-year equation of time and
  declination), a different algorithm from the Almanac one in satellites.py, good to about 0.5 deg.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np


def _quat_rotate(q, v):
    w, x, y, z = q
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return r @ v


def in_view(yaw, pitch, head_quat, north_offset_deg, display_fov_deg, margin_deg=0.0):
    ty = math.radians(yaw - north_offset_deg)
    p = math.radians(pitch)
    v_t = np.array([math.sin(ty) * math.cos(p), math.cos(ty) * math.cos(p), math.sin(p)])
    w, x, y, z = head_quat
    v_h = _quat_rotate((w, -x, -y, -z), v_t)  # inverse rotation: tracking -> head
    if v_h[1] <= 0:
        return False
    h = math.degrees(math.atan2(v_h[0], v_h[1]))
    v = math.degrees(math.atan2(v_h[2], v_h[1]))
    return (
        abs(h) <= display_fov_deg[0] / 2 + margin_deg
        and abs(v) <= display_fov_deg[1] / 2 + margin_deg
    )


def sun_direction(unix_time, lat, lon):
    t = datetime.fromtimestamp(unix_time, UTC)
    doy = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(g)
        - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2 * g)
        - 0.040849 * math.sin(2 * g)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(g)
        + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g)
        + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g)
        + 0.00148 * math.sin(3 * g)
    )
    tst = hour * 60 + eqtime + 4 * lon  # true solar time, minutes
    ha = math.radians(tst / 4 - 180)
    phi = math.radians(lat)
    sin_el = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(ha)
    el = math.asin(max(-1, min(1, sin_el)))
    # azimuth clockwise from north
    az = math.atan2(
        -math.sin(ha) * math.cos(decl),
        math.sin(decl) * math.cos(phi) - math.cos(decl) * math.sin(phi) * math.cos(ha),
    )
    return math.degrees(az) % 360.0, math.degrees(el)


def install(monkeypatch) -> None:
    from pointview import world

    monkeypatch.setattr(world, "in_view", in_view)
    monkeypatch.setattr(world, "sun_direction", sun_direction)
