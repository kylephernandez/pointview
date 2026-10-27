"""Reference WGS84 geodetic -> ENU used to stand in for the world.enu_from_geodetic stub in tests."""

import numpy as np

_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)


def ecef(lat, lon, alt):
    lat, lon, alt = (
        np.radians(np.asarray(v, np.float64)) if i < 2 else np.asarray(v, np.float64)
        for i, v in enumerate((lat, lon, alt))
    )
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    return np.stack(
        [
            (n + alt) * np.cos(lat) * np.cos(lon),
            (n + alt) * np.cos(lat) * np.sin(lon),
            (n * (1 - _E2) + alt) * np.sin(lat),
        ],
        axis=-1,
    )


def enu_from_geodetic(origin_lat, origin_lon, origin_alt_m, lat, lon, alt_m):
    d = ecef(lat, lon, alt_m) - ecef(origin_lat, origin_lon, origin_alt_m)
    la, lo = np.radians(origin_lat), np.radians(origin_lon)
    e = np.array([-np.sin(lo), np.cos(lo), 0.0])
    n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    u = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return np.stack([d @ e, d @ n, d @ u], axis=-1)
