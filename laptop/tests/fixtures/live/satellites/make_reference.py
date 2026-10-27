"""Independent reference values for the satellite tests, computed with Skyfield (MIT).

    pip install skyfield   # in a scratch venv; NOT a PointView dependency
    python make_reference.py > reference.json

Skyfield propagates the same TLEs with the same SGP4 code, but does the TEME -> ITRS -> topocentric
reduction itself (equation of the equinoxes, its own WGS84 code), so it checks satellites.py's
GMST-only TEME->ECEF and its az/el/range. Polar motion is zero in both. It also gives the rise /
culmination / set times of the ISS passes above 10 degrees for the next-pass test, and the
geodetic subpoint (lat, lon, height) of each sample.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from skyfield.api import EarthSatellite, load, wgs84

HERE = Path(__file__).parent
ts = load.timescale(builtin=True)

OBSERVERS = {
    "sf": (37.8060, -122.4680),
    "sydney": (-33.8688, 151.2093),
    "tromso": (69.6496, 18.9560),
}
OFFSETS_S = [0, 600, 1800, 5400]
VISUAL_SAMPLE = 8


def read_tles(path):
    ls = [ln.rstrip() for ln in path.read_text().splitlines() if ln.strip()]
    return [(ls[i].strip(), ls[i + 1], ls[i + 2]) for i in range(0, len(ls) - 2, 3)]


def at(unix):
    return ts.from_datetime(datetime.fromtimestamp(unix, UTC))


stations = read_tles(HERE / "tle_stations.txt")
visual = read_tles(HERE / "tle_visual.txt")
sats = [EarthSatellite(l1, l2, name, ts) for name, l1, l2 in stations + visual[:VISUAL_SAMPLE]]
iss = next(s for s in sats if s.model.satnum == 25544)
T0 = round(iss.epoch.utc_datetime().timestamp() + 7200)  # 2 h after the ISS element epoch

out = {
    "t0_unix": T0,
    "offsets_s": OFFSETS_S,
    "samples": [],
    "iss_passes": {},
    "iss_peak_samples": {},
}
for oname, (lat, lon) in OBSERVERS.items():
    obs = wgs84.latlon(lat, lon, 0.0)
    for sat in sats:
        for off in OFFSETS_S:
            t = at(T0 + off)
            alt, az, dist = (sat - obs).at(t).altaz()
            sub = wgs84.subpoint(sat.at(t))
            out["samples"].append(
                {
                    "observer": oname,
                    "lat": lat,
                    "lon": lon,
                    "norad": sat.model.satnum,
                    "t_unix": T0 + off,
                    "az_deg": az.degrees,
                    "el_deg": alt.degrees,
                    "range_km": dist.km,
                    "sub_lat": sub.latitude.degrees,
                    "sub_lon": sub.longitude.degrees,
                    "sub_h_km": sub.elevation.km,
                }
            )
    times, events = iss.find_events(obs, at(T0), at(T0 + 86400), altitude_degrees=10.0)
    passes, cur = [], {}
    for t, ev in zip(times, events, strict=True):
        cur[["rise", "peak", "set"][ev]] = t.utc_datetime().timestamp()
        if ev == 2:
            peak_alt, _, _ = (iss - obs).at(at(cur["peak"])).altaz()
            cur["max_el_deg"] = peak_alt.degrees
            passes.append(cur)
            cur = {}
    out["iss_passes"][oname] = passes
    # ISS look angles at the whole second nearest each peak (the satellite is above the horizon)
    out["iss_peak_samples"][oname] = []
    for pas in passes:
        tp = round(pas["peak"])
        a, z, d = (iss - obs).at(at(tp)).altaz()
        out["iss_peak_samples"][oname].append(
            {"t_unix": tp, "az_deg": z.degrees, "el_deg": a.degrees, "range_km": d.km}
        )
print(json.dumps(out, indent=1))
