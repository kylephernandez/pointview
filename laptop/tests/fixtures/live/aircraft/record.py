"""Record the real adsb.lol / OpenSky / route responses used by tests/live/test_aircraft.py.

Run by hand (needs internet), from ``laptop/``::

    python -I tests/fixtures/live/aircraft/record.py [lat lon]

Writes next to this script: adsblol_point.json, adsblol_mil.json (trimmed to 3 aircraft),
opensky_states.json, route_<CALLSIGN>.json and manifest.json (when/where recorded). Responses are
stored as received; only the ``ac`` list of the mil response is truncated. Re-record when a source
changes its format; tests do not depend on the data being current (they pin ``unix_time`` from the
manifest).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
UA = "PointView/0.1 (fixture recorder; open-source AR labels; https://github.com/kylephernandez/pointview)"
LAT, LON, RADIUS_NM = 40.64, -73.78, 25  # near JFK: lots of traffic, ~46 km radius


def main() -> None:
    lat = float(sys.argv[1]) if len(sys.argv) > 2 else LAT
    lon = float(sys.argv[2]) if len(sys.argv) > 2 else LON
    c = httpx.Client(headers={"User-Agent": UA}, timeout=20, follow_redirects=True)

    point = c.get(f"https://api.adsb.lol/v2/point/{lat}/{lon}/{RADIUS_NM}")
    point.raise_for_status()
    (HERE / "adsblol_point.json").write_text(point.text)
    ac = point.json()["ac"]
    time.sleep(1.5)

    mil = c.get("https://api.adsb.lol/v2/mil")
    mil.raise_for_status()
    m = mil.json()
    m["ac"] = m.get("ac", [])[:3]
    (HERE / "adsblol_mil.json").write_text(json.dumps(m, indent=1))
    time.sleep(1.5)

    dlat = RADIUS_NM * 1.852 / 111.0
    dlon = dlat / math.cos(math.radians(lat))
    os_url = (
        "https://opensky-network.org/api/states/all"
        f"?lamin={lat - dlat:.4f}&lamax={lat + dlat:.4f}"
        f"&lomin={lon - dlon:.4f}&lomax={lon + dlon:.4f}"
    )
    sky = c.get(os_url)
    sky.raise_for_status()
    (HERE / "opensky_states.json").write_text(sky.text)

    calls = [a["flight"].strip() for a in ac if a.get("flight", "").strip()][:2]
    for cs in calls:
        time.sleep(1.5)
        r = c.get(f"https://vrs-standing-data.adsb.lol/routes/{cs[:2]}/{cs}.json")
        if r.status_code == 200:
            (HERE / f"route_{cs}.json").write_text(r.text)
    (HERE / "manifest.json").write_text(
        json.dumps(
            {
                "recorded_unix": time.time(),
                "adsblol_now_ms": point.json().get("now"),
                "opensky_time": sky.json().get("time"),
                "phone_lat": lat,
                "phone_lon": lon,
                "radius_nm": RADIUS_NM,
                "route_callsigns": calls,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
