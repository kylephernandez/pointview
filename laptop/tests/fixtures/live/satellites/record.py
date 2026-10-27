"""Record real CelesTrak TLE responses for the satellite tests (run by hand; tests never do this).

    python record.py

Writes tle_stations.txt and tle_visual.txt next to this file, unedited, from
https://celestrak.org/NORAD/elements/gp.php?GROUP=<group>&FORMAT=tle
CelesTrak asks for at most one download of the same data per 2 hours.
"""

from pathlib import Path

import httpx

HERE = Path(__file__).parent
for group in ("stations", "visual"):
    r = httpx.get(
        "https://celestrak.org/NORAD/elements/gp.php",
        params={"GROUP": group, "FORMAT": "tle"},
        headers={"User-Agent": "PointView-fixture-recorder/0.1"},
        timeout=30,
    )
    r.raise_for_status()
    (HERE / f"tle_{group}.txt").write_bytes(r.content)
    print(group, len(r.content), "bytes")
