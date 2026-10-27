"""Record real USGS fixtures into this directory.

Usage: python record_usgs.py [quake_id]   (needs network access to earthquake.usgs.gov)

Writes summary_2.5_day.geojson (the live feed, unedited) and detail_<id>.geojson for one quake id
(default: us7000ttci, which has felt reports in the 2026-10-27 recording). If you re-record, the
feed will contain different quakes: update EVENT_ID / the observer in tests/live/test_events.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

HERE = Path(__file__).parent
FEED = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
DETAIL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/{id}.geojson"


def main() -> None:
    qid = sys.argv[1] if len(sys.argv) > 1 else "us7000ttci"
    with httpx.Client(timeout=30, headers={"User-Agent": "PointView fixture recorder"}) as c:
        (HERE / "summary_2.5_day.geojson").write_bytes(c.get(FEED).raise_for_status().content)
        (HERE / f"detail_{qid}.geojson").write_bytes(
            c.get(DETAIL.format(id=qid)).raise_for_status().content
        )


if __name__ == "__main__":
    main()
