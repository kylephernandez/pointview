# Satellite fixtures (W9)

| File | Provenance |
| --- | --- |
| `tle_stations.txt`, `tle_visual.txt` | **Recorded for real** from CelesTrak (`gp.php?GROUP=<g>&FORMAT=tle`) on 2026-10-27, unedited. ISS element epoch 2026-10-26 05:01 UTC. `record.py` re-records them (or use curl). |
| `reference.json` | **Generated** by `make_reference.py` with Skyfield (scratch venv, not a dependency): az/el/range and subpoint of 16 station + 8 visual satellites for 3 observers at 4 times, plus the ISS passes above 10 degrees (rise / peak / set) over the next 24 h. Regenerate it whenever the TLE files change. |

`t0_unix` in the JSON is the fixed test time (2 h after the ISS element epoch). Tests never use the network.
