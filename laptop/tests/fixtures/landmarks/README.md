# Landmark fixtures (W6)

Used by `tests/test_landmarks.py`; reusable from other tests (e.g. the lead's end-to-end mock test).

## Where the wearer stands

**`PHONE_LAT = 37.8060`, `PHONE_LON = -122.4680`** (Crissy Field, San Francisco waterfront).
They are constants in `mock_transport.py`. From there the Golden Gate Bridge is about 1.8 km to the
north-west (bearing about 329 deg), Fort Point about 1.1 km west, the Palace of Fine Arts about
1.6 km east. Six named landmarks are inside the 2 km radius (`EXPECTED_LANDMARKS`), one is outside
(Alcatraz), one has no name (a bus stop; must be skipped).

## Provenance: what is recorded and what is hand-written

| File | Provenance |
| --- | --- |
| `wikipedia/*.json` | **Recorded for real** from the Wikipedia REST API (`/api/rest_v1/page/summary/<title>`) by `record_wikipedia.py`, unedited. See the table below for which tags are present. |
| `overpass_response.json` | **HAND-WRITTEN, not recorded.** The Overpass API hosts are unreachable from the dev container, so this file only imitates the exact Overpass JSON output format (`elements` with `type`/`id`/`lat`+`lon` or `center`/`tags`). The OSM ids (`9000000xx`) are made up. Coordinates of the Golden Gate Bridge come from the recorded Wikipedia summary; the others are approximate (typed from memory, good to a few tens of meters) and "Torpedo Wharf" / "Presidio Yacht Club" are plausible positions, not surveyed ones. The Alcatraz entry sits outside the 2 km radius on purpose; the bus stop has no `name` on purpose. |

Do not treat `overpass_response.json` as proof that the real Overpass query returns these features.
A recorded response should replace it the first time someone has a machine that can reach Overpass
(`@pytest.mark.network` test `test_live_apis` hits the live APIs when `POINTVIEW_NETWORK_TESTS=1`).

## Files

* `mock_transport.py`: `FixtureServer` serves both APIs through `httpx.MockTransport`.
  ```python
  from tests.fixtures.landmarks.mock_transport import FixtureServer, PHONE_LAT, PHONE_LON
  server = FixtureServer()
  layer = LandmarkLayer(tmp_path / "cache.sqlite", client=server.client())
  ```
  Knobs: `server.overpass_script = [429, 504]` (fail first), `server.offline = True`; records
  `overpass_calls`, `wiki_calls`, `user_agents`. It honours the `around:R,lat,lon` in the query.
* `_world_ref.py`: reference `haversine_m` / `bearing_deg`; `install(monkeypatch)` patches them onto
  `pointview.world` while W3's bodies are stubs. Drop the call after merging W3.
* `record_wikipedia.py`: re-records `wikipedia/*.json` (needs `*.wikipedia.org`; it backs off on
  403/429, which a shared egress IP can trigger).

## Tags with a recorded summary

See `wikipedia/`; the file name is the `wikipedia` tag, percent-encoded with `%` replaced by `_`.
