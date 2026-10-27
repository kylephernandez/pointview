# Aircraft fixtures (W8)

Used by `tests/live/test_aircraft.py`. Tests never touch the network.

## Provenance: all recorded from the live services, unedited

Recorded by `record.py` (run by hand: `python -I tests/fixtures/live/aircraft/record.py` from `laptop/`;
re-run when a source changes its format or you want fresher data). `manifest.json` has the recording
time, the point queried (near JFK, 40.64 N, -73.78 E, 25 nm) and which callsigns got a route file.

| File | Source | Notes |
| --- | --- | --- |
| `adsblol_point.json` | `GET https://api.adsb.lol/v2/point/40.64/-73.78/25` | 95 aircraft, 53 on the ground (`alt_baro: "ground"`), none squawking an emergency code. |
| `adsblol_mil.json` | `GET https://api.adsb.lol/v2/mil` | Truncated to 3 aircraft; they carry `dbFlags: 1` (the military flag). The only edit. |
| `opensky_states.json` | `GET https://opensky-network.org/api/states/all?lamin=..&lomin=..&lamax=..&lomax=..` (bbox of the same 25 nm) | 83 state vectors, anonymous access. |
| `route_UAL*.json` | `GET https://vrs-standing-data.adsb.lol/routes/UA/<callsign>.json` | Route lookup used for the focused aircraft. |

Not recorded: a 429 response (tests synthesize one) and an emergency squawk (none were flying; tests
patch a 7700 into a copy of a real record).

## Why the route comes from vrs-standing-data

adsb.lol has `POST /api/0/routeset` (body `{"planes":[{"callsign","lat","lng"}]}`). On 2026-10-27 it
answered `201` with an EMPTY body for real callsigns, and `GET /api/0/route/<callsign>` answers
`302` to `vrs-standing-data.adsb.lol/routes/<2 letters>/<callsign>.json#deprecated`. So the layer uses
that static file directly (404 = unknown callsign). Routes are schedule data: the file does not know
which leg the aircraft is flying, so the layer shows the whole chain ("LAS -> EWR -> IAD").
