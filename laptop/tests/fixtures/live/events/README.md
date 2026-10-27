# Events and vessels fixtures (W10)

Used by `tests/live/test_events.py`.

| File | Provenance |
| --- | --- |
| `summary_2.5_day.geojson` | **RECORDED for real** from `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson` on 2026-10-27 (feed `generated` = 1793064128000 ms, 32 quakes), unedited. Re-record with `record_usgs.py`. |
| `detail_us7000ttci.geojson` | **RECORDED for real** from `.../feed/v1.0/detail/us7000ttci.geojson` (M4.7 south of Encounter Bay, Australia; 27 felt reports, tsunami 0), unedited. |
| `ais_messages.json` | **HAND-WRITTEN, not recorded.** AISStream needs an API key we do not have. The message shapes follow the AISStream documentation (https://aisstream.io/documentation): `MessageType`, `MetaData` (`MMSI`, `ShipName`, `latitude`, `longitude`, `time_utc`) and `Message.<MessageType>` (`PositionReport` with `Cog`/`Sog`/`TrueHeading`/`UserID`, `ShipStaticData` with `Destination`/`Eta`/`Type`/`ImoNumber`). MMSIs, names, IMO and callsign are made up. Includes the AIS "not available" sentinels (Sog 102.3, Cog 360, lat 91 / lon 181) and an error envelope. Field names were typed from the docs, never checked against a live stream. |

Test geometry: the quake test observer stands in Adelaide (-34.93, 138.60), about 170 km from the
Encounter Bay quake; the vessel tests use Crissy Field, San Francisco (37.806, -122.468).
