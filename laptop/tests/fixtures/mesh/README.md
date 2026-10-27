# Mesh fixtures (real data, recorded once)

Recorded on 2026-10-27 around the mock location, lat 37.8060, lon -122.4680 (Crissy Field /
Presidio, San Francisco) by `record.py` in this folder. Tests never touch the network.

## overpass_buildings.json

**Real OSM data, fetched via the OSM API 0.6 (`/api/0.6/map?bbox=-122.472,37.803,-122.464,37.809`)
and converted to the Overpass `out geom` JSON format** by `record.py`. 49 building ways.

Why: Overpass is not reachable from the build container. The production code
(`pointview/mesh/osm_buildings.py`) queries Overpass only. The OSM 0.6 API is meant for editing
and must not be used by the app; it was used here for this one-off recording.

The conversion keeps each way's tags and its node coordinates as `geometry`, which is the shape
`out geom` returns. A real Overpass response also carries `bounds` and a header; the parser does
not use them.

Licence: © OpenStreetMap contributors, Open Database Licence (ODbL) 1.0,
https://www.openstreetmap.org/copyright . Attribution string in the app:
"© OpenStreetMap contributors".

## terrarium_14_{x}_{y}.png

Four AWS Terrain Tiles in Terrarium encoding (zoom 14, x 2617-2618, y 6330-6331), unedited bytes
from `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/14/{x}/{y}.png`. They cover about
1 km around the point. Elevation = (R*256 + G + B/256) - 32768 m.

Licence and attribution: the tiles combine public-domain and open datasets (USGS 3DEP/NED, SRTM,
GMTED2010, NOAA ETOPO1 and national datasets elsewhere). Required attribution for the Mapzen/AWS
hosted tiles, see
https://github.com/tilezen/joerd/blob/master/docs/attribution.md and the data-source details at
https://github.com/tilezen/joerd/blob/master/docs/data-sources.md . The AWS Open Data listing is
https://registry.opendata.aws/terrain-tiles/ . The app shows
"Terrain: Mapzen/AWS Terrain Tiles (USGS 3DEP, SRTM, GMTED2010; NOAA ETOPO1)".

Note the tiles include bathymetry: points over San Francisco Bay have negative elevation
(for example -13 m at 37.8100, -122.4600).
