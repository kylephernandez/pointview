"""One-off recording script for the mesh fixtures (NOT used in production or tests).

Fetches real OSM data via the OSM 0.6 API (``map?bbox=``, meant for editing; never use it in
production code) and converts it to the Overpass ``out geom`` JSON shape that production parses.
Also downloads the Terrarium tiles unedited. Run: python record.py
"""

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

HERE = Path(__file__).parent
LAT, LON = 37.8060, -122.4680
DLAT, DLON = 0.0030, 0.0040  # ~330 m x ~350 m
UA = {"User-Agent": "PointView/0.1 (https://github.com/kylephernandez/pointview) fixture recording"}


def osm_to_overpass(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in root.iter("node")}
    ways = {}
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        refs = [nd.get("ref") for nd in w.findall("nd")]
        ways[w.get("id")] = (tags, refs)

    def geom(refs):
        return [{"lat": nodes[r][0], "lon": nodes[r][1]} for r in refs if r in nodes]

    elements = []
    for wid, (tags, refs) in ways.items():
        if "building" in tags:
            elements.append(
                {
                    "type": "way",
                    "id": int(wid),
                    "nodes": [int(r) for r in refs],
                    "tags": tags,
                    "geometry": geom(refs),
                }
            )
    for r in root.iter("relation"):
        tags = {t.get("k"): t.get("v") for t in r.findall("tag")}
        if "building" not in tags:
            continue
        members = []
        for m in r.findall("member"):
            if m.get("type") == "way" and m.get("ref") in ways:
                members.append(
                    {
                        "type": "way",
                        "ref": int(m.get("ref")),
                        "role": m.get("role"),
                        "geometry": geom(ways[m.get("ref")][1]),
                    }
                )
        elements.append(
            {"type": "relation", "id": int(r.get("id")), "members": members, "tags": tags}
        )
    return {
        "version": 0.6,
        "generator": "OSM API 0.6 map call converted to Overpass out geom",
        "elements": elements,
    }


def tile_xy(lat, lon, z):
    n = 2**z
    return (
        int((lon + 180) / 360 * n),
        int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n),
    )


def main():
    z = 14
    with httpx.Client(headers=UA, timeout=60) as c:
        bbox = f"{LON - DLON},{LAT - DLAT},{LON + DLON},{LAT + DLAT}"
        r = c.get(f"https://api.openstreetmap.org/api/0.6/map?bbox={bbox}")
        r.raise_for_status()
        data = osm_to_overpass(r.text)
        (HERE / "overpass_buildings.json").write_text(json.dumps(data, separators=(",", ":")))
        print(len(data["elements"]), "buildings")
        x0, y0 = tile_xy(LAT + 0.01, LON - 0.013, z)
        x1, y1 = tile_xy(LAT - 0.01, LON + 0.013, z)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
                t = c.get(url)
                t.raise_for_status()
                (HERE / f"terrarium_{z}_{x}_{y}.png").write_bytes(t.content)
                print("tile", x, y, len(t.content))


if __name__ == "__main__":
    sys.exit(main())
