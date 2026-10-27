"""Offline stand-in for Overpass + Wikipedia, serving the files in this folder.

Reusable from any test::

    from tests.fixtures.landmarks.mock_transport import FixtureServer, PHONE_LAT, PHONE_LON

    server = FixtureServer()
    layer = LandmarkLayer(tmp_path / "cache.sqlite", client=server.client())

Overpass: the ``around:R,lat,lon`` of the posted query is honoured (elements outside R are dropped).
Wikipedia: ``/api/rest_v1/page/summary/<title>`` is answered from ``wikipedia/*.json`` (404 else).
Set ``server.overpass_script = [429, 504]`` to make the next Overpass calls fail with those
statuses first; ``server.offline = True`` raises ``httpx.ConnectError`` for everything.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from pathlib import Path

import httpx

from tests.fixtures.landmarks import _world_ref

HERE = Path(__file__).parent

#: Where the fictional wearer stands: Crissy Field, San Francisco (a real place).
PHONE_LAT = 37.8060
PHONE_LON = -122.4680
#: Landmarks inside 2 km of PHONE_LAT/PHONE_LON: OSM entity id -> (name, wikipedia tag or None).
EXPECTED_LANDMARKS = {
    "osm:way/900000001": ("Golden Gate Bridge", "en:Golden Gate Bridge"),
    "osm:way/900000002": ("Fort Point", "en:Fort Point, San Francisco"),
    "osm:way/900000003": ("Palace of Fine Arts", "en:Palace of Fine Arts"),
    "osm:relation/900000005": ("Crissy Field", "en:Crissy Field"),
    "osm:node/900000004": ("Torpedo Wharf", None),
    "osm:node/900000006": ("Presidio Yacht Club", None),
}

_AROUND = re.compile(r"around:(\d+),(-?\d+\.\d+),(-?\d+\.\d+)")


def wiki_fixture_name(tag: str) -> str:
    return urllib.parse.quote(tag, safe="").replace("%", "_") + ".json"


def _position(el: dict) -> tuple[float, float]:
    c = el.get("center", el)
    return c["lat"], c["lon"]


class FixtureServer:
    def __init__(self) -> None:
        self.overpass_calls: list[str] = []  # posted queries
        self.wiki_calls: list[str] = []  # request paths
        self.overpass_script: list[int] = []
        self.offline = False
        self.user_agents: list[str] = []
        self.transport = httpx.MockTransport(self._handle)
        self._overpass = json.loads((HERE / "overpass_response.json").read_text())

    def client(self) -> httpx.Client:
        return httpx.Client(transport=self.transport)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.user_agents.append(request.headers.get("user-agent", ""))
        if self.offline:
            raise httpx.ConnectError("offline", request=request)
        if "wikipedia.org" in request.url.host:
            return self._wiki(request)
        if request.url.path.endswith("/interpreter"):
            return self._overpass_response(request)
        return httpx.Response(404)

    def _overpass_response(self, request: httpx.Request) -> httpx.Response:
        form = urllib.parse.parse_qs(request.content.decode())
        query = (form.get("data") or [request.url.params.get("data", "")])[0]
        self.overpass_calls.append(query)
        if self.overpass_script:
            return httpx.Response(self.overpass_script.pop(0), text="busy")
        elements = self._overpass["elements"]
        m = _AROUND.search(query)
        if m:
            r, lat, lon = float(m[1]), float(m[2]), float(m[3])
            elements = [e for e in elements if _world_ref.haversine_m(lat, lon, *_position(e)) <= r]
        return httpx.Response(200, json={**self._overpass, "elements": elements})

    def _wiki(self, request: httpx.Request) -> httpx.Response:
        self.wiki_calls.append(request.url.path)
        lang = request.url.host.split(".")[0]
        prefix = "/api/rest_v1/page/summary/"
        if not request.url.path.startswith(prefix):
            return httpx.Response(404)
        title = urllib.parse.unquote(request.url.path[len(prefix) :]).replace("_", " ")
        f = HERE / "wikipedia" / wiki_fixture_name(f"{lang}:{title}")
        if not f.exists():
            return httpx.Response(404, json={"title": "Not found."})
        return httpx.Response(200, json=json.loads(f.read_text()))
