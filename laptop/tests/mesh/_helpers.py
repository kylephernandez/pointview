"""Shared test helpers for the mesh tests."""

from __future__ import annotations

import cv2
import httpx
import numpy as np

#: Mock location (Crissy Field / Presidio, San Francisco); the fixtures were recorded around it.
LAT, LON = 37.8060, -122.4680


def encode_terrarium(elev: np.ndarray) -> bytes:
    v = (elev.astype(np.float64) + 32768.0) * 256.0
    v = np.round(v).astype(np.int64)
    r, g, b = v // 65536, (v // 256) % 256, v % 256
    img = np.dstack([b, g, r]).astype(np.uint8)  # OpenCV is BGR
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class FakeServer:
    """Serves the recorded Overpass JSON and Terrarium tiles; counts requests; can go offline."""

    def __init__(self, buildings: str, tiles: dict):
        self.buildings, self.tiles = buildings, tiles
        self.overpass_calls = 0
        self.tile_calls = 0
        self.offline = False
        self.overpass_queries: list[str] = []
        self.user_agents: set[str] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.user_agents.add(request.headers.get("user-agent", ""))
        if self.offline:
            raise httpx.ConnectError("offline", request=request)
        if request.method == "POST" and "interpreter" in request.url.path:
            self.overpass_calls += 1
            self.overpass_queries.append(request.content.decode())
            return httpx.Response(200, text=self.buildings)
        if "amazonaws" in request.url.host:
            self.tile_calls += 1
            parts = request.url.path.strip("/").split("/")  # .../terrarium/z/x/y.png
            z, x, y = int(parts[-3]), int(parts[-2]), int(parts[-1].removesuffix(".png"))
            png = self.tiles.get((z, x, y))
            return httpx.Response(200, content=png) if png else httpx.Response(404)
        return httpx.Response(404)
