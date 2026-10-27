"""Mesh test helpers: the real recorded fixtures behind an httpx MockTransport. No network."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tests.mesh._helpers import FakeServer

MESH_FIXTURES = Path(__file__).parent.parent / "fixtures" / "mesh"


@pytest.fixture(scope="session")
def buildings_json() -> str:
    return (MESH_FIXTURES / "overpass_buildings.json").read_text()


@pytest.fixture(scope="session")
def tile_pngs() -> dict[tuple[int, int, int], bytes]:
    out = {}
    for p in MESH_FIXTURES.glob("terrarium_*.png"):
        _, z, x, y = p.stem.split("_")
        out[(int(z), int(x), int(y))] = p.read_bytes()
    return out


@pytest.fixture
def server(buildings_json, tile_pngs) -> FakeServer:
    return FakeServer(buildings_json, tile_pngs)


@pytest.fixture
def client(server):
    with httpx.Client(transport=httpx.MockTransport(server)) as c:
        yield c
