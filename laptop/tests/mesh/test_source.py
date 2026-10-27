from __future__ import annotations

import json
import time
from urllib.parse import unquote_plus

import httpx
import numpy as np
import pytest

from pointview.interfaces import MeshSource, WorldMesh
from pointview.mesh import OsmTerrainMeshSource
from pointview.mesh import source as S
from tests.mesh._helpers import LAT, LON, FakeServer

M_LAT = 111_320.0


def wait_until(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def make(tmp_path, client, **kw):
    kw.setdefault("backoff_s", 0.05)
    return OsmTerrainMeshSource(tmp_path / "mesh", client, **kw)


def test_implements_protocol_and_builds_mesh(tmp_path, client, server):
    src = make(tmp_path, client)
    try:
        assert isinstance(src, MeshSource)
        assert src.mesh_near(LAT, LON) is None  # nothing yet, and it did not block
        assert src.wait_for_mesh(10)
        m = src.mesh_near(LAT, LON)
        assert isinstance(m, WorldMesh)
        assert m.source == "osm-buildings+terrarium"
        assert "OpenStreetMap" in m.attribution and "Mapzen" in m.attribution
        assert m.vertices.dtype == np.float32 and m.edges.dtype == np.int32
        assert m.origin.lat == LAT and m.origin.lon == LON
        assert m.fetched_at == pytest.approx(time.time(), abs=60)
        assert m.edges.max() < len(m.vertices)
        h = src.ground_alt_m(LAT, LON)
        assert h == pytest.approx(m.origin.alt_m) and 2 < h < 6
        assert src.ground_alt_m(0.0, 0.0) is None
        assert server.overpass_calls == 1 and server.tile_calls == 4
        assert "(around:800,37.806000,-122.468000)" in unquote_plus(server.overpass_queries[0])
        assert server.user_agents == {"PointView/0.1 (https://github.com/kylephernandez/pointview)"}
    finally:
        src.close()


def test_default_client_sends_polite_user_agent(tmp_path):
    src = OsmTerrainMeshSource(tmp_path)
    try:
        assert src._client.headers["user-agent"].startswith("PointView/0.1 (https://github.com")
    finally:
        src.close()


def test_cache_roundtrip_and_offline_startup(tmp_path, client, server):
    src = make(tmp_path, client)
    src.mesh_near(LAT, LON)
    assert src.wait_for_mesh(10)
    first = src.mesh_near(LAT, LON)
    src.close()
    assert (tmp_path / "mesh" / "last_mesh.npz").exists()
    meta = json.loads((tmp_path / "mesh" / "last_mesh.json").read_text())
    assert meta["source"] == "osm-buildings+terrarium" and len(meta["tiles"]) == 4

    server.offline = True
    calls = (server.overpass_calls, server.tile_calls)
    src2 = make(tmp_path, client)
    try:
        t0 = time.perf_counter()
        m = src2.mesh_near(LAT, LON)  # the very first call already has the cached mesh
        assert time.perf_counter() - t0 < 0.05
        assert m is not None
        assert np.array_equal(m.vertices, first.vertices) and np.array_equal(m.edges, first.edges)
        assert m.origin == first.origin and m.source == first.source
        assert m.attribution == first.attribution and m.fetched_at == first.fetched_at
        assert src2.ground_alt_m(LAT, LON) == pytest.approx(first.origin.alt_m)
        time.sleep(0.3)
        assert (server.overpass_calls, server.tile_calls) == calls  # fresh cache: no refetch
    finally:
        src2.close()


def test_offline_with_stale_cache_keeps_mesh_and_backs_off(tmp_path, client, server, monkeypatch):
    src = make(tmp_path, client)
    src.mesh_near(LAT, LON)
    assert src.wait_for_mesh(10)
    first = src.mesh_near(LAT, LON)
    src.close()

    server.offline = True
    monkeypatch.setattr(S, "CACHE_FRESH_S", -1.0)  # everything is "stale": wants a refresh
    src2 = make(tmp_path, client, backoff_s=0.2)
    try:
        assert src2.mesh_near(LAT, LON) is not None
        # The cached Overpass response is stale too, so the worker tries the network and fails;
        # the old mesh stays and nothing crashes.
        assert wait_until(lambda: src2.last_error is not None)
        assert src2.mesh_near(LAT, LON).edges.shape == first.edges.shape
    finally:
        src2.close()


def test_mesh_of_another_place_is_not_returned(tmp_path, client, server):
    src = make(tmp_path, client)
    src.mesh_near(LAT, LON)
    assert src.wait_for_mesh(10)
    src.close()
    server.offline = True
    src2 = make(tmp_path, client)
    try:
        assert src2.mesh_near(LAT, LON) is not None
        assert src2.mesh_near(48.8584, 2.2945) is None  # a stale cache from a different city
    finally:
        src2.close()


def test_refetch_after_moving_more_than_refetch_m(tmp_path, client, server):
    src = make(tmp_path, client, refetch_m=200)
    try:
        src.mesh_near(LAT, LON)
        assert wait_until(lambda: src.builds == 1)
        tiles_after_first = server.tile_calls

        # 100 m away: still the same mesh, no new request
        src.mesh_near(LAT + 100 / M_LAT, LON)
        time.sleep(0.4)
        assert src.builds == 1 and server.overpass_calls == 1

        # 300 m away: a new mesh centred on the new position (terrain tiles come from the cache)
        lat2 = LAT + 300 / M_LAT
        src.mesh_near(lat2, LON)
        assert wait_until(lambda: src.builds == 2)
        assert server.overpass_calls == 2
        assert server.tile_calls == tiles_after_first
        m = src.mesh_near(lat2, LON)
        assert m.origin.lat == pytest.approx(lat2)
        # the previous position is still within the new mesh's 800 m reach
        assert src.mesh_near(LAT, LON) is m
    finally:
        src.close()


def test_returning_near_a_cached_query_skips_overpass(tmp_path, client, server):
    src = make(tmp_path, client, refetch_m=200)
    try:
        src.mesh_near(LAT, LON)
        assert wait_until(lambda: src.builds == 1)
        src.mesh_near(LAT + 300 / M_LAT, LON)
        assert wait_until(lambda: src.builds == 2)
        src.mesh_near(LAT + 10 / M_LAT, LON)  # back within 50 m of the first query
        assert wait_until(lambda: src.builds == 3)
        assert server.overpass_calls == 2
    finally:
        src.close()


def test_buildings_failure_gives_terrain_only_then_retries(tmp_path, buildings_json, tile_pngs):
    srv = FakeServer(buildings_json, tile_pngs)
    real = srv.__call__
    broken = {"on": True}

    def handler(req: httpx.Request) -> httpx.Response:
        if broken["on"] and req.method == "POST":
            return httpx.Response(503)
        return real(req)

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        src = make(tmp_path, c, backoff_s=0.05)
        try:
            src.mesh_near(LAT, LON)
            assert wait_until(lambda: src.builds == 1)
            m = src.mesh_near(LAT, LON)
            assert m.source == "terrarium" and len(m.edges) > 0
            assert src.ground_alt_m(LAT, LON) is not None
            broken["on"] = False
            assert wait_until(
                lambda: src.builds >= 2 and src.mesh_near(LAT, LON).source != "terrarium"
            )
            assert src.mesh_near(LAT, LON).source == "osm-buildings+terrarium"
            assert src.last_error is None
        finally:
            src.close()


def test_failure_backoff_limits_requests(tmp_path, client, server):
    server.offline = True
    src = make(tmp_path, client, backoff_s=0.5)
    try:
        src.mesh_near(LAT, LON)
        assert wait_until(lambda: src.last_error is not None)
        time.sleep(0.3)
        for _ in range(20):
            src.mesh_near(LAT, LON)
            time.sleep(0.01)
        assert src._failures <= 2  # not one attempt per call
        assert src.mesh_near(LAT, LON) is None and src.ground_alt_m(LAT, LON) is None
    finally:
        src.close()


def test_corrupt_disk_cache_is_ignored(tmp_path, client):
    d = tmp_path / "mesh"
    d.mkdir()
    (d / "last_mesh.json").write_text("{not json")
    (d / "last_mesh.npz").write_bytes(b"garbage")
    src = make(tmp_path, client)
    try:
        assert src.mesh_near(LAT, LON) is None
        assert src.wait_for_mesh(10)
        assert src.mesh_near(LAT, LON) is not None
    finally:
        src.close()


def test_mesh_near_never_blocks(tmp_path, buildings_json, tile_pngs):
    """With a network that takes 0.5 s per request, mesh_near still returns in well under 1 ms."""
    srv = FakeServer(buildings_json, tile_pngs)

    def slow(req: httpx.Request) -> httpx.Response:
        time.sleep(0.5)
        return srv(req)

    with httpx.Client(transport=httpx.MockTransport(slow)) as c:
        src = make(tmp_path, c)
        try:
            src.mesh_near(LAT, LON)  # starts the thread
            times = []
            for i in range(300):
                t0 = time.perf_counter()
                src.mesh_near(LAT + (i % 5) * 1e-4, LON)
                times.append(time.perf_counter() - t0)
            assert max(times) < 0.01
            assert float(np.median(times)) < 0.001
            assert src.builds == 0  # still fetching: the loop never waited for it
            assert wait_until(lambda: src.builds == 1, 20)
            times = []
            for _ in range(300):
                t0 = time.perf_counter()
                assert src.mesh_near(LAT, LON) is not None
                src.ground_alt_m(LAT, LON)
                times.append(time.perf_counter() - t0)
            assert float(np.median(times)) < 0.001
        finally:
            src.close()


def test_close_is_idempotent(tmp_path, client):
    src = make(tmp_path, client)
    src.mesh_near(LAT, LON)
    src.close()
    src.close()
