"""LandmarkLayer, Wikipedia summaries and attribution. No live network except the marked test."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from pointview import landmarks, world
from pointview.interfaces import (
    Alignment,
    Calibration,
    LayerContext,
    PhoneState,
)
from tests.fixtures.landmarks import _world_ref
from tests.fixtures.landmarks.mock_transport import (
    EXPECTED_LANDMARKS,
    PHONE_LAT,
    PHONE_LON,
    FixtureServer,
)

LANDMARK_ONLY = frozenset({"landmark"})


@pytest.fixture(autouse=True)
def _world_and_caches(monkeypatch):
    _world_ref.install(monkeypatch)  # lead: drop after merging W3 (world.py bodies)
    monkeypatch.setattr(landmarks, "_memory_cache", {})
    monkeypatch.setattr(landmarks, "_default_cache", None)


def phone(lat: float | None = PHONE_LAT, lon: float | None = PHONE_LON) -> PhoneState:
    return PhoneState(
        t_ms=0.0,
        t=0.0,
        received_at=0.0,
        quat=(1.0, 0.0, 0.0, 0.0),
        omega=(0.0, 0.0, 0.0),
        heading_deg=None,
        heading_acc=None,
        lat=lat,
        lon=lon,
        gps_acc=5.0,
    )


def ctx(p: PhoneState | None, enabled=frozenset({"person", "vehicle", "landmark"}), t=1.0):
    return LayerContext(
        t=t,
        glasses=None,  # type: ignore[arg-type]  (the landmark layer never touches the glasses)
        alignment=Alignment(),
        calibration=Calibration(),
        phone=p,
        enabled=enabled,
    )


def make_layer(tmp_path, server: FixtureServer, **kw) -> landmarks.LandmarkLayer:
    kw.setdefault("backoff_s", 0.05)
    return landmarks.LandmarkLayer(tmp_path / "cache.sqlite", client=server.client(), **kw)


def load(layer: landmarks.LandmarkLayer, p: PhoneState | None = None) -> list:
    layer.update(ctx(p or phone()))
    assert layer.wait_for_data(5), layer.last_error
    return layer.update(ctx(p or phone()))


def wait_until(fn, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if fn():
            return True
        time.sleep(0.01)
    return False


# ------------------------------------------------------------------------------- the layer


def test_attribution_strings():
    assert landmarks.OSM_ATTRIBUTION == "© OpenStreetMap contributors"
    assert landmarks.WIKIPEDIA_ATTRIBUTION == "Wikipedia · CC BY-SA"


def test_entities_from_fixture(tmp_path):
    server = FixtureServer()
    layer = make_layer(tmp_path, server)
    try:
        # never blocks: the very first call has nothing yet
        t0 = time.perf_counter()
        first = layer.update(ctx(phone()))
        assert time.perf_counter() - t0 < 0.005
        assert first == []
        assert layer.wait_for_data(5)
        ents = layer.update(ctx(phone(), t=2.5))
    finally:
        layer.close()

    assert {e.id for e in ents} == set(EXPECTED_LANDMARKS)  # >= 3, Alcatraz/unnamed excluded
    assert len(ents) >= 3
    by_id = {e.id: e for e in ents}
    for eid, (name, wiki) in EXPECTED_LANDMARKS.items():
        e = by_id[eid]
        assert e.kind == "landmark" and e.label == name
        assert e.world_pitch_deg == 0.0 and e.updated_at == 2.5
        assert e.meta["osm"] == eid.removeprefix("osm:")
        assert e.meta.get("wikipedia") == wiki
        assert 0.0 <= e.world_yaw_deg < 360.0
        assert 0 < e.distance_m <= 2000

    gg = by_id["osm:way/900000001"]
    assert gg.world_yaw_deg == pytest.approx(
        _world_ref.bearing_deg(PHONE_LAT, PHONE_LON, 37.81972222, -122.47861111)
    )
    assert gg.distance_m == pytest.approx(
        _world_ref.haversine_m(PHONE_LAT, PHONE_LON, 37.81972222, -122.47861111)
    )
    assert 320 < gg.world_yaw_deg < 340  # the bridge is north-west of the phone position
    assert server.overpass_calls and all(ua.startswith("PointView/") for ua in server.user_agents)
    q = server.overpass_calls[0]
    for needle in ("tourism", "historic", "place_of_worship", "natural", "peak", "building"):
        assert needle in q
    assert "out center" in q and "around:2000" in q and "nwr" in q


def test_uses_world_module_functions(tmp_path, monkeypatch):
    """Production code must go through pointview.world (W3's implementation)."""
    monkeypatch.setattr(world, "bearing_deg", lambda *a: 123.0)
    server = FixtureServer()
    layer = make_layer(tmp_path, server)
    try:
        ents = load(layer)
    finally:
        layer.close()
    assert ents and all(e.world_yaw_deg == 123.0 for e in ents)


def test_bearing_and_distance_follow_the_wearer_without_requery(tmp_path):
    server = FixtureServer()
    layer = make_layer(tmp_path, server)
    try:
        a = {e.id: e for e in load(layer)}
        moved = phone(PHONE_LAT + 0.0009, PHONE_LON)  # ~100 m north, under the 250 m threshold
        b = {e.id: e for e in layer.update(ctx(moved))}
        time.sleep(0.5)  # give a (wrong) requery the chance to happen
    finally:
        layer.close()
    assert len(server.overpass_calls) == 1
    gg = "osm:way/900000001"
    assert b[gg].distance_m == pytest.approx(a[gg].distance_m - 86, abs=6)  # bridge bears ~329 deg
    assert b[gg].world_yaw_deg != pytest.approx(a[gg].world_yaw_deg, abs=0.05)
    yacht = "osm:node/900000006"  # moving north puts it behind-ish: bearing swings
    assert abs(b[yacht].world_yaw_deg - a[yacht].world_yaw_deg) > 5


def test_requery_after_250m(tmp_path):
    server = FixtureServer()
    layer = make_layer(tmp_path, server)
    try:
        load(layer)
        far = phone(PHONE_LAT + 0.0030, PHONE_LON)  # ~330 m
        layer.update(ctx(far))
        assert wait_until(lambda: len(server.overpass_calls) == 2), len(server.overpass_calls)
        assert f"{PHONE_LAT + 0.0030:.6f}" in server.overpass_calls[1]
    finally:
        layer.close()


def test_hidden_without_fix_or_when_disabled(tmp_path):
    server = FixtureServer()
    layer = make_layer(tmp_path, server)
    try:
        assert len(load(layer)) >= 3
        assert layer.update(ctx(None)) == []
        assert layer.update(ctx(phone(None, None))) == []
        assert layer.update(ctx(phone(PHONE_LAT, None))) == []
        assert layer.update(ctx(phone(), enabled=frozenset({"person", "vehicle"}))) == []
        assert len(layer.update(ctx(phone(), enabled=LANDMARK_ONLY))) >= 3
    finally:
        layer.close()


def test_no_query_without_a_fix(tmp_path):
    server = FixtureServer()
    layer = make_layer(tmp_path, server)
    try:
        for _ in range(5):
            layer.update(ctx(None))
        time.sleep(0.4)
    finally:
        layer.close()
    assert server.overpass_calls == []


def test_cap_prefers_notable_then_near(tmp_path):
    server = FixtureServer()
    layer = make_layer(tmp_path, server, max_entities=3)
    try:
        ents = load(layer)
    finally:
        layer.close()
    assert len(ents) == 3
    # ranked by distance minus a bonus for features that have a Wikipedia article
    other = make_layer(tmp_path / "again", server)
    try:
        all_ents = {e.id: e for e in load(other)}
    finally:
        other.close()
    score = {
        i: e.distance_m - (landmarks.NOTABLE_BONUS_M if "wikipedia" in e.meta else 0)
        for i, e in all_ents.items()
    }
    assert {e.id for e in ents} == set(sorted(score, key=score.get)[:3])
    assert any("wikipedia" in e.meta for e in ents)


def test_update_is_fast_with_many_features(tmp_path):
    elements = [
        {
            "type": "node",
            "id": i,
            "lat": PHONE_LAT + (i % 40 - 20) * 0.0004,
            "lon": PHONE_LON + (i // 40 - 5) * 0.0004,
            "tags": {"name": f"Building {i}", "building": "yes"},
        }
        for i in range(700)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"elements": elements})

    layer = landmarks.LandmarkLayer(
        tmp_path / "c.sqlite", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    try:
        ents = load(layer)
        assert len(ents) == landmarks.MAX_ENTITIES
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            layer.update(ctx(phone()))
            times.append(time.perf_counter() - t0)
    finally:
        layer.close()
    assert sorted(times)[len(times) // 2] < 0.005, sorted(times)[-5:]


def test_backoff_on_429_and_504_then_success(tmp_path):
    server = FixtureServer()
    server.overpass_script = [429, 504, 504]
    layer = make_layer(
        tmp_path,
        server,
        endpoints=("https://a.test/api/interpreter", "https://b.test/api/interpreter"),
    )
    try:
        layer.update(ctx(phone()))
        assert wait_until(lambda: layer.last_error is not None)
        assert "HTTP" in layer.last_error
        assert layer.update(ctx(phone())) == []  # still nothing, still not blocking
        # next attempt only after the backoff, and it succeeds on the mirror order
        assert wait_until(lambda: layer.has_data, 5)
    finally:
        layer.close()
    # attempt 1: 429 + 504 (both mirrors), attempt 2: 504 then 200 on the second mirror
    assert len(server.overpass_calls) == 4
    assert layer.last_error is None


def test_backoff_does_not_hammer(tmp_path):
    server = FixtureServer()
    server.overpass_script = [429] * 100
    layer = make_layer(
        tmp_path, server, backoff_s=30.0, endpoints=("https://a.test/api/interpreter",)
    )
    try:
        for _ in range(10):
            layer.update(ctx(phone()))
            time.sleep(0.05)
        time.sleep(0.5)
    finally:
        layer.close()
    assert len(server.overpass_calls) == 1


def test_offline_without_cache_is_quiet(tmp_path):
    server = FixtureServer()
    server.offline = True
    layer = make_layer(tmp_path, server)
    try:
        assert layer.update(ctx(phone())) == []
        assert wait_until(lambda: layer.last_error is not None)
        assert layer.update(ctx(phone())) == []
    finally:
        layer.close()


def test_restart_offline_uses_disk_cache(tmp_path):
    server = FixtureServer()
    layer = make_layer(tmp_path, server)
    try:
        online = load(layer)
    finally:
        layer.close()

    offline = FixtureServer()
    offline.offline = True
    layer2 = make_layer(tmp_path, offline)
    try:
        # restarted a bit away from where it was cached (still within the cached 2 km area)
        ents = load(layer2, phone(PHONE_LAT + 0.0015, PHONE_LON))
    finally:
        layer2.close()
    assert {e.id for e in ents} == {e.id for e in online}
    assert offline.overpass_calls == []  # fresh cache within 250 m... or no calls reached the net


def test_stale_cache_shown_while_offline_and_retries(tmp_path, monkeypatch):
    server = FixtureServer()
    layer = make_layer(tmp_path, server)
    try:
        load(layer)
    finally:
        layer.close()
    monkeypatch.setattr(landmarks, "CACHE_FRESH_S", -1.0)  # everything counts as stale
    offline = FixtureServer()
    offline.offline = True
    layer2 = make_layer(tmp_path, offline)
    try:
        assert len(load(layer2)) >= 3
        assert wait_until(lambda: layer2.last_error is not None)
        assert len(layer2.update(ctx(phone()))) >= 3  # stale data stays visible
    finally:
        layer2.close()
    assert offline.overpass_calls == []  # unreachable: ConnectError before recording is fine


def test_garbage_responses_do_not_crash(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>rate limited</html>")

    layer = landmarks.LandmarkLayer(
        tmp_path / "c.sqlite",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        backoff_s=0.05,
    )
    try:
        layer.update(ctx(phone()))
        assert wait_until(lambda: layer.last_error is not None)
        assert layer.update(ctx(phone())) == []
    finally:
        layer.close()


def test_close_stops_thread(tmp_path):
    layer = make_layer(tmp_path, FixtureServer())
    load(layer)
    thread = layer._thread
    layer.close()
    assert thread is not None and not thread.is_alive()
    layer.close()  # idempotent


def test_parse_overpass_handles_nodes_ways_and_skips_nameless():
    data = json.loads((Path(_world_ref.__file__).parent / "overpass_response.json").read_text())
    feats = landmarks.parse_overpass(data, PHONE_LAT, PHONE_LON)
    ids = [f.id for f in feats]
    assert "osm:node/900000007" not in ids  # unnamed bus stop
    assert "osm:node/900000004" in ids and "osm:relation/900000005" in ids
    dists = [_world_ref.haversine_m(PHONE_LAT, PHONE_LON, f.lat, f.lon) for f in feats]
    assert dists == sorted(dists)


# ----------------------------------------------------------------------------- Wikipedia


def test_parse_wikipedia_tag():
    assert landmarks.parse_wikipedia_tag("en:Golden Gate Bridge") == ("en", "Golden Gate Bridge")
    assert landmarks.parse_wikipedia_tag("de:Brandenburger Tor") == ("de", "Brandenburger Tor")
    assert landmarks.parse_wikipedia_tag("fr:Tour Eiffel: histoire") == (
        "fr",
        "Tour Eiffel: histoire",
    )
    assert landmarks.parse_wikipedia_tag("Golden Gate Bridge") == ("en", "Golden Gate Bridge")
    assert landmarks.parse_wikipedia_tag("  ") is None
    assert landmarks.summary_url("en", "Fort Point, San Francisco").endswith(
        "/page/summary/Fort_Point,_San_Francisco"
    )


def test_fetch_summary_real_recorded_response_and_cache(tmp_path):
    server = FixtureServer()
    cache = landmarks._Cache(tmp_path / "c.sqlite")
    s = landmarks.fetch_summary("en:Golden Gate Bridge", server.client(), cache=cache)
    assert s is not None
    assert s.title == "Golden Gate Bridge"
    assert "suspension bridge" in s.extract
    assert s.url == "https://en.wikipedia.org/wiki/Golden_Gate_Bridge"
    assert server.wiki_calls == ["/api/rest_v1/page/summary/Golden_Gate_Bridge"]
    assert all(ua.startswith("PointView/") for ua in server.user_agents)

    # memory cache
    assert landmarks.fetch_summary("en:Golden Gate Bridge", server.client(), cache=cache) == s
    assert len(server.wiki_calls) == 1
    # disk cache survives a "restart" even when offline
    landmarks._memory_cache.clear()
    server.offline = True
    assert landmarks.fetch_summary("en:Golden Gate Bridge", server.client(), cache=cache) == s
    cache.close()
    cache2 = landmarks._Cache(tmp_path / "c.sqlite")
    landmarks._memory_cache.clear()
    assert landmarks.fetch_summary("en:Golden Gate Bridge", server.client(), cache=cache2) == s
    cache2.close()


def test_fetch_summary_missing_page_and_failures():
    server = FixtureServer()
    assert landmarks.fetch_summary("en:No Such Page At All", server.client()) is None
    server.offline = True
    assert landmarks.fetch_summary("en:Whatever", server.client(), retries=0) is None
    assert landmarks.fetch_summary("", server.client()) is None


def test_fetch_summary_retries_on_429():
    server = FixtureServer()
    hits = {"n": 0}
    real = server.transport.handle_request

    def flaky(request):
        hits["n"] += 1
        if hits["n"] < 3:
            return httpx.Response(429)
        return real(request)

    c = httpx.Client(transport=httpx.MockTransport(flaky))
    s = landmarks.fetch_summary("en:Golden Gate Bridge", c, backoff_s=0.0)
    assert s is not None and hits["n"] == 3


def test_layer_summary_for_is_nonblocking(tmp_path):
    server = FixtureServer()
    layer = make_layer(tmp_path, server)
    try:
        t0 = time.perf_counter()
        first = layer.summary_for("en:Golden Gate Bridge")
        assert time.perf_counter() - t0 < 0.05
        if first is None:
            assert wait_until(lambda: layer.summary_for("en:Golden Gate Bridge") is not None)
        s = layer.summary_for("en:Golden Gate Bridge")
        assert s is not None and s.title == "Golden Gate Bridge"
        # the module-level default cache is this layer's cache
        server.offline = True
        landmarks._memory_cache.clear()
        assert landmarks.fetch_summary("en:Golden Gate Bridge", server.client()) == s
    finally:
        layer.close()


def test_summary_lines():
    s = landmarks.WikiSummary(
        "Golden Gate Bridge",
        "The Golden Gate Bridge is a suspension bridge spanning the Golden Gate, the one-mile-wide "
        "(1.6 km) strait connecting San Francisco Bay and the Pacific Ocean in California.",
        "https://en.wikipedia.org/wiki/Golden_Gate_Bridge",
    )
    lines = landmarks.summary_lines(s, 40)
    assert len(lines) == 2
    assert all(len(ln) <= 40 for ln in lines)
    assert lines[-1].endswith("…")
    assert "(1.6" not in " ".join(lines)  # parenthetical dropped
    assert lines[0].startswith("The Golden Gate Bridge is a")

    short = landmarks.WikiSummary("X", "A small tower.", "u")
    assert landmarks.summary_lines(short, 40) == ("A small tower.",)
    assert landmarks.summary_lines(short, 40, max_lines=1) == ("A small tower.",)
    assert landmarks.summary_lines(landmarks.WikiSummary("X", "", "u"), 40) == ()
    long_word = landmarks.WikiSummary("X", "Supercalifragilisticexpialidocious word", "u")
    assert all(len(ln) <= 10 for ln in landmarks.summary_lines(long_word, 10))
    one = landmarks.summary_lines(s, 30, max_lines=1)
    assert len(one) == 1 and len(one[0]) <= 30 and one[0].endswith("…")


# --------------------------------------------------------------------------------- network


@pytest.mark.network
def test_live_apis(tmp_path):
    layer = landmarks.LandmarkLayer(tmp_path / "live.sqlite")
    try:
        layer.update(ctx(phone()))
        assert layer.wait_for_data(60), layer.last_error
        ents = layer.update(ctx(phone()))
        assert ents
        s = landmarks.fetch_summary("en:Golden Gate Bridge")
        assert s is not None and "bridge" in s.extract.lower()
    finally:
        layer.close()
