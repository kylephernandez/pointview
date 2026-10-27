"""AircraftLayer: parsing real recordings, dead reckoning, view filtering, focus-only enrichment,
backoff and timing. No live network except the marked test (POINTVIEW_NETWORK_TESTS=1)."""

from __future__ import annotations

import copy
import json
import math
import statistics
import time
from pathlib import Path

import httpx
import pytest

from pointview import world
from pointview.entities import LIVE_KINDS
from pointview.interfaces import Alignment, Calibration, LayerContext, PhoneState, View
from pointview.live import aircraft as ac_mod
from pointview.live.aircraft import AircraftLayer
from tests.live import _world_ref

FIX = Path(__file__).parent.parent / "fixtures" / "live" / "aircraft"
MANIFEST = json.loads((FIX / "manifest.json").read_text())
PHONE_LAT, PHONE_LON = MANIFEST["phone_lat"], MANIFEST["phone_lon"]
POINT = json.loads((FIX / "adsblol_point.json").read_text())
SKY = json.loads((FIX / "opensky_states.json").read_text())
MIL = json.loads((FIX / "adsblol_mil.json").read_text())
NOW = POINT["now"] / 1000.0 + 0.5  # the "wall clock" right after the recorded poll
FOV = (40.0, 25.0)


# ------------------------------------------------------------------------------------ helpers


def phone(lat=PHONE_LAT, lon=PHONE_LON) -> PhoneState:
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


def view_at(yaw: float, pitch: float = 0.0, fov=FOV) -> View:
    """A view whose center is the given world direction (north_offset carries the yaw)."""
    return View(
        head_quat=world.quat_from_yaw_pitch_roll(0.0, pitch, 0.0),
        north_offset_deg=yaw,
        display_fov_deg=fov,
        yaw_deg=yaw % 360.0,
        pitch_deg=pitch,
    )


def ctx(p="default", view=None, focus=None, unix=NOW, enabled=None) -> LayerContext:
    kw = {}
    if enabled is not None:
        kw["enabled"] = enabled
    return LayerContext(
        t=1.0,
        glasses=None,  # type: ignore[arg-type]
        alignment=Alignment(),
        calibration=Calibration(),
        phone=phone() if p == "default" else p,
        unix_time=unix,
        view=view,
        focus_id=focus,
        **kw,
    )


class FakeServer:
    """httpx.MockTransport handler for adsb.lol, OpenSky and the route data. Counts requests."""

    def __init__(self):
        self.sky_headers: dict[str, str] = {}
        self.adsb_body: object = POINT
        self.adsb_status = 200
        self.adsb_headers: dict[str, str] = {}
        self.sky_body: object = SKY
        self.sky_status = 200
        self.requests: list[httpx.Request] = []
        self.routes = {
            p.stem.removeprefix("route_"): p.read_text() for p in FIX.glob("route_*.json")
        }

    def count(self, host_part: str) -> int:
        return sum(1 for r in self.requests if host_part in str(r.url))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if "api.adsb.lol/v2/point" in url:
            if self.adsb_status != 200:
                return httpx.Response(self.adsb_status, headers=self.adsb_headers, text="x")
            return httpx.Response(200, json=self.adsb_body)
        if "opensky-network.org" in url:
            if self.sky_status != 200:
                return httpx.Response(self.sky_status, headers=self.sky_headers, text="x")
            return httpx.Response(200, json=self.sky_body)
        if "vrs-standing-data.adsb.lol/routes/" in url:
            cs = url.rsplit("/", 1)[1].removesuffix(".json")
            if cs in self.routes:
                return httpx.Response(200, text=self.routes[cs])
            return httpx.Response(404, text="not found")
        return httpx.Response(500)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self), follow_redirects=True)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def make_layer(server: FakeServer, tmp_path, clock=None, **kw) -> AircraftLayer:
    return AircraftLayer(
        tmp_path / "ac.sqlite", client=server.client(), clock=clock or Clock(), **kw
    )


def polled(server: FakeServer, tmp_path, clock=None, **kw) -> AircraftLayer:
    """A layer that has done one synchronous poll (no background thread involved)."""
    layer = make_layer(server, tmp_path, clock, **kw)
    layer._pos = (PHONE_LAT, PHONE_LON)
    layer.poll_once()
    return layer


def one_plane(**over) -> dict:
    """A synthetic adsb.lol aircraft 10 km east of the phone, observed at NOW."""
    rec = {
        "hex": "abc123",
        "flight": "TEST1   ",
        "r": "N12345",
        "t": "B738",
        "alt_baro": 10000,
        "alt_geom": 10200,
        "gs": 400.0,
        "track": 90.0,
        "baro_rate": 0,
        "squawk": "1200",
        "lat": PHONE_LAT,
        "lon": PHONE_LON + 10_000 / (111_320 * math.cos(math.radians(PHONE_LAT))),
        "seen": 0.0,
        "seen_pos": 0.0,
    }
    rec.update(over)
    return {"now": NOW * 1000.0, "ac": [rec]}


def aim_at(layer: AircraftLayer, hexid: str, unix=NOW, fov=FOV) -> View:
    """A view centered on the (dead-reckoned) aircraft: found by brute force over all directions."""
    # Wide first pass to read the aircraft's direction back, then center on it.
    for e in everywhere(layer, unix=unix):
        if e.id == f"ac:{hexid}":
            return view_at(e.world_yaw_deg, e.world_pitch_deg, fov)
    raise AssertionError("aircraft not found")


def everywhere(layer: AircraftLayer, unix=NOW) -> list:
    """All aircraft the layer would show to someone who could look in every direction.

    ``in_view`` only accepts the front hemisphere, so union four wide views around the compass."""
    seen: dict = {}
    for yaw in (0.0, 90.0, 180.0, 270.0):
        for e in layer.update(ctx(view=view_at(yaw, 0.0, fov=(170.0, 170.0)), unix=unix)):
            seen[e.id] = e
    return list(seen.values())


# --------------------------------------------------------------------------------------- parsing


def test_parse_adsblol_recording():
    acs = ac_mod.parse_adsblol(POINT)
    assert len(acs) == len(POINT["ac"]) == 95
    by = {a.hex: a for a in acs}
    raw = next(r for r in POINT["ac"] if r["hex"] == "a12aa4")
    a = by["a12aa4"]
    assert a.callsign == "UAL2066" and a.reg == "N17417" and a.type == "B39M"
    assert a.alt_m == pytest.approx(raw["alt_geom"] * 0.3048)  # geometric preferred over baro
    assert a.gs_mps == pytest.approx(raw["gs"] * 0.514444)
    assert a.vrate_mps == pytest.approx(raw["baro_rate"] * 0.00508)
    assert a.track_deg == pytest.approx(raw["track"])
    assert a.t_pos == pytest.approx(POINT["now"] / 1000 - raw["seen_pos"])
    assert not a.ground and not a.military
    assert sum(x.ground for x in acs) == sum(r.get("alt_baro") == "ground" for r in POINT["ac"])
    assert all(x.alt_m == 0.0 for x in acs if x.ground)


def test_parse_adsblol_baro_only_and_missing_position():
    rows = {
        "now": 1_700_000_000_000,
        "ac": [
            {"hex": "AAA111", "lat": 1.0, "lon": 2.0, "alt_baro": 1000, "seen": 3.0},
            {"hex": "bbb222", "alt_baro": 1000},  # no position
            {"hex": "ccc333", "lat": 1.0, "lon": 2.0},  # no altitude
            {"hex": "aaa111", "lat": 9.0, "lon": 9.0, "alt_baro": 5},  # duplicate hex
        ],
    }
    out = ac_mod.parse_adsblol(rows)
    assert [a.hex for a in out] == ["aaa111"]
    assert out[0].alt_m == pytest.approx(304.8)
    assert out[0].t_seen == pytest.approx(1_700_000_000 - 3.0)
    assert out[0].t_pos == pytest.approx(1_700_000_000 - 3.0)  # seen_pos falls back to seen


def test_parse_adsblol_military_flag_only_from_source():
    mil = ac_mod.parse_adsblol(MIL)
    assert mil and all(a.military for a in mil)
    flagged = {r["hex"] for r in POINT["ac"] if int(r.get("dbFlags", 0)) & 1}
    got = {a.hex for a in ac_mod.parse_adsblol(POINT) if a.military}
    assert got == flagged


def test_parse_opensky_recording():
    acs = ac_mod.parse_opensky(SKY)
    assert len(acs) == len([s for s in SKY["states"] if s[5] is not None and s[6] is not None])
    s = next(s for s in SKY["states"] if not s[8] and s[13] is not None and s[5] is not None)
    a = next(x for x in acs if x.hex == s[0])
    assert a.lat == pytest.approx(s[6]) and a.lon == pytest.approx(s[5])
    assert a.alt_m == pytest.approx(s[13])
    assert a.gs_mps == pytest.approx(s[9]) and a.track_deg == pytest.approx(s[10])
    assert a.callsign == s[1].strip() and a.type == "" and not a.military
    ground = next(x for x, st in zip(acs, SKY["states"], strict=False) if st[8])
    assert ground.ground and ground.alt_m == 0.0


def test_sticky_metadata_survives_a_poll_that_misses_fields():
    first = {a.hex: a for a in ac_mod.parse_adsblol(one_plane())}
    second_payload = one_plane(flight="", r="", t="")
    plain = ac_mod.parse_adsblol(second_payload)[0]
    sticky = ac_mod.parse_adsblol(second_payload, first)[0]
    assert (plain.callsign, plain.type) == ("", "")
    assert (sticky.callsign, sticky.reg, sticky.type) == ("TEST1", "N12345", "B738")


# --------------------------------------------------------------------------- entities and fields


def test_entity_fields_from_recording(tmp_path):
    server = FakeServer()
    layer = polled(server, tmp_path)
    try:
        out = everywhere(layer)
    finally:
        layer.close()
    airborne = [r for r in POINT["ac"] if r.get("alt_baro") != "ground"]
    assert len(out) == len(airborne) > 20
    e = next(e for e in out if e.id == "ac:a12aa4")
    assert e.kind == "aircraft" and "aircraft" in LIVE_KINDS
    assert e.label == "UAL2066"
    assert e.meta["type"] == "B39M"
    assert int(e.meta["altitude_ft"]) == pytest.approx(5475 + 2240 * 0.842 / 60, abs=5)
    assert e.meta["speed_kt"] == "288"
    assert e.meta["track"] == "214"
    assert "emergency" not in e.meta and "squawk" not in e.meta and "route" not in e.meta
    assert e.geo is not None and e.geo.alt_m > 1000
    yaw, pitch, rng = _world_ref.look_angles(
        PHONE_LAT, PHONE_LON, 0.0, e.geo.lat, e.geo.lon, e.geo.alt_m
    )
    assert (e.world_yaw_deg, e.world_pitch_deg, e.distance_m) == pytest.approx((yaw, pitch, rng))
    assert 0 <= e.world_yaw_deg < 360 and e.updated_at == 1.0


def test_label_falls_back_to_registration_then_hex(tmp_path):
    server = FakeServer()
    server.adsb_body = {
        "now": NOW * 1000,
        "ac": [
            one_plane(flight="", hex="aaa001")["ac"][0],
            one_plane(flight="", r="", hex="aaa002")["ac"][0],
        ],
    }
    layer = polled(server, tmp_path)
    out = layer.update(ctx(view=view_at(90.0)))
    layer.close()
    assert {e.id: e.label for e in out} == {"ac:aaa001": "N12345", "ac:aaa002": "AAA002"}


def test_emergency_squawk_and_military_meta(tmp_path):
    server = FakeServer()
    a = one_plane(squawk="7700", dbFlags=1)["ac"][0]
    b = one_plane(squawk="1200", hex="abc124")["ac"][0]
    server.adsb_body = {"now": NOW * 1000, "ac": [a, b]}
    layer = polled(server, tmp_path)
    out = {e.id: e for e in layer.update(ctx(view=view_at(90.0)))}
    layer.close()
    assert out["ac:abc123"].meta["emergency"] == "general emergency"
    assert out["ac:abc123"].meta["squawk"] == "7700"
    assert out["ac:abc123"].meta["military"] == "1"
    assert "emergency" not in out["ac:abc124"].meta and "military" not in out["ac:abc124"].meta


def test_ground_aircraft_hidden_by_default_and_shown_on_request(tmp_path):
    server = FakeServer()
    server.adsb_body = one_plane(alt_baro="ground", alt_geom=None, gs=5.0)
    layer = polled(server, tmp_path)
    assert layer.update(ctx(view=view_at(90.0))) == []
    layer.close()
    layer = polled(server, tmp_path, include_ground=True)
    assert len(layer.update(ctx(view=view_at(90.0)))) == 1
    layer.close()


# ----------------------------------------------------------------------------- dead reckoning


def test_dead_reckoning_moves_along_track_and_climbs(tmp_path):
    server = FakeServer()
    server.adsb_body = one_plane(
        gs=400.0, track=90.0, baro_rate=1200, alt_geom=10000, alt_baro=10000
    )
    layer = polled(server, tmp_path)
    try:
        v = aim_at(layer, "abc123")
        e0 = layer.update(ctx(view=v, unix=NOW))[0]
        e5 = layer.update(ctx(view=v, unix=NOW + 5.0))[0]
    finally:
        layer.close()
    moved = world.haversine_m(e0.geo.lat, e0.geo.lon, e5.geo.lat, e5.geo.lon)
    assert moved == pytest.approx(400 * 0.514444 * 5.0, rel=0.01)  # ~1029 m
    assert e5.geo.lon > e0.geo.lon and e5.geo.lat == pytest.approx(e0.geo.lat, abs=1e-5)  # due east
    assert e5.geo.alt_m - e0.geo.alt_m == pytest.approx(1200 * 0.00508 * 5.0, rel=0.01)
    assert e5.distance_m > e0.distance_m  # flying away to the east


def test_dead_reckoning_uses_the_reports_own_age(tmp_path):
    # Position was already 4 s old when the poll arrived (seen_pos=4): at NOW it is 4 s along.
    server = FakeServer()
    server.adsb_body = one_plane(gs=500.0, track=0.0, seen_pos=4.0, seen=4.0)
    layer = polled(server, tmp_path)
    try:
        raw_lat = one_plane()["ac"][0]["lat"]
        e = layer.update(ctx(view=view_at(90.0, fov=(120, 60)), unix=NOW))[0]
    finally:
        layer.close()
    north_m = (e.geo.lat - raw_lat) * 111_200
    assert north_m == pytest.approx(500 * 0.514444 * 4.0, rel=0.02)


def test_dead_reckoning_never_goes_underground_and_caps_extrapolation(tmp_path):
    server = FakeServer()
    server.adsb_body = one_plane(baro_rate=-6000, alt_geom=300, alt_baro=300)
    layer = polled(server, tmp_path)
    try:
        out = layer.update(ctx(view=view_at(90.0, fov=(120, 60)), unix=NOW + 20))
    finally:
        layer.close()
    assert out and out[0].geo.alt_m == 0.0
    # 50 s after the report the position is capped at 30 s of travel (and still shown, <60 s old)
    server.adsb_body = one_plane(gs=400.0, track=90.0, baro_rate=0)
    layer = polled(server, tmp_path)
    try:
        e50 = layer.update(ctx(view=view_at(90.0, fov=(120, 60)), unix=NOW + 50))[0]
        e30 = layer.update(ctx(view=view_at(90.0, fov=(120, 60)), unix=NOW + 30))[0]
    finally:
        layer.close()
    assert e50.geo.lon == pytest.approx(e30.geo.lon)


def test_aircraft_unseen_for_over_60s_are_dropped(tmp_path):
    server = FakeServer()
    server.adsb_body = one_plane(seen=10.0, seen_pos=10.0)
    layer = polled(server, tmp_path)
    try:
        v = view_at(90.0, fov=(120, 60))
        assert len(layer.update(ctx(view=v, unix=NOW + 49))) == 1  # last heard 59 s ago
        assert layer.update(ctx(view=v, unix=NOW + 51)) == []  # 61 s ago
    finally:
        layer.close()


# ------------------------------------------------------------------------------ view filtering


def test_no_view_no_fix_or_disabled_returns_empty(tmp_path):
    server = FakeServer()
    layer = polled(server, tmp_path)
    try:
        assert layer.update(ctx(view=None)) == []
        assert layer.update(ctx(p=None, view=view_at(0))) == []
        assert layer.update(ctx(p=phone(lat=None, lon=None), view=view_at(0))) == []
        assert layer.update(ctx(view=view_at(0), enabled=frozenset({"landmark"}))) == []
    finally:
        layer.close()


def test_only_aircraft_in_view_are_returned(tmp_path):
    server = FakeServer()
    layer = polled(server, tmp_path)
    try:
        everything = everywhere(layer)
        target = next(e for e in everything if e.world_pitch_deg > 5)
        v = view_at(target.world_yaw_deg, target.world_pitch_deg)
        seen = layer.update(ctx(view=v))
        away = layer.update(ctx(view=view_at(target.world_yaw_deg + 180, target.world_pitch_deg)))
    finally:
        layer.close()
    assert target.id in {e.id for e in seen}
    assert target.id not in {e.id for e in away}
    assert 0 < len(seen) < len(everything)
    for e in seen:  # every returned entity is inside the 5-degree-widened view
        assert world.in_view(
            e.world_yaw_deg, e.world_pitch_deg, v.head_quat, v.north_offset_deg, FOV, margin_deg=5
        )


def test_prefilter_matches_brute_force_over_many_views(tmp_path):
    """The cheap cone test must never drop something world.in_view would accept."""
    server = FakeServer()
    layer = polled(server, tmp_path)
    try:
        allr = {e.id: e for e in everywhere(layer)}
        for yaw in range(0, 360, 37):
            for pitch in (-20, 0, 15, 40, 75):
                for fov in ((40.0, 25.0), (90.0, 60.0)):
                    v = view_at(float(yaw), float(pitch), fov)
                    got = {e.id for e in layer.update(ctx(view=v))}
                    want = {
                        i
                        for i, e in allr.items()
                        if world.in_view(
                            e.world_yaw_deg,
                            e.world_pitch_deg,
                            v.head_quat,
                            v.north_offset_deg,
                            fov,
                            margin_deg=5,
                        )
                    }
                    assert got == want, (yaw, pitch, fov)
    finally:
        layer.close()


# ------------------------------------------------------------------------- focus enrichment


def _wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_route_is_fetched_only_for_the_focused_aircraft(tmp_path):
    server = FakeServer()
    layer = polled(server, tmp_path)
    try:
        callsign = MANIFEST["route_callsigns"][0]
        focus = next(
            f"ac:{r['hex']}" for r in POINT["ac"] if r.get("flight", "").strip() == callsign
        )
        wide = aim_at(layer, focus.removeprefix("ac:"), fov=(170.0, 170.0))
        for _ in range(5):
            out = layer.update(ctx(view=wide))
        assert len(out) > 20
        assert server.count("vrs-standing-data") == 0  # nothing focused: no enrichment at all
        assert all("route" not in e.meta for e in out)

        first = {e.id: e for e in layer.update(ctx(view=wide, focus=focus))}
        assert "route" not in first[focus].meta  # not blocked on the network: arrives later
        assert _wait(lambda: server.count("vrs-standing-data") == 1)
        assert _wait(
            lambda: (
                "route" in {k for e in layer.update(ctx(view=wide, focus=focus)) for k in e.meta}
            )
        )
        out = {e.id: e for e in layer.update(ctx(view=wide, focus=focus))}
        route = json.loads((FIX / f"route_{callsign}.json").read_text())["_airport_codes_iata"]
        assert out[focus].meta["route"] == route.replace("-", " → ")
        assert [k for k, e in out.items() if "route" in e.meta] == [focus]
        for _ in range(10):  # cached per callsign: no more requests
            layer.update(ctx(view=wide, focus=focus))
        assert server.count("vrs-standing-data") == 1
        urls = [str(r.url) for r in server.requests if "vrs-standing-data" in str(r.url)]
        assert urls == [f"https://vrs-standing-data.adsb.lol/routes/{callsign[:2]}/{callsign}.json"]
        # focus moves to another aircraft: one more request, for that one only
        other = MANIFEST["route_callsigns"][1]
        focus2 = next(f"ac:{r['hex']}" for r in POINT["ac"] if r.get("flight", "").strip() == other)
        layer.update(ctx(view=wide, focus=focus2))
        assert _wait(lambda: server.count("vrs-standing-data") == 2)
    finally:
        layer.close()


def test_route_cache_survives_restart_and_unknown_callsigns_are_negative_cached(tmp_path):
    server = FakeServer()
    layer = polled(server, tmp_path)
    cs = MANIFEST["route_callsigns"][0]
    assert layer.resolve_route(cs)
    assert layer.resolve_route("ZZZ999") is None
    assert server.count("vrs-standing-data") == 2
    layer.close()

    server2 = FakeServer()
    layer2 = make_layer(server2, tmp_path)
    try:
        assert layer2.resolve_route(cs) is not None
        assert layer2.resolve_route("ZZZ999") is None
        assert server2.count("vrs-standing-data") == 0
    finally:
        layer2.close()


def test_route_not_requested_for_junk_callsigns(tmp_path):
    server = FakeServer()
    server.adsb_body = one_plane(flight="../../x ")
    layer = polled(server, tmp_path)
    try:
        layer.update(ctx(view=view_at(90.0), focus="ac:abc123"))
        time.sleep(0.1)
        assert server.count("vrs-standing-data") == 0
    finally:
        layer.close()


# ---------------------------------------------------------------------------- sources and backoff


def test_adsblol_request_shape(tmp_path):
    server = FakeServer()
    layer = polled(server, tmp_path, radius_km=40)
    layer.close()
    (req,) = server.requests
    assert str(req.url) == f"https://api.adsb.lol/v2/point/{PHONE_LAT:.4f}/{PHONE_LON:.4f}/22"
    assert req.headers["user-agent"].startswith("PointView/")


def test_falls_back_to_opensky_with_bbox_and_recovers(tmp_path):
    clock = Clock()
    server = FakeServer()
    server.adsb_status = 503
    layer = polled(server, tmp_path, clock)
    try:
        assert layer.source == "opensky" and layer.last_error is None
        sky = next(r for r in server.requests if "opensky" in str(r.url))
        q = dict(sky.url.params)
        assert float(q["lamin"]) < PHONE_LAT < float(q["lamax"])
        assert float(q["lomin"]) < PHONE_LON < float(q["lomax"])
        assert sky.headers["user-agent"].startswith("PointView/")
        assert (
            len(layer.update(ctx(view=view_at(0, 0, fov=(360.0, 179.0)), unix=SKY["time"] + 1))) > 5
        )
        assert layer.attribution == ac_mod.OPENSKY_ATTRIBUTION
        # adsb.lol comes back: the next due poll goes back to it
        server.adsb_status = 200
        clock.t += 100
        layer.poll_once()
        assert layer.source == "adsb.lol" and layer.attribution == ac_mod.ADSBLOL_ATTRIBUTION
    finally:
        layer.close()


def test_backoff_on_429_honours_retry_after_and_doubles(tmp_path):
    clock = Clock()
    server = FakeServer()
    server.adsb_status = 429
    server.adsb_headers = {"Retry-After": "40"}
    server.sky_status = 429
    server.sky_headers = {"X-Rate-Limit-Retry-After-Seconds": "40"}
    layer = make_layer(server, tmp_path, clock)
    layer._pos = (PHONE_LAT, PHONE_LON)
    try:
        assert layer.poll_once() is True
        assert (server.count("adsb.lol"), server.count("opensky")) == (1, 1)
        assert layer.last_error and "429" in layer.last_error
        assert not layer.has_data
        for dt in (1, 8, 30):  # still inside Retry-After (40 s): nothing is sent
            clock.t += dt
            layer.poll_once()
        assert (server.count("adsb.lol"), server.count("opensky")) == (1, 1)
        clock.t += 10  # 49 s since the failure: both sources are due again
        layer.poll_once()
        assert (server.count("adsb.lol"), server.count("opensky")) == (2, 2)
        # second consecutive failure: the delay doubles (8 s base would give 16 s; Retry-After 40)
        server.adsb_headers = {}
        before = server.count("adsb.lol")
        clock.t += 20
        layer.poll_once()
        assert server.count("adsb.lol") == before  # failures=2 -> 16 s, but retry-after set 40 s
        clock.t += 25
        layer.poll_once()
        assert server.count("adsb.lol") == before + 1
        assert layer._primary.failures == 3
        # success resets everything
        server.adsb_status = 200
        clock.t += 100
        layer.poll_once()
        assert layer._primary.failures == 0 and layer.has_data and layer.last_error is None
        assert layer.source == "adsb.lol"
    finally:
        layer.close()


def test_backoff_without_retry_after_grows_exponentially_and_is_capped(tmp_path):
    clock = Clock()
    server = FakeServer()
    server.adsb_status = 500
    server.sky_status = 500
    layer = make_layer(server, tmp_path, clock, poll_s=8.0)
    layer._pos = (PHONE_LAT, PHONE_LON)
    try:
        delays = []
        for _ in range(8):
            layer.poll_once(force=True)
            delays.append(layer._primary.next_at - clock.t)
        assert delays[:4] == [8.0, 16.0, 32.0, 64.0]
        assert delays[-1] == ac_mod.MAX_BACKOFF_S
    finally:
        layer.close()


def test_bad_json_is_a_failure_not_a_crash(tmp_path):
    server = FakeServer()
    server.adsb_body = ["not", "an", "object"]
    server.sky_body = {"states": "nope"}
    layer = make_layer(server, tmp_path, Clock())
    layer._pos = (PHONE_LAT, PHONE_LON)
    try:
        layer.poll_once()
        assert not layer.has_data and layer.last_error
        assert layer.update(ctx(view=view_at(0))) == []
    finally:
        layer.close()


# ------------------------------------------------------------------------------------ cache


def test_restart_shows_cached_traffic_without_network(tmp_path):
    server = FakeServer()
    layer = polled(server, tmp_path)
    layer.close()

    def refuse(request):
        raise httpx.ConnectError("offline")

    layer2 = AircraftLayer(
        tmp_path / "ac.sqlite", client=httpx.Client(transport=httpx.MockTransport(refuse))
    )
    try:
        wide = view_at(0, 0, fov=(360.0, 179.0))
        # first update starts the thread, which loads the cache before touching the network
        layer2.update(ctx(view=wide))
        assert layer2.wait_for_data(3.0)
        out = layer2.update(ctx(view=wide))
        assert len(out) > 20
        assert layer2.source == "adsb.lol"
    finally:
        layer2.close()


def test_cache_for_a_far_away_place_is_ignored(tmp_path):
    server = FakeServer()
    polled(server, tmp_path).close()
    layer = AircraftLayer(tmp_path / "ac.sqlite", client=FakeServer().client(), clock=Clock())
    layer._pos = (PHONE_LAT + 5.0, PHONE_LON)
    layer._load_cache(layer._pos)
    assert not layer.has_data
    layer.close()


def test_cache_disabled_without_path_still_works():
    server = FakeServer()
    layer = AircraftLayer(None, client=server.client(), clock=Clock())
    layer._pos = (PHONE_LAT, PHONE_LON)
    layer.poll_once()
    assert layer.has_data
    layer.close()


# ----------------------------------------------------------------------------- thread and timing


def test_background_thread_polls_and_update_returns_entities(tmp_path):
    server = FakeServer()
    layer = AircraftLayer(tmp_path / "ac.sqlite", client=server.client(), poll_s=0.2)
    try:
        wide = view_at(0, 0, fov=(360.0, 179.0))
        layer.update(ctx(view=wide))
        assert layer.wait_for_data(3.0)
        assert _wait(lambda: len(layer.update(ctx(view=wide))) > 20)
        # keeps polling on its own
        n = server.count("api.adsb.lol/v2/point")
        assert _wait(lambda: server.count("api.adsb.lol/v2/point") > n, timeout=3.0)
    finally:
        layer.close()


def _crowd(n: int) -> dict:
    import random

    rnd = random.Random(7)
    rows = []
    for i in range(n):
        rng_m = rnd.uniform(3_000, 40_000)
        brg = rnd.uniform(0, 2 * math.pi)
        rows.append(
            {
                "hex": f"{0xA00000 + i:06x}",
                "flight": f"TST{i:04d}",
                "t": "A320",
                "alt_geom": rnd.uniform(2_000, 40_000),
                "alt_baro": 1000,
                "gs": rnd.uniform(100, 500),
                "track": rnd.uniform(0, 360),
                "baro_rate": rnd.uniform(-2000, 2000),
                "squawk": "1200",
                "lat": PHONE_LAT + rng_m * math.cos(brg) / 111_200,
                "lon": PHONE_LON
                + rng_m * math.sin(brg) / (111_200 * math.cos(math.radians(PHONE_LAT))),
                "seen": 0.5,
                "seen_pos": 0.5,
            }
        )
    return {"now": NOW * 1000, "ac": rows}


@pytest.mark.parametrize("fov", [(40.0, 25.0), (360.0, 179.0)])
def test_update_stays_under_5ms_with_300_aircraft(tmp_path, fov):
    """fov (360, 179) is the worst case: every aircraft passes the cone and reaches look_angles."""
    server = FakeServer()
    server.adsb_body = _crowd(300)
    layer = polled(server, tmp_path)
    try:
        v = view_at(45.0, 10.0, fov=fov)
        c = ctx(view=v, focus="ac:a00005")
        layer.update(c)
        times = []
        for k in range(30):
            c = ctx(view=v, unix=NOW + 0.016 * k, focus="ac:a00005")
            t0 = time.perf_counter()
            out = layer.update(c)
            times.append((time.perf_counter() - t0) * 1000)
        med = statistics.median(times)
        print(f"update(): {len(out)} entities, median {med:.2f} ms, max {max(times):.2f} ms")
        assert med < 5.0
        if fov[0] == 360.0:
            assert len(out) > 120  # the whole front hemisphere: each one went through look_angles
    finally:
        layer.close()


def test_update_does_no_io(tmp_path):
    server = FakeServer()
    layer = polled(server, tmp_path)
    try:
        n = len(server.requests)
        for _ in range(20):
            layer.update(ctx(view=view_at(0, 0, fov=(360.0, 179.0))))
        assert len(server.requests) == n
    finally:
        layer.close()


def test_live_entities_do_not_alias_snapshot_state(tmp_path):
    server = FakeServer()
    layer = polled(server, tmp_path)
    try:
        a = layer.update(ctx(view=view_at(0, 0, fov=(360.0, 179.0))))
        snapshot_before = copy.deepcopy([x.lat for x in layer._snap.ac])
        for e in a:
            e.meta["x"] = "y"  # type: ignore[index]
        b = layer.update(ctx(view=view_at(0, 0, fov=(360.0, 179.0))))
        assert all("x" not in e.meta for e in b)
        assert snapshot_before == [x.lat for x in layer._snap.ac]
    finally:
        layer.close()


@pytest.mark.network
def test_live_apis_smoke(tmp_path):
    layer = AircraftLayer(tmp_path / "ac.sqlite", poll_s=1.0)
    try:
        layer._pos = (PHONE_LAT, PHONE_LON)
        layer.poll_once(force=True)
        assert layer.has_data
    finally:
        layer.close()
