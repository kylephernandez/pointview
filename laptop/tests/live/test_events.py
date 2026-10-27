from __future__ import annotations

import json
import logging
import queue
import statistics
import time
from pathlib import Path

import httpx
import pytest

from pointview import world
from pointview.interfaces import (
    Alignment,
    Calibration,
    Layer,
    LayerContext,
    PhoneState,
    View,
)
from pointview.live import events
from pointview.live.events import EventLayer, VesselLayer, dead_reckon, parse_usgs_feed

from . import _world_ref_events as ref

FIX = Path(__file__).parent.parent / "fixtures" / "live" / "events"
FEED_BODY = (FIX / "summary_2.5_day.geojson").read_text()
DETAIL_BODY = (FIX / "detail_us7000ttci.geojson").read_text()
AIS = json.loads((FIX / "ais_messages.json").read_text())

#: Feed generation time: "now" for the quake tests.
NOW = 1793064128.0
ADELAIDE = (-34.93, 138.60)
QUAKE_ID = "us7000ttci"  # M4.7, ~170 km south of Adelaide, 27 felt reports
SF = (37.8060, -122.4680)


@pytest.fixture(autouse=True)
def _no_ais_key(monkeypatch):
    monkeypatch.delenv("AISSTREAM_API_KEY", raising=False)


def phone(pos=ADELAIDE) -> PhoneState | None:
    if pos is None:
        return None
    return PhoneState(
        t_ms=0.0,
        t=0.0,
        received_at=0.0,
        quat=(1.0, 0.0, 0.0, 0.0),
        omega=(0.0, 0.0, 0.0),
        heading_deg=None,
        heading_acc=None,
        lat=pos[0],
        lon=pos[1],
        gps_acc=5.0,
    )


def make_view(yaw: float, pitch: float = 0.0) -> View:
    q = world.quat_from_yaw_pitch_roll(yaw, pitch)
    return View(
        head_quat=q,
        north_offset_deg=0.0,
        display_fov_deg=(46.0, 26.0),
        yaw_deg=yaw,
        pitch_deg=pitch,
    )


def ctx(pos=ADELAIDE, view="south", focus=None, unix=NOW, enabled=None) -> LayerContext:
    if view == "south":
        view = make_view(184.0)
    kw = {} if enabled is None else {"enabled": enabled}
    return LayerContext(
        t=1.0,
        glasses=None,  # type: ignore[arg-type]
        alignment=Alignment(),
        calibration=Calibration(),
        phone=phone(pos),
        unix_time=unix,
        view=view,
        focus_id=focus,
        **kw,
    )


def feed_client(counts: dict | None = None, *, detail_status: int = 200) -> httpx.Client:
    counts = counts if counts is not None else {}

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if "summary" in path:
            counts["feed"] = counts.get("feed", 0) + 1
            return httpx.Response(200, text=FEED_BODY)
        counts["detail"] = counts.get("detail", 0) + 1
        counts.setdefault("detail_ids", []).append(path.rsplit("/", 1)[-1])
        return httpx.Response(detail_status, text=DETAIL_BODY)

    return httpx.Client(transport=httpx.MockTransport(handler))


def make_events(tmp_path, counts=None, **kw) -> EventLayer:
    layer = EventLayer(tmp_path / "c.sqlite", client=feed_client(counts), **kw)
    return layer


def loaded(layer: EventLayer, c: LayerContext) -> list:
    layer.update(c)  # starts the thread
    assert layer.wait_for_data(5)
    return layer.update(c)


# ------------------------------------------------------------------------------------- quakes


def test_parse_real_usgs_recording():
    quakes = parse_usgs_feed(FEED_BODY)
    assert len(quakes) == 32
    q = next(q for q in quakes if q.id == QUAKE_ID)
    assert q.mag == pytest.approx(4.7)
    assert q.place == "98 km S of Encounter Bay, Australia"
    assert q.lat == pytest.approx(-36.4594)
    assert q.lon == pytest.approx(138.5321)
    assert q.depth_km == pytest.approx(10.0)
    assert q.time_s > 1.79e9


def test_parse_skips_garbage():
    assert parse_usgs_feed("not json") == []
    assert (
        parse_usgs_feed({"features": [{"id": "x"}, {"id": "y", "properties": {"mag": None}}]}) == []
    )


def test_layer_protocol_and_entity_shape(tmp_path):
    layer = make_events(tmp_path)
    try:
        assert isinstance(layer, Layer)
        out = loaded(layer, ctx())
        assert [e.id for e in out] == [f"quake:{QUAKE_ID}"]
        e = out[0]
        assert e.kind == "event"
        assert e.label.startswith("M4.7 quake · ")
        assert e.label.endswith(" h ago")
        assert e.geo is not None
        assert e.geo.lat == pytest.approx(-36.4594)
        assert e.geo.alt_m == pytest.approx(-10_000.0)
        assert e.meta["mag"] == "4.7"
        assert e.meta["place"].startswith("98 km S of Encounter Bay")
        assert e.meta["depth_km"] == "10.0"
        assert e.meta["time"].endswith("Z")
        assert "felt_reports" not in e.meta  # not focused
    finally:
        layer.close()


def test_horizon_clamp_keeps_true_range(tmp_path):
    layer = make_events(tmp_path)
    try:
        e = loaded(layer, ctx())[0]
        true_yaw, true_pitch, true_rng = world.look_angles(
            *ADELAIDE, 0.0, -36.4594, 138.5321, -10000.0
        )
        # The spherical reference agrees with the ellipsoid to well under 0.1°.
        ref_yaw, ref_pitch, _ = ref.look_angles(*ADELAIDE, 0.0, -36.4594, 138.5321, -10000.0)
        assert abs(ref_yaw - true_yaw) < 0.1 and abs(ref_pitch - true_pitch) < 0.1
        assert true_pitch < -1.0  # below the horizon: curvature plus depth
        assert e.world_pitch_deg == 0.0
        assert e.world_yaw_deg == pytest.approx(true_yaw)
        assert e.distance_m == pytest.approx(true_rng)
        assert 150_000 < e.distance_m < 200_000
    finally:
        layer.close()


def test_radius_filter(tmp_path):
    wide = make_events(tmp_path / "a", radius_km=500)
    narrow = make_events(tmp_path / "b", radius_km=100)
    try:
        assert len(loaded(wide, ctx())) == 1
        assert loaded(narrow, ctx()) == []
    finally:
        wide.close()
        narrow.close()


def test_min_magnitude_filter(tmp_path):
    layer = make_events(tmp_path, min_mag=5.0)
    try:
        assert loaded(layer, ctx()) == []
    finally:
        layer.close()


def test_other_regions_are_radius_filtered(tmp_path):
    """Tokyo sees the Tomioka quake only; Adelaide's quake is far outside."""
    layer = make_events(tmp_path)
    try:
        tokyo = (35.68, 139.76)
        c = ctx(tokyo, view=make_view(20.0))
        ids = [e.id for e in loaded(layer, c)]
        assert ids == ["quake:us7000ttdt"]
    finally:
        layer.close()


def test_in_view_filtering(tmp_path):
    layer = make_events(tmp_path)
    try:
        assert len(loaded(layer, ctx(view=make_view(184.0)))) == 1
        assert layer.update(ctx(view=make_view(0.0))) == []  # looking away
        assert layer.update(ctx(view=make_view(184.0 + 40.0))) == []  # outside FOV + margin
        # inside half-FOV (23) + margin (5) = 28 deg
        assert len(layer.update(ctx(view=make_view(184.0 + 26.0)))) == 1
    finally:
        layer.close()


def test_no_view_no_fix_or_disabled_gives_empty(tmp_path):
    layer = make_events(tmp_path)
    try:
        loaded(layer, ctx())
        assert layer.update(ctx(view=None)) == []
        assert layer.update(ctx(pos=None)) == []
        assert layer.update(ctx(enabled=frozenset({"landmark"}))) == []
    finally:
        layer.close()


def test_phone_without_lat_lon_gives_empty(tmp_path):
    layer = make_events(tmp_path)
    try:
        loaded(layer, ctx())
        p = phone()
        object.__setattr__(p, "lat", None)
        c = ctx()
        c = LayerContext(
            t=c.t, glasses=c.glasses, alignment=c.alignment, calibration=c.calibration,
            phone=p, unix_time=NOW, view=c.view,
        )  # fmt: skip
        assert layer.update(c) == []
    finally:
        layer.close()


def test_detail_fetched_only_for_focused_quake(tmp_path):
    counts: dict = {}
    layer = make_events(tmp_path, counts)
    try:
        out = loaded(layer, ctx())
        for _ in range(5):
            layer.update(ctx())
        time.sleep(0.2)
        assert counts.get("detail", 0) == 0  # nothing focused: no detail request
        assert "felt_reports" not in out[0].meta

        focus = f"quake:{QUAKE_ID}"
        layer.update(ctx(focus=focus))
        assert layer.wait_for_detail(QUAKE_ID)
        for _ in range(5):
            e = layer.update(ctx(focus=focus))[0]
        assert counts["detail"] == 1  # cached after the first fetch
        assert counts["detail_ids"] == [f"{QUAKE_ID}.geojson"]
        assert e.meta["felt_reports"] == "27"
        assert e.meta["tsunami"] == "no"
        assert e.meta["felt_intensity_cdi"] == "3.8"

        # focus moves away: enrichment is not shown for unfocused entities
        assert "felt_reports" not in layer.update(ctx(focus="quake:other"))[0].meta
    finally:
        layer.close()


def test_detail_and_feed_survive_restart_offline(tmp_path):
    counts: dict = {}
    layer = make_events(tmp_path, counts)
    loaded(layer, ctx())
    layer.update(ctx(focus=f"quake:{QUAKE_ID}"))
    assert layer.wait_for_detail(QUAKE_ID)
    layer.close()

    def offline(req):
        raise httpx.ConnectError("offline")

    layer2 = EventLayer(
        tmp_path / "c.sqlite", client=httpx.Client(transport=httpx.MockTransport(offline))
    )
    try:
        out = loaded(layer2, ctx(focus=f"quake:{QUAKE_ID}"))  # from the SQLite cache
        assert len(out) == 1
        assert layer2.wait_for_detail(QUAKE_ID)
        assert layer2.update(ctx(focus=f"quake:{QUAKE_ID}"))[0].meta["felt_reports"] == "27"
    finally:
        layer2.close()


def test_feed_polled_on_interval(tmp_path):
    counts: dict = {}
    layer = make_events(tmp_path, counts, poll_interval_s=0.05)
    try:
        loaded(layer, ctx())
        deadline = time.monotonic() + 3
        while counts.get("feed", 0) < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert counts["feed"] >= 3
    finally:
        layer.close()
    assert events.POLL_INTERVAL_S == 300.0


def test_feed_error_is_tolerated(tmp_path):
    def boom(req):
        return httpx.Response(503, text="down")

    layer = EventLayer(None, client=httpx.Client(transport=httpx.MockTransport(boom)))
    try:
        layer.update(ctx())
        time.sleep(0.2)
        assert layer.update(ctx()) == []
        assert layer.last_error and "feed" in layer.last_error
    finally:
        layer.close()


def test_label_ages():
    assert events._ago(10) == "just now"
    assert events._ago(600) == "10 min ago"
    assert events._ago(2 * 3600) == "2 h ago"
    assert events._ago(3 * 86400) == "3 d ago"


def test_events_update_is_fast(tmp_path):
    layer = make_events(tmp_path, radius_km=20000)  # every quake passes the radius filter
    try:
        c = ctx(view=make_view(184.0))
        loaded(layer, c)
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            layer.update(c)
            times.append(time.perf_counter() - t0)
        assert statistics.median(times) < 0.005
        assert max(times) < 0.05
    finally:
        layer.close()


# ------------------------------------------------------------------------------------- vessels

T0 = 1793062800.0  # 2026-10-27 01:00:00 UTC, matches the fixtures' time_utc


def test_no_key_is_inactive_and_logs_once(caplog):
    with caplog.at_level(logging.INFO, logger="pointview.live.events"):
        layer = VesselLayer(None)
    assert not layer.active
    msgs = [r for r in caplog.records if "inactive" in r.getMessage()]
    assert len(msgs) == 1
    assert layer.update(ctx(SF, unix=T0)) == []
    assert layer._thread is None  # never opened a socket
    layer.close()
    assert not VesselLayer("   ").active


def test_key_from_environment(monkeypatch):
    monkeypatch.setenv("AISSTREAM_API_KEY", "abc")
    assert VesselLayer().active


def test_parse_position_report():
    r = events.parse_ais_message(AIS["position_report"], 0.0)
    assert isinstance(r, events._Report)
    assert (r.mmsi, r.lat, r.lon) == (366999001, 37.85, -122.468)
    assert r.name == "TEST VESSEL ONE"  # '@' padding stripped
    assert r.sog_kn == 10.0 and r.cog_deg == 90.0 and r.heading_deg == 91
    assert r.t == pytest.approx(T0 + 0.318353)


def test_parse_sentinels_and_invalid():
    r = events.parse_ais_message(json.dumps(AIS["class_b_report"]), 123.0)
    assert isinstance(r, events._Report)
    assert r.sog_kn is None and r.cog_deg is None and r.heading_deg is None
    assert r.name is None
    assert events.parse_ais_message(AIS["invalid_position"], 0.0) is None
    assert events.parse_ais_message(AIS["error_envelope"], 0.0) is None
    assert events.parse_ais_message("garbage", 0.0) is None
    assert events.parse_ais_message({"MessageType": "AidsToNavigationReport"}, 0.0) is None


def test_parse_static_data():
    mmsi, st = events.parse_ais_message(AIS["static_data"], 0.0)  # type: ignore[misc]
    assert mmsi == 366999001
    assert st.destination == "OAKLAND"
    assert st.eta == "10-28 14:30 UTC"
    assert st.ship_type == 70 and events.ship_type_name(70) == "cargo"
    assert st.imo == 9999991 and st.call_sign == "WDX1234"


def test_ship_type_names():
    assert events.ship_type_name(None) is None
    assert events.ship_type_name(0) is None
    assert events.ship_type_name(52) == "tug"
    assert events.ship_type_name(84) == "tanker"
    assert events.ship_type_name(61) == "passenger"


def test_dead_reckoning_math():
    # 10 knots east for 600 s = 3.087 km
    lat, lon = dead_reckon(37.85, -122.468, 10.0, 90.0, 600.0)
    assert world.haversine_m(37.85, -122.468, lat, lon) == pytest.approx(3086.7, rel=0.01)
    assert lat == pytest.approx(37.85, abs=1e-3)
    assert lon > -122.468
    lat, lon = dead_reckon(37.85, -122.468, 10.0, 0.0, 600.0)  # north
    assert lat > 37.85 and lon == pytest.approx(-122.468, abs=1e-6)
    assert dead_reckon(37.85, -122.468, None, 90.0, 600.0) == (37.85, -122.468)
    assert dead_reckon(37.85, -122.468, 10.0, None, 600.0) == (37.85, -122.468)
    assert dead_reckon(37.85, -122.468, 10.0, 90.0, 0.0) == (37.85, -122.468)


def vessel_layer(**kw) -> VesselLayer:
    return VesselLayer("test-key", **kw)


def test_vessel_entity_and_dead_reckoned_position():
    layer = vessel_layer()
    try:
        layer.ingest(AIS["position_report"], T0)
        view = make_view(0.0)  # ship is ~5 km north
        e0 = layer.update(ctx(SF, view=view, unix=T0 + 0.318353))[0]
        assert e0.id == "ais:366999001"
        assert e0.kind == "vessel"
        assert e0.label == "TEST VESSEL ONE"
        assert abs(world.wrap_deg_signed(e0.world_yaw_deg)) < 0.5
        assert 4500 < e0.distance_m < 5500
        assert e0.geo.alt_m == 0.0
        assert e0.meta["mmsi"] == "366999001" and e0.meta["sog_kn"] == "10.0"
        # 5 minutes later it has moved ~1.54 km east
        e1 = layer.update(ctx(SF, view=view, unix=T0 + 300.318353))[0]
        assert e1.geo.lon > e0.geo.lon
        moved = world.haversine_m(e0.geo.lat, e0.geo.lon, e1.geo.lat, e1.geo.lon)
        assert moved == pytest.approx(1543.3, rel=0.01)
        swing = world.wrap_deg_signed(e1.world_yaw_deg - e0.world_yaw_deg)
        assert swing > 10  # swung toward the east
    finally:
        layer.close()


def test_vessel_without_name_uses_mmsi_and_type_from_static():
    layer = vessel_layer()
    try:
        layer.ingest(AIS["class_b_report"], T0)
        layer.ingest(
            {
                "MessageType": "PositionReport",
                "MetaData": {"MMSI": 366999005, "ShipName": "X", "time_utc": ""},
                "Message": {
                    "PositionReport": {
                        "Latitude": 37.85, "Longitude": -122.468, "Sog": 0.0, "Cog": 0.0, "UserID": 366999005,
                    }
                },
            },
            T0,
        )  # fmt: skip
        layer.ingest(AIS["static_data"], T0)
        layer.ingest(AIS["position_report"], T0)
        by_id = {e.id: e for e in layer.update(ctx(SF, view=make_view(70.0), unix=T0 + 1))}
        assert "ais:366999002" in by_id  # 7 km east-ish of SF at Class B position
        assert by_id["ais:366999002"].label == "MMSI 366999002"
        e = layer.update(ctx(SF, view=make_view(0.0), unix=T0 + 1))
        assert "TEST VESSEL ONE · cargo" in {x.label for x in e}
    finally:
        layer.close()


def test_vessel_in_view_and_radius_filtering():
    layer = vessel_layer(radius_km=30)
    try:
        layer.ingest(AIS["position_report"], T0)  # north, ~5 km
        assert len(layer.update(ctx(SF, view=make_view(0.0), unix=T0 + 1))) == 1
        assert layer.update(ctx(SF, view=make_view(180.0), unix=T0 + 1)) == []
        assert layer.update(ctx(SF, view=None, unix=T0 + 1)) == []
        assert layer.update(ctx(None, unix=T0 + 1)) == []
        far = (38.5, -122.468)  # ship now ~72 km south of the observer
        assert layer.update(ctx(far, view=make_view(180.0), unix=T0 + 1)) == []
    finally:
        layer.close()


def test_vessel_expires_after_ten_minutes():
    layer = vessel_layer()
    try:
        layer.ingest(AIS["class_b_report"], T0)  # stationary (sentinels), SF-ish bay
        v = make_view(70.0)
        assert layer.update(ctx(SF, view=v, unix=T0 + 600)) != []
        assert layer.update(ctx(SF, view=v, unix=T0 + 606)) == []
    finally:
        layer.close()


def test_stale_out_of_order_reports_ignored():
    layer = vessel_layer()
    try:
        newer = json.loads(json.dumps(AIS["position_report"]))
        newer["MetaData"]["time_utc"] = "2026-10-27 01:02:00.000000 +0000 UTC"
        newer["Message"]["PositionReport"].update(Latitude=37.9, Sog=0.0)
        layer.ingest(newer, T0 + 120)
        layer.ingest(AIS["position_report"], T0 + 121)  # older timestamp arrives late
        e = layer.update(ctx(SF, view=make_view(0.0), unix=T0 + 125))[0]
        assert e.geo.lat == pytest.approx(37.9)
    finally:
        layer.close()


def test_vessel_enrichment_only_for_focus():
    layer = vessel_layer()
    try:
        layer.ingest(AIS["position_report"], T0)
        layer.ingest(AIS["static_data"], T0)
        v = make_view(0.0)
        plain = layer.update(ctx(SF, view=v, unix=T0 + 1))[0]
        assert "destination" not in plain.meta
        focused = layer.update(ctx(SF, view=v, unix=T0 + 1, focus="ais:366999001"))[0]
        assert focused.meta["destination"] == "OAKLAND"
        assert focused.meta["eta"] == "10-28 14:30 UTC"
        assert focused.meta["ship_type"] == "cargo"
        # focused vessel without static data: no enrichment keys, no crash
        layer.ingest(AIS["class_b_report"], T0)
        layer.update(ctx(SF, view=make_view(70.0), unix=T0 + 1, focus="ais:366999002"))
    finally:
        layer.close()


def test_subscription_message_format():
    layer = vessel_layer(radius_km=30)
    msg = json.loads(layer.subscription_message(*SF))
    assert msg["APIKey"] == "test-key"
    ((a, b),) = msg["BoundingBoxes"]
    assert a[0] < SF[0] < b[0] and a[1] < SF[1] < b[1]
    assert (b[0] - a[0]) * 111.32 == pytest.approx(70.0, rel=0.01)  # 2 x (30 + 5 km)
    assert (
        "PositionReport" in msg["FilterMessageTypes"]
        and "ShipStaticData" in msg["FilterMessageTypes"]
    )


class FakeWS:
    def __init__(self, inbox: queue.Queue, sent: list):
        self.inbox, self.sent = inbox, sent

    async def send(self, data):
        self.sent.append(json.loads(data))

    def __aiter__(self):
        return self

    async def __anext__(self):
        import asyncio

        while True:
            try:
                return self.inbox.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)


class FakeConnect:
    def __init__(self):
        self.inbox: queue.Queue = queue.Queue()
        self.sent: list = []
        self.urls: list = []

    def __call__(self, url):
        self.urls.append(url)
        return self

    async def __aenter__(self):
        return FakeWS(self.inbox, self.sent)

    async def __aexit__(self, *a):
        return False


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_websocket_subscribe_stream_and_resubscribe():
    fake = FakeConnect()
    layer = VesselLayer("k", radius_km=30, connect=fake, check_s=0.01)
    try:
        v = make_view(0.0)
        layer.update(ctx(SF, view=v, unix=T0))
        assert wait_for(lambda: len(fake.sent) == 1)
        assert fake.urls == ["wss://stream.aisstream.io/v0/stream"]
        assert fake.sent[0]["APIKey"] == "k"
        fake.inbox.put(json.dumps(AIS["position_report"]))
        fake.inbox.put("not json")
        fake.inbox.put(json.dumps(AIS["error_envelope"]))
        fake.inbox.put(json.dumps(AIS["static_data"]))
        assert wait_for(lambda: len(layer._static) == 1 and len(layer._reports) == 1)
        out = layer.update(ctx(SF, view=v, unix=T0 + 1))
        assert [e.id for e in out] == ["ais:366999001"]

        layer.update(ctx((SF[0] + 0.03, SF[1]), view=v, unix=T0 + 1))  # ~3.3 km: no resubscribe
        time.sleep(0.15)
        assert len(fake.sent) == 1

        layer.update(ctx((SF[0] + 0.06, SF[1]), view=v, unix=T0 + 1))  # ~6.7 km: resubscribe
        assert wait_for(lambda: len(fake.sent) == 2)
        box = fake.sent[1]["BoundingBoxes"][0]
        assert box[0][0] < SF[0] + 0.06 < box[1][0]
    finally:
        layer.close()
    assert not layer._thread.is_alive()


def test_websocket_reconnects_after_failure():
    attempts = []

    class Flaky(FakeConnect):
        def __call__(self, url):
            attempts.append(url)
            if len(attempts) == 1:
                raise ConnectionError("boom")
            return super().__call__(url)

    fake = Flaky()
    layer = VesselLayer("k", connect=fake, reconnect_s=0.02, check_s=0.01)
    try:
        layer.update(ctx(SF, unix=T0))
        assert wait_for(lambda: len(fake.sent) == 1)
        assert len(attempts) >= 2
        assert layer.last_error == "boom"
    finally:
        layer.close()


def test_vessels_update_is_fast():
    layer = vessel_layer()
    try:
        for i in range(200):
            m = json.loads(json.dumps(AIS["position_report"]))
            m["Message"]["PositionReport"].update(UserID=366000000 + i, Latitude=37.84 + i * 1e-4)
            m["MetaData"]["time_utc"] = "2026-10-27 01:00:00.000000 +0000 UTC"
            layer.ingest(m, T0)
        c = ctx(SF, view=make_view(0.0), unix=T0 + 30)
        layer.update(c)
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            out = layer.update(c)
            times.append(time.perf_counter() - t0)
        assert len(out) > 100
        assert statistics.median(times) < 0.005
    finally:
        layer.close()
