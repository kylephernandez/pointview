"""PhoneLink / PhoneLinkThread against raw websocket clients and the fake phone (localhost, port 0)."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from test_fake_phone import fp
from websockets.asyncio.client import connect

from pointview.interfaces import EXTRA_TOGGLES, LAYER_KINDS, PhoneLinkAPI
from pointview.phone_link import PhoneLink, PhoneLinkThread


class ManualClock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def sample(t_ms: float, **kw) -> dict:
    m = {
        "v": 1,
        "t_ms": t_ms,
        "quat": [1, 0, 0, 0],
        "omega": [0, 0, 0],
        "heading_deg": None,
        "heading_acc": None,
        "lat": None,
        "lon": None,
        "gps_acc": None,
        "buttons": [],
    }
    m.update(kw)
    return m


async def until(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met in time")


@pytest.fixture
async def link():
    clock = ManualClock()
    lk = PhoneLink(host="127.0.0.1", port=0, clock=clock)
    lk.clock = clock  # type: ignore[attr-defined]
    await lk.start()
    yield lk
    await lk.stop()


@pytest.fixture
def url(link):
    return f"ws://127.0.0.1:{link.port}"


async def send_all(ws, *msgs):
    for m in msgs:
        await ws.send(json.dumps(m))


def test_implements_api():
    lk = PhoneLink(host="127.0.0.1", port=0)
    for name in dir(PhoneLinkAPI):
        if not name.startswith("_"):
            assert callable(getattr(lk, name))


async def test_ephemeral_port_and_initial_state(link):
    assert link.port != 0
    assert link.latest() is None
    assert link.history() == []
    assert link.pop_buttons() == []
    assert not link.connected()
    assert not link.stalled()
    await link.send({"type": "haptic", "style": "snap"})  # no phone: no-op


async def test_sample_conversion_and_normalization(link, url):
    async with connect(url) as ws:
        await send_all(
            ws,
            sample(
                1.79e12,
                quat=[1.0005, 0, 0, 0],
                omega=[0.1, 0.2, 0.3],
                heading_deg=10.5,
                heading_acc=4,
                lat=37.8,
                lon=-122.4,
                gps_acc=8,
            ),
        )
        await until(lambda: link.latest() is not None)
        s = link.latest()
        assert link.connected()
        assert s.quat == pytest.approx((1.0, 0, 0, 0))
        assert s.omega == (0.1, 0.2, 0.3)
        assert (s.heading_deg, s.heading_acc, s.lat, s.lon, s.gps_acc) == (10.5, 4, 37.8, -122.4, 8)
        assert s.t_ms == 1.79e12
        assert s.received_at == link.clock.t
        assert s.t == pytest.approx(s.received_at)  # single sample: offset = its own delay
    await until(lambda: not link.connected())


async def test_end_to_end_with_fake_phone(link, url, phone_messages):
    script = fp.SCRIPTS["align_then_point"]()
    link.clock.t = time.monotonic()  # drive with a live-ish clock below
    link._clock = time.monotonic
    stats = await fp.run_fake_phone(url, script, rate_hz=100, duration_s=1.5, north_offset_deg=15)
    await until(lambda: link.stats().messages == stats.sent)
    assert link.stats().invalid == 0
    assert stats.sent >= 140
    assert [b.name for b in link.pop_buttons()] == ["align"]
    assert link.pop_buttons() == []
    s = link.latest()
    assert s.heading_deg == pytest.approx((script.pose_at(1.5)[0] + 15) % 360, abs=0.01)
    assert link.history(10.0)[0].t <= link.history(10.0)[-1].t


async def test_history_order_and_pruning():
    clock = ManualClock(1000.0)
    lk = PhoneLink(host="127.0.0.1", port=0, clock=clock, history_s=2.0)
    await lk.start()
    try:
        async with connect(f"ws://127.0.0.1:{lk.port}") as ws:
            for i in range(60):
                clock.t = 1000.0 + i * 0.1  # 10 Hz for 6 s
                await ws.send(json.dumps(sample(1.79e12 + i * 100, omega=[i / 100, 0, 0])))
                await until(lambda i=i: lk.stats().messages == i + 1)
            hist = lk.history(100.0)
            ts = [s.t for s in hist]
            assert ts == sorted(ts)
            assert ts[-1] - ts[0] <= 2.0 + 1e-9
            assert 20 <= len(hist) <= 21
            assert hist[-1] is lk.latest()
            last_second = lk.history(1.0)
            assert 10 <= len(last_second) <= 11
            assert all(s.t >= clock.t - 1.0 for s in last_second)
    finally:
        await lk.stop()


async def test_buttons_exactly_once_with_multiple_per_message(link, url):
    async with connect(url) as ws:
        await send_all(
            ws,
            sample(1.79e12, buttons=["align", "info"]),
            sample(1.79e12 + 16, buttons=["align_long"]),
            sample(1.79e12 + 32),
        )
        await until(lambda: link.stats().messages == 3)
        ev = link.pop_buttons()
        assert [e.name for e in ev] == ["align", "info", "align_long"]
        assert ev[0].t_ms == 1.79e12
        assert ev[0].t == pytest.approx(link.history(10)[0].t)
        assert ev[2].t_ms == 1.79e12 + 16
        assert link.pop_buttons() == []
        await send_all(ws, sample(1.79e12 + 48, buttons=["info"]))
        await until(lambda: link.stats().messages == 4)
        assert [e.name for e in link.pop_buttons()] == ["info"]
        assert link.pop_buttons() == []
        # PhoneState keeps the per-message buttons for inspection, but the queue is the source of truth.
        assert link.latest().buttons == ("info",)


async def test_invalid_and_non_json_dropped_connection_survives(link, url):
    bad = [
        "not json at all",
        "[1, 2, 3]",
        '{"v": 1}',
        json.dumps(sample(1.79e12, extra="field")),
        json.dumps(sample(1.79e12, quat=[1, 0, 0])),
        json.dumps(sample(1.79e12, heading_deg=400)),
        json.dumps(sample(1.79e12, buttons=["bogus"])),
        json.dumps(sample(1.79e12, v=2)),
        '{"v":1,"t_ms":NaN,"quat":[1,0,0,0],"omega":[0,0,0],"heading_deg":null,'
        '"heading_acc":null,"lat":null,"lon":null,"gps_acc":null,"buttons":[]}',
        json.dumps(sample(1.79e12, quat=[0, 0, 0, 0])),
    ]
    async with connect(url) as ws:
        for b in bad:
            await ws.send(b)
        await ws.send(b"\x00\x01binary")
        await ws.send(json.dumps(sample(1.79e12)))
        await until(lambda: link.stats().messages == 1)
        st = link.stats()
        assert st.invalid == len(bad) + 1
        assert st.messages == 1
        assert link.connected()
        assert link.pop_buttons() == []
        assert link.latest() is not None


async def test_layers_persist_when_omitted(link, url):
    async with connect(url) as ws:
        await send_all(ws, sample(1.79e12))
        await until(lambda: link.latest() is not None)
        assert link.latest().layers == layers()
        await send_all(ws, sample(1.79e12 + 16, layers={"people": False}))
        await until(lambda: link.stats().messages == 2)
        assert link.latest().layers == layers(people=False)
        await send_all(ws, sample(1.79e12 + 32), sample(1.79e12 + 48, layers={"landmarks": False}))
        await until(lambda: link.stats().messages == 4)
        assert link.history(10)[-2].layers == layers(people=False)
        assert link.latest().layers == layers(people=False, landmarks=False)
        # History entries do not alias one dict.
        assert link.history(10)[0].layers["people"] is True


def layers(**off: bool) -> dict:
    return {**dict.fromkeys((*LAYER_KINDS, *EXTRA_TOGGLES), True), **off}


async def test_look_and_passthrough_persist_when_omitted(link, url):
    async with connect(url) as ws:
        await send_all(ws, sample(1.79e12))
        await until(lambda: link.latest() is not None)
        assert link.latest().look == "normal" and link.latest().passthrough is False
        await send_all(ws, {**sample(1.79e12 + 16), "look": "nvg", "passthrough": True})
        await until(lambda: link.stats().messages == 2)
        await send_all(ws, sample(1.79e12 + 32, layers={"aircraft": False}))
        await until(lambda: link.stats().messages == 3)
        st = link.latest()
        assert st.look == "nvg" and st.passthrough is True
        assert st.layers == layers(aircraft=False)


async def test_second_connection_replaces_first(link, url):
    async with connect(url) as first:
        await send_all(first, sample(1.79e12))
        await until(lambda: link.stats().messages == 1)
        async with connect(url) as second:
            await until(lambda: link.stats().connections == 2)
            await asyncio.wait_for(first.wait_closed(), 3.0)
            assert first.close_code == 1000
            assert link.connected()
            await send_all(second, sample(1.79e12 + 16, buttons=["info"]))
            await until(lambda: link.stats().messages == 2)
            assert [e.name for e in link.pop_buttons()] == ["info"]
            # Sends go to the new phone.
            await link.send({"type": "haptic", "style": "select"})
            assert json.loads(await asyncio.wait_for(second.recv(), 3.0)) == {
                "type": "haptic",
                "style": "select",
            }
        await until(lambda: not link.connected())


async def test_old_connection_closing_does_not_disconnect_new(link, url):
    first = await connect(url)
    await send_all(first, sample(1.79e12))
    await until(lambda: link.connected())
    async with connect(url):
        await until(lambda: link.stats().connections == 2)
        await first.wait_closed()
        await asyncio.sleep(0.05)
        assert link.connected()


async def test_stall_detection_with_injected_clock(link, url):
    clock = link.clock
    async with connect(url) as ws:
        assert not link.stalled()
        await send_all(ws, sample(1.79e12))
        await until(lambda: link.stats().messages == 1)
        clock.t += 0.040
        assert not link.stalled()
        clock.t += 0.020  # 60 ms since the last sample
        assert link.stalled()
        await send_all(ws, sample(1.79e12 + 60))
        await until(lambda: link.stats().messages == 2)
        assert not link.stalled()
        st = link.stats()
        assert st.stalls == 1
        assert st.longest_gap_ms == pytest.approx(60.0)
        clock.t += 0.016
        await send_all(ws, sample(1.79e12 + 76))
        await until(lambda: link.stats().messages == 3)
        assert link.stats().stalls == 1
    await until(lambda: not link.connected())
    assert not link.stalled()  # not connected -> not stalled


async def test_stall_with_fake_phone_injected_pause():
    lk = PhoneLink(host="127.0.0.1", port=0)  # real clock
    await lk.start()
    try:
        script = fp.SCRIPTS["still"]()
        task = asyncio.create_task(
            fp.run_fake_phone(
                f"ws://127.0.0.1:{lk.port}",
                script,
                rate_hz=100,
                duration_s=1.0,
                stalls=[(0.3, 200)],
            )
        )
        saw_stalled = False
        while not task.done():
            saw_stalled |= lk.stalled()
            await asyncio.sleep(0.005)
        await task
        assert saw_stalled
        st = lk.stats()
        assert st.stalls >= 1
        assert st.longest_gap_ms >= 150
    finally:
        await lk.stop()


async def test_clock_offset_mapping_with_skewed_phone_clock():
    lk = PhoneLink(host="127.0.0.1", port=0)  # real monotonic clock
    await lk.start()
    try:
        skew_ms = 3 * 3600 * 1000
        script = fp.SCRIPTS["still"]()
        await fp.run_fake_phone(
            f"ws://127.0.0.1:{lk.port}", script, rate_hz=100, duration_s=0.5, clock_skew_ms=skew_ms
        )
        await until(lambda: lk.stats().messages >= 45)
        for s in lk.history(10.0):
            assert -1e-6 <= s.received_at - s.t < 0.05
        assert lk.stats().excess_latency_ms is not None
        assert lk.stats().excess_latency_ms < 50.0
        # Phone time really was 3 h off the laptop's wall clock.
        assert abs(lk.latest().t_ms / 1000.0 - time.time() - 3 * 3600) < 5
    finally:
        await lk.stop()


async def test_min_latency_filter_uses_best_recent_sample(link, url):
    clock = link.clock
    async with connect(url) as ws:
        # Phone clock 5 s ahead; first sample delayed 30 ms, second 10 ms, third 20 ms.
        for i, delay in enumerate([0.030, 0.010, 0.020]):
            clock.t = 200.0 + i * 0.1 + delay
            await ws.send(json.dumps(sample((200.0 + i * 0.1 + 5.0) * 1000)))
            await until(lambda i=i: link.stats().messages == i + 1)
        ts = [(s.t, s.received_at) for s in link.history(10.0)]
        # Causal filter: sample 1 defines its own offset; sample 2 lowers it; sample 3 shows 10 ms excess.
        assert [round(r - t, 3) for t, r in ts] == [0.0, 0.0, 0.010]
        assert link.stats().excess_latency_ms == pytest.approx(10.0)
        # An old best sample ages out of the 2 s window.
        clock.t = 203.0
        await ws.send(json.dumps(sample((203.0 + 5.0 - 0.05) * 1000)))  # delayed 50 ms
        await until(lambda: link.stats().messages == 4)
        assert link.latest().t == pytest.approx(link.latest().received_at)


async def test_send_valid_messages_reach_phone_and_invalid_rejected(link, url):
    async with connect(url) as ws:
        await send_all(ws, sample(1.79e12))
        await until(lambda: link.connected())
        await link.send({"type": "haptic", "style": "snap"})
        await link.send({"type": "status", "aligned": True, "selected": "car", "fps": 58.9})
        await link.send({"type": "status", "aligned": False, "selected": None})
        got = [json.loads(await asyncio.wait_for(ws.recv(), 3.0)) for _ in range(3)]
        assert got[0] == {"type": "haptic", "style": "snap"}
        assert got[1]["selected"] == "car"
        assert got[2]["selected"] is None
        for bad in (
            {"type": "haptic", "style": "boom"},
            {"type": "haptic"},
            {"type": "status", "aligned": "yes", "selected": None},
            {"type": "nope"},
            {"type": "haptic", "style": "snap", "extra": 1},
        ):
            with pytest.raises(ValueError):
                await link.send(bad)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ws.recv(), 0.1)


async def test_send_validates_even_without_phone(link):
    with pytest.raises(ValueError):
        await link.send({"type": "haptic", "style": "boom"})


async def test_fake_phone_receives_haptic(link, url):
    got = []
    script = fp.Script([fp.Keyframe(0, 0), fp.Keyframe(0.6, 0)])
    task = asyncio.create_task(fp.run_fake_phone(url, script, rate_hz=50, on_message=got.append))
    await until(lambda: link.connected())
    await link.send({"type": "haptic", "style": "align_ok"})
    stats = await task
    assert got == [{"type": "haptic", "style": "align_ok"}]
    assert stats.received == got


# ----------------------------------------------------------------------------- threaded wrapper


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met in time")


def run_phone_in_thread(url, script, **kw):
    result = {}

    def target():
        result["stats"] = asyncio.run(fp.run_fake_phone(url, script, **kw))

    t = threading.Thread(target=target)
    t.start()
    return t, result


def test_thread_wrapper_start_stop_clean():
    before = {t.name for t in threading.enumerate()}
    th = PhoneLinkThread(host="127.0.0.1", port=0)
    th.start()
    try:
        assert th.running()
        assert th.port != 0
        assert th.latest() is None
        assert not th.connected()
        th.send_nowait({"type": "haptic", "style": "snap"})  # no phone: no-op
        with pytest.raises(ValueError):
            th.send_nowait({"type": "haptic", "style": "bad"})

        script = fp.Script(
            [fp.Keyframe(0, 0), fp.Keyframe(0.1, 5, buttons=("info",)), fp.Keyframe(1.0, 5)]
        )
        got = []
        t, res = run_phone_in_thread(
            f"ws://127.0.0.1:{th.port}", script, rate_hz=100, on_message=got.append
        )
        wait_for(th.connected)
        th.send_nowait({"type": "haptic", "style": "select"})
        t.join(5)
        assert not t.is_alive()
        assert got == [{"type": "haptic", "style": "select"}]
        assert th.stats().messages == res["stats"].sent
        assert [b.name for b in th.pop_buttons()] == ["info"]
        assert th.pop_buttons() == []
        assert th.latest() is not None
        assert th.history(5.0)
        wait_for(lambda: not th.connected())
    finally:
        th.stop()
    assert not th.running()
    th.stop()  # idempotent
    wait_for(lambda: {t.name for t in threading.enumerate()} <= before)


def test_thread_wrapper_context_manager_and_restart_port_released():
    with PhoneLinkThread(host="127.0.0.1", port=0) as th:
        port = th.port
    # Port is free again: a second link can bind it.
    with PhoneLinkThread(host="127.0.0.1", port=port) as th2:
        assert th2.port == port


def test_thread_wrapper_start_error_is_raised():
    with PhoneLinkThread(host="127.0.0.1", port=0) as th:
        busy = PhoneLinkThread(host="127.0.0.1", port=th.port)
        with pytest.raises(OSError):
            busy.start()
        assert not busy.running()


async def test_tracking_ack_is_session_bound_and_resets_on_reconnect(link, url):
    async with connect(url) as ws:
        await until(link.connected)
        first = link.tracking_target()
        await send_all(
            ws,
            sample(
                1000, tracking={"session": first.session, "enabled": True, "marker_size_m": 0.06}
            ),
        )
        await until(lambda: link.latest() is not None)
        assert link.latest().tracking_enabled
        assert link.latest().marker_size_m == 0.06
        await link.send(
            {
                "type": "status",
                "aligned": False,
                "selected": None,
                "tracking": {**first.message(), "state": "verifying"},
            }
        )
        message = json.loads(await ws.recv())
        assert message["tracking"]["session"] == first.session
    await until(lambda: not link.connected())
    async with connect(url) as ws:
        await until(link.connected)
        second = link.tracking_target()
        assert second.session != first.session
        assert link.latest() is None and link.history() == []
        await send_all(
            ws,
            sample(
                2000, tracking={"session": first.session, "enabled": True, "marker_size_m": 0.06}
            ),
        )
        await until(lambda: link.latest() is not None)
        assert not link.latest().tracking_enabled
        assert link.latest().tracking_session is None
        assert link.latest().marker_size_m is None


@pytest.mark.parametrize("size", [0, -0.1, 0.001, 1, "0.06"])
def test_bad_tracking_scale_is_rejected(size):
    lk = PhoneLink()
    lk._on_frame(
        json.dumps(
            sample(1000, tracking={"session": "a" * 32, "enabled": True, "marker_size_m": size})
        )
    )
    assert lk.latest() is None
    assert lk.stats().invalid == 1
