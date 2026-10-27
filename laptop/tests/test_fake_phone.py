"""fake_phone: quaternion/omega geometry, scripts, schema validity, and a short live run."""

from __future__ import annotations

import asyncio
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import jsonschema
import pytest
from websockets.asyncio.server import serve

from pointview import schemas

_PATH = Path(__file__).resolve().parents[1] / "tools" / "fake_phone.py"


def load_fake_phone():
    if "fake_phone" in sys.modules:
        return sys.modules["fake_phone"]
    spec = importlib.util.spec_from_file_location("fake_phone", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fake_phone"] = mod
    spec.loader.exec_module(mod)
    return mod


fp = load_fake_phone()
validator = jsonschema.Draft202012Validator(schemas.load("phone_message"))


def angle_between(a, b) -> float:
    d = sum(x * y for x, y in zip(a, b, strict=True))
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


@pytest.mark.parametrize(
    ("yaw", "pitch"),
    [(0, 0), (90, 0), (-30, 10), (180, 0), (45, -20), (270, 60), (10, 85)],
)
def test_top_edge_points_at_scripted_yaw_pitch(yaw, pitch):
    q = fp.pointer_quat(yaw, pitch)
    assert math.isclose(sum(c * c for c in q), 1.0, abs_tol=1e-12)
    top = fp.qrotate(q, (0.0, 1.0, 0.0))
    want = fp.pointer_direction(yaw, pitch)
    assert angle_between(top, want) < 1e-6


def test_compass_sense_and_pitch_sign():
    # Yaw 90 is east (+x), yaw 0 is north (+y), pitch up is +z.
    assert fp.pointer_direction(90, 0)[0] == pytest.approx(1.0)
    assert fp.pointer_direction(0, 0)[1] == pytest.approx(1.0)
    assert fp.pointer_direction(0, 30)[2] > 0
    # Increasing yaw turns the top edge clockwise seen from above (north -> east).
    top = fp.qrotate(fp.pointer_quat(10, 0), (0.0, 1.0, 0.0))
    assert top[0] > 0


def test_screen_faces_up_when_level():
    out = fp.qrotate(fp.pointer_quat(37, 0), (0.0, 0.0, 1.0))
    assert out[2] == pytest.approx(1.0)


def test_omega_matches_finite_difference_of_quaternion():
    script = fp.Script([fp.Keyframe(0, 10, 5), fp.Keyframe(2, 50, 25)])
    t, h = 1.0, 1e-4
    _, pitch, yr, pr = script.pose_at(t)
    omega = fp.phone_omega(yr, pr, pitch)
    q0 = fp.pointer_quat(*script.pose_at(t - h)[:2])
    q1 = fp.pointer_quat(*script.pose_at(t + h)[:2])
    # Body-frame delta rotation: q0* q1 ~ exp(omega * 2h / 2)
    q0c = (q0[0], -q0[1], -q0[2], -q0[3])
    dq = fp.qmul(q0c, q1)
    est = [2 * c / (2 * h) for c in dq[1:]]
    for a, b in zip(est, omega, strict=True):
        assert a == pytest.approx(b, abs=1e-3)


def test_script_interpolation_and_events():
    s = fp.Script(
        [
            fp.Keyframe(0, 0),
            fp.Keyframe(2, 20, 10, buttons=("align",)),
            fp.Keyframe(4, 20, 10, layers={"people": False}),
        ]
    )
    assert s.pose_at(1.0)[:2] == pytest.approx((10.0, 5.0))
    assert s.pose_at(1.0)[2] == pytest.approx(10.0)
    assert s.pose_at(9.0) == (20, 10, 0.0, 0.0)
    assert [k.t_s for k in s.events_between(1.5, 2.0)] == [2]
    assert s.events_between(2.0, 3.0) == []
    assert [k.t_s for k in s.events_between(3.0, 5.0)] == [4]


def test_heading_follows_yaw_plus_north_offset():
    s = fp.Script([fp.Keyframe(0, 350.0)])
    m = fp.build_message(s, 0, 1.7e12, north_offset_deg=20.0)
    assert m["heading_deg"] == pytest.approx(10.0)
    assert fp.build_message(fp.Script([fp.Keyframe(0, -1e-9)]), 0, 1.7e12)["heading_deg"] < 360.0


@pytest.mark.parametrize("name", sorted(fp.SCRIPTS))
def test_builtin_scripts_produce_schema_valid_messages(name):
    s = fp.SCRIPTS[name]()
    n = 0
    t = 0.0
    while t <= s.duration_s:
        m = fp.build_message(s, t, 1.79e12 + t * 1000, north_offset_deg=37.0)
        validator.validate(m)
        n += 1
        t += 1 / 60
    assert n > 100


def test_align_then_point_script_shape():
    s = fp.SCRIPTS["align_then_point"]()
    assert s.pose_at(0.5)[0] == 0.0
    aligns = [k for k in s.keyframes if "align" in k.buttons]
    assert [k.t_s for k in aligns] == [1.0]
    assert sum("info" in k.buttons for k in s.keyframes) >= 3
    assert max(abs(k.yaw_deg) for k in s.keyframes) <= 20.0


def test_sweep_range():
    s = fp.SCRIPTS["sweep"]()
    assert max(k.yaw_deg for k in s.keyframes) == 30
    assert min(k.yaw_deg for k in s.keyframes) == -30


async def test_run_fake_phone_sends_valid_paced_messages_and_buttons():
    got: list[dict] = []

    async def handler(ws):
        async for raw in ws:
            got.append(json.loads(raw))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        script = fp.Script(
            [
                fp.Keyframe(0, 0),
                fp.Keyframe(0.25, 10, buttons=("align", "info")),
                fp.Keyframe(0.5, 10),
            ]
        )
        stats = await fp.run_fake_phone(
            f"ws://127.0.0.1:{port}", script, rate_hz=100, stalls=[(0.3, 100)], jitter_ms=2
        )
        await asyncio.sleep(0.05)
    assert stats.sent == len(got)
    assert 30 <= len(got) <= 51  # 51 ticks minus about 10 skipped by the stall
    for m in got:
        validator.validate(m)
    pressed = [b for m in got for b in m["buttons"]]
    assert pressed == ["align", "info"]
    ts = [m["t_ms"] for m in got]
    assert ts == sorted(ts)
    assert max(b - a for a, b in itertools.pairwise(ts)) >= 100
    assert stats.stalls_injected == 1


def test_cli_list(capsys):
    assert fp.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "align_then_point" in out
    assert "sweep" in out
    assert "still" in out
