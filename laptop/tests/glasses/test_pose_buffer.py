from __future__ import annotations

import math
import threading

import pytest

from pointview.glasses.pose_buffer import MAX_EXTRAPOLATION_S, PoseBuffer

from .helpers import analytic_quat, angle_between_deg


def _fill(pose_track, upto: float | None = None, offset: float = 0.0) -> PoseBuffer:
    buf = PoseBuffer(keep_s=12.0)
    for p in pose_track:
        if upto is not None and p.t > upto:
            break
        buf.add(p.t + offset, p.quat)
    return buf


def test_empty_returns_none():
    assert PoseBuffer().pose_at(1.0) is None


def test_returned_t_equals_requested(pose_track):
    buf = _fill(pose_track)
    for t in (-1.0, 0.0, 3.1234, 10.0, 99.0):
        assert buf.pose_at(t).t == t


def test_exact_sample(pose_track):
    buf = _fill(pose_track)
    p = pose_track[400]
    got = buf.pose_at(p.t)
    assert angle_between_deg(got.quat, p.quat) < 1e-4


def test_interpolation_matches_analytic_motion(pose_track):
    buf = _fill(pose_track)
    worst = 0.0
    for i in range(1, 990):
        t = i * 0.01 + 0.0023  # off-grid
        worst = max(worst, angle_between_deg(buf.pose_at(t).quat, analytic_quat(t)))
    assert worst < 0.01  # degrees


def test_before_oldest_returns_oldest(pose_track):
    buf = _fill(pose_track, offset=5.0)
    got = buf.pose_at(1.0)
    assert got.quat == pytest.approx(pose_track[0].quat)
    assert got.t == 1.0


def test_extrapolation_follows_velocity_then_holds(pose_track):
    # Newest sample at t=2.0, where yaw is moving ~13 deg/s.
    buf = _fill(pose_track, upto=2.0)
    newest = 2.0
    for dt in (0.01, 0.03, MAX_EXTRAPOLATION_S):
        err = angle_between_deg(buf.pose_at(newest + dt).quat, analytic_quat(newest + dt))
        hold_err = angle_between_deg(analytic_quat(newest), analytic_quat(newest + dt))
        assert err < 0.5 * hold_err + 0.01
    at_limit = buf.pose_at(newest + MAX_EXTRAPOLATION_S).quat
    for dt in (0.06, 0.5, 5.0):
        assert buf.pose_at(newest + dt).quat == pytest.approx(at_limit, abs=1e-12)


def test_extrapolation_constant_rate_is_exact():
    from pointview.glasses import _quat as q

    buf = PoseBuffer()
    rate = math.radians(90.0)  # rad/s about z
    for i in range(50):
        t = i * 0.005
        buf.add(t, q.from_rotvec((0.0, 0.0, rate * t)))
    tl = 49 * 0.005
    got = buf.pose_at(tl + 0.04).quat
    assert angle_between_deg(got, q.from_rotvec((0.0, 0.0, rate * (tl + 0.04)))) < 1e-6


def test_single_sample_holds():
    buf = PoseBuffer()
    buf.add(1.0, (0.0, 1.0, 0.0, 0.0))
    assert buf.pose_at(1.02).quat == (0.0, 1.0, 0.0, 0.0)
    assert buf.pose_at(0.0).quat == (0.0, 1.0, 0.0, 0.0)


def test_slerp_shortest_path_with_sign_flip():
    from pointview.glasses import _quat as q

    a = q.from_rotvec((0.0, 0.0, math.radians(10)))
    b = q.from_rotvec((0.0, 0.0, math.radians(30)))
    b_neg = tuple(-c for c in b)  # same rotation, opposite hemisphere
    buf = PoseBuffer()
    buf.add(0.0, a)
    buf.add(1.0, b_neg)
    mid = buf.pose_at(0.5).quat
    assert angle_between_deg(mid, q.from_rotvec((0.0, 0.0, math.radians(20)))) < 1e-6


def test_out_of_order_rejected_and_normalized():
    buf = PoseBuffer()
    assert buf.add(1.0, (2.0, 0.0, 0.0, 0.0))
    assert not buf.add(1.0, (1.0, 0.0, 0.0, 0.0))
    assert not buf.add(0.5, (1.0, 0.0, 0.0, 0.0))
    assert buf.pose_at(1.0).quat == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_keeps_at_least_two_seconds():
    buf = PoseBuffer()
    for i in range(5000):
        buf.add(i * 0.005, (1.0, 0.0, 0.0, 0.0))
    assert buf.newest_t() - buf.oldest_t() >= 2.0
    assert len(buf) < 5000


def test_keep_below_two_seconds_rejected():
    with pytest.raises(ValueError):
        PoseBuffer(keep_s=1.0)


def test_discontinuity_not_used_for_velocity():
    from pointview.glasses import _quat as q

    buf = PoseBuffer()
    for i in range(10):
        buf.add(i * 0.005, (1.0, 0.0, 0.0, 0.0))
    # A jump with discontinuity flag: extrapolating right after must hold, not spike.
    buf.add(10 * 0.005, q.from_rotvec((0.0, 0.0, 0.5)), discontinuity=True)
    got = buf.pose_at(10 * 0.005 + 0.04).quat
    assert got == pytest.approx(q.from_rotvec((0.0, 0.0, 0.5)))


def test_thread_safety():
    buf = PoseBuffer()
    stop = threading.Event()
    errors: list[Exception] = []

    def reader():
        while not stop.is_set():
            try:
                p = buf.pose_at(1.0)
                if p is not None:
                    assert abs(sum(c * c for c in p.quat) - 1.0) < 1e-9
            except Exception as e:  # pragma: no cover
                errors.append(e)

    ths = [threading.Thread(target=reader) for _ in range(3)]
    for th in ths:
        th.start()
    for i in range(20000):
        buf.add(i * 0.001, (math.cos(i * 1e-3), 0.0, 0.0, math.sin(i * 1e-3)))
    stop.set()
    for th in ths:
        th.join()
    assert not errors
