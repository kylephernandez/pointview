from __future__ import annotations

import time

import numpy as np
import pytest

from pointview.glasses import MockGlasses
from pointview.interfaces import GlassesSource

from .helpers import ManualClock, analytic_quat, angle_between_deg, mf


def make(sample_video_path, fixtures_dir, fixture_intrinsics, **kw):
    clock = kw.pop("clock", None) or ManualClock()
    g = MockGlasses(
        sample_video_path,
        fixtures_dir / "pose_track.csv",
        fixture_intrinsics,
        clock=clock,
        **kw,
    )
    return g, clock


def test_is_glasses_source(sample_video_path, fixtures_dir, fixture_intrinsics):
    g, _ = make(sample_video_path, fixtures_dir, fixture_intrinsics)
    assert isinstance(g, GlassesSource)
    assert g.display_size == (1920, 1080)
    assert g.display_fov_deg == (46.0, 27.0)
    assert g.camera_intrinsics == fixture_intrinsics


def test_before_start_nothing(sample_video_path, fixtures_dir, fixture_intrinsics):
    g, _ = make(sample_video_path, fixtures_dir, fixture_intrinsics, realtime=False)
    assert g.latest_frame() is None
    assert g.pose_at(1.0) is None


def test_nonrealtime_one_frame_per_call_and_timing(
    sample_video_path, fixtures_dir, fixture_intrinsics
):
    g, clock = make(sample_video_path, fixtures_dir, fixture_intrinsics, realtime=False)
    g.start()
    try:
        t0 = clock()
        for i in range(10):
            f = g.latest_frame()
            assert f.t == pytest.approx(t0 + i / 30)
            assert f.bgr.shape == (540, 960, 3) and f.bgr.dtype == np.uint8
    finally:
        g.stop()


def test_nonrealtime_deterministic(sample_video_path, fixtures_dir, fixture_intrinsics):
    runs = []
    for _ in range(2):
        g, clock = make(sample_video_path, fixtures_dir, fixture_intrinsics, realtime=False)
        g.start()
        out = []
        for _ in range(40):
            f = g.latest_frame()
            out.append((f.t - clock(), f.bgr.tobytes()[:5000], g.pose_at(f.t).quat))
        g.stop()
        runs.append(out)
    assert runs[0] == runs[1]


def test_pose_at_frame_times_matches_analytic(sample_video_path, fixtures_dir, fixture_intrinsics):
    g, clock = make(sample_video_path, fixtures_dir, fixture_intrinsics, realtime=False)
    g.start()
    t0 = clock()
    for _ in range(180):
        f = g.latest_frame()
        p = g.pose_at(f.t)
        assert p.t == f.t
        assert angle_between_deg(p.quat, analytic_quat(f.t - t0)) < 0.01
    g.stop()


def test_loop_wraps_video_and_pose_by_own_length(
    sample_video_path, fixtures_dir, fixture_intrinsics
):
    g, clock = make(sample_video_path, fixtures_dir, fixture_intrinsics, realtime=False)
    g.start()
    t0 = clock()
    frames = [g.latest_frame() for _ in range(200)]  # 180 frames/cycle
    assert frames[180].t == pytest.approx(t0 + 6.0)
    assert np.array_equal(frames[180].bgr, frames[0].bgr)
    assert np.array_equal(frames[185].bgr, frames[5].bgr)
    # Pose track is 10 s: at t0 + 10.5 it has wrapped to fixture time ~0.495.
    for _ in range(200, 330):
        g.latest_frame()
    wrapped = g.pose_at(t0 + 10.5).quat
    assert angle_between_deg(wrapped, analytic_quat(0.495)) < 0.05
    # The wrap is continuous in time: no gaps, no duplicates around the seam.
    pose_a = g.pose_at(t0 + 10.0).quat
    assert angle_between_deg(pose_a, analytic_quat(10.0)) < 0.05
    g.stop()


def test_no_loop_holds_last_frame(sample_video_path, fixtures_dir, fixture_intrinsics):
    g, _ = make(sample_video_path, fixtures_dir, fixture_intrinsics, realtime=False, loop=False)
    g.start()
    frames = [g.latest_frame() for _ in range(200)]
    assert frames[179].t == frames[199].t
    assert np.array_equal(frames[179].bgr, frames[199].bgr)
    g.stop()


def test_realtime_step_follows_clock(sample_video_path, fixtures_dir, fixture_intrinsics):
    g, clock = make(sample_video_path, fixtures_dir, fixture_intrinsics, realtime=True)
    g.start()
    try:
        t0 = clock()
        assert g.latest_frame().t == t0  # frame 0 available immediately
        clock.t = t0 + 1.0 + 0.001
        deadline = time.monotonic() + 3
        while g.latest_frame().t < t0 + 1.0 - 1e-6 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert g.latest_frame().t == pytest.approx(t0 + 1.0)
        # Poses are available up to the clock, extrapolated at most 50 ms beyond.
        p = g.pose_at(clock())
        assert angle_between_deg(p.quat, analytic_quat(1.001)) < 0.05
    finally:
        g.stop()


def test_start_stop_idempotent(sample_video_path, fixtures_dir, fixture_intrinsics):
    g, _ = make(sample_video_path, fixtures_dir, fixture_intrinsics)
    g.stop()  # before start
    g.start()
    g.start()
    g.stop()
    g.stop()
    assert g.latest_frame() is None
    g.start()  # restartable
    assert g.latest_frame() is not None
    g.stop()


def test_real_clock_realtime_smoke(sample_video_path, fixtures_dir, fixture_intrinsics):
    with MockGlasses(sample_video_path, fixtures_dir / "pose_track.csv", fixture_intrinsics) as g:
        t_first = g.latest_frame().t
        time.sleep(0.3)
        assert g.latest_frame().t > t_first
        assert g.pose_at(time.monotonic()) is not None


def test_missing_video_raises(tmp_path, fixtures_dir, fixture_intrinsics):
    g = MockGlasses(tmp_path / "nope.mp4", fixtures_dir / "pose_track.csv", fixture_intrinsics)
    with pytest.raises(RuntimeError):
        g.start()


def test_frames_consistent_with_poses(sample_video_path, fixtures_dir, fixture_intrinsics, scene):
    """A marker's blob in the frame sits where the fixture projection says, at the frame's pose."""
    g, _ = make(sample_video_path, fixtures_dir, fixture_intrinsics, realtime=False)
    g.start()
    checked = 0
    for _ in range(90):
        f = g.latest_frame()
        pose = g.pose_at(f.t)
        for m in scene["markers_tracking_frame"]:
            uv = mf.project(pose.quat, mf.direction(m["yaw_deg"], m["pitch_deg"]))
            if uv is None or not (40 < uv[0] < 920 and 40 < uv[1] < 500):
                continue
            x0, y0 = int(uv[0]) - 28, int(uv[1]) - 28
            win = f.bgr[y0 : y0 + 57, x0 : x0 + 57].astype(np.int16)
            mask = np.abs(win - np.array(m["color_bgr"], np.int16)).max(axis=2) < 45
            ys, xs = np.nonzero(mask)
            assert len(xs) > 200, (m["id"], uv)  # a ~18 px radius disc
            cx, cy = xs.mean() + x0, ys.mean() + y0
            assert abs(cx - uv[0]) < 3 and abs(cy - uv[1]) < 3, (m["id"], uv, cx, cy)
            checked += 1
    g.stop()
    assert checked > 100
