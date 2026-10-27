"""Tracking cameras and device controls: VitureGlasses (fake SDK library) and MockGlasses."""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

from pointview.glasses import MockGlasses
from pointview.glasses import viture_sdk as sdk
from pointview.glasses.viture import VitureGlasses
from pointview.interfaces import IDENTITY_QUAT, TrackingCameraSource

from .helpers import ManualClock
from .test_viture import PID, FakeLib


class ControlLib(FakeLib):
    """FakeLib plus exposure and device-control functions that record their arguments."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.args: list[tuple] = []
        self.brightness = 3
        self.film = 0.25
        self.xr_device_provider_set_manual_exposure_carina = self._rec("manual_exp")
        self.xr_device_provider_set_auto_exposure_carina = self._rec("auto_exp")
        self.xr_device_provider_get_brightness_level = lambda h: self.brightness
        self.xr_device_provider_set_brightness_level = self._rec("set_brightness")
        self.xr_device_provider_get_film_mode = self._get_film
        self.xr_device_provider_set_film_mode = self._rec("set_film")

    def _rec(self, name):
        def f(*a):
            self.args.append((name, *a[1:]))
            return 0

        return f

    def _get_film(self, handle, ptr):
        ptr.contents.value = self.film
        return 0


def stereo_call(lib, left0: bytes, w: int, h: int, ts: float = 5.0) -> None:
    """Invoke the registered callback through its real ctypes type, with four buffers."""
    bufs = [ctypes.create_string_buffer(left0, len(left0)) for _ in range(4)]
    bufs[1] = ctypes.create_string_buffer(b"\xff" * len(left0), len(left0))  # right0 differs
    addrs = [ctypes.addressof(b) for b in bufs]
    lib.stereo_cb(*addrs, ts, w, h)


def started(lib=None, **kw) -> tuple[VitureGlasses, ControlLib, ManualClock]:
    lib = lib or ControlLib()
    clock = ManualClock(100.0)
    g = VitureGlasses(product_id=PID, clock=clock, poll_interval_s=0.01, _lib=lib, **kw)
    g.start()
    return g, lib, clock


def test_stereo_callback_type_has_no_c_char_p():
    # c_char_p would truncate image data at the first 0 byte.
    assert sdk.XRStereoCallback._argtypes_[:4] == (ctypes.c_void_p,) * 4
    assert sdk.XRStereoCallback._argtypes_[4:] == (ctypes.c_double, ctypes.c_int, ctypes.c_int)


def test_stereo_frame_left_only_bgr_with_zero_bytes_and_arrival_stamp():
    g, lib, clock = started()
    try:
        assert g.latest_tracking_frame() is None
        w, h = 8, 6
        gray = bytes(range(w * h))  # contains a 0 byte first
        clock.t = 123.0
        stereo_call(lib, gray, w, h)
        f = g.latest_tracking_frame()
        assert f.t == 123.0  # arrival time on the laptop clock, not the SDK timestamp
        assert f.bgr.shape == (h, w, 3) and f.bgr.dtype == np.uint8
        expect = np.frombuffer(gray, np.uint8).reshape(h, w)
        for c in range(3):
            assert np.array_equal(f.bgr[:, :, c], expect)  # gray replicated, left0 (not right0)
        assert g.latest_tracking_frame() is f  # cached until a new frame arrives
        stereo_call(lib, bytes(w * h), w, h)
        assert g.latest_tracking_frame() is not f
    finally:
        g.stop()


def test_stereo_callback_ignores_bad_sizes_and_copies_data():
    g, lib, _ = started()
    try:
        lib.stereo_cb(None, None, None, None, 1.0, 8, 6)
        lib.stereo_cb(1, 1, 1, 1, 1.0, 0, 0)
        assert g.latest_tracking_frame() is None
        buf = ctypes.create_string_buffer(bytes([7]) * 12, 12)
        a = ctypes.addressof(buf)
        lib.stereo_cb(a, a, a, a, 1.0, 4, 3)
        buf.raw = bytes(12)  # SDK reuses its buffer after the callback returns
        assert int(g.latest_tracking_frame().bgr[0, 0, 0]) == 7
    finally:
        g.stop()


def test_imu_still_works_with_stereo_registered():
    g, lib, _ = started()
    try:
        lib.imu_cb((ctypes.c_float * 6)(0, 0, 9.8, 0.1, 0.2, 0.3), 1.0)
        assert g.latest_imu().accel == pytest.approx((0, 0, 9.8), abs=1e-5)
    finally:
        g.stop()


def test_tracking_defaults_flagged_as_assumed():
    g = VitureGlasses()
    assert isinstance(g, TrackingCameraSource)
    assert g.tracking_intrinsics_are_assumed and g.tracking_mounting_is_assumed
    assert g.tracking_camera_to_head_quat == IDENTITY_QUAT
    i = g.tracking_camera_intrinsics
    assert (i.width, i.height) == (640, 480)  # guess until the first frame
    assert 2 * math.degrees(math.atan((i.width / 2) / i.fx)) == pytest.approx(110.0)


def test_tracking_intrinsics_follow_real_frame_size_and_overrides():
    g, lib, _ = started()
    try:
        stereo_call(lib, bytes(100 * 40), 100, 40)
        i = g.tracking_camera_intrinsics
        assert (i.width, i.height, i.cx, i.cy) == (100, 40, 50, 20)
        assert 2 * math.degrees(math.atan(50 / i.fx)) == pytest.approx(110.0)
    finally:
        g.stop()
    own = VitureGlasses().tracking_camera_intrinsics
    g2 = VitureGlasses(tracking_intrinsics=own, tracking_camera_to_head_quat=(0.0, 1.0, 0.0, 0.0))
    assert not g2.tracking_intrinsics_are_assumed and not g2.tracking_mounting_is_assumed
    assert g2.tracking_camera_to_head_quat == (0.0, 1.0, 0.0, 0.0)


def test_exposure_calls_and_clamping():
    g, lib, _ = started()
    try:
        g.set_tracking_exposure(4.0, 8)
        g.set_tracking_exposure(None)
        g.set_tracking_exposure(100.0, 99)  # clamped to the documented ranges
        g.set_tracking_exposure(0.0)
        assert lib.args == [
            ("manual_exp", 4.0, 8),
            ("auto_exp",),
            ("manual_exp", 8.0, 15),
            ("manual_exp", 0.01, 0),
        ]
    finally:
        g.stop()


def test_brightness_and_film_calls():
    g, lib, _ = started()
    try:
        g.set_brightness(5)
        g.set_film(1.0)
        g.set_film(0)
        assert lib.args == [("set_brightness", 5), ("set_film", 1.0), ("set_film", 0.0)]
        assert g.get_brightness() == 3
        assert g.get_film() == pytest.approx(0.25)
        for bad in (-1, 9):
            with pytest.raises(ValueError):
                g.set_brightness(bad)
        for bad in (-0.1, 1.5, float("nan")):
            with pytest.raises(ValueError):
                g.set_film(bad)
        assert len(lib.args) == 3  # invalid values never reach the SDK
    finally:
        g.stop()


def test_controls_need_started_provider():
    g = VitureGlasses(product_id=PID, _lib=ControlLib())
    for call in (
        lambda: g.set_brightness(1),
        lambda: g.get_brightness(),
        lambda: g.set_film(0.5),
        lambda: g.get_film(),
        lambda: g.set_tracking_exposure(None),
    ):
        with pytest.raises(sdk.SdkError, match="not started"):
            call()


def test_missing_symbols_raise_clear_sdk_error():
    g, _lib, _ = started(FakeLib())  # plain FakeLib exports none of the new functions
    try:
        for call, name in (
            (lambda: g.set_brightness(1), "set_brightness_level"),
            (lambda: g.get_brightness(), "get_brightness_level"),
            (lambda: g.set_film(0.5), "set_film_mode"),
            (lambda: g.get_film(), "get_film_mode"),
            (lambda: g.set_tracking_exposure(None), "set_auto_exposure_carina"),
            (lambda: g.set_tracking_exposure(1.0), "set_manual_exposure_carina"),
        ):
            with pytest.raises(sdk.SdkError, match=f"does not export .*{name}"):
                call()
    finally:
        g.stop()


def test_sdk_error_codes_surface():
    lib = ControlLib()
    lib.xr_device_provider_set_film_mode = lambda h, v: -4  # e.g. Viture Pro 2 has no film
    lib.xr_device_provider_get_brightness_level = lambda h: -5
    g, _, _ = started(lib)
    try:
        with pytest.raises(sdk.SdkError, match="-4"):
            g.set_film(1.0)
        with pytest.raises(sdk.SdkError, match="-5"):
            g.get_brightness()
    finally:
        g.stop()


# ------------------------------------------------------------------------------------- mock


def make_mock(sample_video_path, fixtures_dir, fixture_intrinsics):
    return MockGlasses(
        sample_video_path,
        fixtures_dir / "pose_track.csv",
        fixture_intrinsics,
        clock=ManualClock(),
        realtime=False,
    )


def test_mock_is_tracking_source(sample_video_path, fixtures_dir, fixture_intrinsics):
    g = make_mock(sample_video_path, fixtures_dir, fixture_intrinsics)
    assert isinstance(g, TrackingCameraSource)
    assert g.tracking_camera_to_head_quat == IDENTITY_QUAT
    assert g.latest_tracking_frame() is None  # not started


def test_mock_tracking_frames(sample_video_path, fixtures_dir, fixture_intrinsics):
    g = make_mock(sample_video_path, fixtures_dir, fixture_intrinsics)
    g.start()
    try:
        f = g.latest_tracking_frame()
        i = g.tracking_camera_intrinsics
        assert f.bgr.shape == (i.height, i.width, 3) and f.bgr.dtype == np.uint8
        assert (i.width, i.height) != (fixture_intrinsics.width, fixture_intrinsics.height)
        assert i != fixture_intrinsics
        assert np.array_equal(f.bgr[:, :, 0], f.bgr[:, :, 1])
        assert np.array_equal(f.bgr[:, :, 1], f.bgr[:, :, 2])
        assert g.latest_tracking_frame().t >= f.t  # timestamps follow the video
        # same view, rescaled: the principal point stays centred
        assert i.cx / i.width == pytest.approx(fixture_intrinsics.cx / fixture_intrinsics.width)
        assert i.fx / i.width == pytest.approx(fixture_intrinsics.fx / fixture_intrinsics.width)
    finally:
        g.stop()


def test_mock_exposure_is_recorded_noop(sample_video_path, fixtures_dir, fixture_intrinsics):
    g = make_mock(sample_video_path, fixtures_dir, fixture_intrinsics)
    g.set_tracking_exposure(2.0, 4)
    g.set_tracking_exposure(None)
    assert g.tracking_exposure_calls == [(2.0, 4), (None, None)]
