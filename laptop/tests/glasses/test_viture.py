"""VitureGlasses / SDK binding tests against a fake library object (no DLL, no hardware)."""

from __future__ import annotations

import ctypes
import math
import time

import cv2
import numpy as np
import pytest

from pointview.glasses import _quat as q
from pointview.glasses import viture_sdk as sdk
from pointview.glasses.viture import VitureGlasses, default_intrinsics
from pointview.interfaces import GlassesSource

from .helpers import ManualClock, angle_between_deg

PID = 0x1234  # arbitrary test value; the real Luma Ultra id is unknown


class FakeLib:
    """Stands in for the ctypes CDLL. Functions receive the same Python/ctypes objects the real
    foreign call would get (before C conversion)."""

    def __init__(self, camera: bool = True):
        self.calls: list[str] = []
        self.pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.status = 0
        self.imu_cb = None
        self.cam_cb = None
        self.stereo_cb = None
        self.created = False

        def rec(name, ret=0):
            def f(*a):
                self.calls.append(name)
                return ret

            return f

        self.xr_device_provider_create = lambda pid: (self.calls.append("create"), 0xBEEF)[1]
        self.xr_device_provider_set_dof_type_carina = rec("set_dof")
        self.xr_device_provider_initialize = rec("initialize")
        self.xr_device_provider_start = rec("start")
        self.xr_device_provider_stop = rec("stop")
        self.xr_device_provider_shutdown = rec("shutdown")
        self.xr_device_provider_destroy = rec("destroy")
        self.xr_device_provider_get_gl_pose_carina = self._get_pose
        self.xr_device_provider_register_callbacks_carina = self._register
        if camera:
            self.xr_camera_provider_get_camera_vid = lambda pid: 0x0C45
            self.xr_camera_provider_get_camera_pid = lambda pid: 0x636B
            self.xr_camera_provider_create = lambda vid, pid: 0xCAFE
            self.xr_camera_provider_start = self._cam_start
            self.xr_camera_provider_stop = rec("cam_stop")
            self.xr_camera_provider_destroy = rec("cam_destroy")

    def _get_pose(self, handle, pose, predict, status_ptr):
        for i, v in enumerate(self.pose):
            pose[i] = v
        status_ptr.contents.value = self.status
        return 0

    def _register(self, handle, pose_cb, vsync_cb, imu_cb, stereo_cb):
        self.imu_cb = imu_cb
        self.stereo_cb = stereo_cb
        return 0

    def _cam_start(self, cam, cb, user):
        self.cam_cb = cb
        return 0


def test_import_is_safe_and_load_fails_clearly_off_windows():
    with pytest.raises(sdk.SdkError, match="Windows"):
        sdk.load_library()


def test_start_without_product_id_and_no_glasses_clear_error(monkeypatch):
    monkeypatch.delenv("VITURE_PRODUCT_ID", raising=False)
    with pytest.raises(sdk.SdkError, match="No VITURE glasses found"):
        VitureGlasses(_lib=FakeLib()).start()  # off Windows nothing is detected


def test_start_without_dll_clear_error():
    with pytest.raises(sdk.SdkError):
        VitureGlasses(product_id=PID).start()


def test_defaults():
    g = VitureGlasses()
    assert isinstance(g, GlassesSource)
    assert g.display_size == (1920, 1080)
    assert g.display_fov_deg == (46.0, 27.0)
    assert g.intrinsics_are_assumed
    i = g.camera_intrinsics
    assert (i.width, i.height) == (1920, 1080)
    assert i.fx == pytest.approx(960 / math.tan(math.radians(45)))
    assert VitureGlasses(intrinsics=default_intrinsics()).intrinsics_are_assumed is False


@pytest.mark.parametrize(
    ("quat", "expect_forward", "expect_up"),
    [
        ((1, 0, 0, 0), (0, 1, 0), (0, 0, 1)),
        # GL yaw left 90 deg: rotation about GL +y by +90 -> forward (-z) goes to -x (west).
        ((math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0), (-1, 0, 0), (0, 0, 1)),
        # GL pitch up 90 deg: rotation about GL +x by +90 -> forward (-z) goes to +y (GL up) -> T up; head up then points back.
        ((math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0), (0, 0, 1), (0, -1, 0)),
    ],
)
def test_carina_conversion_axes(quat, expect_forward, expect_up):
    h = sdk.carina_gl_to_head_quat(quat)
    assert sum(c * c for c in h) == pytest.approx(1.0)

    def rot(v):
        r = q.mul(q.mul(h, (0.0, *v)), q.conj(h))
        return r[1:]

    assert rot((0, 1, 0)) == pytest.approx(expect_forward, abs=1e-9)
    assert rot((0, 0, 1)) == pytest.approx(expect_up, abs=1e-9)


def test_carina_conversion_turn_right_is_clockwise_yaw():
    # GL yaw RIGHT by 20 deg = rotation about GL +y by -20 deg.
    a = math.radians(-20)
    h = sdk.carina_gl_to_head_quat((math.cos(a / 2), 0.0, math.sin(a / 2), 0.0))
    r = q.mul(q.mul(h, (0.0, 0.0, 1.0, 0.0)), q.conj(h))[1:]
    yaw = math.degrees(math.atan2(r[0], r[1]))  # compass sense
    assert yaw == pytest.approx(20.0)


def test_world_to_device_flag(monkeypatch):
    a = math.radians(-20)
    gl = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)
    fwd = sdk.carina_gl_to_head_quat(gl)
    monkeypatch.setattr(sdk, "CARINA_POSE_IS_WORLD_TO_DEVICE", True)
    inv = sdk.carina_gl_to_head_quat(gl)
    assert angle_between_deg(inv, q.conj(fwd)) < 1e-9


def test_bindings_pose_roundtrip_and_struct_parsing():
    lib = FakeLib()
    lib.pose = [0.1, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5]
    b = sdk.SdkBindings(lib)
    pose, status = b.get_gl_pose_carina(0xBEEF)
    assert pose == pytest.approx(tuple(lib.pose))
    assert status == 0
    lib.status = 1
    assert b.get_gl_pose_carina(0xBEEF)[1] == 1


def test_bindings_missing_symbols_tolerated():
    b = sdk.SdkBindings(FakeLib(camera=False))
    assert not b.has("xr_camera_provider_create")
    with pytest.raises(sdk.SdkError):
        b.camera_ids(1)


def test_bindings_error_codes():
    lib = FakeLib()
    lib.xr_device_provider_start = lambda h: -2
    b = sdk.SdkBindings(lib)
    with pytest.raises(sdk.SdkError, match="-2"):
        b.start(1)


def test_imu_callback_parsing():
    lib = FakeLib()
    b = sdk.SdkBindings(lib)
    got = []
    b.register_imu_callback(0xBEEF, lambda v, ts: got.append((v, ts)))
    arr = (ctypes.c_float * 6)(1, 2, 3, 4, 5, 6)
    lib.imu_cb(arr, 12.5)
    assert got == [((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), 12.5)]


def _jpeg(color=(10, 200, 30), size=(64, 48)) -> bytes:
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:] = color
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_camera_callback_struct_parsing():
    lib = FakeLib()
    b = sdk.SdkBindings(lib)
    got = []
    b.camera_start(0x0C45, 0x636B, lambda *a: got.append(a))
    data = _jpeg()
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    frame = sdk.XRCameraFrame(
        ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)), len(data), 64, 48, 0, 123456789, 7
    )
    lib.cam_cb(ctypes.pointer(frame), None)
    assert got == [(data, 64, 48, 123456789, 7)]
    # empty frames are ignored
    lib.cam_cb(ctypes.pointer(sdk.XRCameraFrame()), None)
    assert len(got) == 1


def test_viture_full_flow_with_fake_lib():
    lib = FakeLib()
    clock = ManualClock(50.0)
    g = VitureGlasses(
        product_id=PID, clock=clock, exposure_latency_s=0.03, poll_interval_s=0.002, _lib=lib
    )
    g.start()
    try:
        # lifecycle order per the docs: create, set_dof (before initialize), initialize, start
        assert lib.calls[:4] == ["create", "set_dof", "initialize", "start"]
        # camera: SDK stream -> decoded frame stamped arrival - latency
        data = _jpeg((10, 200, 30))
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        fr = sdk.XRCameraFrame(
            ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8)), len(data), 64, 48, 0, 1, 0
        )
        clock.t = 51.0
        lib.cam_cb(ctypes.pointer(fr), None)
        deadline = time.monotonic() + 3
        while g.latest_frame() is None and time.monotonic() < deadline:
            time.sleep(0.005)
        f = g.latest_frame()
        assert f.t == pytest.approx(51.0 - 0.03)
        assert f.bgr.shape == (48, 64, 3)
        assert abs(int(f.bgr[10, 10, 1]) - 200) < 12
        # IMU
        lib.imu_cb((ctypes.c_float * 6)(0, 0, 9.8, 0.1, 0.2, 0.3), 1.0)
        imu = g.latest_imu()
        assert imu.accel == pytest.approx((0, 0, 9.8), abs=1e-5)
        assert imu.gyro == pytest.approx((0.1, 0.2, 0.3), abs=1e-5)
        # pose: identity SDK pose -> identity head pose
        clock.t = 52.0
        assert g.poll_pose_once()
        p = g.pose_at(52.0)
        assert p.quat == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-6)
        assert g.tracking_stable
        lib.status = 1
        g.poll_pose_once()
        assert not g.tracking_stable
    finally:
        g.stop()
    g.stop()
    assert lib.calls[-3:] == ["stop", "shutdown", "destroy"]


def test_pose_thread_rate_and_ordering():
    lib = FakeLib()
    g = VitureGlasses(product_id=PID, poll_interval_s=0.002, _lib=lib)
    g.start()
    try:
        time.sleep(0.4)
        assert g.pose_at(time.monotonic()) is not None
        assert g.measured_pose_rate_hz() > 60  # fake lib is instant; CI-safe bound
    finally:
        g.stop()


def test_no_camera_api_means_pose_only():
    g = VitureGlasses(product_id=PID, _lib=FakeLib(camera=False))
    g.start()
    try:
        assert g.latest_frame() is None
    finally:
        g.stop()


def test_opencv_camera_missing_index_raises_and_cleans_up():
    lib = FakeLib()
    g = VitureGlasses(product_id=PID, camera_index=97, _lib=lib)
    with pytest.raises(sdk.SdkError, match="camera index"):
        g.start()
    assert lib.calls[-1] == "destroy"


def test_usb_unavailable_explains_likely_fixes():
    lib = FakeLib()
    lib.xr_device_provider_initialize = lambda h, a, b: -2
    g = VitureGlasses(product_id=PID, _lib=lib)
    with pytest.raises(sdk.SdkError, match=r"USB not connected.*SpaceWalker for Windows"):
        g.start()


def test_invalid_product_id_falls_back_to_detection(monkeypatch):
    from pointview.glasses import viture

    lib = FakeLib()
    lib.xr_device_provider_is_product_id_valid = lambda pid: int(pid == 0x1104)
    monkeypatch.setattr(viture, "connected_viture_pids", lambda: [0x1102, 0x1104])
    g = VitureGlasses(product_id=0x1102, _lib=lib)
    g.start()
    try:
        assert g._product_id == 0x1104 and "create" in lib.calls
    finally:
        g.stop()


def test_product_id_autodetect_from_windows_instance_ids():
    from pointview.glasses.viture import parse_viture_pids, pick_product_id

    # Real output from a Luma Ultra on Windows (2026-10-26).
    ids = """USB\\VID_35CA&PID_1102&MI_02\\7&315633E2&0&0002
USB\\VID_35CA&PID_1104\\6&61DBEA8&0&1
HID\\VID_35CA&PID_1102&MI_04\\8&DF0244&0&0000
USB\\VID_35CA&PID_1102&MI_03\\7&315633E2&0&0003
USB\\VID_35CA&PID_1102\\1234333333
USB\\VID_35CA&PID_1102&MI_04\\7&315633E2&0&0004
HID\\VID_35CA&PID_1102&MI_03\\8&18A28E89&0&0000
USB\\VID_35CA&PID_1102&MI_00\\7&315633E2&0&0000"""
    assert parse_viture_pids(ids) == [0x1102, 0x1104]
    lib = FakeLib()
    lib.xr_device_provider_is_product_id_valid = lambda pid: int(pid == 0x1104)
    assert pick_product_id([0x1102, 0x1104], sdk.SdkBindings(lib)) == 0x1104
    with pytest.raises(sdk.SdkError, match="No VITURE glasses found"):
        pick_product_id([], sdk.SdkBindings(lib))
