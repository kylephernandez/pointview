"""VitureGlasses: the real glasses (VITURE Luma Ultra) as a ``GlassesSource``.

Windows x86_64 + the VITURE SDK are required at ``start()``; importing this module on any OS is safe
(the DLL is loaded lazily). See ``viture_sdk`` for the API sources and everything unverified.

Pose: a background thread polls ``get_gl_pose_carina`` (~250 Hz target, the docs' own loop uses
8 ms) and stores ``carina_gl_to_head_quat(q)`` in the pose buffer stamped with the poll time on the
laptop clock minus ``pose_latency_s`` (default 0: the SDK pose time is on an undocumented clock).
Use ``measured_pose_rate_hz()`` on hardware to confirm >= 120 Hz.

Camera: with ``camera_index=None`` the SDK pass-through stream (1920x1080 MJPEG) is used; with an
integer ``camera_index`` OpenCV ``VideoCapture`` is used instead (the camera is UVC). To find the
index run ``python -c "from pointview.glasses.viture import list_cameras; print(list_cameras())"``
and pick the entry that reports 1920x1080 (the glasses' front camera, USB id 0C45:636B); the laptop's
own webcam is usually index 0. Frames are stamped ``arrival time - exposure_latency_s``.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pointview.glasses import viture_sdk as sdk
from pointview.glasses.pose_buffer import PoseBuffer
from pointview.interfaces import IDENTITY_QUAT, CameraIntrinsics, Clock, Frame, Pose, Quat, now

log = logging.getLogger(__name__)

#: ASSUMED, not documented: horizontal FOV of the 1920x1080 front camera. MUST BE CALIBRATED
#: (OpenCV checkerboard) before trusting label alignment.
ASSUMED_CAMERA_HFOV_DEG = 90.0
CAMERA_SIZE = (1920, 1080)
#: ASSUMED (contract section 9): horizontal FOV of a side tracking camera. MUST BE MEASURED.
ASSUMED_TRACKING_HFOV_DEG = 110.0
#: Frame size used for the tracking intrinsics before the first stereo frame has arrived (guess;
#: the docs state no fixed size). Replaced by the real size as soon as a frame arrives.
ASSUMED_TRACKING_SIZE = (640, 480)
#: Documented limits.
EXPOSURE_MS_RANGE = (0.01, 8.0)
EXPOSURE_GAIN_RANGE = (0, 15)
#: Gain used when manual exposure is requested without one. NOT documented; a guess.
DEFAULT_MANUAL_GAIN = 0
#: Luma series brightness range per the docs' device-control table.
BRIGHTNESS_RANGE = (0, 8)


def default_intrinsics(
    size: tuple[int, int] = CAMERA_SIZE, hfov_deg: float = ASSUMED_CAMERA_HFOV_DEG
) -> CameraIntrinsics:
    """Pinhole intrinsics from an ASSUMED field of view. MUST-CALIBRATE."""
    import math

    w, h = size
    f = (w / 2) / math.tan(math.radians(hfov_deg / 2))
    return CameraIntrinsics(f, f, w / 2, h / 2, w, h, ())


@dataclass(frozen=True, slots=True)
class ImuSample:
    t: float
    #: Carina raw layout [ax, ay, az, gx, gy, gz]; units unverified, axes are the SDK's.
    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]


def list_cameras(max_index: int = 8) -> list[tuple[int, int, int]]:
    """Probe OpenCV camera indices. Returns (index, width, height) for each that opens."""
    out = []
    for i in range(max_index):
        cap = _open_capture(i)
        if cap.isOpened():
            out.append(
                (i, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )
        cap.release()
    return out


def _open_capture(index: int) -> cv2.VideoCapture:
    if sys.platform == "win32":
        for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
            cap = cv2.VideoCapture(index, api)
            if cap.isOpened():
                return cap
            cap.release()
    return cv2.VideoCapture(index)


VITURE_VID = 0x35CA
_PID_RE = re.compile(r"VID_35CA&PID_([0-9A-F]{4})", re.IGNORECASE)


def parse_viture_pids(instance_ids: str) -> list[int]:
    """VITURE product ids in Windows device instance ids, the one with most interfaces first."""
    counts: dict[int, int] = {}
    for m in _PID_RE.finditer(instance_ids):
        pid = int(m.group(1), 16)
        counts[pid] = counts.get(pid, 0) + 1
    return sorted(counts, key=lambda p: -counts[p])


def connected_viture_pids() -> list[int]:
    """Product ids of VITURE USB devices Windows sees right now (empty elsewhere / on error)."""
    if os.name != "nt":
        return []
    cmd = (
        "Get-PnpDevice -PresentOnly | Where-Object InstanceId -match 'VID_35CA' | "
        "ForEach-Object InstanceId"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        log.warning("could not list USB devices to find the glasses", exc_info=True)
        return []
    return parse_viture_pids(out)


def pick_product_id(found: list[int], bindings: sdk.SdkBindings) -> int:
    """First connected VITURE product id the SDK accepts (docs: validate before create)."""
    for pid in found:
        if bindings.product_id_valid(pid) is not False:
            log.info(
                "using VITURE glasses product id %#06x (found %s)", pid, [hex(p) for p in found]
            )
            return pid
    if found:
        raise sdk.SdkError(
            f"VITURE USB devices found ({', '.join(hex(p) for p in found)}) but the SDK accepts "
            "none of them. Set VITURE_PRODUCT_ID to override."
        )
    raise sdk.SdkError(
        "No VITURE glasses found on USB (no device with VID_35CA). Plug them straight into the "
        "laptop's DP Alt Mode USB-C port with the cable that came with them, check the display "
        "turns on, then retry. Or set VITURE_PRODUCT_ID."
    )


class VitureGlasses:
    def __init__(
        self,
        camera_index: int | None = None,
        display_fov_deg: tuple[float, float] = (46.0, 27.0),
        intrinsics: CameraIntrinsics | None = None,
        sdk_path: Path | None = None,
        *,
        product_id: int | None = None,
        display_size: tuple[int, int] = (1920, 1080),
        clock: Clock = now,
        exposure_latency_s: float = 0.030,
        pose_latency_s: float = 0.0,
        poll_interval_s: float = 0.004,
        six_dof: bool = False,
        tracking_intrinsics: CameraIntrinsics | None = None,
        tracking_camera_to_head_quat: Quat | None = None,
        _lib: Any = None,
    ) -> None:
        self._camera_index = camera_index
        self._display_fov = display_fov_deg
        self._display_size = display_size
        self._intrinsics = intrinsics or default_intrinsics()
        self.intrinsics_are_assumed = intrinsics is None  # True => must calibrate
        self._sdk_path = sdk_path
        if product_id is None and os.environ.get("VITURE_PRODUCT_ID"):
            product_id = int(os.environ["VITURE_PRODUCT_ID"], 0)
        self._product_id = product_id
        self._clock = clock
        self._exposure_latency_s = exposure_latency_s
        self._pose_latency_s = pose_latency_s
        self._poll_interval_s = poll_interval_s
        self._six_dof = six_dof
        self._injected_lib = _lib
        self._tracking_intrinsics = tracking_intrinsics
        #: True => the tracking intrinsics are a guess (110 deg FOV) and must be measured.
        self.tracking_intrinsics_are_assumed = tracking_intrinsics is None
        self._tracking_to_head = tracking_camera_to_head_quat or IDENTITY_QUAT
        #: True => the mounting rotation is the identity placeholder, not measured.
        self.tracking_mounting_is_assumed = tracking_camera_to_head_quat is None
        # Newest raw left0 buffer: (gray bytes, width, height, arrival time). Converted lazily.
        self._tracking_raw: tuple[bytes, int, int, float] | None = None
        self._tracking_cache: tuple[tuple[bytes, int, int, float], Frame] | None = None

        self._buffer = PoseBuffer()
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._imu: ImuSample | None = None
        self._stable = False
        self._poll_times: deque[float] = deque(maxlen=512)
        self._bindings: sdk.SdkBindings | None = None
        self._handle: Any = None
        self._cam: Any = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._cap: cv2.VideoCapture | None = None
        self._jpeg: tuple[bytes, float] | None = None
        self._jpeg_ready = threading.Event()
        self._started = False

    # ---------------------------------------------------------------- GlassesSource properties

    @property
    def camera_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    @property
    def display_size(self) -> tuple[int, int]:
        return self._display_size

    @property
    def display_fov_deg(self) -> tuple[float, float]:
        return self._display_fov

    # ------------------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._started:
            return
        lib = (
            self._injected_lib
            if self._injected_lib is not None
            else sdk.load_library(self._sdk_path)
        )
        b = sdk.SdkBindings(lib)
        self._bindings = b
        self._stop.clear()
        if self._product_id is not None and b.product_id_valid(self._product_id) is False:
            log.warning(
                "the SDK rejects product id %#06x (VITURE_PRODUCT_ID?); detecting instead",
                self._product_id,
            )
            self._product_id = None
        if self._product_id is None:
            self._product_id = pick_product_id(connected_viture_pids(), b)
        handle = b.create(self._product_id)
        try:
            b.set_dof_type(handle, self._six_dof)  # docs: after create, before initialize
            b.initialize(handle)
            b.start(handle)
        except Exception:
            b.destroy(handle)
            raise
        self._handle = handle
        self._started = True
        try:
            b.register_carina_callbacks(handle, on_imu=self._on_imu, on_stereo=self._on_stereo)
        except sdk.SdkError as e:
            log.warning(
                "Carina callbacks unavailable (%s); latest_imu() and tracking frames are None", e
            )
        self._boost_timer()
        self._spawn(self._pose_loop, "viture-pose")
        try:
            self._start_camera()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._stop.set()
        self._jpeg_ready.set()
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=2.0)
        self._threads.clear()
        b = self._bindings
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if b is not None:
            if self._cam is not None:
                try:
                    b.camera_stop(self._cam)
                except Exception:  # pragma: no cover - best effort on shutdown
                    log.exception("camera stop failed")
                self._cam = None
            if self._handle is not None:
                for fn in (b.stop, b.shutdown, b.destroy):
                    try:
                        fn(self._handle)
                    except Exception:  # pragma: no cover
                        log.exception("SDK shutdown step failed")
                self._handle = None
        self._bindings = None

    # ---------------------------------------------------------------------------------- data

    def latest_frame(self) -> Frame | None:
        with self._lock:
            return self._frame

    def pose_at(self, t: float) -> Pose | None:
        return self._buffer.pose_at(t)

    def latest_imu(self) -> ImuSample | None:
        with self._lock:
            return self._imu

    @property
    def tracking_stable(self) -> bool:
        """Last SDK pose status (0 = stable)."""
        return self._stable

    def measured_pose_rate_hz(self) -> float:
        """Distinct pose polls per second over the recent window (hardware check: >= 120)."""
        with self._lock:
            ts = list(self._poll_times)
        if len(ts) < 2 or ts[-1] <= ts[0]:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])

    # ------------------------------------------------------------- TrackingCameraSource (W13)

    @property
    def tracking_camera_intrinsics(self) -> CameraIntrinsics:
        """Explicit intrinsics if given, else a pinhole with an ASSUMED 110 deg horizontal FOV at
        the real frame size (640x480 until the first frame). See ``tracking_intrinsics_are_assumed``."""
        if self._tracking_intrinsics is not None:
            return self._tracking_intrinsics
        raw = self._tracking_raw
        size = (raw[1], raw[2]) if raw is not None else ASSUMED_TRACKING_SIZE
        return default_intrinsics(size, ASSUMED_TRACKING_HFOV_DEG)

    @property
    def tracking_camera_to_head_quat(self) -> Quat:
        """Identity placeholder (UNMEASURED); see ``tracking_mounting_is_assumed``."""
        return self._tracking_to_head

    def latest_tracking_frame(self) -> Frame | None:
        """Newest LEFT tracking frame (left0), BGR with the gray replicated, stamped with arrival
        time. The conversion happens here, not on the SDK thread, and is cached per frame."""
        with self._lock:
            raw = self._tracking_raw
            cached = self._tracking_cache
        if raw is None:
            return None
        if cached is not None and cached[0] is raw:
            return cached[1]
        data, w, h, t = raw
        gray = np.frombuffer(data, np.uint8, count=w * h).reshape(h, w)
        frame = Frame(t, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        with self._lock:
            self._tracking_cache = (raw, frame)
        return frame

    def set_tracking_exposure(self, exposure_ms: float | None, gain: float | None = None) -> None:
        """Manual exposure (ms clamped to 0.01-8, gain to 0-15) or None for automatic.

        SDK: ``set_manual_exposure_carina(handle, ms, gain)`` / ``set_auto_exposure_carina(handle)``."""
        b, h = self._require_symbols(
            "set_tracking_exposure",
            "xr_device_provider_set_auto_exposure_carina"
            if exposure_ms is None
            else "xr_device_provider_set_manual_exposure_carina",
        )
        if exposure_ms is None:
            b.set_auto_exposure(h)
            return
        lo, hi = EXPOSURE_MS_RANGE
        ms = min(max(float(exposure_ms), lo), hi)
        g = DEFAULT_MANUAL_GAIN if gain is None else gain
        g = int(min(max(round(g), EXPOSURE_GAIN_RANGE[0]), EXPOSURE_GAIN_RANGE[1]))
        b.set_manual_exposure(h, ms, g)

    # ------------------------------------------------------------------- device controls (W13)

    def _require_symbols(self, what: str, *names: str) -> tuple[sdk.SdkBindings, Any]:
        b, h = self._bindings, self._handle
        if not self._started or b is None or h is None:
            raise sdk.SdkError(f"{what}: the glasses are not started (the SDK needs STARTED state)")
        for n in names:
            if not b.has(n):
                raise sdk.SdkError(f"{what}: this SDK library does not export {n}")
        return b, h

    def set_brightness(self, level: int) -> None:
        """Display brightness, integer 0-8 (Luma series, per the docs)."""
        lo, hi = BRIGHTNESS_RANGE
        if not lo <= int(level) <= hi:
            raise ValueError(f"brightness level must be {lo}-{hi}, got {level}")
        b, h = self._require_symbols("set_brightness", "xr_device_provider_set_brightness_level")
        b.set_brightness_level(h, int(level))

    def get_brightness(self) -> int:
        b, h = self._require_symbols("get_brightness", "xr_device_provider_get_brightness_level")
        return b.get_brightness_level(h)

    def set_film(self, voltage_0_1: float) -> None:
        """Electrochromic lens tint, voltage 0.0 (clear) to 1.0 (darkest). Gen1 film is binary."""
        if not (math.isfinite(voltage_0_1) and 0.0 <= voltage_0_1 <= 1.0):
            raise ValueError(f"film voltage must be within 0.0-1.0, got {voltage_0_1}")
        b, h = self._require_symbols("set_film", "xr_device_provider_set_film_mode")
        b.set_film_mode(h, float(voltage_0_1))

    def get_film(self) -> float:
        b, h = self._require_symbols("get_film", "xr_device_provider_get_film_mode")
        return b.get_film_mode(h)

    # -------------------------------------------------------------------------------- internals

    def _spawn(self, target: Any, name: str) -> None:
        th = threading.Thread(target=target, name=name, daemon=True)
        self._threads.append(th)
        th.start()

    @staticmethod
    def _boost_timer() -> None:
        # Windows' default timer tick is ~15.6 ms, which would cap polling near 64 Hz.
        if sys.platform == "win32":  # pragma: no cover - Windows only
            try:
                import ctypes

                ctypes.windll.winmm.timeBeginPeriod(1)  # type: ignore[attr-defined]
            except Exception:
                log.warning("timeBeginPeriod(1) failed; pose polling may be limited to ~64 Hz")

    def poll_pose_once(self) -> bool:
        """One poll: read the SDK pose, convert, store. Returns False on an SDK error."""
        b, h = self._bindings, self._handle
        if b is None or h is None:
            return False
        try:
            pose, status = b.get_gl_pose_carina(h)
        except sdk.SdkError as e:
            log.debug("pose poll failed: %s", e)
            return False
        t = self._clock()
        self._stable = status == 0
        quat = sdk.carina_gl_to_head_quat(pose[3:7])  # type: ignore[arg-type]
        self._buffer.add(t - self._pose_latency_s, quat)
        with self._lock:
            self._poll_times.append(t)
        return True

    def _pose_loop(self) -> None:
        while not self._stop.is_set():
            self.poll_pose_once()
            time.sleep(self._poll_interval_s)

    def _on_stereo(self, left0: bytes, w: int, h: int, _sdk_ts: float) -> None:
        # SDK thread: store the already-copied bytes and return. Arrival time on the laptop clock.
        raw = (left0, w, h, self._clock())
        with self._lock:
            self._tracking_raw = raw

    def _on_imu(self, vals: tuple[float, ...], _sdk_ts: float) -> None:
        s = ImuSample(self._clock(), vals[0:3], vals[3:6])  # type: ignore[arg-type]
        with self._lock:
            self._imu = s

    # camera ----------------------------------------------------------------------------------

    def _start_camera(self) -> None:
        if self._camera_index is not None:
            cap = _open_capture(self._camera_index)
            if not cap.isOpened():
                raise sdk.SdkError(
                    f"cannot open camera index {self._camera_index}; see list_cameras()"
                )
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
            cap.set(cv2.CAP_PROP_FPS, 30)
            self._cap = cap
            self._spawn(self._capture_loop, "viture-cv-cam")
            return
        b = self._bindings
        assert b is not None
        if not b.has("xr_camera_provider_create", "xr_camera_provider_start"):
            log.warning("SDK has no pass-through camera API; running pose-only (no frames)")
            return
        try:
            vid, pid = b.camera_ids(self._product_id or 0)
            if not vid or not pid:
                log.warning("these glasses report no pass-through camera; running pose-only")
                return
            self._cam = b.camera_start(vid, pid, self._on_jpeg)
        except sdk.SdkError as e:
            log.warning("SDK camera failed (%s); pass camera_index= to use OpenCV", e)
            return
        self._spawn(self._decode_loop, "viture-decode")

    def _capture_loop(self) -> None:
        cap = self._cap
        while not self._stop.is_set() and cap is not None:
            ok, bgr = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            self._publish(bgr, self._clock())

    def _on_jpeg(self, data: bytes, _w: int, _h: int, _ts_ns: int, _seq: int) -> None:
        # SDK thread: keep it short. Arrival time is the laptop-clock stamp (SDK clock unknown).
        self._jpeg = (data, self._clock())
        self._jpeg_ready.set()

    def _decode_loop(self) -> None:
        while not self._stop.is_set():
            self._jpeg_ready.wait(0.5)
            self._jpeg_ready.clear()
            item, self._jpeg = self._jpeg, None
            if item is None:
                continue
            self.process_jpeg(*item)

    def process_jpeg(self, data: bytes, arrival_t: float) -> None:
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            self._publish(bgr, arrival_t)

    def _publish(self, bgr: np.ndarray, arrival_t: float) -> None:
        f = Frame(arrival_t - self._exposure_latency_s, bgr)
        with self._lock:
            self._frame = f
