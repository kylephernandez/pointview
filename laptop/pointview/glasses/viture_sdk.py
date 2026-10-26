"""ctypes bindings for the VITURE Glasses SDK (C API, Windows x86_64 ``libglasses.dll``).

SOURCES (read 2026-10-26; this is everything that was available)
    https://www.viture.com/developer/glasses-sdk/glasses
        "VITURE XR Glasses SDK" documentation page: function list, callbacks, pose formats, camera
        streaming. The page carries NO download link and no public repository is linked from it.
        Only the page's first ~100k of 142k characters could be read through the fetch tool.
    https://pub.dev/documentation/viture_kit/latest/  (community Dart FFI wrapper; confirms Windows
        support and the ``libglasses`` naming, nothing more was used).
The SDK binary/headers were NOT obtained, so nothing here was run against the real DLL.

WHAT THE DOCS STATE (used below)
    * Library ``libglasses.dll`` (Windows); headers viture_glasses_provider.h, viture_device_carina.h,
      viture_camera_provider.h, viture_result.h. USB vendor id 0x35CA.
    * Lifecycle ``xr_device_provider_create(int product_id) -> XRDeviceProviderHandle`` (void*),
      ``..._initialize(handle, const char* custom_config, const char* cache_file_dir)`` (page passes
      NULL, NULL), ``..._start``, ``..._stop``, ``..._shutdown``, ``..._destroy``; all int status
      (VITURE_GLASSES_SUCCESS = 0) except destroy (void).
    * Luma Ultra is a "Carina" device. Pose is polled:
      ``int xr_device_provider_get_gl_pose_carina(handle, float* pose, double predict_time, int* status)``
      with ``pose = [px, py, pz, qw, qx, qy, qz]``, OpenGL frame (X right, Y up, Z backward),
      position in metres; ``status`` 0 = stable, 1 = unstable. The docs' own loop sleeps 8 ms and
      passes predict_time 0.0.
    * ``xr_device_provider_set_dof_type_carina(handle, int is_6dof)`` is called after create and
      before initialize.
    * ``xr_device_provider_register_callbacks_carina(handle, XRPoseCallback, XRVSyncCallback,
      XRImuCallback, XRCameraCallback)`` with ``void (*XRPoseCallback)(float* pose, double ts)``,
      ``void (*XRVSyncCallback)(double ts)``, ``void (*XRImuCallback)(float* imu, double ts)``
      (carina raw IMU layout ``[ax, ay, az, gx, gy, gz]``) and
      ``void (*XRCameraCallback)(char* l0, char* r0, char* l1, char* r1, double ts, int w, int h)``
      (stereo tracking cameras, not the pass-through camera).
    * Pass-through camera: ``xr_camera_provider_get_camera_vid/pid(int glasses_pid)``,
      ``xr_camera_provider_create(int vid, int pid) -> handle``, ``xr_camera_provider_start(cam,
      XRCameraCallback2, void* user_data)``, ``..._stop``, ``..._destroy``; callback
      ``void (*)(const XRCameraFrame*, void*)``; frame = 1920x1080 MJPEG @ 30 fps. The struct is
      ``{uint8_t* data; uint32_t size, width, height; XRCameraFormat format; uint64_t timestamp
      /* ns */; uint32_t sequence;}``. Camera USB id VID 0x0C45, PID 0x636B for Luma Ultra.

UNVERIFIED / NOT DOCUMENTED (flagged again at each use)
    * The numeric glasses USB product id of the Luma Ultra (the page lists none): it must be
      passed in (``product_id=`` or env ``VITURE_PRODUCT_ID``).
    * ``xr_device_provider_create`` is shown overloaded (with and without a file descriptor, the
      latter for Android). C has no overloads; only the one-int-argument form is assumed to be
      exported under that name on Windows.
    * sizeof(XRCameraFormat): assumed a C ``enum`` = 4 bytes (``c_int``); the struct layout (and
      therefore every field after ``format``) depends on this.
    * Whether NULL callbacks are accepted by ``register_callbacks_carina``; we pass no-op callbacks
      instead. Whether the pose is device->world or world->device: see ``carina_gl_to_head_quat``.
    * The IMU unit (m/s^2 and rad/s? g and deg/s?) and the clock of every SDK timestamp.
    * Calling convention (cdecl assumed: ``ctypes.CDLL``), DLL search path, dependent DLLs.
    * Gen1/Gen2 glasses (pose callback ``[roll, pitch, yaw, qw, qx, qy, qz]`` North-West-Up) are
      NOT bound: ``xr_device_provider_register_imu_pose_callback``'s prototype is not documented.
"""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pointview.glasses import _quat as q
from pointview.interfaces import Quat

log = logging.getLogger(__name__)

DLL_NAME = "libglasses.dll"
#: Names the SDK DLL may have. The Windows SDK ships ``glasses.lib``, whose DLL is ``glasses.dll``.
DLL_NAMES = ("glasses.dll", "libglasses.dll")
VITURE_USB_VID = 0x35CA  # documented
VITURE_GLASSES_SUCCESS = 0  # documented


class SdkError(RuntimeError):
    """A VITURE SDK call failed or the SDK is unavailable."""


class XRCameraFrame(ctypes.Structure):
    _fields_ = (
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("size", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("format", ctypes.c_int),  # ASSUMPTION: enum == int
        ("timestamp", ctypes.c_uint64),  # nanoseconds
        ("sequence", ctypes.c_uint32),
    )


# Callback prototypes (cdecl). CFUNCTYPE works on Linux too, so parsing is testable off-Windows.
XRPoseCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.c_double)
XRVSyncCallback = ctypes.CFUNCTYPE(None, ctypes.c_double)
XRImuCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.c_double)
XRStereoCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_int,
)
XRCameraFrameCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(XRCameraFrame), ctypes.c_void_p)

_P = ctypes.c_void_p
_I = ctypes.c_int

#: name -> (restype, argtypes). Only functions this module calls.
_SIGNATURES: dict[str, tuple[Any, tuple[Any, ...]]] = {
    "xr_device_provider_create": (_P, (_I,)),
    "xr_device_provider_initialize": (_I, (_P, ctypes.c_char_p, ctypes.c_char_p)),
    "xr_device_provider_start": (_I, (_P,)),
    "xr_device_provider_stop": (_I, (_P,)),
    "xr_device_provider_shutdown": (_I, (_P,)),
    "xr_device_provider_destroy": (None, (_P,)),
    "xr_device_provider_get_device_type": (_I, (_P,)),
    "xr_device_provider_is_product_id_valid": (_I, (_I,)),
    "xr_device_provider_set_dof_type_carina": (_I, (_P, _I)),
    "xr_device_provider_get_gl_pose_carina": (
        _I,
        (_P, ctypes.POINTER(ctypes.c_float), ctypes.c_double, ctypes.POINTER(_I)),
    ),
    "xr_device_provider_register_callbacks_carina": (
        _I,
        (_P, XRPoseCallback, XRVSyncCallback, XRImuCallback, XRStereoCallback),
    ),
    "xr_camera_provider_get_camera_vid": (_I, (_I,)),
    "xr_camera_provider_get_camera_pid": (_I, (_I,)),
    "xr_camera_provider_create": (_P, (_I, _I)),
    "xr_camera_provider_start": (_I, (_P, XRCameraFrameCallback, _P)),
    "xr_camera_provider_stop": (_I, (_P,)),
    "xr_camera_provider_destroy": (None, (_P,)),
}


def load_library(sdk_path: Path | None = None) -> Any:
    """Load ``libglasses.dll`` lazily. Raises :class:`SdkError` with a clear message elsewhere."""
    if os.name != "nt":
        raise SdkError(
            "The VITURE SDK binding targets Windows x86_64 (libglasses.dll); this OS is "
            f"{os.name!r}. Use MockGlasses off Windows."
        )
    root: Path | None = None
    if sdk_path is not None:
        root = Path(sdk_path)
    elif os.environ.get("VITURE_SDK_PATH"):
        root = Path(os.environ["VITURE_SDK_PATH"])
    candidates: list[Path | str] = []
    if root is not None and root.is_file():
        candidates.append(root)
    elif root is not None and root.is_dir():
        # The DLL may sit in a subfolder (bin/, lib/x64/ ...): search the whole SDK folder.
        candidates += [f for name in DLL_NAMES for f in sorted(root.rglob(name))]
    candidates += list(DLL_NAMES)  # system / PATH / application directory
    errors = []
    for c in candidates:
        try:
            if isinstance(c, Path) and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(c.parent))  # its dependent DLLs live next to it
            return ctypes.CDLL(str(c))
        except OSError as e:
            errors.append(f"{c}: {e}")
    hint = ""
    if root is not None and root.is_dir() and not any(isinstance(c, Path) for c in candidates):
        libs = sorted(str(f.relative_to(root)) for f in root.rglob("*.lib"))
        if libs:
            hint = (
                f" The SDK folder has {', '.join(libs)} but no {' / '.join(DLL_NAMES)}: run "
                "tools/inspect_sdk_lib.py on the .lib to see whether it is an import library "
                "(then the DLL is missing from the download) or a static library (then a small "
                "C++ wrapper DLL is needed)."
            )
    raise SdkError(
        f"Could not load the VITURE SDK ({' / '.join(DLL_NAMES)}). Download the Windows x86_64 "
        "SDK from https://www.viture.com/developer/glasses-sdk/glasses and pass sdk_path= or set "
        "VITURE_SDK_PATH to its folder. Tried: " + "; ".join(errors) + hint
    )


#: Result codes from VITURE's SDK documentation.
RESULT_CODES = {
    0: "success",
    -1: "invalid parameter",
    -2: "USB not connected or not accessible",
    -3: "USB read/write failed",
    -4: "not supported on this device",
    -5: "no response from the device",
}
USB_UNAVAILABLE_HINT = (
    " Check: (1) the USB driver is installed: VITURE's docs say it comes with 'SpaceWalker for "
    "Windows'; (2) no other app (SpaceWalker, the VITURE web updater) is using the glasses: quit it; "
    "(3) the glasses are plugged straight into the laptop's DP Alt Mode USB-C port, not a hub or "
    "dock, and the display works; (4) VITURE_PRODUCT_ID matches the glasses' PID_xxxx."
)


class SdkBindings:
    """Thin typed wrapper over a loaded library object (a ``ctypes.CDLL`` or a test fake).

    Callbacks handed to the library are kept referenced here for the lifetime of the object.
    """

    def __init__(self, lib: Any) -> None:
        self._lib = lib
        self._keepalive: list[Any] = []
        self._missing: set[str] = set()
        for name, (restype, argtypes) in _SIGNATURES.items():
            try:
                fn = getattr(lib, name)
            except AttributeError:
                self._missing.add(name)
                continue
            try:
                fn.restype = restype
                fn.argtypes = list(argtypes)
            except AttributeError:  # e.g. a plain bound method in a test fake
                pass

    def has(self, *names: str) -> bool:
        return not any(n in self._missing for n in names)

    def _fn(self, name: str) -> Callable[..., Any]:
        if name in self._missing:
            raise SdkError(f"SDK library does not export {name}")
        return getattr(self._lib, name)

    def _check(self, name: str, rc: int) -> int:
        if rc != VITURE_GLASSES_SUCCESS:
            meaning = RESULT_CODES.get(rc, "unknown")
            hint = USB_UNAVAILABLE_HINT if rc == -2 else ""
            raise SdkError(f"{name} failed with VITURE result code {rc} ({meaning}).{hint}")
        return rc

    def product_id_valid(self, product_id: int) -> bool | None:
        """The SDK's own check; None if this SDK build does not export it."""
        if not self.has("xr_device_provider_is_product_id_valid"):
            return None
        return bool(self._fn("xr_device_provider_is_product_id_valid")(product_id))

    # --- device provider -------------------------------------------------------------------

    def create(self, product_id: int) -> Any:
        handle = self._fn("xr_device_provider_create")(product_id)
        if not handle:
            raise SdkError(f"xr_device_provider_create({product_id:#x}) returned NULL")
        return handle

    def set_dof_type(self, handle: Any, six_dof: bool) -> None:
        self._check(
            "set_dof_type_carina",
            self._fn("xr_device_provider_set_dof_type_carina")(handle, 1 if six_dof else 0),
        )

    def initialize(self, handle: Any) -> None:
        self._check("initialize", self._fn("xr_device_provider_initialize")(handle, None, None))

    def start(self, handle: Any) -> None:
        self._check("start", self._fn("xr_device_provider_start")(handle))

    def stop(self, handle: Any) -> None:
        self._fn("xr_device_provider_stop")(handle)

    def shutdown(self, handle: Any) -> None:
        self._fn("xr_device_provider_shutdown")(handle)

    def destroy(self, handle: Any) -> None:
        self._fn("xr_device_provider_destroy")(handle)

    def get_gl_pose_carina(
        self, handle: Any, buf: Any = None, predict_time: float = 0.0
    ) -> tuple[tuple[float, ...], int]:
        """Poll the Carina pose. Returns (``[px, py, pz, qw, qx, qy, qz]``, status 0 stable/1 not)."""
        pose = buf if buf is not None else (ctypes.c_float * 7)()
        status = _I(0)
        rc = self._fn("xr_device_provider_get_gl_pose_carina")(
            handle, pose, predict_time, ctypes.pointer(status)
        )
        self._check("get_gl_pose_carina", rc)
        return tuple(float(pose[i]) for i in range(7)), int(status.value)

    def register_imu_callback(
        self, handle: Any, on_imu: Callable[[tuple[float, ...], float], None]
    ) -> None:
        """Register an IMU callback (``[ax, ay, az, gx, gy, gz]``, units UNVERIFIED).

        The other three callbacks are no-ops: whether the SDK accepts NULL is undocumented."""

        def _imu(ptr: Any, ts: float) -> None:
            on_imu(tuple(float(ptr[i]) for i in range(6)), float(ts))

        cbs = (
            XRPoseCallback(lambda _p, _t: None),
            XRVSyncCallback(lambda _t: None),
            XRImuCallback(_imu),
            XRStereoCallback(lambda *_a: None),
        )
        self._keepalive.extend(cbs)
        self._check(
            "register_callbacks_carina",
            self._fn("xr_device_provider_register_callbacks_carina")(handle, *cbs),
        )

    # --- pass-through camera ---------------------------------------------------------------

    def camera_ids(self, product_id: int) -> tuple[int, int]:
        vid = self._fn("xr_camera_provider_get_camera_vid")(product_id)
        pid = self._fn("xr_camera_provider_get_camera_pid")(product_id)
        return int(vid), int(pid)

    def camera_start(
        self, vid: int, pid: int, on_jpeg: Callable[[bytes, int, int, int, int], None]
    ) -> Any:
        """Start the pass-through camera. ``on_jpeg(data, width, height, timestamp_ns, sequence)``
        is called on an SDK thread with a COPY of the MJPEG bytes."""
        cam = self._fn("xr_camera_provider_create")(vid, pid)
        if not cam:
            raise SdkError("xr_camera_provider_create returned NULL")

        def _frame(fp: Any, _user: Any) -> None:
            f = fp.contents
            if not f.data or f.size == 0:
                return
            on_jpeg(
                ctypes.string_at(f.data, f.size),
                int(f.width),
                int(f.height),
                int(f.timestamp),
                int(f.sequence),
            )

        cb = XRCameraFrameCallback(_frame)
        self._keepalive.append(cb)
        rc = self._fn("xr_camera_provider_start")(cam, cb, None)
        if rc != VITURE_GLASSES_SUCCESS:
            self._fn("xr_camera_provider_destroy")(cam)
            raise SdkError(f"xr_camera_provider_start failed with VITURE result code {rc}")
        return cam

    def camera_stop(self, cam: Any) -> None:
        self._fn("xr_camera_provider_stop")(cam)
        self._fn("xr_camera_provider_destroy")(cam)


# ------------------------------------------------------------------------------ frame conversion

#: Set True if hardware shows the SDK quaternion is world->device (the inverse of what we assume).
CARINA_POSE_IS_WORLD_TO_DEVICE = False

# Rotation M taking OpenGL axes (x right, y up, z backward) to PointView axes (x right/east,
# y forward/north, z up):  x' = x,  y' = -z,  z' = y.   M = Rx(+90 deg), quaternion below.
_M: Quat = (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)


def carina_gl_to_head_quat(sdk_qwxyz: tuple[float, float, float, float]) -> Quat:
    """Convert a Carina pose quaternion ``(qw, qx, qy, qz)`` to ``Pose.quat`` (head H -> tracking T).

    ASSUMPTIONS (to verify on hardware, see the hardware checklist):
      1. The SDK pose is an OpenGL-frame rotation taking device vectors to the SDK's tracking
         vectors (device->world), both frames OpenGL: x right, y up, z backward, in which "level and
         looking along -z" is the identity. (Docs give the frame but not the direction of the
         rotation; flip ``CARINA_POSE_IS_WORLD_TO_DEVICE`` if turning the head right makes yaw
         decrease.)
      2. The tracking frame is gravity-aligned with y up (VIO origin), so after relabelling axes it
         is PointView's T (z up, yaw origin arbitrary = wherever tracking started).
      3. The device frame's "forward" is -z (OpenGL camera convention), "right" is +x, "up" is +y.

    With M the fixed relabelling OpenGL -> (right, forward, up), the same on both sides:
    ``q_HT = M * q_gl * M^-1``. Identity maps to identity (looking along T +y, level), turning the
    head right (clockwise from above) gives a clockwise (compass-sense) yaw increase in T.
    Position is ignored (orientation-only pointing).
    """
    quat = q.normalize(sdk_qwxyz)
    if CARINA_POSE_IS_WORLD_TO_DEVICE:
        quat = q.conj(quat)
    return q.normalize(q.mul(q.mul(_M, quat), q.conj(_M)))
