"""Module boundaries for PointView. FROZEN — see docs/contracts.md for the full conventions.

Coordinate frames (all right-handed, all vectors unit length):

* **World W** — x = east, y = true north, z = up. Yaw is clockwise from +y (north), pitch is up from
  the horizon. ``vector = (sin(yaw)cos(pitch), cos(yaw)cos(pitch), sin(pitch))``.
* **Tracking T** — the glasses' gravity-aligned tracking frame. Same axes as W but with an arbitrary
  yaw origin. ``world_yaw = tracking_yaw + Alignment.north_offset_deg``.
* **Head H** — fixed to the glasses: x = right, y = forward (where you look), z = up.
  ``Pose.quat`` rotates H vectors into T. The identity quaternion = looking along tracking +y, level.
* **Camera C** — OpenCV convention: x = right, y = down, z = forward (optical axis).
  ``head = (c.x, c.z, -c.y)`` then rotated by ``Calibration.camera_to_head_quat``.
* **Phone P** — iOS device frame: x = right edge, y = top edge, z = out of the screen.
* **Phone reference R** — gravity-aligned (z up), arbitrary yaw. ``PhoneState.quat`` rotates P vectors
  into R. The pointer is the top edge, P +y.
* **Display** — pixels, origin top-left, x right, y down, size ``GlassesSource.display_size``.

Quaternions are ``(w, x, y, z)`` tuples, Hamilton convention, unit norm.
All laptop times are seconds on the ``now()`` clock (``time.monotonic``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from pointview.entities import Entity, EntityKind, GeoPos

Quat = tuple[float, float, float, float]
Vec3 = tuple[float, float, float]
IDENTITY_QUAT: Quat = (1.0, 0.0, 0.0, 0.0)

#: The laptop clock. Everything with a ``t`` on the laptop uses it. Tests may inject another.
now: Callable[[], float] = time.monotonic
Clock = Callable[[], float]


# --------------------------------------------------------------------------------------- glasses


@dataclass(frozen=True, slots=True)
class Frame:
    #: Capture time on the laptop clock (best estimate of exposure time, not arrival time).
    t: float
    #: H x W x 3, uint8, BGR (OpenCV order). Lives in memory only — never saved (privacy rule).
    bgr: np.ndarray


@dataclass(frozen=True, slots=True)
class Pose:
    t: float
    #: Rotates head-frame vectors into the tracking frame.
    quat: Quat


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics of the front camera, in pixels of the delivered Frame."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    #: OpenCV distortion coefficients (k1, k2, p1, p2[, k3]); empty = undistorted.
    dist: tuple[float, ...] = ()


@runtime_checkable
class GlassesSource(Protocol):
    """The glasses: front camera frames, head pose, and display geometry."""

    @property
    def camera_intrinsics(self) -> CameraIntrinsics: ...

    @property
    def display_size(self) -> tuple[int, int]:
        """(width, height) of the glasses' display in pixels, e.g. (1920, 1080)."""
        ...

    @property
    def display_fov_deg(self) -> tuple[float, float]:
        """(horizontal, vertical) field of view of the display, about (46.0, 27.0)."""
        ...

    def start(self) -> None:
        """Open the device / file and start background capture and pose polling."""
        ...

    def stop(self) -> None: ...

    def latest_frame(self) -> Frame | None:
        """Newest camera frame, or None before the first one. Never blocks."""
        ...

    def pose_at(self, t: float) -> Pose | None:
        """Head pose at laptop time ``t``.

        Slerp-interpolated between the two samples around ``t``. For ``t`` after the newest sample,
        extrapolate with the last angular velocity for at most 50 ms past it, then hold. For ``t``
        before the oldest buffered sample, return the oldest. None only if no pose has arrived.
        The returned ``Pose.t`` equals the requested ``t``.
        """
        ...


@runtime_checkable
class TrackingCameraSource(Protocol):
    """Optional extra on a GlassesSource: the Luma Ultra's side-mounted grayscale tracking cameras
    (VIO cameras, streamed by the SDK's Carina camera callback). Used as the NVG backdrop, since they
    may cope better with low light than the front color camera. Check with isinstance()."""

    @property
    def tracking_camera_intrinsics(self) -> CameraIntrinsics: ...

    @property
    def tracking_camera_to_head_quat(self) -> Quat:
        """Mounting rotation of the (left) tracking camera relative to the head frame."""
        ...

    def latest_tracking_frame(self) -> Frame | None:
        """Newest left tracking-camera frame, H x W x 3 uint8 BGR (gray replicated). Never blocks."""
        ...

    def set_tracking_exposure(self, exposure_ms: float | None, gain: float | None = None) -> None:
        """Manual exposure (SDK range 0.01-8 ms, gain 0-15), or None for automatic."""
        ...


@dataclass(frozen=True, slots=True)
class Calibration:
    """Fixed per-rig calibration, measured once (see docs/hardware-checklist.md)."""

    #: Small rotation from the camera's nominal mounting to the head frame (camera-display offset).
    camera_to_head_quat: Quat = IDENTITY_QUAT


@dataclass(frozen=True, slots=True)
class Alignment:
    """Result of the Align step (design doc, Pointing math step 5)."""

    #: Added to the phone's yaw (in R) to get the pointer's yaw in the tracking frame T.
    phone_to_head_deg: float = 0.0
    #: Added to a tracking-frame yaw to get a world (true north) yaw.
    north_offset_deg: float = 0.0
    #: Laptop time of the last Align; None = never aligned.
    aligned_at: float | None = None
    #: How north was set: compass heading, a known landmark (long-press), or not at all.
    north_source: Literal["none", "compass", "landmark"] = "none"
    #: Compass accuracy (degrees) at the time north was set, if from the compass.
    heading_acc_deg: float | None = None


# ----------------------------------------------------------------------------------------- phone

ButtonName = Literal["info", "align", "align_long"]
LayerName = Literal[
    "people", "vehicles", "landmarks", "aircraft", "satellites", "vessels", "events", "mesh"
]
#: Phone layer toggle → entity kind.
LAYER_KINDS: dict[str, EntityKind] = {
    "people": "person",
    "vehicles": "vehicle",
    "landmarks": "landmark",
    "aircraft": "aircraft",
    "satellites": "satellite",
    "vessels": "vessel",
    "events": "event",
}
#: Phone toggles that are not entity kinds.
EXTRA_TOGGLES: tuple[str, ...] = ("mesh",)
#: God's Eye View sensor looks (contracts §8). "normal" = no post-processing.
Look = Literal["normal", "crt", "nvg", "flir", "noir", "snow"]
LOOKS: tuple[Look, ...] = ("normal", "crt", "nvg", "flir", "noir", "snow")


@dataclass(frozen=True, slots=True)
class ButtonEvent:
    name: ButtonName
    #: Phone clock, ms.
    t_ms: float
    #: Mapped onto the laptop clock.
    t: float


@dataclass(frozen=True, slots=True)
class Fiducial:
    """What the phone reports about the marker it shows in Track mode."""

    id: int
    #: Marker side (black border included, quiet zone excluded) in millimeters on the screen.
    side_mm: float


@dataclass(frozen=True, slots=True)
class PhoneState:
    """One validated phone sample (docs/phone-protocol.md), plus laptop-side timing."""

    #: Phone clock, ms since epoch, as sent.
    t_ms: float
    #: The sample time mapped onto the laptop clock (phone_link estimates the clock offset).
    t: float
    #: Laptop time the message arrived.
    received_at: float
    #: Rotates phone-frame vectors into the phone reference frame R.
    quat: Quat
    #: Rotation rate, rad/s, in the phone frame P (x, y, z).
    omega: Vec3
    #: True heading of the phone's top edge, degrees clockwise from north; None if unavailable.
    heading_deg: float | None
    heading_acc: float | None
    lat: float | None
    lon: float | None
    #: Horizontal GPS accuracy, meters.
    gps_acc: float | None
    #: Buttons pressed since the previous message, in press order.
    buttons: tuple[ButtonName, ...] = ()
    #: Layer toggles as last reported by the phone.
    layers: dict[LayerName, bool] = field(
        default_factory=lambda: dict.fromkeys((*LAYER_KINDS, *EXTRA_TOGGLES), True)
    )
    #: How tracked objects are drawn.
    marker: MarkerShape = "square"
    #: The tracking marker the phone is showing, if it is in Track mode (contracts §12).
    fiducial: Fiducial | None = None
    #: Sensor look chosen on the phone (persists when a message omits it).
    look: Look = "normal"
    #: Show the camera image behind the overlay (needed for looks to be visible).
    passthrough: bool = False


class PhoneLinkAPI(Protocol):
    """What the main loop needs from phone_link.PhoneLink."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def latest(self) -> PhoneState | None: ...

    def history(self, seconds: float = 1.0) -> list[PhoneState]:
        """Recent samples, oldest first."""
        ...

    def pop_buttons(self) -> list[ButtonEvent]:
        """Button presses received since the last call, oldest first. Each press is returned once."""
        ...

    def connected(self) -> bool: ...

    def stalled(self) -> bool:
        """Connected, but no sample for more than 50 ms."""
        ...

    async def send(self, message: dict) -> None:
        """Send a laptop→phone message (haptic, status) to the connected phone, if any."""
        ...


# ---------------------------------------------------------------------------------------- layers


@dataclass(frozen=True, slots=True)
class View:
    """The glasses' current view. Use world.in_view() to test a direction against it."""

    head_quat: Quat
    north_offset_deg: float
    display_fov_deg: tuple[float, float]
    #: World direction of the view center.
    yaw_deg: float
    pitch_deg: float


@dataclass(frozen=True, slots=True)
class LayerContext:
    t: float
    glasses: GlassesSource
    alignment: Alignment
    calibration: Calibration
    phone: PhoneState | None
    #: Kinds the user has switched on (from the phone's toggles).
    enabled: frozenset[EntityKind] = frozenset(LAYER_KINDS.values())
    #: Wall clock, Unix seconds (UTC). Live layers use it for propagation (orbits, dead reckoning);
    #: the mock pins it to the fixtures' recording time.
    unix_time: float = 0.0
    #: Where the glasses look right now; None before the first head pose.
    view: View | None = None
    #: World (yaw, pitch) of the phone pointer, or None (no phone / not aligned).
    pointer: tuple[float, float] | None = None
    #: Id of the entity the pointer is snapped to. Layers do per-item enrichment (routes, next
    #: pass, details) only for this one.
    focus_id: str | None = None


@runtime_checkable
class Layer(Protocol):
    """A data layer. update() is called every render frame (60 Hz) and must return in < 5 ms;
    heavy work (inference, HTTP) runs on the layer's own thread and update() returns its latest result.
    """

    name: str

    def update(self, ctx: LayerContext) -> list[Entity]: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- phone visual tracking


@dataclass(frozen=True, slots=True)
class PhoneVisualFix:
    """The phone seen by the glasses' front camera (contracts §12).

    The phone shows an ArUco marker (DICT_4X4_50) drawn upright on its screen, so the marker's
    axes equal the phone frame P (x right edge, y top edge, z out of the screen)."""

    #: Laptop time of the camera frame.
    t: float
    #: Rotates phone-frame vectors into the head frame H (camera → head already applied).
    head_from_phone: Quat
    #: Marker center in the head frame, meters (approximate: depends on the screen's real size).
    position_head_m: Vec3
    #: 0..1: how well the camera-seen rotation matches the phone's own gyro (identity check).
    identity_score: float
    #: True once the identity check passed; only confirmed fixes may change the alignment.
    confirmed: bool
    marker_id: int


@runtime_checkable
class PhoneTracker(Protocol):
    """Finds the user's phone in the glasses' camera. Never blocks: works on its own thread."""

    def update(
        self, glasses: GlassesSource, phone_history: Sequence[PhoneState], marker_id: int | None
    ) -> PhoneVisualFix | None:
        """Hand over the latest inputs; returns the newest fix (≤ 0.5 s old) or None."""
        ...

    def close(self) -> None: ...


# ------------------------------------------------------------------------------------ world mesh


@dataclass(frozen=True, slots=True, eq=False)
class WorldMesh:
    """A sparse 3D model of the surroundings (contracts §11).

    Vertices are local East-North-Up meters relative to ``origin`` (ENU at the origin's ellipsoid
    height). ``edges`` index pairs of vertices. Built from the most recent model available (network
    or disk cache) and kept until a newer one replaces it."""

    origin: GeoPos
    vertices: np.ndarray  # float32 (N, 3) east, north, up
    edges: np.ndarray  # int32 (M, 2)
    #: e.g. "osm-buildings+terrarium", for attribution and the status corner.
    source: str
    #: Unix time the underlying data was fetched.
    fetched_at: float
    attribution: str = ""


@runtime_checkable
class MeshSource(Protocol):
    """Provides the WorldMesh around the user. Never blocks: fetches on its own thread."""

    def mesh_near(self, lat: float, lon: float) -> WorldMesh | None:
        """The most recent mesh covering this position (cached or fresh), or None if none yet.
        Calling it also tells the source where the user is (it refetches after moving ~200 m)."""
        ...

    def ground_alt_m(self, lat: float, lon: float) -> float | None:
        """Terrain height (ellipsoidal, m) at this position, if known. Eye height is added by the
        caller."""
        ...

    def close(self) -> None: ...


# ------------------------------------------------------------------------------------- rendering

Color = tuple[int, int, int]  # RGB 0-255

#: One bright color per class (design doc). Black is transparent on the glasses.
CLASS_COLORS: dict[EntityKind, Color] = {
    "person": (255, 200, 0),  # amber
    "vehicle": (0, 220, 255),  # cyan
    "landmark": (140, 255, 120),  # green
    "aircraft": (255, 255, 255),  # white
    "satellite": (190, 160, 255),  # lavender
    "vessel": (80, 170, 255),  # blue
    "event": (255, 90, 70),  # red
}
POINTER_COLOR: Color = (255, 255, 255)
STATUS_OK: Color = (120, 255, 120)
STATUS_WARN: Color = (255, 200, 0)
STATUS_BAD: Color = (255, 80, 80)
MIN_TEXT_PX = 24


@dataclass(frozen=True, slots=True)
class Brackets:
    """Corner brackets around a detection. (x, y) is the top-left corner, display px."""

    x: float
    y: float
    w: float
    h: float
    color: Color
    highlighted: bool = False


@dataclass(frozen=True, slots=True)
class Tag:
    """A text tag like "car · ~15 m". (x, y) is the bottom-center of the text box."""

    x: float
    y: float
    text: str
    color: Color
    highlighted: bool = False
    size_px: int = 28


@dataclass(frozen=True, slots=True)
class LandmarkLabel:
    """Name centered on x with its baseline at y_text, and a thin line down to (x, y_target)."""

    x: float
    y_text: float
    y_target: float
    text: str
    color: Color
    highlighted: bool = False
    size_px: int = 28


@dataclass(frozen=True, slots=True)
class EdgeArrow:
    """An arrow at the display edge for a landmark just out of view. The tip is at (x, y).
    angle_deg is the pointing direction in display coordinates: 0 = right, 90 = down."""

    x: float
    y: float
    angle_deg: float
    text: str
    color: Color
    size_px: int = 24


@dataclass(frozen=True, slots=True)
class PointerRing:
    x: float
    y: float
    radius: float = 18.0
    #: Hollow while free, solid when snapped.
    solid: bool = False
    color: Color = POINTER_COLOR


@dataclass(frozen=True, slots=True)
class Crosshair:
    """The Align crosshair (and the M0 world-fixed test marker)."""

    x: float
    y: float
    size: float = 60.0
    color: Color = POINTER_COLOR


@dataclass(frozen=True, slots=True)
class InfoCard:
    """Card next to the selected item. (x, y) is the top-left corner."""

    x: float
    y: float
    title: str
    lines: tuple[str, ...]
    color: Color
    #: Source / license line, e.g. "Wikipedia · CC BY-SA".
    footer: str | None = None
    size_px: int = 28


@dataclass(frozen=True, slots=True)
class StatusCorner:
    """Status lines drawn top-left, e.g. ("phone ✓", STATUS_OK), ("GPS ±8 m", STATUS_OK)."""

    lines: tuple[tuple[str, Color], ...]
    size_px: int = 24


@dataclass(frozen=True, slots=True)
class Text:
    """Plain text; (x, y) is the top-left of the text box. Used for the OSM attribution."""

    x: float
    y: float
    text: str
    color: Color
    size_px: int = 24


@dataclass(frozen=True, slots=True, eq=False)
class Backdrop:
    """A camera image drawn first, behind everything else (passthrough).

    ``homography`` (row-major 3x3) maps image pixels to display pixels; world.backdrop_homography()
    computes it so the image lines up with the real world. Black stays see-through on the glasses.
    """

    bgr: np.ndarray
    homography: tuple[float, ...]
    #: 0..1 brightness multiplier (night use).
    gain: float = 1.0


MarkerShape = Literal["point", "square"]


@dataclass(frozen=True, slots=True)
class Marker:
    """A tracked object drawn as a point or a square (God's Eye View detection overlay style).
    (x, y) is the center; ``label`` is drawn next to it when given."""

    x: float
    y: float
    shape: MarkerShape
    color: Color
    size: float = 14.0
    label: str | None = None
    highlighted: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class WireMesh:
    """Sparse world wireframe in display pixels, drawn right after the Backdrop.

    ``segments`` is float32 (N, 4): x0, y0, x1, y1, already clipped to the display.
    ``alpha`` is float32 (N,) 0..1 per segment (distance fade), or None for uniform."""

    segments: np.ndarray
    color: Color
    alpha: np.ndarray | None = None
    width_px: float = 1.0


DrawItem = (
    Backdrop
    | WireMesh
    | Marker
    | Brackets
    | Tag
    | LandmarkLabel
    | EdgeArrow
    | PointerRing
    | Crosshair
    | InfoCard
    | StatusCorner
    | Text
)


class Renderer(Protocol):
    """Draws a list of DrawItems on a black (= transparent) background, in display pixels."""

    def show(self, items: Sequence[DrawItem]) -> None:
        """Replace what is on screen with ``items`` (drawn in list order)."""
        ...

    def set_look(self, look: Look, params: Mapping[str, float] | None = None) -> None:
        """Apply a God's Eye View sensor look to everything drawn (Backdrop and overlay)."""
        ...

    def close(self) -> None: ...
