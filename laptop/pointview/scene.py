"""Scene composition: entities + pointer + selection → DrawItems in display pixels.

Pure function of its inputs (no Qt, no I/O), so it is unit-tested directly. Called once per render frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pointview import world
from pointview.entities import Entity
from pointview.interfaces import (
    CLASS_COLORS,
    POINTER_COLOR,
    STATUS_BAD,
    STATUS_OK,
    STATUS_WARN,
    Alignment,
    Brackets,
    Color,
    Crosshair,
    DrawItem,
    EdgeArrow,
    InfoCard,
    LandmarkLabel,
    PointerRing,
    Quat,
    StatusCorner,
    Tag,
    Text,
)

#: Landmarks this far outside the display's horizontal FOV still get an edge arrow.
EDGE_ARROW_MARGIN_DEG = 35.0
#: Landmark names sit this far above their target direction, in px, plus a stagger per level.
LANDMARK_TEXT_RISE_PX = 110
LANDMARK_LEVEL_PX = 40
LANDMARK_MIN_GAP_PX = 260
EDGE_INSET_PX = 30
CARD_W_PX = 560


@dataclass(frozen=True, slots=True)
class InfoContent:
    """What the info card shows for the selected entity."""

    entity_id: str
    title: str
    lines: tuple[str, ...]
    footer: str | None = None


@dataclass(frozen=True, slots=True)
class Status:
    phone_connected: bool
    phone_stalled: bool = False
    gps_acc_m: float | None = None
    alignment: Alignment = field(default_factory=Alignment)
    #: Seconds since the last Align, for the drift indicator.
    since_align_s: float | None = None
    heading_acc_deg: float | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class SceneInput:
    entities: Sequence[Entity]
    head_quat: Quat
    alignment: Alignment
    display_size: tuple[int, int]
    display_fov_deg: tuple[float, float]
    status: Status
    #: World (yaw, pitch) of the pointer ray, or None when no phone.
    pointer: tuple[float, float] | None = None
    snapped_id: str | None = None
    info: InfoContent | None = None
    show_crosshair: bool = False
    attribution: str | None = None


def format_distance(d: float | None) -> str | None:
    """Rounded like the design doc's "~15 m": 1 m steps under 10 m, 5 m under 100 m, then 10 m / km."""
    if d is None:
        return None
    if d < 10:
        return f"~{max(1, round(d))} m"
    if d < 100:
        return f"~{round(d / 5) * 5} m"
    if d < 1000:
        return f"~{round(d / 10) * 10} m"
    return f"~{d / 1000:.1f} km"


def entity_tag_text(e: Entity) -> str:
    dist = format_distance(e.distance_m)
    return f"{e.label} · {dist}" if dist else e.label


def info_for(e: Entity, summary_lines: Sequence[str] = (), footer: str | None = None) -> InfoContent:
    """Card content per the design doc: vehicles type + distance; people "person" + distance only;
    landmarks name, distance and up to two summary lines."""
    dist = format_distance(e.distance_m)
    if e.kind == "person":
        return InfoContent(e.id, "person", (dist,) if dist else ())
    if e.kind == "vehicle":
        return InfoContent(e.id, e.label, (dist,) if dist else ())
    lines = ((dist,) if dist else ()) + tuple(summary_lines[:2])
    return InfoContent(e.id, e.label, lines, footer)


def _status_items(s: Status) -> StatusCorner:
    lines: list[tuple[str, Color]] = []
    if not s.phone_connected:
        lines.append(("phone: searching", STATUS_BAD))
    elif s.phone_stalled:
        lines.append(("phone: stalled", STATUS_WARN))
    else:
        lines.append(("phone ✓", STATUS_OK))
    if s.gps_acc_m is None:
        lines.append(("GPS: no fix", STATUS_WARN))
    else:
        c = STATUS_OK if s.gps_acc_m <= 15 else STATUS_WARN if s.gps_acc_m <= 50 else STATUS_BAD
        lines.append((f"GPS ±{s.gps_acc_m:.0f} m", c))
    a = s.alignment
    if a.aligned_at is None:
        lines.append(("not aligned: tap Align", STATUS_BAD))
    else:
        age = s.since_align_s or 0.0
        c = STATUS_OK if age < 300 else STATUS_WARN if age < 900 else STATUS_BAD
        north = {"landmark": "north ✓ landmark", "compass": "north: compass", "none": "north: none"}
        north_c = STATUS_OK if a.north_source == "landmark" else STATUS_WARN
        if a.north_source == "compass" and (a.heading_acc_deg or 99) <= 10:
            north_c = STATUS_OK
        lines.append((f"aligned {age / 60:.0f} min ago", c))
        lines.append((north[a.north_source], north_c))
    if s.note:
        lines.append((s.note, STATUS_WARN))
    return StatusCorner(tuple(lines))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def build_scene(inp: SceneInput) -> list[DrawItem]:
    w, h = inp.display_size
    north = inp.alignment.north_offset_deg
    half_h_fov = inp.display_fov_deg[0] / 2
    items: list[DrawItem] = []
    anchors: dict[str, tuple[float, float]] = {}

    landmarks: list[tuple[float, float, Entity]] = []
    for e in inp.entities:
        color = CLASS_COLORS[e.kind]
        hl = e.id == inp.snapped_id
        p = world.project_to_display(
            e.world_yaw_deg, e.world_pitch_deg, inp.head_quat, north, inp.display_size, inp.display_fov_deg
        )
        if e.kind == "landmark":
            if p.in_front and p.on_screen:
                landmarks.append((p.x, p.y, e))
                continue
            alpha, _beta, in_front = world.head_angles_deg(
                e.world_yaw_deg, e.world_pitch_deg, inp.head_quat, north
            )
            # Edge arrows only for landmarks *just* out of view, horizontally.
            if in_front and half_h_fov < abs(alpha) <= half_h_fov + EDGE_ARROW_MARGIN_DEG:
                right = alpha > 0
                y = _clamp(p.y, h * 0.3, h * 0.7)
                items.append(
                    EdgeArrow(
                        w - EDGE_INSET_PX if right else EDGE_INSET_PX,
                        y,
                        0.0 if right else 180.0,
                        e.label,
                        color,
                    )
                )
            continue
        if not (p.in_front and p.on_screen):
            continue
        size = e.size_deg
        bw = world.angular_size_to_px(size.w, inp.display_size, inp.display_fov_deg) if size else 60
        bh = world.angular_size_to_px(size.h, inp.display_size, inp.display_fov_deg) if size else 60
        bw, bh = max(bw, 24.0), max(bh, 24.0)
        items.append(Brackets(p.x - bw / 2, p.y - bh / 2, bw, bh, color, hl))
        items.append(Tag(p.x, p.y - bh / 2 - 8, entity_tag_text(e), color, hl))
        anchors[e.id] = (p.x + bw / 2, p.y - bh / 2)

    # Landmark names above the horizon; stagger neighbours vertically so names don't overlap.
    landmarks.sort(key=lambda t: t[0])
    last_x_at_level: dict[int, float] = {}
    for x, y, e in landmarks:
        level = 0
        while x - last_x_at_level.get(level, -1e9) < LANDMARK_MIN_GAP_PX and level < 4:
            level += 1
        last_x_at_level[level] = x
        y_text = _clamp(y - LANDMARK_TEXT_RISE_PX - level * LANDMARK_LEVEL_PX, 40, h - 40)
        hl = e.id == inp.snapped_id
        dist = format_distance(e.distance_m)
        text = f"{e.label} · {dist}" if dist and hl else e.label
        items.append(LandmarkLabel(x, y_text, y, text, CLASS_COLORS["landmark"], hl))
        anchors[e.id] = (x + 20, y_text)

    if inp.show_crosshair:
        items.append(Crosshair(w / 2, h / 2))

    if inp.info is not None:
        ax, ay = anchors.get(inp.info.entity_id, (w / 2 + 40, h / 2 - 40))
        n_lines = 1 + len(inp.info.lines) + (1 if inp.info.footer else 0)
        card_h = 24 + n_lines * 40
        x = _clamp(ax + 24, 10, w - CARD_W_PX - 10)
        y = _clamp(ay, 10, h - card_h - 10)
        kind_color = next(
            (CLASS_COLORS[e.kind] for e in inp.entities if e.id == inp.info.entity_id), POINTER_COLOR
        )
        items.append(InfoCard(x, y, inp.info.title, inp.info.lines, kind_color, inp.info.footer))

    if inp.pointer is not None:
        p = world.project_to_display(
            inp.pointer[0], inp.pointer[1], inp.head_quat, north, inp.display_size, inp.display_fov_deg
        )
        if p.in_front:
            # Off-screen pointers stick to the display edge so you can find them.
            x, y = _clamp(p.x, 20, w - 20), _clamp(p.y, 20, h - 20)
            items.append(PointerRing(x, y, solid=inp.snapped_id is not None))

    items.append(_status_items(inp.status))
    if inp.attribution and any(e.kind == "landmark" for e in inp.entities):
        items.append(Text(w - 420, h - 40, inp.attribution, (160, 160, 160)))
    return items
