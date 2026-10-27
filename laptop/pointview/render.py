"""The glasses window: PySide6 painter code for every DrawItem, plus an offscreen path for tests.

Black is see-through on the glasses, so nothing is ever filled with a "background" color: only
strokes and glyphs are drawn. All text is at least ``MIN_TEXT_PX`` pixels high. Display coordinates
are pixels, origin top-left (see interfaces.py).

* ``render_to_image(items, size)`` and ``save_png(...)`` use exactly the same painter code as the
  window (``paint_items``).
* ``QtRenderer`` is the borderless full-screen window (contracts §3.4).

Passthrough and looks (contracts §8). A ``Backdrop`` is warped by its homography into the canvas
and scaled by ``gain``. With the "normal" look the overlay is then painted at full display
resolution; with any other look the backdrop and the overlay are composed at an internal
resolution (960x540 by default), the CPU look from ``pointview.looks`` is applied to that frame,
and the result is upscaled to the display. Labels are re-tinted per look (``tint_color``) so they
read as part of it. With no Backdrop and the normal look, the plain full-resolution QPainter path
is used, as before. The optional GLSL path is not implemented: ``gpu="auto"`` runs the CPU path.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QApplication, QWidget

from pointview import looks as looks_mod
from pointview.interfaces import (
    LOOKS,
    MIN_TEXT_PX,
    Backdrop,
    Brackets,
    Color,
    Crosshair,
    DrawItem,
    EdgeArrow,
    InfoCard,
    LandmarkLabel,
    Look,
    Marker,
    PointerRing,
    StatusCorner,
    Tag,
    Text,
    WireMesh,
)

log = logging.getLogger(__name__)

#: Preferred families; Qt falls back per glyph (for "✓", "±", "·").
FONT_FAMILIES = ["Segoe UI", "Inter", "DejaVu Sans", "FreeSans", "Arial"]
SCREEN_MARGIN = 16.0
#: Resolution the looks run at (contracts §8); the result is upscaled to the display.
DEFAULT_INTERNAL_SIZE = (960, 540)

# Colour transform applied to every pen/brush/text colour while painting (see ``tint_color``).
_active_tint: Callable[[Color], Color] | None = None


# ------------------------------------------------------------------------------------ helpers


def ensure_app() -> QGuiApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app  # type: ignore[return-value]


def make_font(size_px: float, bold: bool = False) -> QFont:
    """A font whose pixel size is never below MIN_TEXT_PX."""
    f = QFont()
    f.setFamilies(FONT_FAMILIES)
    f.setPixelSize(max(MIN_TEXT_PX, round(size_px)))
    f.setBold(bold)
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return f


def qcolor(c: Color, mix_white: float = 0.0, mix_black: float = 0.0) -> QColor:
    r, g, b = c
    if mix_white:
        r, g, b = (round(v + (255 - v) * mix_white) for v in (r, g, b))
    if mix_black:
        r, g, b = (round(v * (1 - mix_black)) for v in (r, g, b))
    if _active_tint is not None:
        r, g, b = _active_tint((r, g, b))
    return QColor(r, g, b)


#: Label colour per look as a multiple of the label's own brightness (RGB, 0..1).
_TINT_RGB: dict[str, tuple[float, float, float]] = {
    "crt": (0.55, 1.0, 0.60),  # phosphor green
    "nvg": (0.16, 1.0, 0.22),  # P43, the same green the NVG look uses
    "flir": (1.0, 1.0, 1.0),  # white-hot
    "noir": (0.95, 0.95, 0.95),
    "snow": (0.90, 0.95, 1.0),
}
_IRONBOW_HOT_RGB = (1.0, 0.91, 0.32)  # ironbow near the hot end


def tint_color(color: Color, look: Look, params: Mapping[str, float] | None = None) -> Color:
    """Re-tint a label colour so it reads as part of ``look``. ``"normal"`` returns it unchanged.

    The tint keeps the colour's brightness (never below 55 %, so labels stay legible) but replaces
    its hue: phosphor green for CRT and NVG, white-hot for FLIR (ironbow hot yellow when the
    palette is ironbow), grey for noir, cool white for snow.
    """
    if look == "normal":
        return color
    r, g, b = color
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    k = 0.55 + 0.45 * lum
    if look == "flir" and float((params or {}).get("palette", 0.0)) >= 0.5:
        mult = _IRONBOW_HOT_RGB
    else:
        mult = _TINT_RGB[look]
    return (round(255 * k * mult[0]), round(255 * k * mult[1]), round(255 * k * mult[2]))


def _pen(color: QColor, width: float) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _draw_text_baseline(p: QPainter, x: float, baseline: float, text: str) -> None:
    p.drawText(QPointF(x, baseline), text)


# ------------------------------------------------------------------------------------- items


def _brackets(p: QPainter, it: Brackets) -> None:
    color = qcolor(it.color, mix_white=0.35 if it.highlighted else 0.0)
    p.setPen(_pen(color, 5.0 if it.highlighted else 3.0))
    p.setBrush(Qt.BrushStyle.NoBrush)
    n = max(12.0, min(40.0, min(it.w, it.h) * 0.25))
    x0, y0, x1, y1 = it.x, it.y, it.x + it.w, it.y + it.h
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        p.drawPolyline([QPointF(cx + dx * n, cy), QPointF(cx, cy), QPointF(cx, cy + dy * n)])


MESH_ALPHA_BUCKETS = 8
MESH_RASTER_MAX_W = 960


def _wire_mesh(p: QPainter, it: WireMesh, size: tuple[int, int]) -> None:
    """Rasterize the segments with OpenCV into one 8-bit coverage mask (antialiased; each segment
    drawn at its alpha), colour it in one numpy pass and composite it once. Far faster than one
    QPainter call per segment. Keep N ≲ 8k (the scene caps it): OpenCV costs ~0.5 µs per line."""
    seg = np.asarray(it.segments, np.float32).reshape(-1, 4)
    if len(seg) == 0:
        return
    dev = p.device()
    dw, dh = (dev.width(), dev.height()) if dev is not None else size
    # Rasterize at most ~960 px wide and let Qt scale it up smoothly: a quarter of the pixels at
    # 1080p, and a faint thin wireframe looks the same.
    scale = min(1.0, MESH_RASTER_MAX_W / dw)
    w, h = max(1, round(dw * scale)), max(1, round(dh * scale))
    sx, sy = w / size[0], h / size[1]
    pts = np.round(seg.reshape(-1, 2, 2) * (sx, sy)).astype(np.int32)
    alpha = (
        np.ones(len(seg), np.float32)
        if it.alpha is None
        else np.clip(np.asarray(it.alpha, np.float32), 0.0, 1.0)
    )
    buckets = np.minimum((alpha * MESH_ALPHA_BUCKETS).astype(np.int32), MESH_ALPHA_BUCKETS - 1)
    order = np.argsort(buckets, kind="stable")  # brighter buckets drawn last, on top
    mask = np.zeros((h, w), np.uint8)
    thickness = max(1, round(it.width_px * sx))
    for b in np.unique(buckets):
        sel = order[buckets[order] == b]
        level = int(255 * (b + 1) / MESH_ALPHA_BUCKETS)
        cv2.polylines(mask, list(pts[sel]), False, level, thickness, cv2.LINE_AA)
    c = qcolor(it.color)
    layer = np.empty((h, w, 4), np.uint8)
    layer[..., 0], layer[..., 1], layer[..., 2] = c.blue(), c.green(), c.red()
    layer[..., 3] = mask  # QImage ARGB32 is BGRA in memory, straight alpha
    img = QImage(layer.data, w, h, 4 * w, QImage.Format.Format_ARGB32)
    p.save()
    p.resetTransform()
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.drawImage(QRectF(0, 0, dw, dh), img)
    p.restore()


def _marker(p: QPainter, it: Marker) -> None:
    color = qcolor(it.color, mix_white=0.35 if it.highlighted else 0.0)
    r = it.size / 2
    if it.shape == "point":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawEllipse(QPointF(it.x, it.y), r, r)
    else:
        p.setPen(_pen(color, 3.0 if it.highlighted else 2.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(it.x - r, it.y - r, it.size, it.size))
    if it.label:
        p.setFont(make_font(MIN_TEXT_PX, bold=it.highlighted))
        p.setPen(color)
        fm = p.fontMetrics()
        _draw_text_baseline(p, it.x + r + 8, it.y + (fm.ascent() - fm.descent()) / 2, it.label)


def _tag(p: QPainter, it: Tag) -> None:
    p.setFont(make_font(it.size_px, bold=it.highlighted))
    p.setPen(qcolor(it.color, mix_white=0.35 if it.highlighted else 0.0))
    fm = p.fontMetrics()
    tw = fm.horizontalAdvance(it.text)
    _draw_text_baseline(p, it.x - tw / 2, it.y - fm.descent(), it.text)


def _landmark_label(p: QPainter, it: LandmarkLabel) -> None:
    color = qcolor(it.color, mix_white=0.35 if it.highlighted else 0.0)
    p.setFont(make_font(it.size_px, bold=it.highlighted))
    p.setPen(color)
    fm = p.fontMetrics()
    tw = fm.horizontalAdvance(it.text)
    _draw_text_baseline(p, it.x - tw / 2, it.y_text, it.text)
    y_from = it.y_text + fm.descent() + 4
    if it.y_target > y_from:
        p.setPen(_pen(color, 2.5 if it.highlighted else 1.5))
        p.drawLine(QPointF(it.x, y_from), QPointF(it.x, it.y_target))
        p.setBrush(color)
        p.drawEllipse(QPointF(it.x, it.y_target), 3.0, 3.0)
        p.setBrush(Qt.BrushStyle.NoBrush)


def _edge_arrow(p: QPainter, it: EdgeArrow, size: tuple[int, int]) -> None:
    color = qcolor(it.color)
    a = math.radians(it.angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    nx, ny = -dy, dx
    tip = QPointF(it.x, it.y)
    shaft = 30.0
    head = 14.0
    tail = QPointF(it.x - dx * shaft, it.y - dy * shaft)
    p.setPen(_pen(color, 3.0))
    p.drawLine(tail, tip)
    base = QPointF(it.x - dx * head, it.y - dy * head)
    p.setBrush(color)
    p.drawPolygon(
        QPolygonF([tip, QPointF(base.x() + nx * 9, base.y() + ny * 9),
                   QPointF(base.x() - nx * 9, base.y() - ny * 9)])
    )  # fmt: skip
    p.setBrush(Qt.BrushStyle.NoBrush)

    # Text sits behind the arrow's tail, i.e. on the inside of the display edge.
    p.setFont(make_font(it.size_px))
    fm = p.fontMetrics()
    tw, th = fm.horizontalAdvance(it.text), fm.height()
    gap = 10.0
    reach = abs(dx) * tw / 2 + abs(dy) * th / 2 + gap
    cx, cy = tail.x() - dx * reach, tail.y() - dy * reach
    left = min(max(cx - tw / 2, SCREEN_MARGIN), max(SCREEN_MARGIN, size[0] - SCREEN_MARGIN - tw))
    top = min(max(cy - th / 2, SCREEN_MARGIN), max(SCREEN_MARGIN, size[1] - SCREEN_MARGIN - th))
    p.setPen(color)
    _draw_text_baseline(p, left, top + fm.ascent(), it.text)


def _pointer_ring(p: QPainter, it: PointerRing) -> None:
    color = qcolor(it.color)
    p.setPen(_pen(color, 3.0))
    p.setBrush(color if it.solid else Qt.BrushStyle.NoBrush)
    p.drawEllipse(QPointF(it.x, it.y), it.radius, it.radius)
    p.setBrush(Qt.BrushStyle.NoBrush)


def _crosshair(p: QPainter, it: Crosshair) -> None:
    p.setPen(_pen(qcolor(it.color), 3.0))
    h = it.size / 2
    gap = min(10.0, h / 3)
    p.drawLine(QPointF(it.x - h, it.y), QPointF(it.x - gap, it.y))
    p.drawLine(QPointF(it.x + gap, it.y), QPointF(it.x + h, it.y))
    p.drawLine(QPointF(it.x, it.y - h), QPointF(it.x, it.y - gap))
    p.drawLine(QPointF(it.x, it.y + gap), QPointF(it.x, it.y + h))


def _info_card(p: QPainter, it: InfoCard, size: tuple[int, int]) -> None:
    body_px = max(MIN_TEXT_PX, it.size_px)
    title_font = make_font(body_px, bold=True)
    body_font = make_font(body_px)
    foot_font = make_font(MIN_TEXT_PX)
    pad, gap = 16.0, 6.0
    tfm, bfm, ffm = QFontMetrics(title_font), QFontMetrics(body_font), QFontMetrics(foot_font)

    rows: list[tuple[QFont, QFontMetrics, str, QColor]] = [
        (title_font, tfm, it.title, qcolor(it.color, mix_white=0.2))
    ]
    rows += [(body_font, bfm, ln, qcolor((235, 235, 235))) for ln in it.lines]
    if it.footer:
        rows.append((foot_font, ffm, it.footer, qcolor(it.color, mix_black=0.25)))
    w = max(fm.horizontalAdvance(t) for _, fm, t, _ in rows) + 2 * pad
    h = sum(fm.height() for _, fm, _, _ in rows) + gap * (len(rows) - 1) + 2 * pad
    x = min(max(it.x, SCREEN_MARGIN), max(SCREEN_MARGIN, size[0] - SCREEN_MARGIN - w))
    y = min(max(it.y, SCREEN_MARGIN), max(SCREEN_MARGIN, size[1] - SCREEN_MARGIN - h))

    p.setPen(_pen(qcolor(it.color), 2.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    path = QPainterPath()
    path.addRoundedRect(QRectF(x, y, w, h), 10, 10)
    p.drawPath(path)
    cy = y + pad
    for font, fm, text, col in rows:
        p.setFont(font)
        p.setPen(col)
        _draw_text_baseline(p, x + pad, cy + fm.ascent(), text)
        cy += fm.height() + gap


def _status_corner(p: QPainter, it: StatusCorner) -> None:
    p.setFont(make_font(it.size_px))
    fm = p.fontMetrics()
    y = SCREEN_MARGIN + 4
    for text, color in it.lines:
        p.setPen(qcolor(color))
        _draw_text_baseline(p, SCREEN_MARGIN + 8, y + fm.ascent(), text)
        y += fm.height() + 2


def _text(p: QPainter, it: Text) -> None:
    p.setFont(make_font(it.size_px))
    p.setPen(qcolor(it.color))
    _draw_text_baseline(p, it.x, it.y + p.fontMetrics().ascent(), it.text)


def paint_items(p: QPainter, items: Sequence[DrawItem], size: tuple[int, int]) -> None:
    """Draw ``items`` (in list order) onto a painter whose target is ``size`` display pixels.
    Does not clear the background."""
    p.setRenderHints(
        QPainter.RenderHint.Antialiasing
        | QPainter.RenderHint.TextAntialiasing
        | QPainter.RenderHint.SmoothPixmapTransform
    )
    for it in items:
        p.save()
        match it:
            case Brackets():
                _brackets(p, it)
            case Tag():
                _tag(p, it)
            case LandmarkLabel():
                _landmark_label(p, it)
            case EdgeArrow():
                _edge_arrow(p, it, size)
            case PointerRing():
                _pointer_ring(p, it)
            case Crosshair():
                _crosshair(p, it)
            case InfoCard():
                _info_card(p, it, size)
            case StatusCorner():
                _status_corner(p, it)
            case Text():
                _text(p, it)
            case WireMesh():
                _wire_mesh(p, it, size)
            case Marker():
                _marker(p, it)
            case Backdrop():
                pass  # drawn by the compositor before the overlay
            case _:
                log.warning("unknown draw item %r", type(it).__name__)
        p.restore()


# --------------------------------------------------------------------------------- offscreen


def _internal_size(size: tuple[int, int], internal: tuple[int, int] | None) -> tuple[int, int]:
    """Largest size with the display's aspect ratio that fits within ``internal`` pixels wide."""
    if internal is None or size[0] <= internal[0]:
        return size
    return internal[0], max(1, round(size[1] * internal[0] / size[0]))


def _warp_backdrop(bd: Backdrop, size: tuple[int, int], out: tuple[int, int]) -> np.ndarray:
    """``bd.bgr`` warped by its image->display homography into an ``out``-sized BGR canvas
    (``size`` is the display size the homography targets), times ``gain``. Black elsewhere."""
    h = np.asarray(bd.homography, dtype=np.float64).reshape(3, 3)
    scale = np.diag([out[0] / size[0], out[1] / size[1], 1.0])
    warped = cv2.warpPerspective(
        np.ascontiguousarray(bd.bgr),
        scale @ h,
        out,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    if bd.gain != 1.0:
        warped = cv2.convertScaleAbs(warped, alpha=max(0.0, float(bd.gain)))
    return warped


def _paint_over(
    bgr: np.ndarray,
    items: Sequence[DrawItem],
    size: tuple[int, int],
    look: Look,
    params: Mapping[str, float] | None,
) -> np.ndarray:
    """Paint ``items`` (display coordinates, tinted for ``look``) over a BGR canvas of any
    resolution; returns the painted BGR canvas."""
    global _active_tint
    ch, cw = bgr.shape[:2]
    buf = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA))
    img = QImage(buf.data, cw, ch, buf.strides[0], QImage.Format.Format_RGB32)
    p = QPainter(img)
    prev = _active_tint
    _active_tint = (lambda c: tint_color(c, look, params)) if look != "normal" else None
    try:
        if (cw, ch) != tuple(size):
            p.scale(cw / size[0], ch / size[1])
        paint_items(p, items, size)
    finally:
        _active_tint = prev
        p.end()
    return cv2.cvtColor(buf, cv2.COLOR_BGRA2BGR)


def compose_frame(
    items: Sequence[DrawItem],
    size: tuple[int, int],
    look: Look = "normal",
    t_s: float = 0.0,
    params: Mapping[str, float] | None = None,
    internal_size: tuple[int, int] | None = DEFAULT_INTERNAL_SIZE,
) -> np.ndarray:
    """The whole frame as a ``size`` BGR uint8 array: backdrop, overlay, then the look."""
    if look not in LOOKS:
        raise ValueError(f"unknown look {look!r}; expected one of {LOOKS}")
    ensure_app()
    backdrop = next((it for it in reversed(items) if isinstance(it, Backdrop)), None)
    if look == "normal":
        # No post-processing: warp at display size so the overlay stays crisp.
        canvas = (
            _warp_backdrop(backdrop, size, size)
            if backdrop is not None
            else np.zeros((size[1], size[0], 3), np.uint8)
        )
        return _paint_over(canvas, items, size, "normal", None)
    iw, ih = _internal_size(size, internal_size)
    canvas = (
        _warp_backdrop(backdrop, size, (iw, ih))
        if backdrop is not None
        else np.zeros((ih, iw, 3), np.uint8)
    )
    frame = _paint_over(canvas, items, size, look, params)
    frame = looks_mod.apply_look(look, frame, t_s, params)
    if (iw, ih) != tuple(size):
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    return frame


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    buf = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA))
    h, w = buf.shape[:2]
    return QImage(buf.data, w, h, buf.strides[0], QImage.Format.Format_RGB32).copy()


def _is_plain(items: Sequence[DrawItem], look: Look) -> bool:
    return look == "normal" and not any(isinstance(it, Backdrop) for it in items)


def render_to_image(
    items: Sequence[DrawItem],
    size: tuple[int, int] = (1920, 1080),
    look: Look = "normal",
    t_s: float = 0.0,
    params: Mapping[str, float] | None = None,
    internal_size: tuple[int, int] | None = DEFAULT_INTERNAL_SIZE,
) -> QImage:
    """Render ``items`` on black, offscreen, with the same code as the window.

    With no Backdrop and ``look="normal"`` this is the plain full-resolution QPainter path.
    """
    ensure_app()
    if _is_plain(items, look):
        img = QImage(size[0], size[1], QImage.Format.Format_RGB32)
        img.fill(QColor(0, 0, 0))
        p = QPainter(img)
        try:
            paint_items(p, items, size)
        finally:
            p.end()
        return img
    return _bgr_to_qimage(compose_frame(items, size, look, t_s, params, internal_size))


def save_png(
    items: Sequence[DrawItem],
    path: str | Path,
    size: tuple[int, int] = (1920, 1080),
    look: Look = "normal",
    t_s: float = 0.0,
    params: Mapping[str, float] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not render_to_image(items, size, look, t_s, params).save(str(path), "PNG"):
        raise OSError(f"could not write {path}")
    return path


# ----------------------------------------------------------------------------------- window


class _Canvas(QWidget):
    def __init__(self, size: tuple[int, int]):
        super().__init__()
        self.logical_size = size
        self.items: tuple[DrawItem, ...] = ()
        self.look: Look = "normal"
        self.look_params: dict[str, float] = {}
        self.internal_size: tuple[int, int] | None = DEFAULT_INTERNAL_SIZE
        self.t0 = time.monotonic()
        self.setWindowTitle("PointView")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.setStyleSheet("background-color: black;")

    def paintEvent(self, event) -> None:  # Qt API name
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0))
        if not _is_plain(self.items, self.look):
            frame = _bgr_to_qimage(
                compose_frame(
                    self.items,
                    self.logical_size,
                    self.look,
                    time.monotonic() - self.t0,
                    self.look_params,
                    self.internal_size,
                )
            )
            p.drawImage(self.rect(), frame)
            p.end()
            return
        lw, lh = self.logical_size
        if self.width() > 0 and self.height() > 0 and (self.width(), self.height()) != (lw, lh):
            p.scale(self.width() / lw, self.height() / lh)
        paint_items(p, self.items, self.logical_size)
        p.end()


def pick_screen(screens, screen_name: str | None, screen_index: int | None, primary):
    """Choose a screen by name substring (case-insensitive) or index; warn and fall back to the
    primary screen."""
    if screen_name:
        needle = screen_name.lower()
        for s in screens:
            hay = " ".join((s.name(), s.manufacturer(), s.model(), s.serialNumber())).lower()
            if needle in hay:
                return s
        log.warning(
            "no screen matching %r (have: %s); using the primary screen",
            screen_name,
            ", ".join(s.name() for s in screens),
        )
    elif screen_index is not None:
        if 0 <= screen_index < len(screens):
            return screens[screen_index]
        log.warning(
            "screen index %d out of range (%d screens); using primary", screen_index, len(screens)
        )
    return primary


class QtRenderer:
    """Borderless full-screen black window on the glasses' display (implements ``Renderer``).

    ``show(items)`` stores the list and repaints; the main loop calls it every frame (60 Hz).
    With ``pump=True`` (default) it also runs ``processEvents()`` so the paint happens even if the
    caller is not running a Qt event loop.
    """

    def __init__(
        self,
        screen_name: str | None = None,
        screen_index: int | None = None,
        size: tuple[int, int] = (1920, 1080),
        *,
        pump: bool = True,
        gpu: str = "auto",
        internal_size: tuple[int, int] | None = DEFAULT_INTERNAL_SIZE,
    ):
        if gpu not in ("auto", "off"):
            raise ValueError(f"gpu must be 'auto' or 'off', not {gpu!r}")
        self._app = ensure_app()
        #: True only if a GLSL path is running. Not implemented, so always False (CPU looks).
        self.gpu_active = False
        if gpu == "auto":
            log.info("GLSL look path not implemented; using the CPU looks")
        self.size = size
        self._pump = pump
        self._closed = False
        screen = pick_screen(
            QGuiApplication.screens(), screen_name, screen_index, QGuiApplication.primaryScreen()
        )
        self.screen = screen
        self._canvas = _Canvas(size)
        self._canvas.internal_size = internal_size
        if screen is not None:
            self._canvas.setScreen(screen)
            self._canvas.setGeometry(screen.geometry())
        self._canvas.showFullScreen()
        self._canvas.setCursor(Qt.CursorShape.BlankCursor)
        if self._pump:
            self._app.processEvents()

    def show(self, items: Sequence[DrawItem]) -> None:
        if self._closed:
            return
        self._canvas.items = tuple(items)
        self._canvas.update()
        if self._pump:
            self._app.processEvents()

    def set_look(self, look: Look, params: Mapping[str, float] | None = None) -> None:
        """Apply a sensor look to everything drawn (Backdrop and overlay); see contracts §8."""
        if look not in LOOKS:
            raise ValueError(f"unknown look {look!r}; expected one of {LOOKS}")
        self._canvas.look = look
        self._canvas.look_params = dict(params or {})
        if not self._closed:
            self._canvas.update()
            if self._pump:
                self._app.processEvents()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._canvas.close()
        self._canvas.deleteLater()
        if self._pump:
            self._app.processEvents()

    def grab(self) -> QImage:
        """Current window contents (for tests)."""
        return self._canvas.grab().toImage()
