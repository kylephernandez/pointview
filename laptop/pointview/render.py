"""The glasses window: PySide6 painter code for every DrawItem, plus an offscreen path for tests.

Black is see-through on the glasses, so nothing is ever filled with a "background" color: only
strokes and glyphs are drawn. All text is at least ``MIN_TEXT_PX`` pixels high. Display coordinates
are pixels, origin top-left (see interfaces.py).

* ``render_to_image(items, size)`` and ``save_png(...)`` use exactly the same painter code as the
  window (``paint_items``).
* ``QtRenderer`` is the borderless full-screen window (contracts §3.4).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path

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

from pointview.interfaces import (
    MIN_TEXT_PX,
    Brackets,
    Color,
    Crosshair,
    DrawItem,
    EdgeArrow,
    InfoCard,
    LandmarkLabel,
    PointerRing,
    StatusCorner,
    Tag,
    Text,
)

log = logging.getLogger(__name__)

#: Preferred families; Qt falls back per glyph (for "✓", "±", "·").
FONT_FAMILIES = ["Segoe UI", "Inter", "DejaVu Sans", "FreeSans", "Arial"]
SCREEN_MARGIN = 16.0


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
    return QColor(r, g, b)


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
    rows += [(body_font, bfm, ln, QColor(235, 235, 235)) for ln in it.lines]
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
            case _:
                log.warning("unknown draw item %r", type(it).__name__)
        p.restore()


# --------------------------------------------------------------------------------- offscreen


def render_to_image(items: Sequence[DrawItem], size: tuple[int, int] = (1920, 1080)) -> QImage:
    """Render ``items`` on black, offscreen, with the same painter code as the window."""
    ensure_app()
    img = QImage(size[0], size[1], QImage.Format.Format_RGB32)
    img.fill(QColor(0, 0, 0))
    p = QPainter(img)
    try:
        paint_items(p, items, size)
    finally:
        p.end()
    return img


def save_png(
    items: Sequence[DrawItem], path: str | Path, size: tuple[int, int] = (1920, 1080)
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not render_to_image(items, size).save(str(path), "PNG"):
        raise OSError(f"could not write {path}")
    return path


# ----------------------------------------------------------------------------------- window


class _Canvas(QWidget):
    def __init__(self, size: tuple[int, int]):
        super().__init__()
        self.logical_size = size
        self.items: tuple[DrawItem, ...] = ()
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
    ):
        self._app = ensure_app()
        self.size = size
        self._pump = pump
        self._closed = False
        screen = pick_screen(
            QGuiApplication.screens(), screen_name, screen_index, QGuiApplication.primaryScreen()
        )
        self.screen = screen
        self._canvas = _Canvas(size)
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
