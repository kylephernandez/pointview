"""Offscreen render tests. PNGs land in laptop/test-output/ for eyeballing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtGui import QFontMetrics, QGuiApplication, QImage

from pointview import render
from pointview.interfaces import (
    CLASS_COLORS,
    LOOKS,
    MIN_TEXT_PX,
    POINTER_COLOR,
    STATUS_BAD,
    STATUS_OK,
    STATUS_WARN,
    Backdrop,
    Brackets,
    Crosshair,
    EdgeArrow,
    InfoCard,
    LandmarkLabel,
    PointerRing,
    StatusCorner,
    Tag,
    Text,
)

OUT = Path(__file__).resolve().parents[1] / "test-output"
SIZE = (1920, 1080)


@pytest.fixture(scope="module", autouse=True)
def _app():
    render.ensure_app()


def arr(img: QImage) -> np.ndarray:
    """H x W x 3 RGB uint8 copy of the image."""
    img = img.convertToFormat(QImage.Format.Format_RGB32)
    buf = np.frombuffer(img.constBits(), dtype=np.uint8)
    a = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, : img.width()]
    return a[:, :, [2, 1, 0]].copy()  # BGRA -> RGB


def px(img: QImage, x: int, y: int) -> tuple[int, int, int]:
    return tuple(int(v) for v in arr(img)[int(y), int(x)])


def bright_bbox(img: QImage, x0=0, y0=0, x1=None, y1=None, thresh=40):
    """Bounding box (l, t, r, b) of non-black pixels inside a region, or None."""
    a = arr(img)
    x1 = a.shape[1] if x1 is None else x1
    y1 = a.shape[0] if y1 is None else y1
    x0, y0 = max(0, x0), max(0, y0)
    mask = a[y0:y1, x0:x1].max(axis=2) > thresh
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min() + x0), int(ys.min() + y0), int(xs.max() + x0), int(ys.max() + y0)


def near_color(img: QImage, cx, cy, color, radius=40, tol=60) -> bool:
    a = arr(img)
    y0, x0 = max(0, int(cy - radius)), max(0, int(cx - radius))
    region = a[y0 : int(cy + radius), x0 : int(cx + radius)].astype(int)
    if region.size == 0:
        return False
    return bool((np.abs(region - np.array(color)).max(axis=2) <= tol).any())


def save(img: QImage, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    assert img.save(str(OUT / name), "PNG")


# ------------------------------------------------------------------------------- per item


def test_empty_scene_is_black():
    img = render.render_to_image([], SIZE)
    assert img.size().width() == 1920 and img.size().height() == 1080
    assert bright_bbox(img) is None
    save(img, "00_empty.png")


def test_brackets_color_and_highlight():
    items = [
        Brackets(200, 200, 300, 200, CLASS_COLORS["vehicle"]),
        Brackets(800, 200, 300, 200, CLASS_COLORS["person"], highlighted=True),
    ]
    img = render.render_to_image(items, SIZE)
    save(img, "01_brackets.png")
    assert near_color(img, 201, 201, CLASS_COLORS["vehicle"], radius=6)
    assert px(img, 350, 300) == (0, 0, 0)  # inside the box is untouched (see-through)
    assert px(img, 950, 300) == (0, 0, 0)
    # highlighted: thicker strokes and brighter than the base color
    plain = bright_bbox(img, 190, 190, 520, 410)
    hi = bright_bbox(img, 790, 190, 1120, 410)
    assert plain and hi

    # thickness: count lit pixels on a row through the top-left corner vertical stroke
    def run(x0, y):
        return sum(1 for x in range(x0 - 8, x0 + 8) if max(px(img, x, y)) > 60)

    assert run(800, 230) > run(200, 230)
    assert max(px(img, 800, 230)) > 0 and px(img, 800, 230)[2] > CLASS_COLORS["person"][2]


def test_tag_text_height_and_position():
    img = render.render_to_image([Tag(600, 500, "car · ~15 m", CLASS_COLORS["vehicle"])], SIZE)
    save(img, "02_tag.png")
    bb = bright_bbox(img)
    assert bb is not None
    left, t, right, b = bb
    assert (left + right) / 2 == pytest.approx(600, abs=12)  # centered on x
    assert b <= 500  # the text box bottom is y
    assert b > 480
    assert near_color(
        img, (left + right) / 2, (t + b) / 2, CLASS_COLORS["vehicle"], radius=60, tol=80
    )


def test_landmark_label_line_down_to_target():
    img = render.render_to_image(
        [LandmarkLabel(960, 300, 560, "Golden Gate Bridge", CLASS_COLORS["landmark"])], SIZE
    )
    save(img, "03_landmark_label.png")
    assert near_color(img, 960, 450, CLASS_COLORS["landmark"], radius=4, tol=90)  # the line
    assert bright_bbox(img, 960 - 200, 200, 960 + 200, 320) is not None  # the text
    assert near_color(img, 960, 560, CLASS_COLORS["landmark"], radius=6, tol=40)  # target dot
    assert px(img, 900, 450) == (0, 0, 0)


def test_edge_arrow_text_inside_display():
    items = [
        EdgeArrow(1900, 540, 0, "Fort Point 12°", CLASS_COLORS["landmark"]),
        EdgeArrow(20, 300, 180, "Palace of Fine Arts", CLASS_COLORS["landmark"]),
    ]
    img = render.render_to_image(items, SIZE)
    save(img, "04_edge_arrows.png")
    right = bright_bbox(img, 1000, 400, 1920, 700)
    left = bright_bbox(img, 0, 200, 900, 420)
    assert right and left
    assert right[2] <= 1920 - 1 and right[2] >= 1890  # arrow tip at the right edge
    assert left[0] >= 0 and left[0] <= 25
    # text lies inside the display, left of the right arrow's tail
    assert bright_bbox(img, 1400, 500, 1860, 580) is not None
    assert bright_bbox(img, 40, 280, 700, 330) is not None  # left text sits right of the left arrow


def test_pointer_ring_hollow_vs_solid():
    img = render.render_to_image(
        [PointerRing(400, 400), PointerRing(800, 400, solid=True)],
        SIZE,
    )
    save(img, "05_pointer_rings.png")
    assert px(img, 400, 400) == (0, 0, 0)  # hollow
    assert near_color(img, 400 + 18, 400, POINTER_COLOR, radius=3)  # but a ring exists
    assert px(img, 800, 400) == POINTER_COLOR  # solid fills its center


def test_crosshair():
    img = render.render_to_image([Crosshair(960, 540)], SIZE)
    save(img, "06_crosshair.png")
    assert px(img, 960 + 20, 540) == POINTER_COLOR
    assert px(img, 960, 540 - 20) == POINTER_COLOR
    assert px(img, 960 + 20, 540 + 20) == (0, 0, 0)
    assert bright_bbox(img) is not None


def test_info_card():
    card = InfoCard(
        1100,
        300,
        "Golden Gate Bridge",
        ("1.8 km · NNW", "Suspension bridge spanning the", "Golden Gate strait."),
        CLASS_COLORS["landmark"],
        footer="Wikipedia · CC BY-SA",
    )
    img = render.render_to_image([card], SIZE)
    save(img, "07_info_card.png")
    bb = bright_bbox(img)
    assert bb is not None
    assert bb[0] >= 1100 - 3 and bb[1] >= 300 - 3
    assert near_color(img, bb[0] + 2, (bb[1] + bb[3]) / 2, CLASS_COLORS["landmark"], radius=6)


def test_info_card_clamped_into_display():
    card = InfoCard(1900, 1070, "Title", ("a line",), CLASS_COLORS["vehicle"])
    img = render.render_to_image([card], SIZE)
    bb = bright_bbox(img)
    assert bb is not None and bb[2] < 1920 - 1 and bb[3] < 1080 - 1


def test_status_corner_top_left():
    img = render.render_to_image(
        [
            StatusCorner(
                (("phone ✓", STATUS_OK), ("GPS ±8 m", STATUS_WARN), ("align ✗", STATUS_BAD))
            )
        ],
        SIZE,
    )
    save(img, "08_status_corner.png")
    bb = bright_bbox(img)
    assert bb is not None
    assert bb[0] < 120 and bb[1] < 80 and bb[2] < 500
    assert near_color(img, 80, 30, STATUS_OK, radius=30, tol=90)


def test_text_item_and_min_size_clamp():
    # asking for 10 px must still give >= 24 px text
    img_small = render.render_to_image([Text(100, 100, "Hxg", (255, 255, 255), size_px=10)], SIZE)
    img_ok = render.render_to_image([Text(100, 100, "Hxg", (255, 255, 255), size_px=24)], SIZE)
    save(img_small, "09_text_min_size.png")
    assert bright_bbox(img_small) == bright_bbox(img_ok)
    _, t, _, b = bright_bbox(img_small)
    assert b - t + 1 >= 22  # ascender-to-descender of 'Hg' at a 24 px font
    assert QFontMetrics(render.make_font(10)).height() >= MIN_TEXT_PX
    assert render.make_font(10).pixelSize() == MIN_TEXT_PX
    assert render.make_font(40).pixelSize() == 40


def test_all_item_sizes_clamped():
    items = [
        Tag(300, 300, "tiny", (255, 255, 255), size_px=8),
        LandmarkLabel(600, 300, 400, "tiny", (255, 255, 255), size_px=8),
        EdgeArrow(900, 300, 0, "tiny", (255, 255, 255), size_px=8),
        InfoCard(1000, 100, "t", ("l",), (255, 255, 255), footer="f", size_px=8),
        StatusCorner((("tiny", STATUS_OK),), size_px=8),
        Text(1500, 800, "tiny", (255, 255, 255), size_px=8),
    ]
    img = render.render_to_image(items, SIZE)
    for region in ((200, 250, 400, 305), (500, 250, 700, 305), (1500, 800, 1700, 850)):
        bb = bright_bbox(img, *region)
        assert bb is not None
        assert bb[3] - bb[1] + 1 >= 14  # x-height-only word "tiny" with descender at 24 px


def test_full_scene_screenshot():
    lm, veh, per = CLASS_COLORS["landmark"], CLASS_COLORS["vehicle"], CLASS_COLORS["person"]
    items = [
        Brackets(300, 560, 260, 170, veh),
        Tag(430, 550, "car · ~15 m", veh),
        Brackets(700, 520, 90, 230, per, highlighted=True),
        Tag(745, 510, "person · ~8 m", per, highlighted=True),
        LandmarkLabel(1250, 330, 520, "Golden Gate Bridge", lm, highlighted=True),
        LandmarkLabel(1500, 390, 520, "Fort Point", lm),
        EdgeArrow(1895, 540, 0, "Palace of Fine Arts", lm),
        InfoCard(
            1280,
            600,
            "Golden Gate Bridge",
            ("1.8 km · NNW", "Suspension bridge spanning the", "Golden Gate strait."),
            lm,
            footer="Wikipedia · CC BY-SA",
        ),
        Crosshair(960, 540, 40),
        PointerRing(1250, 520, solid=True),
        StatusCorner((("phone ✓", STATUS_OK), ("GPS ±8 m", STATUS_OK), ("align ~", STATUS_WARN))),
        Text(1500, 1030, "© OpenStreetMap contributors", (200, 200, 200)),
    ]
    img = render.render_to_image(items, SIZE)
    save(img, "10_full_scene.png")
    assert bright_bbox(img) is not None
    assert px(img, 5, 1075) == (0, 0, 0)
    # every class color shows up somewhere
    for c in (veh, per, lm):
        assert near_color(img, 960, 540, c, radius=1000, tol=100)
    assert near_color(img, 1250, 520, POINTER_COLOR, radius=3)


def test_save_png_helper(tmp_path):
    p = render.save_png([Crosshair(100, 100)], tmp_path / "x" / "c.png", (320, 200))
    img = QImage(str(p))
    assert (img.width(), img.height()) == (320, 200)


# ------------------------------------------------------------------------------- window


def test_pick_screen_fallbacks(caplog):
    primary = QGuiApplication.primaryScreen()
    screens = QGuiApplication.screens()
    with caplog.at_level("WARNING"):
        assert render.pick_screen(screens, "VITURE", None, primary) is primary
    assert "VITURE" in caplog.text
    assert render.pick_screen(screens, None, 0, primary) is screens[0]
    assert render.pick_screen(screens, None, None, primary) is primary
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert render.pick_screen(screens, None, 99, primary) is primary
    assert "out of range" in caplog.text
    assert render.pick_screen(screens, screens[0].name()[:3], None, primary) is screens[0]


def test_qt_renderer_lifecycle_and_paint():
    r = render.QtRenderer(screen_name="VITURE", size=(800, 600))
    assert r._canvas.isVisible()
    assert r._canvas.testAttribute(render.Qt.WidgetAttribute.WA_OpaquePaintEvent)
    r.show([PointerRing(100, 100, solid=True), Crosshair(300, 300)])
    img = r.grab()
    assert img.width() > 0
    # window-grab scales logical (800x600) coordinates to the actual window size
    sx = img.width() / 800
    sy = img.height() / 600
    assert px(img, 100 * sx, 100 * sy) == POINTER_COLOR
    assert px(img, 5, 5) == (0, 0, 0)
    r.show([])
    assert bright_bbox(r.grab()) is None
    r.close()
    r.close()  # idempotent
    r.show([Crosshair(1, 1)])  # no-op after close, must not raise


# ------------------------------------------------------------------- looks and passthrough

IDENT = (1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)


def _backdrop_square() -> Backdrop:
    """200x100 black image with a white square at x 20..60, y 10..40, mapped by x -> 2x + 200,
    y -> 2y + 100, so on a 960x540 display it must land at x 240..320, y 120..180."""
    src = np.zeros((100, 200, 3), np.uint8)
    src[10:40, 20:60] = 255
    return Backdrop(src, (2.0, 0, 200.0, 0, 2.0, 100.0, 0, 0, 1.0))


def test_backdrop_homography_places_pattern_normal_look():
    img = render.render_to_image([_backdrop_square()], (960, 540))
    assert px(img, 280, 150) == (255, 255, 255)
    for x, y in ((230, 150), (330, 150), (280, 110), (280, 190), (10, 10), (900, 500)):
        assert px(img, x, y) == (0, 0, 0), (x, y)
    bb = bright_bbox(img, thresh=128)
    assert bb is not None
    assert all(abs(a - b) <= 2 for a, b in zip(bb, (240, 120, 319, 179), strict=True))


def test_backdrop_homography_places_pattern_through_internal_resolution():
    # 1920x1080 display, looks run at 960x540: the homography is scaled and must still line up.
    bd = Backdrop(_backdrop_square().bgr, (4.0, 0, 400.0, 0, 4.0, 200.0, 0, 0, 1.0))
    img = render.render_to_image([bd], (1920, 1080), look="noir")
    assert min(px(img, 560, 300)) > 150
    for x, y in ((460, 300), (660, 300), (560, 230), (560, 380), (100, 900)):
        assert max(px(img, x, y)) < 40, (x, y)


def test_backdrop_perspective_homography():
    src = np.full((100, 100, 3), 255, np.uint8)
    h = (1.0, 0.0, 100.0, 0.0, 1.0, 100.0, 0.0, 0.004, 1.0)  # w' = 1 + 0.004 y: lower rows shrink
    a = arr(render.render_to_image([Backdrop(src, h)], (400, 300)))
    top_row = (a[102].max(axis=1) > 128).sum()
    bottom_row = (a[196].max(axis=1) > 128).sum()
    assert top_row > bottom_row + 5
    assert tuple(a[5, 5]) == (0, 0, 0)


def test_backdrop_gain():
    src = np.full((360, 640, 3), (50, 100, 200), np.uint8)  # BGR
    full = render.render_to_image([Backdrop(src, IDENT)], (640, 360))
    assert px(full, 320, 180) == (200, 100, 50)
    half = render.render_to_image([Backdrop(src, IDENT, gain=0.5)], (640, 360))
    assert all(abs(a - b) <= 1 for a, b in zip(px(half, 320, 180), (100, 50, 25), strict=True))
    dark = render.render_to_image([Backdrop(src, IDENT, gain=0.0)], (640, 360))
    assert px(dark, 320, 180) == (0, 0, 0)
    # gain also applies when a look runs
    lit = render.render_to_image([Backdrop(src, IDENT, gain=1.0)], (640, 360), look="noir")
    unlit = render.render_to_image([Backdrop(src, IDENT, gain=0.2)], (640, 360), look="noir")
    assert sum(px(lit, 320, 180)) > sum(px(unlit, 320, 180)) + 30


def test_overlay_draws_over_backdrop():
    src = np.full((540, 960, 3), 40, np.uint8)
    items = [Backdrop(src, IDENT), PointerRing(480, 270, solid=True)]
    img = render.render_to_image(items, (960, 540))
    assert px(img, 480, 270) == POINTER_COLOR
    assert px(img, 100, 100) == (40, 40, 40)


def test_plain_path_is_unchanged_full_resolution_painter():
    items = [Brackets(300, 560, 260, 170, (255, 80, 80)), Tag(430, 550, "car", (255, 80, 80))]
    a = arr(render.render_to_image(items, SIZE))
    ref = QImage(SIZE[0], SIZE[1], QImage.Format.Format_RGB32)
    ref.fill(0)
    p = render.QPainter(ref)
    render.paint_items(p, items, SIZE)
    p.end()
    assert np.array_equal(a, arr(ref))
    assert np.array_equal(a, arr(render.render_to_image(items, SIZE, look="normal")))


def test_render_to_image_rejects_unknown_look():
    with pytest.raises(ValueError):
        render.render_to_image([], (320, 200), look="sepia")


@pytest.mark.parametrize("look", [lk for lk in LOOKS if lk != "normal"])
def test_every_look_renders_at_display_size_and_is_deterministic(look):
    items = [Crosshair(160, 100)]
    img = render.render_to_image(items, (320, 200), look=look, t_s=1.0)
    assert (img.width(), img.height()) == (320, 200)
    assert np.array_equal(arr(img), arr(render.render_to_image(items, (320, 200), look, 1.0)))


def test_tint_color_per_look():
    car = CLASS_COLORS["vehicle"]
    assert render.tint_color(car, "normal") == car
    nvg = render.tint_color(car, "nvg")
    assert nvg[1] > 3 * nvg[0] and nvg[1] > 3 * nvg[2]  # green
    crt = render.tint_color(car, "crt")
    assert crt[1] > crt[0] and crt[1] > crt[2] and crt[1] > 150  # phosphor green
    flir = render.tint_color(car, "flir")
    assert max(flir) - min(flir) <= 1 and flir[0] > 140  # white-hot
    iron = render.tint_color(car, "flir", {"palette": 1.0})
    assert iron[0] > iron[1] > iron[2] and iron[0] > 140  # hot end of ironbow: yellow-white
    noir = render.tint_color(car, "noir")
    assert max(noir) - min(noir) <= 15
    assert render.tint_color((20, 20, 20), "nvg")[1] >= 120  # dim colours are lifted


@pytest.mark.parametrize(
    ("look", "check"),
    [
        ("nvg", lambda r, g, b: g > 2.5 * r and g > 2.5 * b),
        ("crt", lambda r, g, b: g > r and g > b),
        ("flir", lambda r, g, b: abs(r - g) < 12 and abs(g - b) < 12),
    ],
)
def test_labels_are_rendered_in_the_look_colour(look, check):
    items = [
        Brackets(800, 400, 320, 280, CLASS_COLORS["vehicle"]),
        Tag(960, 380, "car · ~15 m", CLASS_COLORS["vehicle"]),
        InfoCard(300, 300, "Fort Point", ("0.8 km", "Brick fort"), CLASS_COLORS["landmark"]),
    ]
    a = arr(render.render_to_image(items, SIZE, look=look, t_s=1.0)).astype(float)
    lit = a.max(axis=2) > 90
    assert lit.sum() > 200
    r, g, b = (a[:, :, c][lit].mean() for c in range(3))
    assert check(r, g, b), (look, r, g, b)


@pytest.mark.parametrize("look", ["crt", "nvg", "flir", "noir", "snow"])
def test_label_text_stays_at_least_24px_after_upscaling(look):
    items = [Tag(960, 580, "tiny", (255, 255, 255), size_px=10)]
    img = render.render_to_image(items, SIZE, look=look)
    bb = bright_bbox(img, 800, 500, 1120, 620, thresh=60)
    assert bb is not None
    assert bb[3] - bb[1] + 1 >= 14  # x-height plus descender of a 24 px "tiny"


def test_qt_renderer_set_look_and_passthrough():
    r = render.QtRenderer(size=(800, 600), gpu="off")
    src = np.full((600, 800, 3), 180, np.uint8)
    r.set_look("nvg", {"gain": 0.8})
    r.show([Backdrop(src, IDENT), Crosshair(400, 300)])
    a = arr(r.grab()).astype(float)
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    c = a[cy - 30 : cy + 30, cx - 30 : cx + 30].reshape(-1, 3).mean(axis=0)
    assert c[1] > 2.5 * c[0] and c[1] > 2.5 * c[2]
    r.set_look("normal")
    r.show([Backdrop(src, IDENT)])
    assert tuple(int(v) for v in arr(r.grab())[cy, cx]) == (180, 180, 180)
    r.show([])
    assert bright_bbox(r.grab()) is None
    with pytest.raises(ValueError):
        r.set_look("sepia")  # type: ignore[arg-type]
    r.close()
    r.set_look("crt")  # harmless after close


def test_qt_renderer_gpu_argument():
    with pytest.raises(ValueError):
        render.QtRenderer(size=(320, 200), gpu="yes")
    r = render.QtRenderer(size=(320, 200), gpu="auto")
    assert r.gpu_active is False  # the GLSL path is not implemented: CPU looks
    r.close()


def test_looks_png_over_fixture_video_frame(sample_video_path):
    import cv2

    cap = cv2.VideoCapture(str(sample_video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (540, 960, 3)
    backdrop = Backdrop(frame, (2.0, 0, 0, 0, 2.0, 0, 0, 0, 1.0))  # 960x540 -> 1920x1080
    lm, veh, per = CLASS_COLORS["landmark"], CLASS_COLORS["vehicle"], CLASS_COLORS["person"]
    items = [
        backdrop,
        Brackets(560, 380, 260, 170, veh),
        Tag(690, 370, "car · ~15 m", veh),
        Brackets(1100, 300, 90, 230, per, highlighted=True),
        Tag(1145, 290, "person · ~8 m", per, highlighted=True),
        LandmarkLabel(1300, 200, 480, "Golden Gate Bridge", lm),
        InfoCard(300, 600, "Fort Point", ("1.8 km · NNW", "Brick fort under the bridge"), lm),
        Crosshair(960, 540, 40),
        PointerRing(1250, 520, solid=True),
        StatusCorner((("phone ✓", STATUS_OK), ("GPS ±8 m", STATUS_OK))),
    ]
    out = OUT / "looks"
    out.mkdir(parents=True, exist_ok=True)
    variants = [(lk, lk, None) for lk in LOOKS]
    variants += [
        ("flir_ironbow", "flir", {"palette": 1.0}),
        ("flir_blackhot", "flir", {"mode": 1.0}),
    ]
    normal = arr(render.render_to_image(items, SIZE))
    for name, look, params in variants:
        img = render.render_to_image(items, SIZE, look=look, t_s=1.0, params=params)
        assert (img.width(), img.height()) == SIZE
        assert img.save(str(out / f"{name}.png"), "PNG")
        if look != "normal":
            assert np.abs(arr(img).astype(int) - normal.astype(int)).mean() > 4, name
    for name, _, _ in variants:
        assert (out / f"{name}.png").stat().st_size > 1000
