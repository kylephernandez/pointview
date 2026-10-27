"""CPU looks: determinism, shape/dtype, visual character, parameters and speed (960x540)."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from pointview.interfaces import LOOKS
from pointview.looks import apply_look, look_defaults, look_params

ACTIVE = [lk for lk in LOOKS if lk != "normal"]
W, H = 960, 540
#: GEV-style looks that animate (everything but noir, whose grain is static in GEV).
ANIMATED = ["crt", "nvg", "flir", "snow"]


def _frame(idx: int = 60) -> np.ndarray:
    from pathlib import Path

    cap = cv2.VideoCapture(str(Path(__file__).parent / "fixtures" / "sample.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    assert ok
    return f


@pytest.fixture(scope="module")
def frame() -> np.ndarray:
    return _frame()


def _scene(w: int = W, h: int = H) -> np.ndarray:
    """A coloured, textured test image: gradients plus a bright blob (BGR)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([60 + 120 * xx / w, 40 + 150 * yy / h, 90 + 100 * (1 - xx / w)], axis=2)
    cv2.circle(img, (w // 2, h // 2), h // 8, (230, 240, 250), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _flat(level: int, w: int = W, h: int = H) -> np.ndarray:
    return np.full((h, w, 3), level, np.uint8)


def _mean(img, y, x, r=40):
    return img[y - r : y + r, x - r : x + r].reshape(-1, 3).mean(axis=0)


def test_normal_is_identity(frame):
    assert apply_look("normal", frame, 1.0) is frame


def test_unknown_look_and_bad_input(frame):
    with pytest.raises(ValueError):
        apply_look("sepia", frame, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        apply_look("crt", frame.astype(np.float32), 0.0)
    with pytest.raises(ValueError):
        apply_look("crt", frame[:, :, 0], 0.0)


@pytest.mark.parametrize("look", ACTIVE)
def test_deterministic_for_a_given_time(look, frame):
    a = apply_look(look, frame, 3.217)
    b = apply_look(look, frame, 3.217)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("look", ANIMATED)
def test_changes_over_time(look):
    img = _scene()
    assert not np.array_equal(apply_look(look, img, 1.0), apply_look(look, img, 1.7))


@pytest.mark.parametrize("look", ACTIVE)
@pytest.mark.parametrize("size", [(960, 540), (640, 360), (321, 181), (1920, 1080)])
def test_preserves_size_and_dtype_and_input(look, size):
    img = _scene(*size)
    before = img.copy()
    out = apply_look(look, img, 0.5)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(img, before)
    assert out is not img


@pytest.mark.parametrize("look", ACTIVE)
def test_visually_distinct_from_normal(look, frame):
    img = _scene()
    out = apply_look(look, img, 0.5)
    diff = np.abs(out.astype(int) - img.astype(int)).mean()
    assert diff > 8, f"{look} barely changes the image (mean abs diff {diff:.1f})"
    # and not a degenerate all-black / all-white result
    assert 3 < out.mean() < 252
    # also on the fixture frame
    f_out = apply_look(look, frame, 0.5)
    assert not np.array_equal(f_out, frame)


def test_nvg_is_phosphor_green():
    out = apply_look("nvg", _scene(), 0.5).astype(float)
    b, g, r = (out[:, :, c].mean() for c in range(3))
    assert g > 3 * r and g > 3 * b
    # circular tube: corners black, centre lit
    assert out[:20, :20].max() == 0
    assert out[H // 2 - 20 : H // 2 + 20, W // 2 - 20 : W // 2 + 20, 1].mean() > 60
    # P43 ratio r:g:b ~ 0.16:1:0.22 on the bright centre
    c = out[H // 2 - 30 : H // 2 + 30, W // 2 - 30 : W // 2 + 30].reshape(-1, 3).mean(axis=0)
    assert c[2] / c[1] == pytest.approx(0.16, abs=0.04)
    assert c[0] / c[1] == pytest.approx(0.22, abs=0.04)


def test_nvg_gain_and_bloom_parameters():
    img = _scene()
    dim = apply_look("nvg", img, 0.5, {"gain": 0.0}).mean()
    bright = apply_look("nvg", img, 0.5, {"gain": 1.0}).mean()
    assert bright > dim * 1.2
    spot = _flat(0)
    cv2.circle(spot, (W // 2, H // 2), 14, (255, 255, 255), -1)
    halo0 = apply_look("nvg", spot, 0.5, {"bloom": 0.0})
    halo1 = apply_look("nvg", spot, 0.5, {"bloom": 1.0})
    ring = (slice(H // 2 + 18, H // 2 + 26), slice(W // 2 - 10, W // 2 + 10))
    assert halo1[ring][:, :, 1].mean() > halo0[ring][:, :, 1].mean() + 8


def test_flir_white_hot_and_black_hot_are_grey_and_inverse():
    img = _flat(40)
    img[:, W // 2 :] = 200  # dark left, bright right
    whot = apply_look("flir", img, 0.5).astype(float)
    bhot = apply_look("flir", img, 0.5, {"mode": 1.0}).astype(float)
    for out in (whot, bhot):
        c = out[H // 2 - 20 : H // 2 + 20, W // 2 - 120 : W // 2 - 60].reshape(-1, 3).mean(axis=0)
        assert np.ptp(c) < 6  # monochrome
    l_w, r_w = _mean(whot, H // 2, W // 2 - 90), _mean(whot, H // 2, W // 2 + 90)
    l_b, r_b = _mean(bhot, H // 2, W // 2 - 90), _mean(bhot, H // 2, W // 2 + 90)
    assert r_w.mean() > l_w.mean() + 30  # hot is bright in white-hot
    assert r_b.mean() < l_b.mean() - 30  # and dark in black-hot


def test_flir_ironbow_dark_purple_bright_yellow_white():
    img = _flat(70)
    img[:, W // 2 :] = 205
    out = apply_look("flir", img, 0.5, {"palette": 1.0}).astype(float)
    dark = _mean(out, H // 2, W // 2 - 90, 25)  # BGR
    hot = _mean(out, H // 2, W // 2 + 90, 25)
    db, dg, dr = dark
    assert db > dr > dg + 15 and db > 50  # purple: blue strongest, red above green
    assert dg < 25
    hb, hg, hr = hot
    assert hr > 220 and hg > 190 and hb < hg - 30  # yellow-white: blue lagging


def test_flir_sensitivity_and_pixelation_parameters():
    img = _scene()
    a = apply_look("flir", img, 0.5, {"sensitivity": 0.0})
    b = apply_look("flir", img, 0.5, {"sensitivity": 1.0})
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 4
    blocky = apply_look("flir", img, 0.5, {"pixelation": 6.0})
    assert blocky.shape == img.shape


def test_noir_is_near_grey_with_warm_tint_and_contrast():
    img = _scene()
    out = apply_look("noir", img, 0.0).astype(float)
    c = out[H // 2 - 100 : H // 2 + 100, W // 2 - 100 : W // 2 + 100].reshape(-1, 3).mean(axis=0)
    b, g, r = c
    assert r > g > b and (r - b) / max(r, 1) < 0.15  # sepia, barely
    grey_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    grey_out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
    c0 = (slice(H // 4, 3 * H // 4), slice(W // 4, 3 * W // 4))
    assert grey_out[c0].std() > grey_in[c0].std() * 1.3  # contrast boosted
    # vignette: corners darker than the centre
    assert out[:30, :30].mean() < out[H // 2 - 15 : H // 2 + 15, W // 2 - 15 : W // 2 + 15].mean()


def test_noir_grain_is_static():
    img = _scene()
    assert np.array_equal(apply_look("noir", img, 0.0), apply_look("noir", img, 9.0))
    flat = apply_look("noir", _flat(128), 0.0, {"grainAmt": 1.0})
    quiet = apply_look("noir", _flat(128), 0.0, {"grainAmt": 0.0})
    mid = (slice(H // 2 - 50, H // 2 + 50), slice(W // 2 - 50, W // 2 + 50))
    assert flat[mid].std() > quiet[mid].std() + 1


def test_snow_is_cool_brighter_and_has_flakes():
    dark = _flat(30)
    out = apply_look("snow", dark, 2.0).astype(int)
    assert out.mean() > dark.mean() + 20  # brightened + fog
    assert (out.min(axis=2) > 200).sum() > 100  # white flakes on a dark scene
    grey = apply_look("snow", _flat(128), 2.0, {"density": 0.0}).astype(float)
    top = grey[:100].reshape(-1, 3).mean(axis=0)  # above the fog
    assert top[0] > top[2]  # BGR: blue > red after the cool shift
    more = apply_look("snow", dark, 2.0, {"density": 1.0}).astype(int)
    assert (more.min(axis=2) > 200).sum() >= (out.min(axis=2) > 200).sum()


def test_snow_flakes_fall():
    a = apply_look("snow", _flat(30), 2.0, {"wind": 0.0}).min(axis=2) > 200
    b = apply_look("snow", _flat(30), 2.5, {"wind": 0.0}).min(axis=2) > 200
    ya, yb = np.nonzero(a)[0].mean(), np.nonzero(b)[0].mean()
    assert not np.array_equal(a, b) and abs(ya - yb) < 60


def test_crt_scanlines_and_pixel_blocks():
    out = apply_look("crt", _flat(160), 0.0).astype(float)
    row_means = out[:, 100:860].mean(axis=(1, 2))
    assert row_means.std() > 2.0  # rolling scanlines modulate the rows
    # pixelation 5: adjacent columns inside a block are mostly identical
    out1 = apply_look("crt", _scene(), 0.0, {"pixelation": 1.0, "instability": 0.0})
    out5 = apply_look("crt", _scene(), 0.0, {"pixelation": 8.0, "instability": 0.0})
    assert (
        np.abs(np.diff(out5.astype(int), axis=1)).mean()
        != np.abs(np.diff(out1.astype(int), axis=1)).mean()
    )


def test_crt_distortion_blacks_the_corners():
    out = apply_look("crt", _flat(200), 0.0, {"distortion": 1.0, "instability": 0.0})
    assert out[:3, :3].max() < 30
    assert out[H // 2, W // 2].max() > 50


def test_parameter_tables_match_gev_defaults():
    assert look_defaults("normal") == {}
    assert look_defaults("crt") == {"pixelation": 5.0, "distortion": 0.0, "instability": 0.4}
    assert look_defaults("nvg") == {
        "gain": 0.55, "bloom": 0.3, "scanlineStr": 1.0, "pixelation": 2.5,
    }  # fmt: skip
    assert look_defaults("flir") == {
        "sensitivity": 0.75, "bloom": 0.65, "mode": 0.0, "pixelation": 1.5, "palette": 0.0,
    }  # fmt: skip
    assert look_defaults("noir") == {"contrastAmt": 1.2, "grainAmt": 0.5, "vignetteAmt": 0.5}
    assert look_defaults("snow") == {"density": 0.6, "wind": 0.5}
    for lk in ACTIVE:
        for default, lo, hi in look_params(lk).values():
            assert lo <= default <= hi


def test_unknown_and_out_of_range_params_are_tolerated(frame):
    a = apply_look("nvg", frame, 0.0, {"gain": 99.0, "nonsense": 1.0})
    b = apply_look("nvg", frame, 0.0, {"gain": 1.0})
    assert np.array_equal(a, b)


@pytest.mark.parametrize("look", ACTIVE)
def test_speed_at_960x540(look, frame):
    """Budget is 8 ms on the reference CPU; CI asserts 15 ms on the best of 20 runs so a noisy
    shared runner does not flake. The measured numbers are printed (run with -s)."""
    img = _frame()
    assert img.shape == (H, W, 3)
    apply_look(look, img, 0.0)  # build caches
    times = []
    for i in range(20):
        t0 = time.perf_counter()
        apply_look(look, img, 0.1 * (i + 1))
        times.append((time.perf_counter() - t0) * 1e3)
    best, med = min(times), float(np.median(times))
    print(f"\nlook speed {look:5s} 960x540: best {best:5.2f} ms  median {med:5.2f} ms")
    assert best <= 15.0
