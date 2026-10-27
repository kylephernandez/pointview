# Ported from God's Eye View (https://github.com/bilawalsidhu/gods-eye-view), MIT License, Copyright (c) 2026 Bilawal Sidhu
# Source file: src/styles/thermal.js (FLIR), uniform ranges from src/ui/styleParameters.js
"""FLIR thermal look: white-hot, black-hot or the ironbow "Predator" ramp.

Ported: circular lens vignette and shading, sensor pixelation, soft IR blur, luminance to
temperature mapping with sensitivity, 12-band temperature contours, WHOT/BHOT, the ironbow ramp
(black, purple, magenta, red, orange, yellow, white) with the monochrome/ironbow ``palette`` blend,
hot-spot bloom and drifting fractal noise.

Skipped (our overlay draws these): the in-shader HUD, meaning the FLIR/mode markers, crosshair,
temperature digits, frame counter and right-hand scale bar. Simplified: every per-pixel step that
depends only on luminance (sensitivity, banding, polarity, palette) is baked into one 256-entry
colour table; the fractal noise is built at half resolution from bilinear value noise.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import cv2
import numpy as np

from pointview.looks._common import (
    cached,
    check_image,
    lut3,
    mask3,
    resolve,
    smoothstep,
    snap_maps,
    uv_grid,
    value_noise,
)

PARAMS = {
    "sensitivity": (0.75, 0.0, 1.0),
    "bloom": (0.65, 0.0, 1.0),
    "mode": (0.0, 0.0, 1.0),
    "pixelation": (1.5, 1.0, 6.0),
    "palette": (0.0, 0.0, 1.0),
}

_IRON = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.13, 0.0, 0.30),
        (0.49, 0.0, 0.45),
        (0.86, 0.10, 0.18),
        (1.0, 0.55, 0.0),
        (1.0, 0.91, 0.32),
        (1.0, 1.0, 1.0),
    ]
)  # RGB, GEV's c0..c6


def ironbow(t: np.ndarray) -> np.ndarray:
    """0..1 temperature -> RGB (0..1), piecewise linear through GEV's seven stops."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    s = t * 6.0
    i = np.minimum(s.astype(int), 5)
    f = (s - i)[..., None]
    return _IRON[i] * (1 - f) + _IRON[i + 1] * f


def _static(w: int, h: int, pix: float) -> SimpleNamespace:
    def build() -> SimpleNamespace:
        ux, uy = uv_grid(w, h)
        radius = np.sqrt(((ux * 2 - 1) * (w / h)) ** 2 + (uy * 2 - 1) ** 2)
        mask = (1.0 - smoothstep(0.6, 1.05, radius)) ** 0.7
        shading = np.maximum(1.0 - radius * radius * 0.25, 0.0)
        # GEV multiplies by the lens mask in the thermal path and again on the final blend
        return SimpleNamespace(snap=snap_maps(w, h, pix, ux, uy), mask=mask3(shading * mask * mask))

    return cached(("flir", w, h, round(pix, 3)), build)


def _tables(sensitivity: float, mode: float, palette: float) -> SimpleNamespace:
    def build() -> SimpleNamespace:
        luma = np.arange(256) / 255.0
        sens = 0.25 + 0.75 * sensitivity
        lo = 0.5 - sens * 0.5

        def mapped(x):
            return np.clip((x - lo) / sens, 0.0, 1.0)

        temp = mapped(luma)
        band = np.abs(temp * 12.0 - np.floor(temp * 12.0) - 0.5)
        contour = smoothstep(0.04, 0.06, band)
        temp = temp * (1.0 + (contour * 0.85 + 0.15 - 1.0) * 0.3)
        black_hot = mode >= 0.5
        mono = np.repeat((1.0 - temp if black_hot else temp)[:, None], 3, axis=1)
        iron = ironbow(temp)
        rgb = mono * (1.0 - palette) + iron * palette
        s_final = 1.0 - mapped(luma) if black_hot else mapped(luma)
        hot = smoothstep(0.6, 1.0, s_final)
        return SimpleNamespace(
            color=lut3(rgb[:, ::-1] * 255.0), hot=(hot * 255).round().astype(np.uint8)
        )

    return cached(("flir-lut", round(sensitivity, 4), mode >= 0.5, round(palette, 4)), build)


def _fbm_u8(hh: int, hw: int, t_s: float) -> np.ndarray:
    """4-octave coherent noise (float32, 0..255 scale), drifting with time."""
    acc = np.zeros((hh, hw), np.float32)
    amp = 0.5
    for i in range(4):
        cells = 80 * 2**i
        n = value_noise(hh, hw, cells, cells, t_s * 0.3 * 2**i, t_s * 0.2 * 2**i, seed=100 + i)
        acc += amp * n
        amp *= 0.5
    return acc


def apply(img: np.ndarray, t_s: float, params: Mapping[str, float] | None = None) -> np.ndarray:
    img = check_image(img)
    p = resolve(PARAMS, params)
    h, w = img.shape[:2]
    st = _static(w, h, p["pixelation"])
    tb = _tables(p["sensitivity"], p["mode"], p["palette"])

    gray = cv2.remap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), *st.snap, cv2.INTER_NEAREST)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.1, borderType=cv2.BORDER_REPLICATE)
    src = cv2.addWeighted(gray, 0.4, blur, 0.6, 0.0)
    out = cv2.LUT(cv2.cvtColor(src, cv2.COLOR_GRAY2BGR), tb.color)

    # hot-spot bloom plus drifting thermal noise, added equally to the three channels. Both are
    # smooth, so they are built at half resolution and upscaled once.
    hh, hw = max(1, h // 2), max(1, w // 2)
    hot = cv2.resize(cv2.LUT(gray, tb.hot), (hw, hh), interpolation=cv2.INTER_AREA)
    hot = cv2.GaussianBlur(hot, (0, 0), 2.83 * 3 / 2, borderType=cv2.BORDER_REPLICATE)
    field = cv2.addWeighted(
        hot.astype(np.float32), p["bloom"] * 0.8, _fbm_u8(hh, hw, t_s), 0.08, -0.04 * 255.0
    )
    field = cv2.resize(np.rint(field).astype(np.int16), (w, h), interpolation=cv2.INTER_LINEAR)
    out16 = cv2.add(out, cv2.merge([field, field, field]), dtype=cv2.CV_16S)
    out = cv2.convertScaleAbs(cv2.max(out16, 0))
    return cv2.multiply(out, st.mask, scale=1.0 / 255.0)
