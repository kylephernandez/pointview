# Ported from God's Eye View (https://github.com/bilawalsidhu/gods-eye-view), MIT License, Copyright (c) 2026 Bilawal Sidhu
# Source file: src/styles/surveillance.js (Night Vision / NVG), uniform ranges from src/ui/styleParameters.js
"""Night-vision (PVS-14 image intensifier) look.

Ported: lens barrel distortion, intensifier-tube pixelation, circular tube vignette and centre
falloff, luminance gain curve, wide Gaussian bloom/corona on bright sources, P43 green phosphor
(0.16, 1.0, 0.22), coherent tube grain, gain-dependent dark-area noise, honeycomb fibre plate and
display scanlines.

Skipped (our overlay draws these): the in-shader HUD, meaning the NVG/AUTO markers, centre
crosshair, 7-segment timestamp and blinking REC dot. Simplified: value noise is bilinear rather than
smoothstep-interpolated; the 11x11 bloom tap loop is a downsample-by-4 Gaussian (same sigma).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import cv2
import numpy as np

from pointview.looks._common import (
    blur_small,
    cached,
    check_image,
    gray3,
    lut3,
    resolve,
    smoothstep,
    snap_maps,
    to_u8,
    uv_grid,
    value_noise,
    white_noise_u8,
)

PARAMS = {
    "gain": (0.55, 0.0, 1.0),
    "bloom": (0.3, 0.0, 1.0),
    "scanlineStr": (1.0, 0.0, 1.0),
    "pixelation": (2.5, 1.0, 6.0),
}

PHOSPHOR_RGB = (0.16, 1.0, 0.22)  # P43, as in GEV
_PHOSPHOR_BGR = np.array(PHOSPHOR_RGB[::-1])


def _static(w: int, h: int, pix: float) -> SimpleNamespace:
    def build() -> SimpleNamespace:
        ux, uy = uv_grid(w, h)
        cx, cy = ux * 2 - 1, uy * 2 - 1
        r2 = cx * cx + cy * cy
        d = 1.0 + r2 * 0.5 * 0.5 + r2 * r2 * 0.5 * 0.15
        dx, dy = (cx * d) * 0.5 + 0.5, (cy * d) * 0.5 + 0.5
        snap = snap_maps(w, h, pix, dx, dy)

        # circular tube mask (aspect-corrected) and centre falloff
        radius = np.sqrt((cx * (w / h)) ** 2 + cy * cy)
        tube = (1.0 - smoothstep(0.6, 1.05, radius)) ** 0.7
        shading = np.maximum(1.0 - radius * radius * 0.3, 0.0)

        # honeycomb fibre-optic plate
        scale = min(w, h) * 0.008
        px, py = dx * w * scale, dy * h * scale
        rx, ry, hx, hy = 1.0, 1.732, 0.5, 0.866
        ax, ay = np.mod(px, rx) - hx, np.mod(py, ry) - hy
        bx, by = np.mod(px - hx, rx) - hx, np.mod(py - hy, ry) - hy
        use_a = (ax * ax + ay * ay) < (bx * bx + by * by)
        gx, gy = np.where(use_a, ax, bx), np.where(use_a, ay, by)
        dd = np.maximum(np.abs(gx), np.abs(gy) * 0.577 + np.abs(gx) * 0.5)
        hc = smoothstep(0.4, 0.45, dd)

        inside = (dx >= 0) & (dx <= 1) & (dy >= 0) & (dy <= 1)
        mult = (shading * tube * tube * (1.0 - hc * 0.04) * inside).astype(np.float32)
        return SimpleNamespace(snap=snap, mult=mult, ys=h - np.arange(h) - 0.5)

    return cached(("nvg", w, h, round(pix, 3)), build)


def _luts(gain: float) -> tuple[np.ndarray, np.ndarray]:
    def build():
        gl = 0.8 + (2.5 - 0.8) * gain
        luma = np.arange(256) / 255.0
        amp = np.clip(luma * gl, 0, 1) ** (1.2 + (0.7 - 1.2) * gain)
        bright = smoothstep(0.4, 0.9, luma * gl)
        return (amp * 255).round().astype(np.uint8), (bright * 255).round().astype(np.uint8)

    return cached(("nvg-lut", round(gain, 4)), build)


def apply(img: np.ndarray, t_s: float, params: Mapping[str, float] | None = None) -> np.ndarray:
    img = check_image(img)
    p = resolve(PARAMS, params)
    h, w = img.shape[:2]
    gain = p["gain"]
    st = _static(w, h, p["pixelation"])
    amp_lut, bright_lut = _luts(gain)

    gray = cv2.remap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), *st.snap, cv2.INTER_NEAREST)
    amp = cv2.LUT(gray, amp_lut).astype(np.float32)

    # intensifier bloom / corona around bright sources
    y = amp + blur_small(cv2.LUT(gray, bright_lut), 4, 3.0, 11).astype(np.float32) * (
        p["bloom"] * 1.5
    )

    # scintillation: slow coherent tube grain plus gain-dependent white noise in the dark
    grain_k = 0.06 + (0.2 - 0.06) * gain
    n = value_noise(h, w, 120, 120, t_s * 0.5, t_s * 0.3, seed=43)
    y += (n.astype(np.float32) - 127.5) * grain_k
    wn = white_noise_u8(h, w, t_s, salt=1).astype(np.float32)
    y += (255.0 - amp) * wn * (0.08 * gain / 255.0)

    # honeycomb, tube shading and mask (static) and display scanlines (per row)
    scan = (np.sin(st.ys * 1.2 + t_s * 2.0) * 0.5 + 0.5) ** 2.5
    rows = (1.0 - scan * p["scanlineStr"] * 0.15).astype(np.float32)
    y *= st.mult
    y *= rows[:, None]

    return cv2.LUT(gray3(to_u8(y)), lut3(np.arange(256)[:, None] * _PHOSPHOR_BGR[None, :]))
