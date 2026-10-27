# Ported from God's Eye View (https://github.com/bilawalsidhu/gods-eye-view), MIT License, Copyright (c) 2026 Bilawal Sidhu
# Source file: src/styles/retro.js (CRT Terminal), uniform ranges from src/ui/styleParameters.js
"""CRT terminal look.

Ported: barrel distortion (default 0, as in GEV), sparse horizontal jitter, chromatic aberration,
pixelation, 8x8 Bayer dithering, 10-level posterisation, +30 % saturation, RGB shadow-mask
subpixels, block-edge darkening, rolling scanlines, phosphor ghosting, 60 Hz flicker, glitch bars,
warm phosphor tint and edge vignette.

Simplified: jitter moves whole rows of the already-composed image rather than shifting the sample
position; the shadow mask, block edge and vignette are folded into one static multiplier (GEV
applies the vignette last); dithering is applied at 0.88 scale in an 8-bit pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import cv2
import numpy as np

from pointview.looks._common import (
    cached,
    check_image,
    frac,
    gray3,
    hash21,
    lut3,
    mask3,
    mul_mask,
    resolve,
    row_factor,
    smoothstep,
    snap_maps,
    uv_grid,
)

#: name -> (default, min, max), from GEV's retro uniforms.
PARAMS = {
    "pixelation": (5.0, 1.0, 10.0),
    "distortion": (0.0, 0.0, 1.0),
    "instability": (0.4, 0.0, 1.0),
}

_BAYER8 = np.array(
    [
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ],
    dtype=np.float64,
)

_DITHER_PEAK = 15.0  # 0.12 * 255 / 2, in 8-bit counts
_PRE_SCALE = 225.0 / 255.0  # leaves head-room for the +/-15 dither without saturating


def _static(w: int, h: int, pix: float, dist: float) -> SimpleNamespace:
    def build() -> SimpleNamespace:
        ux, uy = uv_grid(w, h)
        if dist > 0:
            cx, cy = ux * 2 - 1, uy * 2 - 1
            d = 1.0 + (cx * cx + cy * cy) * dist * 0.4
            dx, dy = (cx * d) * 0.5 + 0.5, (cy * d) * 0.5 + 0.5
        else:
            dx, dy = ux, uy
        pixmap = snap_maps(w, h, pix, dx, dy)

        # chromatic aberration: R samples outwards, B inwards, by |c| * 0.008
        ccx, ccy = dx - 0.5, dy - 0.5
        ca = np.sqrt(ccx * ccx + ccy * ccy) * 0.008
        rmap = snap_maps(w, h, 1.0, dx + ccx * ca, dy + ccy * ca)
        bmap = snap_maps(w, h, 1.0, dx - ccx * ca, dy - ccy * ca)

        # Bayer dither, constant over each pixelation block
        bx = np.floor(dx * w / pix).astype(np.int64) % 8
        by = np.floor(dy * h / pix).astype(np.int64) % 8
        dith = _BAYER8[by, bx] / 64.0 - 0.5
        dpos = np.clip(np.rint(dith * 2 * _DITHER_PEAK) + _DITHER_PEAK, 0, 255).astype(np.uint8)
        dither3 = np.repeat(dpos[:, :, None], 3, axis=2)

        # RGB shadow mask * block-edge darkening * edge vignette
        subx = np.mod(dx * w, 3.0)
        stripes = [smoothstep(0.0, 0.8, 1.0 - np.abs(subx - c)) for c in (2.5, 1.5, 0.5)]  # B,G,R
        sub_s = float(smoothstep(2.0, 6.0, pix)) * 0.3
        fac = np.stack([1.0 - sub_s + sub_s * (s * 0.7 + 0.3) for s in stripes], axis=2)
        fx, fy = frac(dx * w / pix), frac(dy * h / pix)
        edge = smoothstep(0.0, 0.08, np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)))
        fac = fac * (edge * 0.15 + 0.85)[:, :, None]
        vig = np.clip(np.maximum(dx * (1 - dx) * dy * (1 - dy) * 20.0, 0.0) ** 0.4, 0.0, 1.0)
        fac = fac * (1.0 - 0.6 + 0.6 * vig)[:, :, None]
        inside = ((dx >= 0) & (dx <= 1) & (dy >= 0) & (dy <= 1))[:, :, None]

        i = np.arange(256, dtype=np.float64)
        pre = np.rint(i * _PRE_SCALE).astype(np.uint8)
        v = (i - _DITHER_PEAK) / 225.0
        post = np.clip(np.floor(v * 10.0 + 0.5) / 10.0 * 255.0, 0, 255).astype(np.uint8)
        return SimpleNamespace(
            pixmap=pixmap,
            rmap=rmap,
            bmap=bmap,
            dither3=dither3,
            mask=mask3(fac * inside),
            pre=pre,
            post=post,
            ys=(h - np.arange(h) - 0.5),
        )

    return cached(("crt", w, h, round(pix, 3), round(dist, 3)), build)


def apply(img: np.ndarray, t_s: float, params: Mapping[str, float] | None = None) -> np.ndarray:
    img = check_image(img)
    p = resolve(PARAMS, params)
    h, w = img.shape[:2]
    inst = p["instability"]
    st = _static(w, h, p["pixelation"], p["distortion"])

    # sample: pixelated colour, blended 70/30 with the chromatic-aberration sample
    pixd = cv2.remap(img, *st.pixmap, cv2.INTER_NEAREST)
    b, g, r = cv2.split(img)
    r2 = cv2.remap(r, *st.rmap, cv2.INTER_NEAREST)
    b2 = cv2.remap(b, *st.bmap, cv2.INTER_NEAREST)
    base = cv2.addWeighted(pixd, 0.7, cv2.merge([b2, g, r2]), 0.3, 0.0)

    # horizontal jitter on a sparse set of lines (seeded by time)
    if inst > 0:
        rows = np.arange(h, dtype=np.float64)
        seed = hash21(rows, np.full(h, np.floor(t_s * 8.0)))
        active = np.nonzero(seed >= 0.97 - inst * 0.04)[0]
        if active.size:
            amt = hash21(rows[active] * 7.0, np.full(active.size, np.floor(t_s * 12.0))) - 0.5
            shifts = np.rint(amt * 0.008 * inst * w).astype(int)
            for y, s in zip(active, shifts, strict=True):
                if s:
                    base[y] = np.roll(base[y], int(s), axis=0)

    # dither + posterise, saturation boost
    v = cv2.add(cv2.LUT(base, st.pre), st.dither3)
    post = cv2.LUT(v, st.post)
    gp = gray3(cv2.cvtColor(post, cv2.COLOR_BGR2GRAY))
    res = cv2.addWeighted(post, 1.3, gp, -0.3, 0.0)
    res = mul_mask(res, st.mask)

    # rolling scanlines
    scan = (np.sin(st.ys + t_s * 2.5) * 0.5 + 0.5) ** 1.5
    res = mul_mask(res, row_factor(0.65 + 0.35 * scan, w))

    # phosphor ghost: a dim copy of the luminance two pixels to the left
    if inst > 0:
        gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shifted = np.zeros_like(gs)
        shifted[:, 2:] = gs[:, :-2]
        ghost = cv2.convertScaleAbs(shifted, alpha=0.08 * inst)
        res = cv2.add(res, gray3(ghost))

    # flicker and warm phosphor tint in one per-channel table (BGR order)
    flicker = 1.0 - (np.sin(t_s * 188.5) * 0.5 + 0.5) * 0.03 * inst
    tint = np.array([1.0 + 0.4 * (0.94 - 1.0), 1.0, 1.0 + 0.4 * 0.02]) * flicker
    res = cv2.LUT(res, lut3(np.arange(256)[:, None] * tint[None, :]))

    # rare glitch bar
    if inst > 0 and hash21(np.floor(t_s * 2.0), 0.0) >= 0.92 - inst * 0.08:
        gy = float(hash21(np.floor(t_s * 2.0), 1.0))
        y0 = round((1.0 - gy) * h)
        lo, hi = max(0, y0 - 2), min(h, y0 + 2)
        if hi > lo:
            a = int(0.3 * inst * 255)
            res[lo:hi] = cv2.add(res[lo:hi], (a, a, a, 0))
    return res
