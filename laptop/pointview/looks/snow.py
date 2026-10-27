# Ported from God's Eye View (https://github.com/bilawalsidhu/gods-eye-view), MIT License, Copyright (c) 2026 Bilawal Sidhu
# Source file: src/styles/snow.js (Snow), uniform ranges from src/ui/styleParameters.js
"""Snow look: cool colour shift, partial desaturation, brightening, frost on bright areas, five
parallax layers of falling flakes and a light ground fog.

Ported: the grading chain and the layer parameters (depth, speed, size, wind sway, cell hash and
flake wander) exactly. Simplified: instead of evaluating a 3x3-cell distance field per pixel, the
flakes of each layer are enumerated from the same cell hash and stamped as small elliptical soft
discs at whole-pixel positions; frost and grading are table lookups on 8-bit data.
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
    lut3,
    mask3,
    resolve,
    smoothstep,
)

PARAMS = {
    "density": (0.6, 0.0, 1.0),
    "wind": (0.5, 0.0, 1.0),
}

_LAYERS = ((0.0, 0.5), (0.25, 0.6), (0.5, 0.7), (0.75, 0.8), (1.0, 1.0))  # (layer, weight)
_COOL_BGR = np.array([1.1, 0.9, 0.85])  # GEV cool = rgb * (0.85, 0.9, 1.1)
_FROST_BGR = np.array([1.0, 0.97, 0.95])  # vec3(0.95, 0.97, 1.0)
_FOG_BGR = np.array([0.95, 0.88, 0.85])  # vec3(0.85, 0.88, 0.95)


def _hash2(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """GEV snow.js ``hash(vec2)``."""
    x, y = frac(px * 123.34), frac(py * 456.21)
    d = x * (x + 45.32) + y * (y + 45.32)
    x, y = x + d, y + d
    return frac(x * y)


def _static(w: int, h: int) -> SimpleNamespace:
    def build() -> SimpleNamespace:
        r = (np.arange(h) + 0.5) / h  # 1 - uv.y
        fog_r = smoothstep(0.0, 0.4, r) * 0.1
        keep = np.repeat((1.0 - fog_r)[:, None], w, axis=1)
        add = fog_r[:, None, None] * (_FOG_BGR * 255.0)[None, None, :]
        add = np.broadcast_to(add, (h, w, 3))
        luma = np.arange(256) / 255.0
        m = smoothstep(0.3, 0.8, luma)
        q = 1.0 - 0.25 * m
        return SimpleNamespace(
            fog_keep=mask3(keep),
            fog_add=np.ascontiguousarray(np.rint(add).astype(np.uint8)),
            img_tab=lut3(0.51 * _COOL_BGR[None, :] * np.arange(256)[:, None]),
            l_tab=np.rint(0.34 * np.arange(256) + 0.15 * 255.0).clip(0, 255).astype(np.uint8),
            q_tab=np.rint(q * 255.0).astype(np.uint8),
            g_tab=lut3((1.0 - q)[:, None] * (_FROST_BGR * 255.0)[None, :]),
        )

    return cached(("snow", w, h), build)


def _sprite(h_px: int, w_px: int, size: float, depth: float, sx: float, sy: float) -> np.ndarray:
    ry = int(np.ceil(size * h_px / sy)) + 1
    rx = int(np.ceil(size * w_px / sx)) + 1
    yy, xx = np.mgrid[-ry : ry + 1, -rx : rx + 1].astype(np.float64)
    d = np.hypot(xx / w_px * sx, yy / h_px * sy)
    t = np.clip(1.0 - d / size, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t) * depth).astype(np.float32)


def _flakes(w: int, h: int, t_s: float, wind: float) -> np.ndarray:
    """The summed, weighted flake field (float32, un-clamped) for time ``t_s``."""
    field = np.zeros((h, w), np.float32)
    for layer, weight in _LAYERS:
        depth = 0.5 + layer * 0.5
        speed = 0.4 + layer * 0.3
        size = 0.01 + (0.025 - 0.01) * layer
        wf = 0.05 + wind * 0.2
        sx = 4.0 + layer * 4.0
        sy = 0.5 * sx
        offx = -2.0 * np.cos(t_s * 0.5 + layer * 3.14) * wf
        offy = t_s * speed
        cx = np.arange(int(np.floor(offx)) - 1, int(np.floor(offx + sx)) + 2, dtype=np.float64)
        cy = np.arange(int(np.floor(offy)) - 1, int(np.floor(offy + sy)) + 2, dtype=np.float64)
        gx, gy = np.meshgrid(cx, cy)
        pt_x = 0.5 + 0.4 * np.sin(t_s * 0.3 + 6.2831 * _hash2(gx, gy))
        pt_y = 0.5 + 0.4 * np.sin(t_s * 0.3 + 6.2831 * _hash2(gx + 100.0, gy + 100.0))
        u = (gx + pt_x - offx) / sx
        v = (gy + pt_y - offy) / sy  # GLSL v grows upward
        px = np.rint(u * w).astype(int).ravel()
        py = np.rint((1.0 - v) * h).astype(int).ravel()
        spr = _sprite(h, w, size, depth, sx, sy) * weight
        sh, sw = spr.shape
        ry, rx = sh // 2, sw // 2
        for x, y in zip(px, py, strict=True):
            x0, y0 = x - rx, y - ry
            x1, y1 = x0 + sw, y0 + sh
            if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
                continue
            sx0, sy0 = max(0, -x0), max(0, -y0)
            ex, ey = min(sw, w - x0), min(sh, h - y0)
            field[y0 + sy0 : y0 + ey, x0 + sx0 : x0 + ex] += spr[sy0:ey, sx0:ex]
    return field


def apply(img: np.ndarray, t_s: float, params: Mapping[str, float] | None = None) -> np.ndarray:
    img = check_image(img)
    p = resolve(PARAMS, params)
    h, w = img.shape[:2]
    st = _static(w, h)

    # luma of the cool-shifted colour drives frost; everything else is per-channel or per-luma
    luma = cv2.transform(img, np.array([[0.114 * 1.1, 0.587 * 0.9, 0.299 * 0.85]]))
    l3 = gray3(luma)
    base = cv2.add(cv2.LUT(img, st.img_tab), cv2.LUT(l3, st.l_tab))
    out = cv2.multiply(base, cv2.LUT(l3, st.q_tab), scale=1.0 / 255.0)
    out = cv2.add(out, cv2.LUT(l3, st.g_tab))

    snow = np.clip(_flakes(w, h, t_s, p["wind"]) * (0.4 + p["density"] * 1.2), 0.0, 1.0)
    out = cv2.add(out, gray3(cv2.convertScaleAbs(snow, alpha=0.8 * 255.0)))
    out = cv2.multiply(out, st.fog_keep, scale=1.0 / 255.0)
    return cv2.add(out, st.fog_add)
