# Ported from God's Eye View (https://github.com/bilawalsidhu/gods-eye-view), MIT License, Copyright (c) 2026 Bilawal Sidhu
# Source file: src/styles/noir.js (Noir), uniform ranges from src/ui/styleParameters.js
"""Film-noir look: desaturation, high contrast, film grain, vignette and a faint sepia tint.

Ported all of it. GEV's grain is a fixed hash of the pixel position (no time term), so the grain
here is static too; it is therefore the same for every ``t_s``.
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
    resolve,
    to_u8,
    uv_grid,
)

PARAMS = {
    "contrastAmt": (1.2, 0.0, 2.0),
    "grainAmt": (0.5, 0.0, 1.0),
    "vignetteAmt": (0.5, 0.0, 1.0),
}

# GEV's sepia matrix rows summed (the input is grey), mixed in at 0.15
_SEPIA_BGR = 1.0 + 0.15 * (np.array([0.272 + 0.534 + 0.131, 0.349 + 0.686 + 0.168, 1.351]) - 1.0)


def _static(w: int, h: int) -> SimpleNamespace:
    def build() -> SimpleNamespace:
        ux, uy = uv_grid(w, h)
        g = frac(np.sin((ux * w) * 12.9898 + (uy * h) * 78.233) * 43758.5453)
        return SimpleNamespace(
            grain=(g - 0.5).astype(np.float32), vig_base=ux * (1 - ux) * uy * (1 - uy) * 16.0
        )

    return cached(("noir", w, h), build)


def _vignette(w: int, h: int, amt: float) -> np.ndarray:
    base = _static(w, h).vig_base
    return cached(
        ("noir-vig", w, h, round(amt, 4)), lambda: (base ** (0.3 + 0.4 * amt)).astype(np.float32)
    )


def apply(img: np.ndarray, t_s: float, params: Mapping[str, float] | None = None) -> np.ndarray:
    img = check_image(img)
    p = resolve(PARAMS, params)
    h, w = img.shape[:2]
    st = _static(w, h)
    contrast = 1.0 + p["contrastAmt"]
    lut = np.clip(((np.arange(256) / 255.0 - 0.5) * contrast + 0.5) * 255.0, 0, 255)
    y = cv2.LUT(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), np.rint(lut).astype(np.uint8))
    yf = cv2.addWeighted(y.astype(np.float32), 1.0, st.grain, 0.08 * p["grainAmt"] * 255.0, 0.0)
    yf *= _vignette(w, h, p["vignetteAmt"])
    return cv2.LUT(gray3(to_u8(yf)), lut3(np.arange(256)[:, None] * _SEPIA_BGR[None, :]))
