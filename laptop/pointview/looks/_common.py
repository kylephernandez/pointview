"""Shared helpers for the CPU looks: parameter handling, per-size caches, seeded noise.

Everything here is vectorised numpy / OpenCV. Looks are pure functions of (image, t_s, params);
the only state is a small cache of masks and lookup tables keyed by image size and parameters.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

import cv2
import numpy as np

_CACHE: OrderedDict[tuple, Any] = OrderedDict()
_CACHE_MAX = 32


def cached[T](key: tuple, factory: Callable[[], T]) -> T:
    """Tiny LRU for per-size masks and LUTs."""
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    val = factory()
    _CACHE[key] = val
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return val


def resolve(
    defaults: Mapping[str, tuple[float, float, float]], params: Mapping[str, float] | None
) -> dict[str, float]:
    """Merge ``params`` over the defaults. ``defaults`` maps name -> (default, min, max); unknown
    names are ignored and values are clamped to their range."""
    out = {k: v[0] for k, v in defaults.items()}
    for k, v in (params or {}).items():
        if k in defaults:
            lo, hi = defaults[k][1], defaults[k][2]
            out[k] = min(max(float(v), lo), hi)
    return out


def check_image(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 uint8 BGR image, got {img.shape} {img.dtype}")
    return np.ascontiguousarray(img)


def to_u8(x: np.ndarray) -> np.ndarray:
    """float32 array (0..255 scale) -> uint8 with rounding and saturation."""
    return cv2.convertScaleAbs(cv2.max(x, 0.0))


def smoothstep(e0: float, e1: float, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def frac(x):
    return x - np.floor(x)


def hash21(x, y):
    """GLSL ``hash(vec2)`` used by God's Eye View's shaders (Hoskins hash12), on arrays."""
    fx = frac(np.asarray(x, dtype=np.float64) * 0.1031)
    fy = frac(np.asarray(y, dtype=np.float64) * 0.1031)
    d = fx * (fy + 33.33) + fy * (fx + 33.33) + fx * (fx + 33.33)
    return frac(((fx + d) + (fy + d)) * (fx + d))


def time_seed(t_s: float, salt: int = 0) -> int:
    return (round(t_s * 1000.0) * 2654435761 + salt * 97) & 0xFFFFFFFF


def white_noise_u8(h: int, w: int, t_s: float, salt: int = 0) -> np.ndarray:
    """Per-pixel uniform noise, a new field for every millisecond of ``t_s``."""
    rng = np.random.default_rng(time_seed(t_s, salt))
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def noise_grid(cx: int, cy: int, seed: int) -> np.ndarray:
    return cached(
        ("grid", cx, cy, seed),
        lambda: np.random.default_rng(seed).integers(0, 256, size=(cy, cx), dtype=np.uint8),
    )


def value_noise(h: int, w: int, cx: int, cy: int, ox: float, oy: float, seed: int) -> np.ndarray:
    """Smooth tiling noise (uint8) with ``cx`` x ``cy`` cells across the image, scrolled by
    (ox, oy) cells. Bilinear instead of GLSL's smoothstep interpolation."""
    grid = noise_grid(cx, cy, seed)
    m = np.array([[cx / w, 0.0, ox], [0.0, cy / h, oy]], dtype=np.float64)
    return cv2.warpAffine(
        grid,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_WRAP,
    )


def uv_grid(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre normalised coordinates, (h,w) float64 each; v grows downward."""
    ux = np.broadcast_to((np.arange(w, dtype=np.float64) + 0.5)[None, :] / w, (h, w))
    uy = np.broadcast_to((np.arange(h, dtype=np.float64) + 0.5)[:, None] / h, (h, w))
    return ux, uy


def snap_maps(
    w: int, h: int, pix: float, ux: np.ndarray, uy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-point nearest-neighbour remap tables for GLSL's
    ``floor(uv * dims / pix) * pix / dims`` sampling at normalised coordinates (ux, uy).
    Coordinates outside 0..1 sample black (BORDER_CONSTANT)."""
    if pix > 1.0:
        sx = np.floor(ux * w / pix) * pix
        sy = np.floor(uy * h / pix) * pix
    else:
        sx, sy = ux * w - 0.5, uy * h - 0.5
    outside = (ux < 0) | (ux > 1) | (uy < 0) | (uy > 1)
    sx = np.where(outside, -10.0, sx).astype(np.float32)
    sy = np.where(outside, -10.0, sy).astype(np.float32)
    return cv2.convertMaps(sx, sy, cv2.CV_16SC2, nninterpolation=True)


def linear_maps(mx: np.ndarray, my: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cv2.convertMaps(mx.astype(np.float32), my.astype(np.float32), cv2.CV_16SC2)


def gray3(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def lut3(table_bgr: np.ndarray) -> np.ndarray:
    """(256, 3) array -> the (256, 1, 3) uint8 table cv2.LUT wants for a 3-channel image."""
    return np.ascontiguousarray(np.clip(np.rint(table_bgr), 0, 255).astype(np.uint8)[:, None, :])


def mask3(mask01: np.ndarray) -> np.ndarray:
    """float 0..1, (H,W) or (H,W,3) -> uint8 0..255 (H,W,3)."""
    m = np.clip(mask01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(m if m.ndim == 3 else np.repeat(m[:, :, None], 3, axis=2))


def mul_mask(img: np.ndarray, m3: np.ndarray) -> np.ndarray:
    return cv2.multiply(img, m3, scale=1.0 / 255.0)


def blur_small(gray_u8: np.ndarray, down: int, sigma: float, ksize: int) -> np.ndarray:
    """Downsample by ``down``, Gaussian-blur with the kernel GLSL's tap loop uses, upsample."""
    h, w = gray_u8.shape
    small = cv2.resize(
        gray_u8, (max(1, w // down), max(1, h // down)), interpolation=cv2.INTER_AREA
    )
    small = cv2.GaussianBlur(small, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def row_factor(f: np.ndarray, w: int) -> np.ndarray:
    """Per-row 0..1 factor (len h) -> uint8 (h,w,3) multiplier image."""
    col = np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)[:, None, None]
    return cv2.resize(np.repeat(col, 3, axis=2), (w, len(f)), interpolation=cv2.INTER_NEAREST)
