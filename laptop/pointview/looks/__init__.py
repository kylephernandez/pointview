"""CPU reference versions of the God's Eye View sensor looks (contracts section 8).

Each look module exposes ``apply(img_bgr_uint8, t_s, params) -> img_bgr_uint8`` and a ``PARAMS``
table of ``name -> (default, min, max)`` taken from God's Eye View's uniforms. ``apply_look``
dispatches by name; ``"normal"`` is the identity. Output depends only on the image, ``t_s`` and
``params`` (noise is seeded from ``t_s``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from pointview.interfaces import LOOKS, Look
from pointview.looks import crt, flir, noir, nvg, snow
from pointview.looks._common import check_image

_MODULES = {"crt": crt, "nvg": nvg, "flir": flir, "noir": noir, "snow": snow}
_APPLY: dict[str, Callable[..., np.ndarray]] = {k: m.apply for k, m in _MODULES.items()}

__all__ = ["LOOKS", "apply_look", "look_defaults", "look_params"]


def look_params(look: Look) -> dict[str, tuple[float, float, float]]:
    """name -> (default, min, max) for a look's uniforms ({} for ``"normal"``)."""
    mod = _MODULES.get(look)
    return dict(mod.PARAMS) if mod else {}


def look_defaults(look: Look) -> dict[str, float]:
    return {k: v[0] for k, v in look_params(look).items()}


def apply_look(
    look: Look, img: np.ndarray, t_s: float, params: Mapping[str, float] | None = None
) -> np.ndarray:
    """Apply ``look`` to a BGR uint8 image. ``"normal"`` returns ``img`` unchanged."""
    if look == "normal":
        return img
    fn = _APPLY.get(look)
    if fn is None:
        raise ValueError(f"unknown look {look!r}; expected one of {LOOKS}")
    check_image(img)
    return fn(img, float(t_s), params)
