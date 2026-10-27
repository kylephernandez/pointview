"""Shared helpers: access to tools/make_fixtures.py and a manual clock."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[2] / "tools" / "make_fixtures.py"
_spec = importlib.util.spec_from_file_location("make_fixtures", _TOOLS)
assert _spec and _spec.loader
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)


def analytic_quat(t: float):
    return mf.quat_yaw_pitch(*mf.head_yaw_pitch(t))


def angle_between_deg(a, b) -> float:
    d = abs(sum(x * y for x, y in zip(a, b, strict=True)))
    return math.degrees(2 * math.acos(min(1.0, d)))


class ManualClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t
