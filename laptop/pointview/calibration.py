"""Load/save per-rig calibration: camera intrinsics JSON and the camera-to-display offset."""

from __future__ import annotations

import json
from pathlib import Path

from pointview import world
from pointview.interfaces import Calibration, CameraIntrinsics


def load_intrinsics(path: Path) -> CameraIntrinsics:
    """JSON with fx, fy, cx, cy, width, height and optional dist (as written by
    tools/calibrate_camera.py)."""
    d = json.loads(Path(path).read_text())
    return CameraIntrinsics(
        float(d["fx"]),
        float(d["fy"]),
        float(d["cx"]),
        float(d["cy"]),
        int(d["width"]),
        int(d["height"]),
        tuple(float(x) for x in d.get("dist", ())),
    )


def save_intrinsics(intr: CameraIntrinsics, path: Path, **extra) -> None:
    data = {
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.cx,
        "cy": intr.cy,
        "width": intr.width,
        "height": intr.height,
        "dist": list(intr.dist),
        **extra,
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def parse_camera_offset(text: str | None) -> Calibration:
    """ "yaw,pitch[,roll]" in degrees: how far the camera's axis is rotated from the display's center
    (right / up / clockwise positive)."""
    if not text:
        return Calibration()
    parts = [float(x) for x in text.split(",")]
    if len(parts) not in (2, 3):
        raise ValueError("--camera-offset wants yaw,pitch[,roll] in degrees")
    return Calibration(world.quat_from_yaw_pitch_roll(*parts))
