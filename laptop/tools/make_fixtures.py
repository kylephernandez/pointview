"""Generate the shared test fixtures in laptop/tests/fixtures/ (deterministic).

    python tools/make_fixtures.py

Outputs:
  pose_track.csv   t,qw,qx,qy,qz at 200 Hz for 10 s. Head→tracking quaternion (interfaces.py frames).
  sample.mp4       960x540 @ 30 fps, 6 s. A synthetic world (sky, ground, horizon, numbered
                   world-fixed markers) rendered through the pose track, so frame i at t = i/30 s
                   is exactly what a camera at pose_track(t) would see.
  scene.json       Camera intrinsics, video timing, and the markers' tracking-frame directions.

Self-contained on purpose: it does not import pointview.world (that module is tested against it).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

POSE_HZ = 200
POSE_SECONDS = 10.0
VIDEO_FPS = 30
VIDEO_SECONDS = 6.0
W, H = 960, 540
HFOV_DEG = 70.0
FX = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))

# Tracking-frame directions of world-fixed markers (yaw clockwise from tracking +y, pitch up).
MARKERS = [
    {"id": 1, "yaw_deg": 0.0, "pitch_deg": 0.0, "color_bgr": [0, 200, 255]},
    {"id": 2, "yaw_deg": 15.0, "pitch_deg": -5.0, "color_bgr": [255, 220, 0]},
    {"id": 3, "yaw_deg": -20.0, "pitch_deg": 3.0, "color_bgr": [120, 255, 140]},
    {"id": 4, "yaw_deg": 30.0, "pitch_deg": 8.0, "color_bgr": [255, 120, 255]},
]


def head_yaw_pitch(t: float) -> tuple[float, float]:
    """Synthetic head motion: slow ±25° yaw sweep plus a small pitch nod."""
    yaw = 25.0 * math.sin(2 * math.pi * 0.1 * t)
    pitch = 4.0 * math.sin(2 * math.pi * 0.23 * t)
    return yaw, pitch


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_yaw_pitch(yaw_deg: float, pitch_deg: float):
    # Clockwise yaw = rotation about +z by -yaw; then pitch up about the new +x.
    hy = math.radians(-yaw_deg) / 2
    hp = math.radians(pitch_deg) / 2
    qz = (math.cos(hy), 0.0, 0.0, math.sin(hy))
    qx = (math.cos(hp), math.sin(hp), 0.0, 0.0)
    return qmul(qz, qx)


def rotmat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def direction(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p)])


def project(q, d: np.ndarray) -> tuple[float, float] | None:
    h = rotmat(q).T @ d  # tracking → head
    cx_, cy_, cz_ = h[0], -h[2], h[1]  # head → camera (x right, y down, z forward)
    if cz_ <= 1e-6:
        return None
    return W / 2 + FX * cx_ / cz_, H / 2 + FX * cy_ / cz_


def render(t: float) -> np.ndarray:
    q = quat_yaw_pitch(*head_yaw_pitch(t))
    img = np.zeros((H, W, 3), np.uint8)
    # Sky/ground split along the projected horizon (sample two horizon points).
    yaw0, _ = head_yaw_pitch(t)
    a = project(q, direction(yaw0 - 30, 0.0))
    b = project(q, direction(yaw0 + 30, 0.0))
    assert a and b
    slope = (b[1] - a[1]) / (b[0] - a[0])
    ys = a[1] + slope * (np.arange(W) - a[0])
    rows = np.arange(H)[:, None]
    ground = rows > ys[None, :]
    img[~ground] = (70, 40, 20)
    img[ground] = (40, 60, 50)
    cv2.line(img, (0, int(ys[0])), (W - 1, int(ys[-1])), (200, 200, 200), 2, cv2.LINE_AA)
    for m in MARKERS:
        p = project(q, direction(m["yaw_deg"], m["pitch_deg"]))
        if p is None:
            continue
        c = (round(p[0]), round(p[1]))
        cv2.circle(img, c, 18, m["color_bgr"], -1, cv2.LINE_AA)
        cv2.putText(
            img,
            str(m["id"]),
            (c[0] - 8, c[1] + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "pose_track.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "qw", "qx", "qy", "qz"])
        for i in range(int(POSE_SECONDS * POSE_HZ) + 1):
            t = i / POSE_HZ
            wr.writerow([f"{t:.4f}", *(f"{c:.8f}" for c in quat_yaw_pitch(*head_yaw_pitch(t)))])

    vw = cv2.VideoWriter(
        str(OUT / "sample.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (W, H)
    )
    assert vw.isOpened(), "OpenCV cannot write mp4v"
    n = int(VIDEO_SECONDS * VIDEO_FPS)
    for i in range(n):
        vw.write(render(i / VIDEO_FPS))
    vw.release()

    scene = {
        "description": "Synthetic fixture world; see tools/make_fixtures.py",
        "video": {
            "file": "sample.mp4",
            "fps": VIDEO_FPS,
            "frames": n,
            "width": W,
            "height": H,
            "frame_time": "frame i was rendered at t = i / fps on the pose_track clock",
        },
        "pose_track": {
            "file": "pose_track.csv",
            "hz": POSE_HZ,
            "seconds": POSE_SECONDS,
            "head_yaw_pitch": "yaw = 25 sin(2π·0.1 t), pitch = 4 sin(2π·0.23 t), deg",
        },
        "intrinsics": {
            "fx": FX,
            "fy": FX,
            "cx": W / 2,
            "cy": H / 2,
            "width": W,
            "height": H,
            "dist": [],
        },
        "markers_tracking_frame": MARKERS,
    }
    (OUT / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")


if __name__ == "__main__":
    main()
