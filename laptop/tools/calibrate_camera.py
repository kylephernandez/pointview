"""Checkerboard calibration of the glasses' front camera → intrinsics JSON for ``--intrinsics``.

    python tools/calibrate_camera.py capture --camera-index 1 --out calib_frames/   # press SPACE, q
    python tools/calibrate_camera.py solve calib_frames/*.png --cols 9 --rows 6 --square-mm 25 \
        --out intrinsics.json

Print a checkerboard (``cols`` x ``rows`` *inner* corners), capture 15-25 views covering the whole image
at different angles, then solve. Calibration images are only for this step: delete them afterwards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointview.calibration import save_intrinsics
from pointview.interfaces import CameraIntrinsics


def solve(
    images: list[Path], cols: int, rows: int, square_mm: float
) -> tuple[CameraIntrinsics, float, int]:
    """Returns (intrinsics, RMS reprojection error in px, views used)."""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    obj_pts, img_pts, size = [], [], None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for path in images:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        size = (gray.shape[1], gray.shape[0])
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_pts.append(objp)
        img_pts.append(corners)
    if len(obj_pts) < 3 or size is None:
        raise SystemExit(f"need >= 3 views with the board found, got {len(obj_pts)}")
    rms, k, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    d = tuple(float(x) for x in dist.ravel()[:5])
    intr = CameraIntrinsics(k[0, 0], k[1, 1], k[0, 2], k[1, 2], size[0], size[1], d)
    return intr, float(rms), len(obj_pts)


def capture(camera_index: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("camera read failed")
        cv2.imshow("calibrate: SPACE saves, q quits", cv2.resize(frame, None, fx=0.5, fy=0.5))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            cv2.imwrite(str(out / f"view_{n:02d}.png"), frame)
            n += 1
            print(f"saved {n}")
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--camera-index", type=int, required=True)
    c.add_argument("--out", type=Path, default=Path("calib_frames"))
    s = sub.add_parser("solve")
    s.add_argument("images", nargs="+", type=Path)
    s.add_argument("--cols", type=int, default=9)
    s.add_argument("--rows", type=int, default=6)
    s.add_argument("--square-mm", type=float, default=25.0)
    s.add_argument("--out", type=Path, default=Path("intrinsics.json"))
    a = p.parse_args(argv)
    if a.cmd == "capture":
        capture(a.camera_index, a.out)
        return 0
    intr, rms, n = solve(a.images, a.cols, a.rows, a.square_mm)
    hfov = 2 * np.degrees(np.arctan(intr.width / 2 / intr.fx))
    save_intrinsics(intr, a.out, rms_px=rms, views=n, hfov_deg=float(hfov))
    print(
        f"{n} views, RMS {rms:.3f} px, fx={intr.fx:.1f} fy={intr.fy:.1f}, HFOV {hfov:.1f}° → {a.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
