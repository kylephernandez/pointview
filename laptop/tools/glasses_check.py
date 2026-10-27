"""M0 bring-up: read head pose + camera from the glasses, optionally draw a world-fixed crosshair.

    python tools/glasses_check.py                    # print yaw/pitch/roll, pose rate, camera frames
    python tools/glasses_check.py --crosshair --screen VITURE   # M0 test: crosshair stays put in the world
    python tools/glasses_check.py --mock --seconds 3 # same, against MockGlasses (no hardware)

Expected on hardware: level + facing forward ≈ yaw 0 / pitch 0 / roll 0 right after start; turning your
head right increases yaw; looking up increases pitch; tilting your head right increases roll and leaves
yaw/pitch nearly unchanged; pose rate ≥ 120 Hz.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pointview import world
from pointview.interfaces import Crosshair, StatusCorner, now


def head_angles(q) -> tuple[float, float, float]:
    """(yaw, pitch, roll) of the head in the tracking frame, degrees."""
    yaw, pitch = world.yaw_pitch_from_vector(world.quat_rotate(q, (0.0, 1.0, 0.0)))
    right = world.quat_rotate(q, (1.0, 0.0, 0.0))
    up = world.quat_rotate(q, (0.0, 0.0, 1.0))
    roll = math.degrees(math.atan2(-right[2], up[2]))
    return world.wrap_deg_signed(yaw), pitch, roll


def make_glasses(args):
    if args.mock:
        from pointview import mock

        return mock.make_glasses()
    from pointview.glasses import VitureGlasses

    return VitureGlasses(camera_index=args.camera_index)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mock", action="store_true")
    p.add_argument("--camera-index", type=int)
    p.add_argument("--seconds", type=float, default=0, help="0 = until Ctrl+C")
    p.add_argument("--crosshair", action="store_true", help="draw a world-fixed crosshair")
    p.add_argument("--screen", help="glasses display name substring")
    a = p.parse_args(argv)

    glasses = make_glasses(a)
    glasses.start()
    renderer = None
    if a.crosshair:
        from pointview.render import QtRenderer

        renderer = QtRenderer(a.screen, None, glasses.display_size)
    t0 = now()
    anchor = None
    last_print = 0.0
    try:
        while not a.seconds or now() - t0 < a.seconds:
            t = now()
            pose = glasses.pose_at(t)
            if pose is None:
                time.sleep(0.01)
                continue
            yaw, pitch, roll = head_angles(pose.quat)
            if anchor is None:
                anchor = (world.wrap_deg(yaw), pitch)  # the direction you faced at start
            if renderer is not None:
                pt = world.project_to_display(
                    anchor[0],
                    anchor[1],
                    pose.quat,
                    0.0,
                    glasses.display_size,
                    glasses.display_fov_deg,
                )
                items = [StatusCorner(((f"yaw {yaw:+.1f} pitch {pitch:+.1f}", (255, 255, 255)),))]
                if pt.in_front:
                    items.insert(0, Crosshair(pt.x, pt.y))
                renderer.show(items)
            if t - last_print >= 0.2:
                last_print = t
                frame = glasses.latest_frame()
                cam = (
                    f"{frame.bgr.shape[1]}x{frame.bgr.shape[0]} age {1000 * (t - frame.t):.0f} ms"
                    if frame
                    else "no frame"
                )
                rate = getattr(glasses, "measured_pose_rate_hz", lambda: float("nan"))()
                print(
                    f"yaw {yaw:+7.2f}  pitch {pitch:+6.2f}  roll {roll:+6.2f}  pose {rate:6.1f} Hz  camera {cam}",
                    flush=True,
                )
            time.sleep(1 / 120)
    except KeyboardInterrupt:
        pass
    finally:
        glasses.stop()
        if renderer is not None:
            renderer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
