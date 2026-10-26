"""The hardware bring-up tools, exercised against mocks/synthetic data."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from pointview import world
from pointview.calibration import load_intrinsics, parse_camera_offset

LAPTOP = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, LAPTOP / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_calibrate_camera_recovers_synthetic_intrinsics(tmp_path):
    cal = load_tool("calibrate_camera")
    cols, rows, sq = 9, 6, 30.0
    fx, cx, cy, w, h = 800.0, 640.0, 360.0, 1280, 720
    k = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
    board = np.zeros(((rows + 1) * 60 + 120, (cols + 1) * 60 + 120), np.uint8) + 255
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[60 + r * 60 : 120 + r * 60, 60 + c * 60 : 120 + c * 60] = 0
    # Board plane points (mm) for its pixel grid: 60 px = one 30 mm square, origin at the first corner.
    paths = []
    for i, (rx, ry, tz) in enumerate(
        [
            (0.0, 0.0, 600),
            (0.3, 0.0, 650),
            (-0.3, 0.2, 700),
            (0.2, -0.3, 600),
            (0.0, 0.4, 750),
            (-0.25, -0.25, 650),
            (0.35, 0.3, 800),
        ]
    ):
        rvec = np.array([rx, ry, 0.0])
        tvec = np.array([-(cols / 2) * sq, -(rows / 2) * sq, tz])
        src = np.float32(
            [[0, 0], [board.shape[1], 0], [0, board.shape[0]], [board.shape[1], board.shape[0]]]
        )
        mm = np.float32([[(x - 120) / 60 * sq, (y - 120) / 60 * sq, 0] for x, y in src])
        dst, _ = cv2.projectPoints(mm, rvec, tvec, k, None)
        hmat = cv2.getPerspectiveTransform(src, dst.reshape(4, 2).astype(np.float32))
        img = cv2.warpPerspective(board, hmat, (w, h), borderValue=255)
        paths.append(tmp_path / f"v{i}.png")
        cv2.imwrite(str(paths[-1]), img)
    intr, rms, n = cal.solve(paths, cols, rows, sq)
    assert n >= 5 and rms < 1.0
    assert abs(intr.fx - fx) / fx < 0.03 and abs(intr.cx - cx) < 15 and abs(intr.cy - cy) < 15
    out = tmp_path / "intr.json"
    cal.save_intrinsics(intr, out)
    assert load_intrinsics(out) == intr


def test_camera_offset_parsing():
    assert parse_camera_offset(None).camera_to_head_quat == (1.0, 0.0, 0.0, 0.0)
    q = parse_camera_offset("1.5,-0.5").camera_to_head_quat
    y, p = world.yaw_pitch_from_vector(world.quat_rotate(q, (0.0, 1.0, 0.0)))
    assert abs(y - 1.5) < 1e-9 and abs(p + 0.5) < 1e-9


def test_glasses_check_runs_on_mock():
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    out = subprocess.run(
        [sys.executable, "tools/glasses_check.py", "--mock", "--seconds", "1", "--crosshair"],
        cwd=LAPTOP,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "yaw" in out.stdout and "camera 960x540" in out.stdout


def test_glasses_check_head_angles():
    gc = load_tool("glasses_check")
    y, p, r = gc.head_angles(world.quat_from_yaw_pitch_roll(30.0, 10.0, 5.0))
    assert abs(y - 30) < 1e-6 and abs(p - 10) < 1e-6 and abs(r - 5) < 1e-6


def test_phone_monitor_with_fake_phone():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mon = subprocess.Popen(
        [
            sys.executable,
            "tools/phone_monitor.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--seconds",
            "3",
        ],
        cwd=LAPTOP,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert "listening" in mon.stdout.readline()  # wait until the server is bound
    fake = subprocess.run(
        [
            sys.executable,
            "tools/fake_phone.py",
            "--url",
            f"ws://127.0.0.1:{port}",
            "--script",
            "sweep",
            "--duration",
            "2",
        ],
        cwd=LAPTOP,
        capture_output=True,
        text=True,
        timeout=30,
    )
    out, _ = mon.communicate(timeout=30)
    assert fake.returncode == 0, fake.stderr
    assert "Hz" in out and "summary: messages" in out and "invalid 0" in out


def test_inspect_sdk_lib_verdicts():
    ins = load_tool("inspect_sdk_lib").inspect
    imp = b"!<arch>\n/ \x00__imp_xr_device_provider_create\x00glasses.dll\x00get_gl_pose_carina\x00"
    v = ins(imp)
    assert v["verdict"].startswith("import library") and v["dlls"] == ["glasses.dll"]
    assert "xr_device_provider_create" in v["api"]
    static = b"!<arch>\n/ \x00provider.obj/\x00xr_device_provider_create\x00"
    assert ins(static)["verdict"].startswith("static library")
    assert ins(b"MZ...")["verdict"].startswith("not a COFF")
