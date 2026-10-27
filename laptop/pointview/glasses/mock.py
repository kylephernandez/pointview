"""MockGlasses: a ``GlassesSource`` that replays a video file and a pose CSV (no hardware).

Time model: fixture time 0 maps to ``clock()`` at ``start()`` (``t0``). Video frame counter ``k`` is
stamped ``t0 + k / fps`` and shows file frame ``k % n_frames`` when looping. The pose track (CSV
``t,qw,qx,qy,qz``) is fed into a :class:`PoseBuffer` with sample times mapped to ``t0 + cycle * L + t_j``
where ``L`` is the track length plus one sample interval (so a looped cycle has no duplicate
timestamps). The video and the pose track wrap independently, each by its own length, so after the
first video loop frames no longer match the pose at the same fixture time (by contract, §3.1).

At the pose-track wrap the quaternion jumps (the fixture pitch is not periodic over 10 s). The jump is
spread over one sample interval by the interpolation and is flagged as a discontinuity, so it never
feeds the velocity used for extrapolation.

Frames live in memory only.
"""

from __future__ import annotations

import csv
import math
import threading
import time
from pathlib import Path

import cv2

from pointview.glasses.pose_buffer import PoseBuffer
from pointview.interfaces import IDENTITY_QUAT, CameraIntrinsics, Clock, Frame, Pose, Quat, now

#: With realtime=False, poses this far ahead of the current frame are pre-loaded so that
#: interpolation (not extrapolation) is used for the frame time.
_LOOKAHEAD_S = 0.1

#: Size of the mock tracking frames. Deliberately not the front camera's size, so callers cannot
#: assume the two are equal (the real tracking cameras differ from the front camera).
TRACKING_SIZE = (640, 480)


def load_pose_csv(path: Path) -> list[Pose]:
    with Path(path).open() as f:
        return [
            Pose(float(r["t"]), (float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])))
            for r in csv.DictReader(f)
        ]


class MockGlasses:
    def __init__(
        self,
        video_path: Path,
        pose_csv: Path,
        intrinsics: CameraIntrinsics,
        display_size: tuple[int, int] = (1920, 1080),
        display_fov_deg: tuple[float, float] = (46.0, 27.0),
        clock: Clock = now,
        loop: bool = True,
        realtime: bool = True,
    ) -> None:
        self._video_path = Path(video_path)
        self._intrinsics = intrinsics
        self._display_size = display_size
        self._display_fov = display_fov_deg
        self._clock = clock
        self._loop = loop
        self._realtime = realtime

        samples = load_pose_csv(Path(pose_csv))
        if not samples:
            raise ValueError(f"no pose samples in {pose_csv}")
        self._pose_t = [s.t - samples[0].t for s in samples]
        self._pose_q: list[Quat] = [s.quat for s in samples]
        if len(samples) > 1:
            dts = sorted(b - a for a, b in zip(self._pose_t, self._pose_t[1:], strict=False))
            interval = dts[len(dts) // 2]
        else:
            interval = 1.0
        self._pose_cycle_s = self._pose_t[-1] + interval

        self._buffer = PoseBuffer()
        self._lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None
        self._fps = 30.0
        self._n_frames = 0
        self._t0 = 0.0
        self._started = False
        self._next_k = 0  # next video frame counter to produce
        self._next_pose = (0, 0)  # (cycle, index) of the next pose sample to feed
        self._pose_done = False
        self._frame: Frame | None = None
        self._tracking_cache: tuple[Frame, Frame] | None = None  # (source frame, tracking frame)
        #: Every set_tracking_exposure call, as (exposure_ms, gain); the mock only records them.
        self.tracking_exposure_calls: list[tuple[float | None, float | None]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ---------------------------------------------------------------- GlassesSource properties

    @property
    def camera_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    @property
    def display_size(self) -> tuple[int, int]:
        return self._display_size

    @property
    def display_fov_deg(self) -> tuple[float, float]:
        return self._display_fov

    # ------------------------------------------------------- TrackingCameraSource (contract s.9)

    @property
    def tracking_camera_intrinsics(self) -> CameraIntrinsics:
        """The front intrinsics rescaled to the (stretched) 640x480 tracking frame: same view,
        different size and focal length."""
        i = self._intrinsics
        w, h = TRACKING_SIZE
        sx, sy = w / i.width, h / i.height
        return CameraIntrinsics(i.fx * sx, i.fy * sy, i.cx * sx, i.cy * sy, w, h, i.dist)

    @property
    def tracking_camera_to_head_quat(self) -> Quat:
        return IDENTITY_QUAT

    def latest_tracking_frame(self) -> Frame | None:
        """The newest video frame as 640x480 gray replicated to BGR (same timestamp)."""
        src = self.latest_frame()
        if src is None:
            return None
        cached = self._tracking_cache
        if cached is not None and cached[0] is src:
            return cached[1]
        gray = cv2.cvtColor(src.bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, TRACKING_SIZE, interpolation=cv2.INTER_AREA)
        out = Frame(src.t, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        self._tracking_cache = (src, out)
        return out

    def set_tracking_exposure(self, exposure_ms: float | None, gain: float | None = None) -> None:
        """Recorded no-op."""
        self.tracking_exposure_calls.append((exposure_ms, gain))

    @property
    def t0(self) -> float:
        """Laptop time that fixture time 0 was mapped to (valid after start())."""
        return self._t0

    @property
    def fps(self) -> float:
        return self._fps

    # ------------------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._started:
            return
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {self._video_path}")
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else 30.0
        self._n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._n_frames <= 0:
            cap.release()
            self._cap = None
            raise RuntimeError(f"video {self._video_path} has no frames")
        self._buffer.clear()
        self._frame = None
        self._next_k = 0
        self._next_pose = (0, 0)
        self._pose_done = False
        self._t0 = self._clock()
        self._started = True
        self._feed_poses(self._t0 + (_LOOKAHEAD_S if not self._realtime else 0.0))
        if self._realtime:
            self._stop.clear()
            self._step()  # frame 0 is available as soon as start() returns
            self._thread = threading.Thread(target=self._run, name="MockGlasses", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._started = False

    def __enter__(self) -> MockGlasses:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ---------------------------------------------------------------------------------- data

    def latest_frame(self) -> Frame | None:
        if not self._started:
            return None
        if not self._realtime:
            self._advance_one()
        with self._lock:
            return self._frame

    def pose_at(self, t: float) -> Pose | None:
        return self._buffer.pose_at(t)

    # -------------------------------------------------------------------------------- internals

    def _run(self) -> None:
        while not self._stop.is_set():
            self._step()
            time.sleep(0.001)

    def _step(self) -> None:
        """Realtime tick: publish poses up to the clock, and the newest due video frame."""
        with self._lock:
            if not self._started:
                return
            t_now = self._clock()
            self._feed_poses(t_now)
            due = math.floor((t_now - self._t0) * self._fps + 1e-9)
            if due < self._next_k:
                return
            if not self._loop:
                due = min(due, self._n_frames - 1)
                if self._next_k > due:
                    return
            frame = None
            while self._next_k <= due:  # decode sequentially; keep only the newest
                frame = self._read_next()
                if frame is None:
                    break
            if frame is not None:
                self._frame = frame

    def _advance_one(self) -> None:
        with self._lock:
            if not self._started:
                return
            frame = self._read_next()
            if frame is not None:
                self._frame = frame
                self._feed_poses(frame.t + _LOOKAHEAD_S)

    def _read_next(self) -> Frame | None:
        """Decode video frame counter ``_next_k`` (caller holds the lock)."""
        k = self._next_k
        if k >= self._n_frames and not self._loop:
            return None
        if self._cap is None:
            return None
        idx = k % self._n_frames
        if idx == 0 and k > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, bgr = self._cap.read()
        if not ok:  # unexpected short file: rewind once
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = self._cap.read()
            if not ok:
                return None
        self._next_k = k + 1
        return Frame(self._t0 + k / self._fps, bgr)

    def _feed_poses(self, t_limit: float) -> None:
        cycle, j = self._next_pose
        n = len(self._pose_t)
        while not self._pose_done:
            t = self._t0 + cycle * self._pose_cycle_s + self._pose_t[j]
            if t > t_limit:
                break
            self._buffer.add(t, self._pose_q[j], discontinuity=(j == 0 and cycle > 0))
            j += 1
            if j >= n:
                j = 0
                cycle += 1
                if not self._loop:
                    self._pose_done = True
        self._next_pose = (cycle, j)
