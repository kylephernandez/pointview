"""Screen-marker tracking, in the existing P/C/H/T frames (contracts §12).

Motion agreement is a rejection heuristic, not authentication. Camera frames stay in memory.
Only accepted observations refresh the inertial alignment; translation expires quickly.
"""

from __future__ import annotations

import math
import secrets
from collections import deque
from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np

from pointview import world
from pointview.interfaces import Calibration, CameraIntrinsics, Frame, PhoneState, Quat, Vec3

DICTIONARY = cv2.aruco.DICT_6X6_1000
MAX_FRAME_AGE_S = 0.2
MAX_SAMPLE_GAP_S = 0.08
POSITION_TTL_S = 0.2


@dataclass(frozen=True)
class TrackingTarget:
    session: str
    marker_id: int

    @classmethod
    def create(cls) -> TrackingTarget:
        return cls(secrets.token_hex(16), secrets.randbelow(1000))

    def message(self) -> dict:
        pixels = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(DICTIONARY), self.marker_id, 8
        )
        return {
            "session": self.session,
            "dictionary": "DICT_6X6_1000",
            "marker_id": self.marker_id,
            "cells": ["".join("1" if p else "0" for p in row) for row in pixels],
        }


@dataclass(frozen=True)
class Observation:
    t: float
    #: P -> T, with head motion at exposure removed.
    quat: Quat
    #: Marker centre relative to the camera in T, metres if size was supplied.
    position: Vec3 | None
    reprojection_px: float


def rotation_quat(matrix: np.ndarray) -> Quat:
    axis, _ = cv2.Rodrigues(matrix)
    angle = float(np.linalg.norm(axis))
    return world.quat_from_axis_angle(tuple(axis.ravel()), angle)


def rotation_error(a: Quat, b: Quat) -> float:
    dot = min(1.0, abs(sum(x * y for x, y in zip(a, b, strict=True))))
    return math.degrees(2 * math.acos(dot))


def sample_at(samples: list[PhoneState], t: float) -> PhoneState | None:
    """Strictly bracket the exposure time; never compare a stale or future-only phone pose."""
    from dataclasses import replace

    for a, b in pairwise(samples):
        if a.t <= t <= b.t and 0 < b.t - a.t <= MAX_SAMPLE_GAP_S:
            f = (t - a.t) / (b.t - a.t)
            return replace(
                a,
                t=t,
                quat=world.quat_slerp(a.quat, b.quat, f),
                omega=tuple(x + f * (y - x) for x, y in zip(a.omega, b.omega, strict=True)),
            )
    return None


def gyro_delta(samples: list[PhoneState], start: float, end: float) -> Quat | None:
    a, b = sample_at(samples, start), sample_at(samples, end)
    if a is None or b is None:
        return None
    seq = [a, *(s for s in samples if start < s.t < end), b]
    q: Quat = (1.0, 0.0, 0.0, 0.0)
    for a, b in pairwise(seq):
        dt = b.t - a.t
        if not 0 < dt <= MAX_SAMPLE_GAP_S:
            return None
        omega = tuple((x + y) / 2 for x, y in zip(a.omega, b.omega, strict=True))
        rate = math.sqrt(sum(x * x for x in omega))
        q = world.quat_normalize(world.quat_mul(q, world.quat_from_axis_angle(omega, rate * dt)))
    return q


class MarkerDetector:
    def __init__(self) -> None:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(DICTIONARY), params
        )

    def detect(
        self,
        frame: Frame,
        intr: CameraIntrinsics,
        target: TrackingTarget,
        size_m: float | None,
        head: Quat,
        calibration: Calibration,
    ) -> Observation | None:
        if (frame.bgr.shape[1], frame.bgr.shape[0]) != (intr.width, intr.height):
            return None
        # Bound CPU work; solvePnP still uses original full-resolution pixel coordinates.
        scale = min(1.0, 960 / intr.width)
        image = frame.bgr if scale == 1 else cv2.resize(frame.bgr, None, fx=scale, fy=scale)
        corners, ids, _ = self.detector.detectMarkers(image)
        matches = [] if ids is None else np.flatnonzero(ids.ravel() == target.marker_id)
        if (
            len(matches) != 1
        ):  # duplicate IDs are ambiguous, including two copies on different screens
            return None
        pixels = corners[matches[0]].reshape(4, 2).astype(np.float64) / scale
        if min(np.linalg.norm(pixels[(i + 1) % 4] - pixels[i]) for i in range(4)) < 20:
            return None
        s = (size_m or 0.06) / 2
        points = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
        k = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array(intr.dist, dtype=np.float64) if intr.dist else None
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            points, pixels, k, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            return None
        solutions = []
        basis = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # C -> nominal H
        for rvec, tvec in zip(rvecs, tvecs, strict=True):
            rot, _ = cv2.Rodrigues(rvec)
            pos = tvec.ravel()
            if not np.isfinite(rot).all() or not np.isfinite(pos).all():
                continue
            if np.any((rot @ points.T + pos[:, None])[2] <= 0) or float(rot[:, 2] @ pos) >= 0:
                continue
            if size_m is not None and not 0.1 <= np.linalg.norm(pos) <= 2.0:
                continue
            projected, _ = cv2.projectPoints(points, rvec, tvec, k, dist)
            error = float(np.sqrt(np.mean((projected.reshape(4, 2) - pixels) ** 2)))
            if error > 2.0:
                continue
            q = world.quat_mul(
                head, world.quat_mul(calibration.camera_to_head_quat, rotation_quat(basis @ rot))
            )
            position = world.quat_rotate(
                head, world.camera_to_head(tuple(pos), calibration.camera_to_head_quat)
            )
            solutions.append(
                Observation(frame.t, world.quat_normalize(q), position if size_m else None, error)
            )
        solutions.sort(key=lambda o: o.reprojection_px)
        if not solutions:
            return None
        if (
            len(solutions) > 1
            and solutions[1].reprojection_px - solutions[0].reprojection_px < 0.3
            and rotation_error(solutions[0].quat, solutions[1].quat) > 10
        ):
            return None  # planar pose ambiguity: wait for a better view instead of flipping the ray
        return solutions[0]


class PhoneTracker:
    def __init__(self, target: TrackingTarget) -> None:
        self.target = target
        self.detector = MarkerDetector()
        self.observations: deque[Observation] = deque()
        self.last_frame_t = -math.inf
        self.accepted: Observation | None = None
        self.reference_quat: Quat | None = None  # R -> T
        self.verified = False

    def reset_verification(self) -> None:
        self.observations.clear()
        self.verified = False

    def observe(self, obs: Observation, samples: list[PhoneState]) -> bool:
        """Require a moving 0.25–0.8 s camera/gyro agreement window before acquiring/reacquiring."""
        phone = sample_at(samples, obs.t)
        if phone is None:
            self.reset_verification()
            return False
        if self.observations and not 0 < obs.t - self.observations[-1].t <= MAX_FRAME_AGE_S:
            self.reset_verification()
        self.observations.append(obs)
        while self.observations and obs.t - self.observations[0].t > 0.8:
            self.observations.popleft()
        if len(self.observations) < 4 or obs.t - self.observations[0].t < 0.25:
            return False
        first = self.observations[0]
        delta = gyro_delta(samples, first.t, obs.t)
        first_phone = sample_at(samples, first.t)
        if delta is None or first_phone is None:
            self.reset_verification()
            return False
        vision = world.quat_mul(world.quat_conj(first.quat), obs.quat)
        motion = rotation_error((1.0, 0.0, 0.0, 0.0), delta)
        attitude = world.quat_mul(world.quat_conj(first_phone.quat), phone.quat)
        tolerance = 2 + 0.15 * motion
        if rotation_error(vision, delta) > tolerance or rotation_error(attitude, delta) > tolerance:
            self.reset_verification()
            return False
        if not self.verified and motion < 6:
            return False  # stationary agreement cannot establish that this is the moving phone
        self.verified = True
        self.accepted = obs
        self.reference_quat = world.quat_normalize(
            world.quat_mul(obs.quat, world.quat_conj(phone.quat))
        )
        return True

    def update(
        self,
        glasses,
        samples: list[PhoneState],
        calibration: Calibration,
        size_m: float | None,
        now: float,
    ) -> bool:
        frame = glasses.latest_frame()
        if frame is None or not 0 <= now - frame.t <= MAX_FRAME_AGE_S:
            self.reset_verification()
            return False
        if frame.t <= self.last_frame_t or frame.t - self.last_frame_t < 0.05:
            return False
        self.last_frame_t = frame.t
        pose = glasses.pose_at(frame.t)
        if pose is None:
            self.reset_verification()
            return False
        obs = self.detector.detect(
            frame, glasses.camera_intrinsics, self.target, size_m, pose.quat, calibration
        )
        if obs is None:
            self.reset_verification()
            return False
        return self.observe(obs, samples)

    def state(self, now: float) -> str:
        if self.accepted is not None and self.verified and now - self.accepted.t < MAX_FRAME_AGE_S:
            return "tracked"
        if self.reference_quat is not None:
            return "coasting"
        return "verifying" if self.observations else "searching"

    def origin_world(self, now: float, north: float) -> Vec3 | None:
        if self.accepted is None or not self.verified or now - self.accepted.t > POSITION_TTL_S:
            return None
        pos = self.accepted.position
        if pos is None:
            return None
        # No head translation sensor: use the camera origin as the eye origin, fresh positions only.
        return world.head_to_world(pos, (1.0, 0.0, 0.0, 0.0), north)
