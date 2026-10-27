"""Thread-safe pose ring buffer with timestamp interpolation (``GlassesSource.pose_at`` semantics)."""

from __future__ import annotations

import bisect
import threading

from pointview.glasses import _quat as q
from pointview.interfaces import Pose, Quat

#: Never extrapolate further than this past the newest sample (seconds). Then hold.
MAX_EXTRAPOLATION_S = 0.05
#: Angular velocity is measured over a baseline of at least this long when available (noise).
_MIN_VELOCITY_BASELINE_S = 0.004


class PoseBuffer:
    """Ring buffer of head poses keyed by laptop time.

    ``add`` may be called from a producer thread while ``pose_at`` is called from others.
    Samples must arrive in increasing time; an older-or-equal timestamp is ignored.
    """

    def __init__(self, keep_s: float = 3.0) -> None:
        if keep_s < 2.0:
            raise ValueError("keep_s must be at least 2 seconds")
        self._keep_s = keep_s
        self._lock = threading.Lock()
        self._t: list[float] = []
        self._q: list[Quat] = []
        # Samples at or before this time are not used to estimate angular velocity
        # (set at a discontinuity such as a looped pose track).
        self._seam_t = float("-inf")

    def __len__(self) -> int:
        with self._lock:
            return len(self._t)

    def clear(self) -> None:
        with self._lock:
            self._t.clear()
            self._q.clear()
            self._seam_t = float("-inf")

    def add(self, t: float, quat: Quat, *, discontinuity: bool = False) -> bool:
        """Append a sample. ``discontinuity`` marks a jump (not real motion) between the previous
        sample and this one: it is still interpolated across, but never used for extrapolation.
        Returns False if the sample was rejected as out of order."""
        quat = q.normalize(quat)
        with self._lock:
            if self._t and t <= self._t[-1]:
                return False
            self._t.append(t)
            self._q.append(quat)
            if discontinuity:
                self._seam_t = t
            cutoff = t - self._keep_s
            if self._t[0] < cutoff - self._keep_s:  # amortized pruning, always keeps >= keep_s
                i = bisect.bisect_left(self._t, cutoff)
                del self._t[:i]
                del self._q[:i]
            return True

    def newest_t(self) -> float | None:
        with self._lock:
            return self._t[-1] if self._t else None

    def oldest_t(self) -> float | None:
        with self._lock:
            return self._t[0] if self._t else None

    def pose_at(self, t: float) -> Pose | None:
        with self._lock:
            if not self._t:
                return None
            ts, qs = self._t, self._q
            if t <= ts[0]:
                return Pose(t, qs[0])
            if t >= ts[-1]:
                return Pose(t, self._extrapolate(t))
            i = bisect.bisect_right(ts, t)  # ts[i-1] <= t < ts[i]
            t0, t1 = ts[i - 1], ts[i]
            return Pose(t, q.slerp(qs[i - 1], qs[i], (t - t0) / (t1 - t0)))

    def _extrapolate(self, t: float) -> Quat:
        ts, qs = self._t, self._q
        last_t, last_q = ts[-1], qs[-1]
        dt = min(t - last_t, MAX_EXTRAPOLATION_S)
        if dt <= 0.0 or len(ts) < 2:
            return last_q
        # Reference sample at least the minimum baseline back, but never across a seam.
        j = len(ts) - 2
        while j > 0 and last_t - ts[j] < _MIN_VELOCITY_BASELINE_S and ts[j - 1] > self._seam_t:
            j -= 1
        if ts[j] < self._seam_t or last_t <= self._seam_t:
            return last_q
        span = last_t - ts[j]
        if span <= 1e-9:
            return last_q
        rv = q.rotvec(q.mul(last_q, q.conj(qs[j])))  # world-frame delta: last = delta * ref
        k = dt / span
        return q.normalize(q.mul(q.from_rotvec((rv[0] * k, rv[1] * k, rv[2] * k)), last_q))
