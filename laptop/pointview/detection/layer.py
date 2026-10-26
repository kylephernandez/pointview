"""DetectionLayer: the Layer (contracts section 3.2) that turns glasses frames into Entities."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from pointview import interfaces
from pointview.detection.convert import track_to_entity
from pointview.detection.model import Detection, Detector
from pointview.detection.tracker import HOLD_S, IoUTracker
from pointview.entities import Entity
from pointview.interfaces import LayerContext


class DetectorLike(Protocol):
    def detect(self, bgr: np.ndarray) -> list[Detection]: ...


class DetectionLayer:
    """Runs the detector on its own thread; ``update`` only returns the latest snapshot.

    Pipeline per inference: newest frame -> detector -> tracker (3 hits to show, 300 ms hold) ->
    each track matched in this frame is converted to a world direction using the head pose AT THE
    FRAME'S CAPTURE TIME. Tracks that are only being held keep that last world direction.
    The frame is dropped right after inference; nothing is logged or saved.
    """

    name = "detection"

    def __init__(
        self,
        model_path: Path,
        providers: list[str] | None = None,
        max_fps: float = 15.0,
        *,
        detector: DetectorLike | None = None,
        hold_s: float = HOLD_S,
    ) -> None:
        # ``detector`` lets tests inject a fake session; production passes only model_path.
        self._detector: DetectorLike = detector or Detector(Path(model_path), providers)
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._hold_s = hold_s
        self._tracker = IoUTracker(hold_s=hold_s)
        self._lock = threading.Lock()
        self._ctx: LayerContext | None = None
        self._snapshot: list[Entity] = []
        self._cache: dict[int, Entity] = {}
        self._last_frame_t: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        #: Duration of the most recent detector call, seconds (for diagnostics).
        self.last_inference_s: float | None = None
        self.error: BaseException | None = None
        #: Number of frames fully processed (inference + tracking + conversion).
        self.frames_processed = 0

    # ------------------------------------------------------------------ Layer protocol

    def update(self, ctx: LayerContext) -> list[Entity]:
        with self._lock:
            self._ctx = ctx
            snapshot = self._snapshot
        self._ensure_thread()
        return [
            e for e in snapshot if e.kind in ctx.enabled and ctx.t - e.updated_at <= self._hold_s
        ]

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        with self._lock:
            self._snapshot = []
            self._cache.clear()
            self._ctx = None

    # ------------------------------------------------------------------ worker

    def _ensure_thread(self) -> None:
        if self._thread is None and not self._stop.is_set():
            self._thread = threading.Thread(
                target=self._run, name="pointview-detection", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        last_start = -1e9
        while not self._stop.is_set():
            with self._lock:
                ctx = self._ctx
            wait = self._min_interval - (time.monotonic() - last_start)
            if ctx is None or wait > 0:
                self._stop.wait(min(max(wait, 0.0), 0.01) if ctx else 0.01)
                continue
            frame = ctx.glasses.latest_frame()
            if frame is None or frame.t == self._last_frame_t:
                self._stop.wait(0.005)
                continue
            last_start = time.monotonic()
            try:
                self._process(ctx, frame)
            except Exception as exc:  # keep the layer alive; surface for diagnostics
                self.error = exc
            finally:
                del frame  # privacy: drop the pixels as soon as possible

    def _process(self, ctx: LayerContext, frame: interfaces.Frame) -> None:
        self._last_frame_t = frame.t
        # Only enabled kinds matter, but the tracker needs all of them to keep ids stable.
        h, w = frame.bgr.shape[:2]
        t0 = time.perf_counter()
        detections = self._detector.detect(frame.bgr)
        self.last_inference_s = time.perf_counter() - t0
        tracks = self._tracker.update(detections, frame.t)
        pose = ctx.glasses.pose_at(frame.t)
        intr = ctx.glasses.camera_intrinsics
        cache: dict[int, Entity] = {}
        for tr in tracks:
            if tr.last_seen_t == frame.t and pose is not None:
                cache[tr.id] = track_to_entity(
                    tr,
                    intr,
                    pose.quat,
                    ctx.alignment.north_offset_deg,
                    ctx.calibration.camera_to_head_quat,
                    (w, h),
                )
            elif tr.id in self._cache:
                cache[tr.id] = self._cache[tr.id]  # held: keep last world direction
        with self._lock:
            self._cache = cache
            self._snapshot = sorted(cache.values(), key=lambda e: e.id)
            self.frames_processed += 1
