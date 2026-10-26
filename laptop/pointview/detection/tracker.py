"""IoU tracker: stable ids, confirm after N consecutive hits, hold after loss (laptop-time based)."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

from pointview.detection.model import Detection
from pointview.entities import EntityKind

Box = tuple[float, float, float, float]

MIN_HITS = 3
HOLD_S = 0.3
IOU_THRESHOLD = 0.3


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass(slots=True)
class Track:
    id: int
    kind: EntityKind
    label: str
    box: Box
    score: float
    #: Consecutive detection updates this track was matched in.
    hits: int
    #: Laptop time of the last matching detection.
    last_seen_t: float
    #: True once the track reached ``min_hits``; it then stays shown until the hold expires.
    confirmed: bool = False
    extra: dict = field(default_factory=dict)


class IoUTracker:
    """Greedy IoU matching, done separately per class group (person / vehicle).

    * A new track is *tentative*; it is returned only after ``min_hits`` consecutive matches.
      A tentative track that misses one update is discarded.
    * A confirmed track that is not matched is kept (and returned, at its last box) until
      ``hold_s`` seconds of laptop time have passed since it was last matched.
    * Timing uses the ``t`` passed to ``update`` (laptop clock), never frame counts.
    """

    def __init__(
        self,
        min_hits: int = MIN_HITS,
        hold_s: float = HOLD_S,
        iou_threshold: float = IOU_THRESHOLD,
    ) -> None:
        self.min_hits = min_hits
        self.hold_s = hold_s
        self.iou_threshold = iou_threshold
        self._tracks: list[Track] = []
        self._ids = count(1)

    def update(self, detections: list[Detection], t: float) -> list[Track]:
        """Feed one inference result taken at laptop time ``t``; return the tracks to show."""
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self._tracks):
            for di, d in enumerate(detections):
                if d.kind != tr.kind:
                    continue
                v = iou(tr.box, d.box)
                if v >= self.iou_threshold:
                    pairs.append((v, ti, di))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            tr, d = self._tracks[ti], detections[di]
            tr.box, tr.score, tr.label = d.box, d.score, d.label
            tr.hits += 1
            tr.last_seen_t = t
            if tr.hits >= self.min_hits:
                tr.confirmed = True

        survivors: list[Track] = []
        for ti, tr in enumerate(self._tracks):
            if ti in matched_tracks or (tr.confirmed and t - tr.last_seen_t <= self.hold_s):
                survivors.append(tr)
        for di, d in enumerate(detections):
            if di in matched_dets:
                continue
            tr = Track(next(self._ids), d.kind, d.label, d.box, d.score, 1, t)
            if self.min_hits <= 1:
                tr.confirmed = True
            survivors.append(tr)
        self._tracks = survivors
        return self.visible(t)

    def visible(self, t: float) -> list[Track]:
        return [tr for tr in self._tracks if tr.confirmed and t - tr.last_seen_t <= self.hold_s]
