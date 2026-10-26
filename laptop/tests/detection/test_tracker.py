from __future__ import annotations

import pytest

from pointview.detection.model import Detection
from pointview.detection.tracker import IoUTracker, iou


def det(box, label="car", kind="vehicle", score=0.9) -> Detection:
    return Detection(label, kind, score, tuple(box))


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_shown_only_after_three_consecutive_hits():
    tr = IoUTracker()
    box = (100, 100, 200, 200)
    assert tr.update([det(box)], 0.00) == []
    assert tr.update([det(box)], 0.07) == []
    out = tr.update([det(box)], 0.13)
    assert len(out) == 1 and out[0].hits == 3


def test_tentative_track_dies_on_a_miss():
    tr = IoUTracker()
    box = (100, 100, 200, 200)
    tr.update([det(box)], 0.0)
    tr.update([det(box)], 0.07)
    tr.update([], 0.13)  # miss: the two hits are not "consecutive" any more
    assert tr.update([det(box)], 0.20) == []
    assert tr.update([det(box)], 0.27) == []
    assert len(tr.update([det(box)], 0.33)) == 1


def test_hold_300ms_after_last_match_is_time_based():
    tr = IoUTracker()
    box = (100, 100, 200, 200)
    for i in range(3):
        shown = tr.update([det(box)], 1.0 + i * 0.07)
    last = 1.14
    assert len(shown) == 1
    # Many empty inferences inside the window must not drop it (not frame-count based) ...
    for k in range(1, 6):
        out = tr.update([], last + k * 0.05)  # up to 0.25 s
        assert [t.id for t in out] == [shown[0].id]
        assert out[0].last_seen_t == pytest.approx(last)
    # ... and one empty inference after 300 ms must.
    assert tr.update([], last + 0.31) == []
    # A single late update well past the hold also drops it, whatever the frame count.
    tr2 = IoUTracker()
    for i in range(3):
        tr2.update([det(box)], i * 0.07)
    assert len(tr2.visible(0.14 + 0.30)) == 1
    assert tr2.visible(0.14 + 0.31) == []


def test_stable_id_while_moving_and_after_short_gap():
    tr = IoUTracker()
    ids = set()
    t = 0.0
    for i in range(10):
        x = 100 + i * 5
        out = tr.update([det((x, 100, x + 100, 200))], t)
        ids |= {o.id for o in out}
        t += 0.067
    assert len(ids) == 1
    tid = ids.pop()
    tr.update([], t)  # one missed inference
    t += 0.067
    out = tr.update([det((150, 100, 250, 200))], t)
    assert [o.id for o in out] == [tid]


def test_ids_distinct_and_matching_is_per_class_group():
    tr = IoUTracker()
    box = (100, 100, 200, 200)
    person = det(box, "person", "person")
    car = det(box, "car", "vehicle")
    out = []
    for i in range(3):
        out = tr.update([person, car], i * 0.07)
    assert {o.kind for o in out} == {"person", "vehicle"}
    assert len({o.id for o in out}) == 2
    # A car box where the person was must not steal the person track.
    out = tr.update([car], 0.21)
    by_kind = {o.kind: o for o in out}
    assert by_kind["person"].last_seen_t == pytest.approx(0.14)  # held, not updated
    assert by_kind["vehicle"].last_seen_t == 0.21


def test_class_flicker_inside_a_group_keeps_the_id():
    tr = IoUTracker()
    box = (100, 100, 200, 200)
    out = []
    for i, label in enumerate(["car", "truck", "car"]):
        out = tr.update([det(box, label)], i * 0.07)
    assert len(out) == 1 and out[0].label == "car"
    assert out[0].id == 1


def test_greedy_matching_prefers_highest_iou():
    tr = IoUTracker()
    a, b = (0, 0, 100, 100), (60, 0, 160, 100)
    for i in range(3):
        tr.update([det(a), det(b)], i * 0.07)
    ids = {o.id: o.box for o in tr.visible(0.14)}
    out = tr.update([det((5, 0, 105, 100)), det((65, 0, 165, 100))], 0.21)
    assert {o.id: o.box for o in out} == {
        next(i for i, bx in ids.items() if bx == a): (5, 0, 105, 100),
        next(i for i, bx in ids.items() if bx == b): (65, 0, 165, 100),
    }


def test_low_iou_starts_new_track():
    tr = IoUTracker()
    for i in range(3):
        tr.update([det((0, 0, 100, 100))], i * 0.07)
    out = tr.update([det((500, 500, 600, 600))], 0.21)
    assert [o.box for o in out] == [(0, 0, 100, 100)]  # old one held; new one tentative
