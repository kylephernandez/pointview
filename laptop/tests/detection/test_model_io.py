from __future__ import annotations

import numpy as np

from pointview.detection import model as m


def make_outputs(entries, n_queries=300):
    """entries: list of (class_name, score, (x1, y1, x2, y2) normalized)."""
    logits = np.full((1, n_queries, 80), -12.0, dtype=np.float32)
    boxes = np.zeros((1, n_queries, 4), dtype=np.float32)
    for q, (name, score, (x1, y1, x2, y2)) in enumerate(entries):
        logits[0, q, m.COCO80.index(name)] = np.log(score / (1 - score))
        boxes[0, q] = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    return logits, boxes


def test_coco_names_and_kept_classes():
    assert m.COCO80[0] == "person" and m.COCO80[2] == "car" and m.COCO80[7] == "truck"
    assert sorted(n for n, _ in m.KEPT_CLASSES.values()) == sorted(
        ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
    )


def test_preprocess_layout_range_and_channel_order():
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[..., 0] = 255  # BGR blue
    x = m.preprocess(img)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    # RGB order: blue ends up in channel 2; no mean/std, plain /255.
    assert np.allclose(x[0, 2], 1.0) and np.allclose(x[0, :2], 0.0)
    img[:] = 128
    assert np.allclose(m.preprocess(img), 128 / 255, atol=1e-6)


def test_preprocess_is_a_plain_stretch_so_boxes_map_by_scaling():
    """A bright rectangle keeps its relative position/size after the stretch resize."""
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[270:810, 480:1440] = 255  # normalized x 0.25..0.75, y 0.25..0.75
    x = m.preprocess(img)[0, 0]
    ys, xs = np.nonzero(x > 0.5)
    assert abs(xs.min() / 640 - 0.25) < 0.01 and abs((xs.max() + 1) / 640 - 0.75) < 0.01
    assert abs(ys.min() / 640 - 0.25) < 0.01 and abs((ys.max() + 1) / 640 - 0.75) < 0.01


def test_postprocess_box_roundtrip_to_frame_pixels():
    logits, boxes = make_outputs([("car", 0.9, (0.25, 0.40, 0.50, 0.80))])
    (d,) = m.postprocess(logits, boxes, 1920, 1080)
    assert d.label == "car" and d.kind == "vehicle"
    assert abs(d.score - 0.9) < 1e-4
    assert np.allclose(d.box, (480, 432, 960, 864), atol=1e-3)


def test_postprocess_filters_classes_threshold_and_clips():
    logits, boxes = make_outputs(
        [
            ("person", 0.8, (0.1, 0.1, 0.2, 0.5)),
            ("airplane", 0.99, (0.3, 0.3, 0.6, 0.6)),  # not a kept class
            ("dog", 0.99, (0.3, 0.3, 0.6, 0.6)),
            ("bus", 0.3, (0.5, 0.5, 0.9, 0.9)),  # below threshold
            ("motorcycle", 0.6, (-0.1, 0.5, 0.3, 1.2)),  # sticks out of the frame
        ]
    )
    out = m.postprocess(logits, boxes, 1000, 500, score_threshold=0.5)
    assert [d.label for d in out] == ["person", "motorcycle"]  # sorted by score
    moto = out[1]
    assert moto.box[0] == 0.0 and moto.box[3] == 500.0
    assert m.postprocess(logits, boxes, 1000, 500, score_threshold=0.2)[-1].label == "bus"


def test_one_detection_per_query():
    logits, boxes = make_outputs([("car", 0.9, (0.1, 0.1, 0.5, 0.5))])
    logits[0, 0, m.COCO80.index("truck")] = np.log(0.6 / 0.4)
    out = m.postprocess(logits, boxes, 640, 640)
    assert [d.label for d in out] == ["car"]


def test_default_providers():
    assert m.default_providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert m.default_providers(["DmlExecutionProvider", "CPUExecutionProvider"]) == [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
