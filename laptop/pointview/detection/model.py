"""ONNX object detector: preprocessing, inference session, postprocessing.

Model: RT-DETR v2 (ResNet-18 backbone), COCO, exported to ONNX by onnx-community
(see ``laptop/models/README.md``). Apache-2.0 weights and code.

Model I/O (verified with onnxruntime on the downloaded file):

* input  ``pixel_values``  float32 ``[1, 3, 640, 640]``, RGB, values in [0, 1] (rescale 1/255 only,
  NO mean/std normalization, per ``preprocessor_config.json``), plain resize (no letterbox, the
  aspect ratio is stretched exactly as in training).
* output ``logits``     ``[1, 300, 80]`` raw class logits (sigmoid scores, focal-loss head)
* output ``pred_boxes`` ``[1, 300, 4]``  normalized (cx, cy, w, h) in [0, 1] of the model input.

Frames are never stored here; ``Detector.detect`` only reads the array it is handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pointview.entities import DETECTION_CLASSES, EntityKind

INPUT_SIZE = 640
DEFAULT_SCORE_THRESHOLD = 0.5

#: COCO 80-class names in the model's output order (contiguous index 0..79).
_COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic_light",
    "fire_hydrant",
    "stop_sign",
    "parking_meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports_ball",
    "kite",
    "baseball_bat",
    "baseball_glove",
    "skateboard",
    "surfboard",
    "tennis_racket",
    "bottle",
    "wine_glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot_dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted_plant",
    "bed",
    "dining_table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell_phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy_bear",
    "hair_drier",
    "toothbrush",
]
COCO80 = tuple(n.replace("_", " ") for n in _COCO_NAMES)
assert len(COCO80) == 80

#: Model class index -> (label, kind) for the classes we keep.
KEPT_CLASSES: dict[int, tuple[str, EntityKind]] = {
    i: (name, DETECTION_CLASSES[name]) for i, name in enumerate(COCO80) if name in DETECTION_CLASSES
}
_KEPT_IDX = np.array(sorted(KEPT_CLASSES), dtype=np.int64)


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object; box in ORIGINAL frame pixels (x1, y1, x2, y2), origin top-left."""

    label: str
    kind: EntityKind
    score: float
    box: tuple[float, float, float, float]


def preprocess(bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """H x W x 3 uint8 BGR -> [1, 3, size, size] float32 RGB in [0, 1]."""
    resized = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) * np.float32(1.0 / 255.0)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def postprocess(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    frame_w: int,
    frame_h: int,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[Detection]:
    """Raw model outputs (batch of 1) -> detections of the kept classes in frame pixels.

    Each of the 300 queries yields at most one detection (its best kept class), and
    only person/bicycle/car/motorcycle/bus/truck are considered.
    """
    scores = 1.0 / (1.0 + np.exp(-logits[0][:, _KEPT_IDX].astype(np.float64)))  # [300, K]
    # One class per query (the best kept class), so a car/truck query is not reported twice.
    best = scores.argmax(axis=1)
    q_idx = np.nonzero(scores[np.arange(len(scores)), best] >= score_threshold)[0]
    k_idx = best[q_idx]
    out: list[Detection] = []
    boxes = pred_boxes[0]
    for q, k in zip(q_idx.tolist(), k_idx.tolist(), strict=True):
        cx, cy, w, h = (float(v) for v in boxes[q])
        x1 = min(max((cx - w / 2) * frame_w, 0.0), frame_w)
        x2 = min(max((cx + w / 2) * frame_w, 0.0), frame_w)
        y1 = min(max((cy - h / 2) * frame_h, 0.0), frame_h)
        y2 = min(max((cy + h / 2) * frame_h, 0.0), frame_h)
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue
        label, kind = KEPT_CLASSES[int(_KEPT_IDX[k])]
        out.append(Detection(label, kind, float(scores[q, k]), (x1, y1, x2, y2)))
    out.sort(key=lambda d: -d.score)
    return out


def default_providers(available: list[str] | None = None) -> list[str]:
    """DirectML first when present (Windows), always CPU as the fallback."""
    if available is None:
        import onnxruntime as ort

        available = ort.get_available_providers()
    if "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class Detector:
    """ONNX Runtime session + pre/post-processing. Use from one thread at a time."""

    def __init__(
        self,
        model_path: Path,
        providers: list[str] | None = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        import onnxruntime as ort

        self.providers = providers if providers is not None else default_providers()
        opts = ort.SessionOptions()
        if "DmlExecutionProvider" in self.providers:
            # Required by the DirectML execution provider.
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=self.providers
        )
        self.score_threshold = score_threshold
        self._input = self._session.get_inputs()[0].name
        self._outputs = [o.name for o in self._session.get_outputs()]

    @property
    def active_providers(self) -> list[str]:
        return self._session.get_providers()

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        h, w = bgr.shape[:2]
        x = preprocess(bgr)
        logits, boxes = self._session.run(self._outputs, {self._input: x})
        return postprocess(logits, boxes, w, h, self.score_threshold)
