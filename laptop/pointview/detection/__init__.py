"""Vehicle and person detection layer (W2): ONNX detector, IoU tracker, pixel boxes -> Entities."""

from pointview.detection.layer import DetectionLayer
from pointview.detection.model import Detection, Detector
from pointview.detection.tracker import IoUTracker, Track

__all__ = ["Detection", "DetectionLayer", "Detector", "IoUTracker", "Track"]
