"""The shared entity format (docs/design.md, "Shared entity format"). FROZEN — see docs/contracts.md.

Every layer turns its input into Entities. Entities are world directions, never screen pixels:

* ``world_yaw_deg``   — degrees clockwise from TRUE NORTH, normalized to [0, 360).
* ``world_pitch_deg`` — degrees above the horizon (up is positive), in [-90, 90].
* ``distance_m``      — meters from the wearer (estimated for detections, computed for landmarks).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

EntityKind = Literal["person", "vehicle", "landmark"]

#: Detector classes we keep, mapped to their entity kind. Labels are the COCO class names.
DETECTION_CLASSES: Mapping[str, EntityKind] = {
    "person": "person",
    "bicycle": "vehicle",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
}


@dataclass(frozen=True, slots=True)
class SizeDeg:
    """Angular size of an object, for drawing brackets."""

    w: float
    h: float


@dataclass(frozen=True, slots=True)
class Entity:
    #: Stable per tracked object ("det:17") or OSM feature ("osm:node/123").
    id: str
    kind: EntityKind
    #: "car", "bus", "person", "Golden Gate Bridge". Never a person's identity.
    label: str
    #: Degrees clockwise from true north, [0, 360).
    world_yaw_deg: float
    #: Degrees above the horizon, up-positive.
    world_pitch_deg: float
    #: Laptop clock (seconds, see interfaces.now()) when this entity was last refreshed.
    updated_at: float
    distance_m: float | None = None
    size_deg: SizeDeg | None = None
    confidence: float | None = None
    #: Free-form string metadata a layer wants to carry to the info card, e.g.
    #: {"wikipedia": "en:Golden Gate Bridge", "osm": "way/123"}. Never personal data.
    meta: Mapping[str, str] = field(default_factory=dict)
