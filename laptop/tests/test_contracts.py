"""Sanity checks on the frozen contract files and shared fixtures."""

import jsonschema

from pointview import interfaces
from pointview.entities import DETECTION_CLASSES, Entity, SizeDeg
from pointview.schemas import load


def test_phone_fixture_messages_match_schema(phone_messages):
    schema = load("phone_message")
    jsonschema.Draft202012Validator.check_schema(schema)
    assert len(phone_messages) >= 5
    for msg in phone_messages:
        jsonschema.validate(msg, schema)


def test_schema_rejects_bad_messages(phone_messages):
    v = jsonschema.Draft202012Validator(load("phone_message"))
    good = phone_messages[1]
    for bad in (
        {**good, "quat": [1, 0, 0]},
        {**good, "buttons": ["volume_up"]},
        {**good, "heading_deg": 360},
        {**good, "extra": 1},
        {k: x for k, x in good.items() if k != "t_ms"},
    ):
        assert not v.is_valid(bad)


def test_laptop_schema():
    v = jsonschema.Draft202012Validator(load("laptop_message"))
    assert v.is_valid({"type": "haptic", "style": "snap"})
    assert v.is_valid({"type": "status", "aligned": True, "selected": None})
    assert not v.is_valid({"type": "haptic", "style": "buzz"})


def test_entity_defaults():
    e = Entity("det:1", "vehicle", "car", 10.0, 0.0, 1.0, size_deg=SizeDeg(3, 2))
    assert e.distance_m is None and dict(e.meta) == {}
    assert DETECTION_CLASSES["truck"] == "vehicle"


def test_fixtures_consistent(scene, pose_track, fixture_intrinsics, sample_video_path):
    assert sample_video_path.stat().st_size > 10_000
    assert len(pose_track) == 2001 and pose_track[0].t == 0.0
    assert fixture_intrinsics.width == scene["video"]["width"] == 960
    assert interfaces.MIN_TEXT_PX == 24


def test_v2_phone_fields(phone_messages):
    v = jsonschema.Draft202012Validator(load("phone_message"))
    good = phone_messages[1]
    assert v.is_valid({**good, "look": "nvg", "passthrough": True, "layers": {"aircraft": False}})
    assert not v.is_valid({**good, "look": "sepia"})
    assert set(interfaces.LAYER_KINDS) == set(
        load("phone_message")["properties"]["layers"]["properties"]
    )
    assert interfaces.PhoneState.__dataclass_fields__["look"].default == "normal"
