import itertools

from pointview import world
from pointview.entities import Entity, SizeDeg
from pointview.interfaces import (
    Alignment,
    Brackets,
    Crosshair,
    EdgeArrow,
    InfoCard,
    LandmarkLabel,
    PointerRing,
    StatusCorner,
    Tag,
    Text,
)
from pointview.scene import (
    SceneInput,
    Status,
    build_scene,
    entity_tag_text,
    format_distance,
    info_for,
)

DISPLAY = (1920, 1080)
FOV = (46.0, 27.0)
LEVEL_NORTH = world.quat_from_yaw_pitch_roll(0.0, 0.0)


def ent(id_, kind, label, yaw, pitch=0.0, dist=None, size=None):
    return Entity(id_, kind, label, yaw, pitch, 0.0, distance_m=dist, size_deg=size)


def scene(entities, **kw):
    base = dict(
        entities=entities,
        head_quat=LEVEL_NORTH,
        alignment=Alignment(aligned_at=0.0),
        display_size=DISPLAY,
        display_fov_deg=FOV,
        status=Status(phone_connected=True, gps_acc_m=5.0),
    )
    base.update(kw)
    return build_scene(SceneInput(**base))


def of(items, cls):
    return [i for i in items if isinstance(i, cls)]


def test_format_distance():
    assert format_distance(None) is None
    assert format_distance(3.4) == "~3 m"
    assert format_distance(14.0) == "~15 m"
    assert format_distance(123.0) == "~120 m"
    assert format_distance(1834.0) == "~1.8 km"


def test_tag_and_cards_follow_privacy_rules():
    car = ent("det:1", "vehicle", "car", 0, dist=15.2)
    person = ent("det:2", "person", "person", 0, dist=8.0)
    assert entity_tag_text(car) == "car · ~15 m"
    card = info_for(person)
    assert card.title == "person" and card.lines == ("~8 m",)
    lm = ent("osm:node/1", "landmark", "Fort Point", 0, dist=900)
    card = info_for(lm, ["line one", "line two", "line three"], "Wikipedia · CC BY-SA")
    assert card.lines == ("~900 m", "line one", "line two") and card.footer


def test_detection_drawn_centered_with_brackets_and_tag():
    items = scene([ent("det:1", "vehicle", "car", 0.0, dist=15, size=SizeDeg(4, 3))])
    (b,) = of(items, Brackets)
    (t,) = of(items, Tag)
    assert abs(b.x + b.w / 2 - 960) < 1 and abs(b.y + b.h / 2 - 540) < 1
    assert t.text == "car · ~15 m" and t.y < b.y


def test_offscreen_and_behind_are_skipped_or_arrowed():
    items = scene(
        [
            ent("det:1", "person", "person", 90.0),  # far right: not drawn
            ent("osm:a", "landmark", "Behind", 180.0),  # behind: nothing
            ent("osm:b", "landmark", "Just right", 35.0),  # just out of view: arrow
        ]
    )
    assert not of(items, Brackets) and not of(items, LandmarkLabel)
    (arrow,) = of(items, EdgeArrow)
    assert arrow.text == "Just right" and arrow.x > 1800 and arrow.angle_deg == 0.0


def test_edge_arrows_on_one_side_do_not_overlap():
    items = scene([ent(f"osm:{i}", "landmark", f"L{i}", -30.0 - i) for i in range(4)])
    ys = sorted(a.y for a in of(items, EdgeArrow))
    assert len(ys) == 4 and all(b - a >= 40 for a, b in itertools.pairwise(ys))


def test_landmark_labels_staggered_and_highlighted():
    items = scene(
        [ent("osm:a", "landmark", "A", -1.0, dist=500), ent("osm:b", "landmark", "B", 1.0)],
        snapped_id="osm:a",
    )
    a, b = sorted(of(items, LandmarkLabel), key=lambda i: i.x)
    assert a.highlighted and a.text == "A · ~500 m" and not b.highlighted
    assert a.y_text != b.y_text  # neighbours are staggered
    assert a.y_text < a.y_target


def test_pointer_ring_solid_when_snapped_and_clamped_at_edge():
    e = ent("osm:a", "landmark", "A", 0.0)
    (ring,) = of(scene([e], pointer=(0.0, 0.0), snapped_id="osm:a"), PointerRing)
    assert ring.solid and abs(ring.x - 960) < 1
    (ring,) = of(scene([e], pointer=(40.0, 0.0)), PointerRing)
    assert not ring.solid and ring.x == 1900


def test_crosshair_info_card_status_and_attribution():
    from pointview.scene import InfoContent

    e = ent("osm:a", "landmark", "A", 0.0)
    items = scene(
        [e],
        show_crosshair=True,
        info=InfoContent("osm:a", "A", ("~1 km",), "src"),
        attribution="© OpenStreetMap contributors",
    )
    assert of(items, Crosshair)
    (card,) = of(items, InfoCard)
    assert 0 <= card.x <= DISPLAY[0] - 500 and card.title == "A"
    assert of(items, StatusCorner)[0].lines[0][0] == "phone ✓"
    assert of(items, Text)[0].text.startswith("©")


def test_status_unaligned_and_no_phone():
    items = scene([], status=Status(phone_connected=False), alignment=Alignment())
    lines = [t for t, _ in of(items, StatusCorner)[0].lines]
    assert lines[0] == "phone: searching" and "not aligned: tap Align" in lines


def test_crowded_tags_declutter_nearest_first_and_stay_on_screen():
    crowd = [
        ent(f"det:{i}", "person", "person", 0.2 * i, dist=10.0 + i, size=SizeDeg(1, 3))
        for i in range(8)
    ]
    crowd.append(ent("det:edge", "person", "person", -22.9, dist=5, size=SizeDeg(1, 3)))
    items = scene(crowd, snapped_id="det:7")
    tags = of(items, Tag)
    assert len(of(items, Brackets)) == 9  # every object keeps its brackets
    assert any(t.highlighted for t in tags)  # the selected tag always shows
    assert "person · ~5 m" in {t.text for t in tags}  # the nearest unselected one too
    assert len(tags) < 9
    for t in tags:
        half = len(t.text) * 15 / 2
        assert t.x - half >= 0 and t.x + half <= DISPLAY[0]
