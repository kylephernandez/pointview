from __future__ import annotations

import json

import httpx
import pytest

from pointview.mesh import osm_buildings as ob


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("12", 12.0),
        ("12 m", 12.0),
        ("12m", 12.0),
        ("12.5", 12.5),
        ("12,5", 12.5),
        (" 7 meters ", 7.0),
        ("40 ft", 40 * 0.3048),
        ("40'", 40 * 0.3048),
        ("10'6\"", (10 + 0.5) * 0.3048),
        ("0", 0.0),
        (7, 7.0),
        (None, None),
        ("", None),
        ("tall", None),
        ("12 storeys", None),
        ("-5", None),
        ("nan", None),
        ("inf", None),
        ("1e3", None),
        ("5000", None),
        ("12;15", None),
        ("1" * 100, None),
    ],
)
def test_parse_length(value, expected):
    got = ob.parse_length_m(value)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ({"building": "yes"}, (0.0, 6.0)),
        ({"building": "yes", "height": "12 m"}, (0.0, 12.0)),
        ({"building": "yes", "building:levels": "4"}, (0.0, 12.0)),
        ({"building": "yes", "building:levels": "2.5"}, (0.0, 7.5)),
        ({"building": "yes", "height": "9", "building:levels": "10"}, (0.0, 9.0)),  # height wins
        ({"building": "yes", "height": "junk", "building:levels": "3"}, (0.0, 9.0)),
        ({"building": "yes", "height": "junk", "building:levels": "x"}, (0.0, 6.0)),
        ({"building": "yes", "min_height": "10", "height": "25"}, (10.0, 25.0)),
        ({"building": "yes", "building:min_level": "2", "building:levels": "5"}, (6.0, 15.0)),
        ({"building": "yes", "min_height": "10"}, (10.0, 13.0)),  # top missing: default < min
        ({"building": "yes", "min_height": "30", "height": "20"}, (30.0, 33.0)),
        ({"building": "yes", "height": "0"}, (0.0, 6.0)),
        ({"building": "yes", "building:levels": "0"}, (0.0, 6.0)),
        ({"building": "yes", "min_height": "-3", "height": "8"}, (0.0, 8.0)),
    ],
)
def test_building_extent(tags, expected):
    lo, hi = ob.building_extent(tags)
    assert (lo, hi) == pytest.approx(expected)
    assert hi > lo >= 0


def _way(i, pts, **tags):
    return {
        "type": "way",
        "id": i,
        "tags": {"building": "yes", **tags},
        "geometry": [{"lat": a, "lon": b} for a, b in pts],
    }


SQUARE = [(0.0, 0.0), (0.0, 0.001), (0.001, 0.001), (0.001, 0.0), (0.0, 0.0)]


def test_parse_basic_way_drops_closing_point():
    (fp,) = ob.parse_buildings({"elements": [_way(1, SQUARE, height="8")]})
    assert fp.id == "way/1" and len(fp.ring) == 4 and fp.ring[0] != fp.ring[-1]
    assert (fp.min_height_m, fp.height_m) == (0.0, 8.0)


def test_parse_skips_junk():
    body = {
        "elements": [
            "oops",
            {"type": "way", "id": 1},  # no tags
            _way(2, SQUARE[:2]),  # two points: not a polygon
            _way(3, [(0, 0), (0, 0), (0, 0)]),  # degenerate
            {**_way(4, SQUARE), "tags": {"building": "no"}},
            {"type": "node", "id": 5, "tags": {"building": "yes"}, "lat": 1, "lon": 1},
            {"type": "way", "id": 6, "tags": {"building": "yes"}, "geometry": [{"lat": "x"}] * 4},
            _way(7, [(95.0, 0.0), (95.0, 1.0), (96.0, 1.0)]),  # latitude out of range
            _way(8, SQUARE),
        ]
    }
    assert [f.id for f in ob.parse_buildings(body)] == ["way/8"]
    assert ob.parse_buildings("not json") == []
    assert ob.parse_buildings("[1,2]") == []
    assert ob.parse_buildings({}) == []


def test_relation_outer_rings_are_stitched_and_inner_ignored():
    a = [(0.0, 0.0), (0.0, 0.001), (0.001, 0.001)]
    b = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001)]  # reversed half: joins end to end
    hole = [(0.0002, 0.0002), (0.0002, 0.0004), (0.0004, 0.0004), (0.0002, 0.0002)]
    geom = lambda pts: [{"lat": x, "lon": y} for x, y in pts]  # noqa: E731
    rel = {
        "type": "relation",
        "id": 9,
        "tags": {"building": "yes", "height": "20"},
        "members": [
            {"type": "way", "ref": 1, "role": "outer", "geometry": geom(a)},
            {"type": "way", "ref": 2, "role": "outer", "geometry": geom(b)},
            {"type": "way", "ref": 3, "role": "inner", "geometry": geom(hole)},
            {"type": "node", "ref": 4, "role": "outer"},
        ],
    }
    (fp,) = ob.parse_buildings({"elements": [rel]})
    assert fp.id == "relation/9" and fp.height_m == 20.0
    assert len(fp.ring) == 4


def test_relation_unclosable_ring_dropped():
    rel = {
        "type": "relation",
        "id": 9,
        "tags": {"building": "yes"},
        "members": [
            {
                "type": "way",
                "role": "outer",
                "geometry": [{"lat": 0, "lon": 0}, {"lat": 1, "lon": 1}],
            }
        ],
    }
    assert ob.parse_buildings({"elements": [rel]}) == []


def test_real_recording(buildings_json):
    fps = ob.parse_buildings(buildings_json)
    raw = json.loads(buildings_json)["elements"]
    assert len(raw) == 49 and len(fps) == 49
    assert len({f.id for f in fps}) == 49
    for f in fps:
        assert len(f.ring) >= 3 and f.ring[0] != f.ring[-1]
        assert all(37.80 < lat < 37.81 and -122.48 < lon < -122.46 for lat, lon in f.ring)
        assert 0 <= f.min_height_m < f.height_m < 100
    by_id = {f.id: f for f in fps}
    # "Movement San Francisco": tagged height=10
    assert by_id["way/24334661"].height_m == 10.0
    assert len(by_id["way/24334661"].ring) == 9


def test_query_shape():
    q = ob.build_query(37.806, -122.468, 800)
    assert 'way["building"](around:800,37.806000,-122.468000)' in q
    assert 'relation["building"](around:800,37.806000,-122.468000)' in q
    assert "out geom" in q and "[out:json]" in q


def test_fetch_falls_back_to_next_endpoint_and_sets_user_agent():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((str(req.url), req.headers["user-agent"]))
        if "first" in req.url.host:
            return httpx.Response(504)
        return httpx.Response(200, text='{"elements": []}')

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        body = ob.fetch_buildings_raw(
            c, 37.8, -122.4, 800, ["https://first.example/api", "https://second.example/api"]
        )
    assert json.loads(body) == {"elements": []}
    assert [u for u, _ in seen] == ["https://first.example/api", "https://second.example/api"]
    assert all(
        ua.startswith("PointView/0.1 (https://github.com/kylephernandez/pointview)")
        for _, ua in seen
    )


def test_fetch_all_fail_raises():
    def handler(req):
        return httpx.Response(200, text="<html>rate limited</html>")

    with httpx.Client(transport=httpx.MockTransport(handler)) as c, pytest.raises(RuntimeError):
        ob.fetch_buildings_raw(c, 0, 0, 800, ["https://a.example/x"])
