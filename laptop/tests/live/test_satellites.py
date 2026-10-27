"""Tests for pointview.live.satellites. No network: CelesTrak is served from recorded TLE files.

Astronomy checks (see tests/fixtures/live/satellites/make_reference.py):
* GMST against Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., Example 3-5.
* SGP4 against Vallado's SGP4-VER.TLE verification case 00005 (the sgp4 package's own test data).
* TEME -> ECEF -> az/el/range and the geodetic subpoint against Skyfield, which does its own
  reduction (equation of the equinoxes, own WGS84 code) on the same TLEs.
"""

from __future__ import annotations

import json
import math
import re
import time
from datetime import UTC
from pathlib import Path

import httpx
import numpy as np
import pytest
from sgp4.api import Satrec

from pointview import world
from pointview.entities import Entity
from pointview.interfaces import (
    Alignment,
    Calibration,
    LayerContext,
    PhoneState,
    View,
)
from pointview.live import satellites as sat
from tests.live import _world_ref_sat

FIX = Path(__file__).parent.parent / "fixtures" / "live" / "satellites"
REF = json.loads((FIX / "reference.json").read_text())
T0 = REF["t0_unix"]
TLE = {g: (FIX / f"tle_{g}.txt").read_text() for g in ("stations", "visual")}
SF = (37.8060, -122.4680)
FOV = (46.0, 27.0)
ISS = 25544


@pytest.fixture(autouse=True)
def _world_ref(monkeypatch):
    _world_ref_sat.install(monkeypatch)


# ----------------------------------------------------------------------------------- helpers


class FakeCelesTrak:
    """httpx transport serving the recorded TLE files; counts requests."""

    def __init__(self):
        self.requests: list[dict[str, str]] = []
        self.offline = False
        self.status = 200
        self.body_override: str | None = None

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if self.offline:
            raise httpx.ConnectError("offline", request=request)
        q = dict(request.url.params)
        self.requests.append(q)
        assert request.url.host == "celestrak.org"
        assert request.url.path == "/NORAD/elements/gp.php"
        if self.status != 200:
            return httpx.Response(self.status, text="nope")
        if self.body_override is not None:
            return httpx.Response(200, text=self.body_override)
        return httpx.Response(200, text=TLE[q["GROUP"]])


def phone(lat=SF[0], lon=SF[1]) -> PhoneState:
    return PhoneState(
        t_ms=0.0, t=0.0, received_at=0.0, quat=(1.0, 0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0),
        heading_deg=None, heading_acc=None, lat=lat, lon=lon, gps_acc=5.0,
    )  # fmt: skip


def head_quat(yaw_deg: float, pitch_deg: float):
    """Quaternion (w, x, y, z) taking the head's forward axis to (yaw, pitch) in the tracking frame."""
    hy, hp = math.radians(-yaw_deg) / 2, math.radians(pitch_deg) / 2
    qz = (math.cos(hy), 0.0, 0.0, math.sin(hy))
    qx = (math.cos(hp), math.sin(hp), 0.0, 0.0)
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qx
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def view_at(yaw: float, pitch: float) -> View:
    return View(head_quat(yaw, pitch), 0.0, FOV, yaw, pitch)


def ctx(unix, view=None, p="default", focus=None, enabled=None):
    kw = {}
    if enabled is not None:
        kw["enabled"] = enabled
    return LayerContext(
        t=1.0, glasses=None, alignment=Alignment(), calibration=Calibration(),  # type: ignore[arg-type]
        phone=phone() if p == "default" else p, unix_time=float(unix), view=view, focus_id=focus,
        **kw,
    )  # fmt: skip


@pytest.fixture
def server():
    return FakeCelesTrak()


@pytest.fixture
def layer(tmp_path, server):
    lay = sat.SatelliteLayer(tmp_path / "sat.sqlite", client=server.client(), tz=UTC)
    yield lay
    lay.close()


def ready(lay: sat.SatelliteLayer, unix=T0, **kw):
    """Load the catalogue and run the coarse pass at ``unix`` for the SF observer."""
    c = ctx(unix, view=view_at(0, 0), **kw)
    lay.update(c)
    assert lay.wait_for_data(5), lay.last_error
    lay.compute_now(c)


def iss_peak(i=0):
    return REF["iss_peak_samples"]["sf"][i], REF["iss_passes"]["sf"][i]


def dummy_sat(satnum=ISS):
    return next(s for s in sat.parse_tle(TLE["stations"], "stations") if s.norad == satnum)


# ------------------------------------------------------------------ astronomy against references


def test_gmst_vallado_example_3_5():
    # Vallado 4th ed. Example 3-5: 1992 Aug 20 12:14 UT1, JD 2448855.009722 -> GMST 152.578787810 deg
    g = math.degrees(float(sat.gmst_rad(2448854.5, 0.5 + 14 / 1440)))
    assert g == pytest.approx(152.578787810, abs=2e-5)


def test_sgp4_vallado_verification_case_00005():
    # SGP4-VER.TLE satellite 00005, tsince = 0 (Vallado et al., "Revisiting Spacetrack Report #3"):
    # r = 7022.46529266, -1400.08296755, 0.03995155 km
    s = Satrec.twoline2rv(
        "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
        "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
    )
    e, r, _ = s.sgp4(s.jdsatepoch, s.jdsatepochF)
    assert e == 0
    assert r == pytest.approx((7022.46529266, -1400.08296755, 0.03995155), abs=1e-6)


def test_teme_ecef_azel_range_match_skyfield():
    sats = {s.norad: s for g in TLE for s in sat.parse_tle(TLE[g], g)}
    assert len(REF["samples"]) > 250
    worst = np.zeros(4)
    for r in REF["samples"]:
        jd, fr = sat._jd_split(r["t_unix"])
        e, pos, _ = sats[r["norad"]].satrec.sgp4(jd, fr)
        assert e == 0
        ecef = sat.teme_to_ecef(np.array(pos), sat.gmst_rad(jd, fr))
        az, el, rng = sat.look_angles_ecef(ecef, r["lat"], r["lon"])
        lat, lon, _h = sat.ecef_to_geodetic(ecef)
        daz = ((float(az) - r["az_deg"] + 180) % 360 - 180) * math.cos(math.radians(r["el_deg"]))
        worst = np.maximum(
            worst,
            [abs(daz), abs(float(el) - r["el_deg"]), abs(float(rng) - r["range_km"]),
             abs(float(lat) - r["sub_lat"]) + abs(float(lon) - r["sub_lon"])],
        )  # fmt: skip
    # measured: 0.0011 deg, 0.0011 deg, 0.034 km, 0.0004 deg
    assert worst[0] < 0.005 and worst[1] < 0.005  # degrees on the sky
    assert worst[2] < 0.5  # km
    assert worst[3] < 0.005  # subpoint, degrees


def test_ecef_to_geodetic_roundtrip():
    for lat, lon, h in [
        (37.8, -122.4, 420.0),
        (-33.9, 151.2, 20.0),
        (89.0, 10.0, 800.0),
        (0, 179, 35786),
    ]:
        ecef = sat.observer_ecef_km(lat, lon, h)
        la, lo, hh = sat.ecef_to_geodetic(ecef)
        assert (float(la), float(lo), float(hh)) == pytest.approx((lat, lon, h), abs=1e-6)


def test_sun_matches_independent_noaa_reference():
    # sun_unit_teme through the same GMST/topocentric path vs the NOAA equations in the test ref
    for dt in (0, 3 * 3600, 9 * 3600, 15 * 3600, 80 * 86400):
        unix = T0 + dt
        jd, fr = sat._jd_split(unix)
        s = sat.sun_unit_teme(jd, fr) * 149_597_870.7
        az, el, _ = sat.look_angles_ecef(sat.teme_to_ecef(s, sat.gmst_rad(jd, fr)), *SF)
        raz, rel = _world_ref_sat.sun_direction(unix, *SF)
        assert abs(float(el) - rel) < 0.6
        if rel > -5:
            assert abs((float(az) - raz + 180) % 360 - 180) < 1.0


# ------------------------------------------------------------------------- shadow / visibility


def test_cylindrical_shadow():
    sun = np.array([1.0, 0.0, 0.0])
    r = 6371.0 + 420.0
    assert sat.sunlit(np.array([r, 0, 0]), sun)  # sunward of Earth
    assert not sat.sunlit(np.array([-r, 0, 0]), sun)  # straight behind Earth
    assert not sat.sunlit(
        np.array([-r, 0, 6000.0]), sun
    )  # behind, inside the cylinder (6000 < 6371)
    assert sat.sunlit(np.array([-r, 0, 6400.0]), sun)  # behind, outside the cylinder
    assert sat.sunlit(np.array([0, r, 0]), sun)  # on the terminator plane, above the limb
    assert not sat.sunlit(
        np.array([-100.0, 6000.0, 0]), sun
    )  # just behind the plane, 6000 km off axis
    both = sat.sunlit(np.array([[r, 0, 0], [-r, 0, 0]]), sun)
    assert both.tolist() == [True, False]


def _entity_at_iss_peak(lay, monkeypatch=None, i=0):
    ref, _ = iss_peak(i)
    ready(lay, ref["t_unix"])
    v = view_at(ref["az_deg"], ref["el_deg"])
    out = lay.update(ctx(ref["t_unix"], v))
    return next(e for e in out if e.id == f"sat:{ISS}"), ref


@pytest.mark.parametrize("lit", [True, False])
@pytest.mark.parametrize("sun_pitch", [-30.0, -6.5, -5.5, 20.0])
def test_visible_needs_sunlit_and_dark_sky(layer, monkeypatch, lit, sun_pitch):
    monkeypatch.setattr(world, "sun_direction", lambda u, la, lo: (200.0, sun_pitch))
    monkeypatch.setattr(
        sat, "sunlit", lambda r, s: np.full(np.asarray(r).shape[:-1], lit, dtype=bool)
    )
    e, _ = _entity_at_iss_peak(layer)
    dark = sun_pitch < -6.0
    assert e.meta["visible"] == ("yes" if (lit and dark) else "no")
    assert e.meta["sunlit"] == ("yes" if lit else "no")


def test_real_geometry_iss_over_sf_at_night_is_in_shadow(layer):
    # 01:15 PDT in late October, Sun ~60 deg below the horizon: the sky is dark, but the ISS at
    # 430 km is inside Earth's shadow, so it is not visible.
    e, ref = _entity_at_iss_peak(layer)
    assert _world_ref_sat.sun_direction(ref["t_unix"], *SF)[1] < -40
    assert e.meta["sunlit"] == "no" and e.meta["visible"] == "no"


# ------------------------------------------------------------------------------- the ISS fixture


def test_iss_entity_matches_skyfield_at_pass_peak(layer):
    e, ref = _entity_at_iss_peak(layer)
    assert isinstance(e, Entity) and e.kind == "satellite"
    assert e.label == "ISS (ZARYA)"
    assert e.world_yaw_deg == pytest.approx(ref["az_deg"], abs=0.005)
    assert e.world_pitch_deg == pytest.approx(ref["el_deg"], abs=0.005)
    assert 0 <= e.world_yaw_deg < 360
    assert e.distance_m == pytest.approx(ref["range_km"] * 1000, abs=500)
    assert e.updated_at == 1.0
    assert e.meta["class"] == "station"
    assert e.meta["norad"] == "25544"
    assert 400 <= float(e.meta["altitude_km"]) <= 440
    assert e.geo is not None and abs(e.geo.lat) <= 51.7 and 400e3 < e.geo.alt_m < 440e3
    # geometric sanity of a published-style pass: peak ~27 deg, south-east of SF, ~830 km away
    assert 120 < e.world_yaw_deg < 145 and 26 < e.world_pitch_deg < 28
    assert "next_pass" not in e.meta  # not the focus


def test_iss_second_independent_check_ground_distance(layer):
    # Second computation: ground distance from the subpoint with haversine matches slant geometry.
    e, _ = _entity_at_iss_peak(layer)
    assert e.geo is not None
    dlat, dlon = math.radians(e.geo.lat - SF[0]), math.radians(e.geo.lon - SF[1])
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(SF[0]))
        * math.cos(math.radians(e.geo.lat))
        * math.sin(dlon / 2) ** 2
    )
    ground_km = 2 * 6371.0 * math.asin(math.sqrt(a))
    # law of cosines on a sphere: slant^2 = R^2 + (R+h)^2 - 2R(R+h)cos(angle)
    h = e.geo.alt_m / 1000
    ang = ground_km / 6371.0
    slant = math.sqrt(6371**2 + (6371 + h) ** 2 - 2 * 6371 * (6371 + h) * math.cos(ang))
    assert slant == pytest.approx(e.distance_m / 1000, rel=0.01)


# -------------------------------------------------------------------------- in-view filtering


def test_only_satellites_in_view_are_returned(layer):
    ref, _ = iss_peak()
    t = ref["t_unix"]
    ready(layer, t)
    az, el = ref["az_deg"], ref["el_deg"]
    # looking straight at it
    assert any(e.id == f"sat:{ISS}" for e in layer.update(ctx(t, view_at(az, el))))
    # 25 deg off to the side: half FOV 23 + margin 5 = 28 -> still returned
    assert any(e.id == f"sat:{ISS}" for e in layer.update(ctx(t, view_at(az + 25, el))))
    # 32 deg off: out
    assert not any(e.id == f"sat:{ISS}" for e in layer.update(ctx(t, view_at(az + 32, el))))
    # vertically: half 13.5 + 5 = 18.5
    assert any(e.id == f"sat:{ISS}" for e in layer.update(ctx(t, view_at(az, el - 17))))
    assert not any(e.id == f"sat:{ISS}" for e in layer.update(ctx(t, view_at(az, el - 20))))
    # behind
    assert layer.update(ctx(t, view_at(az + 180, 0))) == [] or all(
        e.id != f"sat:{ISS}" for e in layer.update(ctx(t, view_at(az + 180, 0)))
    )


def test_returned_entities_are_all_above_horizon_and_in_view(layer):
    ref, _ = iss_peak()
    t = ref["t_unix"]
    ready(layer, t)
    for yaw in range(0, 360, 20):
        v = view_at(yaw, 20)
        for e in layer.update(ctx(t, v)):
            assert e.world_pitch_deg > 0
            assert world.in_view(
                e.world_yaw_deg, e.world_pitch_deg, v.head_quat, 0.0, FOV, margin_deg=5
            )


def test_wide_view_returns_many(layer):
    # Anything above the horizon within +-28 / +-18.5 of a level view toward the ISS
    ref, _ = iss_peak()
    ready(layer, ref["t_unix"])
    seen = set()
    for yaw in range(0, 360, 10):
        for pitch in (0, 20, 40, 60):
            seen |= {e.id for e in layer.update(ctx(ref["t_unix"], view_at(yaw, pitch)))}
    assert f"sat:{ISS}" in seen


def test_no_view_no_fix_disabled_or_no_data(layer, tmp_path):
    ref, _ = iss_peak()
    t = ref["t_unix"]
    ready(layer, t)
    v = view_at(ref["az_deg"], ref["el_deg"])
    assert layer.update(ctx(t, None)) == []
    assert layer.update(ctx(t, v, p=None)) == []
    assert layer.update(ctx(t, v, p=phone(None, None))) == []
    assert layer.update(ctx(t, v, enabled=frozenset({"landmark"}))) == []
    assert layer.update(ctx(t, v)) != []
    # a fresh layer with nothing cached and no network: quiet empty list
    off = FakeCelesTrak()
    off.offline = True
    quiet = sat.SatelliteLayer(tmp_path / "other.sqlite", client=off.client(), backoff_s=0.05)
    try:
        assert quiet.update(ctx(t, v)) == []
        time.sleep(0.2)
        assert quiet.update(ctx(t, v)) == []
        assert quiet.last_error
    finally:
        quiet.close()


def test_coarse_list_tracks_time_and_moves(layer):
    ref, _ = iss_peak()
    t = ref["t_unix"]
    ready(layer, t)
    n0 = layer.coarse_computations
    layer.update(ctx(t + 1, view_at(0, 0)))  # within 2 s: no new coarse pass requested
    time.sleep(0.1)
    assert layer.coarse_computations == n0
    layer.update(ctx(t + 3, view_at(0, 0)))  # stale: the thread recomputes
    deadline = time.monotonic() + 3
    while layer.coarse_computations == n0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert layer.coarse_computations > n0


# ----------------------------------------------------------------------- focus-only next pass


def test_next_pass_only_for_focus_and_cached(layer):
    ref, _ = iss_peak()
    t = ref["t_unix"]
    ready(layer, t)
    v = view_at(ref["az_deg"], ref["el_deg"])
    # no focus: nothing is computed, however many frames
    for _ in range(5):
        out = layer.update(ctx(t, v))
    assert layer.pass_computations == 0
    assert all("next_pass" not in e.meta for e in out)
    # focus on the ISS: one computation (on the thread), then it shows up in meta
    layer.update(ctx(t, v, focus=f"sat:{ISS}"))
    assert layer.drain(10)
    out = layer.update(ctx(t, v, focus=f"sat:{ISS}"))
    assert layer.pass_computations == 1
    iss = next(e for e in out if e.id == f"sat:{ISS}")
    assert re.fullmatch(
        r"(rises|up now, sets) \d\d:\d\d · max \d+° · (not )?visible", iss.meta["next_pass"]
    )
    assert all("next_pass" not in e.meta for e in out if e.id != iss.id)
    # many frames later (and focus kept): cached, no recomputation
    for _ in range(20):
        layer.update(ctx(t, v, focus=f"sat:{ISS}"))
    assert layer.pass_computations == 1


def test_next_pass_recomputed_after_cache_expiry_and_for_new_focus(layer, monkeypatch):
    ref, _ = iss_peak()
    t = ref["t_unix"]
    ready(layer, t)
    v = view_at(ref["az_deg"], ref["el_deg"])
    layer.update(ctx(t, v, focus=f"sat:{ISS}"))
    assert layer.drain(10)
    assert layer.pass_computations == 1
    monkeypatch.setattr(sat, "PASS_CACHE_S", 0.0)  # everything is stale
    layer.update(ctx(t, v, focus=f"sat:{ISS}"))
    assert layer.drain(10)
    assert layer.pass_computations == 2


def test_focus_on_unknown_or_hidden_satellite_computes_nothing(layer):
    ref, _ = iss_peak()
    t = ref["t_unix"]
    ready(layer, t)
    v = view_at(ref["az_deg"] + 120, 0)  # the ISS is out of view
    layer.update(ctx(t, v, focus=f"sat:{ISS}"))
    layer.update(ctx(t, view_at(ref["az_deg"], ref["el_deg"]), focus="osm:way/1"))
    assert layer.drain(2)
    assert layer.pass_computations == 0


@pytest.mark.parametrize("i", [0, 1, 2, 3])
def test_compute_next_pass_matches_skyfield(i):
    s = dummy_sat()
    passes = REF["iss_passes"]["sf"]
    p = sat.compute_next_pass(s.satrec, T0, *SF)
    assert p is not None and not p.up_now
    ref = passes[0]
    assert abs(p.rise - ref["rise"]) < 10
    assert abs(p.set - ref["set"]) < 10
    assert abs(p.peak - ref["peak"]) < 10
    assert p.max_el_deg == pytest.approx(ref["max_el_deg"], abs=0.2)
    if i:
        # search starting just after the previous pass finds the i-th one
        p2 = sat.compute_next_pass(s.satrec, passes[i - 1]["set"] + 60, *SF)
        assert p2 is not None
        assert abs(p2.rise - passes[i]["rise"]) < 10
        assert p2.max_el_deg == pytest.approx(passes[i]["max_el_deg"], abs=0.2)


def test_compute_next_pass_up_now():
    s = dummy_sat()
    ref = REF["iss_passes"]["sf"][0]
    p = sat.compute_next_pass(s.satrec, ref["peak"], *SF)
    assert p is not None and p.up_now and p.rise == ref["peak"]
    assert abs(p.set - ref["set"]) < 10
    txt = sat.format_pass(p, UTC)
    assert txt.startswith("up now, sets ")


def test_pass_visibility_follows_sun_and_shadow(monkeypatch):
    s = dummy_sat()
    ps = REF["iss_passes"]["sf"]
    # a bright afternoon sky: never visible
    monkeypatch.setattr(world, "sun_direction", lambda u, la, lo: (180.0, 40.0))
    assert not any(sat.compute_next_pass(s.satrec, p["rise"] - 120, *SF).visible for p in ps)
    # forced dark sky: visible exactly when the ISS is sunlit somewhere in the pass
    monkeypatch.setattr(world, "sun_direction", lambda u, la, lo: (0.0, -40.0))
    flags = [sat.compute_next_pass(s.satrec, p["rise"] - 120, *SF).visible for p in ps]
    assert any(flags)  # the daytime passes are sunlit
    assert not flags[0]  # the 01:15 PDT pass is inside Earth's shadow (checked above)


def test_format_pass():
    p = sat.NextPass(rise=T0 + 4460, peak=T0 + 4600, set=T0 + 4800, max_el_deg=62.6, visible=True)
    txt = sat.format_pass(p, UTC)
    hh = time.strftime("%H:%M", time.gmtime(T0 + 4460))
    assert txt == f"rises {hh} · max 63° · visible"
    assert sat.format_pass(None) == "no pass in 24 h"


# ------------------------------------------------------------------------------ catalogue / cache


def test_fetch_urls_cache_and_12h_refresh(tmp_path, server, monkeypatch):
    path = tmp_path / "c.sqlite"
    a = sat.SatelliteLayer(path, client=server.client())
    ready(a)
    a.close()
    assert [r["GROUP"] for r in server.requests] == ["stations", "visual"]
    assert all(r["FORMAT"] == "tle" for r in server.requests)
    # restart within 12 h: served from SQLite, no request
    n = len(server.requests)
    b = sat.SatelliteLayer(path, client=server.client())
    ready(b)
    assert len(server.requests) == n and b.has_data
    b.close()
    # a restart 13 h later refetches
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 13 * 3600)
    c = sat.SatelliteLayer(path, client=server.client())
    ready(c)
    deadline = time.monotonic() + 5
    while len(server.requests) < n + 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(server.requests) == n + 2
    c.close()


def test_offline_restart_uses_stale_cache(tmp_path, server, monkeypatch):
    path = tmp_path / "c.sqlite"
    a = sat.SatelliteLayer(path, client=server.client())
    ready(a)
    a.close()
    server.offline = True
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + 30 * 86400)  # a month later, still offline
    b = sat.SatelliteLayer(path, client=server.client(), backoff_s=0.05)
    ready(b)
    deadline = time.monotonic() + 5
    while not b.last_error and time.monotonic() < deadline:
        time.sleep(0.01)
    assert b.has_data and b.last_error
    ref, _ = iss_peak()
    ready(b, ref["t_unix"])
    out = b.update(ctx(ref["t_unix"], view_at(ref["az_deg"], ref["el_deg"])))
    assert any(e.id == f"sat:{ISS}" for e in out)
    b.close()


def test_http_errors_and_garbage_do_not_crash_or_hammer(tmp_path, server):
    server.status = 503
    lay = sat.SatelliteLayer(tmp_path / "c.sqlite", client=server.client(), backoff_s=5.0)
    try:
        for _ in range(10):
            lay.update(ctx(T0, view_at(0, 0)))
            time.sleep(0.05)
        assert len(server.requests) <= 2  # one try per group, then backoff
        assert not lay.has_data and "503" in (lay.last_error or "")
        assert lay.update(ctx(T0, view_at(0, 0))) == []
    finally:
        lay.close()
    server.status = 200
    server.body_override = "<html>rate limited</html>"
    lay = sat.SatelliteLayer(tmp_path / "d.sqlite", client=server.client(), backoff_s=5.0)
    try:
        lay.update(ctx(T0, view_at(0, 0)))
        time.sleep(0.3)
        assert not lay.has_data
        assert lay.update(ctx(T0, view_at(0, 0))) == []
    finally:
        lay.close()


def test_parse_tle_classes_dedup_and_junk():
    st = sat.parse_tle(TLE["stations"], "stations")
    vis = sat.parse_tle(TLE["visual"], "visual")
    assert len(st) >= 5 and len(vis) > 100
    assert {s.klass for s in st} == {"station"}
    assert {s.klass for s in vis if s.norad != ISS} == {"visual"}
    assert next(s for s in vis if s.norad == ISS).klass == "station"  # ISS is always a station
    assert next(s for s in st if s.norad == ISS).name == "ISS (ZARYA)"
    assert sat.parse_tle("garbage\n1 xx\n2 yy\n", "visual") == []
    assert sat.parse_tle("", "visual") == []
    two_line = "\n".join(TLE["stations"].splitlines()[1:3])
    assert sat.parse_tle(two_line, "stations")[0].name.startswith("NORAD")
    assert sat.parse_tle(TLE["stations"], "gps-ops")[1].klass == "nav"
    assert sat.parse_tle(TLE["stations"], "who-knows")[1].klass == "visual"


def test_duplicate_satellites_across_groups_are_one_entity(layer):
    ready(layer)
    assert len(layer._sats) == len({s.norad for s in layer._sats})
    assert len(layer._sats) == len({s.norad for g in TLE for s in sat.parse_tle(TLE[g], g)})
    assert layer._by_norad[ISS].klass == "station"


def test_close_stops_thread(tmp_path, server):
    lay = sat.SatelliteLayer(tmp_path / "c.sqlite", client=server.client())
    lay.update(ctx(T0, view_at(0, 0)))
    t = lay._thread
    assert t is not None and t.is_alive()
    lay.close()
    assert not t.is_alive()


# --------------------------------------------------------------------------------------- timing


def test_update_under_5ms_with_200_satellites(layer):
    ref, _ = iss_peak()
    t = ref["t_unix"]
    ready(layer, t)
    assert len(layer._sats) >= 150
    co = layer._coarse
    assert co is not None and 0 < len(co.sats) < len(layer._sats)
    v = view_at(ref["az_deg"], ref["el_deg"])
    for _ in range(20):  # warm up
        layer.update(ctx(t, v))
    times = []
    for k in range(300):
        c = ctx(t + (k % 2) * 0.016, v, focus=f"sat:{ISS}")
        t0 = time.perf_counter()
        layer.update(c)
        times.append(time.perf_counter() - t0)
    times.sort()
    assert np.mean(times) < 0.005, f"mean {np.mean(times) * 1e3:.2f} ms"
    assert times[int(0.95 * len(times))] < 0.005, (
        f"p95 {times[int(0.95 * len(times))] * 1e3:.2f} ms"
    )
