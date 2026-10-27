# PointView module contracts

**Status: FROZEN** from the start of Phase 1. Workers must not edit this file, `laptop/pointview/entities.py`,
`laptop/pointview/interfaces.py`, the *signatures* in `laptop/pointview/world.py`, the JSON Schemas in
`laptop/pointview/schemas/`, `phone/src/protocol.ts`, or `laptop/tests/conftest.py` + `laptop/tests/fixtures/`.
If a contract blocks you, stop and describe the change you need in your report; the lead decides.

`docs/design.md` is the product spec. This file pins down what the spec leaves open: frames, units, clocks
and module APIs.

## 1. Units and clocks

| Quantity | Unit / convention |
| --- | --- |
| Angles | Degrees everywhere in APIs, unless a name ends in `_rad`. |
| Yaw | Clockwise from north seen from above (compass sense). World yaw is in `[0, 360)`. |
| Pitch | Up from the horizon is positive, `[-90, 90]`. |
| Distances | Meters. |
| Laptop time | Seconds on `pointview.interfaces.now()` (= `time.monotonic`). Every `t` on the laptop. |
| Phone time | `t_ms`, the phone's `Date.now()` in milliseconds. Only `phone_link` converts it to laptop time. |
| Quaternions | `(w, x, y, z)` tuples, Hamilton product, unit norm. `q` "rotates A into B" means `v_B = q ⊗ v_A ⊗ q*`. |
| Colors | RGB tuples 0–255. Black is see-through on the glasses. |

## 2. Frames

All frames are right-handed. Direction vectors are unit length.

| Frame | Axes | Notes |
| --- | --- | --- |
| **World W** | x east, y true north, z up | `v = (sin ψ cos θ, cos ψ cos θ, sin θ)` for yaw ψ, pitch θ. |
| **Tracking T** | same as W, arbitrary yaw origin | The glasses' gravity-aligned frame. `world_yaw = tracking_yaw + north_offset_deg`. The glasses driver must convert the SDK's frame into this one. |
| **Head H** | x right, y forward, z up | `Pose.quat` rotates H into T. Identity = looking along tracking +y, level. |
| **Camera C** | x right, y down, z forward (OpenCV) | `head = (c.x, c.z, −c.y)`, then rotated by `Calibration.camera_to_head_quat` (measured camera-to-display offset). |
| **Phone P** | x right edge, y top edge, z out of the screen | iOS device frame. The pointer is +y (the top edge). |
| **Phone reference R** | gravity-aligned, z up, arbitrary yaw | `PhoneState.quat` rotates P into R. |
| **Display** | pixels, origin top-left, x right, y down | `display_size` = (W_g, H_g), about (1920, 1080). |

Display projection (design doc step 4) uses the head-frame vector `h` of a world direction:
`x = W_g/2 + f_g·h.x/h.y`, `y = H_g/2 − f_g·h.z/h.y`, `f_g = (W_g/2)/tan(FOV_h/2)`, with FOV about 46° × 27°.

### Alignment (design doc step 5)

At an Align tap the user looks at, and points at, the same far object.

* `phone_to_head_deg = head_yaw_T − phone_pointer_yaw_R` places the pointer in T.
* `north_offset_deg = phone_heading_deg − head_yaw_T` places T in W. When the heading is unavailable it is 0
  and `north_source = "none"`.
* A long-press (`align_long`) while snapped to a landmark replaces only the north offset, so the pointer's
  tracking yaw maps exactly onto that landmark's bearing.

Pointer ray: `R → (rotate yaw by phone_to_head) → T → (rotate yaw by north_offset) → W`. Pitch passes
through unchanged.

## 3. Python modules and ownership

| Path | Owner | Contract |
| --- | --- | --- |
| `pointview/entities.py` | lead (frozen) | `Entity`, `SizeDeg`, `DETECTION_CLASSES` |
| `pointview/interfaces.py` | lead (frozen) | `GlassesSource`, `Frame`, `Pose`, `CameraIntrinsics`, `Calibration`, `Alignment`, `PhoneState`, `ButtonEvent`, `PhoneLinkAPI`, `Layer`, `LayerContext`, `DrawItem` types, `Renderer`, colors |
| `pointview/world.py` | W3 | Signatures frozen; W3 writes the bodies. Pure, no I/O, no numpy required in signatures. |
| `pointview/glasses/` | W1 | `VitureGlasses`, `MockGlasses` (both `GlassesSource`), pose buffer |
| `pointview/detection/` | W2 | `DetectionLayer` (a `Layer`), model download script, tracker |
| `pointview/phone_link.py` | W5 | `PhoneLink` (a `PhoneLinkAPI`) |
| `tools/fake_phone.py` | W5 | Scripted phone simulator |
| `pointview/landmarks.py` | W6 | `LandmarkLayer` (a `Layer`), Wikipedia lookup, attribution |
| `pointview/render.py` | W6 | `QtRenderer` (a `Renderer`) + offscreen `render_to_image` |
| `phone/` (except `src/protocol.ts`) | W4 | Expo Go app |
| `pointview/__main__.py`, `app.py`, `scene.py`, `mock.py`, `calibration.py` | lead | CLI and main loop (`app.py`), entities + pointer + selection → `DrawItem`s (`scene.py`), `--mock` wiring, calibration files |

### 3.1 GlassesSource (W1)

See `interfaces.GlassesSource`. Constructors are not part of the protocol, but the lead will call:

```python
MockGlasses(video_path: Path, pose_csv: Path, intrinsics: CameraIntrinsics,
            display_size=(1920, 1080), display_fov_deg=(46.0, 27.0),
            clock: Clock = now, loop: bool = True, realtime: bool = True)
VitureGlasses(camera_index: int | None = None, display_fov_deg=(46.0, 27.0),
              intrinsics: CameraIntrinsics | None = None, sdk_path: Path | None = None)
```

* `MockGlasses` maps fixture time 0 to `clock()` at `start()`. Frame `i` gets `t = t0 + i/fps`. With
  `loop=True` it wraps both the video and the pose track (the pose track is 10 s, the video 6 s: wrap each by
  its own length). With `realtime=False`, `latest_frame()` advances exactly one frame per call (for
  deterministic tests).
* `pose_at(t)` semantics are in the docstring: slerp, at most 50 ms of extrapolation, never None after the
  first sample.
* `latest_frame()` never blocks. Frames are never written to disk.
* The SDK pose must be converted to frame T/H as defined above. Document the conversion and put the sign
  checks in the hardware checklist.

### 3.2 Layers (W2, W6)

```python
class Layer(Protocol):
    name: str
    def update(self, ctx: LayerContext) -> list[Entity]: ...  # < 5 ms; called at about 60 Hz
    def close(self) -> None: ...
```

* `DetectionLayer(model_path: Path, providers: list[str] | None = None, max_fps: float = 15.0)`. It reads
  `ctx.glasses.latest_frame()` and `ctx.glasses.pose_at(frame.t)` and runs inference on its own thread.
  Entity ids look like `det:<track id>`; `kind` comes from `DETECTION_CLASSES`. It fills `size_deg`,
  `distance_m` (d ≈ f·H/h with H = 1.7 m for people, 1.5 m for vehicles) and `confidence`. Only labels and
  directions leave the module; frames are dropped after inference.
* `LandmarkLayer(cache_path: Path, client: httpx.Client | None = None, radius_m=2000, requery_m=250)`.
  It reads `ctx.phone.lat/lon`. Ids look like `osm:<type>/<id>`. `meta["wikipedia"]` holds the OSM tag
  value (for example `"en:Golden Gate Bridge"`) when present. Landmark pitch is 0 unless elevation data
  exists.
* `landmarks.fetch_summary(wikipedia_tag: str, client=None) -> WikiSummary | None` is cached and called only
  when a landmark is selected. `WikiSummary` has `title`, `extract`, `url`.
* `landmarks.OSM_ATTRIBUTION = "© OpenStreetMap contributors"`.
* A layer drops entities whose kind is not in `ctx.enabled`.

### 3.3 PhoneLink (W5)

`PhoneLink(host="0.0.0.0", port=8765, clock: Clock = now, history_s=2.0, stall_ms=50)` implements
`PhoneLinkAPI`.

* It validates every message against `schemas/phone_message.schema.json`. An invalid message is dropped and
  counted; the connection stays open.
* Clock mapping: `offset = min over the last 2 s of (received_at − t_ms/1000)`. Then
  `PhoneState.t = t_ms/1000 + offset`.
* It holds one phone at a time. A new connection replaces the old one.
* Layer toggles persist across messages that omit `layers`.
* `send()` validates against `schemas/laptop_message.schema.json`.

### 3.4 Renderer (W6)

`QtRenderer(screen_name: str | None = None, screen_index: int | None = None, size=(1920, 1080))`: a
borderless, full-screen, black window on the chosen screen, implementing `Renderer`.

`render.render_to_image(items, size) -> QImage` draws offscreen through the same painter code, for tests and
screenshots. Text is at least `MIN_TEXT_PX` (24 px).

### 3.5 world.py (W3)

See the signatures and docstrings in `laptop/pointview/world.py`. Every other module uses these functions
for frame math. Nobody else re-implements a frame conversion.

## 4. Phone app (W4)

`phone/src/protocol.ts` mirrors the schema and is frozen. W4 owns everything else under `phone/`. The app
must run in Expo Go: Expo SDK modules only, no custom native code. See `docs/phone-protocol.md`.

## 5. Shared fixtures (`laptop/tests/`)

| Fixture | What |
| --- | --- |
| `sample_video_path` | `fixtures/sample.mp4`: 960×540, 30 fps, 6 s. A synthetic world rendered through the pose track. |
| `pose_track` | `fixtures/pose_track.csv` as `list[Pose]`: 200 Hz, 10 s, t from 0. Yaw ±25° sweep, pitch ±4°. |
| `fixture_intrinsics` | 70° horizontal FOV pinhole matching the video. |
| `scene` | `fixtures/scene.json`: the above, plus the tracking-frame directions of the 4 world-fixed markers. |
| `phone_messages` | `fixtures/phone_messages.jsonl`: valid phone messages. |
| `fixtures_dir` | Path to the folder. Modules put their own fixtures in `fixtures/<module>/`. |

Regenerate the fixtures with `python tools/make_fixtures.py`. The output is deterministic.

Tests marked `@pytest.mark.network` are skipped unless `POINTVIEW_NETWORK_TESTS=1`. Tests that need the ONNX
model use `@pytest.mark.model` and skip when the model file is missing. Qt tests run with
`QT_QPA_PLATFORM=offscreen`; `conftest.py` sets it.

## 6. Dev environment

```bash
cd laptop
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"   # or reuse a shared venv
ruff check . && ruff format --check . && pytest
cd ../phone && npm ci && npx tsc --noEmit && npx eslint . && npx jest
```

`pytest` adds `laptop/` to `sys.path` (`pythonpath = ["."]`), so a venv without the editable install also
works from any checkout or worktree.

---

# Contract v2: God's Eye View looks and live layers

**Status: FROZEN** from the start of Phase 3. The additions below are backward compatible, so v1 code still works unchanged. They borrow from [God's Eye View](https://github.com/bilawalsidhu/gods-eye-view) (MIT, © 2026 Bilawal Sidhu): its sensor looks and its live public-data layers, brought onto the real world through the glasses. Any code or shader math ported from it keeps its copyright notice in a comment at the top of the ported file and is listed in the README.

**Division of roles:**
* **Phone:** only a pointer. It sends its rotation and compass heading, and the user picks layers and the look on it. The phone camera is never used.
* **VITURE glasses:** the only source of imagery.
  * The front color camera feeds detection and the passthrough backdrop.
  * The side grayscale tracking cameras feed the NVG backdrop.

## 7. Live layers and "only what you're looking at"

* **New entity kinds:** `aircraft`, `satellite`, `vessel`, `event`. `entities.LIVE_KINDS` lists them. `Entity.geo` (`GeoPos`) carries a position when the layer knows one.
* **New `LayerContext` fields:**
  * `unix_time`: wall clock. The mock pins it to the time the fixtures were recorded.
  * `view`: a `View` describing what the glasses face right now.
  * `pointer`: the phone pointer's world direction.
  * `focus_id`: the entity the pointer is snapped to.
* **Focus rule.** These apply to live layers:
  1. Return only entities that pass `world.in_view(…, ctx.view…, margin_deg=5)`. The app filters again, but layers should not even compute labels for other entities. Live layers get no edge arrows.
  2. Fetch data from the network by area around `ctx.phone.lat/lon`. Feeds are queried by area, not by cone, so fetch the area and filter to the view.
  3. Per-item enrichment (an aircraft's route or type details, a satellite's next pass, quake details) runs only for `ctx.focus_id`. Put the results in `meta`.
  4. With no view (no head pose yet) or no GPS fix, return `[]`.
* **Direction from a position:** use `world.look_angles(observer lat/lon/alt, target lat/lon/alt)`, which accounts for Earth curvature. The observer's altitude is 0 until the phone sends altitude.
* **Layer modules:** each one is a `Layer` that is non-blocking, works in a background thread, and caches to SQLite like `landmarks.py`. Tests use recorded HTTP responses, never the live network.

  | Module | Kind | Sources (keyless first) | Refresh |
  | --- | --- | --- | --- |
  | `pointview/live/aircraft.py` | aircraft | adsb.lol `/v2/point`, falling back to OpenSky `states/all` with a bounding box | 5–10 s, dead-reckoned between polls |
  | `pointview/live/satellites.py` | satellite | CelesTrak GP (`visual`, `stations`), propagated with `sgp4` | TLEs daily, positions each frame for satellites in view |
  | `pointview/live/events.py` | event, vessel | USGS earthquakes GeoJSON (keyless); AISStream vessels only if `AISSTREAM_API_KEY` is set | quakes every 5 min; vessels streamed |

* **Phone toggles:** `aircraft`, `satellites`, `vessels`, `events`. `interfaces.LAYER_KINDS` maps toggle names to kinds.

## 8. Looks and passthrough (`pointview/looks/`, `render.py`)

* **Looks:** `interfaces.LOOKS` = `normal, crt, nvg, flir, noir, snow`, ported from God's Eye View's `src/styles/` (`retro.js`, `surveillance.js`, `thermal.js`, `noir.js`, `snow.js`).
* **What a look applies to:** the whole composed frame, meaning the `Backdrop` plus the overlay.
* **On see-through glasses black means "nothing"**, so a look has no visible effect unless passthrough is on. Labels are re-tinted in every look.
* **`Backdrop(bgr, homography, gain)`** is drawn first. The app builds it from the passthrough camera's latest frame, using `world.backdrop_homography()` with the head pose at capture time and the pose now.
* **Which camera:**
  * NVG uses the tracking camera when `isinstance(glasses, TrackingCameraSource)`.
  * Every other look, and NVG on glasses without tracking frames, uses the front camera.
* **`Renderer.set_look(look, params)`:** `params` are each look's uniforms from God's Eye View (for example `gain`, `bloom`, `pixelation`), with God's Eye View's defaults.
* **CPU reference:** `pointview/looks/` holds a numpy/OpenCV version of each look as a pure function on a uint8 image. It must stay within 8 ms at 960×540 internal resolution on this CPU; the renderer upscales to the display. Tests use this path.
* **GPU path:** a GLSL path through Qt OpenGL is optional and only used when a GL context exists. It can only be verified on Windows hardware.
* **Phone message (optional, additive, still `v: 1`):** `look` (one of `LOOKS`), `passthrough` (boolean), and the new `layers` keys. When a message omits them, the previous value stands. The laptop's status message echoes `look` and `passthrough`.

## 9. Tracking cameras (`TrackingCameraSource`)

* **Interface:** an optional extra protocol on a glasses source. `VitureGlasses` implements it through the Carina stereo callback (8-bit grayscale; left camera only) and the SDK's exposure control. `MockGlasses` implements it by converting its video to grayscale.
* **Intrinsics and mounting rotation** of the tracking cameras are unknown until measured on hardware. Until then the defaults are a guessed wide field of view, about 110°, and identity mounting. They are flagged as such.

## 10. world.py v2

Five functions are added; their signatures are frozen and W7 implements the bodies:
* `ecef_from_geodetic`
* `look_angles`
* `in_view`
* `backdrop_homography`
* `sun_direction`

See the docstrings in `laptop/pointview/world.py`.

## 11. Contract v2.1: world mesh and tracked-object markers

* **What it's for:** walking around, you see a **sparse wireframe of the whole world**, meaning building outlines and a coarse terrain grid. It comes from the most recent 3D model available, in the style of God's Eye View's globe. Tracked objects are drawn on top as a **point or a square**: detections, aircraft, satellites, vessels, events. The mesh is everywhere; live data still only shows where the glasses look (§7).
* **`WorldMesh`** (in `interfaces.py`):
  * `vertices` are local ENU meters at `origin`; `edges` index vertex pairs.
  * The mesh is sparse. Typical sizes are 5k–50k edges within ~800 m.
* **`MeshSource`** (`pointview/mesh/`):
  * `mesh_near(lat, lon)` never blocks. It returns the most recent mesh, cached or fresh, and refetches on its own thread after the user moves about 200 m.
  * `ground_alt_m(lat, lon)` gives terrain height.
  * The default source combines:
    * **OSM buildings.** Fetched with Overpass `way["building"](around:…); out geom;`. Each building is extruded: the bottom is `min_height` (or 0) above terrain, the top is `height`, `building:levels`×3 m, or 6 m otherwise. Each extruded building contributes footprint edges, roof edges, and vertical edges at its corners.
    * **Terrain.** A grid built from AWS Terrain Tiles (Terrarium PNG, `s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`, no key). Lines are ~40 m apart near the user and sparser farther out.
  * It caches the last model on disk, so offline you still get the most recent one.
  * Attribution: "© OpenStreetMap contributors" and the terrain sources (Mapzen/AWS).
  * Google Photorealistic 3D Tiles (what God's Eye View uses) can be added later as another `MeshSource` with the user's own API key.
* **`world.project_mesh(...)`** (vectorized numpy):
  * Input: the mesh, the observer's geodetic position (eye height = terrain + `world.EYE_HEIGHT_M`), and the head pose.
  * Output: display-pixel segments clipped to the near plane and the screen, plus the distance of each segment, used for fade.
  * `world.enu_from_geodetic` handles the coordinate change.
* **Draw items:**
  * `WireMesh(segments, color, alpha, width_px)` is drawn right after `Backdrop`: thin, dim, and fading with distance.
  * `Marker(x, y, shape, color, size, label, highlighted)` draws a tracked object as a point or a square. The label is short, e.g. "UAL123 · 14 km", and is shown in full only for the snapped object.
* **Phone:** new toggle `layers.mesh` and new field `marker` (`"point"` | `"square"`, default `"square"`). Both are additive; an omitted value means unchanged.
* **Accuracy caveat:** the phone's GPS is good to ~5 m. Far geometry lines up well; geometry within ~30 m can be off by several degrees. Two things will improve it later:
  * 6DoF VIO for walking smoothness;
  * a "nudge" alignment, using a long-press on a building corner.
