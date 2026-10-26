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
| `pointview/__main__.py`, `pointview/scene.py` | lead | Main loop; turns entities, the pointer and the selection into `DrawItem`s |

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
