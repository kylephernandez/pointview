# PointView

Real-world labels for **VITURE Luma Ultra** glasses, with your **iPhone as a handheld pointer**.

A Windows laptop in a backpack reads the glasses' front camera and head tracking. It detects cars, bikes
and people, looks up nearby landmarks in OpenStreetMap, and draws labels on the glasses' see-through
display. Aim the phone's top edge at a label to select it, then tap **Info** for details.

* Spec: [docs/design.md](docs/design.md)
* Module contracts (frames, units, APIs): [docs/contracts.md](docs/contracts.md)
* Phone ↔ laptop protocol: [docs/phone-protocol.md](docs/phone-protocol.md)
* Bring-up on real hardware: [docs/hardware-checklist.md](docs/hardware-checklist.md)

> **Status: pre-hardware.** Everything here is tested with mocks: a synthetic glasses video and pose
> track, a scripted fake phone, and recorded or hand-written HTTP responses. Nothing has run on real
> glasses or a real iPhone yet. That is what the hardware checklist is for.

## Repository layout

```
laptop/                      Python 3.12 app (package `pointview`)
  pointview/
    __main__.py              python -m pointview [--mock]
    app.py                   per-frame loop: glasses → layers → world → scene → renderer; Align/Info
    scene.py                 entities + pointer + selection → DrawItems (display pixels)
    world.py                 frames, alignment, pointer ray, snapping, prediction (pure math)
    glasses/                 VitureGlasses (VITURE SDK via ctypes), MockGlasses, pose buffer
    detection/               RT-DETRv2 ONNX detector, IoU tracker, DetectionLayer
    landmarks.py             Overpass + Wikipedia + SQLite cache, LandmarkLayer
    render.py                PySide6 borderless full-screen window on the glasses
    phone_link.py            WebSocket server on :8765 (PhoneLink, PhoneLinkThread)
    entities.py, interfaces.py, schemas/   frozen contracts
  tools/
    download_model.py        fetch + verify the detector model
    glasses_check.py         M0: pose/camera readout, world-fixed crosshair
    calibrate_camera.py      M1: checkerboard → intrinsics.json
    phone_monitor.py         M2: live link rate / stalls / sensor readout
    fake_phone.py            scripted phone simulator
    make_fixtures.py         regenerate the synthetic test video + pose track
  tests/                     pytest (unit, property-based, offscreen render, end-to-end)
phone/                       Expo (SDK 58) TypeScript app for Expo Go
```

## Laptop setup (Windows 11, x86_64)

Requirements:
* a USB-C port with DisplayPort Alt Mode (DP or Thunderbolt logo) for the glasses
* Python **3.12** from python.org
* Git
* ideally a DirectX 12 GPU or NPU for detection through DirectML

```powershell
git clone https://github.com/kylephernandez/pointview
cd pointview\laptop
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"           # installs onnxruntime-directml on Windows
python tools\download_model.py    # RT-DETRv2-R18, 81 MB, sha256-verified
pytest                            # same suite as CI
```

### VITURE SDK

PointView talks to the glasses through VITURE's Glasses SDK for Windows (`libglasses.dll`), loaded with
`ctypes`. Download the SDK from the [VITURE developer site](https://www.viture.com/developer/glasses-sdk/glasses)
(it may require registering). This repo does not ship it. Then set:

```powershell
$env:VITURE_SDK_PATH = "C:\path\to\sdk\folder\containing\libglasses.dll"
$env:VITURE_PRODUCT_ID = "0x...."   # Device Manager → glasses → Hardware Ids → VID_35CA&PID_xxxx
```

The bindings follow VITURE's public documentation, but they have not been run against the real DLL. See
`laptop/pointview/glasses/viture_sdk.py` for every assumption, and the hardware checklist for how to
verify each one.

### Run

```powershell
python -m pointview --screen VITURE                     # real glasses + phone
python -m pointview --mock                              # no hardware: mock glasses + fake phone
python -m pointview --mock --duration 15 --screenshot-dir shots   # headless demo with PNGs
```

Useful flags:
* `--intrinsics intrinsics.json` (from `tools/calibrate_camera.py`)
* `--camera-offset yaw,pitch` (camera-to-display offset in degrees)
* `--camera-index N` (OpenCV camera instead of the SDK stream)
* `--no-detection`, `--no-landmarks`
* `--port 8765`

`python -m pointview --help` lists all of them.

Windows will ask whether Python may accept connections on port 8765. Allow it on **private** networks.
The iPhone hotspot counts as one.

To run PointView in the backpack, set the lid-close action to "Do nothing" and use a power plan that
never sleeps. Start it at login with a shortcut in `shell:startup`.

## iPhone app (Expo Go, no Mac needed)

1. Install **Expo Go** from the App Store. Its version must support **Expo SDK 58**, which this app uses.
2. On the laptop, install Node.js 22 LTS, then:
   ```powershell
   cd pointview\phone
   npm ci
   npx expo start
   ```
3. Scan the QR code with the iPhone camera. The app opens in Expo Go and reloads when code changes.
4. Allow **Location (While Using)**, **Motion** and **Local Network** when asked.
5. Turn on the iPhone's **Personal Hotspot** and connect the laptop to it, by USB cable (needs the Apple
   Devices app) or by Wi-Fi.
6. The app scans `172.20.10.2–14:8765` and connects by itself. On another network, tap the status line
   and enter the laptop's address once.

Controls:
* **Info** (the whole lower half of the screen): show or hide details for the selected label.
* **Align**: tap while looking at, and pointing at, the same far object.
* **Long-press Align** on a landmark label: corrects north exactly.
* Toggles: people, vehicles, landmarks.

## Development

```bash
cd laptop && ruff check . && ruff format --check . && pytest
cd phone && npx tsc --noEmit && npx eslint . && npx jest
```

CI (`.github/workflows/ci.yml`) runs both on every pull request, on Linux, with Qt offscreen.

Test markers:
* Tests that need the ONNX model skip when `laptop/models/*.onnx` is missing.
* Live API tests run only with `POINTVIEW_NETWORK_TESTS=1`.

## Privacy

* Camera frames are processed in memory and dropped. They are never saved, logged or uploaded.
* People are only ever labeled "person". There is no face recognition, no identity and no license-plate
  reading.
* The only network calls are Overpass (your approximate position) and Wikipedia (a landmark's title).
* A camera on your face in public can make people feel recorded. Follow local laws on filming.

## Licenses and attribution

* **PointView** is MIT licensed. See [LICENSE](LICENSE).
* **Object detector**: RT-DETRv2-R18 ([PekingU/rtdetr_v2_r18vd](https://huggingface.co/PekingU/rtdetr_v2_r18vd),
  ONNX export [onnx-community/rtdetr_v2_r18vd-ONNX](https://huggingface.co/onnx-community/rtdetr_v2_r18vd-ONNX)),
  Apache-2.0, trained on COCO. No Ultralytics or AGPL code or models are used. Details:
  [laptop/models/README.md](laptop/models/README.md).
* **Map data** © OpenStreetMap contributors, available under the
  [Open Database License (ODbL)](https://www.openstreetmap.org/copyright). The glasses show
  "© OpenStreetMap contributors" whenever landmarks are on screen.
* **Landmark summaries** come from Wikipedia under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  Each info card shows "Wikipedia · CC BY-SA".
* **Test fixtures**:
  * a CC0 street photo from Wikimedia Commons ([source](laptop/tests/fixtures/detection/README.md))
  * a recorded Wikipedia summary (CC BY-SA)
  * a hand-written Overpass response in OSM format ([provenance](laptop/tests/fixtures/landmarks/README.md))
* **Third-party libraries**:
  * ONNX Runtime (MIT)
  * OpenCV (Apache-2.0)
  * PySide6 / Qt (LGPLv3, used as a dynamically linked library)
  * websockets (BSD-3-Clause)
  * httpx (BSD-3-Clause)
  * NumPy (BSD-3-Clause)
  * jsonschema (MIT)
  * Expo and React Native (MIT)
* The architecture follows [God's Eye View](https://github.com/bilawalsidhu/gods-eye-view)'s
  layer-per-module pattern.
