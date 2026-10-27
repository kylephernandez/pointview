# PointView — Real-World Labels for VITURE Luma Ultra

Design doc · Oct 26, 2026 · @Kyle Hernandez

A Windows laptop in a backpack runs the glasses and puts labels on the real world. Your iPhone is a handheld pointer: aim it at a car, person or landmark to select it.

## Overview and goals

PointView is open source. The laptop reads the glasses' own front camera and head tracking, detects cars and people, and draws labels that sit on the real objects. The iPhone adds GPS, a compass, and a pointer you aim with your hand.

**Goals for v1**

- Label cars, bikes, buses, trucks and people in front of you in real time, with a rough distance.
- Label named landmarks in the direction you face, using the phone's GPS and OpenStreetMap.
- Line each label up with its real object to within about 2°, roughly a car's width at 50 m.
- Point the phone at anything labeled to select it and see details. A cursor in the glasses shows where you're pointing.
- Build everything on Windows. No Mac and no Apple Developer account are needed for v1.
- Publish under the MIT license, with a module layout modeled on God's Eye View.

**Non-goals**

- Identifying people. No face recognition, no names, no license plates. A person is only ever labeled "person".
- Recording. Camera frames stay in the laptop's memory and are never saved or uploaded.
- A pocket-sized setup. The backpack laptop is a prototype rig. A phone-only Android version is a later phase.
- Labels for things behind you or outside the glasses' view.

## Hardware setup

Three devices and one cable to the glasses. The iPhone can be wired or wireless.

| Part | Connects to | Carries | Requirement |
| --- | --- | --- | --- |
| VITURE Luma Ultra | Laptop, USB-C | Video to the glasses; camera, head pose and IMU back | Laptop USB-C port with DisplayPort Alt Mode (a DP or Thunderbolt logo) |
| Windows laptop (in backpack) | — | Runs everything | A GPU or NPU for detection, ideally 3+ hours on battery, set to keep running with the lid closed |
| iPhone 15 or newer (in hand) | Laptop, USB cable or Wi-Fi | Pointer orientation, GPS, compass, button presses | Apple Devices app on Windows for the USB driver |
| Backpack with airflow | — | — | A laptop working hard in a closed bag can overheat |

**Wired or wireless iPhone**

Only the pointer and the GPS and compass data travel from the phone. The labels on cars, people and landmarks come from the glasses' camera and head tracking over the wired glasses cable, so they are just as fast either way.

| Link | Typical added delay | Hiccups | Notes |
| --- | --- | --- | --- |
| USB (Personal Hotspot over the cable) | about 1–3 ms | Rare | Also charges the phone. A cable runs from your hand to the bag. |
| Wi-Fi (laptop joins the iPhone's hotspot) | about 3–15 ms | Occasional 50–200 ms stalls from interference or a dropped packet | No cable. Laptop gets internet from the phone either way. |
| Bluetooth LE | about 15–45 ms | Few | Needs a custom native module. Not worth it for v1. |

These are typical figures to confirm in M2, not measurements.

End to end, wireless pointing should feel close to wired: roughly 40–70 ms from moving your hand to the cursor moving, against about 35–55 ms wired. That includes the phone's sensor, the app, the link, and drawing a frame. Under about 50–70 ms a pointer generally feels responsive. Wi-Fi's main cost is the occasional stall, which makes the cursor jump. Two fixes hide most of it:

- **Timestamp and predict:** each phone sample carries its time and rotation speed. The laptop extends the motion forward to "now", so a sample 30 ms old still lands about where your hand is.
- **Smooth stalls:** if no sample arrives for 50 ms, keep moving the cursor along its last motion briefly, then hold it still.

## User experience

You walk around with labels floating on the world, and you use the phone like a TV remote. Point its top edge at something, and a cursor in the glasses follows. The nearest label snaps to it, and a tap shows details.

**Core flow**

1. Put on the glasses, start the laptop app (it can auto-start at login), and open the PointView app on the iPhone. The phone finds the laptop by itself.
2. Align once, in about 10 seconds. The glasses show a crosshair. Look at something at least 20 m away so the crosshair sits on it, point the phone at the same thing, and tap **Align**. This lines up the phone's direction with your head's.
3. Look around. Cars and people get labels within about a second, and landmarks are labeled along the horizon.
4. Point the phone. A small ring in the glasses shows where it's aimed. When the ring passes within about 5° of a label, it snaps on, the label brightens, and the phone gives a short buzz.
5. Tap **Info** to open a card for the selected item. Tap again to close it.

**What the glasses show**

The background is black, which is see-through on these glasses, so you see only what's drawn.

- **Cars and people:** small corner brackets plus a tag like `car · ~15 m`.
- **Landmarks:** the name above the horizon with a thin line down to its direction, and an arrow at the edge for ones just out of view.
- **Pointer ring:** a small hollow circle. It turns solid when snapped to something.
- **Info card:** next to the selected item. Vehicles: type and distance. People: "person" and distance only. Landmarks: name, distance, and a two-line summary from Wikipedia when OpenStreetMap links one.
- **Status corner:** phone connected, GPS accuracy, and alignment quality.
- **Field of view:** about 46° × 27°, worked out from VITURE's 52° diagonal. Text at least 24 px high on the 1080p display, one bright color per class.

**What the phone shows**

Large buttons you can hit without looking: **Info** (the whole lower half of the screen), **Align**, and toggles for people, vehicles and landmarks. Later, the volume buttons can work as Info and Back. That needs a custom build, so it's out of v1. The screen dims to save battery but stays on.

## Architecture

The laptop does all the work. The glasses supply camera frames and head pose and show the result, and the iPhone supplies pointing, GPS and button presses.

&#91;embedded content: PointView architecture · glasses, iPhone, laptop\]

The highlighted world model is the core: every input becomes a direction relative to true north, so detections, landmarks and the pointer can be compared and drawn the same way. A new layer, such as aircraft or transit borrowed from God's Eye View, only needs to output entities in that form.

## Data layers

v1 has two layers, both running on the laptop. Each one turns its input into the same entity format, so new layers plug in without touching the renderer. This is the God's Eye View pattern.

| Layer | Input | Method | Refresh | License note |
| --- | --- | --- | --- | --- |
| Vehicles and people | Glasses front camera, 1920 × 1080 at 30 fps | Shrink each frame, then run an object detector on the laptop's GPU via ONNX Runtime. COCO classes: person, bicycle, car, motorcycle, bus, truck. | Target 15–30 detections per second | Use an Apache-2.0 model such as EfficientDet or RT-DETR. Avoid Ultralytics YOLO models, which are AGPL-3.0. |
| Landmarks | iPhone GPS + OpenStreetMap | Overpass API query for named features within 2 km (`tourism`, `historic`, `place_of_worship`, `natural=peak`, named buildings) | On start, then every 250 m you move. Cached on disk. | OSM data is ODbL. Show "© OpenStreetMap contributors". |
| Landmark details | OSM `wikipedia` tag | Wikipedia REST summary endpoint, fetched only when you select the landmark | On demand, cached | Wikipedia text is CC BY-SA. Show the source on the card. |

**Shared entity format**

```
Entity {
  id: string             // stable per tracked object or OSM feature
  kind: 'person' | 'vehicle' | 'landmark'
  label: string          // "car", "bus", "Golden Gate Bridge"
  worldYawDeg: number    // direction from you, relative to true north
  worldPitchDeg: number  // up/down from the horizon
  distanceM?: number     // estimated (detections) or computed (landmarks)
  sizeDeg?: { w, h }     // angular size, for drawing brackets
  confidence?: number
  updatedAt: number
}
```

Entities are stored as **world directions**, not screen pixels. When you turn your head between detections, the glasses' head pose moves every label to its new spot right away. Labels stay stuck to the world even at 15 detections per second.

**Smoothing.** A small tracker matches objects between frames. An object shows only after 3 consecutive detections and stays 300 ms after it is lost. This is the same idea as God's Eye View drawing flights one poll behind so they move smoothly.

**Privacy rules in code.** Frames are processed in memory and dropped. Only class labels and directions leave the detection module. Nothing is logged, saved or uploaded. The only network calls are the Overpass and Wikipedia lookups, which send an approximate position or a landmark name.

## Pointing math

Everything becomes a direction in one shared world frame (true north, horizon), then gets drawn through the glasses' current head pose. Detections, landmarks and the phone's pointer all follow the same last step, so they line up with each other.

**1. Detections: glasses pixel to world direction.** A box centered at camera pixel (u, v) is at these angles from the camera's center. f is the camera's focal length in pixels, measured once with an OpenCV checkerboard calibration if the SDK doesn't provide it.

```latex
\alpha = \arctan\frac{u - W/2}{f}, \qquad \beta = \arctan\frac{v - H/2}{f}
```

Rotate that direction by the head pose at the frame's capture time, then add the north offset from alignment (step 5).

**2. Landmarks: bearing from you** (your position φ₁, λ₁ from the iPhone's GPS; the landmark's φ₂, λ₂):

```latex
\theta = \operatorname{atan2}\big(\sin\Delta\lambda\,\cos\varphi_2,\; \cos\varphi_1\sin\varphi_2 - \sin\varphi_1\cos\varphi_2\cos\Delta\lambda\big)
```

**3. Distance to a car or person** from box height h and a typical real height H (1.7 m for a person, 1.5 m for a car). Expect about ±30% error, so tags say `~15 m`.

```latex
d \approx \frac{f \cdot H}{h}
```

**4. Drawing on the glasses.** Turn each world direction into angles α′, β′ from where your head points now, using the latest head pose (polled at 120 Hz or more). Then place it on the display, with the display's horizontal field of view FOV\_g ≈ 46°:

```latex
x_g = \frac{W_g}{2} + f_g\tan\alpha', \qquad f_g = \frac{W_g/2}{\tan(\mathrm{FOV}_g/2)}
```

A small fixed offset between the camera and the display is measured once during setup and added here.

**5. The pointer ray and alignment.** The phone's motion sensors give its rotation, which the app turns into the direction of the phone's top edge. That direction is drawn as the cursor through step 4. The phone and the glasses each have their own idea of "straight ahead", so the one-tap **Align** links them. When you tap, you're looking at and pointing at the same thing, so:

- **Phone-to-head offset** = glasses yaw − phone yaw. This makes the cursor land where you point, and it doesn't depend on the compass.
- **North offset** = the phone's compass heading − glasses yaw. This orients landmarks. Compass error (5–15° near cars and metal) affects only landmarks. Long-press **Align** while pointing at a labeled landmark to correct north exactly.

**6. Snapping and lag**

- **Snap:** pick the label closest in angle to the pointer ray, if it's within 5°. Keep the current pick until another is at least 2° closer, so it doesn't flicker between neighbors.
- **Prediction:** each phone sample carries its timestamp and rotation speed ω. The laptop draws the cursor where the hand is now: ψ\_now = ψ\_sample + ω × (t\_now − t\_sample).

**7. Accuracy limits**

- **Pointer parallax:** your hand is about 40 cm below your eyes, so the ray is about 2° off for a target 10 m away and under 1° beyond 25 m. Snapping hides this.
- **Drift:** head tracking and the phone's gyroscope both drift slowly in yaw. The status corner shows alignment quality, and re-aligning takes one tap.
- **Label depth:** the glasses show every label at the same apparent distance. Placing labels at their real depth with VITURE's 3D side-by-side mode is a possible v2 feature.

## Tech stack

The laptop app is Python, and the iPhone app is a small Expo app that runs inside **Expo Go**. That means no Mac, no cloud iOS builds and no Apple Developer account for v1.

**Laptop app (Python 3.12, Windows)**

| Need | Tool | Notes |
| --- | --- | --- |
| Glasses head pose, IMU, camera control | VITURE Glasses SDK (Windows x86\_64), called through `ctypes` | Pose polled at 120 Hz or more |
| Glasses camera frames | The SDK's pass-through stream, or OpenCV `VideoCapture` | The camera is a standard USB webcam (UVC), so OpenCV may open it directly |
| Object detection | ONNX Runtime with DirectML | Uses any recent Windows GPU or NPU |
| Drawing on the glasses | PySide6 (Qt), a borderless full-screen window on the glasses' display | Black background, 60 fps |
| Phone link | `websockets` server on port 8765 | JSON messages, about 60 per second |
| Landmarks | `httpx` to Overpass and Wikipedia, cached in SQLite | Internet through the phone's hotspot |

**iPhone app (Expo, TypeScript, in Expo Go)**

| Need | Library |
| --- | --- |
| Phone rotation and rotation speed | `expo-sensors` DeviceMotion, at about 60 Hz |
| GPS and compass heading | `expo-location` (`watchPositionAsync`, `watchHeadingAsync`) |
| Snap buzz | `expo-haptics` |
| Keep the screen awake | `expo-keep-awake` |
| Link to the laptop | The built-in `WebSocket` |

**The link**

- Each message carries a timestamp, the phone's rotation (quaternion), rotation speed, compass heading and its accuracy, GPS position and accuracy, and button presses.
- **Finding the laptop:** on the iPhone's hotspot, wired or Wi-Fi, the laptop gets an address in the hotspot's small range, usually 172.20.10.x. The phone tries each address on port 8765 until one answers, which takes under a second. On other Wi-Fi networks you enter the laptop's address once.

**Repo layout**

```
laptop/pointview/
  glasses.py      SDK bindings: camera, head pose
  detection.py    model + tracker → entities
  landmarks.py    Overpass, Wikipedia, cache → entities
  world.py        frames, alignment, pointer ray, snapping (pure math, unit-tested)
  render.py       the glasses window
  phone_link.py   WebSocket server
phone/
  App.tsx         buttons and status
  sensors.ts      motion, heading, GPS
  link.ts         discovery and sending
```

**Daily loop**

1. At your desk, plug the glasses into the laptop and run `python -m pointview`.
2. On Windows, run `npx expo start` in `phone/` and scan the QR code with Expo Go. Code changes reload on the phone in seconds.
3. To go outside, set the laptop app to start at login and the lid to do nothing when closed, and pack it.

**Open source:** MIT license. GitHub Actions runs `pytest` on `world.py`, plus lint and type-checks for both apps, on every pull request. No keys are needed, since Overpass and Wikipedia are free.

**Later, with a custom iPhone build:** volume-button controls, and streaming the phone's camera to sharpen the pointer (see Milestones).

## Milestones, risks and open questions

The first milestone proves the riskiest piece: talking to the glasses from Python on Windows. Everything else builds on it.

**Milestones**

1. **M0 — Glasses bring-up:** glasses work as a second display, Python reads the head pose and camera frames through the SDK, and a test window draws a crosshair that stays fixed in the world as you turn your head.
2. **M1 — Labels on the world:** detection on the glasses' camera, tracker, brackets and tags drawn through the head pose. Measure detections per second and the delay from camera to glasses.
3. **M2 — Phone link:** Expo Go app sending motion, compass and GPS. Measure latency and stalls, wired and on Wi-Fi.
4. **M3 — Pointer:** Align, cursor ring, snapping, haptics, info cards.
5. **M4 — Landmarks:** Overpass, caching, bearings, Wikipedia summaries, north correction on a known landmark.
6. **M5 — Backpack field test:** a 60-minute walk measuring laptop heat, battery for the laptop, glasses and phone, and alignment drift.
7. **M6 — Open-source release:** README, setup guide, `CONTRIBUTING.md` with a "how to add a layer" guide, demo video.

**Phase 2 ideas**

- **Phone-camera sync for exact pointing:** a custom iPhone build streams small phone-camera frames (about 320 px, 5 per second) to the laptop. Matching image features against the glasses' frame finds exactly what the phone is aimed at, which removes alignment drift from the pointer.
- **No laptop:** run the same Python logic on an Android phone with VITURE's Android SDK.
- **Labels at real depth** using the glasses' 3D side-by-side mode.

**Risks**

| Risk | Effect | Mitigation |
| --- | --- | --- |
| VITURE SDK hard to call from Python | M0 slips | Try OpenCV for the camera first. Wrap only the pose calls. Fall back to a small C++ helper. |
| Laptop overheats in the backpack | Throttling, then shutdown | Bag with mesh panel, cap detection at 15 per second, monitor temperature |
| Laptop sleeps when packed | Everything stops | Lid action "do nothing", a power plan that never sleeps, auto-start at login |
| Wi-Fi stalls | Cursor jumps | Prediction and stall smoothing, or switch to the USB cable |
| Compass error | Landmarks 5–15° off | Long-press Align on a known landmark. Hide landmarks when heading accuracy is poor. |
| Head-tracking or gyroscope drift | Pointer slowly misaligns | Alignment-quality indicator, one-tap re-align |
| A visible camera on your face in public | People feel recorded | On-device only, no recording, stated in the README. Follow local laws on filming. |

**Open questions**

- [ ] Which laptop? Check that it has a USB-C port with DisplayPort Alt Mode, and whether it has a GPU or NPU.
- [ ] Does the VITURE SDK give the front camera's focal length, or do we calibrate with a checkerboard?
- [ ] Point with the phone's top edge (like a remote) or its back (like a camera)? The top edge is assumed.
- [ ] Keep the name PointView?

## Sources

- [God's Eye View on GitHub](https://github.com/bilawalsidhu/gods-eye-view): the layer-per-module architecture and smoothing approach this design borrows.
- [VITURE XR Glasses SDK](https://www.viture.com/developer/glasses-sdk/glasses): supported platforms including Windows, Luma Ultra's 1080p30 front camera, stereo tracking cameras, 6DoF pose and IMU.
- [VITURE Luma Ultra review, Gaming Nexus](https://www.gamingnexus.com/Article/16314/VITURE-Luma-Ultra/): 52° field of view, Sony Micro-OLED display, and tracking that relies on the host device's processing.
