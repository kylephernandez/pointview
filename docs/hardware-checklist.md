# Hardware checklist (M0–M3)

Everything in this repo has been tested only with mocks: a synthetic glasses video and pose track, a
scripted fake phone, and recorded or hand-written HTTP responses. This list checks it on the real laptop,
glasses and iPhone, in milestone order.

How to use it:
* Do the steps in order.
* Each step says what to run and what you should see.
* If a step fails, note the actual result in the PR or an issue before going on. Later steps depend on
  earlier ones.

Commands run from `pointview\laptop` with the venv active, unless they say otherwise.

## Setup (before M0)

1. **Laptop video output.** Plug the glasses into the USB-C port marked DP or Thunderbolt.
   *Expected:* Windows › Settings › Display shows a second 1920×1080 display. Write down its name, for
   example "VITURE"; later steps pass it as `--screen`. If no display appears, the port lacks DP Alt
   Mode, so try the other port.

2. **Install and run the test suite on Windows.** Follow README › Laptop setup, then run
   `python tools\download_model.py` and `pytest`.
   *Expected:* `download_model.py` prints a verified sha256. `pytest` reports the same results as CI, with
   only the live-network test skipped.

3. **DirectML is available.** Run
   `python -c "import onnxruntime as o; print(o.get_available_providers())"`.
   *Expected:* the list contains `DmlExecutionProvider`. If it only shows `CPUExecutionProvider`,
   detection runs on the CPU, at about 5 per second on a 4-core machine. Note which GPU or NPU the laptop
   has.

4. **The renderer on the glasses (mock data).** Run
   `python -m pointview --mock --screen VITURE --duration 20`.
   *Expected:* on the glasses, in this order:
   * a crosshair;
   * after about 1 s, green landmark names ("Golden Gate Bridge", "Fort Point", …) with thin lines down
     to the horizon;
   * a white ring that turns solid on "Golden Gate Bridge";
   * a green info card with a two-line Wikipedia summary and "Wikipedia · CC BY-SA";
   * the status corner, top left;
   * "© OpenStreetMap contributors", bottom right.

   The background must look see-through, with no black glow. The text must be readable. Note whether
   24 px text is comfortable, and whether the window covers the glasses' display exactly with no taskbar.

5. **Live Overpass and Wikipedia through the iPhone hotspot.** Connect the laptop to the hotspot. Then run
   `$env:POINTVIEW_NETWORK_TESTS=1; pytest tests\test_landmarks.py -k live -rs`.
   *Expected:* `1 passed`.
   This is the first time PointView's Overpass query runs against the real server; until now the tests
   used a hand-written response. If it fails, paste the error into an issue. The query is
   `landmarks.build_query()` and the parser is `landmarks.parse_overpass()`.

## M0: glasses bring-up

6. **Install the VITURE SDK.** Download it from VITURE's developer site. Set `VITURE_SDK_PATH` to the
   folder that contains `libglasses.dll`. In Device Manager, open the glasses › Details › Hardware Ids,
   then set `VITURE_PRODUCT_ID=0x<PID>` from the `VID_35CA&PID_<PID>` entry.
   *Expected:* the folder has `libglasses.dll`, plus any DLLs it depends on. Record the PID in the PR.

7. **Read the head pose.** Run `python tools\glasses_check.py`.
   *Expected:* about 5 lines per second reading `yaw … pitch … roll … pose … Hz camera …`, with no
   `SdkError`.
   If `start()` fails, the most likely causes are:
   * the DLL name or path;
   * a missing dependent DLL;
   * the overloaded `xr_device_provider_create`.

   See `glasses/viture_sdk.py`.

8. **Pose signs.** Keep `glasses_check.py` running.
   1. Face forward and level. *Expected:* yaw, pitch and roll are all about 0.
   2. Turn your head about 45° to the right. *Expected:* yaw goes to about +45, not −45.
   3. Look up. *Expected:* pitch is positive.
   4. Tilt your head toward your right shoulder. *Expected:* roll is positive, and yaw and pitch change
      by less than 3°.

   If yaw has the wrong sign, set `CARINA_POSE_IS_WORLD_TO_DEVICE = True` in `glasses/viture_sdk.py`. If
   the axes are swapped, fix `_M`, which converts OpenGL axes to PointView's.

9. **Pose rate.** Read the `pose … Hz` column.
   *Expected:* at least 120 Hz. If it is lower, the polling thread is too slow and should use the SDK
   pose callback instead. Note the rate.

10. **Camera frames.**
    * SDK pass-through (the default): the `camera` column shows `1920x1080 age <100 ms`.
    * If it says `no frame`, run `python -c "from pointview.glasses.viture import list_cameras; print(list_cameras())"`,
      then rerun with `--camera-index N`, using the index of the 1920×1080 camera.

    *Expected:* frames arrive at about 30 fps from one of the two paths. Record which path works.

11. **The M0 test: a world-fixed crosshair.** Run `python tools\glasses_check.py --crosshair --screen VITURE`.
    Look at a distant point such as a door frame, then turn your head slowly left, right, up and down.
    *Expected:* the crosshair stays on that point within about 1°. It must not follow your head.
    Drift should be slow, under a few degrees per minute. Fast head turns may lag slightly; note how
    much.

## M1: labels on the world

12. **Calibrate the camera.** Print a 9×6 inner-corner checkerboard with 25 mm squares. Then run
    `python tools\calibrate_camera.py capture --camera-index N`. Take 15–25 views covering the whole
    frame, then run `python tools\calibrate_camera.py solve calib_frames\*.png --out intrinsics.json`.
    *Expected:* RMS under 0.7 px. The tool also prints a horizontal FOV; the code currently assumes 90°,
    so note the real value. Delete `calib_frames\` afterwards.

13. **Detection on the live camera.** Run `python -m pointview --screen VITURE --intrinsics intrinsics.json --no-landmarks`,
    with no phone needed yet. Stand where cars and people pass.
    *Expected:*
    * cyan brackets on vehicles and amber brackets on people within about 1 s, with tags like `car · ~15 m`;
    * a person is never labeled anything but "person";
    * on exit, the log prints `detection: X frames/s`. Target 15; record the number.

14. **Label alignment and camera offset.** With step 13 running, look at a parked car about 20–50 m
    away and turn your head slowly.
    *Expected:* the brackets stay on the car, within about 2° (one car-width at 50 m), while you turn.
    If they sit offset consistently, measure the offset in degrees and rerun with
    `--camera-offset <yaw>,<pitch>` until they line up. Record the values.

15. **Camera-to-glasses delay.** Turn your head quickly from side to side.
    *Expected:* the labels lag behind the real objects only briefly and settle within about 0.1 s. If
    they visibly overshoot or lag, tune `--exposure-latency-ms` (default 30) until they stick best.
    Record the value.

16. **Distance estimates.** Pace out a person at about 10 m and a car at about 20 m.
    *Expected:* the tags read within about ±30% (`~10 m`, `~20 m`).

## M2: phone link

17. **Firewall and address.** Join the laptop to the iPhone's Personal Hotspot, first by USB cable. Run
    `ipconfig`.
    *Expected:* an adapter with an IPv4 address in `172.20.10.2–14`. On first run, Windows asks to allow
    Python on private networks; allow it.

18. **Start the app in Expo Go.** In `phone\`, run `npm ci`, then `npx expo start`, and scan the QR code.
    *Expected:*
    * the PointView app opens in Expo Go;
    * iOS asks for Location (While Using), Motion and Local Network, and you allow all three;
    * if Expo Go refuses the project's SDK version, note the version it supports.

19. **Discovery.** Run `python tools\phone_monitor.py` on the laptop, then open the app.
    *Expected:*
    * the app's status line shows "Connected 172.20.10.x:8765" within about 1 s;
    * the monitor prints `~60 Hz` lines;
    * force-quit and relaunch the app: it reconnects by itself.

20. **Phone axes.** Watch `ptr yaw` and `pitch` in the monitor.
    *Expected:*
    * phone flat, top edge pointing ahead: pitch about 0;
    * tilt the top edge up: pitch rises;
    * rotate the phone clockwise, seen from above: `ptr yaw` increases;
    * roll the phone about its top edge: yaw and pitch barely change;
    * during a clockwise turn, ω[2] is negative.
    Record any sign that is reversed; the fix goes in `phone/src/sensors.ts`.

21. **Compass and GPS.** Point the top edge along a street whose bearing you know, using a map app.
    *Expected:*
    * `heading` is within about 15° of the street's bearing;
    * `±` reads 10–40 (a coarse value mapped from iOS's calibration level);
    * outdoors, `GPS ±` is 3–15 m.

22. **Rate and stalls: wired vs Wi-Fi.** Run `python tools\phone_monitor.py --seconds 120` twice: once
    with the laptop on USB hotspot, once on Wi-Fi hotspot. Move the phone around during each run.
    Fill in:

    | Link | rate Hz | stalls >50 ms per min | longest gap ms | worst jitter ms |
    | --- | --- | --- | --- | --- |
    | USB |  |  |  |  |
    | Wi-Fi |  |  |  |  |

    *Expected:* about 60 Hz on both. USB should have almost no stalls. Wi-Fi may show occasional
    50–200 ms gaps, as the design doc predicts.

## M3: pointer

23. **Align.** Run `python -m pointview --screen VITURE --intrinsics intrinsics.json` with the phone
    connected. The glasses show a crosshair. Put it on something at least 20 m away, point the phone's
    top edge at the same thing, and tap **Align**.
    *Expected:*
    * the phone gives a success buzz;
    * the crosshair disappears;
    * the status corner shows "aligned 0 min ago" and "north: compass";
    * a hollow ring appears where the phone points.

24. **The cursor follows the hand.** Sweep the phone slowly, then quickly.
    *Expected:*
    * the ring tracks the top edge's direction, within about 2° beyond 25 m (parallax is larger up close);
    * it feels responsive, with no obvious lag. Compare USB and Wi-Fi;
    * on Wi-Fi, a stall shows as a short glide, not a jump.

25. **Snap.** Move the ring near a labeled car, person or landmark.
    *Expected:*
    * within about 5°, the ring turns solid, the label brightens, and the phone gives a light buzz;
    * between two nearby labels, the selection does not flicker.

26. **Info.** While snapped, tap **Info**. Tap it again.
    *Expected:*
    * the first tap gives a medium buzz and opens a card next to the item;
    * vehicles show their type and distance;
    * people show only "person" and a distance;
    * landmarks show name, distance and two lines of Wikipedia with its source line;
    * the second tap closes the card.

27. **North correction on a landmark.** Point at a labeled landmark you can see clearly and long-press
    **Align** (at least 0.6 s).
    *Expected:*
    * a success buzz;
    * the status corner shows "north ✓ landmark";
    * the landmark's label moves onto the real landmark, and other landmarks shift by the same amount.

    This is M4 behaviour, but it is wired up already.

28. **Drift over 10 minutes.** Wear the setup for 10 minutes, looking around normally, then point at the
    step 23 object.
    *Expected:* the ring is within about 3° of it, and re-aligning takes one tap. Record the drift.
