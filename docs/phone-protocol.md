# Phone link protocol (v1)

The iPhone pointer and the laptop talk over one WebSocket. The laptop is the server at
`ws://<laptop>:8765/`, and the phone connects to it. Every frame is UTF-8 JSON text.

* Phone → laptop: [`phone_message.schema.json`](../laptop/pointview/schemas/phone_message.schema.json), about 60 per second.
* Laptop → phone: [`laptop_message.schema.json`](../laptop/pointview/schemas/laptop_message.schema.json), when needed.
* TypeScript mirror: [`phone/src/protocol.ts`](../phone/src/protocol.ts).

## Phone → laptop sample

```json
{"v": 1, "t_ms": 1793050000016, "quat": [0.996, 0, 0, -0.087], "omega": [0, 0, -0.5],
 "heading_deg": 10.0, "heading_acc": 12.5, "lat": 37.8078, "lon": -122.475, "gps_acc": 8.0,
 "buttons": ["align"], "layers": {"people": false}}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `v` | `1` | Protocol version. |
| `t_ms` | number | Phone `Date.now()` when the motion sample was taken (ms). It is not the send time. |
| `quat` | `[w,x,y,z]` | Attitude: rotates phone-frame vectors (x = right edge, y = top edge, z = out of the screen) into a gravity-aligned frame (z up, arbitrary yaw). Unit norm. |
| `omega` | `[x,y,z]` | Rotation rate in **rad/s**, in the phone frame. |
| `heading_deg` | number \| null | True heading of the top edge, clockwise from north, `[0, 360)`. If true heading is unavailable (iOS reports −1), send the magnetic heading. null if there is none. |
| `heading_acc` | number \| null | Heading accuracy in degrees. |
| `lat`, `lon` | number \| null | WGS84 degrees. null before the first fix. |
| `gps_acc` | number \| null | Horizontal accuracy in meters. |
| `buttons` | array | Presses since the previous message, in order: `info`, `align`, `align_long` (Align held for 600 ms or more). Each press is sent exactly once. |
| `layers` | object, optional | `{people, vehicles, landmarks}` booleans. Send it whenever a toggle changes; the phone may also resend it. When it is omitted, the previous value stands. All layers start on. |

Unknown fields are rejected (`additionalProperties: false`). Breaking changes need a version bump; the optional screen-tracking extension below is only sent after the laptop advertises support.

### Building `quat` and `omega` in Expo Go

`expo-sensors` `DeviceMotion` reports `rotation: {alpha, beta, gamma}` in radians, as W3C DeviceOrientation
intrinsic Z‑X′‑Y″ Euler angles, and `rotationRate: {alpha, beta, gamma}`. The phone app:

1. Builds `q = qz(alpha) ⊗ qx(beta) ⊗ qy(gamma)`, where `qa(θ)` is a rotation by θ about axis a. This takes
   the device frame into the W3C earth frame (x east-ish, y north-ish, z up).
2. Converts the rotation rate to rad/s, taking the units from the platform: Expo documents degrees per second.
   `omega = [rotationRate.beta, rotationRate.gamma, rotationRate.alpha]` (about x, y, z).
3. Sends at about 60 Hz (`DeviceMotion.setUpdateInterval(16)`), one message per motion sample, with
   `t_ms = Date.now()` taken in the motion callback.

The yaw origin of `alpha` does not matter, because Align removes it. Pitch must be gravity-referenced. The
axis order and signs are checked on the device in the hardware checklist (M2).

## Laptop → phone

```json
{"type": "haptic", "style": "snap"}
{"type": "status", "aligned": true, "selected": "car", "fps": 58.9}
```

* `haptic`: `snap` (light impact) when the pointer snaps to a new item, `select` (medium) when the info card
  opens, `align_ok` (success notification), `error` (error notification).
* `status`: sent at about 2 Hz for the phone's status line.

## Discovery

On the iPhone's Personal Hotspot (USB or Wi-Fi), the laptop gets an address in `172.20.10.2`–`172.20.10.14`.
The phone opens a WebSocket to every address on port 8765 in parallel, with a 700 ms timeout each, and keeps
the first one that opens. It closes the others. If nothing answers, the app shows a field for a manual
`host[:port]`, remembered across launches. On disconnect it retries with backoff (0.5 s, doubling, at most
5 s), last good address first.

## Laptop-side timing

The laptop maps `t_ms` onto its own clock with a minimum-latency filter: `offset = min(received_at −
t_ms/1000)` over the last 2 s. Cursor prediction uses `ψ_now = ψ_sample + ω·(t_now − t_sample)`, capped at
150 ms. A link counts as stalled after 50 ms without a sample. During a stall the cursor keeps extrapolating
up to the cap, then holds.

## Optional screen tracking extension

New laptops add `tracking` to status messages (the normal status fields are still required):

```json
{"type":"status","aligned":false,"selected":null,"tracking":{
  "session":"0123456789abcdef0123456789abcdef",
  "dictionary":"DICT_6X6_1000","marker_id":42,
  "cells":["00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000"],
  "state":"searching"}}
```

The cells above illustrate the shape only; display **the actual received cells**, which encode the
ID in the named OpenCV dictionary. `0` is black, `1` is white. The phone adds a one-cell white quiet
zone outside this 8×8 square. The reference marker is in `laptop/tests/fixtures/phone_tracking.json`.

A tracking-capable phone sends the following optional field **on every motion sample**, after it has
received a target (in addition to the required v1 fields):

```json
{"tracking":{"session":"0123456789abcdef0123456789abcdef","enabled":true,"marker_size_m":0.060}}
```

`enabled` is true only in Track mode with a displayed pattern. `marker_size_m` is the measured black
square's side in metres (0.020–0.200), or null for orientation only. Reconnecting clears the target;
never carry the old session to a new connection. The laptop ignores an echo for another session.

Status states are `off`, `searching`, `verifying` (rotate the phone gently), `tracked`, and `coasting`
(last accepted alignment plus current phone sensors). Status is still sent at about 2 Hz; camera
updates run up to 20 Hz independently. A missing `tracking` status field means a legacy laptop;
continue manual Align and do not send extension fields. These negotiated optional fields extend v1;
other unknown fields remain rejected. See contracts §12 for timing and validation thresholds.
