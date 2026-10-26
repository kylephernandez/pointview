// Sensor access. Pure conversion functions first (unit-tested), subscription code at the bottom.
import { DeviceMotion } from 'expo-sensors';
import * as Location from 'expo-location';

export type Quat = [number, number, number, number];
export type Vec3 = [number, number, number];

const DEG2RAD = Math.PI / 180;

/** Hamilton product a ⊗ b, quaternions as [w, x, y, z]. */
export function quatMul(a: Quat, b: Quat): Quat {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

export function quatNormalize(q: Quat): Quat {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!Number.isFinite(n) || n < 1e-9) return [1, 0, 0, 0];
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/** Rotates vector v by unit quaternion q. */
export function quatRotate(q: Quat, v: Vec3): Vec3 {
  const p = quatMul(quatMul(q, [0, v[0], v[1], v[2]]), [q[0], -q[1], -q[2], -q[3]]);
  return [p[1], p[2], p[3]];
}

/**
 * DeviceMotion `rotation` (radians; W3C intrinsic Z-X'-Y'' Euler angles) to the protocol quaternion
 * q = qz(alpha) ⊗ qx(beta) ⊗ qy(gamma): rotates phone-frame vectors into the gravity-aligned frame.
 */
export function rotationToQuat(alpha: number, beta: number, gamma: number): Quat {
  const qz: Quat = [Math.cos(alpha / 2), 0, 0, Math.sin(alpha / 2)];
  const qx: Quat = [Math.cos(beta / 2), Math.sin(beta / 2), 0, 0];
  const qy: Quat = [Math.cos(gamma / 2), 0, Math.sin(gamma / 2), 0];
  return quatNormalize(quatMul(quatMul(qz, qx), qy));
}

/**
 * DeviceMotion `rotationRate` is documented by Expo as degrees per second on every platform
 * (alpha = about z, beta = about x, gamma = about y). Protocol omega is rad/s in phone axes [x, y, z].
 */
export function rotationRateToOmega(
  rate: { alpha: number; beta: number; gamma: number } | null | undefined,
): Vec3 {
  if (!rate) return [0, 0, 0];
  const f = (d: number) => (Number.isFinite(d) ? d * DEG2RAD : 0);
  return [f(rate.beta), f(rate.gamma), f(rate.alpha)];
}

export interface MotionSample {
  t_ms: number;
  quat: Quat;
  omega: Vec3;
}

export function motionToSample(
  m: {
    rotation?: { alpha: number; beta: number; gamma: number } | null;
    rotationRate?: { alpha: number; beta: number; gamma: number } | null;
  },
  t_ms: number,
): MotionSample | null {
  if (!m.rotation) return null;
  return {
    t_ms,
    quat: rotationToQuat(m.rotation.alpha, m.rotation.beta, m.rotation.gamma),
    omega: rotationRateToOmega(m.rotationRate),
  };
}

/** Wraps degrees into [0, 360). */
export function wrapDeg(d: number): number {
  const r = d % 360;
  return r < 0 ? r + 360 : r;
}

export interface HeadingReading {
  heading_deg: number | null;
  heading_acc: number | null;
}

/**
 * Expo's heading `accuracy` is a calibration level, not degrees: 3 high, 2 medium, 1 low, 0 uncalibrated.
 * Mapped to a rough error in degrees (assumption, see report).
 */
export function headingLevelToDeg(level: number | null | undefined): number | null {
  if (level === null || level === undefined || !Number.isFinite(level)) return null;
  if (level >= 3) return 10;
  if (level >= 2) return 20;
  if (level >= 1) return 40;
  return 90;
}

/** trueHeading, falling back to magHeading when trueHeading is negative (-1: unavailable). */
export function headingFromExpo(h: {
  trueHeading: number;
  magHeading: number;
  accuracy: number;
}): HeadingReading {
  const pick = h.trueHeading >= 0 ? h.trueHeading : h.magHeading;
  if (!Number.isFinite(pick) || pick < 0) return { heading_deg: null, heading_acc: null };
  return { heading_deg: wrapDeg(pick), heading_acc: headingLevelToDeg(h.accuracy) };
}

export interface FixReading {
  lat: number;
  lon: number;
  gps_acc: number | null;
}

export function fixFromExpo(loc: {
  coords: { latitude: number; longitude: number; accuracy: number | null };
}): FixReading {
  const acc = loc.coords.accuracy;
  return {
    lat: loc.coords.latitude,
    lon: loc.coords.longitude,
    gps_acc: acc !== null && Number.isFinite(acc) && acc >= 0 ? acc : null,
  };
}

// ---------------------------------------------------------------- subscriptions (device only)

export const MOTION_INTERVAL_MS = 16;

/** Starts DeviceMotion at ~60 Hz. Returns a stop function. */
export function startMotion(onSample: (s: MotionSample) => void): () => void {
  DeviceMotion.setUpdateInterval(MOTION_INTERVAL_MS);
  const sub = DeviceMotion.addListener((m) => {
    const s = motionToSample(m, Date.now());
    if (s) onSample(s);
  });
  return () => sub.remove();
}

export interface LocationCallbacks {
  onFix: (f: FixReading) => void;
  onHeading: (h: HeadingReading) => void;
  onPermission: (granted: boolean) => void;
}

/** Requests foreground permission, then watches position and heading. Returns a stop function. */
export function startLocation(cb: LocationCallbacks): () => void {
  let stopped = false;
  const subs: { remove: () => void }[] = [];
  (async () => {
    try {
      const perm = await Location.requestForegroundPermissionsAsync();
      cb.onPermission(perm.status === 'granted');
      if (perm.status !== 'granted' || stopped) return;
      const pos = await Location.watchPositionAsync(
        { accuracy: Location.Accuracy.BestForNavigation, distanceInterval: 0, timeInterval: 1000 },
        (loc) => cb.onFix(fixFromExpo(loc)),
      );
      const head = await Location.watchHeadingAsync((h) => cb.onHeading(headingFromExpo(h)));
      if (stopped) {
        pos.remove();
        head.remove();
      } else {
        subs.push(pos, head);
      }
    } catch {
      cb.onPermission(false);
    }
  })();
  return () => {
    stopped = true;
    subs.forEach((s) => s.remove());
  };
}
