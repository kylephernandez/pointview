import {
  fixFromExpo,
  headingFromExpo,
  motionToSample,
  quatRotate,
  rotationRateToOmega,
  rotationToQuat,
  wrapDeg,
  type Vec3,
} from '../sensors';

const D = Math.PI / 180;
const close = (a: readonly number[], b: readonly number[], eps = 1e-9) => {
  expect(a).toHaveLength(b.length);
  a.forEach((x, i) => expect(Math.abs(x - b[i])).toBeLessThan(eps));
};
const norm = (q: readonly number[]) => Math.hypot(...q);

describe('rotationToQuat', () => {
  it('flat face-up with zero Euler angles is the identity', () => {
    close(rotationToQuat(0, 0, 0), [1, 0, 0, 0]);
  });

  it('upright portrait: top edge points up, screen faces -y (towards a user looking north)', () => {
    const q = rotationToQuat(0, 90 * D, 0);
    close(quatRotate(q, [0, 1, 0]), [0, 0, 1]);
    close(quatRotate(q, [0, 0, 1]), [0, -1, 0]);
    close(quatRotate(q, [1, 0, 0]), [1, 0, 0]);
  });

  it('alpha rotates the top edge counter-clockwise about up (W3C): alpha=90 deg turns north to west', () => {
    const q = rotationToQuat(90 * D, 0, 0);
    close(quatRotate(q, [0, 1, 0]), [-1, 0, 0]);
  });

  it('pitched up 30 deg keeps the top edge pitch at +30', () => {
    const q = rotationToQuat(0, 30 * D, 0);
    close(quatRotate(q, [0, 1, 0]), [0, Math.cos(30 * D), Math.sin(30 * D)]);
  });

  it('gamma rolls about the top edge: top edge unchanged, x axis dips', () => {
    const q = rotationToQuat(0, 0, 45 * D);
    close(quatRotate(q, [0, 1, 0]), [0, 1, 0]);
    close(quatRotate(q, [1, 0, 0]), [Math.cos(45 * D), 0, -Math.sin(45 * D)]);
  });

  it('always has unit norm', () => {
    for (const a of [0, 1, 2.5, -3, 6]) {
      for (const b of [0, 0.7, -1.4, 3]) {
        for (const g of [0, 0.3, -1.5, 1.5]) {
          expect(Math.abs(norm(rotationToQuat(a, b, g)) - 1)).toBeLessThan(1e-12);
        }
      }
    }
  });

  it('rotation preserves length and inverts cleanly', () => {
    const q = rotationToQuat(0.4, 0.9, -0.3);
    const v: Vec3 = [0.2, -0.5, 0.8];
    const w = quatRotate(q, v);
    expect(norm(w)).toBeCloseTo(norm(v), 9);
    close(quatRotate([q[0], -q[1], -q[2], -q[3]], w), v);
  });
});

describe('rotationRateToOmega', () => {
  it('converts deg/s to rad/s and maps beta->x, gamma->y, alpha->z', () => {
    close(rotationRateToOmega({ alpha: 180, beta: 90, gamma: 360 }), [Math.PI / 2, 2 * Math.PI, Math.PI]);
  });
  it('null or non-finite becomes zero', () => {
    expect(rotationRateToOmega(null)).toEqual([0, 0, 0]);
    expect(rotationRateToOmega({ alpha: NaN, beta: 0, gamma: 0 })).toEqual([0, 0, 0]);
  });
});

describe('motionToSample', () => {
  it('returns null without rotation', () => {
    expect(motionToSample({}, 5)).toBeNull();
  });
  it('builds a sample with the callback timestamp', () => {
    const s = motionToSample({ rotation: { alpha: 0, beta: 0, gamma: 0 }, rotationRate: null }, 1234);
    expect(s).toEqual({ t_ms: 1234, quat: [1, 0, 0, 0], omega: [0, 0, 0] });
  });
});

describe('heading and fix', () => {
  it('uses trueHeading when available', () => {
    expect(headingFromExpo({ trueHeading: 123.4, magHeading: 100, accuracy: 3 })).toEqual({
      heading_deg: 123.4,
      heading_acc: 10,
    });
  });
  it('falls back to magHeading when trueHeading < 0', () => {
    expect(headingFromExpo({ trueHeading: -1, magHeading: 100, accuracy: 2 })).toEqual({
      heading_deg: 100,
      heading_acc: 20,
    });
  });
  it('is null when neither is usable', () => {
    expect(headingFromExpo({ trueHeading: -1, magHeading: -1, accuracy: 0 })).toEqual({
      heading_deg: null,
      heading_acc: null,
    });
  });
  it('maps calibration levels to degrees', () => {
    const acc = (l: number) => headingFromExpo({ trueHeading: 1, magHeading: 1, accuracy: l }).heading_acc;
    expect([acc(3), acc(2), acc(1), acc(0)]).toEqual([10, 20, 40, 90]);
  });
  it('wraps degrees', () => {
    expect(wrapDeg(-10)).toBe(350);
    expect(wrapDeg(360)).toBe(0);
  });
  it('maps a location fix', () => {
    expect(fixFromExpo({ coords: { latitude: 1, longitude: 2, accuracy: 5 } })).toEqual({
      lat: 1,
      lon: 2,
      gps_acc: 5,
    });
    expect(fixFromExpo({ coords: { latitude: 1, longitude: 2, accuracy: -1 } }).gps_acc).toBeNull();
  });
});
