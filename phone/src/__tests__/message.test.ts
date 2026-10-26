import * as fs from 'fs';
import * as path from 'path';
import Ajv2020 from 'ajv/dist/2020';
import schema from '../../../laptop/pointview/schemas/phone_message.schema.json';
import { MessageBuilder } from '../message';
import type { MotionSample } from '../sensors';

const ajv = new Ajv2020({ strict: false });
const validate = ajv.compile(schema);

const sample = (t_ms: number, over: Partial<MotionSample> = {}): MotionSample => ({
  t_ms,
  quat: [1, 0, 0, 0],
  omega: [0, 0, 0],
  ...over,
});

describe('MessageBuilder', () => {
  it('builds schema-valid messages, with and without sensors', () => {
    const b = new MessageBuilder();
    expect(validate(b.build(sample(1793050000000)))).toBe(true);
    b.setFix(37.80781234567, -122.47501234567, 8.04);
    b.setHeading(359.99, 12.34);
    const m2 = b.build(sample(1793050000016, { quat: [2, 0, 0, 0], omega: [0.123456789, 500, -500] }));
    expect(validate(m2)).toBe(true);
    expect(m2.quat).toEqual([1, 0, 0, 0]);
    expect(m2.omega).toEqual([0.1235, 100, -100]);
    expect(m2.heading_deg).toBe(0);
    expect(m2.heading_acc).toBe(12.3);
    expect(m2.lat).toBe(37.807812);
    expect(m2.gps_acc).toBe(8);
  });

  it('normalizes the quaternion and keeps validity after rounding', () => {
    const b = new MessageBuilder();
    const m = b.build(sample(1, { quat: [0.3, 0.5, -0.2, 0.9] }));
    expect(Math.abs(Math.hypot(...m.quat) - 1)).toBeLessThan(1e-5);
    expect(validate(m)).toBe(true);
  });

  it('sends each button exactly once, in order', () => {
    const b = new MessageBuilder();
    b.pressButton('align');
    b.pressButton('info');
    expect(b.build(sample(1)).buttons).toEqual(['align', 'info']);
    expect(b.build(sample(2)).buttons).toEqual([]);
    b.pressButton('align_long');
    expect(b.build(sample(3)).buttons).toEqual(['align_long']);
    expect(b.build(sample(4)).buttons).toEqual([]);
  });

  it('never exceeds 16 buttons per message and does not drop the overflow', () => {
    const b = new MessageBuilder();
    for (let i = 0; i < 20; i++) b.pressButton('info');
    const m1 = b.build(sample(1));
    const m2 = b.build(sample(2));
    expect(m1.buttons).toHaveLength(16);
    expect(m2.buttons).toHaveLength(4);
    expect(validate(m1)).toBe(true);
  });

  it('includes layers in the first message, then only on change or every second', () => {
    const b = new MessageBuilder(1000);
    expect(b.build(sample(0)).layers).toEqual({ people: true, vehicles: true, landmarks: true });
    expect(b.build(sample(16)).layers).toBeUndefined();
    expect(b.build(sample(999)).layers).toBeUndefined();
    expect(b.build(sample(1000)).layers).toBeDefined();
    expect(b.build(sample(1016)).layers).toBeUndefined();
    b.setLayer('people', false);
    const m = b.build(sample(1032));
    expect(m.layers).toEqual({ people: false, vehicles: true, landmarks: true });
    expect(validate(m)).toBe(true);
    expect(b.build(sample(1048)).layers).toBeUndefined();
    b.setLayer('people', false); // same value: not a change
    expect(b.build(sample(1064)).layers).toBeUndefined();
    expect(b.build(sample(2032)).layers).toBeDefined(); // 1 s after the last send
  });

  it('markLayersDirty forces layers into the next message', () => {
    const b = new MessageBuilder();
    b.build(sample(0));
    expect(b.build(sample(10)).layers).toBeUndefined();
    b.markLayersDirty();
    expect(b.build(sample(20)).layers).toBeDefined();
  });

  it('the shared fixtures validate against the same schema', () => {
    const file = path.join(__dirname, '../../../laptop/tests/fixtures/phone_messages.jsonl');
    const lines = fs.readFileSync(file, 'utf8').trim().split('\n');
    for (const l of lines) expect(validate(JSON.parse(l))).toBe(true);
  });
});
