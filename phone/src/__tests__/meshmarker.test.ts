import Ajv2020 from 'ajv/dist/2020';
import schema from '../../../laptop/pointview/schemas/phone_message.schema.json';
import { parseLaptopMessage } from '../link';
import { MessageBuilder } from '../message';
import { MARKER_SHAPES } from '../protocol';
import type { MotionSample } from '../sensors';
import { defaultSettings, loadSettings, parseSettings, saveSettings, statusMismatch } from '../settings';

const validate = new Ajv2020({ strict: false }).compile(schema);
const sample = (t_ms: number): MotionSample => ({ t_ms, quat: [1, 0, 0, 0], omega: [0, 0, 0] });

describe('mesh and marker', () => {
  it('sends marker square and mesh on in the first message, validated by the schema', () => {
    const b = new MessageBuilder();
    const m = b.build(sample(1));
    expect(m.marker).toBe('square');
    expect(m.layers?.mesh).toBe(true);
    expect(validate(m)).toBe(true);
    expect(b.build(sample(2)).marker).toBeUndefined();
  });

  it('sends marker on change and periodically; mesh goes in layers', () => {
    const b = new MessageBuilder();
    b.build(sample(1000));
    b.setMarker('point');
    b.setLayer('mesh', false);
    const m = b.build(sample(1001));
    expect(m.marker).toBe('point');
    expect(m.layers?.mesh).toBe(false);
    expect(validate(m)).toBe(true);
    expect(b.build(sample(1500)).marker).toBeUndefined();
    expect(b.build(sample(2001)).marker).toBe('point');
    for (const mk of MARKER_SHAPES) {
      const c = new MessageBuilder();
      c.setMarker(mk);
      expect(validate(c.build(sample(1)))).toBe(true);
    }
  });

  it('persists marker and mesh', async () => {
    const m = new Map<string, string>();
    const store = {
      get: async (k: string) => m.get(k) ?? null,
      set: async (k: string, v: string) => void m.set(k, v),
    };
    const s = defaultSettings();
    s.marker = 'point';
    s.layers.mesh = false;
    await saveSettings(store, s);
    expect(await loadSettings(store)).toEqual(s);
    expect(parseSettings(JSON.stringify({ marker: 'blob' })).marker).toBe('square');
    expect(parseSettings(JSON.stringify({ look: 'crt' })).layers.mesh).toBe(true);
  });

  it('parses and flags the echoed marker', () => {
    const st = parseLaptopMessage(
      JSON.stringify({ type: 'status', aligned: true, selected: null, marker: 'point' }),
    );
    expect(st).toMatchObject({ marker: 'point' });
    const bad = parseLaptopMessage(JSON.stringify({ type: 'status', aligned: true, marker: 'x' }));
    expect(bad).not.toHaveProperty('marker');
    const s = { type: 'status' as const, aligned: true, selected: null, marker: 'point' as const };
    expect(statusMismatch(s, 'normal', false, 'point')).toBeNull();
    expect(statusMismatch(s, 'normal', false, 'square')).toContain('marker POINT');
  });
});
