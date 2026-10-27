import Ajv2020 from 'ajv/dist/2020';
import schema from '../../../laptop/pointview/schemas/phone_message.schema.json';
import { parseLaptopMessage } from '../link';
import { MessageBuilder } from '../message';
import { LAYER_NAMES, LOOKS } from '../protocol';
import type { MotionSample } from '../sensors';
import {
  defaultSettings,
  loadSettings,
  nextLook,
  parseSettings,
  saveSettings,
  statusMismatch,
  STORE_KEY_SETTINGS,
} from '../settings';

const validate = new Ajv2020({ strict: false }).compile(schema);
const sample = (t_ms: number): MotionSample => ({
  t_ms,
  quat: [1, 0, 0, 0],
  omega: [0, 0, 0],
});

describe('MessageBuilder look and passthrough', () => {
  it('sends normal/false in the first message, then omits until change', () => {
    const b = new MessageBuilder();
    const m1 = b.build(sample(1));
    expect(m1.look).toBe('normal');
    expect(m1.passthrough).toBe(false);
    expect(validate(m1)).toBe(true);
    const m2 = b.build(sample(2));
    expect(m2.look).toBeUndefined();
    expect(m2.passthrough).toBeUndefined();
    expect(m2.layers).toBeUndefined();
  });

  it('sends on change, alongside layers', () => {
    const b = new MessageBuilder();
    b.build(sample(1));
    b.setLook('nvg');
    const m = b.build(sample(2));
    expect(m.look).toBe('nvg');
    expect(m.passthrough).toBe(false);
    expect(Object.keys(m.layers ?? {})).toHaveLength(LAYER_NAMES.length);
    expect(validate(m)).toBe(true);
    b.setPassthrough(true);
    const m3 = b.build(sample(3));
    expect(m3.passthrough).toBe(true);
    expect(m3.look).toBe('nvg');
    expect(b.build(sample(4)).look).toBeUndefined();
  });

  it('does not resend when set to the same value', () => {
    const b = new MessageBuilder();
    b.build(sample(1));
    b.setLook('normal');
    b.setPassthrough(false);
    expect(b.build(sample(2)).look).toBeUndefined();
  });

  it('resends every second and after markLayersDirty', () => {
    const b = new MessageBuilder();
    b.setLook('flir');
    b.setPassthrough(true);
    b.build(sample(1000));
    expect(b.build(sample(1999)).look).toBeUndefined();
    const m = b.build(sample(2000));
    expect(m.look).toBe('flir');
    expect(m.passthrough).toBe(true);
    expect(validate(m)).toBe(true);
    b.markLayersDirty();
    expect(b.build(sample(2010)).look).toBe('flir');
  });

  it('every look validates against the schema', () => {
    for (const look of LOOKS) {
      const b = new MessageBuilder();
      b.setLook(look);
      expect(validate(b.build(sample(1)))).toBe(true);
    }
  });
});

describe('cycle order', () => {
  it('walks LOOKS in order and wraps', () => {
    expect(LOOKS).toEqual(['normal', 'crt', 'nvg', 'flir', 'noir', 'snow']);
    let l = LOOKS[0];
    const seen = [l];
    for (let i = 0; i < LOOKS.length; i++) seen.push((l = nextLook(l)));
    expect(seen).toEqual([...LOOKS, 'normal']);
  });
});

describe('persistence', () => {
  const memStore = () => {
    const m = new Map<string, string>();
    return {
      m,
      get: async (k: string) => m.get(k) ?? null,
      set: async (k: string, v: string) => void m.set(k, v),
    };
  };

  it('round-trips look, passthrough and layers', async () => {
    const store = memStore();
    const s = defaultSettings();
    s.look = 'crt';
    s.passthrough = true;
    s.layers.aircraft = false;
    s.layers.people = false;
    await saveSettings(store, s);
    expect(store.m.has(STORE_KEY_SETTINGS)).toBe(true);
    expect(await loadSettings(store)).toEqual(s);
  });

  it('falls back to defaults on empty, corrupt or invalid data', async () => {
    expect(await loadSettings(memStore())).toEqual(defaultSettings());
    expect(parseSettings('{nope')).toEqual(defaultSettings());
    const bad = parseSettings(
      JSON.stringify({
        look: 'sepia',
        passthrough: 'yes',
        layers: { people: 0, events: false },
      }),
    );
    expect(bad.look).toBe('normal');
    expect(bad.passthrough).toBe(false);
    expect(bad.layers.people).toBe(true);
    expect(bad.layers.events).toBe(false);
  });

  it('survives a throwing store', async () => {
    const broken = {
      get: async () => {
        throw new Error('x');
      },
      set: async () => {
        throw new Error('x');
      },
    };
    expect(await loadSettings(broken)).toEqual(defaultSettings());
    await expect(saveSettings(broken, defaultSettings())).resolves.toBeUndefined();
  });

  it('applies to a builder', () => {
    const s = parseSettings(
      JSON.stringify({
        look: 'snow',
        passthrough: true,
        layers: { vessels: false },
      }),
    );
    const b = new MessageBuilder();
    b.setLook(s.look);
    b.setPassthrough(s.passthrough);
    b.setLayer('vessels', s.layers.vessels);
    const m = b.build(sample(1));
    expect(m.look).toBe('snow');
    expect(m.layers?.vessels).toBe(false);
    expect(validate(m)).toBe(true);
  });
});

describe('status with look/passthrough', () => {
  it('parses the echoed fields and ignores invalid ones', () => {
    const m = parseLaptopMessage(
      JSON.stringify({
        type: 'status',
        aligned: true,
        selected: null,
        look: 'flir',
        passthrough: true,
      }),
    );
    expect(m).toEqual({
      type: 'status',
      aligned: true,
      selected: null,
      look: 'flir',
      passthrough: true,
    });
    const bad = parseLaptopMessage(
      JSON.stringify({
        type: 'status',
        aligned: true,
        selected: null,
        look: 'sepia',
        passthrough: 'x',
      }),
    );
    expect(bad).toEqual({ type: 'status', aligned: true, selected: null });
  });

  it('flags a mismatch only when the laptop reports a different value', () => {
    const st = (extra: object) => ({
      type: 'status' as const,
      aligned: true,
      selected: null,
      ...extra,
    });
    expect(statusMismatch(null, 'crt', true)).toBeNull();
    expect(statusMismatch(st({}), 'crt', true)).toBeNull();
    expect(statusMismatch(st({ look: 'crt', passthrough: true }), 'crt', true)).toBeNull();
    expect(statusMismatch(st({ look: 'nvg', passthrough: true }), 'crt', true)).toContain('NVG');
    expect(statusMismatch(st({ look: 'crt', passthrough: false }), 'crt', true)).toContain('passthrough off');
  });
});
