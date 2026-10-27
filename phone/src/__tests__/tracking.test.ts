import Ajv2020 from 'ajv/dist/2020';
import phoneSchema from '../../../laptop/pointview/schemas/phone_message.schema.json';
import laptopSchema from '../../../laptop/pointview/schemas/laptop_message.schema.json';
import target from '../../../laptop/tests/fixtures/phone_tracking.json';
import { MessageBuilder } from '../message';
import { parseLaptopMessage } from '../link';
import { markerSizeMeters, parseTrackingTarget } from '../tracking';
import { TrackingPattern } from '../TrackingPattern';

const ajv = new Ajv2020({ strict: false });
const validatePhone = ajv.compile(phoneSchema);
const validateLaptop = ajv.compile(laptopSchema);
const sample = (t_ms: number) => ({ t_ms, quat: [1, 0, 0, 0] as [number,number,number,number], omega: [0, 0, 0] as [number,number,number] });

describe('phone screen tracking', () => {
  it('uses the exact laptop target, including its session, dictionary and cell orientation', () => {
    const m = { type: 'status', aligned: false, selected: null, tracking: target };
    expect(validateLaptop(m)).toBe(true);
    const parsed = parseLaptopMessage(JSON.stringify(m));
    expect(parsed?.type === 'status' && parsed.tracking).toEqual(target);
    const valid = parseTrackingTarget(target)!;
    const element = TrackingPattern({ target: valid, side: 250 });
    const rows = element.props.children.map((row: { props: { children: { props: { style: {backgroundColor: string}[] } }[] } }) =>
      row.props.children.map((cell) => cell.props.style[1].backgroundColor === '#fff' ? '1' : '0').join(''));
    expect(rows).toEqual(['1111111111', ...target.cells.map((r) => `1${r}1`), '1111111111']);
    expect(element.props.style).toEqual({ width: 250, height: 250 });
  });

  it('never enables tracking on a legacy peer, and echoes opt-in every sample', () => {
    const b = new MessageBuilder();
    expect(b.build(sample(1000)).tracking).toBeUndefined();
    const tracking = { session: target.session, enabled: true, marker_size_m: null };
    b.setTracking(tracking);
    for (const t of [1016, 1032, 1048]) {
      const msg = b.build(sample(t));
      expect(validatePhone(msg)).toBe(true);
      expect(msg.tracking).toEqual(tracking);
    }
    b.setTracking({ ...tracking, enabled: false, marker_size_m: .06 });
    expect(b.build(sample(1064)).tracking?.enabled).toBe(false);
    b.setTracking(undefined);
    expect(b.build(sample(1080)).tracking).toBeUndefined();
  });

  it.each([
    { session: 'old' }, { dictionary: 'unknown' }, { marker_id: -1 }, { marker_id: 1000 },
    { marker_id: 1.2 }, { state: 'authenticated' }, { cells: [] },
    { cells: ['11111111', ...target.cells.slice(1)] },
    { cells: ['00000000', '01111112', ...target.cells.slice(2)] },
  ])('rejects malformed marker configuration %j', (change) => {
    expect(parseTrackingTarget({ ...target, ...change })).toBeNull();
  });

  it('validates measured physical widths without assuming screen pixel density', () => {
    expect(markerSizeMeters('60')).toBe(.06);
    expect(markerSizeMeters('62.5')).toBe(.0625);
    for (const text of ['', '0', '-60', 'NaN', 'Infinity', '1e2', '1000', '60mm'])
      expect(markerSizeMeters(text)).toBeNull();
  });
});
