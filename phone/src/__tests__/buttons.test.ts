import { AlignPressDetector, LONG_PRESS_MS } from '../buttons';
import type { ButtonName } from '../protocol';

describe('AlignPressDetector', () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  const make = () => {
    const out: ButtonName[] = [];
    return { out, d: new AlignPressDetector((b) => out.push(b)) };
  };

  it('tap emits align on release', () => {
    const { out, d } = make();
    d.pressIn();
    jest.advanceTimersByTime(LONG_PRESS_MS - 1);
    d.pressOut();
    jest.advanceTimersByTime(1000);
    expect(out).toEqual(['align']);
  });

  it('hold emits align_long at 600 ms and nothing more on release', () => {
    const { out, d } = make();
    d.pressIn();
    jest.advanceTimersByTime(LONG_PRESS_MS);
    expect(out).toEqual(['align_long']);
    d.pressOut();
    expect(out).toEqual(['align_long']);
  });

  it('cancel emits nothing, and a new press works afterwards', () => {
    const { out, d } = make();
    d.pressIn();
    d.cancel();
    jest.advanceTimersByTime(2000);
    expect(out).toEqual([]);
    d.pressIn();
    d.pressOut();
    expect(out).toEqual(['align']);
  });
});
