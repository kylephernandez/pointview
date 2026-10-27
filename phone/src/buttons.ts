// Align button: tap = align, hold >= 600 ms = align_long (fires as soon as the threshold passes).
import type { ButtonName } from './protocol';

export const LONG_PRESS_MS = 600;

export interface TimerApi {
  setTimeout: (fn: () => void, ms: number) => unknown;
  clearTimeout: (h: unknown) => void;
}

export const realTimers: TimerApi = {
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

export class AlignPressDetector {
  private handle: unknown = null;
  private fired = false;
  private down = false;

  constructor(
    private readonly emit: (b: ButtonName) => void,
    private readonly timers: TimerApi = realTimers,
    private readonly longMs = LONG_PRESS_MS,
  ) {}

  pressIn(): void {
    if (this.down) return;
    this.down = true;
    this.fired = false;
    this.handle = this.timers.setTimeout(() => {
      this.handle = null;
      this.fired = true;
      this.emit('align_long');
    }, this.longMs);
  }

  pressOut(): void {
    if (!this.down) return;
    this.down = false;
    if (this.handle !== null) {
      this.timers.clearTimeout(this.handle);
      this.handle = null;
    }
    if (!this.fired) this.emit('align');
  }

  /** Touch cancelled (e.g. system gesture): emit nothing. */
  cancel(): void {
    this.down = false;
    if (this.handle !== null) this.timers.clearTimeout(this.handle);
    this.handle = null;
  }
}
