// Builds schema-valid PhoneMessage objects from the latest sensor state.
import {
  LAYER_NAMES,
  PROTOCOL_VERSION,
  type Look,
  type MarkerShape,
  type ButtonName,
  type LayerName,
  type Layers,
  type PhoneMessage,
} from './protocol';
import { quatNormalize, wrapDeg, type MotionSample } from './sensors';

export const LAYERS_RESEND_MS = 1000;
export const MAX_BUTTONS_PER_MESSAGE = 16;
const OMEGA_LIMIT = 100;

const round = (x: number, digits: number) => {
  const f = 10 ** digits;
  return Math.round(x * f) / f;
};
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

export type FullLayers = Record<LayerName, boolean>;

export class MessageBuilder {
  private tracking: PhoneMessage['tracking'] = undefined;
  private buttons: ButtonName[] = [];
  private layers: FullLayers = Object.fromEntries(LAYER_NAMES.map((n) => [n, true])) as FullLayers;
  private look: Look = 'normal';
  private marker: MarkerShape = 'square';
  private passthrough = false;
  // Layers, look and passthrough travel together: first message, on change, and every resendMs.
  private layersDirty = true;
  private lastLayersSentAt = -Infinity;
  private lat: number | null = null;
  private lon: number | null = null;
  private gpsAcc: number | null = null;
  private headingDeg: number | null = null;
  private headingAcc: number | null = null;

  constructor(private readonly resendMs = LAYERS_RESEND_MS) {}

  setTracking(tracking: PhoneMessage['tracking']): void {
    this.tracking = tracking;
  }

  pressButton(b: ButtonName): void {
    this.buttons.push(b);
  }

  pendingButtons(): readonly ButtonName[] {
    return this.buttons;
  }

  setLayer(name: LayerName, on: boolean): void {
    if (this.layers[name] !== on) {
      this.layers = { ...this.layers, [name]: on };
      this.layersDirty = true;
    }
  }

  setLook(look: Look): void {
    if (this.look !== look) {
      this.look = look;
      this.layersDirty = true;
    }
  }

  setMarker(marker: MarkerShape): void {
    if (this.marker !== marker) {
      this.marker = marker;
      this.layersDirty = true;
    }
  }

  getMarker(): MarkerShape {
    return this.marker;
  }

  getLook(): Look {
    return this.look;
  }

  setPassthrough(on: boolean): void {
    if (this.passthrough !== on) {
      this.passthrough = on;
      this.layersDirty = true;
    }
  }

  getPassthrough(): boolean {
    return this.passthrough;
  }

  getLayers(): FullLayers {
    return this.layers;
  }

  /** Force the layers into the next message (e.g. after a reconnect). */
  markLayersDirty(): void {
    this.layersDirty = true;
  }

  setFix(lat: number | null, lon: number | null, acc: number | null): void {
    this.lat = lat;
    this.lon = lon;
    this.gpsAcc = acc;
  }

  setHeading(deg: number | null, acc: number | null): void {
    this.headingDeg = deg;
    this.headingAcc = acc;
  }

  /**
   * Builds one message and consumes the queued button presses (each is sent exactly once). Only call this
   * when the message will actually be sent; otherwise presses are lost.
   */
  build(sample: MotionSample): PhoneMessage {
    const q = quatNormalize(sample.quat);
    const buttons = this.buttons.splice(0, MAX_BUTTONS_PER_MESSAGE);
    const msg: PhoneMessage = {
      v: PROTOCOL_VERSION,
      t_ms: Math.max(0, Math.round(sample.t_ms)),
      quat: [round(q[0], 6), round(q[1], 6), round(q[2], 6), round(q[3], 6)],
      omega: [
        round(clamp(sample.omega[0], -OMEGA_LIMIT, OMEGA_LIMIT), 4),
        round(clamp(sample.omega[1], -OMEGA_LIMIT, OMEGA_LIMIT), 4),
        round(clamp(sample.omega[2], -OMEGA_LIMIT, OMEGA_LIMIT), 4),
      ],
      heading_deg: this.headingDeg === null ? null : this.roundHeading(this.headingDeg),
      heading_acc: this.headingAcc === null ? null : round(Math.max(0, this.headingAcc), 1),
      lat: this.lat === null ? null : round(clamp(this.lat, -90, 90), 6),
      lon: this.lon === null ? null : round(clamp(this.lon, -180, 180), 6),
      gps_acc: this.gpsAcc === null ? null : round(Math.max(0, this.gpsAcc), 1),
      buttons,
    };
    if (this.layersDirty || sample.t_ms - this.lastLayersSentAt >= this.resendMs) {
      msg.layers = { ...this.layers } as Layers;
      msg.look = this.look;
      msg.marker = this.marker;
      msg.passthrough = this.passthrough;
      this.layersDirty = false;
      this.lastLayersSentAt = sample.t_ms;
    }
    if (this.tracking) msg.tracking = { ...this.tracking };
    return msg;
  }

  private roundHeading(deg: number): number {
    const r = round(wrapDeg(deg), 1);
    return r >= 360 ? 0 : r;
  }
}
