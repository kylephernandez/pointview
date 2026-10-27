// Phone ↔ laptop protocol types. FROZEN — mirrors laptop/pointview/schemas/*.schema.json.
// See docs/phone-protocol.md and docs/contracts.md.

export const PROTOCOL_VERSION = 1 as const;
export const LAPTOP_PORT = 8765;
/** iPhone Personal Hotspot range the laptop gets an address in. */
export const HOTSPOT_SCAN_RANGE = { prefix: '172.20.10.', first: 2, last: 14 } as const;
export const SEND_HZ = 60;

export type ButtonName = 'info' | 'align' | 'align_long';
export type LayerName =
  | 'people'
  | 'vehicles'
  | 'landmarks'
  | 'aircraft'
  | 'satellites'
  | 'vessels'
  | 'events'
  | 'mesh';
export const LAYER_NAMES: readonly LayerName[] = [
  'people',
  'vehicles',
  'landmarks',
  'aircraft',
  'satellites',
  'vessels',
  'events',
  'mesh',
];
/** God's Eye View sensor looks, applied in the glasses (the phone only chooses). */
export type Look = 'normal' | 'crt' | 'nvg' | 'flir' | 'noir' | 'snow';
export const LOOKS: readonly Look[] = ['normal', 'crt', 'nvg', 'flir', 'noir', 'snow'];
/** How tracked objects are drawn in the glasses. */
export type MarkerShape = 'point' | 'square';
export const MARKER_SHAPES: readonly MarkerShape[] = ['square', 'point'];
export type Layers = Partial<Record<LayerName, boolean>>;

/** One phone → laptop sample. Field meanings and units: docs/phone-protocol.md. */
export interface PhoneMessage {
  v: typeof PROTOCOL_VERSION;
  /** Date.now() when the motion sample was taken. */
  t_ms: number;
  /** [w, x, y, z]: phone frame (x right, y top edge, z out of screen) → gravity-aligned frame. */
  quat: [number, number, number, number];
  /** Rotation rate in rad/s, phone frame [x, y, z]. */
  omega: [number, number, number];
  /** Degrees clockwise from true north, [0, 360). */
  heading_deg: number | null;
  heading_acc: number | null;
  lat: number | null;
  lon: number | null;
  /** Meters. */
  gps_acc: number | null;
  /** Presses since the previous message. */
  buttons: ButtonName[];
  layers?: Layers;
  /** Sent on change (and with layers); omitted = unchanged. */
  look?: Look;
  /** Show the VITURE camera image behind the overlay; omitted = unchanged. */
  passthrough?: boolean;
  /** Tracked-object style; omitted = unchanged. */
  marker?: MarkerShape;
  /** Track mode: the ArUco DICT_4X4_50 marker on screen (side in mm), null when stopped; omitted = unchanged. */
  fiducial?: Fiducial | null;
}

/** The tracking marker the phone shows so the glasses' camera can follow it (contracts §12). */
export interface Fiducial {
  id: number;
  side_mm: number;
}
/** ArUco DICT_4X4_50 ids. */
export const FIDUCIAL_ID_MAX = 49;

export type HapticStyle = 'snap' | 'select' | 'align_ok' | 'error';

export type LaptopMessage =
  | { type: 'haptic'; style: HapticStyle }
  | {
      type: 'status';
      aligned: boolean;
      selected: string | null;
      fps?: number;
      message?: string;
      look?: Look;
      passthrough?: boolean;
      marker?: MarkerShape;
    }
  | { type: 'fiducial'; id: number; tracked?: boolean };
