import type { TrackingTarget } from './protocol';

const STATES = ['off', 'searching', 'verifying', 'tracked', 'coasting'];

/** Reject partial/unknown targets; never render a guessed or stale tracking pattern. */
export function parseTrackingTarget(value: unknown): TrackingTarget | null {
  if (!value || typeof value !== 'object') return null;
  const v = value as Record<string, unknown>;
  if (
    typeof v.session !== 'string' || !/^[a-f0-9]{32}$/.test(v.session) ||
    v.dictionary !== 'DICT_6X6_1000' || !Number.isInteger(v.marker_id) ||
    (v.marker_id as number) < 0 || (v.marker_id as number) > 999 ||
    typeof v.state !== 'string' || !STATES.includes(v.state) ||
    !Array.isArray(v.cells) || v.cells.length !== 8 ||
    !v.cells.every((row: unknown) => typeof row === 'string' && /^[01]{8}$/.test(row))
  ) return null;
  const cells = v.cells as string[];
  if (cells[0] !== '00000000' || cells[7] !== '00000000' ||
      cells.some((row) => row[0] !== '0' || row[7] !== '0')) return null;
  return {
    session: v.session, dictionary: v.dictionary, marker_id: v.marker_id as number,
    cells: [...cells], state: v.state as TrackingTarget['state'],
  };
}

export function markerSizeMeters(text: string): number | null {
  if (!/^\d+(\.\d+)?$/.test(text.trim())) return null;
  const mm = Number(text);
  return Number.isFinite(mm) && mm >= 20 && mm <= 200 ? mm / 1000 : null;
}

export const TRACKING_HINT: Record<TrackingTarget['state'], string> = {
  off: 'Show the screen to the glasses camera.',
  searching: 'Show the screen to the glasses camera.',
  verifying: 'Gently rotate the phone to match its motion.',
  tracked: 'Tracking · pointing updates automatically',
  coasting: 'Out of view or motion mismatch · using phone sensors',
};
