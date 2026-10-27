// User choices that survive app launches: look, passthrough, layer toggles.
import type { KeyValueStore } from './link';
import { LAYER_NAMES, LOOKS, type LaptopMessage, type LayerName, type Look } from './protocol';

export const STORE_KEY_SETTINGS = 'pointview.settings';

export interface Settings {
  look: Look;
  passthrough: boolean;
  layers: Record<LayerName, boolean>;
}

export function defaultSettings(): Settings {
  return {
    look: 'normal',
    passthrough: false,
    layers: Object.fromEntries(LAYER_NAMES.map((n) => [n, true])) as Record<LayerName, boolean>,
  };
}

/** Next look in LOOKS order, wrapping after the last. */
export function nextLook(look: Look): Look {
  return LOOKS[(LOOKS.indexOf(look) + 1) % LOOKS.length];
}

export function parseSettings(raw: string | null): Settings {
  const out = defaultSettings();
  if (!raw) return out;
  try {
    const o = JSON.parse(raw) as Record<string, unknown>;
    if (typeof o.look === 'string' && (LOOKS as readonly string[]).includes(o.look))
      out.look = o.look as Look;
    if (typeof o.passthrough === 'boolean') out.passthrough = o.passthrough;
    if (typeof o.layers === 'object' && o.layers !== null) {
      const l = o.layers as Record<string, unknown>;
      for (const n of LAYER_NAMES) if (typeof l[n] === 'boolean') out.layers[n] = l[n];
    }
  } catch {
    // corrupt storage: fall back to defaults
  }
  return out;
}

export async function loadSettings(store: KeyValueStore): Promise<Settings> {
  try {
    return parseSettings(await store.get(STORE_KEY_SETTINGS));
  } catch {
    return defaultSettings();
  }
}

export async function saveSettings(store: KeyValueStore, s: Settings): Promise<void> {
  try {
    await store.set(STORE_KEY_SETTINGS, JSON.stringify(s));
  } catch {
    // best effort
  }
}

/** How the laptop's echoed look/passthrough differ from what the phone asked for; null if they agree. */
export function statusMismatch(
  status: Extract<LaptopMessage, { type: 'status' }> | null,
  look: Look,
  passthrough: boolean,
): string | null {
  if (!status) return null;
  const parts: string[] = [];
  if (status.look !== undefined && status.look !== look) parts.push(`look ${status.look.toUpperCase()}`);
  if (status.passthrough !== undefined && status.passthrough !== passthrough) {
    parts.push(`passthrough ${status.passthrough ? 'on' : 'off'}`);
  }
  return parts.length ? `Laptop has ${parts.join(', ')}` : null;
}
