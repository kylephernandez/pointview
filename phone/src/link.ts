// WebSocket link to the laptop: discovery, reconnect with backoff, send, receive.
import type { TimerApi } from './buttons';
import {
  Backoff,
  discover,
  formatAddress,
  hotspotAddresses,
  parseAddress,
  PROBE_TIMEOUT_MS,
  WS_OPEN,
  type Address,
  type SocketFactory,
  type WebSocketLike,
} from './discovery';
import { LOOKS, type HapticStyle, type LaptopMessage, type Look } from './protocol';

export interface KeyValueStore {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<void>;
}

export const STORE_KEY_LAST_GOOD = 'pointview.lastGood';
export const STORE_KEY_MANUAL = 'pointview.manual';

export type LinkStatus = 'idle' | 'searching' | 'connected' | 'waiting';

export interface LinkState {
  status: LinkStatus;
  /** Connected address, or the address being tried last. */
  address: string | null;
  /** True after a full scan found nothing: the UI should offer the manual address field. */
  needsManual: boolean;
  /** Number of connection rounds since the last success. */
  attempts: number;
}

export interface LinkOptions {
  createSocket: SocketFactory;
  timers: TimerApi;
  store: KeyValueStore;
  onState: (s: LinkState) => void;
  onMessage: (m: LaptopMessage) => void;
  /** Called every time a connection is established. */
  onConnected?: () => void;
  probeTimeoutMs?: number;
  /** Skip sending when the socket buffer holds more than this many bytes. */
  maxBuffered?: number;
}

const HAPTIC_STYLES: readonly string[] = ['snap', 'select', 'align_ok', 'error'];

/** Parses and validates a laptop → phone frame. Returns null for anything that is not a valid message. */
export function parseLaptopMessage(data: unknown): LaptopMessage | null {
  if (typeof data !== 'string') return null;
  let o: unknown;
  try {
    o = JSON.parse(data);
  } catch {
    return null;
  }
  if (typeof o !== 'object' || o === null) return null;
  const m = o as Record<string, unknown>;
  if (m.type === 'haptic' && typeof m.style === 'string' && HAPTIC_STYLES.includes(m.style)) {
    return { type: 'haptic', style: m.style as HapticStyle };
  }
  if (m.type === 'status' && typeof m.aligned === 'boolean') {
    const sel = m.selected;
    if (sel !== null && sel !== undefined && typeof sel !== 'string') return null;
    const out: LaptopMessage = {
      type: 'status',
      aligned: m.aligned,
      selected: sel ?? null,
    };
    if (typeof m.fps === 'number') out.fps = m.fps;
    if (typeof m.message === 'string') out.message = m.message;
    if (typeof m.look === 'string' && (LOOKS as readonly string[]).includes(m.look))
      out.look = m.look as Look;
    if (typeof m.passthrough === 'boolean') out.passthrough = m.passthrough;
    return out;
  }
  return null;
}

export class PhoneLink {
  private state: LinkState = {
    status: 'idle',
    address: null,
    needsManual: false,
    attempts: 0,
  };
  private socket: WebSocketLike | null = null;
  private retryHandle: unknown = null;
  private generation = 0;
  private running = false;
  private readonly backoff = new Backoff();
  private lastGood: Address | null = null;
  private manual: Address | null = null;

  constructor(private readonly opts: LinkOptions) {}

  getState(): LinkState {
    return this.state;
  }

  isOpen(): boolean {
    return this.socket !== null && this.socket.readyState === WS_OPEN;
  }

  /** True when the socket is open and not backed up. */
  canSend(): boolean {
    if (!this.isOpen() || !this.socket) return false;
    const max = this.opts.maxBuffered ?? 64 * 1024;
    return (this.socket.bufferedAmount ?? 0) <= max;
  }

  /** Sends a text frame. Returns false if it was not sent. */
  send(json: string): boolean {
    if (!this.canSend() || !this.socket) return false;
    try {
      this.socket.send(json);
      return true;
    } catch {
      return false;
    }
  }

  async start(): Promise<void> {
    if (this.running) return;
    this.running = true;
    const [lg, man] = await Promise.all([this.read(STORE_KEY_LAST_GOOD), this.read(STORE_KEY_MANUAL)]);
    this.lastGood = lg ? parseAddress(lg) : null;
    this.manual = man ? parseAddress(man) : null;
    if (!this.running) return;
    await this.attempt(this.generation);
  }

  stop(): void {
    this.running = false;
    this.generation++;
    this.clearRetry();
    this.dropSocket();
    this.setState({ status: 'idle' });
  }

  /** Saves a manual address (host[:port]) and reconnects right away. Returns false if it does not parse. */
  async setManualAddress(text: string): Promise<boolean> {
    const a = parseAddress(text);
    if (!a) return false;
    this.manual = a;
    await this.write(STORE_KEY_MANUAL, formatAddress(a));
    if (this.running) this.reconnectNow();
    return true;
  }

  getManualAddress(): string | null {
    return this.manual ? formatAddress(this.manual) : this.lastGood ? formatAddress(this.lastGood) : null;
  }

  reconnectNow(): void {
    this.generation++;
    this.clearRetry();
    this.dropSocket();
    this.backoff.reset();
    void this.attempt(this.generation);
  }

  // ------------------------------------------------------------------ internals

  private async attempt(gen: number): Promise<void> {
    if (!this.running || gen !== this.generation) return;
    this.setState({ status: 'searching', attempts: this.state.attempts + 1 });

    const tried = new Set<string>();
    const priority: Address[] = [];
    for (const a of [this.lastGood, this.manual]) {
      if (a && !tried.has(formatAddress(a))) {
        tried.add(formatAddress(a));
        priority.push(a);
      }
    }
    const dOpts = {
      createSocket: this.opts.createSocket,
      timers: this.opts.timers,
      timeoutMs: this.opts.probeTimeoutMs ?? PROBE_TIMEOUT_MS,
    };

    let found = priority.length ? await discover(priority, dOpts) : null;
    if (gen !== this.generation) {
      this.discard(found);
      return;
    }
    let scanned = false;
    if (!found) {
      const rest = hotspotAddresses().filter((a) => !tried.has(formatAddress(a)));
      found = await discover(rest, dOpts);
      scanned = true;
      if (gen !== this.generation) {
        this.discard(found);
        return;
      }
    }

    if (found) {
      this.adopt(found.socket, found.address, gen);
    } else {
      this.setState({
        status: 'waiting',
        needsManual: this.state.needsManual || scanned,
      });
      this.scheduleRetry(gen);
    }
  }

  private discard(found: { socket: WebSocketLike } | null): void {
    if (found) {
      try {
        found.socket.close();
      } catch {
        // ignore
      }
    }
  }

  private adopt(socket: WebSocketLike, address: Address, gen: number): void {
    this.socket = socket;
    this.lastGood = address;
    void this.write(STORE_KEY_LAST_GOOD, formatAddress(address));
    this.backoff.reset();
    socket.onmessage = (ev) => {
      const m = parseLaptopMessage(ev.data);
      if (m) this.opts.onMessage(m);
    };
    const lost = () => {
      if (gen !== this.generation || this.socket !== socket) return;
      this.socket = null;
      this.setState({ status: 'waiting' });
      this.scheduleRetry(gen);
    };
    socket.onclose = lost;
    socket.onerror = lost;
    this.setState({
      status: 'connected',
      address: formatAddress(address),
      needsManual: false,
      attempts: 0,
    });
    this.opts.onConnected?.();
  }

  private scheduleRetry(gen: number): void {
    this.clearRetry();
    const delay = this.backoff.next();
    this.retryHandle = this.opts.timers.setTimeout(() => {
      this.retryHandle = null;
      void this.attempt(gen);
    }, delay);
  }

  private clearRetry(): void {
    if (this.retryHandle !== null) {
      this.opts.timers.clearTimeout(this.retryHandle);
      this.retryHandle = null;
    }
  }

  private dropSocket(): void {
    const s = this.socket;
    this.socket = null;
    if (s) {
      s.onopen = s.onclose = s.onerror = s.onmessage = null;
      try {
        s.close();
      } catch {
        // ignore
      }
    }
  }

  private setState(patch: Partial<LinkState>): void {
    this.state = { ...this.state, ...patch };
    this.opts.onState(this.state);
  }

  private async read(key: string): Promise<string | null> {
    try {
      return await this.opts.store.get(key);
    } catch {
      return null;
    }
  }

  private async write(key: string, value: string): Promise<void> {
    try {
      await this.opts.store.set(key, value);
    } catch {
      // persistence is best effort
    }
  }
}
