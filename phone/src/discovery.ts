// Finding the laptop: parallel scan of the hotspot range, manual host[:port] fallback, reconnect backoff.
import { HOTSPOT_SCAN_RANGE, LAPTOP_PORT } from './protocol';
import type { TimerApi } from './buttons';

export const PROBE_TIMEOUT_MS = 700;
export const BACKOFF_START_MS = 500;
export const BACKOFF_MAX_MS = 5000;

export interface Address {
  host: string;
  port: number;
}

export interface WebSocketLike {
  readyState: number;
  bufferedAmount?: number;
  onopen: ((ev?: unknown) => void) | null;
  onclose: ((ev?: unknown) => void) | null;
  onerror: ((ev?: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  send(data: string): void;
  close(): void;
}

export type SocketFactory = (url: string) => WebSocketLike;

export const WS_OPEN = 1;

export function formatAddress(a: Address): string {
  return `${a.host}:${a.port}`;
}

export function addressUrl(a: Address): string {
  return `ws://${a.host}:${a.port}/`;
}

/** Parses "host", "host:port", optionally prefixed by ws:// and followed by "/". Returns null if invalid. */
export function parseAddress(input: string, defaultPort: number = LAPTOP_PORT): Address | null {
  let s = input.trim();
  s = s.replace(/^wss?:\/\//i, '').replace(/\/+$/, '');
  if (!s || /[\s/?#@]/.test(s)) return null;
  const parts = s.split(':');
  if (parts.length > 2) return null;
  const host = parts[0];
  if (!/^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$/.test(host)) return null;
  let port = defaultPort;
  if (parts.length === 2) {
    if (!/^\d{1,5}$/.test(parts[1])) return null;
    port = Number(parts[1]);
    if (port < 1 || port > 65535) return null;
  }
  return { host, port };
}

/** 172.20.10.2 – 172.20.10.14 on the laptop port. */
export function hotspotAddresses(): Address[] {
  const { prefix, first, last } = HOTSPOT_SCAN_RANGE;
  const out: Address[] = [];
  for (let i = first; i <= last; i++) out.push({ host: `${prefix}${i}`, port: LAPTOP_PORT });
  return out;
}

export interface DiscoverOptions {
  createSocket: SocketFactory;
  timers: TimerApi;
  timeoutMs?: number;
}

export interface DiscoverResult {
  socket: WebSocketLike;
  address: Address;
}

/**
 * Opens a WebSocket to every address in parallel. Resolves with the first one that opens and closes all
 * others (including any that open later). Resolves null when all fail or the timeout passes.
 */
export function discover(addresses: Address[], opts: DiscoverOptions): Promise<DiscoverResult | null> {
  const timeoutMs = opts.timeoutMs ?? PROBE_TIMEOUT_MS;
  return new Promise((resolve) => {
    if (addresses.length === 0) {
      resolve(null);
      return;
    }
    let done = false;
    let failed = 0;
    const probes: { address: Address; socket: WebSocketLike; failed: boolean }[] = [];

    const finish = (winner: DiscoverResult | null) => {
      if (done) return;
      done = true;
      opts.timers.clearTimeout(timer);
      for (const p of probes) {
        if (winner && p.socket === winner.socket) continue;
        detach(p.socket);
        safeClose(p.socket);
      }
      resolve(winner);
    };

    const timer = opts.timers.setTimeout(() => finish(null), timeoutMs);

    for (const address of addresses) {
      let socket: WebSocketLike;
      try {
        socket = opts.createSocket(addressUrl(address));
      } catch {
        failed++;
        continue;
      }
      const probe = { address, socket, failed: false };
      probes.push(probe);
      const fail = () => {
        if (probe.failed || done) return;
        probe.failed = true;
        failed++;
        if (failed >= addresses.length) finish(null);
      };
      socket.onopen = () => {
        if (done) {
          safeClose(socket);
          return;
        }
        finish({ socket, address });
      };
      socket.onerror = fail;
      socket.onclose = fail;
    }
    if (failed >= addresses.length) finish(null);
  });
}

function detach(s: WebSocketLike): void {
  s.onopen = null;
  s.onerror = null;
  s.onclose = null;
  s.onmessage = null;
}

function safeClose(s: WebSocketLike): void {
  try {
    s.close();
  } catch {
    // ignore
  }
}

/** Reconnect delay: 0.5 s, doubling, capped at 5 s. */
export class Backoff {
  private current = BACKOFF_START_MS;
  constructor(
    private readonly start = BACKOFF_START_MS,
    private readonly max = BACKOFF_MAX_MS,
  ) {
    this.current = start;
  }
  next(): number {
    const d = this.current;
    this.current = Math.min(this.max, this.current * 2);
    return d;
  }
  reset(): void {
    this.current = this.start;
  }
}
