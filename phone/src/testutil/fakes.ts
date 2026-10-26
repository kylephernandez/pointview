// Test doubles: manual timers, fake WebSocket, in-memory store.
import type { TimerApi } from '../buttons';
import type { WebSocketLike } from '../discovery';
import type { KeyValueStore } from '../link';

export class FakeTimers implements TimerApi {
  now = 0;
  private nextId = 1;
  private queue: { id: number; at: number; fn: () => void }[] = [];

  setTimeout = (fn: () => void, ms: number): unknown => {
    const id = this.nextId++;
    this.queue.push({ id, at: this.now + ms, fn });
    return id;
  };

  clearTimeout = (h: unknown): void => {
    this.queue = this.queue.filter((t) => t.id !== h);
  };

  pending(): number {
    return this.queue.length;
  }

  /** Advances time, running due callbacks in order (callbacks may schedule more). */
  advance(ms: number): void {
    const target = this.now + ms;
    for (;;) {
      this.queue.sort((a, b) => a.at - b.at || a.id - b.id);
      const next = this.queue[0];
      if (!next || next.at > target) break;
      this.queue.shift();
      this.now = next.at;
      next.fn();
    }
    this.now = target;
  }
}

export class FakeSocket implements WebSocketLike {
  readyState = 0;
  bufferedAmount = 0;
  onopen: WebSocketLike['onopen'] = null;
  onclose: WebSocketLike['onclose'] = null;
  onerror: WebSocketLike['onerror'] = null;
  onmessage: WebSocketLike['onmessage'] = null;
  sent: string[] = [];
  closed = false;

  constructor(readonly url: string) {}

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.closed = true;
    this.readyState = 3;
  }

  /** Server accepted the connection. */
  open(): void {
    if (this.closed) return;
    this.readyState = 1;
    this.onopen?.();
  }

  /** Connection refused / unreachable. */
  fail(): void {
    this.readyState = 3;
    this.onerror?.();
    this.onclose?.();
  }

  /** Server went away after being open. */
  drop(): void {
    this.readyState = 3;
    this.onclose?.();
  }

  receive(data: unknown): void {
    this.onmessage?.({ data });
  }
}

export class SocketFactoryFake {
  sockets: FakeSocket[] = [];
  create = (url: string): FakeSocket => {
    const s = new FakeSocket(url);
    this.sockets.push(s);
    return s;
  };
  forUrl(host: string): FakeSocket[] {
    return this.sockets.filter((s) => s.url.includes(`//${host}:`));
  }
  clear(): void {
    this.sockets = [];
  }
}

export class MemoryStore implements KeyValueStore {
  data = new Map<string, string>();
  async get(key: string): Promise<string | null> {
    return this.data.get(key) ?? null;
  }
  async set(key: string, value: string): Promise<void> {
    this.data.set(key, value);
  }
}

export async function flush(times = 10): Promise<void> {
  for (let i = 0; i < times; i++) await Promise.resolve();
}
