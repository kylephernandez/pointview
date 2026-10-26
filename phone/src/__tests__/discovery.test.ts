import {
  Backoff,
  discover,
  formatAddress,
  hotspotAddresses,
  parseAddress,
} from '../discovery';
import { FakeTimers, SocketFactoryFake, flush } from '../testutil/fakes';

describe('parseAddress', () => {
  it('parses host and host:port with default port 8765', () => {
    expect(parseAddress('172.20.10.3')).toEqual({ host: '172.20.10.3', port: 8765 });
    expect(parseAddress(' 172.20.10.3:9000 ')).toEqual({ host: '172.20.10.3', port: 9000 });
    expect(parseAddress('my-laptop.local:1')).toEqual({ host: 'my-laptop.local', port: 1 });
    expect(parseAddress('ws://10.0.0.5:8765/')).toEqual({ host: '10.0.0.5', port: 8765 });
  });
  it('rejects bad input', () => {
    for (const bad of ['', '  ', 'a b', ':8765', 'host:', 'host:0', 'host:70000', 'host:12x', 'a:1:2', 'http://x/y', '-x', 'x/y']) {
      expect(parseAddress(bad)).toBeNull();
    }
  });
  it('round-trips through formatAddress', () => {
    expect(formatAddress({ host: 'h', port: 5 })).toBe('h:5');
  });
});

describe('hotspotAddresses', () => {
  it('covers 172.20.10.2 to .14 on 8765', () => {
    const a = hotspotAddresses();
    expect(a).toHaveLength(13);
    expect(formatAddress(a[0])).toBe('172.20.10.2:8765');
    expect(formatAddress(a[12])).toBe('172.20.10.14:8765');
  });
});

describe('discover', () => {
  const setup = () => {
    const timers = new FakeTimers();
    const f = new SocketFactoryFake();
    return { timers, f, opts: { createSocket: f.create, timers, timeoutMs: 700 } };
  };

  it('opens all addresses in parallel, keeps the first to open and closes the rest', async () => {
    const { f, opts } = setup();
    const p = discover(hotspotAddresses(), opts);
    expect(f.sockets).toHaveLength(13);
    f.forUrl('172.20.10.5')[0].open();
    f.forUrl('172.20.10.9')[0].open(); // opens later: must lose
    const r = await p;
    expect(r?.address).toEqual({ host: '172.20.10.5', port: 8765 });
    expect(r?.socket).toBe(f.forUrl('172.20.10.5')[0]);
    expect(f.sockets.filter((s) => !s.closed)).toEqual([r?.socket]);
  });

  it('closes a socket that opens after a winner was chosen', async () => {
    const { f, opts } = setup();
    const p = discover(hotspotAddresses().slice(0, 3), opts);
    f.sockets[1].open();
    await p;
    f.sockets[2].open();
    expect(f.sockets[2].closed).toBe(true);
    expect(f.sockets[1].closed).toBe(false);
  });

  it('times out after 700 ms and closes everything', async () => {
    const { f, timers, opts } = setup();
    const p = discover(hotspotAddresses(), opts);
    timers.advance(699);
    expect(f.sockets.every((s) => !s.closed)).toBe(true);
    timers.advance(1);
    expect(await p).toBeNull();
    expect(f.sockets.every((s) => s.closed)).toBe(true);
    // a late open is ignored and closed
    f.sockets[0].readyState = 1;
    expect(timers.pending()).toBe(0);
  });

  it('resolves null early when every probe fails', async () => {
    const { f, timers, opts } = setup();
    const p = discover(hotspotAddresses(), opts);
    f.sockets.forEach((s) => s.fail());
    expect(await p).toBeNull();
    expect(timers.pending()).toBe(0);
  });

  it('resolves null for an empty list', async () => {
    const { opts } = setup();
    expect(await discover([], opts)).toBeNull();
  });

  it('survives a factory that throws', async () => {
    const { timers } = setup();
    const r = await discover([{ host: 'x', port: 1 }], {
      createSocket: () => {
        throw new Error('nope');
      },
      timers,
    });
    await flush();
    expect(r).toBeNull();
  });
});

describe('Backoff', () => {
  it('goes 0.5 s doubling to 5 s and resets', () => {
    const b = new Backoff();
    expect([1, 2, 3, 4, 5, 6, 7].map(() => b.next())).toEqual([500, 1000, 2000, 4000, 5000, 5000, 5000]);
    b.reset();
    expect(b.next()).toBe(500);
  });
});
