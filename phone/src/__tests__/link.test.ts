import { parseLaptopMessage, PhoneLink, STORE_KEY_LAST_GOOD, STORE_KEY_MANUAL, type LinkState } from '../link';
import type { LaptopMessage } from '../protocol';
import { FakeTimers, MemoryStore, SocketFactoryFake, flush } from '../testutil/fakes';

const setup = (store = new MemoryStore()) => {
  const timers = new FakeTimers();
  const f = new SocketFactoryFake();
  const states: LinkState[] = [];
  const messages: LaptopMessage[] = [];
  const onConnected = jest.fn();
  const link = new PhoneLink({
    createSocket: f.create,
    timers,
    store,
    onState: (s) => states.push(s),
    onMessage: (m) => messages.push(m),
    onConnected,
  });
  return { timers, f, states, messages, store, link, onConnected };
};

describe('parseLaptopMessage', () => {
  it('accepts valid haptic and status messages', () => {
    expect(parseLaptopMessage('{"type":"haptic","style":"snap"}')).toEqual({ type: 'haptic', style: 'snap' });
    expect(parseLaptopMessage('{"type":"status","aligned":true,"selected":"car","fps":58.9}')).toEqual({
      type: 'status',
      aligned: true,
      selected: 'car',
      fps: 58.9,
    });
    expect(parseLaptopMessage('{"type":"status","aligned":false,"selected":null}')).toEqual({
      type: 'status',
      aligned: false,
      selected: null,
    });
  });
  it('rejects junk', () => {
    for (const bad of ['', 'x', '[]', '{"type":"haptic","style":"zap"}', '{"type":"status"}', '{"type":"nope"}']) {
      expect(parseLaptopMessage(bad)).toBeNull();
    }
    expect(parseLaptopMessage(42)).toBeNull();
  });
});

describe('PhoneLink', () => {
  it('scans the hotspot range, connects to the first that opens, and remembers it', async () => {
    const { f, link, states, store, onConnected } = setup();
    const started = link.start();
    await flush();
    expect(f.sockets).toHaveLength(13);
    f.forUrl('172.20.10.4')[0].open();
    await started;
    await flush();
    expect(link.getState()).toMatchObject({ status: 'connected', address: '172.20.10.4:8765' });
    expect(states.map((s) => s.status)).toEqual(['searching', 'connected']);
    expect(store.data.get(STORE_KEY_LAST_GOOD)).toBe('172.20.10.4:8765');
    expect(onConnected).toHaveBeenCalledTimes(1);
    expect(link.isOpen()).toBe(true);
  });

  it('sends frames only while open and not backed up', async () => {
    const { f, link } = setup();
    expect(link.send('x')).toBe(false);
    void link.start();
    await flush();
    const s = f.forUrl('172.20.10.2')[0];
    s.open();
    await flush();
    expect(link.send('{"a":1}')).toBe(true);
    expect(s.sent).toEqual(['{"a":1}']);
    s.bufferedAmount = 10 * 1024 * 1024;
    expect(link.send('y')).toBe(false);
  });

  it('retries with backoff 0.5 s, 1 s, 2 s... and asks for a manual address after a failed scan', async () => {
    const { f, timers, link } = setup();
    void link.start();
    await flush();
    f.sockets.forEach((s) => s.fail());
    await flush();
    expect(link.getState()).toMatchObject({ status: 'waiting', needsManual: true });

    const rounds: number[] = [];
    const delays = [500, 1000, 2000, 4000, 5000, 5000];
    for (const d of delays) {
      f.clear();
      timers.advance(d - 1);
      await flush();
      expect(f.sockets).toHaveLength(0); // not yet
      timers.advance(1);
      await flush();
      expect(f.sockets).toHaveLength(13); // next scan round
      rounds.push(d);
      f.sockets.forEach((s) => s.fail());
      await flush();
    }
    expect(rounds).toEqual(delays);
  });

  it('after a drop, tries the last good address first, with the backoff reset', async () => {
    const { f, timers, link, states } = setup();
    void link.start();
    await flush();
    f.forUrl('172.20.10.7')[0].open();
    await flush();
    f.clear();
    // second drop would otherwise use a longer delay: verify reset on success
    link.reconnectNow();
    await flush();
    f.forUrl('172.20.10.7')[0].open(); // lastGood probed first
    await flush();
    expect(f.sockets).toHaveLength(1);
    f.sockets[0].drop();
    expect(link.getState().status).toBe('waiting');
    f.clear();
    timers.advance(499);
    await flush();
    expect(f.sockets).toHaveLength(0);
    timers.advance(1);
    await flush();
    expect(f.sockets).toHaveLength(1);
    expect(f.sockets[0].url).toBe('ws://172.20.10.7:8765/');
    f.sockets[0].open();
    await flush();
    expect(link.getState().status).toBe('connected');
    expect(states.filter((s) => s.status === 'connected').length).toBeGreaterThanOrEqual(3);
  });

  it('falls back to a full scan when the last good address no longer answers', async () => {
    const store = new MemoryStore();
    store.data.set(STORE_KEY_LAST_GOOD, '172.20.10.7:8765');
    const { f, timers, link } = setup(store);
    void link.start();
    await flush();
    expect(f.sockets).toHaveLength(1);
    timers.advance(700);
    await flush();
    expect(f.sockets).toHaveLength(1 + 12); // scan excludes the one already tried
    f.forUrl('172.20.10.12')[0].open();
    await flush();
    expect(link.getState().address).toBe('172.20.10.12:8765');
  });

  it('manual address is persisted, tried first, and used immediately when set', async () => {
    const store = new MemoryStore();
    const a = setup(store);
    void a.link.start();
    await flush();
    expect(await a.link.setManualAddress('not an address')).toBe(false);
    expect(await a.link.setManualAddress('192.168.1.50')).toBe(true);
    expect(store.data.get(STORE_KEY_MANUAL)).toBe('192.168.1.50:8765');
    await flush();
    expect(a.f.forUrl('192.168.1.50')).toHaveLength(1);
    a.f.forUrl('192.168.1.50')[0].open();
    await flush();
    expect(a.link.getState().address).toBe('192.168.1.50:8765');
    a.link.stop();

    // new launch reads the persisted manual address and probes it first
    const b = setup(store);
    void b.link.start();
    await flush();
    expect(b.f.sockets.map((s) => s.url)).toEqual(['ws://192.168.1.50:8765/']);
  });

  it('delivers laptop messages and ignores junk', async () => {
    const { f, link, messages } = setup();
    void link.start();
    await flush();
    f.sockets[0].open();
    await flush();
    f.sockets[0].receive('{"type":"haptic","style":"error"}');
    f.sockets[0].receive('garbage');
    expect(messages).toEqual([{ type: 'haptic', style: 'error' }]);
  });

  it('stop() closes the socket and cancels retries', async () => {
    const { f, timers, link } = setup();
    void link.start();
    await flush();
    f.sockets[0].open();
    await flush();
    const s = f.sockets[0];
    link.stop();
    expect(s.closed).toBe(true);
    expect(link.getState().status).toBe('idle');
    expect(timers.pending()).toBe(0);
    expect(link.isOpen()).toBe(false);
  });
});
