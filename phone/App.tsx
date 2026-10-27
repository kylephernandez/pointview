import { useEffect, useMemo, useRef, useState } from 'react';
import { Pressable, StyleSheet, Switch, Text, TextInput, View } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { useKeepAwake } from 'expo-keep-awake';

import { AlignPressDetector, realTimers } from './src/buttons';
import { playHaptic } from './src/haptics';
import { PhoneLink, type LinkState } from './src/link';
import { MessageBuilder } from './src/message';
import { LOOKS, type LayerName, type LaptopMessage, type Look } from './src/protocol';
import { defaultSettings, loadSettings, nextLook, saveSettings, statusMismatch } from './src/settings';
import { startLocation, startMotion } from './src/sensors';
import { asyncStore } from './src/storage';

type LaptopStatus = Extract<LaptopMessage, { type: 'status' }>;

const LAYER_GROUPS: { title: string; items: [LayerName, string][] }[] = [
  {
    title: 'Around you',
    items: [
      ['people', 'People'],
      ['vehicles', 'Vehicles'],
      ['landmarks', 'Landmarks'],
    ],
  },
  {
    title: 'Live data',
    items: [
      ['aircraft', 'Aircraft'],
      ['satellites', 'Satellites'],
      ['vessels', 'Vessels'],
      ['events', 'Events'],
    ],
  },
];

const DOUBLE_TAP_MS = 350;

const fmt = (v: number | null, digits = 0, unit = '') => (v === null ? '--' : `${v.toFixed(digits)}${unit}`);

export default function App() {
  useKeepAwake();

  const builder = useMemo(() => new MessageBuilder(), []);
  const [link, setLink] = useState<LinkState>({
    status: 'idle',
    address: null,
    needsManual: false,
    attempts: 0,
  });
  const [laptop, setLaptop] = useState<LaptopStatus | null>(null);
  const [layers, setLayers] = useState(builder.getLayers());
  const [look, setLook] = useState<Look>('normal');
  const [passthrough, setPassthrough] = useState(false);
  const [rate, setRate] = useState(0);
  const [gpsAcc, setGpsAcc] = useState<number | null>(null);
  const [headingAcc, setHeadingAcc] = useState<number | null>(null);
  const [locationOk, setLocationOk] = useState<boolean | null>(null);
  const [manualText, setManualText] = useState('');
  const [manualOpen, setManualOpen] = useState(false);
  const [manualError, setManualError] = useState(false);
  const sentCount = useRef(0);
  // Nothing is sent until the saved look/layers are applied, so defaults never overwrite them.
  const ready = useRef(false);
  const lastStatusTap = useRef(0);

  const phoneLink = useMemo(
    () =>
      new PhoneLink({
        createSocket: (url) => new WebSocket(url) as never,
        timers: realTimers,
        store: asyncStore,
        onState: setLink,
        onConnected: () => builder.markLayersDirty(),
        onMessage: (m) => {
          if (m.type === 'haptic') playHaptic(m.style);
          else setLaptop(m);
        },
      }),
    [builder],
  );

  const align = useMemo(() => new AlignPressDetector((b) => builder.pressButton(b)), [builder]);

  useEffect(() => {
    void loadSettings(asyncStore).then((s) => {
      for (const [n, on] of Object.entries(s.layers)) builder.setLayer(n as LayerName, on);
      builder.setLook(s.look);
      builder.setPassthrough(s.passthrough);
      setLayers(builder.getLayers());
      setLook(s.look);
      setPassthrough(s.passthrough);
      ready.current = true;
    });
    void phoneLink.start().then(() => setManualText(phoneLink.getManualAddress() ?? ''));
    const stopMotion = startMotion((sample) => {
      if (!ready.current || !phoneLink.canSend()) return;
      if (phoneLink.send(JSON.stringify(builder.build(sample)))) sentCount.current++;
    });
    const stopLocation = startLocation({
      onFix: (f) => {
        builder.setFix(f.lat, f.lon, f.gps_acc);
        setGpsAcc(f.gps_acc);
      },
      onHeading: (h) => {
        builder.setHeading(h.heading_deg, h.heading_acc);
        setHeadingAcc(h.heading_acc);
      },
      onPermission: setLocationOk,
    });
    const rateTimer = setInterval(() => {
      setRate(sentCount.current);
      sentCount.current = 0;
    }, 1000);
    return () => {
      stopMotion();
      stopLocation();
      clearInterval(rateTimer);
      phoneLink.stop();
    };
  }, [builder, phoneLink]);

  const persist = () =>
    void saveSettings(asyncStore, {
      ...defaultSettings(),
      look: builder.getLook(),
      passthrough: builder.getPassthrough(),
      layers: builder.getLayers(),
    });

  const toggle = (name: LayerName, on: boolean) => {
    builder.setLayer(name, on);
    setLayers(builder.getLayers());
    persist();
  };

  const chooseLook = (l: Look) => {
    builder.setLook(l);
    setLook(l);
    persist();
  };

  const choosePassthrough = (on: boolean) => {
    builder.setPassthrough(on);
    setPassthrough(on);
    persist();
  };

  /** Double-tap the laptop status line to cycle the look. */
  const onStatusTap = () => {
    const now = Date.now();
    if (now - lastStatusTap.current < DOUBLE_TAP_MS) {
      lastStatusTap.current = 0;
      chooseLook(nextLook(builder.getLook()));
    } else {
      lastStatusTap.current = now;
    }
  };

  const mismatch = statusMismatch(laptop, look, passthrough);

  const submitManual = async () => {
    const ok = await phoneLink.setManualAddress(manualText);
    setManualError(!ok);
    if (ok) setManualOpen(false);
  };

  const connected = link.status === 'connected';
  const showManual = manualOpen || (link.needsManual && !connected);
  const linkText = connected
    ? `Connected ${link.address ?? ''}`
    : link.status === 'searching'
      ? 'Searching for laptop...'
      : link.status === 'waiting'
        ? 'Reconnecting...'
        : 'Idle';

  return (
    <View style={styles.root}>
      <StatusBar style="light" />
      <View style={styles.top}>
        <Pressable onPress={() => setManualOpen((o) => !o)} accessibilityLabel="Connection status">
          <Text style={[styles.linkText, connected ? styles.ok : styles.warn]}>{linkText}</Text>
        </Pressable>
        <Text style={styles.stat}>
          {`${rate}/s   GPS ${fmt(gpsAcc, 0, ' m')}   Compass ${fmt(headingAcc, 0, '°')}`}
          {locationOk === false ? '   (location denied)' : ''}
        </Text>
        <Pressable onPress={onStatusTap} accessibilityLabel="Laptop status. Double-tap to cycle look">
          <Text style={styles.stat}>
            {laptop
              ? `${laptop.aligned ? 'Aligned' : 'Not aligned'}   ${laptop.selected ?? 'nothing selected'}`
              : 'No status from laptop'}
            {laptop?.message ? `   ${laptop.message}` : ''}
          </Text>
          <Text style={styles.stat}>
            {laptop?.look !== undefined || laptop?.passthrough !== undefined
              ? `Glasses: ${(laptop.look ?? look).toUpperCase()}   passthrough ${(laptop.passthrough ?? passthrough) ? 'on' : 'off'}`
              : 'Glasses: look not reported'}
          </Text>
          {mismatch && <Text style={styles.mismatch}>{`${mismatch} (resending)`}</Text>}
        </Pressable>

        {showManual && (
          <View style={styles.manualRow}>
            <TextInput
              style={[styles.input, manualError && styles.inputError]}
              value={manualText}
              onChangeText={(t) => {
                setManualText(t);
                setManualError(false);
              }}
              placeholder="laptop address, e.g. 172.20.10.3:8765"
              placeholderTextColor="#667"
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="numbers-and-punctuation"
              onSubmitEditing={submitManual}
              returnKeyType="go"
            />
            <Pressable style={styles.connectBtn} onPress={submitManual}>
              <Text style={styles.connectText}>Connect</Text>
            </Pressable>
          </View>
        )}

        <Pressable
          style={({ pressed }) => [styles.align, pressed && styles.alignPressed]}
          onPressIn={() => align.pressIn()}
          onPressOut={() => align.pressOut()}
          accessibilityLabel="Align. Tap to align, hold to align on landmark"
        >
          <Text style={styles.alignText}>ALIGN</Text>
          <Text style={styles.alignHint}>hold = landmark</Text>
        </Pressable>

        <View style={styles.looks}>
          {LOOKS.map((l) => (
            <Pressable
              key={l}
              style={[styles.lookBtn, look === l && styles.lookBtnOn]}
              onPress={() => chooseLook(l)}
              accessibilityLabel={`Look ${l}`}
              accessibilityState={{ selected: look === l }}
            >
              <Text style={[styles.lookText, look === l && styles.lookTextOn]}>{l.toUpperCase()}</Text>
            </Pressable>
          ))}
        </View>
        <View style={styles.ptRow}>
          <Text style={styles.toggleText}>Passthrough (camera behind overlay)</Text>
          <Switch value={passthrough} onValueChange={choosePassthrough} />
        </View>
        <Text style={styles.hint}>Looks only show over passthrough.</Text>

        {LAYER_GROUPS.map((g) => (
          <View key={g.title}>
            <Text style={styles.groupTitle}>{g.title}</Text>
            <View style={styles.toggles}>
              {g.items.map(([name, label]) => (
                <View key={name} style={styles.toggle}>
                  <Text style={styles.toggleText}>{label}</Text>
                  <Switch value={layers[name]} onValueChange={(v) => toggle(name, v)} />
                </View>
              ))}
            </View>
          </View>
        ))}
      </View>

      <Pressable
        style={({ pressed }) => [styles.info, pressed && styles.infoPressed]}
        onPressIn={() => builder.pressButton('info')}
        accessibilityLabel="Info"
      >
        <Text style={styles.infoText}>INFO</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000', paddingTop: 48, paddingBottom: 12 },
  top: { flex: 1, padding: 12, gap: 8 },
  linkText: { fontSize: 18, fontWeight: '600' },
  ok: { color: '#4ade80' },
  warn: { color: '#fbbf24' },
  stat: { color: '#9ca3af', fontSize: 14 },
  manualRow: { flexDirection: 'row', gap: 8 },
  input: {
    flex: 1,
    backgroundColor: '#111827',
    color: '#fff',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 8,
    fontSize: 16,
  },
  inputError: { borderWidth: 1, borderColor: '#ef4444' },
  connectBtn: {
    backgroundColor: '#2563eb',
    borderRadius: 8,
    paddingHorizontal: 14,
    justifyContent: 'center',
  },
  connectText: { color: '#fff', fontWeight: '600' },
  align: {
    flex: 1,
    backgroundColor: '#1d4ed8',
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  alignPressed: { backgroundColor: '#3b82f6' },
  alignText: { color: '#fff', fontSize: 40, fontWeight: '800' },
  alignHint: { color: '#bfdbfe', fontSize: 14 },
  toggles: { flexDirection: 'row', justifyContent: 'space-between' },
  mismatch: { color: '#f87171', fontSize: 14, fontWeight: '600' },
  looks: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  lookBtn: {
    flexBasis: '31%',
    flexGrow: 1,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: '#1f2937',
    alignItems: 'center',
  },
  lookBtnOn: { backgroundColor: '#f59e0b' },
  lookText: { color: '#d1d5db', fontSize: 18, fontWeight: '800' },
  lookTextOn: { color: '#000' },
  ptRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  hint: { color: '#6b7280', fontSize: 12, marginTop: -4 },
  groupTitle: {
    color: '#9ca3af',
    fontSize: 12,
    fontWeight: '700',
    marginBottom: 2,
  },
  toggle: { alignItems: 'center', gap: 2 },
  toggleText: { color: '#e5e7eb', fontSize: 12 },
  info: {
    flex: 1,
    backgroundColor: '#065f46',
    margin: 12,
    marginTop: 0,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  infoPressed: { backgroundColor: '#10b981' },
  infoText: { color: '#fff', fontSize: 56, fontWeight: '800' },
});
