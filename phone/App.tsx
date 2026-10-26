import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Pressable,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { useKeepAwake } from 'expo-keep-awake';

import { AlignPressDetector, realTimers } from './src/buttons';
import { playHaptic } from './src/haptics';
import { PhoneLink, type LinkState } from './src/link';
import { MessageBuilder } from './src/message';
import type { LayerName, LaptopMessage } from './src/protocol';
import { startLocation, startMotion } from './src/sensors';
import { asyncStore } from './src/storage';

type LaptopStatus = Extract<LaptopMessage, { type: 'status' }>;

const LAYER_LABELS: [LayerName, string][] = [
  ['people', 'People'],
  ['vehicles', 'Vehicles'],
  ['landmarks', 'Landmarks'],
];

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
  const [rate, setRate] = useState(0);
  const [gpsAcc, setGpsAcc] = useState<number | null>(null);
  const [headingAcc, setHeadingAcc] = useState<number | null>(null);
  const [locationOk, setLocationOk] = useState<boolean | null>(null);
  const [manualText, setManualText] = useState('');
  const [manualOpen, setManualOpen] = useState(false);
  const [manualError, setManualError] = useState(false);
  const sentCount = useRef(0);

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
    void phoneLink.start().then(() => setManualText(phoneLink.getManualAddress() ?? ''));
    const stopMotion = startMotion((sample) => {
      if (!phoneLink.canSend()) return;
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

  const toggle = (name: LayerName, on: boolean) => {
    builder.setLayer(name, on);
    setLayers(builder.getLayers());
  };

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
        <Text style={styles.stat}>
          {laptop
            ? `${laptop.aligned ? 'Aligned' : 'Not aligned'}   ${laptop.selected ?? 'nothing selected'}`
            : 'No status from laptop'}
          {laptop?.message ? `   ${laptop.message}` : ''}
        </Text>

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

        <View style={styles.toggles}>
          {LAYER_LABELS.map(([name, label]) => (
            <View key={name} style={styles.toggle}>
              <Text style={styles.toggleText}>{label}</Text>
              <Switch value={layers[name]} onValueChange={(v) => toggle(name, v)} />
            </View>
          ))}
        </View>
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
  connectBtn: { backgroundColor: '#2563eb', borderRadius: 8, paddingHorizontal: 14, justifyContent: 'center' },
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
  toggle: { alignItems: 'center', gap: 4 },
  toggleText: { color: '#e5e7eb', fontSize: 14 },
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
