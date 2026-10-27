import { StyleSheet, View } from 'react-native';
import type { TrackingTarget } from './protocol';

/** White quiet zone + the exact upright marker received from the laptop. No images or interpolation. */
export function TrackingPattern({ target, side }: { target: TrackingTarget; side: number }) {
  const rows = ['1111111111', ...target.cells.map((row) => `1${row}1`), '1111111111'];
  // A white cell around the black border is the detector quiet zone.
  return (
    <View style={{ width: side, height: side }} accessibilityLabel="Phone tracking pattern">
      {rows.map((row, y) => (
        <View key={y} style={styles.row}>
          {row.split('').map((cell, x) => (
            <View key={x} style={[styles.cell, { backgroundColor: cell === '0' ? '#000' : '#fff' }]} />
          ))}
        </View>
      ))}
    </View>
  );
}
const styles = StyleSheet.create({ row: { flex: 1, flexDirection: 'row' }, cell: { flex: 1 } });
