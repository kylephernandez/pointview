import schema from '../../../laptop/pointview/schemas/phone_message.schema.json';
import { HOTSPOT_SCAN_RANGE, LAPTOP_PORT, PROTOCOL_VERSION, type PhoneMessage } from '../protocol';

describe('protocol contract', () => {
  it('matches the shared JSON Schema', () => {
    const msg: PhoneMessage = {
      v: PROTOCOL_VERSION,
      t_ms: 0,
      quat: [1, 0, 0, 0],
      omega: [0, 0, 0],
      heading_deg: null,
      heading_acc: null,
      lat: null,
      lon: null,
      gps_acc: null,
      buttons: [],
    };
    expect([...schema.required].sort()).toEqual(Object.keys(msg).sort());
    expect(LAPTOP_PORT).toBe(8765);
    expect(HOTSPOT_SCAN_RANGE.last - HOTSPOT_SCAN_RANGE.first + 1).toBe(13);
  });
});
