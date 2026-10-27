import { planHaptic, playHaptic, type HapticsApi } from '../haptics';

describe('haptics', () => {
  it('maps styles', () => {
    expect(planHaptic('snap')).toEqual({ kind: 'impact', style: 'Light' });
    expect(planHaptic('select')).toEqual({ kind: 'impact', style: 'Medium' });
    expect(planHaptic('align_ok')).toEqual({ kind: 'notification', type: 'Success' });
    expect(planHaptic('error')).toEqual({ kind: 'notification', type: 'Error' });
  });

  it('calls the matching expo-haptics function', () => {
    const api: HapticsApi = {
      impactAsync: jest.fn().mockResolvedValue(undefined),
      notificationAsync: jest.fn().mockResolvedValue(undefined),
      ImpactFeedbackStyle: { Light: 'L', Medium: 'M' },
      NotificationFeedbackType: { Success: 'S', Error: 'E' },
    };
    playHaptic('snap', api);
    playHaptic('select', api);
    playHaptic('align_ok', api);
    playHaptic('error', api);
    expect(api.impactAsync).toHaveBeenNthCalledWith(1, 'L');
    expect(api.impactAsync).toHaveBeenNthCalledWith(2, 'M');
    expect(api.notificationAsync).toHaveBeenNthCalledWith(1, 'S');
    expect(api.notificationAsync).toHaveBeenNthCalledWith(2, 'E');
  });
});
