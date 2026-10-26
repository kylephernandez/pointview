// Laptop haptic requests → expo-haptics calls.
import * as Haptics from 'expo-haptics';
import type { HapticStyle } from './protocol';

export type HapticPlan =
  | { kind: 'impact'; style: 'Light' | 'Medium' }
  | { kind: 'notification'; type: 'Success' | 'Error' };

/** snap = light impact, select = medium impact, align_ok = success, error = error notification. */
export function planHaptic(style: HapticStyle): HapticPlan {
  switch (style) {
    case 'snap':
      return { kind: 'impact', style: 'Light' };
    case 'select':
      return { kind: 'impact', style: 'Medium' };
    case 'align_ok':
      return { kind: 'notification', type: 'Success' };
    case 'error':
      return { kind: 'notification', type: 'Error' };
  }
}

export interface HapticsApi {
  impactAsync(style: unknown): Promise<void>;
  notificationAsync(type: unknown): Promise<void>;
  ImpactFeedbackStyle: Record<string, unknown>;
  NotificationFeedbackType: Record<string, unknown>;
}

export function playHaptic(style: HapticStyle, api: HapticsApi = Haptics as unknown as HapticsApi): void {
  const plan = planHaptic(style);
  try {
    const p =
      plan.kind === 'impact'
        ? api.impactAsync(api.ImpactFeedbackStyle[plan.style])
        : api.notificationAsync(api.NotificationFeedbackType[plan.type]);
    void p.catch(() => undefined);
  } catch {
    // haptics are best effort
  }
}
