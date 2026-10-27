// Persistent key-value store (AsyncStorage is bundled with Expo Go).
import AsyncStorage from '@react-native-async-storage/async-storage';
import type { KeyValueStore } from './link';

export const asyncStore: KeyValueStore = {
  get: (key) => AsyncStorage.getItem(key),
  set: (key, value) => AsyncStorage.setItem(key, value),
};
