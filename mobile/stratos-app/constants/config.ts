export const USE_MOCKS = false;

// Local network IP — used when running via Expo Go / WiFi
const WIFI_API_BASE = 'http://192.168.0.148:8080';
// Localhost — used when running via USB with `adb reverse tcp:8080 tcp:8080`
const USB_API_BASE = 'http://localhost:8080';

// Use WiFi base for dev (Expo Go), localhost for release APK (USB tunnel)
export const API_BASE = __DEV__ ? WIFI_API_BASE : USB_API_BASE;

export const FEATURES = {
  enableGaming: true,
  enableNSFW: true,
  enableTavernImport: true,
};

export const Config = {
  API_BASE,
  USE_MOCKS,
  FEATURES,
  STREAM_TIMEOUT_MS: 60_000,
  STALL_TIMEOUT_MS: 30_000,
  MAX_CHAT_SESSIONS: 50,
  MAX_MOCK_LIBRARY: 200,
  PERSIST_DEBOUNCE_MS: 500,
} as const;
