export const USE_MOCKS = false;

const DEV_API_BASE = 'http://192.168.0.148:8080';
const PROD_API_BASE = 'https://api.stratos.app';

export const API_BASE = __DEV__ ? DEV_API_BASE : PROD_API_BASE;

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
