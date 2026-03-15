export const USE_MOCKS = true;

const DEV_API_BASE = 'http://192.168.1.100:8080';
const PROD_API_BASE = 'https://api.stratos.app';

export const API_BASE = __DEV__ ? DEV_API_BASE : PROD_API_BASE;

export const FEATURES = {
  enableGaming: true,
  enableNSFW: false,
  enableTavernImport: true,
};
