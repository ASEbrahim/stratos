import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';
import { API_BASE } from '../constants/config';
import { AuthError, ApiError } from './types';
import { reportError } from './utils';

const TOKEN_KEY = 'stratos_auth_token';
const DEVICE_ID_KEY = 'stratos_device_id';

// ── Device ID (persistent anonymous identity) ──
// Generated once on first launch, persists forever.
// Sent with every request so the backend can create/find an anonymous profile.
// When user later logs in, anonymous profile merges with their account.

let _cachedDeviceId: string | null = null;

export async function getDeviceId(): Promise<string> {
  if (_cachedDeviceId) return _cachedDeviceId;
  try {
    let id: string | null = null;
    if (Platform.OS === 'web') {
      id = localStorage.getItem(DEVICE_ID_KEY);
    } else {
      id = await SecureStore.getItemAsync(DEVICE_ID_KEY);
    }
    if (id) { _cachedDeviceId = id; return id; }
  } catch (err) { reportError('getDeviceId:read', err); }

  // Generate new device ID
  const id = `device-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  try {
    if (Platform.OS === 'web') {
      localStorage.setItem(DEVICE_ID_KEY, id);
    } else {
      await SecureStore.setItemAsync(DEVICE_ID_KEY, id);
    }
  } catch (err) { reportError('getDeviceId:write', err); }
  _cachedDeviceId = id;
  return id;
}

// ── Auth token (for logged-in users) ──

export async function getToken(): Promise<string | null> {
  try {
    if (Platform.OS === 'web') {
      return localStorage.getItem(TOKEN_KEY);
    }
    return await SecureStore.getItemAsync(TOKEN_KEY);
  } catch (err) {
    reportError('getToken', err);
    return null;
  }
}

export async function setToken(token: string): Promise<void> {
  if (Platform.OS === 'web') {
    localStorage.setItem(TOKEN_KEY, token);
    return;
  }
  await SecureStore.setItemAsync(TOKEN_KEY, token);
}

export async function clearToken(): Promise<void> {
  if (Platform.OS === 'web') {
    localStorage.removeItem(TOKEN_KEY);
    return;
  }
  await SecureStore.deleteItemAsync(TOKEN_KEY);
}

// ── API client ──

// Retry only GET requests, only on network errors or 502/503/504
const RETRYABLE_STATUSES = new Set([502, 503, 504]);

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const token = await getToken();
  const deviceId = await getDeviceId();
  const method = (options.method ?? 'GET').toUpperCase();
  const isGet = method === 'GET';
  const url = `${API_BASE.replace(/\/$/, '')}${path.startsWith('/') ? path : `/${path}`}`;

  const doFetch = async (): Promise<Response> => {
    return fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'X-Device-Id': deviceId,
        ...(token ? { 'X-Auth-Token': token } : {}),
        ...options.headers,
      },
    });
  };

  let response: Response;
  try {
    response = await doFetch();
  } catch (err) {
    // Network error — retry once for GET requests
    if (isGet) {
      await new Promise(r => setTimeout(r, 2000));
      response = await doFetch();
    } else {
      throw err;
    }
  }

  // Retry once on 502/503/504 for GET only
  if (isGet && RETRYABLE_STATUSES.has(response.status)) {
    await new Promise(r => setTimeout(r, 2000));
    response = await doFetch();
  }

  if (response.status === 401) {
    if (token) await clearToken();
    throw new AuthError('Session expired');
  }
  if (!response.ok) {
    throw new ApiError(response.status, await response.text());
  }
  return response.json();
}
