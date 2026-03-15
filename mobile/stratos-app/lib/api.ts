import * as SecureStore from 'expo-secure-store';
import { API_BASE } from '../constants/config';
import { AuthError, ApiError } from './types';

const TOKEN_KEY = 'stratos_auth_token';

export async function getToken(): Promise<string | null> {
  try {
    return await SecureStore.getItemAsync(TOKEN_KEY);
  } catch {
    return null;
  }
}

export async function setToken(token: string): Promise<void> {
  await SecureStore.setItemAsync(TOKEN_KEY, token);
}

export async function clearToken(): Promise<void> {
  await SecureStore.deleteItemAsync(TOKEN_KEY);
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const token = await getToken();
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    },
  });

  if (response.status === 401) {
    await clearToken();
    throw new AuthError('Session expired');
  }
  if (!response.ok) {
    throw new ApiError(response.status, await response.text());
  }
  return response.json();
}
