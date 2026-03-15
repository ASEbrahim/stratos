import { USE_MOCKS } from '../constants/config';
import { apiFetch, setToken, clearToken, getToken } from './api';
import { AuthResponse, User } from './types';
import { MOCK_USER } from './mock';

export async function login(email: string, password: string): Promise<AuthResponse> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 800));
    const token = 'mock-token-' + Date.now();
    await setToken(token);
    return { token, user: MOCK_USER };
  }
  const result = await apiFetch<AuthResponse>('/api/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
  await setToken(result.token);
  return result;
}

export async function register(name: string, email: string, password: string): Promise<AuthResponse> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 800));
    const token = 'mock-token-' + Date.now();
    await setToken(token);
    return { token, user: { ...MOCK_USER, name, email } };
  }
  const result = await apiFetch<AuthResponse>('/api/register', {
    method: 'POST',
    body: JSON.stringify({ name, email, password }),
  });
  await setToken(result.token);
  return result;
}

export async function getProfile(): Promise<User | null> {
  const token = await getToken();
  if (!token) return null;
  if (USE_MOCKS) return MOCK_USER;
  try {
    return await apiFetch<User>('/api/profile');
  } catch {
    return null;
  }
}

export async function logout(): Promise<void> {
  await clearToken();
}

export async function isAuthenticated(): Promise<boolean> {
  const token = await getToken();
  return !!token;
}
