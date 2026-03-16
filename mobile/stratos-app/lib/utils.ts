import { API_BASE } from '../constants/config';

/**
 * Safe JSON parse — returns fallback on malformed input instead of crashing.
 */
export function safeParse<T>(json: string, fallback: T): T {
  try { return JSON.parse(json); } catch { return fallback; }
}

/**
 * Build a normalized API URL — prevents double slashes.
 */
export function buildUrl(path: string): string {
  const base = API_BASE.replace(/\/$/, '');
  const clean = path.startsWith('/') ? path : `/${path}`;
  return `${base}${clean}`;
}

/**
 * Centralized error reporting. All catch blocks should call this.
 * Future: wire to Sentry or crash reporting service.
 */
export function reportError(context: string, error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`[StratOS:${context}] ${message}`);
  if (__DEV__ && error instanceof Error && error.stack) {
    console.error(error.stack);
  }
}
