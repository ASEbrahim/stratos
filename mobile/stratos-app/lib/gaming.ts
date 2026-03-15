import { USE_MOCKS } from '../constants/config';
import { apiFetch } from './api';
import { GamingScenario } from './types';
import { MOCK_SCENARIOS } from './mock';

export async function getScenarios(): Promise<GamingScenario[]> {
  if (USE_MOCKS) { await new Promise(r => setTimeout(r, 400)); return MOCK_SCENARIOS; }
  try {
    const { scenarios } = await apiFetch<{ scenarios: any[] }>('/api/scenarios');
    return scenarios;
  } catch {
    return [];
  }
}

export async function getScenario(id: string): Promise<GamingScenario | null> {
  if (USE_MOCKS) { await new Promise(r => setTimeout(r, 200)); return MOCK_SCENARIOS.find(s => s.id === id) ?? null; }
  try {
    return await apiFetch<GamingScenario>(`/api/scenarios/${id}`);
  } catch {
    return null;
  }
}

export async function startGamingSession(scenarioId: string): Promise<string> {
  if (USE_MOCKS) return `gaming-session-${scenarioId}-${Date.now()}`;
  try {
    const result = await apiFetch<{ session_id: string }>(`/api/scenarios/${scenarioId}/start`, { method: 'POST' });
    return result.session_id;
  } catch {
    return `gaming-session-${scenarioId}-${Date.now()}`;
  }
}

export function parseOptions(text: string): { text: string; options: string[] } {
  const lines = text.split('\n');
  const options: string[] = [];
  const textLines: string[] = [];
  for (const line of lines) {
    const match = line.match(/^\s*(\d+)\.\s+(.+)$/);
    if (match) { options.push(match[2]); }
    else { const optMatch = line.match(/^\s*Option\s+(\d+):\s+(.+)$/i); if (optMatch) options.push(optMatch[2]); else textLines.push(line); }
  }
  return { text: textLines.join('\n').trim(), options };
}
