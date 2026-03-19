import { USE_MOCKS } from '../constants/config';
import { apiFetch } from './api';
import { GamingScenario } from './types';
import { MOCK_SCENARIOS } from './mock';
import { reportError } from './utils';

export async function getScenarios(): Promise<GamingScenario[]> {
  if (USE_MOCKS) { await new Promise(r => setTimeout(r, 400)); return MOCK_SCENARIOS; }
  try {
    const { scenarios } = await apiFetch<{ scenarios: GamingScenario[] }>('/api/scenarios');
    return scenarios;
  } catch (err) {
    reportError('getScenarios', err);
    return [];
  }
}

export async function getScenario(id: string): Promise<GamingScenario | null> {
  if (USE_MOCKS) { await new Promise(r => setTimeout(r, 200)); return MOCK_SCENARIOS.find(s => s.id === id) ?? null; }
  try {
    return await apiFetch<GamingScenario>(`/api/scenarios/${id}`);
  } catch (err) {
    reportError('getScenario', err);
    return null;
  }
}

export async function startGamingSession(scenarioId: string): Promise<string> {
  if (USE_MOCKS) return `gaming-session-${scenarioId}-${Date.now()}`;
  try {
    const result = await apiFetch<{ session_id: string }>(`/api/scenarios/${scenarioId}/start`, { method: 'POST' });
    return result.session_id;
  } catch (err) {
    reportError('startGamingSession', err);
    return `gaming-session-${scenarioId}-${Date.now()}`;
  }
}

export interface WorldWizardConfig {
  name: string;
  description: string;
  genre: string;
  wizard_config: {
    story_position: string;
    starting_level: number;
    difficulty: string;
    starting_class: string;
    stats: { STR: number; DEX: number; INT: number };
    extras: string[];
    canon_characters: boolean;
    real_names: boolean;
    lore_depth: string;
  };
}

export async function createWorld(config: WorldWizardConfig): Promise<{
  ok: boolean;
  name: string;
  status: string;
}> {
  return apiFetch('/api/scenarios/create', {
    method: 'POST',
    body: JSON.stringify(config),
  });
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
