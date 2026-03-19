import { USE_MOCKS, Config } from '../constants/config';
import { apiFetch } from './api';
import { CharacterCard, CharacterCardCreate } from './types';
import { BackendCard, mapCardFromBackend, mapCardsFromBackend } from './mappers';
import { MOCK_CHARACTERS, generateId } from './mock';
import { reportError } from './utils';

let mockLibrary: CharacterCard[] = [];

// Default formatting instructions for cards that don't specify speech_pattern
const DEFAULT_SPEECH_PATTERN = 'Use *asterisks* for actions/narration and "quotes" for dialogue. Mix prose with action beats. Keep response length proportional — short inputs (1-3 words) get 1-2 sentences max. Never repeat or echo the user\'s words back. Never over-write.';

/**
 * Auto-fill empty advanced fields with sensible defaults.
 * Called before saving so even simple cards get proper RP formatting.
 */
export function fillCardDefaults(data: CharacterCardCreate): CharacterCardCreate {
  const filled = { ...data };
  if (!filled.speech_pattern?.trim()) {
    filled.speech_pattern = DEFAULT_SPEECH_PATTERN;
  }
  return filled;
}

export async function getTrendingCharacters(): Promise<CharacterCard[]> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 400));
    return [...MOCK_CHARACTERS].sort((a, b) => b.session_count - a.session_count);
  }
  try {
    const { cards } = await apiFetch<{ cards: BackendCard[] }>('/api/cards/trending');
    return cards.length > 0 ? mapCardsFromBackend(cards) : MOCK_CHARACTERS.slice(0, 5);
  } catch (err) {
    reportError('getTrendingCharacters', err);
    return [...MOCK_CHARACTERS].sort((a, b) => b.session_count - a.session_count);
  }
}

export async function getNewCharacters(page = 1): Promise<CharacterCard[]> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 300));
    return [...MOCK_CHARACTERS].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
  }
  try {
    const offset = (page - 1) * 20;
    const { cards } = await apiFetch<{ cards: BackendCard[] }>(`/api/cards/browse?sort=newest&offset=${offset}`);
    return cards.length > 0 ? mapCardsFromBackend(cards) : MOCK_CHARACTERS;
  } catch (err) {
    reportError('getNewCharacters', err);
    return [...MOCK_CHARACTERS].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
  }
}

export async function searchCharacters(query: string, genre?: string): Promise<CharacterCard[]> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 300));
    let results = MOCK_CHARACTERS;
    if (genre) results = results.filter(c => c.genre_tags.includes(genre));
    if (query) {
      const q = query.toLowerCase();
      results = results.filter(c => c.name.toLowerCase().includes(q) || c.description.toLowerCase().includes(q));
    }
    return results;
  }
  try {
    const params = new URLSearchParams();
    if (query) params.set('q', query);
    if (genre) params.set('genre', genre);
    const { cards } = await apiFetch<{ cards: BackendCard[] }>(`/api/cards/search?${params}`);
    return mapCardsFromBackend(cards);
  } catch (err) {
    reportError('searchCharacters', err);
    let results = MOCK_CHARACTERS;
    if (genre) results = results.filter(c => c.genre_tags.includes(genre));
    if (query) { const q = query.toLowerCase(); results = results.filter(c => c.name.toLowerCase().includes(q)); }
    return results;
  }
}

export async function getCharacter(id: string): Promise<CharacterCard | null> {
  // Always check mocks first (works offline)
  const mockMatch = MOCK_CHARACTERS.find(c => c.id === id) ?? mockLibrary.find(c => c.id === id);
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 200));
    return mockMatch ?? null;
  }
  try {
    const raw = await apiFetch<BackendCard>(`/api/cards/${id}`);
    return mapCardFromBackend(raw);
  } catch (err) {
    reportError('getCharacter', err);
    return mockMatch ?? null;
  }
}

export async function createCharacter(data: CharacterCardCreate): Promise<CharacterCard> {
  const filled = fillCardDefaults(data);
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 500));
    const card: CharacterCard = {
      ...filled, id: generateId(), creator_id: 'user-1', creator_name: 'You',
      is_public: false, session_count: 0, rating: 0, rating_count: 0,
      created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
    };
    mockLibrary.push(card);
    if (mockLibrary.length > Config.MAX_MOCK_LIBRARY) mockLibrary = mockLibrary.slice(-Config.MAX_MOCK_LIBRARY);
    return card;
  }
  // No try/catch — let errors propagate to caller (CardEditor shows alert)
  const result = await apiFetch<{ ok: boolean; card_id: string }>('/api/cards', {
    method: 'POST',
    body: JSON.stringify({
      name: filled.name, physical_description: filled.physical_description,
      speech_pattern: filled.speech_pattern, emotional_trigger: filled.emotional_trigger,
      defensive_mechanism: filled.defensive_mechanism, vulnerability: filled.vulnerability,
      specific_detail: filled.specific_detail, personality: filled.personality,
      scenario: filled.scenario, first_message: filled.first_message,
      genre_tags: filled.genre_tags, content_rating: filled.content_rating,
      gender: filled.gender, archetype_override: filled.archetype_override,
      narration_pov: filled.narration_pov, relationship_to_user: filled.relationship_to_user,
      nsfw_comfort: filled.nsfw_comfort, response_length_pref: filled.response_length_pref,
      age_range: filled.age_range, personality_tags: filled.personality_tags,
    }),
  });
  const card = await getCharacter(result.card_id);
  return card!;
}

export async function getSavedCharacters(): Promise<CharacterCard[]> {
  if (USE_MOCKS) return mockLibrary;
  try {
    const { cards } = await apiFetch<{ cards: BackendCard[] }>('/api/cards/my');
    return mapCardsFromBackend(cards);
  } catch (err) {
    reportError('getSavedCharacters', err);
    return mockLibrary;
  }
}

export async function getMyCharacters(): Promise<CharacterCard[]> {
  if (USE_MOCKS) return mockLibrary.filter(c => c.creator_id === 'user-1');
  try {
    const { cards } = await apiFetch<{ cards: BackendCard[] }>('/api/cards/my');
    return mapCardsFromBackend(cards);
  } catch (err) {
    reportError('getMyCharacters', err);
    return mockLibrary;
  }
}

export async function deleteCharacter(cardId: string): Promise<void> {
  if (USE_MOCKS) {
    mockLibrary = mockLibrary.filter(c => c.id !== cardId);
    return;
  }
  await apiFetch(`/api/cards/${cardId}`, { method: 'DELETE' });
}

export async function quickGenerateCharacter(params: {
  prompt?: string;
  genre?: string;
  archetype?: string;
  relationship?: string;
}): Promise<CharacterCardCreate> {
  const result = await apiFetch<{ ok: boolean; card: CharacterCardCreate }>('/api/cards/generate', {
    method: 'POST',
    body: JSON.stringify(params),
  });
  if (!result.ok || !result.card) throw new Error('Character generation failed');
  return fillCardDefaults(result.card);
}
