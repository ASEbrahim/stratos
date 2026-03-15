import { USE_MOCKS } from '../constants/config';
import { apiFetch } from './api';
import { CharacterCard, CharacterCardCreate } from './types';
import { mapCardFromBackend, mapCardsFromBackend } from './mappers';
import { MOCK_CHARACTERS, generateId } from './mock';

let mockLibrary: CharacterCard[] = [];

export async function getTrendingCharacters(): Promise<CharacterCard[]> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 400));
    return [...MOCK_CHARACTERS].sort((a, b) => b.session_count - a.session_count);
  }
  try {
    const { cards } = await apiFetch<{ cards: any[] }>('/api/cards/trending');
    return cards.length > 0 ? mapCardsFromBackend(cards) : MOCK_CHARACTERS.slice(0, 5);
  } catch {
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
    const { cards } = await apiFetch<{ cards: any[] }>(`/api/cards/browse?sort=newest&offset=${offset}`);
    return cards.length > 0 ? mapCardsFromBackend(cards) : MOCK_CHARACTERS;
  } catch {
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
    const { cards } = await apiFetch<{ cards: any[] }>(`/api/cards/search?${params}`);
    return mapCardsFromBackend(cards);
  } catch {
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
    const raw = await apiFetch<any>(`/api/cards/${id}`);
    return mapCardFromBackend(raw);
  } catch {
    return mockMatch ?? null;
  }
}

export async function createCharacter(data: CharacterCardCreate): Promise<CharacterCard> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 500));
    const card: CharacterCard = {
      ...data, id: generateId(), creator_id: 'user-1', creator_name: 'You',
      is_public: false, session_count: 0, rating: 0, rating_count: 0,
      created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
    };
    mockLibrary.push(card);
    return card;
  }
  try {
    const result = await apiFetch<{ ok: boolean; card_id: string }>('/api/cards', {
      method: 'POST',
      body: JSON.stringify({
        name: data.name, physical_description: data.physical_description,
        speech_pattern: data.speech_pattern, emotional_trigger: data.emotional_trigger,
        defensive_mechanism: data.defensive_mechanism, vulnerability: data.vulnerability,
        specific_detail: data.specific_detail, personality: data.personality,
        scenario: data.scenario, first_message: data.first_message,
        genre_tags: data.genre_tags, content_rating: data.content_rating,
      }),
    });
    const card = await getCharacter(result.card_id);
    return card!;
  } catch {
    // Fallback to local creation
    const card: CharacterCard = {
      ...data, id: generateId(), creator_id: 'local', creator_name: 'You',
      is_public: false, session_count: 0, rating: 0, rating_count: 0,
      created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
    };
    mockLibrary.push(card);
    return card;
  }
}

export async function getSavedCharacters(): Promise<CharacterCard[]> {
  if (USE_MOCKS) return mockLibrary;
  try {
    const { cards } = await apiFetch<{ cards: any[] }>('/api/cards/my');
    return mapCardsFromBackend(cards);
  } catch {
    return mockLibrary;
  }
}

export async function getMyCharacters(): Promise<CharacterCard[]> {
  if (USE_MOCKS) return mockLibrary.filter(c => c.creator_id === 'user-1');
  try {
    const { cards } = await apiFetch<{ cards: any[] }>('/api/cards/my');
    return mapCardsFromBackend(cards);
  } catch {
    return mockLibrary;
  }
}
