import { USE_MOCKS } from '../constants/config';
import { apiFetch } from './api';
import { CharacterCard, CharacterCardCreate } from './types';
import { MOCK_CHARACTERS, generateId } from './mock';

let mockLibrary: CharacterCard[] = [];

export async function getTrendingCharacters(): Promise<CharacterCard[]> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 400));
    return [...MOCK_CHARACTERS].sort((a, b) => b.session_count - a.session_count);
  }
  return apiFetch<CharacterCard[]>('/api/characters/trending');
}

export async function getNewCharacters(page = 1): Promise<CharacterCard[]> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 300));
    return [...MOCK_CHARACTERS].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
  }
  return apiFetch<CharacterCard[]>(`/api/characters?page=${page}&sort=new`);
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
  const params = new URLSearchParams();
  if (query) params.set('q', query);
  if (genre) params.set('genre', genre);
  return apiFetch<CharacterCard[]>(`/api/characters/search?${params}`);
}

export async function getCharacter(id: string): Promise<CharacterCard | null> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 200));
    return MOCK_CHARACTERS.find(c => c.id === id) ?? mockLibrary.find(c => c.id === id) ?? null;
  }
  return apiFetch<CharacterCard>(`/api/characters/${id}`);
}

export async function createCharacter(data: CharacterCardCreate): Promise<CharacterCard> {
  if (USE_MOCKS) {
    await new Promise(r => setTimeout(r, 500));
    const card: CharacterCard = {
      ...data, id: generateId(), creator_id: 'user-1', creator_name: 'Ahmad',
      is_public: false, session_count: 0, rating: 0, rating_count: 0,
      created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
    };
    mockLibrary.push(card);
    return card;
  }
  return apiFetch<CharacterCard>('/api/characters', { method: 'POST', body: JSON.stringify(data) });
}

export async function getSavedCharacters(): Promise<CharacterCard[]> {
  if (USE_MOCKS) return mockLibrary;
  return apiFetch<CharacterCard[]>('/api/characters?filter=saved');
}

export async function getMyCharacters(): Promise<CharacterCard[]> {
  if (USE_MOCKS) return mockLibrary.filter(c => c.creator_id === 'user-1');
  return apiFetch<CharacterCard[]>('/api/characters?filter=mine');
}
