import { create } from 'zustand';
import { CharacterCard } from '../lib/types';
import * as characters from '../lib/characters';

interface CharacterState {
  trending: CharacterCard[];
  newCards: CharacterCard[];
  savedCards: CharacterCard[];
  myCards: CharacterCard[];
  isLoading: boolean;
  selectedGenre: string | null;
  searchQuery: string;
  searchResults: CharacterCard[];
  loadTrending: () => Promise<void>;
  loadNew: () => Promise<void>;
  loadSaved: () => Promise<void>;
  loadMyCards: () => Promise<void>;
  search: (query: string, genre?: string) => Promise<void>;
  setGenre: (genre: string | null) => void;
  setSearchQuery: (query: string) => void;
}

export const useCharacterStore = create<CharacterState>((set, get) => ({
  trending: [], newCards: [], savedCards: [], myCards: [], isLoading: false,
  selectedGenre: null, searchQuery: '', searchResults: [],
  loadTrending: async () => { set({ isLoading: true }); try { const trending = await characters.getTrendingCharacters(); set({ trending, isLoading: false }); } catch { set({ isLoading: false }); } },
  loadNew: async () => { try { const newCards = await characters.getNewCharacters(); set({ newCards }); } catch {} },
  loadSaved: async () => { try { const savedCards = await characters.getSavedCharacters(); set({ savedCards }); } catch {} },
  loadMyCards: async () => { try { const myCards = await characters.getMyCharacters(); set({ myCards }); } catch {} },
  search: async (query, genre) => { set({ isLoading: true }); try { const searchResults = await characters.searchCharacters(query, genre); set({ searchResults, isLoading: false }); } catch { set({ isLoading: false }); } },
  setGenre: (genre) => { set({ selectedGenre: genre }); const { searchQuery } = get(); get().search(searchQuery, genre ?? undefined); },
  setSearchQuery: (query) => { set({ searchQuery: query }); const { selectedGenre } = get(); get().search(query, selectedGenre ?? undefined); },
}));
