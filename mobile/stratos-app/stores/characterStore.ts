import { create } from 'zustand';
import { CharacterCard } from '../lib/types';
import * as characters from '../lib/characters';
import { getSavedCards, saveCard, removeSavedCard } from '../lib/storage';
import { reportError } from '../lib/utils';

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
  saveToLibrary: (card: CharacterCard) => Promise<void>;
  removeFromLibrary: (cardId: string) => Promise<void>;
}

export const useCharacterStore = create<CharacterState>((set, get) => ({
  trending: [], newCards: [], savedCards: [], myCards: [], isLoading: false,
  selectedGenre: null, searchQuery: '', searchResults: [],
  loadTrending: async () => { set({ isLoading: true }); try { const trending = await characters.getTrendingCharacters(); set({ trending, isLoading: false }); } catch (err) { reportError('loadTrending', err); set({ isLoading: false }); } },
  loadNew: async () => { try { const newCards = await characters.getNewCharacters(); set({ newCards }); } catch (err) { reportError('loadNew', err); } },
  loadSaved: async () => {
    try {
      const persisted = await getSavedCards();
      const apiCards = await characters.getSavedCharacters();
      // Merge: persisted local saves + mock library cards, deduplicated
      const allIds = new Set<string>();
      const merged: CharacterCard[] = [];
      for (const c of [...persisted, ...apiCards]) {
        if (!allIds.has(c.id)) { allIds.add(c.id); merged.push(c); }
      }
      set({ savedCards: merged });
    } catch (err) { reportError('loadSaved', err); }
  },
  loadMyCards: async () => { try { const myCards = await characters.getMyCharacters(); set({ myCards }); } catch (err) { reportError('loadMyCards', err); } },
  search: async (query, genre) => { set({ isLoading: true }); try { const searchResults = await characters.searchCharacters(query, genre); set({ searchResults, isLoading: false }); } catch (err) { reportError('characterStore:search', err); set({ isLoading: false }); } },
  setGenre: (genre) => { set({ selectedGenre: genre }); const { searchQuery } = get(); get().search(searchQuery, genre ?? undefined); },
  setSearchQuery: (query) => { set({ searchQuery: query }); const { selectedGenre } = get(); get().search(query, selectedGenre ?? undefined); },
  saveToLibrary: async (card) => {
    await saveCard(card);
    set((state) => ({ savedCards: [card, ...state.savedCards.filter(c => c.id !== card.id)] }));
  },
  removeFromLibrary: async (cardId) => {
    await removeSavedCard(cardId);
    set((state) => ({ savedCards: state.savedCards.filter(c => c.id !== cardId) }));
  },
}));
