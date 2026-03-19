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
  isLoadingMore: boolean;
  hasMore: boolean;
  page: number;
  selectedGenre: string | null;
  searchQuery: string;
  searchResults: CharacterCard[];
  loadTrending: () => Promise<void>;
  loadNew: () => Promise<void>;
  loadMore: () => Promise<void>;
  loadSaved: () => Promise<void>;
  loadMyCards: () => Promise<void>;
  deleteMyCard: (cardId: string) => Promise<void>;
  search: (query: string, genre?: string) => Promise<void>;
  setGenre: (genre: string | null) => void;
  setSearchQuery: (query: string) => void;
  saveToLibrary: (card: CharacterCard) => Promise<void>;
  removeFromLibrary: (cardId: string) => Promise<void>;
}

export const useCharacterStore = create<CharacterState>((set, get) => ({
  trending: [], newCards: [], savedCards: [], myCards: [], isLoading: false,
  isLoadingMore: false, hasMore: true, page: 1,
  selectedGenre: null, searchQuery: '', searchResults: [],
  loadTrending: async () => { set({ isLoading: true }); try { const trending = await characters.getTrendingCharacters(); set({ trending, isLoading: false }); } catch (err) { reportError('loadTrending', err); set({ isLoading: false }); } },
  loadNew: async () => { set({ isLoading: true }); try { const newCards = await characters.getNewCharacters(); set({ newCards, page: 1, hasMore: true, isLoading: false }); } catch (err) { reportError('loadNew', err); set({ isLoading: false }); } },
  loadMore: async () => {
    const { isLoadingMore, hasMore, page } = get();
    if (isLoadingMore || !hasMore) return;
    set({ isLoadingMore: true });
    try {
      const nextPage = page + 1;
      const moreCards = await characters.getNewCharacters(nextPage);
      if (moreCards.length === 0) {
        set({ hasMore: false, isLoadingMore: false });
      } else {
        set(state => ({
          newCards: [...state.newCards, ...moreCards.filter(c => !state.newCards.some(e => e.id === c.id))],
          page: nextPage,
          isLoadingMore: false,
        }));
      }
    } catch (err) { reportError('loadMore', err); set({ isLoadingMore: false }); }
  },
  loadSaved: async () => {
    set({ isLoading: true });
    try {
      const persisted = await getSavedCards();
      const apiCards = await characters.getSavedCharacters();
      // Merge: persisted local saves + mock library cards, deduplicated
      const allIds = new Set<string>();
      const merged: CharacterCard[] = [];
      for (const c of [...persisted, ...apiCards]) {
        if (!allIds.has(c.id)) { allIds.add(c.id); merged.push(c); }
      }
      set({ savedCards: merged, isLoading: false });
    } catch (err) { reportError('loadSaved', err); set({ isLoading: false }); }
  },
  loadMyCards: async () => { set({ isLoading: true }); try { const myCards = await characters.getMyCharacters(); set({ myCards, isLoading: false }); } catch (err) { reportError('loadMyCards', err); set({ isLoading: false }); } },
  deleteMyCard: async (cardId) => {
    try {
      await characters.deleteCharacter(cardId);
      set(state => ({
        myCards: state.myCards.filter(c => c.id !== cardId),
        newCards: state.newCards.filter(c => c.id !== cardId),
        trending: state.trending.filter(c => c.id !== cardId),
        savedCards: state.savedCards.filter(c => c.id !== cardId),
      }));
    } catch (err) { reportError('deleteMyCard', err); throw err; }
  },
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
