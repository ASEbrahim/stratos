import { create } from 'zustand';
import { ChatMessage, CharacterCard, Suggestion } from '../lib/types';
import { streamMessage, getSuggestions, createMessageId } from '../lib/chat';

interface ChatState {
  sessionId: string | null;
  character: CharacterCard | null;
  persona: 'roleplay' | 'gaming';
  messages: ChatMessage[];
  suggestions: Suggestion[];
  isStreaming: boolean;
  streamingContent: string;
  isLoadingSuggestions: boolean;
  startSession: (character: CharacterCard, persona?: 'roleplay' | 'gaming') => void;
  sendMessage: (content: string) => Promise<void>;
  loadSuggestions: () => Promise<void>;
  clearSession: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessionId: null, character: null, persona: 'roleplay', messages: [], suggestions: [],
  isStreaming: false, streamingContent: '', isLoadingSuggestions: false,
  startSession: (character, persona = 'roleplay') => {
    const sessionId = `session-${character.id}-${Date.now()}`;
    const messages: ChatMessage[] = [];
    if (character.first_message) {
      messages.push({ id: createMessageId(), role: 'assistant', content: character.first_message, timestamp: new Date().toISOString() });
    }
    set({ sessionId, character, persona, messages, suggestions: [], isStreaming: false, streamingContent: '' });
  },
  sendMessage: async (content) => {
    const { sessionId, character, persona, messages } = get();
    if (!sessionId) return;
    const userMessage: ChatMessage = { id: createMessageId(), role: 'user', content, timestamp: new Date().toISOString() };
    set({ messages: [...messages, userMessage], isStreaming: true, streamingContent: '', suggestions: [] });
    let accumulated = '';
    await streamMessage(sessionId, content, persona, character,
      (chunk) => { accumulated += chunk; set({ streamingContent: accumulated }); },
      () => {
        const assistantMessage: ChatMessage = { id: createMessageId(), role: 'assistant', content: accumulated, timestamp: new Date().toISOString() };
        set((state) => ({ messages: [...state.messages, assistantMessage], isStreaming: false, streamingContent: '' }));
        get().loadSuggestions();
      },
    );
  },
  loadSuggestions: async () => {
    const { sessionId, persona, messages } = get();
    if (!sessionId || messages.length === 0) return;
    set({ isLoadingSuggestions: true });
    try {
      const lastMessage = messages[messages.length - 1]?.content ?? '';
      const suggestions = await getSuggestions(sessionId, persona, lastMessage);
      set({ suggestions, isLoadingSuggestions: false });
    } catch { set({ isLoadingSuggestions: false }); }
  },
  clearSession: () => { set({ sessionId: null, character: null, messages: [], suggestions: [], isStreaming: false, streamingContent: '' }); },
}));
