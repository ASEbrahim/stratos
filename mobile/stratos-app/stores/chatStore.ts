import { create } from 'zustand';
import { ChatMessage, ChatSession, CharacterCard, Suggestion } from '../lib/types';
import { streamMessage, getSuggestions, createMessageId } from '../lib/chat';
import { saveChatSession, loadChatSessions, getChatSession, incrementStat } from '../lib/storage';

interface ChatState {
  sessionId: string | null;
  character: CharacterCard | null;
  persona: 'roleplay' | 'gaming';
  messages: ChatMessage[];
  suggestions: Suggestion[];
  isStreaming: boolean;
  streamingContent: string;
  isLoadingSuggestions: boolean;
  recentSessions: ChatSession[];
  startSession: (character: CharacterCard, persona?: 'roleplay' | 'gaming') => void;
  resumeSession: (session: ChatSession) => void;
  sendMessage: (content: string, directorNote?: string) => Promise<void>;
  loadSuggestions: () => Promise<void>;
  loadRecentSessions: () => Promise<void>;
  clearSession: () => void;
  persistSession: () => Promise<void>;
  regenerateLastMessage: () => Promise<void>;
}

export const useChatStore = create<ChatState>((set, get) => ({
  sessionId: null, character: null, persona: 'roleplay', messages: [], suggestions: [],
  isStreaming: false, streamingContent: '', isLoadingSuggestions: false, recentSessions: [],
  startSession: (character, persona = 'roleplay') => {
    const sessionId = `session-${character.id}-${Date.now()}`;
    const messages: ChatMessage[] = [];
    if (character.first_message) {
      messages.push({ id: createMessageId(), role: 'assistant', content: character.first_message, timestamp: new Date().toISOString() });
    }
    set({ sessionId, character, persona, messages, suggestions: [], isStreaming: false, streamingContent: '' });
    incrementStat('totalSessions').catch(() => {});
  },
  resumeSession: (session: ChatSession) => {
    set({
      sessionId: session.id,
      character: { id: session.character_id, name: session.character_name, avatar_url: session.character_avatar } as CharacterCard,
      persona: session.persona,
      messages: session.messages,
      suggestions: [], isStreaming: false, streamingContent: '',
    });
  },
  sendMessage: async (content, directorNote) => {
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
        incrementStat('totalMessages', 2).catch(() => {}); // user + assistant
        get().persistSession().catch(() => {});
        get().loadSuggestions().catch(() => {});
      },
      directorNote,
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
  loadRecentSessions: async () => {
    try {
      const sessions = await loadChatSessions();
      set({ recentSessions: sessions });
    } catch {}
  },
  clearSession: () => {
    get().persistSession().catch(() => {});
    set({ sessionId: null, character: null, messages: [], suggestions: [], isStreaming: false, streamingContent: '' });
  },
  persistSession: async () => {
    const { sessionId, character, persona, messages } = get();
    if (!sessionId || !character || messages.length === 0) return;
    const session: ChatSession = {
      id: sessionId,
      character_id: character.id,
      character_name: character.name,
      character_avatar: character.avatar_url,
      persona,
      messages,
      created_at: messages[0]?.timestamp ?? new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    await saveChatSession(session);
  },
  regenerateLastMessage: async () => {
    const { sessionId, character, persona, messages } = get();
    if (!sessionId || messages.length < 2) return;
    // Remove the last assistant message from display
    const trimmed = messages.filter((_, i) => i < messages.length - 1 || messages[i].role !== 'assistant');
    set({ messages: trimmed, isStreaming: true, streamingContent: '', suggestions: [] });
    // Use the regenerate endpoint (swipe) — it handles deactivating the old response
    // and generates a fresh one with different sampling
    const { regenerateMessage } = await import('../lib/rp');
    try {
      await regenerateMessage(sessionId, 'main', character?.id);
    } catch { /* swipe endpoint may fail, fall back to re-streaming */ }
    // Re-stream from the regular chat endpoint with the last user message
    const lastUser = trimmed.filter(m => m.role === 'user').pop();
    if (!lastUser) return;
    let accumulated = '';
    await streamMessage(sessionId, lastUser.content, persona, character,
      (chunk) => { accumulated += chunk; set({ streamingContent: accumulated }); },
      () => {
        const assistantMessage: ChatMessage = { id: createMessageId(), role: 'assistant', content: accumulated, timestamp: new Date().toISOString() };
        set((state) => ({ messages: [...state.messages, assistantMessage], isStreaming: false, streamingContent: '' }));
        get().persistSession().catch(() => {});
        get().loadSuggestions().catch(() => {});
      },
    );
  },
}));
