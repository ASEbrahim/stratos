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
    // Keep the old message as fallback in case regenerate fails
    const lastIdx = messages.length - 1;
    if (messages[lastIdx].role !== 'assistant') return;
    const oldMessage = messages[lastIdx];
    const trimmed = messages.slice(0, lastIdx);
    set({ messages: trimmed, isStreaming: true, streamingContent: '', suggestions: [] });

    // Use the same streamMessage function for consistency
    // The backend's /api/rp/chat will see the conversation history
    // WITHOUT the last assistant message (since we just inserted it as turn N
    // and it's the latest), so it generates a fresh response
    const lastUser = trimmed.filter(m => m.role === 'user').pop();
    if (!lastUser) {
      // Restore old message if no user message found
      set({ messages, isStreaming: false });
      return;
    }

    let accumulated = '';
    try {
      await streamMessage(sessionId, lastUser.content, persona, character,
        (chunk) => { accumulated += chunk; set({ streamingContent: accumulated }); },
        () => {
          if (accumulated) {
            const assistantMessage: ChatMessage = { id: createMessageId(), role: 'assistant', content: accumulated, timestamp: new Date().toISOString() };
            set((state) => ({ messages: [...state.messages, assistantMessage], isStreaming: false, streamingContent: '' }));
          } else {
            // Restore old message if generation returned empty
            set({ messages, isStreaming: false, streamingContent: '' });
          }
          get().persistSession().catch(() => {});
          get().loadSuggestions().catch(() => {});
        },
      );
    } catch {
      // Restore old message on error
      set({ messages, isStreaming: false, streamingContent: '' });
    }
  },
}));
