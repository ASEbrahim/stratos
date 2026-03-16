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
    const lastIdx = messages.length - 1;
    if (messages[lastIdx].role !== 'assistant') return;
    const trimmed = messages.slice(0, lastIdx);
    set({ messages: trimmed, isStreaming: true, streamingContent: '', suggestions: [] });

    // Stream from the regenerate (swipe) endpoint directly
    // It deactivates the old response and generates a fresh one — no duplicate user messages
    try {
      const { getToken, getDeviceId } = await import('../lib/api');
      const { API_BASE } = await import('../constants/config');
      const token = await getToken();
      const deviceId = await getDeviceId();
      const response = await fetch(`${API_BASE}/api/rp/regenerate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Device-Id': deviceId,
          ...(token ? { 'X-Auth-Token': token } : {}),
        },
        body: JSON.stringify({
          session_id: sessionId,
          branch_id: 'main',
          character_card_id: character?.id,
          persona,
        }),
      });

      let accumulated = '';
      const reader = response.body?.getReader();
      if (reader) {
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() ?? '';
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.content || data.token) {
                  accumulated += data.content || data.token;
                  set({ streamingContent: accumulated });
                }
                if (data.done) break;
              } catch { /* partial JSON */ }
            }
          }
        }
      }

      if (accumulated) {
        const assistantMessage: ChatMessage = { id: createMessageId(), role: 'assistant', content: accumulated, timestamp: new Date().toISOString() };
        set((state) => ({ messages: [...state.messages, assistantMessage], isStreaming: false, streamingContent: '' }));
      } else {
        set({ isStreaming: false, streamingContent: '' });
      }
      get().persistSession().catch(() => {});
      get().loadSuggestions().catch(() => {});
    } catch {
      set({ isStreaming: false, streamingContent: '' });
    }
  },
}));
