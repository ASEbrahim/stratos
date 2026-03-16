import { create } from 'zustand';
import { ChatMessage, ChatSession, CharacterCard, Suggestion } from '../lib/types';
import { streamMessage, streamRegenerate, getSuggestions, createMessageId } from '../lib/chat';
import { saveChatSession, loadChatSessions, getChatSession, incrementStat } from '../lib/storage';
import { reportError } from '../lib/utils';

interface ChatState {
  sessionId: string | null;
  character: CharacterCard | null;
  persona: 'roleplay' | 'gaming';
  sessionContext: string;  // Persistent context injected on every turn
  messages: ChatMessage[];
  suggestions: Suggestion[];
  isStreaming: boolean;
  streamingContent: string;
  isLoadingSuggestions: boolean;
  recentSessions: ChatSession[];
  startSession: (character: CharacterCard, persona?: 'roleplay' | 'gaming') => void;
  resumeSession: (session: ChatSession) => void;
  setSessionContext: (context: string) => void;
  sendMessage: (content: string, directorNote?: string) => Promise<void>;
  loadSuggestions: () => Promise<void>;
  loadRecentSessions: () => Promise<void>;
  clearSession: () => void;
  persistSession: () => Promise<void>;
  regenerateLastMessage: () => Promise<void>;
}

let _persistTimer: ReturnType<typeof setTimeout> | null = null;

export const useChatStore = create<ChatState>((set, get) => ({
  sessionId: null, character: null, persona: 'roleplay', sessionContext: '', messages: [], suggestions: [],
  isStreaming: false, streamingContent: '', isLoadingSuggestions: false, recentSessions: [],
  startSession: (character, persona = 'roleplay') => {
    const sessionId = `session-${character.id}-${Date.now()}`;
    const messages: ChatMessage[] = [];
    if (character.first_message) {
      messages.push({ id: createMessageId(), role: 'assistant', content: character.first_message, timestamp: new Date().toISOString() });
    }
    set({ sessionId, character, persona, sessionContext: '', messages, suggestions: [], isStreaming: false, streamingContent: '' });
    incrementStat('totalSessions').catch(err => reportError('startSession:incrementStat', err));
  },
  setSessionContext: (context) => set({ sessionContext: context }),
  resumeSession: (session: ChatSession) => {
    set({
      sessionId: session.id,
      character: { id: session.character_id, name: session.character_name, avatar_url: session.character_avatar } as CharacterCard,
      persona: session.persona,
      sessionContext: session.session_context || '',
      messages: session.messages,
      suggestions: [], isStreaming: false, streamingContent: '',
    });
  },
  sendMessage: async (content, directorNote) => {
    const { sessionId, character, persona, sessionContext, messages, isStreaming } = get();
    if (!sessionId || isStreaming) return;
    const userMessage: ChatMessage = { id: createMessageId(), role: 'user', content, timestamp: new Date().toISOString() };
    set({ messages: [...messages, userMessage], isStreaming: true, streamingContent: '', suggestions: [] });
    let accumulated = '';
    await streamMessage(sessionId, content, persona, character,
      (chunk) => { accumulated += chunk; set({ streamingContent: accumulated }); },
      () => {
        const assistantMessage: ChatMessage = { id: createMessageId(), role: 'assistant', content: accumulated, timestamp: new Date().toISOString() };
        set((state) => ({ messages: [...state.messages, assistantMessage], isStreaming: false, streamingContent: '' }));
        incrementStat('totalMessages', 2).catch(err => reportError('sendMessage:incrementStat', err));
        get().persistSession().catch(err => reportError('sendMessage:persistSession', err));
        get().loadSuggestions().catch(err => reportError('sendMessage:loadSuggestions', err));
      },
      directorNote, sessionContext || undefined,
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
    } catch (err) { reportError('loadSuggestions', err); set({ isLoadingSuggestions: false }); }
  },
  loadRecentSessions: async () => {
    try {
      const sessions = await loadChatSessions();
      set({ recentSessions: sessions });
    } catch (err) { reportError('loadRecentSessions', err); }
  },
  clearSession: () => {
    get().persistSession().catch(err => reportError('clearSession:persistSession', err));
    set({ sessionId: null, character: null, messages: [], suggestions: [], isStreaming: false, streamingContent: '' });
  },
  persistSession: () => {
    return new Promise<void>((resolve) => {
      if (_persistTimer) clearTimeout(_persistTimer);
      _persistTimer = setTimeout(async () => {
        _persistTimer = null;
        const { sessionId, character, persona, sessionContext, messages } = get();
        if (!sessionId || !character || messages.length === 0) { resolve(); return; }
        const session: ChatSession = {
          id: sessionId,
          character_id: character.id,
          character_name: character.name,
          character_avatar: character.avatar_url,
          persona,
          session_context: sessionContext || undefined,
          messages,
          created_at: messages[0]?.timestamp ?? new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };
        try {
          await saveChatSession(session);
        } catch (err) {
          reportError('persistSession', err);
        }
        resolve();
      }, 500);
    });
  },
  regenerateLastMessage: async () => {
    const { sessionId, character, persona, messages, isStreaming } = get();
    if (!sessionId || messages.length < 2 || isStreaming) return;
    const lastIdx = messages.length - 1;
    if (messages[lastIdx].role !== 'assistant') return;
    const originalMessages = [...messages]; // Full backup for fallback
    const trimmed = messages.slice(0, lastIdx);
    set({ messages: trimmed, isStreaming: true, streamingContent: '', suggestions: [] });

    let accumulated = '';
    try {
      // Use streamRegenerate — hits /api/rp/regenerate, does NOT create duplicate user messages
      await streamRegenerate(sessionId, persona, character,
        (chunk) => { accumulated += chunk; set({ streamingContent: accumulated }); },
        () => {
          if (accumulated) {
            const assistantMessage: ChatMessage = { id: createMessageId(), role: 'assistant', content: accumulated, timestamp: new Date().toISOString() };
            set((state) => ({ messages: [...state.messages, assistantMessage], isStreaming: false, streamingContent: '' }));
          } else {
            // Restore original if empty response
            set({ messages: originalMessages, isStreaming: false, streamingContent: '' });
          }
          get().persistSession().catch(err => reportError('regenerate:persistSession', err));
          get().loadSuggestions().catch(err => reportError('regenerate:loadSuggestions', err));
        },
      );
    } catch (err) {
      reportError('regenerateLastMessage', err);
      // Restore original on error
      set({ messages: originalMessages, isStreaming: false, streamingContent: '' });
    }
  },
}));
