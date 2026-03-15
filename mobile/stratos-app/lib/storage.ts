import AsyncStorage from '@react-native-async-storage/async-storage';
import { ChatSession, CharacterCard } from './types';

const SESSIONS_KEY = 'stratos_chat_sessions';
const SAVED_CARDS_KEY = 'stratos_saved_cards';
const STATS_KEY = 'stratos_user_stats';

// ─── Chat Session Persistence ───

export async function saveChatSession(session: ChatSession): Promise<void> {
  const sessions = await loadChatSessions();
  const idx = sessions.findIndex(s => s.id === session.id);
  if (idx >= 0) sessions[idx] = session;
  else sessions.unshift(session);
  // Keep max 50 sessions
  const trimmed = sessions.slice(0, 50);
  await AsyncStorage.setItem(SESSIONS_KEY, JSON.stringify(trimmed));
}

export async function loadChatSessions(): Promise<ChatSession[]> {
  const raw = await AsyncStorage.getItem(SESSIONS_KEY);
  if (!raw) return [];
  try { return JSON.parse(raw); } catch { return []; }
}

export async function deleteChatSession(sessionId: string): Promise<void> {
  const sessions = await loadChatSessions();
  const filtered = sessions.filter(s => s.id !== sessionId);
  await AsyncStorage.setItem(SESSIONS_KEY, JSON.stringify(filtered));
}

export async function getChatSession(sessionId: string): Promise<ChatSession | null> {
  const sessions = await loadChatSessions();
  return sessions.find(s => s.id === sessionId) ?? null;
}

// ─── Saved Cards ───

export async function saveCard(card: CharacterCard): Promise<void> {
  const cards = await getSavedCards();
  if (cards.some(c => c.id === card.id)) return; // Already saved
  cards.unshift(card);
  await AsyncStorage.setItem(SAVED_CARDS_KEY, JSON.stringify(cards));
}

export async function removeSavedCard(cardId: string): Promise<void> {
  const cards = await getSavedCards();
  const filtered = cards.filter(c => c.id !== cardId);
  await AsyncStorage.setItem(SAVED_CARDS_KEY, JSON.stringify(filtered));
}

export async function getSavedCards(): Promise<CharacterCard[]> {
  const raw = await AsyncStorage.getItem(SAVED_CARDS_KEY);
  if (!raw) return [];
  try { return JSON.parse(raw); } catch { return []; }
}

export async function isCardSaved(cardId: string): Promise<boolean> {
  const cards = await getSavedCards();
  return cards.some(c => c.id === cardId);
}

// ─── User Stats ───

interface UserStats {
  totalSessions: number;
  totalMessages: number;
  totalCharacters: number;
}

export async function getUserStats(): Promise<UserStats> {
  const raw = await AsyncStorage.getItem(STATS_KEY);
  if (!raw) return { totalSessions: 0, totalMessages: 0, totalCharacters: 0 };
  try { return JSON.parse(raw); } catch { return { totalSessions: 0, totalMessages: 0, totalCharacters: 0 }; }
}

export async function incrementStat(key: keyof UserStats, amount = 1): Promise<void> {
  const stats = await getUserStats();
  stats[key] += amount;
  await AsyncStorage.setItem(STATS_KEY, JSON.stringify(stats));
}
