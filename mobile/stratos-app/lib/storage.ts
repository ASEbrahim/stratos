import AsyncStorage from '@react-native-async-storage/async-storage';
import { ChatSession, CharacterCard } from './types';
import { Config } from '../constants/config';
import { reportError } from './utils';

const SESSIONS_KEY = 'stratos_chat_sessions';
const SAVED_CARDS_KEY = 'stratos_saved_cards';
const STATS_KEY = 'stratos_user_stats';

// ─── Chat Session Persistence ───

export async function saveChatSession(session: ChatSession): Promise<void> {
  const sessions = await loadChatSessions();
  const idx = sessions.findIndex(s => s.id === session.id);
  if (idx >= 0) sessions[idx] = session;
  else sessions.unshift(session);
  // Keep max sessions
  const trimmed = sessions.slice(0, Config.MAX_CHAT_SESSIONS);
  await AsyncStorage.setItem(SESSIONS_KEY, JSON.stringify(trimmed));
}

export async function loadChatSessions(): Promise<ChatSession[]> {
  const raw = await AsyncStorage.getItem(SESSIONS_KEY);
  if (!raw) return [];
  try { return JSON.parse(raw); } catch (err) { reportError('loadChatSessions:parse', err); return []; }
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
  try { return JSON.parse(raw); } catch (err) { reportError('getSavedCards:parse', err); return []; }
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
  try { return JSON.parse(raw); } catch (err) { reportError('getUserStats:parse', err); return { totalSessions: 0, totalMessages: 0, totalCharacters: 0 }; }
}

export async function incrementStat(key: keyof UserStats, amount = 1): Promise<void> {
  const stats = await getUserStats();
  stats[key] += amount;
  await AsyncStorage.setItem(STATS_KEY, JSON.stringify(stats));
}

export interface DetailedStats {
  totalSessions: number;
  totalMessages: number;
  totalWords: number;
  avgSessionLength: number;
  favoriteGenre: string;
  longestSession: number;
  totalCharacters: number;
}

export async function getDetailedStats(): Promise<DetailedStats> {
  const sessions = await loadChatSessions();
  const cards = await getSavedCards();
  let totalMessages = 0;
  let totalWords = 0;
  let longestSession = 0;
  const genreCounts: Record<string, number> = {};

  for (const s of sessions) {
    const userMsgs = s.messages.filter(m => m.role === 'user');
    totalMessages += s.messages.length;
    for (const m of userMsgs) totalWords += m.content.trim().split(/\s+/).length;
    if (s.messages.length > longestSession) longestSession = s.messages.length;
    // Track genre from character name — we don't have genre data in sessions
  }

  // Try to match session characters to saved cards for genre data
  const allCards = cards;
  for (const s of sessions) {
    const card = allCards.find(c => c.id === s.character_id);
    if (card?.genre_tags?.[0]) {
      genreCounts[card.genre_tags[0]] = (genreCounts[card.genre_tags[0]] || 0) + 1;
    }
  }

  const favoriteGenre = Object.entries(genreCounts).sort(([, a], [, b]) => b - a)[0]?.[0] ?? 'None yet';
  const avgSessionLength = sessions.length > 0 ? Math.round(totalMessages / sessions.length) : 0;

  return {
    totalSessions: sessions.length,
    totalMessages,
    totalWords,
    avgSessionLength,
    favoriteGenre: favoriteGenre.charAt(0).toUpperCase() + favoriteGenre.slice(1),
    longestSession,
    totalCharacters: cards.length,
  };
}
