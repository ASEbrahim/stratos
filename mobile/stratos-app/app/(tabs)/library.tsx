import React, { useEffect, useState, useCallback } from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet, Alert, RefreshControl } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { MessageCircle, Clock, Trash2 } from 'lucide-react-native';
import * as Haptics from 'expo-haptics';
import { deleteChatSession } from '../../lib/storage';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useCharacterStore } from '../../stores/characterStore';
import { useChatStore } from '../../stores/chatStore';
import { CharacterCardComponent } from '../../components/cards/CharacterCard';
import { EmptyState } from '../../components/shared/EmptyState';
import { ChatSession } from '../../lib/types';
import { useThemeStore } from '../../stores/themeStore';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

export default function LibraryScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const tc = useThemeStore(s => s.colors);
  const { myCards, savedCards, loadMyCards, loadSaved } = useCharacterStore();
  const { recentSessions, loadRecentSessions, resumeSession } = useChatStore();
  const [tab, setTab] = useState<'mine' | 'saved' | 'history'>('mine');
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => { loadMyCards(); loadSaved(); loadRecentSessions(); }, []);
  const onRefresh = useCallback(async () => { setRefreshing(true); await Promise.all([loadMyCards(), loadSaved(), loadRecentSessions()]); setRefreshing(false); }, []);

  const handleResumeSession = (session: ChatSession) => {
    resumeSession(session);
    router.push(`/chat/${session.character_id}`);
  };

  const handleClearAllHistory = () => {
    Alert.alert('Clear All History', `Delete all ${recentSessions.length} conversations?`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Clear All', style: 'destructive', onPress: async () => {
        await AsyncStorage.removeItem('stratos_chat_sessions');
        loadRecentSessions();
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      }},
    ]);
  };

  const handleDeleteSession = (session: ChatSession) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert('Delete Session', `Delete conversation with ${session.character_name}?`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: async () => {
        await deleteChatSession(session.id);
        loadRecentSessions();
      }},
    ]);
  };

  const cards = tab === 'mine' ? myCards : savedCards;

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <Text style={[styles.title, { color: tc.text.primary }]}>Library</Text>
      <View style={styles.tabs}>
        <TouchableOpacity style={[styles.tab, tab === 'mine' && styles.tabActive]} onPress={() => setTab('mine')}>
          <Text style={[styles.tabText, tab === 'mine' && styles.tabTextActive]}>My Characters{myCards.length > 0 ? ` (${myCards.length})` : ''}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.tab, tab === 'saved' && styles.tabActive]} onPress={() => setTab('saved')}>
          <Text style={[styles.tabText, tab === 'saved' && styles.tabTextActive]}>Saved{savedCards.length > 0 ? ` (${savedCards.length})` : ''}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.tab, tab === 'history' && styles.tabActive]} onPress={() => setTab('history')}>
          <Text style={[styles.tabText, tab === 'history' && styles.tabTextActive]}>History{recentSessions.length > 0 ? ` (${recentSessions.length})` : ''}</Text>
        </TouchableOpacity>
      </View>
      {tab === 'history' ? (
        recentSessions.length === 0 ? (
          <EmptyState title="No conversations yet" subtitle="Start a chat from Discover to see it here." />
        ) : (
          <ScrollView contentContainerStyle={styles.sessionList} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={tc.accent.primary} />}>
            <TouchableOpacity style={styles.clearAllBtn} onPress={handleClearAllHistory}>
              <Text style={[styles.clearAllText, { color: colors.status.error }]}>Clear All History</Text>
            </TouchableOpacity>
            {recentSessions.map(s => (
              <TouchableOpacity key={s.id} style={styles.sessionCard} onPress={() => handleResumeSession(s)} onLongPress={() => handleDeleteSession(s)} delayLongPress={500} activeOpacity={0.7}>
                <View style={[styles.sessionAvatar, { backgroundColor: tc.accent.primary + '15' }]}>
                  <MessageCircle size={20} color={tc.accent.primary} />
                </View>
                <View style={styles.sessionInfo}>
                  <Text style={[styles.sessionName, { color: tc.text.primary }]} numberOfLines={1}>{s.character_name}</Text>
                  <Text style={[styles.sessionPreview, { color: tc.text.secondary }]} numberOfLines={1}>
                    {s.messages[s.messages.length - 1]?.content?.replace(/\*[^*]+\*/g, '').slice(0, 60) ?? 'No messages'}
                  </Text>
                  <View style={styles.sessionMeta}>
                    <Clock size={10} color={tc.text.muted} />
                    <Text style={[styles.sessionTime, { color: tc.text.muted }]}>{formatRelativeTime(s.updated_at)}</Text>
                    <Text style={[styles.sessionMsgCount, { color: tc.text.muted }]}>{s.messages.length} msgs</Text>
                  </View>
                </View>
                <TouchableOpacity onPress={() => handleDeleteSession(s)} style={styles.deleteBtn} hitSlop={8}>
                  <Trash2 size={14} color={colors.status.error + '80'} />
                </TouchableOpacity>
              </TouchableOpacity>
            ))}
          </ScrollView>
        )
      ) : (
        cards.length === 0 ? (
          <EmptyState title={tab === 'mine' ? 'No characters yet' : 'No saved characters'} subtitle={tab === 'mine' ? 'Create your first character!' : 'Browse and save from Discover.'} />
        ) : (
          <ScrollView contentContainerStyle={styles.grid} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={tc.accent.primary} />}>{cards.map(c => <CharacterCardComponent key={c.id} card={c} />)}</ScrollView>
        )
      )}
    </View>
  );
}

function formatRelativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg.primary },
  title: { ...typography.display, color: colors.text.primary, paddingHorizontal: spacing.lg, paddingTop: spacing.lg, paddingBottom: spacing.md },
  tabs: { flexDirection: 'row', paddingHorizontal: spacing.lg, gap: spacing.sm, marginBottom: spacing.lg },
  tab: { paddingVertical: spacing.sm, paddingHorizontal: spacing.lg, borderRadius: borderRadius.full, backgroundColor: colors.bg.tertiary },
  tabActive: { backgroundColor: colors.accent.primary + '20' },
  tabText: { ...typography.subheading, fontSize: 14, color: colors.text.muted },
  tabTextActive: { color: colors.accent.primary },
  grid: { flexDirection: 'row', flexWrap: 'wrap', paddingHorizontal: spacing.lg, gap: spacing.lg, paddingBottom: spacing.xxl },
  sessionList: { paddingHorizontal: spacing.lg, gap: spacing.sm, paddingBottom: spacing.xxl },
  sessionCard: { flexDirection: 'row', alignItems: 'center', backgroundColor: colors.bg.secondary, borderRadius: borderRadius.lg, padding: spacing.lg, gap: spacing.md },
  sessionAvatar: { width: 44, height: 44, borderRadius: 22, backgroundColor: colors.accent.primary + '15', justifyContent: 'center', alignItems: 'center' },
  sessionInfo: { flex: 1 },
  sessionName: { ...typography.subheading, color: colors.text.primary, fontSize: 15 },
  sessionPreview: { ...typography.caption, color: colors.text.secondary, marginTop: 2 },
  sessionMeta: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs, marginTop: spacing.xs },
  sessionTime: { ...typography.small, color: colors.text.muted },
  sessionMsgCount: { ...typography.small, color: colors.text.muted, marginLeft: spacing.sm },
  deleteBtn: { padding: spacing.sm, justifyContent: 'center' },
  clearAllBtn: { alignSelf: 'flex-end', paddingVertical: spacing.xs, paddingHorizontal: spacing.sm, marginBottom: spacing.sm },
  clearAllText: { fontSize: 11, fontWeight: '600' },
});
