import React, { useEffect, useState, useCallback } from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet, Alert, RefreshControl, TextInput, Dimensions } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { MessageCircle, Clock, Trash2, Search } from 'lucide-react-native';
import * as Haptics from 'expo-haptics';
import { deleteChatSession } from '../../lib/storage';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useCharacterStore } from '../../stores/characterStore';
import { reportError } from '../../lib/utils';
import { useChatStore } from '../../stores/chatStore';
import { CharacterCardComponent } from '../../components/cards/CharacterCard';
import { EmptyState } from '../../components/shared/EmptyState';
import { ChatSession, CharacterCard } from '../../lib/types';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';
import Animated, { FadeOut, Layout } from 'react-native-reanimated';

export default function LibraryScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const tc = useThemeStore(s => s.colors);
  const { myCards, savedCards, loadMyCards, loadSaved, deleteMyCard } = useCharacterStore();
  const { recentSessions, loadRecentSessions, resumeSession } = useChatStore();
  const [tab, setTab] = useState<'mine' | 'saved' | 'history'>('mine');
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => { loadMyCards(); loadSaved(); }, []);
  const onRefresh = useCallback(async () => { setRefreshing(true); await Promise.all([loadMyCards(), loadSaved(), loadRecentSessions()]); setRefreshing(false); }, []);

  const handleResumeSession = (session: ChatSession) => {
    resumeSession(session);
    router.push(`/chat/${session.character_id}`);
  };

  const handleClearAllHistory = () => {
    Alert.alert('Clear All History', `Delete all ${recentSessions.length} conversations?`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Clear All', style: 'destructive', onPress: async () => {
        try {
          await AsyncStorage.removeItem('stratos_chat_sessions');
          loadRecentSessions();
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        } catch (err) { reportError('LibraryScreen:clearAllHistory', err); Alert.alert('Error', 'Failed to clear history.'); }
      }},
    ]);
  };

  const handleDeleteSession = (session: ChatSession) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert('Delete Session', `Delete conversation with ${session.character_name}?`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: async () => {
        try {
          await deleteChatSession(session.id);
          loadRecentSessions();
        } catch (err) { reportError('LibraryScreen:deleteSession', err); Alert.alert('Error', 'Failed to delete session.'); }
      }},
    ]);
  };

  const handleDeleteCard = (card: CharacterCard) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert('Delete Character', `Delete "${card.name}"? This cannot be undone.`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: async () => {
        try {
          await deleteMyCard(card.id);
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        } catch (err) { reportError('LibraryScreen:deleteCard', err); Alert.alert('Error', 'Failed to delete character.'); }
      }},
    ]);
  };

  const rawCards = tab === 'mine' ? myCards : savedCards;
  const q = searchQuery.toLowerCase().trim();
  const cards = q ? rawCards.filter(c => c.name.toLowerCase().includes(q) || c.description.toLowerCase().includes(q)) : rawCards;
  const filteredSessions = q ? recentSessions.filter(s => s.character_name.toLowerCase().includes(q)) : recentSessions;

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <Text style={[styles.title, { color: tc.text.primary }]}>Library</Text>
      <View style={[styles.searchBox, { backgroundColor: tc.bg.tertiary }]}>
        <Search size={16} color={tc.text.muted} />
        <TextInput style={[styles.searchInput, { color: tc.text.primary }]} value={searchQuery} onChangeText={setSearchQuery} placeholder="Search library..." placeholderTextColor={tc.text.muted} accessibilityLabel="Search library" accessibilityRole="search" />
      </View>
      <View style={styles.tabs}>
        {(['mine', 'saved', 'history'] as const).map(t => {
          const labels = { mine: 'My Characters', saved: 'Saved', history: 'History' };
          const counts = { mine: myCards.length, saved: savedCards.length, history: recentSessions.length };
          const active = tab === t;
          return (
            <TouchableOpacity key={t} style={[styles.tab, { backgroundColor: tc.bg.tertiary }, active && { backgroundColor: tc.accent.primary + '20' }]} onPress={() => { Haptics.selectionAsync(); setTab(t); }} accessibilityRole="tab" accessibilityState={{ selected: active }}>
              <Text style={[styles.tabText, { color: tc.text.muted }, active && { color: tc.accent.primary }]}>{labels[t]}{counts[t] > 0 ? ` ${counts[t]}` : ''}</Text>
            </TouchableOpacity>
          );
        })}
      </View>
      {tab === 'history' ? (
        filteredSessions.length === 0 ? (
          <EmptyState title="No conversations yet" subtitle="Start a chat from Discover to see it here." />
        ) : (
          <ScrollView contentContainerStyle={styles.sessionList} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={tc.accent.primary} />}>
            <TouchableOpacity style={styles.clearAllBtn} onPress={handleClearAllHistory} accessibilityLabel="Clear all chat history" accessibilityRole="button">
              <Text style={[styles.clearAllText, { color: tc.status.error }]}>Clear All History</Text>
            </TouchableOpacity>
            {filteredSessions.map(s => (
              <Animated.View key={s.id} exiting={FadeOut.duration(200)} layout={Layout.springify()}>
                <TouchableOpacity style={[styles.sessionCard, { backgroundColor: tc.bg.secondary }]} onPress={() => handleResumeSession(s)} onLongPress={() => handleDeleteSession(s)} delayLongPress={500} activeOpacity={0.7} accessibilityLabel={`Resume chat with ${s.character_name}, ${s.messages.length} messages`} accessibilityRole="button">
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
                      <Text style={[styles.sessionMsgCount, { color: tc.text.muted }]}>{s.messages.length} msgs · ~{Math.max(1, Math.round(s.messages.length * 0.8))}min</Text>
                    </View>
                  </View>
                  <TouchableOpacity onPress={() => handleDeleteSession(s)} style={styles.deleteBtn} hitSlop={8} accessibilityLabel={`Delete chat with ${s.character_name}`} accessibilityRole="button">
                    <Trash2 size={14} color={tc.status.error + '80'} />
                  </TouchableOpacity>
                </TouchableOpacity>
              </Animated.View>
            ))}
          </ScrollView>
        )
      ) : (
        cards.length === 0 ? (
          <EmptyState title={tab === 'mine' ? 'No characters yet' : 'No saved characters'} subtitle={tab === 'mine' ? 'Create your first character!' : 'Browse and save from Discover.'} />
        ) : (
          <ScrollView contentContainerStyle={styles.grid} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={tc.accent.primary} />}>{cards.map(c => (
            <View key={c.id} style={{ position: 'relative' }}>
              <CharacterCardComponent card={c} />
              {tab === 'mine' && (
                <TouchableOpacity
                  style={{ position: 'absolute', bottom: 6, right: 6, padding: 4, backgroundColor: 'rgba(0,0,0,0.6)', borderRadius: 12, zIndex: 10 }}
                  onPress={() => handleDeleteCard(c)}
                  hitSlop={8}
                  accessibilityLabel={`Delete ${c.name}`}
                  accessibilityRole="button"
                >
                  <Trash2 size={12} color="#f87171" />
                </TouchableOpacity>
              )}
            </View>
          ))}</ScrollView>
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
  container: { flex: 1 },
  title: { ...typography.display, fontFamily: fonts.logo, paddingHorizontal: spacing.lg, paddingTop: spacing.lg, paddingBottom: spacing.sm },
  searchBox: { flexDirection: 'row', alignItems: 'center', borderRadius: borderRadius.lg, paddingHorizontal: spacing.md, marginHorizontal: spacing.lg, marginBottom: spacing.md, gap: spacing.sm },
  searchInput: { flex: 1, paddingVertical: spacing.sm, fontSize: 14, fontFamily: fonts.body },
  tabs: { flexDirection: 'row', paddingHorizontal: spacing.lg, gap: spacing.sm, marginBottom: spacing.md },
  tab: { flex: 1, paddingVertical: 6, alignItems: 'center', borderRadius: borderRadius.lg },
  tabText: { fontSize: 13, fontFamily: fonts.button },
  grid: { flexDirection: 'row', flexWrap: 'wrap', paddingHorizontal: spacing.lg, gap: spacing.lg, paddingBottom: spacing.xxl },
  sessionList: { paddingHorizontal: spacing.lg, gap: spacing.sm, paddingBottom: spacing.xxl },
  sessionCard: { flexDirection: 'row', alignItems: 'center', borderRadius: borderRadius.lg, padding: spacing.lg, gap: spacing.md },
  sessionAvatar: { width: 44, height: 44, borderRadius: 22, justifyContent: 'center', alignItems: 'center' },
  sessionInfo: { flex: 1 },
  sessionName: { ...typography.subheading, fontSize: 15, fontFamily: fonts.heading },
  sessionPreview: { ...typography.caption, marginTop: 2, fontFamily: fonts.body },
  sessionMeta: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs, marginTop: spacing.xs },
  sessionTime: { ...typography.small },
  sessionMsgCount: { ...typography.small, marginLeft: spacing.sm },
  deleteBtn: { padding: spacing.sm, justifyContent: 'center' },
  clearAllBtn: { alignSelf: 'flex-end', paddingVertical: spacing.xs, paddingHorizontal: spacing.sm, marginBottom: spacing.sm },
  clearAllText: { fontSize: 11, fontFamily: fonts.button },
});
