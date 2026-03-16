import React, { useEffect, useState, useCallback } from 'react';
import { View, Text, ScrollView, TouchableOpacity, TextInput, StyleSheet, RefreshControl, Alert } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import { MessageCircle, Clock, Trash2, Search, Archive } from 'lucide-react-native';
import * as Haptics from 'expo-haptics';
import { deleteChatSession } from '../../lib/storage';
import { useChatStore } from '../../stores/chatStore';
import { useThemeStore } from '../../stores/themeStore';
import { EmptyState } from '../../components/shared/EmptyState';
import { ChatSession } from '../../lib/types';
import { getGenreColor } from '../../constants/genres';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';
import Animated, { FadeIn, FadeOut, Layout } from 'react-native-reanimated';

export default function ChatsScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const tc = useThemeStore(s => s.colors);
  const { recentSessions, loadRecentSessions, resumeSession, sessionId: activeSessionId, character: activeCharacter } = useChatStore();
  const [refreshing, setRefreshing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => { loadRecentSessions(); }, []);
  const onRefresh = useCallback(async () => { setRefreshing(true); await loadRecentSessions(); setRefreshing(false); }, []);

  const q = searchQuery.toLowerCase().trim();
  const filtered = q ? recentSessions.filter(s => s.character_name.toLowerCase().includes(q)) : recentSessions;

  // Group by time: Today, Yesterday, Earlier
  const now = new Date();
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const yesterdayStart = todayStart - 86400000;

  const today = filtered.filter(s => new Date(s.updated_at).getTime() >= todayStart);
  const yesterday = filtered.filter(s => { const t = new Date(s.updated_at).getTime(); return t >= yesterdayStart && t < todayStart; });
  const earlier = filtered.filter(s => new Date(s.updated_at).getTime() < yesterdayStart);

  const handleResume = (session: ChatSession) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    resumeSession(session);
    router.push(`/chat/${session.character_id}`);
  };

  const handleDelete = (session: ChatSession) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert('Delete Chat', `Delete conversation with ${session.character_name}?`, [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: async () => {
        try {
          await deleteChatSession(session.id);
          loadRecentSessions();
        } catch { Alert.alert('Error', 'Failed to delete.'); }
      }},
    ]);
  };

  const renderSession = (s: ChatSession) => {
    const lastMsg = s.messages[s.messages.length - 1];
    const preview = lastMsg?.content?.replace(/\*[^*]+\*/g, '').replace(/\n/g, ' ').trim().slice(0, 80) ?? '';
    const isUser = lastMsg?.role === 'user';
    const timeStr = formatTime(s.updated_at);
    const initial = s.character_name[0] ?? '?';

    return (
      <Animated.View key={s.id} entering={FadeIn.duration(200)} exiting={FadeOut.duration(150)} layout={Layout.springify()}>
        <TouchableOpacity
          style={[styles.chatRow, { backgroundColor: tc.bg.secondary }]}
          onPress={() => handleResume(s)}
          onLongPress={() => handleDelete(s)}
          delayLongPress={500}
          activeOpacity={0.7}
          accessibilityLabel={`Chat with ${s.character_name}, ${s.messages.length} messages, ${timeStr}`}
          accessibilityRole="button"
        >
          <View style={[styles.avatar, { backgroundColor: tc.accent.primary + '15' }]}>
            <Text style={[styles.avatarLetter, { color: tc.accent.primary }]}>{initial}</Text>
            <View style={[styles.onlineDot, { backgroundColor: tc.status.success }]} />
          </View>
          <View style={styles.chatInfo}>
            <View style={styles.nameRow}>
              <Text style={[styles.chatName, { color: tc.text.primary }]} numberOfLines={1}>{s.character_name}</Text>
              <Text style={[styles.chatTime, { color: tc.text.muted }]}>{timeStr}</Text>
            </View>
            <View style={styles.previewRow}>
              <Text style={[styles.chatPreview, { color: tc.text.secondary }]} numberOfLines={2}>
                {isUser ? 'You: ' : ''}{preview || 'No messages yet'}
              </Text>
              <View style={[styles.msgBadge, { backgroundColor: tc.accent.primary + '20' }]}>
                <Text style={[styles.msgBadgeText, { color: tc.accent.primary }]}>{s.messages.length}</Text>
              </View>
            </View>
            {s.persona === 'gaming' && (
              <View style={[styles.modeBadge, { backgroundColor: tc.accent.secondary + '15' }]}>
                <Text style={[styles.modeBadgeText, { color: tc.accent.secondary }]}>Gaming</Text>
              </View>
            )}
          </View>
          <TouchableOpacity
            onPress={() => handleDelete(s)}
            style={styles.deleteBtn}
            hitSlop={8}
            accessibilityLabel={`Delete chat with ${s.character_name}`}
            accessibilityRole="button"
          >
            <Trash2 size={14} color={tc.status.error + '60'} />
          </TouchableOpacity>
        </TouchableOpacity>
      </Animated.View>
    );
  };

  const renderSection = (title: string, sessions: ChatSession[]) => {
    if (sessions.length === 0) return null;
    return (
      <View key={title}>
        <Text style={[styles.sectionLabel, { color: tc.text.muted }]}>{title}</Text>
        {sessions.map(renderSession)}
      </View>
    );
  };

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <View style={styles.header}>
        <Text style={[styles.title, { color: tc.text.primary }]}>Chats</Text>
        {recentSessions.length > 0 && (
          <Text style={[styles.chatCount, { color: tc.text.muted }]}>{recentSessions.length}</Text>
        )}
      </View>
      {recentSessions.length > 0 && (
        <View style={[styles.searchBox, { backgroundColor: tc.bg.tertiary }]}>
          <Search size={16} color={tc.text.muted} />
          <TextInput
            style={[styles.searchInput, { color: tc.text.primary }]}
            value={searchQuery}
            onChangeText={setSearchQuery}
            placeholder="Search conversations..."
            placeholderTextColor={tc.text.muted}
            accessibilityLabel="Search conversations"
            accessibilityRole="search"
          />
        </View>
      )}
      {filtered.length === 0 ? (
        recentSessions.length === 0 ? (
          <EmptyState
            title="No conversations yet"
            subtitle="Start chatting with a character from Discover to see your conversations here."
            icon="💬"
          />
        ) : (
          <EmptyState
            title="No matches"
            subtitle={`No conversations match "${searchQuery}"`}
            icon="🔍"
          />
        )
      ) : (
        <ScrollView
          contentContainerStyle={styles.list}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={tc.accent.primary} />}
          keyboardDismissMode="on-drag"
        >
          {/* Pinned / Active session indicator */}
          {activeSessionId && activeCharacter && (
            <TouchableOpacity
              style={[styles.activeSession, { backgroundColor: tc.accent.primary + '10', borderColor: tc.accent.primary + '30' }]}
              onPress={() => router.push(`/chat/${activeCharacter.id}`)}
              activeOpacity={0.7}
              accessibilityLabel={`Continue active chat with ${activeCharacter.name}`}
              accessibilityRole="button"
            >
              <View style={[styles.activeDot, { backgroundColor: tc.status.success }]} />
              <Text style={[styles.activeText, { color: tc.accent.primary }]}>
                Continue with {activeCharacter.name}
              </Text>
              <Text style={[styles.activeArrow, { color: tc.accent.primary }]}>→</Text>
            </TouchableOpacity>
          )}
          {renderSection('Today', today)}
          {renderSection('Yesterday', yesterday)}
          {renderSection('Earlier', earlier)}
          <View style={{ height: spacing.xxl }} />
        </ScrollView>
      )}
    </View>
  );
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMin = Math.floor(diffMs / 60000);

  if (diffMin < 1) return 'now';
  if (diffMin < 60) return `${diffMin}m`;
  const h = d.getHours();
  const m = d.getMinutes().toString().padStart(2, '0');
  const time = `${h % 12 || 12}:${m} ${h >= 12 ? 'PM' : 'AM'}`;

  if (d.toDateString() === now.toDateString()) return time;
  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  header: { flexDirection: 'row', alignItems: 'baseline', paddingHorizontal: spacing.lg, paddingTop: spacing.lg, paddingBottom: spacing.sm, gap: spacing.sm },
  title: { ...typography.display, fontFamily: fonts.logo },
  chatCount: { fontSize: 16, fontFamily: fonts.button },
  searchBox: { flexDirection: 'row', alignItems: 'center', borderRadius: borderRadius.lg, paddingHorizontal: spacing.md, marginHorizontal: spacing.lg, marginBottom: spacing.md, gap: spacing.sm },
  searchInput: { flex: 1, paddingVertical: spacing.sm, fontSize: 14, fontFamily: fonts.body },
  list: { paddingHorizontal: spacing.lg },
  sectionLabel: { fontSize: 11, fontFamily: fonts.heading, textTransform: 'uppercase', letterSpacing: 1.5, marginTop: spacing.lg, marginBottom: spacing.sm, paddingLeft: spacing.xs },
  chatRow: { flexDirection: 'row', alignItems: 'center', padding: spacing.md, borderRadius: borderRadius.lg, marginBottom: spacing.sm, gap: spacing.md },
  avatar: { width: 50, height: 50, borderRadius: 25, justifyContent: 'center', alignItems: 'center', position: 'relative' },
  avatarLetter: { fontSize: 20, fontFamily: fonts.heading },
  onlineDot: { position: 'absolute', bottom: 1, right: 1, width: 10, height: 10, borderRadius: 5, borderWidth: 2, borderColor: '#0a0a0f' },
  chatInfo: { flex: 1 },
  nameRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 3 },
  chatName: { ...typography.subheading, fontSize: 15, fontFamily: fonts.heading, flex: 1, marginRight: spacing.sm },
  chatTime: { fontSize: 11, fontFamily: fonts.body },
  previewRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  chatPreview: { ...typography.caption, flex: 1, lineHeight: 18, fontFamily: fonts.body },
  msgBadge: { minWidth: 22, height: 22, borderRadius: 11, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 5 },
  msgBadgeText: { fontSize: 10, fontWeight: '800' },
  modeBadge: { alignSelf: 'flex-start', paddingHorizontal: spacing.sm, paddingVertical: 1, borderRadius: borderRadius.sm, marginTop: 4 },
  modeBadgeText: { fontSize: 8, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.5 },
  deleteBtn: { padding: spacing.sm, justifyContent: 'center' },
  activeSession: { flexDirection: 'row', alignItems: 'center', padding: spacing.md, borderRadius: borderRadius.lg, borderWidth: 1, marginBottom: spacing.md, gap: spacing.sm },
  activeDot: { width: 8, height: 8, borderRadius: 4 },
  activeText: { flex: 1, fontSize: 13, fontFamily: fonts.button },
  activeArrow: { fontSize: 16, fontWeight: '700' },
});
