import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { Star, BookmarkPlus, BookmarkCheck, Flag } from 'lucide-react-native';
import { CharacterCard, formatCount } from '../../lib/types';
import { TagPills } from './TagPills';
import { QualityScore } from './QualityScore';
import { useChatStore } from '../../stores/chatStore';
import { useCharacterStore } from '../../stores/characterStore';
import { isCardSaved, loadChatSessions } from '../../lib/storage';
import { ChatSession } from '../../lib/types';
import { useThemeStore } from '../../stores/themeStore';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';
import { getGenreColor } from '../../constants/genres';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, withSequence } from 'react-native-reanimated';

interface CharacterDetailProps { card: CharacterCard; }

export function CharacterDetailView({ card }: CharacterDetailProps) {
  const router = useRouter();
  const tc = useThemeStore(s => s.colors);
  const { startSession, resumeSession } = useChatStore();
  const { saveToLibrary, removeFromLibrary } = useCharacterStore();
  const [saved, setSaved] = useState(false);
  const [existingSession, setExistingSession] = useState<ChatSession | null>(null);
  const accentColor = getGenreColor(card.genre_tags[0] ?? 'default');
  const btnScale = useSharedValue(1);
  const btnAnimStyle = useAnimatedStyle(() => ({ transform: [{ scale: btnScale.value }] }));

  useEffect(() => {
    isCardSaved(card.id).then(setSaved);
    loadChatSessions().then(sessions => {
      const match = sessions.find(s => s.character_id === card.id);
      if (match) setExistingSession(match);
    });
  }, [card.id]);

  const handleStartChat = () => {
    btnScale.value = withSequence(withSpring(0.95, { damping: 15 }), withSpring(1, { damping: 10 }));
    startSession(card, 'roleplay');
    router.push(`/chat/${card.id}`);
  };

  const handleContinueChat = () => {
    if (!existingSession) return;
    btnScale.value = withSequence(withSpring(0.95, { damping: 15 }), withSpring(1, { damping: 10 }));
    resumeSession(existingSession);
    router.push(`/chat/${card.id}`);
  };

  const handleToggleSave = async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    if (saved) {
      await removeFromLibrary(card.id);
      setSaved(false);
    } else {
      await saveToLibrary(card);
      setSaved(true);
    }
  };

  return (
    <ScrollView style={[styles.container, { backgroundColor: tc.bg.primary }]} contentContainerStyle={styles.content}>
      <View style={[styles.avatarContainer, { backgroundColor: accentColor + '15' }]}>
        {card.avatar_url ? <Image source={{ uri: card.avatar_url }} style={styles.avatar} /> : <Text style={[styles.avatarLetter, { color: accentColor }]}>{card.name[0]}</Text>}
      </View>
      <Text style={[styles.name, { color: tc.text.primary }]}>{card.name}</Text>
      <Text style={[styles.creator, { color: tc.text.secondary }]}>by @{card.creator_name}</Text>
      <View style={styles.ratingRow}>
        <Star size={14} color={tc.accent.secondary} fill={tc.accent.secondary} />
        <Text style={[styles.rating, { color: tc.text.primary }]}>{card.rating.toFixed(1)}</Text>
        <Text style={[styles.separator, { color: tc.text.muted }]}>·</Text>
        <Text style={[styles.sessionCount, { color: tc.text.secondary }]}>{formatCount(card.session_count)} chats</Text>
      </View>
      <View style={styles.section}><TagPills tags={card.genre_tags} size="medium" /></View>
      <View style={styles.section}><Text style={[styles.sectionTitle, { color: tc.text.primary }]}>Description</Text><Text style={[styles.sectionBody, { color: tc.text.secondary }]}>{card.description}</Text></View>
      {card.personality && <View style={styles.section}><Text style={[styles.sectionTitle, { color: tc.text.primary }]}>Personality</Text><Text style={[styles.sectionBody, { color: tc.text.secondary }]}>{card.personality}</Text></View>}
      {card.scenario && <View style={styles.section}><Text style={[styles.sectionTitle, { color: tc.text.primary }]}>Scenario</Text><Text style={[styles.sectionBody, { color: tc.text.secondary }]}>{card.scenario}</Text></View>}
      <View style={styles.section}><Text style={[styles.sectionTitle, { color: tc.text.primary }]}>Quality Elements</Text><QualityScore card={card} showElements size="large" /></View>
      <Animated.View style={btnAnimStyle}>
        {existingSession ? (
          <>
            <TouchableOpacity style={[styles.primaryButton, { backgroundColor: accentColor }]} onPress={handleContinueChat} activeOpacity={0.8}>
              <Text style={styles.primaryButtonText}>Continue Conversation</Text>
            </TouchableOpacity>
            <Text style={styles.sessionHint}>{existingSession.messages.length} messages · last active {formatRelativeTime(existingSession.updated_at)}</Text>
            <TouchableOpacity style={[styles.newSessionBtn, { borderColor: accentColor + '40' }]} onPress={handleStartChat} activeOpacity={0.7}>
              <Text style={[styles.newSessionText, { color: accentColor }]}>Start New Session</Text>
            </TouchableOpacity>
          </>
        ) : (
          <TouchableOpacity style={[styles.primaryButton, { backgroundColor: accentColor }]} onPress={handleStartChat} activeOpacity={0.8}>
            <Text style={styles.primaryButtonText}>Start Conversation</Text>
          </TouchableOpacity>
        )}
      </Animated.View>
      <TouchableOpacity style={[styles.secondaryButton, saved && { borderColor: colors.status.success + '60', backgroundColor: colors.status.success + '10' }]} onPress={handleToggleSave} activeOpacity={0.7}>
        {saved ? <BookmarkCheck size={18} color={colors.status.success} /> : <BookmarkPlus size={18} color={colors.text.secondary} />}
        <Text style={[styles.secondaryButtonText, saved && { color: colors.status.success }]}>{saved ? 'Saved' : 'Save to Library'}</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.reportBtn} onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); Alert.alert('Report Character', 'Report this character for inappropriate content?', [{ text: 'Cancel', style: 'cancel' }, { text: 'Report', style: 'destructive', onPress: () => Alert.alert('Reported', 'Thank you for your report. We will review this character.') }]); }} activeOpacity={0.7}>
        <Flag size={12} color={colors.text.muted} />
        <Text style={styles.reportText}>Report Character</Text>
      </TouchableOpacity>
      <View style={{ height: spacing.xxl }} />
    </ScrollView>
  );
}

function formatRelativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg.primary },
  content: { padding: spacing.lg },
  avatarContainer: { width: '100%', aspectRatio: 1, borderRadius: borderRadius.xl, justifyContent: 'center', alignItems: 'center', marginBottom: spacing.lg, overflow: 'hidden' },
  avatar: { width: '100%', height: '100%' },
  avatarLetter: { fontSize: 80, fontWeight: '700', opacity: 0.5 },
  name: { ...typography.display, color: colors.text.primary, marginBottom: spacing.xs },
  creator: { ...typography.body, color: colors.text.secondary, marginBottom: spacing.sm },
  ratingRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs, marginBottom: spacing.lg },
  rating: { ...typography.body, color: colors.text.primary, fontWeight: '600' },
  separator: { ...typography.body, color: colors.text.muted },
  sessionCount: { ...typography.body, color: colors.text.secondary },
  section: { marginBottom: spacing.xl },
  sectionTitle: { ...typography.subheading, color: colors.text.primary, marginBottom: spacing.sm },
  sectionBody: { ...typography.body, color: colors.text.secondary, lineHeight: 24 },
  primaryButton: { paddingVertical: spacing.lg, borderRadius: borderRadius.lg, alignItems: 'center', marginBottom: spacing.md },
  primaryButtonText: { ...typography.subheading, color: '#fff' },
  secondaryButton: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.lg, borderRadius: borderRadius.lg, borderWidth: 1, borderColor: colors.border.medium },
  secondaryButtonText: { ...typography.subheading, color: colors.text.secondary },
  sessionHint: { ...typography.small, color: colors.text.muted, textAlign: 'center', marginTop: spacing.xs, marginBottom: spacing.sm },
  newSessionBtn: { paddingVertical: spacing.md, borderRadius: borderRadius.lg, borderWidth: 1, alignItems: 'center', marginBottom: spacing.md },
  newSessionText: { ...typography.caption, fontWeight: '600' },
  reportBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.xs, paddingVertical: spacing.lg, marginTop: spacing.lg },
  reportText: { ...typography.small, color: colors.text.muted },
});
