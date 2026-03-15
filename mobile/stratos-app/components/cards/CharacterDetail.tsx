import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import { Star, BookmarkPlus } from 'lucide-react-native';
import { CharacterCard, formatCount } from '../../lib/types';
import { TagPills } from './TagPills';
import { QualityScore } from './QualityScore';
import { useChatStore } from '../../stores/chatStore';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';
import { getGenreColor } from '../../constants/genres';

interface CharacterDetailProps { card: CharacterCard; }

export function CharacterDetailView({ card }: CharacterDetailProps) {
  const router = useRouter();
  const startSession = useChatStore((s) => s.startSession);
  const accentColor = getGenreColor(card.genre_tags[0] ?? 'default');
  const handleStartChat = () => { startSession(card, 'roleplay'); router.push(`/chat/${card.id}`); };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={[styles.avatarContainer, { backgroundColor: accentColor + '15' }]}>
        {card.avatar_url ? <Image source={{ uri: card.avatar_url }} style={styles.avatar} /> : <Text style={[styles.avatarLetter, { color: accentColor }]}>{card.name[0]}</Text>}
      </View>
      <Text style={styles.name}>{card.name}</Text>
      <Text style={styles.creator}>by @{card.creator_name}</Text>
      <View style={styles.ratingRow}>
        <Star size={14} color={colors.accent.secondary} fill={colors.accent.secondary} />
        <Text style={styles.rating}>{card.rating.toFixed(1)}</Text>
        <Text style={styles.separator}>·</Text>
        <Text style={styles.sessionCount}>{formatCount(card.session_count)} chats</Text>
      </View>
      <View style={styles.section}><TagPills tags={card.genre_tags} size="medium" /></View>
      <View style={styles.section}><Text style={styles.sectionTitle}>Description</Text><Text style={styles.sectionBody}>{card.description}</Text></View>
      {card.personality && <View style={styles.section}><Text style={styles.sectionTitle}>Personality</Text><Text style={styles.sectionBody}>{card.personality}</Text></View>}
      {card.scenario && <View style={styles.section}><Text style={styles.sectionTitle}>Scenario</Text><Text style={styles.sectionBody}>{card.scenario}</Text></View>}
      <View style={styles.section}><Text style={styles.sectionTitle}>Quality Elements</Text><QualityScore card={card} showElements size="large" /></View>
      <TouchableOpacity style={[styles.primaryButton, { backgroundColor: accentColor }]} onPress={handleStartChat} activeOpacity={0.8}>
        <Text style={styles.primaryButtonText}>Start Conversation</Text>
      </TouchableOpacity>
      <TouchableOpacity style={styles.secondaryButton} activeOpacity={0.7}>
        <BookmarkPlus size={18} color={colors.text.secondary} /><Text style={styles.secondaryButtonText}>Save to Library</Text>
      </TouchableOpacity>
      <View style={{ height: spacing.xxl }} />
    </ScrollView>
  );
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
});
