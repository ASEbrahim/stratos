import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Dimensions } from 'react-native';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import { CharacterCard as CharacterCardType, formatCount } from '../../lib/types';
import { TagPills } from './TagPills';
import { QualityScore } from './QualityScore';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';
import { getGenreColor } from '../../constants/genres';

const CARD_WIDTH = (Dimensions.get('window').width - spacing.lg * 3) / 2;

interface CharacterCardProps { card: CharacterCardType; variant?: 'grid' | 'horizontal'; }

export function CharacterCardComponent({ card, variant = 'grid' }: CharacterCardProps) {
  const router = useRouter();
  const accentColor = getGenreColor(card.genre_tags[0] ?? 'default');

  if (variant === 'horizontal') {
    return (
      <TouchableOpacity style={styles.horizontalCard} onPress={() => router.push(`/character/${card.id}`)} activeOpacity={0.7}>
        <View style={[styles.horizontalAvatar, { backgroundColor: accentColor + '20' }]}>
          {card.avatar_url ? <Image source={{ uri: card.avatar_url }} style={styles.horizontalAvatarImage} /> : <Text style={[styles.avatarInitial, { color: accentColor }]}>{card.name[0]}</Text>}
        </View>
        <Text style={styles.horizontalName} numberOfLines={1}>{card.name}</Text>
        <Text style={[styles.horizontalGenre, { color: accentColor }]}>{card.genre_tags[0] ?? ''}</Text>
      </TouchableOpacity>
    );
  }

  return (
    <TouchableOpacity style={[styles.card, { width: CARD_WIDTH }]} onPress={() => router.push(`/character/${card.id}`)} activeOpacity={0.7}>
      <View style={[styles.avatarContainer, { backgroundColor: accentColor + '15' }]}>
        {card.avatar_url ? <Image source={{ uri: card.avatar_url }} style={styles.avatarImage} /> : <Text style={[styles.avatarLetter, { color: accentColor }]}>{card.name[0]}</Text>}
        {card.content_rating === 'nsfw' && <View style={styles.nsfwBadge}><Text style={styles.nsfwText}>18+</Text></View>}
      </View>
      <View style={styles.info}>
        <Text style={styles.name} numberOfLines={1}>{card.name}</Text>
        <View style={styles.meta}><TagPills tags={card.genre_tags.slice(0, 1)} /><QualityScore card={card} /></View>
        <Text style={styles.sessions}>{formatCount(card.session_count)} chats</Text>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: { backgroundColor: colors.bg.secondary, borderRadius: borderRadius.lg, overflow: 'hidden', borderWidth: 1, borderColor: colors.border.subtle },
  avatarContainer: { width: '100%', aspectRatio: 3/4, justifyContent: 'center', alignItems: 'center' },
  avatarImage: { width: '100%', height: '100%' },
  avatarLetter: { fontSize: 48, fontWeight: '700', opacity: 0.6 },
  avatarInitial: { fontSize: 24, fontWeight: '700', opacity: 0.6 },
  nsfwBadge: { position: 'absolute', top: spacing.sm, right: spacing.sm, backgroundColor: colors.nsfw + 'CC', paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: borderRadius.sm },
  nsfwText: { ...typography.small, color: '#fff', fontWeight: '700' },
  info: { padding: spacing.md, gap: spacing.xs },
  name: { ...typography.subheading, color: colors.text.primary },
  meta: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  sessions: { ...typography.small, color: colors.text.muted },
  horizontalCard: { width: 120, alignItems: 'center', marginRight: spacing.md },
  horizontalAvatar: { width: 100, height: 100, borderRadius: borderRadius.lg, justifyContent: 'center', alignItems: 'center', marginBottom: spacing.sm },
  horizontalAvatarImage: { width: 100, height: 100, borderRadius: borderRadius.lg },
  horizontalName: { ...typography.caption, color: colors.text.primary, fontWeight: '600', textAlign: 'center' },
  horizontalGenre: { ...typography.small, textTransform: 'capitalize' },
});
