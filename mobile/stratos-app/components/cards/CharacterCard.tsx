import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Dimensions } from 'react-native';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import { Star, MessageCircle } from 'lucide-react-native';
import { CharacterCard as CharacterCardType, formatCount } from '../../lib/types';
import { useChatStore } from '../../stores/chatStore';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';
import { getGenreColor } from '../../constants/genres';
const CARD_WIDTH = (Dimensions.get('window').width - spacing.lg * 3) / 2;

interface CharacterCardProps { card: CharacterCardType; variant?: 'grid' | 'horizontal'; }

export function CharacterCardComponent({ card, variant = 'grid' }: CharacterCardProps) {
  const router = useRouter();
  const startSession = useChatStore(s => s.startSession);
  const accentColor = getGenreColor(card.genre_tags[0] ?? 'default');
  const genreLabel = (card.genre_tags[0] ?? '').charAt(0).toUpperCase() + (card.genre_tags[0] ?? '').slice(1);

  const handleQuickChat = () => { startSession(card, 'roleplay'); router.push(`/chat/${card.id}`); };

  if (variant === 'horizontal') {
    return (
      <TouchableOpacity style={styles.horizontalCard} onPress={() => router.push(`/character/${card.id}`)} activeOpacity={0.7}>
        <View style={[styles.horizontalAvatar, { backgroundColor: accentColor + '15', borderColor: accentColor + '25', borderWidth: 1 }]}>
          {card.avatar_url ? (
            <Image source={{ uri: card.avatar_url }} style={styles.horizontalAvatarImage} />
          ) : (
            <>
              <View style={[styles.avatarGlow, { backgroundColor: accentColor, opacity: 0.08 }]} />
              <Text style={[styles.avatarInitial, { color: accentColor }]}>{card.name[0]}</Text>
            </>
          )}
        </View>
        <Text style={styles.horizontalName} numberOfLines={1}>{card.name}</Text>
        <Text style={[styles.horizontalGenre, { color: accentColor }]}>{genreLabel}</Text>
      </TouchableOpacity>
    );
  }

  const starCount = Math.round(card.rating);

  return (
    <TouchableOpacity style={[styles.card, { width: CARD_WIDTH }]} onPress={() => router.push(`/character/${card.id}`)} activeOpacity={0.7}>
      <View style={[styles.avatarContainer, { backgroundColor: accentColor + '08' }]}>
        {card.avatar_url ? (
          <Image source={{ uri: card.avatar_url }} style={styles.avatarImage} />
        ) : (
          <>
            {/* Ambient glow behind letter */}
            <View style={[styles.letterGlow, { backgroundColor: accentColor, shadowColor: accentColor }]} />
            <Text style={[styles.avatarLetter, { color: accentColor }]}>{card.name[0]}</Text>
          </>
        )}
        {/* Quick chat button */}
        <TouchableOpacity style={[styles.quickChatBtn, { backgroundColor: accentColor + 'CC' }]} onPress={handleQuickChat} activeOpacity={0.8} hitSlop={4}>
          <MessageCircle size={12} color="#fff" />
        </TouchableOpacity>
        {/* NSFW badge */}
        {card.content_rating === 'nsfw' && <View style={styles.nsfwBadge}><Text style={styles.nsfwText}>18+</Text></View>}
        {/* Gradient overlay at bottom of avatar */}
        <View style={[styles.avatarGradient, { backgroundColor: colors.bg.secondary }]} />
      </View>
      <View style={styles.info}>
        <Text style={styles.name} numberOfLines={1}>{card.name}</Text>
        <View style={styles.metaRow}>
          <View style={[styles.genrePill, { backgroundColor: accentColor + '18', borderColor: accentColor + '35' }]}>
            <Text style={[styles.genrePillText, { color: accentColor }]}>{genreLabel}</Text>
          </View>
          <View style={styles.ratingRow}>
            {Array.from({ length: 5 }, (_, i) => (
              <Star key={i} size={9} color={i < starCount ? colors.accent.secondary : colors.text.faint} fill={i < starCount ? colors.accent.secondary : 'transparent'} />
            ))}
          </View>
        </View>
        {card.first_message ? (
          <Text style={styles.preview} numberOfLines={2}>{card.first_message.replace(/\*[^*]+\*/g, '').replace(/\n/g, ' ').trim().slice(0, 80)}</Text>
        ) : null}
        <View style={styles.bottomRow}>
          <MessageCircle size={10} color={colors.text.muted} />
          <Text style={styles.sessions}>{formatCount(card.session_count)} chats</Text>
        </View>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: { backgroundColor: colors.bg.secondary, borderRadius: borderRadius.lg, overflow: 'hidden', borderWidth: 1, borderColor: colors.border.subtle },
  avatarContainer: { width: '100%', aspectRatio: 3/4, justifyContent: 'center', alignItems: 'center', position: 'relative' },
  avatarImage: { width: '100%', height: '100%' },
  avatarLetter: { fontSize: 48, fontWeight: '700', opacity: 0.7 },
  letterGlow: {
    position: 'absolute', width: 80, height: 80, borderRadius: 40,
    opacity: 0.12,
    shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.6, shadowRadius: 30, elevation: 3,
  },
  avatarGradient: {
    position: 'absolute', bottom: 0, left: 0, right: 0, height: 30,
    opacity: 0.7,
  },
  quickChatBtn: { position: 'absolute', bottom: spacing.sm + 30, right: spacing.sm, width: 28, height: 28, borderRadius: 14, justifyContent: 'center', alignItems: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 3 },
  nsfwBadge: { position: 'absolute', top: spacing.sm, right: spacing.sm, backgroundColor: colors.nsfw + 'CC', paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: borderRadius.sm },
  nsfwText: { ...typography.small, color: '#fff', fontWeight: '700' },
  info: { padding: spacing.md, gap: 4 },
  name: { ...typography.subheading, color: colors.text.primary, fontSize: 14 },
  metaRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 2 },
  genrePill: { borderWidth: 1, borderRadius: borderRadius.full, paddingHorizontal: spacing.sm, paddingVertical: 1 },
  genrePillText: { fontSize: 10, fontWeight: '600' },
  ratingRow: { flexDirection: 'row', gap: 1 },
  bottomRow: { flexDirection: 'row', alignItems: 'center', gap: 3, marginTop: 2 },
  preview: { ...typography.small, color: colors.text.muted, fontSize: 10, lineHeight: 14, marginTop: 2 },
  sessions: { ...typography.small, color: colors.text.muted, fontSize: 10 },
  avatarInitial: { fontSize: 24, fontWeight: '700', opacity: 0.7 },
  avatarGlow: { position: 'absolute', width: '100%', height: '100%', borderRadius: 16 },
  horizontalCard: { width: 120, alignItems: 'center', marginRight: spacing.md },
  horizontalAvatar: { width: 100, height: 100, borderRadius: borderRadius.lg, justifyContent: 'center', alignItems: 'center', marginBottom: spacing.sm, overflow: 'hidden' },
  horizontalAvatarImage: { width: 100, height: 100, borderRadius: borderRadius.lg },
  horizontalName: { ...typography.caption, color: colors.text.primary, fontWeight: '600', textAlign: 'center' },
  horizontalGenre: { ...typography.small, textTransform: 'capitalize' },
});
