import React, { useRef, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Dimensions, Pressable } from 'react-native';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { Star, MessageCircle, Heart } from 'lucide-react-native';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, withSequence, withTiming, FadeIn, FadeOut } from 'react-native-reanimated';
import { CharacterCard as CharacterCardType, formatCount } from '../../lib/types';
import { useChatStore } from '../../stores/chatStore';
import { useCharacterStore } from '../../stores/characterStore';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { getGenreColor } from '../../constants/genres';
import { useThemeStore } from '../../stores/themeStore';
const CARD_WIDTH = (Dimensions.get('window').width - spacing.lg * 3) / 2;

interface CharacterCardProps { card: CharacterCardType; variant?: 'grid' | 'horizontal'; featured?: boolean; }

function isNew(dateStr: string): boolean {
  return Date.now() - new Date(dateStr).getTime() < 30 * 24 * 60 * 60 * 1000;
}

export function CharacterCardComponent({ card, variant = 'grid', featured = false }: CharacterCardProps) {
  const router = useRouter();
  const startSession = useChatStore(s => s.startSession);
  const tc = useThemeStore(s => s.colors);
  const accentColor = getGenreColor(card.genre_tags[0] ?? 'default');
  const genreLabel = (card.genre_tags[0] ?? '').charAt(0).toUpperCase() + (card.genre_tags[0] ?? '').slice(1);
  const showNew = isNew(card.created_at);

  const handleQuickChat = () => { startSession(card, 'roleplay'); router.push(`/chat/${card.id}`); };

  if (variant === 'horizontal') {
    return (
      <TouchableOpacity style={styles.horizontalCard} onPress={() => router.push(`/character/${card.id}`)} activeOpacity={0.7} accessibilityLabel={`View ${card.name}, ${genreLabel} character`} accessibilityRole="button">
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
        <Text style={[styles.horizontalName, { color: tc.text.primary }]} numberOfLines={1}>{card.name}</Text>
        <Text style={[styles.horizontalGenre, { color: accentColor }]}>{genreLabel}</Text>
      </TouchableOpacity>
    );
  }

  const starCount = Math.round(card.rating);
  const cardScale = useSharedValue(1);
  const cardAnimStyle = useAnimatedStyle(() => ({ transform: [{ scale: cardScale.value }] }));
  const lastTap = useRef(0);
  const [showHeart, setShowHeart] = useState(false);
  const heartScale = useSharedValue(0);
  const heartStyle = useAnimatedStyle(() => ({ transform: [{ scale: heartScale.value }], opacity: heartScale.value > 0.1 ? 1 : 0 }));
  const { saveToLibrary } = useCharacterStore();

  const handlePress = () => {
    const now = Date.now();
    if (now - lastTap.current < 300) {
      // Double tap — save/bookmark
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      saveToLibrary(card);
      setShowHeart(true);
      heartScale.value = withSequence(withSpring(1.2, { damping: 8 }), withTiming(1, { duration: 100 }), withTiming(0, { duration: 400 }));
      setTimeout(() => setShowHeart(false), 800);
    } else {
      // Single tap — navigate
      setTimeout(() => {
        if (Date.now() - lastTap.current >= 300) router.push(`/character/${card.id}`);
      }, 300);
    }
    lastTap.current = now;
  };

  return (
    <Animated.View style={[{ width: CARD_WIDTH }, cardAnimStyle]}>
    <Pressable
      style={[styles.card, { backgroundColor: tc.bg.secondary, borderColor: tc.border.subtle }]}
      onPress={handlePress}
      onPressIn={() => { cardScale.value = withSpring(0.96, { damping: 15 }); }}
      onPressOut={() => { cardScale.value = withSpring(1, { damping: 10 }); }}
      accessibilityLabel={`${card.name}, ${genreLabel} character, ${card.rating.toFixed(1)} stars, ${formatCount(card.session_count)} chats`}
      accessibilityRole="button"
    >
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
        <TouchableOpacity style={[styles.quickChatBtn, { backgroundColor: accentColor + 'CC' }]} onPress={handleQuickChat} activeOpacity={0.8} hitSlop={4} accessibilityLabel={`Quick chat with ${card.name}`} accessibilityRole="button">
          <MessageCircle size={10} color="#fff" />
          <Text style={styles.quickChatText}>Chat</Text>
        </TouchableOpacity>
        {/* NSFW badge */}
        {card.content_rating === 'nsfw' && <View style={[styles.nsfwBadge, { backgroundColor: tc.nsfw + 'CC' }]}><Text style={styles.nsfwText}>18+</Text></View>}
        {/* Featured / NEW badge */}
        {featured && <View style={[styles.newBadge, { backgroundColor: accentColor }]}><Text style={styles.newBadgeText}>PICK OF THE DAY</Text></View>}
        {!featured && showNew && card.content_rating !== 'nsfw' && <View style={[styles.newBadge, { backgroundColor: tc.status.success }]}><Text style={styles.newBadgeText}>NEW</Text></View>}
        {/* Heart animation on double-tap */}
        {showHeart && (
          <Animated.View style={[styles.heartOverlay, heartStyle]}>
            <Heart size={40} color="#fff" fill="#ff4466" />
          </Animated.View>
        )}
        {/* Gradient overlay at bottom of avatar */}
        <View style={[styles.avatarGradient, { backgroundColor: tc.bg.secondary }]} />
      </View>
      <View style={styles.info}>
        <Text style={[styles.name, { color: tc.text.primary }]} numberOfLines={1}>{card.name}</Text>
        <View style={styles.metaRow}>
          <View style={[styles.genrePill, { backgroundColor: accentColor + '18', borderColor: accentColor + '35' }]}>
            <Text style={[styles.genrePillText, { color: accentColor }]}>{genreLabel}</Text>
          </View>
          <View style={styles.ratingRow}>
            {Array.from({ length: 5 }, (_, i) => (
              <Star key={i} size={9} color={i < starCount ? tc.accent.secondary : tc.text.faint} fill={i < starCount ? tc.accent.secondary : 'transparent'} />
            ))}
          </View>
        </View>
        {card.first_message ? (
          <Text style={[styles.preview, { color: tc.text.muted }]} numberOfLines={2}>{card.first_message.replace(/\*[^*]+\*/g, '').replace(/\n/g, ' ').trim().slice(0, 80)}</Text>
        ) : null}
        <View style={styles.bottomRow}>
          <MessageCircle size={10} color={tc.text.muted} />
          <Text style={[styles.sessions, { color: tc.text.muted }]}>{formatCount(card.session_count)} chats</Text>
        </View>
      </View>
    </Pressable>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  card: { borderRadius: borderRadius.lg, overflow: 'hidden', borderWidth: 1 },
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
  heartOverlay: { position: 'absolute', zIndex: 10, justifyContent: 'center', alignItems: 'center', top: '35%', left: '35%' },
  quickChatBtn: { position: 'absolute', bottom: spacing.sm + 30, right: spacing.sm, flexDirection: 'row', alignItems: 'center', gap: 3, paddingHorizontal: 8, height: 26, borderRadius: 13, justifyContent: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 3 },
  quickChatText: { fontSize: 10, fontWeight: '700', color: '#fff' },
  nsfwBadge: { position: 'absolute', top: spacing.sm, right: spacing.sm, paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: borderRadius.sm },
  newBadge: { position: 'absolute', top: spacing.sm, left: spacing.sm, paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: borderRadius.sm },
  newBadgeText: { fontSize: 8, fontWeight: '800', color: '#fff', letterSpacing: 1 },
  nsfwText: { ...typography.small, color: '#fff', fontWeight: '700' },
  info: { padding: spacing.md, gap: 4 },
  name: { ...typography.subheading, fontSize: 14 },
  metaRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 2 },
  genrePill: { borderWidth: 1, borderRadius: borderRadius.full, paddingHorizontal: spacing.sm, paddingVertical: 1 },
  genrePillText: { fontSize: 10, fontWeight: '600' },
  ratingRow: { flexDirection: 'row', gap: 1 },
  bottomRow: { flexDirection: 'row', alignItems: 'center', gap: 3, marginTop: 2 },
  preview: { ...typography.small, fontSize: 10, lineHeight: 14, marginTop: 2 },
  sessions: { ...typography.small, fontSize: 10 },
  avatarInitial: { fontSize: 24, fontWeight: '700', opacity: 0.7 },
  avatarGlow: { position: 'absolute', width: '100%', height: '100%', borderRadius: 16 },
  horizontalCard: { width: 120, alignItems: 'center', marginRight: spacing.md },
  horizontalAvatar: { width: 100, height: 100, borderRadius: borderRadius.lg, justifyContent: 'center', alignItems: 'center', marginBottom: spacing.sm, overflow: 'hidden' },
  horizontalAvatarImage: { width: 100, height: 100, borderRadius: borderRadius.lg },
  horizontalName: { ...typography.caption, fontWeight: '600', textAlign: 'center' },
  horizontalGenre: { ...typography.small, textTransform: 'capitalize' },
});
