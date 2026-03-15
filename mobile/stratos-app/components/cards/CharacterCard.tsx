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
const CARD_WIDTH = (Dimensions.get('window').width - spacing.md * 2 - spacing.sm) / 2;

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
            <View style={[styles.letterGlow, { backgroundColor: accentColor, shadowColor: accentColor }]} />
            <Text style={[styles.avatarLetter, { color: accentColor }]}>{card.name[0]}</Text>
          </>
        )}
        {card.content_rating === 'nsfw' && <View style={[styles.nsfwBadge, { backgroundColor: tc.nsfw + 'CC' }]}><Text style={styles.nsfwText}>18+</Text></View>}
        {featured && <View style={[styles.newBadge, { backgroundColor: accentColor }]}><Text style={styles.newBadgeText}>PICK OF THE DAY</Text></View>}
        {!featured && showNew && card.content_rating !== 'nsfw' && <View style={[styles.newBadge, { backgroundColor: tc.status.success }]}><Text style={styles.newBadgeText}>NEW</Text></View>}
        {showHeart && (
          <Animated.View style={[styles.heartOverlay, heartStyle]}>
            <Heart size={40} color="#fff" fill="#ff4466" />
          </Animated.View>
        )}
      </View>
      <View style={styles.info}>
        <View style={styles.nameRow}>
          <Text style={[styles.name, { color: tc.text.primary }]} numberOfLines={1}>{card.name}</Text>
          <Text style={[styles.genreTag, { color: accentColor }]}>{genreLabel}</Text>
        </View>
      </View>
      <TouchableOpacity style={[styles.quickChatBtn, { backgroundColor: accentColor + '15', borderColor: accentColor + '30' }]} onPress={handleQuickChat} activeOpacity={0.7} accessibilityLabel={`Chat with ${card.name}`} accessibilityRole="button">
        <MessageCircle size={11} color={accentColor} />
        <Text style={[styles.quickChatText, { color: accentColor }]}>Chat</Text>
      </TouchableOpacity>
    </Pressable>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  card: { borderRadius: borderRadius.lg, overflow: 'hidden', borderWidth: 1 },
  avatarContainer: { width: '100%', aspectRatio: 1.1, justifyContent: 'center', alignItems: 'center', position: 'relative' },
  avatarImage: { width: '100%', height: '100%' },
  avatarLetter: { fontSize: 36, fontWeight: '700', opacity: 0.7 },
  letterGlow: {
    position: 'absolute', width: 60, height: 60, borderRadius: 30,
    opacity: 0.12,
    shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.6, shadowRadius: 20, elevation: 3,
  },
  heartOverlay: { position: 'absolute', zIndex: 10, justifyContent: 'center', alignItems: 'center', top: '30%', left: '30%' },
  quickChatBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 3, paddingVertical: spacing.xs, borderTopWidth: 1 },
  quickChatText: { fontSize: 10, fontWeight: '700' },
  nsfwBadge: { position: 'absolute', top: spacing.xs, right: spacing.xs, paddingHorizontal: 4, paddingVertical: 1, borderRadius: borderRadius.sm },
  newBadge: { position: 'absolute', top: spacing.xs, left: spacing.xs, paddingHorizontal: 4, paddingVertical: 1, borderRadius: borderRadius.sm },
  newBadgeText: { fontSize: 7, fontWeight: '800', color: '#fff', letterSpacing: 0.5 },
  nsfwText: { fontSize: 8, color: '#fff', fontWeight: '700' },
  info: { paddingHorizontal: spacing.xs, paddingVertical: 3 },
  nameRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 2 },
  name: { fontSize: 12, fontWeight: '700', flex: 1 },
  genreTag: { fontSize: 9, fontWeight: '600' },
  avatarInitial: { fontSize: 24, fontWeight: '700', opacity: 0.7 },
  avatarGlow: { position: 'absolute', width: '100%', height: '100%', borderRadius: 16 },
  horizontalCard: { width: 120, alignItems: 'center', marginRight: spacing.md },
  horizontalAvatar: { width: 100, height: 100, borderRadius: borderRadius.lg, justifyContent: 'center', alignItems: 'center', marginBottom: spacing.sm, overflow: 'hidden' },
  horizontalAvatarImage: { width: 100, height: 100, borderRadius: borderRadius.lg },
  horizontalName: { ...typography.caption, fontWeight: '600', textAlign: 'center' },
  horizontalGenre: { ...typography.small, textTransform: 'capitalize' },
});
