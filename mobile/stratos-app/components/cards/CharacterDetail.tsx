import React, { useState, useEffect, useMemo } from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet, Alert, Share } from 'react-native';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { Star, BookmarkPlus, BookmarkCheck, Flag, Share2, Wand2, StarIcon, Pencil, Copy } from 'lucide-react-native';
import { CharacterCard, formatCount } from '../../lib/types';
import { TagPills } from './TagPills';
import { CharacterDepthSection } from './CharacterDepthSection';
import { SimilarCharacters } from './SimilarCharacters';
import { useChatStore } from '../../stores/chatStore';
import { useCharacterStore } from '../../stores/characterStore';
import { isCardSaved } from '../../lib/storage';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { getGenreColor } from '../../constants/genres';
import { rateCard } from '../../lib/rp';
import { reportError } from '../../lib/utils';
import { fonts } from '../../constants/fonts';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, withSequence, withRepeat, withTiming } from 'react-native-reanimated';

interface CharacterDetailProps { card: CharacterCard; }

export function CharacterDetailView({ card }: CharacterDetailProps) {
  const router = useRouter();
  const tc = useThemeStore(s => s.colors);
  const { startSession } = useChatStore();
  const { saveToLibrary, removeFromLibrary, newCards } = useCharacterStore();
  const [saved, setSaved] = useState(false);
  const [showDepth, setShowDepth] = useState(false);
  const accentColor = getGenreColor(card.genre_tags[0] ?? 'default');
  const similarCards = useMemo(() => newCards.filter(c => c.id !== card.id && c.genre_tags.some(t => card.genre_tags.includes(t))).slice(0, 3), [newCards, card.id, card.genre_tags]);
  const btnScale = useSharedValue(1);
  const btnAnimStyle = useAnimatedStyle(() => ({ transform: [{ scale: btnScale.value }] }));
  const ctaPulse = useSharedValue(1);
  useEffect(() => {
    ctaPulse.value = withRepeat(withSequence(withTiming(1.02, { duration: 1500 }), withTiming(1, { duration: 1500 })), -1, false);
  }, []);
  const ctaStyle = useAnimatedStyle(() => ({ transform: [{ scale: ctaPulse.value }] }));

  useEffect(() => {
    isCardSaved(card.id).then(setSaved);
  }, [card.id]);

  const handleStartChat = () => {
    btnScale.value = withSequence(withSpring(0.95, { damping: 15 }), withSpring(1, { damping: 10 }));
    startSession(card, 'roleplay');
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

  const handleNavigateToCharacter = (id: string) => {
    router.push(`/character/${id}`);
  };

  return (
    <ScrollView style={[styles.container, { backgroundColor: tc.bg.primary }]} contentContainerStyle={styles.content}>
      <View style={[styles.avatarContainer, { backgroundColor: accentColor + '10' }]}>
        {card.avatar_url ? <Image source={{ uri: card.avatar_url }} style={styles.avatar} /> : (
          <>
            <View style={[styles.avatarGlowBg, { backgroundColor: accentColor, opacity: 0.06 }]} />
            <Text style={[styles.avatarLetter, { color: accentColor }]}>{card.name[0]}</Text>
          </>
        )}
        <View style={[styles.avatarFade, { backgroundColor: tc.bg.primary }]} />
        <View style={[styles.avatarBorderLine, { backgroundColor: accentColor + '30' }]} />
      </View>
      <View style={styles.nameRow}>
        <Text style={[styles.name, { color: tc.text.primary }]}>{card.name}</Text>
        <View style={[styles.ratingBadge, { backgroundColor: card.content_rating === 'nsfw' ? tc.nsfw + '20' : tc.status.success + '15' }]}>
          <Text style={[styles.ratingBadgeText, { color: card.content_rating === 'nsfw' ? tc.nsfw : tc.status.success }]}>{card.content_rating === 'nsfw' ? '18+' : 'SFW'}</Text>
        </View>
      </View>
      <Text style={[styles.creator, { color: tc.text.secondary }]}>by @{card.creator_name} · Created {formatRelativeTime(card.created_at)}</Text>
      <View style={styles.ratingRow}>
        <Star size={14} color={tc.accent.secondary} fill={tc.accent.secondary} />
        <Text style={[styles.rating, { color: tc.text.primary }]}>{card.rating.toFixed(1)}</Text>
        <Text style={[styles.separator, { color: tc.text.muted }]}>·</Text>
        <Text style={[styles.sessionCount, { color: tc.text.secondary }]}>{formatCount(card.session_count)} chats</Text>
        <Text style={[styles.separator, { color: tc.text.muted }]}>·</Text>
        <Text style={[styles.sessionCount, { color: tc.text.muted }]}>{[card.description, card.personality, card.scenario, card.first_message, card.physical_description, card.speech_pattern, card.emotional_trigger, card.defensive_mechanism, card.vulnerability, card.specific_detail].filter(Boolean).join(' ').split(/\s+/).length} words</Text>
      </View>
      <View style={styles.section}><TagPills tags={card.genre_tags} size="medium" /></View>

      {/* ── Action buttons (moved to top for accessibility) ── */}
      <Animated.View style={[btnAnimStyle, ctaStyle]}>
        <TouchableOpacity style={[styles.primaryButton, { backgroundColor: accentColor }]} onPress={handleStartChat} activeOpacity={0.8} accessibilityLabel={`Start conversation with ${card.name}`} accessibilityRole="button">
          <Text style={styles.primaryButtonText}>Start Chat</Text>
        </TouchableOpacity>
      </Animated.View>
      <View style={styles.actionRow}>
        <TouchableOpacity style={[styles.secondaryButton, { flex: 1, borderColor: tc.border.medium }, saved && { borderColor: tc.status.success + '60', backgroundColor: tc.status.success + '10' }]} onPress={handleToggleSave} activeOpacity={0.7} accessibilityLabel={saved ? `Remove ${card.name} from library` : `Save ${card.name} to library`} accessibilityRole="button">
          {saved ? <BookmarkCheck size={18} color={tc.status.success} /> : <BookmarkPlus size={18} color={tc.text.secondary} />}
          <Text style={[styles.secondaryButtonText, { color: tc.text.secondary }, saved && { color: tc.status.success }]}>{saved ? 'Saved' : 'Save'}</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.secondaryButton, { flex: 1, borderColor: accentColor + '40' }]} onPress={() => {
          Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
          router.push({ pathname: '/(tabs)/create', params: { editCard: JSON.stringify(card) } });
        }} activeOpacity={0.7} accessibilityLabel={`Edit ${card.name}`} accessibilityRole="button">
          <Pencil size={18} color={accentColor} />
          <Text style={[styles.secondaryButtonText, { color: accentColor }]}>Edit</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.secondaryButton, { flex: 1, borderColor: tc.border.medium }]} onPress={() => {
          Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
          // Clone card — strip id/creator so CardEditor creates a NEW card instead of updating
          const { id, creator_id, creator_name, ...cardData } = card;
          const clone = { ...cardData, name: `${card.name} (Copy)` };
          router.push({ pathname: '/(tabs)/create', params: { newCard: JSON.stringify(clone) } });
        }} activeOpacity={0.7} accessibilityLabel={`Copy ${card.name} to edit`} accessibilityRole="button">
          <Copy size={18} color={tc.text.secondary} />
          <Text style={[styles.secondaryButtonText, { color: tc.text.secondary }]}>Copy</Text>
        </TouchableOpacity>
      </View>
      {/* Generate Portrait */}
      {card.physical_description ? (
        <TouchableOpacity
          style={[styles.portraitBtn, { borderColor: accentColor + '40', backgroundColor: accentColor + '08' }]}
          onPress={() => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            router.push({ pathname: '/imagegen', params: { name: card.name, description: card.physical_description, card_id: card.id } });
          }}
          activeOpacity={0.7}
        >
          <Wand2 size={16} color={accentColor} />
          <Text style={[styles.portraitBtnText, { color: accentColor }]}>Generate Portrait</Text>
        </TouchableOpacity>
      ) : null}

      {/* Rate this character */}
      <RatingSection card={card} tc={tc} />

      {/* ── Character details (below actions) ── */}
      <View style={styles.section}><Text style={[styles.sectionTitle, { color: tc.text.primary }]}>Description</Text><Text style={[styles.sectionBody, { color: tc.text.secondary }]}>{card.description}</Text></View>
      {card.personality && <View style={styles.section}><Text style={[styles.sectionTitle, { color: tc.text.primary }]}>Personality</Text><Text style={[styles.sectionBody, { color: tc.text.secondary }]}>{card.personality}</Text></View>}
      {card.scenario && <View style={styles.section}><Text style={[styles.sectionTitle, { color: tc.text.primary }]}>Scenario</Text><Text style={[styles.sectionBody, { color: tc.text.secondary }]}>{card.scenario}</Text></View>}

      <CharacterDepthSection
        card={card}
        showDepth={showDepth}
        accentColor={accentColor}
        onToggleDepth={() => setShowDepth(!showDepth)}
      />

      {card.first_message && (
        <View style={[styles.section, styles.firstMsgSection, { backgroundColor: tc.bg.tertiary, borderColor: tc.border.subtle }]}>
          <Text style={[styles.sectionTitle, { color: tc.text.primary, marginBottom: spacing.sm }]}>Opening Line</Text>
          <Text style={[styles.firstMsgText, { color: tc.text.secondary }]}>
            {card.first_message.split('\n').slice(0, 4).map((line, i) => {
              const cleaned = line.replace(/\*{3}([^*]+)\*{3}/g, '$1').replace(/\*{2}([^*]+)\*{2}/g, '$1');
              const isAction = /^\*[^*]+\*$/.test(line.trim());
              return (
                <Text key={i} style={isAction ? { fontStyle: 'italic', color: tc.text.muted } : undefined}>
                  {i > 0 ? '\n' : ''}{cleaned}
                </Text>
              );
            })}
          </Text>
        </View>
      )}

      <SimilarCharacters
        similarCards={similarCards}
        onNavigate={handleNavigateToCharacter}
      />

      <TouchableOpacity style={styles.reportBtn} onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); Alert.alert('Report Character', 'Report this character for inappropriate content?', [{ text: 'Cancel', style: 'cancel' }, { text: 'Report', style: 'destructive', onPress: () => Alert.alert('Reported', 'Thank you for your report. We will review this character.') }]); }} activeOpacity={0.7} accessibilityLabel={`Report ${card.name}`} accessibilityRole="button">
        <Flag size={12} color={tc.text.muted} />
        <Text style={[styles.reportText, { color: tc.text.muted }]}>Report Character</Text>
      </TouchableOpacity>
      <View style={{ height: spacing.xxl }} />
    </ScrollView>
  );
}

function RatingSection({ card, tc }: { card: CharacterCard; tc: any }) {
  const [userRating, setUserRating] = useState(0);
  const displayRating = userRating || Math.round(card.rating);

  return (
    <View style={styles.rateSection}>
      <Text style={[styles.rateLabel, { color: tc.text.muted }]}>Rate this character</Text>
      <View style={styles.rateStars}>
        {[1, 2, 3, 4, 5].map(n => (
          <TouchableOpacity key={n} onPress={() => {
            setUserRating(n);
            Haptics.selectionAsync();
            rateCard(card.id, n).catch(err => reportError('CharacterDetail:rateCard', err));
          }} hitSlop={4}>
            <Star size={22} color={tc.accent.secondary} fill={n <= displayRating ? tc.accent.secondary : 'transparent'} />
          </TouchableOpacity>
        ))}
      </View>
      {userRating > 0 && <Text style={[{ color: tc.text.muted, fontSize: 11, marginTop: 4 }]}>Rated {userRating}/5</Text>}
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
  return `${Math.floor(hours / 24)}d ago`;
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: spacing.lg },
  avatarContainer: { width: '100%', aspectRatio: 1, borderRadius: borderRadius.xl, justifyContent: 'center', alignItems: 'center', marginBottom: spacing.lg, overflow: 'hidden' },
  avatar: { width: '100%', height: '100%' },
  avatarLetter: { fontSize: 80, fontWeight: '700', opacity: 0.5, zIndex: 1 },
  avatarGlowBg: { position: 'absolute', width: 200, height: 200, borderRadius: 100 },
  avatarFade: { position: 'absolute', bottom: 0, left: 0, right: 0, height: 60, opacity: 0.8 },
  avatarBorderLine: { position: 'absolute', bottom: 0, left: 0, right: 0, height: 2 },
  nameRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, marginBottom: spacing.xs },
  name: { ...typography.display, flex: 1 },
  ratingBadge: { paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: borderRadius.sm },
  ratingBadgeText: { fontSize: 10, fontWeight: '800', letterSpacing: 0.5 },
  creator: { ...typography.body, marginBottom: spacing.sm },
  ratingRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs, marginBottom: spacing.lg },
  rating: { ...typography.body, fontWeight: '600' },
  separator: { ...typography.body },
  sessionCount: { ...typography.body },
  section: { marginBottom: spacing.xl },
  sectionTitle: { ...typography.subheading, marginBottom: spacing.sm },
  sectionBody: { ...typography.body, lineHeight: 24 },
  primaryButton: { paddingVertical: spacing.lg, borderRadius: borderRadius.lg, alignItems: 'center', marginBottom: spacing.md },
  primaryButtonText: { ...typography.subheading, color: '#fff' },
  actionRow: { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.sm },
  secondaryButton: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.lg, borderRadius: borderRadius.lg, borderWidth: 1 },
  secondaryButtonText: { ...typography.subheading },
  reportBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.xs, paddingVertical: spacing.lg, marginTop: spacing.lg },
  reportText: { ...typography.small },
  firstMsgSection: { padding: spacing.lg, borderRadius: borderRadius.lg, borderWidth: 1 },
  firstMsgText: { ...typography.body, lineHeight: 22, fontSize: 13 },
  portraitBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.md, borderRadius: borderRadius.lg, borderWidth: 1, marginBottom: spacing.md },
  portraitBtnText: { fontSize: 13, fontFamily: fonts.button },
  rateSection: { alignItems: 'center', marginBottom: spacing.lg, gap: spacing.sm },
  rateLabel: { fontSize: 11, fontFamily: fonts.body },
  rateStars: { flexDirection: 'row', gap: spacing.sm },
});
