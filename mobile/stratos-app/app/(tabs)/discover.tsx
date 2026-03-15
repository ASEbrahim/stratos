import React, { useEffect, useState, useCallback } from 'react';
import { View, Text, ScrollView, TouchableOpacity, TextInput, StyleSheet, RefreshControl } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Search, Shuffle, ChevronUp } from 'lucide-react-native';
import * as Haptics from 'expo-haptics';
import Animated, { FadeInDown } from 'react-native-reanimated';
import { useCharacterStore } from '../../stores/characterStore';
import { useChatStore } from '../../stores/chatStore';
import { useThemeStore } from '../../stores/themeStore';
import { CharacterCardComponent } from '../../components/cards/CharacterCard';
import { ScenarioCard } from '../../components/gaming/ScenarioCard';
import { StarParallaxBg } from '../../components/shared/StarParallax';
import { DiscoverSkeleton } from '../../components/shared/Skeleton';
import { GENRES } from '../../constants/genres';
import { GamingScenario } from '../../lib/types';
import { getScenarios } from '../../lib/gaming';
import { useRouter } from 'expo-router';
import { getGenreColor } from '../../constants/genres';
import { formatCount } from '../../lib/types';
import { typography, spacing, borderRadius } from '../../constants/theme';

export default function DiscoverScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const tc = useThemeStore(s => s.colors);
  const { trending, newCards, selectedGenre, isLoading, loadTrending, loadNew, setGenre, search } = useCharacterStore();
  const { recentSessions, loadRecentSessions, resumeSession } = useChatStore();
  const [scenarios, setScenarios] = useState<GamingScenario[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const [sortBy, setSortBy] = useState<'popular' | 'newest' | 'rating'>('popular');
  const [showScrollTop, setShowScrollTop] = useState(false);
  const scrollRef = React.useRef<ScrollView>(null);

  useEffect(() => { loadTrending(); loadNew(); getScenarios().then(setScenarios); loadRecentSessions(); }, []);
  const onRefresh = useCallback(async () => { setRefreshing(true); await Promise.all([loadTrending(), loadNew(), getScenarios().then(setScenarios)]); setRefreshing(false); }, []);
  const searchTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const handleSearch = (text: string) => {
    setSearchQuery(text);
    if (searchTimer.current) clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => {
      if (text.trim()) search(text.trim(), selectedGenre ?? undefined);
    }, 300);
  };
  const nsfwFilter = useThemeStore(s => s.nsfwFilter);
  const filterNsfw = (cards: typeof newCards) => nsfwFilter ? cards.filter(c => c.content_rating !== 'nsfw') : cards;
  const sortCards = (cards: typeof newCards) => {
    const sorted = [...cards];
    if (sortBy === 'popular') sorted.sort((a, b) => b.session_count - a.session_count);
    else if (sortBy === 'newest') sorted.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    else if (sortBy === 'rating') sorted.sort((a, b) => b.rating - a.rating);
    return sorted;
  };
  const displayCards = sortCards(filterNsfw(searchQuery.trim() ? useCharacterStore.getState().searchResults : (selectedGenre ? newCards.filter(c => c.genre_tags.includes(selectedGenre)) : newCards)));

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <StarParallaxBg />
      <ScrollView ref={scrollRef} style={styles.scroll} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={tc.accent.primary} />}
        onScroll={(e) => setShowScrollTop(e.nativeEvent.contentOffset.y > 600)}
        scrollEventThrottle={200}>
        <Animated.View entering={FadeInDown.duration(400).springify().damping(16)} style={styles.brandRow}>
          <Text style={[styles.brandText, { color: tc.accent.primary }]}>StratOS</Text>
          <Text style={[styles.brandSub, { color: tc.text.muted }]}>Characters</Text>
        </Animated.View>
        <View style={styles.searchRow}>
          <View style={[styles.searchBox, { backgroundColor: tc.bg.tertiary, flex: 1 }]}>
            <Search size={18} color={tc.text.muted} />
            <TextInput style={[styles.searchInput, { color: tc.text.primary }]} value={searchQuery} onChangeText={handleSearch} placeholder="Search characters..." placeholderTextColor={tc.text.muted} />
          </View>
          {newCards.length > 0 && (
            <TouchableOpacity style={[styles.shuffleBtn, { backgroundColor: tc.accent.primary + '15' }]} onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
              const all = filterNsfw(newCards);
              const pick = all[Math.floor(Math.random() * all.length)];
              if (pick) router.push(`/character/${pick.id}`);
            }} activeOpacity={0.7}>
              <Shuffle size={18} color={tc.accent.primary} />
            </TouchableOpacity>
          )}
        </View>
        {isLoading && trending.length === 0 && <DiscoverSkeleton />}
        {/* Featured Character Spotlight */}
        {!searchQuery.trim() && trending.length > 0 && (() => {
          // Character of the Day — rotate daily based on day of year
          const dayOfYear = Math.floor((Date.now() - new Date(new Date().getFullYear(), 0, 0).getTime()) / 86400000);
          const sortedByRating = [...trending].sort((a, b) => b.rating - a.rating);
          const featured = sortedByRating[dayOfYear % sortedByRating.length];
          if (!featured) return null;
          const fc = getGenreColor(featured.genre_tags[0] ?? 'default');
          return (
            <TouchableOpacity style={[styles.spotlightCard, { borderColor: fc + '30' }]} onPress={() => router.push(`/character/${featured.id}`)} activeOpacity={0.8}>
              <View style={[styles.spotlightAvatar, { backgroundColor: fc + '15' }]}>
                <Text style={[styles.spotlightLetter, { color: fc }]}>{featured.name[0]}</Text>
              </View>
              <View style={styles.spotlightInfo}>
                <View style={[styles.spotlightBadge, { backgroundColor: fc + '20' }]}>
                  <Text style={[styles.spotlightBadgeText, { color: fc }]}>Character of the Day</Text>
                </View>
                <Text style={[styles.spotlightName, { color: tc.text.primary }]}>{featured.name}</Text>
                <Text style={[styles.spotlightDesc, { color: tc.text.secondary }]} numberOfLines={2}>{featured.description}</Text>
                <Text style={[styles.spotlightMeta, { color: tc.text.muted }]}>{featured.rating.toFixed(1)} rating · {formatCount(featured.session_count)} chats</Text>
              </View>
            </TouchableOpacity>
          );
        })()}
        {/* Recently Chatted — quick resume */}
        {!searchQuery.trim() && recentSessions.length > 0 && (
          <>
            <View style={styles.recentRow}>
              <Text style={[styles.recentLabel, { color: tc.text.muted }]}>Continue</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.recentScroll}>
                {recentSessions.slice(0, 8).map(s => (
                  <TouchableOpacity key={s.id} style={styles.recentItem} onPress={() => { resumeSession(s); router.push(`/chat/${s.character_id}`); }} activeOpacity={0.7}>
                    <View style={[styles.recentAvatar, { backgroundColor: tc.accent.primary + '15', borderColor: tc.accent.primary + '30' }]}>
                      <Text style={[styles.recentLetter, { color: tc.accent.primary }]}>{s.character_name[0]}</Text>
                      {s.messages.length > 1 && (
                        <View style={[styles.recentBadge, { backgroundColor: tc.accent.primary }]}>
                          <Text style={styles.recentBadgeText}>{s.messages.length > 99 ? '99+' : s.messages.length}</Text>
                        </View>
                      )}
                    </View>
                    <Text style={[styles.recentName, { color: tc.text.secondary }]} numberOfLines={1}>{s.character_name.split(' ')[0]}</Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>
            </View>
          </>
        )}
        {/* Popular This Week — curated picks */}
        {!searchQuery.trim() && trending.length > 0 && (
          <>
            <View style={styles.sectionHdr}>
              <Text style={[styles.sectionTitle, { color: tc.text.primary }]}>Popular This Week</Text>
            </View>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.popularScroll}>
              {trending.slice(0, 5).map(c => {
                const gc = getGenreColor(c.genre_tags[0] ?? 'default');
                return (
                  <TouchableOpacity key={`pop-${c.id}`} style={[styles.popularCard, { borderColor: gc + '25' }]} onPress={() => router.push(`/character/${c.id}`)} activeOpacity={0.7}>
                    <View style={[styles.popularAvatar, { backgroundColor: gc + '15' }]}>
                      <Text style={[styles.popularLetter, { color: gc }]}>{c.name[0]}</Text>
                    </View>
                    <View style={styles.popularInfo}>
                      <Text style={[styles.popularName, { color: tc.text.primary }]} numberOfLines={1}>{c.name}</Text>
                      <Text style={[styles.popularCreator, { color: tc.text.muted }]}>by @{c.creator_name}</Text>
                      <Text style={[styles.popularDesc, { color: tc.text.secondary }]} numberOfLines={2}>{c.description}</Text>
                      {c.first_message && <Text style={[styles.popularQuote, { color: tc.text.muted }]} numberOfLines={1}>"{c.first_message.replace(/\*[^*]+\*/g, '').replace(/\n/g, ' ').trim().slice(0, 60)}"</Text>}
                      <Text style={[styles.popularStat, { color: gc }]}>{formatCount(c.session_count)} chats · {c.rating.toFixed(1)}★</Text>
                    </View>
                  </TouchableOpacity>
                );
              })}
            </ScrollView>
          </>
        )}
        {showWelcome && !searchQuery.trim() && (
          <TouchableOpacity style={[styles.welcomeCard, { backgroundColor: tc.accent.primary + '10', borderColor: tc.accent.primary + '25' }]} onPress={() => setShowWelcome(false)} activeOpacity={0.8}>
            <Text style={[styles.welcomeTitle, { color: tc.text.primary }]}>Welcome to StratOS</Text>
            <Text style={[styles.welcomeBody, { color: tc.text.secondary }]}>Discover AI characters for immersive roleplay and interactive gaming. Tap a character to start a conversation.</Text>
          </TouchableOpacity>
        )}
        {!searchQuery.trim() && trending.length > 0 && (
          <>
            <View style={styles.sectionHdr}><Text style={[styles.sectionTitle, { color: tc.text.primary }]}>Trending</Text></View>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.hScroll}>
              {trending.slice(0, 10).map(c => <CharacterCardComponent key={c.id} card={c} variant="horizontal" />)}
            </ScrollView>
          </>
        )}
        {!searchQuery.trim() && scenarios.length > 0 && (
          <>
            <View style={styles.sectionHdr}><Text style={[styles.sectionTitle, { color: tc.text.primary }]}>Gaming Scenarios</Text></View>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.hScroll}>
              {scenarios.map(s => <ScenarioCard key={s.id} scenario={s} variant="horizontal" />)}
            </ScrollView>
          </>
        )}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.genreScroll}>
          <TouchableOpacity style={[styles.genreChip, { borderColor: tc.border.subtle, backgroundColor: tc.bg.secondary }, !selectedGenre && { backgroundColor: tc.accent.primary + '20', borderColor: tc.accent.primary }]} onPress={() => { Haptics.selectionAsync(); setGenre(null); }}>
            <Text style={[styles.genreText, { color: tc.text.secondary }, !selectedGenre && { color: tc.accent.primary }]}>All</Text>
          </TouchableOpacity>
          {GENRES.map(g => {
            const a = selectedGenre === g.id;
            return (
              <TouchableOpacity key={g.id} style={[styles.genreChip, { borderColor: tc.border.subtle, backgroundColor: tc.bg.secondary }, a && { backgroundColor: g.color + '20', borderColor: g.color }]} onPress={() => { Haptics.selectionAsync(); setGenre(a ? null : g.id); }}>
                <Text style={[styles.genreText, { color: tc.text.secondary }, a && { color: g.color }]}>{g.emoji} {g.label}</Text>
              </TouchableOpacity>
            );
          })}
        </ScrollView>
        <View style={styles.gridHeader}>
          <Text style={[styles.sectionTitle, { color: tc.text.primary }]}>{searchQuery.trim() ? 'Search Results' : 'Characters'} <Text style={{ color: tc.text.muted, fontSize: 14 }}>({displayCards.length})</Text></Text>
          {!searchQuery.trim() && (
            <View style={styles.sortRow}>
              {(['popular', 'newest', 'rating'] as const).map(s => (
                <TouchableOpacity key={s} onPress={() => { Haptics.selectionAsync(); setSortBy(s); }} style={[styles.sortChip, sortBy === s && { backgroundColor: tc.accent.primary + '15' }]}>
                  <Text style={[styles.sortText, { color: sortBy === s ? tc.accent.primary : tc.text.muted }]}>{s === 'popular' ? 'Popular' : s === 'newest' ? 'New' : 'Top Rated'}</Text>
                </TouchableOpacity>
              ))}
            </View>
          )}
        </View>
        {displayCards.length === 0 ? (
          <View style={styles.emptySearch}>
            <Text style={[styles.emptyIcon]}>{searchQuery.trim() ? '🔍' : selectedGenre ? GENRES.find(g => g.id === selectedGenre)?.emoji ?? '🎭' : '✨'}</Text>
            <Text style={[styles.emptyTitle, { color: tc.text.secondary }]}>{searchQuery.trim() ? 'No characters found' : selectedGenre ? `No ${selectedGenre} characters yet` : 'No characters'}</Text>
            <Text style={[styles.emptySubtitle, { color: tc.text.muted }]}>{searchQuery.trim() ? 'Try a different search term or browse by genre' : selectedGenre ? 'Be the first to create one!' : 'Characters will appear here'}</Text>
          </View>
        ) : (
          <View style={styles.grid}>{displayCards.map((c, idx) => (
            <Animated.View key={c.id} entering={FadeInDown.delay(idx * 60).duration(300).springify().damping(14)}>
              <CharacterCardComponent card={c} />
            </Animated.View>
          ))}</View>
        )}
        <View style={{ height: spacing.xxl }} />
      </ScrollView>
      {showScrollTop && (
        <TouchableOpacity style={[styles.scrollTopFab, { backgroundColor: tc.bg.elevated, borderColor: tc.border.subtle }]} onPress={() => scrollRef.current?.scrollTo({ y: 0, animated: true })} activeOpacity={0.8}>
          <ChevronUp size={18} color={tc.text.secondary} />
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  scroll: { flex: 1 },
  brandRow: { flexDirection: 'row', alignItems: 'baseline', paddingHorizontal: spacing.lg, paddingTop: spacing.md, gap: spacing.sm },
  brandText: { fontSize: 22, fontWeight: '800', letterSpacing: -0.5 },
  brandSub: { fontSize: 12, fontWeight: '500' },
  searchRow: { flexDirection: 'row', alignItems: 'center', marginHorizontal: spacing.lg, marginTop: spacing.sm, marginBottom: spacing.lg, gap: spacing.sm },
  searchBox: { flexDirection: 'row', alignItems: 'center', borderRadius: borderRadius.lg, paddingHorizontal: spacing.lg, gap: spacing.sm },
  shuffleBtn: { width: 44, height: 44, borderRadius: borderRadius.lg, justifyContent: 'center', alignItems: 'center' },
  searchInput: { flex: 1, paddingVertical: spacing.md, fontSize: 15 },
  sectionHdr: { paddingHorizontal: spacing.lg, marginTop: spacing.lg, marginBottom: spacing.md },
  sectionTitle: { ...typography.heading },
  hScroll: { paddingHorizontal: spacing.lg },
  genreScroll: { paddingHorizontal: spacing.lg, gap: spacing.sm, paddingVertical: spacing.md },
  genreChip: { paddingHorizontal: spacing.md, paddingVertical: spacing.sm, borderRadius: borderRadius.full, borderWidth: 1 },
  genreText: { ...typography.caption },
  grid: { flexDirection: 'row', flexWrap: 'wrap', paddingHorizontal: spacing.lg, gap: spacing.lg },
  emptySearch: { alignItems: 'center', paddingVertical: spacing.xxl * 2, paddingHorizontal: spacing.xxl },
  emptyIcon: { fontSize: 40, marginBottom: spacing.md },
  emptyTitle: { ...typography.subheading, marginBottom: spacing.xs },
  emptySubtitle: { ...typography.caption, textAlign: 'center' },
  spotlightCard: { flexDirection: 'row', marginHorizontal: spacing.lg, marginBottom: spacing.md, padding: spacing.lg, borderRadius: borderRadius.lg, borderWidth: 1, gap: spacing.lg, backgroundColor: 'rgba(255,255,255,0.02)' },
  spotlightAvatar: { width: 70, height: 70, borderRadius: borderRadius.lg, justifyContent: 'center', alignItems: 'center' },
  spotlightLetter: { fontSize: 28, fontWeight: '700', opacity: 0.7 },
  spotlightInfo: { flex: 1, justifyContent: 'center' },
  spotlightBadge: { alignSelf: 'flex-start', paddingHorizontal: spacing.sm, paddingVertical: 1, borderRadius: borderRadius.sm, marginBottom: 4 },
  spotlightBadgeText: { fontSize: 9, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 1 },
  spotlightName: { ...typography.subheading, marginBottom: 2 },
  spotlightDesc: { ...typography.small, lineHeight: 15, fontSize: 11, marginBottom: 4 },
  spotlightMeta: { fontSize: 10 },
  recentRow: { flexDirection: 'row', alignItems: 'center', paddingLeft: spacing.lg, marginBottom: spacing.sm },
  recentLabel: { ...typography.small, fontSize: 10, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 1, marginRight: spacing.sm },
  recentScroll: { gap: spacing.md, paddingRight: spacing.lg },
  recentItem: { alignItems: 'center', width: 52 },
  recentAvatar: { width: 40, height: 40, borderRadius: 20, justifyContent: 'center', alignItems: 'center', borderWidth: 1.5, marginBottom: 3 },
  recentLetter: { fontSize: 16, fontWeight: '700' },
  recentName: { fontSize: 9, fontWeight: '500', textAlign: 'center' },
  recentBadge: { position: 'absolute', top: -4, right: -4, minWidth: 16, height: 16, borderRadius: 8, justifyContent: 'center', alignItems: 'center', paddingHorizontal: 3 },
  recentBadgeText: { fontSize: 8, fontWeight: '800', color: '#fff' },
  gridHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.lg, marginTop: spacing.lg, marginBottom: spacing.md },
  sortRow: { flexDirection: 'row', gap: 2 },
  sortChip: { paddingHorizontal: spacing.sm, paddingVertical: 3, borderRadius: borderRadius.sm },
  sortText: { fontSize: 10, fontWeight: '600' },
  popularScroll: { paddingHorizontal: spacing.lg, gap: spacing.md },
  popularCard: { width: 280, flexDirection: 'row', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: borderRadius.lg, padding: spacing.md, borderWidth: 1, gap: spacing.md },
  popularAvatar: { width: 56, height: 56, borderRadius: borderRadius.md, justifyContent: 'center', alignItems: 'center' },
  popularLetter: { fontSize: 22, fontWeight: '700', opacity: 0.7 },
  popularInfo: { flex: 1, justifyContent: 'center', gap: 2 },
  popularName: { ...typography.subheading, fontSize: 14 },
  popularCreator: { fontSize: 9, marginBottom: 2 },
  popularDesc: { ...typography.small, lineHeight: 15, fontSize: 10 },
  popularQuote: { ...typography.small, fontSize: 9, fontStyle: 'italic', lineHeight: 13, marginTop: 2 },
  popularStat: { ...typography.small, fontSize: 10, fontWeight: '600', marginTop: 2 },
  welcomeCard: { marginHorizontal: spacing.lg, marginBottom: spacing.md, padding: spacing.lg, borderRadius: borderRadius.lg, borderWidth: 1 },
  welcomeTitle: { ...typography.subheading, marginBottom: spacing.xs },
  welcomeBody: { ...typography.caption, lineHeight: 18 },
  scrollTopFab: { position: 'absolute', right: spacing.lg, bottom: 80, width: 36, height: 36, borderRadius: 18, justifyContent: 'center', alignItems: 'center', borderWidth: 1, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 4 },
});
