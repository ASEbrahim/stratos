import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { View, Text, ScrollView, TouchableOpacity, TextInput, StyleSheet, RefreshControl } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Search, Shuffle, ChevronUp } from 'lucide-react-native';
import * as Haptics from 'expo-haptics';
import Animated, { FadeInDown } from 'react-native-reanimated';
import { useCharacterStore } from '../../stores/characterStore';
import { useChatStore } from '../../stores/chatStore';
import { useThemeStore } from '../../stores/themeStore';
import { THEMES } from '../../constants/themes';
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
import { reportError } from '../../lib/utils';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';

export default function DiscoverScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { colors: tc, themeId, setTheme } = useThemeStore();
  const { trending, newCards, selectedGenre, searchResults, isLoading, isLoadingMore, hasMore, loadTrending, loadNew, loadMore, setGenre, search } = useCharacterStore();
  const { recentSessions, loadRecentSessions, resumeSession } = useChatStore();
  const [scenarios, setScenarios] = useState<GamingScenario[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [sortBy, setSortBy] = useState<'popular' | 'newest' | 'rating' | 'gaming'>('popular');
  const [showScrollTop, setShowScrollTop] = useState(false);
  const scrollRef = React.useRef<ScrollView>(null);

  useEffect(() => { loadTrending(); loadNew(); getScenarios().then(setScenarios).catch(err => reportError('DiscoverScreen:getScenarios', err)); }, []);
  const onRefresh = useCallback(async () => { setRefreshing(true); try { await Promise.all([loadTrending(), loadNew(), getScenarios().then(setScenarios)]); } catch (err) { reportError('DiscoverScreen:onRefresh', err); } setRefreshing(false); }, []);
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
    if (sortBy === 'popular') sorted.sort((a, b) => (b.session_count - a.session_count) || (b.rating - a.rating) || (new Date(b.created_at).getTime() - new Date(a.created_at).getTime()));
    else if (sortBy === 'newest') sorted.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    else if (sortBy === 'rating') sorted.sort((a, b) => (b.rating - a.rating) || (b.rating_count - a.rating_count) || (b.session_count - a.session_count));
    return sorted;
  };
  const displayCards = useMemo(() => sortCards(filterNsfw(searchQuery.trim() ? searchResults : (selectedGenre ? newCards.filter(c => c.genre_tags.includes(selectedGenre)) : newCards))), [searchQuery, searchResults, selectedGenre, newCards, sortBy, nsfwFilter]);

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <StarParallaxBg />
      <ScrollView ref={scrollRef} style={styles.scroll} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={tc.accent.primary} />}
        onScroll={(e) => {
          setShowScrollTop(e.nativeEvent.contentOffset.y > 600);
          // Infinite scroll: load more when near bottom
          const { layoutMeasurement, contentOffset, contentSize } = e.nativeEvent;
          if (layoutMeasurement.height + contentOffset.y >= contentSize.height - 400 && !searchQuery.trim() && sortBy !== 'gaming') {
            loadMore();
          }
        }}
        scrollEventThrottle={200}>
        <Animated.View entering={FadeInDown.duration(400).springify().damping(16)} style={styles.brandRow}>
          {THEMES.slice(0, 2).map(t => {
            const active = themeId === t.id;
            return (
              <TouchableOpacity key={t.id} onPress={() => { Haptics.selectionAsync(); setTheme(t.id); }} activeOpacity={0.7} style={[styles.themeDot, { backgroundColor: t.colors.accent.primary, opacity: active ? 1 : 0.3 }, active && { shadowColor: t.colors.accent.primary, shadowOpacity: 0.6, shadowRadius: 6, elevation: 3 }]} accessibilityLabel={`${t.label} theme${active ? ', selected' : ''}`} accessibilityRole="button" />
            );
          })}
          <Text style={[styles.brandText, { color: tc.accent.primary, textShadowColor: tc.accent.primary + '40', textShadowOffset: { width: 0, height: 0 }, textShadowRadius: 12 }]}>Strat<Text style={[styles.brandAccent, { color: tc.text.primary }]}>OS</Text></Text>
          {THEMES.slice(2, 4).map(t => {
            const active = themeId === t.id;
            return (
              <TouchableOpacity key={t.id} onPress={() => { Haptics.selectionAsync(); setTheme(t.id); }} activeOpacity={0.7} style={[styles.themeDot, { backgroundColor: t.colors.accent.primary, opacity: active ? 1 : 0.3 }, active && { shadowColor: t.colors.accent.primary, shadowOpacity: 0.6, shadowRadius: 6, elevation: 3 }]} accessibilityLabel={`${t.label} theme${active ? ', selected' : ''}`} accessibilityRole="button" />
            );
          })}
        </Animated.View>
        <View style={styles.searchRow}>
          <View style={[styles.searchBox, { backgroundColor: tc.bg.tertiary, flex: 1 }]}>
            <Search size={18} color={tc.text.muted} />
            <TextInput style={[styles.searchInput, { color: tc.text.primary }]} value={searchQuery} onChangeText={handleSearch} placeholder="Search characters..." placeholderTextColor={tc.text.muted} accessibilityLabel="Search characters" accessibilityRole="search" />
          </View>
          {newCards.length > 0 && (
            <TouchableOpacity style={[styles.shuffleBtn, { backgroundColor: tc.accent.primary + '15' }]} onPress={() => {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
              const all = filterNsfw(newCards);
              const pick = all[Math.floor(Math.random() * all.length)];
              if (pick) router.push(`/character/${pick.id}`);
            }} activeOpacity={0.7} accessibilityLabel="Shuffle random character" accessibilityRole="button">
              <Shuffle size={18} color={tc.accent.primary} />
            </TouchableOpacity>
          )}
        </View>
        {isLoading && trending.length === 0 && <DiscoverSkeleton />}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.genreScroll}>
          <TouchableOpacity style={[styles.genreChip, { borderColor: tc.border.subtle, backgroundColor: tc.bg.secondary }, !selectedGenre && { backgroundColor: tc.accent.primary + '20', borderColor: tc.accent.primary }]} onPress={() => { Haptics.selectionAsync(); setGenre(null); }} accessibilityLabel={`Filter by all genres${!selectedGenre ? ', selected' : ''}`} accessibilityRole="button">
            <Text style={[styles.genreText, { color: tc.text.secondary }, !selectedGenre && { color: tc.accent.primary }]}>All</Text>
          </TouchableOpacity>
          {GENRES.map(g => {
            const a = selectedGenre === g.id;
            return (
              <TouchableOpacity key={g.id} style={[styles.genreChip, { borderColor: tc.border.subtle, backgroundColor: tc.bg.secondary }, a && { backgroundColor: g.color + '20', borderColor: g.color }]} onPress={() => { Haptics.selectionAsync(); setGenre(a ? null : g.id); }} accessibilityLabel={`Filter by ${g.label}${a ? ', selected' : ''}`} accessibilityRole="button">
                <Text style={[styles.genreText, { color: tc.text.secondary }, a && { color: g.color }]}>{g.label}</Text>
              </TouchableOpacity>
            );
          })}
        </ScrollView>
        <View style={styles.gridHeader}>
          <Text style={[styles.sectionTitle, { color: tc.text.primary }]}>{searchQuery.trim() ? 'Search Results' : 'Characters'}</Text>
          {!searchQuery.trim() && (
            <View style={styles.sortRow}>
              {(['popular', 'newest', 'rating', 'gaming'] as const).map(s => (
                <TouchableOpacity key={s} onPress={() => { Haptics.selectionAsync(); setSortBy(s); }} style={[styles.sortChip, sortBy === s && { backgroundColor: tc.accent.primary + '15' }]} accessibilityLabel={`${s === 'gaming' ? 'Gaming scenarios' : `Sort by ${s}`}${sortBy === s ? ', selected' : ''}`} accessibilityRole="button">
                  <Text style={[styles.sortText, { color: sortBy === s ? tc.accent.primary : tc.text.muted }]}>{s === 'popular' ? 'Popular' : s === 'newest' ? 'New' : s === 'rating' ? 'Top Rated' : 'Gaming'}</Text>
                </TouchableOpacity>
              ))}
            </View>
          )}
        </View>
        {sortBy === 'gaming' ? (
          scenarios.length === 0 ? (
            <View style={styles.emptySearch}>
              <Text style={styles.emptyIcon}>🎮</Text>
              <Text style={[styles.emptyTitle, { color: tc.text.secondary }]}>No gaming scenarios yet</Text>
              <Text style={[styles.emptySubtitle, { color: tc.text.muted }]}>Interactive story scenarios will appear here</Text>
            </View>
          ) : (
            <View style={styles.scenarioGrid}>
              {scenarios.map(s => <ScenarioCard key={s.id} scenario={s} />)}
            </View>
          )
        ) : displayCards.length === 0 ? (
          <View style={styles.emptySearch}>
            <Text style={styles.emptyIcon}>{searchQuery.trim() ? '🔍' : selectedGenre ? GENRES.find(g => g.id === selectedGenre)?.emoji ?? '🎭' : '✨'}</Text>
            <Text style={[styles.emptyTitle, { color: tc.text.secondary }]}>{searchQuery.trim() ? 'No characters found' : selectedGenre ? `No ${selectedGenre} characters yet` : 'No characters'}</Text>
            <Text style={[styles.emptySubtitle, { color: tc.text.muted }]}>{searchQuery.trim() ? 'Try a different search term or browse by genre' : selectedGenre ? 'Be the first to create one!' : 'Characters will appear here'}</Text>
          </View>
        ) : (
          <>
            <View style={styles.grid}>{displayCards.map((c, idx) => (
              <Animated.View key={c.id} entering={idx < 20 ? FadeInDown.delay(idx * 50).duration(300).springify().damping(20) : undefined}>
                <CharacterCardComponent card={c} featured={idx === 0 && !searchQuery.trim() && !selectedGenre} />
              </Animated.View>
            ))}</View>
            {isLoadingMore && (
              <View style={{ paddingVertical: spacing.lg, alignItems: 'center' }}>
                <Text style={{ color: tc.text.muted, fontSize: 12 }}>Loading more...</Text>
              </View>
            )}
            {!hasMore && displayCards.length > 0 && (
              <View style={{ paddingVertical: spacing.md, alignItems: 'center' }}>
                <Text style={{ color: tc.text.muted, fontSize: 11 }}>That's everything</Text>
              </View>
            )}
          </>
        )}
        <View style={{ height: spacing.xxl }} />
      </ScrollView>
      {showScrollTop && (
        <TouchableOpacity style={[styles.scrollTopFab, { backgroundColor: tc.bg.elevated, borderColor: tc.border.subtle }]} onPress={() => scrollRef.current?.scrollTo({ y: 0, animated: true })} activeOpacity={0.8} accessibilityLabel="Scroll to top" accessibilityRole="button">
          <ChevronUp size={18} color={tc.text.secondary} />
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  scroll: { flex: 1 },
  brandRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingTop: spacing.sm, gap: spacing.md },
  brandText: { fontSize: 26, fontFamily: fonts.logo, letterSpacing: 2 },
  brandAccent: { fontFamily: fonts.bodyLight, letterSpacing: 0 },
  themeDot: { width: 10, height: 10, borderRadius: 5 },
  searchRow: { flexDirection: 'row', alignItems: 'center', marginHorizontal: spacing.md, marginTop: spacing.sm, marginBottom: spacing.xs, gap: spacing.sm },
  searchBox: { flexDirection: 'row', alignItems: 'center', borderRadius: borderRadius.lg, paddingHorizontal: spacing.md, gap: spacing.sm },
  shuffleBtn: { width: 38, height: 38, borderRadius: borderRadius.md, justifyContent: 'center', alignItems: 'center' },
  searchInput: { flex: 1, paddingVertical: spacing.sm, fontSize: 14 },
  sectionHdr: { paddingHorizontal: spacing.md, marginTop: spacing.lg, marginBottom: spacing.sm },
  sectionTitle: { ...typography.heading, fontFamily: fonts.heading },
  hScroll: { paddingHorizontal: spacing.md },
  genreScroll: { paddingHorizontal: spacing.md, gap: spacing.xs, paddingVertical: spacing.xs },
  genreChip: { paddingHorizontal: spacing.md, paddingVertical: 6, borderRadius: 20, borderWidth: 1 },
  genreText: { fontSize: 12, fontFamily: fonts.button },
  grid: { flexDirection: 'row', flexWrap: 'wrap', paddingHorizontal: spacing.md, gap: spacing.sm },
  scenarioGrid: { paddingHorizontal: spacing.lg, gap: spacing.md },
  emptySearch: { alignItems: 'center', paddingVertical: spacing.xxl * 2, paddingHorizontal: spacing.xxl },
  emptyIcon: { fontSize: 40, marginBottom: spacing.md },
  emptyTitle: { ...typography.subheading, marginBottom: spacing.xs },
  emptySubtitle: { ...typography.caption, textAlign: 'center' },
  gridHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: spacing.md, marginTop: spacing.xs, marginBottom: spacing.xs },
  sortRow: { flexDirection: 'row', gap: 2 },
  sortChip: { paddingHorizontal: spacing.sm, paddingVertical: 3, borderRadius: borderRadius.sm },
  sortText: { fontSize: 10, fontFamily: fonts.bodySemiBold },
  scrollTopFab: { position: 'absolute', right: spacing.lg, bottom: 80, width: 36, height: 36, borderRadius: 18, justifyContent: 'center', alignItems: 'center', borderWidth: 1, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.3, shadowRadius: 4, elevation: 4 },
});
