import React, { useEffect, useState, useCallback } from 'react';
import { View, Text, ScrollView, TouchableOpacity, TextInput, StyleSheet, RefreshControl } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Search } from 'lucide-react-native';
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
  const displayCards = filterNsfw(searchQuery.trim() ? useCharacterStore.getState().searchResults : (selectedGenre ? newCards.filter(c => c.genre_tags.includes(selectedGenre)) : newCards));

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <StarParallaxBg />
      <ScrollView style={styles.scroll} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={tc.accent.primary} />}>
        <View style={[styles.searchBox, { backgroundColor: tc.bg.tertiary }]}>
          <Search size={18} color={tc.text.muted} />
          <TextInput style={[styles.searchInput, { color: tc.text.primary }]} value={searchQuery} onChangeText={handleSearch} placeholder="Search characters..." placeholderTextColor={tc.text.muted} />
        </View>
        {isLoading && trending.length === 0 && <DiscoverSkeleton />}
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
              {trending.slice(0, 3).map(c => {
                const gc = getGenreColor(c.genre_tags[0] ?? 'default');
                return (
                  <TouchableOpacity key={`pop-${c.id}`} style={[styles.popularCard, { borderColor: gc + '25' }]} onPress={() => router.push(`/character/${c.id}`)} activeOpacity={0.7}>
                    <View style={[styles.popularAvatar, { backgroundColor: gc + '15' }]}>
                      <Text style={[styles.popularLetter, { color: gc }]}>{c.name[0]}</Text>
                    </View>
                    <View style={styles.popularInfo}>
                      <Text style={[styles.popularName, { color: tc.text.primary }]} numberOfLines={1}>{c.name}</Text>
                      <Text style={[styles.popularDesc, { color: tc.text.secondary }]} numberOfLines={2}>{c.description}</Text>
                      <Text style={[styles.popularStat, { color: gc }]}>{formatCount(c.session_count)} chats</Text>
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
          <TouchableOpacity style={[styles.genreChip, { borderColor: tc.border.subtle, backgroundColor: tc.bg.secondary }, !selectedGenre && { backgroundColor: tc.accent.primary + '20', borderColor: tc.accent.primary }]} onPress={() => setGenre(null)}>
            <Text style={[styles.genreText, { color: tc.text.secondary }, !selectedGenre && { color: tc.accent.primary }]}>All</Text>
          </TouchableOpacity>
          {GENRES.map(g => {
            const a = selectedGenre === g.id;
            return (
              <TouchableOpacity key={g.id} style={[styles.genreChip, { borderColor: tc.border.subtle, backgroundColor: tc.bg.secondary }, a && { backgroundColor: g.color + '20', borderColor: g.color }]} onPress={() => setGenre(a ? null : g.id)}>
                <Text style={[styles.genreText, { color: tc.text.secondary }, a && { color: g.color }]}>{g.emoji} {g.label}</Text>
              </TouchableOpacity>
            );
          })}
        </ScrollView>
        <View style={styles.sectionHdr}><Text style={[styles.sectionTitle, { color: tc.text.primary }]}>{searchQuery.trim() ? 'Search Results' : 'New Characters'}</Text></View>
        {displayCards.length === 0 && searchQuery.trim() ? (
          <View style={styles.emptySearch}>
            <Text style={[styles.emptyIcon]}>🔍</Text>
            <Text style={[styles.emptyTitle, { color: tc.text.secondary }]}>No characters found</Text>
            <Text style={[styles.emptySubtitle, { color: tc.text.muted }]}>Try a different search term or browse by genre</Text>
          </View>
        ) : (
          <View style={styles.grid}>{displayCards.map(c => <CharacterCardComponent key={c.id} card={c} />)}</View>
        )}
        <View style={{ height: spacing.xxl }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  scroll: { flex: 1 },
  searchBox: { flexDirection: 'row', alignItems: 'center', borderRadius: borderRadius.lg, paddingHorizontal: spacing.lg, marginHorizontal: spacing.lg, marginTop: spacing.md, marginBottom: spacing.lg, gap: spacing.sm },
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
  recentRow: { flexDirection: 'row', alignItems: 'center', paddingLeft: spacing.lg, marginBottom: spacing.sm },
  recentLabel: { ...typography.small, fontSize: 10, fontWeight: '600', textTransform: 'uppercase', letterSpacing: 1, marginRight: spacing.sm },
  recentScroll: { gap: spacing.md, paddingRight: spacing.lg },
  recentItem: { alignItems: 'center', width: 52 },
  recentAvatar: { width: 40, height: 40, borderRadius: 20, justifyContent: 'center', alignItems: 'center', borderWidth: 1.5, marginBottom: 3 },
  recentLetter: { fontSize: 16, fontWeight: '700' },
  recentName: { fontSize: 9, fontWeight: '500', textAlign: 'center' },
  popularScroll: { paddingHorizontal: spacing.lg, gap: spacing.md },
  popularCard: { width: 280, flexDirection: 'row', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: borderRadius.lg, padding: spacing.md, borderWidth: 1, gap: spacing.md },
  popularAvatar: { width: 56, height: 56, borderRadius: borderRadius.md, justifyContent: 'center', alignItems: 'center' },
  popularLetter: { fontSize: 22, fontWeight: '700', opacity: 0.7 },
  popularInfo: { flex: 1, justifyContent: 'center', gap: 2 },
  popularName: { ...typography.subheading, fontSize: 14 },
  popularDesc: { ...typography.small, lineHeight: 15, fontSize: 10 },
  popularStat: { ...typography.small, fontSize: 10, fontWeight: '600', marginTop: 2 },
  welcomeCard: { marginHorizontal: spacing.lg, marginBottom: spacing.md, padding: spacing.lg, borderRadius: borderRadius.lg, borderWidth: 1 },
  welcomeTitle: { ...typography.subheading, marginBottom: spacing.xs },
  welcomeBody: { ...typography.caption, lineHeight: 18 },
});
