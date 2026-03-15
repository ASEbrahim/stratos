import React, { useEffect, useState, useCallback } from 'react';
import { View, Text, ScrollView, TouchableOpacity, TextInput, StyleSheet, RefreshControl } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Search } from 'lucide-react-native';
import { useCharacterStore } from '../../stores/characterStore';
import { useThemeStore } from '../../stores/themeStore';
import { CharacterCardComponent } from '../../components/cards/CharacterCard';
import { ScenarioCard } from '../../components/gaming/ScenarioCard';
import { StarParallaxBg } from '../../components/shared/StarParallax';
import { DiscoverSkeleton } from '../../components/shared/Skeleton';
import { GENRES } from '../../constants/genres';
import { GamingScenario } from '../../lib/types';
import { getScenarios } from '../../lib/gaming';
import { typography, spacing, borderRadius } from '../../constants/theme';

export default function DiscoverScreen() {
  const insets = useSafeAreaInsets();
  const tc = useThemeStore(s => s.colors);
  const { trending, newCards, selectedGenre, isLoading, loadTrending, loadNew, setGenre, search } = useCharacterStore();
  const [scenarios, setScenarios] = useState<GamingScenario[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);

  useEffect(() => { loadTrending(); loadNew(); getScenarios().then(setScenarios); }, []);
  const onRefresh = useCallback(async () => { setRefreshing(true); await Promise.all([loadTrending(), loadNew(), getScenarios().then(setScenarios)]); setRefreshing(false); }, []);
  const searchTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const handleSearch = (text: string) => {
    setSearchQuery(text);
    if (searchTimer.current) clearTimeout(searchTimer.current);
    searchTimer.current = setTimeout(() => {
      if (text.trim()) search(text.trim(), selectedGenre ?? undefined);
    }, 300);
  };
  const displayCards = searchQuery.trim() ? useCharacterStore.getState().searchResults : (selectedGenre ? newCards.filter(c => c.genre_tags.includes(selectedGenre)) : newCards);

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <StarParallaxBg />
      <ScrollView style={styles.scroll} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={tc.accent.primary} />}>
        <View style={[styles.searchBox, { backgroundColor: tc.bg.tertiary }]}>
          <Search size={18} color={tc.text.muted} />
          <TextInput style={[styles.searchInput, { color: tc.text.primary }]} value={searchQuery} onChangeText={handleSearch} placeholder="Search characters..." placeholderTextColor={tc.text.muted} />
        </View>
        {isLoading && trending.length === 0 && <DiscoverSkeleton />}
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
  welcomeCard: { marginHorizontal: spacing.lg, marginBottom: spacing.md, padding: spacing.lg, borderRadius: borderRadius.lg, borderWidth: 1 },
  welcomeTitle: { ...typography.subheading, marginBottom: spacing.xs },
  welcomeBody: { ...typography.caption, lineHeight: 18 },
});
