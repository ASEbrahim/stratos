import React, { useEffect, useState, useCallback } from 'react';
import { View, Text, ScrollView, TouchableOpacity, TextInput, StyleSheet, RefreshControl } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Search } from 'lucide-react-native';
import { useCharacterStore } from '../../stores/characterStore';
import { CharacterCardComponent } from '../../components/cards/CharacterCard';
import { ScenarioCard } from '../../components/gaming/ScenarioCard';
import { StarParallaxBg } from '../../components/shared/StarParallax';
import { GENRES } from '../../constants/genres';
import { GamingScenario } from '../../lib/types';
import { getScenarios } from '../../lib/gaming';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

export default function DiscoverScreen() {
  const insets = useSafeAreaInsets();
  const { trending, newCards, selectedGenre, loadTrending, loadNew, setGenre, search } = useCharacterStore();
  const [scenarios, setScenarios] = useState<GamingScenario[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => { loadTrending(); loadNew(); getScenarios().then(setScenarios); }, []);
  const onRefresh = useCallback(async () => { setRefreshing(true); await Promise.all([loadTrending(), loadNew(), getScenarios().then(setScenarios)]); setRefreshing(false); }, []);
  const handleSearch = (text: string) => { setSearchQuery(text); if (text.trim()) search(text.trim(), selectedGenre ?? undefined); };
  const displayCards = searchQuery.trim() ? useCharacterStore.getState().searchResults : (selectedGenre ? newCards.filter(c => c.genre_tags.includes(selectedGenre)) : newCards);

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <StarParallaxBg />
      <ScrollView style={styles.scroll} refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.accent.primary} />}>
        <View style={styles.searchBox}><Search size={18} color={colors.text.muted} /><TextInput style={styles.searchInput} value={searchQuery} onChangeText={handleSearch} placeholder="Search characters..." placeholderTextColor={colors.text.muted} /></View>
        {!searchQuery.trim() && (<><View style={styles.sectionHdr}><Text style={styles.sectionTitle}>Trending</Text></View><ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.hScroll}>{trending.slice(0, 10).map(c => <CharacterCardComponent key={c.id} card={c} variant="horizontal" />)}</ScrollView></>)}
        {!searchQuery.trim() && scenarios.length > 0 && (<><View style={styles.sectionHdr}><Text style={styles.sectionTitle}>Gaming Scenarios</Text></View><ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.hScroll}>{scenarios.map(s => <ScenarioCard key={s.id} scenario={s} variant="horizontal" />)}</ScrollView></>)}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.genreScroll}>
          <TouchableOpacity style={[styles.genreChip, !selectedGenre && styles.genreActive]} onPress={() => setGenre(null)}><Text style={[styles.genreText, !selectedGenre && styles.genreTextActive]}>All</Text></TouchableOpacity>
          {GENRES.map(g => { const a = selectedGenre === g.id; return (<TouchableOpacity key={g.id} style={[styles.genreChip, a && { backgroundColor: g.color + '20', borderColor: g.color }]} onPress={() => setGenre(a ? null : g.id)}><Text style={[styles.genreText, a && { color: g.color }]}>{g.emoji} {g.label}</Text></TouchableOpacity>); })}
        </ScrollView>
        <View style={styles.sectionHdr}><Text style={styles.sectionTitle}>{searchQuery.trim() ? 'Search Results' : 'New Characters'}</Text></View>
        <View style={styles.grid}>{displayCards.map(c => <CharacterCardComponent key={c.id} card={c} />)}</View>
        <View style={{ height: spacing.xxl }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg.primary },
  scroll: { flex: 1 },
  searchBox: { flexDirection: 'row', alignItems: 'center', backgroundColor: colors.bg.tertiary, borderRadius: borderRadius.lg, paddingHorizontal: spacing.lg, marginHorizontal: spacing.lg, marginTop: spacing.md, marginBottom: spacing.lg, gap: spacing.sm },
  searchInput: { flex: 1, paddingVertical: spacing.md, color: colors.text.primary, fontSize: 15 },
  sectionHdr: { paddingHorizontal: spacing.lg, marginTop: spacing.lg, marginBottom: spacing.md },
  sectionTitle: { ...typography.heading, color: colors.text.primary },
  hScroll: { paddingHorizontal: spacing.lg },
  genreScroll: { paddingHorizontal: spacing.lg, gap: spacing.sm, paddingVertical: spacing.md },
  genreChip: { paddingHorizontal: spacing.md, paddingVertical: spacing.sm, borderRadius: borderRadius.full, borderWidth: 1, borderColor: colors.border.subtle, backgroundColor: colors.bg.secondary },
  genreActive: { backgroundColor: colors.accent.primary + '20', borderColor: colors.accent.primary },
  genreText: { ...typography.caption, color: colors.text.secondary },
  genreTextActive: { color: colors.accent.primary },
  grid: { flexDirection: 'row', flexWrap: 'wrap', paddingHorizontal: spacing.lg, gap: spacing.lg },
});
