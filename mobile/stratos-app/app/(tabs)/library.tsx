import React, { useEffect, useState } from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useCharacterStore } from '../../stores/characterStore';
import { CharacterCardComponent } from '../../components/cards/CharacterCard';
import { EmptyState } from '../../components/shared/EmptyState';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

export default function LibraryScreen() {
  const insets = useSafeAreaInsets();
  const { myCards, savedCards, loadMyCards, loadSaved } = useCharacterStore();
  const [tab, setTab] = useState<'mine' | 'saved'>('mine');
  useEffect(() => { loadMyCards(); loadSaved(); }, []);
  const cards = tab === 'mine' ? myCards : savedCards;
  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <Text style={styles.title}>Library</Text>
      <View style={styles.tabs}>
        <TouchableOpacity style={[styles.tab, tab === 'mine' && styles.tabActive]} onPress={() => setTab('mine')}><Text style={[styles.tabText, tab === 'mine' && styles.tabTextActive]}>My Characters</Text></TouchableOpacity>
        <TouchableOpacity style={[styles.tab, tab === 'saved' && styles.tabActive]} onPress={() => setTab('saved')}><Text style={[styles.tabText, tab === 'saved' && styles.tabTextActive]}>Saved</Text></TouchableOpacity>
      </View>
      {cards.length === 0 ? <EmptyState title={tab === 'mine' ? 'No characters yet' : 'No saved characters'} subtitle={tab === 'mine' ? 'Create your first character!' : 'Browse and save from Discover.'} /> : (
        <ScrollView contentContainerStyle={styles.grid}>{cards.map(c => <CharacterCardComponent key={c.id} card={c} />)}</ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg.primary },
  title: { ...typography.display, color: colors.text.primary, paddingHorizontal: spacing.lg, paddingTop: spacing.lg, paddingBottom: spacing.md },
  tabs: { flexDirection: 'row', paddingHorizontal: spacing.lg, gap: spacing.md, marginBottom: spacing.lg },
  tab: { paddingVertical: spacing.sm, paddingHorizontal: spacing.lg, borderRadius: borderRadius.full, backgroundColor: colors.bg.tertiary },
  tabActive: { backgroundColor: colors.accent.primary + '20' },
  tabText: { ...typography.subheading, fontSize: 14, color: colors.text.muted },
  tabTextActive: { color: colors.accent.primary },
  grid: { flexDirection: 'row', flexWrap: 'wrap', paddingHorizontal: spacing.lg, gap: spacing.lg, paddingBottom: spacing.xxl },
});
