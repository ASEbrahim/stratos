import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { Star } from 'lucide-react-native';
import { GamingScenario, formatCount } from '../../lib/types';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { getGenreColor } from '../../constants/genres';

interface ScenarioCardProps { scenario: GamingScenario; variant?: 'horizontal' | 'full'; }

export function ScenarioCard({ scenario, variant = 'full' }: ScenarioCardProps) {
  const router = useRouter();
  const tc = useThemeStore(s => s.colors);
  const accentColor = getGenreColor(scenario.genre);
  if (variant === 'horizontal') {
    return (
      <TouchableOpacity style={[styles.horizontalCard, { borderColor: accentColor + '30', backgroundColor: tc.bg.secondary }]} onPress={() => router.push(`/gaming/${scenario.id}`)} activeOpacity={0.7}>
        <View style={[styles.horizontalIcon, { backgroundColor: accentColor + '15' }]}>
          <Text style={styles.iconText}>{scenario.genre === 'fantasy' ? '⚔️' : scenario.genre === 'scifi' ? '🚀' : '🏚️'}</Text>
        </View>
        <Text style={[styles.horizontalName, { color: tc.text.primary }]} numberOfLines={1}>{scenario.name}</Text>
        <Text style={[styles.horizontalGenre, { color: accentColor }]}>{scenario.subgenre}</Text>
      </TouchableOpacity>
    );
  }
  return (
    <TouchableOpacity style={[styles.card, { backgroundColor: tc.bg.secondary, borderColor: tc.border.subtle }]} onPress={() => router.push(`/gaming/${scenario.id}`)} activeOpacity={0.7}>
      <View style={[styles.cardHeader, { borderLeftColor: accentColor }]}>
        <Text style={[styles.cardName, { color: tc.text.primary }]}>{scenario.name}</Text>
        <View style={styles.cardMeta}>
          <Text style={[styles.genreLabel, { color: accentColor }]}>{scenario.genre} · {scenario.subgenre}</Text>
          <View style={styles.ratingRow}><Star size={12} color={tc.accent.secondary} fill={tc.accent.secondary} /><Text style={[styles.ratingText, { color: tc.text.secondary }]}>{scenario.rating.toFixed(1)}</Text></View>
        </View>
      </View>
      <Text style={[styles.description, { color: tc.text.secondary }]} numberOfLines={2}>{scenario.description}</Text>
      <Text style={[styles.sessions, { color: tc.text.muted }]}>{formatCount(scenario.session_count)} sessions</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: { borderRadius: borderRadius.lg, padding: spacing.lg, borderWidth: 1, gap: spacing.sm },
  cardHeader: { borderLeftWidth: 3, paddingLeft: spacing.md },
  cardName: { ...typography.heading, marginBottom: spacing.xs },
  cardMeta: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  genreLabel: { ...typography.caption, textTransform: 'capitalize' },
  ratingRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  ratingText: { ...typography.caption },
  description: { ...typography.body },
  sessions: { ...typography.small },
  horizontalCard: { width: 140, alignItems: 'center', marginRight: spacing.md, borderRadius: borderRadius.lg, padding: spacing.md, borderWidth: 1 },
  horizontalIcon: { width: 50, height: 50, borderRadius: borderRadius.md, justifyContent: 'center', alignItems: 'center', marginBottom: spacing.sm },
  iconText: { fontSize: 24 },
  horizontalName: { ...typography.caption, fontWeight: '600', textAlign: 'center', marginBottom: 2 },
  horizontalGenre: { ...typography.small },
});
