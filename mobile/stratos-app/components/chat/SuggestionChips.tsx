import React from 'react';
import { ScrollView, TouchableOpacity, Text, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import { Suggestion } from '../../lib/types';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

interface SuggestionChipsProps { suggestions: Suggestion[]; onSelect: (prompt: string) => void; accentColor?: string; }

export function SuggestionChips({ suggestions, onSelect, accentColor }: SuggestionChipsProps) {
  if (suggestions.length === 0) return null;
  const color = accentColor ?? colors.accent.primary;
  return (
    <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
      {suggestions.map((s, i) => (
        <TouchableOpacity key={i} style={[styles.chip, { borderColor: color + '40', backgroundColor: color + '10' }]} onPress={() => { Haptics.selectionAsync(); onSelect(s.prompt); }} activeOpacity={0.7}>
          <Text style={[styles.chipText, { color }]} numberOfLines={1}>{s.label}</Text>
        </TouchableOpacity>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { paddingHorizontal: spacing.lg, paddingVertical: spacing.sm, gap: spacing.sm },
  chip: { paddingHorizontal: spacing.md, paddingVertical: spacing.sm, borderRadius: borderRadius.full, borderWidth: 1 },
  chipText: { ...typography.caption, fontWeight: '500' },
});
