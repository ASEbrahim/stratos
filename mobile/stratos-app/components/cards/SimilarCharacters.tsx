import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { CharacterCard } from '../../lib/types';
import { getGenreColor } from '../../constants/genres';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing, borderRadius } from '../../constants/theme';

interface SimilarCharactersProps {
  similarCards: CharacterCard[];
  onNavigate: (id: string) => void;
}

export const SimilarCharacters = React.memo(function SimilarCharacters({
  similarCards,
  onNavigate,
}: SimilarCharactersProps) {
  const tc = useThemeStore(s => s.colors);

  if (similarCards.length === 0) return null;

  return (
    <View style={[styles.section, { marginTop: spacing.lg }]}>
      <Text style={[styles.sectionTitle, { color: tc.text.primary }]}>Similar Characters</Text>
      <View style={styles.similarRow}>
        {similarCards.map(c => {
          const gc = getGenreColor(c.genre_tags[0] ?? 'default');
          return (
            <TouchableOpacity key={c.id} style={styles.similarCard} onPress={() => onNavigate(c.id)} activeOpacity={0.7} accessibilityLabel={`View similar character ${c.name}`} accessibilityRole="button">
              <View style={[styles.similarAvatar, { backgroundColor: gc + '15' }]}>
                <Text style={[styles.similarLetter, { color: gc }]}>{c.name[0]}</Text>
              </View>
              <Text style={[styles.similarName, { color: tc.text.primary }]} numberOfLines={1}>{c.name}</Text>
              <Text style={[styles.similarGenre, { color: gc }]}>{c.genre_tags[0]}</Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
});

const styles = StyleSheet.create({
  section: { marginBottom: spacing.xl },
  sectionTitle: { ...typography.subheading, marginBottom: spacing.sm },
  similarRow: { flexDirection: 'row', gap: spacing.md },
  similarCard: { flex: 1, alignItems: 'center' },
  similarAvatar: { width: 50, height: 50, borderRadius: borderRadius.md, justifyContent: 'center', alignItems: 'center', marginBottom: spacing.xs },
  similarLetter: { fontSize: 20, fontWeight: '700', opacity: 0.7 },
  similarName: { fontSize: 11, fontWeight: '600', textAlign: 'center' },
  similarGenre: { fontSize: 9, textTransform: 'capitalize', marginTop: 2 },
});
