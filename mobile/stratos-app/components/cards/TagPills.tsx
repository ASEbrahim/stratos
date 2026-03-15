import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { GENRE_MAP } from '../../constants/genres';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing, borderRadius } from '../../constants/theme';

interface TagPillsProps { tags: string[]; size?: 'small' | 'medium'; }

export function TagPills({ tags, size = 'small' }: TagPillsProps) {
  const tc = useThemeStore(s => s.colors);
  return (
    <View style={styles.container}>
      {tags.map((tag) => {
        const genre = GENRE_MAP[tag];
        const tagColor = genre?.color ?? tc.accent.primary;
        return (
          <View key={tag} style={[styles.pill, size === 'medium' && styles.pillMedium, { backgroundColor: tagColor + '20', borderColor: tagColor + '40' }]}>
            <Text style={[styles.text, size === 'medium' && styles.textMedium, { color: tagColor }]}>{genre?.label ?? tag}</Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.xs },
  pill: { paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: borderRadius.full, borderWidth: 1 },
  pillMedium: { paddingHorizontal: spacing.md, paddingVertical: spacing.xs },
  text: { ...typography.small },
  textMedium: { ...typography.caption },
});
