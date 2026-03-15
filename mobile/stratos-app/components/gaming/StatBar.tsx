import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { colors, typography, spacing } from '../../constants/theme';

interface StatBarProps { stats: Record<string, number>; accentColor?: string; }

export function StatBar({ stats, accentColor }: StatBarProps) {
  const color = accentColor ?? colors.accent.primary;
  return (
    <View style={styles.container}>
      {Object.entries(stats).map(([key, value]) => (
        <View key={key} style={styles.stat}>
          <Text style={styles.label}>{key.toUpperCase()}</Text>
          <Text style={[styles.value, key.toLowerCase() === 'hp' ? { color } : null]}>{value}</Text>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flexDirection: 'row', backgroundColor: colors.bg.secondary, borderBottomWidth: 1, borderBottomColor: colors.border.subtle, paddingVertical: spacing.sm, paddingHorizontal: spacing.lg, gap: spacing.lg },
  stat: { alignItems: 'center' },
  label: { ...typography.small, color: colors.text.muted, letterSpacing: 1 },
  value: { ...typography.subheading, color: colors.text.primary },
});
