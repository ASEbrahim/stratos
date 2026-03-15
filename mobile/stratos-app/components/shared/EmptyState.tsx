import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing } from '../../constants/theme';

interface EmptyStateProps { title: string; subtitle?: string; }

export function EmptyState({ title, subtitle }: EmptyStateProps) {
  const tc = useThemeStore(s => s.colors);
  return (
    <View style={styles.container}>
      <Text style={[styles.title, { color: tc.text.secondary }]}>{title}</Text>
      {subtitle && <Text style={[styles.subtitle, { color: tc.text.muted }]}>{subtitle}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: spacing.xxl },
  title: { ...typography.heading, textAlign: 'center', marginBottom: spacing.sm },
  subtitle: { ...typography.body, textAlign: 'center' },
});
