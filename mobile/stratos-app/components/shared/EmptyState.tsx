import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { colors, typography, spacing } from '../../constants/theme';

interface EmptyStateProps { title: string; subtitle?: string; }

export function EmptyState({ title, subtitle }: EmptyStateProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>{title}</Text>
      {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: spacing.xxl },
  title: { ...typography.heading, color: colors.text.secondary, textAlign: 'center', marginBottom: spacing.sm },
  subtitle: { ...typography.body, color: colors.text.muted, textAlign: 'center' },
});
