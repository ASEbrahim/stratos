import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { CharacterCard, getQualityScore } from '../../lib/types';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

interface QualityScoreProps { card: CharacterCard; showElements?: boolean; size?: 'small' | 'large'; }

export function QualityScore({ card, showElements = false, size = 'small' }: QualityScoreProps) {
  const quality = getQualityScore(card);
  const color = colors.quality[quality.level];
  if (size === 'small') {
    return (
      <View style={[styles.badge, { borderColor: color + '60' }]}>
        <Text style={[styles.stars, { color }]}>{'★'.repeat(Math.min(quality.score, 5))}{'☆'.repeat(Math.max(0, 5 - quality.score))}</Text>
      </View>
    );
  }
  return (
    <View style={styles.largeContainer}>
      <View style={[styles.labelBadge, { backgroundColor: color + '20', borderColor: color + '40' }]}>
        <Text style={[styles.labelText, { color }]}>{quality.label}</Text>
      </View>
      {showElements && (
        <View style={styles.elements}>
          {quality.elements.map((el) => (
            <View key={el.name} style={styles.element}>
              <Text style={[styles.elementCheck, { color: el.filled ? colors.status.success : colors.text.muted }]}>{el.filled ? '✓' : '✗'}</Text>
              <Text style={[styles.elementName, { color: el.filled ? colors.text.primary : colors.text.muted }]}>{el.name}</Text>
            </View>
          ))}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  badge: { borderWidth: 1, borderRadius: borderRadius.sm, paddingHorizontal: spacing.xs, paddingVertical: 1 },
  stars: { fontSize: 10, letterSpacing: 1 },
  largeContainer: { gap: spacing.md },
  labelBadge: { alignSelf: 'flex-start', borderWidth: 1, borderRadius: borderRadius.full, paddingHorizontal: spacing.md, paddingVertical: spacing.xs },
  labelText: { ...typography.subheading, fontSize: 14 },
  elements: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  element: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs, width: '45%' as any },
  elementCheck: { ...typography.body, fontSize: 14 },
  elementName: { ...typography.caption },
});
