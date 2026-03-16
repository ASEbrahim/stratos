import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { CharacterCard } from '../../lib/types';
import { QualityScore } from './QualityScore';
import { useThemeStore } from '../../stores/themeStore';
import type { ThemeColors } from '../../constants/themes';
import { typography, spacing, borderRadius } from '../../constants/theme';

interface CharacterDepthSectionProps {
  card: CharacterCard;
  showDepth: boolean;
  accentColor: string;
  onToggleDepth: () => void;
}

function DepthItem({ label, value, tc }: { label: string; value: string; tc: ThemeColors }) {
  return (
    <View style={{ marginBottom: spacing.md }}>
      <Text style={{ fontSize: 10, fontWeight: '700', color: tc.text.muted, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 3 }}>{label}</Text>
      <Text style={{ fontSize: 13, color: tc.text.secondary, lineHeight: 19 }}>{value}</Text>
    </View>
  );
}

export const CharacterDepthSection = React.memo(function CharacterDepthSection({
  card,
  showDepth,
  accentColor,
  onToggleDepth,
}: CharacterDepthSectionProps) {
  const tc = useThemeStore(s => s.colors);
  const hasDepth = !!(card.speech_pattern || card.emotional_trigger || card.defensive_mechanism || card.vulnerability || card.specific_detail || card.physical_description);

  return (
    <View style={styles.section}>
      <Text style={[styles.sectionTitle, { color: tc.text.primary }]}>Quality Elements</Text>
      <QualityScore card={card} showElements size="large" />
      {hasDepth && (
        <TouchableOpacity style={[styles.depthToggle, { borderColor: accentColor + '30' }]} onPress={onToggleDepth} activeOpacity={0.7}>
          <Text style={[styles.depthToggleText, { color: accentColor }]}>{showDepth ? 'Hide Details' : 'Show Character Depth'}</Text>
        </TouchableOpacity>
      )}
      {showDepth && (
        <View style={styles.depthGrid}>
          {card.physical_description && <DepthItem label="Appearance" value={card.physical_description} tc={tc} />}
          {card.speech_pattern && <DepthItem label="Speech Pattern" value={card.speech_pattern} tc={tc} />}
          {card.emotional_trigger && <DepthItem label="Emotional Triggers" value={card.emotional_trigger} tc={tc} />}
          {card.defensive_mechanism && <DepthItem label="Defenses" value={card.defensive_mechanism} tc={tc} />}
          {card.vulnerability && <DepthItem label="Vulnerability" value={card.vulnerability} tc={tc} />}
          {card.specific_detail && <DepthItem label="Signature Detail" value={card.specific_detail} tc={tc} />}
        </View>
      )}
    </View>
  );
});

const styles = StyleSheet.create({
  section: { marginBottom: spacing.xl },
  sectionTitle: { ...typography.subheading, marginBottom: spacing.sm },
  depthToggle: { paddingVertical: spacing.sm, marginTop: spacing.sm, borderRadius: borderRadius.sm, borderWidth: 1, alignItems: 'center' },
  depthToggleText: { fontSize: 12, fontWeight: '600' },
  depthGrid: { marginTop: spacing.md },
});
