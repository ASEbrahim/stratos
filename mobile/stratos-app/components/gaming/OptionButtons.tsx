import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

interface OptionButtonsProps { options: string[]; onSelect: (option: string, index: number) => void; disabled?: boolean; accentColor?: string; }

export function OptionButtons({ options, onSelect, disabled = false, accentColor }: OptionButtonsProps) {
  if (options.length === 0) return null;
  const color = accentColor ?? colors.accent.primary;
  return (
    <View style={styles.container}>
      {options.map((option, index) => (
        <TouchableOpacity key={index} style={[styles.button, { borderColor: color + '40' }]} onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium); onSelect(option, index); }} disabled={disabled} activeOpacity={0.7}>
          <View style={[styles.number, { backgroundColor: color + '20' }]}><Text style={[styles.numberText, { color }]}>{index + 1}</Text></View>
          <Text style={styles.optionText}>{option}</Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { paddingHorizontal: spacing.lg, paddingVertical: spacing.sm, gap: spacing.sm },
  button: { flexDirection: 'row', alignItems: 'center', backgroundColor: colors.bg.secondary, borderRadius: borderRadius.lg, borderWidth: 1, padding: spacing.md, gap: spacing.md },
  number: { width: 28, height: 28, borderRadius: 14, justifyContent: 'center', alignItems: 'center' },
  numberText: { ...typography.subheading, fontSize: 14 },
  optionText: { ...typography.body, color: colors.text.primary, flex: 1 },
});
