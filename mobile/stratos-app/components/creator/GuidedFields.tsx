import React from 'react';
import { View, Text, TextInput, StyleSheet } from 'react-native';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

interface QualityField { key: string; label: string; hint: string; example: string; value: string; onChangeText: (text: string) => void; }
interface GuidedFieldsProps { fields: QualityField[]; }

export function GuidedFields({ fields }: GuidedFieldsProps) {
  return (
    <View style={styles.container}>
      {fields.map((field, index) => {
        const filled = !!field.value.trim();
        return (
          <View key={field.key} style={styles.fieldContainer}>
            <View style={styles.labelRow}>
              <Text style={[styles.indicator, { color: filled ? colors.status.success : colors.text.muted }]}>{filled ? '✓' : `${index + 1}`}</Text>
              <Text style={styles.label}>{field.label}</Text>
            </View>
            <Text style={styles.hint}>{field.hint}</Text>
            <TextInput style={styles.input} value={field.value} onChangeText={field.onChangeText} placeholder={field.example} placeholderTextColor={colors.text.muted + '80'} multiline textAlignVertical="top" />
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { gap: spacing.lg },
  fieldContainer: { gap: spacing.xs },
  labelRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  indicator: { ...typography.subheading, fontSize: 14, width: 20, textAlign: 'center' },
  label: { ...typography.subheading, color: colors.text.primary },
  hint: { ...typography.caption, color: colors.text.secondary, marginLeft: 28 },
  input: { backgroundColor: colors.bg.tertiary, borderRadius: borderRadius.md, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, color: colors.text.primary, fontSize: 14, minHeight: 60, borderWidth: 1, borderColor: colors.border.subtle },
});
