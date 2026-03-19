import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useThemeStore } from '../../stores/themeStore';
import { fonts } from '../../constants/fonts';

interface WizardStepBarProps {
  steps: string[];
  currentStep: number;
}

export default function WizardStepBar({ steps, currentStep }: WizardStepBarProps) {
  const tc = useThemeStore(s => s.colors);

  return (
    <View style={styles.container}>
      {steps.map((label, i) => {
        const isActive = i === currentStep;
        const isCompleted = i < currentStep;
        const dotColor = isActive
          ? tc.accent.primary
          : isCompleted
            ? tc.accent.primary + '80'
            : tc.bg.tertiary;

        return (
          <View key={label} style={styles.step}>
            <View style={[styles.dot, { backgroundColor: dotColor }]} />
            <Text
              style={[
                styles.label,
                {
                  color: isActive ? tc.accent.primary : tc.text.muted,
                  fontFamily: fonts.body,
                },
              ]}
              numberOfLines={1}
            >
              {label}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 16, paddingVertical: 12 },
  step: { alignItems: 'center', flex: 1 },
  dot: { width: 10, height: 10, borderRadius: 5, marginBottom: 4 },
  label: { fontSize: 9, textAlign: 'center' },
});
