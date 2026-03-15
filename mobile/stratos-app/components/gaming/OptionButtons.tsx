import React, { useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import Animated, { useSharedValue, useAnimatedStyle, withDelay, withSpring, FadeIn } from 'react-native-reanimated';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing, borderRadius } from '../../constants/theme';

interface OptionButtonsProps { options: string[]; onSelect: (option: string, index: number) => void; disabled?: boolean; accentColor?: string; }

function AnimatedOption({ option, index, color, onSelect, disabled }: { option: string; index: number; color: string; onSelect: (o: string, i: number) => void; disabled: boolean }) {
  const tc = useThemeStore(s => s.colors);
  const scale = useSharedValue(0.95);
  const opacity = useSharedValue(0);

  useEffect(() => {
    opacity.value = withDelay(index * 80, withSpring(1));
    scale.value = withDelay(index * 80, withSpring(1, { damping: 12 }));
  }, []);

  const animStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [{ scale: scale.value }],
  }));

  return (
    <Animated.View style={animStyle}>
      <TouchableOpacity
        style={[styles.button, { borderColor: color + '30', backgroundColor: tc.bg.secondary }]}
        onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium); onSelect(option, index); }}
        disabled={disabled}
        activeOpacity={0.7}
      >
        <View style={[styles.number, { backgroundColor: color + '20', borderColor: color + '40', borderWidth: 1 }]}>
          <Text style={[styles.numberText, { color }]}>{index + 1}</Text>
        </View>
        <Text style={[styles.optionText, { color: tc.text.primary }]}>{option}</Text>
      </TouchableOpacity>
    </Animated.View>
  );
}

export function OptionButtons({ options, onSelect, disabled = false, accentColor }: OptionButtonsProps) {
  if (options.length === 0) return null;
  const tc = useThemeStore(s => s.colors);
  const color = accentColor ?? tc.accent.primary;
  return (
    <View style={styles.container}>
      {options.map((option, index) => (
        <AnimatedOption key={index} option={option} index={index} color={color} onSelect={onSelect} disabled={disabled} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { paddingHorizontal: spacing.lg, paddingVertical: spacing.sm, gap: spacing.sm },
  button: { flexDirection: 'row', alignItems: 'center', borderRadius: borderRadius.lg, borderWidth: 1, padding: spacing.md, gap: spacing.md },
  number: { width: 28, height: 28, borderRadius: 14, justifyContent: 'center', alignItems: 'center' },
  numberText: { ...typography.subheading, fontSize: 14 },
  optionText: { ...typography.body, flex: 1 },
});
