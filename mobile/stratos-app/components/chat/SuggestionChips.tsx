import React, { useEffect } from 'react';
import { ScrollView, TouchableOpacity, Text, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import Animated, { useSharedValue, useAnimatedStyle, withDelay, withTiming, withSpring } from 'react-native-reanimated';
import { Suggestion } from '../../lib/types';
import { colors, typography, spacing, borderRadius } from '../../constants/theme';

interface SuggestionChipsProps { suggestions: Suggestion[]; onSelect: (prompt: string) => void; accentColor?: string; }

function AnimatedChip({ suggestion, index, color, onSelect }: { suggestion: Suggestion; index: number; color: string; onSelect: (p: string) => void }) {
  const opacity = useSharedValue(0);
  const translateY = useSharedValue(8);
  const scale = useSharedValue(1);

  useEffect(() => {
    opacity.value = withDelay(index * 100, withTiming(1, { duration: 250 }));
    translateY.value = withDelay(index * 100, withSpring(0, { damping: 12 }));
  }, []);

  const animStyle = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [{ translateY: translateY.value }, { scale: scale.value }],
  }));

  const handlePress = () => {
    scale.value = withSpring(0.9, { damping: 15 }, () => { scale.value = withSpring(1, { damping: 10 }); });
    Haptics.selectionAsync();
    onSelect(suggestion.prompt);
  };

  return (
    <Animated.View style={animStyle}>
      <TouchableOpacity style={[styles.chip, { borderColor: color + '40', backgroundColor: color + '10' }]} onPress={handlePress} activeOpacity={0.7}>
        <Text style={[styles.chipText, { color }]} numberOfLines={1}>{suggestion.label}</Text>
      </TouchableOpacity>
    </Animated.View>
  );
}

export function SuggestionChips({ suggestions, onSelect, accentColor }: SuggestionChipsProps) {
  if (suggestions.length === 0) return null;
  const color = accentColor ?? colors.accent.primary;
  return (
    <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.container} keyboardShouldPersistTaps="handled">
      {suggestions.map((s, i) => (
        <AnimatedChip key={`${s.label}-${i}`} suggestion={s} index={i} color={color} onSelect={onSelect} />
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { paddingHorizontal: spacing.lg, paddingVertical: spacing.sm, gap: spacing.sm },
  chip: { paddingHorizontal: spacing.md, paddingVertical: spacing.sm, borderRadius: borderRadius.full, borderWidth: 1 },
  chipText: { ...typography.caption, fontWeight: '500' },
});
