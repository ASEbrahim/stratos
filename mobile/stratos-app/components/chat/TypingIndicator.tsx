import React, { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Animated, { useSharedValue, useAnimatedStyle, withRepeat, withTiming, withDelay, withSequence, FadeIn } from 'react-native-reanimated';
import { useThemeStore } from '../../stores/themeStore';
import { spacing, borderRadius, typography } from '../../constants/theme';

interface TypingIndicatorProps { characterName?: string; }

export function TypingIndicator({ characterName }: TypingIndicatorProps) {
  const tc = useThemeStore(s => s.colors);
  const dot1 = useSharedValue(0);
  const dot2 = useSharedValue(0);
  const dot3 = useSharedValue(0);
  useEffect(() => {
    const anim = (delay: number) => withDelay(delay, withRepeat(withSequence(withTiming(1, { duration: 400 }), withTiming(0, { duration: 400 })), -1));
    dot1.value = anim(0); dot2.value = anim(150); dot3.value = anim(300);
  }, []);
  const s1 = useAnimatedStyle(() => ({ opacity: 0.3 + dot1.value * 0.7, transform: [{ translateY: -dot1.value * 4 }] }));
  const s2 = useAnimatedStyle(() => ({ opacity: 0.3 + dot2.value * 0.7, transform: [{ translateY: -dot2.value * 4 }] }));
  const s3 = useAnimatedStyle(() => ({ opacity: 0.3 + dot3.value * 0.7, transform: [{ translateY: -dot3.value * 4 }] }));
  return (
    <Animated.View entering={FadeIn.duration(200)} style={styles.container}>
      {characterName && <Text style={[styles.label, { color: tc.text.muted }]}>{characterName} is typing</Text>}
      <View style={[styles.bubble, { backgroundColor: tc.bg.tertiary }]}>
        <Animated.View style={[styles.dot, { backgroundColor: tc.text.muted }, s1]} />
        <Animated.View style={[styles.dot, { backgroundColor: tc.text.muted }, s2]} />
        <Animated.View style={[styles.dot, { backgroundColor: tc.text.muted }, s3]} />
      </View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: { paddingHorizontal: spacing.lg, paddingVertical: spacing.sm },
  label: { ...typography.small, fontSize: 10, marginBottom: 4, marginLeft: spacing.xs },
  bubble: { flexDirection: 'row', alignItems: 'center', alignSelf: 'flex-start', gap: spacing.xs, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, borderRadius: borderRadius.lg, borderTopLeftRadius: borderRadius.sm },
  dot: { width: 8, height: 8, borderRadius: 4 },
});
