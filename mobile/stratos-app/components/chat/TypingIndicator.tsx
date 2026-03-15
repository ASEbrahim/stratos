import React, { useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import Animated, { useSharedValue, useAnimatedStyle, withRepeat, withTiming, withDelay, withSequence } from 'react-native-reanimated';
import { colors, spacing, borderRadius } from '../../constants/theme';

export function TypingIndicator() {
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
    <View style={styles.container}><View style={styles.bubble}>
      <Animated.View style={[styles.dot, s1]} /><Animated.View style={[styles.dot, s2]} /><Animated.View style={[styles.dot, s3]} />
    </View></View>
  );
}

const styles = StyleSheet.create({
  container: { flexDirection: 'row', paddingHorizontal: spacing.lg, paddingVertical: spacing.sm },
  bubble: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs, backgroundColor: colors.bg.tertiary, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, borderRadius: borderRadius.lg, borderTopLeftRadius: borderRadius.sm },
  dot: { width: 8, height: 8, borderRadius: 4, backgroundColor: colors.text.muted },
});
