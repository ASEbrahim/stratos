import React, { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, withDelay } from 'react-native-reanimated';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing } from '../../constants/theme';

interface EmptyStateProps { title: string; subtitle?: string; icon?: string; }

export function EmptyState({ title, subtitle, icon }: EmptyStateProps) {
  const tc = useThemeStore(s => s.colors);
  const scale = useSharedValue(0.5);
  const opacity = useSharedValue(0);
  const textOpacity = useSharedValue(0);

  useEffect(() => {
    scale.value = withSpring(1, { damping: 12 });
    opacity.value = withSpring(1);
    textOpacity.value = withDelay(200, withSpring(1));
  }, []);

  const iconStyle = useAnimatedStyle(() => ({ transform: [{ scale: scale.value }], opacity: opacity.value }));
  const textStyle = useAnimatedStyle(() => ({ opacity: textOpacity.value }));

  // Auto-pick icon based on title
  const displayIcon = icon ?? (
    title.includes('conversation') ? '💬' :
    title.includes('character') ? '✨' :
    title.includes('saved') ? '📚' :
    '🎭'
  );

  return (
    <View style={styles.container}>
      <Animated.View style={iconStyle}>
        <Text style={styles.icon}>{displayIcon}</Text>
      </Animated.View>
      <Animated.View style={textStyle}>
        <Text style={[styles.title, { color: tc.text.secondary }]}>{title}</Text>
        {subtitle && <Text style={[styles.subtitle, { color: tc.text.muted }]}>{subtitle}</Text>}
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: spacing.xxl },
  icon: { fontSize: 48, marginBottom: spacing.lg, textAlign: 'center' },
  title: { ...typography.heading, textAlign: 'center', marginBottom: spacing.sm },
  subtitle: { ...typography.body, textAlign: 'center', lineHeight: 22 },
});
