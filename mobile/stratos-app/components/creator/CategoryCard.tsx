import React, { useCallback, useState } from 'react';
import { View, Text, Pressable, StyleSheet, LayoutChangeEvent } from 'react-native';
import Animated, { useSharedValue, useAnimatedStyle, withTiming, FadeIn } from 'react-native-reanimated';
import { ChevronDown, CircleCheckBig, Circle, LucideIcon } from 'lucide-react-native';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';
import { useThemeStore } from '../../stores/themeStore';

// ═══════════════════════════════════════════════════════════
// CategoryCard — Animated collapsible card for editor fields
// ═══════════════════════════════════════════════════════════

interface CategoryCardProps {
  icon: LucideIcon;
  iconColor: string;
  title: string;
  preview: string;
  isComplete: boolean;
  isExpanded: boolean;
  onToggle: () => void;
  index?: number;
  children: React.ReactNode;
}

export const CategoryCard = React.memo(function CategoryCard({
  icon: Icon, iconColor, title, preview, isComplete, isExpanded, onToggle, index = 0, children,
}: CategoryCardProps) {
  const tc = useThemeStore(s => s.colors);
  const [contentHeight, setContentHeight] = useState(0);

  const animatedHeight = useSharedValue(0);
  const chevronRotation = useSharedValue(0);

  // Drive animations when expanded state changes
  React.useEffect(() => {
    if (isExpanded && contentHeight > 0) {
      animatedHeight.value = withTiming(contentHeight, { duration: 250 });
      chevronRotation.value = withTiming(180, { duration: 250 });
    } else {
      animatedHeight.value = withTiming(0, { duration: 200 });
      chevronRotation.value = withTiming(0, { duration: 200 });
    }
  }, [isExpanded, contentHeight]);

  const bodyStyle = useAnimatedStyle(() => ({
    height: animatedHeight.value,
    overflow: 'hidden' as const,
  }));

  const chevronStyle = useAnimatedStyle(() => ({
    transform: [{ rotate: `${chevronRotation.value}deg` }],
  }));

  const handleLayout = useCallback((e: LayoutChangeEvent) => {
    const h = e.nativeEvent.layout.height;
    if (h > 0 && Math.abs(h - contentHeight) > 2) {
      setContentHeight(h);
      // If already expanded, snap to new height
      if (isExpanded) {
        animatedHeight.value = withTiming(h, { duration: 150 });
      }
    }
  }, [contentHeight, isExpanded]);

  return (
    <Animated.View
      entering={FadeIn.duration(200).delay(index * 50)}
      style={[styles.card, { backgroundColor: tc.bg.secondary, borderColor: tc.border.subtle }]}
    >
      {/* Left accent strip */}
      <View style={[styles.accentStrip, { backgroundColor: iconColor }]} />

      {/* Header — always visible */}
      <Pressable onPress={onToggle} style={styles.header}>
        {/* Icon */}
        <View style={[styles.iconCircle, { backgroundColor: iconColor + '15' }]}>
          <Icon size={15} color={iconColor} />
        </View>

        {/* Title + Preview */}
        <View style={styles.titleCol}>
          <Text style={[styles.title, { color: tc.text.primary }]} numberOfLines={1}>{title}</Text>
          <Text style={[styles.preview, { color: isComplete ? tc.text.secondary : tc.text.faint }]} numberOfLines={1}>
            {preview}
          </Text>
        </View>

        {/* Completion indicator */}
        {isComplete ? (
          <CircleCheckBig size={14} color={tc.status.success} />
        ) : (
          <Circle size={14} color={tc.text.faint} />
        )}

        {/* Chevron */}
        <Animated.View style={chevronStyle}>
          <ChevronDown size={16} color={tc.text.muted} />
        </Animated.View>
      </Pressable>

      {/* Collapsible body */}
      <Animated.View style={bodyStyle}>
        <View onLayout={handleLayout} style={styles.bodyInner}>
          {children}
        </View>
      </Animated.View>
    </Animated.View>
  );
});

// ═══════════════════════════════════════════════════════════
// SectionHeader — Group title with progress indicator
// ═══════════════════════════════════════════════════════════

interface SectionHeaderProps {
  title: string;
  progress: string;
}

export const SectionHeader = React.memo(function SectionHeader({ title, progress }: SectionHeaderProps) {
  const tc = useThemeStore(s => s.colors);
  return (
    <View style={styles.sectionRow}>
      <Text style={[styles.sectionTitle, { color: tc.accent.primary }]}>{title}</Text>
      <View style={[styles.progressPill, { backgroundColor: tc.accent.primary + '15' }]}>
        <Text style={[styles.progressText, { color: tc.accent.primary }]}>{progress}</Text>
      </View>
    </View>
  );
});

const styles = StyleSheet.create({
  // Card
  card: {
    borderRadius: borderRadius.lg,
    borderWidth: 1,
    marginBottom: spacing.sm,
    overflow: 'hidden',
  },
  accentStrip: {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    width: 3,
    borderTopLeftRadius: borderRadius.lg,
    borderBottomLeftRadius: borderRadius.lg,
  },
  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingLeft: spacing.lg + 4, // offset for accent strip
    gap: spacing.sm,
  },
  iconCircle: {
    width: 28,
    height: 28,
    borderRadius: 14,
    justifyContent: 'center',
    alignItems: 'center',
  },
  titleCol: {
    flex: 1,
    gap: 1,
  },
  title: {
    fontSize: 14,
    fontFamily: fonts.heading,
  },
  preview: {
    fontSize: 11,
    fontFamily: fonts.body,
  },
  // Body
  bodyInner: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.lg,
    position: 'absolute',
    width: '100%',
  },
  // Section header
  sectionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  sectionTitle: {
    ...typography.subheading,
    fontSize: 13,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  progressPill: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.full,
  },
  progressText: {
    fontSize: 11,
    fontFamily: fonts.button,
  },
});
