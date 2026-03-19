import React, { useCallback, useState } from 'react';
import { View, Text, Pressable, TouchableOpacity, Modal, StyleSheet, LayoutChangeEvent, ScrollView } from 'react-native';
import Animated, { useSharedValue, useAnimatedStyle, withTiming, FadeIn } from 'react-native-reanimated';
import { ChevronDown, ChevronRight, CircleCheckBig, Circle, X, LucideIcon } from 'lucide-react-native';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';
import { useThemeStore } from '../../stores/themeStore';
import type { ThemeColors } from '../../constants/themes';

// ═══════════════════════════════════════════════════════════
// Shared CardHeader — used by both CategoryCard and CategoryPopup
// ═══════════════════════════════════════════════════════════

function CardHeader({ icon: Icon, iconColor, title, preview, isComplete, right, onPress, tc }: {
  icon: LucideIcon; iconColor: string; title: string; preview: string;
  isComplete: boolean; right: React.ReactNode; onPress: () => void;
  tc: ThemeColors;
}) {
  return (
    <Pressable onPress={onPress} style={styles.header}>
      <View style={[styles.iconCircle, { backgroundColor: iconColor + '15' }]}>
        <Icon size={15} color={iconColor} />
      </View>
      <View style={styles.titleCol}>
        <Text style={[styles.title, { color: tc.text.primary }]} numberOfLines={1}>{title}</Text>
        <Text style={[styles.preview, { color: isComplete ? tc.text.secondary : tc.text.faint }]} numberOfLines={1}>
          {preview}
        </Text>
      </View>
      {isComplete ? (
        <CircleCheckBig size={14} color={tc.status.success} />
      ) : (
        <Circle size={14} color={tc.text.faint} />
      )}
      {right}
    </Pressable>
  );
}

// ═══════════════════════════════════════════════════════════
// CategoryCard — Animated collapsible card (inline expand)
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
  icon, iconColor, title, preview, isComplete, isExpanded, onToggle, index = 0, children,
}: CategoryCardProps) {
  const tc = useThemeStore(s => s.colors);
  const [contentHeight, setContentHeight] = useState(0);

  const animatedHeight = useSharedValue(0);
  const chevronRotation = useSharedValue(0);

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
      <View style={[styles.accentStrip, { backgroundColor: iconColor }]} />
      <CardHeader
        icon={icon} iconColor={iconColor} title={title} preview={preview}
        isComplete={isComplete} onPress={onToggle} tc={tc}
        right={<Animated.View style={chevronStyle}><ChevronDown size={16} color={tc.text.muted} /></Animated.View>}
      />
      <Animated.View style={bodyStyle}>
        <View onLayout={handleLayout} style={styles.bodyInner}>
          {children}
        </View>
      </Animated.View>
    </Animated.View>
  );
});

// ═══════════════════════════════════════════════════════════
// CategoryPopup — Wizard-style full modal (for pill selectors)
// ═══════════════════════════════════════════════════════════

interface CategoryPopupProps {
  icon: LucideIcon;
  iconColor: string;
  title: string;
  preview: string;
  isComplete: boolean;
  index?: number;
  children: React.ReactNode;
}

export const CategoryPopup = React.memo(function CategoryPopup({
  icon, iconColor, title, preview, isComplete, index = 0, children,
}: CategoryPopupProps) {
  const tc = useThemeStore(s => s.colors);
  const [visible, setVisible] = useState(false);

  return (
    <>
      {/* Card trigger — tapping opens the modal */}
      <Animated.View
        entering={FadeIn.duration(200).delay(index * 50)}
        style={[styles.card, { backgroundColor: tc.bg.secondary, borderColor: tc.border.subtle }]}
      >
        <View style={[styles.accentStrip, { backgroundColor: iconColor }]} />
        <CardHeader
          icon={icon} iconColor={iconColor} title={title} preview={preview}
          isComplete={isComplete} onPress={() => setVisible(true)} tc={tc}
          right={<ChevronRight size={16} color={tc.text.muted} />}
        />
      </Animated.View>

      {/* Wizard-style modal */}
      <Modal visible={visible} transparent animationType="fade" onRequestClose={() => setVisible(false)}>
        <View style={styles.wizBackdrop}>
          <View style={[styles.wizPanel, { backgroundColor: tc.bg.primary, borderColor: tc.border.subtle }]}>
            {/* Header with accent strip */}
            <View style={[styles.wizHeader, { borderBottomColor: tc.border.subtle }]}>
              <View style={[styles.wizIconLarge, { backgroundColor: iconColor + '15' }]}>
                {React.createElement(icon, { size: 22, color: iconColor })}
              </View>
              <View style={{ flex: 1 }}>
                <Text style={[styles.wizTitle, { color: tc.text.primary }]}>{title}</Text>
                <Text style={[styles.wizSubtitle, { color: tc.text.muted }]}>
                  {isComplete ? preview : 'Tap to select'}
                </Text>
              </View>
              <TouchableOpacity
                onPress={() => setVisible(false)}
                style={[styles.wizClose, { backgroundColor: tc.bg.tertiary }]}
                hitSlop={8}
              >
                <X size={18} color={tc.text.secondary} />
              </TouchableOpacity>
            </View>

            {/* Pill content — wrapping, all visible */}
            <ScrollView
              style={styles.wizBody}
              contentContainerStyle={styles.wizBodyContent}
              showsVerticalScrollIndicator={false}
            >
              {children}
            </ScrollView>

            {/* Done button */}
            <View style={styles.wizFooter}>
              <TouchableOpacity
                style={[styles.wizDone, { backgroundColor: tc.accent.primary }]}
                onPress={() => setVisible(false)}
                activeOpacity={0.7}
              >
                <Text style={styles.wizDoneText}>Done</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </>
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
  // Card (shared between CategoryCard and CategoryPopup)
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
    paddingLeft: spacing.lg + 4,
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
  // Body (inline expand)
  bodyInner: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.lg,
    position: 'absolute',
    width: '100%',
  },
  // Wizard-style modal
  wizBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.lg,
  },
  wizPanel: {
    width: '100%',
    maxHeight: '85%',
    borderRadius: borderRadius.xl,
    borderWidth: 1,
    overflow: 'hidden',
  },
  wizHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingHorizontal: spacing.xl,
    paddingVertical: spacing.lg,
    borderBottomWidth: 1,
  },
  wizIconLarge: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
  },
  wizTitle: {
    fontSize: 20,
    fontFamily: fonts.heading,
  },
  wizSubtitle: {
    fontSize: 12,
    fontFamily: fonts.body,
    marginTop: 2,
  },
  wizClose: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  wizBody: {
    paddingHorizontal: spacing.xl,
  },
  wizBodyContent: {
    paddingVertical: spacing.xl,
  },
  wizFooter: {
    paddingHorizontal: spacing.xl,
    paddingBottom: spacing.lg,
    paddingTop: spacing.sm,
  },
  wizDone: {
    paddingVertical: spacing.md,
    borderRadius: borderRadius.lg,
    alignItems: 'center',
  },
  wizDoneText: {
    fontSize: 16,
    fontFamily: fonts.heading,
    color: '#fff',
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
