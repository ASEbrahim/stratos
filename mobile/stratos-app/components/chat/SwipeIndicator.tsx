import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import * as Haptics from 'expo-haptics';
import { ChevronLeft, ChevronRight, RefreshCw } from 'lucide-react-native';
import { useThemeStore } from '../../stores/themeStore';
import { spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';

interface SwipeIndicatorProps {
  currentIndex: number;
  totalCount: number;
  onPrev: () => void;
  onNext: () => void;
  onRegenerate: () => void;
  isRegenerating?: boolean;
  accentColor?: string;
}

export const SwipeIndicator = React.memo(function SwipeIndicator({
  currentIndex, totalCount, onPrev, onNext, onRegenerate, isRegenerating, accentColor,
}: SwipeIndicatorProps) {
  const tc = useThemeStore(s => s.colors);
  const accent = accentColor ?? tc.accent.primary;

  const handlePrev = () => { if (currentIndex > 0) { Haptics.selectionAsync(); onPrev(); } };
  const handleNext = () => { if (currentIndex < totalCount - 1) { Haptics.selectionAsync(); onNext(); } };
  const handleRegen = () => { if (!isRegenerating) { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium); onRegenerate(); } };

  return (
    <View style={styles.container}>
      {totalCount > 1 && (
        <View style={styles.navRow}>
          <TouchableOpacity onPress={handlePrev} disabled={currentIndex === 0} style={styles.navBtn} hitSlop={8}>
            <ChevronLeft size={14} color={currentIndex > 0 ? tc.text.secondary : tc.text.faint} />
          </TouchableOpacity>
          <Text style={[styles.counter, { color: tc.text.muted }]}>
            {currentIndex + 1}/{totalCount}
          </Text>
          <TouchableOpacity onPress={handleNext} disabled={currentIndex >= totalCount - 1} style={styles.navBtn} hitSlop={8}>
            <ChevronRight size={14} color={currentIndex < totalCount - 1 ? tc.text.secondary : tc.text.faint} />
          </TouchableOpacity>
        </View>
      )}
      <TouchableOpacity
        onPress={handleRegen}
        disabled={isRegenerating}
        style={[styles.regenBtn, { borderColor: accent + '30' }, isRegenerating && { opacity: 0.5 }]}
        hitSlop={8}
      >
        <RefreshCw size={11} color={accent} />
      </TouchableOpacity>
    </View>
  );
});

const styles = StyleSheet.create({
  container: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm, paddingLeft: spacing.lg, marginTop: 2 },
  navRow: { flexDirection: 'row', alignItems: 'center', gap: 2 },
  navBtn: { padding: 2 },
  counter: { fontSize: 10, fontFamily: fonts.body, minWidth: 24, textAlign: 'center' },
  regenBtn: { padding: 4, borderRadius: 8, borderWidth: 1 },
});
