import React, { useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import Animated, { useSharedValue, useAnimatedStyle, withRepeat, withTiming, Easing } from 'react-native-reanimated';
import { colors, spacing, borderRadius } from '../../constants/theme';

interface SkeletonProps {
  width: number | string;
  height: number;
  borderRadius?: number;
  style?: any;
}

export function Skeleton({ width, height, borderRadius: br = borderRadius.md, style }: SkeletonProps) {
  const shimmer = useSharedValue(0.3);

  useEffect(() => {
    shimmer.value = withRepeat(
      withTiming(0.6, { duration: 1000, easing: Easing.inOut(Easing.ease) }),
      -1,
      true,
    );
  }, []);

  const animStyle = useAnimatedStyle(() => ({
    opacity: shimmer.value,
  }));

  return (
    <Animated.View style={[{
      width: width as any, height, borderRadius: br,
      backgroundColor: colors.bg.elevated,
    }, animStyle, style]} />
  );
}

// Pre-built skeleton layouts

export function SkeletonCard() {
  return (
    <View style={skStyles.card}>
      <Skeleton width="100%" height={120} borderRadius={borderRadius.lg} />
      <View style={skStyles.cardInfo}>
        <Skeleton width="70%" height={14} />
        <View style={skStyles.cardMeta}>
          <Skeleton width={50} height={10} borderRadius={borderRadius.full} />
          <Skeleton width={60} height={10} />
        </View>
        <Skeleton width={40} height={8} />
      </View>
    </View>
  );
}

export function SkeletonHorizontalCard() {
  return (
    <View style={skStyles.hCard}>
      <Skeleton width={100} height={100} borderRadius={borderRadius.lg} />
      <Skeleton width={80} height={12} style={{ marginTop: spacing.sm }} />
      <Skeleton width={50} height={9} style={{ marginTop: 3 }} />
    </View>
  );
}

export function SkeletonRow() {
  return (
    <View style={skStyles.row}>
      <Skeleton width={44} height={44} borderRadius={22} />
      <View style={skStyles.rowText}>
        <Skeleton width="60%" height={14} />
        <Skeleton width="40%" height={10} style={{ marginTop: 4 }} />
      </View>
    </View>
  );
}

export function DiscoverSkeleton() {
  return (
    <View style={skStyles.container}>
      <Skeleton width="100%" height={40} borderRadius={borderRadius.lg} style={{ marginHorizontal: spacing.lg, marginBottom: spacing.lg }} />
      <Skeleton width={100} height={20} style={{ marginLeft: spacing.lg, marginBottom: spacing.md }} />
      <View style={skStyles.hScroll}>
        <SkeletonHorizontalCard />
        <SkeletonHorizontalCard />
        <SkeletonHorizontalCard />
      </View>
      <Skeleton width={140} height={20} style={{ marginLeft: spacing.lg, marginTop: spacing.xl, marginBottom: spacing.md }} />
      <View style={skStyles.grid}>
        <SkeletonCard />
        <SkeletonCard />
      </View>
    </View>
  );
}

const skStyles = StyleSheet.create({
  container: { paddingTop: spacing.md },
  card: { width: '47%', backgroundColor: colors.bg.secondary, borderRadius: borderRadius.lg, overflow: 'hidden', borderWidth: 1, borderColor: colors.border.subtle },
  cardInfo: { padding: spacing.md, gap: 6 },
  cardMeta: { flexDirection: 'row', justifyContent: 'space-between' },
  hCard: { width: 120, alignItems: 'center', marginRight: spacing.md },
  hScroll: { flexDirection: 'row', paddingHorizontal: spacing.lg },
  grid: { flexDirection: 'row', paddingHorizontal: spacing.lg, gap: spacing.lg },
  row: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: spacing.lg, paddingVertical: spacing.md, gap: spacing.md },
  rowText: { flex: 1, gap: 2 },
});
