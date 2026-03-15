/**
 * StarParallax — Native animated star field
 *
 * Uses Reanimated withRepeat/withTiming for smooth GPU-driven animations.
 * No frame callbacks, no manual time tracking. Stars twinkle and drift
 * using native driver animations for best performance.
 */
import React, { useMemo } from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import Animated, {
  useSharedValue, useAnimatedStyle, withRepeat, withTiming, withDelay,
  withSequence, Easing, FadeIn,
} from 'react-native-reanimated';
import { useThemeStore } from '../../stores/themeStore';
import type { ThemeColors } from '../../constants/themes';
import { useEffect } from 'react';

const { width: W, height: H } = Dimensions.get('window');

function rand(a: number, b: number) { return a + Math.random() * (b - a); }

function pickColor(c: ThemeColors): { r: number; g: number; b: number } {
  const r = Math.random();
  if (r < 0.35) return c.star.color1;
  if (r < 0.65) return c.star.color2;
  return c.star.color3;
}

// ─── Twinkling Star ───
function TwinkleStar({ x, y, size, color, delay }: {
  x: number; y: number; size: number; color: string; delay: number;
}) {
  const opacity = useSharedValue(0.1);
  const translateY = useSharedValue(0);

  useEffect(() => {
    // Twinkle: fade in/out with random timing
    opacity.value = withDelay(delay, withRepeat(
      withSequence(
        withTiming(rand(0.4, 0.9), { duration: rand(1500, 3000), easing: Easing.inOut(Easing.sin) }),
        withTiming(rand(0.05, 0.2), { duration: rand(1500, 3000), easing: Easing.inOut(Easing.sin) }),
      ), -1, true
    ));
    // Gentle upward drift
    translateY.value = withDelay(delay, withRepeat(
      withTiming(-rand(15, 40), { duration: rand(8000, 15000), easing: Easing.linear }),
      -1, false
    ));
  }, []);

  const style = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [{ translateY: translateY.value }],
  }));

  return (
    <Animated.View style={[{
      position: 'absolute', left: x, top: y,
      width: size, height: size, borderRadius: size / 2,
      backgroundColor: color,
      shadowColor: color,
      shadowOffset: { width: 0, height: 0 },
      shadowOpacity: 0.7,
      shadowRadius: size * 2,
      elevation: 1,
    }, style]} />
  );
}

// ─── Floating Orb (soft glow particle) ───
function FloatingOrb({ x, y, size, color, delay }: {
  x: number; y: number; size: number; color: string; delay: number;
}) {
  const opacity = useSharedValue(0);
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const scale = useSharedValue(0.6);

  useEffect(() => {
    opacity.value = withDelay(delay, withRepeat(
      withSequence(
        withTiming(rand(0.15, 0.35), { duration: rand(3000, 5000), easing: Easing.inOut(Easing.quad) }),
        withTiming(0.05, { duration: rand(3000, 5000), easing: Easing.inOut(Easing.quad) }),
      ), -1, true
    ));
    translateX.value = withDelay(delay, withRepeat(
      withSequence(
        withTiming(rand(-20, 20), { duration: rand(5000, 9000), easing: Easing.inOut(Easing.sin) }),
        withTiming(rand(-20, 20), { duration: rand(5000, 9000), easing: Easing.inOut(Easing.sin) }),
      ), -1, true
    ));
    translateY.value = withDelay(delay, withRepeat(
      withTiming(-rand(30, 60), { duration: rand(10000, 18000), easing: Easing.inOut(Easing.sin) }),
      -1, false
    ));
    scale.value = withDelay(delay, withRepeat(
      withSequence(
        withTiming(rand(0.8, 1.2), { duration: rand(4000, 7000), easing: Easing.inOut(Easing.sin) }),
        withTiming(rand(0.5, 0.8), { duration: rand(4000, 7000), easing: Easing.inOut(Easing.sin) }),
      ), -1, true
    ));
  }, []);

  const style = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value },
      { scale: scale.value },
    ],
  }));

  return (
    <Animated.View style={[{
      position: 'absolute', left: x, top: y,
      width: size, height: size, borderRadius: size / 2,
      backgroundColor: color,
      shadowColor: color,
      shadowOffset: { width: 0, height: 0 },
      shadowOpacity: 0.5,
      shadowRadius: size,
      elevation: 1,
    }, style]} />
  );
}

// ─── Shooting Star ───
function ShootingStar({ color, minInterval, maxInterval }: { color: string; minInterval: number; maxInterval: number }) {
  const x = useSharedValue(-100);
  const y = useSharedValue(-100);
  const opacity = useSharedValue(0);
  const rotation = useSharedValue(30);

  useEffect(() => {
    let timeout: ReturnType<typeof setTimeout>;
    const animate = () => {
      const startX = rand(W * 0.05, W * 0.7);
      const startY = rand(H * 0.05, H * 0.5);
      const angle = rand(25, 50);
      const travel = rand(80, 180);
      const duration = rand(500, 900);

      x.value = startX;
      y.value = startY;
      rotation.value = angle;

      opacity.value = withSequence(
        withTiming(rand(0.4, 0.7), { duration: 100, easing: Easing.out(Easing.quad) }),
        withTiming(0, { duration: duration, easing: Easing.in(Easing.quad) }),
      );
      const rad = (angle * Math.PI) / 180;
      x.value = withTiming(startX + Math.cos(rad) * travel, { duration: duration, easing: Easing.out(Easing.quad) });
      y.value = withTiming(startY + Math.sin(rad) * travel, { duration: duration, easing: Easing.out(Easing.quad) });

      timeout = setTimeout(animate, rand(minInterval, maxInterval));
    };

    timeout = setTimeout(animate, rand(2000, 6000));
    return () => clearTimeout(timeout);
  }, []);

  const style = useAnimatedStyle(() => ({
    opacity: opacity.value,
    transform: [{ translateX: x.value }, { translateY: y.value }, { rotate: `${rotation.value}deg` }],
  }));

  return (
    <Animated.View style={[{
      position: 'absolute', width: 30, height: 1, borderRadius: 0.5,
      backgroundColor: color,
      shadowColor: color,
      shadowOffset: { width: 0, height: 0 },
      shadowOpacity: 0.5,
      shadowRadius: 3,
      elevation: 1,
    }, style]} />
  );
}

// ─── Generate star/orb data ───
function generateField(count: number, orbCount: number, c: ThemeColors) {
  const stars = Array.from({ length: count }, () => {
    const col = pickColor(c);
    return {
      x: rand(0, W), y: rand(0, H),
      size: rand(1.5, 3.5),
      color: `rgb(${col.r},${col.g},${col.b})`,
      delay: rand(0, 3000),
    };
  });
  const orbs = Array.from({ length: orbCount }, () => {
    const col = pickColor(c);
    return {
      x: rand(W * 0.1, W * 0.9), y: rand(H * 0.1, H * 0.8),
      size: rand(6, 14),
      color: `rgba(${col.r},${col.g},${col.b},0.4)`,
      delay: rand(0, 5000),
    };
  });
  const shootColor = `rgb(${c.star.color1.r},${c.star.color1.g},${c.star.color1.b})`;
  return { stars, orbs, shootColor };
}

// ─── Full StarParallax (auth screens) ───
export function StarParallax({ children }: { children?: React.ReactNode }) {
  const tc = useThemeStore(s => s.colors);
  const { stars, orbs, shootColor } = useMemo(() => generateField(40, 8, tc), [tc]);

  return (
    <View style={[localStyles.container, { backgroundColor: tc.bg.primary }]}>
      {/* Ambient glow */}
      <View style={[localStyles.glowTop, { backgroundColor: tc.glow.top }]} />
      <View style={[localStyles.glowBottom, { backgroundColor: tc.glow.bottom }]} />

      <View style={StyleSheet.absoluteFill} pointerEvents="none">
        {stars.map((s, i) => <TwinkleStar key={`s${i}`} {...s} />)}
        {orbs.map((o, i) => <FloatingOrb key={`o${i}`} {...o} />)}
        <ShootingStar color={shootColor} minInterval={8000} maxInterval={15000} />
        <ShootingStar color={shootColor} minInterval={12000} maxInterval={20000} />
      </View>

      {children}
    </View>
  );
}

// ─── Lightweight ambient background (discover etc) ───
export function StarParallaxBg() {
  const tc = useThemeStore(s => s.colors);
  const { stars, orbs, shootColor } = useMemo(() => generateField(20, 4, tc), [tc]);

  return (
    <Animated.View
      entering={FadeIn.duration(800)}
      style={[StyleSheet.absoluteFill, { backgroundColor: tc.bg.primary }]}
      pointerEvents="none"
    >
      <View style={[localStyles.glowTop, { backgroundColor: tc.glow.top }]} />
      <View style={[localStyles.glowBottom, { backgroundColor: tc.glow.bottom }]} />
      {stars.map((s, i) => <TwinkleStar key={`bs${i}`} {...s} />)}
      {orbs.map((o, i) => <FloatingOrb key={`bo${i}`} {...o} />)}
      <ShootingStar color={shootColor} minInterval={10000} maxInterval={18000} />
    </Animated.View>
  );
}

const localStyles = StyleSheet.create({
  container: { flex: 1 },
  glowTop: {
    position: 'absolute', top: 0, left: 0, right: 0, height: 200,
    opacity: 0.6,
  },
  glowBottom: {
    position: 'absolute', bottom: 0, left: 0, right: 0, height: 200,
    opacity: 0.4,
  },
});
