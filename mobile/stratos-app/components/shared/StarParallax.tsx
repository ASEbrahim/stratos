/**
 * StarParallax — Multi-theme particle system
 *
 * Adapts to current theme colors (Arcane, Sakura, Nebula, Cosmos, Noir).
 * Stars + motes + shooting stars, all driven by a single SharedValue (t).
 * Optimized for mobile: reduced particle count, no canvas, pure Reanimated.
 * pointerEvents="none" — children receive all touches.
 */
import React, { useEffect, useMemo, useState } from 'react';
import { View, StyleSheet, Dimensions, LayoutChangeEvent } from 'react-native';
import Animated, {
  useSharedValue, useAnimatedStyle, useFrameCallback, SharedValue,
} from 'react-native-reanimated';
import { useThemeStore } from '../../stores/themeStore';
import { colors as defaultColors } from '../../constants/theme';
import type { ThemeColors } from '../../constants/themes';

const STAR_COUNT = 35;
const MOTE_COUNT = 12;
const SHOOTING_INTERVAL = 8000;
const DRIFT_SPEED = 0.12;

interface StarData {
  startX: number; startY: number; radius: number; baseAlpha: number;
  speed: number; phase: number; cr: number; cg: number; cb: number;
}

interface MoteData {
  startX: number; startY: number; size: number; baseAlpha: number;
  fallSpeed: number; swayFreq: number; swayAmp: number; spinSpeed: number;
  phase: number; cr: number; cg: number; cb: number;
  spiralR: number; spiralSpeed: number;
}

function rand(a: number, b: number) { return a + Math.random() * (b - a); }

function pickStarColor(c: ThemeColors) {
  const r = Math.random();
  if (r < 0.35) return c.star.color1;
  if (r < 0.65) return c.star.color2;
  return c.star.color3;
}

function pickMoteColor(c: ThemeColors) {
  const p = [c.petal.pink, c.petal.lightPink, c.petal.blush, c.petal.lavender];
  return p[Math.floor(Math.random() * p.length)];
}

function genStars(w: number, h: number, c: ThemeColors): StarData[] {
  return Array.from({ length: STAR_COUNT }, () => {
    const col = pickStarColor(c);
    return {
      startX: Math.random() * w, startY: Math.random() * h, radius: rand(0.8, 2.0),
      baseAlpha: rand(0.15, 0.50), speed: rand(0.03, 0.15), phase: rand(0, Math.PI * 2),
      cr: col.r, cg: col.g, cb: col.b,
    };
  });
}

function genMotes(w: number, h: number, c: ThemeColors): MoteData[] {
  return Array.from({ length: MOTE_COUNT }, () => {
    const col = pickMoteColor(c);
    return {
      startX: rand(w * 0.1, w * 0.9), startY: rand(-h * 0.2, h * 0.6),
      size: rand(3, 6), baseAlpha: rand(0.18, 0.45), fallSpeed: rand(0.2, 0.5),
      swayFreq: rand(0.4, 1.0), swayAmp: rand(12, 30), spinSpeed: rand(0.01, 0.03),
      phase: rand(0, Math.PI * 2), cr: col.r, cg: col.g, cb: col.b,
      spiralR: rand(6, 20), spiralSpeed: rand(0.4, 1.2),
    };
  });
}

// ─── Star ───
function Star({ data, t, h }: { data: StarData; t: SharedValue<number>; h: number }) {
  const d = useMemo(() => Object.freeze({ ...data }), []);

  const style = useAnimatedStyle(() => {
    const time = t.value;
    let y = d.startY - (time * DRIFT_SPEED * d.speed * 60);
    const totalH = h + 20;
    y = ((y % totalH) + totalH) % totalH - 10;
    const x = d.startX + Math.sin(time * d.speed * 0.5 + d.phase) * 6;
    const alpha = d.baseAlpha * (0.6 + 0.4 * Math.sin(time * 1.5 + d.phase));
    return {
      transform: [{ translateX: x }, { translateY: y }],
      opacity: alpha,
    };
  });

  return (
    <Animated.View style={[{
      position: 'absolute', width: d.radius * 2, height: d.radius * 2, borderRadius: d.radius,
      backgroundColor: `rgb(${d.cr},${d.cg},${d.cb})`,
      shadowColor: `rgb(${d.cr},${d.cg},${d.cb})`,
      shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.5, shadowRadius: d.radius * 1.5,
      elevation: 1,
    }, style]} />
  );
}

// ─── Mote (theme-adaptive particle — hex for Arcane, petal for Sakura, orb for others) ───
function Mote({ data, t, w, h }: { data: MoteData; t: SharedValue<number>; w: number; h: number }) {
  const d = useMemo(() => Object.freeze({ ...data }), []);

  const style = useAnimatedStyle(() => {
    const time = t.value;
    const age = (time * 30 + d.phase * 100) % 200;
    const sp = Math.min(age / 80, 1);
    const sf = 1 - sp;
    const sX = Math.cos(time * d.spiralSpeed + d.phase) * d.spiralR * sf;
    const sY = Math.sin(time * d.spiralSpeed + d.phase) * d.spiralR * sf * 0.6;
    const wind = Math.sin(time * 0.3 + d.phase) * 0.3;
    const dX = (-0.4 - wind) * sp;
    const fY = d.fallSpeed * sp;
    const swX = Math.sin(time * d.swayFreq + d.phase) * d.swayAmp * sp * 0.3;
    let x = d.startX + sX + dX * time * 20 + swX;
    let y = d.startY + sY + fY * time * 20;
    y = ((y % (h + 40)) + (h + 40)) % (h + 40) - 20;
    x = ((x % (w + 40)) + (w + 40)) % (w + 40) - 20;
    const rot = time * d.spinSpeed * 60 + d.phase;
    const pulse = 0.7 + 0.3 * Math.sin(time * 2 + d.phase);
    const scale = sp < 1 ? pulse : 0.85 + 0.15 * pulse;
    const alpha = d.baseAlpha * (1 - sp * 0.15) * pulse;
    return {
      transform: [
        { translateX: x }, { translateY: y },
        { rotate: `${rot}rad` }, { scaleX: scale }, { scaleY: scale },
      ],
      opacity: alpha,
    };
  });

  return (
    <Animated.View style={[{ position: 'absolute', width: d.size * 2, height: d.size * 2 }, style]}>
      <View style={{
        position: 'absolute', width: d.size * 1.4, height: d.size * 1.4,
        left: d.size * 0.3, top: d.size * 0.3,
        transform: [{ rotate: '45deg' }],
        backgroundColor: `rgba(${d.cr},${d.cg},${d.cb},0.5)`,
        borderRadius: d.size * 0.25,
        shadowColor: `rgb(${d.cr},${d.cg},${d.cb})`,
        shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.4, shadowRadius: 4, elevation: 1,
      }} />
    </Animated.View>
  );
}

// ─── Shooting Star ───
function Shooter({ t, w, h, c }: { t: SharedValue<number>; w: number; h: number; c: ThemeColors }) {
  const shootX = useSharedValue(0);
  const shootY = useSharedValue(0);
  const shootAngle = useSharedValue(0.5);
  const shootSpeed = useSharedValue(8);
  const shootLen = useSharedValue(35);
  const shootSpawn = useSharedValue(-999);

  const { r, g, b } = c.star.color1;

  useEffect(() => {
    const interval = setInterval(() => {
      shootX.value = rand(0, w * 0.7);
      shootY.value = rand(0, h * 0.35);
      shootAngle.value = rand(0.25, 0.75);
      shootSpeed.value = rand(5, 12);
      shootLen.value = rand(25, 50);
      shootSpawn.value = t.value;
    }, SHOOTING_INTERVAL + rand(0, 4000));
    return () => clearInterval(interval);
  }, [w, h]);

  const headStyle = useAnimatedStyle(() => {
    const elapsed = t.value - shootSpawn.value;
    const life = Math.max(0, 1 - elapsed * 0.8);
    if (life <= 0) return { opacity: 0 };
    const dist = elapsed * shootSpeed.value * 60;
    const x = shootX.value + Math.cos(shootAngle.value) * dist;
    const y = shootY.value + Math.sin(shootAngle.value) * dist;
    return {
      transform: [{ translateX: x }, { translateY: y }],
      opacity: life * 0.85,
    };
  });

  const tailStyle = useAnimatedStyle(() => {
    const elapsed = t.value - shootSpawn.value;
    const life = Math.max(0, 1 - elapsed * 0.8);
    if (life <= 0) return { opacity: 0, width: 0 };
    const dist = elapsed * shootSpeed.value * 60;
    const x = shootX.value + Math.cos(shootAngle.value) * dist;
    const y = shootY.value + Math.sin(shootAngle.value) * dist;
    const len = shootLen.value;
    const angle = shootAngle.value;
    return {
      transform: [
        { translateX: x - Math.cos(angle) * len },
        { translateY: y - Math.sin(angle) * len },
        { rotate: `${angle}rad` },
      ],
      opacity: life * 0.4,
      width: len,
    };
  });

  return (
    <>
      <Animated.View style={[{
        position: 'absolute', height: 1.5, borderRadius: 1,
        backgroundColor: `rgb(${r},${g},${b})`,
      }, tailStyle]} />
      <Animated.View style={[{
        position: 'absolute', width: 3, height: 3, borderRadius: 1.5,
        backgroundColor: '#fff',
        shadowColor: '#fff', shadowOffset: { width: 0, height: 0 },
        shadowOpacity: 0.8, shadowRadius: 3, elevation: 2,
      }, headStyle]} />
    </>
  );
}

// ─── Full StarParallax (auth screens) ───
export function StarParallax({ children }: { children?: React.ReactNode }) {
  const themeColors = useThemeStore(s => s.colors);
  const [layout, setLayout] = useState({
    w: Dimensions.get('window').width,
    h: Dimensions.get('window').height,
  });
  const t = useSharedValue(0);

  const stars = useMemo(() => genStars(layout.w, layout.h, themeColors), [layout.w, layout.h, themeColors]);
  const motes = useMemo(() => genMotes(layout.w, layout.h, themeColors), [layout.w, layout.h, themeColors]);

  useFrameCallback(fi => {
    if (fi.timeSincePreviousFrame) t.value += fi.timeSincePreviousFrame / 1000;
  });

  return (
    <View
      style={[localStyles.container, { backgroundColor: themeColors.bg.primary }]}
      onLayout={(e: LayoutChangeEvent) => {
        const { width: w, height: h } = e.nativeEvent.layout;
        if (w > 0) setLayout({ w, h });
      }}
    >
      <View style={[localStyles.glow, { shadowColor: themeColors.glow.top, shadowOffset: { width: 0, height: -100 } }]} />
      <View style={[localStyles.glow, { shadowColor: themeColors.glow.bottom, shadowOffset: { width: 0, height: 100 } }]} />

      <View style={StyleSheet.absoluteFill} pointerEvents="none">
        {stars.map((d, i) => <Star key={`s${i}`} data={d} t={t} h={layout.h} />)}
      </View>
      <View style={StyleSheet.absoluteFill} pointerEvents="none">
        {motes.map((d, i) => <Mote key={`m${i}`} data={d} t={t} w={layout.w} h={layout.h} />)}
      </View>
      <View style={StyleSheet.absoluteFill} pointerEvents="none">
        <Shooter t={t} w={layout.w} h={layout.h} c={themeColors} />
        <Shooter t={t} w={layout.w} h={layout.h} c={themeColors} />
      </View>

      {children}
    </View>
  );
}

// ─── Lightweight ambient background ───
export function StarParallaxBg() {
  const themeColors = useThemeStore(s => s.colors);
  const [layout, setLayout] = useState({
    w: Dimensions.get('window').width,
    h: Dimensions.get('window').height,
  });
  const t = useSharedValue(0);

  const stars = useMemo(() => genStars(layout.w, layout.h, themeColors).slice(0, 16), [layout.w, layout.h, themeColors]);
  const motes = useMemo(() => genMotes(layout.w, layout.h, themeColors).slice(0, 5), [layout.w, layout.h, themeColors]);

  useFrameCallback(fi => {
    if (fi.timeSincePreviousFrame) t.value += fi.timeSincePreviousFrame / 1000;
  });

  return (
    <View
      style={[StyleSheet.absoluteFill, { backgroundColor: themeColors.bg.primary }]}
      pointerEvents="none"
      onLayout={(e: LayoutChangeEvent) => {
        const { width: w, height: h } = e.nativeEvent.layout;
        if (w > 0) setLayout({ w, h });
      }}
    >
      <View style={[localStyles.glow, { shadowColor: themeColors.glow.top, shadowOffset: { width: 0, height: -100 } }]} />
      <View style={[localStyles.glow, { shadowColor: themeColors.glow.bottom, shadowOffset: { width: 0, height: 100 } }]} />
      {stars.map((d, i) => <Star key={`bs${i}`} data={d} t={t} h={layout.h} />)}
      {motes.map((d, i) => <Mote key={`bm${i}`} data={d} t={t} w={layout.w} h={layout.h} />)}
    </View>
  );
}

const localStyles = StyleSheet.create({
  container: { flex: 1 },
  glow: {
    ...StyleSheet.absoluteFillObject, backgroundColor: 'transparent',
    shadowOpacity: 1, shadowRadius: 200,
  },
});
