/**
 * StarParallax — Arcane-themed particle system
 *
 * Renders hextech energy motes, drifting stars, and shooting stars.
 * Uses a single reanimated SharedValue (t) to drive all animation at 60fps.
 * Touch parallax via PanResponder — drag to shift the field, spring-back on release.
 *
 * Two exports:
 *   <StarParallax>  — Full interactive (auth screens): 40 stars + 18 motes + 2 shooters
 *   <StarParallaxBg> — Lightweight ambient (behind scroll): 20 stars + 6 motes, no touch
 */
import React, { useEffect, useMemo, useState } from 'react';
import { View, StyleSheet, Dimensions, LayoutChangeEvent, PanResponder } from 'react-native';
import Animated, {
  useSharedValue, useAnimatedStyle, useFrameCallback, withSpring, SharedValue,
} from 'react-native-reanimated';
import { colors } from '../../constants/theme';

const STAR_COUNT = 40;
const MOTE_COUNT = 18;
const SHOOTING_INTERVAL = 6000;
const DRIFT_SPEED = 0.15;
const PARALLAX_STR = 25;

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

function pickStarColor() {
  const r = Math.random();
  if (r < 0.35) return colors.star.color1;
  if (r < 0.65) return colors.star.color2;
  return colors.star.color3;
}

function pickMoteColor() {
  const p = [colors.petal.pink, colors.petal.lightPink, colors.petal.blush, colors.petal.lavender];
  return p[Math.floor(Math.random() * p.length)];
}

function genStars(w: number, h: number): StarData[] {
  return Array.from({ length: STAR_COUNT }, () => {
    const c = pickStarColor();
    return { startX: Math.random() * w, startY: Math.random() * h, radius: rand(0.8, 2.2),
      baseAlpha: rand(0.15, 0.55), speed: rand(0.03, 0.18), phase: rand(0, Math.PI * 2),
      cr: c.r, cg: c.g, cb: c.b };
  });
}

function genMotes(w: number, h: number): MoteData[] {
  return Array.from({ length: MOTE_COUNT }, () => {
    const c = pickMoteColor();
    return { startX: rand(w * 0.1, w * 0.9), startY: rand(-h * 0.2, h * 0.6),
      size: rand(3, 7), baseAlpha: rand(0.2, 0.55), fallSpeed: rand(0.2, 0.6),
      swayFreq: rand(0.4, 1.2), swayAmp: rand(15, 40), spinSpeed: rand(0.01, 0.04),
      phase: rand(0, Math.PI * 2), cr: c.r, cg: c.g, cb: c.b,
      spiralR: rand(8, 25), spiralSpeed: rand(0.5, 1.5) };
  });
}

// ─── Star ───
function Star({ data, t, px, py, h }: { data: StarData; t: SharedValue<number>; px: SharedValue<number>; py: SharedValue<number>; h: number }) {
  const style = useAnimatedStyle(() => {
    const time = t.value;
    let y = data.startY - (time * DRIFT_SPEED * data.speed * 60);
    const totalH = h + 20;
    y = ((y % totalH) + totalH) % totalH - 10;
    const x = data.startX + Math.sin(time * data.speed * 0.5 + data.phase) * 8;
    const alpha = data.baseAlpha * (0.6 + 0.4 * Math.sin(time * 1.5 + data.phase));
    return {
      transform: [{ translateX: x + px.value * data.speed * PARALLAX_STR }, { translateY: y + py.value * data.speed * PARALLAX_STR }],
      opacity: alpha,
    };
  });
  return (
    <Animated.View style={[{
      position: 'absolute', width: data.radius * 2, height: data.radius * 2, borderRadius: data.radius,
      backgroundColor: `rgb(${data.cr},${data.cg},${data.cb})`,
      shadowColor: `rgb(${data.cr},${data.cg},${data.cb})`,
      shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.6, shadowRadius: data.radius * 1.5, elevation: 1,
    }, style]} />
  );
}

// ─── Hex Mote (replaces sakura petals — diamond/hex energy fragment) ───
function HexMote({ data, t, px, py, w, h }: { data: MoteData; t: SharedValue<number>; px: SharedValue<number>; py: SharedValue<number>; w: number; h: number }) {
  const style = useAnimatedStyle(() => {
    const time = t.value;
    const age = (time * 30 + data.phase * 100) % 200;
    const sp = Math.min(age / 80, 1);
    const sf = 1 - sp;
    const sX = Math.cos(time * data.spiralSpeed + data.phase) * data.spiralR * sf;
    const sY = Math.sin(time * data.spiralSpeed + data.phase) * data.spiralR * sf * 0.6;
    const wind = Math.sin(time * 0.3 + data.phase) * 0.3;
    const dX = (-0.4 - wind) * sp;
    const fY = data.fallSpeed * sp;
    const swX = Math.sin(time * data.swayFreq + data.phase) * data.swayAmp * sp * 0.3;
    let x = data.startX + sX + dX * time * 20 + swX;
    let y = data.startY + sY + fY * time * 20;
    y = ((y % (h + 40)) + (h + 40)) % (h + 40) - 20;
    x = ((x % (w + 40)) + (w + 40)) % (w + 40) - 20;
    const rot = time * data.spinSpeed * 60 + data.phase;
    // Pulsing glow (hextech energy throb)
    const pulse = 0.7 + 0.3 * Math.sin(time * 2 + data.phase);
    const scale = sp < 1 ? pulse : 0.85 + 0.15 * pulse;
    const alpha = data.baseAlpha * (1 - sp * 0.15) * pulse;
    return {
      transform: [
        { translateX: x + px.value * 0.12 * PARALLAX_STR },
        { translateY: y + py.value * 0.12 * PARALLAX_STR },
        { rotate: `${rot}rad` }, { scaleX: scale }, { scaleY: scale },
      ],
      opacity: alpha,
    };
  });
  return (
    <Animated.View style={[{ position: 'absolute', width: data.size * 2, height: data.size * 2 }, style]}>
      {/* Diamond/hex shape via 45deg rotation + border-radius */}
      <View style={{
        position: 'absolute', width: data.size * 1.4, height: data.size * 1.4,
        left: data.size * 0.3, top: data.size * 0.3,
        transform: [{ rotate: '45deg' }],
        backgroundColor: `rgba(${data.cr},${data.cg},${data.cb},0.6)`,
        borderRadius: data.size * 0.2,
        shadowColor: `rgb(${data.cr},${data.cg},${data.cb})`,
        shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.5, shadowRadius: 4, elevation: 1,
      }} />
    </Animated.View>
  );
}

// ─── Shooting Star ───
function Shooter({ t, w, h }: { t: SharedValue<number>; w: number; h: number }) {
  const [vis, setVis] = useState(false);
  const spawn = useSharedValue(0);
  const ref = React.useRef({ x: 0, y: 0, a: 0.5, s: 8, l: 35 });
  useEffect(() => {
    const iv = setInterval(() => {
      const d = ref.current;
      d.x = rand(0, w * 0.7); d.y = rand(0, h * 0.35);
      d.a = rand(0.25, 0.75); d.s = rand(5, 12); d.l = rand(25, 50);
      spawn.value = t.value;
      setVis(true);
      setTimeout(() => setVis(false), 1200);
    }, SHOOTING_INTERVAL + rand(0, 4000));
    return () => clearInterval(iv);
  }, [w, h]);
  const { r, g, b } = colors.star.color1;
  const head = useAnimatedStyle(() => {
    if (!vis) return { opacity: 0 };
    const d = ref.current; const el = t.value - spawn.value; const life = Math.max(0, 1 - el * 0.8);
    if (life <= 0) return { opacity: 0 };
    const dist = el * d.s * 60;
    return { transform: [{ translateX: d.x + Math.cos(d.a) * dist }, { translateY: d.y + Math.sin(d.a) * dist }], opacity: life * 0.85 };
  });
  const tail = useAnimatedStyle(() => {
    if (!vis) return { opacity: 0, width: 0 };
    const d = ref.current; const el = t.value - spawn.value; const life = Math.max(0, 1 - el * 0.8);
    if (life <= 0) return { opacity: 0, width: 0 };
    const dist = el * d.s * 60; const x = d.x + Math.cos(d.a) * dist; const y = d.y + Math.sin(d.a) * dist;
    return { transform: [{ translateX: x - Math.cos(d.a) * d.l }, { translateY: y - Math.sin(d.a) * d.l }, { rotate: `${d.a}rad` }], opacity: life * 0.4, width: d.l };
  });
  return (
    <>
      <Animated.View style={[{ position: 'absolute', height: 1.5, borderRadius: 1, backgroundColor: `rgb(${r},${g},${b})` }, tail]} />
      <Animated.View style={[{ position: 'absolute', width: 3, height: 3, borderRadius: 1.5, backgroundColor: '#fff', shadowColor: '#fff', shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.8, shadowRadius: 3, elevation: 2 }, head]} />
    </>
  );
}

// ─── Full interactive StarParallax ───
export function StarParallax({ children }: { children?: React.ReactNode }) {
  const [layout, setLayout] = useState({ w: Dimensions.get('window').width, h: Dimensions.get('window').height });
  const t = useSharedValue(0);
  const pxV = useSharedValue(0);
  const pyV = useSharedValue(0);
  const stars = useMemo(() => genStars(layout.w, layout.h), [layout.w, layout.h]);
  const motes = useMemo(() => genMotes(layout.w, layout.h), [layout.w, layout.h]);
  useFrameCallback(fi => { if (fi.timeSincePreviousFrame) t.value += fi.timeSincePreviousFrame / 1000; });
  const pan = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => true, onMoveShouldSetPanResponder: () => true,
    onPanResponderMove: (_e, gs) => { pxV.value = (gs.moveX / layout.w - 0.5) * 2; pyV.value = (gs.moveY / layout.h - 0.5) * 2; },
    onPanResponderRelease: () => { pxV.value = withSpring(0, { damping: 20, stiffness: 60 }); pyV.value = withSpring(0, { damping: 20, stiffness: 60 }); },
  }), [layout.w, layout.h]);
  return (
    <View style={s.container} onLayout={(e: LayoutChangeEvent) => { const { width: w, height: h } = e.nativeEvent.layout; if (w > 0) setLayout({ w, h }); }} {...pan.panHandlers}>
      <View style={s.glowTop} /><View style={s.glowBot} />
      <View style={StyleSheet.absoluteFill} pointerEvents="none">
        {stars.map((d, i) => <Star key={`s${i}`} data={d} t={t} px={pxV} py={pyV} h={layout.h} />)}
      </View>
      <View style={StyleSheet.absoluteFill} pointerEvents="none">
        {motes.map((d, i) => <HexMote key={`m${i}`} data={d} t={t} px={pxV} py={pyV} w={layout.w} h={layout.h} />)}
      </View>
      <View style={StyleSheet.absoluteFill} pointerEvents="none">
        <Shooter t={t} w={layout.w} h={layout.h} />
        <Shooter t={t} w={layout.w} h={layout.h} />
      </View>
      {children}
    </View>
  );
}

// ─── Lightweight ambient background ───
export function StarParallaxBg() {
  const [layout, setLayout] = useState({ w: Dimensions.get('window').width, h: Dimensions.get('window').height });
  const t = useSharedValue(0);
  const z = useSharedValue(0);
  const stars = useMemo(() => genStars(layout.w, layout.h).slice(0, 20), [layout.w, layout.h]);
  const motes = useMemo(() => genMotes(layout.w, layout.h).slice(0, 6), [layout.w, layout.h]);
  useFrameCallback(fi => { if (fi.timeSincePreviousFrame) t.value += fi.timeSincePreviousFrame / 1000; });
  return (
    <View style={[StyleSheet.absoluteFill, { backgroundColor: colors.bg.primary }]} pointerEvents="none"
      onLayout={(e: LayoutChangeEvent) => { const { width: w, height: h } = e.nativeEvent.layout; if (w > 0) setLayout({ w, h }); }}>
      <View style={s.glowTop} /><View style={s.glowBot} />
      {stars.map((d, i) => <Star key={`bs${i}`} data={d} t={t} px={z} py={z} h={layout.h} />)}
      {motes.map((d, i) => <HexMote key={`bm${i}`} data={d} t={t} px={z} py={z} w={layout.w} h={layout.h} />)}
    </View>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg.primary },
  glowTop: { ...StyleSheet.absoluteFillObject, backgroundColor: 'transparent',
    shadowColor: 'rgba(79, 168, 212, 0.05)', shadowOffset: { width: 0, height: -100 }, shadowOpacity: 1, shadowRadius: 200 },
  glowBot: { ...StyleSheet.absoluteFillObject,
    shadowColor: 'rgba(68, 212, 128, 0.03)', shadowOffset: { width: 0, height: 100 }, shadowOpacity: 1, shadowRadius: 200 },
});
