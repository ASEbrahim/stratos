// ═══════════════════════════════════════════════════════════
// STRAT_OS MOBILE THEME SYSTEM
// Ported from web themes.css — 5 atmospheric themes
// Each theme: bg palette, accent colors, star/particle colors
// ═══════════════════════════════════════════════════════════

export interface ThemeColors {
  bg: { primary: string; secondary: string; tertiary: string; elevated: string };
  text: { primary: string; secondary: string; muted: string; faint: string; inverse: string };
  accent: {
    primary: string; light: string; dim: string; secondary: string;
    fantasy: string; scifi: string; romance: string; horror: string;
    modern: string; anime: string; historical: string; default: string;
  };
  border: { subtle: string; medium: string };
  status: { success: string; error: string; warning: string };
  quality: { basic: string; good: string; great: string; exceptional: string };
  sfw: string; nsfw: string;
  star: { color1: { r: number; g: number; b: number }; color2: { r: number; g: number; b: number }; color3: { r: number; g: number; b: number } };
  petal: { pink: { r: number; g: number; b: number }; lightPink: { r: number; g: number; b: number }; blush: { r: number; g: number; b: number }; lavender: { r: number; g: number; b: number } };
  panel: string; sidebar: string;
  glow: { top: string; bottom: string };
}

export interface ThemeDef {
  id: string;
  label: string;
  icon: string;
  description: string;
  colors: ThemeColors;
}

// ──────────────── ARCANE (default) ────────────────
const arcane: ThemeDef = {
  id: 'arcane', label: 'Arcane', icon: '✦', description: 'Hextech energy + Zaun depths',
  colors: {
    bg: { primary: '#06070e', secondary: '#0c0e1a', tertiary: '#151828', elevated: '#1e2340' },
    text: { primary: '#e4e8f4', secondary: '#8e94b0', muted: '#5a6080', faint: '#3a4060', inverse: '#06070e' },
    accent: {
      primary: '#4fa8d4', light: '#7cc4e8', dim: '#1a4060', secondary: '#d4a044',
      fantasy: '#d4a044', scifi: '#4fa8d4', romance: '#c468e0', horror: '#44d480',
      modern: '#8e78d4', anime: '#e478b4', historical: '#c4a878', default: '#4fa8d4',
    },
    border: { subtle: 'rgba(30, 35, 64, 0.5)', medium: 'rgba(40, 48, 80, 0.6)' },
    status: { success: '#44d480', error: '#d44060', warning: '#d4a044' },
    quality: { basic: '#5a6080', good: '#d4a044', great: '#7cc4e8', exceptional: '#c468e0' },
    sfw: '#44d480', nsfw: '#d44060',
    star: { color1: { r: 79, g: 168, b: 212 }, color2: { r: 212, g: 160, b: 68 }, color3: { r: 68, g: 212, b: 128 } },
    petal: { pink: { r: 79, g: 168, b: 212 }, lightPink: { r: 124, g: 196, b: 232 }, blush: { r: 196, g: 104, b: 224 }, lavender: { r: 212, g: 160, b: 68 } },
    panel: 'rgba(12, 14, 26, 0.85)', sidebar: 'rgba(6, 7, 14, 0.95)',
    glow: { top: 'rgba(79, 168, 212, 0.05)', bottom: 'rgba(68, 212, 128, 0.03)' },
  },
};

// ──────────────── SAKURA ────────────────
const sakura: ThemeDef = {
  id: 'sakura', label: 'Sakura', icon: '🌸', description: 'Nighttime cherry blossoms',
  colors: {
    bg: { primary: '#08050c', secondary: '#120a1a', tertiary: '#1e1028', elevated: '#2a1838' },
    text: { primary: '#f0e4ea', secondary: '#b090a0', muted: '#8a6a7a', faint: '#5a3e4e', inverse: '#08050c' },
    accent: {
      primary: '#f0a0b8', light: '#f8c4d4', dim: '#6b3048', secondary: '#d4809a',
      fantasy: '#d4a044', scifi: '#f0a0b8', romance: '#f8c4d4', horror: '#a0d4a0',
      modern: '#c4a0d4', anime: '#f0a0b8', historical: '#d4b890', default: '#f0a0b8',
    },
    border: { subtle: 'rgba(38, 20, 56, 0.5)', medium: 'rgba(50, 28, 68, 0.6)' },
    status: { success: '#a0d4a0', error: '#d44060', warning: '#d4a044' },
    quality: { basic: '#8a6a7a', good: '#d4a044', great: '#f0a0b8', exceptional: '#f8c4d4' },
    sfw: '#a0d4a0', nsfw: '#d44060',
    star: { color1: { r: 240, g: 160, b: 184 }, color2: { r: 255, g: 220, b: 240 }, color3: { r: 200, g: 160, b: 220 } },
    petal: { pink: { r: 240, g: 160, b: 184 }, lightPink: { r: 255, g: 200, b: 225 }, blush: { r: 248, g: 196, b: 212 }, lavender: { r: 200, g: 160, b: 220 } },
    panel: 'rgba(18, 9, 28, 0.85)', sidebar: 'rgba(6, 3, 10, 0.95)',
    glow: { top: 'rgba(240, 160, 184, 0.04)', bottom: 'rgba(200, 160, 220, 0.03)' },
  },
};

// ──────────────── NEBULA ────────────────
const nebula: ThemeDef = {
  id: 'nebula', label: 'Nebula', icon: '🔮', description: 'Deep space violet + cyan',
  colors: {
    bg: { primary: '#08060f', secondary: '#140e25', tertiary: '#1c1438', elevated: '#28204a' },
    text: { primary: '#e8e0f0', secondary: '#8b7eb5', muted: '#584e78', faint: '#403860', inverse: '#08060f' },
    accent: {
      primary: '#38bdf8', light: '#7dd3fc', dim: '#0c4a6e', secondary: '#a78bfa',
      fantasy: '#f0cc55', scifi: '#38bdf8', romance: '#a78bfa', horror: '#34d399',
      modern: '#7dd3fc', anime: '#a78bfa', historical: '#f0cc55', default: '#38bdf8',
    },
    border: { subtle: 'rgba(42, 31, 74, 0.5)', medium: 'rgba(54, 40, 90, 0.6)' },
    status: { success: '#34d399', error: '#d44060', warning: '#f0cc55' },
    quality: { basic: '#584e78', good: '#f0cc55', great: '#38bdf8', exceptional: '#a78bfa' },
    sfw: '#34d399', nsfw: '#d44060',
    star: { color1: { r: 167, g: 139, b: 250 }, color2: { r: 56, g: 189, b: 248 }, color3: { r: 255, g: 255, b: 255 } },
    petal: { pink: { r: 167, g: 139, b: 250 }, lightPink: { r: 125, g: 211, b: 252 }, blush: { r: 196, g: 160, b: 255 }, lavender: { r: 56, g: 189, b: 248 } },
    panel: 'rgba(21, 16, 40, 0.85)', sidebar: 'rgba(10, 7, 26, 0.95)',
    glow: { top: 'rgba(167, 139, 250, 0.04)', bottom: 'rgba(56, 189, 248, 0.03)' },
  },
};

// ──────────────── COSMOS ────────────────
const cosmos: ThemeDef = {
  id: 'cosmos', label: 'Cosmos', icon: '🌌', description: 'Deep blue + golden starlight',
  colors: {
    bg: { primary: '#07080f', secondary: '#0d1225', tertiary: '#162040', elevated: '#1e2a55' },
    text: { primary: '#e2e8f0', secondary: '#7a8bb5', muted: '#4a5578', faint: '#333d58', inverse: '#07080f' },
    accent: {
      primary: '#e8b931', light: '#f0cc55', dim: '#6b5010', secondary: '#7a8bb5',
      fantasy: '#e8b931', scifi: '#7a8bb5', romance: '#f0a0b8', horror: '#34d399',
      modern: '#e8b931', anime: '#f0cc55', historical: '#e8b931', default: '#e8b931',
    },
    border: { subtle: 'rgba(30, 42, 82, 0.5)', medium: 'rgba(40, 55, 100, 0.6)' },
    status: { success: '#34d399', error: '#d44060', warning: '#e8b931' },
    quality: { basic: '#4a5578', good: '#e8b931', great: '#7a8bb5', exceptional: '#f0cc55' },
    sfw: '#34d399', nsfw: '#d44060',
    star: { color1: { r: 232, g: 185, b: 49 }, color2: { r: 150, g: 180, b: 255 }, color3: { r: 255, g: 255, b: 255 } },
    petal: { pink: { r: 232, g: 185, b: 49 }, lightPink: { r: 240, g: 204, b: 85 }, blush: { r: 150, g: 180, b: 255 }, lavender: { r: 255, g: 240, b: 200 } },
    panel: 'rgba(17, 22, 49, 0.85)', sidebar: 'rgba(10, 13, 26, 0.95)',
    glow: { top: 'rgba(232, 185, 49, 0.04)', bottom: 'rgba(150, 180, 255, 0.03)' },
  },
};

// ──────────────── NOIR ────────────────
const noir: ThemeDef = {
  id: 'noir', label: 'Noir', icon: '◆', description: 'Pure black + violet accent',
  colors: {
    bg: { primary: '#050506', secondary: '#0a0a0c', tertiary: '#141418', elevated: '#1e1e24' },
    text: { primary: '#e4e4e7', secondary: '#a1a1aa', muted: '#71717a', faint: '#52525b', inverse: '#050506' },
    accent: {
      primary: '#8b5cf6', light: '#a78bfa', dim: '#3b1f7a', secondary: '#a78bfa',
      fantasy: '#d4a044', scifi: '#8b5cf6', romance: '#a78bfa', horror: '#34d399',
      modern: '#8b5cf6', anime: '#a78bfa', historical: '#d4a044', default: '#8b5cf6',
    },
    border: { subtle: 'rgba(63, 63, 70, 0.3)', medium: 'rgba(63, 63, 70, 0.5)' },
    status: { success: '#34d399', error: '#d44060', warning: '#d4a044' },
    quality: { basic: '#71717a', good: '#d4a044', great: '#a78bfa', exceptional: '#8b5cf6' },
    sfw: '#34d399', nsfw: '#d44060',
    star: { color1: { r: 139, g: 92, b: 246 }, color2: { r: 167, g: 139, b: 250 }, color3: { r: 255, g: 255, b: 255 } },
    petal: { pink: { r: 139, g: 92, b: 246 }, lightPink: { r: 167, g: 139, b: 250 }, blush: { r: 196, g: 160, b: 255 }, lavender: { r: 100, g: 100, b: 120 } },
    panel: 'rgba(22, 22, 28, 0.85)', sidebar: 'rgba(5, 5, 6, 0.95)',
    glow: { top: 'rgba(139, 92, 246, 0.03)', bottom: 'rgba(100, 100, 120, 0.02)' },
  },
};

export const THEMES: ThemeDef[] = [nebula, sakura, cosmos, noir];

export function getTheme(id: string): ThemeDef {
  return THEMES.find(t => t.id === id) ?? nebula;
}
