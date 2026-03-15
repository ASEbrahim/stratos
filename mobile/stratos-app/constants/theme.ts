// StratOS Mobile — Arcane Theme (Hextech energy + Zaun depths)

export const colors = {
  bg: {
    primary: '#06070e',        // Deep indigo-black
    secondary: '#0c0e1a',     // Zaun dark
    tertiary: '#151828',       // Panel dark blue
    elevated: '#1e2340',       // Elevated surface
  },
  text: {
    primary: '#e4e8f4',       // Cool white
    secondary: '#8e94b0',     // Muted lavender
    muted: '#5a6080',         // Dim steel
    faint: '#3a4060',         // Barely visible
    inverse: '#06070e',
  },
  accent: {
    primary: '#4fa8d4',       // Hextech blue (main accent)
    light: '#7cc4e8',         // Bright hextech
    dim: '#1a4060',           // Deep hextech
    secondary: '#d4a044',     // Piltover gold
    zaun: '#44d480',          // Zaun toxic green
    shimmer: '#c468e0',       // Shimmer violet
    // Genre accents
    fantasy: '#d4a044',
    scifi: '#4fa8d4',
    romance: '#c468e0',
    horror: '#44d480',
    modern: '#8e78d4',
    anime: '#e478b4',
    historical: '#c4a878',
    default: '#4fa8d4',
  },
  border: {
    subtle: 'rgba(30, 35, 64, 0.5)',
    medium: 'rgba(40, 48, 80, 0.6)',
  },
  status: {
    success: '#44d480',
    error: '#d44060',
    warning: '#d4a044',
  },
  quality: {
    basic: '#5a6080',
    good: '#d4a044',
    great: '#7cc4e8',
    exceptional: '#c468e0',
  },
  sfw: '#44d480',
  nsfw: '#d44060',
  // Star parallax colors — hextech energy particles
  star: {
    color1: { r: 79, g: 168, b: 212 },   // Hextech blue
    color2: { r: 212, g: 160, b: 68 },    // Piltover gold
    color3: { r: 68, g: 212, b: 128 },    // Zaun green
  },
  // Petal equivalents — energy motes / hex fragments
  petal: {
    pink: { r: 79, g: 168, b: 212 },      // Blue hex energy
    lightPink: { r: 124, g: 196, b: 232 }, // Bright blue
    blush: { r: 196, g: 104, b: 224 },     // Shimmer violet
    lavender: { r: 212, g: 160, b: 68 },   // Gold spark
  },
  panel: 'rgba(12, 14, 26, 0.85)',
  sidebar: 'rgba(6, 7, 14, 0.95)',
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
};

export const borderRadius = {
  sm: 6,
  md: 10,
  lg: 16,
  xl: 24,
  full: 999,
};

export const typography = {
  display: {
    fontSize: 28,
    fontWeight: '700' as const,
    letterSpacing: -0.5,
  },
  heading: {
    fontSize: 20,
    fontWeight: '600' as const,
    letterSpacing: -0.3,
  },
  subheading: {
    fontSize: 16,
    fontWeight: '600' as const,
  },
  body: {
    fontSize: 15,
    fontWeight: '400' as const,
    lineHeight: 22,
  },
  caption: {
    fontSize: 12,
    fontWeight: '400' as const,
  },
  small: {
    fontSize: 11,
    fontWeight: '500' as const,
  },
};
