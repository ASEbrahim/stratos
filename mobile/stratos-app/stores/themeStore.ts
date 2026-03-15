import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { ThemeColors, getTheme } from '../constants/themes';

const THEME_KEY = 'stratos_theme';

interface ThemeState {
  themeId: string;
  colors: ThemeColors;
  setTheme: (id: string) => void;
  loadTheme: () => Promise<void>;
}

export const useThemeStore = create<ThemeState>((set) => ({
  themeId: 'arcane',
  colors: getTheme('arcane').colors,
  setTheme: (id) => {
    const theme = getTheme(id);
    set({ themeId: theme.id, colors: theme.colors });
    AsyncStorage.setItem(THEME_KEY, theme.id).catch(() => {});
  },
  loadTheme: async () => {
    try {
      const saved = await AsyncStorage.getItem(THEME_KEY);
      if (saved) {
        const theme = getTheme(saved);
        set({ themeId: theme.id, colors: theme.colors });
      }
    } catch {}
  },
}));
