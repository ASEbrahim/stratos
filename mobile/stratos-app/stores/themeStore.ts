import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { ThemeColors, getTheme } from '../constants/themes';
import { reportError } from '../lib/utils';

const THEME_KEY = 'stratos_theme';
const NSFW_KEY = 'stratos_nsfw_filter';

interface ThemeState {
  themeId: string;
  colors: ThemeColors;
  nsfwFilter: boolean; // true = SFW only (hide NSFW)
  setTheme: (id: string) => void;
  setNsfwFilter: (enabled: boolean) => void;
  loadTheme: () => Promise<void>;
}

export const useThemeStore = create<ThemeState>((set) => ({
  themeId: 'nebula',
  colors: getTheme('nebula').colors,
  nsfwFilter: true,
  setTheme: (id) => {
    const theme = getTheme(id);
    set({ themeId: theme.id, colors: theme.colors });
    AsyncStorage.setItem(THEME_KEY, theme.id).catch(err => reportError('setTheme:persist', err));
  },
  setNsfwFilter: (enabled) => {
    set({ nsfwFilter: enabled });
    AsyncStorage.setItem(NSFW_KEY, enabled ? '1' : '0').catch(err => reportError('setNsfwFilter:persist', err));
  },
  loadTheme: async () => {
    try {
      const [saved, nsfw] = await Promise.all([
        AsyncStorage.getItem(THEME_KEY),
        AsyncStorage.getItem(NSFW_KEY),
      ]);
      const updates: Partial<ThemeState> = {};
      if (saved) {
        const theme = getTheme(saved);
        updates.themeId = theme.id;
        updates.colors = theme.colors;
      }
      if (nsfw !== null) updates.nsfwFilter = nsfw !== '0';
      set(updates);
    } catch (err) { reportError('loadTheme', err); }
  },
}));
