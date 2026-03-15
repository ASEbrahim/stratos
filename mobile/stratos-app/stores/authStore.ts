import { create } from 'zustand';
import { User } from '../lib/types';
import * as auth from '../lib/auth';

interface AuthState {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  checkAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null, isLoading: true, isAuthenticated: false,
  login: async (email, password) => {
    set({ isLoading: true });
    try { const result = await auth.login(email, password); set({ user: result.user, isAuthenticated: true, isLoading: false }); }
    catch (error) { set({ isLoading: false }); throw error; }
  },
  register: async (name, email, password) => {
    set({ isLoading: true });
    try { const result = await auth.register(name, email, password); set({ user: result.user, isAuthenticated: true, isLoading: false }); }
    catch (error) { set({ isLoading: false }); throw error; }
  },
  logout: async () => { await auth.logout(); set({ user: null, isAuthenticated: false }); },
  checkAuth: async () => {
    set({ isLoading: true });
    try { const user = await auth.getProfile(); set({ user, isAuthenticated: !!user, isLoading: false }); }
    catch { set({ user: null, isAuthenticated: false, isLoading: false }); }
  },
}));
