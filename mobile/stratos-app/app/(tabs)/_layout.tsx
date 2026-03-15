import React from 'react';
import { Tabs } from 'expo-router';
import { Compass, BookOpen, PlusCircle, User } from 'lucide-react-native';
import { useThemeStore } from '../../stores/themeStore';
import { typography } from '../../constants/theme';

export default function TabsLayout() {
  const tc = useThemeStore(s => s.colors);
  return (
    <Tabs screenOptions={{
      headerShown: false,
      tabBarStyle: { backgroundColor: tc.bg.secondary, borderTopColor: tc.border.subtle, borderTopWidth: 1, height: 60, paddingBottom: 8, paddingTop: 4 },
      tabBarActiveTintColor: tc.accent.primary,
      tabBarInactiveTintColor: tc.text.muted,
      tabBarLabelStyle: { ...typography.small, fontSize: 10 },
    }}>
      <Tabs.Screen name="discover" options={{ title: 'Discover', tabBarIcon: ({ color, size }) => <Compass size={size} color={color} /> }} />
      <Tabs.Screen name="library" options={{ title: 'Library', tabBarIcon: ({ color, size }) => <BookOpen size={size} color={color} /> }} />
      <Tabs.Screen name="create" options={{ title: 'Create', tabBarIcon: ({ color, size }) => <PlusCircle size={size} color={color} /> }} />
      <Tabs.Screen name="profile" options={{ title: 'Profile', tabBarIcon: ({ color, size }) => <User size={size} color={color} /> }} />
    </Tabs>
  );
}
