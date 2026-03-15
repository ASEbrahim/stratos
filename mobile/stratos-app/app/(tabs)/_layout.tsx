import React from 'react';
import { View } from 'react-native';
import { Tabs } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Compass, MessageSquare, BookOpen, PlusCircle, User } from 'lucide-react-native';
import { useThemeStore } from '../../stores/themeStore';
import { typography } from '../../constants/theme';
import { fonts } from '../../constants/fonts';

function TabIcon({ Icon, color, size, focused, accentColor }: { Icon: any; color: string; size: number; focused: boolean; accentColor: string }) {
  return (
    <View style={{ alignItems: 'center' }}>
      <Icon size={size} color={color} />
      {focused && <View style={{ width: 4, height: 4, borderRadius: 2, backgroundColor: accentColor, marginTop: 2 }} />}
    </View>
  );
}

export default function TabsLayout() {
  const tc = useThemeStore(s => s.colors);
  const insets = useSafeAreaInsets();
  const bottomPad = Math.max(insets.bottom, 8);
  return (
    <Tabs screenOptions={{
      headerShown: false,
      tabBarStyle: { backgroundColor: tc.bg.secondary, borderTopColor: tc.border.subtle, borderTopWidth: 1, height: 52 + bottomPad, paddingBottom: bottomPad, paddingTop: 4 },
      tabBarActiveTintColor: tc.accent.primary,
      tabBarInactiveTintColor: tc.text.muted,
      tabBarLabelStyle: { fontFamily: fonts.button, fontSize: 10 },
    }}>
      <Tabs.Screen name="discover" options={{ title: 'Discover', tabBarIcon: ({ color, size, focused }) => <TabIcon Icon={Compass} color={color} size={size} focused={focused} accentColor={tc.accent.primary} /> }} />
      <Tabs.Screen name="chats" options={{ title: 'Chats', tabBarIcon: ({ color, size, focused }) => <TabIcon Icon={MessageSquare} color={color} size={size} focused={focused} accentColor={tc.accent.primary} /> }} />
      <Tabs.Screen name="library" options={{ title: 'Library', tabBarIcon: ({ color, size, focused }) => <TabIcon Icon={BookOpen} color={color} size={size} focused={focused} accentColor={tc.accent.primary} /> }} />
      <Tabs.Screen name="create" options={{ title: 'Create', tabBarIcon: ({ color, size, focused }) => <TabIcon Icon={PlusCircle} color={color} size={size} focused={focused} accentColor={tc.accent.primary} /> }} />
      <Tabs.Screen name="profile" options={{ title: 'Profile', tabBarIcon: ({ color, size, focused }) => <TabIcon Icon={User} color={color} size={size} focused={focused} accentColor={tc.accent.primary} /> }} />
    </Tabs>
  );
}
