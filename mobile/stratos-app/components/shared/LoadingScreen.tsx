import React from 'react';
import { View, ActivityIndicator, StyleSheet } from 'react-native';
import { useThemeStore } from '../../stores/themeStore';

export function LoadingScreen() {
  const tc = useThemeStore(s => s.colors);
  return (
    <View style={[styles.container, { backgroundColor: tc.bg.primary }]}>
      <ActivityIndicator size="large" color={tc.accent.primary} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center' },
});
