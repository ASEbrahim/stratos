import React, { useEffect } from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { View, StyleSheet } from 'react-native';
import { useAuthStore } from '../stores/authStore';
import { useThemeStore } from '../stores/themeStore';
import { LoadingScreen } from '../components/shared/LoadingScreen';
import { ErrorBoundary } from '../components/shared/ErrorBoundary';

export default function RootLayout() {
  const { isLoading, checkAuth } = useAuthStore();
  const { colors: tc, loadTheme } = useThemeStore();

  useEffect(() => { checkAuth(); loadTheme(); }, []);

  if (isLoading) return <LoadingScreen />;
  return (
    <ErrorBoundary>
      <View style={[styles.container, { backgroundColor: tc.bg.primary }]}>
        <StatusBar style="light" />
        <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: tc.bg.primary }, animation: 'slide_from_right' }} />
      </View>
    </ErrorBoundary>
  );
}

const styles = StyleSheet.create({ container: { flex: 1 } });
