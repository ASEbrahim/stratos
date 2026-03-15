import React, { useEffect } from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { View, StyleSheet } from 'react-native';
import { useAuthStore } from '../stores/authStore';
import { LoadingScreen } from '../components/shared/LoadingScreen';
import { ErrorBoundary } from '../components/shared/ErrorBoundary';
import { colors } from '../constants/theme';

export default function RootLayout() {
  const { isLoading, checkAuth } = useAuthStore();
  useEffect(() => { checkAuth(); }, []);
  if (isLoading) return <LoadingScreen />;
  return (
    <ErrorBoundary>
      <View style={styles.container}>
        <StatusBar style="light" />
        <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: colors.bg.primary }, animation: 'slide_from_right' }} />
      </View>
    </ErrorBoundary>
  );
}

const styles = StyleSheet.create({ container: { flex: 1, backgroundColor: colors.bg.primary } });
