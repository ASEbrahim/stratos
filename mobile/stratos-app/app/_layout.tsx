import React, { useEffect } from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { View, StyleSheet } from 'react-native';
import { useFonts } from 'expo-font';
import { Nunito_400Regular, Nunito_600SemiBold, Nunito_700Bold, Nunito_800ExtraBold, Nunito_900Black } from '@expo-google-fonts/nunito';
import { Poppins_300Light, Poppins_400Regular, Poppins_500Medium, Poppins_600SemiBold, Poppins_700Bold } from '@expo-google-fonts/poppins';
import { Quicksand_400Regular, Quicksand_500Medium, Quicksand_600SemiBold, Quicksand_700Bold } from '@expo-google-fonts/quicksand';
import { useAuthStore } from '../stores/authStore';
import { useThemeStore } from '../stores/themeStore';
import { useChatStore } from '../stores/chatStore';
import { LoadingScreen } from '../components/shared/LoadingScreen';
import { ErrorBoundary } from '../components/shared/ErrorBoundary';
import { OfflineBanner } from '../components/shared/OfflineBanner';

export default function RootLayout() {
  const { isLoading, checkAuth } = useAuthStore();
  const { colors: tc, loadTheme } = useThemeStore();

  const [fontsLoaded] = useFonts({
    Nunito_400Regular, Nunito_600SemiBold, Nunito_700Bold, Nunito_800ExtraBold, Nunito_900Black,
    Poppins_300Light, Poppins_400Regular, Poppins_500Medium, Poppins_600SemiBold, Poppins_700Bold,
    Quicksand_400Regular, Quicksand_500Medium, Quicksand_600SemiBold, Quicksand_700Bold,
  });

  useEffect(() => { checkAuth(); loadTheme(); useChatStore.getState().loadRecentSessions(); }, []);

  if (isLoading || !fontsLoaded) return <LoadingScreen />;
  return (
    <ErrorBoundary>
      <View style={[styles.container, { backgroundColor: tc.bg.primary }]}>
        <StatusBar style="light" />
        <OfflineBanner />
        <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: tc.bg.primary }, animation: 'slide_from_right' }} />
      </View>
    </ErrorBoundary>
  );
}

const styles = StyleSheet.create({ container: { flex: 1 } });
