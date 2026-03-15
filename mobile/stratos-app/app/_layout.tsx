import React, { useEffect } from 'react';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { View, StyleSheet } from 'react-native';
import { useFonts } from 'expo-font';
import { Nunito_400Regular, Nunito_600SemiBold, Nunito_700Bold, Nunito_800ExtraBold, Nunito_900Black } from '@expo-google-fonts/nunito';
import { Poppins_300Light, Poppins_400Regular, Poppins_500Medium, Poppins_600SemiBold, Poppins_700Bold } from '@expo-google-fonts/poppins';
import { Comfortaa_400Regular, Comfortaa_500Medium, Comfortaa_600SemiBold, Comfortaa_700Bold } from '@expo-google-fonts/comfortaa';
import { useAuthStore } from '../stores/authStore';
import { useThemeStore } from '../stores/themeStore';
import { LoadingScreen } from '../components/shared/LoadingScreen';
import { ErrorBoundary } from '../components/shared/ErrorBoundary';
import { OfflineBanner } from '../components/shared/OfflineBanner';

export default function RootLayout() {
  const { isLoading, checkAuth } = useAuthStore();
  const { colors: tc, loadTheme } = useThemeStore();

  const [fontsLoaded] = useFonts({
    Nunito_400Regular, Nunito_600SemiBold, Nunito_700Bold, Nunito_800ExtraBold, Nunito_900Black,
    Poppins_300Light, Poppins_400Regular, Poppins_500Medium, Poppins_600SemiBold, Poppins_700Bold,
    Comfortaa_400Regular, Comfortaa_500Medium, Comfortaa_600SemiBold, Comfortaa_700Bold,
  });

  useEffect(() => { checkAuth(); loadTheme(); }, []);

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
