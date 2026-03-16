import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, Platform } from 'react-native';
import Animated, { useSharedValue, useAnimatedStyle, withTiming } from 'react-native-reanimated';
import { WifiOff } from 'lucide-react-native';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing } from '../../constants/theme';

export function OfflineBanner() {
  const tc = useThemeStore(s => s.colors);
  const [isOffline, setIsOffline] = useState(false);
  const height = useSharedValue(0);
  const animStyle = useAnimatedStyle(() => ({
    height: height.value,
    opacity: height.value > 0 ? 1 : 0,
  }));

  useEffect(() => {
    if (Platform.OS === 'web') {
      const handleOnline = () => setIsOffline(false);
      const handleOffline = () => setIsOffline(true);
      window.addEventListener('online', handleOnline);
      window.addEventListener('offline', handleOffline);
      setIsOffline(!navigator.onLine);
      return () => { window.removeEventListener('online', handleOnline); window.removeEventListener('offline', handleOffline); };
    }
    // On native, would use @react-native-community/netinfo
    return undefined;
  }, []);

  useEffect(() => {
    height.value = withTiming(isOffline ? 36 : 0, { duration: 200 });
  }, [isOffline]);

  return (
    <Animated.View style={[styles.container, { backgroundColor: tc.status.warning + '20' }, animStyle]}>
      <WifiOff size={14} color={tc.status.warning} />
      <Text style={[styles.text, { color: tc.status.warning }]}>No internet connection</Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, overflow: 'hidden' },
  text: { ...typography.small, fontWeight: '600' },
});
