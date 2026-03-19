import React from 'react';
import { View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useThemeStore } from '../../stores/themeStore';
import WorldWizard from '../../components/gaming/WorldWizard';

export default function ImportWorldScreen() {
  const insets = useSafeAreaInsets();
  const tc = useThemeStore(s => s.colors);
  return (
    <View style={{ flex: 1, paddingTop: insets.top, backgroundColor: tc.bg.primary }}>
      <WorldWizard />
    </View>
  );
}
