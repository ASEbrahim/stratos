import React from 'react';
import { View, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Header } from '../../components/shared/Header';
import { CardEditor } from '../../components/creator/CardEditor';
import { useThemeStore } from '../../stores/themeStore';

export default function CreateScreen() {
  const insets = useSafeAreaInsets();
  const tc = useThemeStore(s => s.colors);
  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <Header title="Create Character" />
      <CardEditor />
    </View>
  );
}

const styles = StyleSheet.create({ container: { flex: 1 } });
