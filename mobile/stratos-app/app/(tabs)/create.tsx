import React, { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Header } from '../../components/shared/Header';
import { CardEditor } from '../../components/creator/CardEditor';
import { useThemeStore } from '../../stores/themeStore';
import { useCharacterStore } from '../../stores/characterStore';
import { spacing, typography } from '../../constants/theme';

export default function CreateScreen() {
  const insets = useSafeAreaInsets();
  const tc = useThemeStore(s => s.colors);
  const { myCards, loadMyCards } = useCharacterStore();

  useEffect(() => { loadMyCards(); }, []);

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <Header title="Create Character" subtitle={myCards.length > 0 ? `${myCards.length} created` : undefined} />
      <CardEditor />
    </View>
  );
}

const styles = StyleSheet.create({ container: { flex: 1 } });
