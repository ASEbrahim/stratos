import React, { useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useLocalSearchParams } from 'expo-router';
import { Header } from '../../components/shared/Header';
import { CardEditor } from '../../components/creator/CardEditor';
import { useThemeStore } from '../../stores/themeStore';
import { useCharacterStore } from '../../stores/characterStore';
import { CharacterCard } from '../../lib/types';

export default function CreateScreen() {
  const insets = useSafeAreaInsets();
  const tc = useThemeStore(s => s.colors);
  const { myCards, loadMyCards } = useCharacterStore();
  const params = useLocalSearchParams<{ editCard?: string }>();

  useEffect(() => { loadMyCards(); }, []);

  // Parse edit card from params (passed from character detail "Edit" button)
  let editCard: CharacterCard | undefined;
  if (params.editCard) {
    try { editCard = JSON.parse(params.editCard); } catch {}
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <Header title={editCard ? 'Edit Character' : 'Create Character'} subtitle={!editCard && myCards.length > 0 ? `${myCards.length} created` : undefined} showBack={!!editCard} />
      <CardEditor initialCard={editCard} />
    </View>
  );
}

const styles = StyleSheet.create({ container: { flex: 1 } });
