import React, { useEffect } from 'react';
import { View, TouchableOpacity, StyleSheet } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { Wand2 } from 'lucide-react-native';
import { Header } from '../../components/shared/Header';
import { CardEditor } from '../../components/creator/CardEditor';
import { useThemeStore } from '../../stores/themeStore';
import { useCharacterStore } from '../../stores/characterStore';
import { CharacterCard } from '../../lib/types';
import { reportError } from '../../lib/utils';

export default function CreateScreen() {
  const insets = useSafeAreaInsets();
  const tc = useThemeStore(s => s.colors);
  const router = useRouter();
  const { myCards, loadMyCards } = useCharacterStore();
  const params = useLocalSearchParams<{ editCard?: string; newCard?: string }>();

  useEffect(() => { loadMyCards(); }, []);

  // Parse edit card from params (passed from character detail "Edit" button)
  let editCard: CharacterCard | undefined;
  if (params.editCard) {
    try { editCard = JSON.parse(params.editCard); } catch (err) { reportError('CreateScreen:parseEditCard', err); }
  }

  // Parse copied card — pre-fills form but creates NEW (no initialCard = no isEditing)
  let prefillCard: CharacterCard | undefined;
  if (params.newCard && !editCard) {
    try { prefillCard = JSON.parse(params.newCard); } catch (err) { reportError('CreateScreen:parsePrefillCard', err); }
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <Header
        title={editCard ? 'Edit Character' : 'Create Character'}
        subtitle={!editCard && myCards.length > 0 ? `${myCards.length} created` : undefined}
        showBack={!!editCard || !!prefillCard}
        right={!editCard && !prefillCard ? (
          <TouchableOpacity onPress={() => router.push('/generate')} hitSlop={8}>
            <Wand2 size={20} color={tc.accent.secondary} />
          </TouchableOpacity>
        ) : undefined}
      />
      <CardEditor initialCard={editCard} prefillData={prefillCard} />
    </View>
  );
}

const styles = StyleSheet.create({ container: { flex: 1 } });
