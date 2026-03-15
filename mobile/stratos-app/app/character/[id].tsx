import React, { useEffect, useState } from 'react';
import { View, StyleSheet } from 'react-native';
import { useLocalSearchParams } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Header } from '../../components/shared/Header';
import { CharacterDetailView } from '../../components/cards/CharacterDetail';
import { LoadingScreen } from '../../components/shared/LoadingScreen';
import { CharacterCard } from '../../lib/types';
import { getCharacter } from '../../lib/characters';
import { colors } from '../../constants/theme';

export default function CharacterDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const insets = useSafeAreaInsets();
  const [card, setCard] = useState<CharacterCard | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => { if (id) getCharacter(id).then(c => { setCard(c); setLoading(false); }); }, [id]);
  if (loading || !card) return <LoadingScreen />;
  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <Header showBack /><CharacterDetailView card={card} />
    </View>
  );
}

const styles = StyleSheet.create({ container: { flex: 1, backgroundColor: colors.bg.primary } });
