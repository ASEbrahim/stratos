import React, { useEffect, useState, useCallback } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useLocalSearchParams } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Header } from '../../components/shared/Header';
import { CharacterDetailView } from '../../components/cards/CharacterDetail';
import { LoadingScreen } from '../../components/shared/LoadingScreen';
import { CharacterCard } from '../../lib/types';
import { getCharacter } from '../../lib/characters';
import { useThemeStore } from '../../stores/themeStore';
import { reportError } from '../../lib/utils';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';

export default function CharacterDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const insets = useSafeAreaInsets();
  const tc = useThemeStore(s => s.colors);
  const [card, setCard] = useState<CharacterCard | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCharacter = useCallback(async () => {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const c = await getCharacter(id);
      if (c) {
        setCard(c);
      } else {
        setError('Character not found.');
      }
    } catch (err) {
      reportError('CharacterDetailScreen:fetch', err);
      setError(err instanceof Error ? err.message : 'Failed to load character.');
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => { fetchCharacter(); }, [fetchCharacter]);

  if (loading) return <LoadingScreen />;

  if (error || !card) {
    return (
      <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
        <Header showBack />
        <View style={styles.errorContainer}>
          <Text style={[styles.errorTitle, { color: tc.text.primary }]}>Something went wrong</Text>
          <Text style={[styles.errorMessage, { color: tc.text.secondary }]}>{error || 'Character not found.'}</Text>
          <TouchableOpacity style={[styles.retryButton, { backgroundColor: tc.accent.primary }]} onPress={fetchCharacter} activeOpacity={0.7}>
            <Text style={styles.retryText}>Retry</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <Header showBack /><CharacterDetailView card={card} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  errorContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: spacing.xxl },
  errorTitle: { fontSize: 18, fontFamily: fonts.heading, marginBottom: spacing.sm },
  errorMessage: { fontSize: 14, fontFamily: fonts.body, textAlign: 'center', marginBottom: spacing.xl, lineHeight: 20 },
  retryButton: { paddingHorizontal: spacing.xxl, paddingVertical: spacing.md, borderRadius: borderRadius.lg },
  retryText: { fontSize: 15, fontFamily: fonts.heading, color: '#fff' },
});
