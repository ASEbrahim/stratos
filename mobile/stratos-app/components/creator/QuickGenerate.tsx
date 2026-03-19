import React, { useState, useCallback } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, StyleSheet, ActivityIndicator } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { Wand2 } from 'lucide-react-native';
import { Header } from '../shared/Header';
import { PillSelector, ARCHETYPE_OPTIONS, RELATIONSHIP_OPTIONS, PillOption } from './PillSelector';
import { GENRES } from '../../constants/genres';
import { useThemeStore } from '../../stores/themeStore';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';
import { quickGenerateCharacter } from '../../lib/characters';
import { reportError } from '../../lib/utils';

const GENRE_PILLS: PillOption[] = GENRES.map(g => ({ value: g.id, label: g.label }));

export default function QuickGenerateScreen() {
  const insets = useSafeAreaInsets();
  const tc = useThemeStore(s => s.colors);
  const router = useRouter();

  const [prompt, setPrompt] = useState('');
  const [genre, setGenre] = useState<string | undefined>();
  const [archetype, setArchetype] = useState<string | undefined>();
  const [relationship, setRelationship] = useState<string | undefined>();
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState('');

  const canGenerate = !!(prompt.trim() || genre || archetype);

  const handleGenerate = useCallback(async () => {
    if (!canGenerate || generating) return;
    setGenerating(true);
    setError('');
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

    try {
      const card = await quickGenerateCharacter({
        prompt: prompt.trim() || undefined,
        genre: genre || undefined,
        archetype: archetype || undefined,
        relationship: relationship || undefined,
      });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      router.push({
        pathname: '/(tabs)/create',
        params: { newCard: JSON.stringify(card) },
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Generation failed — is the server running?';
      setError(msg);
      reportError('QuickGenerate', e);
    } finally {
      setGenerating(false);
    }
  }, [canGenerate, generating, prompt, genre, archetype, relationship, router]);

  return (
    <View style={[styles.container, { paddingTop: insets.top, backgroundColor: tc.bg.primary }]}>
      <Header title="Quick Generate" showBack />
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">

        <Text style={[styles.subtitle, { color: tc.text.muted }]}>
          Describe a character or pick filters — AI generates the rest.
        </Text>

        <PillSelector
          label="Genre"
          options={GENRE_PILLS}
          value={genre}
          onChange={setGenre}
          wrap
        />

        <PillSelector
          label="Archetype"
          options={ARCHETYPE_OPTIONS}
          value={archetype}
          onChange={setArchetype}
          wrap
        />

        <PillSelector
          label="Relationship"
          options={RELATIONSHIP_OPTIONS}
          value={relationship}
          onChange={setRelationship}
          wrap
        />

        <Text style={[styles.label, { color: tc.text.secondary }]}>Description (optional)</Text>
        <TextInput
          style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]}
          value={prompt}
          onChangeText={setPrompt}
          placeholder="A shy elf librarian who speaks in riddles..."
          placeholderTextColor={tc.text.muted}
          multiline
          textAlignVertical="top"
        />

        <TouchableOpacity
          style={[styles.generateBtn, { backgroundColor: tc.accent.primary }, (!canGenerate || generating) && { opacity: 0.5 }]}
          onPress={handleGenerate}
          disabled={!canGenerate || generating}
          activeOpacity={0.7}
        >
          {generating ? (
            <>
              <ActivityIndicator size={18} color="#fff" />
              <Text style={styles.generateText}>Generating...</Text>
            </>
          ) : (
            <>
              <Wand2 size={18} color="#fff" />
              <Text style={styles.generateText}>Generate Character</Text>
            </>
          )}
        </TouchableOpacity>

        {error ? <Text style={[styles.error, { color: tc.status.error }]}>{error}</Text> : null}

        <View style={{ height: spacing.xxl * 2 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: spacing.lg },
  subtitle: { fontSize: 13, fontFamily: fonts.body, marginBottom: spacing.lg, lineHeight: 20 },
  label: { fontSize: 12, fontWeight: '600', marginBottom: 6, marginTop: spacing.md, textTransform: 'uppercase', letterSpacing: 0.5 },
  input: { borderRadius: borderRadius.md, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, fontSize: 15, fontFamily: fonts.body, minHeight: 80, borderWidth: 1, textAlignVertical: 'top' },
  generateBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.lg, borderRadius: borderRadius.lg, marginTop: spacing.xl },
  generateText: { fontSize: 16, fontFamily: fonts.heading, color: '#fff' },
  error: { fontSize: 13, fontFamily: fonts.body, textAlign: 'center', marginTop: spacing.md },
});
