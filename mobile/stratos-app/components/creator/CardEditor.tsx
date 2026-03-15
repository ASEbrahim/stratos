import React, { useState } from 'react';
import { View, Text, TextInput, ScrollView, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import * as DocumentPicker from 'expo-document-picker';
import { Upload } from 'lucide-react-native';
import { CharacterCardCreate, getQualityScore } from '../../lib/types';
import { createCharacter } from '../../lib/characters';
import { parseTavernCard } from '../../lib/tavern-import';
import { incrementStat } from '../../lib/storage';
import { GuidedFields } from './GuidedFields';
import { AvatarPicker } from './AvatarPicker';
import { GENRES } from '../../constants/genres';
import { FEATURES } from '../../constants/config';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { useThemeStore } from '../../stores/themeStore';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, withSequence } from 'react-native-reanimated';

function WordCount({ text }: { text: string }) {
  const tc = useThemeStore(s => s.colors);
  if (!text.trim()) return null;
  const words = text.trim().split(/\s+/).length;
  return <Text style={{ fontSize: 9, color: tc.text.muted, textAlign: 'right', marginTop: 2 }}>{words} words · {text.length} chars</Text>;
}

export function CardEditor() {
  const router = useRouter();
  const tc = useThemeStore(s => s.colors);
  const [mode, setMode] = useState<'quick' | 'advanced'>('quick');
  const [saving, setSaving] = useState(false);
  const [importing, setImporting] = useState(false);
  const [card, setCard] = useState<CharacterCardCreate>({
    name: '', description: '', personality: '', scenario: '', first_message: '',
    physical_description: '', speech_pattern: '', emotional_trigger: '',
    defensive_mechanism: '', vulnerability: '', specific_detail: '',
    genre_tags: [], content_rating: 'sfw', avatar_url: '',
  });

  const saveBtnScale = useSharedValue(1);
  const saveBtnAnimStyle = useAnimatedStyle(() => ({ transform: [{ scale: saveBtnScale.value }] }));

  const update = (key: keyof CharacterCardCreate, value: string | string[]) => setCard(prev => ({ ...prev, [key]: value }));
  const toggleGenre = (id: string) => setCard(prev => ({
    ...prev, genre_tags: prev.genre_tags.includes(id) ? prev.genre_tags.filter(g => g !== id) : [...prev.genre_tags, id],
  }));

  const quality = getQualityScore(card);
  const qualityColor = tc.quality[quality.level];

  const handleSave = async () => {
    if (!card.name.trim()) { Alert.alert('Missing Name', 'Give your character a name.'); return; }
    setSaving(true);
    saveBtnScale.value = withSequence(withSpring(0.95, { damping: 15 }), withSpring(1, { damping: 10 }));
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    try {
      await createCharacter(card);
      await incrementStat('totalCharacters');
      Alert.alert('Created!', 'Your character has been created.', [{ text: 'OK', onPress: () => router.back() }]);
    } catch { Alert.alert('Error', 'Failed to create character.'); }
    finally { setSaving(false); }
  };

  const handleImportTavern = async () => {
    try {
      setImporting(true);
      // Use file picker for PNG files
      const result = await DocumentPicker.getDocumentAsync({
        type: 'image/png',
        copyToCacheDirectory: true,
      });
      if (result.canceled || !result.assets?.[0]) { setImporting(false); return; }
      const parsed = await parseTavernCard(result.assets[0].uri);
      if (!parsed) {
        Alert.alert('Import Failed', 'No TavernCard V2 data found in this PNG. Make sure it\'s a valid character card file.');
        setImporting(false);
        return;
      }
      setCard(parsed);
      setMode('advanced'); // Show all imported fields
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Alert.alert('Imported!', `"${parsed.name}" loaded. Review and edit before saving.`);
    } catch {
      Alert.alert('Error', 'Failed to import card file.');
    } finally { setImporting(false); }
  };

  return (
    <ScrollView style={[styles.container, { backgroundColor: tc.bg.primary }]} contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
      {/* Mode toggle */}
      <View style={[styles.modeToggle, { backgroundColor: tc.bg.tertiary }]}>
        <TouchableOpacity style={[styles.modeBtn, mode === 'quick' && { backgroundColor: tc.bg.elevated }]} onPress={() => setMode('quick')}>
          <Text style={[styles.modeText, { color: tc.text.muted }, mode === 'quick' && { color: tc.text.primary }]}>Quick</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.modeBtn, mode === 'advanced' && { backgroundColor: tc.bg.elevated }]} onPress={() => setMode('advanced')}>
          <Text style={[styles.modeText, { color: tc.text.muted }, mode === 'advanced' && { color: tc.text.primary }]}>Advanced</Text>
        </TouchableOpacity>
      </View>

      {/* TavernCard import */}
      {FEATURES.enableTavernImport && (
        <TouchableOpacity style={[styles.importBtn, { borderColor: tc.accent.primary + '40', backgroundColor: tc.accent.primary + '08' }]} onPress={handleImportTavern} disabled={importing} activeOpacity={0.7}>
          <Upload size={16} color={tc.accent.primary} />
          <Text style={[styles.importText, { color: tc.accent.primary }]}>{importing ? 'Importing...' : 'Import TavernCard V2'}</Text>
        </TouchableOpacity>
      )}

      {/* Quality indicator */}
      <View style={[styles.qualityBanner, { borderColor: qualityColor + '40' }]}>
        <Text style={[styles.qualityLabel, { color: qualityColor }]}>{quality.label} ({quality.score}/6 elements)</Text>
      </View>

      {/* Avatar picker */}
      <AvatarPicker
        avatarUri={card.avatar_url}
        onPick={(uri) => update('avatar_url', uri)}
        onClear={() => update('avatar_url', '')}
      />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Name *</Text>
      <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.name} onChangeText={v => update('name', v)} placeholder="Character name" placeholderTextColor={tc.text.muted} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Genre</Text>
      <View style={styles.genreGrid}>
        {GENRES.map(g => {
          const sel = card.genre_tags.includes(g.id);
          return (
            <TouchableOpacity key={g.id} style={[styles.genreChip, { backgroundColor: sel ? g.color + '20' : tc.bg.tertiary, borderColor: sel ? g.color + '60' : tc.border.subtle }]} onPress={() => toggleGenre(g.id)}>
              <Text style={{ color: sel ? g.color : tc.text.secondary, fontSize: 13 }}>{g.emoji} {g.label}</Text>
            </TouchableOpacity>
          );
        })}
      </View>

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Content Rating</Text>
      <View style={styles.contentRatingRow}>
        <TouchableOpacity style={[styles.contentRatingBtn, { backgroundColor: card.content_rating === 'sfw' ? tc.status.success + '20' : tc.bg.tertiary, borderColor: card.content_rating === 'sfw' ? tc.status.success + '60' : tc.border.subtle }]} onPress={() => { update('content_rating', 'sfw'); Haptics.selectionAsync(); }}>
          <Text style={{ color: card.content_rating === 'sfw' ? tc.status.success : tc.text.muted, fontSize: 13, fontWeight: '600' }}>SFW</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.contentRatingBtn, { backgroundColor: card.content_rating === 'nsfw' ? tc.nsfw + '20' : tc.bg.tertiary, borderColor: card.content_rating === 'nsfw' ? tc.nsfw + '60' : tc.border.subtle }]} onPress={() => { update('content_rating', 'nsfw'); Haptics.selectionAsync(); }}>
          <Text style={{ color: card.content_rating === 'nsfw' ? tc.nsfw : tc.text.muted, fontSize: 13, fontWeight: '600' }}>18+</Text>
        </TouchableOpacity>
      </View>

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Description</Text>
      <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.description} onChangeText={v => update('description', v)} placeholder="Who is this character?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.description} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Personality</Text>
      <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.personality} onChangeText={v => update('personality', v)} placeholder="How does your character act and speak?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.personality} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>First Message</Text>
      <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>Sets the tone for every conversation — the most important field.</Text>
      <TextInput style={[styles.input, styles.multilineLg, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.first_message} onChangeText={v => update('first_message', v)} placeholder="What does your character say or do first?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.first_message} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Scenario</Text>
      <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.scenario} onChangeText={v => update('scenario', v)} placeholder="The starting situation" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.scenario} />

      {mode === 'advanced' && (
        <>
          <View style={[styles.divider, { backgroundColor: tc.border.subtle }]} />
          <Text style={[styles.sectionTitle, { color: tc.text.primary }]}>Quality Elements</Text>
          <GuidedFields fields={[
            { key: 'physical_description', label: 'Physical Description', hint: 'What do they look like? One unique detail.', example: 'Tall, clouded left eye...', value: card.physical_description, onChangeText: v => update('physical_description', v) },
            { key: 'speech_pattern', label: 'Speech Pattern', hint: 'How do they talk?', example: 'Formal, archaic...', value: card.speech_pattern, onChangeText: v => update('speech_pattern', v) },
            { key: 'emotional_trigger', label: 'Emotional Trigger', hint: 'What sets them off?', example: 'Any mention of cowardice...', value: card.emotional_trigger, onChangeText: v => update('emotional_trigger', v) },
            { key: 'defensive_mechanism', label: 'Defensive Mechanism', hint: 'How do they protect themselves?', example: 'Deflects with formality...', value: card.defensive_mechanism, onChangeText: v => update('defensive_mechanism', v) },
            { key: 'vulnerability', label: 'Vulnerability', hint: 'The crack in the armor?', example: 'Children playing...', value: card.vulnerability, onChangeText: v => update('vulnerability', v) },
            { key: 'specific_detail', label: 'Specific Detail', hint: 'One concrete, grounding detail.', example: 'Traces sigil when anxious...', value: card.specific_detail, onChangeText: v => update('specific_detail', v) },
          ]} />
        </>
      )}

      <View style={styles.ratingRow}>
        <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Content Rating</Text>
        <View style={styles.contentRatingRow}>
          <TouchableOpacity style={[styles.ratingOpt, card.content_rating === 'sfw' && { backgroundColor: tc.status.success + '20', borderColor: tc.status.success }]} onPress={() => update('content_rating', 'sfw')}>
            <Text style={{ color: card.content_rating === 'sfw' ? tc.status.success : tc.text.muted }}>SFW</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.ratingOpt, card.content_rating === 'nsfw' && { backgroundColor: tc.nsfw + '20', borderColor: tc.nsfw }]} onPress={() => update('content_rating', 'nsfw')}>
            <Text style={{ color: card.content_rating === 'nsfw' ? tc.nsfw : tc.text.muted }}>NSFW</Text>
          </TouchableOpacity>
        </View>
      </View>

      <Animated.View style={saveBtnAnimStyle}>
        <TouchableOpacity style={[styles.saveBtn, { backgroundColor: tc.accent.primary }, saving && { opacity: 0.6 }]} onPress={handleSave} disabled={saving}>
          <Text style={styles.saveBtnText}>{saving ? 'Creating...' : 'Create Character'}</Text>
        </TouchableOpacity>
      </Animated.View>
      <View style={{ height: spacing.xxl }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: spacing.lg },
  modeToggle: { flexDirection: 'row', borderRadius: borderRadius.lg, padding: 3, marginBottom: spacing.lg },
  modeBtn: { flex: 1, paddingVertical: spacing.sm, alignItems: 'center', borderRadius: borderRadius.md },
  modeBtnActive: {},
  modeText: { ...typography.subheading, fontSize: 14 },
  modeTextActive: {},
  qualityBanner: { borderWidth: 1, borderRadius: borderRadius.md, padding: spacing.md, alignItems: 'center', marginBottom: spacing.lg },
  qualityLabel: { ...typography.subheading, fontSize: 14 },
  fieldLabel: { ...typography.subheading, marginBottom: spacing.xs, marginTop: spacing.lg },
  fieldHint: { ...typography.caption, marginBottom: spacing.sm },
  input: { borderRadius: borderRadius.md, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, fontSize: 15, borderWidth: 1 },
  multiline: { minHeight: 80, textAlignVertical: 'top' },
  multilineLg: { minHeight: 140, textAlignVertical: 'top' },
  genreGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  genreChip: { paddingHorizontal: spacing.md, paddingVertical: spacing.sm, borderRadius: borderRadius.full, borderWidth: 1 },
  contentRatingRow: { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.md },
  contentRatingBtn: { flex: 1, paddingVertical: spacing.md, borderRadius: borderRadius.md, borderWidth: 1, alignItems: 'center' },
  divider: { height: 1, marginVertical: spacing.xl },
  sectionTitle: { ...typography.heading, marginBottom: spacing.lg },
  ratingRow: { marginTop: spacing.lg },
  ratingToggle: { flexDirection: 'row', gap: spacing.sm, marginTop: spacing.sm },
  ratingOpt: { paddingHorizontal: spacing.xl, paddingVertical: spacing.sm, borderRadius: borderRadius.md, borderWidth: 1 },
  importBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.md, borderRadius: borderRadius.md, borderWidth: 1, marginBottom: spacing.lg },
  importText: { ...typography.caption, fontWeight: '600' },
  saveBtn: { paddingVertical: spacing.lg, borderRadius: borderRadius.lg, alignItems: 'center', marginTop: spacing.xxl },
  saveBtnText: { ...typography.subheading, color: '#fff' },
});
