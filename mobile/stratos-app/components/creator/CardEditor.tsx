import React, { useState, useCallback } from 'react';
import { View, Text, TextInput, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { useThemedAlert } from '../shared/ThemedAlert';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import * as DocumentPicker from 'expo-document-picker';
import { CharacterCardCreate } from '../../lib/types';
import { createCharacter } from '../../lib/characters';
import { parseTavernCard } from '../../lib/tavern-import';
import { incrementStat } from '../../lib/storage';
import { reportError } from '../../lib/utils';
import { AvatarPicker } from './AvatarPicker';
import { SimpleEditor } from './SimpleEditor';
import { AdvancedEditor } from './AdvancedEditor';
import { EditorActions } from './EditorActions';
import { GENRES } from '../../constants/genres';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { useThemeStore } from '../../stores/themeStore';
import { useCharacterStore } from '../../stores/characterStore';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, withSequence } from 'react-native-reanimated';

interface CardEditorProps {
  initialCard?: import('../../lib/types').CharacterCard;
  prefillData?: import('../../lib/types').CharacterCard;
}

export function CardEditor({ initialCard, prefillData }: CardEditorProps) {
  const router = useRouter();
  const tc = useThemeStore(s => s.colors);
  const isEditing = !!initialCard;  // Only true for Edit, NOT for Copy
  const source = initialCard || prefillData;
  const [mode, setMode] = useState<'quick' | 'advanced'>((isEditing || prefillData) ? 'advanced' : 'quick');
  const [saving, setSaving] = useState(false);
  const [importing, setImporting] = useState(false);
  const [card, setCard] = useState<CharacterCardCreate>(source ? {
    name: source.name,
    description: source.description || source.personality || '',
    personality: source.personality,
    scenario: source.scenario,
    first_message: source.first_message,
    physical_description: source.physical_description,
    speech_pattern: source.speech_pattern,
    emotional_trigger: source.emotional_trigger,
    defensive_mechanism: source.defensive_mechanism,
    vulnerability: source.vulnerability,
    specific_detail: source.specific_detail,
    genre_tags: source.genre_tags || [],
    content_rating: source.content_rating || 'sfw',
    avatar_url: '',  // Don't copy avatar — user generates their own
    // Pill fields
    gender: source.gender,
    archetype_override: source.archetype_override,
    narration_pov: source.narration_pov,
    relationship_to_user: source.relationship_to_user,
    nsfw_comfort: source.nsfw_comfort,
    response_length_pref: source.response_length_pref,
    age_range: source.age_range,
    personality_tags: source.personality_tags || [],
  } : {
    name: '', description: '', personality: '', scenario: '', first_message: '',
    physical_description: '', speech_pattern: '', emotional_trigger: '',
    defensive_mechanism: '', vulnerability: '', specific_detail: '',
    genre_tags: [], content_rating: 'sfw', avatar_url: '',
    personality_tags: [],
  });

  const { loadMyCards, loadNew } = useCharacterStore();
  const saveBtnScale = useSharedValue(1);
  const saveBtnAnimStyle = useAnimatedStyle(() => ({ transform: [{ scale: saveBtnScale.value }] }));
  const { alert: showAlert, AlertComponent } = useThemedAlert();

  const update = useCallback((key: keyof CharacterCardCreate, value: string | string[] | undefined) => setCard(prev => ({ ...prev, [key]: value })), []);
  const toggleGenre = (id: string) => setCard(prev => ({
    ...prev, genre_tags: prev.genre_tags.includes(id) ? prev.genre_tags.filter(g => g !== id) : [...prev.genre_tags, id],
  }));

  const handleSave = async () => {
    if (!card.name.trim()) { showAlert('Missing Name', 'Give your character a name.'); return; }
    setSaving(true);
    saveBtnScale.value = withSequence(withSpring(0.95, { damping: 15 }), withSpring(1, { damping: 10 }));
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    try {
      if (isEditing && initialCard) {
        // Update existing card via API
        const { apiFetch } = await import('../../lib/api');
        await apiFetch(`/api/cards/${initialCard.id}`, {
          method: 'PUT',
          body: JSON.stringify({
            name: card.name, personality: card.personality, scenario: card.scenario,
            first_message: card.first_message, physical_description: card.physical_description,
            speech_pattern: card.speech_pattern, emotional_trigger: card.emotional_trigger,
            defensive_mechanism: card.defensive_mechanism, vulnerability: card.vulnerability,
            specific_detail: card.specific_detail, genre_tags: card.genre_tags,
            content_rating: card.content_rating,
            gender: card.gender, archetype_override: card.archetype_override,
            narration_pov: card.narration_pov, relationship_to_user: card.relationship_to_user,
            nsfw_comfort: card.nsfw_comfort, response_length_pref: card.response_length_pref,
            age_range: card.age_range, personality_tags: card.personality_tags,
          }),
        });
        loadMyCards().catch(err => reportError('CardEditor:loadMyCards', err));
        loadNew().catch(err => reportError('CardEditor:loadNew', err));
        showAlert('Updated!', 'Character has been updated.', [{ text: 'OK', onPress: () => router.back() }]);
      } else {
        await createCharacter(card);
        await incrementStat('totalCharacters');
        loadMyCards().catch(err => reportError('CardEditor:loadMyCards', err));
        loadNew().catch(err => reportError('CardEditor:loadNew', err));
        showAlert('Created!', 'Your character has been created.', [{ text: 'OK', onPress: () => router.back() }]);
      }
    } catch (err) { reportError('CardEditor:handleSave', err); showAlert('Error', isEditing ? 'Failed to update character.' : 'Failed to create character.'); }
    finally { setSaving(false); }
  };

  const handleImportTavern = async () => {
    try {
      setImporting(true);
      const result = await DocumentPicker.getDocumentAsync({
        type: ['image/png', 'application/json'],
        copyToCacheDirectory: true,
      });
      if (result.canceled || !result.assets?.[0]) { setImporting(false); return; }
      const asset = result.assets[0];
      const parsed = await parseTavernCard(asset.uri, asset.mimeType ?? undefined, (asset as any).file);
      if (!parsed) {
        showAlert('Import Failed', 'No character data found. Supports TavernCard V2 PNG and character JSON files.');
        setImporting(false);
        return;
      }
      setCard(parsed);
      setMode('advanced'); // Show all imported fields
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      showAlert('Imported!', `"${parsed.name}" loaded. Review and edit before saving.`);
    } catch (err) {
      reportError('CardEditor:handleImportTavern', err);
      showAlert('Error', 'Failed to import card file.');
    } finally { setImporting(false); }
  };

  const handleGenerateImage = useCallback(() => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    const desc = card.physical_description?.trim() || card.personality?.trim() || card.description?.trim() || card.name;
    router.push({ pathname: '/imagegen', params: { name: card.name, description: desc, ...(initialCard?.id ? { card_id: initialCard.id } : {}) } });
  }, [card.physical_description, card.personality, card.description, card.name, initialCard?.id, router]);

  const handleUpdateContentRating = useCallback((rating: string) => {
    update('content_rating', rating);
  }, [update]);

  return (
    <ScrollView style={[styles.container, { backgroundColor: tc.bg.primary }]} contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
      {/* Mode toggle */}
      <View style={[styles.modeToggle, { backgroundColor: tc.bg.tertiary }]}>
        <TouchableOpacity style={[styles.modeBtn, mode === 'quick' && { backgroundColor: tc.bg.elevated }]} onPress={() => setMode('quick')} accessibilityLabel={`Simple mode${mode === 'quick' ? ', selected' : ''}`} accessibilityRole="button">
          <Text style={[styles.modeText, { color: tc.text.muted }, mode === 'quick' && { color: tc.text.primary }]}>Simple</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.modeBtn, mode === 'advanced' && { backgroundColor: tc.bg.elevated }]} onPress={() => setMode('advanced')} accessibilityLabel={`Advanced mode${mode === 'advanced' ? ', selected' : ''}`} accessibilityRole="button">
          <Text style={[styles.modeText, { color: tc.text.muted }, mode === 'advanced' && { color: tc.text.primary }]}>Advanced</Text>
        </TouchableOpacity>
      </View>

      {/* Editor actions: import + generate image (top) */}
      <EditorActions
        card={card}
        isEditing={isEditing}
        saving={saving}
        importing={importing}
        saveBtnAnimStyle={saveBtnAnimStyle}
        onSave={handleSave}
        onImportTavern={handleImportTavern}
        onGenerateImage={handleGenerateImage}
        onUpdateContentRating={handleUpdateContentRating}
      />

      {/* Avatar + Name row — shared between both modes */}
      <View style={styles.avatarNameRow}>
        <AvatarPicker
          avatarUri={card.avatar_url}
          onPick={(uri) => update('avatar_url', uri)}
          onClear={() => update('avatar_url', '')}
          compact
        />
        <View style={styles.nameCol}>
          <Text style={[styles.fieldLabel, { color: tc.text.primary, marginTop: 0 }]}>Name *</Text>
          <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.name} onChangeText={v => update('name', v)} placeholder="Character name" placeholderTextColor={tc.text.muted} accessibilityLabel="Character name" />
        </View>
      </View>

      {/* Genre + Rating row — shared */}
      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Genre</Text>
      <View style={styles.genreGrid}>
        {GENRES.map(g => {
          const sel = card.genre_tags.includes(g.id);
          return (
            <TouchableOpacity key={g.id} style={[styles.genreChip, { backgroundColor: sel ? g.color + '20' : tc.bg.tertiary, borderColor: sel ? g.color + '60' : tc.border.subtle }]} onPress={() => toggleGenre(g.id)} accessibilityLabel={`${g.label} genre${sel ? ', selected' : ''}`} accessibilityRole="button">
              <Text style={{ color: sel ? g.color : tc.text.secondary, fontSize: 12, fontWeight: '600' }} numberOfLines={1}>{g.label}</Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {mode === 'quick' ? (
        <SimpleEditor card={card} onUpdate={update} />
      ) : (
        <AdvancedEditor card={card} onUpdate={update} />
      )}

      <View style={{ height: spacing.xxl }} />
      {AlertComponent}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: spacing.lg },
  modeToggle: { flexDirection: 'row', borderRadius: borderRadius.lg, padding: 3, marginBottom: spacing.lg },
  modeBtn: { flex: 1, paddingVertical: spacing.sm, alignItems: 'center', borderRadius: borderRadius.md },
  modeText: { ...typography.subheading, fontSize: 14 },
  avatarNameRow: { flexDirection: 'row', gap: spacing.md, alignItems: 'flex-start', marginBottom: spacing.sm },
  nameCol: { flex: 1, justifyContent: 'center' },
  fieldLabel: { ...typography.subheading, marginBottom: spacing.xs, marginTop: spacing.lg },
  input: { borderRadius: borderRadius.md, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, fontSize: 15, borderWidth: 1 },
  genreGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  genreChip: { paddingHorizontal: spacing.md, paddingVertical: spacing.sm, borderRadius: borderRadius.full, borderWidth: 1 },
});
