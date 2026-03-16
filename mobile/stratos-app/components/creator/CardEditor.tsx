import React, { useState } from 'react';
import { View, Text, TextInput, ScrollView, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import * as DocumentPicker from 'expo-document-picker';
import { Upload, Wand2 } from 'lucide-react-native';
import { CharacterCardCreate, getQualityScore } from '../../lib/types';
import { createCharacter } from '../../lib/characters';
import { parseTavernCard } from '../../lib/tavern-import';
import { incrementStat } from '../../lib/storage';
import { GuidedFields } from './GuidedFields';
import { AvatarPicker } from './AvatarPicker';
import { GENRES } from '../../constants/genres';
import { FEATURES } from '../../constants/config';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';
import { useThemeStore } from '../../stores/themeStore';
import { useCharacterStore } from '../../stores/characterStore';
import Animated, { useSharedValue, useAnimatedStyle, withSpring, withSequence } from 'react-native-reanimated';

function WordCount({ text }: { text: string }) {
  const tc = useThemeStore(s => s.colors);
  if (!text.trim()) return null;
  const words = text.trim().split(/\s+/).length;
  return <Text style={{ fontSize: 9, color: tc.text.muted, textAlign: 'right', marginTop: 2 }}>{words} words · {text.length} chars</Text>;
}

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
  } : {
    name: '', description: '', personality: '', scenario: '', first_message: '',
    physical_description: '', speech_pattern: '', emotional_trigger: '',
    defensive_mechanism: '', vulnerability: '', specific_detail: '',
    genre_tags: [], content_rating: 'sfw', avatar_url: '',
  });

  const { loadMyCards, loadNew } = useCharacterStore();
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
          }),
        });
        loadMyCards().catch(() => {});
        loadNew().catch(() => {});
        Alert.alert('Updated!', 'Character has been updated.', [{ text: 'OK', onPress: () => router.back() }]);
      } else {
        await createCharacter(card);
        await incrementStat('totalCharacters');
        loadMyCards().catch(() => {});
        loadNew().catch(() => {});
        Alert.alert('Created!', 'Your character has been created.', [{ text: 'OK', onPress: () => router.back() }]);
      }
    } catch { Alert.alert('Error', isEditing ? 'Failed to update character.' : 'Failed to create character.'); }
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
        <TouchableOpacity style={[styles.modeBtn, mode === 'quick' && { backgroundColor: tc.bg.elevated }]} onPress={() => setMode('quick')} accessibilityLabel={`Simple mode${mode === 'quick' ? ', selected' : ''}`} accessibilityRole="button">
          <Text style={[styles.modeText, { color: tc.text.muted }, mode === 'quick' && { color: tc.text.primary }]}>Simple</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.modeBtn, mode === 'advanced' && { backgroundColor: tc.bg.elevated }]} onPress={() => setMode('advanced')} accessibilityLabel={`Advanced mode${mode === 'advanced' ? ', selected' : ''}`} accessibilityRole="button">
          <Text style={[styles.modeText, { color: tc.text.muted }, mode === 'advanced' && { color: tc.text.primary }]}>Advanced</Text>
        </TouchableOpacity>
      </View>

      {/* TavernCard import */}
      {FEATURES.enableTavernImport && (
        <TouchableOpacity style={[styles.importBtn, { borderColor: tc.accent.primary + '40', backgroundColor: tc.accent.primary + '08' }]} onPress={handleImportTavern} disabled={importing} activeOpacity={0.7} accessibilityLabel={importing ? 'Importing TavernCard' : 'Import TavernCard V2'} accessibilityRole="button">
          <Upload size={16} color={tc.accent.primary} />
          <Text style={[styles.importText, { color: tc.accent.primary }]}>{importing ? 'Importing...' : 'Import TavernCard V2'}</Text>
        </TouchableOpacity>
      )}

      {/* Generate Image — accessible from top of editor */}
      {card.name.trim() && (
        <TouchableOpacity
          style={[styles.importBtn, { borderColor: tc.accent.secondary + '40', backgroundColor: tc.accent.secondary + '08' }]}
          onPress={() => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            const desc = card.physical_description?.trim() || card.personality?.trim() || card.description?.trim() || card.name;
            router.push({ pathname: '/imagegen', params: { name: card.name, description: desc, ...(initialCard?.id ? { card_id: initialCard.id } : {}) } });
          }}
          activeOpacity={0.7}
        >
          <Wand2 size={16} color={tc.accent.secondary} />
          <Text style={[styles.importText, { color: tc.accent.secondary }]}>Generate Image</Text>
        </TouchableOpacity>
      )}

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
        <>
          {/* ─── QUICK MODE ─── */}
          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Description</Text>
          <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.description} onChangeText={v => update('description', v)} placeholder="Who is this character?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" accessibilityLabel="Character description" />
          <WordCount text={card.description} />

          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Personality</Text>
          <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.personality} onChangeText={v => update('personality', v)} placeholder="How does your character act and speak?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" accessibilityLabel="Character personality" />
          <WordCount text={card.personality} />

          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>First Message</Text>
          <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>Sets the tone for every conversation.</Text>
          <TextInput style={[styles.input, styles.multilineLg, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.first_message} onChangeText={v => update('first_message', v)} placeholder="What does your character say or do first?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" accessibilityLabel="First message" />
          <WordCount text={card.first_message} />

          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Scenario</Text>
          <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.scenario} onChangeText={v => update('scenario', v)} placeholder="The starting situation" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" accessibilityLabel="Scenario" />
          <WordCount text={card.scenario} />
        </>
      ) : (
        <>
          {/* ─── ADVANCED MODE ─── */}
          {/* Section: Identity */}
          <Text style={[styles.sectionTitle, { color: tc.accent.primary }]}>Identity</Text>
          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Description</Text>
          <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.description} onChangeText={v => update('description', v)} placeholder="Who is this character?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
          <WordCount text={card.description} />

          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Physical Description</Text>
          <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>What do they look like? Include one unique detail.</Text>
          <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.physical_description} onChangeText={v => update('physical_description', v)} placeholder="Tall, clouded left eye, worn leather armor..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
          <WordCount text={card.physical_description} />

          {/* Section: Behavior */}
          <View style={[styles.divider, { backgroundColor: tc.border.subtle }]} />
          <Text style={[styles.sectionTitle, { color: tc.accent.primary }]}>Behavior</Text>

          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Personality</Text>
          <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.personality} onChangeText={v => update('personality', v)} placeholder="How does your character act and speak?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
          <WordCount text={card.personality} />

          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Speech Pattern</Text>
          <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>How do they talk? Accent, vocabulary, quirks.</Text>
          <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.speech_pattern} onChangeText={v => update('speech_pattern', v)} placeholder="Formal, archaic phrasing, never uses contractions..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
          <WordCount text={card.speech_pattern} />

          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>First Message</Text>
          <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>Sets the tone for every conversation — the most important field.</Text>
          <TextInput style={[styles.input, styles.multilineLg, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.first_message} onChangeText={v => update('first_message', v)} placeholder="What does your character say or do first?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
          <WordCount text={card.first_message} />

          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Scenario</Text>
          <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.scenario} onChangeText={v => update('scenario', v)} placeholder="The starting situation" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
          <WordCount text={card.scenario} />

          {/* Section: Depth */}
          <View style={[styles.divider, { backgroundColor: tc.border.subtle }]} />
          <Text style={[styles.sectionTitle, { color: tc.accent.primary }]}>Depth</Text>

          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Emotional Trigger</Text>
          <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>What sets them off emotionally?</Text>
          <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.emotional_trigger} onChangeText={v => update('emotional_trigger', v)} placeholder="Any mention of cowardice sends them into a rage..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
          <WordCount text={card.emotional_trigger} />

          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Defensive Mechanism</Text>
          <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>How do they protect themselves?</Text>
          <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.defensive_mechanism} onChangeText={v => update('defensive_mechanism', v)} placeholder="Deflects with cold formality, changes the subject..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
          <WordCount text={card.defensive_mechanism} />

          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Vulnerability</Text>
          <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>The crack in the armor. What breaks through?</Text>
          <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.vulnerability} onChangeText={v => update('vulnerability', v)} placeholder="Children playing reminds them of what they lost..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
          <WordCount text={card.vulnerability} />

          <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Specific Detail</Text>
          <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>One concrete, grounding detail that makes them real.</Text>
          <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.specific_detail} onChangeText={v => update('specific_detail', v)} placeholder="Traces the sigil on their ring when anxious..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
          <WordCount text={card.specific_detail} />
        </>
      )}

      <View style={styles.ratingRow}>
        <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Content Rating</Text>
        <View style={styles.contentRatingRow}>
          <TouchableOpacity style={[styles.ratingOpt, card.content_rating === 'sfw' && { backgroundColor: tc.status.success + '20', borderColor: tc.status.success }]} onPress={() => update('content_rating', 'sfw')} accessibilityLabel={`SFW content rating${card.content_rating === 'sfw' ? ', selected' : ''}`} accessibilityRole="button">
            <Text style={{ color: card.content_rating === 'sfw' ? tc.status.success : tc.text.muted }}>SFW</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.ratingOpt, card.content_rating === 'nsfw' && { backgroundColor: tc.nsfw + '20', borderColor: tc.nsfw }]} onPress={() => update('content_rating', 'nsfw')} accessibilityLabel={`NSFW content rating${card.content_rating === 'nsfw' ? ', selected' : ''}`} accessibilityRole="button">
            <Text style={{ color: card.content_rating === 'nsfw' ? tc.nsfw : tc.text.muted }}>NSFW</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Generate Image button — navigates to full image gen screen with NSFW toggle */}
      {card.name.trim() ? (
        <TouchableOpacity
          style={[styles.genImageBtn, { borderColor: tc.accent.secondary + '40', backgroundColor: tc.accent.secondary + '08' }]}
          onPress={() => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            const desc = card.physical_description?.trim() || card.personality?.trim() || card.description?.trim() || card.name;
            router.push({ pathname: '/imagegen', params: { name: card.name, description: desc, ...(initialCard?.id ? { card_id: initialCard.id } : {}) } });
          }}
          activeOpacity={0.7}
        >
          <Wand2 size={14} color={tc.accent.secondary} />
          <Text style={[styles.genImageText, { color: tc.accent.secondary }]}>Generate Character Image</Text>
        </TouchableOpacity>
      ) : null}

      <Animated.View style={saveBtnAnimStyle}>
        <TouchableOpacity style={[styles.saveBtn, { backgroundColor: tc.accent.primary }, saving && { opacity: 0.6 }]} onPress={handleSave} disabled={saving} accessibilityLabel={saving ? (isEditing ? 'Saving changes' : 'Creating character') : (isEditing ? 'Save changes' : 'Create character')} accessibilityRole="button">
          <Text style={styles.saveBtnText}>{saving ? (isEditing ? 'Saving...' : 'Creating...') : (isEditing ? 'Save Changes' : 'Create Character')}</Text>
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
  avatarNameRow: { flexDirection: 'row', gap: spacing.md, alignItems: 'flex-start', marginBottom: spacing.sm },
  nameCol: { flex: 1, justifyContent: 'center' },
  fieldLabel: { ...typography.subheading, marginBottom: spacing.xs, marginTop: spacing.lg },
  fieldHint: { ...typography.caption, marginBottom: spacing.sm },
  input: { borderRadius: borderRadius.md, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, fontSize: 15, borderWidth: 1 },
  multiline: { minHeight: 80, textAlignVertical: 'top' },
  multilineLg: { minHeight: 140, textAlignVertical: 'top' },
  genreGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  genreChip: { paddingHorizontal: spacing.md, paddingVertical: spacing.sm, borderRadius: borderRadius.full, borderWidth: 1 },
  contentRatingRow: { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.md },
  contentRatingBtn: { flex: 1, paddingVertical: spacing.md, borderRadius: borderRadius.md, borderWidth: 1, alignItems: 'center' },
  divider: { height: 1, marginVertical: spacing.lg },
  sectionTitle: { ...typography.subheading, fontSize: 13, textTransform: 'uppercase', letterSpacing: 1, marginTop: spacing.lg, marginBottom: spacing.xs },
  importBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.md, borderRadius: borderRadius.md, borderWidth: 1, marginBottom: spacing.lg },
  ratingRow: { marginTop: spacing.lg },
  ratingToggle: { flexDirection: 'row', gap: spacing.sm, marginTop: spacing.sm },
  ratingOpt: { paddingHorizontal: spacing.xl, paddingVertical: spacing.sm, borderRadius: borderRadius.md, borderWidth: 1 },
  importText: { ...typography.caption, fontWeight: '600' },
  genImageBtn: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.md, borderRadius: borderRadius.lg, borderWidth: 1, marginTop: spacing.lg },
  genImageText: { fontSize: 13, fontFamily: fonts.button },
  saveBtn: { paddingVertical: spacing.lg, borderRadius: borderRadius.lg, alignItems: 'center', marginTop: spacing.md },
  saveBtnText: { ...typography.subheading, color: '#fff' },
  previewOverlay: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.7)', padding: spacing.xl },
  previewCard: { width: '100%', borderRadius: borderRadius.lg, borderWidth: 1, overflow: 'hidden', padding: spacing.sm },
  previewImage: { width: '100%', aspectRatio: 3 / 4, borderRadius: borderRadius.md },
  previewHint: { fontSize: 10, textAlign: 'center', marginTop: spacing.sm },
});
