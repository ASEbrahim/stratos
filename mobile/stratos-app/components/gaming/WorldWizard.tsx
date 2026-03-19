import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, Alert, StyleSheet } from 'react-native';
import { useRouter } from 'expo-router';
import { useThemeStore } from '../../stores/themeStore';
import { ThemeColors } from '../../constants/themes';
import { spacing, borderRadius } from '../../constants/theme';
import { fonts } from '../../constants/fonts';
import { PillSelector, MultiPillSelector, PillOption } from '../creator/PillSelector';
import WizardStepBar from './WizardStepBar';
import LabeledSlider from './LabeledSlider';
import GeneratedCharacters from './GeneratedCharacters';
import { createWorld, WorldWizardConfig } from '../../lib/gaming';
import { createCharacter } from '../../lib/characters';
import { ScenarioNpc, mapNpcToCard } from '../../lib/mappers';
import { CharacterCardCreate } from '../../lib/types';

// ── Wizard-specific pill options ──

const POPULAR_WORLDS: PillOption[] = [
  { value: 'SAO', label: 'SAO' },
  { value: 'Witcher', label: 'Witcher' },
  { value: 'Naruto', label: 'Naruto' },
  { value: 'Skyrim', label: 'Skyrim' },
  { value: 'Dark Souls', label: 'Dark Souls' },
  { value: 'Elden Ring', label: 'Elden Ring' },
  { value: 'Pokemon', label: 'Pokemon' },
  { value: 'One Piece', label: 'One Piece' },
  { value: 'Zelda', label: 'Zelda' },
  { value: 'Final Fantasy', label: 'Final Fantasy' },
];

const STORY_POSITIONS: PillOption[] = [
  { value: 'Beginning', label: 'Beginning' },
  { value: 'Middle', label: 'Middle' },
  { value: 'End', label: 'End' },
];

const DIFFICULTIES: PillOption[] = [
  { value: 'Casual', label: 'Casual' },
  { value: 'Normal', label: 'Normal' },
  { value: 'Hard', label: 'Hard' },
];

const CLASSES: PillOption[] = [
  { value: 'Warrior', label: 'Warrior' },
  { value: 'Mage', label: 'Mage' },
  { value: 'Rogue', label: 'Rogue' },
  { value: 'Ranger', label: 'Ranger' },
  { value: 'Cleric', label: 'Cleric' },
  { value: 'Paladin', label: 'Paladin' },
  { value: 'Bard', label: 'Bard' },
  { value: 'Monk', label: 'Monk' },
];

const EXTRAS: PillOption[] = [
  { value: 'Companions', label: 'Companions' },
  { value: 'Pets', label: 'Pets' },
  { value: 'Housing', label: 'Housing' },
  { value: 'Mounts', label: 'Mounts' },
  { value: 'Crafting', label: 'Crafting' },
  { value: 'Romance', label: 'Romance' },
];

const CHARACTER_SOURCES: PillOption[] = [
  { value: 'Canon', label: 'Canon' },
  { value: 'AI-Generated', label: 'AI-Generated' },
];

const NAME_STYLES: PillOption[] = [
  { value: 'Real', label: 'Real' },
  { value: 'Changed', label: 'Changed' },
];

const LORE_DEPTHS: PillOption[] = [
  { value: 'Light', label: 'Light' },
  { value: 'Standard', label: 'Standard' },
  { value: 'Deep', label: 'Deep' },
];

const STEPS = ['World', 'Setup', 'Character', 'Options', 'Confirm'];

export default function WorldWizard() {
  const router = useRouter();
  const tc = useThemeStore(s => s.colors);

  // Mode toggle
  const [advancedMode, setAdvancedMode] = useState(false);
  const [step, setStep] = useState(0);

  // Wizard data
  const [worldName, setWorldName] = useState('');
  const [storyPosition, setStoryPosition] = useState<string | undefined>('Beginning');
  const [startingLevel, setStartingLevel] = useState(1);
  const [difficulty, setDifficulty] = useState<string | undefined>('Normal');
  const [startingClass, setStartingClass] = useState<string | undefined>('Warrior');
  const [statSTR, setStatSTR] = useState(10);
  const [statDEX, setStatDEX] = useState(10);
  const [statINT, setStatINT] = useState(10);
  const [extras, setExtras] = useState<string[]>([]);
  const [characterSource, setCharacterSource] = useState<string | undefined>('Canon');
  const [nameStyle, setNameStyle] = useState<string | undefined>('Real');
  const [loreDepth, setLoreDepth] = useState<string | undefined>('Standard');

  // Generation state
  const [loading, setLoading] = useState(false);
  const [generatedNpcs, setGeneratedNpcs] = useState<ScenarioNpc[]>([]);
  const [worldScenario, setWorldScenario] = useState('');

  const buildConfig = (): WorldWizardConfig => ({
    name: worldName,
    description: `${worldName} world`,
    genre: 'fantasy',
    wizard_config: {
      story_position: storyPosition || 'Beginning',
      starting_level: startingLevel,
      difficulty: difficulty || 'Normal',
      starting_class: startingClass || 'Warrior',
      stats: { STR: statSTR, DEX: statDEX, INT: statINT },
      extras,
      canon_characters: characterSource === 'Canon',
      real_names: nameStyle === 'Real',
      lore_depth: loreDepth || 'Standard',
    },
  });

  const handleGenerate = async () => {
    if (!worldName.trim()) {
      Alert.alert('Missing Name', 'Please enter or select a world name.');
      return;
    }
    setLoading(true);
    try {
      const config = buildConfig();
      const result = await createWorld(config) as Record<string, unknown>;
      setWorldScenario(config.name);
      // Parse characters from response if present
      const charsRaw = result.characters_json;
      if (typeof charsRaw === 'string') {
        try {
          const parsed = JSON.parse(charsRaw);
          if (Array.isArray(parsed)) setGeneratedNpcs(parsed as ScenarioNpc[]);
        } catch { /* ignore parse errors */ }
      } else if (Array.isArray(charsRaw)) {
        setGeneratedNpcs(charsRaw as ScenarioNpc[]);
      }
      Alert.alert('World Created', `${worldName} has been generated!`);
    } catch (err) {
      Alert.alert('Error', err instanceof Error ? err.message : 'Failed to create world');
    } finally {
      setLoading(false);
    }
  };

  const handleStartChat = (card: CharacterCardCreate) => {
    // Navigate to chat with the card data
    router.push({ pathname: '/chat/[id]', params: { id: `new-${card.name}` } });
  };

  const handleImport = async (card: CharacterCardCreate) => {
    try {
      await createCharacter(card);
      Alert.alert('Imported', `${card.name} has been added to your library.`);
    } catch (err) {
      Alert.alert('Error', err instanceof Error ? err.message : 'Failed to import character');
    }
  };

  // ── Quick mode ──
  const renderQuickMode = () => (
    <ScrollView contentContainerStyle={styles.scrollContent}>
      <Text style={[styles.sectionLabel, { color: tc.text.secondary, fontFamily: fonts.bodySemiBold }]}>
        World Name
      </Text>
      <TextInput
        style={[styles.input, { color: tc.text.primary, backgroundColor: tc.bg.secondary, borderColor: tc.border.subtle, fontFamily: fonts.body }]}
        value={worldName}
        onChangeText={setWorldName}
        placeholder="Enter world name..."
        placeholderTextColor={tc.text.muted}
      />
      <PillSelector
        label="Popular Worlds"
        options={POPULAR_WORLDS}
        value={worldName || undefined}
        onChange={(v) => { if (v) setWorldName(v); }}
        wrap
      />
      <TouchableOpacity
        style={[styles.generateButton, { backgroundColor: tc.accent.primary, opacity: loading ? 0.6 : 1 }]}
        onPress={handleGenerate}
        disabled={loading}
        activeOpacity={0.7}
      >
        <Text style={[styles.generateText, { color: tc.text.inverse, fontFamily: fonts.button }]}>
          {loading ? 'Generating...' : 'Generate World'}
        </Text>
      </TouchableOpacity>
      {generatedNpcs.length > 0 && (
        <GeneratedCharacters
          characters={generatedNpcs}
          worldScenario={worldScenario}
          onStartChat={handleStartChat}
          onImport={handleImport}
        />
      )}
    </ScrollView>
  );

  // ── Advanced mode steps ──
  const renderStep = () => {
    switch (step) {
      case 0:
        return (
          <View>
            <Text style={[styles.sectionLabel, { color: tc.text.secondary, fontFamily: fonts.bodySemiBold }]}>
              World Name
            </Text>
            <TextInput
              style={[styles.input, { color: tc.text.primary, backgroundColor: tc.bg.secondary, borderColor: tc.border.subtle, fontFamily: fonts.body }]}
              value={worldName}
              onChangeText={setWorldName}
              placeholder="Enter world name..."
              placeholderTextColor={tc.text.muted}
            />
            <PillSelector
              label="Popular Worlds"
              options={POPULAR_WORLDS}
              value={worldName || undefined}
              onChange={(v) => { if (v) setWorldName(v); }}
              wrap
            />
          </View>
        );
      case 1:
        return (
          <View>
            <PillSelector label="Story Position" options={STORY_POSITIONS} value={storyPosition} onChange={setStoryPosition} allowDeselect={false} />
            <LabeledSlider label="Starting Level" value={startingLevel} min={1} max={100} onChange={setStartingLevel} />
            <PillSelector label="Difficulty" options={DIFFICULTIES} value={difficulty} onChange={setDifficulty} allowDeselect={false} />
          </View>
        );
      case 2:
        return (
          <View>
            <PillSelector label="Class" options={CLASSES} value={startingClass} onChange={setStartingClass} wrap allowDeselect={false} />
            <LabeledSlider label="STR" value={statSTR} min={1} max={20} onChange={setStatSTR} />
            <LabeledSlider label="DEX" value={statDEX} min={1} max={20} onChange={setStatDEX} />
            <LabeledSlider label="INT" value={statINT} min={1} max={20} onChange={setStatINT} />
            <MultiPillSelector label="Extras" options={EXTRAS} values={extras} onChange={setExtras} wrap />
          </View>
        );
      case 3:
        return (
          <View>
            <PillSelector label="Character Source" options={CHARACTER_SOURCES} value={characterSource} onChange={setCharacterSource} allowDeselect={false} />
            <PillSelector label="Name Style" options={NAME_STYLES} value={nameStyle} onChange={setNameStyle} allowDeselect={false} />
            <PillSelector label="Lore Depth" options={LORE_DEPTHS} value={loreDepth} onChange={setLoreDepth} allowDeselect={false} />
          </View>
        );
      case 4:
        return (
          <View>
            <View style={[styles.summaryCard, { backgroundColor: tc.bg.secondary, borderColor: tc.border.subtle }]}>
              <Text style={[styles.summaryTitle, { color: tc.accent.primary, fontFamily: fonts.heading }]}>Summary</Text>
              <SummaryRow label="World" value={worldName || '(none)'} tc={tc} />
              <SummaryRow label="Story Position" value={storyPosition || 'Beginning'} tc={tc} />
              <SummaryRow label="Level" value={String(startingLevel)} tc={tc} />
              <SummaryRow label="Difficulty" value={difficulty || 'Normal'} tc={tc} />
              <SummaryRow label="Class" value={startingClass || 'Warrior'} tc={tc} />
              <SummaryRow label="Stats" value={`STR ${statSTR} / DEX ${statDEX} / INT ${statINT}`} tc={tc} />
              <SummaryRow label="Extras" value={extras.length > 0 ? extras.join(', ') : 'None'} tc={tc} />
              <SummaryRow label="Characters" value={characterSource || 'Canon'} tc={tc} />
              <SummaryRow label="Names" value={nameStyle || 'Real'} tc={tc} />
              <SummaryRow label="Lore Depth" value={loreDepth || 'Standard'} tc={tc} />
            </View>
            <TouchableOpacity
              style={[styles.generateButton, { backgroundColor: tc.accent.primary, opacity: loading ? 0.6 : 1 }]}
              onPress={handleGenerate}
              disabled={loading}
              activeOpacity={0.7}
            >
              <Text style={[styles.generateText, { color: tc.text.inverse, fontFamily: fonts.button }]}>
                {loading ? 'Generating...' : 'Generate World'}
              </Text>
            </TouchableOpacity>
            {generatedNpcs.length > 0 && (
              <GeneratedCharacters
                characters={generatedNpcs}
                worldScenario={worldScenario}
                onStartChat={handleStartChat}
                onImport={handleImport}
              />
            )}
          </View>
        );
      default:
        return null;
    }
  };

  const renderAdvancedMode = () => (
    <>
      <WizardStepBar steps={STEPS} currentStep={step} />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {renderStep()}
      </ScrollView>
      <View style={[styles.navRow, { borderTopColor: tc.border.subtle }]}>
        {step > 0 ? (
          <TouchableOpacity
            style={[styles.navButton, { backgroundColor: tc.bg.tertiary }]}
            onPress={() => setStep(s => s - 1)}
            activeOpacity={0.7}
          >
            <Text style={[styles.navButtonText, { color: tc.text.primary, fontFamily: fonts.button }]}>Back</Text>
          </TouchableOpacity>
        ) : (
          <View style={styles.navButton} />
        )}
        {step < STEPS.length - 1 && (
          <TouchableOpacity
            style={[styles.navButton, { backgroundColor: tc.accent.primary }]}
            onPress={() => setStep(s => s + 1)}
            activeOpacity={0.7}
          >
            <Text style={[styles.navButtonText, { color: tc.text.inverse, fontFamily: fonts.button }]}>Next</Text>
          </TouchableOpacity>
        )}
      </View>
    </>
  );

  return (
    <View style={styles.container}>
      {/* Mode toggle */}
      <View style={[styles.toggleRow, { borderBottomColor: tc.border.subtle }]}>
        <TouchableOpacity
          style={[styles.toggleButton, !advancedMode && { borderBottomColor: tc.accent.primary, borderBottomWidth: 2 }]}
          onPress={() => { setAdvancedMode(false); setStep(0); }}
          activeOpacity={0.7}
        >
          <Text style={[styles.toggleText, { color: !advancedMode ? tc.accent.primary : tc.text.muted, fontFamily: fonts.button }]}>
            Quick
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.toggleButton, advancedMode && { borderBottomColor: tc.accent.primary, borderBottomWidth: 2 }]}
          onPress={() => setAdvancedMode(true)}
          activeOpacity={0.7}
        >
          <Text style={[styles.toggleText, { color: advancedMode ? tc.accent.primary : tc.text.muted, fontFamily: fonts.button }]}>
            Advanced
          </Text>
        </TouchableOpacity>
      </View>

      {advancedMode ? renderAdvancedMode() : renderQuickMode()}
    </View>
  );
}

// ── Summary row helper ──

function SummaryRow({ label, value, tc }: { label: string; value: string; tc: ThemeColors }) {
  return (
    <View style={styles.summaryRow}>
      <Text style={[styles.summaryLabel, { color: tc.text.muted, fontFamily: fonts.body }]}>{label}</Text>
      <Text style={[styles.summaryValue, { color: tc.text.primary, fontFamily: fonts.bodyMedium }]}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  toggleRow: { flexDirection: 'row', borderBottomWidth: 1 },
  toggleButton: { flex: 1, alignItems: 'center', paddingVertical: 12 },
  toggleText: { fontSize: 14 },
  scrollContent: { padding: spacing.lg, gap: spacing.md, paddingBottom: 100 },
  sectionLabel: { fontSize: 13, textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: spacing.xs },
  input: {
    borderWidth: 1,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    fontSize: 15,
    marginBottom: spacing.md,
  },
  generateButton: {
    paddingVertical: 14,
    borderRadius: borderRadius.md,
    alignItems: 'center',
    marginTop: spacing.md,
  },
  generateText: { fontSize: 16 },
  navRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    padding: spacing.lg,
    borderTopWidth: 1,
    gap: spacing.md,
  },
  navButton: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: borderRadius.md,
    alignItems: 'center',
  },
  navButtonText: { fontSize: 14 },
  summaryCard: {
    borderWidth: 1,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    gap: spacing.sm,
  },
  summaryTitle: { fontSize: 18, marginBottom: spacing.sm },
  summaryRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  summaryLabel: { fontSize: 13 },
  summaryValue: { fontSize: 13 },
});
