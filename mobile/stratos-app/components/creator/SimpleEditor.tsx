import React from 'react';
import { View, Text, TextInput, StyleSheet } from 'react-native';
import { CharacterCardCreate } from '../../lib/types';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { useThemeStore } from '../../stores/themeStore';
import {
  PillSelector,
  GENDER_OPTIONS, ARCHETYPE_OPTIONS, RELATIONSHIP_OPTIONS,
  SCENARIO_TEMPLATES, PERSONALITY_TEMPLATES,
} from './PillSelector';

function WordCount({ text }: { text: string }) {
  const tc = useThemeStore(s => s.colors);
  if (!text.trim()) return null;
  const words = text.trim().split(/\s+/).length;
  return <Text style={{ fontSize: 9, color: tc.text.muted, textAlign: 'right', marginTop: 2 }}>{words} words · {text.length} chars</Text>;
}

interface SimpleEditorProps {
  card: CharacterCardCreate;
  onUpdate: (key: keyof CharacterCardCreate, value: string | string[] | undefined) => void;
}

export const SimpleEditor = React.memo(function SimpleEditor({ card, onUpdate }: SimpleEditorProps) {
  const tc = useThemeStore(s => s.colors);

  const handleArchetypeChange = (arch: string | undefined) => {
    onUpdate('archetype_override', arch);
    // Auto-populate personality template if personality is empty or was a previous template
    if (arch && (!card.personality?.trim() || Object.values(PERSONALITY_TEMPLATES).includes(card.personality))) {
      onUpdate('personality', PERSONALITY_TEMPLATES[arch] || '');
    }
  };

  const handleScenarioTemplate = (template: string | undefined) => {
    if (template) {
      onUpdate('scenario', template);
    }
  };

  return (
    <>
      {/* P0 Pills — Gender + Archetype */}
      <PillSelector
        label="Gender"
        options={GENDER_OPTIONS}
        value={card.gender}
        onChange={(v) => onUpdate('gender', v)}
      />

      <PillSelector
        label="Archetype"
        options={ARCHETYPE_OPTIONS}
        value={card.archetype_override}
        onChange={handleArchetypeChange}
      />

      <PillSelector
        label="Relationship to You"
        options={RELATIONSHIP_OPTIONS}
        value={card.relationship_to_user}
        onChange={(v) => onUpdate('relationship_to_user', v)}
      />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Description</Text>
      <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.description} onChangeText={v => onUpdate('description', v)} placeholder="Who is this character?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" accessibilityLabel="Character description" />
      <WordCount text={card.description} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Personality</Text>
      <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.personality} onChangeText={v => onUpdate('personality', v)} placeholder="How does your character act and speak?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" accessibilityLabel="Character personality" />
      <WordCount text={card.personality} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>First Message</Text>
      <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>Sets the tone for every conversation.</Text>
      <TextInput style={[styles.input, styles.multilineLg, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.first_message} onChangeText={v => onUpdate('first_message', v)} placeholder="What does your character say or do first?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" accessibilityLabel="First message" />
      <WordCount text={card.first_message} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Scenario</Text>
      <PillSelector
        label=""
        options={SCENARIO_TEMPLATES}
        value={undefined}
        onChange={handleScenarioTemplate}
        allowDeselect={false}
      />
      <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.scenario} onChangeText={v => onUpdate('scenario', v)} placeholder="The starting situation" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" accessibilityLabel="Scenario" />
      <WordCount text={card.scenario} />
    </>
  );
});

const styles = StyleSheet.create({
  fieldLabel: { ...typography.subheading, marginBottom: spacing.xs, marginTop: spacing.lg },
  fieldHint: { ...typography.caption, marginBottom: spacing.sm },
  input: { borderRadius: borderRadius.md, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, fontSize: 15, borderWidth: 1 },
  multiline: { minHeight: 80, textAlignVertical: 'top' },
  multilineLg: { minHeight: 140, textAlignVertical: 'top' },
});
