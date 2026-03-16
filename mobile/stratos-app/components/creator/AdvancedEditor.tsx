import React from 'react';
import { View, Text, TextInput, StyleSheet } from 'react-native';
import { CharacterCardCreate } from '../../lib/types';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { useThemeStore } from '../../stores/themeStore';

function WordCount({ text }: { text: string }) {
  const tc = useThemeStore(s => s.colors);
  if (!text.trim()) return null;
  const words = text.trim().split(/\s+/).length;
  return <Text style={{ fontSize: 9, color: tc.text.muted, textAlign: 'right', marginTop: 2 }}>{words} words · {text.length} chars</Text>;
}

interface AdvancedEditorProps {
  card: CharacterCardCreate;
  onUpdate: (key: keyof CharacterCardCreate, value: string | string[]) => void;
}

export const AdvancedEditor = React.memo(function AdvancedEditor({ card, onUpdate }: AdvancedEditorProps) {
  const tc = useThemeStore(s => s.colors);

  return (
    <>
      {/* Section: Identity */}
      <Text style={[styles.sectionTitle, { color: tc.accent.primary }]}>Identity</Text>
      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Description</Text>
      <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.description} onChangeText={v => onUpdate('description', v)} placeholder="Who is this character?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.description} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Physical Description</Text>
      <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>What do they look like? Include one unique detail.</Text>
      <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.physical_description} onChangeText={v => onUpdate('physical_description', v)} placeholder="Tall, clouded left eye, worn leather armor..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.physical_description} />

      {/* Section: Behavior */}
      <View style={[styles.divider, { backgroundColor: tc.border.subtle }]} />
      <Text style={[styles.sectionTitle, { color: tc.accent.primary }]}>Behavior</Text>

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Personality</Text>
      <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.personality} onChangeText={v => onUpdate('personality', v)} placeholder="How does your character act and speak?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.personality} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Speech Pattern</Text>
      <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>How do they talk? Accent, vocabulary, quirks.</Text>
      <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.speech_pattern} onChangeText={v => onUpdate('speech_pattern', v)} placeholder="Formal, archaic phrasing, never uses contractions..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.speech_pattern} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>First Message</Text>
      <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>Sets the tone for every conversation — the most important field.</Text>
      <TextInput style={[styles.input, styles.multilineLg, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.first_message} onChangeText={v => onUpdate('first_message', v)} placeholder="What does your character say or do first?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.first_message} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Scenario</Text>
      <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.scenario} onChangeText={v => onUpdate('scenario', v)} placeholder="The starting situation" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.scenario} />

      {/* Section: Depth */}
      <View style={[styles.divider, { backgroundColor: tc.border.subtle }]} />
      <Text style={[styles.sectionTitle, { color: tc.accent.primary }]}>Depth</Text>

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Emotional Trigger</Text>
      <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>What sets them off emotionally?</Text>
      <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.emotional_trigger} onChangeText={v => onUpdate('emotional_trigger', v)} placeholder="Any mention of cowardice sends them into a rage..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.emotional_trigger} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Defensive Mechanism</Text>
      <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>How do they protect themselves?</Text>
      <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.defensive_mechanism} onChangeText={v => onUpdate('defensive_mechanism', v)} placeholder="Deflects with cold formality, changes the subject..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.defensive_mechanism} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Vulnerability</Text>
      <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>The crack in the armor. What breaks through?</Text>
      <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.vulnerability} onChangeText={v => onUpdate('vulnerability', v)} placeholder="Children playing reminds them of what they lost..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.vulnerability} />

      <Text style={[styles.fieldLabel, { color: tc.text.primary }]}>Specific Detail</Text>
      <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>One concrete, grounding detail that makes them real.</Text>
      <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.specific_detail} onChangeText={v => onUpdate('specific_detail', v)} placeholder="Traces the sigil on their ring when anxious..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
      <WordCount text={card.specific_detail} />
    </>
  );
});

const styles = StyleSheet.create({
  fieldLabel: { ...typography.subheading, marginBottom: spacing.xs, marginTop: spacing.lg },
  fieldHint: { ...typography.caption, marginBottom: spacing.sm },
  input: { borderRadius: borderRadius.md, paddingHorizontal: spacing.lg, paddingVertical: spacing.md, fontSize: 15, borderWidth: 1 },
  multiline: { minHeight: 80, textAlignVertical: 'top' },
  multilineLg: { minHeight: 140, textAlignVertical: 'top' },
  divider: { height: 1, marginVertical: spacing.lg },
  sectionTitle: { ...typography.subheading, fontSize: 13, textTransform: 'uppercase', letterSpacing: 1, marginTop: spacing.lg, marginBottom: spacing.xs },
});
