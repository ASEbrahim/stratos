import React, { useState } from 'react';
import { View, Text, TextInput, StyleSheet } from 'react-native';
import { User, Crown, Heart, Sparkles, ScrollText, Brain, Feather, Globe } from 'lucide-react-native';
import { CharacterCardCreate } from '../../lib/types';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { useThemeStore } from '../../stores/themeStore';
import { CategoryCard, SectionHeader } from './CategoryCard';
import {
  PillSelector, MultiPillSelector,
  GENDER_OPTIONS, ARCHETYPE_OPTIONS, RELATIONSHIP_OPTIONS,
  SCENARIO_TEMPLATES, PERSONALITY_TEMPLATES, PERSONALITY_TAG_OPTIONS,
  PillOption,
} from './PillSelector';

function WordCount({ text }: { text: string }) {
  const tc = useThemeStore(s => s.colors);
  if (!text.trim()) return null;
  const words = text.trim().split(/\s+/).length;
  return <Text style={{ fontSize: 9, color: tc.text.muted, textAlign: 'right', marginTop: 2 }}>{words} words · {text.length} chars</Text>;
}

/** Find the label for a pill value, or return fallback */
function pillPreview(options: PillOption[], value: string | undefined): string {
  if (!value) return 'Not set';
  const match = options.find(o => o.value === value);
  return match ? match.label : 'Not set';
}

/** Preview for a text field: first 30 chars + ellipsis, or 'Empty' */
function textPreview(text: string | undefined): string {
  if (!text?.trim()) return 'Empty';
  const trimmed = text.trim();
  return trimmed.length > 30 ? trimmed.slice(0, 30) + '...' : trimmed;
}

/** Preview for multi-select tags */
function tagsPreview(tags: string[] | undefined, options: PillOption[]): string {
  if (!tags?.length) return 'Not set';
  return tags
    .map(v => options.find(o => o.value === v)?.label ?? v)
    .join(', ');
}

interface SimpleEditorProps {
  card: CharacterCardCreate;
  onUpdate: (key: keyof CharacterCardCreate, value: string | string[] | undefined) => void;
}

export const SimpleEditor = React.memo(function SimpleEditor({ card, onUpdate }: SimpleEditorProps) {
  const tc = useThemeStore(s => s.colors);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const toggle = (id: string) => setExpandedId(prev => prev === id ? null : id);

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

  // Progress counters
  const identityDone = [card.gender, card.archetype_override].filter(Boolean).length;
  const dynamicsDone = [card.relationship_to_user, card.personality_tags?.length].filter(Boolean).length;
  const storyDone = [card.description, card.personality, card.first_message, card.scenario]
    .filter(v => typeof v === 'string' && v.trim().length > 0).length;

  return (
    <>
      {/* ── Identity ── */}
      <SectionHeader title="Identity" progress={`${identityDone}/2`} />

      <CategoryCard
        icon={User}
        iconColor={tc.accent.primary}
        title="Gender"
        preview={pillPreview(GENDER_OPTIONS, card.gender)}
        isComplete={!!card.gender}
        isExpanded={expandedId === 'gender'}
        onToggle={() => toggle('gender')}
        index={0}
      >
        <PillSelector
          label=""
          options={GENDER_OPTIONS}
          value={card.gender}
          onChange={(v) => onUpdate('gender', v)}
        />
      </CategoryCard>

      <CategoryCard
        icon={Crown}
        iconColor={tc.accent.secondary}
        title="Archetype"
        preview={pillPreview(ARCHETYPE_OPTIONS, card.archetype_override)}
        isComplete={!!card.archetype_override}
        isExpanded={expandedId === 'archetype'}
        onToggle={() => toggle('archetype')}
        index={1}
      >
        <PillSelector
          label=""
          options={ARCHETYPE_OPTIONS}
          value={card.archetype_override}
          onChange={handleArchetypeChange}
        />
      </CategoryCard>

      {/* ── Dynamics ── */}
      <SectionHeader title="Dynamics" progress={`${dynamicsDone}/2`} />

      <CategoryCard
        icon={Heart}
        iconColor={tc.accent.romance}
        title="Relationship"
        preview={pillPreview(RELATIONSHIP_OPTIONS, card.relationship_to_user)}
        isComplete={!!card.relationship_to_user}
        isExpanded={expandedId === 'relationship'}
        onToggle={() => toggle('relationship')}
        index={2}
      >
        <PillSelector
          label=""
          options={RELATIONSHIP_OPTIONS}
          value={card.relationship_to_user}
          onChange={(v) => onUpdate('relationship_to_user', v)}
        />
      </CategoryCard>

      <CategoryCard
        icon={Sparkles}
        iconColor={tc.accent.anime}
        title="Personality Tags"
        preview={tagsPreview(card.personality_tags, PERSONALITY_TAG_OPTIONS)}
        isComplete={!!card.personality_tags?.length}
        isExpanded={expandedId === 'tags'}
        onToggle={() => toggle('tags')}
        index={3}
      >
        <MultiPillSelector
          label=""
          options={PERSONALITY_TAG_OPTIONS}
          values={card.personality_tags || []}
          onChange={(v) => onUpdate('personality_tags' as keyof CharacterCardCreate, v as unknown as string)}
        />
      </CategoryCard>

      {/* ── Story ── */}
      <SectionHeader title="Story" progress={`${storyDone}/4`} />

      <CategoryCard
        icon={ScrollText}
        iconColor={tc.accent.fantasy}
        title="Description"
        preview={textPreview(card.description)}
        isComplete={!!card.description?.trim()}
        isExpanded={expandedId === 'description'}
        onToggle={() => toggle('description')}
        index={4}
      >
        <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.description} onChangeText={v => onUpdate('description', v)} placeholder="Who is this character?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" accessibilityLabel="Character description" />
        <WordCount text={card.description} />
      </CategoryCard>

      <CategoryCard
        icon={Brain}
        iconColor={tc.accent.modern}
        title="Personality"
        preview={textPreview(card.personality)}
        isComplete={!!card.personality?.trim()}
        isExpanded={expandedId === 'personality'}
        onToggle={() => toggle('personality')}
        index={5}
      >
        <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.personality} onChangeText={v => onUpdate('personality', v)} placeholder="How does your character act and speak?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" accessibilityLabel="Character personality" />
        <WordCount text={card.personality} />
      </CategoryCard>

      <CategoryCard
        icon={Feather}
        iconColor={tc.accent.fantasy}
        title="First Message"
        preview={textPreview(card.first_message)}
        isComplete={!!card.first_message?.trim()}
        isExpanded={expandedId === 'first_message'}
        onToggle={() => toggle('first_message')}
        index={6}
      >
        <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>Sets the tone for every conversation.</Text>
        <TextInput style={[styles.input, styles.multilineLg, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.first_message} onChangeText={v => onUpdate('first_message', v)} placeholder="What does your character say or do first?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" accessibilityLabel="First message" />
        <WordCount text={card.first_message} />
      </CategoryCard>

      <CategoryCard
        icon={Globe}
        iconColor={tc.accent.horror}
        title="Scenario"
        preview={textPreview(card.scenario)}
        isComplete={!!card.scenario?.trim()}
        isExpanded={expandedId === 'scenario'}
        onToggle={() => toggle('scenario')}
        index={7}
      >
        <PillSelector
          label=""
          options={SCENARIO_TEMPLATES}
          value={undefined}
          onChange={handleScenarioTemplate}
          allowDeselect={false}
        />
        <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.scenario} onChangeText={v => onUpdate('scenario', v)} placeholder="The starting situation" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" accessibilityLabel="Scenario" />
        <WordCount text={card.scenario} />
      </CategoryCard>
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
