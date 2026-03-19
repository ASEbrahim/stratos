import React, { useState } from 'react';
import { View, Text, TextInput, StyleSheet } from 'react-native';
import {
  User, Clock, Crown, Heart, Sparkles, ScrollText, Eye, Brain,
  MessageCircle, Feather, Globe, BookOpen, AlignLeft, Shield,
  Flame, ShieldAlert, HeartCrack, Zap,
} from 'lucide-react-native';
import { CharacterCardCreate } from '../../lib/types';
import { typography, spacing, borderRadius } from '../../constants/theme';
import { useThemeStore } from '../../stores/themeStore';
import { CategoryCard, CategoryPopup, SectionHeader } from './CategoryCard';
import {
  PillSelector, MultiPillSelector,
  GENDER_OPTIONS, ARCHETYPE_OPTIONS, RELATIONSHIP_OPTIONS,
  POV_OPTIONS, NSFW_COMFORT_OPTIONS, AGE_RANGE_OPTIONS,
  RESPONSE_LENGTH_OPTIONS, SCENARIO_TEMPLATES, SPEECH_STYLE_TEMPLATES,
  PERSONALITY_TEMPLATES, PERSONALITY_TAG_OPTIONS,
} from './PillSelector';

function WordCount({ text }: { text: string }) {
  const tc = useThemeStore(s => s.colors);
  if (!text.trim()) return null;
  const words = text.trim().split(/\s+/).length;
  return <Text style={{ fontSize: 9, color: tc.text.muted, textAlign: 'right', marginTop: 2 }}>{words} words · {text.length} chars</Text>;
}

// Preview helpers
function pillPreview(value: string | undefined, options: { value: string; label: string }[]): string {
  if (!value) return 'Not set';
  const match = options.find(o => o.value === value);
  return match ? match.label : 'Not set';
}

function textPreview(value: string | undefined): string {
  if (!value?.trim()) return 'Empty';
  return value.length > 30 ? value.slice(0, 30) + '...' : value;
}

interface AdvancedEditorProps {
  card: CharacterCardCreate;
  onUpdate: (key: keyof CharacterCardCreate, value: string | string[] | undefined) => void;
}

export const AdvancedEditor = React.memo(function AdvancedEditor({ card, onUpdate }: AdvancedEditorProps) {
  const tc = useThemeStore(s => s.colors);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const toggle = (id: string) => setExpandedId(prev => prev === id ? null : id);

  const handleArchetypeChange = (arch: string | undefined) => {
    onUpdate('archetype_override', arch);
    if (arch && (!card.personality?.trim() || Object.values(PERSONALITY_TEMPLATES).includes(card.personality))) {
      onUpdate('personality', PERSONALITY_TEMPLATES[arch] || '');
    }
  };

  // Progress counts
  const identityProgress = [card.gender, card.age_range, card.description, card.physical_description].filter(Boolean).length;
  const behaviorProgress = [card.archetype_override, card.relationship_to_user, card.personality_tags?.length, card.personality, card.speech_pattern].filter(Boolean).length;
  const worldProgress = [card.first_message, card.scenario].filter(Boolean).length;
  const styleProgress = [card.narration_pov, card.response_length_pref, card.nsfw_comfort].filter(Boolean).length;
  const depthProgress = [card.emotional_trigger, card.defensive_mechanism, card.vulnerability, card.specific_detail].filter(Boolean).length;

  return (
    <>
      {/* ═══ Identity ═══ */}
      <SectionHeader title="Identity" progress={`${identityProgress}/4`} />

      <CategoryPopup
        icon={User}
        iconColor={tc.accent.primary}
        title="Gender"
        preview={pillPreview(card.gender, GENDER_OPTIONS)}
        isComplete={!!card.gender}
        index={0}
      >
        <PillSelector label="" options={GENDER_OPTIONS} value={card.gender} onChange={(v) => onUpdate('gender', v)} />
      </CategoryPopup>

      <CategoryPopup
        icon={Clock}
        iconColor={tc.accent.primary}
        title="Age Range"
        preview={pillPreview(card.age_range, AGE_RANGE_OPTIONS)}
        isComplete={!!card.age_range}
        index={1}
      >
        <PillSelector label="" options={AGE_RANGE_OPTIONS} value={card.age_range} onChange={(v) => onUpdate('age_range', v)} />
      </CategoryPopup>

      <CategoryCard
        icon={ScrollText}
        iconColor={tc.accent.fantasy}
        title="Description"
        preview={textPreview(card.description)}
        isComplete={!!card.description?.trim()}
        isExpanded={expandedId === 'description'}
        onToggle={() => toggle('description')}
        index={2}
      >
        <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.description} onChangeText={v => onUpdate('description', v)} placeholder="Who is this character?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
        <WordCount text={card.description} />
      </CategoryCard>

      <CategoryCard
        icon={Eye}
        iconColor={tc.accent.scifi}
        title="Physical Description"
        preview={textPreview(card.physical_description)}
        isComplete={!!card.physical_description?.trim()}
        isExpanded={expandedId === 'physical_description'}
        onToggle={() => toggle('physical_description')}
        index={3}
      >
        <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>What do they look like? Include one unique detail.</Text>
        <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.physical_description} onChangeText={v => onUpdate('physical_description', v)} placeholder="Tall, clouded left eye, worn leather armor..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
        <WordCount text={card.physical_description} />
      </CategoryCard>

      {/* ═══ Behavior ═══ */}
      <SectionHeader title="Behavior" progress={`${behaviorProgress}/5`} />

      <CategoryPopup
        icon={Crown}
        iconColor={tc.accent.secondary}
        title="Archetype"
        preview={pillPreview(card.archetype_override, ARCHETYPE_OPTIONS)}
        isComplete={!!card.archetype_override}
        index={0}
      >
        <PillSelector label="" options={ARCHETYPE_OPTIONS} value={card.archetype_override} onChange={handleArchetypeChange} />
      </CategoryPopup>

      <CategoryPopup
        icon={Heart}
        iconColor={tc.accent.romance}
        title="Relationship"
        preview={pillPreview(card.relationship_to_user, RELATIONSHIP_OPTIONS)}
        isComplete={!!card.relationship_to_user}
        index={1}
      >
        <PillSelector label="" options={RELATIONSHIP_OPTIONS} value={card.relationship_to_user} onChange={(v) => onUpdate('relationship_to_user', v)} />
      </CategoryPopup>

      <CategoryPopup
        icon={Sparkles}
        iconColor={tc.accent.anime}
        title="Personality Tags"
        preview={card.personality_tags?.length ? `${card.personality_tags.length} selected` : 'Not set'}
        isComplete={!!card.personality_tags?.length}
        index={2}
      >
        <MultiPillSelector
          label=""
          options={PERSONALITY_TAG_OPTIONS}
          values={card.personality_tags || []}
          onChange={(v) => onUpdate('personality_tags' as keyof CharacterCardCreate, v as unknown as string)}
          wrap
        />
      </CategoryPopup>

      <CategoryCard
        icon={Brain}
        iconColor={tc.accent.modern}
        title="Personality"
        preview={textPreview(card.personality)}
        isComplete={!!card.personality?.trim()}
        isExpanded={expandedId === 'personality'}
        onToggle={() => toggle('personality')}
        index={3}
      >
        <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.personality} onChangeText={v => onUpdate('personality', v)} placeholder="How does your character act and speak?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
        <WordCount text={card.personality} />
      </CategoryCard>

      <CategoryCard
        icon={MessageCircle}
        iconColor={tc.accent.historical}
        title="Speech Pattern"
        preview={textPreview(card.speech_pattern)}
        isComplete={!!card.speech_pattern?.trim()}
        isExpanded={expandedId === 'speech_pattern'}
        onToggle={() => toggle('speech_pattern')}
        index={4}
      >
        <PillSelector label="" options={SPEECH_STYLE_TEMPLATES} value={undefined} onChange={(v) => { if (v) onUpdate('speech_pattern', v); }} allowDeselect={false} />
        <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.speech_pattern} onChangeText={v => onUpdate('speech_pattern', v)} placeholder="Formal, archaic phrasing, never uses contractions..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
        <WordCount text={card.speech_pattern} />
      </CategoryCard>

      {/* ═══ World ═══ */}
      <SectionHeader title="World" progress={`${worldProgress}/2`} />

      <CategoryCard
        icon={Feather}
        iconColor={tc.accent.fantasy}
        title="First Message"
        preview={textPreview(card.first_message)}
        isComplete={!!card.first_message?.trim()}
        isExpanded={expandedId === 'first_message'}
        onToggle={() => toggle('first_message')}
        index={0}
      >
        <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>Sets the tone for every conversation — the most important field.</Text>
        <TextInput style={[styles.input, styles.multilineLg, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.first_message} onChangeText={v => onUpdate('first_message', v)} placeholder="What does your character say or do first?" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
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
        index={1}
      >
        <PillSelector label="" options={SCENARIO_TEMPLATES} value={undefined} onChange={(v) => { if (v) onUpdate('scenario', v); }} allowDeselect={false} />
        <TextInput style={[styles.input, styles.multiline, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.scenario} onChangeText={v => onUpdate('scenario', v)} placeholder="The starting situation" placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
        <WordCount text={card.scenario} />
      </CategoryCard>

      {/* ═══ Style ═══ */}
      <SectionHeader title="Style" progress={`${styleProgress}/3`} />

      <CategoryPopup
        icon={BookOpen}
        iconColor={tc.accent.modern}
        title="Narration POV"
        preview={pillPreview(card.narration_pov, POV_OPTIONS)}
        isComplete={!!card.narration_pov}
        index={0}
      >
        <PillSelector label="" options={POV_OPTIONS} value={card.narration_pov} onChange={(v) => onUpdate('narration_pov', v)} />
      </CategoryPopup>

      <CategoryPopup
        icon={AlignLeft}
        iconColor={tc.accent.primary}
        title="Response Length"
        preview={pillPreview(card.response_length_pref, RESPONSE_LENGTH_OPTIONS)}
        isComplete={!!card.response_length_pref}
        index={1}
      >
        <PillSelector label="" options={RESPONSE_LENGTH_OPTIONS} value={card.response_length_pref} onChange={(v) => onUpdate('response_length_pref', v)} />
      </CategoryPopup>

      <CategoryPopup
        icon={Shield}
        iconColor={tc.nsfw}
        title="NSFW Comfort"
        preview={pillPreview(card.nsfw_comfort, NSFW_COMFORT_OPTIONS)}
        isComplete={!!card.nsfw_comfort}
        index={2}
      >
        <PillSelector label="" options={NSFW_COMFORT_OPTIONS} value={card.nsfw_comfort} onChange={(v) => onUpdate('nsfw_comfort', v)} />
      </CategoryPopup>

      {/* ═══ Depth ═══ */}
      <SectionHeader title="Depth" progress={`${depthProgress}/4`} />

      <CategoryCard
        icon={Flame}
        iconColor={tc.status.error}
        title="Emotional Trigger"
        preview={textPreview(card.emotional_trigger)}
        isComplete={!!card.emotional_trigger?.trim()}
        isExpanded={expandedId === 'emotional_trigger'}
        onToggle={() => toggle('emotional_trigger')}
        index={0}
      >
        <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>What sets them off emotionally?</Text>
        <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.emotional_trigger} onChangeText={v => onUpdate('emotional_trigger', v)} placeholder="Any mention of cowardice sends them into a rage..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
        <WordCount text={card.emotional_trigger} />
      </CategoryCard>

      <CategoryCard
        icon={ShieldAlert}
        iconColor={tc.accent.scifi}
        title="Defensive Mechanism"
        preview={textPreview(card.defensive_mechanism)}
        isComplete={!!card.defensive_mechanism?.trim()}
        isExpanded={expandedId === 'defensive_mechanism'}
        onToggle={() => toggle('defensive_mechanism')}
        index={1}
      >
        <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>How do they protect themselves?</Text>
        <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.defensive_mechanism} onChangeText={v => onUpdate('defensive_mechanism', v)} placeholder="Deflects with cold formality, changes the subject..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
        <WordCount text={card.defensive_mechanism} />
      </CategoryCard>

      <CategoryCard
        icon={HeartCrack}
        iconColor={tc.accent.romance}
        title="Vulnerability"
        preview={textPreview(card.vulnerability)}
        isComplete={!!card.vulnerability?.trim()}
        isExpanded={expandedId === 'vulnerability'}
        onToggle={() => toggle('vulnerability')}
        index={2}
      >
        <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>The crack in the armor. What breaks through?</Text>
        <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.vulnerability} onChangeText={v => onUpdate('vulnerability', v)} placeholder="Children playing reminds them of what they lost..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
        <WordCount text={card.vulnerability} />
      </CategoryCard>

      <CategoryCard
        icon={Zap}
        iconColor={tc.accent.secondary}
        title="Specific Detail"
        preview={textPreview(card.specific_detail)}
        isComplete={!!card.specific_detail?.trim()}
        isExpanded={expandedId === 'specific_detail'}
        onToggle={() => toggle('specific_detail')}
        index={3}
      >
        <Text style={[styles.fieldHint, { color: tc.text.secondary }]}>One concrete, grounding detail that makes them real.</Text>
        <TextInput style={[styles.input, { backgroundColor: tc.bg.tertiary, color: tc.text.primary, borderColor: tc.border.subtle }]} value={card.specific_detail} onChangeText={v => onUpdate('specific_detail', v)} placeholder="Traces the sigil on their ring when anxious..." placeholderTextColor={tc.text.muted} multiline textAlignVertical="top" />
        <WordCount text={card.specific_detail} />
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
