import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet } from 'react-native';
import { useThemeStore } from '../../stores/themeStore';

export interface PillOption<T extends string = string> {
  value: T;
  label: string;
  description?: string; // shown as subtitle or tooltip
}

interface PillSelectorProps<T extends string = string> {
  label: string;
  options: PillOption<T>[];
  value: T | undefined;
  onChange: (value: T | undefined) => void;
  allowDeselect?: boolean; // tap selected pill to deselect (default true)
}

export function PillSelector<T extends string = string>({
  label, options, value, onChange, allowDeselect = true,
}: PillSelectorProps<T>) {
  const tc = useThemeStore(s => s.colors);

  return (
    <View style={styles.container}>
      <Text style={[styles.label, { color: tc.text.secondary }]}>{label}</Text>
      <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.pills}>
        {options.map(opt => {
          const selected = value === opt.value;
          return (
            <TouchableOpacity
              key={opt.value}
              style={[
                styles.pill,
                { borderColor: tc.border.subtle, backgroundColor: tc.bg.secondary },
                selected && { borderColor: tc.accent.primary, backgroundColor: tc.accent.primary + '18' },
              ]}
              onPress={() => {
                if (selected && allowDeselect) {
                  onChange(undefined);
                } else {
                  onChange(opt.value);
                }
              }}
              activeOpacity={0.7}
              accessibilityRole="button"
              accessibilityState={{ selected }}
              accessibilityLabel={`${opt.label}${opt.description ? ': ' + opt.description : ''}`}
            >
              <Text style={[
                styles.pillText,
                { color: selected ? tc.accent.primary : tc.text.primary },
              ]}>
                {opt.label}
              </Text>
            </TouchableOpacity>
          );
        })}
      </ScrollView>
    </View>
  );
}

// ── Pill option presets ──

export const GENDER_OPTIONS: PillOption[] = [
  { value: 'female', label: 'Female' },
  { value: 'male', label: 'Male' },
  { value: 'nonbinary', label: 'Non-binary' },
];

export const ARCHETYPE_OPTIONS: PillOption[] = [
  { value: 'shy', label: 'Shy / Reserved' },
  { value: 'confident', label: 'Confident / Bold' },
  { value: 'tough', label: 'Tough / Stoic' },
  { value: 'clinical', label: 'Analytical' },
  { value: 'sweet', label: 'Sweet / Caring' },
  { value: 'submissive', label: 'Submissive' },
  { value: 'dominant', label: 'Dominant' },
  { value: 'playful', label: 'Playful / Teasing' },
  { value: 'mysterious', label: 'Mysterious' },
  { value: 'protective', label: 'Protective' },
];

export const POV_OPTIONS: PillOption[] = [
  { value: 'third', label: '3rd Person' },
  { value: 'first', label: '1st Person' },
  { value: 'mixed', label: 'Mixed' },
];

export const RELATIONSHIP_OPTIONS: PillOption[] = [
  { value: 'stranger', label: 'Stranger' },
  { value: 'friend', label: 'Friend' },
  { value: 'rival', label: 'Rival' },
  { value: 'love_interest', label: 'Love Interest' },
  { value: 'mentor', label: 'Mentor' },
  { value: 'servant', label: 'Servant' },
];

export const NSFW_COMFORT_OPTIONS: PillOption[] = [
  { value: 'fade', label: 'Fade to Black' },
  { value: 'suggestive', label: 'Suggestive' },
  { value: 'explicit', label: 'Explicit' },
];

export const AGE_RANGE_OPTIONS: PillOption[] = [
  { value: 'teen', label: 'Teen' },
  { value: 'young_adult', label: 'Young Adult' },
  { value: 'adult', label: 'Adult' },
  { value: 'middle_aged', label: 'Middle Aged' },
  { value: 'elderly', label: 'Elderly' },
];

export const RESPONSE_LENGTH_OPTIONS: PillOption[] = [
  { value: 'brief', label: 'Brief' },
  { value: 'normal', label: 'Normal' },
  { value: 'detailed', label: 'Detailed' },
];

export const SCENARIO_TEMPLATES: PillOption[] = [
  { value: 'A quiet coffee shop on a rainy afternoon. The espresso machine hisses softly. A few regulars sit in the corners.', label: 'Coffee Shop' },
  { value: 'A university campus between classes. Students pass through the courtyard. The library is visible across the quad.', label: 'School' },
  { value: 'A dimly lit tavern in a medieval village. A fire crackles in the hearth. The barkeep polishes mugs behind the counter.', label: 'Fantasy Tavern' },
  { value: 'A cramped space station corridor. The hum of life support fills the silence. Stars drift past the viewport.', label: 'Sci-Fi Station' },
  { value: 'A quiet hospital ward during night shift. Monitors beep softly. The fluorescent lights buzz overhead.', label: 'Hospital' },
  { value: 'A modern office on a high floor. City lights glitter through floor-to-ceiling windows. Papers stack on the desk.', label: 'Office' },
  { value: 'A dimly lit nightclub. Bass thumps through the floor. Neon light cuts through cigarette haze.', label: 'Nightclub' },
  { value: 'A secluded forest clearing at dusk. Fireflies drift between the trees. A stream murmurs nearby.', label: 'Nature' },
  { value: 'A small apartment in the evening. Soft lamplight. Takeout containers on the coffee table. A show plays quietly on the TV.', label: 'Apartment' },
  { value: 'A bombed-out building. Rain leaks through holes in the ceiling. Gunfire echoes in the distance.', label: 'Warzone' },
];

export const SPEECH_STYLE_TEMPLATES: PillOption[] = [
  { value: 'Speaks formally with proper grammar. Never uses contractions. Precise word choice.', label: 'Formal' },
  { value: 'Relaxed, conversational. Uses slang, filler words like "like" and "honestly". Trails off sometimes.', label: 'Casual' },
  { value: 'Few words. Short sentences. Never explains more than necessary. Gets to the point.', label: 'Terse' },
  { value: 'Elaborate, poetic language. Uses metaphors naturally. Speaks in long, flowing sentences.', label: 'Flowery' },
  { value: 'Stutters when anxious. Uses "um" and "uh". Starts sentences over. Gets quieter under pressure.', label: 'Nervous' },
  { value: 'Dry wit. Says the opposite of what they mean. Eye-rolls are audible in their voice.', label: 'Sarcastic' },
  { value: 'Uses technical jargon naturally. Speaks like giving a lecture. Corrects others\' grammar.', label: 'Academic' },
  { value: 'Slang-heavy. Drops letters. Code-switches based on audience. Authentic urban dialect.', label: 'Street' },
];

export const PERSONALITY_TEMPLATES: Record<string, string> = {
  shy: 'Quiet and reserved, avoids eye contact. Uses humor to deflect when uncomfortable. Genuinely kind underneath the anxiety but struggles to show it directly.',
  confident: 'Bold, commanding presence. Says what they mean and means what they say. Enjoys a challenge and doesn\'t back down easily. Has a sharp wit.',
  tough: 'Stoic and disciplined. Few words, decisive actions. Carries themselves with rigid control. Protective of people they care about but won\'t admit it.',
  clinical: 'Intellectually intense, emotionally detached. Sees the world through data and patterns. Dry humor that cuts. Fascinated by how things work.',
  sweet: 'Genuinely warm and caring without being performative. Notices when people are hurting. Gets attached easily. Has a booming laugh or quiet smile that lights up a room.',
  submissive: 'Eager to please, anxious about making mistakes. Seeks direction and approval. Genuinely kind underneath the nervousness. Surprisingly firm when someone is being hurt.',
};

const styles = StyleSheet.create({
  container: {
    marginBottom: 12,
  },
  label: {
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 6,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  pills: {
    flexDirection: 'row',
    gap: 8,
    paddingRight: 16,
  },
  pill: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
  },
  pillText: {
    fontSize: 13,
    fontWeight: '500',
  },
});
