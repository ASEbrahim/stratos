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
      {label ? <Text style={[styles.label, { color: tc.text.secondary }]}>{label}</Text> : null}
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

// ── Multi-select pill selector (for tags, mix-and-match) ──

interface MultiPillSelectorProps {
  label: string;
  options: PillOption[];
  values: string[];
  onChange: (values: string[]) => void;
  maxSelections?: number;
  wrap?: boolean; // use wrapping layout instead of horizontal scroll
}

export function MultiPillSelector({
  label, options, values, onChange, maxSelections = 8, wrap = false,
}: MultiPillSelectorProps) {
  const tc = useThemeStore(s => s.colors);

  const toggle = (val: string) => {
    if (values.includes(val)) {
      onChange(values.filter(v => v !== val));
    } else if (values.length < maxSelections) {
      onChange([...values, val]);
    }
  };

  const content = options.map(opt => {
    const selected = values.includes(opt.value);
    return (
      <TouchableOpacity
        key={opt.value}
        style={[
          styles.pill,
          { borderColor: tc.border.subtle, backgroundColor: tc.bg.secondary },
          selected && { borderColor: tc.accent.primary, backgroundColor: tc.accent.primary + '18' },
        ]}
        onPress={() => toggle(opt.value)}
        activeOpacity={0.7}
        accessibilityRole="button"
        accessibilityState={{ selected }}
      >
        <Text style={[
          styles.pillText,
          { color: selected ? tc.accent.primary : tc.text.primary },
        ]}>
          {opt.label}
        </Text>
      </TouchableOpacity>
    );
  });

  return (
    <View style={styles.container}>
      {label ? (
        <View style={styles.labelRow}>
          <Text style={[styles.label, { color: tc.text.secondary }]}>{label}</Text>
          {values.length > 0 && (
            <Text style={[styles.countBadge, { color: tc.accent.primary }]}>{values.length}</Text>
          )}
        </View>
      ) : null}
      {wrap ? (
        <View style={styles.wrapPills}>{content}</View>
      ) : (
        <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.pills}>
          {content}
        </ScrollView>
      )}
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
  { value: 'tsundere', label: 'Tsundere' },
  { value: 'yandere', label: 'Yandere' },
  { value: 'kuudere', label: 'Kuudere' },
  { value: 'manipulative', label: 'Manipulative' },
  { value: 'chaotic', label: 'Chaotic / Wild' },
  { value: 'elegant', label: 'Elegant / Refined' },
  { value: 'nurturing', label: 'Nurturing / Maternal' },
  { value: 'brooding', label: 'Brooding / Dark' },
];

export const POV_OPTIONS: PillOption[] = [
  { value: 'third', label: '3rd Person' },
  { value: 'first', label: '1st Person' },
  { value: 'mixed', label: 'Mixed' },
];

export const RELATIONSHIP_OPTIONS: PillOption[] = [
  { value: 'stranger', label: 'Stranger' },
  { value: 'friend', label: 'Friend' },
  { value: 'childhood_friend', label: 'Childhood Friend' },
  { value: 'rival', label: 'Rival' },
  { value: 'love_interest', label: 'Love Interest' },
  { value: 'ex', label: 'Ex' },
  { value: 'mentor', label: 'Mentor' },
  { value: 'student', label: 'Student' },
  { value: 'servant', label: 'Servant' },
  { value: 'master', label: 'Master' },
  { value: 'coworker', label: 'Coworker' },
  { value: 'roommate', label: 'Roommate' },
  { value: 'enemy', label: 'Enemy' },
  { value: 'sibling', label: 'Sibling' },
  { value: 'bodyguard', label: 'Bodyguard' },
  { value: 'captor', label: 'Captor' },
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
  { value: 'A luxury penthouse at midnight. Floor-to-ceiling windows overlook the city. Wine glasses sit on the marble counter.', label: 'Penthouse' },
  { value: 'A dark dungeon with stone walls. Chains hang from the ceiling. Torchlight flickers and shadows dance.', label: 'Dungeon' },
  { value: 'A rooftop at sunset. The wind tugs at hair and clothes. The city stretches below in orange light.', label: 'Rooftop' },
  { value: 'A high school hallway after hours. Lockers line the walls. Footsteps echo on polished floors.', label: 'After School' },
  { value: 'A royal court in a fantasy kingdom. Marble pillars rise to vaulted ceilings. Nobles whisper behind silk fans.', label: 'Royal Court' },
  { value: 'A cyberpunk back alley. Neon signs in foreign scripts. Rain slicks the asphalt. A drone buzzes overhead.', label: 'Cyberpunk Alley' },
  { value: 'A beach at night. Waves lap at the shore. The moon reflects on dark water. A bonfire crackles nearby.', label: 'Night Beach' },
  { value: 'A train compartment, late at night. Rain streaks the window. The rhythmic clatter of tracks fills the silence.', label: 'Night Train' },
  { value: 'A fighting arena. Sand and blood on the floor. The crowd roars from the stands above.', label: 'Arena' },
  { value: 'A library in an ancient manor. Bookshelves tower to the ceiling. Dust motes float in lamplight.', label: 'Manor Library' },
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
  { value: 'Commanding, authoritative. Expects to be obeyed. Speaks in orders and statements, never questions.', label: 'Commanding' },
  { value: 'Soft-spoken, almost a whisper. Chooses words carefully. Pauses between sentences. Calming presence.', label: 'Soft' },
  { value: 'Theatrical and dramatic. Grand gestures in words. Everything is an event. Never boring.', label: 'Dramatic' },
  { value: 'Cold and clinical. Speaks about emotions like symptoms. Uses precise, detached language.', label: 'Detached' },
  { value: 'Crude and vulgar. Swears freely. Says what everyone is thinking but won\'t say. No filter.', label: 'Vulgar' },
  { value: 'Old-fashioned, archaic vocabulary. "Pray tell", "indeed", "I daresay". Chivalrous or courtly diction.', label: 'Archaic' },
];

// ── Personality tags (multi-select, mix-and-match) ──
export const PERSONALITY_TAG_OPTIONS: PillOption[] = [
  // Temperament
  { value: 'flirty', label: 'Flirty' },
  { value: 'aggressive', label: 'Aggressive' },
  { value: 'gentle', label: 'Gentle' },
  { value: 'cold', label: 'Cold' },
  { value: 'hot-tempered', label: 'Hot-Tempered' },
  { value: 'patient', label: 'Patient' },
  { value: 'jealous', label: 'Jealous' },
  { value: 'possessive', label: 'Possessive' },
  { value: 'sadistic', label: 'Sadistic' },
  { value: 'masochistic', label: 'Masochistic' },
  // Social
  { value: 'sarcastic', label: 'Sarcastic' },
  { value: 'witty', label: 'Witty' },
  { value: 'awkward', label: 'Awkward' },
  { value: 'charming', label: 'Charming' },
  { value: 'blunt', label: 'Blunt' },
  { value: 'manipulative-tag', label: 'Scheming' },
  { value: 'loyal', label: 'Loyal' },
  { value: 'rebellious', label: 'Rebellious' },
  // Emotional
  { value: 'emotionally-guarded', label: 'Guarded' },
  { value: 'emotionally-open', label: 'Open' },
  { value: 'insecure', label: 'Insecure' },
  { value: 'cocky', label: 'Cocky' },
  { value: 'traumatized', label: 'Traumatized' },
  { value: 'lonely', label: 'Lonely' },
  { value: 'cheerful', label: 'Cheerful' },
  { value: 'melancholic', label: 'Melancholic' },
  // Quirks
  { value: 'clumsy', label: 'Clumsy' },
  { value: 'perfectionist', label: 'Perfectionist' },
  { value: 'workaholic', label: 'Workaholic' },
  { value: 'lazy', label: 'Lazy' },
  { value: 'bookworm', label: 'Bookworm' },
  { value: 'foodie', label: 'Foodie' },
  { value: 'competitive', label: 'Competitive' },
  { value: 'oblivious', label: 'Oblivious' },
];

export const PERSONALITY_TEMPLATES: Record<string, string> = {
  shy: 'Quiet and reserved, avoids eye contact. Uses humor to deflect when uncomfortable. Genuinely kind underneath the anxiety but struggles to show it directly.',
  confident: 'Bold, commanding presence. Says what they mean and means what they say. Enjoys a challenge and doesn\'t back down easily. Has a sharp wit.',
  tough: 'Stoic and disciplined. Few words, decisive actions. Carries themselves with rigid control. Protective of people they care about but won\'t admit it.',
  clinical: 'Intellectually intense, emotionally detached. Sees the world through data and patterns. Dry humor that cuts. Fascinated by how things work.',
  sweet: 'Genuinely warm and caring without being performative. Notices when people are hurting. Gets attached easily. Has a booming laugh or quiet smile that lights up a room.',
  submissive: 'Eager to please, anxious about making mistakes. Seeks direction and approval. Genuinely kind underneath the nervousness. Surprisingly firm when someone is being hurt.',
  dominant: 'Commands the room. Expects obedience but earns respect. Precise, deliberate, controlled. Gets dangerously quiet when angry instead of loud.',
  playful: 'Mischievous energy that never stops. Loves pushing buttons and finding boundaries. Quick with a grin. Hides genuine feelings behind jokes and pranks.',
  mysterious: 'Enigmatic, speaks in riddles. Reveals nothing about their past. Watches more than participates. When they finally open up, it means everything.',
  protective: 'Puts others before themselves every time. Hyper-aware of danger. Won\'t let anyone they care about get hurt. Struggles to accept help in return.',
  tsundere: 'Harsh and dismissive on the outside, secretly caring underneath. Denies any feelings aggressively. Blushes easily. Insults are their love language.',
  yandere: 'Obsessively devoted. Sweet and loving on the surface but intensely possessive. Will do anything to keep their person close. Unstable when threatened.',
  kuudere: 'Emotionless facade. Speaks in monotone. Shows affection through tiny, almost invisible gestures. Ice that slowly melts.',
  manipulative: 'Always three steps ahead. Charming when they want something. Every word is calculated. Genuinely surprised when someone sees through them.',
  chaotic: 'Unpredictable, lives for the moment. Might start a fight or a party with equal enthusiasm. No filter, no plan, no regrets.',
  elegant: 'Refined and composed. Every movement is deliberate. Speaks with precision. Has standards and expectations. Vulnerability is deeply private.',
  nurturing: 'Motherly instinct whether they want it or not. Feeds everyone, worries about everyone. Surprisingly fierce when someone threatens their people.',
  brooding: 'Carries a weight they won\'t share. Dark sense of humor. Drawn to shadows and solitude. Opens up only when they think no one is watching.',
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
  labelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 6,
  },
  countBadge: {
    fontSize: 11,
    fontWeight: '700',
  },
  pills: {
    flexDirection: 'row',
    gap: 8,
    paddingRight: 16,
  },
  wrapPills: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
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
