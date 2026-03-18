// Pill field types
export type Gender = 'female' | 'male' | 'nonbinary';
export type Archetype = 'shy' | 'confident' | 'tough' | 'clinical' | 'sweet' | 'submissive';
export type NarrationPOV = 'first' | 'third' | 'mixed';
export type Relationship = 'stranger' | 'friend' | 'rival' | 'love_interest' | 'mentor' | 'servant';
export type NSFWComfort = 'fade' | 'suggestive' | 'explicit';
export type ResponseLength = 'brief' | 'normal' | 'detailed';
export type AgeRange = 'teen' | 'young_adult' | 'adult' | 'middle_aged' | 'elderly';

export interface CharacterCard {
  id: string;
  name: string;
  description: string;
  personality: string;
  scenario: string;
  first_message: string;
  physical_description: string;
  speech_pattern: string;
  emotional_trigger: string;
  defensive_mechanism: string;
  vulnerability: string;
  specific_detail: string;
  genre_tags: string[];
  content_rating: 'sfw' | 'nsfw';
  avatar_url: string;
  creator_id: string;
  creator_name: string;
  is_public: boolean;
  session_count: number;
  rating: number;
  rating_count: number;
  created_at: string;
  updated_at: string;
  // Pill fields (optional — NULL for cards created before pills)
  gender?: Gender;
  archetype_override?: Archetype;
  narration_pov?: NarrationPOV;
  relationship_to_user?: Relationship;
  nsfw_comfort?: NSFWComfort;
  response_length_pref?: ResponseLength;
  age_range?: AgeRange;
}

export interface CharacterCardCreate {
  name: string;
  description: string;
  personality: string;
  scenario: string;
  first_message: string;
  physical_description: string;
  speech_pattern: string;
  emotional_trigger: string;
  defensive_mechanism: string;
  vulnerability: string;
  specific_detail: string;
  genre_tags: string[];
  content_rating: 'sfw' | 'nsfw';
  avatar_url: string;
  // Pill fields (optional)
  gender?: Gender;
  archetype_override?: Archetype;
  narration_pov?: NarrationPOV;
  relationship_to_user?: Relationship;
  nsfw_comfort?: NSFWComfort;
  response_length_pref?: ResponseLength;
  age_range?: AgeRange;
}

export interface GamingScenario {
  id: string;
  name: string;
  description: string;
  genre: string;
  subgenre: string;
  rating: number;
  session_count: number;
  image_url: string;
  entities: ScenarioEntity[];
  initial_message: string;
}

export interface ScenarioEntity {
  name: string;
  type: string;
  stats?: Record<string, number>;
  description: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  dbId?: number;  // Backend rp_messages.id — used for edit/feedback API calls
}

export interface ChatSession {
  id: string;
  character_id: string;
  character_name: string;
  character_avatar: string;
  persona: 'roleplay' | 'gaming';
  session_context?: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
}

export interface Suggestion {
  label: string;
  prompt: string;
}

export interface User {
  id: string;
  name: string;
  email: string;
  avatar_url: string;
  created_at: string;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface TavernCardV2 {
  name: string;
  description: string;
  personality: string;
  scenario: string;
  first_mes: string;
  mes_example: string;
  creator_notes: string;
  tags: string[];
  spec: string;
  spec_version: string;
}

export class AuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AuthError';
  }
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export function getQualityScore(card: CharacterCard | CharacterCardCreate): {
  score: number;
  level: 'basic' | 'good' | 'great' | 'exceptional';
  label: string;
  elements: { name: string; filled: boolean }[];
} {
  const elements = [
    { name: 'Physical', filled: !!card.physical_description?.trim() },
    { name: 'Speech', filled: !!card.speech_pattern?.trim() },
    { name: 'Trigger', filled: !!card.emotional_trigger?.trim() },
    { name: 'Defense', filled: !!card.defensive_mechanism?.trim() },
    { name: 'Vulnerability', filled: !!card.vulnerability?.trim() },
    { name: 'Detail', filled: !!card.specific_detail?.trim() },
  ];
  const score = elements.filter(e => e.filled).length;
  const hasFirstMessage = 'first_message' in card && !!card.first_message?.trim();
  const hasScenario = !!card.scenario?.trim();

  let level: 'basic' | 'good' | 'great' | 'exceptional';
  let label: string;

  if (score >= 6 && hasFirstMessage && hasScenario) {
    level = 'exceptional';
    label = 'Exceptional';
  } else if (score >= 5) {
    level = 'great';
    label = 'Great';
  } else if (score >= 3) {
    level = 'good';
    label = 'Good';
  } else {
    level = 'basic';
    label = 'Basic';
  }

  return { score, level, label, elements };
}

export function formatCount(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return String(n);
}
