/**
 * Backend → Mobile field mappers.
 *
 * The backend character_cards table uses different field names than
 * the mobile CharacterCard interface. These functions normalize.
 */

import { CharacterCard } from './types';
import { safeParse, buildUrl } from './utils';

/** Shape of a character card as returned by the backend API. */
export interface BackendCard {
  id: string;
  name?: string;
  personality?: string;
  description?: string;
  scenario?: string;
  first_message?: string;
  physical_description?: string;
  speech_pattern?: string;
  emotional_trigger?: string;
  defensive_mechanism?: string;
  vulnerability?: string;
  specific_detail?: string;
  genre_tags?: string | string[];
  content_rating?: string;
  avatar_image_path?: string;
  avatar_url?: string;
  // Pill fields
  gender?: string;
  archetype_override?: string;
  narration_pov?: string;
  relationship_to_user?: string;
  nsfw_comfort?: string;
  response_length_pref?: string;
  age_range?: string;
  personality_tags?: string | string[];
  creator_profile_id?: string | number;
  creator_id?: string | number;
  creator_name?: string;
  is_published?: boolean;
  is_public?: boolean;
  sessions?: number;
  session_count?: number;
  avg_rating_val?: number;
  avg_rating?: number;
  rating?: number;
  total_ratings?: number;
  rating_count?: number;
  created_at?: string;
  updated_at?: string;
}

/**
 * Map a backend character card response to mobile CharacterCard interface.
 */
export function mapCardFromBackend(raw: BackendCard): CharacterCard {
  return {
    id: raw.id,
    name: raw.name || '',
    description: raw.personality || raw.description || '',
    personality: raw.personality || '',
    scenario: raw.scenario || '',
    first_message: raw.first_message || '',
    physical_description: raw.physical_description || '',
    speech_pattern: raw.speech_pattern || '',
    emotional_trigger: raw.emotional_trigger || '',
    defensive_mechanism: raw.defensive_mechanism || '',
    vulnerability: raw.vulnerability || '',
    specific_detail: raw.specific_detail || '',
    genre_tags: typeof raw.genre_tags === 'string' ? safeParse(raw.genre_tags, []) : (raw.genre_tags || []),
    content_rating: (raw.content_rating === 'nsfw' ? 'nsfw' : 'sfw') as 'sfw' | 'nsfw',
    avatar_url: raw.avatar_image_path ? buildUrl(`/api/image/${raw.avatar_image_path}`) : (raw.avatar_url || ''),
    creator_id: String(raw.creator_profile_id || raw.creator_id || ''),
    creator_name: raw.creator_name || 'Unknown',
    is_public: raw.is_published ?? raw.is_public ?? false,
    session_count: raw.sessions ?? raw.session_count ?? 0,
    rating: raw.avg_rating_val ?? raw.avg_rating ?? raw.rating ?? 0,
    rating_count: raw.total_ratings ?? raw.rating_count ?? 0,
    created_at: raw.created_at || new Date().toISOString(),
    updated_at: raw.updated_at || new Date().toISOString(),
    // Pill fields — backend strings are validated server-side, safe to narrow
    gender: raw.gender as CharacterCard['gender'],
    archetype_override: raw.archetype_override as CharacterCard['archetype_override'],
    narration_pov: raw.narration_pov as CharacterCard['narration_pov'],
    relationship_to_user: raw.relationship_to_user as CharacterCard['relationship_to_user'],
    nsfw_comfort: raw.nsfw_comfort as CharacterCard['nsfw_comfort'],
    response_length_pref: raw.response_length_pref as CharacterCard['response_length_pref'],
    age_range: raw.age_range as CharacterCard['age_range'],
    personality_tags: typeof raw.personality_tags === 'string' ? safeParse(raw.personality_tags, []) : (raw.personality_tags || []),
  };
}

/**
 * Map an array of backend cards.
 */
export function mapCardsFromBackend(raw: BackendCard[]): CharacterCard[] {
  return raw.map(mapCardFromBackend);
}
