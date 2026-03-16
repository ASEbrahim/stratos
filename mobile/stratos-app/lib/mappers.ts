/**
 * Backend → Mobile field mappers.
 *
 * The backend character_cards table uses different field names than
 * the mobile CharacterCard interface. These functions normalize.
 */

import { CharacterCard } from './types';
import { safeParse, buildUrl } from './utils';

/**
 * Map a backend character card response to mobile CharacterCard interface.
 */
export function mapCardFromBackend(raw: any): CharacterCard {
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
    content_rating: raw.content_rating || 'sfw',
    avatar_url: raw.avatar_image_path ? buildUrl(`/api/image/${raw.avatar_image_path}`) : (raw.avatar_url || ''),
    creator_id: String(raw.creator_profile_id || raw.creator_id || ''),
    creator_name: raw.creator_name || 'Unknown',
    is_public: raw.is_published ?? raw.is_public ?? false,
    session_count: raw.sessions ?? raw.session_count ?? 0,
    rating: raw.avg_rating_val ?? raw.avg_rating ?? raw.rating ?? 0,
    rating_count: raw.total_ratings ?? raw.rating_count ?? 0,
    created_at: raw.created_at || new Date().toISOString(),
    updated_at: raw.updated_at || new Date().toISOString(),
  };
}

/**
 * Map an array of backend cards.
 */
export function mapCardsFromBackend(raw: any[]): CharacterCard[] {
  return raw.map(mapCardFromBackend);
}
