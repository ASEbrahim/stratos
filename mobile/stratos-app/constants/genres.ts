import { colors } from './theme';

export interface Genre {
  id: string;
  label: string;
  emoji: string;
  color: string;
}

export const GENRES: Genre[] = [
  { id: 'anime', label: 'Anime', emoji: '🌸', color: colors.accent.anime },
  { id: 'fantasy', label: 'Fantasy', emoji: '⚔️', color: colors.accent.fantasy },
  { id: 'scifi', label: 'Sci-Fi', emoji: '🚀', color: colors.accent.scifi },
  { id: 'romance', label: 'Romance', emoji: '💕', color: colors.accent.romance },
  { id: 'horror', label: 'Horror', emoji: '👻', color: colors.accent.horror },
  { id: 'modern', label: 'Modern', emoji: '🌆', color: colors.accent.modern },
  { id: 'historical', label: 'Historical', emoji: '📜', color: colors.accent.historical },
];

export const GENRE_MAP = Object.fromEntries(GENRES.map(g => [g.id, g]));

export function getGenreColor(genreId: string): string {
  return GENRE_MAP[genreId]?.color ?? colors.accent.primary;
}
