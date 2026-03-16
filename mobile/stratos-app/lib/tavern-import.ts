import * as FileSystem from 'expo-file-system';
import { TavernCardV2, CharacterCardCreate } from './types';
import { reportError } from './utils';

/**
 * Parse a TavernCard V2 PNG file and extract the embedded character data.
 * TavernCard V2 stores JSON in the PNG tEXt chunk with keyword "chara" (base64 encoded).
 */
export async function parseTavernCard(uri: string): Promise<CharacterCardCreate | null> {
  try {
    // Read file as base64
    const fileBase64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' as any });
    const bytes = base64ToBytes(fileBase64);

    // PNG signature: 137 80 78 71 13 10 26 10
    if (bytes.length < 8 || bytes[0] !== 137 || bytes[1] !== 80 || bytes[2] !== 78 || bytes[3] !== 71) {
      return null; // Not a valid PNG
    }

    // Walk PNG chunks looking for tEXt with keyword "chara"
    let offset = 8; // Skip PNG signature
    while (offset < bytes.length - 12) {
      const length = readUint32(bytes, offset);
      const type = String.fromCharCode(bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7]);

      if (type === 'tEXt') {
        const chunkData = bytes.slice(offset + 8, offset + 8 + length);
        // keyword is null-terminated
        const nullIdx = chunkData.indexOf(0);
        if (nullIdx !== -1) {
          const keyword = bytesToString(chunkData.slice(0, nullIdx));
          if (keyword === 'chara') {
            const valueBytes = chunkData.slice(nullIdx + 1);
            const valueStr = bytesToString(valueBytes);
            // Value is base64-encoded JSON
            const jsonStr = atob(valueStr);
            const tavernCard: TavernCardV2 = JSON.parse(jsonStr);
            return mapTavernToCard(tavernCard);
          }
        }
      }

      // Move to next chunk: length(4) + type(4) + data(length) + crc(4)
      offset += 12 + length;
    }

    return null; // No chara tEXt chunk found
  } catch (err) {
    reportError('parseTavernCard', err);
    return null;
  }
}

function mapTavernToCard(tavern: TavernCardV2): CharacterCardCreate {
  // Split description heuristically — first paragraph as physical, rest as description
  const descParagraphs = (tavern.description ?? '').split('\n').filter(p => p.trim());
  const physicalDesc = descParagraphs[0] ?? '';
  const remainingDesc = descParagraphs.slice(1).join('\n') || tavern.description;

  // Split personality into components
  const personalityParts = (tavern.personality ?? '').split(/[.;]\s*/);
  const speechPattern = personalityParts[0] ?? '';
  const emotionalTrigger = personalityParts[1] ?? '';

  // Map tags to our genre format
  const genreTags = mapTavernTags(tavern.tags ?? []);

  return {
    name: tavern.name ?? 'Imported Character',
    description: remainingDesc,
    personality: tavern.personality ?? '',
    scenario: tavern.scenario ?? '',
    first_message: tavern.first_mes ?? '',
    physical_description: physicalDesc,
    speech_pattern: speechPattern,
    emotional_trigger: emotionalTrigger,
    defensive_mechanism: '',
    vulnerability: '',
    specific_detail: '',
    genre_tags: genreTags,
    content_rating: 'sfw',
    avatar_url: '',
  };
}

function mapTavernTags(tags: string[]): string[] {
  const tagMap: Record<string, string> = {
    fantasy: 'fantasy', magic: 'fantasy', 'high fantasy': 'fantasy',
    medieval: 'fantasy',
    'sci-fi': 'scifi', scifi: 'scifi', 'science fiction': 'scifi', cyberpunk: 'scifi', space: 'scifi',
    romance: 'romance', love: 'romance', dating: 'romance',
    horror: 'horror', dark: 'horror', supernatural: 'horror', vampire: 'horror',
    modern: 'modern', 'slice of life': 'modern', realistic: 'modern',
    anime: 'anime', manga: 'anime',
    historical: 'historical', victorian: 'historical',
  };

  const mapped = new Set<string>();
  for (const tag of tags) {
    const lower = tag.toLowerCase().trim();
    if (tagMap[lower]) mapped.add(tagMap[lower]);
  }
  return Array.from(mapped);
}

// ─── Binary helpers ───

function base64ToBytes(base64: string): Uint8Array {
  const raw = atob(base64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

function readUint32(bytes: Uint8Array, offset: number): number {
  return (bytes[offset] << 24) | (bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3];
}

function bytesToString(bytes: Uint8Array): string {
  try {
    return new TextDecoder('utf-8').decode(bytes);
  } catch (err) {
    reportError('bytesToString', err);
    // Fallback for environments without TextDecoder
    let str = '';
    for (let i = 0; i < bytes.length; i++) str += String.fromCharCode(bytes[i]);
    return str;
  }
}
