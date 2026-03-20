"""
Lens Extraction System — Configurable per-channel analysis prompts.

Each lens is a focused prompt that extracts specific types of insight
from a video transcript. Lenses run individually against qwen3.5:9b
via Ollama.

Available lenses: summary, eloquence, history, spiritual, politics, narrations

Usage:
    result = extract_lens(transcript, 'eloquence', 'Video Title', ollama_host, model)
"""

import json
import logging
import re
import requests
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Max transcript chars to send per lens call (prevents context overflow for 9B)
MAX_TRANSCRIPT_CHUNK = 6000

# ═══════════════════════════════════════════════════════════
# LENS PROMPT DEFINITIONS (each under 200 words)
# ═══════════════════════════════════════════════════════════

LENS_PROMPTS = {
    'summary': {
        'system': 'You extract concise summaries from lecture transcripts. Output valid JSON only.',
        'user': """Summarize this lecture transcript.

Title: {title}
Transcript: {transcript}

Output JSON:
{{"summary": "2-3 paragraph summary", "key_takeaways": ["point 1", "point 2", "point 3"]}}

JSON only, no other text.""",
    },

    'eloquence': {
        'system': 'You extract advanced vocabulary and phrases from transcripts. Output valid JSON only.',
        'user': """Extract uncommon and rare words or phrases from this lecture transcript. Focus on vocabulary that would expand a learner's proficiency.

Title: {title}
Transcript: {transcript}

For each word/phrase provide: the term, its definition, the context it was used in, and rarity ("uncommon" or "rare").

Output JSON array:
[{{"term": "word", "definition": "meaning", "context_quote": "sentence where used", "rarity": "uncommon"}}]

JSON array only, no other text.""",
    },

    'history': {
        'system': 'You extract historical narratives from lecture transcripts. Output valid JSON only.',
        'user': """Identify historical events, battles, or figures discussed in this lecture.

Title: {title}
Transcript: {transcript}

For each: event name, time period, key actors, summary, and the speaker's interpretation.

Output JSON array:
[{{"event": "name", "period": "era", "actors": ["person1"], "summary": "what happened", "speaker_interpretation": "speaker's view"}}]

JSON array only. If no historical events found, output [].""",
    },

    'spiritual': {
        'system': 'You extract spiritual and philosophical insights from lecture transcripts. Output valid JSON only.',
        'user': """Extract core spiritual lessons, philosophical arguments, or moral teachings from this lecture.

Title: {title}
Transcript: {transcript}

For each: the lesson, supporting evidence cited, and the speaker's framing.

Output JSON array:
[{{"lesson": "teaching", "supporting_evidence": "what was cited", "speaker_framing": "how presented"}}]

JSON array only. If no spiritual content, output [].""",
    },

    'politics': {
        'system': 'You extract political commentary from lecture transcripts. Output valid JSON only.',
        'user': """Identify political commentary, policy analysis, or geopolitical observations in this lecture.

Title: {title}
Transcript: {transcript}

For each: topic, analysis, speaker's position, and any cited sources.

Output JSON array:
[{{"topic": "subject", "analysis": "what was said", "speaker_position": "their stance", "cited_sources": ["source1"]}}]

JSON array only. If no political content, output [].""",
    },

    'narrations': {
        'system': 'You detect quotes, narrations, citations, and attributed statements in transcripts. This includes religious texts (any religion), philosophical works, historical accounts, scholarly references, proverbs, and any attributed speech. Output valid JSON only.',
        'user': """Detect any quotes, narrations, citations, or attributed statements in this lecture. This includes references from any tradition — religious (Islamic, Christian, Jewish, Hindu, Buddhist, etc.), philosophical, historical, literary, or scholarly.

Title: {title}
Transcript: {transcript}

For each narration provide:
- narration_text: the quote or narration
- speaker_attribution: who the speaker attributed it to (e.g. "Prophet Muhammad", "Jesus", "Aristotle", "Shakespeare")
- source_claimed: the source mentioned (e.g. "Sahih Bukhari", "Gospel of Matthew", "Republic by Plato", "Torah")
- source_reference: specific chapter/verse/number if mentioned (e.g. "Bukhari 6094", "Matthew 5:44", "Genesis 1:1", ""). Empty string if not identifiable.
- needs_verification: true if the exact source cannot be pinpointed

Output JSON array:
[{{"narration_text": "the quote", "speaker_attribution": "who said it", "source_claimed": "source book or text", "source_reference": "specific ref if any", "needs_verification": true}}]

JSON array only. If no narrations found, output [].""",
    },
}

AVAILABLE_LENSES = list(LENS_PROMPTS.keys()) + ['transcript']

# Language display names for prompts
LANGUAGE_NAMES = {
    'en': 'English', 'ar': 'Arabic', 'ja': 'Japanese',
    'ko': 'Korean', 'zh': 'Chinese', 'fr': 'French',
    'de': 'German', 'es': 'Spanish', 'ru': 'Russian',
}


def _build_language_instruction(target_language: str) -> str:
    """Build a language instruction suffix for lens prompts."""
    if target_language == 'en':
        return ''  # Default — no extra instruction needed
    lang_name = LANGUAGE_NAMES.get(target_language, target_language)
    return f'Respond in {lang_name}. All output text (summaries, definitions, analysis) must be in {lang_name}. Keep JSON keys in English.'


def extract_lens(transcript: str, lens_name: str, video_title: str,
                 ollama_host: str, model: str,
                 target_language: str = 'en') -> Optional[Any]:
    """Run a single lens extraction against a transcript.

    Args:
        transcript: Full transcript text
        lens_name: Lens name (must be in LENS_PROMPTS)
        video_title: Video title for context
        ollama_host: Ollama API host
        model: Model name (e.g., 'qwen3.5:9b')
        target_language: Language to extract in ('en', 'ar', 'ja', etc.)

    Returns:
        Parsed JSON result (dict or list) or None on failure
    """
    if lens_name not in LENS_PROMPTS:
        logger.warning(f"Unknown lens: {lens_name}")
        return None

    lens = LENS_PROMPTS[lens_name]

    # Chunk transcript if too long
    if len(transcript) > MAX_TRANSCRIPT_CHUNK:
        return _extract_lens_chunked(transcript, lens_name, video_title, ollama_host, model, target_language)

    # Build language instruction
    lang_instruction = _build_language_instruction(target_language)
    user_prompt = lens['user'].format(title=video_title, transcript=transcript)
    if lang_instruction:
        user_prompt += f'\n\n{lang_instruction}'

    try:
        resp = requests.post(
            f'{ollama_host}/api/chat',
            json={
                'model': model,
                'messages': [
                    {'role': 'system', 'content': lens['system']},
                    {'role': 'user', 'content': user_prompt},
                ],
                'stream': False,
                'think': False,
                'options': {'temperature': 0.2, 'num_predict': 2000, 'num_ctx': 8192},
            },
            timeout=120,
        )

        if resp.status_code != 200:
            logger.error(f"Lens '{lens_name}': Ollama returned {resp.status_code}")
            return None

        content = resp.json().get('message', {}).get('content', '')
        return _parse_json_response(content, lens_name)

    except Exception as e:
        logger.error(f"Lens '{lens_name}' extraction error: {e}")
        return None


def _extract_lens_chunked(transcript: str, lens_name: str, video_title: str,
                          ollama_host: str, model: str,
                          target_language: str = 'en') -> Optional[Any]:
    """Process long transcripts in chunks, then merge results."""
    chunks = _split_transcript(transcript, MAX_TRANSCRIPT_CHUNK)
    all_results = []

    for i, chunk in enumerate(chunks):
        chunk_title = f"{video_title} (part {i+1}/{len(chunks)})"
        result = extract_lens(chunk, lens_name, chunk_title, ollama_host, model, target_language)
        if result:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, dict):
                all_results.append(result)

    if not all_results:
        return None

    # For summary lens, merge the chunk summaries
    if lens_name == 'summary':
        return _merge_summaries(all_results)

    return all_results


def _split_transcript(text: str, max_chars: int) -> List[str]:
    """Split transcript at sentence boundaries."""
    chunks = []
    while len(text) > max_chars:
        # Find last sentence boundary before max_chars
        cut = max_chars
        for delim in ['. ', '? ', '! ', '؟ ', '。']:
            last = text[:max_chars].rfind(delim)
            if last > max_chars * 0.5:
                cut = last + len(delim)
                break
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def _merge_summaries(summaries: List[Any]) -> Dict[str, Any]:
    """Merge multiple chunk summaries into one."""
    all_takeaways = []
    summary_parts = []
    for s in summaries:
        if isinstance(s, dict):
            if s.get('summary'):
                summary_parts.append(s['summary'])
            if isinstance(s.get('key_takeaways'), list):
                all_takeaways.extend(s['key_takeaways'])
    return {
        'summary': ' '.join(summary_parts),
        'key_takeaways': all_takeaways[:7],  # Cap at 7 takeaways
    }


def _parse_json_response(content: str, lens_name: str) -> Optional[Any]:
    """Parse JSON from LLM response, handling common formatting issues."""
    # Strip think blocks
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding array or object
    for start, end in [('[', ']'), ('{', '}')]:
        idx_start = content.find(start)
        idx_end = content.rfind(end)
        if idx_start >= 0 and idx_end > idx_start:
            try:
                return json.loads(content[idx_start:idx_end+1])
            except json.JSONDecodeError:
                pass

    logger.warning(f"Lens '{lens_name}': could not parse JSON from response")
    return None
