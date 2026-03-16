"""
Narration Source Resolution Engine — Pattern-based cascade resolver.

Stage A: Reference pattern matching against known API endpoints (sync, free).
Stage B: Serper web search with authority domain scoring (async, costs ~$0.001/query).
Stage C: Google Search fallback (frontend-only, no backend involvement).

Usage:
    from processors.source_resolver import resolve_source, resolve_sources_async

    # Sync — Strategy A only (inline during extraction)
    result = resolve_source(source_claimed, source_reference, narration_text)

    # Async — Strategy A + B (background thread after extraction)
    resolve_sources_async(video_id, profile_id, narrations, db, config, sse_manager)
"""

import hashlib
import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlparse

logger = logging.getLogger("STRAT_OS")


# ═══════════════════════════════════════════════════════════
# STRATEGY A — Reference Pattern Resolvers
# ═══════════════════════════════════════════════════════════

REFERENCE_RESOLVERS = [
    {
        'name': 'quran_direct',
        'source_pattern': re.compile(
            r'(?i)quran|qur.an|surah|سورة|ayah|آية|قرآن'
        ),
        'ref_pattern': re.compile(r'(\d{1,3})[:\s](\d{1,3})'),
        'url_builder': lambda m, **_: f'https://quran.com/{m.group(1)}:{m.group(2)}',
        'confidence': 0.95,
    },
    {
        'name': 'bible_direct',
        'source_pattern': re.compile(
            r'(?i)bible|genesis|exodus|leviticus|numbers|deuteronomy|joshua|judges|ruth|'
            r'samuel|kings|chronicles|ezra|nehemiah|esther|job|psalm|proverbs|ecclesiastes|'
            r'isaiah|jeremiah|lamentations|ezekiel|daniel|hosea|joel|amos|obadiah|jonah|'
            r'micah|nahum|habakkuk|zephaniah|haggai|zechariah|malachi|matthew|mark|luke|'
            r'john|acts|romans|corinthians|galatians|ephesians|philippians|colossians|'
            r'thessalonians|timothy|titus|philemon|hebrews|james|peter|jude|revelation|gospel'
        ),
        'ref_pattern': re.compile(r'(\w+)\s+(\d+):(\d+)'),
        'url_builder': lambda m, **_: (
            f'https://www.biblegateway.com/passage/?search='
            f'{quote(m.group(1))}+{m.group(2)}:{m.group(3)}&version=NIV'
        ),
        'confidence': 0.90,
    },
    {
        'name': 'gita_direct',
        'source_pattern': re.compile(r'(?i)bhagavad.?gita|gita'),
        'ref_pattern': re.compile(r'(\d+)[:\.\s]+(\d+)'),
        'url_builder': lambda m, **_: (
            f'https://www.holy-bhagavad-gita.org/chapter/{m.group(1)}/verse/{m.group(2)}'
        ),
        'confidence': 0.85,
    },
    {
        'name': 'sefaria',
        'source_pattern': re.compile(
            r'(?i)torah|talmud|mishnah|midrash|zohar|shemot|vayikra|bamidbar|devarim|'
            r'berakhot|shabbat|sanhedrin|bava|ketubbot|gittin|kiddushin'
        ),
        'ref_pattern': re.compile(r'.+'),
        'url_builder': lambda m, source='', ref='', **_: (
            f'https://www.sefaria.org/search?q={quote((ref or source).strip())}'
        ),
        'confidence': 0.80,
    },
    {
        'name': 'sunnah_search',
        'source_pattern': re.compile(
            r'(?i)hadith|bukhari|بخاري|muslim|مسلم|tirmidhi|ترمذي|'
            r'dawud|داود|abu dawud|nasa.i|نسائي|ibn majah|ماجه|'
            r'muwatta|malik|sahih|sunan|musnad|riyadh|nawawi|bulugh'
        ),
        'ref_pattern': re.compile(r'(\d+)'),
        'url_builder': lambda m, source='', **_: (
            f'https://sunnah.com/search?q={quote(source.strip())}+{m.group(1)}'
        ),
        'confidence': 0.70,
    },
    {
        # Sunnah.com fallback — no number, just search by source name
        'name': 'sunnah_search_noref',
        'source_pattern': re.compile(
            r'(?i)hadith|bukhari|بخاري|muslim|مسلم|tirmidhi|ترمذي|'
            r'dawud|داود|nasa.i|نسائي|majah|ماجه|muwatta|sahih|sunan'
        ),
        'ref_pattern': re.compile(r'.*'),  # matches anything
        'url_builder': lambda m, source='', text='', **_: (
            f'https://sunnah.com/search?q={quote((source + " " + text[:50]).strip())}'
        ),
        'confidence': 0.55,
    },
    {
        'name': 'suttacentral',
        'source_pattern': re.compile(
            r'(?i)dhammapada|pali canon|sutta|tipitaka|tripitaka|nikaya|vinaya'
        ),
        'ref_pattern': re.compile(r'.+'),
        'url_builder': lambda m, source='', ref='', **_: (
            f'https://suttacentral.net/search?query={quote((ref or source).strip())}'
        ),
        'confidence': 0.70,
    },
    {
        'name': 'thaqalayn',
        'source_pattern': re.compile(
            r'(?i)al-kafi|kafi|الكافي|nahj al-balagha|نهج البلاغة|'
            r'bihar al-anwar|بحار|man la yahduruhu|tahdhib|istibsar|wasail'
        ),
        'ref_pattern': re.compile(r'.+'),
        'url_builder': lambda m, source='', ref='', **_: (
            f'https://thaqalayn.net/search?q={quote((ref or source).strip())}'
        ),
        'confidence': 0.65,
    },
    {
        'name': 'al_islam',
        'source_pattern': re.compile(
            r'(?i)nahj|نهج|imam ali|امام علي|shia|shiite|twelver'
        ),
        'ref_pattern': re.compile(r'.+'),
        'url_builder': lambda m, source='', ref='', **_: (
            f'https://www.al-islam.org/search?keys={quote((ref or source).strip())}'
        ),
        'confidence': 0.65,
    },
]


def _hash_narration(text: str) -> str:
    """SHA256 hash of narration text for dedup."""
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:32]


def resolve_source(
    source_claimed: str,
    source_reference: str,
    narration_text: str,
) -> Optional[Dict[str, Any]]:
    """
    Strategy A: Try pattern-based resolvers. Returns first confident match.
    Returns { url, confidence, method } or None.
    """
    combined = f'{source_claimed} {source_reference} {narration_text}'

    for resolver in REFERENCE_RESOLVERS:
        if not resolver['source_pattern'].search(combined):
            continue

        # Try matching reference pattern against source_reference first, then combined
        ref_text = source_reference or combined
        match = resolver['ref_pattern'].search(ref_text)
        if not match and ref_text != combined:
            match = resolver['ref_pattern'].search(combined)
        if not match:
            continue

        try:
            url = resolver['url_builder'](
                match,
                source=source_claimed,
                ref=source_reference,
                text=narration_text,
            )
            if url:
                return {
                    'url': url,
                    'confidence': resolver['confidence'],
                    'method': f'pattern:{resolver["name"]}',
                }
        except Exception as e:
            logger.debug(f"Resolver {resolver['name']} failed: {e}")
            continue

    return None


# ═══════════════════════════════════════════════════════════
# STRATEGY B — Async Serper Web Search Resolution
# ═══════════════════════════════════════════════════════════

AUTHORITY_DOMAINS = {
    'sunnah.com', 'quran.com', 'biblegateway.com', 'sefaria.org',
    'sacred-texts.com', 'britannica.com', 'plato.stanford.edu',
    'gutenberg.org', 'jstor.org', 'archive.org', 'scholar.google.com',
    'thaqalayn.net', 'suttacentral.net', 'vedabase.io',
    'newadvent.org', 'jewishvirtuallibrary.org', 'al-islam.org',
    'holy-bhagavad-gita.org', 'wisdomlib.org',
}


def _resolve_via_search(
    source_claimed: str,
    source_reference: str,
    narration_text: str,
    serper_client,
) -> Optional[Dict[str, Any]]:
    """
    Strategy B: Use Serper API to find the canonical source page.
    One query per narration — never batch.
    """
    # Build targeted query
    parts = []
    if source_claimed:
        parts.append(f'"{source_claimed}"')
    if source_reference:
        parts.append(source_reference)
    query = ' '.join(parts).strip()
    if not query:
        query = narration_text[:60]
    if not query:
        return None

    try:
        results = serper_client.search(query, num_results=3)
    except Exception as e:
        logger.debug(f"Serper search failed for '{query[:50]}': {e}")
        return None

    if not results:
        return None

    # Check authority domains first
    for result in results:
        url = result.get('url') or result.get('link', '')
        if not url:
            continue
        domain = urlparse(url).netloc.replace('www.', '')
        if domain in AUTHORITY_DOMAINS:
            return {'url': url, 'confidence': 0.85, 'method': 'search_authority'}

    # Fallback: top result with lower confidence
    top_url = results[0].get('url') or results[0].get('link', '')
    if top_url:
        return {'url': top_url, 'confidence': 0.50, 'method': 'search_top'}

    return None


def resolve_sources_async(
    video_id: int,
    profile_id: int,
    narrations: List[Dict],
    db,
    config: Dict,
    sse_manager=None,
):
    """
    Background resolution: Strategy A (inline) + Strategy B (Serper) for unresolved.
    Stores results in narration_sources table. Fires SSE on completion.
    Max 3 Serper queries per video to control API costs.
    """
    # Get DB path for thread-safe independent connection
    db_path = str(db.db_path) if hasattr(db, 'db_path') else None
    if not db_path:
        logger.warning("Source resolver: no db_path available, cannot run async resolution")
        return

    def _run():
        import sqlite3
        serper_client = None
        serper_queries_used = 0
        max_serper_queries = 3
        resolved_count = 0

        # Use independent DB connection — never share db.conn across threads
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
        except Exception as e:
            logger.error(f"Source resolver: failed to open DB: {e}")
            return

        try:
            for narration in narrations:
                text = narration.get('narration_text', '')
                source = narration.get('source_claimed', '')
                ref = narration.get('source_reference', '')
                if not text:
                    continue

                nar_hash = _hash_narration(text)

                # Check if already resolved in DB
                cursor.execute(
                    "SELECT resolved_url FROM narration_sources "
                    "WHERE video_id = ? AND profile_id = ? AND narration_hash = ?",
                    (video_id, profile_id, nar_hash)
                )
                if cursor.fetchone():
                    continue  # Already cached

                # Strategy A: pattern matching (free, sync)
                result = resolve_source(source, ref, text)

                # Strategy B: Serper search (if A failed and budget remains)
                if not result and serper_queries_used < max_serper_queries:
                    if serper_client is None:
                        try:
                            from fetchers.serper_search import get_serper_client
                            serper_client = get_serper_client(config)
                        except Exception as e:
                            logger.debug(f"Serper client unavailable: {e}")
                            serper_client = False  # Mark as unavailable
                    if serper_client:
                        result = _resolve_via_search(source, ref, text, serper_client)
                        serper_queries_used += 1

                if result:
                    try:
                        cursor.execute(
                            """INSERT OR REPLACE INTO narration_sources
                               (video_id, profile_id, narration_hash, source_claimed,
                                source_reference, resolved_url, resolution_method, confidence)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            (video_id, profile_id, nar_hash, source, ref,
                             result['url'], result['method'], result['confidence'])
                        )
                        conn.commit()
                        resolved_count += 1
                    except Exception as e:
                        logger.error(f"Failed to store resolved source: {e}")

            if resolved_count > 0 and sse_manager:
                sse_manager.broadcast('narration_resolved', {
                    'video_id': video_id,
                    'resolved_count': resolved_count,
                })
                logger.info(
                    f"Source resolver: {resolved_count} narrations resolved for video {video_id} "
                    f"({serper_queries_used} Serper queries used)"
                )
        except Exception as e:
            logger.error(f"Source resolver thread error: {e}")
        finally:
            conn.close()

    threading.Thread(target=_run, daemon=True).start()
