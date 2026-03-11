"""
Scholarly Citation Verification — Anti-Hallucination RAG

Verifies hadith, historical narrations, and scholarly citations detected
by the 'narrations' lens against real databases. The LLM detects narrations;
this module does deterministic verification via web search.

ABSOLUTE RULE: Never fabricate citations. If not found → "requires manual verification".

Trusted domains:
  - thaqalayn.net — Shia hadith database with chain verification
  - shiaonlinelibrary.com — Comprehensive Shia scholarly library
  - al-islam.org — Major Islamic text repository
  - hadith.net — Hadith search engine
  - lib.eshia.ir — Extensive Arabic scholarly archive
"""

import logging
import os
import re
import requests
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

TRUSTED_DOMAINS = [
    'thaqalayn.net',
    'shiaonlinelibrary.com',
    'al-islam.org',
    'hadith.net',
    'lib.eshia.ir',
]


def verify_narration(narration_text: str, speaker_attribution: str = '',
                     source_claimed: str = '', config: dict = None) -> Dict[str, Any]:
    """Attempt to verify a scholarly narration against trusted databases.

    Uses Serper/Google search restricted to trusted scholarly domains.
    NEVER fabricates citations — returns "requires manual verification" if not found.

    Args:
        narration_text: The narration text to verify
        speaker_attribution: Who the speaker said narrated it
        source_claimed: Book or chain the speaker mentioned
        config: App config dict (for search API keys)

    Returns:
        Verification result dict
    """
    config = config or {}
    result = {
        'verified': False,
        'search_attempted': True,
        'domains_searched': [],
        'results': [],
        'message': '',
    }

    # Extract search keywords from the narration
    keywords = _extract_search_keywords(narration_text, speaker_attribution, source_claimed)
    if not keywords:
        result['search_attempted'] = False
        result['message'] = 'Could not extract search keywords from narration.'
        return result

    # Try Serper search restricted to trusted domains
    serper_key = os.environ.get('SERPER_API_KEY', '') or config.get('search', {}).get('serper_api_key', '')
    if serper_key:
        search_results = _search_serper(keywords, serper_key)
        result['domains_searched'] = TRUSTED_DOMAINS
        if search_results:
            result['results'] = search_results
            result['verified'] = True
            result['message'] = f'Found {len(search_results)} potential match(es) in scholarly databases.'
            # Extract source details from best match
            best = search_results[0]
            result['source_url'] = best.get('url', '')
            result['source_title'] = best.get('title', '')
            result['source_snippet'] = best.get('snippet', '')
            result['confidence'] = _assess_confidence(narration_text, search_results)
            return result

    # Try SearXNG if configured
    searxng_host = config.get('search', {}).get('searxng_host', '')
    if searxng_host:
        search_results = _search_searxng(keywords, searxng_host)
        result['domains_searched'] = TRUSTED_DOMAINS
        if search_results:
            result['results'] = search_results
            result['verified'] = True
            result['message'] = f'Found {len(search_results)} potential match(es) via SearXNG.'
            best = search_results[0]
            result['source_url'] = best.get('url', '')
            result['source_title'] = best.get('title', '')
            result['source_snippet'] = best.get('snippet', '')
            result['confidence'] = _assess_confidence(narration_text, search_results)
            return result

    # No search provider available or no results found
    if not serper_key and not searxng_host:
        result['search_attempted'] = False
        result['message'] = 'No search provider configured. Narration requires manual verification.'
    else:
        result['message'] = 'Narration detected, but exact scholarly citation requires manual verification.'

    return result


def verify_narrations_batch(narrations: List[Dict[str, Any]],
                            config: dict = None) -> List[Dict[str, Any]]:
    """Verify a batch of narrations from the 'narrations' lens output.

    Args:
        narrations: List of narration dicts from the lens
        config: App config

    Returns:
        List of narration dicts with 'verification' field added
    """
    results = []
    for narr in narrations:
        if not isinstance(narr, dict):
            continue
        verification = verify_narration(
            narration_text=narr.get('narration_text', ''),
            speaker_attribution=narr.get('speaker_attribution', ''),
            source_claimed=narr.get('source_claimed', ''),
            config=config,
        )
        narr['verification'] = verification
        results.append(narr)
    return results


def _extract_search_keywords(narration_text: str, attribution: str = '',
                             source: str = '') -> str:
    """Extract meaningful search keywords from a narration.

    Combines narration text with attribution and source for better search results.
    """
    parts = []

    # Key phrases from narration (first 100 chars, avoiding common words)
    if narration_text:
        # Take first meaningful chunk
        clean = narration_text.strip()[:150]
        parts.append(clean)

    # Attribution (narrator name)
    if attribution:
        parts.append(attribution.strip())

    # Source reference
    if source:
        parts.append(source.strip())

    query = ' '.join(parts)
    if len(query) < 10:
        return ''

    # Cap query length for search APIs
    return query[:200]


def _search_serper(query: str, api_key: str) -> List[Dict[str, str]]:
    """Search trusted scholarly domains via Serper API."""
    # Build site-restricted query
    site_filter = ' OR '.join(f'site:{d}' for d in TRUSTED_DOMAINS)
    full_query = f'{query} ({site_filter})'

    try:
        resp = requests.post(
            'https://google.serper.dev/search',
            json={'q': full_query, 'num': 5},
            headers={'X-API-KEY': api_key},
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning(f"Serper verification search returned {resp.status_code}")
            return []

        data = resp.json()
        results = []
        for item in data.get('organic', [])[:5]:
            url = item.get('link', '')
            # Only include results from trusted domains
            if any(domain in url for domain in TRUSTED_DOMAINS):
                results.append({
                    'url': url,
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                })
        return results

    except Exception as e:
        logger.error(f"Serper verification search failed: {e}")
        return []


def _search_searxng(query: str, searxng_host: str) -> List[Dict[str, str]]:
    """Search trusted scholarly domains via SearXNG."""
    # Build site-restricted query
    site_filter = ' OR '.join(f'site:{d}' for d in TRUSTED_DOMAINS)
    full_query = f'{query} ({site_filter})'

    try:
        resp = requests.get(
            f'{searxng_host}/search',
            params={'q': full_query, 'format': 'json', 'engines': 'google'},
            timeout=15,
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        results = []
        for item in data.get('results', [])[:5]:
            url = item.get('url', '')
            if any(domain in url for domain in TRUSTED_DOMAINS):
                results.append({
                    'url': url,
                    'title': item.get('title', ''),
                    'snippet': item.get('content', ''),
                })
        return results

    except Exception as e:
        logger.error(f"SearXNG verification search failed: {e}")
        return []


def _assess_confidence(narration_text: str, results: List[Dict[str, str]]) -> str:
    """Assess confidence level of verification match.

    Returns: 'high', 'medium', or 'low'
    """
    if not results:
        return 'low'

    best = results[0]
    snippet = (best.get('snippet', '') + ' ' + best.get('title', '')).lower()
    narration_lower = narration_text.lower()

    # Extract key words from narration (3+ chars)
    narr_words = set(w for w in re.findall(r'\w{3,}', narration_lower))

    # Check overlap
    if not narr_words:
        return 'low'

    snippet_words = set(re.findall(r'\w{3,}', snippet))
    overlap = narr_words & snippet_words
    ratio = len(overlap) / len(narr_words) if narr_words else 0

    if ratio > 0.5:
        return 'high'
    elif ratio > 0.25:
        return 'medium'
    return 'low'
