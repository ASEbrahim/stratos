"""
Profile Generator — AI-powered category generation pipeline.

Extracts the generate-profile logic from main.py into a reusable module.
Pipeline: LLM Generate → Sanitize → Enrichment → Dedup → Merge Tiny → Career Opt-out
"""

import json
import logging
import re
import requests as req

from routes.helpers import strip_think_blocks, extract_json

logger = logging.getLogger("STRAT_OS")

# --- Constants ---

BANNED_ITEMS = {
    'news', 'article', 'report', 'update', 'analysis', 'review',
    'latest', 'breaking', 'trending', 'top', 'best', 'new',
    'the', 'and', 'for', 'with', 'from', 'about', 'market',
    'watch', 'alert', 'signal', 'feed', 'monitor', 'track',
    'general', 'other', 'misc', 'category', 'item', 'entry',
    'data', 'info', 'information', 'insights', 'trends',
}

ALIASES = {
    'schlumberger': 'slb', 'slb': 'slb',
    'halliburton': 'hal', 'hal': 'hal',
    'baker hughes': 'bkr', 'bkr': 'bkr',
    'exxonmobil': 'xom', 'xom': 'xom',
    'chevron': 'cvx', 'cvx': 'cvx',
    'weatherford': 'wft', 'wft': 'wft',
}

CAREER_OPTOUT_SIGNALS = [
    r'not looking for career', r'not looking for job', r'not seeking career',
    r'not seeking job', r'not interested in career', r'not interested in job',
    r'not interested in hiring', r'not looking for work', r'not looking for employment',
    r'deprioritize.*job', r'deprioritize.*career', r'deprioritize.*hiring',
    r'deprioritize.*recruitment',
]

MAX_CATEGORIES = 7
MIN_ITEMS_FOR_MERGE = 2


# --- Helpers ---

def _canonical(item_str: str) -> str:
    """Get canonical key for dedup (handles abbreviation aliases)."""
    low = item_str.strip().lower()
    return ALIASES.get(low, low)


def _sanitize_item(item_str: str) -> str:
    """Clean a single item string from LLM output."""
    result = str(item_str).strip()
    # Remove quotes, markdown, numbering
    result = result.strip('"\'`*_•–—')
    result = re.sub(r'^\d+[\.\)]\s*', '', result)
    # Remove parenthetical suffixes: "ADNOC (Abu Dhabi National)" → "ADNOC"
    result = re.sub(r'\s*\([^)]*\)\s*$', '', result).strip()
    # Remove trailing prepositions: "King Abdullah University of" → "King Abdullah University"
    result = re.sub(r'\s+(of|for|in|at|the|and|or)\s*$', '', result, flags=re.IGNORECASE)
    # Cap at 6 words
    words = result.split()
    if len(words) > 6:
        result = ' '.join(words[:6])
    # Must be 2+ chars
    if len(result) < 2:
        return ''
    return result


def _split_comma_items(text: str) -> list:
    """Split on commas that are NOT inside parentheses.

    'ANSYS (reactor simulation), MCNP (neutron transport)' → ['ANSYS (reactor simulation)', 'MCNP (neutron transport)']
    'Kuwait Petroleum Corporation (KPC, KNPC)' → ['Kuwait Petroleum Corporation (KPC, KNPC)']
    'Equate, KNPC, KPC' → ['Equate', 'KNPC', 'KPC']
    """
    parts = []
    depth = 0
    current = []
    for ch in text:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current).strip())
    return [p for p in parts if p]


def _detect_career_optout(role: str, context: str) -> bool:
    """Check if user has opted out of career-related content."""
    combined = f"{role} {context}".lower()
    return any(re.search(sig, combined) for sig in CAREER_OPTOUT_SIGNALS)


# --- Pipeline Stages ---

def sanitize_categories(categories: list) -> list:
    """Clean LLM-generated categories: sanitize items, remove banned, deduplicate within each."""
    for cat in categories:
        raw_items = cat.get('items', [])
        seen = set()
        clean = []
        for item in raw_items:
            # Split comma-separated items (LLM sometimes jams multiple into one string)
            # Respects parenthetical groups: "ANSYS (sim, model), MCNP" → 2 items, not 3
            fragments = _split_comma_items(str(item))
            for frag in fragments:
                sanitized = _sanitize_item(frag)
                canon = _canonical(sanitized)
                if sanitized and canon not in seen and sanitized.lower() not in BANNED_ITEMS:
                    seen.add(canon)
                    clean.append(sanitized)
        cat['items'] = clean
    return categories


def migrate_assets_to_tickers(categories: list, tickers: list) -> tuple:
    """Move investment assets from category items to ticker list."""
    ASSET_PATTERNS = re.compile(
        r'^(gold|silver|copper|platinum|palladium|crude\s*oil|natural\s*gas|bitcoin|ethereum|'
        r'btc|eth|xrp|bnb|solana|doge|litecoin|s&p\s*500|nasdaq|dow\s*jones|nikkei|ftse)$',
        re.IGNORECASE
    )
    ASSET_TO_TICKER = {
        'gold': 'GC=F', 'silver': 'SI=F', 'copper': 'HG=F',
        'platinum': 'PL=F', 'palladium': 'PA=F',
        'crude oil': 'CL=F', 'crudeoil': 'CL=F', 'natural gas': 'NG=F',
        'bitcoin': 'BTC-USD', 'btc': 'BTC-USD',
        'ethereum': 'ETH-USD', 'eth': 'ETH-USD',
        'xrp': 'XRP-USD', 'solana': 'SOL-USD',
        's&p 500': '^GSPC', 's&p500': '^GSPC',
        'nasdaq': '^IXIC', 'dow jones': '^DJI',
        'nikkei': '^N225', 'ftse': '^FTSE',
    }
    migrated = 0
    existing_tickers = set(t.upper() for t in tickers)

    for cat in categories:
        kept = []
        for item in cat.get('items', []):
            if ASSET_PATTERNS.match(item.strip()):
                ticker = ASSET_TO_TICKER.get(item.strip().lower())
                if ticker and ticker.upper() not in existing_tickers:
                    tickers.append(ticker)
                    existing_tickers.add(ticker.upper())
                    migrated += 1
            else:
                kept.append(item)
        cat['items'] = kept

    if migrated:
        logger.info(f"Asset→ticker migration: moved {migrated} items")
    return categories, tickers


def cap_categories(categories: list) -> list:
    """Enforce maximum category count, keeping largest."""
    if len(categories) <= MAX_CATEGORIES:
        return categories
    categories.sort(key=lambda c: len(c.get('items', [])), reverse=True)
    keepers = categories[:MAX_CATEGORIES]
    logger.info(f"Capped categories: {len(categories)} → {len(keepers)}")
    return keepers


def enrich_categories(categories: list, role: str, location: str, context: str,
                      ollama_host: str, model: str) -> list:
    """Second LLM pass to discover missing entities in each category."""
    try:
        cats_summary = []
        for cat in categories:
            items_str = ', '.join(cat.get('items', [])[:8])
            cats_summary.append(f"  {cat.get('label', '?')}: [{items_str}]")
        cats_text = '\n'.join(cats_summary)

        location_hint = ""
        if location:
            loc_lower = location.lower()
            if any(k in loc_lower for k in ['kuwait', 'gcc', 'gulf', 'saudi', 'uae', 'qatar', 'bahrain', 'oman']):
                location_hint = f"\nIMPORTANT: The user is based in {location}. For banking/finance categories, ONLY add banks and financial institutions that operate in {location} or the GCC region. Do NOT add international banks with no presence there (e.g. no European or American retail banks). For employer categories, focus on companies with offices or operations in {location}."

        prompt = f"""Role: {role}
Location: {location or 'Not specified'}
Context: {context or 'Not specified'}
{location_hint}

Current tracking categories:
{cats_text}

TASK: For EACH category, identify 3-6 MISSING entities that are critically relevant but not yet listed. Focus on:
- For company/employer categories: key service providers, contractors, partners, and competitors.
- For technology categories: specific software platforms, methodologies, and tools.
- For regulatory categories: specific regulatory bodies, standards organizations.

Be SPECIFIC — use exact company names, not generic terms.

Respond with ONLY valid JSON — no markdown, no backticks:
{{"additions": [{{"category_label": "exact label from above", "new_items": ["item1", "item2", "item3"]}}]}}"""

        response = req.post(
            f"{ollama_host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a professional industry knowledge assistant. Return ONLY valid JSON with no explanation, no reasoning."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                # Don't set think:false — Qwen3 leaks reasoning into content.
                # num_predict must cover thinking tokens + JSON output.
                "options": {"temperature": 0.3, "num_predict": 8000, "num_ctx": 12288}
            },
            timeout=180
        )

        if response.status_code == 200:
            raw = response.json().get("message", {}).get("content", "")
            raw = strip_think_blocks(raw)
            # Don't strip reasoning — this is a JSON response. Use extract_json instead.
            json_str = extract_json(raw)
            match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if match:
                data = json.loads(match.group())
                additions = data.get("additions", [])

                label_map = {cat.get('label', '').lower(): cat for cat in categories}
                all_existing = set()
                for cat in categories:
                    for item in cat.get('items', []):
                        all_existing.add(_canonical(item))

                added = 0
                for addition in additions:
                    label = addition.get("category_label", "").lower()
                    new_items = addition.get("new_items", [])
                    if label in label_map and isinstance(new_items, list):
                        cat = label_map[label]
                        for item in new_items:
                            # Split comma-separated items from enrichment LLM pass
                            fragments = _split_comma_items(str(item))
                            for frag in fragments:
                                clean = _sanitize_item(frag)
                                canon = _canonical(clean)
                                if clean and canon not in all_existing and clean.lower() not in BANNED_ITEMS:
                                    cat['items'].append(clean)
                                    all_existing.add(canon)
                                    added += 1

                if added:
                    logger.info(f"Entity enrichment: added {added} items across categories")

    except Exception as e:
        logger.debug(f"Entity enrichment pass failed (non-critical): {e}")

    return categories


def dedup_across_categories(categories: list) -> list:
    """Remove items that appear in multiple categories (keep in largest)."""
    categories.sort(key=lambda c: len(c.get('items', [])), reverse=True)
    global_seen = {}
    removed = 0

    for i, cat in enumerate(categories):
        cleaned = []
        for item in cat.get('items', []):
            key = _canonical(item)
            if key not in global_seen:
                global_seen[key] = i
                cleaned.append(item)
            else:
                removed += 1
        cat['items'] = cleaned

    if removed:
        logger.info(f"Cross-category dedup: removed {removed} duplicate items")
    return categories


def merge_tiny_categories(categories: list) -> list:
    """Merge categories with too few items into the most similar larger one."""
    large = [c for c in categories if len(c.get('items', [])) >= MIN_ITEMS_FOR_MERGE]
    tiny = [c for c in categories if len(c.get('items', [])) < MIN_ITEMS_FOR_MERGE]

    for t in tiny:
        if not large:
            large.append(t)
            continue
        t_type = t.get('scorer_type', 'auto')
        best = None
        for lc in large:
            if lc.get('scorer_type', 'auto') == t_type:
                if best is None or len(lc.get('items', [])) < len(best.get('items', [])):
                    best = lc
        if best is None:
            best = min(large, key=lambda c: len(c.get('items', [])))

        existing = set(_canonical(i) for i in best.get('items', []))
        for item in t.get('items', []):
            if _canonical(item) not in existing:
                best['items'].append(item)
                existing.add(_canonical(item))
        logger.info(f"Merged tiny category '{t.get('label', '?')}' ({len(t.get('items', []))} items) into '{best.get('label', '?')}'")

    return large


def apply_career_optout(categories: list, role: str, context: str) -> list:
    """If user opted out of career content, flip career scorer_type to auto."""
    if not _detect_career_optout(role, context):
        return categories

    for cat in categories:
        if cat.get('scorer_type') == 'career':
            cat['scorer_type'] = 'auto'
            logger.info(f"Career opt-out: flipped '{cat.get('label', '?')}' from career → auto")

    return categories


# --- Main Pipeline ---

def run_pipeline(categories: list, tickers: list, role: str, location: str,
                 context: str, ollama_host: str, model: str) -> tuple:
    """
    Run the full post-generation pipeline on AI-generated categories.
    
    Returns: (processed_categories, updated_tickers)
    """
    # 1. Sanitize
    categories = sanitize_categories(categories)

    # 2. Migrate assets to tickers
    categories, tickers = migrate_assets_to_tickers(categories, tickers)

    # 3. Validate tickers against Yahoo Finance
    tickers = validate_tickers(tickers)

    # 4. Cap category count
    categories = cap_categories(categories)

    # 5. Enrich with second LLM pass (BEFORE dedup so categories get filled first)
    categories = enrich_categories(categories, role, location, context, ollama_host, model)

    # 6. Cross-category deduplication
    categories = dedup_across_categories(categories)

    # 7. Merge tiny categories
    categories = merge_tiny_categories(categories)

    # 8. Career opt-out
    categories = apply_career_optout(categories, role, context)

    return categories, tickers


def validate_tickers(tickers: list) -> list:
    """Validate tickers against Yahoo Finance. Remove any that don't exist.
    
    Uses a quick yfinance price check — if no data comes back, the ticker
    is hallucinated or delisted and gets dropped.
    """
    if not tickers:
        return tickers
    
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed, skipping ticker validation")
        return tickers
    
    valid = []
    for t in tickers:
        t = str(t).strip()
        if not t:
            continue
        try:
            ticker = yf.Ticker(t)
            # Fast check: just grab 1 day of data
            hist = ticker.history(period="1d")
            if hist is not None and not hist.empty:
                valid.append(t)
            else:
                logger.warning(f"Ticker validation: {t} — no data (hallucinated or delisted), dropping")
        except Exception as e:
            logger.warning(f"Ticker validation: {t} — error ({e}), dropping")
    
    if len(valid) < len(tickers):
        dropped = set(tickers) - set(valid)
        logger.info(f"Ticker validation: kept {len(valid)}/{len(tickers)}, dropped {dropped}")
    
    return valid
