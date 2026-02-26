#!/usr/bin/env python3
"""
V2 Training Pipeline — Stage 2: Article Collection
====================================================
For each of 30 profiles:
  1. Generate 15-20 search queries from profile fields
  2. Run through DDG .news() AND Serper (parallel sources)
  3. Collect into unified article pool
  4. Deduplicate, strip empty, and report

Target: ~700 unique articles after dedup.
"""

import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Setup
os.chdir(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.v2_pipeline.profiles_v2 import ALL_PROFILES

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("V2_COLLECT")
logger.setLevel(logging.INFO)

for lib in ['urllib3', 'requests', 'duckduckgo_search', 'yfinance', 'peewee']:
    logging.getLogger(lib).setLevel(logging.ERROR)

# Import search tools
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    logger.warning("duckduckgo_search not available")

try:
    from fetchers.serper_search import SerperSearchClient, get_serper_client
    HAS_SERPER = True
except ImportError:
    HAS_SERPER = False
    logger.warning("Serper not available")

# DDG region from profile location
try:
    from processors.scorer_base import location_to_lang
except ImportError:
    def location_to_lang(loc):
        return ({'latin'}, 'wt-wt', 'English')

# ─── Constants ────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent
ARTICLES_FILE = OUTPUT_DIR / "articles_v2.json"
TARGET_ARTICLES = 700
DDG_PAUSE = 2.0        # Seconds between DDG queries (avoid rate limit)
PROFILE_PAUSE = 10.0   # Seconds between profiles
MAX_DDG_PER_PROFILE = 12  # Max DDG queries per profile (rate limit budget)
MAX_SERPER_PER_PROFILE = 8  # Max Serper queries per profile


def generate_queries(profile: dict) -> list:
    """Generate 15-20 search queries from a profile's fields."""
    role = profile['role']
    location = profile['location']
    context = profile.get('context', '')
    interests = profile.get('interests', [])
    companies = profile.get('tracked_companies', '')
    institutions = profile.get('tracked_institutions', '')
    industries = profile.get('tracked_industries', '')

    queries = []

    # 1. Role-based queries (3-4)
    queries.append(f'"{role}" news 2026')
    loc_short = location.split(',')[0].strip()
    queries.append(f'{role} {loc_short} 2026')
    if 'student' in role.lower() or 'graduate' in role.lower():
        queries.append(f'{role} internship OR entry-level 2026')
    else:
        queries.append(f'{role} industry trends 2026')

    # 2. Interest-based queries (5-6)
    for interest in interests[:5]:
        queries.append(f'"{interest}" 2026')

    # 3. Company-based queries (3-4)
    if companies and companies != 'None' and companies != 'None specific':
        company_list = [c.strip() for c in companies.split(',') if c.strip()]
        for company in company_list[:4]:
            queries.append(f'"{company}" news 2026')

    # 4. Industry + location queries (2-3)
    if industries and industries != 'None' and industries != 'varied':
        ind_list = [i.strip() for i in industries.split(',') if i.strip()]
        for ind in ind_list[:2]:
            queries.append(f'{ind} {loc_short} 2026')

    # 5. Context-derived queries (2-3)
    if context:
        # Extract key noun phrases from context
        words = context.split()
        if len(words) > 5:
            # Take a meaningful chunk
            chunk = ' '.join(words[:6])
            queries.append(f'{chunk} 2026')

    # 6. Institution queries (1-2)
    if institutions and institutions != 'None':
        inst_list = [i.strip() for i in institutions.split(',') if i.strip()]
        for inst in inst_list[:2]:
            queries.append(f'"{inst}" 2026')

    # Deduplicate and cap at 20
    seen = set()
    unique = []
    for q in queries:
        q_lower = q.lower().strip()
        if q_lower not in seen:
            seen.add(q_lower)
            unique.append(q)
    return unique[:20]


def search_ddg(query: str, region: str = 'wt-wt', max_results: int = 5) -> list:
    """Search DDG .news() endpoint."""
    if not HAS_DDGS:
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(
                query, region=region, max_results=max_results, timelimit='w'
            ))
            items = []
            for r in results:
                title = r.get('title', '').strip()
                url = r.get('url', '').strip()
                body = r.get('body', '').strip()
                source = r.get('source', 'DDG News')
                if title and url:
                    items.append({
                        'title': title,
                        'url': url,
                        'summary': body[:500] if body else '',
                        'source': source,
                        'search_engine': 'ddg',
                    })
            return items
    except Exception as e:
        if 'Ratelimit' in str(e):
            logger.warning(f"DDG rate limited on: {query[:40]}...")
        else:
            logger.warning(f"DDG error on '{query[:40]}': {e}")
        return []


def search_serper(query: str, config: dict, max_results: int = 5) -> list:
    """Search Serper (Google News)."""
    if not HAS_SERPER:
        return []
    try:
        client = get_serper_client(config)
        if not client:
            return []
        results = client.search_news(query, num_results=max_results)
        items = []
        for r in results:
            title = r.get('title', '').strip()
            url = r.get('link', '').strip()
            snippet = r.get('snippet', '').strip()
            source = r.get('source', 'Serper')
            if title and url:
                items.append({
                    'title': title,
                    'url': url,
                    'summary': snippet[:500] if snippet else '',
                    'source': source,
                    'search_engine': 'serper',
                })
        return items
    except Exception as e:
        logger.warning(f"Serper error on '{query[:40]}': {e}")
        return []


def url_hash(url: str) -> str:
    """Generate hash for URL dedup."""
    return hashlib.md5(url.strip().lower().encode()).hexdigest()[:12]


def collect_for_profile(profile: dict, serper_config: dict) -> list:
    """Collect articles for a single profile."""
    profile_id = profile['id']
    location = profile['location']
    _, ddg_region, _ = location_to_lang(location)

    queries = generate_queries(profile)
    logger.info(f"  Generated {len(queries)} queries")

    all_items = []
    ddg_count = 0
    serper_count = 0

    # Split queries between DDG and Serper
    ddg_queries = queries[:MAX_DDG_PER_PROFILE]
    serper_queries = queries[:MAX_SERPER_PER_PROFILE]

    # DDG queries (sequential to avoid rate limit)
    for q in ddg_queries:
        items = search_ddg(q, region=ddg_region, max_results=5)
        for item in items:
            item['profile_id'] = profile_id
            item['query'] = q
        all_items.extend(items)
        ddg_count += len(items)
        time.sleep(DDG_PAUSE)

    # Serper queries (can be slightly more parallel)
    for q in serper_queries:
        items = search_serper(q, serper_config, max_results=5)
        for item in items:
            item['profile_id'] = profile_id
            item['query'] = q
        all_items.extend(items)
        serper_count += len(items)

    logger.info(f"  Collected: {ddg_count} DDG + {serper_count} Serper = {len(all_items)} total")
    return all_items


def main():
    import yaml

    print("=" * 80)
    print("V2 TRAINING PIPELINE — STAGE 2: ARTICLE COLLECTION")
    print("=" * 80)
    print(f"Profiles: {len(ALL_PROFILES)}")
    print(f"Target: ~{TARGET_ARTICLES} unique articles after dedup")
    print()

    # Load serper config
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    serper_config = {}
    if config_path.exists():
        with open(config_path) as f:
            serper_config = yaml.safe_load(f) or {}

    all_articles = []
    profile_stats = {}

    for i, profile in enumerate(ALL_PROFILES, 1):
        print(f"\n[{i}/{len(ALL_PROFILES)}] {profile['id']} — {profile['role'][:50]}")
        logger.info(f"[{i}/{len(ALL_PROFILES)}] Collecting for: {profile['id']}")

        items = collect_for_profile(profile, serper_config)
        all_articles.extend(items)
        profile_stats[profile['id']] = len(items)

        # Progress
        print(f"  -> {len(items)} raw articles")

        # Pause between profiles
        if i < len(ALL_PROFILES):
            time.sleep(PROFILE_PAUSE)

    # ── Deduplication ──
    print(f"\n{'='*80}")
    print("DEDUPLICATION")
    print(f"{'='*80}")
    print(f"Total raw articles: {len(all_articles)}")

    seen_urls = set()
    unique_articles = []
    for item in all_articles:
        h = url_hash(item['url'])
        if h not in seen_urls:
            seen_urls.add(h)
            # Strip articles with no title or no snippet
            if item.get('title') and (item.get('summary') or item.get('url')):
                unique_articles.append(item)

    print(f"After URL dedup: {len(unique_articles)}")

    # Title dedup (catch near-duplicates with same title from different URLs)
    seen_titles = set()
    title_deduped = []
    for item in unique_articles:
        title_key = item['title'].lower().strip()[:80]
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            title_deduped.append(item)
    unique_articles = title_deduped
    print(f"After title dedup: {len(unique_articles)}")

    # ── Stratified sampling if over target ──
    if len(unique_articles) > TARGET_ARTICLES * 1.2:
        print(f"\nOver target — stratified sampling to {TARGET_ARTICLES}")
        # Group by profile_id
        by_profile = defaultdict(list)
        for item in unique_articles:
            by_profile[item.get('profile_id', 'unknown')].append(item)

        # Proportional sampling
        total = len(unique_articles)
        sampled = []
        for pid, items in by_profile.items():
            proportion = len(items) / total
            n_sample = max(5, int(TARGET_ARTICLES * proportion))
            if len(items) <= n_sample:
                sampled.extend(items)
            else:
                sampled.extend(random.sample(items, n_sample))

        # If still over, randomly trim
        if len(sampled) > TARGET_ARTICLES:
            sampled = random.sample(sampled, TARGET_ARTICLES)
        unique_articles = sampled
        print(f"After sampling: {len(unique_articles)}")

    # ── Source distribution ──
    source_dist = Counter(item.get('search_engine', 'unknown') for item in unique_articles)

    # ── Per-profile distribution ──
    per_profile = Counter(item.get('profile_id', 'unknown') for item in unique_articles)

    # ── Save ──
    # Assign stable IDs
    for item in unique_articles:
        item['id'] = url_hash(item['url'])

    with open(ARTICLES_FILE, 'w') as f:
        json.dump(unique_articles, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(unique_articles)} articles to {ARTICLES_FILE}")

    # ── Report ──
    print(f"\n{'='*80}")
    print("STAGE 2 REPORT")
    print(f"{'='*80}")
    print(f"Total raw articles fetched: {len(all_articles)}")
    print(f"Post-dedup count: {len(unique_articles)}")
    print(f"Source distribution: {dict(source_dist)}")
    print(f"\nArticles per profile (top 10):")
    for pid, count in per_profile.most_common(10):
        print(f"  {pid}: {count}")
    print(f"  ...")
    print(f"  Min per profile: {min(per_profile.values()) if per_profile else 0}")
    print(f"  Max per profile: {max(per_profile.values()) if per_profile else 0}")
    print(f"  Mean per profile: {sum(per_profile.values()) / len(per_profile) if per_profile else 0:.1f}")

    # Check thresholds
    if len(unique_articles) < 400:
        print(f"\n!! WARNING: Only {len(unique_articles)} articles — below 400 minimum!")
        print("!! Consider increasing queries per profile and re-fetching")
    elif len(unique_articles) < 600:
        print(f"\n!! NOTE: {len(unique_articles)} articles — slightly below 700 target")
    else:
        print(f"\n OK: {len(unique_articles)} articles — meets target range")

    return unique_articles


if __name__ == "__main__":
    main()
