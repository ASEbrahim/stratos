#!/usr/bin/env python3
"""
V2 Training Pipeline — Stage 3b: Expand Dataset with New Articles
==================================================================
Selects new articles from the live DB (not in existing V2 data),
scores them via Batch API across all 30 profiles, and merges with
existing V2 scores.

Usage:
    python3 data/v2_pipeline/stage3_expand.py              # Select + submit batch
    python3 data/v2_pipeline/stage3_expand.py --poll        # Poll + download results
    python3 data/v2_pipeline/stage3_expand.py --merge       # Merge into V2 scores
"""

import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

os.chdir(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load API key
if not os.environ.get("ANTHROPIC_API_KEY"):
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

from data.v2_pipeline.profiles_v2 import ALL_PROFILES
from data.v2_pipeline.stage3_score import (
    SCORER_SYSTEM_PROMPT, build_user_message, estimate_cost,
    submit_batch, poll_batch, download_results, process_results,
    parse_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("V2_EXPAND")

OUTPUT_DIR = Path(__file__).parent
NEW_ARTICLES_FILE = OUTPUT_DIR / "articles_v2_new.json"
NEW_BATCH_INPUT = OUTPUT_DIR / "batch_input_expand.jsonl"
NEW_BATCH_OUTPUT = OUTPUT_DIR / "batch_output_expand.jsonl"
NEW_SCORES_FILE = OUTPUT_DIR / "scores_v2_new.json"
MERGED_SCORES_FILE = OUTPUT_DIR / "scores_v2.json"  # Overwrites original
MERGED_ARTICLES_FILE = OUTPUT_DIR / "articles_v2.json"  # Overwrites original
BATCH_ID_FILE = OUTPUT_DIR / "batch_id_expand.txt"

# How many new articles to select
TARGET_NEW_ARTICLES = 130
BUDGET_CEILING = 10.0

# DB path
DB_PATH = Path(__file__).parent.parent.parent / "strat_os.db"


def select_new_articles(db_path: str, existing_articles_file: str, target: int = 200) -> list:
    """Select diverse new articles from the live DB not in existing V2 data."""
    # Load existing article URLs
    existing_urls = set()
    with open(existing_articles_file) as f:
        for a in json.load(f):
            existing_urls.add(a.get('url', ''))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get all unique articles with summaries, not in existing data
    c.execute("""
        SELECT id, title, url, summary, source, root, category, score, score_reason
        FROM news_items
        WHERE score IS NOT NULL AND score > 0
          AND summary IS NOT NULL AND summary != ''
          AND LENGTH(summary) > 50
        GROUP BY url
        ORDER BY fetched_at DESC
    """)

    candidates = []
    for row in c.fetchall():
        row = dict(row)
        if row['url'] in existing_urls:
            continue
        candidates.append(row)

    conn.close()
    logger.info(f"Found {len(candidates)} candidate articles (not in existing V2 data)")

    # Stratified sampling by score band
    bands = {
        'noise': [], 'tangential': [], 'moderate': [], 'high': [], 'critical': []
    }
    for a in candidates:
        s = a['score']
        if s >= 8.5: bands['critical'].append(a)
        elif s >= 7.0: bands['high'].append(a)
        elif s >= 4.5: bands['moderate'].append(a)
        elif s >= 2.5: bands['tangential'].append(a)
        else: bands['noise'].append(a)

    # Target distribution: balanced to fill gaps
    # V2 had too much noise (88.8%) — aim for more balanced
    targets = {
        'noise': int(target * 0.30),       # 60
        'tangential': int(target * 0.25),  # 50
        'moderate': int(target * 0.15),    # 30
        'high': int(target * 0.15),        # 30
        'critical': int(target * 0.15),    # 30
    }

    selected = []
    for band, band_target in targets.items():
        pool = bands[band]
        random.shuffle(pool)
        n = min(band_target, len(pool))
        selected.extend(pool[:n])
        logger.info(f"  {band}: selected {n}/{len(pool)} (target {band_target})")

    random.shuffle(selected)
    logger.info(f"Total selected: {len(selected)} articles")
    return selected


def create_batch_requests(articles: list, profiles: list) -> list:
    """Create batch API request objects for all article×profile pairs."""
    requests = []
    for article in articles:
        article_id = article.get('id', hashlib.md5(article['url'].encode()).hexdigest()[:12])
        for profile in profiles:
            custom_id = f"{article_id}__{profile['id']}"
            user_msg = build_user_message(profile, article)
            request = {
                "custom_id": custom_id,
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 512,
                    "messages": [
                        {"role": "user", "content": f"{SCORER_SYSTEM_PROMPT}\n\n{user_msg}"}
                    ]
                }
            }
            requests.append(request)
    return requests


def merge_scores():
    """Merge new scores into existing V2 scores file."""
    # Load existing
    with open(MERGED_SCORES_FILE) as f:
        existing = json.load(f)
    logger.info(f"Existing scores: {len(existing)}")

    # Load new
    with open(NEW_SCORES_FILE) as f:
        new_scores = json.load(f)
    logger.info(f"New scores: {len(new_scores)}")

    # Deduplicate by custom_id
    seen = set()
    merged = []
    for s in existing:
        key = s.get('custom_id', f"{s['article_id']}__{s['profile_id']}")
        if key not in seen:
            seen.add(key)
            merged.append(s)
    for s in new_scores:
        key = s.get('custom_id', f"{s['article_id']}__{s['profile_id']}")
        if key not in seen:
            seen.add(key)
            merged.append(s)

    # Save merged
    with open(MERGED_SCORES_FILE, 'w') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info(f"Merged total: {len(merged)} scores")

    # Merge articles too
    with open(MERGED_ARTICLES_FILE) as f:
        existing_articles = json.load(f)
    with open(NEW_ARTICLES_FILE) as f:
        new_articles = json.load(f)

    existing_urls = {a['url'] for a in existing_articles}
    for a in new_articles:
        if a['url'] not in existing_urls:
            existing_articles.append(a)

    with open(MERGED_ARTICLES_FILE, 'w') as f:
        json.dump(existing_articles, f, indent=2, ensure_ascii=False)
    logger.info(f"Merged articles: {len(existing_articles)}")

    return len(merged)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", action="store_true", help="Poll and download batch results")
    parser.add_argument("--merge", action="store_true", help="Merge new scores into V2 data")
    parser.add_argument("--target", type=int, default=TARGET_NEW_ARTICLES, help="Target new articles")
    args = parser.parse_args()

    if args.merge:
        merge_scores()
        return

    if args.poll:
        if not BATCH_ID_FILE.exists():
            logger.error("No batch ID file found. Submit batch first.")
            sys.exit(1)
        batch_id = BATCH_ID_FILE.read_text().strip()
        logger.info(f"Polling batch {batch_id}...")
        status = poll_batch(batch_id)
        logger.info(f"Batch completed: {status}")

        logger.info("Downloading results...")
        results = download_results(batch_id, str(NEW_BATCH_OUTPUT))

        logger.info("Processing results...")
        scored, malformed, empty_think, sparse_think, score_dist = process_results(results)

        with open(NEW_SCORES_FILE, 'w') as f:
            json.dump(scored, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(scored)} new scores to {NEW_SCORES_FILE}")
        logger.info(f"Malformed: {malformed}, Empty think: {empty_think}, Sparse: {sparse_think}")
        return

    # ── Select new articles ──
    print("=" * 80)
    print("V2 EXPAND — SELECT NEW ARTICLES FOR OPUS SCORING")
    print("=" * 80)

    articles = select_new_articles(str(DB_PATH), str(MERGED_ARTICLES_FILE), target=args.target)

    if not articles:
        logger.error("No new articles found!")
        sys.exit(1)

    # Save new articles
    with open(NEW_ARTICLES_FILE, 'w') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(articles)} new articles to {NEW_ARTICLES_FILE}")

    # Create batch requests
    n_profiles = len(ALL_PROFILES)
    n_requests = len(articles) * n_profiles
    est_cost = estimate_cost(n_requests)

    print(f"\nNew articles: {len(articles)}")
    print(f"Profiles: {n_profiles}")
    print(f"Total requests: {n_requests}")
    print(f"Estimated cost: ${est_cost:.2f}")

    if est_cost > BUDGET_CEILING:
        print(f"!! OVER BUDGET CEILING of ${BUDGET_CEILING}!")
        sys.exit(1)

    requests = create_batch_requests(articles, ALL_PROFILES)
    with open(NEW_BATCH_INPUT, 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
    logger.info(f"Wrote {len(requests)} requests to {NEW_BATCH_INPUT}")

    # Submit batch
    logger.info("Submitting batch...")
    batch_id = submit_batch(str(NEW_BATCH_INPUT))
    with open(BATCH_ID_FILE, 'w') as f:
        f.write(batch_id)
    logger.info(f"Batch submitted: {batch_id}")
    logger.info("Run with --poll to check status and download results")

    # Auto-poll
    logger.info("Auto-polling for completion...")
    status = poll_batch(batch_id)
    logger.info(f"Batch completed: {status}")

    # Download results
    logger.info("Downloading results...")
    results = download_results(batch_id, str(NEW_BATCH_OUTPUT))

    # Process
    logger.info("Processing results...")
    scored, malformed, empty_think, sparse_think, score_dist = process_results(results)
    with open(NEW_SCORES_FILE, 'w') as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(scored)} new scores")
    logger.info(f"Malformed: {malformed}, Empty think: {empty_think}")

    # Auto-merge
    logger.info("Merging into V2 data...")
    total = merge_scores()
    logger.info(f"Total merged scores: {total}")
    logger.info("Done! Run stage4_prepare.py next.")


if __name__ == "__main__":
    main()
