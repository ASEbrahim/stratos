#!/usr/bin/env python3
"""
V2.2 Article Expansion Pipeline
================================
Phase 1: Extract articles from DB, deduplicate, exclude holdout + existing
Phase 2: Score with Gemini 3 Flash (all 30 Opus profiles)
Phase 3: Apply isotonic calibration, save calibration model
Phase 5: Merge scores, prepare training_v2.2.jsonl

Usage:
  python3 v22_expansion.py --phase 1          # Extract articles from DB
  python3 v22_expansion.py --phase 2          # Score with Gemini (run daily, 9.5K/day)
  python3 v22_expansion.py --phase 3          # Calibrate scores (isotonic regression)
  python3 v22_expansion.py --phase 5          # Prepare training_v2.2.jsonl

  Phase 2 auto-resumes from checkpoint. Run once per day for ~5 days.
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import random
import re
import sqlite3
import sys
import time
import functools
from collections import Counter, defaultdict
from pathlib import Path

os.chdir(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("V22_EXPAND")
# Suppress noisy Google SDK logs
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ══════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════
BASE = Path(__file__).parent
DB_PATH = Path("strat_os.db")
HOLDOUT_FILE = BASE / "holdout_articles.json"
EXISTING_ARTICLES_FILE = BASE / "articles_v2.json"
EXISTING_SCORES_FILE = BASE / "scores_v2.json"
GEMINI_VALIDATION_RESULTS = BASE / "gemini_validation_results.json"

# New files
EXPANSION_ARTICLES_FILE = BASE / "expansion_articles.json"
EXPANSION_SCORES_FILE = BASE / "expansion_scores.json"
CALIBRATION_MODEL_FILE = BASE / "calibration_model.pkl"
CALIBRATION_TABLE_FILE = BASE / "calibration_table.json"
MERGED_SCORES_FILE = BASE / "scores_v2.json"  # We update in-place
TRAINING_V22_FILE = BASE / "training_v2.2.jsonl"
EVAL_V22_FILE = BASE / "eval_v2.2.jsonl"

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
TARGET_ARTICLES = 2000
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_TEMPERATURE = 0.1
GEMINI_MAX_TOKENS = 1024
RANDOM_SEED = 42

# For EXISTING scores (scores_v2.json): only use 30 Opus-scored profiles
# For NEW Gemini scores: use all 49 profiles (Gemini quality is uniform)
# real_estate_dubai has 0 existing scores but will get Gemini scores
AGENT_PROFILES = {
    'marketing_director_london', 'auditor_toronto', 'auto_engineer_stuttgart',
    'journalist_dc', 'ecommerce_istanbul', 'env_consultant_amsterdam',
    'hr_director_sydney', 'dentist_riyadh', 'airline_pilot_dubai',
    'social_media_la', 'electrician_manchester', 'construction_pm_doha',
    'investment_analyst_hk', 'public_health_geneva', 'trucking_owner_atlanta',
    'veterinarian_stockholm', 'sysadmin_osaka', 'math_teacher_texas',
    'backend_engineer_singapore',
}
# Phase 2 scoring: 25 non-agent profiles (10K RPD limit = ~5 days for 50K pairs)
# Dropped: bonsai_competitor_kyoto, filmmaker_mumbai, retired_diplomat_vienna,
#           chef_mexico_city, kpop_marketer_seoul, agribiz_sao_paulo (niche/unlikely markets)
# All 19 agent profiles also skipped (no existing Opus data to merge with)
SKIP_PROFILES_SCORING = AGENT_PROFILES | {
    'bonsai_competitor_kyoto', 'filmmaker_mumbai', 'retired_diplomat_vienna',
    'chef_mexico_city', 'kpop_marketer_seoul', 'agribiz_sao_paulo',
}
# Phase 5 existing data: exclude agent-scored profiles (quality concerns)
SKIP_PROFILES_EXISTING = AGENT_PROFILES

# Training config
MAX_TRAIN = 19000
NOISE_CAP = 9000  # Cap noise band at 9K (47% of 19K)

# Score bands and loss weights (same as V2)
SCORE_BANDS = {
    "noise": (0.0, 2.5),
    "tangential": (2.5, 4.5),
    "moderate": (4.5, 6.5),
    "high": (6.5, 8.5),
    "critical": (8.5, 10.0),
}
LOSS_WEIGHTS = {"noise": 0.5, "tangential": 1.0, "moderate": 1.5, "high": 2.0, "critical": 3.0}

FORMAT_SUFFIX = (
    "\n\n---\nIMPORTANT: Your response MUST start with the score on the very first line "
    "in this exact format:\nSCORE: X.X | REASON: brief explanation\n"
    "Do NOT write any text before the SCORE line."
)

RETRY_PROMPT = "Rate this article's relevance to the user 0.0-10.0. Respond with ONLY: SCORE: X.X"


def score_to_band(score: float) -> str:
    if score >= 8.5: return "critical"
    if score >= 6.5: return "high"
    if score >= 4.5: return "moderate"
    if score >= 2.5: return "tangential"
    return "noise"


def _is_student(role: str) -> bool:
    role_lower = role.lower()
    return any(kw in role_lower for kw in [
        'student', 'freshman', 'sophomore', 'junior', 'senior year',
        'undergraduate', 'graduate student', 'phd candidate',
        'fresh graduate', 'undeclared'
    ])


def parse_score(text: str):
    """Extract score from Gemini response."""
    if not text:
        return None, None
    # SCORE: X.X | REASON: ...
    m = re.search(r'SCORE:\s*([\d.]+)(?:\s*\|\s*REASON:\s*(.+))?', text, re.IGNORECASE)
    if m:
        return float(m.group(1)), (m.group(2) or '').strip()
    # SCORE: X.X (no reason)
    m = re.search(r'SCORE:\s*([\d.]+)', text)
    if m:
        return float(m.group(1)), ''
    # **Score:** format
    m = re.search(r'\*?\*?Score\*?\*?[:\s]+([\d.]+)', text, re.IGNORECASE)
    if m:
        return float(m.group(1)), ''
    # Bare number
    m = re.search(r'^\s*([\d]+\.[\d]+)\s*$', text, re.MULTILINE)
    if m:
        return float(m.group(1)), ''
    return None, None


# ══════════════════════════════════════════════════════════════
# PHASE 1: EXTRACT ARTICLES FROM DB
# ══════════════════════════════════════════════════════════════
def phase1_extract():
    """Extract articles from DB, deduplicate, exclude holdout + existing, stratified sample."""
    logger.info("PHASE 1: Extract articles from DB")

    # Load exclusion sets
    holdout_ids = set()
    if HOLDOUT_FILE.exists():
        holdout_data = json.load(open(HOLDOUT_FILE))
        holdout_ids = {a['id'] if isinstance(a, dict) else a for a in holdout_data}
        logger.info(f"Holdout articles to exclude: {len(holdout_ids)}")

    existing_urls = set()
    existing_ids = set()
    if EXISTING_ARTICLES_FILE.exists():
        existing = json.load(open(EXISTING_ARTICLES_FILE))
        existing_urls = {a['url'] for a in existing if a.get('url')}
        existing_ids = {a['id'] for a in existing if a.get('id')}
        logger.info(f"Existing articles to exclude: {len(existing)} (by URL: {len(existing_urls)}, by ID: {len(existing_ids)})")

    # Query DB
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, title, url, summary, source, category, fetched_at
        FROM news_items
        WHERE summary IS NOT NULL AND summary != ''
        ORDER BY fetched_at DESC
    """).fetchall()
    conn.close()
    logger.info(f"Total DB articles with summaries: {len(rows)}")

    # Deduplicate by URL + exclude holdout + exclude existing
    seen_urls = set()
    pool = []
    excluded_holdout = 0
    excluded_existing = 0
    excluded_dup = 0

    for row in rows:
        art_id = row['id']
        url = row['url']

        # Generate consistent ID if missing
        if not art_id:
            art_id = hashlib.md5(url.encode()).hexdigest()[:12]

        if art_id in holdout_ids:
            excluded_holdout += 1
            continue
        if art_id in existing_ids or url in existing_urls:
            excluded_existing += 1
            continue
        if url in seen_urls:
            excluded_dup += 1
            continue
        seen_urls.add(url)

        # Skip very short summaries
        summary = row['summary'] or ''
        if len(summary) < 50:
            continue

        pool.append({
            'id': art_id,
            'title': row['title'] or '',
            'url': url,
            'summary': summary,
            'source': row['source'] or '',
            'category': row['category'] or 'general',
            'fetched_at': row['fetched_at'] or '',
        })

    logger.info(f"After filtering: {len(pool)} articles")
    logger.info(f"  Excluded holdout: {excluded_holdout}")
    logger.info(f"  Excluded existing: {excluded_existing}")
    logger.info(f"  Excluded duplicate URLs: {excluded_dup}")

    # Stratified sampling by category
    random.seed(RANDOM_SEED)
    by_category = defaultdict(list)
    for art in pool:
        by_category[art['category']].append(art)

    # Group small categories together
    major_cats = {}
    minor_pool = []
    for cat, arts in by_category.items():
        if len(arts) >= 20:
            major_cats[cat] = arts
        else:
            minor_pool.extend(arts)

    logger.info(f"Major categories (>=20 articles): {len(major_cats)}")
    logger.info(f"Minor categories (<20 articles): {len(minor_pool)} articles across {len(by_category) - len(major_cats)} categories")

    # Proportional sampling from major categories + random from minor
    total_major = sum(len(arts) for arts in major_cats.values())
    sampled = []

    for cat, arts in sorted(major_cats.items(), key=lambda x: -len(x[1])):
        # Proportional allocation with a minimum of 3
        n = max(3, int(TARGET_ARTICLES * len(arts) / (total_major + len(minor_pool))))
        n = min(n, len(arts))
        random.shuffle(arts)
        sampled.extend(arts[:n])

    # Fill remaining from minor pool + overflow from major
    remaining = TARGET_ARTICLES - len(sampled)
    if remaining > 0:
        random.shuffle(minor_pool)
        sampled.extend(minor_pool[:remaining])

    # If still short, sample more from largest categories
    remaining = TARGET_ARTICLES - len(sampled)
    if remaining > 0:
        sampled_ids = {a['id'] for a in sampled}
        extra_pool = [a for a in pool if a['id'] not in sampled_ids]
        random.shuffle(extra_pool)
        sampled.extend(extra_pool[:remaining])

    sampled = sampled[:TARGET_ARTICLES]
    logger.info(f"Sampled {len(sampled)} articles for expansion")

    # Report
    cat_dist = Counter(a['category'] for a in sampled)
    src_dist = Counter(a['source'] for a in sampled)
    dates = [a['fetched_at'] for a in sampled if a['fetched_at']]

    logger.info(f"\nCategory distribution (top 20):")
    for cat, count in cat_dist.most_common(20):
        logger.info(f"  {cat:30s}: {count:4d}")

    logger.info(f"\nSource distribution:")
    for src, count in src_dist.most_common(10):
        logger.info(f"  {src:30s}: {count:4d}")

    if dates:
        logger.info(f"\nDate range: {min(dates)[:10]} to {max(dates)[:10]}")

    # Save
    with open(EXPANSION_ARTICLES_FILE, 'w') as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved {len(sampled)} expansion articles to {EXPANSION_ARTICLES_FILE}")

    return sampled


# ══════════════════════════════════════════════════════════════
# PHASE 2: SCORE WITH GEMINI 3 FLASH
# ══════════════════════════════════════════════════════════════
def build_scoring_prompt(profile: dict, article: dict) -> str:
    """Build scoring prompt matching stage4_prepare.py format with real category."""
    role = profile['role']
    location = profile['location']
    context = profile.get('context', 'Not specified')
    companies = profile.get('tracked_companies', 'None specified')
    institutions = profile.get('tracked_institutions', 'None specified')
    interests = ', '.join(profile.get('interests', []))
    industries = profile.get('tracked_industries', 'None specified')

    if _is_student(role):
        level_note = "IMPORTANT: User is a student. Senior/Lead/Director roles requiring years of experience score 0-3. Entry-level, internship, and graduate positions score highest."
    else:
        level_note = "IMPORTANT: User is an experienced professional. Entry-level/intern positions score low. Senior and specialist roles score highest."

    system_prompt = f"""You are a relevance scorer for a {role} in {location}.
User context: {context}
Tracked companies: {companies}
Tracked institutions: {institutions}
Tracked interests: {interests if interests else 'None specified'}
Tracked industries: {industries}
{level_note}

Score each article 0.0-10.0:
9-10: Directly actionable (hiring match, breakthrough in tracked area)
7-8.9: Highly relevant to user's field/interests
5-6.9: Somewhat relevant
0-4.9: Not relevant / noise / wrong level

Reply ONLY with: SCORE: X.X | REASON: brief explanation"""

    title = article.get('title', '')[:200]
    content = article.get('summary', '')[:500]
    category = article.get('category', 'general')

    user_message = f"""Score this article:
Category: {category}
Keywords: {interests}
Title: {title}
Content: {content}"""

    return system_prompt + "\n\n" + user_message + FORMAT_SUFFIX


def _score_one(client, art_id, prof_id, article, profile):
    """Score a single (article, profile) pair. Returns a result dict."""
    prompt = build_scoring_prompt(profile, article)

    for attempt in range(3):
        try:
            if attempt == 2:
                # Final attempt: stripped-down prompt
                simple_prompt = (
                    f"Rate this article's relevance to a {profile['role']} in "
                    f"{profile['location']} on a scale of 0.0-10.0.\n\n"
                    f"Title: {article['title'][:200]}\n"
                    f"Content: {article['summary'][:300]}\n\n"
                    f"Respond with ONLY: SCORE: X.X"
                )
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=simple_prompt,
                    config={"temperature": GEMINI_TEMPERATURE, "max_output_tokens": 256},
                )
            else:
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt,
                    config={"temperature": GEMINI_TEMPERATURE, "max_output_tokens": GEMINI_MAX_TOKENS},
                )

            text = resp.text or ''
            score, reason = parse_score(text)

            if score is not None:
                return {
                    'article_id': art_id, 'profile_id': prof_id,
                    'score': round(max(0.0, min(10.0, score)), 1),
                    'reason': (reason or '')[:300],
                    'source': 'gemini-3-flash',
                }

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                time.sleep(30 * (attempt + 1) + random.random() * 10)
            else:
                time.sleep(5 * (attempt + 1) + random.random() * 3)

    # All attempts failed
    return {
        'article_id': art_id, 'profile_id': prof_id,
        'score': None, 'reason': '', 'source': 'gemini-3-flash',
        'parse_failure': True,
    }


def _get_api_key():
    """Get Google API key from env or .env file."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        for p in [Path(".env"), Path(__file__).parent.parent / ".env"]:
            if p.exists():
                for line in open(p):
                    if line.strip().startswith("GOOGLE_API_KEY="):
                        api_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        break
            if api_key:
                break
    if not api_key:
        logger.error("GOOGLE_API_KEY not found. Set in environment or .env")
        sys.exit(1)
    return api_key


def _load_articles_and_profiles():
    """Load expansion articles and profiles, return maps and all pairs."""
    if not EXPANSION_ARTICLES_FILE.exists():
        logger.error(f"{EXPANSION_ARTICLES_FILE} not found. Run --phase 1 first.")
        sys.exit(1)
    articles = json.load(open(EXPANSION_ARTICLES_FILE))
    article_map = {a['id']: a for a in articles}

    from data.v2_pipeline.profiles_v2 import ALL_PROFILES
    profiles = [p for p in ALL_PROFILES if p['id'] not in SKIP_PROFILES_SCORING]
    profile_map = {p['id']: p for p in profiles}

    all_pairs = [(a['id'], p['id']) for a in articles for p in profiles]
    return article_map, profile_map, all_pairs


def phase2_score(resume=False):
    """Score expansion articles with Gemini 3 Flash.

    Daily RPD limit: 10,000 requests/day (Tier 1).
    Run once per day — scores up to 9,500 pairs per run (with safety margin).
    Automatically resumes from checkpoint. ~5 days for 50K total pairs.

    Usage:
      # Run daily (cron or manual):
      GOOGLE_API_KEY=... python3 v22_expansion.py --phase 2
      # Or with resume flag (same behavior, flag is for compatibility):
      GOOGLE_API_KEY=... python3 v22_expansion.py --phase 2 --resume
    """
    DAILY_LIMIT = 9500  # Safety margin under 10K RPD
    WORKERS = 15  # Conservative concurrency (~180 RPM, well under 2000 RPM)
    SAVE_INTERVAL = 500

    logger.info("PHASE 2: Score with Gemini 3 Flash (daily batch)")

    api_key = _get_api_key()
    from google import genai
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = genai.Client(api_key=api_key)
    article_map, profile_map, all_pairs = _load_articles_and_profiles()
    total = len(all_pairs)
    logger.info(f"Loaded {len(article_map)} articles, {len(profile_map)} profiles")
    logger.info(f"Total scoring pairs: {total}")

    # Resume from existing scores
    results = []
    scored_pairs = set()
    if EXPANSION_SCORES_FILE.exists():
        results = json.load(open(EXPANSION_SCORES_FILE))
        scored_pairs = {(s['article_id'], s['profile_id']) for s in results if s.get('score') is not None}
        logger.info(f"Already scored (valid): {len(scored_pairs)}")

    remaining = [(aid, pid) for aid, pid in all_pairs if (aid, pid) not in scored_pairs]
    logger.info(f"Remaining to score: {len(remaining)}")

    if not remaining:
        logger.info("All pairs already scored!")
        return results

    # Cap this run at daily limit
    today_batch = remaining[:DAILY_LIMIT]
    logger.info(f"This run: {len(today_batch)} (daily limit {DAILY_LIMIT})")
    logger.info(f"After this run: {len(remaining) - len(today_batch)} will remain")

    batch_start = time.time()
    done_count = 0
    fail_count = 0
    lock = __import__('threading').Lock()

    def do_score(pair):
        aid, pid = pair
        return _score_one(client, aid, pid, article_map[aid], profile_map[pid])

    logger.info(f"Starting {WORKERS} concurrent workers...")

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(do_score, pair): pair for pair in today_batch}

        for future in as_completed(futures):
            result = future.result()
            with lock:
                results.append(result)
                done_count += 1
                if result.get('parse_failure'):
                    fail_count += 1

                if done_count % 100 == 0 or done_count <= 10:
                    elapsed = time.time() - batch_start
                    rate = done_count / elapsed if elapsed > 0 else 0
                    eta_min = (len(today_batch) - done_count) / rate / 60 if rate > 0 else 0
                    logger.info(
                        f"[{len(scored_pairs)+done_count}/{total}] "
                        f"done={done_count}/{len(today_batch)} "
                        f"fail={fail_count} rate={rate:.1f}/s ETA={eta_min:.0f}min"
                    )

                if done_count % SAVE_INTERVAL == 0:
                    with open(EXPANSION_SCORES_FILE, 'w') as f:
                        json.dump(results, f, ensure_ascii=False)
                    logger.info(f"  Checkpoint saved ({len(results)} scores)")

                # Stop early if we hit rate limits (high failure rate)
                if done_count > 100 and fail_count / done_count > 0.3:
                    logger.warning(f"High failure rate ({fail_count}/{done_count} = {100*fail_count/done_count:.0f}%). Likely RPD exhausted.")
                    logger.warning("Saving checkpoint and stopping. Run again tomorrow.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    # Final save
    with open(EXPANSION_SCORES_FILE, 'w') as f:
        json.dump(results, f, ensure_ascii=False)

    valid = [r for r in results if r.get('score') is not None]
    failed = [r for r in results if r.get('parse_failure')]
    elapsed = time.time() - batch_start

    logger.info(f"\nDaily run complete in {elapsed/60:.1f} minutes:")
    logger.info(f"  Scored this run: {done_count}")
    logger.info(f"  Parse failures: {fail_count}")
    logger.info(f"  Total valid scores: {len(valid)}/{total}")
    logger.info(f"  Remaining: {total - len(valid)}")
    if total - len(valid) > 0:
        days_left = (total - len(valid)) / DAILY_LIMIT
        logger.info(f"  Estimated days remaining: {days_left:.1f}")
    logger.info(f"  Saved to {EXPANSION_SCORES_FILE}")

    # Score distribution
    band_dist = Counter(score_to_band(r['score']) for r in valid)
    logger.info(f"\nScore band distribution:")
    for band in ['noise', 'tangential', 'moderate', 'high', 'critical']:
        count = band_dist.get(band, 0)
        pct = 100 * count / len(valid) if valid else 0
        logger.info(f"  {band:>12}: {count:>6} ({pct:.1f}%)")

    return results


# ══════════════════════════════════════════════════════════════
# PHASE 3: ISOTONIC CALIBRATION
# ══════════════════════════════════════════════════════════════
def phase3_calibrate():
    """Fit isotonic calibration on validation data, apply to expansion scores."""
    logger.info("PHASE 3: Isotonic Calibration")

    # Load Gemini validation results (283 valid points)
    if not GEMINI_VALIDATION_RESULTS.exists():
        logger.error(f"{GEMINI_VALIDATION_RESULTS} not found.")
        sys.exit(1)

    val_results = json.load(open(GEMINI_VALIDATION_RESULTS))
    valid_val = [r for r in val_results if not r.get('parse_failure') and r.get('gemini_score') is not None]
    logger.info(f"Calibration training points: {len(valid_val)}")

    gem_scores = [r['gemini_score'] for r in valid_val]
    opus_scores = [r['opus_score'] for r in valid_val]

    # Fit isotonic regression
    from sklearn.isotonic import IsotonicRegression
    import numpy as np

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(gem_scores, opus_scores)

    # LOO validation
    loo_errors = []
    for i in range(len(gem_scores)):
        train_g = gem_scores[:i] + gem_scores[i+1:]
        train_o = opus_scores[:i] + opus_scores[i+1:]
        iso_loo = IsotonicRegression(out_of_bounds='clip')
        iso_loo.fit(train_g, train_o)
        pred = iso_loo.predict([gem_scores[i]])[0]
        loo_errors.append(abs(pred - opus_scores[i]))
    loo_mae = np.mean(loo_errors)
    logger.info(f"Isotonic LOO MAE: {loo_mae:.3f}")

    # Save calibration model
    cal_metadata = {
        'model': 'gemini-3-flash-preview',
        'validation_date': '2026-03-07',
        'training_points': len(valid_val),
        'loo_mae': float(loo_mae),
        'prompt_version': 'v22_score_first_suffix',
    }

    with open(CALIBRATION_MODEL_FILE, 'wb') as f:
        pickle.dump({'model': iso, 'metadata': cal_metadata}, f)
    logger.info(f"Saved calibration model to {CALIBRATION_MODEL_FILE}")

    # Also save as JSON lookup table for portability
    # Sample the calibration curve at 0.1 intervals
    test_points = [i / 10.0 for i in range(0, 101)]
    calibrated = iso.predict(test_points)
    lookup = {f"{p:.1f}": round(float(c), 2) for p, c in zip(test_points, calibrated)}
    cal_json = {'metadata': cal_metadata, 'lookup': lookup}
    with open(CALIBRATION_TABLE_FILE, 'w') as f:
        json.dump(cal_json, f, indent=2)
    logger.info(f"Saved calibration table to {CALIBRATION_TABLE_FILE}")

    # Apply calibration to expansion scores
    if not EXPANSION_SCORES_FILE.exists():
        logger.warning(f"{EXPANSION_SCORES_FILE} not found. Run --phase 2 first.")
        return

    scores = json.load(open(EXPANSION_SCORES_FILE))
    valid_scores = [s for s in scores if s.get('score') is not None]
    raw_scores = [s['score'] for s in valid_scores]

    if raw_scores:
        calibrated_scores = iso.predict(raw_scores)
        for s, cal in zip(valid_scores, calibrated_scores):
            s['raw_score'] = s['score']
            s['score'] = round(float(max(0.0, min(10.0, cal))), 1)

        # Save updated scores
        with open(EXPANSION_SCORES_FILE, 'w') as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

        logger.info(f"\nCalibrated {len(valid_scores)} scores")

        # Distribution comparison
        logger.info(f"\nBand distribution (raw → calibrated):")
        raw_bands = Counter(score_to_band(s['raw_score']) for s in valid_scores)
        cal_bands = Counter(score_to_band(s['score']) for s in valid_scores)
        for band in ['noise', 'tangential', 'moderate', 'high', 'critical']:
            r = raw_bands.get(band, 0)
            c = cal_bands.get(band, 0)
            logger.info(f"  {band:>12}: {r:>6} → {c:>6}")


# ══════════════════════════════════════════════════════════════
# PHASE 5: PREPARE TRAINING DATA
# ══════════════════════════════════════════════════════════════
def phase5_prepare():
    """Merge expansion scores with existing, prepare training_v2.2.jsonl."""
    logger.info("PHASE 5: Prepare training data")

    from data.v2_pipeline.profiles_v2 import ALL_PROFILES
    profile_map = {p['id']: p for p in ALL_PROFILES}

    # Load existing scores + articles
    existing_scores = json.load(open(EXISTING_SCORES_FILE))
    existing_articles = json.load(open(EXISTING_ARTICLES_FILE))
    logger.info(f"Existing: {len(existing_scores)} scores, {len(existing_articles)} articles")

    # Load expansion scores + articles
    expansion_scores = json.load(open(EXPANSION_SCORES_FILE))
    expansion_articles = json.load(open(EXPANSION_ARTICLES_FILE))
    logger.info(f"Expansion: {len(expansion_scores)} scores, {len(expansion_articles)} articles")

    # Filter valid expansion scores (non-null, non-parse-failure)
    valid_expansion = [s for s in expansion_scores if s.get('score') is not None and not s.get('parse_failure')]
    logger.info(f"Valid expansion scores: {len(valid_expansion)}")

    # Load holdout article IDs
    holdout_ids = set()
    if HOLDOUT_FILE.exists():
        holdout_data = json.load(open(HOLDOUT_FILE))
        holdout_ids = {a['id'] if isinstance(a, dict) else a for a in holdout_data}

    # Build combined article map
    all_articles = {a['id']: a for a in existing_articles}
    for a in expansion_articles:
        if a['id'] not in all_articles:
            all_articles[a['id']] = a

    # Merge scores: existing + valid expansion (exclude agent profiles)
    merged_scores = []

    # Add existing Opus scores (exclude agent-scored profiles — quality concerns)
    for s in existing_scores:
        if s['profile_id'] in SKIP_PROFILES_EXISTING:
            continue
        if s['article_id'] in holdout_ids:
            continue
        merged_scores.append(s)

    # Add expansion scores
    for s in valid_expansion:
        if s['article_id'] in holdout_ids:
            continue
        merged_scores.append(s)

    logger.info(f"Merged scores (after holdout exclusion): {len(merged_scores)}")

    # Load user feedback from DB
    feedback_scores = []
    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute("""
            SELECT ni.id as article_id, uf.user_score, uf.note, ni.title, ni.summary, ni.category
            FROM user_feedback uf
            JOIN news_items ni ON uf.url = ni.url
            WHERE uf.user_score IS NOT NULL AND ni.summary IS NOT NULL AND ni.summary != ''
        """).fetchall()
        conn.close()

        # User feedback maps to cpeg_student_kw profile (Ahmad's profile)
        for row in rows:
            art_id = row[0]
            if art_id in holdout_ids:
                continue
            if art_id not in all_articles:
                # Add the article
                all_articles[art_id] = {
                    'id': art_id,
                    'title': row[3] or '',
                    'summary': row[4] or '',
                    'category': row[5] or 'general',
                }
            feedback_scores.append({
                'article_id': art_id,
                'profile_id': 'cpeg_student_kw',
                'score': row[1],
                'reason': (row[2] or '')[:200],
                'source': 'user_feedback',
            })
        logger.info(f"User feedback entries: {len(feedback_scores)} (1.0x weight)")
    except Exception as e:
        logger.warning(f"Could not load user feedback: {e}")

    # Combine all scores
    all_scored = merged_scores + feedback_scores
    logger.info(f"Total scored examples: {len(all_scored)}")

    # Band distribution before undersampling
    band_dist = Counter(score_to_band(s['score']) for s in all_scored)
    logger.info(f"\nBand distribution (before noise cap):")
    for band in ['noise', 'tangential', 'moderate', 'high', 'critical']:
        count = band_dist.get(band, 0)
        logger.info(f"  {band:>12}: {count:>6} ({100*count/len(all_scored):.1f}%)")

    # Noise undersample: cap at NOISE_CAP
    random.seed(RANDOM_SEED)
    noise = [s for s in all_scored if score_to_band(s['score']) == 'noise']
    non_noise = [s for s in all_scored if score_to_band(s['score']) != 'noise']

    if len(noise) > NOISE_CAP:
        random.shuffle(noise)
        noise = noise[:NOISE_CAP]
        logger.info(f"Noise capped: {band_dist['noise']} → {NOISE_CAP}")

    all_scored = non_noise + noise
    random.shuffle(all_scored)

    # Cap at MAX_TRAIN
    if len(all_scored) > MAX_TRAIN:
        # Stratified cap: preserve band proportions
        by_band = defaultdict(list)
        for s in all_scored:
            by_band[score_to_band(s['score'])].append(s)

        capped = []
        total_available = len(all_scored)
        for band, examples in by_band.items():
            n = int(MAX_TRAIN * len(examples) / total_available)
            random.shuffle(examples)
            capped.extend(examples[:n])

        # Fill any remainder
        remaining = MAX_TRAIN - len(capped)
        if remaining > 0:
            capped_ids = {(s['article_id'], s['profile_id']) for s in capped}
            extras = [s for s in all_scored if (s['article_id'], s['profile_id']) not in capped_ids]
            random.shuffle(extras)
            capped.extend(extras[:remaining])

        all_scored = capped[:MAX_TRAIN]
        logger.info(f"Capped at {MAX_TRAIN} training examples")

    # Final band distribution
    band_dist = Counter(score_to_band(s['score']) for s in all_scored)
    logger.info(f"\nFinal band distribution:")
    for band in ['noise', 'tangential', 'moderate', 'high', 'critical']:
        count = band_dist.get(band, 0)
        logger.info(f"  {band:>12}: {count:>6} ({100*count/len(all_scored):.1f}%)")

    # Build training examples
    # Split 90/10 for train/eval
    by_band_profile = defaultdict(list)
    for s in all_scored:
        key = f"{score_to_band(s['score'])}_{s['profile_id']}"
        by_band_profile[key].append(s)

    train_examples = []
    eval_examples = []
    for key, examples in by_band_profile.items():
        random.shuffle(examples)
        n_eval = max(1, int(len(examples) * 0.1))
        eval_examples.extend(examples[:n_eval])
        train_examples.extend(examples[n_eval:])

    logger.info(f"\nTrain: {len(train_examples)}, Eval: {len(eval_examples)}")

    # Format as ChatML
    def build_chatml(scored, article, profile):
        role = profile['role']
        location = profile['location']
        context = profile.get('context', 'Not specified')
        companies = profile.get('tracked_companies', 'None specified')
        institutions = profile.get('tracked_institutions', 'None specified')
        interests = ', '.join(profile.get('interests', []))
        industries = profile.get('tracked_industries', 'None specified')

        if _is_student(role):
            level_note = "IMPORTANT: User is a student. Senior/Lead/Director roles requiring years of experience score 0-3. Entry-level, internship, and graduate positions score highest."
        else:
            level_note = "IMPORTANT: User is an experienced professional. Entry-level/intern positions score low. Senior and specialist roles score highest."

        system_prompt = f"""You are a relevance scorer for a {role} in {location}.
User context: {context}
Tracked companies: {companies}
Tracked institutions: {institutions}
Tracked interests: {interests if interests else 'None specified'}
Tracked industries: {industries}
{level_note}

Score each article 0.0-10.0:
9-10: Directly actionable (hiring match, breakthrough in tracked area)
7-8.9: Highly relevant to user's field/interests
5-6.9: Somewhat relevant
0-4.9: Not relevant / noise / wrong level

Reply ONLY with: SCORE: X.X | REASON: brief explanation"""

        title = article.get('title', '')[:200]
        content = article.get('summary', '')[:500]
        category = article.get('category', 'general')

        user_message = f"""Score this article:
Category: {category}
Keywords: {interests}
Title: {title}
Content: {content}"""

        # ALWAYS use short reason — never think_text (V2.1 lesson)
        score_val = scored['score']
        reason = scored.get('reason', 'No reason provided')
        if not reason or len(reason) < 5:
            reason = 'No reason provided'
        # Cap reason length to match V2 format (~170 chars avg)
        reason = reason[:300]

        assistant_message = f"SCORE: {score_val:.1f} | REASON: {reason}"

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        }

    train_jsonl = []
    eval_jsonl = []
    skipped = 0

    for s in train_examples:
        profile = profile_map.get(s['profile_id'])
        article = all_articles.get(s['article_id'])
        if not profile or not article:
            skipped += 1
            continue
        example = build_chatml(s, article, profile)
        example['sample_weight'] = LOSS_WEIGHTS[score_to_band(s['score'])]
        train_jsonl.append(example)

    for s in eval_examples:
        profile = profile_map.get(s['profile_id'])
        article = all_articles.get(s['article_id'])
        if not profile or not article:
            skipped += 1
            continue
        example = build_chatml(s, article, profile)
        eval_jsonl.append(example)

    if skipped:
        logger.warning(f"Skipped {skipped} examples (missing profile or article)")

    # Save training data
    with open(TRAINING_V22_FILE, 'w') as f:
        for ex in train_jsonl:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    with open(EVAL_V22_FILE, 'w') as f:
        for ex in eval_jsonl:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    # Update articles_v2.json with expansion articles
    updated_articles = list(existing_articles)
    existing_ids = {a['id'] for a in existing_articles}
    new_count = 0
    for a in expansion_articles:
        if a['id'] not in existing_ids:
            updated_articles.append(a)
            new_count += 1
    with open(EXISTING_ARTICLES_FILE, 'w') as f:
        json.dump(updated_articles, f, indent=2, ensure_ascii=False)
    logger.info(f"Updated articles_v2.json: {len(existing_articles)} → {len(updated_articles)} (+{new_count})")

    # Verify format
    if train_jsonl:
        first = train_jsonl[0]
        msgs = first['messages']
        assert len(msgs) == 3
        assert msgs[0]['role'] == 'system'
        assert msgs[1]['role'] == 'user'
        assert msgs[2]['role'] == 'assistant'
        assert msgs[2]['content'].startswith('SCORE:')
        # Verify no think_text leak
        assert len(msgs[2]['content']) < 500, f"Assistant message too long ({len(msgs[2]['content'])} chars) — possible think_text leak"
        # Verify real category (not all "general")
        categories = set()
        for ex in train_jsonl[:100]:
            m = re.search(r'Category: (\S+)', ex['messages'][1]['content'])
            if m:
                categories.add(m.group(1))
        assert len(categories) > 1, f"Only {categories} categories found — category fix may not be applied"
        logger.info(f"Format verification: PASSED (categories: {len(categories)} distinct)")

    # Report
    unique_articles = len({s['article_id'] for s in train_examples})
    unique_profiles = len({s['profile_id'] for s in train_examples})
    train_band = Counter(score_to_band(s['score']) for s in train_examples)

    logger.info(f"\n{'='*70}")
    logger.info(f"V2.2 TRAINING DATA REPORT")
    logger.info(f"{'='*70}")
    logger.info(f"Training examples: {len(train_jsonl)}")
    logger.info(f"Eval examples:     {len(eval_jsonl)}")
    logger.info(f"Unique articles:   {unique_articles}")
    logger.info(f"Unique profiles:   {unique_profiles}")
    logger.info(f"Training file:     {TRAINING_V22_FILE} ({TRAINING_V22_FILE.stat().st_size / 1024:.0f} KB)")
    logger.info(f"Eval file:         {EVAL_V22_FILE} ({EVAL_V22_FILE.stat().st_size / 1024:.0f} KB)")
    logger.info(f"\nBand distribution (training):")
    for band in ['noise', 'tangential', 'moderate', 'high', 'critical']:
        count = train_band.get(band, 0)
        logger.info(f"  {band:>12}: {count:>6} ({100*count/len(train_examples):.1f}%)")

    # Changes vs V2
    logger.info(f"\n--- Changes vs V2 ---")
    logger.info(f"Articles: 813 → {unique_articles} (+{unique_articles - 813})")
    logger.info(f"Format: Category=real (was 'general'), short reason only (no think_text)")
    logger.info(f"Noise cap: {NOISE_CAP} (was unlimited ~17K)")
    logger.info(f"User feedback: {len(feedback_scores)} entries at 1.0x weight")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="V2.2 Article Expansion Pipeline")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 5],
                        help="Pipeline phase: 1=extract, 2=score (daily), 3=calibrate, 5=prepare")
    parser.add_argument("--resume", action="store_true", help="Resume (always resumes, kept for compat)")
    args = parser.parse_args()

    if args.phase == 1:
        phase1_extract()
    elif args.phase == 2:
        phase2_score(resume=args.resume)
    elif args.phase == 3:
        phase3_calibrate()
    elif args.phase == 5:
        phase5_prepare()


if __name__ == "__main__":
    print = functools.partial(print, flush=True)
    main()
