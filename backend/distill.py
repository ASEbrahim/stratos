#!/usr/bin/env python3
"""
STRAT_OS — Model Distillation Pipeline
========================================
Uses Claude Opus 4.5 (teacher) to evaluate items that the local Ollama model
(student) already scored. Disagreements become training signals that improve
future scoring via the Tier 1 feedback loop.

Usage:
    python distill.py                    # Score all recent items (last 7 days)
    python distill.py --hours 48         # Last 48 hours only
    python distill.py --limit 50         # Max 50 items
    python distill.py --dry-run          # Show what would be sent, don't call API
    python distill.py --threshold 1.5    # Flag disagreements >= 1.5 (default 2.0)
    python distill.py --batch            # Use Batch API (50% cheaper, async)

API Key Setup:
    Option A (recommended): Set environment variable
        Windows:  set ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
        Linux:    export ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

    Option B: Add to config.yaml under 'distillation':
        distillation:
          api_key: sk-ant-api03-your-key-here

    Option C: Create backend/.env file:
        ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("DISTILL")

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

TEACHER_MODEL = "claude-opus-4-5-20251101"
API_URL = "https://api.anthropic.com/v1/messages"
BATCH_API_URL = "https://api.anthropic.com/v1/messages/batches"
API_VERSION = "2023-06-01"

# How many items to score per API call (batched in prompt, not API batching)
ITEMS_PER_CALL = 10

# Default disagreement threshold
DEFAULT_THRESHOLD = 2.0

# ═══════════════════════════════════════════════════════════════════
# API Key Resolution
# ═══════════════════════════════════════════════════════════════════

def get_api_key() -> Optional[str]:
    """Resolve API key from environment, .env file, or config.yaml."""
    # 1. Environment variable (highest priority)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        logger.info("API key loaded from environment variable")
        return key

    # 2. .env file
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    logger.info("API key loaded from .env file")
                    return key

    # 3. config.yaml
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            key = config.get("distillation", {}).get("api_key", "")
            if key:
                logger.info("API key loaded from config.yaml")
                return key
        except Exception:
            pass

    return None


# ═══════════════════════════════════════════════════════════════════
# Database Access
# ═══════════════════════════════════════════════════════════════════

def get_scored_items(db_path: str, hours: int = 168, limit: int = 200) -> List[Dict]:
    """Pull recently scored items from the database, deduplicated by URL."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    since = (datetime.now() - timedelta(hours=hours)).isoformat()

    # Use GROUP BY url to avoid re-scoring duplicate articles across cycles
    cursor.execute("""
        SELECT id, title, url, summary, source, root, category,
               score, score_reason, timestamp, MAX(fetched_at) as fetched_at
        FROM news_items
        WHERE fetched_at > ? AND score > 0
        GROUP BY url
        ORDER BY fetched_at DESC
        LIMIT ?
    """, (since, limit))

    items = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return items


def get_profile(config_path: str) -> Dict:
    """Load user profile from config."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        return config.get("profile", {})
    except Exception:
        return {}


def get_dynamic_categories(config_path: str) -> List[Dict]:
    """Load dynamic categories from config."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        return config.get("dynamic_categories", [])
    except Exception:
        return []


def save_corrections(db_path: str, corrections: List[Dict], profile: Dict = None):
    """Save teacher corrections into user_feedback table so Tier 1 picks them up.
    
    Also stores the profile context (role, location) so training data can be
    paired with the correct system prompt later.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    saved = 0
    
    p_role = (profile or {}).get('role', '')
    p_location = (profile or {}).get('location', '')
    p_context = (profile or {}).get('context', '')

    for c in corrections:
        try:
            cursor.execute("""
                INSERT INTO user_feedback
                (news_id, title, url, root, category, ai_score, user_score, note, action, created_at,
                 profile_role, profile_location, profile_context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                c.get('id', ''),
                c.get('title', ''),
                c.get('url', ''),
                c.get('root', ''),
                c.get('category', ''),
                c.get('local_score'),         # ai_score = what local model gave
                c.get('teacher_score'),        # user_score = what teacher says it should be
                f"Distillation: {c.get('teacher_reason', '')}",
                'rate',                        # Treated same as a user rating
                datetime.now().isoformat(),
                p_role,
                p_location,
                p_context[:500],  # Cap context length
            ))
            saved += 1
        except Exception as e:
            logger.warning(f"Failed to save correction: {e}")

    conn.commit()
    conn.close()
    return saved


# ═══════════════════════════════════════════════════════════════════
# Teacher Prompt
# ═══════════════════════════════════════════════════════════════════

def build_teacher_prompt(items: List[Dict], profile: Dict, categories: List[Dict]) -> Tuple[str, str]:
    """
    Build system + user prompt for the teacher model.
    Returns (system_prompt, user_prompt).
    """
    role = profile.get('role', 'professional')
    location = profile.get('location', 'unspecified')
    context = profile.get('context', '')

    # Build category summary
    cat_summary = ""
    if categories:
        cat_lines = []
        for cat in categories[:10]:
            cat_lines.append(f"  - {cat.get('label', cat.get('id', ''))}: {', '.join(cat.get('items', [])[:6])}")
        cat_summary = "\nTracked categories:\n" + "\n".join(cat_lines)

    system = f"""You are an expert relevance scorer for a personalized intelligence dashboard.

User profile: {role} in {location}
User context: {context}
{cat_summary}

Your job: Score each article 0.0-10.0 based on how actionable and relevant it is to THIS SPECIFIC USER.

Scoring guide:
  9.0-10.0: Directly actionable — job match at right level, breakthrough in tracked area, money-saving deal
  7.0-8.9:  Highly relevant — useful skill info, regional growth, notable trend in their field
  5.0-6.9:  Somewhat relevant — tangentially useful, worth knowing
  0.0-4.9:  Noise — generic ads, clickbait, wrong experience level, irrelevant topic, company profile pages

IMPORTANT RULES:
- Never score exactly 5.0 (decide: useful or noise)
- Generic company "About" pages, stock quote pages, and job listing INDEX pages score 0-3
- "Top 10 lists" and listicles score 0-4 unless they contain specific actionable info
- Consider the user is in {location} — local opportunities score higher
- Entry-level/student content scores higher if user is a student; senior roles score lower

REASONING INSTRUCTIONS:
For each article, evaluate along these dimensions BEFORE scoring:
- PROFILE MATCH: Does the topic directly relate to the user's profession or stated interests?
- LOCATION RELEVANCE: Is this geographically relevant to them? (exact city > country > region > global)
- ACTIONABILITY: Can they DO something with this? (apply, invest, learn, save money, prepare)
- LEVEL FIT: Does the seniority/experience level match theirs?
- NOISE CHECK: Is this genuine content or a generic page/clickbait/aggregator?

Then synthesize into a clear 2-3 sentence reason that explicitly connects the user's profile to the article.
A smaller model will learn from your reasoning — be specific about WHY, not just WHAT.

Good: "User is a geophysicist at KOC — this seismic acquisition contract in Kuwait's northern fields directly matches their daily work in exploration. They may need to prepare for data interpretation from this new survey."
Bad: "Relevant to user's field" (too vague — teaches nothing)
Bad: "Strong match + actionable: energy, oil" (keyword lists don't teach reasoning)

For EACH article, respond with a JSON object on its own line:
{{"idx": N, "score": X.X, "reason": "2-3 sentence explanation connecting user profile to article content"}}

One line per article. No other text."""

    # Build article list
    article_lines = []
    for i, item in enumerate(items):
        title = item.get('title', '')[:120]
        source = item.get('source', '')[:50]
        category = item.get('category', '')
        summary = (item.get('summary', '') or '')[:300]
        local_score = item.get('score', 0)

        article_lines.append(
            f"[{i}] Title: {title}\n"
            f"    Source: {source} | Category: {category} | Local score: {local_score:.1f}\n"
            f"    Summary: {summary}"
        )

    user_prompt = f"Score these {len(items)} articles:\n\n" + "\n\n".join(article_lines)

    return system, user_prompt


# ═══════════════════════════════════════════════════════════════════
# API Calls
# ═══════════════════════════════════════════════════════════════════

def call_teacher(system: str, user_prompt: str, api_key: str) -> Optional[str]:
    """Call Claude API and return text response."""
    import urllib.request
    import urllib.error

    payload = json.dumps({
        "model": TEACHER_MODEL,
        "max_tokens": 4096,
        "system": system,
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }).encode('utf-8')

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            # Extract text from content blocks
            text_parts = [
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            ]
            full_text = "\n".join(text_parts)

            # Log usage
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            # Opus 4.5: $5/M input, $25/M output
            cost = (input_tokens * 5.0 / 1_000_000) + (output_tokens * 25.0 / 1_000_000)
            logger.info(f"API call: {input_tokens} in + {output_tokens} out = ${cost:.4f}")

            return full_text

    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8') if e.fp else ''
        logger.error(f"API error {e.code}: {body[:500]}")
        # Raise fatal errors so the pipeline can stop immediately
        if e.code in (400, 401, 403) and ('credit balance' in body or 'auth' in body.lower() or 'api_key' in body.lower()):
            raise SystemExit(f"\n✗ FATAL: {body[:200]}\n  → Buy credits at https://platform.claude.com or check your API key.")
        return None
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return None


def parse_teacher_response(response: str, batch_size: int) -> List[Optional[Tuple[float, str]]]:
    """Parse teacher's JSON lines into (score, reason) tuples."""
    import re
    results = [None] * batch_size

    for line in response.strip().splitlines():
        line = line.strip()
        if not line or not line.startswith('{'):
            continue
        try:
            data = json.loads(line)
            idx = int(data.get("idx", -1))
            score = float(data.get("score", -1))
            reason = data.get("reason", "")
            if 0 <= idx < batch_size and 0 <= score <= 10:
                results[idx] = (score, reason)
        except (json.JSONDecodeError, ValueError, TypeError):
            # Try regex fallback
            m = re.search(r'"idx"\s*:\s*(\d+).*?"score"\s*:\s*([\d.]+).*?"reason"\s*:\s*"([^"]*)"', line)
            if m:
                idx = int(m.group(1))
                score = float(m.group(2))
                reason = m.group(3)
                if 0 <= idx < batch_size and 0 <= score <= 10:
                    results[idx] = (score, reason)

    return results


# ═══════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════

def run_distillation(
    db_path: str,
    config_path: str,
    api_key: str,
    hours: int = 168,
    limit: int = 200,
    threshold: float = DEFAULT_THRESHOLD,
    dry_run: bool = False,
    items_per_call: int = ITEMS_PER_CALL
):
    """
    Main distillation pipeline:
    1. Pull locally scored items from DB
    2. Send to Claude Opus 4.5 for re-scoring
    3. Compare scores, find disagreements
    4. Save corrections into feedback table for Tier 1
    """
    logger.info("=" * 60)
    logger.info("STRAT_OS Distillation Pipeline")
    logger.info("=" * 60)

    # Load items
    items = get_scored_items(db_path, hours=hours, limit=limit)
    if not items:
        logger.warning("No scored items found in database. Run a scan first.")
        return

    logger.info(f"Loaded {len(items)} scored items (last {hours}h)")

    # Load profile
    profile = get_profile(config_path)
    categories = get_dynamic_categories(config_path)
    logger.info(f"Profile: {profile.get('role', '?')[:60]} in {profile.get('location', '?')}")

    if dry_run:
        logger.info("\n[DRY RUN] Would send these items to Claude Opus 4.5:")
        for i, item in enumerate(items[:20]):
            logger.info(f"  [{i}] {item['score']:.1f} | {item['title'][:80]}")
        logger.info(f"\n  Total: {len(items)} items in {(len(items) + items_per_call - 1) // items_per_call} API calls")
        est_tokens = len(items) * 500  # ~500 tokens per item estimate
        est_cost = (est_tokens * 5.0 / 1_000_000) + (len(items) * 30 * 25.0 / 1_000_000)
        logger.info(f"  Estimated cost: ~${est_cost:.2f}")
        return

    # Process in batches
    all_corrections = []
    total_cost = 0.0
    total_agreed = 0
    total_disagreed = 0

    batches = [items[i:i + items_per_call] for i in range(0, len(items), items_per_call)]
    logger.info(f"Processing {len(batches)} batches of ~{items_per_call} items each...")

    for batch_idx, batch in enumerate(batches):
        logger.info(f"\nBatch {batch_idx + 1}/{len(batches)} ({len(batch)} items)")

        # Build prompt
        system, user_prompt = build_teacher_prompt(batch, profile, categories)

        # Call teacher
        response = call_teacher(system, user_prompt, api_key)
        if not response:
            logger.error(f"Batch {batch_idx + 1} failed — skipping")
            continue

        # Parse response
        results = parse_teacher_response(response, len(batch))
        parsed_count = sum(1 for r in results if r is not None)
        logger.info(f"  Parsed {parsed_count}/{len(batch)} scores")

        # Compare
        for i, item in enumerate(batch):
            if results[i] is None:
                continue

            teacher_score, teacher_reason = results[i]
            local_score = item.get('score', 0)
            delta = abs(teacher_score - local_score)

            if delta >= threshold:
                direction = "↑" if teacher_score > local_score else "↓"
                logger.info(f"  {direction} DISAGREE [{delta:.1f}]: \"{item['title'][:70]}\" "
                           f"Local:{local_score:.1f} → Teacher:{teacher_score:.1f}")
                all_corrections.append({
                    'id': item.get('id', ''),
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'root': item.get('root', ''),
                    'category': item.get('category', ''),
                    'local_score': local_score,
                    'teacher_score': teacher_score,
                    'teacher_reason': teacher_reason,
                    'delta': delta
                })
                total_disagreed += 1
            else:
                total_agreed += 1

        # Rate limit: 5 requests/min on free tier
        if batch_idx < len(batches) - 1:
            logger.info("  Waiting 13s (rate limit)...")
            time.sleep(13)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DISTILLATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Items scored by teacher: {total_agreed + total_disagreed}")
    logger.info(f"Agreed (within ±{threshold}): {total_agreed}")
    logger.info(f"Disagreements: {total_disagreed}")

    if all_corrections:
        # Sort by magnitude
        all_corrections.sort(key=lambda c: c['delta'], reverse=True)

        logger.info(f"\nTop disagreements:")
        for c in all_corrections[:10]:
            d = "↑" if c['teacher_score'] > c['local_score'] else "↓"
            logger.info(f"  {d} [{c['delta']:.1f}] \"{c['title'][:70]}\" "
                       f"L:{c['local_score']:.1f} → T:{c['teacher_score']:.1f} — {c['teacher_reason'][:60]}")

        # Save to database
        saved = save_corrections(db_path, all_corrections, profile=profile)
        logger.info(f"\n✓ Saved {saved} corrections to feedback table")
        logger.info("  These will be injected into the scorer's prompt on the next scan.")
    else:
        logger.info("\n✓ Local model agrees with teacher — no corrections needed!")

    # Save full report
    report_path = Path(__file__).parent / "data" / "distillation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "teacher_model": TEACHER_MODEL,
        "items_scored": total_agreed + total_disagreed,
        "agreed": total_agreed,
        "disagreements": total_disagreed,
        "threshold": threshold,
        "corrections": all_corrections
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Full report saved to: {report_path}")

    # Return summary for programmatic callers (e.g. autopilot.py)
    return {
        "items_scored": total_agreed + total_disagreed,
        "agreed": total_agreed,
        "disagreements": total_disagreed,
        "corrections_saved": len(all_corrections),
        "estimated_cost": round(len(batches) * 0.019, 4),  # ~$0.019 per API call
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="STRAT_OS Distillation — Train local model using Claude Opus 4.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python distill.py --dry-run          # Preview what would be sent
  python distill.py                    # Run distillation (last 7 days)
  python distill.py --hours 24         # Only last 24 hours
  python distill.py --limit 50         # Cap at 50 items
  python distill.py --threshold 1.5    # More sensitive disagreement detection
        """
    )
    parser.add_argument("--hours", type=int, default=168,
                        help="Look back N hours for scored items (default: 168 = 7 days)")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max items to process (default: 200)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Score disagreement threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be sent without calling API")
    parser.add_argument("--batch-size", type=int, default=ITEMS_PER_CALL,
                        help=f"Items per API call (default: {ITEMS_PER_CALL})")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to database (auto-detected if not set)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (auto-detected if not set)")

    args = parser.parse_args()

    # Resolve paths
    backend_dir = Path(__file__).parent
    db_path = args.db or str(backend_dir / "strat_os.db")
    config_path = args.config or str(backend_dir / "config.yaml")

    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Run a scan first so there are items to distill.")
        sys.exit(1)

    if not Path(config_path).exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    # Get API key
    api_key = get_api_key()
    if not api_key and not args.dry_run:
        logger.error("No API key found! Set it using one of these methods:")
        logger.error("  1. Environment:  set ANTHROPIC_API_KEY=sk-ant-api03-...")
        logger.error("  2. .env file:    echo ANTHROPIC_API_KEY=sk-ant-api03-... > backend/.env")
        logger.error("  3. config.yaml:  Add 'distillation: api_key: sk-ant-...' section")
        logger.error("\nGet your key at: https://platform.claude.com")
        sys.exit(1)

    # Run
    run_distillation(
        db_path=db_path,
        config_path=config_path,
        api_key=api_key or "",
        hours=args.hours,
        limit=args.limit,
        threshold=args.threshold,
        dry_run=args.dry_run,
        items_per_call=args.batch_size
    )


if __name__ == "__main__":
    main()
