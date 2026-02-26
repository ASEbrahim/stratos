#!/usr/bin/env python3
"""
V2 Training Pipeline — Stage 3: Score Articles via Batch API
==============================================================
Score every article against every profile: articles × 30 profiles
Uses Claude Sonnet via Anthropic Batch API.

Budget ceiling: $64
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.chdir(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load API key from .env if not in environment (same pattern as distill.py)
if not os.environ.get("ANTHROPIC_API_KEY"):
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

from data.v2_pipeline.profiles_v2 import ALL_PROFILES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("V2_SCORE")

OUTPUT_DIR = Path(__file__).parent
ARTICLES_FILE = OUTPUT_DIR / "articles_v2.json"
BATCH_INPUT_FILE = OUTPUT_DIR / "batch_input.jsonl"
BATCH_OUTPUT_FILE = OUTPUT_DIR / "batch_output.jsonl"
SCORES_FILE = OUTPUT_DIR / "scores_v2.json"
BATCH_ID_FILE = OUTPUT_DIR / "batch_id.txt"

# Budget
BUDGET_CEILING = 64.0
# Sonnet pricing (per 1M tokens)
SONNET_INPUT_PRICE = 3.0   # $3/1M input tokens
SONNET_OUTPUT_PRICE = 15.0  # $15/1M output tokens
# Batch API gets 50% discount
BATCH_INPUT_PRICE = SONNET_INPUT_PRICE * 0.5
BATCH_OUTPUT_PRICE = SONNET_OUTPUT_PRICE * 0.5

SCORER_SYSTEM_PROMPT = """You are a relevance scorer for a personalized news intelligence system. You receive a user profile and a news article. Score the article's relevance to this specific user on a scale of 0.0-10.0.

You MUST think step-by-step before scoring. Your reasoning MUST:
1. Identify specific elements of the user's profile (role, location, interests) that relate to this article
2. Explain WHY those elements make this article more or less relevant to THIS user specifically (not to a general audience)
3. Consider both direct relevance (exact match to their role/interests) and indirect relevance (affects their industry, location, or career trajectory)
4. A score of 5.0 is FORBIDDEN — you must decide if the article is positively relevant (6.0+) or noise (4.0-)

Your reasoning must be at least 3 sentences. Do not skip reasoning and jump to a score.

Reply with your reasoning first, then on the final line: SCORE: X.X | REASON: one-sentence summary"""


def build_user_message(profile: dict, article: dict) -> str:
    """Build the user message for scoring."""
    role = profile['role']
    location = profile['location']
    context = profile.get('context', '')
    interests = ', '.join(profile.get('interests', []))
    companies = profile.get('tracked_companies', 'None')
    institutions = profile.get('tracked_institutions', 'None')
    industries = profile.get('tracked_industries', 'None')

    title = article.get('title', '')[:200]
    summary = article.get('summary', '')[:500]

    return f"""USER PROFILE:
Role: {role}
Location: {location}
Context: {context}
Interests: {interests}
Tracked Companies: {companies}
Tracked Institutions: {institutions}
Tracked Industries: {industries}

ARTICLE:
Title: {title}
Content: {summary}

Score this article's relevance to this specific user (0.0-10.0)."""


def estimate_cost(n_requests: int, avg_input_tokens: int = 350, avg_output_tokens: int = 200) -> float:
    """Estimate batch API cost."""
    total_input = n_requests * avg_input_tokens
    total_output = n_requests * avg_output_tokens
    input_cost = (total_input / 1_000_000) * BATCH_INPUT_PRICE
    output_cost = (total_output / 1_000_000) * BATCH_OUTPUT_PRICE
    return input_cost + output_cost


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


def submit_batch(input_file: str) -> str:
    """Submit batch to Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()

    # Upload the batch
    with open(input_file, 'r') as f:
        requests = [json.loads(line) for line in f]

    logger.info(f"Submitting batch with {len(requests)} requests...")

    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    logger.info(f"Batch submitted: {batch_id}")

    # Save batch ID
    with open(BATCH_ID_FILE, 'w') as f:
        f.write(batch_id)

    return batch_id


def poll_batch(batch_id: str) -> dict:
    """Poll batch status until complete."""
    import anthropic
    client = anthropic.Anthropic()

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts

        logger.info(
            f"Batch {batch_id}: {status} — "
            f"processing={counts.processing}, "
            f"succeeded={counts.succeeded}, "
            f"errored={counts.errored}, "
            f"canceled={counts.canceled}"
        )

        if status == "ended":
            return {
                "status": status,
                "succeeded": counts.succeeded,
                "errored": counts.errored,
                "canceled": counts.canceled,
            }

        time.sleep(30)


def download_results(batch_id: str, output_file: str) -> list:
    """Download batch results."""
    import anthropic
    client = anthropic.Anthropic()

    results = []
    for result in client.messages.batches.results(batch_id):
        results.append(json.loads(result.json()))

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Downloaded {len(results)} results to {output_file}")
    return results


def parse_score(text: str) -> tuple:
    """Parse score and reason from model output.
    Returns (score, reason, think_text)."""
    # Try to extract SCORE: X.X | REASON: ...
    score_match = re.search(r'SCORE:\s*(\d+\.?\d*)', text)
    reason_match = re.search(r'REASON:\s*(.+)', text)

    score = float(score_match.group(1)) if score_match else -1.0
    reason = reason_match.group(1).strip() if reason_match else ''

    # Everything before the SCORE line is the reasoning/think block
    if score_match:
        think_text = text[:score_match.start()].strip()
    else:
        think_text = text.strip()

    return score, reason, think_text


def process_results(results: list) -> list:
    """Process batch results into scored examples."""
    scored = []
    malformed = 0
    empty_think = 0
    sparse_think = 0
    score_dist = Counter()

    for r in results:
        custom_id = r.get('custom_id', '')
        result = r.get('result', {})

        # Extract text from response
        if result.get('type') == 'succeeded':
            message = result.get('message', {})
            content_blocks = message.get('content', [])
            text = ''
            for block in content_blocks:
                if block.get('type') == 'text':
                    text += block.get('text', '')

            score, reason, think_text = parse_score(text)

            # Validate
            if score < 0 or score > 10:
                malformed += 1
                continue

            # Count think tokens (rough: 1 token ≈ 4 chars)
            think_tokens = len(think_text.split())
            if think_tokens == 0:
                empty_think += 1
            elif think_tokens < 20:
                sparse_think += 1

            # Score distribution
            if score < 2:
                score_dist['0-2'] += 1
            elif score < 4:
                score_dist['2-4'] += 1
            elif score < 5:
                score_dist['4-5'] += 1
            elif score < 6:
                score_dist['5-6'] += 1
            elif score < 8:
                score_dist['6-8'] += 1
            else:
                score_dist['8-10'] += 1

            # Parse article_id and profile_id from custom_id
            parts = custom_id.split('__', 1)
            article_id = parts[0] if len(parts) > 0 else ''
            profile_id = parts[1] if len(parts) > 1 else ''

            scored.append({
                'article_id': article_id,
                'profile_id': profile_id,
                'score': score,
                'reason': reason,
                'think_text': think_text,
                'think_tokens': think_tokens,
                'custom_id': custom_id,
            })
        else:
            malformed += 1

    return scored, malformed, empty_think, sparse_think, score_dist


def main():
    print("=" * 80)
    print("V2 TRAINING PIPELINE — STAGE 3: BATCH API SCORING")
    print("=" * 80)

    # Load articles
    if not ARTICLES_FILE.exists():
        print(f"ERROR: {ARTICLES_FILE} not found. Run stage2_collect.py first.")
        sys.exit(1)

    with open(ARTICLES_FILE) as f:
        articles = json.load(f)

    print(f"Articles: {len(articles)}")
    print(f"Profiles: {len(ALL_PROFILES)}")
    n_requests = len(articles) * len(ALL_PROFILES)
    print(f"Total scoring requests: {n_requests}")

    # Cost estimate
    est_cost = estimate_cost(n_requests)
    print(f"Estimated cost: ${est_cost:.2f}")
    if est_cost > BUDGET_CEILING:
        print(f"!! OVER BUDGET CEILING of ${BUDGET_CEILING}!")
        print(f"!! Reduce articles to {int(BUDGET_CEILING / est_cost * len(articles))} max")
        sys.exit(1)

    # Create batch input file
    print(f"\nCreating batch input...")
    requests = create_batch_requests(articles, ALL_PROFILES)
    with open(BATCH_INPUT_FILE, 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
    print(f"Wrote {len(requests)} requests to {BATCH_INPUT_FILE}")
    print(f"File size: {BATCH_INPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")

    # Submit batch
    print(f"\nSubmitting batch...")
    batch_id = submit_batch(str(BATCH_INPUT_FILE))
    print(f"Batch ID: {batch_id}")

    # Poll until complete
    print(f"\nPolling for completion...")
    status = poll_batch(batch_id)
    print(f"Batch completed: {status}")

    # Download results
    print(f"\nDownloading results...")
    results = download_results(batch_id, str(BATCH_OUTPUT_FILE))

    # Process
    print(f"\nProcessing results...")
    scored, malformed, empty_think, sparse_think, score_dist = process_results(results)

    # Save scored results
    with open(SCORES_FILE, 'w') as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(scored)} scored examples to {SCORES_FILE}")

    # Calculate actual cost from usage
    total_input_tokens = 0
    total_output_tokens = 0
    for r in results:
        result = r.get('result', {})
        if result.get('type') == 'succeeded':
            usage = result.get('message', {}).get('usage', {})
            total_input_tokens += usage.get('input_tokens', 0)
            total_output_tokens += usage.get('output_tokens', 0)

    actual_cost = (total_input_tokens / 1_000_000 * BATCH_INPUT_PRICE +
                   total_output_tokens / 1_000_000 * BATCH_OUTPUT_PRICE)

    # Think block stats
    think_lengths = [s['think_tokens'] for s in scored if s['think_tokens'] > 0]

    # Report
    print(f"\n{'='*80}")
    print("STAGE 3 REPORT")
    print(f"{'='*80}")
    print(f"Total requests: {len(requests)}")
    print(f"Succeeded: {len(scored)}")
    print(f"Malformed: {malformed} ({malformed/max(len(results),1)*100:.1f}%)")
    print(f"Empty think blocks: {empty_think}")
    print(f"Sparse reasoning rejects (<20 tokens): {sparse_think}")
    print(f"\nCost:")
    print(f"  Input tokens: {total_input_tokens:,}")
    print(f"  Output tokens: {total_output_tokens:,}")
    print(f"  Total cost: ${actual_cost:.2f}")
    print(f"  Budget remaining: ${BUDGET_CEILING - actual_cost:.2f}")
    print(f"\nScore distribution:")
    for bucket in ['0-2', '2-4', '4-5', '5-6', '6-8', '8-10']:
        count = score_dist.get(bucket, 0)
        pct = count / max(len(scored), 1) * 100
        bar = '#' * int(pct / 2)
        print(f"  {bucket:>5}: {count:>6} ({pct:>5.1f}%) {bar}")
    print(f"\nThink block lengths (word count):")
    if think_lengths:
        print(f"  Min: {min(think_lengths)}")
        print(f"  Median: {sorted(think_lengths)[len(think_lengths)//2]}")
        print(f"  Mean: {sum(think_lengths)/len(think_lengths):.0f}")
        print(f"  Max: {max(think_lengths)}")
    else:
        print(f"  No think blocks found")

    # Threshold checks
    if malformed / max(len(results), 1) > 0.05:
        print(f"\n!! WARNING: Malformed rate {malformed/max(len(results),1)*100:.1f}% exceeds 5% threshold")
    if len(scored) < n_requests * 0.9:
        print(f"\n!! WARNING: Only {len(scored)}/{n_requests} succeeded")


if __name__ == "__main__":
    main()
