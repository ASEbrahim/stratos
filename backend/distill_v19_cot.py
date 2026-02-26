#!/usr/bin/env python3
"""
StratOS v19 — Chain-of-Thought Distillation via Anthropic Batch API
====================================================================

Sends articles × profiles to Claude Opus for CoT scoring.
Uses the Batch API (50% discount, async).

Usage:
    # Extract articles and estimate cost
    python distill_v19_cot.py --prepare

    # Submit batch (requires ANTHROPIC_API_KEY)
    python distill_v19_cot.py --submit

    # Check batch status
    python distill_v19_cot.py --status

    # Download and parse results
    python distill_v19_cot.py --collect
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("DISTILL-V19")

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

BACKEND_DIR = Path(__file__).parent
DATA_DIR = BACKEND_DIR / "data"

PROFILES_PATH = DATA_DIR / "profiles_v19.json"
POLARIZING_PATH = DATA_DIR / "polarizing_articles_v19.json"
TRAINING_MERGED_PATH = DATA_DIR / "training_merged.jsonl"

# Output files
BATCH_REQUESTS_PATH = DATA_DIR / "v19_batch_requests.jsonl"
BATCH_STATUS_PATH = DATA_DIR / "v19_batch_status.json"
RAW_RESULTS_PATH = DATA_DIR / "v19_raw_results.jsonl"
VALIDATED_PATH = DATA_DIR / "v19_validated_cot.jsonl"

TEACHER_MODEL = "claude-sonnet-4-6"
BATCH_API_URL = "https://api.anthropic.com/v1/messages/batches"
API_VERSION = "2023-06-01"

# The CoT system prompt for Opus (from STRATOS_V19_MASTER_PLAN.md)
COT_TEACHER_PROMPT = """You are generating training data for a relevance scoring model. Your task is to score an article's relevance to a specific user profile AND show your complete reasoning process.

## Output Format

<think>
PROFILE: [1-2 sentence summary of who this user is and what matters to them]
CONTENT: [1-2 sentence summary of what this article is about]
ANALYSIS:
- Role/interest relevance: [STRONG/WEAK/NONE] — [brief explanation]
- Location relevance: [STRONG/WEAK/NONE] — [brief explanation]
- Entity relevance: [STRONG/WEAK/NONE] — [brief explanation]
- Topic relevance: [STRONG/WEAK/NONE] — [brief explanation]
- Actionability: [HIGH/MEDIUM/LOW/NONE] — [brief explanation]
- Source quality: [OFFICIAL/NEWS/BLOG/UNKNOWN] — [brief explanation]
CALIBRATION: Strongest signal is [identify it]. [Map to score range based on signal STRENGTH, not match count]. [Adjust for timeliness/source quality if needed].
</think>

SCORE: X.X | REASON: [One concise sentence naming the strongest factor]

## Scoring Scale

- 9.0-10.0 (Critical): At least one STRONG tracked entity/role match AND directly actionable AND timely. User must act now.
- 7.0-8.9 (High): At least one STRONG match on a tracked dimension. Clearly valuable to this specific user.
- 5.5-6.9 (Moderate): WEAK matches or STRONG interest match without actionability. Worth knowing.
- 2.0-5.4 (Low): Tangential or indirect connections only.
- 0.0-1.9 (Noise): No meaningful relevance to this user's profile.

## Reasoning Depth Rules

- NOISE (0.0-1.9): Keep think block under 50 words. State the mismatch and move on.
- STANDARD (2.0-8.9): Full analysis, 80-150 words.
- CRITICAL (9.0-10.0): Thorough justification, 120-180 words. Explicitly confirm actionability, timeliness, source quality.

## Critical Rules

1. The score MUST reflect THIS SPECIFIC USER'S profile. The same article scored for different profiles MUST produce different scores when their dimensions differ.
2. Never score exactly 5.0 — commit above or below.
3. Score based on the STRONGEST signal, not by counting matching dimensions. One perfect tracked-company-hiring-your-role match beats four vague industry-adjacent connections.
4. Be honest about mismatches. If it's obvious noise, say so in 20 words, not 150.
5. Actionability is a multiplier: job posting with deadline > general trend article.
6. Location specificity is a multiplier: "Equate hiring in Kuwait" > "Oil industry hiring globally."
7. Source quality matters at the high end: official announcements support 9.0+; blog rumors cap at ~8.0.
8. Timeliness: expired opportunities (closed applications, past events) score Low regardless of match quality.
9. For interest-driven profiles with no career dimension, relevance is determined by interest alignment and source quality alone — do not penalize for lack of "career match."
"""


# ═══════════════════════════════════════════════════════════════════
# API Key Resolution
# ═══════════════════════════════════════════════════════════════════

def get_api_key() -> Optional[str]:
    """Resolve API key from environment or .env file."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    env_path = BACKEND_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key:
                    return key
    return None


# ═══════════════════════════════════════════════════════════════════
# Article Extraction
# ═══════════════════════════════════════════════════════════════════

def load_polarizing_articles() -> List[Dict]:
    """Load the 100 polarizing articles from Phase 2."""
    with open(POLARIZING_PATH) as f:
        articles = json.load(f)
    result = []
    for a in articles:
        text = f"Title: {a['headline']}\nContent: {a['summary']}"
        result.append({
            "article_id": f"polar_{a['id']}",
            "text": text,
            "source_type": a.get("source_type", "unknown"),
        })
    return result


def load_multiprofile_articles() -> List[Dict]:
    """Extract unique multi-profile articles from training_merged.jsonl."""
    from collections import defaultdict
    article_profiles = defaultdict(set)
    article_content = {}

    with open(TRAINING_MERGED_PATH) as f:
        for line in f:
            r = json.loads(line)
            msgs = r.get("messages", [])
            user_msg = ""
            sys_msg = ""
            for m in msgs:
                if m["role"] == "user":
                    user_msg = m["content"]
                if m["role"] == "system":
                    sys_msg = m["content"]
            if user_msg:
                key = user_msg[:200]
                article_profiles[key].add(sys_msg[:100])
                article_content[key] = user_msg

    # Keep only multi-profile articles (appeared with 2+ profiles)
    multi = {k: v for k, v in article_profiles.items() if len(v) >= 2}

    result = []
    for i, (key, _profiles) in enumerate(multi.items()):
        content = article_content[key]
        # Clean the article text — remove "Score this article:" prefix if present
        text = content
        if text.startswith("Score this article:"):
            text = text[len("Score this article:"):].strip()
        # Remove "Category:" and "Keywords:" lines if present at the start
        lines = text.split("\n")
        clean_lines = []
        for line in lines:
            if line.strip().startswith("Category:") or line.strip().startswith("Keywords:"):
                continue
            clean_lines.append(line)
        text = "\n".join(clean_lines).strip()

        result.append({
            "article_id": f"multi_{i}",
            "text": text,
            "source_type": "unknown",
        })

    return result


def load_profiles() -> List[Dict]:
    """Load the 10 v19 profiles."""
    with open(PROFILES_PATH) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# Batch Request Builder
# ═══════════════════════════════════════════════════════════════════

def build_batch_requests(articles: List[Dict], profiles: List[Dict]) -> List[Dict]:
    """Build Anthropic Batch API request objects for all article × profile pairs."""
    requests = []
    for article in articles:
        for profile in profiles:
            custom_id = f"{article['article_id']}__{profile['profile_id']}"

            # The user message is just the article text
            user_content = article["text"]

            # The system message combines the CoT teacher prompt + the profile context
            system_content = (
                COT_TEACHER_PROMPT.strip()
                + "\n\n## User Profile for This Scoring\n\n"
                + profile["system_prompt"]
            )

            request = {
                "custom_id": custom_id,
                "params": {
                    "model": TEACHER_MODEL,
                    "max_tokens": 1024,
                    "system": system_content,
                    "messages": [
                        {"role": "user", "content": user_content}
                    ]
                }
            }
            requests.append(request)

    return requests


# ═══════════════════════════════════════════════════════════════════
# Batch API Operations
# ═══════════════════════════════════════════════════════════════════

def api_call(method: str, url: str, api_key: str, data: bytes = None,
             content_type: str = "application/json") -> dict:
    """Make an authenticated API call."""
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
            "Content-Type": content_type,
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def submit_batch(api_key: str) -> dict:
    """Submit the batch request file to the Anthropic Batch API."""
    if not BATCH_REQUESTS_PATH.exists():
        logger.error(f"Batch requests file not found: {BATCH_REQUESTS_PATH}")
        logger.error("Run --prepare first.")
        sys.exit(1)

    # Read the JSONL file
    with open(BATCH_REQUESTS_PATH, "rb") as f:
        file_data = f.read()

    # Count requests
    num_requests = sum(1 for line in file_data.decode().strip().split("\n") if line.strip())
    logger.info(f"Submitting batch with {num_requests} requests...")

    # The Batch API expects multipart upload or direct JSONL
    # Using the requests array approach
    requests_list = []
    for line in file_data.decode().strip().split("\n"):
        if line.strip():
            requests_list.append(json.loads(line))

    payload = json.dumps({"requests": requests_list}).encode("utf-8")

    result = api_call("POST", BATCH_API_URL, api_key, data=payload)

    # Save batch status
    with open(BATCH_STATUS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Batch submitted! ID: {result.get('id')}")
    logger.info(f"Status: {result.get('processing_status')}")
    logger.info(f"Status saved to: {BATCH_STATUS_PATH}")
    return result


def check_status(api_key: str) -> dict:
    """Check the status of a submitted batch."""
    if not BATCH_STATUS_PATH.exists():
        logger.error("No batch status file found. Submit a batch first with --submit.")
        sys.exit(1)

    with open(BATCH_STATUS_PATH) as f:
        status = json.load(f)

    batch_id = status.get("id")
    if not batch_id:
        logger.error("No batch ID found in status file.")
        sys.exit(1)

    result = api_call("GET", f"{BATCH_API_URL}/{batch_id}", api_key)

    # Update status file
    with open(BATCH_STATUS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    counts = result.get("request_counts", {})
    logger.info(f"Batch {batch_id}:")
    logger.info(f"  Status: {result.get('processing_status')}")
    logger.info(f"  Succeeded: {counts.get('succeeded', 0)}")
    logger.info(f"  Errored: {counts.get('errored', 0)}")
    logger.info(f"  Processing: {counts.get('processing', 0)}")
    logger.info(f"  Canceled: {counts.get('canceled', 0)}")

    if result.get("processing_status") == "ended":
        logger.info(f"  Results URL: {result.get('results_url', 'N/A')}")

    return result


def collect_results(api_key: str):
    """Download and parse batch results."""
    if not BATCH_STATUS_PATH.exists():
        logger.error("No batch status file found.")
        sys.exit(1)

    with open(BATCH_STATUS_PATH) as f:
        status = json.load(f)

    # Refresh status
    batch_id = status.get("id")
    status = api_call("GET", f"{BATCH_API_URL}/{batch_id}", api_key)

    if status.get("processing_status") != "ended":
        logger.error(f"Batch not done yet. Status: {status.get('processing_status')}")
        counts = status.get("request_counts", {})
        logger.info(f"  Processing: {counts.get('processing', 0)} remaining")
        sys.exit(1)

    results_url = status.get("results_url")
    if not results_url:
        logger.error("No results URL in batch status.")
        sys.exit(1)

    logger.info(f"Downloading results from {results_url}...")

    # Download results
    req = urllib.request.Request(
        results_url,
        headers={
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw_data = resp.read().decode("utf-8")

    # Save raw results
    with open(RAW_RESULTS_PATH, "w") as f:
        f.write(raw_data)
    logger.info(f"Raw results saved to: {RAW_RESULTS_PATH}")

    # Parse and validate
    validate_results()


# ═══════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════

def parse_cot_response(text: str) -> Optional[Dict]:
    """Parse a CoT response into think_block + score + reason."""
    # Extract think block
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not think_match:
        return None
    think_block = think_match.group(1).strip()

    # Extract SCORE and REASON
    after_think = text[think_match.end():].strip()
    score_match = re.search(r"SCORE:\s*(\d+\.?\d*)\s*\|\s*REASON:\s*(.+)", after_think, re.IGNORECASE)
    if not score_match:
        return None

    score = float(score_match.group(1))
    reason = score_match.group(2).strip()

    return {
        "think_block": think_block,
        "score": score,
        "reason": reason,
    }


def validate_results():
    """Parse raw batch results and validate CoT format."""
    if not RAW_RESULTS_PATH.exists():
        logger.error("No raw results file found. Run --collect first.")
        sys.exit(1)

    # Load profiles for system_prompt lookup
    profiles = {p["profile_id"]: p for p in load_profiles()}

    valid = []
    invalid = []
    scores_5 = 0

    with open(RAW_RESULTS_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            result = json.loads(line)
            custom_id = result.get("custom_id", "")
            result_data = result.get("result", {})

            # Check for API errors
            if result_data.get("type") == "error":
                invalid.append({"custom_id": custom_id, "error": "API error", "detail": str(result_data)})
                continue

            # Extract text from message content
            message = result_data.get("message", {})
            content_blocks = message.get("content", [])
            text = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    text += block.get("text", "")

            if not text:
                invalid.append({"custom_id": custom_id, "error": "empty response"})
                continue

            # Parse CoT
            parsed = parse_cot_response(text)
            if not parsed:
                invalid.append({"custom_id": custom_id, "error": "format mismatch", "text": text[:200]})
                continue

            # Validate think block word count
            word_count = len(parsed["think_block"].split())
            if word_count < 15:
                invalid.append({"custom_id": custom_id, "error": f"think block too short ({word_count} words)"})
                continue
            if word_count > 250:
                invalid.append({"custom_id": custom_id, "error": f"think block too long ({word_count} words)"})
                continue

            # Validate score range
            if parsed["score"] < 0 or parsed["score"] > 10:
                invalid.append({"custom_id": custom_id, "error": f"score out of range: {parsed['score']}"})
                continue

            # Check for forbidden 5.0
            if parsed["score"] == 5.0:
                scores_5 += 1
                # Nudge slightly rather than reject
                parsed["score"] = 4.9
                parsed["reason"] += " (adjusted from 5.0)"

            # Parse custom_id to get article_id and profile_id
            parts = custom_id.split("__", 1)
            if len(parts) != 2:
                invalid.append({"custom_id": custom_id, "error": "bad custom_id format"})
                continue

            article_id, profile_id = parts

            # Build training example
            if profile_id not in profiles:
                invalid.append({"custom_id": custom_id, "error": f"unknown profile: {profile_id}"})
                continue

            profile_prompt = profiles[profile_id]["system_prompt"]

            # The assistant content = think block + score
            assistant_content = (
                f"<think>\n{parsed['think_block']}\n</think>\n\n"
                f"SCORE: {parsed['score']:.1f} | REASON: {parsed['reason']}"
            )

            valid.append({
                "custom_id": custom_id,
                "article_id": article_id,
                "profile_id": profile_id,
                "score": parsed["score"],
                "reason": parsed["reason"],
                "think_block": parsed["think_block"],
                "think_word_count": word_count,
                "assistant_content": assistant_content,
                "system_prompt": profile_prompt,
            })

    # Save validated results
    with open(VALIDATED_PATH, "w") as f:
        for item in valid:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"\nValidation Results:")
    logger.info(f"  Valid: {len(valid)}")
    logger.info(f"  Invalid: {len(invalid)}")
    logger.info(f"  Forbidden 5.0 (adjusted): {scores_5}")

    if invalid:
        logger.info(f"\nSample invalid responses:")
        for inv in invalid[:10]:
            logger.info(f"  {inv['custom_id']}: {inv['error']}")

    # Score distribution
    from collections import Counter
    score_bands = Counter()
    for item in valid:
        s = item["score"]
        if s >= 9.0:
            score_bands["Critical (9-10)"] += 1
        elif s >= 7.0:
            score_bands["High (7-8.9)"] += 1
        elif s >= 5.5:
            score_bands["Moderate (5.5-6.9)"] += 1
        elif s >= 2.0:
            score_bands["Low (2-5.4)"] += 1
        else:
            score_bands["Noise (0-1.9)"] += 1

    logger.info(f"\nScore distribution:")
    for band in ["Critical (9-10)", "High (7-8.9)", "Moderate (5.5-6.9)", "Low (2-5.4)", "Noise (0-1.9)"]:
        count = score_bands.get(band, 0)
        pct = count / len(valid) * 100 if valid else 0
        logger.info(f"  {band}: {count} ({pct:.1f}%)")

    # Think block word count by tier
    noise_wc = [i["think_word_count"] for i in valid if i["score"] < 2.0]
    standard_wc = [i["think_word_count"] for i in valid if 2.0 <= i["score"] < 9.0]
    critical_wc = [i["think_word_count"] for i in valid if i["score"] >= 9.0]

    if noise_wc:
        logger.info(f"\nThink block avg words - Noise: {sum(noise_wc)/len(noise_wc):.0f}")
    if standard_wc:
        logger.info(f"Think block avg words - Standard: {sum(standard_wc)/len(standard_wc):.0f}")
    if critical_wc:
        logger.info(f"Think block avg words - Critical: {sum(critical_wc)/len(critical_wc):.0f}")

    # Profile coverage
    profile_counts = Counter(i["profile_id"] for i in valid)
    logger.info(f"\nPer-profile counts:")
    for pid, count in sorted(profile_counts.items()):
        logger.info(f"  {pid}: {count}")

    logger.info(f"\nValidated results saved to: {VALIDATED_PATH}")


# ═══════════════════════════════════════════════════════════════════
# Prepare Phase
# ═══════════════════════════════════════════════════════════════════

def prepare():
    """Load articles, build batch requests, estimate cost."""
    logger.info("Loading profiles...")
    profiles = load_profiles()
    logger.info(f"  {len(profiles)} profiles loaded")

    logger.info("Loading polarizing articles...")
    polar = load_polarizing_articles()
    logger.info(f"  {len(polar)} polarizing articles")

    logger.info("Loading multi-profile articles from training_merged.jsonl...")
    multi = load_multiprofile_articles()
    logger.info(f"  {len(multi)} multi-profile articles")

    all_articles = polar + multi
    logger.info(f"\nTotal articles: {len(all_articles)}")
    logger.info(f"Total profiles: {len(profiles)}")
    logger.info(f"Total requests: {len(all_articles) * len(profiles)}")

    # Build batch requests
    logger.info("\nBuilding batch requests...")
    requests = build_batch_requests(all_articles, profiles)

    # Save to JSONL
    with open(BATCH_REQUESTS_PATH, "w") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    logger.info(f"Batch requests saved to: {BATCH_REQUESTS_PATH}")

    # Estimate cost
    # Opus: $15/M input, $75/M output (regular)
    # Batch API: 50% discount → $7.5/M input, $37.5/M output
    avg_input_tokens = 900  # system + article
    avg_output_tokens = 200  # think block + score
    total_input = len(requests) * avg_input_tokens
    total_output = len(requests) * avg_output_tokens
    batch_cost = (total_input * 7.5 / 1_000_000) + (total_output * 37.5 / 1_000_000)

    logger.info(f"\n{'='*60}")
    logger.info(f"COST ESTIMATE (Batch API, 50% discount)")
    logger.info(f"{'='*60}")
    logger.info(f"  Requests: {len(requests)}")
    logger.info(f"  Est. input tokens: {total_input:,} (~{avg_input_tokens}/request)")
    logger.info(f"  Est. output tokens: {total_output:,} (~{avg_output_tokens}/request)")
    logger.info(f"  Input cost: ${total_input * 7.5 / 1_000_000:.2f}")
    logger.info(f"  Output cost: ${total_output * 37.5 / 1_000_000:.2f}")
    logger.info(f"  TOTAL ESTIMATED: ${batch_cost:.2f}")
    logger.info(f"{'='*60}")

    # Show sample requests
    logger.info(f"\nSample requests:")
    for req in requests[:3]:
        cid = req["custom_id"]
        sys_len = len(req["params"]["system"])
        user_len = len(req["params"]["messages"][0]["content"])
        logger.info(f"  {cid}: system={sys_len} chars, user={user_len} chars")

    # Show one full request for review
    logger.info(f"\n{'='*60}")
    logger.info(f"SAMPLE FULL REQUEST (first polarizing article × first profile)")
    logger.info(f"{'='*60}")
    sample = requests[0]
    logger.info(f"Custom ID: {sample['custom_id']}")
    logger.info(f"System prompt ({len(sample['params']['system'])} chars):")
    logger.info(sample['params']['system'][:500] + "...")
    logger.info(f"\nUser message:")
    logger.info(sample['params']['messages'][0]['content'])


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="StratOS v19 CoT Distillation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true", help="Build batch requests and estimate cost")
    group.add_argument("--submit", action="store_true", help="Submit batch to Anthropic API")
    group.add_argument("--status", action="store_true", help="Check batch status")
    group.add_argument("--collect", action="store_true", help="Download and validate results")
    group.add_argument("--validate-only", action="store_true", help="Re-validate already downloaded results")

    args = parser.parse_args()

    if args.prepare:
        prepare()
    elif args.submit:
        api_key = get_api_key()
        if not api_key:
            logger.error("No API key found. Set ANTHROPIC_API_KEY environment variable.")
            sys.exit(1)
        submit_batch(api_key)
    elif args.status:
        api_key = get_api_key()
        if not api_key:
            logger.error("No API key found.")
            sys.exit(1)
        check_status(api_key)
    elif args.collect:
        api_key = get_api_key()
        if not api_key:
            logger.error("No API key found.")
            sys.exit(1)
        collect_results(api_key)
    elif args.validate_only:
        validate_results()


if __name__ == "__main__":
    main()
