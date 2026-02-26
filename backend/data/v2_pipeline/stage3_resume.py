#!/usr/bin/env python3
"""
V2 Pipeline — Stage 3 Resume: Poll existing batch, download, parse, report, then run Stage 4.
Does NOT create a new batch. Only retrieves results from an already-submitted batch.
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── Setup ──
os.chdir(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load API key from .env
env_path = Path(__file__).parent.parent.parent / ".env"
if not os.environ.get("ANTHROPIC_API_KEY") and env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            break

import anthropic

# ── Constants ──
BATCH_ID = "msgbatch_013jLkFM5J9dctuTQH7MqB9o"
POLL_INTERVAL = 60  # seconds
OUTPUT_DIR = Path(__file__).parent
BATCH_OUTPUT_FILE = OUTPUT_DIR / "batch_output.jsonl"
SCORES_FILE = OUTPUT_DIR / "scores_v2.json"
REPORT_FILE = OUTPUT_DIR / "v2_pipeline_report.md"

# Sonnet batch pricing (50% discount)
BATCH_INPUT_PRICE = 1.5   # $3/1M * 0.5
BATCH_OUTPUT_PRICE = 7.5  # $15/1M * 0.5


def parse_score(text: str) -> tuple:
    """Parse score and reason from model output. Returns (score, reason, think_text)."""
    score_match = re.search(r'SCORE:\s*(\d+\.?\d*)', text)
    reason_match = re.search(r'REASON:\s*(.+)', text)

    score = float(score_match.group(1)) if score_match else -1.0
    reason = reason_match.group(1).strip() if reason_match else ''

    if score_match:
        think_text = text[:score_match.start()].strip()
    else:
        think_text = text.strip()

    return score, reason, think_text


def process_results(results: list) -> tuple:
    """Process batch results into scored examples."""
    scored = []
    malformed = 0
    empty_think = 0
    sparse_think = 0
    score_dist = Counter()

    for r in results:
        custom_id = r.get('custom_id', '')
        result = r.get('result', {})

        if result.get('type') == 'succeeded':
            message = result.get('message', {})
            content_blocks = message.get('content', [])
            text = ''
            for block in content_blocks:
                if block.get('type') == 'text':
                    text += block.get('text', '')

            score, reason, think_text = parse_score(text)

            if score < 0 or score > 10:
                malformed += 1
                continue

            think_tokens = len(think_text.split())
            if think_tokens == 0:
                empty_think += 1
            elif think_tokens < 20:
                sparse_think += 1

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
    print("V2 PIPELINE — STAGE 3 RESUME (poll + download + parse)")
    print(f"Batch ID: {BATCH_ID}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    client = anthropic.Anthropic()

    # ── Poll until complete ──
    poll_count = 0
    while True:
        batch = client.messages.batches.retrieve(BATCH_ID)
        status = batch.processing_status
        c = batch.request_counts
        total = c.processing + c.succeeded + c.errored + c.expired + c.canceled
        pct = c.succeeded / max(total, 1) * 100

        poll_count += 1
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] Poll #{poll_count}: {status} — "
              f"succeeded={c.succeeded}, processing={c.processing}, "
              f"errored={c.errored}, expired={c.expired} "
              f"({pct:.1f}%)")

        if status == "ended":
            print(f"\nBatch ended. Succeeded: {c.succeeded}, Errored: {c.errored}, Expired: {c.expired}")
            break

        time.sleep(POLL_INTERVAL)

    # ── Download results ──
    print(f"\nDownloading results...")
    results = []
    for result in client.messages.batches.results(BATCH_ID):
        results.append(json.loads(result.json()))

    with open(BATCH_OUTPUT_FILE, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"Downloaded {len(results)} results to {BATCH_OUTPUT_FILE}")
    print(f"File size: {BATCH_OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")

    # ── Process & parse ──
    print(f"\nProcessing results...")
    scored, malformed, empty_think, sparse_think, score_dist = process_results(results)

    # Save scores
    with open(SCORES_FILE, 'w') as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(scored)} scored examples to {SCORES_FILE}")

    # ── Compute cost from usage ──
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

    # ── Think block stats ──
    think_lengths = [s['think_tokens'] for s in scored if s['think_tokens'] > 0]
    think_min = min(think_lengths) if think_lengths else 0
    think_max = max(think_lengths) if think_lengths else 0
    think_median = sorted(think_lengths)[len(think_lengths) // 2] if think_lengths else 0
    think_mean = sum(think_lengths) / len(think_lengths) if think_lengths else 0

    # ── Unique profiles/articles ──
    unique_profiles = set(s['profile_id'] for s in scored)
    unique_articles = set(s['article_id'] for s in scored)

    # ── Console report ──
    print(f"\n{'=' * 80}")
    print("STAGE 3 REPORT")
    print(f"{'=' * 80}")
    print(f"Total results:    {len(results)}")
    print(f"Succeeded:        {len(scored)}")
    print(f"Malformed:        {malformed} ({malformed / max(len(results), 1) * 100:.1f}%)")
    print(f"Empty think:      {empty_think}")
    print(f"Sparse reasoning: {sparse_think} (<20 words)")
    print(f"Unique profiles:  {len(unique_profiles)}")
    print(f"Unique articles:  {len(unique_articles)}")
    print(f"\nCost:")
    print(f"  Input tokens:   {total_input_tokens:,}")
    print(f"  Output tokens:  {total_output_tokens:,}")
    print(f"  Total cost:     ${actual_cost:.2f}")
    print(f"\nScore distribution:")
    for bucket in ['0-2', '2-4', '4-5', '5-6', '6-8', '8-10']:
        count = score_dist.get(bucket, 0)
        pct = count / max(len(scored), 1) * 100
        bar = '#' * int(pct / 2)
        print(f"  {bucket:>5}: {count:>6} ({pct:>5.1f}%) {bar}")
    print(f"\nThink block lengths (word count):")
    print(f"  Min:    {think_min}")
    print(f"  Median: {think_median}")
    print(f"  Mean:   {think_mean:.0f}")
    print(f"  Max:    {think_max}")

    # ── Write markdown report ──
    report_lines = [
        f"# V2 Pipeline — Stage 3 Report",
        f"",
        f"**Batch ID:** `{BATCH_ID}`",
        f"**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total results | {len(results)} |",
        f"| Succeeded | {len(scored)} |",
        f"| Malformed | {malformed} ({malformed / max(len(results), 1) * 100:.1f}%) |",
        f"| Empty think blocks | {empty_think} |",
        f"| Sparse reasoning (<20 words) | {sparse_think} |",
        f"| Unique profiles | {len(unique_profiles)} |",
        f"| Unique articles | {len(unique_articles)} |",
        f"",
        f"## Cost",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Input tokens | {total_input_tokens:,} |",
        f"| Output tokens | {total_output_tokens:,} |",
        f"| **Total cost** | **${actual_cost:.2f}** |",
        f"| Budget ceiling | $64.00 |",
        f"| Budget remaining | ${64.0 - actual_cost:.2f} |",
        f"",
        f"## Score Distribution",
        f"",
        f"| Range | Count | % | Histogram |",
        f"|-------|-------|---|-----------|",
    ]
    for bucket in ['0-2', '2-4', '4-5', '5-6', '6-8', '8-10']:
        count = score_dist.get(bucket, 0)
        pct = count / max(len(scored), 1) * 100
        bar = '█' * int(pct / 2)
        report_lines.append(f"| {bucket} | {count} | {pct:.1f}% | {bar} |")

    report_lines += [
        f"",
        f"## Think Block Length Distribution (word count)",
        f"",
        f"| Stat | Value |",
        f"|------|-------|",
        f"| Min | {think_min} |",
        f"| Median | {think_median} |",
        f"| Mean | {think_mean:.0f} |",
        f"| Max | {think_max} |",
        f"| Empty | {empty_think} |",
        f"| Sparse (<20) | {sparse_think} |",
        f"",
    ]

    with open(REPORT_FILE, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\nReport saved to {REPORT_FILE}")

    # ── Threshold checks ──
    malformed_rate = malformed / max(len(results), 1)
    if malformed_rate > 0.10:
        print(f"\n!! FAIL: Malformed rate {malformed_rate * 100:.1f}% exceeds 10% threshold")
        print("!! Stopping — do NOT proceed to Stage 4")
        sys.exit(1)

    if len(scored) < len(results) * 0.85:
        print(f"\n!! WARNING: Only {len(scored)}/{len(results)} succeeded — below 85%")

    # ── Auto-run Stage 4 ──
    print(f"\n{'=' * 80}")
    print("AUTO-LAUNCHING STAGE 4: DATA PREPARATION")
    print(f"{'=' * 80}\n")

    stage4_path = Path(__file__).parent / "stage4_prepare.py"
    result = subprocess.run(
        [sys.executable, str(stage4_path)],
        cwd=str(Path(__file__).parent.parent.parent),
    )

    if result.returncode != 0:
        print(f"\n!! Stage 4 exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n{'=' * 80}")
    print("V2 PIPELINE COMPLETE — Stage 3 + Stage 4 finished successfully")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
