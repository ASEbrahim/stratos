#!/usr/bin/env python3
"""
Gemini 3 Flash Scoring Validation Test
Compare Gemini scores against Opus ground truth on 300 stratified examples.
Same methodology as deepseek_validation.py for direct comparison.
"""

import json
import random
import re
import time
import sys
import os
import functools
from collections import defaultdict
from pathlib import Path

from google import genai

# Config
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    # Try loading from .env
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in open(env_path):
            line = line.strip()
            if line.startswith("GOOGLE_API_KEY="):
                GOOGLE_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

MODEL = "gemini-3-flash-preview"
TEMPERATURE = 0.1
MAX_TOKENS = 512
SAMPLE_SIZE = 300
RESULTS_FILE = Path(__file__).parent / "gemini_validation_results.json"

# Band config: target samples per band
BAND_CONFIG = {
    "0-2": (0, 2, 60),
    "2-4": (2, 4, 60),
    "4-6": (4, 6, 60),
    "6-8": (6, 8, 60),
    "8-10": (8, 10, 60),
}


def load_data():
    """Load scores and build prompt lookup from batch input files."""
    base = Path(__file__).parent

    # Load scores
    scores = json.load(open(base / "scores_v2.json"))
    print(f"Loaded {len(scores)} scores")

    # Build prompt lookup from batch input files
    prompts = {}
    for fname in ["batch_input.jsonl", "batch_input_expand.jsonl"]:
        fpath = base / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    d = json.loads(line)
                    prompts[d["custom_id"]] = d["params"]["messages"]
            print(f"Loaded prompts from {fname}: {len(prompts)} total")

    return scores, prompts


def stratified_sample(scores, prompts):
    """Sample SAMPLE_SIZE examples stratified across score bands."""
    bands = defaultdict(list)
    for s in scores:
        cid = s.get("custom_id") or f"{s['article_id']}__{s['profile_id']}"
        if cid not in prompts:
            continue
        sc = s["score"]
        if sc < 2: bands["0-2"].append(s)
        elif sc < 4: bands["2-4"].append(s)
        elif sc < 6: bands["4-6"].append(s)
        elif sc < 8: bands["6-8"].append(s)
        else: bands["8-10"].append(s)

    sampled = []
    for band_name, (lo, hi, target) in BAND_CONFIG.items():
        pool = bands[band_name]
        n = min(target, len(pool))
        sampled.extend(random.sample(pool, n))
        print(f"Band {band_name}: sampled {n}/{len(pool)}")

    random.shuffle(sampled)
    print(f"Total sampled: {len(sampled)}")
    return sampled


SCORE_FIRST_SUFFIX = (
    "\n\n---\nIMPORTANT: Your response MUST start with the score on the very first line "
    "in this exact format:\nSCORE: X.X\n"
    "Then provide a brief reason on the next line starting with REASON:. "
    "Do NOT write any text before the SCORE line."
)


def messages_to_prompt(messages):
    """Convert OpenAI-format messages to a single prompt string for Gemini.
    Appends instruction to output SCORE first to avoid truncation issues."""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(content)
    return "\n\n".join(parts) + SCORE_FIRST_SUFFIX


def call_gemini(client, prompt, retries=3):
    """Call Gemini 3 Flash API with retries."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={"temperature": TEMPERATURE, "max_output_tokens": MAX_TOKENS},
            )
            return response.text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "rate" in err_str.lower():
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            elif attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  Retry {attempt+1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def parse_score(text):
    """Extract SCORE: X.X from response text."""
    if not text:
        return None
    # Try standard format
    m = re.search(r'SCORE:\s*([\d.]+)', text)
    if m:
        return float(m.group(1))
    # Try **Score:** markdown format
    m = re.search(r'\*?\*?Score\*?\*?[:\s]+([\d.]+)', text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Try Chinese format
    m = re.search(r'评分[：:]\s*([\d.]+)', text)
    if m:
        return float(m.group(1))
    # Try bare number on its own line (some models just output "7.5")
    m = re.search(r'^\s*([\d]+\.[\d]+)\s*$', text, re.MULTILINE)
    if m:
        return float(m.group(1))
    return None


def run_validation():
    random.seed(42)

    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY not found in environment or .env file")
        sys.exit(1)

    client = genai.Client(api_key=GOOGLE_API_KEY)
    print(f"Using model: {MODEL}")

    scores, prompts = load_data()
    samples = stratified_sample(scores, prompts)

    results = []
    parse_failures = 0
    errors = 0

    for i, sample in enumerate(samples):
        cid = sample.get("custom_id") or f"{sample['article_id']}__{sample['profile_id']}"
        opus_score = sample["score"]
        messages = prompts[cid]
        prompt = messages_to_prompt(messages)

        try:
            response = call_gemini(client, prompt)
            gem_score = parse_score(response)

            if gem_score is None:
                parse_failures += 1
                resp_preview = (response or "")[:100]
                print(f"[{i+1}/{len(samples)}] PARSE FAIL | Opus: {opus_score} | Response: {resp_preview}")
                results.append({
                    "custom_id": cid,
                    "opus_score": opus_score,
                    "gemini_score": None,
                    "parse_failure": True,
                    "response": (response or "")[:500],
                })
            else:
                diff = abs(gem_score - opus_score)
                results.append({
                    "custom_id": cid,
                    "opus_score": opus_score,
                    "gemini_score": gem_score,
                    "parse_failure": False,
                    "abs_error": diff,
                })
                valid = [r for r in results if not r["parse_failure"]]
                running_mae = sum(r["abs_error"] for r in valid) / len(valid) if valid else 0
                print(f"[{i+1}/{len(samples)}] Gem: {gem_score:.1f} | Opus: {opus_score:.1f} | Diff: {diff:.1f} | MAE: {running_mae:.3f}")

        except Exception as e:
            errors += 1
            print(f"[{i+1}/{len(samples)}] ERROR: {e}")
            results.append({
                "custom_id": cid,
                "opus_score": opus_score,
                "gemini_score": None,
                "parse_failure": True,
                "error": str(e),
            })
            if "429" in str(e) or "rate" in str(e).lower():
                print("Rate limited, waiting 15s...")
                time.sleep(15)
            continue

        # Small delay to avoid rate limits (free tier)
        time.sleep(1.0)

    # Save raw results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Compute metrics
    compute_metrics(results)


def compute_metrics(results):
    """Compute and print validation metrics."""
    valid = [r for r in results if not r["parse_failure"]]
    failed = [r for r in results if r["parse_failure"]]

    print(f"\n{'='*60}")
    print(f"GEMINI 3 FLASH VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")
    print(f"Parse failures: {len(failed)} ({100*len(failed)/len(results):.1f}%)")
    print(f"Valid comparisons: {len(valid)}")

    if not valid:
        print("No valid results!")
        return

    # Overall MAE
    errors_list = [r["abs_error"] for r in valid]
    mae = sum(errors_list) / len(errors_list)
    print(f"\nOverall MAE vs Opus: {mae:.3f}")

    # Median absolute error
    sorted_errors = sorted(errors_list)
    median_ae = sorted_errors[len(sorted_errors) // 2]
    print(f"Median AE: {median_ae:.3f}")

    # Within-X accuracy
    within_05 = sum(1 for e in errors_list if e <= 0.5) / len(errors_list)
    within_1 = sum(1 for e in errors_list if e <= 1.0) / len(errors_list)
    within_2 = sum(1 for e in errors_list if e <= 2.0) / len(errors_list)
    print(f"Within 0.5: {100*within_05:.1f}%")
    print(f"Within 1.0: {100*within_1:.1f}%")
    print(f"Within 2.0: {100*within_2:.1f}%")

    # Per-band MAE
    print(f"\nPer-band MAE:")
    band_results = defaultdict(list)
    for r in valid:
        sc = r["opus_score"]
        if sc < 2: band_results["0-2"].append(r)
        elif sc < 4: band_results["2-4"].append(r)
        elif sc < 6: band_results["4-6"].append(r)
        elif sc < 8: band_results["6-8"].append(r)
        else: band_results["8-10"].append(r)

    for band in ["0-2", "2-4", "4-6", "6-8", "8-10"]:
        br = band_results[band]
        if br:
            band_mae = sum(r["abs_error"] for r in br) / len(br)
            band_bias = sum(r["gemini_score"] - r["opus_score"] for r in br) / len(br)
            print(f"  {band}: MAE={band_mae:.3f}, Bias={band_bias:+.3f}, n={len(br)}")

    # Spearman correlation
    try:
        from scipy.stats import spearmanr
        opus = [r["opus_score"] for r in valid]
        gem = [r["gemini_score"] for r in valid]
        corr, pval = spearmanr(opus, gem)
        print(f"\nSpearman correlation: {corr:.3f} (p={pval:.2e})")
    except ImportError:
        opus = [r["opus_score"] for r in valid]
        gem = [r["gemini_score"] for r in valid]
        n = len(opus)

        def rank(arr):
            indexed = sorted(enumerate(arr), key=lambda x: x[1])
            ranks = [0] * n
            for rank_val, (orig_idx, _) in enumerate(indexed):
                ranks[orig_idx] = rank_val + 1
            return ranks

        r_opus = rank(opus)
        r_gem = rank(gem)
        d_sq = sum((a - b) ** 2 for a, b in zip(r_opus, r_gem))
        rho = 1 - (6 * d_sq) / (n * (n**2 - 1))
        print(f"\nSpearman correlation: {rho:.3f}")

    # Direction agreement (both high or both low relative to 5.0)
    direction_agree = sum(
        1 for r in valid
        if (r["opus_score"] >= 5.0) == (r["gemini_score"] >= 5.0)
    ) / len(valid)
    print(f"Direction agreement (>5 vs <5): {100*direction_agree:.1f}%")

    # Score clustering analysis
    print(f"\nGemini score clustering (top 15 most common scores):")
    score_counts = defaultdict(list)
    for r in valid:
        # Round to nearest 0.5 for clustering
        rounded = round(r["gemini_score"] * 2) / 2
        score_counts[rounded].append(r["opus_score"])

    sorted_clusters = sorted(score_counts.items(), key=lambda x: -len(x[1]))[:15]
    for gem_sc, opus_scores in sorted_clusters:
        avg_opus = sum(opus_scores) / len(opus_scores)
        print(f"  Gem {gem_sc:4.1f}: n={len(opus_scores):3d}, avg_opus={avg_opus:.2f}")

    # Worst outliers
    print(f"\nTop 10 worst disagreements:")
    worst = sorted(valid, key=lambda r: r["abs_error"], reverse=True)[:10]
    for r in worst:
        print(f"  {r['custom_id'][:40]:40s} Opus={r['opus_score']:.1f} Gem={r['gemini_score']:.1f} Diff={r['abs_error']:.1f}")

    # Verdict
    print(f"\n{'='*60}")
    if mae < 1.0:
        print(f"VERDICT: VIABLE - MAE {mae:.3f} < 1.0 threshold")
        print(f"Gemini 3 Flash can replace Opus for bulk scoring")
    else:
        print(f"VERDICT: NOT VIABLE (raw) - MAE {mae:.3f} >= 1.0 threshold")
        print(f"Consider calibration or hybrid approach")
    print(f"{'='*60}")

    # DeepSeek comparison
    print(f"\n--- Comparison with DeepSeek V3.2 ---")
    print(f"{'Metric':<30s} {'DeepSeek':>10s} {'Gemini':>10s}")
    print(f"{'MAE vs Opus':<30s} {'1.521':>10s} {mae:>10.3f}")
    print(f"{'Within 1.0':<30s} {'55.3%':>10s} {100*within_1:>9.1f}%")
    print(f"{'Within 2.0':<30s} {'74.7%':>10s} {100*within_2:>9.1f}%")
    print(f"{'Direction agreement':<30s} {'80.3%':>10s} {100*direction_agree:>9.1f}%")


if __name__ == "__main__":
    print = functools.partial(print, flush=True)
    run_validation()
