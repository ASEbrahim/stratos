#!/usr/bin/env python3
"""
DeepSeek V3.2 Scoring Validation Test
Compare DeepSeek scores against Opus ground truth on 300 stratified examples.
"""

import json
import random
import re
import time
import sys
import os
import requests
from collections import defaultdict
from pathlib import Path

# Config
DEEPSEEK_API_KEY = "sk-0359d8db5bdb4918ae3cc176f1b869df"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
TEMPERATURE = 0.1
MAX_TOKENS = 512
SAMPLE_SIZE = 300
RESULTS_FILE = Path(__file__).parent / "deepseek_validation_results.json"

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
    # Group by band
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


def call_deepseek(messages, retries=3):
    """Call DeepSeek V3.2 API with retries."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                DEEPSEEK_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": messages,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  Retry {attempt+1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def parse_score(text):
    """Extract SCORE: X.X from response text."""
    # Try standard format
    m = re.search(r'SCORE:\s*([\d.]+)', text)
    if m:
        return float(m.group(1))
    # Try Chinese format
    m = re.search(r'评分[：:]\s*([\d.]+)', text)
    if m:
        return float(m.group(1))
    return None


def run_validation():
    random.seed(42)

    scores, prompts = load_data()
    samples = stratified_sample(scores, prompts)

    results = []
    parse_failures = 0
    errors = 0

    for i, sample in enumerate(samples):
        cid = sample.get("custom_id") or f"{sample['article_id']}__{sample['profile_id']}"
        opus_score = sample["score"]
        messages = prompts[cid]

        try:
            response = call_deepseek(messages)
            ds_score = parse_score(response)

            if ds_score is None:
                parse_failures += 1
                print(f"[{i+1}/{len(samples)}] PARSE FAIL | Opus: {opus_score} | Response: {response[:100]}", flush=True)
                results.append({
                    "custom_id": cid,
                    "opus_score": opus_score,
                    "deepseek_score": None,
                    "parse_failure": True,
                    "response": response[:500],
                })
            else:
                diff = abs(ds_score - opus_score)
                results.append({
                    "custom_id": cid,
                    "opus_score": opus_score,
                    "deepseek_score": ds_score,
                    "parse_failure": False,
                    "abs_error": diff,
                })
                valid = [r for r in results if not r["parse_failure"]]
                running_mae = sum(r["abs_error"] for r in valid) / len(valid) if valid else 0
                print(f"[{i+1}/{len(samples)}] DS: {ds_score:.1f} | Opus: {opus_score:.1f} | Diff: {diff:.1f} | MAE: {running_mae:.3f}", flush=True)

        except Exception as e:
            errors += 1
            print(f"[{i+1}/{len(samples)}] ERROR: {e}")
            results.append({
                "custom_id": cid,
                "opus_score": opus_score,
                "deepseek_score": None,
                "parse_failure": True,
                "error": str(e),
            })
            if "429" in str(e) or "rate" in str(e).lower():
                print("Rate limited, waiting 10s...")
                time.sleep(10)
            continue

        # Small delay to avoid rate limits
        time.sleep(0.3)

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
    print(f"DEEPSEEK V3.2 VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")
    print(f"Parse failures: {len(failed)} ({100*len(failed)/len(results):.1f}%)")
    print(f"Valid comparisons: {len(valid)}")

    if not valid:
        print("No valid results!")
        return

    # Overall MAE
    errors = [r["abs_error"] for r in valid]
    mae = sum(errors) / len(errors)
    print(f"\nOverall MAE vs Opus: {mae:.3f}")

    # Median absolute error
    sorted_errors = sorted(errors)
    median_ae = sorted_errors[len(sorted_errors) // 2]
    print(f"Median AE: {median_ae:.3f}")

    # Within-X accuracy
    within_05 = sum(1 for e in errors if e <= 0.5) / len(errors)
    within_1 = sum(1 for e in errors if e <= 1.0) / len(errors)
    within_2 = sum(1 for e in errors if e <= 2.0) / len(errors)
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
            band_bias = sum(r["deepseek_score"] - r["opus_score"] for r in br) / len(br)
            print(f"  {band}: MAE={band_mae:.3f}, Bias={band_bias:+.3f}, n={len(br)}")

    # Spearman correlation
    try:
        from scipy.stats import spearmanr
        opus = [r["opus_score"] for r in valid]
        ds = [r["deepseek_score"] for r in valid]
        corr, pval = spearmanr(opus, ds)
        print(f"\nSpearman correlation: {corr:.3f} (p={pval:.2e})")
    except ImportError:
        # Manual Spearman
        opus = [r["opus_score"] for r in valid]
        ds = [r["deepseek_score"] for r in valid]
        n = len(opus)

        def rank(arr):
            indexed = sorted(enumerate(arr), key=lambda x: x[1])
            ranks = [0] * n
            for rank_val, (orig_idx, _) in enumerate(indexed):
                ranks[orig_idx] = rank_val + 1
            return ranks

        r_opus = rank(opus)
        r_ds = rank(ds)
        d_sq = sum((a - b) ** 2 for a, b in zip(r_opus, r_ds))
        rho = 1 - (6 * d_sq) / (n * (n**2 - 1))
        print(f"\nSpearman correlation: {rho:.3f}")

    # Direction agreement (both high or both low relative to 5.0)
    direction_agree = sum(
        1 for r in valid
        if (r["opus_score"] >= 5.0) == (r["deepseek_score"] >= 5.0)
    ) / len(valid)
    print(f"Direction agreement (>5 vs <5): {100*direction_agree:.1f}%")

    # Worst outliers
    print(f"\nTop 10 worst disagreements:")
    worst = sorted(valid, key=lambda r: r["abs_error"], reverse=True)[:10]
    for r in worst:
        print(f"  {r['custom_id'][:30]:30s} Opus={r['opus_score']:.1f} DS={r['deepseek_score']:.1f} Diff={r['abs_error']:.1f}")

    # Verdict
    print(f"\n{'='*60}")
    if mae < 1.0:
        print(f"VERDICT: VIABLE - MAE {mae:.3f} < 1.0 threshold")
        print(f"DeepSeek V3.2 can replace Opus for bulk scoring")
        cost_36k = 36000 * 0.00001  # rough estimate
        print(f"Estimated cost for 36K examples: ~${cost_36k:.0f}-6")
    else:
        print(f"VERDICT: NOT VIABLE - MAE {mae:.3f} >= 1.0 threshold")
        print(f"DeepSeek V3.2 too noisy to replace Opus")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Unbuffered output
    import functools
    print = functools.partial(print, flush=True)
    run_validation()
