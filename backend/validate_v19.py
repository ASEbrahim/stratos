#!/usr/bin/env python3
"""
StratOS v19 — Phase 6 Validation Suite
=======================================
Scores all eval examples with stratos-scorer-v18, compares to Opus ground truth.
Reports: PSR, MAE, Spearman ρ, format compliance, think block depth.

Usage:
    python3 validate_v19.py                # Full validation (572 examples)
    python3 validate_v19.py --resume       # Resume from checkpoint
    python3 validate_v19.py --metrics-only # Recalculate from saved results
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from scipy import stats
import numpy as np

MODEL = "stratos-scorer-v18"
OLLAMA_URL = "http://localhost:11434/api/generate"
EVAL_FILE = Path("data/eval_v19_cot.jsonl")
RESULTS_FILE = Path("data/v19_validation_results.json")
CHECKPOINT_FILE = Path("data/v19_validation_checkpoint.json")


def load_eval_data():
    """Load eval examples and extract Opus ground truth scores."""
    data = []
    with open(EVAL_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            msgs = ex["messages"]
            system_prompt = msgs[0]["content"]
            user_content = msgs[1]["content"]
            assistant_content = msgs[2]["content"]

            # Extract Opus ground truth score from assistant content
            clean = re.sub(r'<think>.*?</think>', '', assistant_content, flags=re.DOTALL).strip()
            score_match = re.search(r'SCORE:\s*(\d+\.?\d*)', clean, re.IGNORECASE)
            opus_score = float(score_match.group(1)) if score_match else None

            # Extract think block from training data
            think_match = re.search(r'<think>(.*?)</think>', assistant_content, re.DOTALL)
            opus_think = think_match.group(1).strip() if think_match else ""

            # Extract profile_id from system prompt
            profile_id = "unknown"
            if "Computer Engineering" in system_prompt and "Kuwait" in system_prompt:
                profile_id = "kuwait_cpeg"
            elif "ER nurse" in system_prompt:
                profile_id = "texas_nurse"
            elif "investment analyst" in system_prompt and "London" in system_prompt:
                profile_id = "london_finance"
            elif "Automotive engineer" in system_prompt and "Munich" in system_prompt:
                profile_id = "munich_mecheng"
            elif "data science" in system_prompt and "Bangal" in system_prompt:
                profile_id = "bangalore_ds"
            elif "Cybersecurity" in system_prompt and "Washington" in system_prompt:
                profile_id = "dc_cybersec"
            elif "gaming" in system_prompt and "Seoul" in system_prompt:
                profile_id = "seoul_gamer_investor"
            elif "geopolitics" in system_prompt and "Ankara" in system_prompt:
                profile_id = "ankara_politics_physics"
            elif "space" in system_prompt and "Portland" in system_prompt:
                profile_id = "portland_retired_space"
            elif "fintech" in system_prompt and "Lagos" in system_prompt:
                profile_id = "lagos_student_entrepreneur"

            # Extract article_id from user content (hash for grouping)
            article_hash = hash(user_content[:200])

            data.append({
                "system_prompt": system_prompt,
                "user_content": user_content,
                "opus_score": opus_score,
                "opus_think_words": len(opus_think.split()) if opus_think else 0,
                "profile_id": profile_id,
                "article_hash": article_hash,
                "sample_weight": ex.get("sample_weight", 1.0),
            })
    return data


def call_scorer(system_prompt, article_text):
    """Call stratos-scorer-v18 via Ollama."""
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": article_text,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "num_predict": 512,
                "repeat_penalty": 1.1,
            }
        }, timeout=120)
        return resp.json().get("response", "")
    except Exception as e:
        return f"ERROR: {e}"


def parse_response(text):
    """Extract score, reason, and think block from model response."""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    think_block = think_match.group(1).strip() if think_match else ""
    think_words = len(think_block.split()) if think_block else 0

    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    score_match = re.search(r'SCORE:\s*(\d+\.?\d*)', clean, re.IGNORECASE)
    reason_match = re.search(r'REASON:\s*(.+)', clean, re.IGNORECASE | re.DOTALL)

    score = float(score_match.group(1)) if score_match else None
    reason = reason_match.group(1).strip() if reason_match else ""

    return {
        "score": score,
        "reason": reason,
        "think_words": think_words,
        "has_think": bool(think_match),
        "has_score": score is not None,
        "raw": text,
    }


def score_band(score):
    if score >= 9.0: return "critical"
    elif score >= 7.0: return "high"
    elif score >= 5.5: return "moderate"
    elif score >= 2.0: return "low"
    else: return "noise"


def compute_metrics(results):
    """Compute all validation metrics from scored results."""
    # Filter to successfully parsed examples
    valid = [r for r in results if r["model_score"] is not None and r["opus_score"] is not None]
    total = len(results)
    parsed = len(valid)

    print(f"\n{'='*70}")
    print(f"VALIDATION METRICS — {parsed}/{total} successfully scored")
    print(f"{'='*70}")

    if parsed < 10:
        print("ERROR: Too few valid results for metrics")
        return

    opus_scores = np.array([r["opus_score"] for r in valid])
    model_scores = np.array([r["model_score"] for r in valid])

    # 1. Format Compliance
    format_ok = sum(1 for r in results if r.get("has_score", False))
    think_ok = sum(1 for r in results if r.get("has_think", False))
    format_pct = format_ok / total * 100
    think_pct = think_ok / total * 100

    print(f"\n--- Format Compliance ---")
    print(f"  SCORE/REASON format: {format_ok}/{total} ({format_pct:.1f}%)")
    print(f"  Think tags present:  {think_ok}/{total} ({think_pct:.1f}%)")

    # 2. MAE
    mae = np.mean(np.abs(opus_scores - model_scores))
    print(f"\n--- Mean Absolute Error (MAE) ---")
    print(f"  MAE: {mae:.3f} (target: <1.5, acceptable: <2.5)")
    print(f"  Status: {'✓ PASS' if mae < 1.5 else '~ ACCEPTABLE' if mae < 2.5 else '✗ FAIL'}")

    # Per-band MAE
    print(f"\n  Per-band MAE:")
    for band in ["critical", "high", "moderate", "low", "noise"]:
        band_items = [r for r in valid if score_band(r["opus_score"]) == band]
        if band_items:
            band_mae = np.mean([abs(r["opus_score"] - r["model_score"]) for r in band_items])
            print(f"    {band:12s}: MAE={band_mae:.2f} ({len(band_items)} examples)")

    # 3. Spearman ρ
    spearman_r, spearman_p = stats.spearmanr(opus_scores, model_scores)
    print(f"\n--- Spearman Rank Correlation ---")
    print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"  Target: >0.80, acceptable: >0.70")
    print(f"  Status: {'✓ PASS' if spearman_r > 0.80 else '~ ACCEPTABLE' if spearman_r > 0.70 else '✗ FAIL'}")

    # Pearson for reference
    pearson_r, pearson_p = stats.pearsonr(opus_scores, model_scores)
    print(f"  Pearson r:  {pearson_r:.4f} (for reference)")

    # 4. PSR (Profile Sensitivity Rate)
    # Group by article, check if max-min score > 2.0 across profiles
    article_groups = defaultdict(list)
    for r in valid:
        article_groups[r["article_hash"]].append(r)

    multi_profile_articles = {k: v for k, v in article_groups.items() if len(v) >= 2}
    psr_pass = 0
    psr_total = len(multi_profile_articles)

    for article_hash, items in multi_profile_articles.items():
        model_max = max(r["model_score"] for r in items)
        model_min = min(r["model_score"] for r in items)
        if model_max - model_min > 2.0:
            psr_pass += 1

    psr = psr_pass / psr_total * 100 if psr_total > 0 else 0

    print(f"\n--- Profile Sensitivity Rate (PSR) ---")
    print(f"  Articles with 2+ profiles: {psr_total}")
    print(f"  Score gap >2.0: {psr_pass}/{psr_total} ({psr:.1f}%)")
    print(f"  Target: >80%, acceptable: >60%")
    print(f"  Status: {'✓ PASS' if psr > 80 else '~ ACCEPTABLE' if psr > 60 else '✗ FAIL'}")

    # 5. Think Block Depth Analysis
    print(f"\n--- Think Block Depth ---")
    for tier, (lo, hi), (min_w, max_w) in [
        ("noise", (0.0, 1.9), (15, 50)),
        ("standard", (2.0, 8.9), (80, 150)),
        ("critical", (9.0, 10.0), (120, 180)),
    ]:
        tier_items = [r for r in valid if lo <= r["model_score"] <= hi]
        if tier_items:
            words = [r.get("model_think_words", 0) for r in tier_items]
            median = np.median(words) if words else 0
            mean = np.mean(words) if words else 0
            print(f"  {tier:12s}: median={median:.0f} words, mean={mean:.0f}, n={len(tier_items)}")
            print(f"    Target: {min_w}-{max_w} words | {'✓' if min_w <= median <= max_w else '✗ (empty think blocks)'}")

    # 6. Per-Profile Breakdown
    print(f"\n--- Per-Profile Breakdown ---")
    profile_groups = defaultdict(list)
    for r in valid:
        profile_groups[r["profile_id"]].append(r)

    interest_profiles = {"seoul_gamer_investor", "ankara_politics_physics",
                         "portland_retired_space", "lagos_student_entrepreneur"}

    for pid in sorted(profile_groups.keys()):
        items = profile_groups[pid]
        if len(items) < 2:
            continue
        p_opus = [r["opus_score"] for r in items]
        p_model = [r["model_score"] for r in items]
        p_mae = np.mean(np.abs(np.array(p_opus) - np.array(p_model)))
        p_spearman, _ = stats.spearmanr(p_opus, p_model) if len(items) >= 5 else (0, 1)
        tag = " [interest]" if pid in interest_profiles else ""
        print(f"  {pid:35s}: MAE={p_mae:.2f}, ρ={p_spearman:.3f}, n={len(items)}{tag}")

    # 7. Score Distribution Comparison
    print(f"\n--- Score Distribution ---")
    print(f"  {'Band':12s} | {'Opus':>6s} | {'Model':>6s} | {'Delta':>6s}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    for band in ["critical", "high", "moderate", "low", "noise"]:
        opus_n = sum(1 for r in valid if score_band(r["opus_score"]) == band)
        model_n = sum(1 for r in valid if score_band(r["model_score"]) == band)
        print(f"  {band:12s} | {opus_n:6d} | {model_n:6d} | {model_n - opus_n:+6d}")

    # 8. Direction Accuracy (does model agree on high vs low?)
    direction_correct = sum(1 for r in valid
                           if (r["opus_score"] >= 5.0 and r["model_score"] >= 5.0) or
                              (r["opus_score"] < 5.0 and r["model_score"] < 5.0))
    direction_pct = direction_correct / parsed * 100
    print(f"\n--- Direction Accuracy (agree on ≥5 vs <5) ---")
    print(f"  {direction_correct}/{parsed} ({direction_pct:.1f}%)")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Format Compliance: {format_pct:.0f}% {'✓' if format_pct > 95 else '✗'}")
    print(f"  MAE:               {mae:.3f} {'✓' if mae < 1.5 else '~' if mae < 2.5 else '✗'}")
    print(f"  Spearman ρ:        {spearman_r:.4f} {'✓' if spearman_r > 0.80 else '~' if spearman_r > 0.70 else '✗'}")
    print(f"  PSR:               {psr:.1f}% {'✓' if psr > 80 else '~' if psr > 60 else '✗'}")
    print(f"  Think Blocks:      {'Empty (cosmetic)' if all(r.get('model_think_words', 0) == 0 for r in valid) else 'Present'}")
    print(f"  Direction:         {direction_pct:.1f}%")

    overall = "PASS" if (mae < 1.5 and spearman_r > 0.80 and psr > 80) else \
              "PARTIAL" if (mae < 2.5 and spearman_r > 0.70 and psr > 60) else "FAIL"
    print(f"\n  OVERALL: {overall}")

    return {
        "mae": float(mae),
        "spearman_r": float(spearman_r),
        "psr": float(psr),
        "format_compliance": float(format_pct),
        "direction_accuracy": float(direction_pct),
        "total_eval": total,
        "parsed": parsed,
        "overall": overall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--metrics-only", action="store_true", help="Recalculate metrics from saved results")
    args = parser.parse_args()

    # Load eval data
    print("Loading eval data...")
    eval_data = load_eval_data()
    print(f"  {len(eval_data)} eval examples")

    profiles = Counter(d["profile_id"] for d in eval_data)
    print(f"  Profiles: {dict(profiles)}")

    if args.metrics_only:
        if not RESULTS_FILE.exists():
            print("ERROR: No results file found. Run full validation first.")
            return
        results = json.load(open(RESULTS_FILE))
        compute_metrics(results)
        return

    # Load checkpoint if resuming
    scored = {}
    if args.resume and CHECKPOINT_FILE.exists():
        checkpoint = json.load(open(CHECKPOINT_FILE))
        scored = {r["idx"]: r for r in checkpoint}
        print(f"  Resuming from checkpoint: {len(scored)} already scored")

    # Score all eval examples
    results = []
    total = len(eval_data)
    start_time = time.time()
    errors = 0

    for i, item in enumerate(eval_data):
        if i in scored:
            results.append(scored[i])
            continue

        response = call_scorer(item["system_prompt"], item["user_content"])
        parsed = parse_response(response)

        result = {
            "idx": i,
            "profile_id": item["profile_id"],
            "opus_score": item["opus_score"],
            "model_score": parsed["score"],
            "model_think_words": parsed["think_words"],
            "has_think": parsed["has_think"],
            "has_score": parsed["has_score"],
            "article_hash": item["article_hash"],
            "sample_weight": item["sample_weight"],
        }
        results.append(result)

        if parsed["score"] is None:
            errors += 1

        # Progress
        if (i + 1) % 25 == 0 or i == total - 1:
            elapsed = time.time() - start_time
            rate = (i + 1 - len(scored)) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            valid_so_far = sum(1 for r in results if r["model_score"] is not None)
            print(f"  [{i+1}/{total}] scored={valid_so_far}, errors={errors}, "
                  f"rate={rate:.1f}/s, ETA={eta/60:.1f}min")

        # Checkpoint every 50
        if (i + 1) % 50 == 0:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump(results, f)

    # Save final results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Clean up checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    # Compute and report metrics
    metrics = compute_metrics(results)

    # Save metrics summary
    metrics_file = Path("data/v19_validation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
