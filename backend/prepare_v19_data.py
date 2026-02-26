#!/usr/bin/env python3
"""
StratOS v19 — Training Data Preparation (Phase 4)
===================================================

Takes validated CoT results from distillation and prepares training data:
1. Formats into messages format (system/user/assistant)
2. Assigns per-sample loss weights based on score band (no upsampling)
3. Sorts by curriculum difficulty (easy->hard)
4. Splits 90/10 train/eval (stratified by profile + score band)
5. Reports statistics

Usage:
    python prepare_v19_data.py                    # Default: loss weighting, no upsampling
    python prepare_v19_data.py --moderate-upsample  # Fallback: 5x cap upsampling
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

BACKEND_DIR = Path(__file__).parent
DATA_DIR = BACKEND_DIR / "data"

VALIDATED_PATH = DATA_DIR / "v19_validated_cot.jsonl"
TRAIN_OUTPUT = DATA_DIR / "training_v19_cot.jsonl"
EVAL_OUTPUT = DATA_DIR / "eval_v19_cot.jsonl"


def get_score_band(score: float) -> str:
    """Classify score into a band."""
    if score >= 9.0:
        return "critical"
    elif score >= 7.0:
        return "high"
    elif score >= 5.5:
        return "moderate"
    elif score >= 2.0:
        return "low"
    else:
        return "noise"


def load_validated_data() -> list:
    """Load validated CoT results."""
    data = []
    with open(VALIDATED_PATH) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_as_messages(item: dict) -> dict:
    """Convert validated CoT item to messages format for training."""
    return {
        "messages": [
            {"role": "system", "content": item["system_prompt"]},
            {"role": "user", "content": item.get("article_text", "")},
            {"role": "assistant", "content": item["assistant_content"]},
        ]
    }


def add_sample_weights(data: list) -> list:
    """Assign per-sample loss weights based on score band rarity.

    Instead of upsampling (which causes overfitting when rare bands have
    few unique examples), we keep the natural distribution and weight the
    loss function so rare examples contribute more to gradient updates.

    Weights are calibrated so that each band contributes roughly equally
    to total training loss despite different sample counts.
    """
    BAND_WEIGHTS = {
        "critical": 10.0,   # 51 examples — rarest, most important
        "high": 8.0,        # 64 examples
        "moderate": 8.0,    # 64 examples
        "low": 1.5,         # 654 examples
        "noise": 1.0,       # 4846 examples — most common
    }

    bands = defaultdict(list)
    for item in data:
        band = get_score_band(item["score"])
        bands[band].append(item)
        item["sample_weight"] = BAND_WEIGHTS[band]

    total = len(data)
    print(f"\nScore band distribution (natural, with loss weights):")
    for band in ["critical", "high", "moderate", "low", "noise"]:
        count = len(bands.get(band, []))
        pct = count / total * 100 if total > 0 else 0
        weight = BAND_WEIGHTS[band]
        effective = count * weight
        print(f"  {band:12s}: {count:5d} ({pct:.1f}%)  weight={weight:.1f}  effective={effective:.0f}")

    total_effective = sum(len(bands.get(b, [])) * w for b, w in BAND_WEIGHTS.items())
    print(f"\n  Total examples: {total}")
    print(f"  Total effective weight: {total_effective:.0f}")

    return data


def moderate_upsample(data: list, max_factor: int = 5) -> list:
    """Fallback: upsample underrepresented bands with a cap of max_factor per band.

    This keeps repetition manageable (max 5x) instead of 60x.
    Result: ~255 critical, ~320 high, ~320 moderate, ~654 low, ~4846 noise = ~6,395
    """
    bands = defaultdict(list)
    for item in data:
        band = get_score_band(item["score"])
        bands[band].append(item)

    total = len(data)
    print(f"\nScore band distribution (before moderate upsampling, {max_factor}x cap):")
    for band in ["critical", "high", "moderate", "low", "noise"]:
        count = len(bands.get(band, []))
        pct = count / total * 100 if total > 0 else 0
        print(f"  {band:12s}: {count:5d} ({pct:.1f}%)")

    result = []
    for band in ["critical", "high", "moderate", "low", "noise"]:
        items = bands.get(band, [])
        if not items:
            continue

        target = min(len(items) * max_factor, len(items))
        # Only upsample if band is small
        target = len(items) * max_factor

        if len(items) >= target:
            result.extend(items)
        else:
            result.extend(items)
            shortfall = target - len(items)
            result.extend(random.choices(items, k=shortfall))

    # Add sample_weight = 1.0 for all (weights are uniform since we upsampled)
    for item in result:
        item["sample_weight"] = 1.0

    print(f"\nAfter moderate upsampling ({max_factor}x cap): {len(result)} examples")
    band_counts = Counter(get_score_band(i["score"]) for i in result)
    for band in ["critical", "high", "moderate", "low", "noise"]:
        count = band_counts.get(band, 0)
        pct = count / len(result) * 100 if result else 0
        print(f"  {band:12s}: {count:5d} ({pct:.1f}%)")

    return result


def curriculum_sort(data: list) -> list:
    """Sort by difficulty: easy (far from 5.0) first, hard (near 5.0) last."""
    return sorted(data, key=lambda x: abs(x["score"] - 5.0), reverse=True)


def stratified_split(data: list, eval_ratio: float = 0.10):
    """Split train/eval stratified by profile AND score band."""
    # Group by (profile, band)
    groups = defaultdict(list)
    for item in data:
        key = (item["profile_id"], get_score_band(item["score"]))
        groups[key].append(item)

    train, eval_ = [], []
    for key, items in groups.items():
        n_eval = max(1, int(len(items) * eval_ratio))
        random.shuffle(items)
        eval_.extend(items[:n_eval])
        train.extend(items[n_eval:])

    return train, eval_


def main():
    parser = argparse.ArgumentParser(description="StratOS v19 — Training Data Preparation")
    parser.add_argument("--moderate-upsample", action="store_true",
                        help="Use moderate upsampling (5x cap) instead of loss weighting")
    parser.add_argument("--max-upsample-factor", type=int, default=5,
                        help="Maximum upsampling factor per band (default: 5)")
    args = parser.parse_args()

    use_loss_weighting = not args.moderate_upsample

    print("=" * 60)
    print("StratOS v19 — Training Data Preparation")
    print(f"Strategy: {'Loss weighting (no upsampling)' if use_loss_weighting else f'Moderate upsampling ({args.max_upsample_factor}x cap)'}")
    print("=" * 60)

    # Load validated CoT data
    print(f"\nLoading validated data from {VALIDATED_PATH}...")
    data = load_validated_data()
    print(f"  Loaded {len(data)} validated examples")

    if not data:
        print("ERROR: No validated data found. Run distill_v19_cot.py --collect first.")
        return

    # Load article texts (need to re-associate since validated data may not have them)
    polar_texts = {}
    polar_path = DATA_DIR / "polarizing_articles_v19.json"
    if polar_path.exists():
        with open(polar_path) as f:
            for a in json.load(f):
                key = f"polar_{a['id']}"
                polar_texts[key] = f"Title: {a['headline']}\nContent: {a['summary']}"

    # Load multi-profile article texts from batch requests
    multi_texts = {}
    batch_req_path = DATA_DIR / "v19_batch_requests.jsonl"
    if batch_req_path.exists():
        with open(batch_req_path) as f:
            for line in f:
                req = json.loads(line)
                article_id = req["custom_id"].split("__")[0]
                if article_id not in multi_texts:
                    multi_texts[article_id] = req["params"]["messages"][0]["content"]

    # Merge text into validated data
    all_texts = {**polar_texts, **multi_texts}
    missing_text = 0
    for item in data:
        article_id = item.get("article_id", "")
        if article_id in all_texts:
            item["article_text"] = all_texts[article_id]
        else:
            item["article_text"] = ""
            missing_text += 1

    if missing_text > 0:
        print(f"  WARNING: {missing_text} examples missing article text")

    # Stats
    print(f"\nProfile distribution:")
    profile_counts = Counter(i["profile_id"] for i in data)
    for pid, count in sorted(profile_counts.items()):
        print(f"  {pid:35s}: {count}")

    print(f"\nThink block word count by tier:")
    noise_wc = [i["think_word_count"] for i in data if i["score"] < 2.0]
    standard_wc = [i["think_word_count"] for i in data if 2.0 <= i["score"] < 9.0]
    critical_wc = [i["think_word_count"] for i in data if i["score"] >= 9.0]
    if noise_wc:
        print(f"  Noise (< 2.0):     avg {sum(noise_wc)/len(noise_wc):.0f} words ({len(noise_wc)} examples)")
    if standard_wc:
        print(f"  Standard (2-8.9):  avg {sum(standard_wc)/len(standard_wc):.0f} words ({len(standard_wc)} examples)")
    if critical_wc:
        print(f"  Critical (9-10):   avg {sum(critical_wc)/len(critical_wc):.0f} words ({len(critical_wc)} examples)")

    # Count contrastive pairs (same article, different profiles, score diff > 3.0)
    article_scores = defaultdict(list)
    for item in data:
        article_scores[item["article_id"]].append((item["profile_id"], item["score"]))

    contrastive_pairs = 0
    for article_id, scores in article_scores.items():
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                if abs(scores[i][1] - scores[j][1]) > 3.0:
                    contrastive_pairs += 1

    print(f"\nContrastive pairs (same article, score diff > 3.0): {contrastive_pairs}")

    # Apply rebalancing strategy
    if use_loss_weighting:
        data = add_sample_weights(data)
    else:
        data = moderate_upsample(data, max_factor=args.max_upsample_factor)

    # Stratified split
    train_data, eval_data = stratified_split(data)

    # Curriculum sort (train only — eval stays shuffled)
    train_data = curriculum_sort(train_data)

    print(f"\nFinal split:")
    print(f"  Train: {len(train_data)}")
    print(f"  Eval:  {len(eval_data)}")

    # Save as messages format with sample_weight
    print(f"\nSaving training data...")
    with open(TRAIN_OUTPUT, "w") as f:
        for item in train_data:
            example = {
                "messages": [
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["article_text"]},
                    {"role": "assistant", "content": item["assistant_content"]},
                ],
                "sample_weight": item.get("sample_weight", 1.0),
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    with open(EVAL_OUTPUT, "w") as f:
        for item in eval_data:
            example = {
                "messages": [
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["article_text"]},
                    {"role": "assistant", "content": item["assistant_content"]},
                ],
                "sample_weight": item.get("sample_weight", 1.0),
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"  Train saved to: {TRAIN_OUTPUT}")
    print(f"  Eval saved to:  {EVAL_OUTPUT}")

    # Final stats
    print(f"\n{'='*60}")
    print(f"FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Total examples: {len(train_data) + len(eval_data)}")
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    train_bands = Counter(get_score_band(i["score"]) for i in train_data)
    eval_bands = Counter(get_score_band(i["score"]) for i in eval_data)

    print(f"\nTrain score bands:")
    for band in ["critical", "high", "moderate", "low", "noise"]:
        count = train_bands.get(band, 0)
        weight = {
            "critical": 10.0, "high": 8.0, "moderate": 8.0,
            "low": 1.5, "noise": 1.0
        }[band] if use_loss_weighting else 1.0
        effective = count * weight
        print(f"  {band:12s}: {count:5d}  weight={weight:.1f}  effective={effective:.0f}")

    print(f"\nEval score bands:")
    for band in ["critical", "high", "moderate", "low", "noise"]:
        print(f"  {band:12s}: {eval_bands.get(band, 0)}")

    train_profiles = Counter(i["profile_id"] for i in train_data)
    print(f"\nTrain per-profile counts:")
    for pid, count in sorted(train_profiles.items()):
        print(f"  {pid:35s}: {count}")

    # Unique article diversity stats
    unique_articles = len(set(i.get("article_id", "") for i in train_data))
    print(f"\nUnique articles in train set: {unique_articles}")
    if use_loss_weighting:
        print(f"Duplication factor: 1.0x (no upsampling)")
    else:
        print(f"Duplication factor: {len(train_data) / unique_articles:.1f}x avg")

    print(f"\nDONE")


if __name__ == "__main__":
    main()
