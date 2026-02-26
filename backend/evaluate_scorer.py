"""
StratOS Scorer Evaluation — Cross-reference live model vs Opus ground truth.

Usage:
    python3 evaluate_scorer.py                    # Evaluate current model against V2 eval set
    python3 evaluate_scorer.py --limit 50         # Quick test with 50 samples
    python3 evaluate_scorer.py --model stratos-scorer-v1   # Evaluate a specific model
    python3 evaluate_scorer.py --export drift.jsonl        # Export disagreements for training

This script scores the V2 evaluation set (2,048 Opus-scored samples) using the
live Ollama model and computes comparison metrics.  Use it to:
  1. Validate a newly deployed model before trusting it in production
  2. Detect model drift over time (re-run periodically)
  3. Identify weak profiles or score bands for targeted retraining
  4. Export disagreements as V3 training signal
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_scorer")

EVAL_FILE = Path(__file__).parent / "data" / "v2_pipeline" / "eval_v2.jsonl"
OLLAMA_HOST = "http://localhost:11434"
SCORE_RE = re.compile(r"SCORE:\s*(\d+\.?\d*)")


def load_eval_set(path: Path, limit: int = 0) -> List[dict]:
    """Load eval samples from JSONL."""
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            # Extract the ground truth score from the assistant message
            assistant_msg = d["messages"][-1]["content"]
            m = SCORE_RE.search(assistant_msg)
            if not m:
                continue
            d["ground_truth_score"] = float(m.group(1))
            # Extract profile from system prompt
            sys_msg = d["messages"][0]["content"]
            role_m = re.search(r"relevance scorer for (?:a |an )?(.+?) in (.+?)\.", sys_msg)
            d["profile_role"] = role_m.group(1) if role_m else "unknown"
            d["profile_location"] = role_m.group(2) if role_m else "unknown"
            samples.append(d)
            if limit and len(samples) >= limit:
                break
    return samples


def score_with_ollama(messages: List[dict], model: str, host: str) -> Tuple[float, str]:
    """Score a single sample using the Ollama model.  Returns (score, raw_text)."""
    try:
        resp = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": messages[:2],  # system + user only
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=120,
        )
        if resp.status_code != 200:
            return -1.0, f"HTTP {resp.status_code}"
        raw = resp.json().get("message", {}).get("content", "")
        # Strip think blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        m = SCORE_RE.search(raw)
        if m:
            return float(m.group(1)), raw
        return -1.0, raw
    except Exception as e:
        return -1.0, str(e)


def compute_metrics(results: List[dict]) -> dict:
    """Compute evaluation metrics from scored results."""
    valid = [r for r in results if r["predicted_score"] >= 0]
    if not valid:
        return {"error": "No valid predictions"}

    # Basic metrics
    n = len(valid)
    deltas = [r["predicted_score"] - r["ground_truth_score"] for r in valid]
    abs_deltas = [abs(d) for d in deltas]
    mae = sum(abs_deltas) / n
    mse = sum(d * d for d in deltas) / n
    rmse = mse ** 0.5

    # Direction accuracy: does the model agree on relevant (>=5) vs noise (<5)?
    direction_correct = sum(
        1 for r in valid
        if (r["predicted_score"] >= 5.0) == (r["ground_truth_score"] >= 5.0)
    )
    direction_acc = direction_correct / n

    # Spearman rank correlation
    gt_ranks = _rank([r["ground_truth_score"] for r in valid])
    pred_ranks = _rank([r["predicted_score"] for r in valid])
    spearman = _spearman(gt_ranks, pred_ranks)

    # Score band breakdown
    bands = {"noise": (0, 2.5), "tangential": (2.5, 4.5), "moderate": (4.5, 6.5),
             "high": (6.5, 8.5), "critical": (8.5, 10.1)}
    band_metrics = {}
    for band_name, (lo, hi) in bands.items():
        band_items = [r for r in valid if lo <= r["ground_truth_score"] < hi]
        if band_items:
            band_mae = sum(abs(r["predicted_score"] - r["ground_truth_score"]) for r in band_items) / len(band_items)
            band_metrics[band_name] = {"count": len(band_items), "mae": round(band_mae, 3)}

    # Per-profile breakdown
    profiles = {}
    for r in valid:
        key = r["profile_role"]
        if key not in profiles:
            profiles[key] = []
        profiles[key].append(r)

    profile_metrics = {}
    for role, items in sorted(profiles.items()):
        p_mae = sum(abs(r["predicted_score"] - r["ground_truth_score"]) for r in items) / len(items)
        p_dir = sum(1 for r in items if (r["predicted_score"] >= 5.0) == (r["ground_truth_score"] >= 5.0)) / len(items)
        profile_metrics[role] = {
            "count": len(items),
            "mae": round(p_mae, 3),
            "direction_acc": round(p_dir, 4),
        }

    # Worst disagreements
    worst = sorted(valid, key=lambda r: abs(r["predicted_score"] - r["ground_truth_score"]), reverse=True)[:10]

    return {
        "total_samples": n,
        "parse_failures": len(results) - n,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "direction_accuracy": round(direction_acc, 4),
        "spearman_rho": round(spearman, 4),
        "agreement_pct_1pt": round(100 * sum(1 for d in abs_deltas if d <= 1.0) / n, 1),
        "agreement_pct_2pt": round(100 * sum(1 for d in abs_deltas if d <= 2.0) / n, 1),
        "band_breakdown": band_metrics,
        "profile_breakdown": profile_metrics,
        "worst_disagreements": [
            {
                "profile": r["profile_role"],
                "predicted": r["predicted_score"],
                "ground_truth": r["ground_truth_score"],
                "delta": round(r["predicted_score"] - r["ground_truth_score"], 2),
                "title": r.get("title", ""),
            }
            for r in worst
        ],
    }


def _rank(values):
    """Compute ranks (1-based, average ties)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _spearman(ranks_a, ranks_b):
    """Compute Spearman rank correlation."""
    n = len(ranks_a)
    if n < 2:
        return 0.0
    mean_a = sum(ranks_a) / n
    mean_b = sum(ranks_b) / n
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b)) / n
    std_a = (sum((a - mean_a) ** 2 for a in ranks_a) / n) ** 0.5
    std_b = (sum((b - mean_b) ** 2 for b in ranks_b) / n) ** 0.5
    if std_a == 0 or std_b == 0:
        return 0.0
    return cov / (std_a * std_b)


def main():
    parser = argparse.ArgumentParser(description="Evaluate StratOS scorer against Opus ground truth")
    parser.add_argument("--model", default=None, help="Ollama model to evaluate (default: from config.yaml)")
    parser.add_argument("--host", default=OLLAMA_HOST, help="Ollama host URL")
    parser.add_argument("--eval-file", default=str(EVAL_FILE), help="Path to eval JSONL")
    parser.add_argument("--limit", type=int, default=0, help="Max samples to evaluate (0=all)")
    parser.add_argument("--export", default=None, help="Export disagreements (delta>=2) to JSONL for V3 training")
    parser.add_argument("--output", default="eval_report.json", help="Output report path")
    args = parser.parse_args()

    # Resolve model
    model = args.model
    if not model:
        try:
            import yaml
            with open(Path(__file__).parent / "config.yaml") as f:
                cfg = yaml.safe_load(f)
            model = cfg.get("scoring", {}).get("model", "stratos-scorer-v2")
        except Exception:
            model = "stratos-scorer-v2"
    logger.info(f"Evaluating model: {model}")

    # Load eval set
    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        logger.error(f"Eval file not found: {eval_path}")
        sys.exit(1)
    samples = load_eval_set(eval_path, limit=args.limit)
    logger.info(f"Loaded {len(samples)} eval samples")

    # Check Ollama availability
    try:
        resp = requests.get(f"{args.host}/api/tags", timeout=5)
        if resp.status_code != 200:
            logger.error("Ollama not reachable")
            sys.exit(1)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(model in m for m in models):
            logger.error(f"Model '{model}' not found in Ollama. Available: {models}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Cannot reach Ollama: {e}")
        sys.exit(1)

    # Score each sample
    results = []
    start_time = time.time()
    for i, sample in enumerate(samples):
        # Extract title from user message
        user_msg = sample["messages"][1]["content"]
        title_m = re.search(r"Title:\s*(.+?)(?:\n|$)", user_msg)
        title = title_m.group(1) if title_m else ""

        score, raw = score_with_ollama(sample["messages"], model, args.host)
        result = {
            "ground_truth_score": sample["ground_truth_score"],
            "predicted_score": score,
            "profile_role": sample["profile_role"],
            "title": title,
            "raw_output": raw[:200],
        }
        results.append(result)

        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate if rate > 0 else 0
            valid = sum(1 for r in results if r["predicted_score"] >= 0)
            logger.info(f"  [{i+1}/{len(samples)}] {rate:.1f} samples/s, ETA {eta:.0f}s, valid={valid}")

    elapsed = time.time() - start_time
    logger.info(f"Scoring complete: {len(results)} samples in {elapsed:.1f}s ({len(results)/elapsed:.1f}/s)")

    # Compute metrics
    metrics = compute_metrics(results)
    metrics["model"] = model
    metrics["eval_file"] = str(eval_path)
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Print report
    print("\n" + "=" * 60)
    print(f"SCORER EVALUATION REPORT — {model}")
    print("=" * 60)
    print(f"Samples:            {metrics['total_samples']}")
    print(f"Parse failures:     {metrics['parse_failures']}")
    print(f"MAE:                {metrics['mae']}")
    print(f"RMSE:               {metrics['rmse']}")
    print(f"Direction accuracy: {metrics['direction_accuracy']:.1%}")
    print(f"Spearman rho:       {metrics['spearman_rho']}")
    print(f"Within 1.0 point:   {metrics['agreement_pct_1pt']}%")
    print(f"Within 2.0 points:  {metrics['agreement_pct_2pt']}%")
    print(f"\nScore Band Breakdown:")
    for band, data in metrics.get("band_breakdown", {}).items():
        print(f"  {band:12s}: n={data['count']:4d}, MAE={data['mae']:.3f}")
    print(f"\nPer-Profile MAE (top 5 worst):")
    sorted_profiles = sorted(metrics.get("profile_breakdown", {}).items(), key=lambda x: x[1]["mae"], reverse=True)
    for role, data in sorted_profiles[:5]:
        print(f"  {role[:50]:50s}: MAE={data['mae']:.3f} dir={data['direction_acc']:.1%} (n={data['count']})")
    print(f"\nWorst Disagreements:")
    for w in metrics.get("worst_disagreements", [])[:5]:
        print(f"  delta={w['delta']:+.1f}  pred={w['predicted']:.1f} gt={w['ground_truth']:.1f}  [{w['profile'][:30]}] {w['title'][:60]}")

    # Save report
    report_path = Path(args.output)
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    # Export disagreements for V3 training
    if args.export:
        disagreements = [
            r for r in results
            if r["predicted_score"] >= 0 and abs(r["predicted_score"] - r["ground_truth_score"]) >= 2.0
        ]
        if disagreements:
            export_path = Path(args.export)
            with open(export_path, "w") as f:
                for d in disagreements:
                    f.write(json.dumps(d) + "\n")
            logger.info(f"Exported {len(disagreements)} disagreements to {export_path}")
        else:
            logger.info("No disagreements >= 2.0 found")

    # V3 trigger recommendation
    print(f"\n{'=' * 60}")
    if metrics["mae"] > 0.6:
        print("RECOMMENDATION: V3 training warranted (MAE > 0.6)")
    elif metrics["direction_accuracy"] < 0.95:
        print("RECOMMENDATION: V3 training warranted (direction accuracy < 95%)")
    else:
        print("RECOMMENDATION: V2 performing well, V3 not needed yet")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
