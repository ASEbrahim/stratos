#!/usr/bin/env python3
"""
StratOS Multi-Model Scorer Comparison Pipeline

Compare multiple Ollama-hosted scorer models against Opus ground truth.

Usage:
    python3 compare_models.py --models stratos-scorer-v2 qwen3.5:9b
    python3 compare_models.py --models stratos-scorer-v2 qwen3.5:9b qwen3:8b --samples-per-band 10
    python3 compare_models.py --models stratos-scorer-v2 --limit 50
    python3 compare_models.py --models stratos-scorer-v2 qwen3.5:9b --output comparison.json

Loads the V2 eval set, performs stratified sampling across score bands, scores
each sample with each model, and prints a side-by-side comparison table with
per-model metrics, per-band breakdowns, and a final recommendation.
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_models")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_FILE = Path(__file__).parent / "data" / "v2_pipeline" / "eval_v2.jsonl"
OLLAMA_HOST = "http://localhost:11434"

SCORE_RE = re.compile(r"SCORE:\s*(\d+\.?\d*)")
THINK_SCORE_RE = re.compile(r"[Ff]inal\s+[Ss]core:\s*(\d+\.?\d*)")
PROFILE_RE = re.compile(r"relevance scorer for (?:a |an )?(.+?) in (.+?)\.")

BANDS = {
    "noise":      (0.0,  2.5),
    "tangential": (2.5,  4.5),
    "moderate":   (4.5,  6.5),
    "high":       (6.5,  8.5),
    "critical":   (8.5, 10.01),
}


# ---------------------------------------------------------------------------
# Data loading & sampling
# ---------------------------------------------------------------------------
def load_eval_set(path: Path) -> List[dict]:
    """Load all eval samples from JSONL, attaching ground truth and metadata."""
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            assistant_msg = d["messages"][-1]["content"]
            m = SCORE_RE.search(assistant_msg)
            if not m:
                continue
            d["ground_truth_score"] = float(m.group(1))
            sys_msg = d["messages"][0]["content"]
            role_m = PROFILE_RE.search(sys_msg)
            d["profile_role"] = role_m.group(1) if role_m else "unknown"
            d["profile_location"] = role_m.group(2) if role_m else "unknown"
            # Extract title from user message
            user_msg = d["messages"][1]["content"]
            title_m = re.search(r"Title:\s*(.+?)(?:\n|$)", user_msg)
            d["title"] = title_m.group(1).strip() if title_m else ""
            samples.append(d)
    return samples


def band_for_score(score: float) -> str:
    """Return the band name for a given score."""
    for name, (lo, hi) in BANDS.items():
        if lo <= score < hi:
            return name
    return "critical"  # fallback for 10.0


def stratified_sample(samples: List[dict], per_band: int, seed: int = 42) -> List[dict]:
    """Sample up to `per_band` items from each score band."""
    rng = random.Random(seed)
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        buckets[band_for_score(s["ground_truth_score"])].append(s)

    selected = []
    for band_name in BANDS:
        pool = buckets.get(band_name, [])
        n = min(per_band, len(pool))
        selected.extend(rng.sample(pool, n))
        logger.info(f"  Band '{band_name}': {len(pool)} available, sampled {n}")
    rng.shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Ollama scoring
# ---------------------------------------------------------------------------
def extract_score(raw_content: str, think_text: str = "") -> float:
    """
    Extract numeric score from model output.

    Strategy:
      1. Strip <think>...</think> blocks from content, search for SCORE: pattern
      2. If not found, search the think text for "final score:" pattern
      3. Return -1.0 on parse failure
    """
    # Strip think blocks (closed and unclosed) from content
    clean = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    if "<think>" in clean and "</think>" not in clean:
        clean = re.sub(r"<think>.*$", "", clean, flags=re.DOTALL).strip()

    # Primary: SCORE: pattern in clean content
    m = SCORE_RE.search(clean)
    if m:
        val = float(m.group(1))
        return min(val, 10.0)

    # Fallback 1: SCORE: pattern in the raw content (inside think blocks)
    m = SCORE_RE.search(raw_content)
    if m:
        val = float(m.group(1))
        return min(val, 10.0)

    # Fallback 2: "Final Score:" in think text (native Qwen3.5 thinking field)
    if think_text:
        m = THINK_SCORE_RE.search(think_text)
        if m:
            val = float(m.group(1))
            return min(val, 10.0)
        # Also try SCORE: in think text
        m = SCORE_RE.search(think_text)
        if m:
            val = float(m.group(1))
            return min(val, 10.0)

    return -1.0


def score_with_model(
    messages: List[dict],
    model: str,
    host: str,
    timeout: int = 120,
) -> Tuple[float, str, float]:
    """
    Score a single sample. Returns (score, raw_output, latency_seconds).
    Score is -1.0 on failure.
    """
    t0 = time.time()
    try:
        resp = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": messages[:2],  # system + user only
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 2048},
            },
            timeout=timeout,
        )
        latency = time.time() - t0
        if resp.status_code != 200:
            return -1.0, f"HTTP {resp.status_code}", latency

        body = resp.json()
        raw_content = body.get("message", {}).get("content", "")
        # Qwen3.5 native thinking field
        think_text = body.get("message", {}).get("thinking", "") or ""
        score = extract_score(raw_content, think_text)
        # Truncate for storage
        display = raw_content[:300]
        return score, display, latency

    except Exception as e:
        return -1.0, str(e), time.time() - t0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _rank(values: List[float]) -> List[float]:
    """Compute 1-based ranks with average ties."""
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


def _spearman(a: List[float], b: List[float]) -> float:
    """Spearman rank correlation coefficient."""
    n = len(a)
    if n < 2:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    ma = sum(ra) / n
    mb = sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / n
    sa = (sum((x - ma) ** 2 for x in ra) / n) ** 0.5
    sb = (sum((y - mb) ** 2 for y in rb) / n) ** 0.5
    if sa == 0 or sb == 0:
        return 0.0
    return cov / (sa * sb)


def compute_metrics(results: List[dict]) -> dict:
    """Compute full metrics suite from a list of scored results."""
    total = len(results)
    valid = [r for r in results if r["predicted_score"] >= 0]
    n = len(valid)
    if n == 0:
        return {
            "total_samples": total,
            "valid_samples": 0,
            "parse_failures": total,
            "parse_failure_rate": 1.0,
            "error": "No valid predictions",
        }

    deltas = [r["predicted_score"] - r["ground_truth_score"] for r in valid]
    abs_deltas = [abs(d) for d in deltas]

    mae = sum(abs_deltas) / n
    rmse = (sum(d * d for d in deltas) / n) ** 0.5

    direction_correct = sum(
        1 for r in valid
        if (r["predicted_score"] >= 5.0) == (r["ground_truth_score"] >= 5.0)
    )
    direction_acc = direction_correct / n

    spearman = _spearman(
        [r["ground_truth_score"] for r in valid],
        [r["predicted_score"] for r in valid],
    )

    within_1 = 100 * sum(1 for d in abs_deltas if d <= 1.0) / n
    within_2 = 100 * sum(1 for d in abs_deltas if d <= 2.0) / n

    avg_latency = sum(r.get("latency", 0) for r in results) / total

    # Per-band breakdown
    band_metrics = {}
    for band_name, (lo, hi) in BANDS.items():
        items = [r for r in valid if lo <= r["ground_truth_score"] < hi]
        if items:
            b_mae = sum(abs(r["predicted_score"] - r["ground_truth_score"]) for r in items) / len(items)
            b_dir = sum(
                1 for r in items
                if (r["predicted_score"] >= 5.0) == (r["ground_truth_score"] >= 5.0)
            ) / len(items)
            band_metrics[band_name] = {
                "count": len(items),
                "mae": round(b_mae, 3),
                "direction_acc": round(b_dir, 4),
            }

    # Per-profile breakdown
    profiles: Dict[str, List[dict]] = defaultdict(list)
    for r in valid:
        profiles[r["profile_role"]].append(r)

    profile_metrics = {}
    for role, items in sorted(profiles.items()):
        p_mae = sum(abs(r["predicted_score"] - r["ground_truth_score"]) for r in items) / len(items)
        p_dir = sum(
            1 for r in items
            if (r["predicted_score"] >= 5.0) == (r["ground_truth_score"] >= 5.0)
        ) / len(items)
        profile_metrics[role] = {
            "count": len(items),
            "mae": round(p_mae, 3),
            "direction_acc": round(p_dir, 4),
        }

    return {
        "total_samples": total,
        "valid_samples": n,
        "parse_failures": total - n,
        "parse_failure_rate": round((total - n) / total, 4) if total else 0,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "direction_accuracy": round(direction_acc, 4),
        "spearman_rho": round(spearman, 4),
        "within_1pt_pct": round(within_1, 1),
        "within_2pt_pct": round(within_2, 1),
        "avg_latency_s": round(avg_latency, 2),
        "band_breakdown": band_metrics,
        "profile_breakdown": profile_metrics,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def fmt_pct(val: float) -> str:
    return f"{val:.1%}"


def fmt_f3(val: float) -> str:
    return f"{val:.3f}"


def fmt_f4(val: float) -> str:
    return f"{val:.4f}"


def print_comparison_table(model_metrics: Dict[str, dict]):
    """Print a side-by-side comparison table for all models."""
    models = list(model_metrics.keys())
    col_w = max(24, max(len(m) for m in models) + 2)

    def header_line():
        return "-" * 22 + "+" + (("-" * col_w + "+") * len(models))

    def row(label: str, values: List[str]):
        cells = "".join(f" {v:>{col_w - 2}} |" for v in values)
        return f" {label:<20s} |{cells}"

    print("\n" + "=" * (23 + (col_w + 1) * len(models)))
    print("  MULTI-MODEL COMPARISON")
    print("=" * (23 + (col_w + 1) * len(models)))

    # Header
    hdr = "".join(f" {m:>{col_w - 2}} |" for m in models)
    print(f" {'Metric':<20s} |{hdr}")
    print(header_line())

    # Core metrics rows
    rows = [
        ("Samples (valid)", [str(model_metrics[m].get("valid_samples", 0)) for m in models]),
        ("Parse failures",  [str(model_metrics[m].get("parse_failures", 0)) for m in models]),
        ("Parse fail rate", [fmt_pct(model_metrics[m].get("parse_failure_rate", 0)) for m in models]),
        ("MAE",             [fmt_f3(model_metrics[m].get("mae", 99)) for m in models]),
        ("RMSE",            [fmt_f3(model_metrics[m].get("rmse", 99)) for m in models]),
        ("Direction acc",   [fmt_pct(model_metrics[m].get("direction_accuracy", 0)) for m in models]),
        ("Spearman rho",    [fmt_f4(model_metrics[m].get("spearman_rho", 0)) for m in models]),
        ("Within 1pt",      [f"{model_metrics[m].get('within_1pt_pct', 0):.1f}%" for m in models]),
        ("Within 2pt",      [f"{model_metrics[m].get('within_2pt_pct', 0):.1f}%" for m in models]),
        ("Avg latency (s)", [f"{model_metrics[m].get('avg_latency_s', 0):.2f}" for m in models]),
    ]
    for label, vals in rows:
        print(row(label, vals))

    # Band breakdown
    print(header_line())
    print(row("BAND BREAKDOWN", [""] * len(models)))
    print(header_line())
    for band_name in BANDS:
        mae_vals = []
        for m in models:
            bd = model_metrics[m].get("band_breakdown", {}).get(band_name, {})
            if bd:
                mae_vals.append(f"MAE {bd['mae']:.3f} (n={bd['count']})")
            else:
                mae_vals.append("--")
        print(row(f"  {band_name}", mae_vals))

    # Per-profile breakdown (top 5 profiles by sample count across all models)
    print(header_line())
    print(row("PROFILE BREAKDOWN", [""] * len(models)))
    print(header_line())

    # Gather all profiles, rank by total count
    all_profiles: Dict[str, int] = defaultdict(int)
    for m in models:
        for role, data in model_metrics[m].get("profile_breakdown", {}).items():
            all_profiles[role] += data["count"]
    top_profiles = sorted(all_profiles.items(), key=lambda x: x[1], reverse=True)[:8]

    for role, _ in top_profiles:
        vals = []
        for m in models:
            pd = model_metrics[m].get("profile_breakdown", {}).get(role, {})
            if pd:
                vals.append(f"MAE {pd['mae']:.2f} dir {pd['direction_acc']:.0%}")
            else:
                vals.append("--")
        display_role = role[:20] if len(role) > 20 else role
        print(row(f"  {display_role}", vals))

    print(header_line())


def print_recommendation(model_metrics: Dict[str, dict]):
    """Print a recommendation section based on composite scoring."""
    models = list(model_metrics.keys())
    if not models:
        return

    print("\n" + "=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)

    # Composite score: lower is better
    # Weights: MAE (40%), direction error (25%), parse failure (15%), 1-spearman (10%), RMSE (10%)
    scores: Dict[str, float] = {}
    details: Dict[str, str] = {}
    for m in models:
        mx = model_metrics[m]
        if "error" in mx:
            scores[m] = 999.0
            details[m] = "  FAILED: no valid predictions"
            continue

        mae = mx.get("mae", 10)
        rmse = mx.get("rmse", 10)
        dir_err = 1.0 - mx.get("direction_accuracy", 0)
        parse_fail = mx.get("parse_failure_rate", 1)
        spearman_gap = 1.0 - mx.get("spearman_rho", 0)

        composite = (
            0.40 * mae +
            0.25 * (dir_err * 10) +  # scale to ~same range as MAE
            0.15 * (parse_fail * 10) +
            0.10 * (spearman_gap * 10) +
            0.10 * rmse
        )
        scores[m] = composite
        details[m] = (
            f"  composite={composite:.3f}  "
            f"MAE={mae:.3f}  RMSE={rmse:.3f}  "
            f"dir={mx.get('direction_accuracy', 0):.1%}  "
            f"rho={mx.get('spearman_rho', 0):.4f}  "
            f"parse_fail={parse_fail:.1%}"
        )

    ranked = sorted(scores.items(), key=lambda x: x[1])
    for i, (m, score) in enumerate(ranked):
        marker = " << BEST" if i == 0 else ""
        print(f"\n  #{i+1}  {m}{marker}")
        print(details[m])

    winner = ranked[0][0]
    runner = ranked[1][0] if len(ranked) > 1 else None

    print(f"\n  VERDICT: Use '{winner}' as the production scorer.")
    if runner:
        gap = scores[runner] - scores[winner]
        if gap < 0.05:
            print(f"  NOTE: Very close to '{runner}' (delta={gap:.3f}). Consider latency and VRAM.")
        elif gap < 0.2:
            print(f"  NOTE: Moderate lead over '{runner}' (delta={gap:.3f}).")
        else:
            print(f"  NOTE: Clear winner over '{runner}' (delta={gap:.3f}).")

    # Actionable thresholds
    wmx = model_metrics[winner]
    warnings = []
    if wmx.get("mae", 10) > 0.6:
        warnings.append("MAE > 0.6 -- consider V3 retraining")
    if wmx.get("direction_accuracy", 0) < 0.95:
        warnings.append("Direction accuracy < 95% -- noise/signal boundary needs work")
    if wmx.get("parse_failure_rate", 1) > 0.05:
        warnings.append("Parse failure rate > 5% -- output format may need prompt tuning")
    if wmx.get("spearman_rho", 0) < 0.85:
        warnings.append("Spearman rho < 0.85 -- rank ordering is weak")

    if warnings:
        print("\n  WARNINGS for best model:")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("\n  No warnings. Best model is performing well across all metrics.")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_model(
    model: str,
    samples: List[dict],
    host: str,
    timeout: int = 120,
) -> Tuple[List[dict], float]:
    """Score all samples with a single model. Returns (results, elapsed)."""
    results = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        score, raw, latency = score_with_model(sample["messages"], model, host, timeout)
        results.append({
            "ground_truth_score": sample["ground_truth_score"],
            "predicted_score": score,
            "profile_role": sample["profile_role"],
            "profile_location": sample["profile_location"],
            "title": sample.get("title", ""),
            "latency": latency,
            "raw_output": raw[:200],
        })
        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(samples) - i - 1) / rate if rate > 0 else 0
            valid_n = sum(1 for r in results if r["predicted_score"] >= 0)
            logger.info(
                f"  [{model}] {i+1}/{len(samples)}  "
                f"{rate:.1f}/s  ETA {eta:.0f}s  valid={valid_n}"
            )
    elapsed = time.time() - t0
    return results, elapsed


def check_ollama(host: str, models: List[str]):
    """Verify Ollama is reachable and requested models exist."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=5)
        if resp.status_code != 200:
            logger.error("Ollama not reachable")
            sys.exit(1)
        available = [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        logger.error(f"Cannot reach Ollama at {host}: {e}")
        sys.exit(1)

    missing = []
    for model in models:
        if not any(model in a for a in available):
            missing.append(model)
    if missing:
        logger.error(f"Models not found in Ollama: {missing}")
        logger.error(f"Available: {available}")
        sys.exit(1)

    logger.info(f"Ollama OK. Available models include: {', '.join(available[:10])}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple Ollama scorer models against Opus ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model names to compare (e.g. stratos-scorer-v2 qwen3.5:9b)",
    )
    parser.add_argument(
        "--host", default=OLLAMA_HOST,
        help=f"Ollama host URL (default: {OLLAMA_HOST})",
    )
    parser.add_argument(
        "--eval-file", default=str(EVAL_FILE),
        help="Path to eval JSONL file",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max total samples (overrides stratified sampling; 0=use stratified)",
    )
    parser.add_argument(
        "--samples-per-band", type=int, default=20,
        help="Samples per score band for stratified sampling (default: 20, total ~100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--output", default="comparison_report.json",
        help="Output JSON report path (default: comparison_report.json)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Per-request timeout in seconds (default: 120)",
    )
    args = parser.parse_args()

    models = args.models
    logger.info(f"Comparing {len(models)} models: {models}")

    # Load eval set
    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        logger.error(f"Eval file not found: {eval_path}")
        sys.exit(1)

    all_samples = load_eval_set(eval_path)
    logger.info(f"Loaded {len(all_samples)} total eval samples")

    # Select samples
    if args.limit > 0:
        # Simple truncation for quick tests
        samples = all_samples[:args.limit]
        logger.info(f"Using first {len(samples)} samples (--limit)")
    else:
        logger.info(f"Stratified sampling: {args.samples_per_band} per band")
        samples = stratified_sample(all_samples, args.samples_per_band, seed=args.seed)
        logger.info(f"Selected {len(samples)} samples across {len(BANDS)} bands")

    if not samples:
        logger.error("No samples selected")
        sys.exit(1)

    # Check Ollama
    check_ollama(args.host, models)

    # Run each model
    all_results: Dict[str, List[dict]] = {}
    all_metrics: Dict[str, dict] = {}

    for model in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Scoring with model: {model}")
        logger.info(f"{'='*60}")

        results, elapsed = run_model(model, samples, args.host, args.timeout)
        metrics = compute_metrics(results)
        metrics["model"] = model
        metrics["elapsed_seconds"] = round(elapsed, 1)
        metrics["throughput"] = round(len(results) / elapsed, 2) if elapsed > 0 else 0

        all_results[model] = results
        all_metrics[model] = metrics

        valid_n = sum(1 for r in results if r["predicted_score"] >= 0)
        logger.info(
            f"Done: {model} -- {valid_n}/{len(results)} valid, "
            f"MAE={metrics.get('mae', '?')}, "
            f"dir={metrics.get('direction_accuracy', '?')}, "
            f"{elapsed:.1f}s"
        )

    # Print comparison
    print_comparison_table(all_metrics)
    print_recommendation(all_metrics)

    # Save full report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "eval_file": str(eval_path),
        "sample_count": len(samples),
        "sampling": "stratified" if args.limit == 0 else "limit",
        "samples_per_band": args.samples_per_band if args.limit == 0 else None,
        "seed": args.seed,
        "models": models,
        "metrics": all_metrics,
        "per_sample": {
            model: [
                {
                    "ground_truth": r["ground_truth_score"],
                    "predicted": r["predicted_score"],
                    "delta": round(r["predicted_score"] - r["ground_truth_score"], 2)
                            if r["predicted_score"] >= 0 else None,
                    "profile": r["profile_role"],
                    "title": r["title"][:80],
                    "latency": round(r["latency"], 2),
                }
                for r in results_list
            ]
            for model, results_list in all_results.items()
        },
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Full report saved to {output_path}")


if __name__ == "__main__":
    main()
