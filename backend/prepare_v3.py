#!/usr/bin/env python3
"""
StratOS V3 Training Data Preparation
======================================
Merges all available training data sources into a unified V3 dataset:
  1. V2 pipeline data (18,502 train + 2,048 eval from Opus batch API)
  2. V1 incremental data (5,977 from distillation corrections)
  3. New user_feedback corrections (since V2 training, Feb 19 2026)

Usage:
    python3 prepare_v3.py                     # Merge all sources, output v3 training set
    python3 prepare_v3.py --dry-run           # Just show counts, don't write
    python3 prepare_v3.py --include-eval      # Also include V2 eval set in training
    python3 prepare_v3.py --export-corrections # Export only new corrections for incremental training

Output:
    data/v3_pipeline/training_v3.jsonl       — Merged deduplicated training set
    data/v3_pipeline/eval_v3.jsonl           — Held-out eval set (from V2 eval)
    data/v3_pipeline/v3_prep_report.json     — Preparation statistics
"""

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("PREPARE_V3")

# Paths
BASE_DIR = Path(__file__).parent
V2_TRAIN = BASE_DIR / "data" / "v2_pipeline" / "training_v2.jsonl"
V2_EVAL = BASE_DIR / "data" / "v2_pipeline" / "eval_v2.jsonl"
V1_MERGED = BASE_DIR / "data" / "training_merged.jsonl"
V1_DATA = BASE_DIR / "data" / "training_data.jsonl"
DB_PATH = BASE_DIR / "strat_os.db"
V3_DIR = BASE_DIR / "data" / "v3_pipeline"

SCORE_RE = re.compile(r"SCORE:\s*(\d+\.?\d*)")


def content_hash(messages: list) -> str:
    """Hash the system+user messages for deduplication (ignores assistant response)."""
    key = messages[0]["content"][:200] + "|" + messages[1]["content"]
    return hashlib.md5(key.encode()).hexdigest()


def extract_metadata(sample: dict) -> dict:
    """Extract profile and score info from a training sample."""
    msgs = sample.get("messages", [])
    if len(msgs) < 3:
        return {"valid": False}

    sys_msg = msgs[0]["content"]
    user_msg = msgs[1]["content"]
    asst_msg = msgs[2]["content"]

    # Extract score
    m = SCORE_RE.search(asst_msg)
    score = float(m.group(1)) if m else -1

    # Extract profile
    role_m = re.search(r"relevance scorer for (?:a |an )?(.+?) in (.+?)\.", sys_msg)
    role = role_m.group(1) if role_m else "unknown"
    location = role_m.group(2) if role_m else "unknown"

    # Extract title
    title_m = re.search(r"Title:\s*(.+?)(?:\n|$)", user_msg)
    title = title_m.group(1).strip() if title_m else ""

    return {
        "valid": score >= 0,
        "score": score,
        "role": role,
        "location": location,
        "title": title,
        "weight": sample.get("sample_weight", 1.0),
    }


def load_jsonl(path: Path) -> list:
    """Load samples from a JSONL file."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return []
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_new_corrections(db_path: Path, since: str = "2026-02-19") -> list:
    """Load new corrections from user_feedback since a given date."""
    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return []

    try:
        from export_training import build_system_prompt, build_user_message
    except ImportError:
        logger.warning("Cannot import export_training — skipping DB corrections")
        return []

    try:
        db = sqlite3.connect(str(db_path))
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """SELECT * FROM user_feedback
               WHERE created_at > ? AND action = 'rate'
               ORDER BY created_at""",
            (since,),
        ).fetchall()
        db.close()

        samples = []
        for row in rows:
            row = dict(row)
            score = row.get("user_score")
            if score is None:
                continue

            # Build ChatML training sample
            role = row.get("profile_role", "")
            location = row.get("profile_location", "")
            context = row.get("profile_context", "")
            if not role:
                continue

            sys_prompt = build_system_prompt(role=role, location=location, context=context)
            user_msg = build_user_message(
                title=row.get("title", ""),
                summary=row.get("note", "")[:500],
                category=row.get("category", "general"),
                keywords="",
            )
            asst_msg = f"SCORE: {float(score):.1f} | REASON: Opus correction (delta from local score {row.get('ai_score', '?')})"

            samples.append({
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": asst_msg},
                ],
                "sample_weight": 2.0,  # Corrections get high weight
                "source": "db_correction",
            })
        return samples

    except Exception as e:
        logger.warning(f"Failed to load corrections from DB: {e}")
        return []


def score_to_band(score: float) -> str:
    """Map a score to its band name."""
    if score < 2.5:
        return "noise"
    if score < 4.5:
        return "tangential"
    if score < 6.5:
        return "moderate"
    if score < 8.5:
        return "high"
    return "critical"


def main():
    parser = argparse.ArgumentParser(description="Prepare V3 training data")
    parser.add_argument("--dry-run", action="store_true", help="Just show counts, don't write")
    parser.add_argument("--include-eval", action="store_true", help="Include V2 eval set in training")
    parser.add_argument("--export-corrections", action="store_true",
                        help="Export only new corrections for incremental training")
    parser.add_argument("--since", default="2026-02-19", help="Corrections since date (default: V2 training date)")
    parser.add_argument("--output-dir", default=str(V3_DIR), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    # === Load all data sources ===
    logger.info("Loading data sources...")

    v2_train = load_jsonl(V2_TRAIN)
    logger.info(f"  V2 training:      {len(v2_train):,} samples")

    v2_eval = load_jsonl(V2_EVAL)
    logger.info(f"  V2 eval:          {len(v2_eval):,} samples")

    v1_merged = load_jsonl(V1_MERGED)
    logger.info(f"  V1 merged:        {len(v1_merged):,} samples")

    new_corrections = load_new_corrections(DB_PATH, since=args.since)
    logger.info(f"  New corrections:  {len(new_corrections):,} samples (since {args.since})")

    # Quick export mode
    if args.export_corrections:
        if new_corrections:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "new_corrections.jsonl"
            with open(path, "w") as f:
                for s in new_corrections:
                    f.write(json.dumps(s) + "\n")
            logger.info(f"Exported {len(new_corrections)} corrections to {path}")
        else:
            logger.info("No new corrections to export")
        return

    # === Deduplicate across sources ===
    logger.info("Deduplicating...")

    seen_hashes = set()
    deduped_train = []
    source_counts = Counter()
    dup_count = 0

    # Priority: V2 train > new corrections > V1 merged
    for source_name, source_data in [
        ("v2_train", v2_train),
        ("v2_eval", v2_eval if args.include_eval else []),
        ("new_corrections", new_corrections),
        ("v1_merged", v1_merged),
    ]:
        for sample in source_data:
            msgs = sample.get("messages", [])
            if len(msgs) < 3:
                continue
            h = content_hash(msgs)
            if h in seen_hashes:
                dup_count += 1
                continue
            seen_hashes.add(h)
            sample["_source"] = source_name
            deduped_train.append(sample)
            source_counts[source_name] += 1

    logger.info(f"  Deduplicated: {dup_count:,} duplicates removed")
    for src, count in source_counts.most_common():
        logger.info(f"    {src}: {count:,}")
    logger.info(f"  Total unique: {len(deduped_train):,}")

    # === Analyze score distribution ===
    band_counts = Counter()
    profile_counts = Counter()
    valid_count = 0
    for sample in deduped_train:
        meta = extract_metadata(sample)
        if meta["valid"]:
            valid_count += 1
            band_counts[score_to_band(meta["score"])] += 1
            profile_counts[meta["role"]] += 1

    logger.info(f"\nScore band distribution ({valid_count:,} valid):")
    for band in ["noise", "tangential", "moderate", "high", "critical"]:
        c = band_counts.get(band, 0)
        pct = 100 * c / valid_count if valid_count else 0
        logger.info(f"  {band:12s}: {c:5,} ({pct:5.1f}%)")

    logger.info(f"\nProfile distribution ({len(profile_counts)} profiles):")
    for role, count in profile_counts.most_common(10):
        logger.info(f"  {role[:60]:60s}: {count:,}")

    # === Prepare eval set ===
    # Use V2 eval as the held-out set (not included in training by default)
    eval_set = v2_eval if not args.include_eval else []

    if args.dry_run:
        logger.info("\n[DRY RUN] Would write:")
        logger.info(f"  Training: {len(deduped_train):,} samples")
        logger.info(f"  Eval:     {len(eval_set):,} samples")
        return

    # === Write output ===
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "training_v3.jsonl"
    with open(train_path, "w") as f:
        for sample in deduped_train:
            # Remove internal metadata before writing
            output = {k: v for k, v in sample.items() if not k.startswith("_")}
            f.write(json.dumps(output) + "\n")
    logger.info(f"\nWrote {len(deduped_train):,} training samples to {train_path}")

    if eval_set:
        eval_path = out_dir / "eval_v3.jsonl"
        with open(eval_path, "w") as f:
            for sample in eval_set:
                f.write(json.dumps(sample) + "\n")
        logger.info(f"Wrote {len(eval_set):,} eval samples to {eval_path}")

    # === Save report ===
    report = {
        "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": dict(source_counts),
        "duplicates_removed": dup_count,
        "total_training": len(deduped_train),
        "total_eval": len(eval_set),
        "valid_samples": valid_count,
        "profiles": len(profile_counts),
        "band_distribution": dict(band_counts),
        "include_eval_in_train": args.include_eval,
        "corrections_since": args.since,
    }
    report_path = out_dir / "v3_prep_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")

    # === V3 training recommendation ===
    print(f"\n{'=' * 60}")
    print("V3 DATA PREPARATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Training samples: {len(deduped_train):,}")
    print(f"Eval samples:     {len(eval_set):,}")
    print(f"Profiles:         {len(profile_counts)}")
    print(f"New corrections:  {len(new_corrections):,}")
    print()
    if len(new_corrections) >= 200:
        print("READY for V3 training (200+ new corrections)")
        print(f"Run: python3 train_lora.py --data {train_path}")
    elif len(new_corrections) >= 50:
        print("Moderate new signal — incremental training recommended")
        print(f"Run: python3 train_lora.py --data {train_path} --epochs 1")
    else:
        print(f"Only {len(new_corrections)} new corrections — V3 not recommended yet")
        print("Continue collecting feedback and run distill.py for more corrections")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
