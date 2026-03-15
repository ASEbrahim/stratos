#!/usr/bin/env python3
"""Merge base (LimaRP+PIPPA) and synthetic gap-filling data into final training JSONL."""

import json
import hashlib
from pathlib import Path
from collections import Counter

PIPELINE_DIR = Path(__file__).parent.parent
BASE_PATH = PIPELINE_DIR / "training_data" / "base_training_data.jsonl"
SYNTHETIC_PATH = PIPELINE_DIR / "training_data" / "synthetic" / "synthetic_gaps.jsonl"
OUTPUT_PATH = PIPELINE_DIR / "training_data" / "final_training_data.jsonl"


def validate_conversation(conv: dict) -> list:
    errors = []
    msgs = conv.get("messages", [])
    if len(msgs) < 3:
        errors.append(f"Too few messages ({len(msgs)})")
        return errors
    if msgs[0]["role"] != "system":
        errors.append("First message is not system")
    for i in range(2, len(msgs)):
        if msgs[i]["role"] == msgs[i-1]["role"] and msgs[i]["role"] != "system":
            errors.append(f"Consecutive same-role at position {i}")
    for i, msg in enumerate(msgs):
        if not msg.get("content", "").strip():
            errors.append(f"Empty content at position {i}")
    return errors


def main():
    all_data = []
    error_count = 0

    if BASE_PATH.exists():
        with open(BASE_PATH) as f:
            for line in f:
                conv = json.loads(line)
                errors = validate_conversation(conv)
                if errors:
                    error_count += 1
                else:
                    all_data.append(conv)
        print(f"Base data: {len(all_data)} valid (skipped {error_count} invalid)")

    synth_count = 0
    if SYNTHETIC_PATH.exists():
        with open(SYNTHETIC_PATH) as f:
            for line in f:
                conv = json.loads(line)
                errors = validate_conversation(conv)
                if errors:
                    error_count += 1
                else:
                    all_data.append(conv)
                    synth_count += 1
        print(f"Synthetic data: {synth_count} valid")

    print(f"\nTotal training conversations: {len(all_data)}")
    print(f"Total validation errors: {error_count}")

    sources = Counter(c.get("source", "unknown") for c in all_data)
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count}")

    gaps = Counter(c.get("gap", "none") for c in all_data if c.get("gap"))
    if gaps:
        print(f"\nGap-filling breakdown:")
        for gap, count in sorted(gaps.items()):
            print(f"  {gap}: {count}")

    with open(OUTPUT_PATH, "w") as f:
        for conv in all_data:
            f.write(json.dumps(conv) + "\n")

    fingerprint = hashlib.sha256(open(OUTPUT_PATH, "rb").read()).hexdigest()[:16]
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Fingerprint: {fingerprint}")
    print(f"\nTraining data ready for Phase 2.")


if __name__ == "__main__":
    main()
