#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  STRAT_OS v2 — Automated Learning Cycle (Linux/ROCm)
#  Exports corrections, merges with distilled data, trains DoRA
# ═══════════════════════════════════════════════════════════════
set -e

echo "════════════════════════════════════════════════════════════"
echo "  STRAT_OS v2 Learning Cycle (DoRA)"
echo "  $(date)"
echo "════════════════════════════════════════════════════════════"

cd "$(dirname "$0")"

# Step 1: Re-export corrections from DB with v2 profile metadata
echo ""
echo "[1/4] Exporting corrections with v2 profile metadata..."
python3 export_training.py --min-delta 1.5 --output data/corrections_v2.jsonl

# Step 2: Merge with distilled data
echo ""
echo "[2/4] Merging training data..."
python3 export_training.py --merge data/distill_v2_train.jsonl data/corrections_v2.jsonl \
    --output data/training_merged.jsonl

# Step 3: Train DoRA (Q8_0)
echo ""
echo "[3/4] Starting DoRA training..."
python3 train_lora.py \
    --training-data data/training_merged.jsonl \
    --epochs 3 \
    --quant q8_0 \
    --full-retrain

# Step 4: Verify
echo ""
echo "[4/4] Verifying..."
echo "  Checking model registration..."
ollama list | grep stratos-scorer || echo "  WARNING: No stratos-scorer model found in ollama"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Learning cycle complete!"
echo "  config.yaml was auto-updated with the new model."
echo "════════════════════════════════════════════════════════════"
