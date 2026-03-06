#!/bin/bash
# Dual-temperature holdout evaluation runner
# Usage: bash run_eval.sh <model_name> [eval_file]
#
# Examples:
#   bash run_eval.sh stratos-scorer-v2-baseline                    # V2 holdout eval
#   bash run_eval.sh stratos-scorer-v2.1                           # V2.1 holdout eval
#   bash run_eval.sh stratos-scorer-v2.1 data/v2_pipeline/eval_v2.jsonl  # contaminated eval

set -e

MODEL="${1:?Usage: bash run_eval.sh <model_name> [eval_file]}"
EVAL_FILE="${2:-data/v2_pipeline/eval_holdout_v2.jsonl}"

echo "============================================"
echo "StratOS Scorer Evaluation"
echo "Model:     $MODEL"
echo "Eval file: $EVAL_FILE"
echo "============================================"

# Check Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running. Start with: sudo systemctl start ollama"
    exit 1
fi

# Check model exists
if ! curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; models=[m['name'] for m in json.load(sys.stdin)['models']]; sys.exit(0 if any('$MODEL' in m for m in models) else 1)" 2>/dev/null; then
    echo "ERROR: Model '$MODEL' not found in Ollama."
    echo "Available models:"
    curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; [print(f'  {m[\"name\"]}') for m in json.load(sys.stdin)['models']]"
    exit 1
fi

echo ""
echo "--- Pass 1: Temperature 0.1 (reproducible benchmark) ---"
python3 evaluate_scorer.py \
    --model "$MODEL" \
    --eval-file "$EVAL_FILE" \
    --temperature 0.1 \
    --output "eval_report_${MODEL}_t0.1.json"

echo ""
echo "--- Pass 2: Temperature 0.6 (production-realistic) ---"
python3 evaluate_scorer.py \
    --model "$MODEL" \
    --eval-file "$EVAL_FILE" \
    --temperature 0.6 \
    --output "eval_report_${MODEL}_t0.6.json"

echo ""
echo "============================================"
echo "Both passes complete. Reports saved:"
echo "  eval_report_${MODEL}_t0.1.json"
echo "  eval_report_${MODEL}_t0.6.json"
echo ""
echo "Quick comparison:"
python3 -c "
import json
t01 = json.load(open('eval_report_${MODEL}_t0.1.json'))
t06 = json.load(open('eval_report_${MODEL}_t0.6.json'))
print(f'  temp=0.1: MAE={t01[\"mae\"]:.3f}  dir={t01[\"direction_accuracy\"]:.1%}  parse_fail={t01[\"parse_failures\"]}  spearman={t01[\"spearman_rho\"]:.4f}')
print(f'  temp=0.6: MAE={t06[\"mae\"]:.3f}  dir={t06[\"direction_accuracy\"]:.1%}  parse_fail={t06[\"parse_failures\"]}  spearman={t06[\"spearman_rho\"]:.4f}')
gap = t06['mae'] - t01['mae']
print(f'  Temperature gap: {gap:+.3f} MAE ({\"model is temp-sensitive\" if abs(gap) > 0.1 else \"model is temp-stable\"})')
"
echo "============================================"
