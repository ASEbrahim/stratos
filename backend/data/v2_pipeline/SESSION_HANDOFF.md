# StratOS V2 Scorer — Session Handoff Document
**Date**: 2026-03-06
**Status**: Training complete, Phase 3 (GGUF export) in progress

---

## What Happened This Session

### 1. V2 Scorer Was Degraded
The deployed V2 scorer (trained Feb 22) had degraded since deployment. Post-training code changes to `scorer_adaptive.py` (think-block extraction, score parsing hardening, profile isolation) caused training/inference format drift:
- **Original V2**: MAE 0.393, 98.1% direction accuracy, 0 parse failures, Spearman 0.750
- **Degraded V2**: MAE 0.761, 237/2048 parse failures

The model weights were fine — the inference prompts had drifted from what the model was trained on.

### 2. Full V2 Retrain on Expanded Dataset
Retrained from scratch using the proven V2 config on Qwen3-8B base (`v1_merged_base/`):

**Training config** (DO NOT CHANGE — this is battle-tested across 19 iterations):
```
DoRA rank=16, alpha=32, dropout=0.05
Targets: q/k/v/o/gate/up/down_proj
batch_size=2, grad_accum=8, effective_batch=16
lr=1e-5, cosine schedule, 5% warmup, 1 epoch
max_seq_length=1024, completion_only_loss=True
AdamW, bf16, gradient_checkpointing (use_reentrant=False)
WeightedRandomSampler + per-sample loss weighting
```

**Critical environment variables**:
```bash
ROCR_VISIBLE_DEVICES=0          # MUST — excludes iGPU that causes memory access faults
HSA_OVERRIDE_GFX_VERSION=11.0.0 # ROCm compatibility
# DO NOT set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 — causes NaN gradients
```

**Training results**:
- 19,000 training examples, 2,423 eval examples
- 1,188 optimizer steps, ~31s/step, 10.4 hours total
- Final loss: 4.582 (started ~7.0, healthy convergence)
- Zero NaN gradients, zero crashes
- All checkpoints saved: 200, 400, 600, 800, 1000, 1188

### 3. Dataset Expansion
- Added 128 new articles scored by Opus via Batch API ($7.78)
- Merged into scores_v2.json: 20,550 → 24,387 scored examples
- Capped training at 19,000 (V2 succeeded at 18,502, V3 crashed at 24,479)
- `stage3_expand.py` was created for this — selects from DB, batch scores via API, merges

### 4. Holdout Eval Set Created (CRITICAL — DO NOT TRAIN ON THIS)
**File**: `data/v2_pipeline/eval_holdout_v2.jsonl` (1,500 examples)
- 50 articles **completely unseen** by training data (not in articles_v2.json)
- Scored across all 30 profiles by 6 parallel Claude Code agents ($0 cost)
- Each score has: article_id, profile_id, score, reason, think_text
- The original `eval_v2.jsonl` has **100% article overlap** with training data — it's contaminated
- Always use `eval_holdout_v2.jsonl` for real accuracy measurement

### 5. Key Discovery: Claude Code Agents Replace Batch API
Claude Code IS Opus. Deployed 6 agents in parallel, each scoring 250 article×profile pairs. Completed 1,500 scores in ~8 minutes at $0 API cost. Quality is identical to Batch API. Use this for batches under ~500 scores.

---

## Current State — What Needs to Be Done

### IMMEDIATE: Phase 3 Completion
Phase 3 (`train_v2.py` auto-runs after training) is currently:
1. ✅ Profile-awareness sanity check (adapter inference — slow on ROCm)
2. ⬜ Merge DoRA adapters into base model
3. ⬜ Export to GGUF
4. ⬜ Save as `training_output/v2_scorer.gguf`

**Check status**:
```bash
tail -30 data/v2_pipeline/training_output/full_training_run.log
ps aux | grep train_v2 | grep -v grep
```

**If Phase 3 completed successfully**, the GGUF will be at:
`data/v2_pipeline/training_output/v2_scorer.gguf`

**If Phase 3 crashed** (adapter inference on ROCm can be flaky):
- Checkpoints are safe at `training_output/checkpoint-1188/`
- Can manually merge + export:
```python
# Load base + adapters, merge, save, then use llama.cpp for GGUF conversion
```

### AFTER Phase 3: Deploy & Verify

1. **Back up the model** (user explicitly requested this):
```bash
mkdir -p data/v2_pipeline/v2_model_backup
cp data/v2_pipeline/training_output/v2_scorer.gguf data/v2_pipeline/v2_model_backup/
cp -r data/v2_pipeline/training_output/checkpoint-1188 data/v2_pipeline/v2_model_backup/
cp -r data/v2_pipeline/training_output/final_checkpoint data/v2_pipeline/v2_model_backup/
```

2. **Register with Ollama**:
```bash
# Check/create Modelfile in training_output/ directory
ollama create stratos-scorer-v2 -f data/v2_pipeline/training_output/Modelfile
sudo systemctl start ollama
```

3. **Evaluate against holdout set** (the REAL test):
```bash
python3 evaluate_scorer.py --eval-file data/v2_pipeline/eval_holdout_v2.jsonl --output eval_holdout_report.json
```
Target metrics: MAE < 0.5, direction accuracy > 95%, parse failures = 0

4. **Also evaluate against original eval set** for comparison:
```bash
python3 evaluate_scorer.py --eval-file data/v2_pipeline/eval_v2.jsonl --output eval_original_report.json
```

5. **Commit everything** with backup path filled in.

---

## Data Landscape

| Dataset | Count | Location |
|---------|-------|----------|
| DB articles (total) | 17,340 unique | `strat_os.db` |
| V2 training articles | 813 | `articles_v2.json` |
| Holdout eval articles | 50 | `holdout_articles.json` |
| **Untapped articles** | **16,477** | In DB, not in any training/eval |
| V2 scored examples | 24,387 | `scores_v2.json` |
| Training examples | 19,000 | `training_v2.jsonl` |
| Eval examples (contaminated) | 2,423 | `eval_v2.jsonl` |
| Eval examples (clean holdout) | 1,500 | `eval_holdout_v2.jsonl` |
| Training profiles | 30 | `profiles_v2.py` |

**Training data distribution** (heavily noise-skewed):
```
noise:       16,816 (88.5%)  weight=0.5
tangential:   1,439 (7.6%)   weight=1.0
moderate:       363 (1.9%)   weight=1.5
high:           245 (1.3%)   weight=2.0
critical:       137 (0.7%)   weight=3.0
```

---

## Planned V3 Improvements (User-Approved Direction)

### 1. Expand to 50 Profiles
20 new profiles drafted (common professions, likely-user countries only):
1. High school math teacher — Austin, Texas, USA
2. Backend software engineer — Singapore
3. Real estate developer — Dubai, UAE
4. Marketing director — London, UK
5. Public accountant / auditor — Toronto, Canada
6. Automotive engineer — Stuttgart, Germany
7. Investigative journalist — Washington DC, USA
8. E-commerce founder — Istanbul, Turkey
9. Environmental consultant — Amsterdam, Netherlands
10. HR director — Sydney, Australia
11. Dentist with private practice — Riyadh, Saudi Arabia
12. Commercial airline pilot — Dubai, UAE
13. Social media strategist — Los Angeles, USA
14. Electrical contractor — Manchester, UK
15. Construction project manager — Doha, Qatar
16. Investment analyst — Hong Kong
17. Public health researcher — Geneva, Switzerland
18. Trucking / logistics company owner — Atlanta, Georgia, USA
19. Veterinarian / clinic owner — Stockholm, Sweden
20. IT systems administrator — Osaka, Japan

**User preferences for profiles**:
- NO uncommon/niche professions (user laughed at the bonsai artist)
- NO countries unlikely to use the product (no India, Kenya, etc.)
- Focus on: Gulf states, US, UK, EU, Australia, East Asia, Turkey, Singapore

### 2. Include think_text in Training
Opus generates `think_text` (~1,125 chars, full reasoning) but only the short `reason` (~154 chars) is used in training. Including think_text would teach Qwen WHY articles score the way they do for specific profiles. Tradeoff: longer inference output.

### 3. Two-Stage Scoring
~80% of articles are obvious noise. A keyword pre-filter before LLM scoring could halve inference time:
- Stage 1: Does article mention ANY tracked companies/industries/interests? If zero overlap → score 0.5, skip LLM
- Stage 2: LLM scoring for everything that passes

### 4. Score with Claude Code Agents (not API)
For future dataset expansion, deploy Claude Code agents instead of paying Batch API. 6 agents in parallel, each handling 5 profiles × N articles. Free.

---

## File Map

```
backend/
├── data/v2_pipeline/
│   ├── profiles_v2.py              # 30 training profile definitions
│   ├── stage2_collect.py            # Article collection
│   ├── stage3_score.py              # Opus batch scoring (original)
│   ├── stage3_expand.py             # NEW: Expand dataset with new articles
│   ├── stage4_prepare.py            # MODIFIED: Added 19K training cap
│   ├── train_v2.py                  # Training script (Phase 2 + Phase 3)
│   ├── articles_v2.json             # 813 training articles
│   ├── scores_v2.json               # 24,387 scored examples
│   ├── training_v2.jsonl            # 19,000 training examples (ChatML format)
│   ├── eval_v2.jsonl                # 2,423 eval examples (CONTAMINATED — articles overlap training)
│   ├── eval_holdout_v2.jsonl        # 1,500 eval examples (CLEAN — unseen articles)
│   ├── holdout_articles.json        # 50 holdout articles
│   ├── holdout_scores_all.json      # 1,500 holdout scores with full reasoning
│   ├── holdout_scores_batch[1-6].json # Individual agent outputs
│   ├── scores_v2_backup.json        # Backup before expansion
│   ├── articles_v2_backup.json      # Backup before expansion
│   └── training_output/
│       ├── full_training_run.log    # Complete training log
│       ├── checkpoint-{200,400,600,800,1000,1188}/  # All checkpoints
│       ├── final_checkpoint/        # DoRA adapters (may be old — check dates)
│       ├── v2_scorer.gguf           # GGUF model (check if new or old Feb 22 version)
│       └── V2_TRAINING_REPORT.md    # Original V2 report (still relevant)
├── evaluate_scorer.py               # Eval script — use --eval-file for holdout
├── processors/
│   ├── scorer_adaptive.py           # Live scoring — prompt format MUST match training
│   └── scorer_base.py               # Base class with Ollama client
└── strat_os.db                      # SQLite DB (17,340 articles)
```

---

## Hardware Notes
- AMD RX 7900 XTX (24GB VRAM), Ryzen 7 7800X3D, 30GB RAM
- ROCm 6.2, PyTorch 2.5.1+rocm6.2
- Training uses ~22GB VRAM at batch_size=2
- No thermal issues at sustained 100% load (junction 88C, limit 110C)
- Stop Ollama before training: `sudo systemctl stop ollama`
- Unload models: `ollama stop stratos-scorer-v2` (if Ollama running)

## Key Lessons Learned (V0.1 through V2)
1. AOTRITON causes NaN gradients — took 7 hours to diagnose originally
2. iGPU causes memory access faults — ROCR_VISIBLE_DEVICES=0 is mandatory
3. Training set over ~20K examples causes OOM — cap at 19K
4. Adapter inference on ROCm is ~30x slower than merged model — always merge before bulk eval
5. Training/inference prompt format MUST be character-for-character identical
6. After `get_peft_model()`, delete `hf_device_map` and force `.to("cuda:0")` to prevent gradient crashes
7. Gradient checkpointing adds ~30% compute time but saves ~40% VRAM — required for 8B on 24GB
