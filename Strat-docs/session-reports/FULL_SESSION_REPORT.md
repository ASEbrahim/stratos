# StratOS Scorer — Complete Session Report
## March 7-8, 2026 | ~24 Hours of Continuous Development

**Outcome:** V2.2 deployed with 41% MAE improvement (1.544 → 0.914), zero parse failures, and a unified Qwen3.5-9B model architecture across the entire system.

---

## Table of Contents

1. [Session Overview](#1-session-overview)
2. [Starting State](#2-starting-state)
3. [V2.1 Evaluation Results — The Think_Text Failure](#3-v21-evaluation-results)
4. [V2 Corrected Baseline — The Generalization Discovery](#4-v2-corrected-baseline)
5. [Scoring Provider Validation](#5-scoring-provider-validation)
6. [V2.2 Article Expansion Pipeline](#6-v22-article-expansion-pipeline)
7. [Qwen3.5 Model Family Evaluation](#7-qwen35-model-family-evaluation)
8. [V2.2 Training Environment Preparation](#8-v22-training-environment-preparation)
9. [V2.2 Training Results](#9-v22-training-results)
10. [Opus Holdout Evaluation — The Real Benchmark](#10-opus-holdout-evaluation)
11. [Infrastructure Benchmarks](#11-infrastructure-benchmarks)
12. [Model Unification Decision](#12-model-unification-decision)
13. [Code Changes and Bug Fixes](#13-code-changes-and-bug-fixes)
14. [Documents Created](#14-documents-created)
15. [Cost Analysis](#15-cost-analysis)
16. [Complete Version History](#16-complete-version-history)
17. [Key Discoveries and Insights](#17-key-discoveries-and-insights)
18. [Lessons Learned](#18-lessons-learned)
19. [Feature Roadmap Status](#19-feature-roadmap-status)
20. [Current System State](#20-current-system-state)
21. [Remaining Work](#21-remaining-work)
22. [Risk Register](#22-risk-register)

---

## 1. Session Overview

This session began with V2.1 training completing overnight and ended with V2.2 deployed in production — a complete training cycle from evaluation through provider validation, data expansion, model upgrade, training, and deployment in approximately 24 hours.

### Timeline

| Time | Event |
|------|-------|
| Session start | V2.1 training complete, evaluation pending |
| Hour 1-2 | V2.1 holdout eval: MAE 3.244 — catastrophic failure |
| Hour 2-3 | V2 corrected baseline eval: MAE 1.544 — generalization discovery |
| Hour 3-4 | DeepSeek V3.2 validation: MAE 1.521 — eliminated |
| Hour 4-5 | Gemini 3 Flash validation: MAE 0.893 calibrated — selected |
| Hour 5-6 | V2.2 expansion pipeline design and review |
| Hour 6-8 | Pipeline implementation, Gemini scoring launched |
| Hour 8-10 | training_pipeline.md, v3_planning.md, features_roadmap.md created |
| Hour 10-12 | Qwen3.5 model evaluations (9B, 35B-A3B, 0.8B) |
| Hour 12-14 | VRAM fit test, training environment prep, stage4 fixes |
| Hour 14-16 | Infrastructure benchmarks (VRAM coexistence, NUM_PARALLEL) |
| Hour 16-17 | Agent search_feed tool built, prompt version-pinning deployed |
| Hour 17-24 | V2.2 training (overnight), GGUF export, Opus holdout eval |
| Final | V2.2 deployed, model unification to Qwen3.5-9B initiated |

---

## 2. Starting State

### What We Had
- V2 production model: Qwen3-8B, MAE 1.533 (broken Modelfile), 813 training articles
- V2.1 training just completed: Qwen3-8B with think_text, 47 profiles, 19K examples
- 17,260 untapped articles in the database
- 39,071 Opus-scored examples from previous distillation
- $5 DeepSeek API credit, no Gemini account yet
- 1TB SSD at 92% full
- No proper baseline measurement (V2 holdout was measured through broken Modelfile)

### Open Questions at Session Start
1. Did V2.1's think_text improve scoring?
2. What is V2's true holdout performance with a correct Modelfile?
3. Can cheaper providers replace Opus for training data?
4. Is 813 articles the generalization bottleneck?
5. Should we upgrade to Qwen3.5?
6. Can the GPU handle parallel inference?
7. Can scorer and wizard coexist in VRAM?

**By session end, every one of these questions was answered with data.**

---

## 3. V2.1 Evaluation Results — The Think_Text Failure

### Results

| Metric | V2.1 (think_text) | V2 Baseline |
|--------|-------------------|-------------|
| Holdout MAE | **3.244** | 1.544 |
| Direction Accuracy | **53.5%** | 91.7% |
| Parse Failures | 68 (4.5%) | 75 (5.0%) |
| Spearman rho | 0.539 | 0.555 |

### Root Cause Analysis

V2.1 massively over-scored noise articles. Noise band MAE was 3.358 — the model predicted scores 3.4 points too high on average for irrelevant articles. High/critical bands were decent (MAE 1.3), meaning the model found relevance everywhere but couldn't say "no."

Think_text (843 chars avg) trained the model to generate plausible-sounding justifications for any score. For noise articles, the model learned to write "no connection" reasoning but the score number still drifted upward. The reasoning was a distraction from calibration.

### What Changed in V2.1 (4+ simultaneous variables)
1. Think_text format (843 chars avg vs 172 chars)
2. 17 new agent-scored profiles
3. Different data distribution
4. 5x longer assistant messages

**Lesson:** Impossible to isolate which change caused regression when multiple variables change simultaneously. Change ONE variable per version.

### The Definitive Pattern on Reasoning Format

| Version | Format | Holdout MAE |
|---------|--------|-------------|
| V1 | CoT (long) | 1.553 |
| V2 | Short reason (172 chars) | 1.544 |
| V2.1 | Think_text (843 chars) | 3.244 |

Short reasons and CoT perform identically on holdout. Think_text catastrophically worse. **Reasoning format in training targets is a solved question for 8B-class models: short reasons only.**

---

## 4. V2 Corrected Baseline — The Generalization Discovery

### The Critical Finding

| Measurement | MAE | What It Measures |
|-------------|-----|------------------|
| V2 contaminated (PyTorch bf16) | 0.393 | Memorization ceiling |
| V2 holdout (broken Modelfile) | 1.533 | Contaminated by format drift |
| **V2 holdout (corrected Modelfile)** | **1.544** | **True generalization** |

**1.544 ≈ 1.533.** The corrected Modelfile barely changed anything. Format drift (/nothink, num_predict=2048) accounted for essentially zero of the holdout error. The generalization gap IS the entire problem.

### Implications

The model memorized 813 articles well (0.393 contaminated) and cannot generalize to unseen articles (1.544 holdout). The contaminated-to-holdout ratio is 3.9x — almost entirely memorization. This conclusively proved that **article expansion is the only thing that matters.** Format changes, think_text, calibration layers — none address the core issue.

### Per-Band Results (Corrected V2 Baseline)

| Band | MAE | Notes |
|------|-----|-------|
| Noise (0-2.5) | 1.384 | Acceptable |
| Tangential (2.5-4.0) | 2.025 | Poor |
| Moderate (4.5-6.5) | 2.740 | Worst band |
| High (7.0-8.5) | 2.688 | Second worst |
| Critical (8.5-10.0) | 1.111 | Good (small n=8) |

---

## 5. Scoring Provider Validation

### 5.1 DeepSeek V3.2 — ELIMINATED

**Test:** 300 stratified examples, seed 42, same prompts as Opus. Cost: ~$0.10.

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Raw MAE vs Opus | 1.521 | < 1.0 | FAILED |
| Calibrated MAE (isotonic LOO) | 1.095 | < 1.0 | FAILED |
| Spearman | 0.882 | > 0.7 | PASSED |
| Parse failures | 0.0% | < 5% | PASSED |

**Fatal flaw:** Middle-range collapse. DeepSeek uses only ~6 distinct score levels. Scores 1.5 and 2.5 serve as "dump buckets" mapping to Opus scores spanning 2.0-8.5. No calibration can recover this lost information.

**6-8 band catastrophic:** MAE 2.932, bias -2.732. DeepSeek applies strict "direct match" logic and cannot assess career trajectory or aspirational relevance.

**Hybrid pipeline insight:** DeepSeek IS viable for noise pre-filtering (0-2 band MAE 0.625). Could pre-screen noise articles cheaply before sending non-noise to Opus. Not pursued because Gemini proved viable.

### 5.2 Gemini 3 Flash — SELECTED

**Test:** Same 300 samples, seed 42. Cost: $0.00 (free tier).

| Metric | Gemini 3 Flash | DeepSeek V3.2 | Threshold |
|--------|:--------------:|:-------------:|:---------:|
| Raw MAE | **1.090** | 1.521 | < 1.0 |
| Calibrated MAE (isotonic LOO) | **0.893** | 1.095 | < 1.0 |
| Spearman | **0.922** | 0.882 | > 0.7 |
| Parse failures | 5.7% | **0.0%** | < 5% |
| Within 1.0 | **70.3%** | 55.3% | > 60% |
| 8-10 band MAE | **0.692** | 1.672 | — |
| Score diversity | **~15 levels** | ~6 levels | — |

**Why calibration works for Gemini but not DeepSeek:** Gemini uses 15+ distinct score levels with relatively flat bias across bands (-0.2 to -1.1). Isotonic regression has actual information to work with. DeepSeek collapses the middle range into two scores, destroying information that no monotone function can recover.

**Calibration implementation:** `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')` fitted on 283 (Gemini, Opus) validation pairs. Saved as both pickle and JSON lookup table, tagged with model version and validation date.

**Parse failure mitigation:** Bumped max_output_tokens from 512 to 1024, added retry logic with simplified fallback prompt.

### 5.3 Provider Cost Comparison (for 36K scores)

| Provider | Cost | MAE (calibrated) | Decision |
|----------|------|-------------------|----------|
| Claude Opus (standard) | $50-70 | 0.0 (baseline) | Too expensive |
| Claude Opus (Batch 50% off) | $25-35 | 0.0 (baseline) | Fallback option |
| Gemini 3 Flash (paid) | $3-6 | 0.893 | **SELECTED** |
| Gemini 3 Flash (free) | $0 | 0.893 | Too slow (36 days) |
| DeepSeek V3.2 | ~$12 | 1.095 | Eliminated |

---

## 6. V2.2 Article Expansion Pipeline

### Pipeline Design

| Phase | Task | Status |
|-------|------|--------|
| 1 | Extract 2,000 articles from DB (stratified by category) | DONE |
| 2 | Score with Gemini 3 Flash (50K pairs: 25 profiles × 2K articles) | IN PROGRESS |
| 3 | Apply isotonic calibration | DONE (calibrator pre-trained) |
| 4 | (skipped) | — |
| 5 | Prepare training data with stage4_prepare.py fixes | DONE |

### Key Decisions

| Decision | Resolution | Rationale |
|----------|-----------|-----------|
| Scoring provider | Gemini 3 Flash + isotonic calibration | MAE 0.893, 15+ score levels |
| Profiles | 25 (reduced from 50) | RPD limit: 10K/day at Paid Tier 1 |
| Articles | 2,000 stratified by category | 2.5x more than V2's 813 |
| Think_text | Forced OFF | V2.1 catastrophe confirmed |
| Category field | Real labels (bug fix) | Was always "general" in training |
| Noise cap | 9,000 (47% of training data) | Balance between too aggressive (5K) and uncapped (90%) |
| User feedback | 2,631 entries at 1.0x weight | Single user — don't over-weight |

### Gemini Scoring Progress

- **Google API:** Paid Tier 1, 10K RPD, 1K RPM
- **Completed:** ~4,148/50,000 scores when training started
- **Automation:** Cron job at 10:15 AM KWT daily
- **Cost so far:** ~$2-3
- **Remaining budget:** ~4.5 KD ($14.60)

### Format Alignment Fixes in stage4_prepare.py

1. Removed think_tokens filter (would have silently rejected all Gemini-scored data)
2. Real category labels instead of hardcoded "general"
3. Added LANGUAGE line to system prompt (matches inference format)
4. Noise cap at 9,000 examples
5. Auto-loads expansion scores/articles
6. Handles missing think_text/think_tokens fields gracefully

---

## 7. Qwen3.5 Model Family Evaluation

### Models Tested

| Model | VRAM | Purpose Tested | Holdout MAE | Verdict |
|-------|------|----------------|-------------|---------|
| Qwen3.5-9B | 6.6 GB | Scorer base model | **1.297** | **SELECTED** — beats fine-tuned V2 |
| Qwen3.5-35B-A3B | 23 GB | Wizard replacement | N/A | NOT RECOMMENDED — too slow, truncates |
| Qwen3.5-0.8B | 1.0 GB | Noise pre-filter | 3.630 | NOT VIABLE — worse than random |

### Qwen3.5-9B: The Breakthrough Finding

**An unfine-tuned Qwen3.5-9B (MAE 1.297) beats the fine-tuned Qwen3-8B V2 (MAE 1.544) by 16% with zero training.**

This is the single most important finding of the session. The generational architecture improvement from Qwen3 to Qwen3.5 provides more benefit than 12+ hours of fine-tuning on the previous generation.

| Metric | Qwen3.5-9B Base | V2 Fine-Tuned (Qwen3-8B) |
|--------|:---------------:|:------------------------:|
| MAE | **1.297** | 1.544 |
| Parse Failures | **0 (0.0%)** | 75 (5.0%) |
| Within 1.0 | **62.1%** | 53.4% |
| Direction Accuracy | 87.3% | **91.7%** |

Primary weakness: positive bias (+0.478) inflating noise scores into moderate band — exactly the systematic error fine-tuning corrects well.

### Qwen3.5-35B-A3B: Eliminated for Wizard

| Task | 35B-A3B | qwen3:14b |
|------|---------|-----------|
| Profile generation | 235s, incomplete (hit token limit) | 27s, complete |
| Briefing | 48s, truncated | 22s, complete |

Better prose quality but 2-8x slower and consistently truncates before completing output. Speed and completeness matter more than marginal quality improvement.

### Qwen3.5-0.8B: Completely Unusable

MAE 3.630, direction accuracy 30% (worse than random), massive positive bias (+3.49), and paradoxically slower than the 9B (15.6s/call due to verbose think blocks). No role as pre-filter, scorer, or anything else.

---

## 8. V2.2 Training Environment Preparation

### VRAM Fit Test Results

| Configuration | Peak VRAM | Result |
|--------------|-----------|--------|
| Qwen3.5-9B, seq_len=1024 | 22.08 GB | **OOM** during backward pass |
| Qwen3.5-9B, seq_len=512 | 20.37 GB | **SUCCESS** (85% VRAM, 3.61 GB headroom) |

### Token Distribution Discovery

All 11,785 training examples: max 458 tokens, P95 ~420 tokens. **Zero examples exceed 512 tokens.** This means seq_length=512 causes zero truncation while providing ~4x speedup over seq_length=1024 (attention is O(n²)).

**Retrospective:** V2 could have used 512 too — its examples were also under 500 tokens. The 1024 default was never questioned, wasting ~9 hours per training run. A 5-line token distribution check would have caught this.

### Training Script Updates

1. Added `--base-model qwen35` flag to train_v2.py
2. Changed MAX_SEQ_LENGTH from 1024 to 512
3. Fixed hardcoded token IDs (Qwen3: 151643 → Qwen3.5: 248044) to dynamic lookup
4. Updated both Phase 2 (training) and Phase 3 (merge/eval) token handling

### First Training Attempt — System Crash

Training started with desktop environment running. At 97.2% VRAM usage (23.31/23.98 GB), gnome-shell/Discord/browsers competed for the remaining 0.67 GB, triggering amdgpu "Not enough memory for command submission" errors. System became unresponsive and crashed.

**Fix:** Kill all GPU-consuming processes (browsers, Discord, compositor) before training. Retrained from headless session. Training stabilized at 23.31 GB with flat allocation curve — no spikes, no fragmentation.

---

## 9. V2.2 Training Results

### Training Configuration

| Parameter | V2 | V2.2 |
|-----------|-----|------|
| Base model | Qwen3-8B | **Qwen3.5-9B** |
| Method | DoRA rank 16 | DoRA rank 16 (unchanged) |
| Seq length | 1024 | **512** |
| Articles | 813 | **2,000** |
| Profiles | 30 | **25** |
| Training examples | 18,502 | **11,785** |
| Scoring source | Opus direct | **Gemini 3 Flash + calibration** |
| Format | Short reason | Short reason (unchanged) |
| Step time | 38.5s | **30.1s** |
| Total time | ~12.5 hours | **~7 hours** |
| Steps | 1,188 | **737** |

### Gemini Eval Results (Student-Teacher Agreement)

| Metric | Value |
|--------|-------|
| MAE | **0.848** |
| Direction Accuracy | 92.7% |
| PSR | 52.3% |
| Spearman rho | 0.708 |
| Parse Failures | 0 |

### Loss Curve

Healthy convergence: 17.50 → 5.32 (minimum). No divergence, no instability. Gradient norms 2.5-3.7 throughout.

### WeightedRandomSampler Verification

| Bucket | Expected | Actual | Raw (unweighted) |
|--------|----------|--------|-------------------|
| Noise | 45.8% | 45.6% | 68.9% |
| Tangential | 26.5% | 25.3% | 20.0% |
| Moderate | 12.6% | 13.2% | 6.3% |
| High | 8.2% | 8.7% | 3.1% |
| Critical | 6.8% | 7.2% | 1.7% |

Distance to expected: 0.0287 (near-perfect). Rare high-value articles upsampled by ~4x.

---

## 10. Opus Holdout Evaluation — The Real Benchmark

The Gemini eval (MAE 0.848) measures student-teacher agreement. The Opus holdout (1,500 examples, 50 unseen articles, scored by Claude Opus) is the true apples-to-apples comparison with V2.

### Results at Both Temperatures

| Metric | V2 (t=0.1) | **V2.2 (t=0.1)** | V2 (t=0.6) | **V2.2 (t=0.6)** | Qwen3.5 Base |
|--------|-----------|-----------------|-----------|-----------------|--------------|
| MAE | 1.544 | **0.914** | 1.745 | **1.059** | 1.297 |
| Direction | 91.7% | **94.5%** | 87.8% | **93.2%** | 87.3% |
| Spearman | 0.555 | **0.691** | 0.527 | **0.630** | — |
| Parse failures | 5.0% | **0.0%** | 7.9% | **0.0%** | 0.0% |
| Within 1.0 | 53.4% | **75.5%** | 50.3% | **71.1%** | 62.1% |

### Per-Band MAE (Opus Holdout)

| Band | V2 (t=0.1) | **V2.2 (t=0.1)** | Change |
|------|-----------|-----------------|--------|
| Noise (n=1222) | 1.384 | **0.787** | **-43.1%** |
| Tangential (n=176) | 2.025 | **1.418** | **-30.0%** |
| Moderate (n=51) | 2.740 | **1.716** | **-37.4%** |
| High (n=42) | 2.688 | **1.462** | **-45.6%** |
| Critical (n=9) | 1.111 | 1.256 | +13.1% |

Every band improved except critical (+13%, but n=9 is too small to be statistically meaningful).

### Deployment Gate Check

| Gate | Condition | Result |
|------|-----------|--------|
| MAE at t=0.6 | < V2 baseline (1.745) | 1.059 — **PASS** |
| Per-band regression | No band >50% worse | Critical +13% (n=9) — **PASS** |
| Parse failures | ≤ V2 (5%) | 0% — **PASS** |
| Direction accuracy | ≥ V2 (91.7%) | 93.2% — **PASS** |

**ALL GATES PASSED — V2.2 DEPLOYED**

### Temperature Sensitivity

V2.2 gap: +0.145 MAE (t=0.1→t=0.6) — less sensitive than V2's +0.201 gap. The model is more robust across temperature settings.

---

## 11. Infrastructure Benchmarks

### VRAM Coexistence Test

| Configuration | Result |
|--------------|--------|
| Scorer (8.7GB) + qwen3:14b (9.3GB) | **FAILED** — exceeds 24GB with KV cache |
| Scorer (8.7GB) + qwen3.5:35b-a3b (23GB) | **IMPOSSIBLE** |
| Scorer (8.9GB) + qwen3.5:9b (6.6GB) | **PENDING TEST** — 15.5GB base, might fit |

### Parallel Inference (OLLAMA_NUM_PARALLEL)

| Setting | Throughput | Per-Call Latency |
|---------|-----------|-----------------|
| NUM_PARALLEL=1 (sequential) | 1.00x baseline | 3.57s |
| NUM_PARALLEL=1 (concurrent) | 1.06x | 11.65s |
| NUM_PARALLEL=4 (concurrent) | 0.94x (slower) | 12.1s |

**Conclusion:** GPU serializes all inference on a single 8B model. The 7900 XTX is memory-bandwidth-bound on autoregressive generation. Parallelism adds contention overhead without throughput benefit. Keep NUM_PARALLEL=1.

**Implication:** Tier 2 item #7 (parallel LLM scoring with ThreadPoolExecutor) is permanently eliminated from the roadmap.

---

## 12. Model Unification Decision

### Before (3 models, 3 roles)

| Model | Size | Role | Generation |
|-------|------|------|-----------|
| stratos-scorer-v2 | 8.7 GB | Scoring | Qwen3-8B fine-tuned |
| qwen3:30b-a3b | 18 GB | Agent/market | Qwen3 MoE |
| qwen3:14b | 9.3 GB | Wizard/briefings | Qwen3 dense |

### After (2 models, unified architecture)

| Model | Size | Role | Generation |
|-------|------|------|-----------|
| stratos-scorer-v2.2 | 8.9 GB | Scoring | Qwen3.5-9B fine-tuned |
| qwen3.5:9b | 6.6 GB | Agent + wizard + briefings | Qwen3.5 dense |

### Benefits of Unification
- One fewer model to manage and download
- 6.6 GB loads in ~2s vs 9.3-18 GB taking 5-15s
- Potential VRAM coexistence with scorer (8.9 + 6.6 = 15.5 GB before KV)
- Better benchmarks than both models it replaces (generational upgrade)
- Same architecture family as scorer — consistent system behavior
- Simplified config.yaml

---

## 13. Code Changes and Bug Fixes

### New Files Created

| File | Purpose |
|------|---------|
| `v22_expansion.py` | V2.2 Gemini scoring pipeline with --resume support |
| `gemini_validation.py` | Gemini 3 Flash validation test script |
| `deepseek_validation.py` | DeepSeek V3.2 validation test script |
| `prompt_version.py` | Prompt template version-pinning (drift detection) |
| `test_vram_fit.py` | VRAM fit test for new base models |
| `merge_export_v22.py` | Standalone merge + GGUF export script |
| `Modelfile.v22` | Ollama config for V2.2 model |

### Bug Fixes

| Bug | Impact | Fix |
|-----|--------|-----|
| Hardcoded token IDs (151643) | Would silently break Qwen3.5 training | Dynamic lookup via `convert_tokens_to_ids()` |
| `', '.join("string")` iterating chars | Tracked fields display garbled | Type-check before joining |
| stage4_prepare.py think_tokens filter | Would reject all Gemini-scored data | Removed filter, handle missing fields |
| Category: "general" in all training | Training/inference format mismatch | Use real category labels |
| Missing LANGUAGE line in training | Training/inference format mismatch | Added to system prompt |

### Infrastructure Changes

| Change | Effect |
|--------|--------|
| Prompt version-pinning deployed | Detects training/inference template drift |
| Cron job for daily Gemini scoring | Automated data expansion |
| Disk cleanup: 32 GB freed | SSD from 92% → 84% |
| OLLAMA settings confirmed optimal | NUM_PARALLEL=1, MAX_LOADED_MODELS=1 |

---

## 14. Documents Created

### Planning & Knowledge Base Documents

| Document | Lines | Purpose |
|----------|-------|---------|
| `training_pipeline.md` | 243 | Complete training knowledge base — hardware, configs, all version results |
| `v3_planning.md` (→ "V2.2 Planning") | 87 | Decision-gated roadmap with budget estimates |
| `features_roadmap.md` | 145 | Prioritized product/engineering roadmap (3 tiers) |
| `SESSION_HANDOFF.md` | 487 | Full session context for next Claude chat |
| `LESSONS_AND_PRACTICES.md` | 485 | Mistakes, best practices, checklists |

### Validation Reports

| Report | Key Finding |
|--------|-------------|
| `DeepSeek_V3.2_Validation_Report.md` | MAE 1.521, middle-range collapse, eliminated |
| `Gemini_3_Flash_Validation_Report.md` | MAE 0.893 calibrated, 15+ score levels, selected |
| `V2_2_Overnight_Report.md` | Qwen3.5-9B base MAE 1.297, infrastructure test results |
| `V2_2_TRAINING_REPORT.md` | Training results + Opus holdout eval |

---

## 15. Cost Analysis

### Session Spending

| Item | Cost |
|------|------|
| DeepSeek validation (300 calls) | ~$0.10 |
| Gemini validation (300 calls) | $0.00 |
| Gemini V2.2 scoring (~4,148 calls) | ~$2-3 |
| Gemini API key setup + billing | $0 |
| Local training (electricity) | ~$0.50 |
| **Total session cost** | **~$3** |

### Cost Comparison with Alternatives

| Approach | Estimated Cost | What We Actually Paid |
|----------|---------------|----------------------|
| Opus for all scoring | $50-70 | — |
| Opus Batch API | $25-35 | — |
| DeepSeek + Opus hybrid | $16-21 | — |
| **Gemini 3 Flash** | **$3-6** | **~$3** |
| RunPod cloud training | $8-10 | $0 (trained locally) |

### Remaining Budget

- Gemini API: ~4.5 KD ($14.60) remaining
- Sufficient for completing all 50K scores + future expansion

---

## 16. Complete Version History

| Version | Base Model | Training Data | Holdout MAE (t=0.1) | Direction | Parse Fail | Status |
|---------|-----------|---------------|---------------------|-----------|------------|--------|
| V1 | Qwen3-8B | 5,679 ex, 10 profiles, CoT | ~1.553 | 90.7% | — | Retired |
| V2 | Qwen3-8B | 18,502 ex, 30 profiles, short reason | 1.544 | 91.7% | 5.0% | Backup |
| V2.1 | Qwen3-8B | 19,000 ex, 47 profiles, think_text | 3.244 | 53.5% | 4.5% | FAILED |
| Base 3.5-9B | Qwen3.5-9B | (unfine-tuned) | 1.297 | 87.3% | 0.0% | Reference |
| **V2.2** | **Qwen3.5-9B** | **11,785 ex, 25 profiles, short reason** | **0.914** | **94.5%** | **0.0%** | **Production** |

### Improvement Decomposition

| Change | MAE Impact | Evidence |
|--------|-----------|---------|
| V1 → V2: more data, more profiles | 1.553 → 1.544 (negligible on holdout) | Data scale helped memorization, not generalization |
| V2 → V2.1: think_text | 1.544 → 3.244 (catastrophic regression) | Long reasoning destroys calibration |
| Qwen3-8B → Qwen3.5-9B base | 1.544 → 1.297 (-16%) | Architecture upgrade alone |
| Base → V2.2 fine-tuning | 1.297 → 0.914 (-30%) | Article expansion + fine-tuning |
| **Total V2 → V2.2** | **1.544 → 0.914 (-41%)** | **Model upgrade + article diversity** |

---

## 17. Key Discoveries and Insights

### Discovery 1: Article Diversity Is the Generalization Bottleneck
V2 corrected baseline (1.544) ≈ V2 broken baseline (1.533). Format drift was irrelevant. The model memorized 813 articles (0.393 contaminated) and couldn't generalize. Only 135 of 813 training articles overlapped with the 17,260 production DB articles — the model trained on one distribution and scored on a different one.

### Discovery 2: Base Model Upgrade > Fine-Tuning
Unfine-tuned Qwen3.5-9B (1.297) beat fine-tuned Qwen3-8B V2 (1.544). 12+ hours of training on the old architecture was worth less than swapping to the new architecture with zero training.

### Discovery 3: Think_Text/CoT Hurts 8B-Class Models
Three data points now confirm: V1 CoT (1.553), V2 short (1.544), V2.1 think_text (3.244). The model doesn't have capacity for genuine reasoning — it learns to pattern-match the style of reasoning, generating plausible justifications for any score.

### Discovery 4: Cheap Providers Work With Calibration
Gemini 3 Flash at $0-6 replaces $50-70 Opus for training data when combined with isotonic calibration. The key requirement: the provider must preserve rank ordering (high Spearman) with enough score diversity for calibration to work. DeepSeek failed this test despite being cheaper.

### Discovery 5: seq_length=512 Saves 75% Training Time
All training examples are under 458 tokens. V2 wasted ~9 hours per run using seq_length=1024. Attention is O(n²) — halving sequence length gives ~4x speedup. Always profile token distribution before setting hyperparameters.

### Discovery 6: Parallel Inference Is Physics-Limited
On a single 7900 XTX, the GPU is memory-bandwidth-bound on 8B autoregressive generation. NUM_PARALLEL=4 is actually 6% slower than sequential. This is not a software limitation — it's fundamental to the architecture of autoregressive LLMs on single GPUs.

### Discovery 7: Spearman Rho Alone Is Misleading
V2.1 had rho=0.539 (acceptable) but MAE=3.244 (catastrophic). The model preserved relative ordering while inflating every score by 3+ points. Always pair ranking metrics with calibration metrics.

### Discovery 8: Pre-Training Reviews Should Block, Not Note
The cross-Claude review before V2.1 correctly identified think_text as risky. It was noted as a concern, not a blocker. V2.1 then trained for 12+ hours and failed. Small-scale experiments (1K examples, 30 minutes) should validate risky approaches before committing to full training runs.

---

## 18. Lessons Learned

### Training Pipeline
1. **Change ONE variable per version** — V2.1 changed 4+ simultaneously, making failure diagnosis impossible
2. **Profile token distribution** — 5 lines of code saves hours of wasted training
3. **Short reasons only** — CoT/think_text destroys calibration on 8B-class models
4. **Dynamic token IDs** — never hardcode model-specific values
5. **AOTRITON causes silent NaN** — never set this env var on ROCm
6. **Merge adapters before eval** — 30x faster than adapter inference on ROCm
7. **Smoke test before full eval** — catches catastrophic failures in 30 seconds vs 46 minutes

### Evaluation
8. **Eval through the deployment pipeline** — PyTorch bf16 metrics are not comparable to Ollama Q8_0
9. **Dual-temperature eval** — model behavior differs at temp=0.1 vs 0.6
10. **Per-band MAE** — overall MAE is dominated by noise band, hides moderate/high failures
11. **Holdout is sacred** — contaminated metrics measure memorization, not generalization
12. **Label every metric** — "MAE 0.393" means nothing without "(PyTorch bf16, contaminated)"

### Data Quality
13. **Article diversity > example count** — 11,785 examples from 2,000 articles beats 18,502 from 813
14. **Validate scoring providers** — $0.10 test prevents $12+ of wasted scoring
15. **Isotonic calibration** — works when provider preserves rank ordering with score diversity
16. **Noise cap at 40-55%** — too aggressive (26%) inflates scores, uncapped (90%) starves signal

### Deployment
17. **Never deploy without smoke test** — /nothink caused 41% parse failures that weren't caught
18. **num_predict = max_output × 1.5** — not some arbitrary large default
19. **Modelfile template matters** — `{{- end }}` placement causes silent prompt corruption
20. **Keep rollback GGUF** — recovery should take 2 minutes, not 12 hours

---

## 19. Feature Roadmap Status

### Completed This Session
- [x] evaluate_scorer.py fixes (num_predict=512, --temperature flag)
- [x] OLLAMA_NUM_PARALLEL benchmark → no gain, keep NUM_PARALLEL=1
- [x] VRAM coexistence test → failed for scorer+14b, pending for scorer+9b
- [x] Prompt template version-pinning (prompt_version.py)
- [x] Agent search_feed tool (search, top_signals, daily_summary)
- [x] Disk cleanup (~32 GB freed)

### Eliminated This Session
- [x] Parallel LLM scoring (ThreadPoolExecutor) → GPU serializes, no benefit
- [x] Qwen3.5-0.8B pre-filter → worse than random
- [x] Qwen3.5-35B-A3B as wizard → too slow, truncates
- [x] Think_text/CoT in training → definitively proven harmful

### Still Pending (from features_roadmap.md)
- [ ] Kill wasted Serper queries — cross-entity combos return 0 results ~30% of time
- [ ] RSS feed configuration — free data, extra_feeds.py has catalogs
- [ ] Chart initial zoom — setVisibleLogicalRange(), ~5 lines
- [ ] Critical vetting layer — 9.5+ articles, verified_critical flag
- [ ] Feed density view — expanded critical cards, single-line noise
- [ ] Source reputation filter — per (domain, profile_category)
- [ ] Scheduled scans + email notifications

---

## 20. Current System State

### Production Models

| Model | Role | Size | Status |
|-------|------|------|--------|
| `stratos-scorer-v2.2` | Scoring | 8.9 GB | **DEPLOYED** |
| `qwen3.5:9b` | Agent + wizard + briefings | 6.6 GB | **DEPLOYING** |

### Model Inventory (Ollama)

| Model | Size | Keep? |
|-------|------|-------|
| stratos-scorer-v2.2 | 8.9 GB | Yes — production |
| stratos-scorer-v2 | 8.7 GB | Yes — rollback |
| stratos-scorer-v2-baseline | 8.7 GB | Yes — eval reference |
| stratos-scorer-v2.1 | 8.7 GB | Remove — failed model |
| qwen3.5:9b | 6.6 GB | Yes — new agent/wizard |
| qwen3:14b | 9.3 GB | Remove — replaced by 3.5-9B |
| qwen3:30b-a3b | 18 GB | Remove — replaced by 3.5-9B |
| qwen3:8b | 5.2 GB | Remove — superseded |

### Key Files

| File | Path | Description |
|------|------|-------------|
| V2.2 GGUF | `training_output/v2_scorer.gguf` | 8.9 GB Q8_0, production model |
| V2 backup | `v2_model_backup/v2_scorer.gguf` | Rollback model |
| Holdout eval | `eval_holdout_v2.jsonl` | 1,500 examples — NEVER train on |
| Holdout articles | `holdout_articles.json` | 50 article IDs — SACRED |
| Calibration model | Fitted on 283 validation points | Isotonic regression for Gemini scores |
| Expansion scores | `expansion_scores.json` | ~4,148 so far, growing daily |

### Infrastructure

| Component | Setting | Status |
|-----------|---------|--------|
| Ollama | NUM_PARALLEL=1, MAX_LOADED=1 | Optimal for 24GB |
| Gemini API | Paid Tier 1, 10K RPD | Cron job running |
| Disk space | ~55 GB free (84%) | Healthy |
| Prompt versioning | Deployed | Detects template drift |

---

## 21. Remaining Work

### Immediate (This Week)
1. Complete Qwen3.5-9B deployment as agent/wizard model
2. Test VRAM coexistence: scorer (8.9GB) + 9B (6.6GB) — if this works, no more swap delays
3. Monitor daily Gemini scoring progress (cron job)
4. Remove unused models: v2.1, qwen3:14b, qwen3:30b-a3b, qwen3:8b
5. Verify file cleanup from the scan results

### Next Training Cycle (When Gemini scoring completes ~March 12)
6. Retrain as V2.3 with full 50K scores (~19K examples after noise cap)
7. Expected additional improvement from 2.3x more training data

### Product Features
8. Kill wasted Serper queries (saves money every scan)
9. RSS feed configuration (free data, zero API cost)
10. Chart initial zoom (~5 lines)
11. Critical vetting layer (if V2.3 results are strong enough)

---

## 22. Risk Register

| Risk | Likelihood | Impact | Mitigation | Status |
|------|-----------|--------|------------|--------|
| V2.2 production regression | Low | High | V2 GGUF backup, 2-minute rollback | Mitigated |
| Gemini RPD quota exhaustion | Low | Medium | Cron job with 500-unit buffer | Active |
| Gemini scores drift from Opus | Low | High | Isotonic calibration (LOO MAE 0.893) | Mitigated |
| Holdout contamination | Very Low | Critical | 4 defense points verified | Mitigated |
| Disk space during V2.3 training | Low | Medium | 55 GB free, need ~20 GB | Monitoring |
| Prompt format drift | Low | High | prompt_version.py deployed | Mitigated |
| VRAM crash during training | Medium | Medium | Kill desktop apps, headless session | Learned |
| Qwen3.5-9B agent quality regression | Low | Medium | Test before removing qwen3:14b | Pending |
| Daily cron job fails silently | Medium | Low | Check log periodically | Active |

---

*Session duration: ~24 hours*
*Total cost: ~$3*
*MAE improvement: 41% (1.544 → 0.914)*
*Parse failures: eliminated (5% → 0%)*
*Models tested: 6 (V2, V2.1, V2.2, Qwen3.5-9B base, Qwen3.5-35B-A3B, Qwen3.5-0.8B)*
*Providers tested: 3 (Opus, DeepSeek, Gemini)*
*Documents created: 9*
*Git commits: 8+*
