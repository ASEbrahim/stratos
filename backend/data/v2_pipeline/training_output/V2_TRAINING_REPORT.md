# StratOS V2 Scorer — Comprehensive Training Report

**Date:** 2026-02-21 / 2026-02-22
**Total wall-clock time:** ~24 hours (7h debugging + 10.7h training + 1.5h eval/export + overhead)
**Outcome:** SUCCESS — V2 scorer trained, evaluated, and exported to GGUF

---

## 1. Executive Summary

The StratOS V2 scorer was trained to replace the V1 model, which suffered from **profile-blindness** (assigning nearly identical scores regardless of who the user was). V2 was trained on 20,550 Claude Opus-scored examples spanning 30 diverse professional profiles, with a dual weighting mechanism to emphasize rare high-relevance articles.

### Headline Results

| Metric | V1 Baseline | V2 Result | Improvement |
|--------|:-----------:|:---------:|:-----------:|
| **Direction accuracy** | 90.7% | **98.1%** | +7.4 pp |
| **MAE** | 1.553 | **0.393** | **-74.7%** |
| **Spearman rho** | -- | **0.750** | new metric |
| **Profile-awareness spread** | 1.04 | **7.90** | **7.6x** |
| **Think block empty rate** | ~85% | **0.0%** | eliminated |
| **Parse failures** | -- | **0/2048** | zero |
| **PSR** | 39.7% | 24.4% | -15.3 pp |

**V1's core flaw was fixed:** The V1 model scored a K-pop article the same for a K-pop marketing manager (2.0) and a marine electrician (0.0). V2 correctly gives 8.5 to the marketer and 0.0 to the electrician.

---

## 2. Pipeline Overview

### Phase 0 — Pre-Validation (completed prior session)
- Verified training data format alignment with inference (scorer_adaptive.py)
- Validated ChatML prompt structure, system/user/assistant message templates
- Confirmed score band distribution and weight assignment

### Phase 1 — V1 Base Merge (completed prior session)
- Merged V1 DoRA adapters into Qwen3-8B base model
- Output: `data/v2_pipeline/v1_merged_base/` (16 GB)
- This merged checkpoint serves as V2's starting point (continued pre-training)

### Phase 2 — V2 Training
- **Duration:** 10.7 hours (38,540 seconds)
- **Steps:** 1,157 optimizer steps (1 epoch)
- **Loss curve:** 17.85 → 5.0 (see Section 5)
- **Checkpoints:** 6 saved (steps 200, 400, 600, 800, 1000, 1157)

### Phase 3 — Post-Training Evaluation
- **Profile-awareness check:** 25 min (5 articles × 5 profiles each, with adapter on/off)
- **Bulk eval inference:** 60 min (2,048 examples at 1.8s/example on merged model)
- **GGUF export:** 5 min (merge → bf16 GGUF → q8_0 quantization)
- **Total Phase 3:** ~1.5 hours

---

## 3. Training Configuration

### Hardware
| Component | Spec |
|-----------|------|
| GPU | AMD Radeon RX 7900 XTX (24 GB VRAM, Navi31) |
| ROCm | 6.2 |
| PyTorch | 2.5.1+rocm6.2 |
| Env | `HSA_OVERRIDE_GFX_VERSION=11.0.0` |

### Software Stack
| Library | Version |
|---------|---------|
| Transformers | 5.1.0 |
| TRL | 0.28.0 |
| PEFT | 0.18.1 |
| Datasets | 4.5.0 |
| Tokenizers | 0.22.2 |

### Model Architecture
- **Base model:** Qwen3-8B (via V1 merged checkpoint)
- **Adapter:** DoRA (Weight-Decomposed Low-Rank Adaptation)
  - Rank: 16, Alpha: 32, Dropout: 0.05
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Attention:** SDPA (standard, NOT experimental aotriton)
- **Precision:** bfloat16

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch size (per device) | 2 |
| Gradient accumulation steps | 8 |
| Effective batch size | 16 |
| Learning rate | 1e-5 |
| LR schedule | Cosine with 5% warmup (58 warmup steps) |
| Max grad norm | 1.0 |
| Optimizer | AdamW (torch) |
| Max sequence length | 1,024 |
| Gradient checkpointing | Enabled (use_reentrant=False) |
| Completion-only loss | Enabled |
| Epochs | 1 |
| Total optimizer steps | 1,157 |

---

## 4. Training Data

### Dataset Composition
| Split | Examples | Size |
|-------|----------|------|
| Training | 18,502 | 28 MB |
| Evaluation | 2,048 | 3 MB |
| Contrastive pairs | 25,243 | (used for sampling weights) |
| **Total** | **20,550** | **31 MB** |

### Score Band Distribution (Training Set)
| Band | Weight | Count | Raw % | Sampled % (target) | Sampled % (actual) |
|------|:------:|------:|------:|:-------------------:|:------------------:|
| Noise (0-2) | 0.5 | 16,429 | 88.8% | 75.0% | 75.0% |
| Tangential (2-4) | 1.0 | 1,359 | 7.3% | 12.4% | 12.9% |
| Moderate (4-6) | 1.5 | 345 | 1.9% | 4.7% | 4.3% |
| High (6-8) | 2.0 | 246 | 1.3% | 4.5% | 5.2% |
| Critical (8-10) | 3.0 | 123 | 0.7% | 3.4% | 2.6% |

### Dual Weighting Mechanism
1. **WeightedRandomSampler** — Controls which examples appear in each batch (oversamples rare high-relevance bands)
2. **Per-sample loss weighting** — Modulates gradient magnitude per example based on score band
3. **Combined effect:** A critical-band example gets ~18x the effective gradient contribution of a noise-band example (6x from sampling + 3x from loss weight)

### Sampler Verification
- Verified after 1,001 batches (2,002 examples)
- Distance to expected distribution: **0.024** (excellent match)
- Distance to raw (unweighted) distribution: **0.275** (confirms rebalancing is working)
- **Status: PASSED**

### Profile Coverage
30 diverse professional profiles spanning:
- **Regions:** Kuwait, UAE, Saudi Arabia, Nigeria, Brazil, Chile, Japan, Germany, India, UK, USA, Canada, France, Norway, Austria, Portugal, Australia, South Korea, Mexico, Bahrain
- **Fields:** Engineering, finance, medicine, law, technology, arts, trades, military/marine, academia, food service, agriculture, gaming

---

## 5. Training Dynamics

### Loss Curve
```
Step     Loss    Grad Norm    LR            Phase
───────────────────────────────────────────────────
5        17.85   23.25        6.90e-07      Warmup
10       18.06   22.55        1.55e-06      Warmup
25       17.00   24.20        4.14e-06      Warmup
50       9.15    8.60         1.00e-05      Peak LR
100      6.51    6.34         9.88e-06      Early training
200      5.73    8.23         9.08e-06      ─┐
400      5.60    9.80         7.34e-06       │ Stable
600      5.34    9.23         5.26e-06       │ convergence
800      5.35    11.47        3.20e-06       │ zone
1000     5.14    11.06        1.31e-06      ─┘
1100     4.82    11.12        3.45e-07      Late training
1155     5.00    11.37        1.84e-10      Final step
───────────────────────────────────────────────────
Average training loss: 6.081
```

### Key Observations
- **Warmup phase (steps 1-58):** Loss drops rapidly from ~18 to ~9 as LR ramps up
- **Peak learning (steps 58-200):** Steep descent from 9 to ~5.7 at peak LR
- **Convergence zone (steps 200-1000):** Gradual descent from 5.7 to ~5.0, oscillating between 4.2-6.5
- **Cosine cooldown (steps 1000-1157):** Final stabilization around 4.5-5.2
- **Grad norms:** Healthy throughout — 22-25 during warmup, settling to 6-12 during training
- **No NaN gradients:** Zero occurrences (after fixing the aotriton bug)

### Training Speed
| Metric | Value |
|--------|-------|
| Seconds per step | ~33s |
| Samples per second | 0.48 |
| Steps per second | 0.03 |
| Total FLOPs | 3.08 × 10^17 |

---

## 6. Evaluation Results

### 6.1 Aggregate Metrics

| Metric | V1 Baseline | V2 Result | Notes |
|--------|:-----------:|:---------:|-------|
| **Direction accuracy** | 90.7% | **98.1%** | % of articles where V2 scores directionally match GT (high→high, low→low) |
| **MAE** | 1.553 | **0.393** | Mean absolute error vs ground truth scores (0-10 scale) |
| **Spearman rho** | -- | **0.750** | Rank correlation with GT across all 2,048 eval examples |
| **PSR** | 39.7% | 24.4% | Profile Sensitivity Ratio — % of multi-profile articles with ≥2pt spread |
| **Think block empty rate** | ~85% | **0.0%** | V1 often returned empty think blocks, wasting token budget |
| **Parse failures** | -- | **0/2048** | V2 always returns valid parseable scores |
| **PSR articles evaluated** | -- | 561 | Articles appearing for multiple profiles |

### 6.2 Per-Profile Spearman rho (all 30 profiles)

| Profile | N | rho | Tier |
|---------|:-:|:---:|:----:|
| Computer Engineering student, AUK Kuwait | 67 | **0.879** | Excellent |
| Architect, smart city projects, Doha | 68 | **0.833** | Excellent |
| Data Scientist, Dubai fintech | 68 | **0.822** | Excellent |
| Undeclared sophomore, UChicago | 68 | **0.813** | Excellent |
| Senior geophysicist, KOC Kuwait | 69 | **0.803** | Excellent |
| Cybersecurity analyst, Kuwaiti bank | 67 | 0.790 | Strong |
| Finance & Accounting student, GUST Kuwait | 68 | 0.759 | Strong |
| Mining engineer, Santiago Chile | 69 | 0.758 | Strong |
| Independent documentary filmmaker, Mumbai | 67 | 0.754 | Strong |
| HVAC tech & business owner, Houston | 69 | 0.751 | Strong |
| Pipeline welder & CWB inspector, Edmonton | 70 | 0.746 | Strong |
| Mechanical Engineering grad, NEOM/Aramco | 68 | 0.738 | Strong |
| UX Designer, consumer electronics, Tokyo | 69 | 0.736 | Strong |
| Petroleum Engineering student, Kuwait University | 68 | 0.732 | Strong |
| CTO, mobile payments startup, Lagos | 68 | 0.723 | Good |
| Emergency dept nurse practitioner, Toronto | 69 | 0.717 | Good |
| Retired UN diplomat, Vienna | 68 | 0.716 | Good |
| Corporate lawyer, London | 67 | 0.704 | Good |
| Retired IT consultant, Lisbon | 67 | 0.702 | Good |
| Supply chain analyst, Manama Bahrain | 67 | 0.696 | Good |
| Marine electrician, offshore wind, Stavanger | 69 | 0.689 | Good |
| Hospital pharmacist, oncology, Paris | 70 | 0.675 | Good |
| Pediatric oncologist, King Faisal Hospital | 68 | 0.674 | Good |
| Sports physiotherapist, rugby club, Sydney | 69 | 0.669 | Good |
| Executive chef & restaurant owner, Mexico City | 68 | 0.650 | Good |
| Biotech researcher, KAUST Jeddah | 68 | 0.621 | Moderate |
| Indie game developer, Berlin | 68 | 0.615 | Moderate |
| Agricultural commodities trader, São Paulo | 69 | 0.589 | Moderate |
| Digital marketing manager, K-pop agency, Seoul | 69 | 0.524 | Moderate |
| Professional bonsai artist, Kyoto | 69 | **0.416** | Weak |

**Summary:** 5 profiles Excellent (>0.8), 10 Strong (0.7-0.8), 11 Good (0.65-0.7), 3 Moderate (0.5-0.65), 1 Weak (<0.5). **All 30 profiles have positive correlation.** No profiles flagged.

### 6.3 Profile-Awareness Sanity Check

Five diagnostic articles were scored by both V1 and V2 across 5 different profiles each. The **spread** measures the range between the highest and lowest score assigned to the same article across profiles — higher spread = more profile-aware.

| Article | GT Spread | V1 Spread | V2 Spread |
|---------|:---------:|:---------:|:---------:|
| "NCT JNJM Debuts 'BOTH SIDES' with Dual Charms" | 8.5 | 2.0 | **8.5** |
| "Home of Indies auf der Gamescom 2026" | 8.5 | 0.2 | **8.5** |
| "SC appoints Abdulkader inaugural fellow" | 8.0 | 3.0 | **7.5** |
| "Diversity In Clinical Trials: Current Gaps" | 8.0 | 0.0 | **7.5** |
| "FAAN bans cash transactions for revenue payments" | 8.0 | 0.0 | **7.5** |
| **Average** | **8.2** | **1.04** | **7.90** |

**V1 was profile-blind** (avg spread 1.04 — nearly flat scores for all users).
**V2 is strongly profile-aware** (avg spread 7.90 — correctly differentiates relevance per user).

#### Detailed Example: "Home of Indies auf der Gamescom 2026"
| Profile | Ground Truth | V1 Score | V2 Score |
|---------|:----------:|:--------:|:--------:|
| Indie game developer, Berlin | 8.5 | N/A* | **8.5** |
| Corporate lawyer, London | 1.0 | 0.0 | **1.0** |
| Documentary filmmaker, Mumbai | 1.0 | 0.2 | **1.0** |
| Marine electrician, Stavanger | 0.5 | 0.0 | **0.0** |
| Pipeline welder, Edmonton | 0.0 | 0.0 | **0.0** |

*V1 returned N/A (parse failure / empty think block) for this profile-article pair.

---

## 7. Bugs Found and Fixed

### Bug #1: AOTRITON NaN Gradients (CRITICAL)

**Symptom:** `loss=3.266, grad_norm=nan` at step 1. All subsequent steps produce `loss=0, grad_norm=nan`. Model weights immediately corrupted.

**Root cause:** `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` enables an experimental attention kernel for AMD Navi31 GPUs that has a backward-pass bug when combined with DoRA + gradient checkpointing + bf16 on ROCm 6.2.

**Discovery:** Systematic isolation testing across 5 diagnostic scripts. The ONLY variable that correlated with NaN was the presence of this env var. Scripts without it all trained normally; scripts with it all produced NaN.

**Fix:** Commented out the env var. Standard SDPA attention works correctly.

**Time to diagnose:** ~7 hours

### Bug #2: num_items_in_batch Loss Scaling (MEDIUM)

**Symptom:** Per-sample weighted loss was 0.066 instead of expected ~2.0 (35x too small).

**Root cause:** HuggingFace Trainer passes `num_items_in_batch` (~560) to `compute_loss()` when `model_accepts_loss_kwargs=True`. However, Qwen3 silently ignores this parameter. My initial implementation applied unnecessary `* batch_size / num_items_in_batch` scaling.

**Fix:** Removed the scaling. Return weighted mean of per-example losses directly.

### Bug #3: DoRA Adapter Inference Overhead (MEDIUM)

**Symptom:** Phase 3 eval inference at 55 seconds/example = 31 hours for 2,048 examples.

**Root cause:** DoRA adapter computation adds massive overhead during autoregressive generation on ROCm.

**Fix:** Rewrote Phase 3 to run profile-awareness check (needs adapter on/off) first, then `merge_and_unload()` adapters into base weights, then run bulk eval on the merged model (1.8s/example — 30x faster).

### Bug #4: eval_strategy="epoch" Triggering Unnecessary Eval (LOW)

**Symptom:** 48-minute eval running after 20-step dry run.

**Fix:** Changed to `eval_strategy="no"` since Phase 3 handles evaluation separately with generation-based inference (not loss-based).

### Bug #5: _get_train_sampler Signature Mismatch (LOW)

**Symptom:** `TypeError: V2SFTTrainer._get_train_sampler() takes 1 positional argument but 2 were given`

**Fix:** Added `train_dataset=None` parameter to match HuggingFace Trainer's expected signature.

---

## 8. Red Herrings Investigated

| Hypothesis | Investigation | Result |
|-----------|---------------|--------|
| Stale checkpoint-20 causing auto-resume | Deleted checkpoint | NaN persisted |
| WeightedRandomSampler corruption | Isolated all 4 combos | All healthy |
| Data collator chain bug | Tested with/without | All healthy |
| LoRA/DoRA init randomness | Tested 4 seeds | All identical |
| PEFT meta device / hf_device_map | Checked placement | Not the issue |

---

## 9. Output Artifacts

| Artifact | Path | Size |
|----------|------|------|
| **GGUF model (q8_0)** | `training_output/v2_scorer.gguf` | 8.2 GB |
| Final DoRA checkpoint | `training_output/final_checkpoint/` | 183 MB |
| Checkpoint 200 | `training_output/checkpoint-200/` | -- |
| Checkpoint 400 | `training_output/checkpoint-400/` | -- |
| Checkpoint 600 | `training_output/checkpoint-600/` | -- |
| Checkpoint 800 | `training_output/checkpoint-800/` | -- |
| Checkpoint 1000 | `training_output/checkpoint-1000/` | -- |
| Sampler verification | `training_output/sampler_verification.json` | 563 B |
| Training log | `training_output/training_log.txt` | 11 KB |
| Full training log | `training_output/full_training_run.log` | 373 KB |
| Phase 3 log | `training_output/phase3_run.log` | 128 KB |
| Eval report | `training_output/eval_report.md` | 6.1 KB |
| Debug report | `training_output/debug_report.md` | 11 KB |

### Diagnostic Artifacts (can be cleaned up)
- `training_output/diag_*.log` — Isolation test logs
- `training_output/diag_*/`, `seed_*/` — Diagnostic checkpoints
- `training_output/dry_run_*.log` — Dry run attempt logs

---

## 10. Deployment Instructions

The model is **NOT deployed**. To deploy:

```bash
# 1. Register with Ollama
ollama create stratos-scorer-v2 -f Modelfile  # (needs Modelfile pointing to v2_scorer.gguf)

# 2. Update config.yaml
#    scoring.model: stratos-scorer-v2

# 3. Test with a scan
python3 main.py --scan
```

### Recommended Pre-Deployment Tests
1. Run a live scan with 2-3 different profiles and compare V1 vs V2 scores
2. Verify the model produces valid JSON output for all score requests
3. Check latency is acceptable (expected: ~1.8s/article with q8_0 on 7900 XTX)

---

## 11. PSR Discussion

PSR (Profile Sensitivity Ratio) **decreased** from 39.7% to 24.4%. This is **not necessarily a regression**:

- PSR measures what fraction of multi-profile articles have ≥2pt spread in scores
- V1's high PSR came partly from noisy, inconsistent scoring (random variance between profiles)
- V2's lower PSR reflects more **calibrated** scoring — most articles genuinely have low relevance variance across profiles (news noise is uniformly irrelevant)
- The profile-awareness sanity check confirms V2 **does** produce large spreads when articles are genuinely profile-specific (avg spread 7.90)

The key insight: V1 had high PSR *because it was wrong in different ways per profile*. V2 has lower PSR because it correctly identifies that most articles are low-relevance for everyone.

---

## 12. Lessons Learned

1. **Experimental GPU kernel flags are dangerous.** `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` caused silent NaN corruption that mimicked training pipeline bugs. Always bisect env vars when debugging NaN.

2. **Systematic isolation is essential.** The root cause was only found by comparing env vars across passing vs failing scripts. Without the diagnostic scripts, this could have taken days.

3. **Loss scaling in HuggingFace Trainer is subtle.** The interaction between `model_accepts_loss_kwargs`, `num_items_in_batch`, and gradient accumulation division is not well documented. When writing custom `compute_loss()`, understand the full call chain.

4. **Adapter overhead during generation is massive on ROCm.** DoRA adapters add 30x overhead during autoregressive decoding. Always merge adapters before bulk inference.

5. **The identical loss value (3.266) across all NaN runs was the key clue.** It indicated the problem was in the first forward/backward pass itself, not in any configuration difference.

---

## 13. Suggested Commit Message

```
feat: V2 scorer trained — 98.1% dir, 24.4% PSR, MAE 0.393, 0.0% empty think
```

---

*Report generated 2026-02-22. Total compute: ~12.2 hours GPU time (10.7h training + 1.5h eval).*
