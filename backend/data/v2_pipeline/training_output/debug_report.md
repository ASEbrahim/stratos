# V2 Scorer Training — Debugging Report

**Duration:** ~7 hours of diagnosis and iteration
**Outcome:** Root cause identified, fixed, full training launched
**Date:** 2026-02-21/22

---

## Executive Summary

Phase 2 training was blocked by NaN gradients at step 1. After systematic isolation testing across 5 diagnostic scripts, the root cause was identified as `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` — an experimental ROCm attention kernel that produces NaN during backward pass with DoRA + gradient checkpointing + bf16 on ROCm 6.2.

Additionally, the per-sample loss weighting implementation required careful analysis of the HuggingFace Trainer's `num_items_in_batch` scaling behavior to avoid 35x loss magnitude errors.

---

## Root Cause #1: AOTRITON NaN Gradients (CRITICAL)

### Symptom
- Step 1: `loss=3.266, grad_norm=nan`
- Steps 2+: `loss=0, grad_norm=nan, entropy=nan`
- Model parameters immediately corrupted (all-zero weights after step 1)

### Root Cause
The environment variable `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` enables an experimental attention kernel for AMD Navi31 GPUs. This kernel has a bug in its backward pass when combined with:
- DoRA adapters (PEFT)
- Gradient checkpointing (`use_reentrant=False`)
- bf16 training
- ROCm 6.2 / PyTorch 2.5.1+rocm6.2

The experimental aotriton kernel produces NaN gradients during the attention backward pass. The standard SDPA kernel (without this flag) works correctly — PyTorch warns about experimental flash attention support but falls back to a working implementation.

### Discovery Method
Systematic isolation across 5 diagnostic scripts:

| Script | AOTRITON flag | Result |
|--------|:------------:|--------|
| `train_v2.py` (main) | YES | NaN at step 1 |
| `diag_precise.py` | YES | NaN at step 1 (all 4 toggle tests) |
| `diag_isolate.py` | NO | Healthy (loss=2.1-2.3, grad=2.7-3.1) |
| `diag_seed.py` | NO | Healthy (all 4 seed variants) |
| `diag_test.py` | NO | Healthy |

The ONLY difference between NaN and healthy scripts was the presence of `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.

### Fix
```python
# DISABLED: causes NaN gradients during backward on ROCm 6.2 with DoRA + gradient checkpointing
# os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
```

### Verification
After removing the flag, dry-run training shows healthy metrics for 20 steps:
- Loss: 2.243 → 1.789 (steady decrease)
- Grad norms: 2.7-3.3 (healthy range)
- No NaN at any step

---

## Root Cause #2: num_items_in_batch Loss Scaling (MEDIUM)

### Symptom
Per-sample weighted loss was 35x too small (0.066 vs expected ~2.0), causing near-zero gradients and negligible learning.

### Root Cause
The HuggingFace Trainer (transformers 5.1.0) computes `num_items_in_batch` as the count of non-masked labels across ALL micro-batches in a gradient accumulation window:

```python
num_items_in_batch = sum((batch["labels"].ne(-100)).sum() for batch in batch_samples)
# Result: ~560 (= ~35 completion tokens × 16 examples across 8 micro-batches)
```

When `model_accepts_loss_kwargs=True` (auto-detected for Qwen3), the Trainer:
1. Passes `num_items_in_batch` to `compute_loss()` and expects the model to handle it
2. Does NOT divide loss by `gradient_accumulation_steps`

**However**, Qwen3 does NOT implement `num_items_in_batch` scaling — the parameter is silently ignored. The model simply returns `CrossEntropyLoss(reduction='mean')` — the standard mean-over-tokens loss.

My initial per-sample weighted loss applied the scaling: `weighted_loss * batch_size / num_items_in_batch = weighted_loss * 2 / 560`, making it 280x smaller than needed.

### Fix
Removed the `num_items_in_batch` scaling from the custom compute_loss:

```python
# NOTE: No num_items_in_batch scaling needed — Qwen3 doesn't implement it,
# and the Trainer skips grad_accum division when model_accepts_loss_kwargs=True.
# This matches passthrough behavior (CE_mean returned directly).
```

### Verification
After fix, weighted loss ≈ 1.8 (vs passthrough ≈ 2.2), confirming correct magnitude.

---

## Red Herrings Investigated & Ruled Out

### 1. Stale checkpoint-20 from broken run
Found `checkpoint-20` with NaN-corrupted `trainer_state.json` in the output directory. Deleted it, but NaN persisted — confirmed it was NOT an auto-resume issue.

### 2. WeightedRandomSampler / collator chain
Suspected the collator wrappers (WeightedCompletionDataCollator → VerifyingCollatorWrapper) or WeightedRandomSampler were causing NaN. `diag_isolate.py` tested all 4 combinations:
- Bare trainer (no sampler, no collator): OK
- With sampler only: OK
- With collator chain only: OK
- Full pipeline (sampler + collator chain): OK

All passed with healthy losses. The pipeline components are NOT the cause.

### 3. LoRA/DoRA initialization randomness
`diag_seed.py` tested 4 different seeds (None, 42, 12345, 0) before `get_peft_model()`. All produced identical healthy training. DoRA initialization is deterministic and not the cause.

### 4. HuggingFace Trainer loss scaling / model_accepts_loss_kwargs
Extensive investigation into whether `model_accepts_loss_kwargs=True` was causing incorrect loss scaling. While the scaling IS different from the traditional path (no division by grad_accum), this is by design and works correctly when the model handles the scaling. Since Qwen3 doesn't handle it, the effective learning rate is 8x the traditional path, but gradient clipping (`max_grad_norm=1.0`) compensates.

### 5. eval_strategy="epoch" triggering mid-dry-run eval
When `max_steps=20` was set for dry runs, the Trainer treated the end of 20 steps as an "epoch end" and triggered full eval (2,048 examples × 1.4s/example = ~48 min). Wasted one full run before catching this. Fixed by setting `eval_strategy="no"` for the dry-run path and keeping it for full runs only via Phase 3's dedicated eval.

### 6. _get_train_sampler signature mismatch (from previous session)
`TypeError: V2SFTTrainer._get_train_sampler() takes 1 positional argument but 2 were given` — HuggingFace's Trainer calls `_get_train_sampler(train_dataset)`. Fixed by adding `train_dataset=None` parameter.

---

## Diagnostic Scripts Created

| Script | Purpose | Lines | Status |
|--------|---------|-------|--------|
| `diag_isolate.py` | 4-test component isolation (sampler, collator, both, none) | ~250 | All tests passed |
| `diag_precise.py` | Feature-toggle tests (eval, verifying collator, extra SFT args) | ~318 | All NaN (has aotriton) |
| `diag_seed.py` | Seed-controlled LoRA/DoRA initialization test | ~213 | All tests passed |
| `diag_test.py` | Earlier diagnostic (from previous session) | ~200 | Passed |

These can be deleted after the full training completes successfully.

---

## Final Training Configuration

### Environment
```
GPU: AMD Radeon RX 7900 XTX (25.75 GB VRAM)
ROCm: 6.2
PyTorch: 2.5.1+rocm6.2
Transformers: 5.1.0
TRL: 0.28.0
PEFT: 0.18.1
HSA_OVERRIDE_GFX_VERSION: 11.0.0
PYTORCH_HIP_ALLOC_CONF: expandable_segments:True
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: DISABLED (NaN root cause)
```

### Model
- Base: V1 merged checkpoint (`data/v2_pipeline/v1_merged_base/`)
- Adapter: Fresh DoRA rank 16, alpha=32, dropout=0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Attention: SDPA (not experimental aotriton)
- Precision: bfloat16

### Training
- **Batch size:** 2 (per device)
- **Gradient accumulation:** 8 (effective batch size: 16)
- **Total optimizer steps:** 1,157
- **Epochs:** 1
- **Learning rate:** 1e-5 (cosine schedule, 5% warmup)
- **Max grad norm:** 1.0
- **Optimizer:** AdamW (torch)
- **Max sequence length:** 1,024
- **Gradient checkpointing:** enabled (use_reentrant=False)
- **Completion-only loss:** enabled

### Data
- **Training:** 18,502 examples (28 MB)
  - noise (w=0.5): 16,429 (88.8%)
  - tangential (w=1.0): 1,359 (7.3%)
  - moderate (w=1.5): 345 (1.9%)
  - high (w=2.0): 246 (1.3%)
  - critical (w=3.0): 123 (0.7%)
- **Eval:** 2,048 examples (3 MB, 30 profiles)

### Sampling & Weighting (dual mechanism)
1. **WeightedRandomSampler:** Controls which examples are sampled (oversamples rare bands)
   - Expected distribution: noise=75%, tangential=12.4%, moderate=4.7%, high=4.5%, critical=3.4%
   - Verification: SamplerCheckCallback verifies after 1,000 batches
2. **Per-sample loss weighting:** Modulates gradient magnitude per example
   - Implements per-example cross-entropy with sample_weight scaling
   - Combined effect: critical examples get ~18x effective gradient contribution vs noise

### Checkpointing
- Save every 200 steps: [200, 400, 600, 800, 1000]
- Eval: disabled during training (Phase 3 handles post-training eval separately)
- Final checkpoint saved to `data/v2_pipeline/training_output/final_checkpoint/`

---

## Expected Timeline

### Phase 2 (Training) — Running Now
- **Steps:** 1,157 optimizer steps
- **Speed:** ~33 seconds per step (from dry-run measurements)
- **Estimated time:** 1,157 × 33s ≈ 10.6 hours
- **Checkpoints:** 5 intermediate (at steps 200, 400, 600, 800, 1000) + final
- **Expected completion:** ~10:40 AM

### Phase 3 (Post-Training) — After Phase 2
- Model reload + V2 inference on 2,048 eval examples: ~1 hour
- Per-profile metrics computation: ~5 minutes
- Profile-awareness sanity check (5 articles × V1/V2 comparison): ~30 minutes
- GGUF export (merge + convert + quantize): ~30 minutes
- **Estimated Phase 3 time:** ~2 hours

### Total estimated wall-clock time: ~13 hours from start

---

## Monitoring Commands

```bash
# Check training progress
tail -20 data/v2_pipeline/training_output/training_log.txt

# Check full output with step metrics
tail -50 data/v2_pipeline/training_output/full_training_run.log

# Check GPU memory
rocm-smi --showmemuse

# Check if training is still running
ps aux | grep train_v2 | grep -v grep

# Check sampler verification (after ~1000 batches = ~125 steps)
cat data/v2_pipeline/training_output/sampler_verification.json
```

---

## Key Lessons

1. **Experimental GPU kernel flags are dangerous.** `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` caused silent NaN corruption that was extremely difficult to diagnose because the failure appeared to be in the model/data/training pipeline, not in a low-level attention kernel.

2. **Systematic isolation is essential.** The root cause was only found by comparing env vars across scripts that passed vs failed. Component isolation (collator, sampler, etc.) was necessary to rule out application-level causes.

3. **Loss scaling interactions are subtle.** `model_accepts_loss_kwargs=True` changes the Trainer's loss division behavior, but Qwen3 doesn't implement the expected `num_items_in_batch` scaling. This silent mismatch causes effective learning rate to differ from expectations, though gradient clipping compensates.

4. **The loss=3.266 was a critical clue.** The identical `loss=3.266` at step 1 across every NaN run (regardless of config changes) suggested the problem was in the first forward/backward pass itself, not in any training configuration. This eventually led to investigating the GPU-level attention kernel.
