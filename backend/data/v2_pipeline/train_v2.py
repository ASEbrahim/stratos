#!/usr/bin/env python3
"""
StratOS V2 Scorer — Phase 2 (Training) + Phase 3 (Eval & Export)

Trains V2 scorer on top of V1 merged base with:
- WeightedRandomSampler for score-band rebalancing (MANDATORY)
- Per-sample loss weighting for gradient emphasis
- DoRA rank 16, 1 epoch, lr=1e-5, cosine schedule

Then evaluates on held-out set, computes per-profile metrics, and exports to GGUF.

Usage:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 data/v2_pipeline/train_v2.py
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 data/v2_pipeline/train_v2.py --phase3-only
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 data/v2_pipeline/train_v2.py --dry-run
"""

import argparse
import gc
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# ROCm setup — MUST be before any torch import
# ═══════════════════════════════════════════════════════════════════
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
# DISABLED: causes NaN gradients during backward on ROCm 6.2 with DoRA + gradient checkpointing
# os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
V1_MERGED_BASE = BASE_DIR / "data" / "v2_pipeline" / "v1_merged_base"
TRAINING_FILE = BASE_DIR / "data" / "v2_pipeline" / "training_v2.jsonl"
EVAL_FILE = BASE_DIR / "data" / "v2_pipeline" / "eval_v2.jsonl"
OUTPUT_DIR = BASE_DIR / "data" / "v2_pipeline" / "training_output"
FINAL_CHECKPOINT = OUTPUT_DIR / "final_checkpoint"

# ═══════════════════════════════════════════════════════════════════
# Hyperparameters
# ═══════════════════════════════════════════════════════════════════
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 1e-5     # Lower than V1's 2e-5 — fine-tuning, not from scratch
GRAD_ACCUM = 8
WARMUP_RATIO = 0.05
EPOCHS = 1
SAVE_STEPS = 200
DEFAULT_QUANT = "q8_0"

# Weight → bucket mapping (from Stage 4 preparation)
WEIGHT_BUCKET = {0.5: "noise", 1.0: "tangential", 1.5: "moderate", 2.0: "high", 3.0: "critical"}

# ═══════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(OUTPUT_DIR / "training_log.txt"), mode='w'),
    ]
)
logger = logging.getLogger("V2_TRAIN")


# ═══════════════════════════════════════════════════════════════════
# Per-Sample Loss Weighting (proven from V1/v19)
# ═══════════════════════════════════════════════════════════════════

class WeightedCompletionDataCollator:
    """Wraps the base data collator to preserve sample_weight through tokenization."""

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        import torch as _torch
        weights = []
        for f in features:
            w = f.pop("sample_weight", None)
            if w is None:
                weights.append(1.0)
            elif isinstance(w, (list, tuple)):
                weights.append(float(w[0]) if w else 1.0)
            else:
                weights.append(float(w))
        batch = self.base_collator(features)
        batch["sample_weight"] = _torch.tensor(weights, dtype=_torch.float32)
        return batch


class VerifyingCollatorWrapper:
    """Wraps WeightedCompletionDataCollator to track sample distributions
    for WeightedRandomSampler verification."""

    def __init__(self, inner_collator, target_batches=1000):
        self.inner = inner_collator
        self.seen_weights = []
        self.batch_count = 0
        self.target_batches = target_batches
        self.verified = False
        self.verification_passed = None

    def __call__(self, features):
        # Record weights BEFORE inner collator pops them
        if not self.verified:
            for f in features:
                w = f.get("sample_weight", 1.0)
                if isinstance(w, (list, tuple)):
                    w = float(w[0]) if w else 1.0
                self.seen_weights.append(float(w))
            self.batch_count += 1
        return self.inner(features)


def create_v2_trainer_class():
    """Create V2SFTTrainer: per-sample loss weighting + WeightedRandomSampler."""
    from trl import SFTTrainer as _BaseSFTTrainer
    import torch
    from torch.nn import CrossEntropyLoss
    from torch.utils.data import WeightedRandomSampler

    class V2SFTTrainer(_BaseSFTTrainer):
        """SFTTrainer with:
        1. WeightedRandomSampler (controls which examples are seen — oversamples rare bands)
        2. Per-sample loss weighting via compute_loss override
        Uses standard Trainer training_step for backward/logging (no custom training_step).
        """

        def __init__(self, *args, sampler_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._sampler_weights = sampler_weights
            # Keep model_accepts_loss_kwargs as auto-detected (True for Qwen3)
            # Setting to False was causing NaN gradients in combination with the pipeline
            self._cl_count = 0

        def _get_train_sampler(self, train_dataset=None):
            """Override to inject WeightedRandomSampler."""
            if self._sampler_weights is not None:
                return WeightedRandomSampler(
                    weights=self._sampler_weights,
                    num_samples=len(self._sampler_weights),
                    replacement=True,
                )
            return super()._get_train_sampler(train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """Per-sample weighted loss: emphasize gradient from high-value score bands.

            Computes per-example cross-entropy, weights by sample_weight, and scales
            for gradient accumulation compatibility (num_items_in_batch).
            """
            self._cl_count += 1
            sample_weights = inputs.pop("sample_weight", None)

            if sample_weights is None:
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch
                )

            # Pop labels — we compute loss manually for per-example weighting
            labels = inputs.pop("labels")
            inputs.pop("num_items_in_batch", None)

            # Forward pass (logits only, no internal loss)
            outputs = model(**inputs)
            logits = outputs.logits

            # Causal LM shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            bs, seq_len, vocab_size = shift_logits.shape

            # Per-token cross-entropy (prompt tokens masked via ignore_index=-100)
            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
            per_token_loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            ).view(bs, seq_len)

            # Per-example mean over completion tokens only
            mask = (shift_labels != -100).float()
            per_example_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # Weighted average across batch
            weights = sample_weights.to(per_example_loss.device)
            weighted_loss = (per_example_loss * weights).sum() / weights.sum()

            # NOTE: No num_items_in_batch scaling needed — Qwen3 doesn't implement it,
            # and the Trainer skips grad_accum division when model_accepts_loss_kwargs=True.
            # This matches passthrough behavior (CE_mean returned directly).

            if self._cl_count <= 2:
                unweighted = per_example_loss.mean().item()
                logger.info("[V2] compute_loss — per-sample weighting ACTIVE")
                logger.info("[V2]   weights: %s, per-example loss: %s",
                            [round(w.item(), 2) for w in weights],
                            [round(l.item(), 4) for l in per_example_loss])
                logger.info("[V2]   weighted=%.4f unweighted=%.4f",
                            weighted_loss.item(), unweighted)

            return (weighted_loss, outputs) if return_outputs else weighted_loss

    return V2SFTTrainer


# ═══════════════════════════════════════════════════════════════════
# Sampler Verification
# ═══════════════════════════════════════════════════════════════════

def verify_sampler(collator_wrapper, raw_weight_counts, save_path):
    """Compare actual vs expected distribution after 1000+ batches.
    Returns True if sampler is working, False if silently failed."""
    seen = collator_wrapper.seen_weights

    # Actual distribution
    actual_counts = Counter()
    for w in seen:
        bucket = WEIGHT_BUCKET.get(round(w, 1), f"unknown_{w}")
        actual_counts[bucket] += 1
    total_seen = len(seen)

    # Expected distribution (weighted sampling probabilities)
    total_weighted = sum(count * weight for weight, count in raw_weight_counts.items())
    expected_fracs = {}
    for weight, count in raw_weight_counts.items():
        bucket = WEIGHT_BUCKET.get(weight, f"unknown_{weight}")
        expected_fracs[bucket] = (count * weight) / total_weighted

    # Raw (unweighted) distribution — what we'd see if sampler failed
    total_raw = sum(raw_weight_counts.values())
    raw_fracs = {}
    for weight, count in raw_weight_counts.items():
        bucket = WEIGHT_BUCKET.get(weight, f"unknown_{weight}")
        raw_fracs[bucket] = count / total_raw

    # Actual fractions
    actual_fracs = {bucket: count / total_seen for bucket, count in actual_counts.items()}

    # Log comparison
    all_buckets = ["noise", "tangential", "moderate", "high", "critical"]
    logger.info("=" * 70)
    logger.info("SAMPLER VERIFICATION (%d batches, %d examples)", collator_wrapper.batch_count, total_seen)
    logger.info("%-15s %10s %10s %10s", "Bucket", "Expected", "Actual", "Raw(bad)")
    logger.info("-" * 50)
    for bucket in all_buckets:
        exp = expected_fracs.get(bucket, 0)
        act = actual_fracs.get(bucket, 0)
        raw = raw_fracs.get(bucket, 0)
        logger.info("%-15s %9.1f%% %9.1f%% %9.1f%%", bucket, exp * 100, act * 100, raw * 100)

    # Is actual closer to expected or to raw?
    exp_distance = sum(abs(actual_fracs.get(b, 0) - expected_fracs.get(b, 0)) for b in all_buckets)
    raw_distance = sum(abs(actual_fracs.get(b, 0) - raw_fracs.get(b, 0)) for b in all_buckets)

    passed = exp_distance < raw_distance

    if passed:
        logger.info("SAMPLER WORKING — actual matches weighted expectation")
        logger.info("  Distance to expected: %.4f, distance to raw: %.4f", exp_distance, raw_distance)
    else:
        logger.error("SAMPLER FAILED — actual matches raw (unweighted) distribution")
        logger.error("  Distance to expected: %.4f, distance to raw: %.4f", exp_distance, raw_distance)
        logger.error("  STOP TRAINING — WeightedRandomSampler is not working")

    # Save verification data
    result = {
        "batches_seen": collator_wrapper.batch_count,
        "examples_seen": total_seen,
        "expected_distribution": {b: round(expected_fracs.get(b, 0), 4) for b in all_buckets},
        "actual_distribution": {b: round(actual_fracs.get(b, 0), 4) for b in all_buckets},
        "raw_distribution": {b: round(raw_fracs.get(b, 0), 4) for b in all_buckets},
        "distance_to_expected": round(exp_distance, 4),
        "distance_to_raw": round(raw_distance, 4),
        "passed": passed,
    }
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info("Verification saved to %s", save_path)

    collator_wrapper.verified = True
    collator_wrapper.verification_passed = passed
    return passed


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Training
# ═══════════════════════════════════════════════════════════════════

def run_phase2(batch_size=4, dry_run=False):
    """Train V2 scorer with WeightedRandomSampler + loss weighting."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset
    from trl import SFTConfig

    max_steps = 20 if dry_run else -1

    logger.info("=" * 70)
    logger.info("PHASE 2 — V2 SCORER TRAINING")
    logger.info("=" * 70)
    logger.info("Base model: %s", V1_MERGED_BASE)
    logger.info("Training data: %s (%d MB)", TRAINING_FILE, TRAINING_FILE.stat().st_size // (1024*1024))
    logger.info("Eval data: %s (%d MB)", EVAL_FILE, EVAL_FILE.stat().st_size // (1024*1024))
    logger.info("Batch size: %d, Grad accum: %d, LR: %s", batch_size, GRAD_ACCUM, LEARNING_RATE)
    logger.info("Epochs: %d, Dry run: %s", EPOCHS, dry_run)

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(str(V1_MERGED_BASE), trust_remote_code=True)

    # v19 CRITICAL: Qwen3 eos_token fix
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    logger.info("Set eos_token = <|im_end|> (id=%d)", tokenizer.eos_token_id)

    if tokenizer.pad_token_id != 151643:
        tokenizer.pad_token_id = 151643
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        logger.info("Set pad_token_id = 151643 (endoftext)")

    # ── Load model in bf16 ──
    logger.info("Loading V1 merged base in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        str(V1_MERGED_BASE),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    # ── Fresh DoRA adapters ──
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=True,
    )
    model = get_peft_model(model, lora_config)

    # PEFT meta device fix (ROCm)
    if hasattr(model, 'hf_device_map'):
        del model.hf_device_map
    model = model.to("cuda:0")
    logger.info("Forced all parameters to cuda:0 (meta device cleanup)")
    model.print_trainable_parameters()

    # ── Load training data ──
    dataset = load_dataset("json", data_files=str(TRAINING_FILE), split="train")
    logger.info("Training examples: %d", len(dataset))

    raw_weight_counts = Counter()
    for w in dataset["sample_weight"]:
        w = w if w is not None else 1.0
        raw_weight_counts[round(w, 1)] += 1
    for w, c in sorted(raw_weight_counts.items()):
        logger.info("  weight=%.1f (%s): %d examples (%.1f%%)",
                     w, WEIGHT_BUCKET.get(w, "?"), c, c / len(dataset) * 100)

    # ── Convert messages → prompt/completion ──
    def convert_to_prompt_completion(example):
        msgs = example["messages"]
        prompt = tokenizer.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
        completion = msgs[2]["content"] + "<|im_end|>"
        result = {"prompt": prompt, "completion": completion}
        w = example.get("sample_weight")
        result["sample_weight"] = w if w is not None else 1.0
        return result

    dataset = dataset.map(convert_to_prompt_completion, remove_columns=["messages"])
    logger.info("Converted messages -> prompt/completion")

    # ── Build WeightedRandomSampler weights ──
    sampler_weights = [float(w) if w is not None else 1.0 for w in dataset["sample_weight"]]
    total_w = sum(sampler_weights)
    logger.info("WeightedRandomSampler: %d weights, sum=%.1f", len(sampler_weights), total_w)

    # Log expected sampling distribution
    expected_by_bucket = defaultdict(float)
    for w in sampler_weights:
        bucket = WEIGHT_BUCKET.get(round(w, 1), "?")
        expected_by_bucket[bucket] += w
    logger.info("Expected sampling distribution:")
    for bucket in ["noise", "tangential", "moderate", "high", "critical"]:
        frac = expected_by_bucket[bucket] / total_w
        logger.info("  %-15s %.1f%%", bucket, frac * 100)

    # ── Load eval data ──
    eval_dataset = load_dataset("json", data_files=str(EVAL_FILE), split="train")
    eval_remove = [c for c in eval_dataset.column_names if c not in ["prompt", "completion"]]
    eval_dataset = eval_dataset.map(convert_to_prompt_completion, remove_columns=eval_remove)
    logger.info("Eval dataset: %d examples", len(eval_dataset))

    # ── Adjust grad accum for batch_size=1 last resort ──
    grad_accum = GRAD_ACCUM
    if batch_size == 1:
        grad_accum = 16
        logger.warning(">>> BATCH_SIZE=1 LAST RESORT — grad_accum=16, loss weighting effectiveness degraded <<<")

    total_steps = math.ceil(len(dataset) / (batch_size * grad_accum))
    ckpt_steps = list(range(SAVE_STEPS, total_steps + 1, SAVE_STEPS))
    logger.info("Total optimizer steps: %d (batch=%d, accum=%d, effective=%d)",
                total_steps, batch_size, grad_accum, batch_size * grad_accum)
    logger.info("Checkpoints at steps: %s", ckpt_steps)

    # ── Create trainer ──
    V2SFTTrainer = create_v2_trainer_class()

    # Sampler verification callback
    class SamplerCheckCallback(TrainerCallback):
        def __init__(self, cw, rw, sp):
            self.collator_wrapper = cw
            self.raw_weights = rw
            self.save_path = sp
            self.checked = False

        def on_step_begin(self, args, state, control, **kwargs):
            if not self.checked and self.collator_wrapper.batch_count >= 1000:
                self.checked = True
                passed = verify_sampler(self.collator_wrapper, self.raw_weights, self.save_path)
                if not passed:
                    logger.error(">>> STOPPING TRAINING — sampler verification failed <<<")
                    control.should_training_stop = True

    trainer = V2SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        sampler_weights=sampler_weights,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            max_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,           # v19 OOM fix
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=WARMUP_RATIO,
            weight_decay=0.01,
            max_grad_norm=1.0,
            logging_steps=1 if dry_run else 5,
            save_steps=SAVE_STEPS,
            save_strategy="steps",
            eval_strategy="no",                  # Phase 3 handles eval with full generation
            eval_accumulation_steps=4,           # v19 OOM fix
            load_best_model_at_end=False,        # Single epoch, no early stopping
            bf16=True,
            fp16=False,
            optim="adamw_torch",
            seed=42,
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            completion_only_loss=True,           # Only train on assistant response tokens
            remove_unused_columns=False,         # Preserve sample_weight column
            average_tokens_across_devices=False,
            max_steps=max_steps,
        ),
    )

    # Wrap collator: base → weighted → verifying
    weighted_collator = WeightedCompletionDataCollator(trainer.data_collator)
    verifying_collator = VerifyingCollatorWrapper(weighted_collator, target_batches=1000)
    trainer.data_collator = verifying_collator

    # Add verification callback
    verification_path = OUTPUT_DIR / "sampler_verification.json"
    trainer.add_callback(SamplerCheckCallback(verifying_collator, raw_weight_counts, str(verification_path)))

    # ── Train ──
    logger.info("Starting V2 training...")
    start_time = time.time()
    stats = trainer.train()
    elapsed = time.time() - start_time
    logger.info("Training complete! Loss: %.4f, Time: %.1fh", stats.training_loss, elapsed / 3600)

    # Verify sampler if not yet checked (dry run with <1000 batches)
    if not verifying_collator.verified and verifying_collator.batch_count > 0:
        verify_sampler(verifying_collator, raw_weight_counts, str(verification_path))

    # ── Phase 3 Step 1: Save final checkpoint ──
    FINAL_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(FINAL_CHECKPOINT))
    tokenizer.save_pretrained(str(FINAL_CHECKPOINT))
    logger.info("Final V2 adapter saved to: %s", FINAL_CHECKPOINT)

    # ── Phase 3 Step 2: Clear GPU memory ──
    del model, trainer
    gc.collect()
    import torch as _t
    _t.cuda.empty_cache()
    logger.info("GPU memory cleared after Phase 2")

    return stats.training_loss, elapsed


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Post-Training Evaluation
# ═══════════════════════════════════════════════════════════════════

def extract_profile(system_prompt):
    """Extract profile description from system prompt."""
    m = re.search(r'You are a relevance scorer for (?:a |an )?(.*?)(?:\.\n|\n)', system_prompt)
    return m.group(1).strip() if m else "UNKNOWN"


def extract_score(text):
    """Extract numeric score from model output."""
    clean = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    m = re.search(r'SCORE:\s*(\d+\.?\d*)', clean, re.IGNORECASE)
    return float(m.group(1)) if m else None


def extract_title(user_msg):
    """Extract article title from user message."""
    m = re.search(r'Title:\s*(.*?)(?:\n|$)', user_msg)
    return m.group(1).strip() if m else "UNKNOWN"


def run_inference(model, tokenizer, eval_examples, batch_size=1, desc="Inference"):
    """Generate predictions for eval examples. Returns list of result dicts."""
    import torch

    model.eval()
    tokenizer.padding_side = "left"
    results = []
    total = len(eval_examples)

    for i in range(0, total, batch_size):
        batch_exs = eval_examples[i:i + batch_size]

        # Build prompts (system + user only, no assistant)
        prompts = []
        for ex in batch_exs:
            msgs = ex["messages"][:2]
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        # Tokenize with left padding for generation
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_SEQ_LENGTH
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, ex in enumerate(batch_exs):
            gen_tokens = outputs[j][input_len:]
            generated = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            results.append({
                "generated": generated.strip(),
                "pred_score": extract_score(generated),
                "gt_score": extract_score(ex["messages"][2]["content"]),
                "profile": extract_profile(ex["messages"][0]["content"]),
                "title": extract_title(ex["messages"][1]["content"]),
            })

        done = min(i + batch_size, total)
        if (i // max(batch_size, 1)) % 100 == 0:
            logger.info("  %s: %d/%d (%.0f%%)", desc, done, total, done / total * 100)

    logger.info("  %s: %d/%d complete", desc, total, total)
    return results


def compute_metrics(results):
    """Compute direction accuracy, MAE, PSR, Spearman rho."""
    from scipy.stats import spearmanr

    valid = [r for r in results if r["pred_score"] is not None and r["gt_score"] is not None]
    total = len(results)
    parse_failures = total - len(valid)

    if not valid:
        return {"error": "No valid predictions", "parse_failures": parse_failures}

    preds = [r["pred_score"] for r in valid]
    gts = [r["gt_score"] for r in valid]

    # Direction accuracy: both above or both below 5.0
    correct_dir = sum(1 for p, g in zip(preds, gts) if (p >= 5.0) == (g >= 5.0))
    direction_acc = correct_dir / len(valid)

    # MAE
    mae = sum(abs(p - g) for p, g in zip(preds, gts)) / len(valid)

    # Spearman rho (aggregate)
    rho, p_value = spearmanr(preds, gts)

    # Think block analysis
    think_empty = sum(1 for r in results if "<think></think>" in r.get("generated", ""))
    think_present = sum(1 for r in results if "<think>" in r.get("generated", ""))

    # PSR: for articles in 2+ profiles, % with score spread >= 2.0
    title_scores = defaultdict(list)
    for r in valid:
        title_scores[r["title"]].append((r["profile"], r["pred_score"]))

    multi_profile = {t: ps for t, ps in title_scores.items() if len(ps) >= 2}
    if multi_profile:
        sensitive = sum(1 for ps in multi_profile.values()
                        if max(s for _, s in ps) - min(s for _, s in ps) >= 2.0)
        psr = sensitive / len(multi_profile)
    else:
        psr = 0.0

    return {
        "total_examples": total,
        "valid_predictions": len(valid),
        "parse_failures": parse_failures,
        "direction_accuracy": direction_acc,
        "mae": mae,
        "spearman_rho": rho,
        "spearman_p": p_value,
        "psr": psr,
        "psr_articles": len(multi_profile),
        "think_empty": think_empty,
        "think_present": think_present,
        "think_empty_rate": think_empty / total if total > 0 else 0,
    }


def per_profile_metrics(results):
    """Compute Spearman rho for each profile individually."""
    from scipy.stats import spearmanr

    by_profile = defaultdict(lambda: {"pred": [], "gt": []})
    for r in results:
        if r["pred_score"] is not None and r["gt_score"] is not None:
            by_profile[r["profile"]]["pred"].append(r["pred_score"])
            by_profile[r["profile"]]["gt"].append(r["gt_score"])

    profile_results = []
    for profile, data in sorted(by_profile.items()):
        n = len(data["pred"])
        if n < 3:
            rho, flag = None, "insufficient_data"
        else:
            try:
                rho, _ = spearmanr(data["pred"], data["gt"])
                if rho is None or (isinstance(rho, float) and rho != rho):
                    flag = "undefined"
                    rho = None
                elif n < 20:
                    flag = "low_n"
                elif rho < 0:
                    flag = "NEGATIVE"
                elif rho < 0.3:
                    flag = "WEAK"
                else:
                    flag = ""
            except Exception:
                rho, flag = None, "error"

        profile_results.append({
            "profile": profile,
            "n": n,
            "spearman_rho": rho,
            "flag": flag,
        })

    return profile_results


def profile_awareness_check(model, tokenizer, eval_examples):
    """Phase 3 Step 6: Score 5 articles with V1 (adapters off) and V2 (adapters on).
    Tests whether V2 produces larger score spreads across different profiles."""
    import torch

    # Group eval examples by title
    by_title = defaultdict(list)
    for ex in eval_examples:
        title = extract_title(ex["messages"][1]["content"])
        profile = extract_profile(ex["messages"][0]["content"])
        gt_score = extract_score(ex["messages"][2]["content"])
        by_title[title].append({"profile": profile, "gt_score": gt_score, "example": ex})

    # Find articles with 3+ profiles and ground truth spread >= 3.0
    candidates = []
    for title, entries in by_title.items():
        if len(entries) >= 3:
            gt_scores = [e["gt_score"] for e in entries if e["gt_score"] is not None]
            if gt_scores:
                spread = max(gt_scores) - min(gt_scores)
                if spread >= 3.0:
                    candidates.append((title, entries, spread))

    candidates.sort(key=lambda x: (-x[2], -len(x[1])))
    selected = candidates[:5]

    if not selected:
        logger.warning("No suitable articles for profile-awareness check (need spread >= 3.0)")
        return []

    tokenizer.padding_side = "left"
    sanity_results = []

    for title, entries, gt_spread in selected:
        article_result = {"title": title, "gt_spread": gt_spread, "profiles": []}

        for entry in sorted(entries, key=lambda e: e["gt_score"] or 0, reverse=True):
            msgs = entry["example"]["messages"][:2]
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=MAX_SEQ_LENGTH
            ).to(model.device)

            gen_kwargs = dict(max_new_tokens=200, do_sample=False,
                              eos_token_id=tokenizer.eos_token_id,
                              pad_token_id=tokenizer.pad_token_id)
            input_len = inputs["input_ids"].shape[1]

            # V2 score (adapters on)
            model.enable_adapter_layers()
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            v2_text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
            v2_score = extract_score(v2_text)

            # V1 score (adapters off = raw V1 merged base)
            model.disable_adapter_layers()
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            v1_text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
            v1_score = extract_score(v1_text)

            # Re-enable adapters for next iteration
            model.enable_adapter_layers()

            article_result["profiles"].append({
                "profile": entry["profile"],
                "gt_score": entry["gt_score"],
                "v1_score": v1_score,
                "v2_score": v2_score,
            })

        v2_scores = [p["v2_score"] for p in article_result["profiles"] if p["v2_score"] is not None]
        v1_scores = [p["v1_score"] for p in article_result["profiles"] if p["v1_score"] is not None]
        article_result["v2_spread"] = (max(v2_scores) - min(v2_scores)) if len(v2_scores) >= 2 else 0
        article_result["v1_spread"] = (max(v1_scores) - min(v1_scores)) if len(v1_scores) >= 2 else 0

        sanity_results.append(article_result)

    return sanity_results


def export_to_gguf(merged_path, output_dir):
    """Convert merged HF model to GGUF via llama.cpp, quantize to q8_0."""
    llama_convert = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"

    if not llama_convert.exists():
        logger.warning("llama.cpp not found at ~/llama.cpp — GGUF export skipped")
        logger.info("Manual conversion:")
        logger.info("  python ~/llama.cpp/convert_hf_to_gguf.py %s --outfile v2_scorer_bf16.gguf --outtype bf16", merged_path)
        return None

    bf16_path = Path(output_dir) / "v2_scorer_bf16.gguf"

    logger.info("Converting to GGUF (bf16 intermediate)...")
    result = subprocess.run(
        [sys.executable, str(llama_convert), merged_path,
         "--outfile", str(bf16_path), "--outtype", "bf16"],
        capture_output=True, text=True, timeout=1800
    )

    if result.returncode != 0 or not bf16_path.exists():
        logger.error("GGUF conversion failed: %s", result.stderr[:500])
        return None

    logger.info("BF16 GGUF: %s (%d MB)", bf16_path, bf16_path.stat().st_size // (1024 * 1024))

    # Quantize to q8_0
    llama_quantize = Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize"
    if not llama_quantize.exists():
        llama_quantize = Path.home() / "llama.cpp" / "llama-quantize"

    if llama_quantize.exists():
        q8_path = Path(output_dir) / "v2_scorer.gguf"
        logger.info("Quantizing to %s...", DEFAULT_QUANT)
        qresult = subprocess.run(
            [str(llama_quantize), str(bf16_path), str(q8_path), DEFAULT_QUANT.upper()],
            capture_output=True, text=True, timeout=1800
        )
        if qresult.returncode == 0 and q8_path.exists():
            logger.info("q8_0 GGUF: %s (%d MB)", q8_path, q8_path.stat().st_size // (1024 * 1024))
            bf16_path.unlink(missing_ok=True)
            logger.info("Cleaned bf16 intermediate")
            return str(q8_path)
        else:
            logger.warning("Quantization failed, keeping bf16: %s", qresult.stderr[:200])
            return str(bf16_path)
    else:
        logger.info("llama-quantize not found — keeping bf16 GGUF")
        return str(bf16_path)


def generate_report(metrics, profile_metrics, sanity_results, gguf_path, training_loss, training_time):
    """Generate eval_report.md with all Phase 3 results."""
    lines = ["# StratOS V2 Scorer — Evaluation Report\n"]
    lines.append(f"Training loss: {training_loss:.4f}, Training time: {training_time / 3600:.1f}h\n")

    # Metrics summary
    lines.append("## Metrics Summary\n")
    lines.append("| Metric | V1 Baseline | V2 Result |")
    lines.append("|--------|-------------|-----------|")
    lines.append(f"| Direction accuracy | 90.7% | {metrics['direction_accuracy']*100:.1f}% |")
    lines.append(f"| PSR (Profile Sensitivity Ratio) | 39.7% | {metrics['psr']*100:.1f}% |")
    lines.append(f"| MAE | 1.553 | {metrics['mae']:.3f} |")
    lines.append(f"| Spearman rho (aggregate) | -- | {metrics['spearman_rho']:.3f} |")
    lines.append(f"| Think block emptiness rate | ~85% | {metrics['think_empty_rate']*100:.1f}% |")
    lines.append(f"| Parse failures | -- | {metrics['parse_failures']}/{metrics['total_examples']} |")
    lines.append(f"| PSR articles evaluated | -- | {metrics['psr_articles']} |")
    lines.append("")

    # Per-profile
    lines.append("## Per-Profile Spearman rho\n")
    lines.append("| Profile | N | rho | Flag |")
    lines.append("|---------|---|-----|------|")
    for pm in profile_metrics:
        rho = f"{pm['spearman_rho']:.3f}" if pm['spearman_rho'] is not None else "N/A"
        flag = pm['flag'] if pm['flag'] else ""
        lines.append(f"| {pm['profile'][:60]} | {pm['n']} | {rho} | {flag} |")
    lines.append("")

    flagged = [pm for pm in profile_metrics if pm['flag'] in ('NEGATIVE', 'WEAK', 'insufficient_data')]
    if flagged:
        lines.append(f"**{len(flagged)} profiles flagged:**\n")
        for pm in flagged:
            rho_s = f"{pm['spearman_rho']:.3f}" if pm['spearman_rho'] is not None else "N/A"
            lines.append(f"- **{pm['profile'][:60]}**: rho={rho_s}, flag={pm['flag']}")
        lines.append("")

    # Profile-awareness sanity check
    lines.append("## Profile-Awareness Sanity Check\n")
    for art in sanity_results:
        lines.append(f"### \"{art['title'][:80]}\"\n")
        lines.append(f"GT spread: {art['gt_spread']:.1f}, V1 spread: {art['v1_spread']:.1f}, V2 spread: {art['v2_spread']:.1f}\n")
        lines.append("| Profile | GT | V1 | V2 |")
        lines.append("|---------|----|----|-----|")
        for p in art["profiles"]:
            v1 = f"{p['v1_score']:.1f}" if p['v1_score'] is not None else "N/A"
            v2 = f"{p['v2_score']:.1f}" if p['v2_score'] is not None else "N/A"
            gt = f"{p['gt_score']:.1f}" if p['gt_score'] is not None else "N/A"
            lines.append(f"| {p['profile'][:50]} | {gt} | {v1} | {v2} |")
        lines.append("")

    if sanity_results:
        v2_avg = sum(a["v2_spread"] for a in sanity_results) / len(sanity_results)
        v1_avg = sum(a["v1_spread"] for a in sanity_results) / len(sanity_results)
        verdict = "PASS" if v2_avg >= 2.0 else "NEEDS REVIEW"
        lines.append(f"**Verdict:** Avg V1 spread: {v1_avg:.2f}, Avg V2 spread: {v2_avg:.2f} — {verdict}\n")

    # Artifacts
    lines.append("## Artifacts\n")
    lines.append(f"- GGUF: `{gguf_path or 'NOT EXPORTED'}`")
    lines.append(f"- Final checkpoint: `{FINAL_CHECKPOINT}`")
    lines.append(f"- Sampler verification: `{OUTPUT_DIR / 'sampler_verification.json'}`")
    lines.append(f"- Training log: `{OUTPUT_DIR / 'training_log.txt'}`")
    lines.append("")
    lines.append("**NOT deployed. NOT registered in Ollama. Review this report before deploying.**")

    return "\n".join(lines)


def run_phase3(training_loss=None, training_time=None):
    """Phase 3: eval, metrics, profile check, GGUF export."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info("=" * 70)
    logger.info("PHASE 3 — POST-TRAINING EVALUATION")
    logger.info("=" * 70)

    # Step 2: Clear GPU
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Load trained model fresh
    logger.info("Loading V1 merged base from %s...", V1_MERGED_BASE)
    tokenizer = AutoTokenizer.from_pretrained(str(V1_MERGED_BASE), trust_remote_code=True)
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tokenizer.pad_token_id != 151643:
        tokenizer.pad_token_id = 151643
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)

    model = AutoModelForCausalLM.from_pretrained(
        str(V1_MERGED_BASE),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    logger.info("Loading V2 adapters from %s...", FINAL_CHECKPOINT)
    model = PeftModel.from_pretrained(model, str(FINAL_CHECKPOINT))
    if hasattr(model, 'hf_device_map'):
        del model.hf_device_map
    model = model.to("cuda:0")
    model.eval()
    logger.info("V2 model loaded for evaluation")

    # Load eval data (raw JSON for generation)
    eval_examples = []
    with open(EVAL_FILE) as f:
        for line in f:
            eval_examples.append(json.loads(line))
    logger.info("Loaded %d eval examples", len(eval_examples))

    # ── Step 6 (FIRST): Profile-awareness sanity check ──
    # Must run BEFORE merge_and_unload() — needs adapter on/off for V1/V2 comparison
    logger.info("=" * 70)
    logger.info("PROFILE-AWARENESS SANITY CHECK (before merge)")
    logger.info("=" * 70)

    sanity = profile_awareness_check(model, tokenizer, eval_examples)

    for art in sanity:
        logger.info("")
        logger.info("Article: \"%s\"", art['title'][:80])
        logger.info("  GT spread: %.1f, V1 spread: %.1f, V2 spread: %.1f",
                     art['gt_spread'], art['v1_spread'], art['v2_spread'])
        for p in art["profiles"]:
            v1 = f"{p['v1_score']:.1f}" if p['v1_score'] is not None else "N/A"
            v2 = f"{p['v2_score']:.1f}" if p['v2_score'] is not None else "N/A"
            logger.info("    %-50s GT=%.1f V1=%s V2=%s", p['profile'][:50], p['gt_score'] or 0, v1, v2)

    if sanity:
        v2_spreads = [a["v2_spread"] for a in sanity]
        v1_spreads = [a["v1_spread"] for a in sanity]
        avg_v2 = sum(v2_spreads) / len(v2_spreads)
        avg_v1 = sum(v1_spreads) / len(v1_spreads)
        logger.info("")
        logger.info("Avg V1 spread: %.2f, Avg V2 spread: %.2f", avg_v1, avg_v2)
        if avg_v2 >= 2.0:
            logger.info("V2 shows profile-awareness (avg spread >= 2.0)")
        else:
            logger.warning("V2 spread < 2.0 — profile-awareness may be insufficient")

    # ── Merge adapters for fast inference ──
    # DoRA adapter overhead causes ~55s/example during generation.
    # After merge: adapters baked into weights, generation is ~5-10x faster.
    logger.info("Merging V2 adapters into base for fast inference...")
    model = model.merge_and_unload()
    model.eval()
    logger.info("Adapters merged — running bulk eval on merged model")

    # ── Run V2 inference on full eval set (merged model, much faster) ──
    logger.info("Running V2 inference on eval set...")
    start = time.time()
    results = run_inference(model, tokenizer, eval_examples, batch_size=1, desc="V2 eval")
    eval_time = time.time() - start
    logger.info("Inference complete in %.0fs (%.1fs/example)", eval_time, eval_time / len(eval_examples))

    # ── Step 4: Compute metrics ──
    metrics = compute_metrics(results)

    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info("%-30s %15s %15s", "Metric", "V1 Baseline", "V2 Result")
    logger.info("-" * 62)
    logger.info("%-30s %15s %14.1f%%", "Direction accuracy", "90.7%", metrics['direction_accuracy'] * 100)
    logger.info("%-30s %15s %14.1f%%", "PSR", "39.7%", metrics['psr'] * 100)
    logger.info("%-30s %15s %15.3f", "MAE", "1.553", metrics['mae'])
    logger.info("%-30s %15s %15.3f", "Spearman rho", "--", metrics['spearman_rho'])
    logger.info("%-30s %15s %14.1f%%", "Think block empty rate", "~85%", metrics['think_empty_rate'] * 100)
    logger.info("%-30s %15s %15d", "Parse failures", "--", metrics['parse_failures'])

    # ── Step 5: Per-profile breakdown ──
    logger.info("=" * 70)
    logger.info("PER-PROFILE SPEARMAN rho")
    logger.info("=" * 70)

    profile_mets = per_profile_metrics(results)
    logger.info("%-65s %5s %8s %10s", "Profile", "N", "rho", "Flag")
    logger.info("-" * 90)
    for pm in profile_mets:
        rho_str = f"{pm['spearman_rho']:.3f}" if pm['spearman_rho'] is not None else "N/A"
        logger.info("%-65s %5d %8s %10s", pm['profile'][:64], pm['n'], rho_str, pm['flag'])

    flagged = [pm for pm in profile_mets if pm['flag'] in ('NEGATIVE', 'WEAK', 'insufficient_data')]
    if flagged:
        logger.warning("%d profiles flagged:", len(flagged))
        for pm in flagged:
            rho_s = f"{pm['spearman_rho']:.3f}" if pm['spearman_rho'] is not None else "N/A"
            logger.warning("  %s: rho=%s, flag=%s", pm['profile'][:60], rho_s, pm['flag'])

    # ── Step 7: GGUF Export ──
    logger.info("=" * 70)
    logger.info("GGUF EXPORT")
    logger.info("=" * 70)

    # Model already merged earlier (before bulk eval)
    merged_path = OUTPUT_DIR / "merged_v2"
    merged_path.mkdir(parents=True, exist_ok=True)
    model = model.cpu()  # Free GPU before saving
    model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    logger.info("Merged model saved to %s", merged_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    gguf_path = export_to_gguf(str(merged_path), str(OUTPUT_DIR))

    # Cleanup merged model dir (~16GB)
    if gguf_path and merged_path.exists():
        shutil.rmtree(str(merged_path), ignore_errors=True)
        logger.info("Cleaned merged_v2/ to free disk")

    # ── Generate report ──
    tl = training_loss if training_loss is not None else 0
    tt = training_time if training_time is not None else 0
    report = generate_report(metrics, profile_mets, sanity, gguf_path, tl, tt)
    report_path = OUTPUT_DIR / "eval_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info("Eval report saved to %s", report_path)

    # ── Step 8: Do NOT deploy ──
    logger.info("=" * 70)
    logger.info("PHASE 3 COMPLETE")
    logger.info("GGUF: %s", gguf_path or "NOT EXPORTED")
    logger.info("Report: %s", report_path)
    logger.info("NOT deploying. NOT registering with Ollama. Review report first.")
    logger.info("=" * 70)

    return metrics


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="StratOS V2/V3 Scorer Training + Evaluation")
    parser.add_argument("--phase3-only", action="store_true", help="Skip training, run Phase 3 only")
    parser.add_argument("--dry-run", action="store_true", help="Train for 20 steps only")
    parser.add_argument("--batch-size", type=int, default=4, help="Starting batch size (fallback: 4->2->1)")
    parser.add_argument("--training-file", type=str, default=None, help="Custom training JSONL path (default: V2 data)")
    parser.add_argument("--eval-file", type=str, default=None, help="Custom eval JSONL path (default: V2 eval)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    # Override paths if custom data provided (for V3 training)
    global TRAINING_FILE, EVAL_FILE, OUTPUT_DIR, FINAL_CHECKPOINT
    if args.training_file:
        TRAINING_FILE = Path(args.training_file)
    if args.eval_file:
        EVAL_FILE = Path(args.eval_file)
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FINAL_CHECKPOINT = OUTPUT_DIR / "final_checkpoint"

    training_loss = None
    training_time = None

    if not args.phase3_only:
        # Phase 2: Training with batch size fallback chain
        batch_chain = [bs for bs in [4, 2, 1] if bs <= args.batch_size]

        for bs in batch_chain:
            try:
                logger.info("Attempting training with batch_size=%d...", bs)
                training_loss, training_time = run_phase2(batch_size=bs, dry_run=args.dry_run)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM with batch_size=%d — falling back to smaller batch size", bs)
                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()
                    time.sleep(5)  # Let GPU memory settle
                    if bs == 1:
                        logger.error("OOM even with batch_size=1 — cannot train on this GPU")
                        raise
                else:
                    raise

        if training_loss is None:
            logger.error("Training failed at all batch sizes")
            return
    else:
        # Verify checkpoint exists
        if not (FINAL_CHECKPOINT / "adapter_config.json").exists():
            logger.error("No final checkpoint at %s — run Phase 2 first", FINAL_CHECKPOINT)
            return

    # Phase 3
    run_phase3(training_loss=training_loss, training_time=training_time)


if __name__ == "__main__":
    main()
