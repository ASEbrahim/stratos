#!/usr/bin/env python3
"""
StratOS V3 Scorer — Multi-Model Training Pipeline
===================================================
Trains V3 scorer supporting both Qwen3-8B (proven) and Qwen3.5-9B (new) as base models.

Adapted from the proven V2 training pipeline with:
- Multi-base-model support (auto-detects architecture)
- WeightedRandomSampler for score-band rebalancing
- Per-sample loss weighting for gradient emphasis
- DoRA rank 16, 1 epoch, lr=1e-5, cosine schedule
- Full eval pipeline + GGUF export + Ollama registration
- Rollback support at every stage

Usage:
    # Train on Qwen3-8B (proven base, uses V2 merged model):
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 data/v3_pipeline/train_v3.py --base qwen3-8b

    # Train on Qwen3.5-9B (new base, downloads from HuggingFace):
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 data/v3_pipeline/train_v3.py --base qwen3.5-9b

    # Dry run (5 steps, verify pipeline works):
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 data/v3_pipeline/train_v3.py --base qwen3.5-9b --dry-run

    # Phase 3 only (eval + export, requires checkpoint):
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 data/v3_pipeline/train_v3.py --base qwen3.5-9b --phase3-only

    # Resume from checkpoint:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 data/v3_pipeline/train_v3.py --base qwen3-8b --resume
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
# DISABLED: causes NaN gradients on ROCm 6.2 with DoRA + gradient checkpointing
# os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════
BACKEND_DIR = Path(__file__).resolve().parent.parent  # backend/
V3_DIR = BACKEND_DIR / "data" / "v3_pipeline"
TRAINING_FILE = V3_DIR / "training_v3.jsonl"
EVAL_FILE = BACKEND_DIR / "data" / "v2_pipeline" / "eval_v2.jsonl"  # Shared eval set

# Base model paths
BASE_MODELS = {
    "qwen3-8b": {
        "hf_id": None,  # Use local V2 merged base
        "local_path": BACKEND_DIR / "data" / "v2_pipeline" / "v1_merged_base",
        "arch": "qwen3",
        "auto_class": "AutoModelForCausalLM",
        "eos_token": "<|im_end|>",
        "eos_token_id_override": None,  # Auto-detect
        "pad_token_id": 151643,
    },
    "qwen3.5-9b": {
        "hf_id": "Qwen/Qwen3.5-9B",
        "local_path": None,  # Download from HF
        "arch": "qwen3_5",
        "auto_class": "AutoModelForCausalLM",
        "eos_token": "<|im_end|>",
        "eos_token_id_override": None,
        "pad_token_id": None,  # Auto-detect
    },
}

# ═══════════════════════════════════════════════════════════════════
# Hyperparameters
# ═══════════════════════════════════════════════════════════════════
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 1e-5
GRAD_ACCUM = 8
WARMUP_RATIO = 0.05
EPOCHS = 1
SAVE_STEPS = 200
DEFAULT_QUANT = "q8_0"

WEIGHT_BUCKET = {0.5: "noise", 1.0: "tangential", 1.5: "moderate", 2.0: "high", 3.0: "critical"}


# ═══════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════
def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(output_dir / "training_log.txt"), mode='a'),
        ]
    )
    return logging.getLogger("V3_TRAIN")


# ═══════════════════════════════════════════════════════════════════
# Per-Sample Loss Weighting (proven from V2)
# ═══════════════════════════════════════════════════════════════════

class WeightedCompletionDataCollator:
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
    def __init__(self, inner_collator, target_batches=1000):
        self.inner = inner_collator
        self.seen_weights = []
        self.batch_count = 0
        self.target_batches = target_batches
        self.verified = False
        self.verification_passed = None

    def __call__(self, features):
        if not self.verified:
            for f in features:
                w = f.get("sample_weight", 1.0)
                if isinstance(w, (list, tuple)):
                    w = float(w[0]) if w else 1.0
                self.seen_weights.append(float(w))
            self.batch_count += 1
        return self.inner(features)


def create_v3_trainer_class():
    from trl import SFTTrainer as _BaseSFTTrainer
    import torch
    from torch.nn import CrossEntropyLoss
    from torch.utils.data import WeightedRandomSampler

    logger = logging.getLogger("V3_TRAIN")

    class V3SFTTrainer(_BaseSFTTrainer):
        def __init__(self, *args, sampler_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._sampler_weights = sampler_weights
            self._cl_count = 0

        def _get_train_sampler(self, train_dataset=None):
            if self._sampler_weights is not None:
                return WeightedRandomSampler(
                    weights=self._sampler_weights,
                    num_samples=len(self._sampler_weights),
                    replacement=True,
                )
            return super()._get_train_sampler(train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            self._cl_count += 1
            sample_weights = inputs.pop("sample_weight", None)

            if sample_weights is None:
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch
                )

            labels = inputs.pop("labels")
            inputs.pop("num_items_in_batch", None)

            outputs = model(**inputs)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            bs, seq_len, vocab_size = shift_logits.shape

            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
            per_token_loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            ).view(bs, seq_len)

            mask = (shift_labels != -100).float()
            per_example_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            weights = sample_weights.to(per_example_loss.device)
            weighted_loss = (per_example_loss * weights).sum() / weights.sum()

            if self._cl_count <= 2:
                logger.info("[V3] compute_loss — per-sample weighting ACTIVE")
                logger.info("[V3]   weighted=%.4f unweighted=%.4f",
                            weighted_loss.item(), per_example_loss.mean().item())

            return (weighted_loss, outputs) if return_outputs else weighted_loss

    return V3SFTTrainer


def verify_sampler(collator_wrapper, raw_weight_counts, save_path):
    logger = logging.getLogger("V3_TRAIN")
    seen = collator_wrapper.seen_weights

    actual_counts = Counter()
    for w in seen:
        bucket = WEIGHT_BUCKET.get(round(w, 1), f"unknown_{w}")
        actual_counts[bucket] += 1
    total_seen = len(seen)

    total_weighted = sum(count * weight for weight, count in raw_weight_counts.items())
    expected_fracs = {}
    for weight, count in raw_weight_counts.items():
        bucket = WEIGHT_BUCKET.get(weight, f"unknown_{weight}")
        expected_fracs[bucket] = (count * weight) / total_weighted

    total_raw = sum(raw_weight_counts.values())
    raw_fracs = {}
    for weight, count in raw_weight_counts.items():
        bucket = WEIGHT_BUCKET.get(weight, f"unknown_{weight}")
        raw_fracs[bucket] = count / total_raw

    actual_fracs = {bucket: count / total_seen for bucket, count in actual_counts.items()}

    all_buckets = ["noise", "tangential", "moderate", "high", "critical"]
    logger.info("SAMPLER VERIFICATION (%d batches, %d examples)", collator_wrapper.batch_count, total_seen)
    for bucket in all_buckets:
        exp = expected_fracs.get(bucket, 0)
        act = actual_fracs.get(bucket, 0)
        logger.info("%-15s expected=%5.1f%% actual=%5.1f%%", bucket, exp * 100, act * 100)

    exp_distance = sum(abs(actual_fracs.get(b, 0) - expected_fracs.get(b, 0)) for b in all_buckets)
    raw_distance = sum(abs(actual_fracs.get(b, 0) - raw_fracs.get(b, 0)) for b in all_buckets)
    passed = exp_distance < raw_distance

    if passed:
        logger.info("SAMPLER WORKING — distance to expected: %.4f", exp_distance)
    else:
        logger.error("SAMPLER FAILED — actual matches raw distribution")

    result = {
        "batches_seen": collator_wrapper.batch_count,
        "examples_seen": total_seen,
        "passed": passed,
        "distance_to_expected": round(exp_distance, 4),
        "distance_to_raw": round(raw_distance, 4),
    }
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)

    collator_wrapper.verified = True
    collator_wrapper.verification_passed = passed
    return passed


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Model Loading (multi-base support)
# ═══════════════════════════════════════════════════════════════════

def load_base_model(base_name, logger):
    """Load base model and tokenizer. Handles both local and HF models."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = BASE_MODELS[base_name]
    model_path = config["local_path"] or config["hf_id"]

    logger.info("Loading base model: %s (%s)", base_name, model_path)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    # Set EOS token for ChatML compatibility
    eos_token = config["eos_token"]
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    if eos_id is not None and eos_id != tokenizer.eos_token_id:
        tokenizer.eos_token = eos_token
        tokenizer.eos_token_id = eos_id
        logger.info("Set eos_token = %s (id=%d)", eos_token, eos_id)

    # Set pad token
    if config["pad_token_id"] is not None:
        tokenizer.pad_token_id = config["pad_token_id"]
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(config["pad_token_id"])
    elif tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Tokenizer: vocab=%d, eos=%s(%d), pad=%s(%d)",
                tokenizer.vocab_size, tokenizer.eos_token, tokenizer.eos_token_id,
                tokenizer.pad_token, tokenizer.pad_token_id)

    # Load model
    logger.info("Loading model in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info("Model loaded: %.2fB parameters, arch=%s", param_count, config["arch"])

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Training
# ═══════════════════════════════════════════════════════════════════

def run_phase2(base_name, output_dir, batch_size=4, dry_run=False, resume=False):
    logger = logging.getLogger("V3_TRAIN")
    import torch
    from transformers import TrainerCallback
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset
    from trl import SFTConfig

    max_steps = 5 if dry_run else -1

    logger.info("=" * 70)
    logger.info("PHASE 2 — V3 SCORER TRAINING")
    logger.info("Base model: %s", base_name)
    logger.info("Training data: %s", TRAINING_FILE)
    logger.info("=" * 70)

    model, tokenizer = load_base_model(base_name, logger)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    # Apply DoRA
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

    try:
        model = get_peft_model(model, lora_config)
        logger.info("DoRA adapters applied")
    except Exception as e:
        logger.warning("DoRA failed (%s), falling back to standard LoRA", e)
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            bias="none", task_type=TaskType.CAUSAL_LM, use_dora=False,
        )
        model = get_peft_model(model, lora_config)

    # PEFT meta device fix (ROCm)
    if hasattr(model, 'hf_device_map'):
        del model.hf_device_map
    model = model.to("cuda:0")
    model.print_trainable_parameters()

    # Load training data
    dataset = load_dataset("json", data_files=str(TRAINING_FILE), split="train")
    logger.info("Training examples: %d", len(dataset))

    raw_weight_counts = Counter()
    for w in dataset["sample_weight"]:
        w = w if w is not None else 1.0
        raw_weight_counts[round(w, 1)] += 1
    for w, c in sorted(raw_weight_counts.items()):
        logger.info("  weight=%.1f (%s): %d examples (%.1f%%)",
                     w, WEIGHT_BUCKET.get(w, "?"), c, c / len(dataset) * 100)

    # Convert messages to prompt/completion
    def convert_to_prompt_completion(example):
        msgs = example["messages"]
        prompt = tokenizer.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
        completion = msgs[2]["content"] + tokenizer.eos_token
        result = {"prompt": prompt, "completion": completion}
        w = example.get("sample_weight")
        result["sample_weight"] = w if w is not None else 1.0
        return result

    dataset = dataset.map(convert_to_prompt_completion, remove_columns=["messages"])
    logger.info("Converted messages -> prompt/completion")

    # Log a sample
    logger.info("Sample prompt (first 200 chars): %s", dataset[0]["prompt"][:200])
    logger.info("Sample completion: %s", dataset[0]["completion"][:100])

    # WeightedRandomSampler
    sampler_weights = [float(w) if w is not None else 1.0 for w in dataset["sample_weight"]]

    # Load eval data
    eval_dataset = load_dataset("json", data_files=str(EVAL_FILE), split="train")
    eval_remove = [c for c in eval_dataset.column_names if c not in ["prompt", "completion"]]
    eval_dataset = eval_dataset.map(convert_to_prompt_completion, remove_columns=eval_remove)
    logger.info("Eval dataset: %d examples", len(eval_dataset))

    # Adjust grad accum
    grad_accum = GRAD_ACCUM
    if batch_size == 1:
        grad_accum = 16
        logger.warning("BATCH_SIZE=1 — grad_accum=16")

    total_steps = math.ceil(len(dataset) / (batch_size * grad_accum))
    logger.info("Total optimizer steps: %d", total_steps)

    # Create trainer
    V3SFTTrainer = create_v3_trainer_class()
    final_checkpoint = output_dir / "final_checkpoint"

    # Check for resume
    resume_ckpt = None
    if resume:
        ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            resume_ckpt = str(ckpts[-1])
            logger.info("Resuming from checkpoint: %s", resume_ckpt)

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
                    logger.error("STOPPING — sampler verification failed")
                    control.should_training_stop = True

    trainer = V3SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        sampler_weights=sampler_weights,
        args=SFTConfig(
            output_dir=str(output_dir),
            max_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
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
            eval_strategy="no",
            load_best_model_at_end=False,
            bf16=True,
            fp16=False,
            optim="adamw_torch",
            seed=42,
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            completion_only_loss=True,
            remove_unused_columns=False,
            average_tokens_across_devices=False,
            max_steps=max_steps,
        ),
    )

    # Wrap collator
    weighted_collator = WeightedCompletionDataCollator(trainer.data_collator)
    verifying_collator = VerifyingCollatorWrapper(weighted_collator, target_batches=1000)
    trainer.data_collator = verifying_collator

    verification_path = output_dir / "sampler_verification.json"
    trainer.add_callback(SamplerCheckCallback(verifying_collator, raw_weight_counts, str(verification_path)))

    # Train
    logger.info("Starting V3 training...")
    start_time = time.time()
    stats = trainer.train(resume_from_checkpoint=resume_ckpt)
    elapsed = time.time() - start_time
    logger.info("Training complete! Loss: %.4f, Time: %.1fh", stats.training_loss, elapsed / 3600)

    # Verify sampler
    if not verifying_collator.verified and verifying_collator.batch_count > 0:
        verify_sampler(verifying_collator, raw_weight_counts, str(verification_path))

    # Save final checkpoint
    final_checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_checkpoint))
    tokenizer.save_pretrained(str(final_checkpoint))
    logger.info("Final adapter saved to: %s", final_checkpoint)

    # Save training metadata
    meta = {
        "base_model": base_name,
        "base_path": str(BASE_MODELS[base_name].get("local_path") or BASE_MODELS[base_name]["hf_id"]),
        "training_file": str(TRAINING_FILE),
        "eval_file": str(EVAL_FILE),
        "training_loss": stats.training_loss,
        "training_time_hours": elapsed / 3600,
        "hyperparams": {
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "lr": LEARNING_RATE, "epochs": EPOCHS, "max_seq_length": MAX_SEQ_LENGTH,
            "batch_size": batch_size, "grad_accum": grad_accum,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    del model, trainer
    gc.collect()
    import torch as _t
    _t.cuda.empty_cache()

    return stats.training_loss, elapsed


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Evaluation + Export
# ═══════════════════════════════════════════════════════════════════

def extract_profile(system_prompt):
    m = re.search(r'You are a relevance scorer for (?:a |an )?(.*?)(?:\.\n|\n)', system_prompt)
    return m.group(1).strip() if m else "UNKNOWN"


def extract_score(text):
    clean = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    if '<think>' in clean and '</think>' not in clean:
        clean = re.sub(r'<think>.*$', '', clean, flags=re.DOTALL)
    m = re.search(r'SCORE:\s*(\d+\.?\d*)', clean, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Fallback: extract from think block
    m = re.search(r'[Ff]inal\s+[Ss]core:\s*(\d+\.?\d*)', text)
    if m:
        return float(m.group(1))
    return None


def extract_title(user_msg):
    m = re.search(r'Title:\s*(.*?)(?:\n|$)', user_msg)
    return m.group(1).strip() if m else "UNKNOWN"


def run_inference(model, tokenizer, eval_examples, logger, desc="Inference"):
    import torch

    model.eval()
    tokenizer.padding_side = "left"
    results = []
    total = len(eval_examples)

    for i, ex in enumerate(eval_examples):
        msgs = ex["messages"][:2]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        results.append({
            "generated": generated.strip(),
            "pred_score": extract_score(generated),
            "gt_score": extract_score(ex["messages"][2]["content"]),
            "profile": extract_profile(ex["messages"][0]["content"]),
            "title": extract_title(ex["messages"][1]["content"]),
        })

        if (i + 1) % 100 == 0:
            logger.info("  %s: %d/%d (%.0f%%)", desc, i + 1, total, (i + 1) / total * 100)

    logger.info("  %s: %d/%d complete", desc, total, total)
    return results


def compute_metrics(results):
    valid = [r for r in results if r["pred_score"] is not None and r["gt_score"] is not None]
    total = len(results)
    parse_failures = total - len(valid)

    if not valid:
        return {"error": "No valid predictions", "parse_failures": parse_failures}

    preds = [r["pred_score"] for r in valid]
    gts = [r["gt_score"] for r in valid]

    correct_dir = sum(1 for p, g in zip(preds, gts) if (p >= 5.0) == (g >= 5.0))
    direction_acc = correct_dir / len(valid)
    mae = sum(abs(p - g) for p, g in zip(preds, gts)) / len(valid)

    # Spearman
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(preds, gts)
    except ImportError:
        # Manual Spearman
        def rank(vals):
            indexed = sorted(enumerate(vals), key=lambda x: x[1])
            ranks = [0.0] * len(vals)
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
        ra, rb = rank(preds), rank(gts)
        n = len(ra)
        ma, mb = sum(ra)/n, sum(rb)/n
        cov = sum((a-ma)*(b-mb) for a, b in zip(ra, rb)) / n
        sa = (sum((a-ma)**2 for a in ra) / n) ** 0.5
        sb = (sum((b-mb)**2 for b in rb) / n) ** 0.5
        rho = cov / (sa * sb) if sa > 0 and sb > 0 else 0.0

    within_1 = sum(1 for p, g in zip(preds, gts) if abs(p - g) <= 1.0) / len(valid) * 100
    within_2 = sum(1 for p, g in zip(preds, gts) if abs(p - g) <= 2.0) / len(valid) * 100

    return {
        "total_examples": total,
        "valid_predictions": len(valid),
        "parse_failures": parse_failures,
        "direction_accuracy": direction_acc,
        "mae": mae,
        "spearman_rho": rho,
        "within_1pt_pct": within_1,
        "within_2pt_pct": within_2,
    }


def export_to_gguf(merged_path, output_dir, model_name="v3_scorer"):
    logger = logging.getLogger("V3_TRAIN")
    llama_convert = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"

    if not llama_convert.exists():
        logger.warning("llama.cpp not found at ~/llama.cpp — GGUF export skipped")
        logger.info("Manual: python ~/llama.cpp/convert_hf_to_gguf.py %s --outfile %s_bf16.gguf --outtype bf16",
                     merged_path, model_name)
        return None

    bf16_path = Path(output_dir) / f"{model_name}_bf16.gguf"

    logger.info("Converting to GGUF (bf16)...")
    result = subprocess.run(
        [sys.executable, str(llama_convert), merged_path,
         "--outfile", str(bf16_path), "--outtype", "bf16"],
        capture_output=True, text=True, timeout=1800
    )

    if result.returncode != 0 or not bf16_path.exists():
        logger.error("GGUF conversion failed: %s", result.stderr[:500])
        return None

    logger.info("BF16 GGUF: %s (%d MB)", bf16_path, bf16_path.stat().st_size // (1024 * 1024))

    # Quantize
    llama_quantize = Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize"
    if not llama_quantize.exists():
        llama_quantize = Path.home() / "llama.cpp" / "llama-quantize"

    if llama_quantize.exists():
        q8_path = Path(output_dir) / f"{model_name}.gguf"
        logger.info("Quantizing to %s...", DEFAULT_QUANT)
        qresult = subprocess.run(
            [str(llama_quantize), str(bf16_path), str(q8_path), DEFAULT_QUANT.upper()],
            capture_output=True, text=True, timeout=1800
        )
        if qresult.returncode == 0 and q8_path.exists():
            logger.info("q8_0 GGUF: %s (%d MB)", q8_path, q8_path.stat().st_size // (1024 * 1024))
            bf16_path.unlink(missing_ok=True)
            return str(q8_path)
        else:
            logger.warning("Quantization failed, keeping bf16")
            return str(bf16_path)
    else:
        logger.info("llama-quantize not found — keeping bf16 GGUF")
        return str(bf16_path)


def run_phase3(base_name, output_dir, training_loss=None, training_time=None):
    logger = logging.getLogger("V3_TRAIN")
    import torch
    from peft import PeftModel

    logger.info("=" * 70)
    logger.info("PHASE 3 — POST-TRAINING EVALUATION")
    logger.info("=" * 70)

    gc.collect()
    torch.cuda.empty_cache()

    final_checkpoint = output_dir / "final_checkpoint"

    # Load base + adapters
    model, tokenizer = load_base_model(base_name, logger)
    logger.info("Loading V3 adapters from %s...", final_checkpoint)
    model = PeftModel.from_pretrained(model, str(final_checkpoint))
    if hasattr(model, 'hf_device_map'):
        del model.hf_device_map
    model = model.to("cuda:0")
    model.eval()

    # Load eval data
    eval_examples = []
    with open(EVAL_FILE) as f:
        for line in f:
            eval_examples.append(json.loads(line))
    logger.info("Loaded %d eval examples", len(eval_examples))

    # Merge adapters
    logger.info("Merging adapters into base weights...")
    model = model.merge_and_unload()
    model.eval()

    # Run inference
    logger.info("Running V3 inference on eval set...")
    start = time.time()
    results = run_inference(model, tokenizer, eval_examples, logger, desc="V3 eval")
    eval_time = time.time() - start
    logger.info("Inference: %.0fs (%.1fs/example)", eval_time, eval_time / len(eval_examples))

    # Metrics
    metrics = compute_metrics(results)

    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info("Direction accuracy: %.1f%%", metrics.get('direction_accuracy', 0) * 100)
    logger.info("MAE:                %.3f", metrics.get('mae', 0))
    logger.info("Spearman rho:       %.3f", metrics.get('spearman_rho', 0))
    logger.info("Within 1.0 point:   %.1f%%", metrics.get('within_1pt_pct', 0))
    logger.info("Within 2.0 points:  %.1f%%", metrics.get('within_2pt_pct', 0))
    logger.info("Parse failures:     %d/%d", metrics.get('parse_failures', 0), metrics.get('total_examples', 0))

    # GGUF Export
    logger.info("=" * 70)
    logger.info("GGUF EXPORT")
    logger.info("=" * 70)

    merged_path = output_dir / "merged_v3"
    merged_path.mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    logger.info("Merged model saved to %s", merged_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    model_name = f"v3_scorer_{base_name.replace('.', '_').replace('-', '_')}"
    gguf_path = export_to_gguf(str(merged_path), str(output_dir), model_name=model_name)

    # Cleanup merged dir
    if gguf_path and merged_path.exists():
        shutil.rmtree(str(merged_path), ignore_errors=True)
        logger.info("Cleaned merged_v3/")

    # Save report
    report = {
        "base_model": base_name,
        "metrics": metrics,
        "gguf_path": gguf_path,
        "training_loss": training_loss,
        "training_time_hours": training_time / 3600 if training_time else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", report_path)

    # Registration instructions
    if gguf_path:
        ollama_name = f"stratos-scorer-v3-{base_name.replace('.', '')}"
        logger.info("")
        logger.info("=" * 70)
        logger.info("TO REGISTER IN OLLAMA:")
        logger.info("  python3 model_manager.py register %s %s", gguf_path, ollama_name)
        logger.info("  python3 model_manager.py switch %s", ollama_name)
        logger.info("")
        logger.info("TO ROLLBACK:")
        logger.info("  python3 model_manager.py rollback")
        logger.info("=" * 70)

    return metrics


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="StratOS V3 Scorer Training Pipeline")
    parser.add_argument("--base", choices=list(BASE_MODELS.keys()), required=True,
                        help="Base model to train on")
    parser.add_argument("--phase3-only", action="store_true",
                        help="Skip training, run evaluation + export only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Train for 5 steps only (verify pipeline works)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Starting batch size (fallback: 4->2->1)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    # Output dir per base model
    output_dir = V3_DIR / f"training_output_{args.base.replace('.', '_').replace('-', '_')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    logger.info("=" * 70)
    logger.info("StratOS V3 Scorer Training Pipeline")
    logger.info("Base model:  %s", args.base)
    logger.info("Output dir:  %s", output_dir)
    logger.info("Training:    %s", TRAINING_FILE)
    logger.info("Eval:        %s", EVAL_FILE)
    logger.info("Dry run:     %s", args.dry_run)
    logger.info("Resume:      %s", args.resume)
    logger.info("=" * 70)

    # Verify training data exists
    if not TRAINING_FILE.exists():
        logger.error("Training file not found: %s", TRAINING_FILE)
        sys.exit(1)
    if not EVAL_FILE.exists():
        logger.error("Eval file not found: %s", EVAL_FILE)
        sys.exit(1)

    # Verify base model exists
    base_cfg = BASE_MODELS[args.base]
    if base_cfg["local_path"] and not base_cfg["local_path"].exists():
        logger.error("Local base model not found: %s", base_cfg["local_path"])
        logger.info("For Qwen3.5-9B, it will be downloaded from HuggingFace automatically.")
        if base_cfg["hf_id"] is None:
            sys.exit(1)

    training_loss = None
    training_time = None

    if not args.phase3_only:
        # Unload Ollama models to free VRAM
        logger.info("Unloading Ollama models to free VRAM...")
        try:
            import requests
            for model_name in ["stratos-scorer-v2", "qwen3:30b-a3b", "qwen3:14b", "qwen3.5:9b"]:
                try:
                    requests.post("http://localhost:11434/api/generate",
                        json={"model": model_name, "keep_alive": 0}, timeout=10)
                except Exception:
                    pass
            time.sleep(3)
        except Exception:
            pass

        # Train with batch size fallback
        batch_chain = [bs for bs in [4, 2, 1] if bs <= args.batch_size]
        for bs in batch_chain:
            try:
                logger.info("Attempting training with batch_size=%d...", bs)
                training_loss, training_time = run_phase2(
                    args.base, output_dir, batch_size=bs,
                    dry_run=args.dry_run, resume=args.resume
                )
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM with batch_size=%d — falling back", bs)
                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()
                    time.sleep(5)
                    if bs == 1:
                        logger.error("OOM even with batch_size=1")
                        raise
                else:
                    raise

        if training_loss is None:
            logger.error("Training failed at all batch sizes")
            return
    else:
        final_ckpt = output_dir / "final_checkpoint" / "adapter_config.json"
        if not final_ckpt.exists():
            logger.error("No final checkpoint at %s — run Phase 2 first", output_dir / "final_checkpoint")
            return

    # Phase 3
    run_phase3(args.base, output_dir, training_loss, training_time)


if __name__ == "__main__":
    main()
