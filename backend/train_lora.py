#!/usr/bin/env python3
"""
STRAT_OS — LoRA Fine-Tuning Pipeline
======================================
Fine-tunes Qwen 3 on distillation corrections using LoRA, then exports
to GGUF and registers with Ollama as a new model.

Prerequisites:
    pip install unsloth transformers datasets peft trl --break-system-packages
    # OR if on Windows without WSL:
    pip install transformers datasets peft trl bitsandbytes accelerate

Usage:
    # Step 1: Export training data (run export_training.py first)
    python export_training.py

    # Step 2: Fine-tune
    python train_lora.py                           # Uses default settings
    python train_lora.py --epochs 3                # More training passes
    python train_lora.py --base-model qwen3:1.7b   # Specify base model
    python train_lora.py --skip-register            # Train but don't register with Ollama

    # Step 3: Verify
    ollama run stratos-scorer-v1 "Score this: TSMC announces new fab in Kuwait"

Pipeline:
    training_data.jsonl → LoRA fine-tune → Merged model → GGUF export → Ollama model

Alternative: If local GPU is insufficient, use the --export-colab flag to generate
a Google Colab notebook that runs training on free T4 GPUs.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("TRAIN")

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

# Model tiers — script auto-selects based on available VRAM
MODEL_TIERS = [
    # (min_vram_gb, hf_model_id, ollama_equivalent, lora_r, batch_size, description)
    (20, "Qwen/Qwen3-8B",   "qwen3:8b",   16, 1, "8B — best quality for 24GB VRAM (DoRA)"),
    (10, "Qwen/Qwen3-4B",   "qwen3:4b",   16, 8, "4B — balanced quality/speed (DoRA)"),
    (4,  "Qwen/Qwen3-1.7B", "qwen3:1.7b", 16, 16, "1.7B — lightweight, fast training (DoRA)"),
]

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"               # Default for 24GB VRAM
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
OUTPUT_MODEL_NAME = "stratos-scorer"
DEFAULT_QUANT = "q8_0"                               # Q8_0 mandatory for fine-tuned scoring models
LORA_R = 16                                          # DoRA works well with lower rank
LORA_ALPHA = 32                                      # 2:1 ratio for DoRA (v19: higher alpha for CoT)
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 1024                                # Accommodates richer training examples with tracked fields
LEARNING_RATE = 1e-4                                 # Gentler learning rate for DoRA
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 8

# ═══════════════════════════════════════════════════════════════════
# v19: Per-Sample Loss Weighting (replaces upsampling rebalancing)
# ═══════════════════════════════════════════════════════════════════

class WeightedCompletionDataCollator:
    """Wraps the base data collator to preserve sample_weight through tokenization.

    The base collator (DataCollatorForCompletionOnlyLM) handles tokenization
    and label masking. This wrapper extracts sample_weight before the base
    collator processes features, then adds it back to the batch tensor.
    """

    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
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

        import torch
        batch["sample_weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch


def _create_weighted_trainer_class():
    """Create WeightedSFTTrainer class (deferred import to avoid import errors)."""
    from trl import SFTTrainer as _BaseSFTTrainer
    import torch
    from torch.nn import CrossEntropyLoss

    class WeightedSFTTrainer(_BaseSFTTrainer):
        """SFTTrainer with per-sample loss weighting for score band rebalancing.

        Instead of upsampling rare score bands (which causes overfitting when
        there are few unique examples), this multiplies the loss for each
        sample by its weight. Rare bands (critical, high) get higher weights
        so the model pays more attention to them despite seeing fewer examples.
        """

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("sample_weight", None)

            # If no weights or all weights are 1.0, use default loss
            if weights is None:
                outputs = model(**inputs)
                loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Per-token cross entropy (no reduction — we reduce manually)
            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
            flat_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            per_token_loss = flat_loss.view(shift_labels.size())

            # Per-sample mean loss (ignoring masked/padding tokens)
            mask = (shift_labels != -100).float()
            tokens_per_sample = mask.sum(dim=1).clamp(min=1)
            per_sample_loss = (per_token_loss * mask).sum(dim=1) / tokens_per_sample

            # Weighted mean across batch
            weights = weights.to(per_sample_loss.device).float()
            loss = (per_sample_loss * weights).sum() / weights.sum()

            return (loss, outputs) if return_outputs else loss

    return WeightedSFTTrainer


def auto_select_model(vram_gb: float, is_rocm: bool = False) -> tuple:
    """Pick the best model that fits in available VRAM.
    ROCm needs more headroom since bitsandbytes 4-bit quantization isn't available."""
    # ROCm loads in fp16 (no 4-bit), so needs ~2x more VRAM per parameter
    # Also need headroom for optimizer states + gradients during training
    rocm_tiers = [
        # (min_vram_gb, hf_model_id, ollama_equivalent, lora_r, batch_size, description)
        # Ollama is stopped during training, so full VRAM is available
        # 8B bf16 = ~16GB weights + ~5GB training overhead = ~21GB (fits in 24GB)
        (22, "Qwen/Qwen3-8B",   "qwen3:8b",   16, 1, "8B bf16 — best quality for 24GB ROCm (DoRA)"),
        (16, "Qwen/Qwen3-4B",   "qwen3:4b",   16, 1, "4B bf16 — good quality for 24GB ROCm (DoRA)"),
        (8,  "Qwen/Qwen3-1.7B", "qwen3:1.7b", 16, 4, "1.7B bf16 — lightweight ROCm training (DoRA)"),
    ]
    tiers = rocm_tiers if is_rocm else MODEL_TIERS
    for min_vram, hf_id, ollama_id, r, bs, desc in tiers:
        if vram_gb >= min_vram:
            return hf_id, ollama_id, r, bs, desc
    # Fallback to smallest
    t = tiers[-1]
    return t[1], t[2], t[3], t[4], t[5]

# ═══════════════════════════════════════════════════════════════════
# Check Environment
# ═══════════════════════════════════════════════════════════════════

def check_gpu():
    """Check for CUDA or ROCm GPU availability."""
    try:
        import torch
        # torch.cuda works for both NVIDIA CUDA and AMD ROCm (HIP backend)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            # Detect if this is ROCm (AMD) or CUDA (NVIDIA)
            is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            backend = "ROCm" if is_rocm else "CUDA"
            logger.info(f"GPU: {gpu_name} ({vram:.1f} GB VRAM) [{backend}]")
            return True, vram, is_rocm
        else:
            logger.warning("No GPU detected (no CUDA or ROCm)")
            return False, 0, False
    except ImportError:
        logger.warning("PyTorch not installed")
        return False, 0, False


def check_unsloth():
    """Check if unsloth is available."""
    try:
        import unsloth
        logger.info(f"Unsloth {unsloth.__version__} available")
        return True
    except ImportError:
        return False


def check_peft():
    """Check if peft/transformers are available as fallback."""
    try:
        import peft
        import transformers
        logger.info(f"PEFT {peft.__version__} + Transformers {transformers.__version__}")
        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════
# Training Data Rebalancing (v16 fix)
# ═══════════════════════════════════════════════════════════════════

SCORE_BANDS = {
    "noise":      (0.0, 2.0),
    "tangential": (2.5, 4.0),
    "moderate":   (4.5, 6.5),
    "high":       (7.0, 8.0),
    "critical":   (8.5, 10.0),
}

def _extract_score(messages: list) -> float:
    """Extract SCORE from assistant message content."""
    import re
    asst = messages[-1]["content"]
    # Strip think blocks first
    clean = re.sub(r'<think>.*?</think>\s*', '', asst, flags=re.DOTALL)
    m = re.search(r'SCORE:\s*(\d+\.?\d*)', clean, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return -1.0

def _get_band(score: float) -> str:
    """Classify a score into a band."""
    for band, (lo, hi) in SCORE_BANDS.items():
        if lo <= score <= hi:
            return band
    return "noise"  # Default for unmatched (e.g., 2.1-2.4 gap)

def rebalance_dataset(dataset, target_pct: float = 0.20):
    """Upsample underrepresented score bands to target percentage.

    v15 had 65% noise, causing bimodal collapse. This upsamples
    mid-range and high-score examples so each band is ~20%.

    Args:
        dataset: HuggingFace Dataset with 'messages' column
        target_pct: Target fraction per band (default 0.20 = 20%)

    Returns:
        Rebalanced Dataset
    """
    import random
    random.seed(42)

    # Classify examples by band
    band_indices = {band: [] for band in SCORE_BANDS}
    for i, example in enumerate(dataset):
        score = _extract_score(example["messages"])
        band = _get_band(score)
        band_indices[band].append(i)

    total = len(dataset)
    target_per_band = int(total * target_pct)

    logger.info("Score band distribution (before rebalancing):")
    for band, indices in band_indices.items():
        pct = len(indices) / total * 100 if total > 0 else 0
        logger.info(f"  {band:12s}: {len(indices):5d} ({pct:.1f}%)")

    # Build rebalanced index list
    all_indices = []
    for band, indices in band_indices.items():
        if len(indices) == 0:
            logger.warning(f"  Band '{band}' has 0 examples — cannot upsample")
            continue
        if len(indices) >= target_per_band:
            # Downsample majority class (noise) to target
            all_indices.extend(random.sample(indices, target_per_band))
        else:
            # Upsample: include all originals + random repeats to reach target
            all_indices.extend(indices)
            shortfall = target_per_band - len(indices)
            all_indices.extend(random.choices(indices, k=shortfall))

    random.shuffle(all_indices)

    rebalanced = dataset.select(all_indices)

    logger.info(f"Rebalanced: {total} → {len(rebalanced)} examples")
    # Log new distribution
    new_bands = {band: 0 for band in SCORE_BANDS}
    for example in rebalanced:
        score = _extract_score(example["messages"])
        band = _get_band(score)
        new_bands[band] += 1
    logger.info("Score band distribution (after rebalancing):")
    for band, count in new_bands.items():
        pct = count / len(rebalanced) * 100 if len(rebalanced) > 0 else 0
        logger.info(f"  {band:12s}: {count:5d} ({pct:.1f}%)")

    return rebalanced


# ═══════════════════════════════════════════════════════════════════
# Training with Unsloth (Preferred — 2x faster, less VRAM)
# ═══════════════════════════════════════════════════════════════════

def train_with_unsloth(
    training_file: str,
    output_dir: str,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 3,
    lr: float = LEARNING_RATE,
):
    """Fine-tune using unsloth (recommended)."""
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    logger.info(f"Loading base model: {base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # 4-bit quantization for less VRAM
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    # Load training data
    dataset = load_dataset("json", data_files=training_file, split="train")
    logger.info(f"Training examples: {len(dataset)}")

    # Rebalance score distribution (v16 fix for bimodal collapse)
    dataset = rebalance_dataset(dataset)

    # v16: Convert messages → prompt/completion for correct loss masking
    # completion_only_loss=True only works with prompt/completion columns, NOT messages
    def convert_to_prompt_completion(example):
        msgs = example["messages"]
        prompt = tokenizer.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
        completion = msgs[2]["content"] + "<|im_end|>"
        return {"prompt": prompt, "completion": completion}

    dataset = dataset.map(convert_to_prompt_completion, remove_columns=["messages"])
    logger.info("Converted messages → prompt/completion for assistant-only loss masking")

    from trl import SFTConfig as _SFTConfig
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=_SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            num_train_epochs=epochs,
            learning_rate=lr,
            warmup_steps=10,
            logging_steps=5,
            save_strategy="epoch",
            fp16=True,
            optim="adamw_8bit",
            seed=42,
            report_to="none",
            completion_only_loss=True,  # v16: Only train on assistant (completion) tokens
        ),
    )
    
    logger.info("Starting training...")
    stats = trainer.train()
    logger.info(f"Training complete! Loss: {stats.training_loss:.4f}")
    
    # Save LoRA adapter
    lora_path = Path(output_dir) / "lora_adapter"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    logger.info(f"LoRA adapter saved to: {lora_path}")
    
    return model, tokenizer, str(lora_path)


# ═══════════════════════════════════════════════════════════════════
# Training with PEFT (Fallback — more compatible)
# ═══════════════════════════════════════════════════════════════════

def train_with_peft(
    training_file: str,
    output_dir: str,
    base_model: str = "Qwen/Qwen3-1.7B",
    epochs: int = 3,
    lr: float = LEARNING_RATE,
    is_rocm: bool = False,
    max_steps: int = -1,
):
    """Fine-tune using standard PEFT (works on both CUDA and ROCm)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch
    
    # ROCm memory optimizations
    if is_rocm:
        os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # v19 CRITICAL: Qwen3's tokenizer silently changed eos_token to <|endoftext|>
    # but the chat template still uses <|im_end|> as the turn terminator.
    # Without this fix, the model won't stop generating at turn boundaries.
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    logger.info(f"Set eos_token = <|im_end|> (id={tokenizer.eos_token_id})")

    # Ensure pad_token_id = 151643 (endoftext), NOT eos (151645 = im_end)
    if tokenizer.pad_token_id != 151643:
        tokenizer.pad_token_id = 151643
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        logger.info(f"Set pad_token_id = 151643 (endoftext)")

    # ROCm: bitsandbytes not supported, use bfloat16 directly (fine with 24GB VRAM)
    # CUDA: can use 4-bit quantization to save VRAM
    if is_rocm:
        logger.info("ROCm detected — using bfloat16 with SDPA attention")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},  # Force all params onto GPU 0 — avoids meta device gradient crash
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    else:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            logger.info("CUDA detected — using 4-bit quantization")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except (ImportError, Exception) as e:
            logger.info(f"4-bit unavailable ({e}), falling back to float16")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
    
    # Enable gradient checkpointing to reduce VRAM (trades compute for memory)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False  # Required when gradient checkpointing is on
    
    # DoRA config (Weight-Decomposed Low-Rank Adaptation)
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
    
    # ── Critical fix for ROCm: PEFT wrapping leaves some params on meta device ──
    # The Trainer sees hf_device_map and refuses to relocate them, causing:
    #   RuntimeError: expected device meta but got cuda:0 during backprop
    # Fix: delete the attribute and force everything to GPU 0
    if hasattr(model, 'hf_device_map'):
        del model.hf_device_map
    model = model.to("cuda:0")
    logger.info("Forced all parameters to cuda:0 (meta device cleanup)")
    
    model.print_trainable_parameters()
    
    # Load data
    dataset = load_dataset("json", data_files=training_file, split="train")
    logger.info(f"Training examples: {len(dataset)}")

    # v19: Check if dataset has sample_weight (loss weighting instead of upsampling)
    has_sample_weights = "sample_weight" in dataset.column_names
    if has_sample_weights:
        logger.info("v19 loss weighting: sample_weight column detected — skipping rebalancing")
        # Log weight distribution
        from collections import Counter as _Counter
        weight_dist = _Counter(round(w, 1) for w in dataset["sample_weight"])
        for w, c in sorted(weight_dist.items()):
            logger.info(f"  weight={w:.1f}: {c} examples")
    else:
        # Legacy path: rebalance via upsampling (v16 fix for bimodal collapse)
        logger.info("No sample_weight column — using legacy upsampling rebalance")
        dataset = rebalance_dataset(dataset)

    # ── v16: Convert messages → prompt/completion for correct loss masking ──
    # v15 bug: pre-formatting to "text" trained on ALL tokens (system+user+assistant).
    # Fix: Convert to prompt/completion format. completion_only_loss=True creates a mask
    # that zeroes loss on prompt tokens and only trains on completion (assistant) tokens.
    # NOTE: "messages" column + completion_only_loss does NOT work because Qwen3's chat
    # template lacks {% generation %} markers needed for assistant_only_loss masking.
    def convert_to_prompt_completion(example):
        msgs = example["messages"]
        prompt = tokenizer.apply_chat_template(msgs[:2], tokenize=False, add_generation_prompt=True)
        completion = msgs[2]["content"] + "<|im_end|>"
        result = {"prompt": prompt, "completion": completion}
        # v19: preserve sample_weight through format conversion
        if "sample_weight" in example:
            result["sample_weight"] = example["sample_weight"]
        return result

    remove_cols = ["messages"]
    dataset = dataset.map(convert_to_prompt_completion, remove_columns=remove_cols)
    logger.info("Converted messages → prompt/completion for assistant-only loss masking")

    # Load eval dataset if available
    # v19: look for eval_v19_cot.jsonl first, fall back to distill_v2_eval.jsonl
    eval_dataset = None
    eval_candidates = [
        Path(training_file).parent / "eval_v19_cot.jsonl",
        Path(training_file).parent / "distill_v2_eval.jsonl",
    ]
    eval_file = None
    for candidate in eval_candidates:
        if candidate.exists():
            eval_file = candidate
            break

    if eval_file:
        eval_dataset = load_dataset("json", data_files=str(eval_file), split="train")
        # Convert eval dataset too (drop sample_weight from eval — not needed for eval loss)
        eval_remove_cols = [c for c in eval_dataset.column_names if c not in ["prompt", "completion"]]
        eval_dataset = eval_dataset.map(convert_to_prompt_completion, remove_columns=eval_remove_cols)
        logger.info(f"Eval dataset: {eval_file.name} ({len(eval_dataset)} examples)")
    else:
        logger.info("No eval dataset found — training without evaluation")

    # Adjust batch size for ROCm (no quantization = more VRAM usage)
    effective_batch = BATCH_SIZE
    effective_grad_accum = GRADIENT_ACCUMULATION
    if is_rocm:
        effective_batch = 1
        # Target effective batch = 32. BATCH_SIZE may have been overridden by auto_select_model(),
        # so use a fixed target instead of BATCH_SIZE * GRADIENT_ACCUMULATION.
        effective_grad_accum = 32

    # v19: Curriculum ordering — sort by difficulty (distance from 5.0)
    # Easy examples (obvious noise/hits) first, hard mid-range cases last.
    # First epoch uses curriculum order; SFTTrainer shuffles subsequent epochs.
    def _sort_by_curriculum(dataset):
        """Sort dataset by absolute distance from score 5.0 (descending = easy first)."""
        scores = []
        for ex in dataset:
            s = _extract_score_from_completion(ex.get("completion", ""))
            scores.append(s)
        # Sort indices by |score - 5.0| descending (easy first)
        sorted_indices = sorted(range(len(scores)), key=lambda i: abs(scores[i] - 5.0), reverse=True)
        return dataset.select(sorted_indices)

    def _extract_score_from_completion(text):
        """Extract score from completion text."""
        import re
        clean = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        m = re.search(r'SCORE:\s*(\d+\.?\d*)', clean, re.IGNORECASE)
        return float(m.group(1)) if m else 5.0

    dataset = _sort_by_curriculum(dataset)
    logger.info("Applied curriculum ordering (easy→hard by distance from 5.0)")

    # Early stopping callback (patience 3 evaluations)
    callbacks = []
    if eval_dataset:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        logger.info("Early stopping enabled (patience=3)")

    # v19: Use WeightedSFTTrainer if sample weights are present
    if has_sample_weights:
        _TrainerClass = _create_weighted_trainer_class()
        logger.info("Using WeightedSFTTrainer for per-sample loss weighting")
    else:
        _TrainerClass = SFTTrainer

    trainer = _TrainerClass(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks if callbacks else None,
        args=SFTConfig(
            output_dir=output_dir,
            max_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=effective_batch,
            per_device_eval_batch_size=1,  # v19: match train batch to avoid OOM during eval (151k vocab logits)
            gradient_accumulation_steps=effective_grad_accum,
            num_train_epochs=epochs,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            max_grad_norm=1.0,
            logging_steps=5,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            eval_accumulation_steps=4,  # v19: accumulate eval preds to reduce peak VRAM
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            bf16=True,
            fp16=False,
            optim="adamw_torch",
            seed=42,
            report_to="none",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            completion_only_loss=True,  # v16: Only train on assistant response tokens
            remove_unused_columns=False,  # v19: preserve sample_weight column through pipeline
            max_steps=max_steps,  # v19: for dry runs (e.g. --max-steps 20)
        ),
    )

    # v19: Wrap data collator to pass sample_weight through to compute_loss
    if has_sample_weights:
        trainer.data_collator = WeightedCompletionDataCollator(trainer.data_collator)
        logger.info("Wrapped data collator for sample_weight passthrough")
    
    logger.info("Starting training...")
    stats = trainer.train()
    logger.info(f"Training complete! Loss: {stats.training_loss:.4f}")

    # Explicitly load best checkpoint if available (load_best_model_at_end
    # can silently fail with PEFT + hf_device_map deletion workaround)
    if eval_dataset and hasattr(trainer.state, 'best_model_checkpoint') and trainer.state.best_model_checkpoint:
        best_ckpt = trainer.state.best_model_checkpoint
        best_score = getattr(trainer.state, 'best_metric', None)
        logger.info(f"Best checkpoint: {best_ckpt} (eval_loss={best_score})")
        try:
            from peft import PeftModel as _PM
            best_adapter = Path(best_ckpt)
            if (best_adapter / "adapter_model.safetensors").exists():
                model.load_adapter(str(best_adapter), "default", is_trainable=True)
                logger.info(f"✓ Reloaded best checkpoint adapter weights")
            else:
                logger.warning(f"Best checkpoint adapter not found at {best_adapter}")
        except Exception as e:
            logger.warning(f"Could not reload best checkpoint: {e}")

    # Save
    lora_path = Path(output_dir) / "lora_adapter"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    logger.info(f"LoRA adapter saved to: {lora_path}")
    
    return model, tokenizer, str(lora_path)


# ═══════════════════════════════════════════════════════════════════
# GGUF Export & Ollama Registration
# ═══════════════════════════════════════════════════════════════════

def export_gguf_unsloth(model, tokenizer, output_dir: str, quant: str = "q4_k_m") -> str:
    """Export merged model to GGUF using unsloth's built-in exporter."""
    gguf_path = Path(output_dir) / "gguf"
    gguf_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting to GGUF ({quant})...")
    model.save_pretrained_gguf(
        str(gguf_path),
        tokenizer,
        quantization_method=quant,
    )
    
    # Find the output file
    gguf_files = list(gguf_path.glob("*.gguf"))
    if gguf_files:
        logger.info(f"GGUF exported: {gguf_files[0]} ({gguf_files[0].stat().st_size / 1024 / 1024:.0f} MB)")
        return str(gguf_files[0])
    
    raise FileNotFoundError(f"No .gguf file found in {gguf_path}")


def export_gguf_manual(lora_path: str, output_dir: str, base_model: str = "Qwen/Qwen3-1.7B", quant: str = "q4_k_m") -> str:
    """Merge LoRA + convert to GGUF using llama.cpp (auto-detected)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    
    logger.info("Merging LoRA weights with base model...")
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, lora_path)
    merged = merged.merge_and_unload()
    
    merged_path = Path(output_dir) / "merged"
    merged_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_path))
    
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    tokenizer.save_pretrained(str(merged_path))
    
    logger.info(f"Merged model saved to {merged_path}")
    
    # Auto-detect llama.cpp convert script
    gguf_path = None
    llama_convert = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"
    
    if llama_convert.exists():
        logger.info("Found llama.cpp — converting to GGUF (bf16 intermediate)...")
        bf16_path = Path(output_dir) / "stratos-scorer-bf16.gguf"

        result = subprocess.run(
            [sys.executable, str(llama_convert), str(merged_path),
             "--outfile", str(bf16_path), "--outtype", "bf16"],
            capture_output=True, text=True
        )

        if result.returncode == 0 and bf16_path.exists():
            logger.info(f"✓ GGUF exported: {bf16_path} ({bf16_path.stat().st_size / 1024 / 1024:.0f} MB)")

            # Delete merged dir to free ~16GB before quantization
            if merged_path.exists():
                shutil.rmtree(str(merged_path), ignore_errors=True)
                logger.info(f"  Cleaned merged/ to free disk for quantization")

            # Quantize if not bf16/f16
            if quant not in ("f16", "bf16"):
                llama_quantize = Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize"
                if not llama_quantize.exists():
                    llama_quantize = Path.home() / "llama.cpp" / "llama-quantize"

                if llama_quantize.exists():
                    quant_out = Path(output_dir) / f"stratos-scorer-{quant}.gguf"

                    qresult = subprocess.run(
                        [str(llama_quantize), str(bf16_path), str(quant_out), quant.upper().replace("_", "_")],
                        capture_output=True, text=True
                    )
                    if qresult.returncode == 0:
                        logger.info(f"✓ Quantized to {quant}: {quant_out}")
                        gguf_path = str(quant_out)
                        # Delete bf16 intermediate to free ~16GB
                        bf16_path.unlink(missing_ok=True)
                        logger.info(f"  Cleaned bf16 intermediate GGUF")
                    else:
                        logger.warning(f"Quantization failed, using bf16: {qresult.stderr[:200]}")
                        gguf_path = str(bf16_path)
                else:
                    logger.info(f"llama-quantize not found — GGUF is bf16 (larger but works)")
                    gguf_path = str(bf16_path)
            else:
                gguf_path = str(bf16_path)
        else:
            logger.warning(f"GGUF conversion failed: {result.stderr[:300]}")
    else:
        logger.info("llama.cpp not found at ~/llama.cpp")
        logger.info("To convert manually:")
        logger.info(f"  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        logger.info(f"  python ~/llama.cpp/convert_hf_to_gguf.py {merged_path} --outfile model.gguf --outtype f16")
    
    return gguf_path or str(merged_path)


def register_with_ollama(gguf_path: str, model_name: str = OUTPUT_MODEL_NAME, version: int = 1):
    """Create Ollama Modelfile and register the fine-tuned model."""
    model_tag = f"{model_name}-v{version}"
    
    # v19: Think-enabled template — model generates <think> blocks naturally.
    # No pre-filled think blocks. Temperature 0.6 required for Qwen3 think mode
    # (greedy decoding causes loops). num_predict 512 for think block + SCORE/REASON.
    modelfile_content = f"""FROM {gguf_path}

TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER num_predict 512
PARAMETER repeat_penalty 1.1
"""
    
    modelfile_path = Path(gguf_path).parent / "Modelfile"
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    logger.info(f"Creating Ollama model: {model_tag}")
    result = subprocess.run(
        ["ollama", "create", model_tag, "-f", str(modelfile_path)],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        logger.info(f"✓ Model registered: {model_tag}")
        logger.info(f"  Test it: ollama run {model_tag}")
        return model_tag
    else:
        logger.error(f"Ollama registration failed: {result.stderr}")
        logger.info(f"Manual registration:")
        logger.info(f"  1. Copy {gguf_path} to a known location")
        logger.info(f"  2. Create a Modelfile with: FROM /path/to/{Path(gguf_path).name}")
        logger.info(f"  3. Run: ollama create {model_tag} -f Modelfile")
        return None


# ═══════════════════════════════════════════════════════════════════
# Google Colab Export (Free GPU fallback)
# ═══════════════════════════════════════════════════════════════════

def generate_colab_notebook(training_file: str, output_path: str):
    """Generate a Colab notebook for training on free T4 GPU."""
    
    # Read training data to embed in notebook
    with open(training_file, 'r') as f:
        training_data = f.read()
    
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU"
        },
        "cells": [
            {
                "cell_type": "markdown",
                "source": ["# STRAT_OS LoRA Fine-Tuning (Colab)\n",
                          "Run this notebook on Google Colab with a **T4 GPU** (free tier).\n",
                          "After training, download the GGUF file and register with Ollama."],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": [
                    "# Install dependencies\n",
                    "!pip install unsloth transformers datasets trl\n",
                    "!pip install --upgrade --no-deps bitsandbytes"
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "source": [
                    "# Write training data\n",
                    "training_data = " + json.dumps(training_data) + "\n",
                    "\n",
                    "with open('training_data.jsonl', 'w') as f:\n",
                    "    f.write(training_data)\n",
                    "\n",
                    "lines = len(training_data.strip().split('\\n'))\n",
                    "print(f'Training examples: {lines}')"
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "source": [
                    "from unsloth import FastLanguageModel\n",
                    "from datasets import load_dataset\n",
                    "from trl import SFTTrainer\n",
                    "from transformers import TrainingArguments\n",
                    "\n",
                    "# Load model\n",
                    "model, tokenizer = FastLanguageModel.from_pretrained(\n",
                    "    model_name='unsloth/Qwen3-1.7B',\n",
                    "    max_seq_length=1024,\n",
                    "    load_in_4bit=True,\n",
                    ")\n",
                    "\n",
                    "# Apply LoRA\n",
                    "model = FastLanguageModel.get_peft_model(\n",
                    "    model, r=16,\n",
                    "    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],\n",
                    "    lora_alpha=32, lora_dropout=0.05,\n",
                    "    bias='none', use_gradient_checkpointing='unsloth',\n",
                    ")\n",
                    "\n",
                    "# Load and format data\n",
                    "dataset = load_dataset('json', data_files='training_data.jsonl', split='train')\n",
                    "def fmt(ex):\n",
                    "    return {'text': tokenizer.apply_chat_template(ex['messages'], tokenize=False, add_generation_prompt=False)}\n",
                    "dataset = dataset.map(fmt)\n",
                    "\n",
                    "# Train\n",
                    "trainer = SFTTrainer(\n",
                    "    model=model, processing_class=tokenizer,\n",
                    "    max_seq_length=1024,\n",
                    "    args=TrainingArguments(\n",
                    "        output_dir='output', per_device_train_batch_size=4,\n",
                    "        gradient_accumulation_steps=2, num_train_epochs=3,\n",
                    "        learning_rate=2e-4, warmup_steps=10, logging_steps=5,\n",
                    "        fp16=True, optim='adamw_8bit', report_to='none',\n",
                    "    ),\n",
                    ")\n",
                    "trainer.train()\n",
                    "print('Training complete!')"
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "source": [
                    "# Export to GGUF\n",
                    "model.save_pretrained_gguf('gguf_out', tokenizer, quantization_method='q4_k_m')\n",
                    "\n",
                    "# Download\n",
                    "import glob\n",
                    "gguf_files = glob.glob('gguf_out/*.gguf')\n",
                    "if gguf_files:\n",
                    "    from google.colab import files\n",
                    "    print(f'Downloading: {gguf_files[0]}')\n",
                    "    files.download(gguf_files[0])\n",
                    "else:\n",
                    "    print('No GGUF file found')"
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "source": [
                    "## After Download\n",
                    "1. Place the `.gguf` file somewhere on your machine\n",
                    "2. Create a `Modelfile`:\n",
                    "```\n",
                    "FROM /path/to/your-model.gguf\n",
                    "PARAMETER temperature 0.3\n",
                    "PARAMETER num_ctx 1024\n",
                    "```\n",
                    "3. Register with Ollama:\n",
                    "```bash\n",
                    "ollama create stratos-scorer-v1 -f Modelfile\n",
                    "```\n",
                    "4. Update `config.yaml`:\n",
                    "```yaml\n",
                    "scoring:\n",
                    "  model: stratos-scorer-v1\n",
                    "```"
                ],
                "metadata": {}
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    logger.info(f"✓ Colab notebook saved to: {output_path}")
    logger.info("  Upload to https://colab.research.google.com and run with T4 GPU")


# ═══════════════════════════════════════════════════════════════════
# Version Management
# ═══════════════════════════════════════════════════════════════════

def get_next_version(output_dir: str) -> int:
    """Get next model version number."""
    versions_file = Path(output_dir) / "versions.json"
    if versions_file.exists():
        with open(versions_file, 'r') as f:
            versions = json.load(f)
        return versions.get("latest", 0) + 1
    return 1


def save_version_info(output_dir: str, version: int, info: Dict):
    """Track model versions."""
    versions_file = Path(output_dir) / "versions.json"
    versions = {}
    if versions_file.exists():
        with open(versions_file, 'r') as f:
            versions = json.load(f)
    
    versions["latest"] = version
    versions[f"v{version}"] = info
    
    with open(versions_file, 'w') as f:
        json.dump(versions, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# Post-Training Cleanup (saves 15-25GB disk)
# ═══════════════════════════════════════════════════════════════

def cleanup_training_artifacts(version_dir: str, base_model: str):
    """Delete intermediate files after GGUF export to reclaim disk space.
    
    Removes:
      - checkpoint-*/     (training checkpoints, ~600MB each)
      - HuggingFace cache for the base model (~8-16GB, only for HF models)
    Preserves (for incremental training):
      - merged/ → moved to current_base/ (the trained model becomes next cycle's base)
    Keeps:
      - *.gguf            (the final model Ollama uses)
      - lora_adapter/     (small, useful for re-export)
      - Modelfile, README.md
    """
    freed = 0
    
    # 1. Delete training checkpoints
    version_path = Path(version_dir)
    for ckpt in version_path.glob("checkpoint-*"):
        if ckpt.is_dir():
            size = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())
            shutil.rmtree(str(ckpt), ignore_errors=True)
            freed += size
            logger.info(f"  Cleaned: {ckpt.name} ({size / 1024 / 1024:.0f} MB)")
    
    # 2. Move merged/ → current_base/ for incremental training
    #    (merged/ may already be deleted by export_gguf_manual for disk space)
    merged_dir = version_path / "merged"
    models_dir = version_path.parent
    current_base = models_dir / "current_base"

    if merged_dir.exists():
        # Delete old current_base first to free disk space
        if current_base.exists():
            old_size = sum(f.stat().st_size for f in current_base.rglob("*") if f.is_file())
            shutil.rmtree(str(current_base), ignore_errors=True)
            freed += old_size
            logger.info(f"  Cleaned: old current_base/ ({old_size / 1024 / 1024:.0f} MB)")
        
        # Move merged → current_base (rename is instant on same filesystem)
        try:
            merged_dir.rename(current_base)
            logger.info(f"  ★ Saved merged model as current_base/ (for incremental training)")
        except OSError:
            # Cross-filesystem: fall back to copy + delete
            shutil.copytree(str(merged_dir), str(current_base))
            size = sum(f.stat().st_size for f in merged_dir.rglob("*") if f.is_file())
            shutil.rmtree(str(merged_dir), ignore_errors=True)
            freed += size
            logger.info(f"  ★ Copied merged → current_base/, cleaned version copy ({size / 1024 / 1024:.0f} MB)")
    
    # 3. Delete HuggingFace cache for this model (only for remote HF models, not local paths)
    if "/" in base_model and not Path(base_model).exists():
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        cache_name = f"models--{base_model.replace('/', '--')}"
        cache_dir = hf_cache / cache_name
        if cache_dir.exists():
            size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            shutil.rmtree(str(cache_dir), ignore_errors=True)
            freed += size
            logger.info(f"  Cleaned: HF cache {cache_name} ({size / 1024 / 1024:.0f} MB)")
    
    # 4. Delete old version directories (keep only current + one previous)
    versions = sorted(models_dir.glob("v*"), key=lambda p: p.name)
    if len(versions) > 2:
        for old_v in versions[:-2]:  # Keep last 2
            size = sum(f.stat().st_size for f in old_v.rglob("*") if f.is_file())
            shutil.rmtree(str(old_v), ignore_errors=True)
            freed += size
            logger.info(f"  Cleaned: old {old_v.name}/ ({size / 1024 / 1024:.0f} MB)")
    
    if freed > 0:
        logger.info(f"  Total disk reclaimed: {freed / 1024 / 1024 / 1024:.1f} GB")
    else:
        logger.info(f"  No intermediate files to clean up")


def main():
    global LORA_R, LORA_ALPHA, BATCH_SIZE
    
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen for STRAT_OS scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_lora.py                     # Auto-detect best method and train
  python train_lora.py --epochs 5          # More training passes
  python train_lora.py --export-colab      # Generate Colab notebook (free GPU)
  python train_lora.py --skip-register     # Don't auto-register with Ollama
        """
    )
    parser.add_argument("--training-data", type=str, default=None,
                        help="Path to training JSONL (default: data/training_data.jsonl)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/models/)")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base model name (auto-detected)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--quant", type=str, default=DEFAULT_QUANT,
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help=f"GGUF quantization (default: {DEFAULT_QUANT})")
    parser.add_argument("--skip-register", action="store_true",
                        help="Don't register with Ollama after training")
    parser.add_argument("--export-colab", action="store_true",
                        help="Generate Google Colab notebook instead of training locally")
    parser.add_argument("--incremental", action="store_true", default=True,
                        help="Incremental training: use previous trained model as base (default)")
    parser.add_argument("--full-retrain", action="store_true",
                        help="Force full retrain from original base model (ignores current_base)")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max training steps (for dry runs, e.g. --max-steps 20)")

    args = parser.parse_args()

    backend_dir = Path(__file__).parent
    training_file = args.training_data or str(backend_dir / "data" / "training_v19_cot.jsonl")
    output_dir = args.output_dir or str(backend_dir / "data" / "models")
    
    # Verify training data exists
    if not Path(training_file).exists():
        logger.error(f"Training data not found: {training_file}")
        logger.error("Run 'python export_training.py' first!")
        sys.exit(1)
    
    # Count examples
    with open(training_file) as f:
        n_examples = sum(1 for _ in f)
    logger.info(f"Training data: {n_examples} examples from {training_file}")
    
    if n_examples < 10:
        logger.warning(f"Only {n_examples} examples — recommend at least 50 for good results.")
        logger.warning("Run more distillation cycles first (python distill.py)")
    
    # Colab export mode
    if args.export_colab:
        colab_path = str(backend_dir / "data" / "StratOS_LoRA_Training.ipynb")
        generate_colab_notebook(training_file, colab_path)
        return
    
    # Local training
    logger.info("=" * 60)
    logger.info("STRAT_OS LoRA Fine-Tuning")
    logger.info("=" * 60)
    
    has_gpu, vram, is_rocm = check_gpu()
    if not has_gpu:
        logger.error("No GPU detected! Options:")
        logger.error("  1. python train_lora.py --export-colab  (free T4 GPU on Google Colab)")
        logger.error("  2. Install CUDA toolkit + PyTorch with CUDA support")
        logger.error("  3. Install ROCm + PyTorch with ROCm support (AMD GPUs)")
        sys.exit(1)
    
    if vram < 4:
        logger.warning(f"Low VRAM ({vram:.1f} GB). Qwen 1.7B needs ~4GB with 4-bit quantization.")
        logger.warning("Consider using Google Colab: python train_lora.py --export-colab")
    
    # Auto-select best model for available VRAM (unless user specified one)
    if not args.base_model:
        auto_hf, auto_ollama, auto_r, auto_bs, auto_desc = auto_select_model(vram, is_rocm)
        logger.info(f"Auto-selected: {auto_desc}")
        logger.info(f"  Model: {auto_hf}, LoRA rank: {auto_r}, Batch size: {auto_bs}")
    else:
        auto_hf, auto_ollama, auto_r, auto_bs = args.base_model, None, LORA_R, BATCH_SIZE
    
    # ── Incremental training: use previous trained model as base ──
    current_base_dir = str(Path(output_dir) / "current_base")
    use_incremental = args.incremental and not args.full_retrain
    
    if use_incremental and Path(current_base_dir).exists():
        # Verify current_base has actual model files
        has_model = (Path(current_base_dir) / "config.json").exists()
        if has_model:
            logger.info(f"  ★ Incremental mode: using previous trained model from {current_base_dir}")
            auto_hf = current_base_dir  # Override base model to our trained version
        else:
            logger.info(f"  current_base/ exists but no model found — falling back to base model")
            use_incremental = False
    elif use_incremental:
        logger.info(f"  No current_base/ found — first training will use original base model")
        use_incremental = False
    
    if args.full_retrain:
        logger.info(f"  Full retrain mode: starting from original base model ({auto_hf})")
    
    # Determine training method
    # Unsloth is CUDA-only; ROCm must use PEFT
    use_unsloth = False
    if not is_rocm:
        use_unsloth = check_unsloth()
    else:
        logger.info("ROCm GPU — using PEFT training (unsloth is CUDA-only)")
    
    use_peft = check_peft() if not use_unsloth else False
    
    if not use_unsloth and not use_peft:
        logger.error("No training framework found! Install one:")
        logger.error("  pip install unsloth          (recommended, faster)")
        logger.error("  pip install peft trl         (fallback)")
        logger.error("  python train_lora.py --export-colab  (use free Colab GPU)")
        sys.exit(1)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    version = get_next_version(output_dir)
    version_dir = str(Path(output_dir) / f"v{version}")
    
    logger.info(f"Training version: v{version}")
    logger.info(f"Method: {'unsloth' if use_unsloth else 'peft'}{' (ROCm)' if is_rocm else ''}")
    logger.info(f"Base model: {auto_hf}")
    logger.info(f"LoRA rank: {auto_r}, Batch size: {auto_bs}")
    logger.info(f"Epochs: {args.epochs}")
    
    # Override global settings with auto-selected values
    LORA_R = auto_r
    LORA_ALPHA = 32              # 2:1 ratio for DoRA (v19)
    BATCH_SIZE = auto_bs
    
    # Train
    gguf_path = None
    if use_unsloth:
        base = auto_hf if not args.base_model else args.base_model
        if not args.base_model:
            base = auto_hf.replace("Qwen/", "unsloth/")
        model, tokenizer, lora_path = train_with_unsloth(
            training_file, version_dir, base, args.epochs, args.lr
        )
        gguf_path = export_gguf_unsloth(model, tokenizer, version_dir, args.quant)
    else:
        base = auto_hf if not args.base_model else args.base_model
        model, tokenizer, lora_path = train_with_peft(
            training_file, version_dir, base, args.epochs, args.lr,
            is_rocm=is_rocm, max_steps=args.max_steps
        )
        # Skip export for dry runs
        if args.max_steps > 0:
            logger.info(f"Dry run complete ({args.max_steps} steps) — skipping GGUF export")
        else:
            result_path = export_gguf_manual(lora_path, version_dir, base, args.quant)
            # If export_gguf_manual found llama.cpp, result_path is a .gguf file
            if result_path.endswith('.gguf'):
                gguf_path = result_path
            else:
                logger.info("PEFT training complete. Install llama.cpp for auto GGUF conversion.")
                logger.info("  Run: git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")

    # Register with Ollama (skip for dry runs)
    model_tag = None
    if gguf_path and not args.skip_register and args.max_steps <= 0:
        model_tag = register_with_ollama(gguf_path, OUTPUT_MODEL_NAME, version)
    
    # ── Auto-update config.yaml to use new model ──
    if model_tag:
        config_path = Path(output_dir).parent / "config.yaml"
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                old_model = config.get("scoring", {}).get("model", "")
                if "scoring" not in config:
                    config["scoring"] = {}
                config["scoring"]["model"] = model_tag
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                logger.info(f"  ✓ config.yaml updated: scoring.model {old_model} → {model_tag}")
            except Exception as e:
                logger.warning(f"  Could not auto-update config.yaml: {e}")
                logger.info(f"  Manual update needed: scoring.model: {model_tag}")
    
    # ── Auto-cleanup: purge intermediate files to save disk ──
    # Skip cleanup for dry runs (preserves HF cache for the full training run)
    if args.max_steps <= 0:
        cleanup_training_artifacts(version_dir, auto_hf)
    else:
        logger.info("Dry run — skipping artifact cleanup")
    
    # Save version info
    save_version_info(output_dir, version, {
        "created": datetime.now().isoformat() if 'datetime' in dir() else "unknown",
        "base_model": auto_hf,
        "training_mode": "incremental" if use_incremental else "full_retrain",
        "adapter_type": "dora",
        "lora_rank": auto_r,
        "lora_alpha": auto_r,
        "batch_size": auto_bs,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "max_seq_length": MAX_SEQ_LENGTH,
        "examples": n_examples,
        "quant": args.quant,
        "gguf_path": gguf_path or "",
        "ollama_tag": model_tag or "",
    })
    
    # Final instructions
    logger.info("\n" + "=" * 60)
    logger.info("DONE! Next steps:")
    logger.info("=" * 60)
    if model_tag:
        logger.info(f"  1. Test: ollama run {model_tag}")
        logger.info(f"  2. config.yaml already updated automatically")
        logger.info(f"  3. Restart STRAT_OS and run a scan")
    else:
        logger.info(f"  1. Convert to GGUF if needed")
        logger.info(f"  2. Register: ollama create {OUTPUT_MODEL_NAME}-v{version} -f Modelfile")
        logger.info(f"  3. Update config.yaml scoring model")


if __name__ == "__main__":
    # Need datetime for version info
    from datetime import datetime
    main()
