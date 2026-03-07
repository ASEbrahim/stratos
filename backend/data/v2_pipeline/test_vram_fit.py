#!/usr/bin/env python3
"""
Test if Qwen3.5-9B fits in 24GB VRAM for DoRA training.

Loads the model with the same config as V2 training:
  - DoRA rank 16, batch_size=2, gradient_accumulation=8
  - max_seq_length=1024 (fallback: 512)
  - Runs ONE training step on a tiny dummy dataset

Reports: peak VRAM usage, whether it OOMs, time per step.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

# ROCm setup — MUST be before any torch import
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
# DO NOT set AOTRITON — causes NaN gradients on ROCm 6.2

MODEL_DIR = Path(__file__).parent / "qwen35_9b_base"
MAX_SEQ_LENGTH = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
BATCH_SIZE = 2

print(f"=" * 70)
print(f"VRAM FIT TEST: Qwen3.5-9B + DoRA rank 16")
print(f"  max_seq_length={MAX_SEQ_LENGTH}, batch_size={BATCH_SIZE}")
print(f"  Model dir: {MODEL_DIR}")
print(f"=" * 70)

if not MODEL_DIR.exists():
    print(f"ERROR: Model directory not found: {MODEL_DIR}")
    print("Run: huggingface-cli download Qwen/Qwen3.5-9B-Base --local-dir qwen35_9b_base")
    sys.exit(1)

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total VRAM: {total_vram:.1f} GB")

def report_vram(label):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  [{label}] Allocated: {alloc:.2f}GB, Reserved: {reserved:.2f}GB, Peak: {peak:.2f}GB")

# Step 1: Load model
print("\n--- Loading model ---")
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
report_vram("after tokenizer")

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
)
load_time = time.time() - t0
print(f"Model loaded in {load_time:.1f}s")
report_vram("after model load")

# Step 2: Apply DoRA
print("\n--- Applying DoRA (rank 16) ---")
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_dora=True,
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# PEFT meta device fix (needed on ROCm)
if hasattr(model, 'hf_device_map'):
    del model.hf_device_map
model = model.to("cuda:0")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")
report_vram("after DoRA")

# Step 3: Enable gradient checkpointing
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
print("Gradient checkpointing enabled")

# Step 4: Create dummy dataset
print("\n--- Creating dummy training data ---")
dummy_examples = []
for i in range(5):
    system = f"You are a relevance scorer for a Software Engineer in San Francisco.\nUser context: ML infrastructure\nTracked companies: Google, Meta\nTracked interests: machine learning, distributed systems\nLANGUAGE: Articles must be in English.\n\nScore each article 0.0-10.0:\n9-10: Directly actionable\n7-8.9: Highly relevant\n5-6.9: Somewhat relevant\n0-4.9: Not relevant\n\nReply ONLY with: SCORE: X.X | REASON: brief explanation"
    user = f"Score this article:\nCategory: tech\nKeywords: machine learning\nTitle: New GPU architecture announced by NVIDIA\nContent: NVIDIA today unveiled its next-generation GPU architecture designed for large language model training."
    assistant = f"SCORE: {7.0 + i * 0.5:.1f} | REASON: Highly relevant to ML infrastructure work"

    text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"
    tokens = tokenizer(text, return_tensors="pt", max_length=MAX_SEQ_LENGTH, truncation=True, padding="max_length")
    dummy_examples.append({
        "input_ids": tokens["input_ids"].squeeze(),
        "attention_mask": tokens["attention_mask"].squeeze(),
        "labels": tokens["input_ids"].squeeze().clone(),
    })

from torch.utils.data import DataLoader, Dataset

class DummyDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

dataset = DummyDataset(dummy_examples)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Step 5: Run ONE training step
print(f"\n--- Running 1 training step (batch_size={BATCH_SIZE}) ---")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

torch.cuda.reset_peak_memory_stats()

try:
    model.train()
    batch = next(iter(dataloader))
    batch = {k: v.to("cuda:0") for k, v in batch.items()}

    t0 = time.time()
    outputs = model(**batch)
    loss = outputs.loss
    print(f"  Forward pass: loss={loss.item():.4f}")
    report_vram("after forward")

    loss.backward()
    print(f"  Backward pass complete")
    report_vram("after backward")

    optimizer.step()
    optimizer.zero_grad()
    step_time = time.time() - t0
    print(f"  Optimizer step complete")
    report_vram("after optimizer step")

    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\n{'=' * 70}")
    print(f"RESULT: SUCCESS")
    print(f"  Peak VRAM: {peak_vram:.2f} GB / {total_vram:.1f} GB ({100*peak_vram/total_vram:.0f}%)")
    print(f"  Step time: {step_time:.2f}s")
    print(f"  Headroom: {total_vram - peak_vram:.2f} GB")
    print(f"  Max seq length: {MAX_SEQ_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE}")

    if peak_vram > total_vram * 0.95:
        print(f"  WARNING: Very tight VRAM fit ({100*peak_vram/total_vram:.0f}%). May OOM with longer sequences.")
    elif peak_vram > total_vram * 0.85:
        print(f"  NOTE: Fits but tight. Monitor during full training.")
    else:
        print(f"  Good VRAM headroom for full training run.")
    print(f"{'=' * 70}")

except torch.cuda.OutOfMemoryError as e:
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n{'=' * 70}")
    print(f"RESULT: OOM")
    print(f"  Peak VRAM before OOM: {peak_vram:.2f} GB / {total_vram:.1f} GB")
    print(f"  Max seq length: {MAX_SEQ_LENGTH}")
    print(f"  Batch size: {BATCH_SIZE}")
    if MAX_SEQ_LENGTH > 512:
        print(f"  Try: python3 {__file__} 512")
    else:
        print(f"  Need RunPod A100 80GB (~$1.74/hr, ~$8-10 total)")
    print(f"{'=' * 70}")
    sys.exit(1)

except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
