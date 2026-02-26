#!/usr/bin/env python3
"""
V2 Pipeline — Phase 1: Merge V1 DoRA adapters into Qwen3-8B base.
Produces a single merged model for V2 fine-tuning.
"""

import gc
import os
import sys
from pathlib import Path

import torch

os.chdir(Path(__file__).parent.parent.parent)

BASE_MODEL = "Qwen/Qwen3-8B"
V1_ADAPTER = "data/models/V1/lora_adapter"
OUTPUT_DIR = Path("data/v2_pipeline/v1_merged_base")

print("=" * 80)
print("V2 PIPELINE — PHASE 1: MERGE V1 INTO BASE MODEL")
print("=" * 80)

# Step 1: Load base model in bf16
print(f"\n[1/5] Loading base model: {BASE_MODEL}")
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cpu",  # Keep on CPU — we just need to merge, not train
)
print(f"  Base model loaded. Parameters: {model.num_parameters():,}")

# Step 2: Load V1 DoRA adapters
print(f"\n[2/5] Loading V1 DoRA adapters: {V1_ADAPTER}")
from peft import PeftModel

model = PeftModel.from_pretrained(model, V1_ADAPTER)
print(f"  Adapters loaded. Adapter type: DoRA rank 16")

# Step 3: Merge and unload
print(f"\n[3/5] Merging adapters into base weights...")
model = model.merge_and_unload()
print(f"  Merged. Parameters: {model.num_parameters():,}")

# Step 4: Save merged model
print(f"\n[4/5] Saving merged model to: {OUTPUT_DIR}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"  Saved.")

# Step 5: Verify
size_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
size_gb = size_bytes / (1024 ** 3)
print(f"\n[5/5] Verification:")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Total size: {size_gb:.1f} GB")
files = sorted(OUTPUT_DIR.iterdir())
for f in files:
    print(f"  {f.name}: {f.stat().st_size / (1024**2):.1f} MB")

if size_gb < 10:
    print(f"\n!! WARNING: Merged model is only {size_gb:.1f} GB — expected ~16 GB")
elif size_gb > 20:
    print(f"\n!! WARNING: Merged model is {size_gb:.1f} GB — larger than expected")
else:
    print(f"\nPASS: Size looks correct for Qwen3-8B bf16")

# Verify it loads
print(f"\nVerifying merged model loads correctly...")
del model
gc.collect()
test_model = AutoModelForCausalLM.from_pretrained(
    str(OUTPUT_DIR), torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu"
)
print(f"  Loaded successfully. Parameters: {test_model.num_parameters():,}")
del test_model
gc.collect()

print(f"\n{'=' * 80}")
print("PHASE 1 COMPLETE — Merged model saved")
print(f"{'=' * 80}")
