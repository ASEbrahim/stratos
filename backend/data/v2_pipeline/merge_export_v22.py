#!/usr/bin/env python3
"""Merge V2.2 DoRA adapters into base model and export to GGUF.

Lightweight script — no training overhead, just merge + save + convert.
"""
import gc
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = Path(__file__).parent / "qwen35_9b_base"
ADAPTER_PATH = Path(__file__).parent / "training_output" / "final_checkpoint"
OUTPUT_DIR = Path(__file__).parent / "training_output"
MERGED_PATH = OUTPUT_DIR / "merged_v2"

def main():
    print(f"Loading base model from {BASE_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL),
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Stay on CPU to avoid VRAM issues
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_PATH))

    print(f"Loading DoRA adapters from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))

    print("Merging adapters into base weights...")
    model = model.merge_and_unload()
    model.eval()

    print(f"Saving merged model to {MERGED_PATH}...")
    MERGED_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_PATH), max_shard_size="4GB")
    tokenizer.save_pretrained(str(MERGED_PATH))

    # Free RAM before GGUF conversion
    del model
    gc.collect()
    print("Merged model saved. RAM freed.")

    # GGUF export
    llama_convert = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"
    bf16_path = OUTPUT_DIR / "v2_scorer_bf16.gguf"

    print("Converting to GGUF (bf16)...")
    result = subprocess.run(
        [sys.executable, str(llama_convert), str(MERGED_PATH),
         "--outfile", str(bf16_path), "--outtype", "bf16"],
        timeout=1800
    )
    if result.returncode != 0 or not bf16_path.exists():
        print("GGUF conversion failed!")
        return

    size_mb = bf16_path.stat().st_size // (1024 * 1024)
    print(f"BF16 GGUF: {bf16_path} ({size_mb} MB)")

    # Quantize to q8_0
    llama_quantize = Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize"
    if not llama_quantize.exists():
        llama_quantize = Path.home() / "llama.cpp" / "llama-quantize"

    if llama_quantize.exists():
        q8_path = OUTPUT_DIR / "v2_scorer.gguf"
        print("Quantizing to Q8_0...")
        qresult = subprocess.run(
            [str(llama_quantize), str(bf16_path), str(q8_path), "Q8_0"],
            timeout=1800
        )
        if qresult.returncode == 0 and q8_path.exists():
            q8_mb = q8_path.stat().st_size // (1024 * 1024)
            print(f"Q8_0 GGUF: {q8_path} ({q8_mb} MB)")
            bf16_path.unlink(missing_ok=True)
            print("Cleaned bf16 intermediate.")
        else:
            print("Quantization failed, keeping bf16.")
    else:
        print("llama-quantize not found, keeping bf16 GGUF.")

    print("Done!")

if __name__ == "__main__":
    main()
