#!/usr/bin/env python3
"""
Qwen3.5-9B Training Feasibility Test
=====================================
Tests whether we can:
1. Load Qwen3.5-9B in bf16 on ROCm
2. Apply DoRA/LoRA adapters
3. Run a few training steps without errors
4. Export to GGUF

This is a quick sanity check before committing to a full 12-hour training run.
"""

import gc
import json
import logging
import os
import subprocess
import sys
import time
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FEASIBILITY")

# Suppress Qwen3.5 vision warnings for text-only training
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

RESULTS = {}

def log_vram():
    """Log current VRAM usage."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "Used" in line and "GPU[0]" in line:
                used_bytes = int(line.split(":")[-1].strip())
                used_gb = used_bytes / 1e9
                logger.info(f"VRAM used: {used_gb:.1f} GB")
                return used_gb
    except Exception:
        pass
    return -1


def test_1_model_loading():
    """Test: Can we load Qwen3.5-9B on ROCm?"""
    logger.info("=" * 60)
    logger.info("TEST 1: Model Loading (Qwen3.5-9B bf16 on ROCm)")
    logger.info("=" * 60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        model_id = "Qwen/Qwen3.5-9B"

        # First try loading config
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        logger.info(f"Config loaded: arch={config.architectures}, type={config.model_type}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

        # Load model — text backbone only
        logger.info("Loading model in bf16 (this may take a few minutes)...")
        log_vram()

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        vram = log_vram()
        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        logger.info(f"Model loaded: {param_count:.2f}B parameters")

        RESULTS["test_1"] = {
            "status": "PASS",
            "params_b": round(param_count, 2),
            "vram_gb": round(vram, 1),
            "arch": str(config.architectures),
        }
        return model, tokenizer

    except Exception as e:
        logger.error(f"TEST 1 FAILED: {e}")
        RESULTS["test_1"] = {"status": "FAIL", "error": str(e)}
        return None, None


def test_2_lora_application(model):
    """Test: Can we apply DoRA/LoRA to the text backbone?"""
    logger.info("=" * 60)
    logger.info("TEST 2: LoRA/DoRA Application")
    logger.info("=" * 60)

    try:
        from peft import LoraConfig, get_peft_model, TaskType

        # Target the text backbone projection layers
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        # Try DoRA first (what V2 used)
        try:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                use_dora=True,
            )
            model = get_peft_model(model, lora_config)
            adapter_type = "DoRA"
        except Exception as dora_err:
            logger.warning(f"DoRA failed ({dora_err}), falling back to standard LoRA")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                use_dora=False,
            )
            model = get_peft_model(model, lora_config)
            adapter_type = "LoRA"

        # Fix PEFT meta device issue (from V2 Bug #3)
        if hasattr(model, "hf_device_map"):
            del model.hf_device_map
        model = model.to("cuda:0")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"{adapter_type} applied: {trainable/1e6:.1f}M trainable / {total/1e9:.2f}B total ({100*trainable/total:.2f}%)")

        vram = log_vram()
        RESULTS["test_2"] = {
            "status": "PASS",
            "adapter_type": adapter_type,
            "trainable_m": round(trainable / 1e6, 1),
            "vram_gb": round(vram, 1),
        }
        return model

    except Exception as e:
        logger.error(f"TEST 2 FAILED: {e}")
        RESULTS["test_2"] = {"status": "FAIL", "error": str(e)}
        return None


def test_3_training_steps(model, tokenizer):
    """Test: Can we run 5 training steps without NaN or errors?"""
    logger.info("=" * 60)
    logger.info("TEST 3: Training Steps (5 steps)")
    logger.info("=" * 60)

    try:
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

        # Create a tiny synthetic dataset matching our scoring format
        examples = []
        for i in range(20):
            messages = [
                {"role": "system", "content": f"You are a relevance scorer for a Software Engineer in San Francisco.\nTracked companies: None specified\nTracked institutions: None specified\nTracked interests: None specified\nTracked industries: None specified\nIMPORTANT: User is an experienced professional.\n\nScore each article 0.0-10.0:\n9-10: Directly actionable\n7-8.9: Highly relevant\n5-6.9: Somewhat relevant\n0-4.9: Not relevant\n\nReply ONLY with: SCORE: X.X | REASON: brief explanation"},
                {"role": "user", "content": f"Score this article:\nCategory: Tech\nKeywords: AI, ML\nTitle: Test Article {i}\nContent: This is test content about technology."},
                {"role": "assistant", "content": f"SCORE: {i % 10}.0 | REASON: Test reason {i}"},
            ]
            examples.append({"messages": messages})

        dataset = Dataset.from_list(examples)

        # Apply chat template
        def format_fn(example):
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        dataset = dataset.map(format_fn)

        # Check what the formatted text looks like
        logger.info(f"Sample formatted text (first 200 chars): {dataset[0]['text'][:200]}")

        # Setup trainer
        training_args = SFTConfig(
            output_dir="/tmp/qwen35_feasibility",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=5,
            learning_rate=1e-5,
            logging_steps=1,
            save_steps=999,  # Don't save checkpoints
            bf16=True,
            max_seq_length=512,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_text_field="text",
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        logger.info("Starting 5 training steps...")
        log_vram()
        train_result = trainer.train()

        # Check for NaN
        final_loss = train_result.training_loss
        logger.info(f"Training complete: loss={final_loss:.4f}")

        if final_loss != final_loss:  # NaN check
            raise ValueError(f"NaN loss detected: {final_loss}")

        vram = log_vram()
        RESULTS["test_3"] = {
            "status": "PASS",
            "final_loss": round(final_loss, 4),
            "vram_gb": round(vram, 1),
        }
        return True

    except Exception as e:
        logger.error(f"TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        RESULTS["test_3"] = {"status": "FAIL", "error": str(e)}
        return False


def test_4_gguf_check():
    """Test: Check if GGUF export path exists."""
    logger.info("=" * 60)
    logger.info("TEST 4: GGUF Export Path Check")
    logger.info("=" * 60)

    try:
        # Check if llama.cpp conversion tools exist
        import shutil

        # Method 1: llama-cpp-python or llama.cpp convert script
        has_llama_cpp = shutil.which("llama-quantize") is not None

        # Method 2: transformers GGUF export (newer versions)
        has_transformers_gguf = False
        try:
            from transformers import GGUFModel
            has_transformers_gguf = True
        except ImportError:
            pass

        # Method 3: Ollama can create from safetensors directly
        has_ollama_create = shutil.which("ollama") is not None

        logger.info(f"llama-quantize: {'found' if has_llama_cpp else 'not found'}")
        logger.info(f"transformers GGUF: {'available' if has_transformers_gguf else 'not available'}")
        logger.info(f"ollama create: {'available' if has_ollama_create else 'not available'}")

        # For Qwen3.5, we need either llama.cpp or Ollama's built-in converter
        can_export = has_llama_cpp or has_ollama_create

        RESULTS["test_4"] = {
            "status": "PASS" if can_export else "WARN",
            "llama_cpp": has_llama_cpp,
            "ollama": has_ollama_create,
            "method": "ollama create" if has_ollama_create else ("llama-quantize" if has_llama_cpp else "none"),
        }
        return can_export

    except Exception as e:
        logger.error(f"TEST 4 FAILED: {e}")
        RESULTS["test_4"] = {"status": "FAIL", "error": str(e)}
        return False


if __name__ == "__main__":
    logger.info("Qwen3.5-9B Training Feasibility Test")
    logger.info("=" * 60)

    # First, stop Ollama models to free VRAM
    logger.info("Unloading Ollama models to free VRAM...")
    try:
        import requests
        # Send keep_alive: 0 to unload all models
        for model_name in ["stratos-scorer-v2", "qwen3:30b-a3b", "qwen3:14b", "qwen3.5:9b"]:
            try:
                requests.post("http://localhost:11434/api/generate",
                    json={"model": model_name, "keep_alive": 0}, timeout=10)
            except:
                pass
        time.sleep(3)
    except:
        pass

    log_vram()

    # Run tests
    model, tokenizer = test_1_model_loading()

    if model is not None:
        model = test_2_lora_application(model)

    if model is not None:
        test_3_training_steps(model, tokenizer)

    # Free GPU memory before GGUF check
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    test_4_gguf_check()

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("FEASIBILITY TEST SUMMARY")
    logger.info("=" * 60)

    all_pass = True
    for test_name, result in RESULTS.items():
        status = result["status"]
        emoji = "PASS" if status == "PASS" else ("WARN" if status == "WARN" else "FAIL")
        logger.info(f"  {test_name}: {emoji} — {json.dumps({k:v for k,v in result.items() if k != 'status'})}")
        if status == "FAIL":
            all_pass = False

    logger.info("")
    if all_pass:
        logger.info("VERDICT: Qwen3.5-9B training is FEASIBLE. Proceed with full training.")
    else:
        logger.info("VERDICT: Qwen3.5-9B training has BLOCKERS. Fall back to Qwen3-8B V3.")

    # Save results
    with open("/tmp/qwen35_feasibility.json", "w") as f:
        json.dump(RESULTS, f, indent=2)

    sys.exit(0 if all_pass else 1)
