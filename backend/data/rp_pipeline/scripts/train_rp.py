#!/usr/bin/env python3
"""
RP Model DoRA Fine-Tuning.

Fine-tunes the abliterated Qwen 3.5-9B for roleplay using DoRA.

Usage:
    python3 data/rp_pipeline/scripts/train_rp.py
    python3 data/rp_pipeline/scripts/train_rp.py --epochs 2 --lr 5e-6
    python3 data/rp_pipeline/scripts/train_rp.py --max-steps 10  # Dry run
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("RP_TRAIN")

BASE_MODEL_ID = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"
PIPELINE_DIR = Path(__file__).parent.parent
TRAINING_DATA = PIPELINE_DIR / "training_data" / "final_training_data.jsonl"
OUTPUT_DIR = PIPELINE_DIR / "training_output"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 5e-6
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16
EPOCHS = 1

# Conservative: attention only to preserve base capabilities
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_training_data(path: Path) -> list:
    conversations = []
    with open(path) as f:
        for line in f:
            conv = json.loads(line)
            messages = conv.get("messages", [])
            if len(messages) >= 3:
                conversations.append({"messages": messages})
    return conversations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=LORA_R)
    parser.add_argument("--seq-length", type=int, default=MAX_SEQ_LENGTH)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RP Model DoRA Fine-Tuning")
    logger.info("=" * 60)

    import torch
    if not torch.cuda.is_available():
        logger.error("No GPU detected.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    os.environ["ROCR_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.pop("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", None)

    if not TRAINING_DATA.exists():
        logger.error(f"Training data not found: {TRAINING_DATA}")
        sys.exit(1)

    conversations = load_training_data(TRAINING_DATA)
    logger.info(f"Training conversations: {len(conversations)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    logger.info(f"Loading base model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        use_dora=True,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # PEFT meta device fix (from scorer V2.2 experience)
    if hasattr(model, 'hf_device_map'):
        del model.hf_device_map
    model = model.to("cuda:0")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total:,} total, {trainable:,} trainable ({100*trainable/total:.2f}%)")

    from trl import SFTTrainer, SFTConfig

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        max_length=args.seq_length,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=args.max_steps,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=conversations,
        args=training_args,
    )

    logger.info(f"\nStarting training:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    logger.info(f"  Seq length: {args.seq_length}")
    logger.info(f"  DoRA rank: {args.rank}")
    logger.info(f"  Target modules: {TARGET_MODULES}")

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start

    logger.info(f"\nTraining complete in {elapsed/3600:.1f} hours")
    logger.info(f"  Final loss: {result.training_loss:.4f}")

    adapter_path = OUTPUT_DIR / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info(f"Adapter saved: {adapter_path}")

    logger.info("\nPhase 2 complete. Next: merge adapter -> GGUF export -> evaluation.")


if __name__ == "__main__":
    main()
