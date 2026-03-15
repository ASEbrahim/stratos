#!/usr/bin/env python3
"""Merge DoRA adapter into base model, export as Q8_0 GGUF, register with Ollama."""

import subprocess
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("MERGE")

PIPELINE_DIR = Path(__file__).parent.parent
ADAPTER_PATH = PIPELINE_DIR / "training_output" / "adapter"
MERGED_PATH = PIPELINE_DIR / "merged_model"
GGUF_PATH = PIPELINE_DIR / "stratos-rp-v2.gguf"
MODELFILE_PATH = PIPELINE_DIR / "Modelfile.rp_v2"
BASE_MODEL_ID = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"


def merge_adapter():
    logger.info("Merging adapter into base model...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Disable caching_allocator_warmup (OOMs on 24GB card with 18GB model)
    import transformers.modeling_utils as _mu
    _mu.caching_allocator_warmup = lambda *a, **kw: None

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True)

    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    model = model.merge_and_unload()

    MERGED_PATH.mkdir(parents=True, exist_ok=True)
    # Save in 4GB shards to avoid 2x peak RAM (18GB model + 18GB write buffer)
    model.save_pretrained(str(MERGED_PATH), max_shard_size="4GB")
    tokenizer.save_pretrained(str(MERGED_PATH))

    # Free RAM before GGUF export
    del model, base_model
    import gc; gc.collect()
    logger.info(f"Merged model saved: {MERGED_PATH}")


def export_gguf():
    logger.info("Exporting to GGUF Q8_0...")
    llama_cpp = Path.home() / "llama.cpp"
    if not llama_cpp.exists():
        logger.info("Cloning llama.cpp...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", str(llama_cpp)], check=True)

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        logger.error("Cannot find convert_hf_to_gguf.py")
        sys.exit(1)

    f16_path = PIPELINE_DIR / "stratos-rp-v2-f16.gguf"
    subprocess.run([sys.executable, str(convert_script), str(MERGED_PATH),
                    "--outfile", str(f16_path), "--outtype", "f16"], check=True)

    quantize = llama_cpp / "build" / "bin" / "llama-quantize"
    if quantize.exists():
        subprocess.run([str(quantize), str(f16_path), str(GGUF_PATH), "Q8_0"], check=True)
        f16_path.unlink()
    else:
        logger.warning("llama-quantize not built. Using F16 directly.")
        f16_path.rename(GGUF_PATH)

    logger.info(f"GGUF exported: {GGUF_PATH} ({GGUF_PATH.stat().st_size / 1024**3:.1f} GB)")


def register_ollama():
    modelfile_content = f"""FROM {GGUF_PATH}

PARAMETER temperature 0.85
PARAMETER top_p 0.95
PARAMETER num_predict 4000
PARAMETER num_ctx 8192

TEMPLATE \"\"\"{{{{- range .Messages }}}}
<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}
<|im_end|>
{{{{ end }}}}<|im_start|>assistant
\"\"\"

SYSTEM \"\"\"You are an immersive roleplay partner.

STYLE MATCHING:
- MIRROR the user's length. One-liner input = one-liner response. Paragraph input = paragraph response.
- Count the user's sentences. Your response should have a SIMILAR number of sentences.
- Match the FORMAT: chat gets chat, asterisk actions get asterisk actions, prose gets prose.

PACING AND TENSION:
- In slow-burn scenes, let tension build through what is NOT said.
- As intimacy increases, your responses should get SHORTER, not longer. The most intense moments need the fewest words.
- Let the character's defenses genuinely erode across turns.
- Physical detail: small and specific over grand gestures.

CHARACTER RULES:
- Stay in character. Never break character or add OOC unless asked.
- Your character has AGENCY. If the user writes your character doing something out of character, acknowledge the moment but redirect to stay true to the character's established personality.
- When multiple NPCs are present, give each named NPC DISTINCT dialogue lines.
- Remember and reference earlier conversation details.
- Show emotional depth through action and subtext, not just dialogue.\"\"\"
"""
    with open(MODELFILE_PATH, "w") as f:
        f.write(modelfile_content)

    logger.info("Registering with Ollama as stratos-rp-v2...")
    subprocess.run(["ollama", "create", "stratos-rp-v2", "-f", str(MODELFILE_PATH)], check=True)
    logger.info("Model registered: stratos-rp-v2")


def main():
    merge_adapter()
    export_gguf()
    register_ollama()
    logger.info("\nPhase 3a complete. Run baseline tests against stratos-rp-v2.")


if __name__ == "__main__":
    main()
