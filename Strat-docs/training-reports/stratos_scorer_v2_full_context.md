# StratOS Scorer — Complete Training Context for Claude Code

**Purpose:** This document contains EVERYTHING a Claude Code instance needs to understand the scorer training pipeline, what made V2 successful, what V2.1 is doing wrong, and how to fix it. Read this ENTIRE document before making any changes.

**Date:** March 6, 2026
**Author:** Compiled from ~20 conversation threads spanning Feb 15 – Mar 6, 2026

---

## PART 1: THE NAMING CONFUSION — READ THIS FIRST

There are TWO different version naming systems that overlap confusingly:

### Internal iteration names (v1–v19)
These were pipeline development iterations, NOT production models:
- **v1–v14**: Infrastructure development, first training attempts
- **v15**: Loss masking bug discovered (75% gradient wasted on prompt tokens)
- **v16**: Wrong TRL parameter (`completion_only_loss` vs `assistant_only_loss`)
- **v17**: Disk full during training
- **v18**: Loss masking fixed, answer-only format → PSR 0%, model outputs flat 8.5. **Critical finding: answer-only training cannot teach profile-aware reasoning.**
- **v19**: First CoT attempt (PRISM framework), 5,679 examples from Sonnet 4.6 ($17). This became the "V1 scorer."

### Production scorer names (V1, V2, V2.1, V3)
- **V1 scorer** = The result of v19 training. 90.7% direction accuracy, 1.553 MAE, 39.7% PSR, profile spread 1.04. Empty think blocks. Grade: C+/B-.
- **V2 scorer** = The successful production model. 98.1% direction accuracy, 0.393 MAE, 0.750 Spearman ρ, 24.4% PSR (vs Opus 26.4%), profile spread 7.90 (vs Opus 8.20). Zero parse failures.
- **V2.1** = Current training run. Attempting to fix V2's post-deployment degradation. **This is the contaminated run.**
- **V3** = Merged V1+V2 data (24,479 samples). Training launched Feb 24, evaluation pending. May or may not improve over V2.

**The v0.1–v19 iterations taught us HOW to build the pipeline. The actual V2 production model was trained fresh using the lessons from all 19 failures.**

---

## PART 2: WHAT MADE V2 SUCCESSFUL — THE EXACT PIPELINE

### Stage 1: Profile Generation
- 30 profiles total (10 existing V1 profiles + 20 new)
- Diversity requirements: non-English markets (Japan, Brazil, Saudi, France), blue-collar/trades (electrician, HVAC, pipeline welder, marine mechanic), creative industries (game dev, UX designer, filmmaker), healthcare specialties, pure hobbyists (crypto + whisky + F1), contradictory interests (lawyer + AI ethics), generalists
- 20 countries represented
- Both career-focused professionals AND interest-driven users (gamers, retirees, hobbyists)
- Claude generated them, Ahmad screened for gaps

### Stage 2: Article Collection
- Each of 30 profiles generated 15-20 search queries
- Sources: DuckDuckGo + Serper (parallel)
- Result: 813 unique articles after dedup
- **Critical:** Articles came from the StratOS pipeline itself — real messy search results with garbage, duplicates, clickbait — matching the actual distribution the production scorer faces. NOT curated RSS.

### Stage 3: Scoring via Batch API
- Every article scored against every profile: 813 articles × ~25-30 profiles = 20,550 scored examples
- **Teacher model: Claude Opus** via Anthropic Batch API
- Cost: ~$52
- Opus system prompt instructed step-by-step reasoning (PRISM-style):
  1. Identify specific profile elements (role, location, interests)
  2. Explain WHY those elements make the article more/less relevant to THIS user
  3. Consider direct AND indirect relevance
  4. Forbidden 5.0 — must decide if positively relevant (6.0+) or noise (4.0-)
- Output fields per example: `score`, `think_text` (full reasoning, ~1,133 chars avg), `reason` (short 1-sentence, ~154 chars avg)
- Validation: reject examples where think block < 20 tokens, score outside 0-10, malformed output
- All 20,550 examples were Opus-scored. **No agent data. No Claude Code data. Pure Opus quality.**

### Stage 4: Data Preparation
**THIS IS THE MOST IMPORTANT PART — GET THIS RIGHT:**

- **Training format**: ChatML via `tokenizer.apply_chat_template()` with **`enable_thinking=False`**
- **Assistant message content**: `SCORE: X.X | REASON: {short reason field (~154 chars)}`
- **The think_text was intentionally DISCARDED from training data.** It was generated and paid for to ensure Opus actually reasoned deeply before scoring, but the model only learned from the concise output.
- **NO `<think>` blocks in training data.** The `enable_thinking=False` flag prevents any think block insertion during tokenization.
- **Loss weighting**: WeightedRandomSampler ONLY (not per-sample loss weights — per-sample was dropped after debugging showed it fought newer TRL/Transformers versions)
- Weight bands:
  - noise (0-2.5): 0.5x sampler weight
  - tangential (2.5-4.0): 1.0x
  - moderate (4.5-6.5): 1.5x
  - high (7.0-8.0): 2.0x
  - critical (8.5-10.0): 3.0x
- **Contrastive pairs**: 25,243 automatically extracted (same article, different profiles, score gap ≥ 3.0 points). 30 profiles per article = C(30,2) = 435 possible pairs per article.
- **Train/eval split**: 18,502 training / 2,048 eval — stratified by score bucket AND profile
- **No curriculum sorting** in final V2 run (was planned but dropped for simplicity)

### Stage 5: Base Model Preparation (Merge-First Approach)
1. Load Qwen3-8B base model in bf16
2. Load V1's DoRA adapters
3. `model.merge_and_unload()` → saved to `v1_merged_base/` (~16GB)
4. This merged model already understood PRISM format, score ranges, basic article comprehension
5. **Fresh DoRA adapters trained on top** — do NOT stack adapters on adapters (causes optimization instability)

### Stage 6: Training Configuration
```
Method:           DoRA (LoRA with magnitude decomposition, use_dora=True)
Rank:             16
Alpha:            32 (note: V2 used alpha=32, not alpha=16)
Dropout:          0.05
Target modules:   q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Batch size:       2 (fell back from 4 due to VRAM)
Gradient accum:   8 (effective batch size = 16)
Learning rate:    1e-5 with cosine schedule, 5% warmup
Epochs:           1 (3.6x more data than V1, single pass sufficient)
Max seq length:   1024
Loss:             completion_only (system + user tokens masked)
Optimizer:        AdamW, bf16
Grad checkpoint:  Yes (use_reentrant=False)
Hardware:         AMD RX 7900 XTX (24GB VRAM), ROCm 6.2
Steps:            ~1,157 total, ~34s/step, ~11 hours
```

**Critical environment:**
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
# DO NOT SET: TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1  (causes NaN with DoRA+gradient checkpointing+bf16)
```

**Tokenizer setup:**
```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
# CRITICAL: Qwen3 silently changed eos_token from <|im_end|> to <|endoftext|>
# Must explicitly set:
tokenizer.eos_token = '<|im_end|>'
tokenizer.eos_token_id = 151645
tokenizer.pad_token_id = 151643  # <|endoftext|>
```

**Model loading:**
```python
model = AutoModelForCausalLM.from_pretrained(
    "path/to/v1_merged_base",
    device_map="auto",
    attn_implementation="sdpa",  # NOT flash_attention_2
    torch_dtype=torch.bfloat16,
)
# PEFT meta device fix: After get_peft_model(), delete hf_device_map and force .to("cuda:0")
```

### Stage 7: Post-Training
1. DoRA adapters merged with base model
2. Exported to GGUF **Q8_0** via llama.cpp (~8.2GB). **Q8_0 is mandatory for scoring tasks** — Q4_K_M introduces noticeable scoring errors because fine-tuned weight deltas are fragile under aggressive quantization.
3. Registered in Ollama as `stratos-scorer-v2`

### Stage 8: Modelfile (CRITICAL — USE THIS EXACT TEMPLATE)
```
FROM path/to/v2_scorer.gguf

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 1024
PARAMETER num_predict 256
PARAMETER stop <|im_end|>
PARAMETER stop <|endoftext|>
PARAMETER stop <|im_start|>
```

**DO NOT include `/nothink` in the Modelfile** — it causes 41% parse failures from degenerate token loops (`/soft`, `/nostart`, `/assistant`...). Let Qwen3 produce natural empty `<think>\n</think>` blocks which get stripped by the parser.

**DO NOT include a SYSTEM directive in the Modelfile** — the system prompt is dynamically generated per-profile by `scorer_adaptive.py` and passed via the Ollama API call.

### Stage 9: Evaluation Results (V2 Production)
```
Direction accuracy:  98.1%
MAE:                 0.393
Spearman ρ:          0.750
PSR:                 24.4% (vs Opus 26.4% — within margin of teacher)
Profile spread:      7.90 (vs Opus 8.20 — near-perfect)
Parse failures:      0
Empty think blocks:  0

Per-profile Spearman:
  Best:  0.879 (AUK CS student)
  Worst: 0.416 (bonsai artist — very niche, limited training signal)
  10 profiles > 0.7 (strong)
  6 profiles 0.65-0.7 (good)
  3 profiles 0.5-0.65 (moderate — niche content underrepresented)
```

---

## PART 3: FORMAT ALIGNMENT — THE #1 CAUSE OF FAILURES

Training data format MUST match inference format character-for-character. Multiple past iterations failed specifically because of format misalignment.

### Training System Prompt Format
```
You are a relevance scorer for a {role} in {location}.
User context: {context if context else 'Not specified'}
Tracked companies: {companies if companies else 'None specified'}
Tracked institutions: {institutions if institutions else 'None specified'}
Tracked interests: {interests if interests else 'None specified'}
Tracked industries: {industries if industries else 'None specified'}
{level_note}

Score each article 0.0-10.0:
9-10: Directly actionable (hiring match, breakthrough in tracked area)
7-8.9: Highly relevant to user's field/interests
5-6.9: Somewhat relevant
0-4.9: Not relevant / noise / wrong level

Reply ONLY with: SCORE: X.X | REASON: brief explanation
```

### Training User Message Format
```
Score this article:
Category: {category_label}
Keywords: {category_items or profile interests}
Title: {article_title[:150]}
Content: {article_content[:500]}
```

### Training Assistant Message Format
```
SCORE: X.X | REASON: {short reason, ~154 chars}
```

**NO `<think>` blocks. NO verbose reasoning. NO preamble before SCORE.**

### Inference Format (scorer_adaptive.py)
The inference system prompt is built by `_build_batch_prompt()` in `scorer_adaptive.py`. It MUST match the training format. Known differences to watch for:

| Element | Training | Inference | Risk Level |
|---------|----------|-----------|------------|
| Category field | Always `"general"` in original V2 | Actual category label | MEDIUM |
| Keywords field | Profile `interests` list | Category-specific `cat_items` | MEDIUM |
| LANGUAGE line | Absent | May be present | LOW |
| Feedback text | Absent | Present (user corrections) | LOW |
| Tracked fields | Direct from profile dict | Via `_tracked_fields_block()` | NONE if identical output |
| Scoring rubric | Identical | Identical | NONE |
| Level note | Identical | Identical | NONE |
| Reply format | Identical | Identical | NONE |

### Parser Regex
```python
score_match = re.search(r'SCORE:\s*(\d+\.?\d*)', response)
reason_match = re.search(r'REASON:\s*(.+)', response)
```

Also add Chinese fallback:
```python
if not score_match:
    score_match = re.search(r'[力评]分[：:]\s*(\d+\.?\d*)', response)
```

### 35 Format Alignment Tests
V2 development created 35 tests covering: format alignment across 3 profiles, forbidden 5.0 rule, score parsing (valid + edge + malformed), batch parsing, and language filtering. A `\n` prefix on `level_note` in `_build_llm_prompt_v2` that wasn't in `export_training.py` was caught by these tests. **Run these tests before any training.**

---

## PART 4: WHAT V2.1 IS DOING WRONG

### Problem 1: Training from V1 base, not V2 base
V2.1 starts from `v1_merged_base` (Qwen3-8B with V1 adapters merged). This throws away everything V2 learned. If the goal is to improve V2, you need to merge V2's adapters into the base first, then train fresh adapters on top.

**Fix:** Create `v2_merged_base` by merging V2's DoRA adapters into the base model, then use that as the starting point.

### Problem 2: 37.6% of training data is low-quality agent data
14,684 examples from Claude Code agents with formulaic 269-char "think_text" (template-based keyword matching). The original V2 used EXCLUSIVELY Opus-scored data (20,550 examples, all genuine reasoning).

**Fix:** Filter out ALL agent-sourced examples. Train only on Opus-scored data. If you need more data, generate it via Batch API, don't use agent shortcuts.

### Problem 3: Training format changed — verbose think_text instead of short reason
V2 trained on: `SCORE: X.X | REASON: short 1-sentence (~154 chars)`
V2.1 trains on: `SCORE: X.X | REASON: full think_text (~843 chars, up to 1,798 chars)`

This is a fundamental change that:
- Changes the output distribution the model learns
- Creates truncation risk: `num_predict=256` may cut off before the SCORE token if the model learns to produce verbose preamble
- Reverses the deliberate V2 design decision to use short inline reasoning

**Fix:** Use the short `reason` field, not `think_text`. The think_text exists to ensure Opus thought deeply before scoring — it's teacher scaffolding, not student output format.

### Problem 4: 90.8% noise in training distribution
Only 101 critical examples (0.5%). Average score 1.26. Even with 3x weighting, the critical signal is severely underrepresented. V2 had better natural distribution because 30 profiles × 813 articles created more contrastive examples across the full range.

### Problem 5: Category always "general" in training
At inference, the model sees real categories like "AI & Technology" but was only trained with `Category: general`. This distribution shift was present in V2 too but was less impactful with short reasons. With verbose think_text, the model may develop stronger expectations about the Category field.

### Problem 6: Post-training format drift (the original V2 degradation)
After V2 was deployed, code changes to `scorer_adaptive.py` modified the prompt format. The model saw different prompt structures at inference than it was trained on. This caused MAE to degrade from 0.393 to higher values and introduced parse failures.

**The correct fix for this is NOT retraining on a different format — it's restoring the inference format to match V2's training format.**

### Problem 7: Dual weighting may fight the framework
V2.1 uses BOTH WeightedRandomSampler AND per-sample loss weights. V2 used sampler-only (per-sample was dropped after debugging showed it fought newer TRL versions). The custom `compute_loss` override that V2.1 uses for per-sample weights can cause NaN errors on newer Transformers/TRL.

**Fix:** Use WeightedRandomSampler only, drop per-sample loss weighting. Sampler ensures the model SEES balanced data. That's sufficient.

---

## PART 5: THE CORRECT FIX PATH

Instead of V2.1 as currently designed, the correct approach is:

### Option A: Fix the format drift (cheapest, fastest)
1. Identify exactly what changed in `scorer_adaptive.py` since V2 training
2. Revert the inference prompt format to match V2's training format
3. Re-register the original V2 GGUF with a correct Modelfile (no `/nothink`)
4. Add Chinese parse fallback
5. Add keyword pre-filter (skip obvious noise before LLM)
6. Re-evaluate on holdout set

This preserves V2's excellent model weights and just fixes the deployment environment.

### Option B: Retrain on expanded data (if Option A isn't enough)
1. Merge V2's adapters into Qwen3-8B base → `v2_merged_base`
2. Train on Opus-only data (filter out all agent data)
3. Use the V2 format: short reason (~154 chars), `enable_thinking=False`
4. Expand profiles from 30 → 47+ if additional Opus-scored data is available
5. WeightedRandomSampler only (no per-sample loss weights)
6. Same hyperparameters as V2 (1e-5 LR, DoRA rank 16, 1 epoch)

### Option C: Full V3 with new data collection (most expensive, highest potential)
1. Generate 50+ diverse profiles
2. Collect 1,000+ articles via StratOS pipeline
3. Score ALL articles × ALL profiles via Opus Batch API
4. Prepare training data using V2's exact format
5. Train from v2_merged_base
6. Target: 30,000+ Opus-scored examples

---

## PART 6: EVERY TECHNICAL GOTCHA AND FAILURE MODE

### ROCm / AMD GPU
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` → NaN during backward pass with DoRA + gradient checkpointing + bf16 on ROCm 6.2. **KEEP DISABLED for training. Safe for inference.**
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` must be set for 7900 XTX
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` prevents OOM from fragmentation
- `attn_implementation="sdpa"` — do NOT use `flash_attention_2` (HuggingFace flash_attn may not compile for RDNA3)

### Qwen3-8B Specific
- **Tokenizer EOS silently changed**: `<|im_end|>` → `<|endoftext|>`. Must explicitly set `eos_token='<|im_end|>'` (ID 151645)
- **SFTTrainer `assistant_only_loss=True`** breaks because Qwen3's default template lacks `{% generation %}` keyword. Use `completion_only_loss` or modify template.
- **Multi-turn template bug**: Second assistant turn gets spurious `<think>\n\n</think>`. StratOS uses single-turn so low risk.
- **8-bit loading fails.** 4-bit works. BF16 is the proven path for training.
- **`/nothink` in Modelfile** causes 41% parse failures — degenerate token loops. Never use it.
- **Greedy decoding + think mode = infinite loops.** Never use temperature=0 with Qwen3 in thinking mode. Use temp=0.3 for scoring, temp=0.6 for general inference.
- **Chinese output**: ~7% of the time Qwen3 outputs Chinese score format. Add regex fallback.

### HuggingFace / TRL
- **Transformers 5.1.0** changed `num_items_in_batch` handling — affects loss scaling. Custom `compute_loss` overrides written for older versions may produce NaN.
- **`model_accepts_loss_kwargs`** in newer Trainer versions changes how loss flows through internals. Custom loss code that worked on older TRL may break.
- **Recommendation: don't fight the framework.** Use standard SFTTrainer with WeightedRandomSampler. Avoid custom `compute_loss`, custom collators, wrapper chains.

### PEFT / DoRA
- After `get_peft_model()`, delete `hf_device_map` attribute and force `.to("cuda:0")` to prevent gradient crashes on meta devices.
- DoRA adapter weights are architecture-specific — cannot transfer between model versions (Qwen3-8B → Qwen3.5 requires full redistillation).
- DoRA rank 16 is sufficient for <20K examples. Rank 32 is an option if you go to 50K+ examples.

### Training Data
- **90/10 stratified split** by score bucket AND profile — ensure every profile appears in both train and eval
- **Every example must have a non-empty, varied system prompt.** Empty system prompts teach the model to ignore system prompts at inference.
- **The forbidden 5.0 rule**: Scores 4.8-5.2 should be nudged to 4.8 or 5.3. Models hedge at 5.0 when uncertain.
- **Think block enforcement**: If using CoT, reject any example where think block < 20 tokens
- **Verify WeightedRandomSampler is actually working**: Log the class distribution of the first 1000 training batches. If noise is still >85%, the sampler is being silently ignored.

### Deployment
- **Q8_0 quantization for scoring.** Q4_K_M introduces noticeable errors in precision scoring.
- **Ollama Modelfile MUST have the ChatML TEMPLATE block.** Without it, the model receives unstructured text and degenerates (every score becomes 3.0-3.5).
- **No SYSTEM directive in Modelfile** — system prompt is dynamic per-profile, passed via API
- **`num_predict: 256`** — the SCORE token comes first, so we only need ~50-100 tokens. Prevents the model from generating verbose responses.
- **Stop tokens**: `<|im_end|>`, `<|endoftext|>`, `<|im_start|>`

### Evaluation
- **Holdout eval set must use completely unseen articles.** V2.1 correctly identified that the original eval used the same articles as training (contaminated). The holdout set uses 50 articles never seen during training.
- **Key metrics**: MAE, direction accuracy (agrees on relevant ≥5 vs noise <5), Spearman ρ (per-profile rank correlation), parse failure rate, per-band MAE
- **The PSR metric was misleading.** Opus's own PSR is 26.4%. V2's 24.4% is within margin. V1's 39.7% was inflated from noisy scoring, not real profile awareness. Don't optimize for PSR > 30%.

---

## PART 7: THE `feedback_text` DYNAMIC CONTEXT

At inference, `scorer_adaptive.py` injects recent user feedback (saves/dismissals) into the scoring prompt as few-shot examples. This is NOT in training data — it's runtime-only calibration. This is correct and should stay this way. Training on feedback text would make the model dependent on it.

---

## PART 8: THE INFERENCE PIPELINE (HOW SCORING ACTUALLY WORKS)

### Three-Phase Scoring
1. **Phase 1 — Rule Scoring**: Regex/keyword noise filters (garbage domains, stale patterns, job aggregators). Items conclusively noise (≤4.0) skip LLM entirely.
2. **Phase 2 — Batch LLM Scoring**: Unresolved items sent to Ollama scorer. AdaptiveScorer batches 4 items per call.
3. **Phase 3 — Re-score Cascade**: Uncertain items (4.5-6.5) re-scored with richer rubric. Capped at 12 items per scan.

### Keyword Pre-filter (NEW in V2.1, good idea)
Before LLM scoring, check keyword overlap between article text and profile interests. Zero keyword matches + low rule score → score as noise (skip LLM). Saves ~50-60% of LLM calls.

### Two Scorer Implementations
- **AIScorer** (`scorer.py`, 2,138 lines): Hardcoded for CPEG/Kuwait profile with extensive pattern lists. Used for default profile.
- **AdaptiveScorer** (`scorer_adaptive.py`, 1,421 lines): Generic, builds rules dynamically from any profile. Used for all non-default profiles.
- Both implement same interface: `score_items()`, `score_item()`, `get_score_category()`

### Streaming + Cancellation
All Ollama scoring calls use `stream=True` with `cancel_check` every 10 tokens for graceful scan cancellation. Partial results preserved.

---

## PART 9: WHAT TO DO RIGHT NOW

1. **STOP the current V2.1 training** if it hasn't finished — it's training on contaminated data with the wrong format
2. **Diagnose the V2 deployment issue**: What exactly changed in `scorer_adaptive.py` that caused the format drift? Compare the current inference format against V2's training format line by line.
3. **Check if the original V2 GGUF still exists**: `data/v2_pipeline/training_output/v2_scorer.gguf`. If yes, Option A (fix deployment) is fastest.
4. **Check if V2's DoRA adapter checkpoint exists**: Needed for Option B (retrain from V2 base).
5. **Run format alignment tests**: 35 tests exist in the codebase. Run them.
6. **If retraining**: Use ONLY Opus-scored data, short reason format, `enable_thinking=False`, WeightedRandomSampler only, start from v2_merged_base.

---

## PART 10: V2 PERFORMANCE TARGETS (for reference)

| Metric | V1 | V2 (target to match or beat) | Opus (ceiling) |
|--------|-----|---------------------------|----------------|
| Direction Accuracy | 90.7% | 98.1% | 100% |
| MAE | 1.553 | 0.393 | 0 |
| PSR | 39.7% (noisy) | 24.4% (calibrated) | 26.4% |
| Spearman ρ | — | 0.750 | 1.0 |
| Profile Spread | 1.04 | 7.90 | 8.20 |
| Parse Failures | — | 0 | 0 |
| Within 1pt | — | — | — |
| Within 2pt | — | — | — |

V2.1 holdout baseline (what the degraded V2 shows):
```
MAE:                 1.533
Direction accuracy:  92.3%
Spearman rho:        0.535
Parse failures:      7.2% (108/1500)
Moderate band MAE:   2.77 (worst — defaults to 8.5)
High band MAE:       2.56 (second worst)
```

This degradation from 0.393→1.533 MAE is almost entirely from format drift and `/nothink`, NOT from model quality issues.
