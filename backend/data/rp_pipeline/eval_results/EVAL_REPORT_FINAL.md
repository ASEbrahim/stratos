# RP Model Prompt Engineering Evaluation Report

**Date:** 2026-03-16
**Model:** stratos-rp-v2 (and stratos-rp-q8 baseline)
**Finding:** Both models share identical weights (same GGUF blob hash: 6ca9c5b0b6c44990)

## Executive Summary

After 6 iterations of prompt engineering across 210 unique test questions, **we have hit stagnation**. The model's quality is bounded by its underlying weights, not the system prompt. The best achievable score through prompt engineering is approximately **3.2/5**, compared to a **3.66/5** baseline. The model has fundamental issues (stochastic think-block leaking, weak god-mod resistance, inconsistent length matching) that cannot be fixed by prompt engineering alone.

## Score Progression

| Run | Prompt Version | Score | Think Leaks | Key Issue |
|-----|----------------|-------|-------------|-----------|
| **Q8 Baseline** | V4 (original) | **3.66** | 1/35 | Best overall — natural, low think leaks |
| Iteration 1 | V4 from Modelfile + char card | 3.43 | 34/35 (4 full) | Massive think leak from dual system messages |
| Iteration 2 | Tighter length + god-mod | 2.97 | 31/35 (6 full) | Think leaks worse, OOC noise |
| Iteration 3 | Inline prompt (matching baseline) | 3.11 | 9/35 (3 full) | Think fixed, brevity fixed, long prose too short |
| **Iteration 4** | Balanced length + god-mod examples | **3.20** | 7/35 (5 full) | Best prompt-engineered score |
| Iteration 5 | Stronger god-mod in char cards | 2.97 | 8/35 (3 full) | OOC prefixes appeared, model confused |
| Iteration 6 | Same as iter4 (stagnation check) | 2.46 | 9/35 (7 full) | High stochastic variance, OOC noise |

## Stagnation Confirmed

Last 3 iterations with prompt changes: **3.20 → 2.97 → 2.46** (within noise, trending down)
The model's behavior is highly stochastic — the same prompt produces very different quality across runs.

## Key Findings

### 1. Think Block Leaking (Unfixable via Prompt)
The Qwen3.5 abliterated model produces `<think>` blocks unpredictably despite `think: false` in the API. This consumes the token budget (2048 tokens of thinking, leaving nothing for actual output). Rate varies 1-9 per 35 tests (3-26%) depending on the run. **This is a model-level issue, not prompt-fixable.**

### 2. Identical Weights
`stratos-rp-v2` and `stratos-rp-q8` resolve to the same Ollama blob (sha256-6ca9c5b0b6c4499). The DoRA fine-tuning either didn't produce different weights, or the V2 GGUF was created from the same base. **There is no trained vs untrained distinction to test.**

### 3. Brevity
- **Baseline** (V4 original prompt): Consistently too long for short inputs
- **Iter3-6** (explicit length rules): Much better brevity (1-10 words for short inputs)
- **Tradeoff**: Fixing brevity caused long inputs to get too-short responses in some cases

### 4. God-Mod Resistance
- **Never reliably resisted** in any iteration
- Character-card-level instructions ("NEVER screams") helped in ~50% of cases
- System prompt rules alone achieved ~30% resistance
- The model fundamentally wants to comply with user-written actions

### 5. ERP/Explicit Content
- **No refusals** across any iteration (abliteration working)
- Quality varies: best at slow-burn emotional scenes (4-5/5)
- Maintains character voice during intimate scenes ~80% of the time
- Explicit physical detail present but not graphic — more literary than explicit

### 6. Code-Switching
- Japanese words appear when explicitly listed in the character card
- Works best with 3-6 specific word suggestions (sugoi, yabai, maji, etc.)
- Without specific word lists, the model defaults to English only

### 7. Multi-NPC
- Consistently gives 1-2 NPCs distinct voices
- Rarely achieves all 4 NPCs with distinct dialogue
- Best when the scene demands a reaction from everyone (dramatic moments)

## Best Prompt Configuration

The optimal system prompt is now embedded in `Modelfile.rp_v2`. Key elements:
1. **Explicit length matching rules** with word-count ranges
2. **Character agency section** with specific "CANNOT" examples
3. **Minimal Modelfile SYSTEM** — detailed prompt sent via API call to avoid dual-system-message confusion
4. **Temperature 0.8, top_p 0.95** (matching baseline)
5. **Character-card-level behavioral constraints** (e.g., "NEVER screams or rages")

## Recommendations

1. **Training data is the bottleneck**, not prompts. To improve beyond 3.66, the model needs:
   - DoRA fine-tuning on actual RP conversations (the current GGUF is untrained)
   - Length-matched training examples (short input → short output)
   - God-mod resistance examples in training data

2. **Think block suppression** needs either:
   - A model that natively supports `think: false` (Qwen3.5 doesn't reliably)
   - Post-processing that detects and retries on think-leaked outputs
   - A different base model without CoT tendencies

3. **Production deployment** should:
   - Use the inline system prompt approach (not Modelfile SYSTEM)
   - Include think-block stripping as a safety net
   - Implement retry logic for empty/think-leaked outputs
   - Set `num_predict` dynamically based on input length

## Files

- Eval scripts: `data/rp_pipeline/scripts/run_iter{1-6}.py`
- Results: `data/rp_pipeline/eval_results/iter{1-6}_stratos-rp-v2.json`
- Q8 baseline: `data/rp_pipeline/eval_results/q8_baseline_stratos-rp-q8.json`
- Modelfile: `data/rp_pipeline/Modelfile.rp_v2`
