# StratOS Scorer — Comprehensive Improvement Recommendations

**Date:** March 6, 2026  
**Context:** V2.1 training at 60%, full system review complete  
**Goal:** Absolute best scoring performance, no constraints on structural changes

---

## Executive Summary

After reviewing the full codebase, training pipeline, data landscape, inference flow, and V2/V2.1 training history, I've identified **10 recommendations** ordered by expected impact. The single biggest insight is this: **V2.1 is probably optimizing the wrong variable.** The evidence from V1→V2 shows that data scale and diversity drove improvement, not reasoning format. V2.1 doubles down on reasoning format (think_text) while keeping the same 813 articles. The highest-ROI move is expanding article diversity. But there are also several structural issues in the pipeline that are silently degrading performance.

---

## 1. CRITICAL — Expand Article Diversity (813 → 2,000+)

**Impact: HIGH | Effort: MEDIUM | Confidence: HIGH**

This is the single most important change. Here's why:

Your own data proves it. V1 had 5,679 examples across 10 profiles and got 1.553 MAE. V2 had 18,502 examples across 30 profiles and got 0.393 MAE (contaminated). The improvement came from seeing more articles and more profile perspectives, not from changing the reasoning format. V1 had CoT reasoning and it didn't help. V2 dropped CoT and improved massively through scale.

V2.1 repeats V1's mistake in a different form — it changes the reasoning format (short reason → think_text) while keeping the same 813 articles. The model has seen every one of those 813 articles from dozens of perspectives. It can memorize article-specific patterns (specific company names, specific headlines, specific content snippets) rather than learning generalizable relevance signals. This is why your contaminated-to-holdout ratio is 3.9x (0.393/1.533) — the model relies heavily on memorization.

You have 16,477 untapped articles in your database. Even scoring 1,200 more articles across 20 profiles via Opus Batch API would roughly double your article diversity at a cost of maybe $35-50. This is the highest-ROI investment you can make.

**Specific plan:**
- Sample 1,500 articles from the 16,477 untapped pool, stratified by source and date
- Score them across 20-25 profiles (not all 47 — diminishing returns past ~25 per article)
- Target: 30,000-37,000 new scored examples
- Combined with existing data: ~70,000 total examples, 2,300+ articles
- Cap training at 19,000 as before, but now the 19K is sampled from a much more diverse pool
- This alone could cut your holdout MAE by 30-50%

**Why this works better than think_text:** The model needs to learn "articles about X are relevant to profiles interested in Y" — generalizable patterns. Seeing 2,300 different articles teaches 2.8x more patterns than seeing the same 813 articles with longer explanations.

---

## 2. CRITICAL — Use Qwen3's Native Think Blocks Instead of Stuffing REASON

**Impact: HIGH | Effort: MEDIUM | Confidence: MEDIUM-HIGH**

V2.1 puts think_text into the REASON field: `SCORE: X.X | REASON: <843 chars of reasoning>`. This creates a fundamental tension:

- At training time, the model learns to produce long reasoning after SCORE
- At inference time, `num_predict=256` truncates most of that reasoning
- The SCORE comes first, so it doesn't get truncated — but the model's internal representations are shaped by the expectation of generating long text after it

A better approach: put reasoning in Qwen3's native `<think>` blocks, BEFORE the score:

```
<think>
Looking at this user's profile, they are a Petroleum Engineering student 
at Kuwait University. The article discusses pediatric oncology treatment 
advances. Zero overlap with petroleum engineering interests...
</think>
SCORE: 1.0 | REASON: Oncology article unrelated to petroleum engineering.
```

**Why this is better:**

1. **Think-then-answer** — the model reasons before committing to a score. This is how chain-of-thought actually helps: the reasoning happens before the answer, not after it. In the current V2.1 format, SCORE comes first, so the model has already committed to a number before it "reasons."

2. **Qwen3 naturally wants to do this** — you're fighting the base model's instincts by training SCORE-first. Qwen3 produces `<think>` blocks by default. Training it to think before scoring aligns with its architecture.

3. **Clean inference** — Ollama separates `<think>` blocks into a `thinking` field. The `content` field contains only `SCORE: X.X | REASON: short explanation`. You parse content, ignore thinking. No truncation risk.

4. **You keep the short REASON too** — the model outputs both: detailed reasoning in `<think>` (for learning) and a short reason after SCORE (for display). Best of both worlds.

**Implementation:**
- Modify `stage4_prepare.py`'s `build_training_example()` to format assistant messages as:
  ```
  <think>\n{think_text}\n</think>\nSCORE: {score} | REASON: {short_reason}
  ```
- Set `num_predict` to 768 or 1024 at inference (think blocks + score)
- Ollama handles separation automatically — parse only `content` field
- Switch `_call_ollama` from `/api/generate` to `/api/chat` to get proper think/content separation

**Risk:** Longer inference time per article (maybe 2-3x). Mitigated by the rule-based pre-filter that already skips 50-60% of articles.

---

## 3. HIGH — Fix Training/Inference Format Misalignment

**Impact: HIGH | Effort: LOW | Confidence: HIGH**

This is a free performance gain you're leaving on the table. The misalignments documented in Section 10 of your report are real and measurable — V2's format drift alone doubled MAE from 0.393 to 0.789.

Current misalignments:

| Element | Training | Inference | Fix |
|---------|----------|-----------|-----|
| Category | Always `"general"` | Real labels like "AI & Technology" | Use real categories in training |
| Keywords | Profile `interests` list | Category-specific `cat_items` | Use cat_items in training |
| LANGUAGE line | Absent | Present | Add to training |
| Feedback text | Absent | Present | Add placeholder or remove from inference |

The Category and Keywords mismatches are MEDIUM concern and easy to fix:

- In `stage3_score.py` (or a new scoring pass), when you score articles, you already know each article's category. Pass the real category label and its keywords into the training example instead of hardcoding "general".
- For the LANGUAGE line: either add it to training data (with the profile's language) or remove it from inference. Adding it to training is better since it's a real signal.
- For feedback text: this one is harder since feedback is dynamic. Best approach is to remove it from the system prompt at inference and instead use it only for rule-based pre-filtering.

**Quick fix (today):** Even just changing `Category: general` to `Category: {real_category}` in `build_training_example()` would help, since the model currently learns that category is always "general" and then sees real categories at inference.

---

## 4. HIGH — Aggressive Noise Undersampling

**Impact: MEDIUM-HIGH | Effort: LOW | Confidence: MEDIUM-HIGH**

Your training data is 90.8% noise (scores 0-2.5). Even with 0.5x loss weight, the model processes 17,255 noise examples and only 101 critical examples. The dual weighting helps, but the ratio is still extreme.

The model has already learned "most articles are noise." It doesn't need 17,000 examples to confirm this. What it needs is many more examples of *why* an article scores 5.0 vs 7.0 vs 9.0.

**Recommendation:** Cap noise examples at 5,000-6,000 (instead of 17,255) and fill the freed-up slots with more moderate/high/critical examples. Since you're VRAM-limited to 19,000 examples, this means:

| Band | Current | Proposed | Change |
|------|---------|----------|--------|
| noise (0-2.5) | 17,255 (90.8%) | 5,000 (26.3%) | -71% |
| tangential (2.5-4) | 1,125 (5.9%) | 1,125 (5.9%) | same |
| moderate (4.5-6.5) | 344 (1.8%) | 344 x 5 = 1,720* | +400% |
| high (7-8) | 175 (0.9%) | 175 x 8 = 1,400* | +700% |
| critical (8.5-10) | 101 (0.5%) | 101 x 10 = 1,010* | +900% |

*Upsampled by repeating examples with slight augmentation (see #5).

This gives the model a much more balanced view of the scoring spectrum. Combined with the dual loss weighting you already have, this should dramatically improve mid-range discrimination (your worst band at 2.77 MAE).

**Why this works:** The model's job is to discriminate between "noise," "somewhat relevant," "highly relevant," and "critical." Seeing 17,000 examples of "this is noise" teaches almost nothing new after the first 3,000. But seeing 1,000+ examples of "this is a 7.0 because..." is transformative for mid-range accuracy.

---

## 5. HIGH — Data Augmentation for Rare Bands

**Impact: MEDIUM-HIGH | Effort: MEDIUM | Confidence: MEDIUM**

You only have 101 critical examples and 175 high examples. Even with upsampling, the model sees the same examples repeatedly. Simple augmentation can create meaningfully different training examples:

**Profile-swap augmentation:** For each high-scoring example (article A scores 8.5 for profile P), find other profiles where article A also scores high. The same article scored from a different profile's perspective is a genuinely different training example — different system prompt, same article, similar score.

**Content truncation augmentation:** Randomly truncate article content to different lengths (300, 400, 500 chars). This teaches the model to score based on partial information, which is what happens in production when summaries vary in length.

**Keyword perturbation:** For the Keywords field, randomly drop 1-2 keywords or reorder them. This prevents the model from relying on keyword position.

**Why not synthetic articles?** Generating fake articles would introduce distribution shift. These augmentations keep real data but vary the presentation, which is exactly what production variability looks like.

---

## 6. HIGH — Contrastive Training (Actually Use Your Contrastive Pairs)

**Impact: MEDIUM-HIGH | Effort: MEDIUM-HIGH | Confidence: MEDIUM**

You extract 25,243 contrastive pairs in `stage4_prepare.py` (same article, different profiles, score gap ≥ 3.0) — but then you save them to a JSON file and never use them in training. The actual training uses standard completion-only cross-entropy loss.

Contrastive pairs are incredibly valuable for teaching the model *why* the same article scores differently for different people. This is the core of profile-aware scoring.

**Two approaches:**

**A) Interleaved contrastive examples (simpler):** During training, after every N standard examples, insert a paired example: the same article scored from two different profiles. The model sees "Profile A → SCORE: 2.0" immediately followed by "Profile B → SCORE: 8.5" for the same article. The gradient signal teaches it that profile context determines the score.

**B) Contrastive loss (more complex):** Add a margin-based contrastive loss term that penalizes the model when it fails to score the same article differently for profiles that should produce different scores. This requires modifying `train_v2.py` to extract scores from generated outputs during training and compute an auxiliary loss.

Approach A is simpler and probably gets you 70% of the benefit. Structure training batches so that every effective batch of 16 includes at least 2-4 contrastive pairs.

---

## 7. MEDIUM — Filter or Down-Weight Agent-Scored Data

**Impact: MEDIUM | Effort: LOW | Confidence: MEDIUM**

37.6% of your total scores (14,684) come from agent scoring. Your report flags that these have "formulaic reasoning" (269 chars avg vs Opus's 1,133 chars). The concern is real: if agents used keyword-matching heuristics rather than genuine reasoning, their scores may be noisy — correct on average but wrong on edge cases.

**Diagnostic first:** Before filtering, measure this: take 200 agent-scored examples and re-score them with Opus. Compute the MAE between agent and Opus scores. If MAE > 1.5, the agent data is hurting you. If MAE < 0.8, it's fine.

**If agent data is noisy:** Either remove it from training entirely (reducing to ~12,000 Opus-scored examples — still enough with better sampling), or apply a 0.3x loss weight to agent-scored examples (lower than noise's 0.5x).

**If agent data is clean:** Keep it but be aware that its formulaic think_text may teach the model surface-level patterns rather than genuine reasoning.

---

## 8. MEDIUM — Score Bucketing with Ordinal Regression

**Impact: MEDIUM | Effort: HIGH | Confidence: MEDIUM**

This is a structural change that reframes the problem. Currently, the model generates text (`SCORE: 7.5 | REASON: ...`) and you parse out a number. This is inefficient — the model's text generation capacity is being used to produce a single number.

**Alternative: Train a classification head on top of the model's hidden states.**

Instead of generating "SCORE: 7.5", the model classifies into ordered buckets:
- Bucket 0: score 0-2 (noise)
- Bucket 1: score 2-4 (tangential)
- Bucket 2: score 4-6 (moderate)
- Bucket 3: score 6-8 (high)
- Bucket 4: score 8-10 (critical)

Use ordinal regression loss (cumulative link model) so the model learns that predicting bucket 4 when the true bucket is 0 is much worse than predicting bucket 1.

**Advantages:**
- No parse failures (output is a softmax over 5 classes)
- Faster inference (no autoregressive text generation needed)
- The model focuses entirely on learning relevance representations
- You can still have a reason by generating text in a second pass for high-scoring items only

**Disadvantage:** Requires modifying the architecture — adding a linear head on top of the final hidden state. Can't deploy via Ollama's standard text generation. You'd need a custom inference server (e.g., vLLM or a simple FastAPI wrapper).

**Compromise version:** Keep the generative approach but train with a coarser score granularity. Instead of 0.0-10.0 in 0.5 increments (21 possible scores), use integer scores only (11 values). This reduces the output space the model needs to learn and may improve accuracy in the mid-range where it currently collapses.

---

## 9. MEDIUM — Multi-Epoch Training with Early Stopping

**Impact: MEDIUM | Effort: LOW | Confidence: MEDIUM**

You train for exactly 1 epoch. Your loss curve shows a plateau starting around step 250 (~4.5-5.2), with slow decline through step 718. The cosine schedule will decay LR to near zero by step 1188, so the back half of training contributes little new learning.

**Recommendation:** Train for 2-3 epochs with a holdout validation loss computed every 200 steps. Stop when validation loss stops improving for 2 consecutive checkpoints.

**Why 1 epoch might not be enough:** With aggressive noise undersampling (#4 above), the model sees only 5,000 noise examples once. But the moderate/high/critical examples that are upsampled need multiple passes to learn nuanced scoring logic. A second epoch lets the model refine its understanding of these rare-but-important examples.

**Why more than 3 epochs is risky:** With only 813 articles, additional epochs increase memorization. But with expanded article diversity (#1), 2-3 epochs becomes safer.

**Implementation:** Modify `train_v2.py` to evaluate on a small held-out subset (500 examples) every 200 steps and save the checkpoint with the best validation loss.

---

## 10. LOW-MEDIUM — Calibration Layer for Deployment

**Impact: LOW-MEDIUM | Effort: LOW | Confidence: HIGH**

After all training improvements, there will still be a gap between PyTorch bf16 inference and Ollama Q8_0 serving. Instead of trying to eliminate this gap, add a lightweight calibration layer.

After training and GGUF export, run the Ollama model on 500-1000 calibration examples where you know the ground truth. Fit a simple isotonic regression or Platt scaling function that maps Ollama output scores to calibrated scores. This takes 5 minutes to implement and can recover 0.1-0.2 MAE lost to quantization and serving differences.

You already have a `_calibrate_score()` method in `scorer_adaptive.py` with a V1 calibration table. Just update it with a proper calibration curve fit to V2.1's actual Ollama output distribution.

---

## Priority Roadmap

**Before V2.1 finishes (now):**
- Plan article expansion (#1) — select 1,500 articles from DB
- Design think-block training format (#2)

**After V2.1 finishes (today/tonight):**
- Complete the 4-measurement eval matrix from the training report
- Run the V2 re-eval with corrected Modelfile
- Smoke test V2.1

**Next training cycle (V3):**
1. Score 1,500 new articles via Opus Batch API (~$40, ~24h) → #1
2. Fix format alignment → #3
3. Implement noise undersampling + upsampling → #4
4. Switch to think-block format → #2
5. Add contrastive pair interleaving → #6
6. Train 2 epochs with early stopping → #9
7. Calibrate Ollama output → #10

**Future iteration:**
- Data augmentation (#5)
- Agent data audit (#7)
- Score bucketing (#8) — only if generative approach hits a ceiling

---

## What I Wouldn't Change

- **DoRA over LoRA** — the magnitude decomposition is well-suited for this task
- **Qwen3 8B as the base** — right size for your VRAM, good multilingual handling
- **The dual weighting system** — sound approach, just needs more extreme noise undersampling
- **Q8_0 quantization** — minimal accuracy loss vs bf16, proven stable
- **The three-phase scoring pipeline** (rules → LLM → rescore) — good architecture, the LLM just needs to be better at the middle range
- **Hyperparameters** — rank 16, alpha 32, lr 1e-5, batch 2 with grad accum 8 — all reasonable for this scale

---

## Bottom Line

If you could only do one thing: **expand article diversity.** The model has squeezed nearly everything it can from 813 articles. Doubling the article pool will teach it more generalizable relevance patterns than any format change, loss function tweak, or hyperparameter adjustment.

If you could do two things: expand articles AND switch to native think blocks. Together, these address the two biggest bottlenecks — data diversity and reasoning architecture.

Everything else is optimization on top of those two foundations.
