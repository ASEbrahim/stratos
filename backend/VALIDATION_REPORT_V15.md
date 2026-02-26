# StratOS v15 — Phase 3 Validation Report

**Date:** 2026-02-18
**Model:** stratos-scorer-v15 (Qwen3-8B, DoRA rank-16, 1 epoch, 5977 examples)
**Eval set:** 800 examples (100 articles x 8 profiles, all scored by Claude Opus)

---

## Executive Summary

The model is **deployed and functional** (empty response bug fixed) but **fails all core scoring metrics**. It has a severe score inflation bias — outputting 8.5 for ~50% of inputs regardless of profile or content. Root cause identified: **SFTTrainer loss computed on all tokens (system+user+assistant) instead of assistant-only**, diluting the score-learning signal to ~25% of total loss.

---

## Fixes Applied (Modelfile + scorer_adaptive.py)

### Modelfile (data/models/v15/Modelfile)
1. Added `/no_think` to template user prompt — suppresses Qwen3 native thinking
2. Pre-filled `<think>\n</think>` in assistant preamble — eliminates stochastic empty responses
3. Empty response rate: **~100% → ~4%**

### scorer_adaptive.py
1. Removed `"think": True` from Ollama payload (was causing complete empty responses via API)
2. Made SCORE/REASON regex case-insensitive (`re.IGNORECASE`) — handles model's `Score:`/`SCORE:` variants
3. Updated docstring to reflect think-mode-disabled rationale

---

## Metric Results

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **PSR** (Profile Sensitivity Rate) | > 80% | **31.0%** | FAIL |
| **MAE** (Mean Absolute Error) | < 1.5 | **5.335** | FAIL |
| **Spearman rho** | > 0.80 | **0.1259** | FAIL |
| Anti-memorization (entity swap) | delta < 2.0 | **0.5** | PASS |
| Smoke tests | 5/5 | **2/5** | FAIL |
| Novel profile (9th profile) | directionally correct | **3/7** | FAIL |
| Format compliance | SCORE: X.X \| REASON: | **~94%** | OK |

---

## 3a. Profile Sensitivity Rate: 31.0% (FAIL)

671 contrastive pairs tested (where teacher score delta >= 2.0 across profiles for same article).

Only 208 pairs (31.0%) show the student reproducing the correct direction with magnitude >= 1.0. The model barely distinguishes between profiles — a Kuwait engineering job gets similar scores from both the Kuwait CPEG student and the Texas nurse.

---

## 3b. Teacher-Student Agreement: MAE=5.335, rho=0.126 (FAIL)

767 valid scored examples (33 parse failures / empty responses = 4.1%).

**Score distributions:**
- Teacher: mean=1.04, std=1.44
- Student: mean=6.18, std=3.43

**Per-profile breakdown:**

| Profile | n | MAE | rho | Teacher mean | Student mean |
|---------|---|-----|-----|-------------|-------------|
| bangalore_ds | 100 | 4.43 | 0.137 | 1.08 | 5.30 |
| dc_cybersec | 92 | 6.06 | -0.042 | 0.68 | 6.65 |
| dubai_founder | 91 | 5.29 | 0.231 | 1.26 | 6.38 |
| kuwait_cpeg | 98 | 4.76 | -0.024 | 2.51 | 6.43 |
| london_finance | 98 | 5.60 | 0.184 | 0.84 | 6.41 |
| munich_mecheng | 100 | 5.47 | 0.235 | 0.60 | 5.94 |
| texas_nurse | 96 | 5.11 | 0.022 | 0.57 | 5.65 |
| toronto_teacher | 92 | 6.04 | 0.143 | 0.74 | 6.73 |

All profiles show the same pattern: teacher scores are low (0.6-2.5 mean), student scores are high (5.3-6.7 mean).

**Student score distribution (bimodal collapse):**
```
8.5:  382 (49.8%) ████████████████████████
1.5:  128 (16.7%) ████████
1.0:   92 (12.0%) █████
9.5:   85 (11.1%) █████
2.5:   24 ( 3.1%) █
(all others < 2%)
```

**Confusion matrix (teacher band -> student band):**
```
               noise   tang    mod   high   crit
noise    →      206     26     18     12    398   (31% accuracy)
tang     →       15      1      3      2     48   ( 1% accuracy)
moderate →        5      0      0      1     23   ( 0% accuracy)
high     →        1      0      0      0      5   ( 0% accuracy)
critical →        1      0      0      0      2   (67% accuracy)
```

60% of teacher-noise items are scored CRITICAL by the student.

---

## 3c. Novel Profile Ablation (Marine Biologist in Lisbon): 3/7 FAIL

| Article | Expected | Got | Status |
|---------|----------|-----|--------|
| EU coral reef restoration fund | HIGH (>6) | 9.5 | PASS |
| Kuwait grad engineering program | LOW (<3) | 9.5 | FAIL |
| GPT-5 announcement | MODERATE (2-7) | 8.5 | FAIL |
| Lisbon marine biotech investment | MOD-HIGH (>5) | 8.5 | PASS |
| Kim Kardashian skincare | NOISE (<3) | 8.5 | FAIL |

The model scores everything 8.5-9.5 for the novel profile. It correctly ranks the ocean article highest but fails to suppress irrelevant content.

---

## 3d. Anti-Memorization: PASS

| Test | Score | Expected |
|------|-------|----------|
| Equate (original, Kuwait) | 9.5 | High |
| FakeCorp (same article, Kuwait) | 10.0 | Should be similar if generalized |
| Equate (wrong location: Sydney) | 8.5 | Should be lower |

- Entity swap delta: 0.5 (< 2.0 threshold) — model generalized, not memorizing "Equate"
- Location change lowered score by 1.0 — some location awareness

---

## 3e. Smoke Tests: 2/5 FAIL

| Test | Expected | Got | Status |
|------|----------|-----|--------|
| Kuwait student + Kuwait career | 9+ | 10.0 | PASS |
| Texas nurse + Kuwait career | 0-2 | 8.5 | FAIL |
| Texas nurse + Houston hospital | 9+ | 8.5 | FAIL |
| Novel profile + generic tech | 2-7 | 8.5 | FAIL |
| Blank profile std | < 1.5 | 0.00 | PASS (degenerate) |

Profile separation (Test 1 - Test 2) = 1.5 points. Target: >5.0 points.

---

## Root Cause Analysis

### Primary: Loss Dilution from Unmasked Input Tokens

**Location:** `train_lora.py` lines 320-376 (PEFT path used for v15)

```python
def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    return {"text": text}

dataset = dataset.map(format_chat)  # Pre-formats messages → text

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,  # Dataset has "text" column, NOT "messages"
    ...
)
```

**The problem:** When SFTTrainer receives a pre-formatted `text` column, it computes loss on **ALL tokens** — system prompt, user article, AND assistant response. There is no automatic input masking.

**Measured impact:**
- Average total content: ~1,403 characters
- Average assistant content: ~345 characters (24.6%)
- **~75% of loss signal is wasted** predicting system prompts and article text
- Only ~25% trains the model to produce correct scores

**This explains the misleading training metrics:**
- eval_loss = 0.5337 (looks good — but 75% is from easy prefix prediction)
- token_accuracy = 88.3% (looks good — dominated by predicting known prompts)
- The model barely learns score discrimination because that signal is drowned out

### Secondary: Bimodal Collapse

The model learned a degenerate scoring function:
```
if (obviously_garbage_content): return ~1.5
else: return ~8.5
```

It correctly identifies obvious noise (Chinese text, data pages, gibberish) but defaults to 8.5 for anything that looks like real content — regardless of profile match.

---

## Recommendations for v16

### Fix 1 (Critical): Enable Assistant-Only Loss Masking

**Option A — Use messages format directly (simplest):**
```python
# Don't pre-format — let SFTTrainer handle masking
dataset = load_dataset("json", data_files=training_file, split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,  # Keep "messages" column
    args=SFTConfig(
        dataset_text_field=None,  # Not needed
        ...
    ),
)
```

**Option B — Use DataCollatorForCompletionOnlyLM:**
```python
from trl import DataCollatorForCompletionOnlyLM

response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    data_collator=collator,
    ...
)
```

### Fix 2: Balance Training Distribution

Current: 65% noise, 8.8% tangential, 12.5% moderate, 7.6% high, 1.8% critical

Target: Upsample mid-range and high-score examples to ~20% each band.

### Fix 3: Add Score-Specific Eval Metric

Add a custom metric that extracts SCORE from model output and computes MAE against teacher, ignoring prefix tokens:

```python
def compute_score_metrics(eval_preds):
    # Decode outputs, extract SCORE: X.X, compare to teacher
    ...
```

### Fix 4: Consider 2-3 Epochs

CLAUDE.md warns about 3-epoch overfitting (eval_loss rises), but the current 1 epoch may be insufficient with proper masking. Try 2 epochs with early stopping on a score-specific eval metric.

---

## Files Modified

| File | Change |
|------|--------|
| `/tmp/Modelfile` → `data/models/v15/Modelfile` | Pre-filled think block, /no_think in template |
| `processors/scorer_adaptive.py` | Removed think:True, case-insensitive regex |
| `validate_phase3.py` (new) | Full validation suite |
| `data/validation_results.json` (new) | Raw scoring results |

## Validation Data

- Full results: `data/validation_results.json` (800 examples with teacher + student scores)
- Validation script: `validate_phase3.py` (rerunnable with --metrics-only, --quick-only, --eval-only)
