# StratOS Scorer — Lessons Learned & Engineering Best Practices

**Created:** March 7, 2026
**Scope:** Everything learned across V1, V2, V2.1, V2.2 development cycles. Covers training pipeline, inference, evaluation, data quality, hardware, and project management.
**Purpose:** Prevent repeated mistakes and establish engineering discipline for all future scorer iterations.

---

## Part 1: Mistakes Made and How to Prevent Them

### 1.1 The /nothink Disaster (V2 Deployment)

**What happened:** V2's Modelfile included `/nothink` to suppress Qwen3's think blocks. This caused the model to enter degenerate token loops, outputting slash-commands (`/soft`, `/nostart`, `/assistant`) instead of scores. 41% of outputs were unparseable.

**Why it wasn't caught:** No smoke test was run after Modelfile changes. The Modelfile was edited, registered with Ollama, and deployed directly to production without verifying output.

**Prevention:**
- Never deploy a Modelfile change without sending at least 3 test prompts through Ollama and manually inspecting raw output
- Parse failure rate should be a monitored metric — any spike above 5% triggers investigation
- The prompt_version.py system now detects template changes, but Modelfile parameters (num_predict, temperature, stop tokens) are outside its scope. Consider extending version-pinning to hash Modelfile contents too

---

### 1.2 The num_predict 2048 Waste (V2 Deployment)

**What happened:** V2's Modelfile set `num_predict 2048` when the model outputs ~50-100 tokens. Every scoring call generated up to 2048 tokens of padding/rambling before hitting the stop token, wasting 10-20x inference time.

**Why it wasn't caught:** The default was never questioned. Training worked, eval worked, nobody profiled actual inference token counts.

**Prevention:**
- Profile your model's actual output length on 100 representative examples before setting num_predict
- Set num_predict to max_expected_output × 1.5, not some large default
- For StratOS scoring: output is `SCORE: X.X | REASON: ~30 words` = ~50-80 tokens. num_predict=256 is generous. 512 for eval (think block headroom).

---

### 1.3 The seq_length=1024 Waste (V2 Training)

**What happened:** V2 trained with max_seq_length=1024 when all training examples were under 500 tokens. This wasted ~9 hours per training run (38.5s/step instead of ~9.6s/step) and nearly blocked Qwen3.5-9B from fitting in 24GB VRAM.

**Why it wasn't caught:** 1024 was set as a conservative default and nobody checked the actual token distribution until Qwen3.5-9B OOMed at 1024, forcing a fallback to 512.

**Prevention:**
- **Always run a token distribution check before training:**
  ```python
  from transformers import AutoTokenizer
  lengths = [len(tokenizer.encode(example)) for example in dataset]
  print(f"Max: {max(lengths)}, P95: {sorted(lengths)[int(len(lengths)*0.95)]}, P99: {sorted(lengths)[int(len(lengths)*0.99)]}")
  ```
- Set max_seq_length to P99 + 10% buffer, not an arbitrary power of 2
- Attention is O(n²) — halving seq_length gives ~4x speedup. This is always worth checking.
- Add this check to stage4_prepare.py so it reports token stats every time training data is generated

---

### 1.4 The Think_Text Catastrophe (V2.1)

**What happened:** V2.1 added Opus think_text (843 chars avg) to training assistant messages. The model learned to generate plausible-sounding justifications for everything, inflating all scores to 8-9. MAE went from 1.544 to 3.244 — worse than random for direction accuracy (53.5%).

**Why it wasn't caught before training:** A cross-Claude review predicted this risk but it was noted as a concern rather than a blocker. The review said "V1 had CoT and got 1.553 — data scale drove V2's improvement, not reasoning format." This was correct.

**Prevention:**
- **The pattern is definitive for 8B-class models:**
  - V1 (CoT long) → 1.553 MAE
  - V2 (short reason) → 1.544 MAE
  - V2.1 (think_text) → 3.244 MAE
  - Short reasons ≈ CoT on holdout. Think_text catastrophically worse.
- Do NOT revisit chain-of-thought or verbose reasoning in training targets for models under ~30B parameters. They lack capacity for genuine reasoning — they learn to pattern-match the style of reasoning instead.
- If a pre-training review flags a risk, treat it as a blocker unless you can run a small-scale experiment first (e.g., train on 1,000 examples, eval on 100 holdout samples — 30 minutes, not 12 hours)

---

### 1.5 Changing Multiple Variables Simultaneously (V2.1)

**What happened:** V2.1 changed 4+ variables at once: think_text format, 17 new agent-scored profiles, different data distribution, 5x longer assistant messages. When it failed catastrophically, it was impossible to determine which change caused the regression.

**Prevention:**
- **Change ONE variable per version.** If you want to test think_text AND new profiles AND more data, that's three separate training runs:
  - V2.2: expand articles only (same format, same profiles)
  - V2.3: if V2.2 works, try new profiles
  - V2.4: if V2.3 works, try format changes
- Each run is ~3 hours now (not 12). You can afford sequential experiments.
- Document what changed in every version's training config. The training_pipeline.md format works — keep using it.

---

### 1.6 Incomparable Evaluation Metrics (V2)

**What happened:** V2's "0.393 MAE" was measured via direct PyTorch bf16 inference during training. V2's "1.533 holdout MAE" was measured through Ollama with a broken Modelfile. These numbers were compared as if they measured the same thing, leading to months of confusion about true model quality.

**Prevention:**
- **Always evaluate through the same pipeline the model will be served through.** If you serve via Ollama Q8_0, evaluate via Ollama Q8_0. PyTorch bf16 eval during training is a sanity check, not a deployment metric.
- Maintain a four-measurement eval matrix:
  | | Contaminated | Holdout |
  |---|---|---|
  | **PyTorch bf16** | Memorization ceiling | True generalization (research) |
  | **Ollama Q8_0** | Quantization + serving cost | **Real-world performance (ship metric)** |
- Only the bottom-right cell matters for deployment decisions
- Label every metric with its measurement pipeline: "MAE 0.393 (PyTorch bf16, contaminated)" not just "MAE 0.393"

---

### 1.7 Eval Contamination (V2 Early)

**What happened:** The original eval set used the same 813 articles as training (different profiles). The 0.393 MAE was measuring memorization, not generalization. Only after creating a proper holdout with 50 unseen articles did the true performance gap (1.544 MAE) become visible.

**Prevention:**
- **Holdout articles must be selected BEFORE any scoring or training begins.** Once articles enter the training pipeline, they're contaminated forever.
- The 50 holdout article IDs in `holdout_articles.json` are sacred. Every pipeline stage must check against them. Currently verified at 4 defense points in v22_expansion.py.
- When expanding to new articles, always exclude holdout IDs from the selection pool — even if you think "these are different articles." Belt and suspenders.
- Consider periodically refreshing the holdout set with new unseen articles to prevent implicit contamination through prompt template optimization

---

### 1.8 Hardcoded Token IDs (V2 → V2.2 Migration)

**What happened:** train_v2.py hardcoded Qwen3's token IDs (eos=151643, pad=151643). When switching to Qwen3.5 (eos=248044, pad=248044), these would have silently broken training — the model would never learn when to stop generating.

**Why it wasn't caught:** It worked for Qwen3. Nobody anticipated a base model change.

**Prevention:**
- **Never hardcode token IDs.** Always use dynamic lookup:
  ```python
  endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
  im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
  ```
- This applies to any magic number that comes from a specific model: vocab size, layer count, hidden dimension, special token IDs
- When switching base models, grep the entire codebase for the old model's specific values

---

### 1.9 The Tracked Fields Join Bug

**What happened:** `', '.join("Equate, SLB")` iterates characters, producing `"E, q, u, a, t, e, ,,  , S, L, B"` instead of the intended comma-separated company list. The string was treated as an iterable of characters.

**Prevention:**
- Always type-check before calling `.join()` on data from external sources (YAML profiles, DB, API):
  ```python
  if isinstance(value, list):
      result = ', '.join(value)
  else:
      result = str(value)
  ```
- This class of bug affects any Python code that expects a list but receives a string. Both are iterable, so no error is raised — just wrong output.

---

### 1.10 Format Drift Between Training and Inference (V2)

**What happened:** After V2 training, code changes to `scorer_adaptive.py` prompt format caused training/inference misalignment. MAE degraded from 0.393 to 0.789 on contaminated eval — format drift alone doubled the error.

**Prevention:**
- **prompt_version.py is now deployed.** It hashes the prompt template skeleton and warns if training and inference templates diverge.
- But it only catches template structure changes, not parameter changes (temperature, num_predict, stop tokens) or subtle whitespace differences
- When modifying scorer_adaptive.py, always check if the same change needs to be reflected in stage4_prepare.py (and vice versa)
- The ideal solution: both files import from a shared `prompt_templates.py` module. Not yet implemented.

---

## Part 2: Data Quality Lessons

### 2.1 Article Diversity Matters More Than Example Count

V2 had 18,502 examples from 813 articles. V2.2 will have ~12,000 examples from 2,000 articles. The expanded article set is almost certainly better training data despite having fewer total examples, because:

- The model memorized the 813 articles (contaminated MAE 0.393, holdout MAE 1.544 — a 3.9x gap)
- Only 135 of the 813 training articles overlapped with the 17,000+ articles in the production database
- The model trained on one distribution and scored on a completely different one

**Rule:** When holdout MAE is much worse than contaminated MAE, the problem is data diversity, not model capacity or training format.

---

### 2.2 Noise Distribution Must Reflect Production

V2 training was 90.8% noise (score 0-2.5). Production feeds are 80-90% noise. The noise cap for V2.2 is 9,000 examples (~47% of training data), with WeightedRandomSampler further adjusting per-batch distribution.

**Danger of over-aggressive noise undersampling:** If you cap noise at 5,000 (26% of data), the model's prior shifts toward "most things are somewhat relevant" and noise scores inflate. The model needs to see enough noise to maintain a realistic base rate.

**Danger of no undersampling:** At 90%+ noise, the model rarely sees high-relevance examples and can't learn to score them. The dual weighting system (sampler + loss weights) helps but doesn't fully compensate.

**Sweet spot:** 40-55% noise in training data, with weighted sampling handling the rest.

---

### 2.3 Scoring Provider Validation Is Cheap and Essential

Before using any provider's scores as training data, run a 300-sample stratified validation test:

| Provider | Raw MAE | Calibrated MAE | Viable? | Cost of test |
|----------|---------|---------------|---------|-------------|
| DeepSeek V3.2 | 1.521 | 1.095 | No (middle-range collapse) | $0.10 |
| Gemini 3 Flash | 1.090 | 0.893 | Yes (with isotonic calibration) | $0.00 |

**The $0.10 DeepSeek test saved potentially $12 of wasted scoring + weeks of training on bad data.** Always validate before bulk scoring.

**What to look for:**
- MAE < 1.0 (raw or calibrated)
- Score diversity: how many distinct score levels does the provider use? DeepSeek used ~6, Gemini used 15+. More diversity = better calibration potential.
- Per-band bias: is the error systematic (fixable with calibration) or random (not fixable)?
- Middle-range collapse: does the provider dump everything into 2-3 scores? If one score maps to a 6+ point range in ground truth, calibration cannot help.

---

### 2.4 Isotonic Calibration — When It Works and When It Doesn't

**Works when:** The provider has systematic bias that preserves rank ordering. Gemini's Spearman rho of 0.922 means it agrees with Opus on relative ordering — it just needs a monotone transformation to fix absolute values.

**Doesn't work when:** The provider collapses multiple ground truth values into a single output score. DeepSeek's score 2.5 mapped to Opus scores from 2.0 to 8.5 — no monotone function can separate those.

**Implementation:** `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')` fitted on 283 (provider, ground_truth) pairs. Save as both pickle (for pipeline use) and JSON (for inspection/portability). Tag with provider model version + prompt hash + validation date. Re-validate if anything changes.

---

### 2.5 User Feedback Data Requires Care

2,631 user feedback entries exist, all from a single user (Ahmad/CPEG student). Including these at >1.0x weight would bias the model toward one profile's relevance judgments. At 1.0x weight they provide genuine signal about real-world scoring preferences but only for one perspective.

**Rule:** User feedback weight should be ≤1.0x until you have feedback from 5+ diverse profiles.

---

## Part 3: Hardware & Environment Lessons

### 3.1 ROCm Survival Guide (AMD 7900 XTX)

| Setting | Value | Why |
|---------|-------|-----|
| `ROCR_VISIBLE_DEVICES=0` | Always set | Excludes 512MB iGPU that causes memory faults |
| `HSA_OVERRIDE_GFX_VERSION=11.0.0` | Always set | Required for ROCm gfx1100 compatibility |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | NEVER set | Causes NaN gradients with DoRA — destroys training silently |
| `OLLAMA_FLASH_ATTENTION=0` | Keep disabled | Unreliable on ROCm 6.2 |

**The AOTRITON trap:** This env var is sometimes recommended in ROCm guides. It causes NaN gradients that don't crash training — loss just stops decreasing. You can waste an entire 12-hour run before noticing. If you see loss plateau at an unusually high value with NaN gradient norms, check for this variable first.

---

### 3.2 VRAM Budget (24GB)

| Configuration | VRAM Used | Fits? |
|--------------|-----------|-------|
| Qwen3.5-9B DoRA training, seq_len=512 | 20.37 GB | Yes (3.61 GB headroom) |
| Qwen3.5-9B DoRA training, seq_len=1024 | 22.08 GB | No (OOM during backward) |
| Scorer (8.7GB) + wizard 14B (9.3GB) | ~19.2 GB + KV | No (exceeds with KV cache) |
| Scorer alone, NUM_PARALLEL=1 | ~9.3 GB | Yes |
| Scorer alone, NUM_PARALLEL=4 | ~10.5 GB | Yes (but no throughput gain) |

**Key insight:** Parallel inference (NUM_PARALLEL>1) provides zero throughput benefit on a single 7900 XTX with an 8B model. The GPU is memory-bandwidth-bound on autoregressive generation. Parallelism just adds contention overhead. This is a physics limitation, not a software limitation.

---

### 3.3 PEFT Meta Device Fix (ROCm-specific)

After `get_peft_model()`, you MUST:
```python
if hasattr(model, 'hf_device_map'):
    del model.hf_device_map
model = model.to("cuda:0")
```
Otherwise gradients crash during the backward pass on ROCm. This is not needed on CUDA.

---

### 3.4 Adapter Inference Is 30x Slower

Running inference through DoRA adapters on ROCm is ~30x slower than through a merged model. Always merge adapters before bulk evaluation or deployment. Phase 3 of train_v2.py handles this automatically.

---

### 3.5 Disk Space During Training

GGUF conversion needs ~25GB temporary space (full bf16 model + quantized output). Monitor disk before starting Phase 3 (merge + export). Current state: 55GB free, needs ~20GB temp = safe.

**Rule:** Keep disk below 85% usage before any training run. GGUF export failures from disk full are unrecoverable — you'd need to re-merge and re-export.

---

## Part 4: Evaluation Best Practices

### 4.1 Always Run Dual-Temperature Eval

The model behaves differently at temp=0.1 (deterministic) vs temp=0.6 (production). V2 showed +0.201 MAE gap between temperatures. Always eval at both and report both numbers.

```bash
python3 evaluate_scorer.py --model MODEL --eval-file holdout.jsonl --temperature 0.1
python3 evaluate_scorer.py --model MODEL --eval-file holdout.jsonl --temperature 0.6
```

---

### 4.2 Spearman Rho Alone Is Misleading

V2.1 had Spearman rho=0.539 (looks acceptable) but MAE=3.244 (catastrophic). The model preserved relative ordering while inflating every score by 3+ points. Always report both ranking metrics (Spearman) AND calibration metrics (MAE, within-1-point, direction accuracy).

---

### 4.3 Per-Band MAE Is More Informative Than Overall MAE

Overall MAE is dominated by the noise band (80%+ of examples). A model could score noise perfectly and fail completely on moderate/high bands while still showing a "good" overall MAE. Always report per-band breakdown:

| Band | What it measures |
|------|-----------------|
| Noise (0-2.5) | Can the model say "no"? |
| Tangential (2.5-4.0) | Low-relevance discrimination |
| Moderate (4.5-6.5) | Nuanced relevance judgment (hardest) |
| High (7.0-8.5) | Strong relevance detection |
| Critical (8.5-10) | Can the model say "critical"? (smallest sample) |

---

### 4.4 Smoke Test Before Full Eval

Before running a 46-minute full holdout eval, send 3-5 manual scoring prompts through Ollama and inspect raw output:

1. A clearly irrelevant article → should score 0-2
2. A clearly relevant article → should score 7-9
3. A borderline article → should score 4-6
4. Check for: think blocks before SCORE, Chinese output, format compliance, reasonable reasoning

This catches catastrophic failures (like V2.1's score inflation) in 30 seconds instead of 46 minutes.

---

### 4.5 Parse Failure Rate Is a First-Class Metric

Every parse failure is a lost data point in eval and a lost score in production. V2's 5-7.9% parse failure rate meant 75-119 articles per scan got no score. Track parse failures by cause:

| Cause | V2 Rate | Fix |
|-------|---------|-----|
| /nothink slash-command loops | 41% → 0% | Remove /nothink from Modelfile |
| Chinese output (力评分：) | ~5-7% | Chinese regex fallback in parser |
| Think block overflow | ~1-2% | Increase num_predict, strip think blocks |
| Format non-compliance | ~1% | Retry with simplified prompt |

Qwen3.5-9B achieved 0% parse failures on the full holdout — a generational improvement.

---

## Part 5: Project Management Lessons

### 5.1 Profile Your Data Before Setting Hyperparameters

Five lines of code checking token distribution would have saved 9+ hours per V2 training run:
```python
lengths = [len(tokenizer.encode(ex)) for ex in dataset]
print(f"Max: {max(lengths)}, P95: {percentile(lengths, 95)}, Mean: {mean(lengths)}")
```
Set seq_length to P99 + 10% buffer. Set num_predict to max_output_length × 1.5. Never use arbitrary defaults.

---

### 5.2 Pre-Training Reviews Should Block, Not Just Note

The cross-Claude review before V2.1 correctly identified that think_text was risky and that V1's CoT didn't help. These were noted as "open risks" rather than blockers. V2.1 then trained for 12+ hours and failed.

**New rule:** If a review flags a risk that could cause >2x MAE regression, it's a blocker. Run a small-scale experiment (1,000 examples, 30 minutes) to validate or reject the approach before committing to a full training run.

---

### 5.3 Cheap Experiments Before Expensive Ones

| Experiment | Cost | Time | Information value |
|-----------|------|------|-------------------|
| Score 300 articles with new provider | $0-0.10 | 10 min | Viable or eliminated |
| Train on 1,000 examples | $0 | 15 min | Catches catastrophic failures |
| Pull new base model and test on holdout | $0 | 30 min | Architecture upgrade worth pursuing? |
| Full training run (19K examples) | $0-10 | 3-12 hrs | Final quality measurement |
| Full Opus scoring of 36K examples | $50-70 | 24 hrs | Ground truth (do last) |

Always run experiments in this order: cheapest first. The $0 Qwen3.5-9B base model test (MAE 1.297) was more informative than any $50 training run.

---

### 5.4 Version Everything

Every training run should have:
- A git commit hash
- A training config dump (hyperparameters, data paths, base model)
- A prompt template hash (from prompt_version.py)
- The exact Modelfile used for serving
- Holdout eval results at both temperatures
- Per-band breakdown

Store all of this in training_pipeline.md. When something goes wrong six weeks later, you need to reconstruct exactly what was different.

---

### 5.5 Keep Rollback Artifacts

Always maintain:
- The previous version's GGUF (currently `v2_model_backup/v2_scorer.gguf`)
- The previous version's Modelfile
- The holdout eval JSONL (never modify, never delete)
- The holdout article IDs (never modify, never delete)

Rolling back should take 2 minutes: `ollama create stratos-scorer-v2 -f OldModelfile`, restart.

---

## Part 6: Forward-Looking Practices

### 6.1 Base Model Upgrades Are Free Performance

The Qwen3-8B → Qwen3.5-9B upgrade gave 16% MAE improvement with zero training. When a new model generation drops, test the base model on your holdout before doing anything else. It takes 30 minutes and costs nothing. If the base model alone beats your fine-tuned previous generation, you know the upgrade is worth pursuing.

Check quarterly: Qwen releases, Llama releases, Mistral releases. Run holdout eval. The landscape is moving fast.

---

### 6.2 Check Token Distributions After Every Data Change

After modifying stage4_prepare.py, adding new data sources, or changing prompt templates, re-run the token distribution check. A prompt template change that adds 100 tokens could push examples past your seq_length limit, causing silent truncation.

---

### 6.3 Validate Calibration on New Data Distributions

The isotonic calibration model was fitted on 283 examples from the V2 article distribution. When scoring articles from a different distribution (DB articles, RSS feeds, different time period), the calibration curve may not transfer perfectly. Budget $1-2 for a 20-30 sample spot-check against Opus whenever the article source changes.

---

### 6.4 Monitor Production Score Distribution

If the production score distribution shifts significantly from training (e.g., suddenly 50% of articles score 5-7 when training was 90% noise), something has changed — either the news landscape, the user's profile, or the model is drifting. Log score distributions per scan and alert on statistical shift.

---

### 6.5 Never Trust a Single Metric

The full eval dashboard should always include:
- MAE (calibration quality)
- Spearman rho (rank ordering quality)
- Direction accuracy (binary signal/noise classification)
- Within-1-point percentage (practical accuracy)
- Per-band MAE (where is the model weak?)
- Parse failure rate (operational reliability)
- Temperature sensitivity (MAE gap between temp=0.1 and temp=0.6)

A model that looks good on 5 of 7 metrics but fails on the other 2 (like V2.1: Spearman OK, MAE catastrophic) should not ship.

---

## Part 7: Checklists

### Pre-Training Checklist

- [ ] Token distribution checked — seq_length set to P99 + 10%
- [ ] Only ONE variable changed from previous version
- [ ] Holdout articles excluded from training data (verify at code level)
- [ ] No think_text in assistant messages (force short reason)
- [ ] Category field uses real labels (not "general")
- [ ] LANGUAGE line present in system prompt
- [ ] Token IDs are dynamically resolved (not hardcoded)
- [ ] AOTRITON env var is NOT set
- [ ] ROCR_VISIBLE_DEVICES=0 is set
- [ ] Disk space >50GB free
- [ ] Previous version's GGUF backed up
- [ ] Training config documented in training_pipeline.md

### Post-Training Checklist

- [ ] Smoke test: 3-5 manual prompts through Ollama, inspect raw output
- [ ] Full holdout eval at temp=0.1 AND temp=0.6
- [ ] Per-band MAE breakdown reviewed
- [ ] Parse failure rate <2%
- [ ] Compare to previous version on same holdout (apples-to-apples)
- [ ] Modelfile reviewed: num_predict=256, no /nothink, correct GGUF path
- [ ] prompt_version.py shows no template drift warnings
- [ ] Git commit with all results documented
- [ ] Previous GGUF kept as rollback

### Pre-Deployment Checklist

- [ ] Holdout MAE is better than current production model
- [ ] No regression on any individual band >20% vs previous version
- [ ] Parse failure rate is equal to or better than previous version
- [ ] Modelfile parameters match training conditions (temperature, stop tokens)
- [ ] config.yaml updated with new model name
- [ ] Ollama model registered and responding
- [ ] One full production scan completed successfully with new model
- [ ] Rollback command documented and tested

---

*This document should be updated after every training cycle with new lessons learned.*
