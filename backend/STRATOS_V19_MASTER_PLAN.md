# StratOS v19 — Master Training Plan

**Date:** 2026-02-18
**Version:** 3.0 FINAL
**Status:** Pending v18 results, ready for immediate execution after

---

## 1. What StratOS Is

StratOS is a self-hosted strategic intelligence platform that scrapes news and scores every article's relevance to a specific user's profile. Not a generic news aggregator — a personal intelligence agent.

The critical capability: **any user with any combination of interests should get scores that reflect THEIR world, not a default world.** A Computer Engineering student in Kuwait sees Equate job postings as critical. A gaming student in Seoul who invests sees NVIDIA GPU launches as critical. A retired dad in Ankara who follows politics and the LHC sees CERN upgrades as critical. Same articles, different scores, because different people care about different things.

This is what makes StratOS unique. Without dynamic profile-aware scoring, it's just another RSS reader with categories.

---

## 2. Why v15–v18 Failed

Every model from v15–v18 was trained on answer-only data:

```
System: You are a relevance scorer for [profile]...
User: [article text]
Assistant: SCORE: 9.5 | REASON: Direct career match
```

The model saw the input and the output but never learned the reasoning process between them. It memorized "noise → 1.5, everything else → 8.5" — a degenerate two-mode function.

**Specific failures and their root causes:**

| Version | Failure | Root Cause |
|---------|---------|------------|
| v14 | Repetition loops, overfitting | 3 epochs + missing Ollama chat template |
| v15 | PSR 31%, MAE 5.3, bimodal collapse | Loss computed on all tokens (75% wasted on prompt prediction) |
| v16 | All scores -1.0 (parse failures) | Wrong TRL parameter (`completion_only_loss` vs `assistant_only_loss`) |
| v17 | Stalled (disk full) | <1MB disk space on Ubuntu |
| v18 | Pending | Loss masking fixed, data cleaned — first valid attempt, but still answer-only format |

The loss masking bug (v15/v16) is now fixed and verified via dry run (initial loss 4.82 vs v15's 3.73). But even with correct loss masking, answer-only training teaches pattern matching, not reasoning. The model needs to learn HOW to score, not just WHAT score to give.

---

## 3. The v19 Approach: Chain-of-Thought Distillation

### 3.1 Core Idea

Train on structured reasoning traces inside Qwen3's native `<think>` blocks. The model learns to:
1. Read the user profile and extract what matters
2. Classify the article content
3. Map overlaps between profile and article, weighting by signal strength
4. Calibrate a score based on the strongest signal, not by counting matches
5. Output the score with a concise reason

```
System: [profile description]
User: [article text]
Assistant: <think>
[Step-by-step reasoning — the HOW]
</think>

SCORE: 9.5 | REASON: [concise summary — the WHAT]
```

### 3.2 Research Backing

This isn't speculative. The evidence is overwhelming:

- **Microsoft Orca (2023):** Prior imitation models learned teacher style but not reasoning. Only "explanation tuning" — rich step-by-step traces — transferred actual reasoning capability.
- **DeepSeek-R1:** Distilling ~800K reasoning traces into Llama-3.1-8B produced a model scoring ~50% on AIME 2024 vs GPT-4o at 9.3%. Distillation outperforms RL on small models.
- **Orca 2 (2023):** A 13B model trained on reasoning traces outperformed Llama-2-Chat-70B — 5× size compression.
- **Faithfulness at 8B:** Anthropic research found smaller models produce MORE faithful reasoning than larger ones, because they genuinely need the scaffolding (inverse scaling effect). Qwen3-8B is weak enough that the reasoning will actually influence its outputs.
- **ECLIPTICA/CITA (2025):** Achieved 86.7% instruction-alignment efficiency on Llama-3.1-8B using contrastive instruction tuning — the exact approach StratOS needs for profile switching.
- **Qwen3-8B think mode:** 63% on AIME24 with thinking vs 24% without. The `<think>` block isn't cosmetic.

### 3.3 The Reasoning Framework: PRISM (v3)

**P**rofile → **R**ecognize → **I**dentify (weighted) → **S**core (signal-anchored) → **M**atch verdict

**Step 1 — Profile:** Summarize who this user is and what matters to them. Not just their job — their interests, tracked entities, location, career stage.

**Step 2 — Recognize:** Classify the article. What is it about? Where? Which companies/entities? How actionable is it? How credible is the source? Is it timely?

**Step 3 — Identify (WEIGHTED, not counted):** Map overlaps between profile and article. Classify each as STRONG / WEAK / NONE. Critical change from v1: **assess match intensity, not match count.** One perfect tracked-company-hiring-your-role match outweighs four vague industry-adjacent connections. The interactions and weights matter more than the tally.

**Step 4 — Score (signal-anchored):** The score is driven by the STRONGEST signal, not by counting how many dimensions match:

- **Critical (9.0–10.0):** At least one STRONG entity/role match AND directly actionable AND timely AND from a credible source. User must act.
- **High (7.0–8.9):** At least one STRONG match on a tracked dimension. Clearly valuable to this specific user.
- **Moderate (5.5–6.9):** WEAK matches or STRONG interest match without actionability. Worth seeing.
- **Low (2.0–5.4):** Tangential connections only. Filtered from main dashboard view.
- **Noise (0.0–1.9):** No meaningful connection whatsoever.

Score band alignment: The dashboard filters items < 5.0 from the executive summary. The Moderate band starts at 5.5. The forbidden 5.0 rule is maintained. There is no contradiction between training rubric and frontend behavior.

Multipliers that adjust scores within a band:
- Actionability (job posting with deadline > trend article)
- Location specificity ("Equate hiring in Kuwait" > "Oil industry hiring globally")
- Source quality (official announcement supports 9.0+; blog rumors cap at ~8.0)
- Timeliness (expired opportunities score Low regardless of match quality)

**Step 5 — Match verdict:** Final SCORE: X.X and a one-sentence REASON naming the strongest factor.

### 3.4 Adaptive Reasoning Depth

Research finding (ACoTD 2025): Naive CoT distillation HURTS performance on easy problems because small models produce unnecessarily long, incoherent reasoning for trivial decisions. Celebrity gossip for an engineer doesn't need 150 words of analysis.

Three tiers of reasoning depth in training data:

**Tier 1 — Quick dismiss (0.0–1.9):** 30-50 words. State the mismatch immediately.
```
<think>
PROFILE: CPEG graduate in Kuwait, tracks oil/gas companies.
CONTENT: Celebrity relationship drama.
ANALYSIS: Zero overlap — wrong field, wrong topic, no tracked entities. Clear noise.
</think>
SCORE: 0.5 | REASON: Celebrity entertainment irrelevant to engineering career
```

**Tier 2 — Standard analysis (2.0–8.9):** 80-150 words. Full PRISM with weighted assessment.

**Tier 3 — Deep justification (9.0–10.0):** 120-180 words. Explicitly confirm actionability, timeliness, source quality, and why each strong match warrants the critical score.

This teaches the model to match reasoning effort to decision difficulty — validated by Orca 2's strategy selection framework.

---

## 4. Training Profiles (10 Total)

### 4.1 Career Profiles (6)

| ID | Profile | Location | Industry |
|----|---------|----------|----------|
| 1 | kuwait_cpeg | Kuwait | Computer Engineering / Oil & Gas |
| 2 | texas_nurse | Houston, Texas | Healthcare / ER Nursing |
| 3 | london_finance | London, UK | Investment Banking |
| 4 | munich_mecheng | Munich, Germany | Automotive / Mechanical Engineering |
| 5 | bangalore_ds | Bangalore, India | Data Science / IT |
| 6 | dc_cybersec | Washington DC, USA | Cybersecurity / Government |

### 4.2 Interest-Driven Profiles (4)

| ID | Profile | Location | Primary Interests |
|----|---------|----------|-------------------|
| 7 | seoul_gamer_investor | Seoul, South Korea | Gaming, esports, tech stocks, GPU tech, crypto |
| 8 | ankara_politics_physics | Ankara, Turkey | Geopolitics, Middle East policy, CERN, LHC, particle physics |
| 9 | portland_retired_space | Portland, Oregon | Space exploration, NASA, JWST, astronomy, gardening, cooking |
| 10 | lagos_student_entrepreneur | Lagos, Nigeria | Fintech, mobile apps, African startup ecosystem, digital marketing |

Interest-driven profiles force the model to learn that relevance ≠ career utility. The Portland retiree has no "role match" or "career stage" — relevance is entirely driven by interest alignment. Without these profiles, the model learns a career-centric scoring heuristic that fails for anyone who doesn't define themselves by their job.

---

## 5. Opus Distillation Prompt

This is the system prompt sent to Claude Opus when generating chain-of-thought training data:

```
You are generating training data for a relevance scoring model. Your task is to score an article's relevance to a specific user profile AND show your complete reasoning process.

## Output Format

<think>
PROFILE: [1-2 sentence summary of who this user is and what matters to them]
CONTENT: [1-2 sentence summary of what this article is about]
ANALYSIS:
- Role/interest relevance: [STRONG/WEAK/NONE] — [brief explanation]
- Location relevance: [STRONG/WEAK/NONE] — [brief explanation]
- Entity relevance: [STRONG/WEAK/NONE] — [brief explanation]
- Topic relevance: [STRONG/WEAK/NONE] — [brief explanation]
- Actionability: [HIGH/MEDIUM/LOW/NONE] — [brief explanation]
- Source quality: [OFFICIAL/NEWS/BLOG/UNKNOWN] — [brief explanation]
CALIBRATION: Strongest signal is [identify it]. [Map to score range based on signal STRENGTH, not match count]. [Adjust for timeliness/source quality if needed].
</think>

SCORE: X.X | REASON: [One concise sentence naming the strongest factor]

## Scoring Scale

- 9.0-10.0 (Critical): At least one STRONG tracked entity/role match AND directly actionable AND timely. User must act now.
- 7.0-8.9 (High): At least one STRONG match on a tracked dimension. Clearly valuable to this specific user.
- 5.5-6.9 (Moderate): WEAK matches or STRONG interest match without actionability. Worth knowing.
- 2.0-5.4 (Low): Tangential or indirect connections only.
- 0.0-1.9 (Noise): No meaningful relevance to this user's profile.

## Reasoning Depth Rules

- NOISE (0.0-1.9): Keep think block under 50 words. State the mismatch and move on.
- STANDARD (2.0-8.9): Full analysis, 80-150 words.
- CRITICAL (9.0-10.0): Thorough justification, 120-180 words. Explicitly confirm actionability, timeliness, source quality.

## Critical Rules

1. The score MUST reflect THIS SPECIFIC USER'S profile. The same article scored for different profiles MUST produce different scores when their dimensions differ.
2. Never score exactly 5.0 — commit above or below.
3. Score based on the STRONGEST signal, not by counting matching dimensions. One perfect tracked-company-hiring-your-role match beats four vague industry-adjacent connections.
4. Be honest about mismatches. If it's obvious noise, say so in 20 words, not 150.
5. Actionability is a multiplier: job posting with deadline > general trend article.
6. Location specificity is a multiplier: "Equate hiring in Kuwait" > "Oil industry hiring globally."
7. Source quality matters at the high end: official announcements support 9.0+; blog rumors cap at ~8.0.
8. Timeliness: expired opportunities (closed applications, past events) score Low regardless of match quality.
9. For interest-driven profiles with no career dimension, relevance is determined by interest alignment and source quality alone — do not penalize for lack of "career match."
```

---

## 6. Training Data Pipeline

### Phase 1: Article Collection

**Source A — Existing multi-profile articles (490 articles):**
These already exist in `training_merged.jsonl`. They appeared across all 8 original profiles. Re-score them through Opus with the new CoT prompt across all 10 profiles.

**Source B — 100 polarizing articles (NEW):**
Generate/curate articles designed to be STRONGLY relevant to exactly 1-2 profiles and irrelevant to all others. This directly addresses the weak contrastive signal identified in validation.

Examples:
- "Equate opens CPEG graduate program in Kuwait" → 9.5 for kuwait_cpeg, 0.5 for everyone else
- "Houston Methodist ER department expanding, hiring travel nurses" → 9.0 for texas_nurse, 0.5 for everyone else
- "CERN announces High-Luminosity LHC upgrade completion" → 8.5 for ankara_politics_physics, 0.5 for most others
- "Epic Games announces Unreal Engine 6 with free student license" → 8.0 for seoul_gamer_investor, moderate for bangalore_ds
- "Nigeria Central Bank launches mobile fintech sandbox" → 9.0 for lagos_student_entrepreneur, 0.5 for most others
- "New NASA exoplanet catalog released via JWST data" → 8.5 for portland_retired_space, 0.5 for most others

**Total: ~590 articles × 10 profiles = ~5,900 training examples**

### Phase 2: Batch API Distillation

Send all article-profile pairs through Claude Opus via Batch API with the CoT system prompt.

Each call:
- Input: ~700 tokens (system profile + article)
- Output: ~200 tokens (think block + score)
- Total: ~5,900 calls
- Batch API (50% discount): **~$7-10**

Post-processing validation:
- Reject responses where think block > 200 words or < 15 words
- Reject responses missing SCORE: or REASON: format
- Reject responses scoring exactly 5.0
- Validate that Tier 1 articles (score < 2.0) have short think blocks and Tier 3 (> 9.0) have thorough ones

### Phase 3: Data Preparation

1. Clean and format all CoT-distilled data into messages format
2. Merge with any surviving high-quality examples from existing data (single-profile articles scoring >= 5.5 that already have good format)
3. **Rebalance across score bands:**
   - Target: ~25% noise/low (0-5.4), ~30% moderate (5.5-6.9), ~30% high (7.0-8.9), ~15% critical (9.0-10.0)
   - This reflects real-world article distribution while ensuring mid-range examples (where reasoning matters most) are well-represented
4. **Every training example MUST have a non-empty, varied system prompt.** Research shows empty system prompts teach the model to ignore them at inference time. This is a hard requirement.
5. Fix the Bangalore seniority mismatch (profile says "experienced professional" but context says "fresh MSc graduate")
6. Split 90/10 train/eval, stratified by profile and score band
7. **Sort by difficulty for curriculum learning** (see Phase 5)

### Phase 4: Pipeline Code Changes

**`distill.py` changes:**
- Update Opus system prompt to the CoT version (Section 5)
- Add reasoning_tier field based on score
- Add post-processing validation (word count, format check)
- Score across all 10 profiles per article

**`export_training.py` changes:**
- Preserve `<think>` blocks in assistant content (currently strips them)
- Add tiered depth validation

**`train_lora.py` changes:**
- Keep prompt/completion conversion with `completion_only_loss=True` (verified working)
- The completion now includes `<think>` block + SCORE — all gets gradient signal
- **Explicitly set `eos_token='<|im_end|>'`** — Qwen3's tokenizer was silently updated to use `<|endoftext|>`. Without this, model won't stop generating.
- Target ALL linear layers (q/k/v/o_proj + gate/up/down_proj), not just attention. Research: attention-only underperforms by 5-15%.
- Learning rate: 1e-4 (slightly lower for DoRA's slower convergence) with cosine annealing, 5% warmup
- Implement curriculum data ordering (see Phase 5)

**`Modelfile` changes:**
- Remove `/no_think` suppression
- Remove pre-filled `<think></think>` from assistant preamble
- Let model generate think blocks naturally
- Set proper stop tokens: `<|im_end|>` and `<|endoftext|>`

**`scorer_adaptive.py` changes:**
- **Set inference parameters: temperature=0.6, top_p=0.95, top_k=20**
- **NEVER use greedy decoding (temperature ~0) with Qwen3 think mode** — causes degradation and endless repetition loops. This likely contributed to v14/v15 issues.
- Strip `<think>...</think>` from response before parsing SCORE/REASON
- Log think blocks for debugging/transparency
- Update SCORE/REASON regex (already case-insensitive from v15 fix)

### Phase 5: Curriculum Training

Research shows easy→hard ordering consistently outperforms random ordering for SFT (1-5% gains — modest but free).

**Implementation:** Sort training data by absolute distance from the decision boundary (5.0). Articles scoring 0.5 and 9.5 are "easy" (obvious noise/obvious hit). Articles scoring 5.5 and 6.0 are "hard" (require nuanced judgment). The sort is descending by distance from 5.0.

**Effect:** The first ~30% of training steps are easy examples with clear PRISM outcomes. The model learns the reasoning framework on unambiguous cases first. Then it encounters the hard mid-range cases where the reasoning actually matters.

**Training parameters:**
- Base model: Qwen3-8B (fresh from HuggingFace, NOT incremental from v18)
- DoRA rank 16, alpha 32
- BF16 precision
- Target modules: all linear layers
- Learning rate: 1e-4, cosine with 5% warmup
- Effective batch size: 32
- **2 epochs** with checkpoint saves every 200 steps
- Early stopping on eval loss, patience 3
- Gradient clipping: 1.0 (prevents NaN loss, documented Qwen3 issue)
- Hardware: AMD 7900 XTX 24GB, ROCm

**Why 2 epochs:** The task complexity quadrupled (50-token completions → 200-token structured reasoning). 1 epoch may not be enough for the model to internalize the PRISM framework. Research says 1-3 epochs for SFT, with 2 as a reasonable experiment for reasoning traces. Early stopping prevents overfitting.

### Phase 6: Validate

**Standard metrics (same Phase 3 suite):**
- PSR (Profile Sensitivity Rate): target > 80%
- MAE (Mean Absolute Error vs Opus): target < 1.5
- Spearman ρ (rank correlation): target > 0.80
- Anti-memorization: delta < 2.0 on held-out articles
- Format compliance: ~100%

**New CoT-specific metrics:**

- **Think block quality audit:** Manually review 50 think blocks across score ranges. Categorize failures:
  - Wrong profile reading (model misidentified who the user is)
  - Wrong article classification (model misunderstood the content)
  - Wrong overlap assessment (saw a match that isn't there, or missed one)
  - Wrong calibration (correct reasoning → wrong score mapping)
  - Unfaithful reasoning (plausible reasoning that doesn't match the actual score)

- **Faithfulness test:** Truncate think blocks at various points on 100 examples. Check if scores change. If >70% of scores remain unchanged when reasoning is removed, the CoT is cosmetic and not actually influencing outputs.

- **Interest-profile validation:** Separately measure PSR and MAE on interest-driven profiles vs career profiles. If interest profiles underperform, add more interest-profile training data.

- **Adaptive depth check:** Verify noise articles get short think blocks (~30-50 words) and critical articles get thorough ones (~120-180 words).

---

## 7. Inference Architecture

### 7.1 Short-term: Direct CoT Scoring

Every article goes through the CoT model with full `<think>` reasoning. Simpler to implement, fully debuggable.

**Estimated latency:** 200 tokens output × ~30-60 tokens/sec on 7900 XTX = ~3-7 seconds per article. For 200 articles per cycle: ~10-23 minutes. Acceptable for a 15-60 minute scrape cycle.

### 7.2 Medium-term: Two-pass Cascading

If latency becomes a problem as article volume grows:
1. **Fast pass:** Score all articles using base model with simple prompt (no think, just SCORE/REASON). ~1 second per article.
2. **Reasoning pass:** Re-score only articles where fast pass scored 4.0–8.0 (the ambiguous zone) using CoT model with full thinking.
3. Accept fast-pass scores for < 4.0 (obvious noise) and > 8.0 (obvious hits).

Expected: ~60-70% fast-scored, ~30-40% get full CoT. Total latency roughly doubles instead of 7.5×.

### 7.3 Future: System 2 → System 1 Distillation

Research (Yu et al. 2024) shows for scoring/classification tasks, you can:
1. Run full CoT on all data
2. Filter with self-consistency (multiple samples, keep agreements)
3. Fine-tune a non-CoT model on the CoT model's outputs

The distilled non-CoT model outperformed both original models on human agreement scores. This is the endgame for inference speed — but requires v19 to work first.

---

## 8. Qwen3-8B Known Issues

| Issue | Severity | Mitigation |
|-------|----------|------------|
| Tokenizer eos_token silently changed to `<\|endoftext\|>` | **Critical** | Explicitly set `eos_token='<\|im_end\|>'` |
| `assistant_only_loss` incompatible with Qwen3 template | High | Use prompt/completion format + `completion_only_loss=True` (already fixed) |
| Greedy decoding causes loops in think mode | **Critical** | Always use temp=0.6, top_p=0.95, top_k=20 |
| NaN loss during training | Medium | Lower learning rate (1e-4), gradient clipping (1.0) |
| 8-bit loading fails | Medium | Use 4-bit for training, Q8_0 GGUF for Ollama inference |
| Multi-turn template bug with think tags | Low | StratOS uses single-turn only |
| Context-parametric inversion (model ignores prompts after too much training) | Medium | 2 epochs max, monitor eval loss, early stopping |

---

## 9. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Opus generates inconsistent CoT format | Low | Strict format in prompt + post-processing validation + rejection |
| Think blocks too verbose (slow training) | Medium | Tiered depth caps: 50/150/180 words. Trim outliers. |
| Correct reasoning but wrong final score | Medium | More examples in the miscalibrated range; explicit calibration anchors in system prompt |
| Wrong reasoning but right score (unfaithful) | Medium | Faithfulness test (truncation); FRODO-style DPO if persistent |
| Repetition loops in think blocks | Low | Never greedy decode; temp=0.6; clean training data |
| Interest profiles underperform career profiles | Medium | Separate validation; rebalance if needed |
| Model ignores system prompts | Low | Every training example has varied, meaningful system prompts |
| CoT latency too slow for scrape cycle | Medium | Two-pass cascading; eventual System 2→1 distillation |
| 2 epochs causes overfitting | Medium | Early stopping on eval loss with patience 3; checkpoint saves |

---

## 10. Cost & Timeline

| Phase | Time | Cost |
|-------|------|------|
| Define 4 interest-driven profiles | 15 min | $0 |
| Generate 100 polarizing articles | 30 min | $0 |
| Batch API distillation (~5,900 examples) | 1-4 hours | ~$7-10 |
| Data prep + curriculum sorting + pipeline code changes | 1-2 hours | $0 |
| Training (2 epochs, ~5,900 examples) | 6-8 hours | electricity |
| Validation + faithfulness testing | 1 hour | $0 |
| **Total** | **~10-16 hours elapsed** | **~$7-10** |

---

## 11. Success Criteria

**v19 is a success if:**
- PSR > 60% (a significant jump from v18's baseline; 80% is the stretch goal)
- MAE < 2.5 (within ~2.5 points of Opus on average)
- Think blocks show coherent PRISM reasoning (manual audit of 50 examples)
- The model differentiates profiles on the same article (sanity check: Kuwait CPEG vs Texas nurse, same article, >3 point gap)
- Interest-driven profiles work (Portland retiree scores JWST article > 7.0 despite no career match)

**v19 is a partial success if:**
- PSR 40-60% and MAE 2.5-3.5 (clear improvement over v18 but not passing thresholds)
- Think blocks show correct reasoning with calibration errors → fix with more mid-range examples
- Next step: targeted data augmentation for weak score bands

**v19 is a failure if:**
- PSR < 40% or MAE > 4.0 (minimal improvement)
- Think blocks show repetition, echolalia, or copied input text
- Next step: investigate base model capability with prompt-only approach; consider different base model

---

## 12. Rollback Plan

If CoT distillation fails entirely:
1. The prompted base Qwen3-8B with a strong system prompt (PRISM instructions in the prompt, no fine-tuning) serves as the fallback scorer
2. The PRISM framework is still valuable as a prompt engineering tool even without fine-tuning
3. The Opus-distilled CoT dataset has value as evaluation data regardless
4. The platform (dashboard, auth, scraping, charting) works independently of the scorer
