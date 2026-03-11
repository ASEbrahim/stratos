# StratOS v19 — Claude Code Implementation Prompt

Read CLAUDE.md first for project conventions.

## CONTEXT: What Is StratOS

StratOS is a self-hosted news intelligence platform. It scrapes articles, scores each one's
relevance to the logged-in user's profile (0-10), and displays a personalized dashboard.
The AI scorer is a fine-tuned Qwen3-8B model running locally via Ollama on an AMD 7900
XTX (24GB VRAM, ROCm). The platform is at ~/Downloads/StratOS/StratOS1/backend/.

The CRITICAL capability: the same article must get different scores for different users.
An Equate job posting = 9.5 for a Kuwait CPEG student, 0.5 for a Texas nurse. An NVIDIA
GPU launch = 9.0 for a Seoul gamer-investor, 1.0 for a Portland retiree. The model must
learn the METHOD of profile-aware scoring, not memorize patterns.

## CONTEXT: Why We're Here (v15-v18 Failure History)

Every model from v15-v18 was trained on answer-only data (SCORE: X.X | REASON: ...).
The model never learned the reasoning process, just memorized "noise → 1.5, else → 8.5."

- v15: Loss computed on ALL tokens (75% wasted on prompt prediction). PSR 31%, MAE 5.3.
- v16: Wrong TRL parameter (completion_only_loss doesn't work with messages column).
- v17: Disk full, stalled.
- v18: Currently training. First run with correct loss masking + cleaned data. Still answer-only.

The loss masking fix IS verified (dry run: initial loss 4.82 vs v15's 3.73, token accuracy 50%
vs 88%). But answer-only training is insufficient — research proves that rich explanation
traces are what differentiate successful distillation from failed imitation.

## WHAT v19 DOES DIFFERENTLY: Chain-of-Thought Distillation

Instead of training on "SCORE: 9.5 | REASON: career match", we train on the full
reasoning process inside Qwen3's native <think> blocks:

```
<think>
PROFILE: CPEG fresh graduate in Kuwait, tracks Equate and oil/gas companies.
CONTENT: Equate hiring CPEG graduates at Ahmadi facility with deadline.
ANALYSIS:
- Role/interest relevance: STRONG — CPEG position matches user's degree
- Location relevance: STRONG — Kuwait, Ahmadi facility
- Entity relevance: STRONG — Equate on tracked list
- Topic relevance: WEAK — engineering, not specifically AI/semiconductors
- Actionability: HIGH — deadline March 15, salary listed
- Source quality: OFFICIAL — company announcement
CALIBRATION: Strongest signal is tracked company hiring for exact degree in user's
city with deadline. Clear Critical range.
</think>

SCORE: 9.5 | REASON: Tracked company Equate hiring CPEG graduates in Kuwait with deadline
```

This is the PRISM framework: Profile → Recognize → Identify (weighted) → Score → Match.

Key principles:
- Score by STRONGEST SIGNAL, not by counting matching dimensions
- One perfect tracked-company-hiring-your-role match beats four vague connections
- Adaptive reasoning depth: noise gets 30-50 word think blocks, critical gets 120-180 words
- Score bands: Critical 9-10, High 7-8.9, Moderate 5.5-6.9, Low 2-5.4, Noise 0-1.9
- Never score exactly 5.0 (the "Forbidden 5.0")
- Source quality caps blog rumors at ~8.0; official sources can reach 9.0+
- Expired opportunities score Low regardless of match quality

## THE EXECUTION PLAN

There are 6 phases. Do NOT skip ahead. Complete each phase and report results before
proceeding to the next. Ask me before starting any phase that costs money (Batch API).

### PHASE 1: Define Training Profiles (10 total)

Create a file `data/profiles_v19.json` with 10 profile definitions. Each profile needs:
- profile_id (string)
- system_prompt (the full system prompt sent to the model during scoring)
- description (1-2 sentences for documentation)

6 career profiles:
1. kuwait_cpeg — CPEG fresh graduate in Kuwait. Tracks: Equate, SLB, Halliburton, KOC,
   KNPC. Interests: AI, semiconductors, embedded systems, quantum computing.
2. texas_nurse — Experienced ER nurse in Houston. Tracks: Houston Methodist, MD Anderson,
   HCA Healthcare. Interests: emergency medicine, trauma care, nursing certs, telehealth.
3. london_finance — Junior investment analyst in London. Tracks: Goldman Sachs, JPMorgan,
   Barclays, LSE. Interests: equity markets, fintech, derivatives, ESG investing.
4. munich_mecheng — Automotive engineer in Munich. Tracks: BMW, Siemens, Bosch, Continental.
   Interests: EVs, autonomous driving, Industry 4.0, materials science.
5. bangalore_ds — Fresh MSc graduate seeking first data science role in Bangalore. Tracks:
   Infosys, TCS, Wipro, Flipkart, Razorpay. Interests: ML, NLP, deep learning, MLOps, Python.
   NOTE: This profile is a FRESH GRADUATE, not experienced. Fix the seniority mismatch
   from previous training data.
6. dc_cybersec — Cybersecurity analyst in Washington DC. Tracks: CISA, NSA, CrowdStrike,
   Palo Alto Networks. Interests: threat intelligence, zero-trust, cloud security, CTF.

4 interest-driven profiles (these are NOT defined by career):
7. seoul_gamer_investor — CS sophomore in Seoul. Tracks: Nintendo, Valve, Epic Games,
   AMD, NVIDIA. Interests: game development, esports, GPU tech, tech stocks, cryptocurrency.
8. ankara_politics_physics — Government employee in Ankara. Tracks: no specific companies.
   Interests: geopolitics, Middle East policy, NATO, CERN, particle physics, LHC, Higgs boson.
9. portland_retired_space — Retired engineer in Portland. Tracks: no specific companies.
   Interests: NASA, SpaceX, JWST, exoplanets, astronomy, gardening, cooking.
10. lagos_student_entrepreneur — University student in Lagos, Nigeria. Tracks: Paystack,
    Flutterwave, Andela. Interests: fintech, mobile apps, African startup ecosystem,
    digital marketing, social media growth.

Each profile's system_prompt should follow this template:
```
You are a relevance scorer for [role description] in [location].
User context: [1-2 sentences about what they're doing/seeking]
Tracked companies: [list or "none specific"]
Tracked interests: [list]
Location: [city, country]
```

### PHASE 2: Generate Polarizing Articles (100 articles)

Create a file `data/polarizing_articles_v19.json` with 100 articles specifically designed
to be STRONGLY relevant to exactly 1-2 profiles and noise for all others.

Target distribution: ~10 articles per profile that are critical (9.0+) for that profile.

Examples of what these should look like:
- "Equate opens CPEG graduate program in Kuwait, applications close March 15" → 9.5 for
  kuwait_cpeg, 0.5 for everyone else
- "Houston Methodist ER expanding, hiring 50 travel nurses for summer 2026" → 9.0 for
  texas_nurse, 0.5 for everyone else
- "CERN announces first collisions at record energy in High-Luminosity LHC" → 8.5 for
  ankara_politics_physics, ~1.0 for most others
- "Epic Games releases Unreal Engine 6 with free student license" → 8.0 for
  seoul_gamer_investor, moderate for bangalore_ds, low for others
- "Nigeria Central Bank launches mobile fintech regulatory sandbox" → 9.0 for
  lagos_student_entrepreneur, 0.5 for most others
- "NASA publishes new JWST exoplanet catalog with 200 candidates" → 8.5 for
  portland_retired_space, 0.5 for most others
- "Barclays opens summer internship program for investment banking analysts" → 8.5 for
  london_finance, 0.5 for most others
- "BMW announces 500 new positions in autonomous driving division Munich" → 9.0 for
  munich_mecheng, 0.5 for most others
- "CrowdStrike releases critical zero-day advisory for federal agencies" → 9.0 for
  dc_cybersec, moderate for others in tech
- "Flipkart acquires Bangalore ML startup, hiring 200 data scientists" → 9.0 for
  bangalore_ds, 0.5 for most others

Also include ~10 "cross-profile" articles that should score differently for multiple profiles:
- "NVIDIA announces RTX 6090 — stock jumps 8%" → Critical for seoul_gamer_investor
  (investment + gaming), High for kuwait_cpeg (tech interest), moderate for others
- "Saudi Arabia announces $50B tech investment zone near Kuwait border" → High for
  kuwait_cpeg (regional), moderate for london_finance (investment angle), low for others
- "Global AI chip shortage worsens, affecting automotive and cloud sectors" → High for
  munich_mecheng, moderate for bangalore_ds, moderate for kuwait_cpeg

Each article should be a realistic headline + 2-3 sentence summary. Include a "source_type"
field: "official", "news", "blog", or "unknown".

### PHASE 3: Batch API Distillation

This is the most important phase. Send every article (490 existing multi-profile articles +
100 new polarizing articles) through Claude Opus via the Batch API with the CoT system
prompt for all 10 profiles.

**Before running the batch:**

1. Read the existing distill.py to understand the current distillation pipeline
2. Update the Opus system prompt to request chain-of-thought format. The FULL prompt is:

```
You are generating training data for a relevance scoring model. Your task is to score an
article's relevance to a specific user profile AND show your complete reasoning process.

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
CALIBRATION: Strongest signal is [identify it]. [Map to score range based on signal
STRENGTH, not match count]. [Adjust for timeliness/source quality if needed].
</think>

SCORE: X.X | REASON: [One concise sentence naming the strongest factor]

## Scoring Scale
- 9.0-10.0 (Critical): STRONG tracked entity/role match AND actionable AND timely
- 7.0-8.9 (High): STRONG match on a tracked dimension
- 5.5-6.9 (Moderate): WEAK matches or STRONG interest without actionability
- 2.0-5.4 (Low): Tangential connections only
- 0.0-1.9 (Noise): No meaningful relevance

## Reasoning Depth Rules
- NOISE (0.0-1.9): Under 50 words in think block
- STANDARD (2.0-8.9): 80-150 words
- CRITICAL (9.0-10.0): 120-180 words, confirm actionability + timeliness + source quality

## Rules
1. Score MUST reflect THIS SPECIFIC USER'S profile
2. Never score exactly 5.0
3. Score by STRONGEST signal, not dimension count
4. Match reasoning depth to difficulty — obvious noise gets short analysis
5. Actionability multiplier: deadline > trend > opinion
6. Location specificity multiplier: specific > regional > global
7. Source quality: official supports 9.0+; blog rumors cap ~8.0
8. Timeliness: expired opportunities score Low regardless
9. Interest-driven profiles: don't penalize for lack of "career match"
```

3. For each article, send it paired with each of the 10 profile system prompts
4. Use the Batch API (50% discount). Estimated cost: ~$7-10 total.
5. Validate every response:
   - Must contain <think>...</think> block
   - Must contain SCORE: X.X | REASON: format
   - Think block must be 15-200 words
   - Score must not be exactly 5.0
   - Reject and retry failures

**IMPORTANT:** Ask me for approval before submitting the batch. Show me the estimated
cost and a few sample requests first.

### PHASE 4: Training Data Preparation

1. Parse all Opus responses into training format:
```json
{"messages": [
  {"role": "system", "content": "[profile system prompt]"},
  {"role": "user", "content": "[article text]"},
  {"role": "assistant", "content": "<think>\n...\n</think>\n\nSCORE: X.X | REASON: ..."}
]}
```

2. Merge with any high-quality existing data (single-profile articles scoring >= 5.5 that
   were correctly formatted in previous training runs — optional, CoT data is primary)

3. Rebalance across score bands:
   - Target: ~25% noise/low (0-5.4), ~30% moderate (5.5-6.9), ~30% high (7-8.9), ~15% critical (9-10)
   - Use upsampling of underrepresented bands, not downsampling

4. Sort by curriculum difficulty: order examples by absolute distance from 5.0 (descending).
   Items scoring 0.5 and 9.5 appear first (easy cases). Items scoring 5.5 and 6.0 appear
   last (hard cases requiring nuanced judgment). This implements easy→hard curriculum
   learning.

5. Split 90/10 train/eval, stratified by profile AND score band

6. Save as data/training_v19_cot.jsonl and data/eval_v19_cot.jsonl

7. Report final stats:
   - Total examples
   - Distribution by profile
   - Distribution by score band
   - Average think block length by tier
   - Number of contrastive pairs (same article, different profiles, score diff > 3.0)

### PHASE 5: Training Pipeline Updates + Training

**Update train_lora.py:**

1. Keep the prompt/completion conversion with completion_only_loss=True (VERIFIED WORKING).
   The completion now includes the <think> block + SCORE line — all gets gradient.
   This is correct: the reasoning IS what we want the model to learn.

2. CRITICAL: Explicitly set eos_token='<|im_end|>' in the tokenizer config.
   Qwen3's tokenizer was silently changed to use <|endoftext|> as eos_token, but the
   chat template still uses <|im_end|>. Without this, the model won't stop generating.

3. Target ALL linear layers: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
   Research: attention-only LoRA underperforms by 5-15%.

4. Training parameters:
   - Base model: Qwen/Qwen3-8B (fresh from HuggingFace, NOT from v18 weights)
   - DoRA rank 16, alpha 32
   - BF16 precision
   - Learning rate: 1e-4 (lower than v15's 2e-4; DoRA converges slower)
   - Cosine scheduler with 5% warmup
   - Effective batch size: 32 (batch_size × gradient_accumulation)
   - 2 epochs (task complexity quadrupled; 1 epoch may not be enough)
   - Early stopping on eval loss, patience 3 evaluations
   - Gradient clipping: max_grad_norm=1.0 (prevents NaN loss, documented Qwen3 issue)
   - Checkpoint saves every 200 steps
   - load_best_model_at_end=True

5. Implement curriculum ordering: the training dataset should NOT be shuffled randomly.
   Load the pre-sorted data (sorted by difficulty in Phase 4) and disable shuffling in the
   DataLoader. If SFTTrainer forces shuffling, implement a custom Sampler that follows
   the pre-sorted order for the first epoch and shuffles for the second epoch.

6. Stop Ollama before training: sudo systemctl stop ollama

**Update Modelfile:**
```
FROM [path to merged GGUF]

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER num_predict 512
PARAMETER repeat_penalty 1.1
```

NOTE: Do NOT add /no_think or pre-fill <think></think>. The model should generate
think blocks naturally — the training data teaches it to do this.

**Update scorer_adaptive.py:**
- Set Ollama API parameters: temperature=0.6, top_p=0.95, top_k=20
  NEVER use temperature=0.1 or anything near greedy decoding. Research proves this
  causes degradation and infinite loops with Qwen3 in think mode.
- Strip <think>...</think> from response before parsing SCORE/REASON
- Optionally log think blocks for debugging
- Keep the case-insensitive SCORE/REASON regex from v15 fix

**Run training:**
- Run a 20-step dry run first. Compare initial loss to v18. Think-block completions are
  longer, so initial loss may be different. Verify loss is decreasing, no NaN.
- If dry run looks good, start full training. Expected: 6-8 hours for 2 epochs.
- Monitor: if eval_loss starts rising before epoch 2 completes, early stopping should kick in.

### PHASE 6: Validation

After training completes:

1. Register model in Ollama with the Modelfile above (with think mode enabled)

2. Sanity check first (3 articles × 3 profiles):
   - Article: "Equate hiring CPEG graduates in Kuwait, deadline March 15"
     * kuwait_cpeg: expect 9.0+ with think block mentioning tracked company + role match
     * texas_nurse: expect < 2.0 with short think block noting wrong field
     * seoul_gamer_investor: expect < 2.0 with short think block

   - Article: "NVIDIA RTX 6090 announced, stock up 8%"
     * seoul_gamer_investor: expect 8.5+ with think block noting tracked company + investment
     * kuwait_cpeg: expect 5.0-7.0 (tech interest but not career-critical)
     * texas_nurse: expect < 2.0

   - Article: "CERN completes High-Luminosity LHC upgrade ahead of schedule"
     * ankara_politics_physics: expect 8.0+ with think block noting tracked CERN interest
     * portland_retired_space: expect 4.0-6.0 (science-adjacent but not space)
     * texas_nurse: expect < 2.0

3. If sanity check shows meaningful profile separation (>3 point gap on the same article),
   run full Phase 3 validation suite (validate_phase3.py) with the 800 eval examples.

4. Additional CoT quality checks:
   - Read 20 think blocks manually. Are they following PRISM? Are mismatches identified?
   - Check that noise articles have SHORT think blocks (< 50 words)
   - Check that critical articles have THOROUGH think blocks (> 100 words)
   - Faithfulness spot-check: for 10 examples, does the score match the reasoning?

5. Report all metrics: PSR, MAE, Spearman ρ, format compliance, think block quality.

---

## IMPORTANT TECHNICAL NOTES

- ROCm GPU: Set HSA_OVERRIDE_GFX_VERSION=11.0.0 and
  PYTORCH_HIP_ALLOC_CONF=expandable_segments:True for all training commands
- Disk space: Verify > 30GB free before training (du -h --max-depth=1 ~/Downloads/)
- Delete old model artifacts: data/models/v15/, v16/, v17/, v18/ can all be removed
  (all failed validation or are superseded)
- GGUF conversion: After LoRA merge, convert to GGUF Q8_0 for Ollama deployment
  using llama.cpp's convert scripts
- The training data file from Phase 4 should be saved as data/training_v19_cot.jsonl
- Version tracking: This will auto-increment to the next version number. The Modelfile
  and all artifacts go in data/models/v[N]/

## FILES YOU'LL NEED TO MODIFY

- distill.py — Update Opus prompt for CoT format, add 10-profile support
- export_training.py — Preserve <think> blocks, add validation
- train_lora.py — eos_token fix, all-linear-layers targeting, curriculum ordering,
  2 epochs, learning rate 1e-4, gradient clipping
- data/models/v[N]/Modelfile — Think-enabled template with correct temperature
- processors/scorer_adaptive.py — temperature 0.6, think block stripping, logging
- validate_phase3.py — Add CoT quality metrics (think block length, faithfulness)

## FILES YOU'LL CREATE

- data/profiles_v19.json — 10 profile definitions
- data/polarizing_articles_v19.json — 100 targeted articles
- data/training_v19_cot.jsonl — Final training data
- data/eval_v19_cot.jsonl — Evaluation split
- STRATOS_V19_MASTER_PLAN.md — Already exists, for reference

## DO NOT

- Do NOT start training without showing me Phase 4 stats first
- Do NOT use temperature < 0.5 for Qwen3 think mode inference
- Do NOT use assistant_only_loss (Qwen3 template lacks {% generation %} markers)
- Do NOT train from v18 weights — use fresh Qwen3-8B base
- Do NOT include empty system prompts in training data (teaches model to ignore them)
- Do NOT skip the 20-step dry run before full training
- Do NOT delete training_merged.jsonl or distill_v2_eval.jsonl (still needed for comparison)
