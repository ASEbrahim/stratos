# StratOS Scorer — Complete Session Handoff (March 7, 2026)

**Purpose:** Self-contained context document for the next Claude session. Covers everything discussed, decided, and in-progress across a full-day session reviewing V2.1 training results, validating scoring providers, and launching the V2.2 article expansion pipeline.

**Project root:** `~/Downloads/StratOS/StratOS1/`
**Backend:** `~/Downloads/StratOS/StratOS1/backend/`

---

## 1. What Is StratOS?

StratOS is a personalized strategic intelligence dashboard. It fetches news from multiple sources (RSS, DuckDuckGo, Serper/Google), scores them for relevance using a local Ollama-hosted fine-tuned LLM (Qwen 3 8B), and presents a prioritized feed via a web UI. The scorer is the core intelligence layer — during a scan (~300 articles), each article is scored 0-10 for relevance to the user's profile. Built by a Computer Engineering student in Kuwait. 45,000+ lines across 40+ modules.

**Three Ollama Models:**
- `stratos-scorer-v2` (8.7GB Q8_0) — fine-tuned scoring model (PRODUCTION — current best)
- `qwen3:30b-a3b` (~18GB) — inference model for strat agent chat and market analysis
- `qwen3:14b` (~9GB) — wizard model for briefings, profile generation

---

## 2. Hardware & Environment

- **AMD RX 7900 XTX** (24GB VRAM, gfx1100/Navi 31), Ryzen 7 7800X3D, 30GB RAM
- **ROCm 6.2** (NOT CUDA — this affects everything)
- 1TB SSD at ~92% full — monitor disk space before training

### CRITICAL Environment Variables
```bash
ROCR_VISIBLE_DEVICES=0          # MUST set — excludes 512MB 7900X3D iGPU
HSA_OVERRIDE_GFX_VERSION=11.0.0 # MUST set — required for ROCm gfx1100
```

### DO NOT SET
```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1  # Causes NaN gradients — DESTROYS training
```

### VRAM Limits
- Max training set: ~19,000 examples before OOM
- batch_size=2, gradient_accumulation=8 (effective batch = 16)
- Gradient checkpointing (use_reentrant=False) required

---

## 3. Scorer Version History — Complete Results

### V1 (Early)
- Chain-of-Thought reasoning, 5,679 examples, 10 profiles
- **Holdout MAE: 1.553**

### V2 (February 2026) — CURRENT PRODUCTION
- Short reason (~172 chars), 18,502 examples, 30 profiles, 813 articles
- DoRA rank 16, alpha 32, lr=1e-5, 1 epoch, cosine schedule
- **Contaminated MAE: 0.393** (direct PyTorch bf16 — measures memorization)
- **Corrected Holdout MAE: 1.544** (Ollama Q8_0, fixed Modelfile, temp=0.1)
- **Holdout temp=0.6: 1.745** (production runs at 0.6)
- Parse failures: 5% (temp=0.1), 7.9% (temp=0.6)
- Band MAE: noise=1.384, tangential=2.025, moderate=2.740, high=2.688, critical=1.111

### V2.1 (March 6-7, 2026) — FAILED, DO NOT USE
- Added think_text (843 chars avg), 47 profiles, 19,000 examples
- **Holdout MAE: 3.244** — catastrophic regression
- **Direction accuracy: 53.5%** — near random
- Root cause: model inflates ALL scores to ~8-9 range. Cannot say "no."
- Think_text trained the model to justify, not evaluate
- Noise band MAE: 3.358 (predicts 3.4 points too high on average)
- High/critical bands decent (MAE 1.3) — relevance inflation everywhere
- **Decision: V2.1 is dead. Ship V2. Drop think_text permanently.**

### Key Insight: 1.544 ≈ 1.533
The corrected V2 baseline (1.544) barely differs from the broken V2 (1.533). Format drift (/nothink, num_predict=2048) accounted for essentially nothing. The generalization gap IS the entire problem. The model memorized 813 articles (0.393 contaminated) and cannot generalize. **Article expansion is the only thing that matters.**

### Reasoning Format — SOLVED QUESTION
| Version | Format | Holdout MAE |
|---------|--------|-------------|
| V1 | CoT (long) | 1.553 |
| V2 | Short reason (172 chars) | 1.544 |
| V2.1 | Think_text (843 chars) | 3.244 |

Short reasons ≈ CoT on holdout. Think_text catastrophically worse. **Never revisit think_text/CoT for this 8B model.**

---

## 4. Training Configuration (Proven V2 Config)

```
Method:           DoRA (NOT plain LoRA)
Rank:             16
Alpha:            32
Dropout:          0.05
Target modules:   q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Batch size:       2
Gradient accum:   8 (effective batch = 16)
Learning rate:    1e-5, cosine schedule, 5% warmup
Epochs:           1
Max seq length:   1024 tokens
Loss:             completion_only (only assistant tokens)
Optimizer:        AdamW, bf16
Speed:            ~38.5s/step, ~12.5 hours for ~1188 steps
```

### Dual Weighting System
| Band | Score Range | Sampler Weight | Loss Weight |
|------|-----------|---------------|-------------|
| noise | 0-2.5 | 0.5x | 0.5x |
| tangential | 2.5-4.0 | 1.0x | 1.0x |
| moderate | 4.5-6.5 | 1.5x | 1.5x |
| high | 7.0-8.0 | 2.0x | 2.0x |
| critical | 8.5-10.0 | 3.0x | 3.0x |

---

## 5. Modelfile Configuration (FIXED)

```
FROM .../v2_model_backup/v2_scorer.gguf
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER num_predict 256
PARAMETER repeat_penalty 1.1
```

**Rules:**
- `{{- end }}` MUST be on its own line, `<|im_start|>user` on next line
- Do NOT add `/nothink` — causes 41% parse failures
- `num_predict` MUST be 256, not 2048
- Let Qwen3 produce natural empty think blocks — they get stripped

---

## 6. Scoring Provider Validation Results

### DeepSeek V3.2 — ELIMINATED
- **Raw MAE vs Opus: 1.521** (threshold: < 1.0) — FAILED
- **Calibrated (isotonic LOO): 1.095** — still above threshold
- Spearman: 0.882, Parse failures: 0%
- Fatal flaw: middle-range collapse. Only uses ~6 distinct score levels. Score 2.5 maps to Opus 2.0-8.5. Calibration cannot recover lost information.
- 6-8 band catastrophic: MAE 2.932, bias -2.732
- Student/junior profiles especially bad — DeepSeek applies strict "direct match" logic
- Report: `backend/data/v2_pipeline/DeepSeek_V3.2_Validation_Report.md`
- Script: `backend/data/v2_pipeline/deepseek_validation.py`

### Gemini 3 Flash — SELECTED PROVIDER
- **Raw MAE vs Opus: 1.090**
- **Calibrated (isotonic LOO): 0.893** — PASSES 1.0 threshold
- Spearman: 0.922, Parse failures: 5.7%, Within 1.0: 70.3%
- Uses 15+ distinct score levels — no middle-range collapse
- Bias is relatively flat across bands (-0.2 to -1.1) — correctable
- 8-10 band MAE: 0.692 (2.4x better than DeepSeek's 1.672)
- 6-8 band still weakest: 1.691 raw, 1.353 calibrated
- Parse failures fixable: bump max_output_tokens to 1024, add retry logic
- Pricing: $0.50/$3.00 per MTok (Gemini 3 Flash), free tier: 1,000 RPD
- Report: `backend/data/v2_pipeline/Gemini_3_Flash_Validation_Report.md`
- Script: `backend/data/v2_pipeline/gemini_validation.py`
- Calibration model: fit `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')` on 283 validation points

### Cost Comparison for 36K scores
| Provider | Cost | MAE (calibrated) |
|----------|------|-------------------|
| Claude Opus (standard) | $50-70 | 0.0 (baseline) |
| Claude Opus (Batch 50% off) | $25-35 | 0.0 (baseline) |
| Gemini 3 Flash (paid) | $3-6 | 0.893 |
| DeepSeek V3.2 | ~$12 | 1.095 (above threshold) |

---

## 7. V2.2 Article Expansion — ACTIVE PIPELINE

### Status: Phase 2 (Gemini Scoring) IN PROGRESS

A Claude Code instance is running the V2.2 expansion pipeline. The script is at `backend/data/v2_pipeline/v22_expansion.py`.

### Pipeline Overview
1. **Phase 1 (DONE):** Extracted 2,000 articles from DB, stratified by category, excluding holdout 50 and existing 813
2. **Phase 2 (IN PROGRESS):** Scoring 2,000 articles × 25 profiles = 50,000 pairs with Gemini 3 Flash
3. **Phase 3 (pending):** Apply isotonic calibration to all scores
4. **Phase 4 (pending):** Prepare training data with `stage4_prepare.py` fixes
5. **Phase 5 (pending):** Train on 7900 XTX (~12.5 hours)

### Phase 2 Status
- Google API: Paid Tier 1, 10,000 RPD limit, 1,000 RPM limit
- ~8,286 scores completed on first day before hitting RPD
- Script has `--resume` support, checkpoints every 500 scores
- ETA: ~4.8 days at 9,500/day (conservative)
- API Key: in `.env` as `GOOGLE_API_KEY` (also hardcoded in launch command)
- 25 profiles selected (down from 50 to fit ~5 days of RPD limits)
- 15 concurrent workers, ~180 RPM

### Daily Command (run after midnight PT RPD reset)
```bash
GOOGLE_API_KEY="AIzaSy..." python3 data/v2_pipeline/v22_expansion.py --phase 2 --resume
```

### Key V2.2 Decisions (LOCKED)
| Decision | Resolution |
|----------|-----------|
| Scoring provider | Gemini 3 Flash + isotonic calibration (MAE 0.893) |
| Profiles | 25 (initially 50, reduced for RPD limits) |
| Articles | 2,000 stratified by category from 17,394 DB articles |
| Think_text | Forced OFF — V2.1 proved catastrophic |
| Category field | Fixed: use real category, not "general" |
| Noise cap | 9,000 examples (47% of 19K) |
| Training format | Short reason: `SCORE: X.X | REASON: brief explanation` (~170 chars) |
| Holdout protection | 50 article IDs excluded from expansion pool |
| User feedback | 2,631 entries at 1.0x weight (single user — Ahmad/CPEG) |
| Sparse filter | Updated to handle Gemini data (no think_tokens field) |

### stage4_prepare.py Fixes Needed for V2.2
1. **Force short reason** — never use think_text regardless of availability
2. **Real category labels** — not "general" (bug fix)
3. **Handle Gemini-scored data** — no think_tokens/think_text fields, don't filter on those
4. **LANGUAGE line** — add to system prompt for alignment with inference

### Calibration
- Isotonic regression fitted on 283 (Gemini, Opus) validation pairs
- Save as both pickle and JSON lookup table
- Tag with model version + prompt hash + validation date
- Spot-check: score 20-30 new articles against Opus ($1-2) after first Gemini batch to verify calibration transfers to new article distribution
- If spot-check MAE > 1.2, re-fit isotonic with new points added

---

## 8. V2.1 Post-Mortem — WHY IT FAILED

1. **Think_text trained the model to justify, not evaluate.** Long REASON text (843 chars avg) taught model to generate plausible arguments for relevance. Score calibration collapsed.
2. **Score inflation is systematic, not random.** Noise articles (GT 0-2) predicted 5-9. Spearman rho nearly unchanged (0.539 vs 0.555) — ordering preserved, absolute calibration destroyed.
3. **Chinese output leak.** Some V2.1 outputs in Chinese. Agent-scored profiles may have triggered Qwen3's multilingual tendencies.
4. **Multiple confounders changed simultaneously.** Think_text + 17 new agent profiles + different data distribution. Impossible to isolate cause.
5. **The pre-training cross-Claude review predicted this.** It flagged think_text as unproven and noted V1 CoT got 1.553. The review was right.

### Lessons (DO NOT REPEAT)
- **Change ONE variable at a time.** V2.2 changes articles only, keeps V2's proven format.
- **Never include untested data sources.** Agent profiles should be validated against Opus first.
- **Short reason > long reason for distillation on 8B model.** Not enough capacity for nuanced reasoning.
- **Spearman rho alone is misleading.** Always check calibration metrics (MAE, within-1-point) alongside ranking.
- **Think_text/CoT is dead for this model architecture.** V1 (CoT) → 1.553. V2 (short) → 1.544. V2.1 (think_text) → 3.244.

---

## 9. Feature Roadmap (Prioritized)

Full details in `features_roadmap.md` (project knowledge). Summary:

### Tier 1: Build Now
0. evaluate_scorer.py fixes — **DONE** (num_predict=512, --temperature flag)
1. Agent search_feed tool (~2h) — lets users query feed history
2. OLLAMA_NUM_PARALLEL benchmark (~15 min) — gates parallel scoring decision
3. VRAM coexistence test (~2 min) — scorer 8.7GB + 14b 9GB in 24GB?
4. Kill wasted Serper queries — cross-entity combos return 0 results ~30% of time

### Tier 2: Build Soon
5. Critical vetting layer (9.5+ articles) — specific checklist, `verified_critical` flag
6. RSS feed configuration — free data, extra_feeds.py has catalogs
7. Parallel LLM scoring — after benchmark confirms gain
8. Prompt template version-pinning — detect training/inference drift
9. Feed density view — expanded critical cards, single-line noise

### Tier 3: Build Later
10. Universal garbage classifier — TF-IDF + scikit-learn, 15-20% filter rate
11. Source reputation filter — per (domain, profile_category) noise probability
12. Streaming score-as-articles-arrive pipeline
13. Scheduled scans + email notifications (NOT now — burns Serper credits)
14. Chart initial zoom — `setVisibleLogicalRange()`, ~5 lines
15. Export/data portability

### Key Design Decisions
- `verified_critical` flag > synthetic 10.0 — don't pollute score scale
- Same-model vetting > different-model — 14b/30b don't know scoring rubric
- Staleness in vetting: fetched_at > 7 days = fail
- Per-profile source reputation > global
- Benchmark before restructuring parallel scoring
- RSS value = free data, not time savings

---

## 10. Known Pitfalls & Gotchas

### Training
- PEFT Meta Device Fix: After `get_peft_model()`, delete `hf_device_map` and force `.to("cuda:0")`
- Adapter inference is 30x slower on ROCm — always merge before eval/deployment
- Qwen3 8B bf16 merge needs GPU VRAM (32GB RAM not enough). Use layer-by-layer shard saving.
- Disk space: 1TB SSD at 92% full. GGUF conversion needs ~25GB temp space.

### Inference
- Do NOT set `"think": False` in Ollama API — Qwen3 ignores it, leaks reasoning as plain text
- Do NOT use `/nothink` in Modelfile — causes 41% parse failures
- Chinese output from Qwen3: ~5-7% of outputs. Parser handles `评分: X.X` format.
- evaluate_scorer.py uses /api/chat vs production's /api/generate — acceptable for now

### Format Alignment
- Training uses `Category: general` → inference uses real categories (bug, being fixed in V2.2)
- Training lacks LANGUAGE line present in inference (minor)
- `stage4_prepare.py` has think_text fallback that MUST be removed for V2.2

---

## 11. Data Landscape

| Dataset | Count | Notes |
|---------|-------|-------|
| Articles in DB | 17,394 | Unique articles with summaries |
| Articles used for V2 training | 813 | Scored by Opus |
| Untapped articles | ~16,500 | Available for expansion |
| V2.2 expansion articles | 2,000 | Selected, stratified by category |
| Total scored examples (V2) | 39,071 | 30 Opus profiles + 17 agent profiles |
| Opus-scored examples | 24,387 | 813 articles × 30 profiles |
| Agent-scored examples | 14,684 | 19 profiles (formulaic reasoning — quality concern) |
| Training examples | 19,000 | VRAM cap |
| Holdout eval | 1,500 | 50 unseen articles × 30 profiles — NEVER TRAIN ON |
| Holdout articles | 50 | In `holdout_articles.json` — SACRED |
| User feedback | 2,631 | Single user (Ahmad/CPEG student) |
| Only 135 of 813 training articles overlap with production DB | | Distribution mismatch is real |

---

## 12. Qwen3.5 Model Family (Released Feb-Mar 2026)

Evaluated but NOT yet used:
- **Qwen3.5-9B:** Best local option. Q8_0 ~10GB, fits in 24GB. Generationally better than Qwen3 8B. Would need cloud training (RunPod A100 80GB ~$1.74/hr, ~$8-10/run), serve locally.
- **Qwen3.5-27B:** Q8_0 ~28GB — doesn't fit locally. Q4_K_M fits but Unsloth warns of accuracy degradation.
- 7800 XT: NOT ROCm supported (Navi 32 / gfx1102). Don't add it.
- **Decision:** Test `ollama run qwen3.5:9b` locally before any cloud spend. 5 minutes, zero cost, gates $10+ decision. Only pursue AFTER V2.2 proves article expansion works with current Qwen3 8B.

---

## 13. Cloud Training (Future)

- RunPod A100 80GB: ~$1.74/hr Community Cloud
- CUDA eliminates ALL ROCm workarounds
- DoRA on 9B needs ~same VRAM as current 8B setup
- Start from clean Qwen3.5-9B base (no stacked adapters)
- Only pursue after V2.2 validates the article expansion approach

---

## 14. File Locations

All paths relative to `~/Downloads/StratOS/StratOS1/backend/`:

```
# Training Pipeline
data/v2_pipeline/train_v2.py              — Training script
data/v2_pipeline/stage4_prepare.py        — Data preparation (NEEDS V2.2 FIXES)
data/v2_pipeline/profiles_v2.py           — 47 profile definitions
data/v2_pipeline/v22_expansion.py         — V2.2 expansion pipeline (NEW — active)

# Data Files
data/v2_pipeline/scores_v2.json           — 39,071 scored examples
data/v2_pipeline/articles_v2.json         — 813 original articles
data/v2_pipeline/expansion_articles.json  — 2,000 new articles (NEW)
data/v2_pipeline/expansion_scores.json    — Gemini scores for new articles (IN PROGRESS)
data/v2_pipeline/training_v2.jsonl        — V2.1 training data (think_text — DO NOT REUSE)
data/v2_pipeline/eval_holdout_v2.jsonl    — 1,500 holdout examples (SACRED)
data/v2_pipeline/holdout_articles.json    — 50 holdout article IDs

# Validation Reports
data/v2_pipeline/DeepSeek_V3.2_Validation_Report.md
data/v2_pipeline/Gemini_3_Flash_Validation_Report.md
data/v2_pipeline/deepseek_validation.py
data/v2_pipeline/gemini_validation.py
data/v2_pipeline/deepseek_validation_results.json
data/v2_pipeline/gemini_validation_results.json

# Models
data/v2_pipeline/v2_model_backup/v2_scorer.gguf  — V2 GGUF (ALWAYS KEEP)
data/v2_pipeline/training_output/v2.1_scorer.gguf — V2.1 GGUF (failed — reference only)
data/v2_pipeline/v1_merged_base/          — Starting checkpoint

# Planning Documents (in project knowledge)
data/v2_pipeline/training_pipeline.md     — Training knowledge base
v3_planning.md → now "V2.2 Planning"      — V2.2 roadmap
features_roadmap.md                       — Product roadmap

# Inference
processors/scorer_adaptive.py             — Production scorer
processors/scorer_base.py                 — Ollama client, parsing
evaluate_scorer.py                        — Eval script (fixed: num_predict=512, --temperature)
data/v2_pipeline/Modelfile                — Ollama model config (FIXED)

# Config
config.yaml                               — Central config
.env                                       — API keys (ANTHROPIC, SERPER, GOOGLE_API_KEY, DEEPSEEK)
```

---

## 15. API Keys & Services

| Service | Key Location | Purpose |
|---------|-------------|---------|
| Anthropic | `.env` ANTHROPIC_API_KEY | Opus scoring (spot-check), Batch API |
| Google/Gemini | `.env` GOOGLE_API_KEY | Gemini 3 Flash scoring (V2.2 expansion) |
| DeepSeek | `.env` DEEPSEEK_API_KEY | Eliminated — don't use |
| Serper | `.env` SERPER_API_KEY | News search |

Gemini: Paid Tier 1, 10K RPD, 1K RPM. $0.50/$3.00 per MTok.

---

## 16. Success Criteria (V2.2 Holdout Eval)

| Metric | V2 Baseline | V2.2 Target |
|--------|-------------|-------------|
| MAE | 1.544 | < 1.0 |
| Direction accuracy | 91.7% | > 95% |
| Parse failures | 5% | < 2% |
| Moderate band (4-6) MAE | 2.740 | < 2.0 |
| Within 1 point | 53.4% | > 60% |
| Spearman rho | 0.555 | > 0.65 |

---

## 17. Immediate Next Steps (When Phase 2 Completes)

1. **Run Phase 2 daily** until all 50,000 scores done (~4-5 days)
2. **Spot-check calibration** — score 20-30 new articles with Opus, verify isotonic calibration holds (MAE < 1.2)
3. **Fix stage4_prepare.py** — force short reason, real categories, handle Gemini data
4. **Prepare training_v2.2.jsonl** — merge new scores, noise cap at 9,000, 19K total
5. **Train V2.2** on 7900 XTX (~12.5 hours)
6. **Eval on holdout** — compare to V2 baseline (1.544)
7. **If V2.2 beats V2** → ship it, update Modelfile, register with Ollama
8. **If not** → investigate per-band, consider Opus hybrid for moderate band, Qwen3.5-9B

### While Waiting for Phase 2 (parallel work)
- OLLAMA_NUM_PARALLEL benchmark (15 min)
- VRAM coexistence test (2 min)
- Agent search_feed tool (2 hours)
- Kill wasted Serper queries
- RSS feed configuration

---

## 18. Rollback Plan

- V2 GGUF backup at `v2_model_backup/v2_scorer.gguf` — always deployable
- If V2.2 holdout regresses vs V2, do NOT deploy
- Test V2 data + V2.2 base model to isolate data vs model effect
- Keep all intermediate data files — never delete scores_v2.json or eval_holdout_v2.jsonl

---

## 19. User Preferences

- **Always commit after each task** — git commits as checkpoints
- Call training versions V2.1, V2.2 etc — NOT V3
- Don't choose uncommon/niche profiles (bonsai artist)
- Don't include countries unlikely to use the product (India, Kenya)
- User prefers autonomous execution — don't ask permission mid-task
- Server: `cd backend && python3 main.py --serve --background` on port 8080
- Kill server: `fuser -k 8080/tcp`
- DB path: `backend/strat_os.db` (NOT `backend/data/strat_os.db`)

---

## 20. Budget Spent This Session

| Item | Cost |
|------|------|
| DeepSeek validation (300 calls) | ~$0.10 |
| Gemini validation (300 calls) | $0.00 (free tier) |
| Gemini V2.2 scoring (~8,286 so far) | ~$2-3 |
| **Total** | **~$3** |
| **Remaining Gemini budget** | ~4.5 KD ($14.60) |

---

## 21. Key Documents to Read

These are in project knowledge and contain the full details:

1. **training_pipeline.md** — Complete training knowledge base with all V2/V2.1 results, hardware config, Modelfile template, eval procedures
2. **v3_planning.md** (now "V2.2 Planning") — Decision-gated V2.2 roadmap, scoring provider analysis, budget estimates, success criteria
3. **features_roadmap.md** — Prioritized product/engineering roadmap with all tiers
4. **MEMORY.md** — Project-level memory for Claude sessions
5. **CLAUDE.md** — Project coding conventions and architecture notes
6. **DeepSeek_V3.2_Validation_Report.md** — Why DeepSeek was eliminated
7. **Gemini_3_Flash_Validation_Report.md** — Why Gemini was selected, calibration analysis
