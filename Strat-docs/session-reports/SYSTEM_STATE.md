# StratOS System State
**Snapshot Date:** February 22, 2026
**Version:** V2.1 Production
**Branch:** master

---

## 1. Current Version & Deployment

- **Scoring model:** `stratos-scorer-v2` registered in Ollama (8.7 GB Q8_0 GGUF)
- **Inference model:** `qwen3:30b-a3b` (agent chat, briefings, suggestions, profile generation)
- **Ollama host:** `http://localhost:11434`
- **Dashboard:** `http://localhost:8080`
- **Base architecture:** Qwen3-8B + DoRA fine-tuning

## 2. Codebase Statistics

| Component | Lines | Notes |
|-----------|------:|-------|
| Backend (Python) | ~19,700 | 33 .py files across root, processors/, fetchers/, routes/, data/v2_pipeline/ |
| Frontend (JS/HTML/CSS) | ~17,400 | 22 files: 14 .js, 5 .html, 3 .css |
| Shell scripts | ~480 | 4 scripts: stratos.sh, learn_cycle.sh, stratos_overnight.sh, setup_rocm_training.sh |
| Docs (Markdown) | ~2,400 | CLAUDE.md, PATCH_README.md, STRATOS_FRS_v4.md, pipeline reports |
| Config (YAML) | ~260 | config.yaml |
| **Total** | **~39,500** | **40+ modules** |

### Major file sizes (backend)

| File | Lines | Purpose |
|------|------:|---------|
| main.py | 2,110 | Core orchestrator + HTTP server |
| processors/scorer.py | 2,138 | Hardcoded CPEG/Kuwait scorer |
| processors/scorer_adaptive.py | 1,421 | Profile-adaptive scorer |
| train_lora.py | 1,344 | LoRA fine-tuning pipeline |
| data/v2_pipeline/train_v2.py | 1,120 | V2 DoRA training script |
| autopilot.py | 959 | Autonomous scan/distill/train loop |
| fetchers/news.py | 917 | Multi-source news fetcher |
| routes/agent.py | 817 | Chat agent with tool-use |
| database.py | 685 | SQLite database layer |
| distill.py | 601 | Opus distillation |
| fetchers/kuwait_scrapers.py | 589 | Kuwait news scrapers |
| export_training.py | 539 | Training data exporter |
| processors/briefing.py | 455 | Intelligence briefings |
| data/v2_pipeline/profiles_v2.py | 423 | 30 training profile definitions |
| processors/profile_generator.py | 400 | AI category generation |

## 3. Active Models

### stratos-scorer-v2 (scoring)
- **Base:** Qwen3-8B
- **Fine-tuning:** DoRA (Weight-Decomposed Low-Rank Adaptation)
- **Training data:** 20,550 scored examples, 18,502 train, 2,048 eval, 25,243 contrastive pairs
- **Profile coverage:** 30 diverse professional profiles
- **Metrics:** 98.1% direction accuracy, MAE 0.393, Spearman 0.750, PSR 24.4%
- **Format:** Q8_0 GGUF, 8.7 GB
- **Think blocks:** Disabled (`"think": False`), 0.0% empty think rate

### qwen3:30b-a3b (inference)
- Used for: agent chat, context suggestions, profile generation, briefings
- Accessed via Ollama API at localhost:11434

## 4. Active Profiles

| Profile | Role | Location | Status |
|---------|------|----------|--------|
| Ahmad | Senior Geophysicist / Computer Engineer | Kuwait | Primary (config.yaml default) |
| Abdullah | — | — | Secondary |

Profiles stored in `profiles/` as YAML directories + .yaml files.

## 5. Database State

- **File:** `strat_os.db` (SQLite, WAL mode)
- **7 tables:** `news_items`, `market_snapshots`, `entities`, `entity_mentions`, `scan_log`, `briefings`, `user_feedback`
- **Thread-safe:** Singleton via `get_database()`, commits via lock

## 6. Recent Changes (last 5 commits)

```
243dea6 chore: V2.1 — deploy to Ollama, cleanup artifacts, pin deps
fb17317 feat: V2 scorer trained — 98.1% dir, 24.4% PSR, MAE 0.393, 0.0% empty think
b4ba2d8 feat: V2 scorer training data — 20,550 scored examples, 18,502 train, 2,048 eval, 25,243 contrastive pairs
0e9fc44 feat: V2 training pipeline stages 1-4 + 685 collected articles
539b61c fix: use DDG news endpoint and remove broken OR-news boolean from query templates
```

## 7. Git Tags

```
v2.0          — V2 scorer trained
v2.1          — V2.1 deploy + cleanup
pre-cancel-fix — Before scan cancellation refactor
post-qa-session
pre-qa-session
```

## 8. Key File Layout

```
backend/
  main.py                          (2,110)  Core orchestrator + HTTP server
  database.py                        (685)  SQLite layer
  config.yaml                        (262)  Central configuration
  distill.py                         (601)  Opus distillation
  export_training.py                 (539)  Training data export
  train_lora.py                    (1,344)  LoRA fine-tuning
  autopilot.py                       (959)  Autonomous loop
  stratos.sh                          (86)  Launcher script
  learn_cycle.sh                      (45)  Manual learning cycle
  stratos_overnight.sh               (190)  Overnight autonomous mode
  setup_rocm_training.sh             (163)  ROCm training env setup
  processors/
    scorer.py                      (2,138)  Hardcoded CPEG/Kuwait scorer
    scorer_adaptive.py             (1,421)  Profile-adaptive scorer
    briefing.py                      (455)  LLM briefings
    profile_generator.py             (400)  AI category generation
  fetchers/
    news.py                          (917)  Multi-source news fetcher
    market.py                        (316)  Yahoo Finance market data
    discovery.py                     (314)  Entity discovery
    serper_search.py                 (332)  Serper API client
    google_search.py                 (300)  Google API client
    extra_feeds.py                   (274)  RSS feed registry
    kuwait_scrapers.py               (589)  Kuwait news scrapers
  routes/
    agent.py                         (817)  Chat agent + tool-use
    wizard.py                        (342)  Onboarding wizard endpoints
    config.py                        (215)  Config save handler
    generate.py                      (254)  Profile generation endpoint
    helpers.py                        (55)  JSON/SSE/gzip utilities
  profiles/
    Ahmad.yaml                              Primary profile
    Abdullah.yaml                           Secondary profile
  data/v2_pipeline/
    profiles_v2.py                   (423)  30 training profiles
    stage2_collect.py                (377)  Article collection
    stage3_score.py                  (395)  Opus batch scoring
    stage3_resume.py                 (319)  Resume interrupted scoring
    stage4_prepare.py                (346)  Training data preparation
    train_v2.py                    (1,120)  V2 DoRA training
    phase1_merge.py                   (87)  Merge V1 base + DoRA adapters

frontend/
  index.html                       (1,104)  Main dashboard
  app.js                           (1,285)  App initialization
  feed.js                           (762)  News feed rendering
  settings.js                      (2,529)  Settings panel
  agent.js                          (908)  Chat agent UI
  wizard.js                        (2,543)  Onboarding wizard logic
  market.js                        (1,004)  Market data
  markets-panel.js                 (1,843)  Markets panel UI
  wizard_v4.html                   (1,131)  Latest wizard template
  onboarding-wizard.html           (1,413)  Onboarding wizard template
  styles.css                         (518)  Custom styles
```

## 9. Configuration Summary

| Setting | Value |
|---------|-------|
| Ollama host | `http://localhost:11434` |
| Server port | `8080` |
| Scoring model | `stratos-scorer-v2` |
| Inference model | `qwen3:30b-a3b` |
| Search provider | `serper` (DuckDuckGo also used in fetcher) |
| News timelimit | `w` (weekly) |
| Score thresholds | Critical: 9.0, High: 7.0, Medium: 5.0, Filter below: 5.0 |
| Database | `strat_os.db` (SQLite, WAL) |
| Background scan | Disabled (30-min interval when enabled) |
| Max news items | 100 |

## 10. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.5.1+rocm6.2 | Training (AMD ROCm) |
| transformers | 5.1.0 | Model loading/training |
| peft | 0.18.1 | DoRA/LoRA adapters |
| accelerate | 1.12.0 | Training acceleration |
| datasets | 4.5.0 | Dataset handling |
| anthropic | 0.83.0 | Claude Opus API (distillation) |
| duckduckgo_search | 8.1.1 | DDG news fetcher |
| feedparser | 6.0.12 | RSS parsing |
| yfinance | 1.1.0 | Market data |
| beautifulsoup4 | 4.14.3 | HTML parsing |
| lxml | 6.0.2 | XML/HTML parsing |
| PyYAML | 6.0.1 | Config/profile parsing |
| requests | 2.32.5 | HTTP client |

**GPU:** AMD Radeon RX 7900 XTX (ROCm 6.2)

## 11. Known Issues

- **AIScorer (scorer.py) hardcoded for CPEG/Kuwait** -- should be removed eventually; `scorer_adaptive.py` handles all profiles generically
- **Qwen3 think blocks** -- always use `"think": False` in Ollama options and strip `<think>...</think>` from responses; without this, think blocks consume the token budget and return empty answers
- **PEFT meta device** -- after `get_peft_model()`, delete `hf_device_map` and force `.to("cuda:0")` to prevent gradient crashes
- **PSR metric** -- V2's 24.4% is correct behavior (Opus's own PSR is 26.4%, so this is near-ceiling)
- **AOTRITON** -- do NOT set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`; causes NaN gradients on 7900 XTX
- **Incremental vs full retrain** -- default is incremental (builds on `data/models/current_base/`); full retrain triggers when agreement rate drops below 35% or profile diversity is too low

## 12. Running the System

```bash
# Standard start (dashboard at http://localhost:8080)
python3 main.py --serve --background

# Launcher script (checks Ollama, registers models, then starts)
bash stratos.sh

# One-shot scan without server
python3 main.py --scan

# Manual learning cycle
bash learn_cycle.sh

# Autonomous overnight mode
bash stratos_overnight.sh          # Default budget
bash stratos_overnight.sh 10.00    # Custom budget

# Autopilot (diverse profile cycling)
python3 autopilot.py --cycles 10 --budget 5.00
```

## 13. External Services

| Service | Status | Purpose |
|---------|--------|---------|
| Ollama | **Required**, local | LLM inference (scoring + agent) |
| Claude Opus API | Optional | Distillation (teacher model) |
| Serper | Optional | Google search for news |
| DuckDuckGo | Free, no key | Primary news search |
| Yahoo Finance | Free, no key | Market data via yfinance |

---

*This snapshot is manually maintained. Update after significant changes (model retraining, architecture refactors, new features).*
