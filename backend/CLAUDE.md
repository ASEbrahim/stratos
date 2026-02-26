# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is StratOS

StratOS is a personalized strategic intelligence dashboard. It fetches news from multiple sources (RSS, DuckDuckGo, Serper/Google), scores them for relevance using a local Ollama-hosted LLM, and presents a prioritized feed via a web UI. The system continuously improves its scoring model through a distillation loop: Claude Opus re-scores articles, disagreements become training data, and LoRA fine-tuning updates the local model.

## Running the Application

```bash
# Start the full server (dashboard at http://localhost:8080)
python3 main.py --serve --background

# Or use the launcher script (checks Ollama, registers models, then starts)
bash stratos.sh

# Run a one-shot scan without the server
python3 main.py --scan
```

**Prerequisites**: Ollama must be running (`ollama serve`). The system uses three Ollama models:
- `scoring.model` from config.yaml (e.g. `stratos-scorer-v2`) — fine-tuned scoring model
- `scoring.inference_model` (e.g. `qwen3:30b-a3b`) — used for agent chat, suggestions, briefings
- `scoring.wizard_model` (e.g. `qwen3:14b`) — lighter model for wizard/generate-profile (faster, less over-thinking). Falls back to inference_model if not set.

## Training Pipeline

```bash
# Manual learning cycle (distill → export → train)
bash learn_cycle.sh

# Or step by step:
python3 distill.py --hours 168 --limit 200    # Opus re-scores recent items
python3 export_training.py --min-delta 1.5     # Export corrections as JSONL
python3 train_lora.py --epochs 3               # LoRA fine-tune → GGUF → Ollama register

# Fully autonomous overnight mode (scan → distill → train loop)
bash stratos_overnight.sh          # Default budget
bash stratos_overnight.sh 10.00    # Custom budget

# Autonomous autopilot (cycles through diverse professional profiles)
python3 autopilot.py --cycles 10 --budget 5.00
python3 autopilot.py --dry-run     # Preview which profiles will run
```

## Architecture

### Core Orchestrator
- `main.py` — Contains the `StratOS` class: scan pipeline, config loading, background scheduler, and entry point. Delegates HTTP serving to `server.py`.
- `server.py` — HTTP server and route dispatch. Contains `CORSHandler` (all API routes via `do_GET`/`do_POST`), `ThreadedHTTPServer`, and `start_server()`. Serves the frontend from `../frontend/`.
- `auth.py` — `AuthManager` class: profile-based authentication, session management, rate limiting, API key masking.
- `sse.py` — `SSEManager` class: SSE client tracking and event broadcasting for live dashboard updates.

### Data Flow
```
News Sources → NewsFetcher → AI Scorer → SQLite DB → JSON API → Frontend
                                ↑                        ↓
                         Feedback Loop              User Feedback
                                ↑                        ↓
                    LoRA Training ← Export ← Distillation (Opus)
```

### Fetchers (`fetchers/`)
- `news.py` — Multi-source news fetcher (DuckDuckGo, Serper, RSS feeds). Uses `ThreadPoolExecutor` for parallel fetching.
- `market.py` — Yahoo Finance market data via `yfinance`. Handles multiple intervals (1m to 1wk).
- `discovery.py` — Entity discovery: tracks keyword frequency changes to detect rising topics.
- `extra_feeds.py` — RSS feed URL registry (finance + politics feeds).
- `serper_search.py` / `google_search.py` — Search API clients.

### Processors (`processors/`)
- `scorer_base.py` — `ScorerBase` class: Ollama client, calibration tables, noise pattern lists, language filtering, `ScoringMemory`. Shared infrastructure for scoring.
- `scorer_adaptive.py` — `AdaptiveScorer(ScorerBase)`: sole scorer. Profile-adaptive, dynamically builds relevance rules from user's categories/keywords/role. Hybrid: rule-based noise filters + Ollama LLM scoring.
- `briefing.py` — Generates LLM-powered intelligence briefings.
- `profile_generator.py` — AI-powered category generation from role/location/context.

### Routes (`routes/`)
- `agent.py` — Chat agent with Ollama tool-use (web_search, manage_watchlist, manage_categories). Handles `/api/agent-chat`, `/api/suggest-context`, `/api/ask`.
- `generate.py` — `/api/generate-profile` — AI generates categories + tickers from user profile.
- `wizard.py` — `/api/wizard-preselect`, `/api/wizard-tab-suggest`, `/api/wizard-rv-items` — onboarding wizard backend.
- `config.py` — `/api/config` save handler.
- `helpers.py` — JSON response, SSE, gzip utilities.

### Self-Improvement Pipeline
- `distill.py` — Teacher-student distillation using Claude Opus. Sends locally-scored items to Opus, saves disagreements (≥2.0 delta) as corrections in `user_feedback` table.
- `export_training.py` — Converts corrections + user feedback into ChatML JSONL. **Critical**: training format must be character-for-character identical to inference format in `scorer_adaptive.py`.
- `train_lora.py` — LoRA fine-tuning pipeline. Auto-selects model tier by VRAM. Supports both Unsloth (CUDA) and PEFT (ROCm). Pipeline: train → merge → GGUF export → Ollama register → config update.
- `autopilot.py` — Fully autonomous loop: picks diverse profiles, generates context via Opus, scans, distills, trains. Manages budget, persistent state, profile diversity guards.

### V2 Training Pipeline (`data/v2_pipeline/`)
- `profiles_v2.py` — 30 training profile definitions
- `stage2_collect.py` — Article collection from DDG/Serper/RSS
- `stage3_score.py` / `stage3_resume.py` — Claude Opus batch scoring
- `stage4_prepare.py` — Training data preparation with dual weighting
- `train_v2.py` — V2-specific DoRA training script
- `phase1_merge.py` — Merge V1 base with DoRA adapters

### Scorer Architecture
`_create_scorer()` in `main.py` always returns `AdaptiveScorer` (the sole scorer since B3.3).
- `scorer_base.py` — `ScorerBase` class with shared infrastructure: Ollama client, calibration, score classification, noise patterns, language filtering, `ScoringMemory`.
- `scorer_adaptive.py` — `AdaptiveScorer(ScorerBase)`: profile-adaptive scorer that dynamically builds relevance rules from any user profile. Interface: `score_items()`, `score_item()`, `get_score_category()`.

### Database
`database.py` — SQLite with WAL mode. Singleton via `get_database()`. Tables: `news_items`, `market_snapshots`, `entities`, `entity_mentions`, `scan_log`, `briefings`, `user_feedback`. Thread-safe commits via lock.

### Profiles
Per-user profiles stored as YAML in `profiles/`. Each has: role, location, context, market tickers, news categories with keywords, dynamic_categories, feed preferences, and a PIN hash for auth.

### Configuration
`config.yaml` — Central config. Key sections: `scoring` (model names, Ollama host, score thresholds), `news` (categories with keywords), `market` (tickers), `search` (API keys/provider), `profile` (role/location/context), `discovery`, `cache` TTLs.

## Key Design Decisions

- **Qwen3 think blocks**: Do NOT set `"think": False` in Ollama API calls — Qwen3 ignores it and leaks reasoning as plain text into the `content` field, consuming the entire `num_predict` token budget before producing any useful output. Instead, omit the `think` parameter entirely. In default mode, Ollama separates reasoning into a `thinking` field while `content` holds the clean answer. Always strip `<think>...</think>` blocks from content as a safety net. Set `num_predict` large enough to cover both thinking tokens (~1000-2000) and output tokens.
- **Training/inference alignment**: `export_training.py` must produce system/user/assistant messages that exactly match what `scorer_adaptive.py` sends at inference time. Misalignment causes the trained model to fail at parsing.
- **PEFT meta device fix**: After `get_peft_model()`, delete `hf_device_map` and force `.to("cuda:0")` to prevent gradient crashes during training.
- **Incremental vs full retrain**: Default is incremental (builds on `data/models/current_base/`). Full retrain triggers when agreement rate drops below 35% or profile diversity is too low (prevents catastrophic forgetting).
- **Streaming Ollama for cancellation**: `_call_ollama` uses `stream=True` and checks `cancel_check` every 10 chunks for responsive scan cancellation.
- **V2 dual weighting**: WeightedRandomSampler for batch balance + per-sample loss weighting for score-band emphasis.
- **AOTRITON workaround**: Do NOT set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` — causes NaN gradients on 7900 XTX.

## Secrets and API Keys

All API keys and secrets live in `backend/.env` (gitignored), NOT in `config.yaml`.

```bash
# backend/.env
ANTHROPIC_API_KEY=sk-ant-...   # Claude Opus API for distillation
SERPER_API_KEY=...              # Serper.dev Google search
GOOGLE_API_KEY=...              # Google Custom Search API
GOOGLE_CSE_ID=...               # Google Custom Search Engine ID
```

`main.py` loads `.env` on startup and overrides `config.yaml` values with environment variables. If `.env` doesn't exist, falls back to config.yaml values (backward compatible).

## Ollama Environment Variables

The inference model (`qwen3:30b-a3b`, ~18-19GB) and scorer model (`stratos-scorer-v2`, 8.7GB) cannot coexist in 24GB VRAM. Ollama handles model swapping automatically. These env vars should be set for the Ollama systemd service or shell:

```bash
OLLAMA_MAX_LOADED_MODELS=1    # Only one model in VRAM at a time
OLLAMA_KEEP_ALIVE=10m         # Auto-unload after 10 min idle
OLLAMA_NUM_PARALLEL=1         # Single-user, minimize VRAM duplication
OLLAMA_FLASH_ATTENTION=0      # ROCm 6.2 — flash attention unreliable
ROCR_VISIBLE_DEVICES=0        # Prevent Ollama from using 7900X3D iGPU
```

## API Endpoints (served from main.py)

Key endpoints: `/api/news`, `/api/market`, `/api/refresh` (trigger scan), `/api/scan/status` (GET), `/api/scan/cancel` (POST), `/api/status`, `/api/feedback`, `/api/agent-chat`, `/api/suggest-context`, `/api/generate-profile`, `/api/config`, `/api/events` (SSE stream), `/api/briefing`, `/api/wizard-preselect`, `/api/wizard-tab-suggest`, `/api/wizard-rv-items`.
