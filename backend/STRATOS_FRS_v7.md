# StratOS Functional Requirements Specification (FRS)

**Project:** StratOS -- Strategic Intelligence Operating System
**Version:** 7.0 (V2.1 Production + Mobile)
**Date:** February 25, 2026
**Status:** V2.1 Production
**Author:** Ahmad (Computer Engineering student, American University of Kuwait)

---

## 1. Executive Summary

StratOS is a self-hosted, profile-driven strategic intelligence platform. Its core mission: **"Tell the system who you are, and it tells you what matters today."**

The system aggregates news from multiple sources, tracks real-time markets, and scores content relevance using a locally fine-tuned AI model. It achieves zero cloud dependency for inference (all scoring runs on a local Ollama server), with optional Claude Opus distillation at approximately $0.40 per cycle for self-improvement.

**Scale:** 40,000+ lines across 40+ modules. Solo project by a Computer Engineering student in Kuwait.

**Intelligence Verticals:**
- Career Intel -- job openings, employer news, hiring trends
- Financial Advantage -- bank deals, student offers, investment opportunities
- Future Tech Trends -- AI, cloud, cybersecurity, emerging tech
- Market Intelligence -- real-time prices, charts, alerts, watchlists
- Regional Industry -- GCC-specific developments, government projects

**Technology Stack:**
- Backend: Python 3.12, built-in threaded HTTP server (no Flask/FastAPI), ~19,700 lines
- Frontend: SPA, vanilla JavaScript + Tailwind CSS + TradingView Lightweight Charts, ~13,400+ lines across 13 JS files + HTML
- AI: Ollama (local LLM server), Qwen3-8B fine-tuned with DoRA, Claude Opus 4.5 (teacher model for distillation)
- Database: SQLite with WAL mode
- Market Data: Yahoo Finance via yfinance
- Search: DuckDuckGo, Serper API, Google Custom Search API, RSS (30+ feeds)
- Mobile: Progressive Web App (PWA), touch gestures, offline-capable

---

## 2. System Architecture

### 2.1 High-Level Data Flow

```
News Sources (DDG, Serper, RSS, Scrapers)
         |
         v
    NewsFetcher (ThreadPoolExecutor, parallel)
         |
         v
    Dynamic Category Reclassification
         |
         v
    AI Scorer (3-phase pipeline)
    [Phase 1: Rule scoring] -> [Phase 2: Batch LLM] -> [Phase 3: Re-score cascade]
         |
         v
    SQLite Database (WAL mode)
         |
         v
    JSON API (/api/data)  ->  Frontend SPA
         |
         v
    User Feedback (save/dismiss/rate)
         |
         v
    Distillation (Claude Opus re-scores)  ->  Export JSONL  ->  LoRA Training
         |
         v
    Updated Scorer Model (Ollama register)
```

### 2.2 Two Ollama Models

| Model | Purpose | Size | Config Key |
|-------|---------|------|------------|
| `stratos-scorer-v2` | DoRA fine-tuned Qwen3-8B for relevance scoring | 8.7GB Q8_0 GGUF | `scoring.model` |
| `qwen3:30b-a3b` | General inference for agent chat, suggestions, profile generation, briefings | 14B params | `scoring.inference_model` |

Both models are served by a local Ollama instance at `http://localhost:11434`. The `think` parameter is omitted entirely from all Ollama calls (see §10.1). All responses are post-processed to strip `<think>...</think>` blocks as a safety net.

### 2.3 Backend Architecture (Python 3.12)

The backend is a monolithic Python application centered on `main.py`, which contains:
- The `StratOS` class (main orchestrator)
- A built-in threaded HTTP server (subclass of `http.server.HTTPServer` with `ThreadingMixIn`)
- All API route handling via `do_GET` / `do_POST` methods on `CORSHandler`
- SSE (Server-Sent Events) push system for live dashboard updates
- Session management (device-isolated, 7-day TTL, persisted to `.sessions.json`)
- Rate limiting (per-endpoint, sliding window)
- Profile loading and switching
- Graceful shutdown with signal handlers (SIGINT/SIGTERM)

There is no dependency on Flask, FastAPI, Django, or any web framework. The server is pure stdlib `http.server` extended with CORS headers, gzip compression, ETag caching, and JSON API routing.

### 2.4 Frontend Architecture (Vanilla JS SPA)

The frontend is a single-page application with no build step required (Tailwind is pre-compiled). Key architectural decisions:
- No React/Vue/Svelte -- pure vanilla JavaScript with module separation
- Tailwind CSS for styling with custom theme support (7 color themes)
- TradingView Lightweight Charts for market visualization
- SSE (`EventSource`) for real-time scan progress updates
- Service Worker (`sw.js`) for offline resilience and PWA support
- Device fingerprinting for profile isolation across devices
- Mobile-first responsive design with touch gesture engine (`mobile.js`)
- Progressive Web App: installable on home screens with standalone display mode

### 2.5 Authentication System

- Device-isolated sessions: each session is bound to a `device_id` + `profile_name`
- SHA-256 PIN hashing: PINs are hashed before storage in per-profile YAML configs
- Session tokens: 64-character hex tokens with 7-day TTL
- Sessions persist across server restarts via `.sessions.json`
- Auth-exempt endpoints: `/api/auth-check`, `/api/auth`, `/api/register`, `/api/logout`, `/api/suggest-context`, `/api/generate-profile`, `/api/wizard-*`, `/api/refresh`, `/api/status`, `/api/scan/status`, `/api/scan/cancel`
- Rate limiting: per-endpoint sliding window (e.g., `/api/refresh` = 2/min, `/api/agent-chat` = 20/min)
- API key masking: keys are masked in GET responses (show only last 4 chars), masked values are rejected on save

---

## 3. Module A: Data Ingestion

### 3.1 News Intelligence Fetcher (`fetchers/news.py`, 917 lines)

The `NewsFetcher` class aggregates news from multiple sources using `ThreadPoolExecutor` for parallel fetching.

**Sources:**
- **DuckDuckGo News:** Primary free source. Uses the `ddgs` library's news endpoint (`DDGS().news()`). Generates per-profile search queries from categories and keywords.
- **Serper API:** Google Search results via Serper's REST API (`fetchers/serper_search.py`, 332 lines). Configurable, tracks remaining credits. Query tracker prevents duplicate queries within configurable windows.
- **Google Custom Search API:** Alternative to Serper (`fetchers/google_search.py`, 300 lines). Requires API key + custom search engine ID.
- **RSS Feeds:** 30+ feeds registered in `fetchers/extra_feeds.py` (274 lines). Includes:
  - Finance: CNBC, Bloomberg, MarketWatch, Yahoo Finance, Seeking Alpha, CoinDesk, OilPrice, Boursa Kuwait, Zawya, etc.
  - Politics: BBC World/Mideast, Al Jazeera, NPR, NYT, CNN, Guardian, Al Arabiya, Middle East Eye, SCMP, Nikkei Asia, etc.
  - User-configurable custom RSS feeds per profile.

**Data Model (`NewsItem` dataclass):**
```python
@dataclass
class NewsItem:
    title: str
    url: str
    summary: str = ""
    content: str = ""        # Full article text from deep scraping
    timestamp: str = ""
    source: str = "Unknown"
    root: str = "global"     # global, regional, kuwait, ai
    category: str = "general"
    score: float = 0.0
    score_reason: str = ""
```

**Article Scraper:** The `ArticleScraper` class performs deep scraping of full article text from URLs using `ThreadPoolExecutor`. Implements rate limiting per domain, rotates user agents, and uses BeautifulSoup for HTML parsing with content selectors (`article`, `main`, `.post-content`, `.article-body`).

**Query Generation:** Queries are dynamically built from the user's profile categories and keywords. Each category's keywords are expanded into search queries, which are then dispatched in parallel to configured search providers.

**Time Limit:** Configurable via `news.timelimit` in config (values: `d` = day, `w` = week, `m` = month).

### 3.2 Market Data Fetcher (`fetchers/market.py`, 316 lines)

The `MarketFetcher` class retrieves market data from Yahoo Finance via the `yfinance` library.

**Supported Intervals:**

| Key | Period | Max Points | Description |
|-----|--------|------------|-------------|
| `1m` | 3 days | 5,000 | 1-minute candles (1D button) |
| `5m` | 15 days | 5,000 | 5-minute candles (5D button) |
| `1d_1mo` | 1 year | 365 | Daily candles (1M button) |
| `1d_1y` | 5 years | 1,260 | Daily candles (1Y button) |
| `1wk` | 10 years | 520 | Weekly candles (5Y button) |

**Features:**
- Per-profile ticker watchlists (stored in profile YAML and config.yaml)
- Built-in caching with configurable TTL (default 60s)
- Threshold-based price alerts (default 5% change triggers alert)
- Returns OHLCV history for charting
- Downsampling: OHLCV aggregation (open of first candle, high of group, low of group, close of last candle)

### 3.3 Kuwait Scrapers (`fetchers/kuwait_scrapers.py`, 589 lines)

Specialized scrapers for Kuwait-specific sources and career portals:
- Kuwait Oil Company (KOC) careers
- Kuwait National Petroleum Company (KNPC)
- Kuwait Integrated Petroleum Industries Company (KIPIC)
- Kuwait-specific news sources

### 3.4 Entity Discovery (`fetchers/discovery.py`, 314 lines)

The `EntityDiscovery` class tracks keyword frequency changes to detect rising topics.

**Algorithm:**
1. Extract potential entities (proper nouns, company names) from scored news items
2. Compare current mention frequency against historical baseline
3. Flag entities with significant frequency increase as "rising"
4. Optionally auto-add rising entities to tracking list

**Configuration:**
- `baseline_window_hours`: 168 (7 days)
- `min_mentions`: 3
- `rising_multiplier`: 3.0x (current frequency must be 3x baseline to flag)
- `auto_track`: true/false

**Entity Categories:** Automatically categorized via regex patterns into `kuwait_companies`, `tech_terms`, `financial_products`.

### 3.5 Extra Feeds System (`fetchers/extra_feeds.py`, 274 lines)

Provides curated RSS feed catalogs for Finance and Politics tabs. Each feed has:
- `id`: unique identifier
- `name`: display name
- `url`: RSS feed URL
- `on`: default enabled state

Users can toggle feeds per-profile. The frontend provides dedicated Finance and Politics tabs that fetch from these feeds without AI scoring (real-time pass-through).

---

## 4. Module B: AI Processing

### 4.1 V2 Scorer Model (`stratos-scorer-v2`)

**Architecture:**
- Base model: Qwen3-8B
- Fine-tuning: DoRA (Weight-Decomposed Low-Rank Adaptation)
  - Rank: 16
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Dropout: 0.05
- Quantization: Q8_0 GGUF (8.7GB)

**Performance Metrics:**
- Direction accuracy: 98.1%
- Mean Absolute Error (MAE): 0.393
- Spearman rho: 0.750
- Profile Sensitivity Ratio (PSR): 24.4%
- Profile-awareness spread: 7.90 (vs V1's 1.04 -- a 7.6x improvement)
- Zero parse failures
- Zero empty think blocks

**Training Data:**
- 30 trained profiles spanning 20 countries
- 20,550 scored examples (Claude Opus teacher)
- 18,502 training examples, 2,048 evaluation examples
- 25,243 contrastive pairs
- Training cost: ~$52 (Claude Opus Batch API)

**Output Format:**
```
SCORE: X.X | REASON: brief explanation
```

### 4.2 Scorer Selection Logic

The `_create_scorer()` factory function in `main.py` picks the appropriate scorer:

```python
def _create_scorer(config, db=None):
    if should_use_adaptive_scorer(config):
        return AdaptiveScorer(config, db=db)
    return AIScorer(config, db=db)
```

- `AIScorer` (`processors/scorer.py`, 2,138 lines): Highly tuned for the default CPEG-in-Kuwait profile. Contains extensive hardcoded pattern lists for noise detection specific to Kuwait tech, oil/gas, and banking contexts.
- `AdaptiveScorer` (`processors/scorer_adaptive.py`, 1,421 lines): Generic, builds relevance rules dynamically from any user profile's categories, keywords, role, and context. Used for all non-default profiles.

Both scorers implement the same interface:
- `score_items(items, progress_callback, cancel_check)` -- batch scoring
- `score_item(item)` -- single item scoring
- `get_score_category(score)` -- returns "critical", "high", "medium", or "noise"

### 4.3 Three-Phase Scoring Pipeline

**Phase 1: Rule Scoring**
- Regex/keyword noise filters detect: garbage domains, stale patterns, job aggregator pages, social media noise, company profile pages, stock data pages, flight pages, technical data pages, generic page patterns
- Confidence thresholds determine whether an item can be scored by rules alone or needs LLM
- Items conclusively identified as noise (score <= 4.0 with high confidence) skip LLM entirely
- Items conclusively identified as highly relevant (strong keyword matches) may receive rule-based scores without LLM

**Phase 2: Batch LLM Scoring**
- Items not resolved by Phase 1 are sent to the Ollama scoring model
- `AdaptiveScorer`: batches of 4 items per LLM call with numbered format
- `AIScorer`: 1 item per call
- Streaming Ollama responses with `cancel_check` called every 10 tokens for graceful scan cancellation
- System prompt dynamically built from user profile (role, location, context, tracked entities)
- Temperature: 0.3, top_p: 0.9, num_ctx: 2048, num_predict: 128

**Phase 3: Prompt Re-score Cascade**
- Items in the "uncertain zone" (scores 4.5-6.5) are re-scored with a richer rubric
- Capped at 12 items per scan to limit API cost
- Uses a more detailed prompt with explicit category breakdowns and scoring guidance

### 4.4 Scoring Rubric

| Score Range | Category | Description |
|-------------|----------|-------------|
| 9.0 - 10.0 | Critical | Direct career/financial opportunity requiring immediate action |
| 7.0 - 8.9 | High | Strongly relevant to professional development or strategic interests |
| 5.0 - 6.9 | Medium | Tangentially relevant or general industry context |
| 0.0 - 4.9 | Noise | Irrelevant, clickbait, paywalled, or wrong geography |

**Forbidden 5.0 Rule:** Scores in the range 4.8-5.2 are nudged to 4.8 or 5.3 to prevent the model from hedging on the noise/signal boundary. This is enforced at the config level (`scoring.forbidden_score: 5.0`).

### 4.5 Adaptive Scorer Internals (`scorer_adaptive.py`)

The `AdaptiveScorer` builds its scoring intelligence dynamically:

**KeywordIndex:** Constructed from the user's dynamic categories. For each category, it extracts:
- Primary keywords: the category items themselves (e.g., "Zain", "Ooredoo")
- Secondary keywords: individual significant words from multi-word items
- Label keywords: words from category labels
- Universal keywords: words from the user's role and context fields
- Location parts: geographic terms for geo-matching

**Universal Signals (domain-agnostic):**
- `HIRING_SIGNALS`: 22 hiring-related terms
- `ENTRY_LEVEL_SIGNALS`: 17 entry-level/student terms
- `SENIOR_LEVEL_SIGNALS`: 10 senior/leadership terms
- `OFFER_DEAL_SIGNALS`: 18 offer/deal terms
- `BREAKTHROUGH_SIGNALS`: 12 breakthrough/discovery terms
- `NEWS_ACTION_SIGNALS`: 16 news action terms
- `EXPERIENCE_REGEX`: 6 regex patterns for experience requirements

**Language Filtering:** `location_to_lang()` maps user locations to expected languages, filtering out non-target-language content. `_is_non_latin_title()` and `_is_non_target_language()` handle script detection.

**ScoringMemory:** Persistent memory class that tracks scoring patterns and adjustments across sessions, stored in `data/memory.json`.

### 4.6 V2 Training Pipeline (`data/v2_pipeline/`)

The V2 training pipeline is a 4-phase system:

**Phase 1: Article Collection (`stage2_collect.py`, 377 lines)**
- Collects articles from DuckDuckGo, Serper, and RSS sources
- 685 unique articles collected
- Stored in `articles_v2.json`

**Phase 2: Batch API Scoring (`stage3_score.py`, 395 lines)**
- Sends articles to Claude Opus via the Batch API
- Each article scored from the perspective of all 30 profiles
- Uses PRISM reasoning framework (Profile analysis, Relevance mapping, Impact assessment, Score calibration, Meta-check)
- 20,550 total scoring results (685 articles x 30 profiles)
- 0 malformed responses, 0 empty think blocks, 0 sparse reasoning
- Cost: $52.37 (9.3M input tokens, 5.1M output tokens)

**Score Distribution:**
| Range | Count | Percentage |
|-------|-------|------------|
| 0-2 | 14,876 | 72.4% |
| 2-4 | 4,671 | 22.7% |
| 4-5 | 194 | 0.9% |
| 5-6 | 0 | 0.0% |
| 6-8 | 647 | 3.1% |
| 8-10 | 162 | 0.8% |

**Phase 3: Data Preparation (`stage4_prepare.py`, 346 lines)**
- Dual weighting strategy:
  - `WeightedRandomSampler`: oversamples underrepresented score bands during training
  - Per-sample loss weighting: gradient emphasis on high-value examples
- Contrastive pairs: 25,243 pairs (same article, different profiles, different scores)
- Train/eval split: 18,502 / 2,048 (stratified by profile and score band)

**Band Distribution (training set):**
| Band | Count |
|------|-------|
| noise | 16,429 |
| tangential | 1,359 |
| moderate | 345 |
| high | 246 |
| critical | 123 |

**Phase 4: Training (`train_v2.py`, 1,120 lines)**
- DoRA fine-tuning with:
  - Rank 16, Alpha 32, Dropout 0.05
  - Learning rate: 1e-5 (cosine schedule with 5% warmup)
  - Gradient accumulation: 8
  - Max sequence length: 1024
  - 1 epoch, 1,157 training steps
  - Training time: ~10.7 hours on AMD RX 7900 XTX (ROCm 6.2)
- Exports to GGUF Q8_0 and registers with Ollama

**Profile Definitions (`profiles_v2.py`, 423 lines):**
30 profiles across 20 countries, including:
- 10 V1 profiles from autopilot.py (Kuwait core + GCC)
- 20 new V2 profiles (global diversity: Nigeria, Brazil, Japan, Germany, India, Turkey, Kenya, Australia, Mexico, Chile, etc.)
- Each profile includes: id, role, location, context, interests, tracked_companies, tracked_institutions, tracked_industries, diversity_tag

### 4.7 Self-Improvement Pipeline

The self-improvement system operates on five tiers:

**Tier 1: Implicit User Feedback**
- Actions: save, dismiss, rate (1-10 user score), thumbs up/down
- Stored in `user_feedback` table with `action`, `ai_score`, `user_score`, `note`
- Collected from the dashboard via `POST /api/feedback`

**Tier 2: Claude Opus Distillation (`distill.py`, 601 lines)**
- Teacher model: `claude-opus-4-5-20251101`
- Re-scores items that the local model already scored
- Saves corrections when disagreement >= threshold (default 2.0 delta)
- Corrections stored in `user_feedback` table with `action='distill_correction'`
- Supports both real-time API calls and Batch API (50% cheaper, async)
- Cost: ~$0.40 per distillation cycle

**Tier 3: Training Data Export (`export_training.py`, 539 lines)**
- Converts corrections + user feedback into ChatML JSONL format
- **Critical invariant:** Training format must be character-for-character identical to inference format in `scorer_adaptive.py`. Misalignment causes the trained model to fail at parsing.
- Supports merging with base distilled training data
- Legacy profile map reconstructs tracked fields for old DB corrections

**Tier 4: LoRA Fine-Tuning (`train_lora.py`, 1,344 lines)**
- Auto-selects model tier based on available VRAM:
  - 20+ GB: Qwen3-8B (best quality)
  - 10+ GB: Qwen3-4B (balanced)
  - 4+ GB: Qwen3-1.7B (lightweight)
- Supports both Unsloth (CUDA) and PEFT (ROCm)
- Pipeline: train -> merge -> GGUF export -> Ollama register -> config.yaml update
- `WeightedCompletionDataCollator`: preserves per-sample loss weights through tokenization
- PEFT meta device fix: after `get_peft_model()`, deletes `hf_device_map` and forces `.to("cuda:0")` to prevent gradient crashes
- Incremental vs full retrain: default is incremental (builds on `data/models/current_base/`). Full retrain triggers when agreement rate drops below 35% or profile diversity is too low.

**Tier 5: Automation Scripts**
- `learn_cycle.sh` (45 lines): Export corrections -> merge training data -> DoRA training -> verify
- `stratos_overnight.sh` (190 lines): Full overnight loop: start dashboard -> autopilot (scan -> distill -> train) -> manage budget -> prevent system sleep
- `autopilot.py` (959 lines): Fully autonomous loop:
  1. Picks a diverse professional profile from 17+ templates
  2. Calls `/api/suggest-context` for AI-generated tracking context
  3. Calls `/api/generate-profile` for categories + tickers
  4. Saves config, runs scan with diverse data
  5. Sends to Claude Opus for re-scoring (distillation)
  6. Restores original profile
  7. Every N cycles: triggers LoRA fine-tune
  8. Manages budget, persistent state, profile diversity guards

### 4.8 Briefing Generator (`processors/briefing.py`, 455 lines)

Generates LLM-powered intelligence summaries. Dynamically builds system prompts from the user's profile (role, location, tracked categories, interests). The briefing is the "voice" of the agent -- formal, professional, focused on actionable intelligence.

Uses the general inference model (`qwen3:30b-a3b`) rather than the scorer model.

### 4.9 Profile Generator (`processors/profile_generator.py`, 400 lines)

AI-powered category generation from role/location/context. Given a user's professional profile, generates:
- 4-7 broad tracking categories with 5-8 keyword items each
- Relevant market tickers (Yahoo Finance format)
- News time limit recommendation
- Profile context summary for AI scoring

Enforces design rules: employers and banks must be separate categories, investment assets go in tickers not category items, labels must be full readable names.

---

## 5. Module C: Output and Display

### 5.1 Dashboard (`index.html`, ~1,115 lines)

The main dashboard provides:
- **Executive summary:** Briefing section with AI-generated intelligence summary
- **News feed:** Articles sorted by relevance score with visual indicators:
  - Color-coded score bars (red = critical, orange = high, yellow = medium)
  - Score badge with numeric value and reason tooltip
  - Source attribution and timestamp
- **Score-based filtering:** Filter by score range (critical, high, medium, all)
- **Category tabs:** Dynamic tabs matching user's configured categories
- **Scan control:** Start/Stop Scan button with 3 states (idle green, scanning red with progress, stopping grey)
- **Responsive layout:** Three-column desktop, single-column mobile with bottom navigation

### 5.2 Markets Panel (`markets-panel.js`, 1,844 lines)

Advanced market visualization using TradingView Lightweight Charts:
- **Chart types:** Candlestick, line, area
- **Timeframes:** 1D (1m candles), 5D (5m), 1M (1d), 1Y (1d), 5Y (1wk)
- **Technical overlays:** Fibonacci retracement, auto-trend lines, pattern detection
- **Layout:** Side-by-side comparison with draggable cards (up to 4 mini charts)
- **Export:** PNG chart export
- **Keyboard shortcuts:** Navigation between tickers and timeframes
- **Price alerts:** Visual indicators for threshold-breaking movements
- **Fullscreen focus mode:** Expand any chart to full viewport with backdrop overlay (desktop and mobile)
- **Market agent:** In-panel AI chat for market analysis queries

### 5.3 Strat Agent (`agent.js` 937 lines + `routes/agent.py` 817 lines)

An AI chat assistant with Ollama tool-use capabilities:

**Tools available to the agent:**
| Tool | Description |
|------|-------------|
| `web_search` | Real-time Google search via Serper API (web or news mode) |
| `manage_watchlist` | View, add, or remove market tickers from watchlist |
| `manage_categories` | Add/remove keywords, list/toggle news categories |

**Features:**
- Streaming Ollama responses with tool-call detection
- Ticker commands: `$NVDA` inline syntax for quick ticker info
- Conversation export/import
- Context-aware: agent knows user's profile, categories, and current feed data
- Uses `qwen3:30b-a3b` inference model
- Dedicated full-screen mobile agent view (see §5.11)

### 5.4 Onboarding Wizard (`wizard.js` 2,626 lines, `wizard_v4.html` 1,131 lines)

Multi-step guided onboarding with Quick/Deep mode toggle:

| Step | Purpose | AI Assistance |
|------|---------|---------------|
| 1 | Role / Location / Career stage | -- |
| 2 | Category selection | `/api/wizard-preselect` -- AI pre-selects relevant categories |
| 3 | Per-category keyword customization | `/api/wizard-tab-suggest` -- AI suggests keywords; `/api/wizard-rv-items` -- AI generates role-aware entities |
| 4 | Market watchlist + RSS feeds | -- |

Wizard routes are handled by `routes/wizard.py` (342 lines) which makes lightweight Ollama calls using the inference model.

### 5.5 Settings Panel (`settings.js`, 2,529 lines)

Comprehensive configuration UI with Simple and Advanced modes:

**Simple Mode:**
- Quick profile (role, location, context) with AI context suggestion
- Category generation with rollback/redo support
- Dynamic tracking categories management
- Category library (saved presets)
- News freshness selector (24h, 7d, 30d)

**Advanced Mode:**
- Full category editor with keyword CRUD (add, remove, reorder, toggle)
- Ticker management (add/remove symbols, human-readable names, drag-reorder)
- RSS feed configuration (finance catalog, politics catalog, custom URLs)
- Search provider settings (Serper API key, Google API key/CX)
- Dynamic category sync: changes in settings automatically sync to legacy keyword fields

### 5.6 Theme System (`theme-editor.js`, 356 lines + `styles.css`)

7 built-in color themes with full CSS custom property customization:

| Theme | Base | Accent | Character |
|-------|------|--------|-----------|
| Midnight | Deep navy | Emerald (#10b981) | Default dark |
| Arctic | Ocean blue | Cyan (#38bdf8) | Cool blue |
| Noir | Pure black | Violet (#8b5cf6) | High contrast |
| Coffee | Chocolate | Amber (#f59e0b) | Warm dark |
| Latte | Warm beige | Golden (#d4a855) | Light mode |
| Rose | Deep plum | Hot pink (#f43f5e) | Vibrant |
| Terminal | Black | Phosphor green (#39ff14) | Hacker aesthetic |

Each theme defines 20+ CSS custom properties (`--bg-primary`, `--accent`, `--text-primary`, `--chart-line`, etc.). The theme editor allows per-variable overrides with real-time preview and localStorage persistence.

### 5.7 Scan History (`scan-history.js`, 209 lines)

Scan log viewer with performance metrics:
- Timestamp, elapsed time, items fetched/scored
- Score breakdown (critical/high/medium/noise)
- Rule-scored vs LLM-scored counts
- Error logging

### 5.8 SSE Protocol

Server-Sent Events for live updates. The `/api/events` endpoint maintains persistent connections with 15-second heartbeats.

**Event Types:**

| Event | Payload | Description |
|-------|---------|-------------|
| `scan` | `{status, progress, scored, total}` | Scan progress update |
| `complete` | `{data_version}` | Scan completed successfully |
| `scan_cancelled` | `{scored, total, high, medium, data_version}` | Scan was cancelled by user |
| `scan_error` | `{message}` | Scan failed with error |
| `status` | `{is_scanning, stage, ...}` | Initial status on connect |
| `critical_signal` | `{title, score, reason, url, category}` | Item scored 9.0+ |
| `shutdown` | `{reason}` | Server shutting down |

### 5.9 Scan Cancellation

Graceful scan cancellation is implemented via `threading.Event`:
1. User clicks "Stop Scan" -> `POST /api/scan/cancel` sets `self._scan_cancelled`
2. During scoring, `cancel_check` callback is called every 10 tokens during Ollama streaming
3. When cancellation is detected, partial results are saved to DB and output file
4. SSE broadcasts `scan_cancelled` event with partial statistics
5. Button transitions: idle (green) -> scanning (red with progress) -> stopping (grey) -> idle

### 5.10 Dashboard Export

`GET /api/export` supports two formats:
- **CSV:** Flat export of news signals (score, title, source, category, URL, etc.)
- **JSON:** Full diagnostic export including current feed, scan history, category stats, top signals, profile info

### 5.11 Mobile Experience (`mobile.js`, 722 lines)

A complete mobile-native experience built with vanilla JavaScript touch event handling. The entire mobile engine is wrapped in an IIFE that only activates on touch-capable devices.

#### 5.11.1 Touch Gesture Engine

| Gesture | Trigger Zone | Action | Threshold |
|---------|-------------|--------|-----------|
| Swipe right from edge | Left 28px edge | Open sidebar drawer | 60px or 0.35 px/ms velocity |
| Swipe left on sidebar | Sidebar + backdrop | Close sidebar drawer | 60px or 0.35 px/ms velocity |
| Swipe right on card | News card (`[data-card-idx]`) | Save signal | 85px |
| Swipe left on card | News card | Dismiss signal | 85px |
| Pull down at top | Main content at `scrollTop <= 5` | Trigger scan (pull-to-refresh) | 70px with 0.5x resistance |

**Conflict Resolution:**
- Edge zone (left 28px) is exclusive to sidebar swipe
- Card swipe has priority over view-level gestures (excluded via target check)
- Vertical scroll cancels horizontal gesture (`|dy| > |dx|`)
- Pull-to-refresh only activates when `scrollTop <= 5`

#### 5.11.2 Bottom Navigation Bar

Fixed 5-tab navigation at the bottom of the screen (visible at ≤768px):

| Tab | Icon | Action |
|-----|------|--------|
| Home | `layout-dashboard` | Shows main feed (highlights for any category/feed tab) |
| Markets | `trending-up` | Switches to markets panel view |
| Agent | `bot` | Opens dedicated mobile agent full-screen view |
| Saved | `bookmark` | Shows saved signals |
| Settings | `settings` | Opens settings panel |

- Active state uses theme `var(--accent)` color
- Home tab highlights for any non-markets/settings/saved view
- Agent tab manages its own `_mobileAgentOpen` state
- Syncs with `setActive()` via monkey-patching
- Landscape mode: horizontal layout with row-direction tabs
- Safe area padding for iOS home bar (`env(safe-area-inset-bottom)`)

#### 5.11.3 Mobile Agent View

Dedicated full-screen agent chat accessible only from the mobile bottom nav:

- Full-page overlay (`#mobile-agent-view`) with header, messages, and input
- Syncs messages from the desktop `#agent-messages` container via polling (500ms interval, 60s max)
- Sends messages by copying to the real `#agent-input` and calling `sendAgentMessage()`
- Supports hardware back button (pushes history state, listens for `popstate`)
- Dark-themed with theme-aware styling

#### 5.11.4 Mobile Layout Adaptations (CSS-only, ≤768px)

All mobile layout changes are CSS-only and do not affect desktop:

- **Hidden on mobile:** `#sidebar-column` (markets sidebar), `#agent-panel`, keyboard shortcuts button
- **Compact header:** Single row with reduced buttons, hidden redundant controls (scan history, help)
- **Feed cards:** 2-column compact grid with hidden summaries, action buttons, and score reasons
- **Feed card interactions:** Full content visible on tap/expand
- **Bottom nav clearance:** Main content has `padding-bottom: 60px` for bottom bar

#### 5.11.5 Fullscreen Chart Focus Mode

Available on both desktop and mobile. Expands any chart container to full viewport:

- Uses **CSS `position: fixed`** on the existing chart element (not DOM movement) to preserve Lightweight Charts canvas rendering
- Background backdrop with `backdrop-filter: blur(4px)`
- Header bar with ticker name and close button
- ResizeObserver on chart container fires automatically when dimensions change
- Escape key (desktop) and back button (mobile) to close
- History state management for mobile back button

#### 5.11.6 Saved View Grid/List Toggle

Toggle button in the Saved tab header switches between:
- **List view** (default): Full card display with summaries and actions
- **Grid view:** Compact 2-column grid (3 columns on desktop ≥1024px) with truncated titles and hidden details

Preference persists via `localStorage.getItem('savedViewGrid')`.

### 5.12 Progressive Web App (PWA)

StratOS is installable as a standalone app on mobile and desktop:

**manifest.json:**
```json
{
  "name": "STRAT_OS - Strategic Intelligence",
  "short_name": "STRAT_OS",
  "display": "standalone",
  "start_url": "/",
  "background_color": "#050810",
  "theme_color": "#050810",
  "orientation": "any",
  "icons": [
    {"src": "/icon-192.png", "sizes": "192x192", "purpose": "any maskable"},
    {"src": "/icon-512.png", "sizes": "512x512", "purpose": "any maskable"}
  ],
  "categories": ["news", "productivity", "utilities"]
}
```

**Service Worker (`sw.js`, ~92 lines):**
- Cache name: `stratos-v3`
- Pre-caches 25 app shell assets (HTML, CSS, JS, icons)
- API calls: network-first strategy with cache fallback
- Static assets: cache-first strategy with background update
- SSE connections (`/api/events`) excluded from caching
- Cache invalidation on version bump

**Install Prompt (`mobile.js`):**
- Captures `beforeinstallprompt` event and shows styled install banner
- iOS fallback: shows "tap Share → Add to Home Screen" instructions
- Auto-dismisses for 7 days after user dismisses
- Styled with theme-aware CSS variables

### 5.13 User Profile Settings

Account management section embedded within the Settings panel:

**Display:**
- Profile avatar preview (large circle with initials, themed colors)
- Avatar initials input (1-3 characters, uppercase, live preview)
- Display name field
- Collapsible "Change PIN" section with current/new PIN fields

**Backend (`POST /api/update-profile`):**
- Validates session token (requires authentication)
- Updates profile YAML fields: `profile.name`, `profile.avatar`, `security.pin_hash`
- Current PIN verification required before PIN change (SHA-256 hash comparison)
- Removes plain-text PIN if present (migration to hash-only)
- Updates live config in memory for name/role/location/avatar changes

**Mobile Integration:**
- Theme picker + profile section duplicated into mobile settings footer (`initMobileSettings` in `mobile.js`)
- MutationObserver syncs sidebar profile changes to mobile settings in real-time
- Login/logout section shown based on auth state

**Search Fallback (DuckDuckGo):**
- Fixed broken import (`from duckduckgo_search import DDGS`)
- Added automatic DDG fallback when Serper returns 0 results (e.g., credit exhaustion)
- Fallback applies per-query: each failed Serper query individually falls back to DDG

---

## 6. Module D: API Reference

### 6.1 GET Endpoints

| Endpoint | Description | Auth Required |
|----------|-------------|---------------|
| `/` | Serve index.html (root) | No |
| `/api/auth-check` | Check authentication status, list profiles for device | No |
| `/api/data`, `/news_data.json` | Get latest news items (with ETag + gzip) | Yes |
| `/api/refresh` | Trigger full scan (news + market) in background | No |
| `/api/refresh-market` | Trigger market-only scan (fast, no API calls) | Yes |
| `/api/refresh-news` | Trigger news-only scan (slower, uses search APIs) | Yes |
| `/api/status` | Get system status + recent scan log | No |
| `/api/scan/status` | Get current scan progress (scored/total/high/medium/cancelled/stage) | No |
| `/api/events` | SSE event stream (persistent connection) | Yes |
| `/api/search-status` | Serper/Google credits remaining | Yes |
| `/api/scan-log` | Scan history (last 50 entries) | Yes |
| `/api/config` | Get current editable configuration (profile, market, news, search) | Yes |
| `/api/profiles` | List available profiles | Yes |
| `/api/top-movers` | Top moving market assets (alerts) | Yes |
| `/api/agent-status` | Chat agent status | Yes |
| `/api/feedback-stats` | User feedback statistics | Yes |
| `/api/finance-news` | Finance RSS feed articles (unscored) | Yes |
| `/api/politics-news` | Politics RSS feed articles (unscored) | Yes |
| `/api/custom-news` | User-defined custom RSS feed articles | Yes |
| `/api/feed-catalog/finance` | Finance RSS feed catalog with toggle states | Yes |
| `/api/feed-catalog/politics` | Politics RSS feed catalog with toggle states | Yes |
| `/api/export` | Dashboard data export (CSV or JSON format) | Yes |
| `/api/health` | System health (uptime, Ollama status, DB size, last scan, SSE clients, memory) | Yes |
| `/api/shadow-scores` | Shadow scoring comparisons + agreement stats | Yes |

### 6.2 POST Endpoints

| Endpoint | Description | Auth Required |
|----------|-------------|---------------|
| `/api/scan/cancel` | Cancel active scan (sets `threading.Event`) | No |
| `/api/auth` | Login with PIN (returns session token) | No |
| `/api/register` | Register new user with profile and PIN | No |
| `/api/logout` | Logout session (delete token) | No |
| `/api/config` | Save updated configuration (syncs to profile YAML) | Yes |
| `/api/serper-credits` | Check or refill Serper API credits | Yes |
| `/api/feedback` | Submit user feedback (save/dismiss/rate/thumbs) | Yes |
| `/api/ask` | Quick AI query (single-shot Ollama call) | Yes |
| `/api/suggest-context` | Role-aware context suggestions via LLM | No |
| `/api/generate-profile` | AI generate categories + tickers from profile | No |
| `/api/wizard-preselect` | AI preselect categories for wizard Step 2 | No |
| `/api/wizard-tab-suggest` | AI suggest keywords for wizard tab | No |
| `/api/wizard-rv-items` | AI generate role-aware entities for wizard Step 3 review | No |
| `/api/agent-chat` | Send chat message to Strat Agent (streaming SSE response) | Yes |
| `/api/profiles` | Load/switch/save/delete active profile presets | Yes |
| `/api/update-profile` | Update user profile (name, avatar, PIN) with current-PIN verification | Yes |

### 6.3 Response Formats

**News Data Response (`GET /api/data`):**
```json
{
  "market": {
    "SYMBOL": {
      "name": "Display Name",
      "data": {
        "1m": {"price": 0.0, "change": 0.0, "high": 0.0, "low": 0.0, "history": [...]},
        "5m": {...},
        "1d_1mo": {...},
        "1d_1y": {...},
        "1wk": {...}
      }
    }
  },
  "news": [
    {
      "id": "abc123def456",
      "title": "Article Title",
      "url": "https://...",
      "summary": "Brief description",
      "content": "Full article text",
      "timestamp": "2026-02-22T12:00:00",
      "source": "Bloomberg",
      "root": "global",
      "category": "tech_trends",
      "score": 8.2,
      "score_reason": "Directly relevant to Kuwait AI strategy"
    }
  ],
  "last_updated": "Feb 22, 12:00 PM",
  "alerts": [...],
  "briefing": {...},
  "timestamps": {"market": "...", "news": "..."},
  "meta": {"version": "3.0", "generated_at": "...", "news_count": 50, "critical_count": 2, "high_count": 12}
}
```

**Scan Status Response (`GET /api/scan/status`):**
```json
{
  "running": true,
  "progress": "42/100",
  "scored": 42,
  "total": 100,
  "high": 8,
  "medium": 15,
  "cancelled": false,
  "stage": "scoring"
}
```

**Scan Stages:** `idle` -> `starting` -> `market` -> `news` -> `scoring` -> `discovery` -> `briefing` -> `output` -> `complete` | `cancelled` | `error`

---

## 7. Database Schema

SQLite database with WAL mode (`database.py`, 685 lines). Singleton pattern via `get_database()`. Thread-safe commits via `threading.Lock`.

**Performance Pragmas:**
```sql
PRAGMA foreign_keys = ON
PRAGMA journal_mode = WAL
PRAGMA busy_timeout = 5000
PRAGMA synchronous = NORMAL
PRAGMA cache_size = -8000     -- 8MB cache
PRAGMA temp_store = MEMORY
```

### 7.1 Tables

**`news_items`** -- Scored news articles
| Column | Type | Constraints |
|--------|------|-------------|
| id | TEXT | PRIMARY KEY |
| title | TEXT | NOT NULL |
| url | TEXT | UNIQUE NOT NULL |
| summary | TEXT | |
| source | TEXT | |
| root | TEXT | |
| category | TEXT | |
| score | REAL | DEFAULT 0.0 |
| score_reason | TEXT | |
| timestamp | TEXT | |
| fetched_at | TEXT | NOT NULL |
| shown_to_user | INTEGER | DEFAULT 0 |
| dismissed | INTEGER | DEFAULT 0 |

**`market_snapshots`** -- Market data history
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| symbol | TEXT | NOT NULL |
| name | TEXT | |
| interval | TEXT | NOT NULL |
| price | REAL | |
| change_percent | REAL | |
| high | REAL | |
| low | REAL | |
| history_json | TEXT | |
| snapshot_at | TEXT | NOT NULL |

**`entities`** -- Tracked and discovered entities
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| name | TEXT | UNIQUE NOT NULL |
| category | TEXT | |
| is_discovered | INTEGER | DEFAULT 0 |
| discovered_at | TEXT | |
| is_active | INTEGER | DEFAULT 1 |
| mention_count | INTEGER | DEFAULT 0 |

**`entity_mentions`** -- Entity frequency history
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| entity_name | TEXT | NOT NULL |
| mention_count | INTEGER | DEFAULT 1 |
| recorded_at | TEXT | NOT NULL |

**`scan_log`** -- Scan performance history
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| started_at | TEXT | NOT NULL |
| elapsed_secs | REAL | |
| items_fetched | INTEGER | DEFAULT 0 |
| items_scored | INTEGER | DEFAULT 0 |
| critical | INTEGER | DEFAULT 0 |
| high | INTEGER | DEFAULT 0 |
| medium | INTEGER | DEFAULT 0 |
| noise | INTEGER | DEFAULT 0 |
| rule_scored | INTEGER | DEFAULT 0 |
| llm_scored | INTEGER | DEFAULT 0 |
| error | TEXT | |

**`briefings`** -- Generated intelligence briefings
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| content_json | TEXT | NOT NULL |
| generated_at | TEXT | NOT NULL |

**`user_feedback`** -- User feedback and distillation corrections
| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| news_id | TEXT | NOT NULL |
| title | TEXT | |
| url | TEXT | |
| root | TEXT | |
| category | TEXT | |
| ai_score | REAL | |
| user_score | REAL | |
| note | TEXT | |
| action | TEXT | NOT NULL |
| created_at | TEXT | NOT NULL |
| profile_role | TEXT | (migration) |
| profile_location | TEXT | (migration) |
| profile_context | TEXT | (migration) |

**`shadow_scores`** -- Shadow scoring comparisons
| Column | Type | Constraints |
|--------|------|-------------|
| scan_id | INTEGER | |
| news_id | TEXT | |
| title | TEXT | |
| category | TEXT | |
| primary_scorer | TEXT | |
| primary_score | REAL | |
| shadow_scorer | TEXT | |
| shadow_score | REAL | |
| delta | REAL | |
| created_at | TEXT | |

### 7.2 Indexes

```sql
CREATE INDEX idx_news_fetched ON news_items(fetched_at);
CREATE INDEX idx_news_score ON news_items(score);
CREATE INDEX idx_market_symbol ON market_snapshots(symbol, snapshot_at);
CREATE INDEX idx_mentions_date ON entity_mentions(recorded_at);
CREATE INDEX idx_feedback_news ON user_feedback(news_id);
CREATE INDEX idx_feedback_date ON user_feedback(created_at);
```

### 7.3 Key Operations

- `save_news_item()`: INSERT OR IGNORE (dedup by URL)
- `update_news_score()`: Update score and reason for existing item
- `get_recent_news(hours, min_score)`: Fetch recent items above threshold
- `dismiss_news()`: Mark item as dismissed (filtered from feed)
- `is_url_seen()`: Deduplication check before fetching
- `cleanup_old_data(days)`: Periodic cleanup of old records (every 10th scan)
- Auto-distillation trigger: configurable via `distillation.auto_every` in config

---

## 8. Configuration System

### 8.1 Global Configuration (`config.yaml`, 262 lines)

The central configuration file with these sections:

**`scoring:`**
```yaml
scoring:
  model: stratos-scorer-v2      # Fine-tuned scoring model name in Ollama
  inference_model: qwen3:30b-a3b     # General inference model for agent/suggestions
  wizard_model: qwen3:14b            # Lighter model for wizard (faster, less over-thinking)
  ollama_host: http://localhost:11434
  critical_min: 9.0
  high_min: 7.0
  medium_min: 5.0
  filter_below: 5.0
  forbidden_score: 5.0
```

**`news:`** Category definitions with keywords and root classification.

**`market:`** Ticker watchlist with symbols, names, intervals, alert threshold.

**`search:`** Provider selection (serper/google/duckduckgo), API keys, credit tracking.

**`profile:`** User role, location, context description, interests.

**`dynamic_categories:`** Array of category objects with id, label, icon, items (keywords), enabled, scorer_type, root.

**`discovery:`** Entity discovery settings (enabled, baseline window, min mentions, rising multiplier, auto-track).

**`cache:`** TTL settings (news: 900s, market: 60s, discovery: 3600s).

**`schedule:`** Background scanning (enabled, interval minutes -- default 30).

**`system:`** Database file, frontend port (8080), log level, max news items, output file.

**`extra_feeds_finance:` / `extra_feeds_politics:`** Toggle maps for curated RSS feeds.

**`custom_feeds:`** User-defined RSS feed URLs.

### 8.2 Profile Configuration (`profiles/*.yaml`)

Per-user profiles stored as YAML files. Each profile contains:

```yaml
profile:
  role: "Computer Engineer"
  location: "Kuwait"
  context: "Track: ... Deprioritize: ..."
  interests: []
security:
  pin_hash: "sha256-hex-string"
  devices: ["device-fingerprint-id"]
market:
  tickers: [{symbol, name, category}]
  intervals: {}
  alert_threshold_percent: 5.0
news:
  timelimit: w
  career: {root, keywords, queries}
  finance: {root, keywords, queries}
  regional: {root, keywords, queries}
  tech_trends: {root, keywords, queries}
  rss_feeds: []
dynamic_categories:
  - {id, label, icon, items, enabled, scorer_type, root}
extra_feeds_finance: {feed_id: true/false}
extra_feeds_politics: {feed_id: true/false}
custom_feeds: []
custom_tab_name: "Custom"
```

When a user logs in, their profile YAML is loaded into memory (not written to `config.yaml`). Profile switching clears all profile-specific keys and applies the new profile's config, preserving API keys and system-level settings.

### 8.3 Dynamic Category Reclassification

Items fetched from RSS/scrapers arrive with generic categories (`tech`, `general`). The `_reclassify_dynamic()` method in `main.py` re-tags them with dynamic category IDs so the scorer routes them correctly and the frontend filters them into the right tabs.

**Matching Strategy:**
- Short items (1-2 words, <=12 chars): word-boundary regex for precision
- Longer items (3+ words): match if ALL significant words appear in text
- Single long words: substring match (safe for words > 4 chars)
- Best-match wins: item assigned to category with highest match count

### 8.4 Environment Variables (`.env`)

```bash
ANTHROPIC_API_KEY=sk-ant-...   # Claude Opus API for distillation
SERPER_API_KEY=...              # Serper.dev Google search
GOOGLE_API_KEY=...              # Google Custom Search API
GOOGLE_CSE_ID=...               # Google Custom Search Engine ID
```

---

## 9. File Structure

### 9.1 Backend (`backend/`)

```
backend/
|-- main.py                    # 2,110 lines — Main orchestrator + HTTP server
|-- server.py                  # HTTP server, route dispatch, CORSHandler
|-- database.py                # 685 lines — SQLite database manager
|-- auth.py                    # AuthManager: sessions, rate limiting, PIN auth
|-- sse.py                     # SSEManager: Server-Sent Events broadcasting
|-- config.yaml                # 262 lines — Central configuration
|-- CLAUDE.md                  # Project instructions for Claude Code
|-- STRATOS_FRS_v7.md          # This document
|-- strat_os.db                # SQLite database file
|
|-- fetchers/
|   |-- __init__.py
|   |-- news.py                # 917 lines — Multi-source news fetcher
|   |-- market.py              # 316 lines — Yahoo Finance market data
|   |-- discovery.py           # 314 lines — Entity discovery engine
|   |-- extra_feeds.py         # 274 lines — RSS feed catalog
|   |-- serper_search.py       # 332 lines — Serper API client
|   |-- google_search.py       # 300 lines — Google Custom Search client
|   |-- kuwait_scrapers.py     # 589 lines — Kuwait-specific scrapers
|
|-- processors/
|   |-- __init__.py
|   |-- scorer.py              # 2,138 lines — Hardcoded CPEG/Kuwait scorer
|   |-- scorer_base.py         # Shared scorer infrastructure
|   |-- scorer_adaptive.py     # 1,421 lines — Profile-adaptive scorer
|   |-- briefing.py            # 455 lines — LLM briefing generator
|   |-- profile_generator.py   # 400 lines — AI profile/category generator
|
|-- routes/
|   |-- agent.py               # 817 lines — Chat agent with tool-use
|   |-- config.py              # 215 lines — Config save handler
|   |-- generate.py            # 254 lines — Profile generation route
|   |-- wizard.py              # 342 lines — Onboarding wizard routes
|   |-- helpers.py             # JSON response / LLM output cleaning utilities
|
|-- distill.py                 # 601 lines — Claude Opus distillation pipeline
|-- export_training.py         # 539 lines — Training data exporter
|-- train_lora.py              # 1,344 lines — LoRA fine-tuning pipeline
|-- autopilot.py               # 959 lines — Autonomous self-improvement loop
|-- migrations.py              # Database migrations
|-- evaluate_scorer.py         # Scorer evaluation tool
|
|-- stratos.sh                 # 86 lines — Launcher script
|-- learn_cycle.sh             # 45 lines — Manual learning cycle
|-- stratos_overnight.sh       # 190 lines — Overnight autonomous training
|-- setup_rocm_training.sh     # 163 lines — ROCm environment setup
|
|-- profiles/
|   |-- Ahmad.yaml             # Example profile
|   |-- Abdullah.yaml          # Example profile
|   |-- Ahmad/                 # Per-profile data directory
|   |-- Abdullah/              # Per-profile data directory
|
|-- data/
|   |-- memory.json            # Scoring memory (persistent patterns)
|   |-- serper_query_tracker.json
|   |-- google_query_tracker.json
|   |-- training_data.jsonl    # V1 training data
|   |-- training_merged.jsonl  # Merged training data
|   |-- v1_calibration_table.json
|   |-- models/                # Trained model checkpoints + GGUF files
|   |-- v2_pipeline/
|   |   |-- profiles_v2.py     # 423 lines — 30 profile definitions
|   |   |-- stage2_collect.py  # 377 lines — Article collection
|   |   |-- stage3_score.py    # 395 lines — Batch API scoring
|   |   |-- stage3_resume.py   # 319 lines — Resume interrupted scoring
|   |   |-- stage4_prepare.py  # 346 lines — Data preparation
|   |   |-- train_v2.py        # 1,120 lines — V2 training script
|   |   |-- phase1_merge.py    # 87 lines — Merge utility
|   |   |-- training_output/   # Training checkpoints + logs
|   |-- v3_pipeline/
|       |-- training_output/   # V3 training attempt checkpoints
```

### 9.2 Frontend (`frontend/`)

```
frontend/
|-- index.html                 # ~1,115 lines — Main dashboard
|-- onboarding-wizard.html     # 1,413 lines — Wizard (legacy version)
|-- wizard_v2.html             # 716 lines — Wizard iteration 2
|-- wizard_v3.html             # 641 lines — Wizard iteration 3
|-- wizard_v4.html             # 1,131 lines — Wizard iteration 4 (current)
|
|-- app.js                     # 1,358 lines — Main application logic
|-- wizard.js                  # 2,626 lines — Onboarding wizard logic
|-- settings.js                # 2,529 lines — Settings panel
|-- markets-panel.js           # 1,844 lines — TradingView charts panel
|-- agent.js                   # 937 lines — Chat agent UI
|-- market.js                  # 1,004 lines — Market data display
|-- feed.js                    # 784 lines — News feed rendering
|-- mobile.js                  # 722 lines — Mobile gestures, bottom nav, PWA
|-- auth.js                    # 429 lines — Authentication UI
|-- theme-editor.js            # 356 lines — Theme customization
|-- ui.js                      # 264 lines — UI utilities
|-- nav.js                     # 245 lines — Navigation logic
|-- scan-history.js            # 209 lines — Scan log viewer
|-- sw.js                      # 92 lines — Service worker
|-- tailwind.config.js         # 26 lines — Tailwind configuration
|
|-- styles.css                 # ~896 lines — Custom styles + 7 themes + mobile
|-- tailwind-input.css         # 3 lines — Tailwind directives
|-- tailwind-built.css         # Pre-compiled Tailwind output (37KB)
|
|-- manifest.json              # PWA manifest
|-- icon-192.png               # PWA icon (192x192)
|-- icon-512.png               # PWA icon (512x512)
```

---

## 10. Key Design Decisions and Technical Notes

### 10.1 Qwen3 Think Block Handling

Do NOT set `"think": False` in Ollama API calls — Qwen3 ignores it and leaks reasoning as plain text into the `content` field, consuming the entire `num_predict` token budget before producing any useful output. Instead, omit the `think` parameter entirely. In default mode, Ollama separates reasoning into a `thinking` field while `content` holds the clean answer. Always strip `<think>...</think>` blocks from content as a safety net. Set `num_predict` large enough to cover both thinking tokens (~1000-2000) and output tokens.

### 10.2 Training/Inference Format Alignment

The training data format (`export_training.py`) must be character-for-character identical to what the model sees at inference time in `scorer_adaptive.py`. Both paths produce/expect:
- System prompt with profile context, tracked entities, and scoring rubric
- User message with article title, source, category, and summary
- Assistant response: `SCORE: X.X | REASON: <explanation>`

Misalignment causes the fine-tuned model to fail at parsing output, producing garbled or empty scores.

### 10.3 PEFT Meta Device Fix

After `get_peft_model()`, the code deletes `hf_device_map` and forces `.to("cuda:0")` to prevent gradient crashes during training on ROCm. This is a workaround for a known issue where PEFT creates model tensors on a "meta" device that cannot compute gradients.

### 10.4 Incremental vs Full Retrain

Default training mode is incremental (builds on `data/models/current_base/`). Full retrain from the base Qwen3 model triggers when:
- Agreement rate drops below 35% (model has drifted too far)
- Profile diversity is too low (training data is too narrow, risking catastrophic forgetting)

### 10.5 Forbidden 5.0 Enforcement

Scores in the range 4.8-5.2 are nudged away from 5.0 because it sits on the exact noise/signal boundary. Models tend to hedge by outputting 5.0 when uncertain, which provides no useful signal. The nudge forces a commit: either noise (4.8) or signal (5.3).

### 10.6 No Web Framework

The backend uses Python's built-in `http.server.HTTPServer` with `ThreadingMixIn`. This was a deliberate choice to minimize dependencies and keep the system self-contained. All routing, middleware (auth, rate limiting, CORS, gzip, ETag), and SSE are implemented inline in the `CORSHandler` class. This approach trades framework conveniences for zero dependency risk and full control.

### 10.7 Dynamic Category System

Categories are not hardcoded in the backend. The user's profile defines dynamic categories (via onboarding wizard or settings), and all downstream systems adapt:
- News fetcher generates queries from category keywords
- Scorer builds relevance rules from category items
- Frontend renders category tabs dynamically
- RSS items are re-classified into matching dynamic categories
- Config save syncs dynamic categories to legacy keyword fields for backward compatibility

### 10.8 Profile Isolation

Each device gets its own view of profiles. When a user registers on a device, their `device_id` is stored in the profile YAML. Other devices cannot see or access that profile. This enables multiple users on shared networks without cross-contamination.

### 10.9 ROCm-Specific Notes

Training on AMD GPUs (ROCm 6.2) requires:
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` environment variable
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` must be DISABLED (causes NaN gradients with DoRA + gradient checkpointing)
- Setup script: `setup_rocm_training.sh` (163 lines)

### 10.10 Mobile Chart Fullscreen Architecture

Lightweight Charts canvases break when DOM elements are moved to a different container. The fullscreen chart mode uses CSS `position: fixed` on the existing chart element instead of DOM movement. The existing `ResizeObserver` attached to the chart container fires automatically when dimensions change, handling the resize without any direct chart API calls.

### 10.11 Streaming Ollama for Cancellation

All scoring Ollama calls use `stream=True` and check `cancel_check` every 10 chunks. This enables responsive scan cancellation without waiting for full model output.

### 10.12 V2 Dual Weighting

Training uses two complementary weighting strategies:
- `WeightedRandomSampler`: oversamples underrepresented score bands for batch balance
- Per-sample loss weighting: gradient emphasis on high-value examples via `WeightedCompletionDataCollator`

### 10.13 AOTRITON Workaround

Do NOT set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` — causes NaN gradients on RX 7900 XTX with DoRA + gradient checkpointing.

---

## 11. Infrastructure Requirements

### 11.1 Reference Hardware

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 9 5900X (12C/24T) |
| RAM | 32 GB DDR4 |
| GPU | AMD Radeon RX 7900 XTX (24 GB VRAM) |
| Storage | 2 TB NVMe SSD |

### 11.2 Ollama Environment Variables

```bash
OLLAMA_MAX_LOADED_MODELS=1    # Only one model in VRAM at a time
OLLAMA_KEEP_ALIVE=10m         # Auto-unload after 10 min idle
OLLAMA_NUM_PARALLEL=1         # Single-user, minimize VRAM duplication
OLLAMA_FLASH_ATTENTION=0      # ROCm 6.2 — flash attention unreliable
ROCR_VISIBLE_DEVICES=0        # Prevent Ollama from using iGPU
```

### 11.3 Software Dependencies

**Runtime:**
- Python 3.12
- Ollama (local LLM server)
- SQLite 3.x (built-in)

**Python Packages (Runtime):**
- `yfinance` -- Yahoo Finance market data
- `feedparser` -- RSS feed parsing
- `duckduckgo-search` (`ddgs`) -- DuckDuckGo news search
- `requests` -- HTTP client
- `beautifulsoup4` -- HTML parsing for article scraping
- `pyyaml` -- YAML configuration parsing

**Python Packages (Training):**
- `transformers` 5.1.0
- `trl` 0.28.0
- `peft` 0.18.1
- `datasets` 4.5.0
- `tokenizers` 0.22.2
- `torch` 2.5.1+rocm6.2 (or CUDA equivalent)

### 11.4 Cost Structure

| Activity | Cost | Frequency |
|----------|------|-----------|
| V2 training pipeline (Batch API) | ~$52 | One-time |
| Distillation cycle | ~$0.40 | Per cycle |
| Overnight autopilot (10 cycles) | ~$4-5 | As needed |
| Runtime scoring | Free | Continuous (local Ollama) |
| Market data | Free | Continuous (yfinance) |
| DuckDuckGo search | Free | Per scan |
| Serper API | Per credit | Per search query |

---

## 12. Running the Application

### 12.1 Quick Start

```bash
# Ensure Ollama is running
ollama serve

# Start the full server (dashboard at http://localhost:8080)
cd backend/
python3 main.py --serve --background

# Or use the launcher script (checks Ollama, registers models, starts server)
bash stratos.sh
```

### 12.2 One-Shot Scan

```bash
python3 main.py --scan
```

### 12.3 Manual Learning Cycle

```bash
# Step by step:
python3 distill.py --hours 168 --limit 200       # Opus re-scores recent items
python3 export_training.py --min-delta 1.5        # Export corrections as JSONL
python3 train_lora.py --epochs 3                  # LoRA fine-tune -> GGUF -> Ollama

# Or automated:
bash learn_cycle.sh
```

### 12.4 Overnight Autonomous Training

```bash
# Default budget (~$8)
bash stratos_overnight.sh

# Custom budget
bash stratos_overnight.sh 10.00

# The script:
# 1. Disables system sleep
# 2. Starts Ollama + registers models
# 3. Starts dashboard server
# 4. Runs autopilot loop (scan -> distill -> train)
# 5. Stops when budget exhausted or Ctrl+C
```

### 12.5 Autopilot

```bash
python3 autopilot.py --cycles 10 --budget 5.00    # 10 cycles, $5 limit
python3 autopilot.py --dry-run                      # Preview without executing
```

---

## 13. Scan Pipeline (Detailed Flow)

A full scan executes the following 7-step pipeline:

1. **Reload Configuration:** Re-reads `config.yaml`, rebuilds all components (fetchers, scorer, briefing generator) with latest settings.

2. **Fetch Market Data:** `MarketFetcher.fetch_all()` retrieves OHLCV data for all tickers across all intervals. Saves snapshots to `market_snapshots` table.

3. **Fetch News:** `NewsFetcher.fetch_all()` dispatches parallel queries to DDG, Serper, RSS. Returns `NewsItem` objects. Items re-classified via `_reclassify_dynamic()`.

4. **Score News Items:** 3-phase scoring pipeline (Phase 1: rules, Phase 2: batch LLM, Phase 3: re-score cascade). Progress reported via SSE. Supports graceful cancellation. Each scored item saved to `news_items` table.

5. **Entity Discovery:** `EntityDiscovery.discover()` analyzes scored items for rising entities. Updates `entities` and `entity_mentions` tables.

6. **Generate Briefing:** `BriefingGenerator.generate_briefing()` creates an AI intelligence summary from market data, alerts, scored news, and discoveries. Saved to `briefings` table.

7. **Build Output:** Compiles market data, scored news, briefing, and alerts into the FRS-compliant JSON schema. Writes to `output/news_data.json`. Broadcasts SSE `complete` event.

**Post-Scan:**
- Scan log saved to `scan_log` table with timing and statistics
- Every 10th scan: periodic DB cleanup (30-day retention)
- Auto-distillation: configurable trigger every N scans (if API key configured)

---

## 14. Profiles System (Detailed)

### 14.1 Profile Structure

Each profile is a YAML file in `profiles/` containing everything needed to personalize the system for a specific user:

**Identity:** role, location, context (natural language description of tracking priorities and depriorities)

**Intelligence Config:** dynamic categories (each with keywords/entities), news time limit, RSS feed preferences, custom feeds

**Market Config:** ticker watchlist, alert thresholds, interval preferences

**Security:** SHA-256 PIN hash, list of authorized device IDs

### 14.2 Profile Lifecycle

1. **Registration:** User completes onboarding wizard or POST to `/api/register`. Profile YAML created with PIN hash and device ID.

2. **Login:** POST to `/api/auth` with profile name, PIN, and device ID. PIN verified against stored hash. Session token issued.

3. **Loading:** On successful login, `_load_profile_config()` clears all profile-specific keys from the live config, resets defaults, then overlays the profile's YAML config. API keys are preserved.

4. **Switching:** User can switch profiles by logging out and logging in to a different profile. Each switch triggers a full config reload.

5. **Saving:** Changes made via Settings or agent commands save to both the live config and the profile YAML file, ensuring persistence across restarts.

### 14.3 Autopilot Profile Templates

The autopilot system includes 17+ profile templates for diverse training data generation:

- 9 Kuwait profiles: CPEG student, ChemE student, PetrolE student, EEE student, CivilE student, Finance student, Medical student, Cybersecurity analyst, Senior geophysicist
- 4 GCC profiles: MechE grad (Saudi), Data Scientist (Dubai), EnviroE grad (Oman), Supply chain analyst (Bahrain)
- 4+ High-contrast profiles: Quant finance analyst, STEM teacher, Biotech researcher (KAUST), Architect (Qatar)

---

## 15. Roadmap

### 15.1 Completed

- V2 scorer model (DoRA fine-tuned Qwen3-8B, 30 profiles, 20 countries)
- Multi-profile system with device isolation
- 4-step onboarding wizard with AI assistance (Quick/Deep modes)
- Stop/cancel scan with graceful partial save
- SSE real-time updates (replaced polling)
- Theme system with 7 color themes + live customization editor
- Markets panel with TradingView charts (5 timeframes, Fibonacci, trend lines)
- Fullscreen chart focus mode (desktop + mobile)
- Strat Agent chat with tool-use (web search, watchlist, categories)
- Self-improvement pipeline (distillation -> export -> training)
- Overnight autonomous training mode
- Dashboard export (CSV and JSON)
- Dynamic category system with reclassification
- Kuwait-specific scrapers (KOC, KNPC, KIPIC)
- **Mobile UI overhaul:** touch gestures, bottom navigation, compact layout
- **Mobile agent page:** dedicated full-screen agent chat for mobile
- **Progressive Web App:** installable with manifest, service worker, offline support
- **Saved view grid/list toggle:** user-selectable compact grid or list display

### 15.2 In Progress

- V2 scorer production hardening and monitoring
- Scoring memory optimization

### 15.3 Future

- **WebSocket Upgrade:** Replace SSE with WebSocket for bidirectional communication
- **Critical Signal Notifications:** Push notifications for critical-scored items (9.0+)
- **Multi-Profile LoRA Adapters:** Separate LoRA weights per user profile instead of one merged model
- **Agent Model Fine-Tuning:** Fine-tune the agent chat model on StratOS-specific conversations
- **Additional Data Sources:** LinkedIn, government portals, specialized industry databases
- **Collaborative Intelligence:** Multi-user shared feeds with team-level scoring

---

## Appendix A: Rate Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/api/refresh` | 2 requests | 60 seconds |
| `/api/refresh-news` | 2 requests | 60 seconds |
| `/api/refresh-market` | 3 requests | 60 seconds |
| `/api/generate-profile` | 5 requests | 60 seconds |
| `/api/suggest-context` | 10 requests | 60 seconds |
| `/api/agent-chat` | 20 requests | 60 seconds |
| `/api/top-movers` | 3 requests | 60 seconds |

## Appendix B: Ollama Model Parameters

**Scoring Model (`stratos-scorer-v2`):**
```
temperature: 0.3
top_p: 0.9
num_ctx: 2048
num_predict: 128
repeat_penalty: 1.3
stop: <|im_end|>, <|endoftext|>, <|im_start|>, <think>, </think>
```

**Inference Model (`qwen3:30b-a3b`):**
```
temperature: 0.2 - 0.3 (varies by use case)
num_predict: 128 - 1500 (varies by use case)
num_ctx: 4096
think: omitted (never set to false — see §10.1)
```

## Appendix C: Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude API key for distillation |
| `SERPER_API_KEY` | Serper.dev Google search |
| `GOOGLE_API_KEY` | Google Custom Search API |
| `GOOGLE_CSE_ID` | Google Custom Search Engine ID |
| `HSA_OVERRIDE_GFX_VERSION` | ROCm GPU version override (set to `11.0.0`) |
| `PYTORCH_HIP_ALLOC_CONF` | PyTorch ROCm memory config (`expandable_segments:True`) |
| `OLLAMA_MAX_LOADED_MODELS` | Limit loaded models (set to `1`) |
| `OLLAMA_KEEP_ALIVE` | Model idle timeout (set to `10m`) |
| `ROCR_VISIBLE_DEVICES` | Restrict GPU visibility (set to `0`) |

## Appendix D: Git Commit History (Recent)

```
a838995 feat: saved grid toggle, fullscreen chart fix, PWA install prompt
c13dca6 fix: mobile-only layout — hide markets on Home, add Agent page, compact card grid
f63d54a fix: mobile UI redesign — compact header, declutter, chart fullscreen
3aade4e feat: mobile gestures — swipe nav, card actions, pull-to-refresh, bottom bar
973bc07 chore: V3 training attempt (killed), fix None weight crash, gitignore
69e4e11 feat: wizard Quick/Deep mode toggle, Qwen3 token fix, CSS overhaul
c68d678 feat: B2 — V3 training data (24,479 samples) and custom path support
1981906 feat: B2 — scorer evaluation tool, V3 data preparation, version tracking
bf2213e feat: A3.3 — PWA offline mode with manifest and app shell caching
ef1e38b feat: A3.2 — mobile-responsive layout enhancements
9ccba88 feat: A2.1 — multi-profile session isolation
```

---

*This document serves as the definitive specification for StratOS V2.1 Production + Mobile. It is intended to provide complete context for any new development session or contributor onboarding.*
