# StratOS Master Plan V2

**Date:** February 22, 2026
**Status:** V2.1 Production System
**Codebase:** ~33,300 lines across 40+ modules (15,400 backend Python + 18,000 frontend JS/HTML/CSS)

---

## 1. System Overview

StratOS is a self-hosted strategic intelligence dashboard. It fetches news from multiple sources, scores each article for personal relevance using a local Ollama-hosted LLM, and presents a prioritized feed via a web UI. The system continuously improves its scoring model through a distillation loop: Claude Opus re-scores articles, disagreements become training data, and LoRA/DoRA fine-tuning updates the local model.

**Core principles:**
- **Profile-driven:** Tell it who you are (role, location, interests, tracked entities), and it tells you what matters. A petroleum engineer in Kuwait and a nurse in Chicago see completely different feeds from the same news sources.
- **Self-improving:** The local 8B-parameter scoring model gets better over time via two feedback loops: (1) user clicks/dismissals/ratings, and (2) Claude Opus teacher distillation where Opus re-scores items and disagreements become training corrections.
- **Zero cloud dependency at runtime:** All inference runs on a local Ollama instance. The only external API calls are optional: DuckDuckGo/Serper for news fetching, and Anthropic API for distillation (~$0.40/cycle).
- **Built by a Computer Engineering student in Kuwait as a solo project.** Every module, from the threaded HTTP server to the LoRA training pipeline, is hand-written without frameworks like Flask, FastAPI, or React.

---

## 2. Architecture Overview

### Data Flow

```
News Sources ─────→ NewsFetcher ─────→ AI Scorer ─────→ SQLite DB ─────→ JSON API ─────→ Frontend
(DDG, Serper,        (parallel,         (rule-based       (WAL mode,       (built-in        (vanilla JS,
 RSS, scrapers)       ThreadPool)        + LLM hybrid)     7 tables)        HTTP server)     SSE push)
                                            ↑                                    ↓
                                     Feedback Loop                         User Feedback
                                            ↑                            (click/dismiss/rate)
                                            ↓                                    ↓
                            LoRA Training ← Export ←──── Distillation (Claude Opus)
```

### Component Map

```
backend/
├── main.py                    # Orchestrator + HTTP server (2,110 lines)
├── database.py                # SQLite manager (685 lines)
├── config.yaml                # Central configuration
├── distill.py                 # Opus teacher distillation (601 lines)
├── export_training.py         # JSONL training data export (539 lines)
├── train_lora.py              # LoRA/DoRA fine-tuning pipeline (1,344 lines)
├── autopilot.py               # Autonomous self-improvement loop (959 lines)
├── fetchers/
│   ├── news.py                # Multi-source news aggregation (917 lines)
│   ├── market.py              # Yahoo Finance market data (316 lines)
│   ├── discovery.py           # Entity frequency tracking (314 lines)
│   ├── extra_feeds.py         # RSS feed catalog (274 lines)
│   ├── serper_search.py       # Serper API client (332 lines)
│   ├── google_search.py       # Google Custom Search API (300 lines)
│   └── kuwait_scrapers.py     # Kuwait-specific news + career portals (589 lines)
├── processors/
│   ├── scorer.py              # AIScorer — hardcoded CPEG/Kuwait (2,138 lines)
│   ├── scorer_adaptive.py     # AdaptiveScorer — any profile (1,421 lines)
│   ├── briefing.py            # LLM intelligence briefings (455 lines)
│   └── profile_generator.py   # AI category generation pipeline (400 lines)
├── routes/
│   ├── agent.py               # Chat agent with tool-use (817 lines)
│   ├── wizard.py              # 4-step onboarding wizard (342 lines)
│   ├── generate.py            # Profile generation endpoint (254 lines)
│   ├── config.py              # Config save/load handler (215 lines)
│   └── helpers.py             # JSON response, SSE, gzip utilities (55 lines)
├── profiles/                  # Per-user YAML profile storage
└── data/
    └── v2_pipeline/           # V2 scorer training pipeline
        ├── profiles_v2.py     # 30 training profiles
        ├── stage2_collect.py  # Article collection (685 articles)
        ├── stage3_score.py    # Claude Opus batch scoring
        ├── stage4_prepare.py  # Train/eval split + contrastive pairs
        └── train_v2.py        # V2 fine-tuning script

frontend/
├── index.html                 # Main SPA shell (1,104 lines)
├── app.js                     # Core orchestration + SSE (1,285 lines)
├── auth.js                    # Login/register + fetch monkey-patch (429 lines)
├── feed.js                    # News feed rendering + feedback (762 lines)
├── market.js                  # Market data + chart integration (1,004 lines)
├── markets-panel.js           # TradingView chart system (1,843 lines)
├── agent.js                   # Chat interface with tool-use (908 lines)
├── settings.js                # Configuration UI (2,529 lines)
├── wizard.js                  # 4-step onboarding (2,543 lines)
├── wizard_v4.html             # Wizard template (1,131 lines)
├── nav.js                     # Sidebar routing (245 lines)
├── ui.js                      # Toasts, modals (264 lines)
├── scan-history.js            # Scan history viewer (209 lines)
├── theme-editor.js            # CSS variable editor (356 lines)
├── sw.js                      # Service worker (37 lines)
└── styles.css                 # Custom styles (518 lines)
```

---

## 3. Backend Modules

### 3.1 Orchestrator — main.py (2,110 lines)

The central brain of StratOS. Contains the `StratOS` class and a built-in threaded HTTP server. No Flask, no FastAPI -- just `http.server.HTTPServer` with `ThreadingMixIn`.

**StratOS class:**
- `__init__()`: Loads config.yaml, initializes Database, MarketFetcher, NewsFetcher, EntityDiscovery, scorer (via `_create_scorer()` factory), BriefingGenerator. Creates `scan_status` dict, SSE client list, cancellation Event.
- `run_scan()`: Full 6-stage pipeline: reload config, fetch market, fetch news, reclassify dynamic categories, AI score (with progress callback + cancellation), entity discovery, generate briefing, build output JSON, save scan log. Every 10th scan triggers DB cleanup. Auto-distillation available every N scans.
- `run_market_refresh()`: Fast market-only refresh. Bypasses cache, updates existing output JSON, broadcasts SSE complete.
- `run_news_refresh()`: News+scoring+briefing refresh. Preserves existing market data. Same scoring pipeline as full scan.
- `_reclassify_dynamic()`: Re-tags RSS items from generic categories (tech, general) into dynamic category IDs. Uses 3 matching strategies: word-boundary regex for short terms (<=4 chars), substring match for 1-2 word terms, all-significant-words match for 3+ word terms.
- `_build_output()`: Constructs FRS-schema JSON with extensions: market, news array (capped at max_news_items), briefing, alerts, timestamps, meta (version, counts).

**Scan cancellation:** Uses `threading.Event`. The `cancel_check` lambda (`lambda: self._scan_cancelled.is_set()`) is passed to `scorer.score_items()`. The scorer checks it between articles (rule scoring) and every 10 tokens (streaming Ollama). POST `/api/scan/cancel` sets the event. Partial results are saved to DB even on cancel.

**scan_status dict:**
```python
{
    "is_scanning": bool,
    "stage": "idle|starting|market|news|scoring|discovery|briefing|output|complete|error|cancelled",
    "progress": str,          # Human-readable progress text
    "scored": int,            # Items scored so far
    "total": int,             # Total items to score
    "high": int,              # Items scoring >= 7.0
    "medium": int,            # Items scoring 5.0-6.9
    "cancelled": bool,
    "last_completed": str,    # ISO timestamp
    "data_version": str,      # File mtime hash for cache busting
}
```

**SSE Push System:**
- `sse_register(wfile)` / `sse_unregister(wfile)`: Manages connected clients in a thread-safe list.
- `sse_broadcast(event_type, data)`: Pushes to all clients. Dead connections are auto-cleaned.
- `/api/events` endpoint: Sends initial status immediately, then 15-second heartbeats to keep the connection alive.
- Event types: `scan` (progress updates), `complete` (scan done, includes data_version), `scan_cancelled`, `scan_error`, `status`.

**HTTP Server:**
- `ThreadedHTTPServer(ThreadingMixIn, HTTPServer)` with `daemon_threads = True`.
- `CORSHandler(SimpleHTTPRequestHandler)` serves frontend from `../frontend/`, handles all API routes via `do_GET` and `do_POST`.
- No-cache headers on HTML/JS/CSS. CORS `Access-Control-Allow-Origin: *`.
- ETag support on `/api/data` (news_data.json) with gzip compression.
- Static file serving falls through to `SimpleHTTPRequestHandler.do_GET()`.

**Security Layer (inline in serve_frontend):**
- Profile-based authentication with SHA-256 hashed PINs.
- Session tokens: `secrets.token_hex(32)`, 7-day TTL, persisted to `.sessions.json`.
- Device-scoped profiles: `X-Device-Id` header binds profiles to devices.
- Session purge every 10 minutes.
- `AUTH_EXEMPT` set: auth-check, login, register, logout, wizard endpoints, refresh, status, scan control.
- Rate limiting per endpoint: sliding window, configurable per-path (e.g., 2/min for refresh, 20/min for agent-chat).
- API key masking: keys displayed with `"****" + last 4 chars`.

**Profile management:**
- `_list_profiles(device_id)`: Lists YAML profiles, optionally filtered by device.
- `_verify_profile_pin(name, pin)`: SHA-256 comparison with plaintext backward compat.
- `_load_profile_config(name)`: Loads profile YAML into live config. Clears profile-specific fields first (dynamic_categories, market tickers, etc.), then applies profile. Preserves API keys. Security block stripped from live config.
- Registration creates empty profile YAML with hashed PIN + device binding, then auto-logs in.

**Scorer factory:**
```python
def _create_scorer(config, db=None):
    if should_use_adaptive_scorer(config):
        return AdaptiveScorer(config, db=db)
    return AIScorer(config, db=db)
```

### 3.2 Database — database.py (685 lines)

SQLite with WAL mode, thread-safe commits, singleton pattern.

**Connection pragmas:**
- `journal_mode = WAL` (concurrent reads)
- `busy_timeout = 5000` (5s wait instead of immediate failure)
- `synchronous = NORMAL` (safety/speed balance)
- `cache_size = -8000` (8MB, up from 2MB default)
- `temp_store = MEMORY` (temp tables in RAM)

**7 tables:**

| Table | Primary Key | Purpose |
|-------|------------|---------|
| `news_items` | `id TEXT` (MD5 hash of URL) | Scored articles. `url` is UNIQUE. Columns: title, url, summary, source, root, category, score, score_reason, timestamp, fetched_at, shown_to_user, dismissed. |
| `market_snapshots` | `id INTEGER AUTOINCREMENT` | Historical price data per symbol/interval. Columns: symbol, name, interval, price, change_percent, high, low, history_json, snapshot_at. |
| `entities` | `id INTEGER AUTOINCREMENT` | Tracked entities for discovery. Columns: name (UNIQUE), category, is_discovered, discovered_at, is_active, mention_count. |
| `entity_mentions` | `id INTEGER AUTOINCREMENT` | Frequency tracking for discovery algorithm. Columns: entity_name, mention_count, recorded_at. |
| `scan_log` | `id INTEGER AUTOINCREMENT` | Scan completion records. Columns: started_at, elapsed_secs, items_fetched, items_scored, critical, high, medium, noise, rule_scored, llm_scored, error. |
| `briefings` | `id INTEGER AUTOINCREMENT` | Generated intelligence briefings. Columns: content_json, generated_at. |
| `user_feedback` | `id INTEGER AUTOINCREMENT` | User interactions. Columns: news_id, title, url, root, category, ai_score, user_score, note, action (click/dismiss/rate/save/thumbs_up/thumbs_down), created_at, profile_role, profile_location, profile_context. |

**Indexes:** idx_news_fetched, idx_news_score, idx_market_symbol, idx_mentions_date, idx_feedback_news, idx_feedback_date.

**Key methods:**
- `save_news_item()`: INSERT OR IGNORE (url uniqueness). Thread-safe commit.
- `get_recent_news(hours, min_score)`: Filters by fetched_at, min score, not dismissed. Ordered by score DESC.
- `get_feedback_for_scoring(days, limit)`: Returns `{positive, negative, corrections}` dicts for scorer prompt injection.
- `get_top_signals()`, `get_category_stats()`, `search_news_history()`, `get_daily_signal_counts()`: Agent query methods.
- `cleanup_old_data(days)`: Deletes news, market, mentions, and briefings older than N days.

**Singleton:** `get_database(db_path)` returns a module-level `_db_instance`.

### 3.3 Fetchers (7 files, ~3,050 lines)

**news.py (917 lines) -- Multi-Source News Aggregation:**
- `NewsFetcher` class: Orchestrates DDG, Serper, RSS, and Kuwait scrapers.
- `NewsItem` dataclass: title, url, summary, content, timestamp, source, root, category, score, score_reason. ID is MD5 of URL (first 12 chars).
- `ArticleScraper`: Full-text extraction from article URLs using `requests` + `BeautifulSoup`. Rotates User-Agent strings. Content selectors: article, main, .post-content, .article-body. Rate limiting per domain.
- `fetch_all(cache_ttl_seconds)`: Dispatches to DDG news endpoint, Serper search, RSS feeds, and Kuwait scrapers using `ThreadPoolExecutor`. Deduplicates by URL. Content caching to avoid re-scraping.
- DuckDuckGo: Uses `ddgs` library (DDGS class). News endpoint (`d.news()`) for each category's keywords. Region-aware (DDG region code derived from profile location).
- Serper: Parallel category searches via `SerperSearchClient`. Credit tracking via `SerperQueryTracker`.
- RSS: `feedparser` for configured feeds (extra_feeds_finance, extra_feeds_politics, custom_feeds, rss_feeds).
- Pre-filtering: Basic noise check on fetched items before scoring, stores `pre_filter_score` and `pre_filter_reason` on the item dict.

**market.py (316 lines) -- Yahoo Finance:**
- `MarketFetcher`: Fetches all configured tickers at 5 intervals using `yfinance`.
- Intervals: 1m (3d period, 5000 pts), 5m (15d, 5000 pts), 1d_1mo (1y, 365 pts), 1d_1y (5y, 1260 pts), 1wk (10y, 520 pts).
- Alert threshold: Configurable percent change trigger (default 5%).
- Cache with TTL (default 60s).
- Returns `(market_data_dict, alerts_list)`.

**discovery.py (314 lines) -- Entity Frequency Tracking:**
- `EntityDiscovery`: Extracts proper nouns from news, compares current frequency vs. 7-day baseline.
- Rising detection: `current_mentions > baseline * rising_multiplier` (default 3x).
- Auto-track: Optionally adds rising entities to tracked list.
- Category classification by regex patterns (kuwait_companies, tech_terms, financial_products).

**extra_feeds.py (274 lines) -- RSS Feed Catalog:**
- Master catalog of finance and politics RSS feeds with metadata (id, url, name, region, category, default on/off).
- Finance: CNBC, MarketWatch, Yahoo Finance, Bloomberg, Investing.com, FT, Seeking Alpha, CoinDesk, GCC/ME finance feeds.
- Politics: BBC, Al Jazeera, NPR, NYT, CNN, Fox, Guardian, DW, France24, GCC-specific news.
- `fetch_extra_feeds(type, config)`: Fetches enabled feeds in parallel, returns sorted items.
- `get_catalog(type)`: Returns feed metadata for Settings UI.

**serper_search.py (332 lines) -- Serper.dev Google Search:**
- `SerperSearchClient`: Wraps Serper API (search, search_news). Tracks query counts.
- `SerperQueryTracker`: Persistent JSON file tracking total queries, daily counts, last 50 queries. Thread-safe with lock. Warning at 80% of 2500 free tier.
- Credit management: Manual override via `/api/serper-credits`, auto-reset on key change.

**google_search.py (300 lines) -- Google Custom Search:**
- `GoogleSearchClient`: Alternative search provider using Google Custom Search API.
- `get_search_status(config)`: Returns provider, limit_reached, remaining credits.

**kuwait_scrapers.py (589 lines) -- Kuwait-Specific Sources:**
- Scrapers for Kuwait-specific news portals and career sites.
- Returns items with root="kuwait" for local relevance scoring.

### 3.4 Processors (4 files, ~4,414 lines)

**scorer.py (2,138 lines) -- AIScorer (Hardcoded CPEG/Kuwait):**

The original scorer, highly tuned for a Computer Engineering student/graduate in Kuwait. Contains extensive hand-crafted pattern lists:

- `NOISE_EXACT` (16 terms): horoscope, lottery, gossip, etc.
- `NOISE_PATTERNS` (8 regex): job aggregator pages, cookie/privacy pages, 404s.
- `SOCIAL_NOISE_PATTERNS` (6 regex): Instagram spam, social media noise.
- `COMPANY_PROFILE_PATTERNS` (8 regex): LinkedIn about pages, Wikipedia company descriptions.
- `STOCK_PAGE_PATTERNS` (12 regex): Yahoo Finance data pages, stock screeners.
- `FLIGHT_PAGE_PATTERNS` (4 regex): Flight tracking (catches bank abbreviation collisions like PR-NBK).
- `TECHNICAL_DATA_PATTERNS` (5 regex): BGP routing tables, AS numbers, corporate actions.
- `GARBAGE_DOMAINS`: Known junk domains.
- `STALE_PATTERNS`: Outdated content markers.
- `GENERIC_PAGE_PATTERNS`: Empty/login/form pages.

Category-specific system prompts for career, finance, tech, regional. Each has its own scoring scale (a 9.0 in career means "target company hiring fresh grads"; in tech means "groundbreaking advancement").

`ScoringMemory`: Maintains few-shot examples from past high-scoring items. Injects user feedback (positive/negative/corrections) into scoring prompts. Persisted to `data/memory.json`.

`_is_non_latin_title()`, `_is_non_target_language()`: Language detection using Unicode script analysis. `location_to_lang()` maps profile location to expected language scripts and DDG region codes.

**scorer_adaptive.py (1,421 lines) -- AdaptiveScorer (Any Profile):**

The universal scorer that builds relevance rules dynamically from any user profile.

`KeywordIndex`: Builds weighted keyword dictionaries from dynamic_categories. Splits multi-word items into individual tokens (filtering stop words). Methods: `match_category(text, cat_id)`, `match_any(text)`, `match_location(text)`.

**3-Phase Scoring Pipeline:**

1. **Phase 1 -- Rule Scoring (all items):** Universal noise check (`_universal_noise_check`) + profile-adaptive scoring functions routed by `scorer_type`:
   - `score_career_adaptive()`: Student vs. experienced paths. Students: senior titles hard-killed, experience requirements hard-killed, entry-level+hiring+location = 9.5. Experienced: entry-level killed, hiring+keywords = 9.0.
   - `score_finance_adaptive()`: Tracked entity + offer/deal = 9.5. Keyword + offer + location = 9.0.
   - `score_tech_adaptive()`: Keywords + breakthrough = 9.5. Keywords + news action = 8.0.
   - `score_generic_adaptive()`: Pure keyword relevance fallback.
   - Confident items (noise, strong matches) get their score immediately. Ambiguous items go to Phase 2.

2. **Phase 2 -- Batch LLM Scoring (ambiguous items, batches of 4):** `_build_batch_prompt()` creates system+user prompt with profile context, tracked fields, feedback examples, and 4 articles. `_parse_batch_response()` extracts numbered `[N] SCORE: X.X | REASON: ...` lines. Failed parses fall back to individual `_llm_score()` calls. Streaming Ollama with `cancel_check` every 10 tokens.

3. **Phase 3 -- Prompt Rescore Cascade (uncertain zone 4.5-6.5, max 12 items):** `_prompt_rescore()` uses a structured 5-dimension rubric (profile match, location match, actionability, level match, noise check). Only accepts re-scores with >= 1.0 point change.

**V1 Isotonic Calibration:** Corrects systematic +1.33 score inflation. Loaded from `data/v1_calibration_table.json`. Applied only to LLM-scored items, not rule-based or pre-filtered.

**Forbidden 5.0 Rule:** Scores in 4.8-5.2 are nudged to 4.8 or 5.3. This prevents the model from fence-sitting at the exact noise/medium boundary.

**Student Detection:** `_detect_student(profile)` checks role for student signals (student, bachelor, undergraduate, etc.) vs. experienced signals (manager, director, senior, etc.). Role takes priority over AI-generated context.

**should_use_adaptive_scorer(config):** Returns True unless the profile is Kuwait + engineering-related role (which uses the highly-tuned AIScorer).

**briefing.py (455 lines) -- BriefingGenerator:**
- Builds dynamic system prompts from config: profile (role, location, context), target companies from dynamic_categories, category summaries.
- Uses the inference_model (qwen3:30b-a3b) via Ollama `/api/generate`.
- Generates structured intelligence briefings with critical signals, market summary, and recommendations.
- Think parameter omitted (do NOT use `"think": False` — Qwen3 ignores it and leaks reasoning into content). `<think>` blocks stripped as safety net.

**profile_generator.py (400 lines) -- AI Category Generation:**
- Post-processing pipeline: LLM Generate -> Sanitize -> Enrichment -> Dedup -> Merge Tiny -> Career Opt-out.
- `_sanitize_item()`: Strips quotes, markdown, numbering, parentheticals, trailing prepositions. Cap at 6 words.
- `_split_comma_items()`: Paren-aware comma splitting.
- `ALIASES`: Canonical names for common entities (Schlumberger->SLB, etc.).
- `CAREER_OPTOUT_SIGNALS`: Regex patterns for users who explicitly don't want career content.
- `MAX_CATEGORIES = 7`, `MIN_ITEMS_FOR_MERGE = 2`.

### 3.5 Routes (5 files, ~1,683 lines)

**agent.py (817 lines) -- Chat Agent:**
- `AGENT_TOOLS`: Three Ollama tool definitions: `web_search` (Serper), `manage_watchlist` (add/remove/list tickers), `manage_categories` (add/remove keywords, enable/disable categories).
- `handle_agent_chat()`: SSE streaming chat. Builds system prompt with profile, feed context (top 15 signals). Sends to Ollama `/api/chat` with tool definitions. Handles tool call responses (both structured JSON and Qwen3 text-format tool calls via `_parse_text_tool_calls()`). Multi-round: executes tool, sends result back to model for final response.
- `_tool_web_search()`: Uses Serper client, returns formatted results with source/date.
- `_tool_manage_watchlist()`: Add/remove/list tickers. Persists to config.yaml via `_save_tickers()`.
- `_tool_manage_categories()`: Fuzzy-matched category lookup. Add/remove keywords, enable/disable categories. Persists to config.yaml.
- `handle_suggest_context()`: Generates AI-suggested profile context from role+location.
- `handle_ask()`: Quick Q&A about a specific news item.
- `handle_agent_status()`: Returns Ollama model availability.

**wizard.py (342 lines) -- Onboarding Wizard:**
- `handle_wizard_preselect()`: AI picks relevant categories for Step 1 based on role/location. Fallback defaults on failure.
- `handle_wizard_tab_suggest()`: AI suggests 5-8 keywords per category. Career-stage-aware (student/senior/fresh graduate) with anti-example guidance.
- `handle_wizard_rv_items()`: AI generates role-appropriate entities for Step 3 review. Rich system prompt with anti-examples (pastry chef should NOT see AWS, Google, CCNA).
- All use `_call_ollama()` helper with `<think>` stripping. Think parameter is omitted (Ollama separates thinking into a `thinking` field automatically).

**generate.py (254 lines) -- Profile Generation:**
- `handle_generate_profile()`: POST `/api/generate-profile`. Extensive system prompt (136 lines) with category design rules, separation rules (employers vs banks, assets in tickers not categories), role diversity guidance, location-specific knowledge (Kuwait K-sector, banks, tech employers).
- 2-attempt retry loop with lowered temperature on retry.
- Post-processes via `profile_generator.run_pipeline()`.

**config.py (215 lines) -- Config Save:**
- `handle_config_save()`: Merges incoming config into live config. Handles profile, market tickers, news, search (masked key detection), dynamic categories (with legacy sync), extra feed toggles.
- `_sync_dynamic_to_legacy()`: Maps dynamic_categories to legacy news.career/finance/tech_trends/regional keyword buckets.
- `_update_search_config()`: Handles masked API keys, Serper credit resets on key change.
- `_sync_to_profile_yaml()`: Persists config changes to user's profile YAML.
- `TICKER_NAMES`: 50+ ticker-to-name mappings for display.

**helpers.py (55 lines):**
- `json_response()`: JSON with optional gzip compression (>1024 bytes + client supports gzip).
- `error_response()`: JSON error wrapper.
- `read_json_body()`: Parses request body.
- `sse_event()`, `start_sse()`: SSE streaming utilities.

### 3.6 Profiles System

Per-user profiles stored as YAML in `profiles/` directory.

**Profile YAML structure:**
```yaml
profile:
  role: "Computer Engineer"
  location: "Kuwait"
  context: "Track: ... Deprioritize: ..."
  interests: []
security:
  pin_hash: "sha256_hex_string"
  devices: ["device_id_hex"]
market:
  tickers: [{symbol: "ZAIN.KW", name: "ZAIN.KW", category: "custom"}, ...]
  intervals: {}
  alert_threshold_percent: 5.0
news:
  timelimit: "w"  # d, w, or m
  career: {root: "kuwait", keywords: [...], queries: []}
  finance: {root: "kuwait", keywords: [...], queries: []}
  regional: {root: "regional", keywords: [...], queries: []}
  tech_trends: {root: "global", keywords: [...], queries: []}
  rss_feeds: []
dynamic_categories:
  - {id: "ktech", label: "Kuwait Tech Employers", icon: "briefcase",
     items: ["Zain", "Ooredoo", ...], enabled: true, scorer_type: "career", root: "kuwait"}
extra_feeds_finance: {cnbc_top: false, bloomberg: true, ...}
extra_feeds_politics: {bbc_world: true, aljazeera: true, ...}
custom_feeds: []
custom_tab_name: "Custom"
```

**Authentication flow:**
1. Frontend generates a random `device_id` (stored in localStorage).
2. Registration: name + PIN (>= 4 chars) + device_id. PIN hashed with SHA-256. Profile YAML created.
3. Login: Verifies PIN hash, loads profile config into memory, creates session token.
4. Session token sent as `X-Auth-Token` header on all API calls. Auth monkey-patched into fetch() by `auth.js`.

---

## 4. Frontend Architecture (18,000 lines)

Vanilla JavaScript SPA. No React, no Vue, no build step. Tailwind CSS for styling (CDN). All state management is manual.

**app.js (1,285 lines):** Core orchestration. SSE connection to `/api/events` via EventSource. `toggleScan()` starts/cancels scans. Data loading from `/api/data` with ETag support. Auto-refresh compares `data_version`.

**auth.js (429 lines):** Login/register forms. Device ID generation. `monkeyPatchFetch()` wraps global `fetch()` to inject `X-Auth-Token` + `X-Device-Id` headers. Intercepts 401 for login redirect.

**feed.js (762 lines):** Scored news rendering with color-coded badges (critical=red, high=orange, medium=blue, noise=gray). Category tab filtering, sort by score/date/source. Feedback actions: click, dismiss, save, thumbs up/down, rate.

**market.js (1,004 lines) + markets-panel.js (1,843 lines):** TradingView-style charts with canvas rendering (zoom, pan, crosshair, volume). 5 time intervals (1D/5D/1M/1Y/5Y). Candlestick + line modes. Top Movers panel (50+ asset universe).

**agent.js (908 lines):** SSE streaming chat. Tool-use display (web_search, manage_watchlist, manage_categories). Markdown rendering. Context-aware (knows current feed, profile, watchlist).

**settings.js (2,529 lines):** Profile editor, dynamic category management (add/remove/reorder/keywords/toggle/scorer_type/root), ticker management, RSS feed catalog toggles, search API key management (masked), AI Generate button for auto-category creation.

**wizard.js (2,543 lines) + wizard_v4.html (1,131 lines):** 4-step onboarding: (1) Role+location with AI category selection, (2) per-category keyword customization with AI suggestions, (3) review generated entities, (4) market tickers + RSS feeds. Each step calls a wizard API endpoint.

**Other:** nav.js (245) sidebar routing, scan-history.js (209) scan log viewer, theme-editor.js (356) CSS variable editor, ui.js (264) toasts/modals, sw.js (37) service worker.

**SSE Event Protocol:**
- `scan` -> update progress bar; `complete` -> reload feed; `scan_cancelled` -> show partial results; `scan_error` -> show error; `status` -> initial sync on connect.

---

## 5. Scoring System Deep Dive

### Scorer Selection

`should_use_adaptive_scorer(config)` checks if profile diverges from the default:
- Kuwait + engineering-related role (computer, cpeg, software, electrical, etc.) -> **AIScorer** (hardcoded, highly tuned).
- Everything else -> **AdaptiveScorer** (dynamic, keyword-driven).
- Empty profile (no role, no location) -> AIScorer (sensible defaults).

Both scorers share the identical interface: `score_items()`, `score_item()`, `get_score_category()`.

### Phase 1: Rule Scoring

Every item passes through rule-based checks first:

1. **Universal noise detection** (both scorers): Garbage domains, NOISE_EXACT terms, NOISE_PATTERNS regex, STALE_PATTERNS, GENERIC_PAGE_PATTERNS, SOCIAL_NOISE_PATTERNS, COMPANY_PROFILE_PATTERNS, STOCK_PAGE_PATTERNS, FLIGHT_PAGE_PATTERNS, TECHNICAL_DATA_PATTERNS. Language filtering based on profile location.

2. **Profile-adaptive scoring** (AdaptiveScorer): Routes by `scorer_type` from dynamic_categories. Each function returns `(score, reason, is_confident)`. Confident results skip LLM. Ambiguous results (is_confident=False) go to Phase 2.

3. **Title deduplication**: Same title (case-insensitive) = score 2.0 automatic.

4. **Pre-filtered items**: Items tagged with `pre_filter_score` by NewsFetcher skip scoring entirely.

### Phase 2: LLM Scoring (Batch)

Ambiguous items are batched in groups of 4 for the Ollama model:

- **System prompt**: Profile role, location, context, tracked fields (companies, institutions, interests, industries), student/experienced level note, feedback examples, language requirement, scoring rubric.
- **User prompt**: Numbered articles with category, keywords, title, content (300 chars).
- **Expected output**: `[1] SCORE: X.X | REASON: ...` per article.
- **Parse**: Regex extracts numbered score/reason pairs. Failed parses fall back to individual Ollama calls.
- **Streaming**: `requests.post(stream=True)`, tokens concatenated. `cancel_check` called every 10 chunks.
- **Options**: `temperature: 0.6`, `top_p: 0.95`, `top_k: 20`, `num_predict: 512`.

### Phase 3: Prompt Rescore Cascade

Items scoring 4.5-6.5 (uncertain zone) after Phase 2 get a second opinion:

- Max 12 items rescored per scan.
- Richer prompt with 5-dimension structured rubric: Profile Match, Location Match, Actionability, Level Match, Noise Check.
- Includes the first automated score for reference.
- Only accepted if the new score differs by >= 1.0 from the first score.
- System prompt uses inference_model (qwen3:30b-a3b) rather than the fine-tuned scorer.

### V1 Calibration Table

A pre-computed isotonic calibration table (`data/v1_calibration_table.json`) corrects systematic score inflation (+1.33 average) from the V1 fine-tuned model. Maps raw score (0.1 granularity) to calibrated score. Applied only to LLM-scored items after Phase 2/3.

### Forbidden 5.0 Rule

Score 5.0 is the exact noise/medium boundary. The model tends to cluster scores here when uncertain. To force decisiveness:
- Scores 4.8-4.9 stay as-is (below medium).
- Score 5.0-5.2 nudges to 5.3 (above medium) or 4.8 (below).
- Applied in rule scoring, LLM scoring, batch parsing, and prompt rescore.

### ScoringMemory

- Maintains a rolling window of high-scoring examples (min_score >= 8.5, max 10 examples).
- `format_feedback_for_prompt(max_each)`: Injects recent user feedback into LLM prompts. Formats positive, negative, and correction examples.
- Feedback comes from `database.get_feedback_for_scoring()`: positive (saved/thumbs_up/high-rated clicks), negative (dismissed/thumbs_down/low-rated), corrections (ai_score vs user_score delta >= 2.0).
- Persisted to `data/memory.json`.

---

## 6. Training Pipeline

### V1 Failures and Lessons (v15-v18)

| Version | Failure Mode | Root Cause |
|---------|-------------|------------|
| v15 | Loss dilution, model outputs garbage | LoRA trained on full input+output tokens. Input tokens dominated loss, assistant response signal drowned. |
| v16 | Bimodal collapse (scores cluster at 2.0 and 8.0) | Imbalanced training data, no WeightedRandomSampler. |
| v17 | Greedy decoding loops ("SCORE: 7.5 SCORE: 7.5 SCORE: 7.5...") | Think mode + temperature=0 causes infinite repetition. Must use temp > 0. |
| v18 | DoRA inference overhead, slow scoring | GGUF export stripped DoRA weight decomposition. Fixed in v19 with proper merge before export. |

**Key lesson:** Training data format must be **character-for-character identical** to inference format. `export_training.py` must produce system/user/assistant messages that exactly match what `scorer_adaptive.py` sends at inference time. Even whitespace differences cause the trained model to fail at parsing.

### V2 Training Pipeline

**Philosophy:** Train a single model that is profile-aware. Instead of one profile's corrections, use 30 diverse profiles across 20 countries to teach the model that the SAME article should score differently for different users.

**Stage 1 -- Profile Creation (profiles_v2.py):**
- 30 profiles: 10 from V1 (autopilot.py templates) + 20 new.
- Diversity across: geography (Kuwait, Saudi, UAE, Oman, Bahrain, Qatar, Singapore, UK, Japan, Brazil, Nigeria, Chile, Italy, Germany, Australia, Canada, India), career level (student, fresh grad, mid-career, senior), and domain (engineering, finance, medicine, education, marketing, architecture, agriculture, marine biology, nuclear engineering).
- Each profile has: id, role, location, context, interests, tracked_companies, tracked_institutions, tracked_industries, diversity_tag.

**Stage 2 -- Article Collection (stage2_collect.py):**
- 685 articles collected from DDG, Serper, and RSS feeds.
- Each article tagged with source, category, content.
- Stored as JSONL for batch processing.

**Stage 3 -- Claude Opus Scoring (stage3_score.py):**
- Uses Claude Opus 4.5 via Anthropic Batch API (50% cheaper than real-time).
- PRISM reasoning framework: Profile relevance, Recency, Impact, Specificity, Maturity.
- Each article scored by EVERY profile that has overlapping relevance.
- 20,550 scored examples total. ~$52 API cost.
- `stage3_resume.py`: Resumes interrupted batch jobs.

**Stage 4 -- Data Preparation (stage4_prepare.py):**
- Train/eval split: 18,502 training / 2,048 evaluation.
- Contrastive pairs: 25,243 pairs where the same article gets different scores from different profiles. These teach the model that relevance is profile-dependent.
- Per-sample loss weighting by score band: noise=0.5, tangential=1.0, moderate=1.5, high=2.0, critical=3.0.
- WeightedRandomSampler weights to rebalance score distribution (oversamples rare critical/noise bands).

**Training (train_v2.py):**
- Base: V1 merged model (Qwen3-8B with V1 LoRA merged in).
- DoRA (Weight-Decomposed Low-Rank Adaptation) rank 16, alpha 32, dropout 0.05.
- 1 epoch, learning rate 1e-5 (lower than V1's 2e-5 since fine-tuning, not from scratch), cosine schedule with 5% warmup.
- Gradient accumulation 8. Max sequence length 1024.
- Effective batch size: 1 * 8 = 8 (limited by 24GB VRAM on 7900 XTX).
- 1,157 steps. ~10.7 hours on AMD Radeon RX 7900 XTX.
- ROCm setup: `HSA_OVERRIDE_GFX_VERSION=11.0.0`, expandable memory segments. AOTRITON experimental mode DISABLED (causes NaN gradients).
- PEFT meta device fix: After `get_peft_model()`, delete `hf_device_map` attribute and force `.to("cuda:0")` to prevent gradient crashes.
- Checkpoints saved every 200 steps.
- Export: Merge LoRA -> GGUF Q8_0 -> Ollama create.

**V2 Results:**

| Metric | V1 | V2 | Notes |
|--------|----|----|-------|
| Direction accuracy | 87% | **98.1%** | Correct high/low classification |
| MAE (Mean Absolute Error) | 0.78 | **0.393** | Average score distance from Opus |
| Spearman rho | 0.52 | **0.750** | Rank correlation with Opus |
| Profile-awareness spread | 1.04 | **7.90** | Score range for same article across profiles (7.6x improvement) |
| Parse failures | 3.2% | **0%** | Zero empty responses, zero malformed outputs |
| Empty think blocks | 8% | **0%** | No more `<think></think>` with empty content |
| All 30 profiles positive correlation | No | **Yes** | Range 0.416-0.879 |

**PSR (Profile Separation Ratio) note:** V2's 24.4% PSR appears lower than V1's 39.7%. However, Opus's own PSR is 26.4%. This means V2 is correctly calibrated -- it matches the teacher's separation behavior rather than artificially inflating differences.

### Legacy Pipeline (still operational)

For incremental improvement on the production system:

1. **distill.py**: Claude Opus 4.5 re-scores recent items from the DB. Uses PRISM rubric. 10 items per API call. Saves disagreements (delta >= 2.0) as corrections in `user_feedback` table with `action='distill'`.
2. **export_training.py**: Converts corrections + user feedback into ChatML JSONL. Mirrors `scorer_adaptive._build_batch_prompt()` format exactly. Legacy profile map reconstructs tracked fields for old corrections.
3. **train_lora.py**: Auto-selects model tier by VRAM (8B/4B/1.7B). DoRA rank 16, alpha 32. `WeightedCompletionDataCollator` for per-sample loss weighting. Supports both Unsloth (CUDA) and PEFT (ROCm). Pipeline: train -> merge -> GGUF Q8_0 -> Ollama register -> config.yaml update.
4. **autopilot.py**: Fully autonomous loop. 17+ profile templates across Kuwait, GCC, and high-contrast domains. Each cycle: pick unused profile (diversity guard), generate context via Ollama, generate categories, save config, trigger scan, run distillation, restore original profile. Budget tracking. Persistent state. Every N cycles triggers training.

**Automation scripts:**
- `learn_cycle.sh`: `distill.py -> export_training.py -> train_lora.py`
- `stratos_overnight.sh`: Continuous `scan -> distill -> train` with budget limit.

---

## 7. API Reference

### GET Endpoints

| Path | Auth | Description |
|------|------|-------------|
| `/api/auth-check` | No | Returns auth state, profiles list (device-scoped), needs_registration flag |
| `/api/data` | Yes | Main data feed (news_data.json). ETag support, gzip compression |
| `/api/refresh` | No | Triggers full scan in background thread. Returns immediately |
| `/api/refresh-market` | Yes | Triggers market-only refresh. Rate limited: 3/min |
| `/api/refresh-news` | Yes | Triggers news+scoring+briefing refresh. Rate limited: 2/min |
| `/api/status` | No | Scan status + last 5 scan log entries |
| `/api/scan/status` | No | Detailed scan progress (running, scored/total, high, medium, cancelled, stage) |
| `/api/events` | Yes | SSE event stream. Heartbeat every 15s |
| `/api/search-status` | Yes | Search provider quota status |
| `/api/scan-log` | Yes | Last 50 scan log entries |
| `/api/export` | Yes | Dashboard export. `?format=csv` for flat CSV, default JSON diagnostic |
| `/api/feedback-stats` | Yes | User feedback summary statistics |
| `/api/finance-news` | Yes | RSS-only finance news (no scoring) |
| `/api/politics-news` | Yes | RSS-only politics news (no scoring) |
| `/api/custom-news` | Yes | User-defined custom RSS feeds |
| `/api/feed-catalog/finance` | Yes | Finance RSS feed catalog with user toggles |
| `/api/feed-catalog/politics` | Yes | Politics RSS feed catalog with user toggles |
| `/api/config` | Yes | Current editable config (keys masked) |
| `/api/profiles` | Yes | User's saved profile presets |
| `/api/top-movers` | Yes | Top 10 movers from 50+ asset universe. Rate limited: 3/min |
| `/api/agent-status` | Yes | Ollama model availability check |

### POST Endpoints

| Path | Auth | Description |
|------|------|-------------|
| `/api/scan/cancel` | No | Sets scan cancellation flag |
| `/api/auth` | No | Login: `{profile, pin}` -> `{token, profile}` |
| `/api/register` | No | Register: `{name, pin, device_id}` -> `{token, profile}` |
| `/api/logout` | No | Invalidates session token |
| `/api/config` | Yes | Save configuration changes |
| `/api/serper-credits` | Yes | Manually set Serper credit count |
| `/api/feedback` | Yes | User feedback: `{news_id, action: click|dismiss|rate|save|thumbs_up|thumbs_down, ...}` |
| `/api/feedback-stats` | Yes | Feedback stats (also available as GET) |
| `/api/ask` | Yes | Quick AI Q&A about a news item |
| `/api/suggest-context` | No | AI-generated profile context from role+location |
| `/api/generate-profile` | No | AI-generated categories+tickers from role/location. Rate limited: 5/min |
| `/api/wizard-preselect` | No | AI picks relevant categories for wizard Step 1 |
| `/api/wizard-tab-suggest` | No | AI suggests keywords for wizard Step 2 tab |
| `/api/wizard-rv-items` | No | AI generates role-aware entities for wizard Step 3 |
| `/api/profiles` | Yes | Preset management: `{action: save|load|delete, name}` |
| `/api/agent-chat` | Yes | Streaming chat with tool-use. Rate limited: 20/min |

### Rate Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/api/refresh` | 2 calls | 60 seconds |
| `/api/refresh-news` | 2 calls | 60 seconds |
| `/api/refresh-market` | 3 calls | 60 seconds |
| `/api/generate-profile` | 5 calls | 60 seconds |
| `/api/suggest-context` | 10 calls | 60 seconds |
| `/api/agent-chat` | 20 calls | 60 seconds |
| `/api/top-movers` | 3 calls | 60 seconds |

---

## 8. Configuration Reference

### config.yaml

```yaml
# Cache TTLs
cache:
  discovery_ttl_seconds: 3600
  market_ttl_seconds: 60
  news_ttl_seconds: 900

# Entity discovery
discovery:
  auto_track: true
  baseline_window_hours: 168    # 7-day baseline
  enabled: true
  min_mentions: 3
  rising_multiplier: 3.0

# Market watchlist
market:
  tickers:
    - {symbol: "BTC-USD", name: "Bitcoin", category: "custom"}
  intervals: {}                 # Empty = use DEFAULT_INTERVALS
  alert_threshold_percent: 5.0

# News sources
news:
  timelimit: "w"                # d=day, w=week, m=month
  career: {root: "kuwait", keywords: [...], queries: []}
  finance: {root: "kuwait", keywords: [...], queries: []}
  regional: {root: "regional", keywords: [...], queries: []}
  tech_trends: {root: "global", keywords: [...], queries: []}
  rss_feeds: []

# Scoring models
scoring:
  model: "stratos-scorer-v2"              # Fine-tuned scoring model
  inference_model: "qwen3:30b-a3b"            # Agent chat, suggestions, profile generation, briefings
  ollama_host: "http://localhost:11434"
  critical_min: 9.0
  high_min: 7.0
  medium_min: 5.0
  filter_below: 5.0
  forbidden_score: 5.0

# Search API
search:
  provider: "serper"            # serper, google, or duckduckgo
  serper_api_key: "..."
  serper_credits: 2500
  google_api_key: "..."
  google_cx: "..."

# System
system:
  database_file: "strat_os.db"
  frontend_port: 8080
  log_level: "INFO"
  max_news_items: 100
  output_file: "output/news_data.json"

# Background scheduler
schedule:
  background_enabled: false
  background_interval_minutes: 30

# User profile
profile:
  role: "Computer Engineer"
  location: "Kuwait"
  context: "Track: ... Deprioritize: ..."
  interests: []

# Dynamic categories (populated by generate-profile or wizard)
dynamic_categories:
  - id: "ktech"
    label: "Kuwait Tech Employers"
    icon: "briefcase"
    items: ["Zain", "Ooredoo", "STC Kuwait", ...]
    enabled: true
    scorer_type: "career"       # career, banks, tech, regional, auto
    root: "kuwait"              # kuwait, regional, global, ai

# RSS feed toggles
extra_feeds_finance: {bloomberg: true, yahoo_fin: true, ...}
extra_feeds_politics: {bbc_world: true, aljazeera: true, ...}
custom_feeds: [{url: "...", name: "...", on: true}]
custom_tab_name: "Custom"

# Auto-distillation (optional)
distillation:
  auto_every: 0                 # Scans between auto-distill (0 = disabled)
  hours: 168
  limit: 200
  threshold: 2.0
```

### Environment Variables

| Variable | Source | Purpose |
|----------|--------|---------|
| `ANTHROPIC_API_KEY` | `.env` or shell | Claude Opus API key for distillation |
| `HSA_OVERRIDE_GFX_VERSION` | Shell (training only) | ROCm GPU version override (11.0.0 for 7900 XTX) |
| `PYTORCH_HIP_ALLOC_CONF` | Shell (training only) | `expandable_segments:True` for ROCm memory |

### Ollama Model Registration

```bash
# Fine-tuned scorer (output of train_lora.py or train_v2.py)
ollama create stratos-scorer-v2 -f data/models/Modelfile

# Inference model (for agent, briefing, profile generation)
ollama pull qwen3:30b-a3b
```

Modelfile format:
```
FROM /path/to/model.gguf
TEMPLATE "{{ .System }}\n{{ .Prompt }}"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
```

---

## 9. Deployment Guide

### Prerequisites

- **Python 3.12+** with pip
- **Ollama** (must be running: `ollama serve`)
- Required Ollama models:
  - `stratos-scorer-v2` (fine-tuned scorer, or `qwen3:8b` as fallback)
  - `qwen3:30b-a3b` (inference model for agent/briefing/generation)
- **GPU** (optional): Required only for training. 24GB VRAM recommended (AMD 7900 XTX or NVIDIA 4090).

### Setup

```bash
# Clone and enter backend
cd backend/

# Install Python dependencies
pip install pyyaml requests beautifulsoup4 feedparser yfinance duckduckgo-search

# Optional: Serper API key for better search results
# Add to .env: ANTHROPIC_API_KEY=sk-ant-...
# Add to config.yaml: search.serper_api_key: ...

# Pull Ollama models
ollama pull qwen3:30b-a3b
# If you have a trained scorer model:
ollama create stratos-scorer-v2 -f data/models/Modelfile
```

### Running

```bash
# Start the full server (dashboard at http://localhost:8080)
python3 main.py --serve --background

# Or use the launcher script (checks Ollama, registers models, then starts)
bash stratos.sh

# Run a one-shot scan without the server
python3 main.py --scan

# Custom port
python3 main.py --serve --port 9090
```

### Training

```bash
# Manual learning cycle
bash learn_cycle.sh

# Step by step:
python3 distill.py --hours 168 --limit 200     # Opus re-scores recent items
python3 export_training.py --min-delta 1.5      # Export corrections as JSONL
python3 train_lora.py --epochs 3                # LoRA fine-tune -> GGUF -> Ollama register

# Fully autonomous overnight mode
bash stratos_overnight.sh 10.00                 # $10 budget

# Autopilot (cycles through diverse profiles)
python3 autopilot.py --cycles 10 --budget 5.00
python3 autopilot.py --dry-run                  # Preview which profiles will run

# V2 pipeline (full retrain)
HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 data/v2_pipeline/train_v2.py
```

### Remote Access

Cloudflare Tunnel for secure external access without port forwarding:
```bash
cloudflared tunnel --url http://localhost:8080
```

---

## 10. Known Issues and Technical Debt

### Critical Constraints

**AOTRITON NaN Gradients:**
Do NOT set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` during ROCm training. This causes NaN gradients during backward pass with DoRA + gradient checkpointing on ROCm 6.2. The flag is explicitly commented out in train_v2.py.

**Qwen3 Think Blocks:**
Do NOT set `"think": False` in Ollama API calls — Qwen3 ignores it and leaks reasoning as plain text into the `content` field, consuming the entire `num_predict` budget before producing useful output. Instead, omit the `think` parameter entirely. In default mode, Ollama separates reasoning into a `thinking` field while `content` holds the clean answer. Always strip `<think>...</think>` blocks from content as a safety net. Set `num_predict` large enough for both thinking tokens (~1000-2000) and output tokens. The exception is the fine-tuned scorer which uses think mode intentionally (trained with it enabled).

**Training/Inference Alignment:**
`export_training.py` must produce system/user/assistant messages that are character-for-character identical to what `scorer_adaptive.py` sends at inference time. The system prompt, user format, tracked fields block, level note -- all must match exactly. Misalignment causes the trained model to output unparseable responses.

**PEFT Meta Device Fix:**
After `get_peft_model()`, you must delete the `hf_device_map` attribute and force `.to("cuda:0")`. Without this, some parameters remain on a "meta" device and gradients crash during backward pass:
```python
if hasattr(model, 'hf_device_map'):
    del model.hf_device_map
model = model.to("cuda:0")
```

**Greedy Decoding Loops:**
Think mode + temperature=0 causes infinite repetition loops ("SCORE: 7.5 SCORE: 7.5 SCORE: 7.5..."). Always use temperature > 0. The scorer uses temp=0.6, top_p=0.95, top_k=20.

**Qwen3 Tokenizer eos_token:**
Some Qwen3 tokenizer versions have the wrong `eos_token`. Verify during training that the tokenizer's eos_token matches the model's actual end-of-sequence token.

### Technical Debt

**AIScorer (scorer.py) is hardcoded for CPEG/Kuwait:** 2,138 lines of hand-crafted patterns specific to one user profile. Should eventually be retired in favor of AdaptiveScorer once V2 model is proven across all profile types. Currently kept because it is the most battle-tested scorer for the primary user.

**PSR Metric Confusion:** V2's 24.4% PSR looks lower than V1's 39.7%, but this is misleading. Opus's own PSR is 26.4%, meaning V2 is correctly calibrated to the teacher. V1 was artificially inflating profile separation.

**No WebSocket:** SSE is unidirectional (server->client). Agent chat uses POST for messages and SSE for streaming responses. A WebSocket upgrade would simplify bidirectional communication.

**Single-threaded Ollama Scoring:** Items are scored sequentially (one Ollama call at a time). The batch-of-4 approach helps but is still fundamentally sequential. Ollama's own concurrency is the bottleneck.

**Profile Config in Memory:** When a user logs in, their profile config replaces the live config in memory. This means only one profile can be active at a time on a single server instance. Multiple concurrent users would need separate server instances.

**Legacy Keyword Sync:** `_sync_dynamic_to_legacy()` in config.py maps dynamic_categories to legacy news.career/finance/tech_trends/regional buckets. This exists only because the original AIScorer reads from legacy buckets. Once AIScorer is removed, this sync becomes unnecessary.

**No Database Migrations Framework:** Schema changes are handled with `ALTER TABLE ... ADD COLUMN` wrapped in try/except. Works but does not support column removal, renaming, or type changes.

---

## 11. Future Roadmap

### Near-Term

- **Mobile-optimized dashboard layout:** Current UI works on mobile but is not optimized. Needs responsive breakpoints for charts, feed cards, and settings panels.
- **WebSocket upgrade from SSE:** Enables bidirectional communication, reduces HTTP overhead for agent chat, enables multi-client awareness.
- **Critical signal push notifications:** Browser Notification API or Push API for items scoring >= 9.0. Requires service worker enhancement.

### Medium-Term

- **Multi-profile LoRA adapters:** One base model + profile-specific LoRA adapters loaded at inference time. Allows multiple profiles to run simultaneously without model switching overhead.
- **Agent model fine-tuning:** Train the inference model (qwen3:30b-a3b) on tool-use patterns specific to StratOS. Currently relies on base model's tool-use capabilities.
- **Additional data sources:** LinkedIn job postings (authenticated scraping), government tender portals (Kuwait, GCC), academic databases (arXiv, PubMed), patent filings.
- **Automated A/B testing for scorer versions:** Run two scorer versions simultaneously on the same feed, compare user engagement metrics (click-through, save rate, dismiss rate) to determine which version performs better.

### Long-Term

- **Air-gapped deployment documentation:** Full offline deployment guide for environments without internet access. Pre-bundled models, RSS-only data sources.
- **Federated learning across profiles:** Multiple StratOS instances share training signals without sharing raw data. Privacy-preserving model improvement.
- **Multi-language scoring:** Currently English-primary with language filtering. Extend to native-language scoring for Arabic, French, German, etc.
- **Dashboard embedding API:** Allow other applications to embed StratOS feed widgets via iframe or API.
