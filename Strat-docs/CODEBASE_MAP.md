# StratOS — Codebase Dependency Map
> Generated: 2026-03-16 | Total: ~111,344 lines across 192 files

---

## Section A: File Inventory

### Backend Core (7 files, 5,467 lines)
| File | Lines | Role |
|------|-------|------|
| main.py | 1,682 | StratOS orchestrator — scan pipeline, profile management, scheduler, SSE |
| server.py | 810 | HTTP dispatcher — CORS, auth middleware, route delegation, static serving |
| database.py | 1,019 | SQLite singleton — all table operations, per-thread connections |
| auth.py | 352 | AuthManager — bcrypt passwords, session tokens, rate limiting |
| sse.py | 62 | SSEManager — client tracking, profile-scoped broadcast |
| migrations.py | 932 | Schema evolution — migrations 001-028 |
| email_service.py | 110 | SMTP email sending for verification codes |
| prompt_version.py | 115 | Prompt format version tracking for training alignment |
| user_data.py | 73 | User data directory helpers |

### Backend Routes (21 files, 11,561 lines)
| File | Lines | Role |
|------|-------|------|
| routes/agent.py | 1,041 | Agent chat — streaming, tool use, suggestions, entity memory |
| routes/agent_tools.py | 1,067 | 11 tool definitions + dispatcher for agent |
| routes/auth.py | 946 | Email auth — register, verify, login, reset, OTP, profiles CRUD |
| routes/personas.py | 951 | Persona registry, prompt builders, _pack_context (16K budget) |
| routes/data_endpoints.py | 965 | /api/data, /api/briefing, /api/status, /api/feedback, /api/search |
| routes/youtube_endpoints.py | 935 | YouTube channel/video CRUD, insight extraction |
| routes/persona_data.py | 696 | Conversation/scenario/entity/workspace per-persona storage |
| routes/rp_chat.py | 625 | RP chat — streaming, swipe, branch, edit, feedback |
| routes/dev_endpoints.py | 587 | Dev tools — prompt builder, codex, sprint log |
| routes/feeds.py | 522 | RSS feeds — finance, politics, jobs, custom + catalog |
| routes/wizard.py | 428 | Onboarding wizard — preselect, tab suggest, rv items |
| routes/rp_memory.py | 400 | Tiered RP memory — extraction + context assembly |
| routes/media.py | 390 | File upload/serve, TTS, STT, proxy |
| routes/character_cards.py | 350 | Character card CRUD + TavernCard V2 import + ratings |
| routes/image_gen.py | 316 | ComfyUI image generation + gallery + GPU swap |
| routes/config.py | 289 | Config save handler |
| routes/persona_prompts.py | 269 | Persona system prompt definitions |
| routes/generate.py | 258 | AI profile generation from role/location/context |
| routes/controls.py | 232 | Scan trigger, cancel, market refresh |
| routes/helpers.py | 228 | JSON response, SSE helpers, gzip, read_json_body |
| routes/gpu_manager.py | 168 | Ollama ↔ ComfyUI GPU swap via API |
| routes/url_validation.py | 91 | SSRF prevention for proxy/RSS |

### Backend Processors (21 files, 10,003 lines)
| File | Lines | Role |
|------|-------|------|
| processors/scorer_adaptive.py | 1,482 | Profile-adaptive scoring engine (sole scorer) |
| processors/youtube.py | 1,230 | YouTube transcript acquisition (3-tier) |
| processors/scorer_base.py | 891 | Shared Ollama infrastructure, calibration, ScoringMemory |
| processors/canon_import.py | 853 | Canon document import pipeline |
| processors/briefing.py | 573 | LLM-powered intelligence briefing generation |
| processors/tts.py | 531 | Dual-engine TTS (Kokoro GPU + Edge-TTS cloud) |
| processors/workspace.py | 431 | Workspace signal management |
| processors/profile_generator.py | 421 | AI category generation from profile |
| processors/scenario_updater.py | 399 | Auto-update scenarios based on agent conversations |
| processors/scenario_generator.py | 391 | 4-pass LLM scenario generation |
| processors/scenario_templates.py | 335 | Scenario template library |
| processors/source_resolver.py | 332 | Source verification for YouTube narrations |
| processors/persona_context.py | 319 | Persona context builder |
| processors/lenses.py | 292 | YouTube lens extraction (6 types) |
| processors/file_handler.py | 276 | File processing for workspace uploads |
| processors/scenarios.py | 262 | Scenario management utilities |
| processors/verification.py | 260 | Citation verification (anti-hallucination RAG) |
| processors/context_compression.py | 245 | Context window compression |
| processors/youtube_worker.py | 215 | Background video processing daemon |
| processors/stt.py | 132 | Speech-to-text (faster-whisper large-v3-turbo) |
| processors/ocr.py | 132 | OCR for image text extraction |

### Backend Fetchers (8 files, 3,530 lines)
| File | Lines | Role |
|------|-------|------|
| fetchers/news.py | 1,010 | Multi-source news (DDG, Serper, RSS, ThreadPoolExecutor) |
| fetchers/kuwait_scrapers.py | 596 | Kuwait-specific news scrapers |
| fetchers/market.py | 478 | Yahoo Finance + Binance market data |
| fetchers/serper_search.py | 353 | Serper.dev API + query tracking |
| fetchers/extra_feeds.py | 351 | RSS aggregation (finance/politics/jobs) |
| fetchers/discovery.py | 314 | Entity frequency tracking |
| fetchers/google_search.py | 300 | Google Custom Search API |
| fetchers/searxng_search.py | 128 | SearXNG metasearch fallback |

### Backend Training Pipeline (4 files, 3,457 lines)
| File | Lines | Role |
|------|-------|------|
| train_lora.py | 1,344 | LoRA fine-tuning → merge → GGUF export → Ollama register |
| autopilot.py | 959 | Autonomous profile cycling + distill + train loop |
| distill.py | 602 | Claude Opus re-scoring to find disagreements |
| export_training.py | 552 | JSONL export aligned to inference format |

### Backend Utilities (6 files, 3,230 lines)
| File | Lines | Role |
|------|-------|------|
| obsidian_export.py | 1,895 | Export data to Obsidian vault |
| model_manager.py | 630 | Ollama model management + GGUF operations |
| evaluate_scorer.py | 356 | Scorer evaluation harness |
| compare_models.py | 678 | Model A/B comparison |
| scripts/regenerate_codex.py | 291 | Codex documentation regeneration |
| scripts/export_obsidian.py | 266 | Obsidian export script |

### Backend Data Pipeline Scripts (38 files, ~11,000 lines)
Located in `data/v2_pipeline/` and `data/rp_pipeline/scripts/`. Training-specific scripts for V2 scorer and RP model. All use direct `sqlite3` imports.

### Frontend (38 active JS files + HTML/CSS, ~40,000 lines)
| File | Lines | Role |
|------|-------|------|
| fullscreen-chart.js | 2,432 | Fullscreen chart modal with drawing tools |
| app.js | 2,144 | Main app state, SSE, data refresh, profile management |
| wizard.js | 2,172 | 7-stage onboarding wizard |
| agent.js | 2,023 | AI chat agent — conversations, SSE streaming, tool use |
| markets-panel.js | 1,920 | Markets widget + asset analysis |
| settings.js | 1,770 | Config management, profile settings |
| ui.js | 1,618 | Main UI controls, Ask AI mode |
| feed.js | 1,518 | News feed display + filtering |
| youtube.js | 1,467 | YouTube channel management + insights |
| auth-star-canvas.js | 1,371 | Auth background animation |
| market.js | 1,360 | Market data processing |
| youtube-kb.js | 1,227 | YouTube knowledge base |
| theme-editor.js | 1,068 | Theme customizer |
| file-browser.js | 1,012 | Workspace file browser + AI assist |
| mobile.js | 860 | Mobile-specific UI adaptations |
| auth.js | 842 | Auth token management, fetch wrapper |
| tour.js | 830 | Guided tour system |
| settings-sources.js | 661 | RSS feed source management |
| prompt-builder.js | 559 | Dev tools for prompt generation |
| wizard-data.js | 487 | Wizard question/answer data |
| games-ui.js | 440 | Gaming scenarios UI |
| ui-sync.js | 388 | UI settings persistence to DB |
| settings-tickers.js | 354 | Ticker symbol management |
| agent-customizer.js | 354 | Agent system prompt customizer |
| stt.js | 344 | Speech-to-text recording |
| persona-context.js | 303 | Persona context editor |
| agent-tickers.js | 298 | Agent ticker context panel |
| codex.js | 279 | Codex documentation modal |
| nav.js | 273 | Navigation sidebar |
| agent-suggestions.js | 267 | Agent suggestion chips |
| settings-categories.js | 245 | Dynamic categories config |
| scan-history.js | 216 | Scan/search history display |
| tts-settings.js | 173 | TTS settings |
| ui-dialogs.js | 173 | Dialog utilities |
| image-gen.js | 172 | Image generation UI |
| workspace.js | 169 | Workspace signal management |
| auth-styles.js | 141 | Auth page styling |
| fullscreen-chart-utils.js | 128 | Chart utility functions |
| sw.js | 88 | Service worker |
| index.html | 1,601 | Main HTML entry point |
| styles.css | 3,559 | Main stylesheet |
| themes.css | 348 | Theme definitions |
| wizard-styles.css | 425 | Wizard-specific styles |

---

## Section B: Backend Dependency Graph

### main.py (1,682 lines) — CENTRAL NODE
```
IMPORTS FROM PROJECT: database, fetchers.market, fetchers.news, fetchers.discovery,
  processors.scorer_adaptive, processors.briefing, sse, user_data
IMPORTS EXTERNAL: copy, hashlib, json, logging, os, yaml, time, threading, datetime, pathlib
READS DB TABLES: news_items, profiles, scan_log
WRITES DB TABLES: news_items, market_snapshots, scan_log, shadow_scores
CALLS OLLAMA: Yes (via AdaptiveScorer, BriefingGenerator)
ACCESSES FILESYSTEM: Yes (output JSON, profile YAML, data/)
USES PROFILE_ID: Yes (active_profile_id, per-scan param)
BROADCASTS SSE: Yes (scan_progress, pass1_complete, briefing_ready, complete)
USES THREADING: Yes (6+ daemon threads: briefing, shadow, distill, scheduler)
CALLED BY: server.py
CALLS INTO: database, fetchers/*, processors/scorer_adaptive, processors/briefing, sse
```

### server.py (810 lines) — CENTRAL NODE
```
IMPORTS FROM PROJECT: routes/* (all 20 modules), email_service, routes.helpers
IMPORTS EXTERNAL: gzip, json, logging, mimetypes, os, sys, signal, time, threading,
  yaml, webbrowser, pathlib, urllib.parse, http.server, socketserver
READS DB TABLES: sessions, profiles, users (auth middleware)
WRITES DB TABLES: sessions (device profile creation)
CALLS OLLAMA: No (delegates to routes)
ACCESSES FILESYSTEM: Yes (static files, gzip cache)
USES PROFILE_ID: Yes (resolves per-request, sets handler._profile_id)
BROADCASTS SSE: No (delegates to routes)
USES THREADING: Yes (ThreadingMixIn, gzip cache lock)
CALLED BY: main.py (serve_frontend)
CALLS INTO: all route modules, auth middleware
```

### database.py (1,019 lines) — CENTRAL NODE
```
IMPORTS FROM PROJECT: migrations, user_data
IMPORTS EXTERNAL: sqlite3, json, threading, datetime, pathlib, logging
READS DB TABLES: ALL (universal read access)
WRITES DB TABLES: ALL (universal write access)
CALLS OLLAMA: No
ACCESSES FILESYSTEM: Yes (strat_os.db file)
USES PROFILE_ID: Yes (most methods accept profile_id param)
BROADCASTS SSE: No
USES THREADING: Yes (threading.local() for per-thread connections, commit lock)
CALLED BY: main.py, server.py, ALL route modules, processors/youtube_worker
CALLS INTO: migrations
```

### auth.py (352 lines)
```
IMPORTS FROM PROJECT: None
IMPORTS EXTERNAL: copy, json, hashlib, secrets, time, threading, logging, yaml, pathlib
READS DB TABLES: sessions, profiles (via db param)
WRITES DB TABLES: sessions (token creation)
CALLS OLLAMA: No
ACCESSES FILESYSTEM: Yes (.sessions.json legacy, profile YAML)
USES PROFILE_ID: Yes (session→profile resolution)
USES THREADING: Yes (_rate_lock)
CALLED BY: server.py
```

### sse.py (62 lines)
```
IMPORTS FROM PROJECT: None
IMPORTS EXTERNAL: json, threading, logging
USES PROFILE_ID: Yes (profile-scoped broadcast)
USES THREADING: Yes (_lock)
CALLED BY: main.py, routes/agent.py, routes/rp_chat.py, routes/data_endpoints.py
```

### Route Modules — All follow the same pattern:
```
IMPORTS FROM PROJECT: routes.helpers (all), routes.personas (agent), routes.agent_tools (agent)
HANDLER SIGNATURE: handle_get/post/delete(handler, strat, auth, path) → bool
PROFILE_ID ACCESS: handler._profile_id (set by server.py middleware)
DB ACCESS: strat.db (via handler parameter)
```

**Ollama-calling routes:** agent.py, rp_chat.py, wizard.py, generate.py, controls.py (triggers scan), media.py (TTS)

**DB-heavy routes:** data_endpoints.py, persona_data.py, youtube_endpoints.py, character_cards.py, rp_chat.py, feeds.py

### Processors — Key dependencies:
```
scorer_adaptive → scorer_base (inheritance)
scorer_base → None (standalone, uses requests for Ollama)
briefing → routes.helpers
youtube_worker → processors.youtube, database (direct sqlite3)
scenario_generator → routes.helpers
profile_generator → routes.helpers
tts → (standalone, uses kokoro/edge-tts)
stt → (standalone, uses faster-whisper)
```

### Training Pipeline — All use direct sqlite3:
```
distill.py → sqlite3, anthropic API (no project imports)
export_training.py → sqlite3, prompt_version (no project imports)
train_lora.py → torch, transformers, peft (no DB, no project imports)
autopilot.py → sqlite3, distill (imports run_distillation)
```

---

## Section D: API Endpoint Map

### Auth Endpoints (routes/auth.py)
| Method | Path | Auth | Profile | Ollama | Tables |
|--------|------|------|---------|--------|--------|
| POST | /api/auth/register | No | No | No | pending_registrations |
| POST | /api/auth/verify | No | No | No | users, profiles, sessions |
| POST | /api/auth/login | No | No | No | users, sessions |
| POST | /api/auth/refresh | Yes | No | No | sessions |
| POST | /api/auth/reset-password | No | No | No | users |
| POST | /api/auth/verify-reset | No | No | No | users |
| POST | /api/auth/otp-request | Yes | No | No | users |
| POST | /api/auth/otp-verify | Yes | No | No | users |
| GET | /api/auth/me | Yes | Yes | No | users, profiles |
| POST | /api/auth/change-password | Yes | No | No | users |
| GET | /api/profiles | Yes | Yes | No | profiles |
| POST | /api/profiles | Yes | Yes | No | profiles |
| DELETE | /api/profiles/:id | Yes | Yes | No | profiles |
| POST | /api/admin/invite | Yes | No | No | invite_codes |

### Intelligence Endpoints
| Method | Path | Auth | Profile | Ollama | Tables | Used By |
|--------|------|------|---------|--------|--------|---------|
| GET | /api/data | Yes | Yes | No | news_items, market_snapshots | web |
| GET | /api/news | Yes | Yes | No | news_items | web |
| GET | /api/market | Yes | No | No | market_snapshots | web |
| GET | /api/status | Yes | No | No | scan_log | web |
| GET | /api/briefing | Yes | Yes | No | briefings | web |
| GET | /api/health | No | No | Yes | None | both |
| GET | /api/config | Yes | No | No | None (YAML) | web |
| POST | /api/config | Yes | No | No | None (YAML) | web |
| POST | /api/feedback | Yes | Yes | No | user_feedback | web |
| GET | /api/export | Yes | Yes | No | news_items | web |
| GET | /api/search | Yes | Yes | No | news_items | web |
| GET | /api/events | Yes | Yes | No | None (SSE) | web |

### Agent Chat
| Method | Path | Auth | Profile | Ollama | Tables | Used By |
|--------|------|------|---------|--------|--------|---------|
| POST | /api/agent-chat | Yes | Yes | **Yes** | conversations, scenarios | web |
| POST | /api/ask | Yes | Yes | **Yes** | news_items | web |
| POST | /api/suggest-context | Yes | Yes | **Yes** | profiles, scenarios | web+mobile |
| GET | /api/agent-status | No | No | **Yes** | None | web |

### RP Chat (mobile + web)
| Method | Path | Auth | Profile | Ollama | Tables | Used By |
|--------|------|------|---------|--------|--------|---------|
| POST | /api/rp/chat | Yes | Yes | **Yes** | rp_messages, character_cards | both |
| POST | /api/rp/regenerate | Yes | Yes | **Yes** | rp_messages | both |
| POST | /api/rp/edit | Yes | Yes | No | rp_edits | both |
| POST | /api/rp/branch | Yes | Yes | **Yes** | rp_messages | both |
| POST | /api/rp/director-note | Yes | Yes | No | rp_suggestions | both |
| POST | /api/rp/feedback | Yes | Yes | No | rp_feedback | both |
| GET | /api/rp/history/:sid | Yes | Yes | No | rp_messages | both |
| GET | /api/rp/branches/:sid | Yes | Yes | No | rp_messages | both |

### Character Cards
| Method | Path | Auth | Profile | Ollama | Tables | Used By |
|--------|------|------|---------|--------|--------|---------|
| POST | /api/cards | Yes | Yes | No | character_cards | both |
| GET | /api/cards/:id | Yes | No | No | character_cards | both |
| PUT | /api/cards/:id | Yes | Yes | No | character_cards | both |
| DELETE | /api/cards/:id | Yes | Yes | No | character_cards | both |
| GET | /api/cards/browse | Yes | No | No | character_cards | both |
| GET | /api/cards/trending | Yes | No | No | character_cards, character_card_stats | mobile |
| GET | /api/cards/search | Yes | No | No | character_cards | both |
| GET | /api/cards/my | Yes | Yes | No | character_cards | mobile |
| POST | /api/cards/:id/rate | Yes | Yes | No | character_card_ratings | both |
| POST | /api/cards/:id/publish | Yes | Yes | No | character_cards | both |
| POST | /api/cards/:id/save | Yes | Yes | No | character_cards | mobile |

### Image Generation
| Method | Path | Auth | Profile | Ollama | Tables | Used By |
|--------|------|------|---------|--------|--------|---------|
| POST | /api/image/generate | Yes | Yes | GPU swap | generated_images | both |
| POST | /api/image/character-portrait | Yes | Yes | GPU swap | generated_images | both |
| GET | /api/image/gallery | Yes | Yes | No | generated_images | both |
| GET | /api/image/:id | Yes | No | No | generated_images | both |
| DELETE | /api/image/:id | Yes | Yes | No | generated_images | both |

### Scan Controls
| Method | Path | Auth | Profile | Ollama | Used By |
|--------|------|------|---------|--------|---------|
| POST | /api/refresh | Yes | Yes | **Yes** | web |
| POST | /api/refresh-market | Yes | Yes | No | web |
| POST | /api/refresh-news | Yes | Yes | **Yes** | web |
| POST | /api/scan/cancel | Yes | Yes | No | web |
| GET | /api/scan/status | Yes | Yes | No | web |

### Wizard
| Method | Path | Auth | Profile | Ollama | Used By |
|--------|------|------|---------|--------|---------|
| POST | /api/wizard-preselect | Yes | Yes | **Yes** | web |
| POST | /api/wizard-tab-suggest | Yes | Yes | **Yes** | web |
| GET | /api/wizard-rv-items | Yes | Yes | **Yes** | web |

### Feeds, YouTube, Persona, Media, Files
*(~30 additional endpoints — all auth-required, profile-scoped, DB-backed, web-only except /api/scenarios which mobile uses)*

---

## Section E: Database Table Ownership

### Critical Tables
| Table | Rows | Written By | Read By |
|-------|------|-----------|---------|
| news_items | 16,950 | main.py, data_endpoints | main.py, data_endpoints, feed.js, distill.py, export_training.py |
| market_snapshots | 43,030 | main.py (via MarketFetcher) | data_endpoints, market.js |
| user_feedback | 2,637 | data_endpoints, distill.py, autopilot.py | export_training.py, scorer_base (ScoringMemory) |
| rp_messages | 1,137 | rp_chat.py | rp_chat.py, rp_memory.py |
| character_cards | 14 | character_cards.py | character_cards.py, rp_chat.py |
| users | 82 | routes/auth.py | routes/auth.py, server.py |
| profiles | 83 | routes/auth.py, server.py | main.py, server.py, routes/auth.py |
| sessions | 61 | routes/auth.py, server.py | server.py (auth middleware) |
| briefings | 244 | main.py (BriefingGenerator) | data_endpoints |
| youtube_videos | 124 | youtube_endpoints, youtube_worker | youtube_endpoints |
| video_insights | 194 | youtube_endpoints, youtube_worker | youtube_endpoints |
| conversations | 19 | persona_data | persona_data, agent.py |
| scenarios | 8 | persona_data, scenario_generator | persona_data, games-ui.js |
| generated_images | 19 | image_gen.py | image_gen.py |
| scan_log | 252 | main.py | data_endpoints |

---

## Section F: Shared Resource Map

| Resource | Accessed By |
|----------|------------|
| **Database singleton** | main.py, server.py, ALL route modules, youtube_worker.py |
| **Config (YAML)** | main.py, server.py, auth.py, scorer_adaptive, distill.py, autopilot.py, train_lora.py |
| **SSE manager** | main.py, routes/agent.py, routes/rp_chat.py, routes/data_endpoints.py, routes/controls.py |
| **Ollama (localhost:11434)** | scorer_base, scorer_adaptive, briefing, agent.py, rp_chat.py, wizard.py, generate.py, lenses, scenario_generator, youtube.py |
| **Profile YAML files** | auth.py, main.py, server.py (legacy) |
| **User data directories** | user_data.py, media.py, file_handler.py, persona_data.py |
| **Auth manager** | server.py (middleware), routes/auth.py |
| **GPU manager** | image_gen.py, rp_chat.py (ensure_ollama) |

---

## Section G: Coupling Clusters

### Tightly Coupled (must audit together)
1. **Scoring Pipeline** — scorer_base.py + scorer_adaptive.py + main.py (scan logic) + export_training.py (format alignment)
   - Reason: Training format must be CHARACTER-FOR-CHARACTER identical to inference format
2. **Auth + Sessions** — auth.py + routes/auth.py + server.py (middleware) + migrations.py (user tables)
   - Reason: Auth middleware in server.py depends on auth.py session model + DB schema
3. **RP Chat System** — rp_chat.py + rp_memory.py + character_cards.py + database.py (RP methods)
   - Reason: Branching logic, message insertion, memory extraction all interdependent
4. **Agent System** — agent.py + agent_tools.py + personas.py + persona_prompts.py + persona_context.py
   - Reason: Agent chat builds context from personas, uses tools, generates suggestions — all tightly wired
5. **YouTube Pipeline** — youtube_endpoints.py + youtube.py + youtube_worker.py + lenses.py + verification.py + source_resolver.py
   - Reason: Background worker depends on processor, endpoints depend on both

### Loosely Coupled (1-2 dependencies)
- **feeds.py** — depends on helpers.py only
- **controls.py** — depends on helpers.py, delegates to main.py
- **config.py** — depends on helpers.py only
- **wizard.py** — depends on helpers.py + Ollama
- **generate.py** — depends on helpers.py + Ollama
- **tts.py** — standalone (Kokoro + Edge-TTS engines)
- **stt.py** — standalone (faster-whisper)
- **ocr.py** — standalone

### Independent (zero internal project deps)
- **fetchers/market.py** — yfinance only
- **fetchers/discovery.py** — pure state tracking
- **fetchers/extra_feeds.py** — pure RSS
- **fetchers/searxng_search.py** — HTTP only
- **email_service.py** — SMTP only
- All training data scripts in `data/v2_pipeline/` and `data/rp_pipeline/` (direct sqlite3)

---

## Section H: Critical Path Analysis

Ranked by blast radius (files that break the most things if changed):

| Rank | File | Depended On By | Provides |
|------|------|---------------|----------|
| 1 | **database.py** | ~30 files | ALL data access — every route, processor, and training script |
| 2 | **server.py** | ALL routes (21) | HTTP dispatch, auth middleware, profile resolution |
| 3 | **main.py** | server.py, controls.py | Scan pipeline, StratOS orchestrator, background scheduler |
| 4 | **routes/helpers.py** | ALL route modules (20) | json_response, error_response, read_json_body, SSE helpers |
| 5 | **auth.py** | server.py, routes/auth.py | Session validation, rate limiting |
| 6 | **sse.py** | main.py, agent.py, rp_chat.py, data_endpoints.py | Real-time event broadcasting |
| 7 | **scorer_base.py** | scorer_adaptive.py | Ollama client, calibration, ScoringMemory |
| 8 | **migrations.py** | database.py | Schema definition for ALL tables |
| 9 | **routes/personas.py** | agent.py | Context packing, persona prompts |
| 10 | **processors/youtube.py** | youtube_worker.py, youtube_endpoints.py | Transcript + insight extraction |

---

## Section I: Parallel Session Plan

### Group 1: Central Nodes (run FIRST, sequentially)

**Session 1A: database.py + migrations.py** (1,951 lines)
- Files to modify: database.py, migrations.py
- Read-only deps: ALL route modules (understand the interface)
- Verify: Every `db.` method still works, all table schemas correct
- Complexity: **LARGE** — every other file depends on this

**Session 1B: server.py + auth.py + sse.py** (1,224 lines)
- Files to modify: server.py, auth.py, sse.py
- Read-only deps: routes/helpers.py, main.py
- Verify: Server boots, auth middleware resolves profiles, SSE connects
- Complexity: **LARGE** — HTTP dispatch + auth + profile isolation

**Session 1C: main.py** (1,682 lines)
- Files to modify: main.py
- Read-only deps: database.py, all fetchers, scorer_adaptive, briefing, sse
- Verify: Scan pipeline works, scheduler starts, briefing generates
- Complexity: **LARGE** — orchestrator with 6 thread types

### Group 2: Route Modules (run in PARALLEL after Group 1)

**Session 2A: Agent System** (3,328 lines)
- Files: routes/agent.py, routes/agent_tools.py, routes/personas.py, routes/persona_prompts.py
- No overlap with 2B-2F
- Complexity: **LARGE**

**Session 2B: RP + Cards System** (1,775 lines)
- Files: routes/rp_chat.py, routes/rp_memory.py, routes/character_cards.py, routes/image_gen.py, routes/gpu_manager.py
- No overlap with 2A/2C-2F
- Complexity: **MEDIUM**

**Session 2C: Data + Feeds + Controls** (2,684 lines)
- Files: routes/data_endpoints.py, routes/feeds.py, routes/controls.py, routes/config.py, routes/media.py
- No overlap with 2A-2B/2D-2F
- Complexity: **MEDIUM**

**Session 2D: YouTube Pipeline** (2,380 lines)
- Files: routes/youtube_endpoints.py, processors/youtube.py, processors/youtube_worker.py, processors/lenses.py, processors/verification.py, processors/source_resolver.py
- No overlap with 2A-2C/2E-2F
- Complexity: **MEDIUM**

**Session 2E: Auth + Wizard + Generate + Persona Data** (2,328 lines)
- Files: routes/auth.py, routes/wizard.py, routes/generate.py, routes/persona_data.py, routes/dev_endpoints.py
- No overlap with 2A-2D/2F
- Complexity: **MEDIUM**

**Session 2F: Scoring Pipeline** (2,373 lines)
- Files: processors/scorer_adaptive.py, processors/scorer_base.py
- Read-only deps: export_training.py (format alignment)
- No overlap with 2A-2E
- Complexity: **MEDIUM** — but CRITICAL for training alignment

### Group 3: Processors (run in PARALLEL after Group 2)

**Session 3A: Briefing + Profile + Context** (1,558 lines)
- Files: processors/briefing.py, processors/profile_generator.py, processors/persona_context.py, processors/context_compression.py, processors/workspace.py
- Complexity: **SMALL**

**Session 3B: Scenarios** (1,387 lines)
- Files: processors/scenario_generator.py, processors/scenario_updater.py, processors/scenario_templates.py, processors/scenarios.py
- Complexity: **SMALL**

**Session 3C: TTS + STT + OCR + File Handler + Canon** (1,924 lines)
- Files: processors/tts.py, processors/stt.py, processors/ocr.py, processors/file_handler.py, processors/canon_import.py
- Complexity: **SMALL** — largely standalone

### Group 4: Fetchers (run in PARALLEL, independent)

**Session 4A: News + Search** (1,491 lines)
- Files: fetchers/news.py, fetchers/serper_search.py, fetchers/searxng_search.py
- Complexity: **SMALL**

**Session 4B: Market + Discovery + Extra Feeds + Kuwait** (1,739 lines)
- Files: fetchers/market.py, fetchers/discovery.py, fetchers/extra_feeds.py, fetchers/kuwait_scrapers.py, fetchers/google_search.py
- Complexity: **SMALL**

### Group 5: Training Pipeline (run after Group 1, independent of 2-4)

**Session 5A: Training Scripts** (3,457 lines)
- Files: distill.py, export_training.py, train_lora.py, autopilot.py
- Read-only deps: prompt_version.py, config.yaml format
- Complexity: **MEDIUM** — direct sqlite3 usage needs attention

### Group 6: Frontend (run in PARALLEL, independent of backend)

**Session 6A: Core Frontend** (6,877 lines)
- Files: app.js, auth.js, nav.js, ui.js, ui-sync.js, ui-dialogs.js
- Complexity: **LARGE**

**Session 6B: Agent + Games** (3,514 lines)
- Files: agent.js, agent-suggestions.js, agent-customizer.js, agent-tickers.js, games-ui.js
- Complexity: **MEDIUM**

**Session 6C: Markets + Charts** (5,840 lines)
- Files: market.js, markets-panel.js, fullscreen-chart.js, fullscreen-chart-utils.js
- Complexity: **MEDIUM**

**Session 6D: Settings + Wizard** (5,679 lines)
- Files: settings.js, settings-tickers.js, settings-sources.js, settings-categories.js, wizard.js, wizard-data.js
- Complexity: **MEDIUM**

**Session 6E: Feed + YouTube** (4,212 lines)
- Files: feed.js, youtube.js, youtube-kb.js
- Complexity: **MEDIUM**

**Session 6F: Specialized** (4,636 lines)
- Files: theme-editor.js, tour.js, mobile.js, file-browser.js, prompt-builder.js, codex.js, persona-context.js, workspace.js, scan-history.js, image-gen.js, stt.js, tts-settings.js, auth-star-canvas.js, auth-styles.js
- Complexity: **MEDIUM** — many small files

---

## Section J: External Dependencies

### Backend (Python)
| Package | Used By | Purpose |
|---------|---------|---------|
| sqlite3 | database.py, distill.py, autopilot.py, export_training.py, training scripts | Database |
| requests | scorer_base, fetchers/*, routes/agent, routes/rp_chat | HTTP client |
| yaml | main.py, auth.py, config.py, autopilot.py | Config parsing |
| bcrypt | routes/auth.py | Password hashing |
| yfinance | fetchers/market.py | Market data |
| torch | train_lora.py, train_v2.py, train_rp.py | ML training |
| transformers | train_lora.py, train_v2.py | Model loading |
| peft | train_lora.py, train_v2.py | LoRA adapters |
| unsloth | train_lora.py | CUDA-optimized training |
| datasets | train_lora.py | HuggingFace datasets |
| kokoro | processors/tts.py | Local TTS |
| edge_tts | processors/tts.py | Cloud TTS (Arabic) |
| faster_whisper | processors/stt.py | Local STT |
| pytesseract | processors/ocr.py | OCR |

### Frontend (JavaScript)
| Library | Used By | Purpose |
|---------|---------|---------|
| Lightweight Charts | fullscreen-chart.js, markets-panel.js | TradingView charting |
| Lucide | Multiple | Icon library |
| Tailwind CSS | styles | Utility CSS (built offline) |

---

## Section K: Known Anomalies

### Dead Code
- `frontend/Prototypes/` — 20+ prototype HTML files, a copy of `markets-panel.js`, copies of `settings-categories.js` and `workspace.js`. None imported by index.html. Safe to delete.
- `backend/compare_models.py` (678 lines) — standalone evaluation script, not imported by anything
- `backend/evaluate_scorer.py` (356 lines) — standalone evaluation script
- `backend/obsidian_export.py` (1,895 lines) — export utility, not part of server

### Inconsistent Patterns
- **Auth header**: Web uses fetch wrapper in auth.js (auto-injects X-Auth-Token). Mobile uses manual headers in apiFetch. Both work but different mechanisms.
- **Profile resolution**: server.py has TWO auth paths — modern email-based (sessions table) and legacy PIN-based (config YAML). The PIN path sets `active_profile_id` on the StratOS instance, which is NOT per-request safe under ThreadingMixIn.
- **Database access**: Route modules use `strat.db` via handler parameter. Training scripts use direct `sqlite3.connect()`. youtube_worker uses its own connection. Three different access patterns.
- **SSE**: Main app events go through SSEManager. Agent chat streaming goes through direct `handler.wfile.write()`. RP chat streaming also writes directly. Inconsistent streaming patterns.

### Duplicated Logic
- `backend/routes/agent.py` has its own Ollama streaming (`_stream_ollama_raw`). `backend/routes/rp_chat.py` has another (`_stream_ollama`). `backend/processors/scorer_base.py` has a third (`_call_ollama`). Three separate Ollama client implementations.
- `frontend/youtube.js` and `frontend/youtube-kb.js` share significant overlap in channel management and insight display.

### Files Doing Too Many Things
- **main.py** (1,682 lines) — orchestrator + scan pipeline + market refresh + news refresh + briefing spawn + shadow scoring + scheduler + profile management + output building
- **server.py** (810 lines) — HTTP dispatch + CORS + auth middleware + device profile creation + static serving + gzip caching
- **routes/agent.py** (1,041 lines) — chat handling + suggestion generation + entity memory + multi-modal tool use

### Threading Risks
- `main.py` spawns 6+ daemon threads that all access `strat.db` through the singleton. SQLite WAL handles this but under load (multiple scans + briefing + shadow scoring), "database is locked" errors occur.
- `youtube_worker.py` creates its own sqlite3 connection per operation specifically to avoid this — a workaround for the shared singleton problem.
- Legacy PIN auth path sets `strat.active_profile_id` on the shared StratOS instance — NOT safe with ThreadingMixIn if two requests arrive simultaneously for different profiles.

### Hardcoded Values
- `routes/rp_chat.py:162` — Ollama timeout hardcoded to 120s
- `fetchers/news.py` — ThreadPoolExecutor max_workers=5 hardcoded
- `processors/tts.py` — 15s synthesis timeout hardcoded
- `routes/helpers.py:19` — max JSON body 10MB hardcoded
- `processors/scorer_base.py` — ScoringMemory 30-day window hardcoded

---

### Not Mapped (intentionally excluded)
- `__init__.py` files (empty, 0-1 lines)
- `test_feeds_quick.py`, `test_media_feeds.py` (ad-hoc test scripts)
- `tests/test_agent_personas.py` (351 lines), `tests/test_format_alignment.py` (634 lines) — test files, not production code
- `frontend/tailwind.config.js` (26 lines) — build config
- `frontend/Prototypes/` (20+ files) — dead prototype HTML/JS, not loaded by index.html

---

*This map covers 77 backend Python files (~37,000 lines), 38 frontend JS files (~33,000 lines), plus ~11,000 lines of training pipeline scripts. Total: ~81,000 lines mapped.*
