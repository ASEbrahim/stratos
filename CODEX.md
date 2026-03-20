# StratOS Codebase Codex (فهرس)

Generated: 2026-03-13 | 13 categories | 139 terms

## 🏗️ Architecture

### Dispatch Dict
**Files:** `backend/server.py`  
**Related:** Route Module, CORSHandler  
**Added:** Sprint 5K

Sequential route dispatch in server.py — do_GET/do_POST call handle_get/handle_post on each route module in order, first to return True wins. Replaced a 2,860-line if/elif chain in Sprint 5K.

---

### Route Module
**Files:** `backend/routes/`  
**Related:** Dispatch Dict, CORSHandler  
**Added:** Sprint 5K

A Python file in backend/routes/ that exports handle_get(), handle_post(), and/or handle_delete(). Returns True if handled, False to pass to next module. 16 route modules total.

---

### CORSHandler
**Files:** `backend/server.py`  
**Related:** Dispatch Dict, ThreadedHTTPServer  
**Added:** Sprint 1

Custom HTTP handler extending SimpleHTTPRequestHandler. Adds CORS headers, auth middleware, gzip compression for static files, and SSE support.

---

### ThreadedHTTPServer
**Files:** `backend/server.py`  
**Related:** CORSHandler  
**Added:** Sprint 1

ThreadingMixIn + HTTPServer — handles concurrent requests via thread pool. Each request gets its own thread.

---

### SSE Event System
**Files:** `backend/sse.py`, `backend/server.py`  
**Related:** SSEManager  
**Added:** Sprint 1

Server-Sent Events via GET /api/events. SSEManager tracks connected clients and broadcasts events (scan_progress, briefing_ready, lens_extracted, youtube_status). Frontend listens via EventSource.

---

### SSEManager
**Files:** `backend/sse.py`  
**Related:** SSE Event System  
**Added:** Sprint 1

Singleton class — register(wfile), unregister(wfile), broadcast(event_type, data, profile_id). Profile-scoped broadcasting.

---

### Context Packing
**Files:** `backend/routes/personas.py`  
**Related:** Selective Loading, Token Budget, Persona  
**Added:** Sprint 5K

_pack_context() in routes/personas.py — builds per-persona system prompts within a 16K token budget (word count × 1.3). Universal sections always included, persona-specific fill remaining.

---

### Config Overlay System
**Files:** `backend/database.py`, `backend/routes/config.py`  
**Related:** Profile Isolation  
**Added:** Sprint 3

Per-profile config overrides in profiles.config_overlay (JSON). Merged on top of base config.yaml at runtime. Per-user tickers, categories, feed preferences.

---

### Profile Isolation
**Files:** `backend/server.py`, `backend/routes/data_endpoints.py`  
**Related:** Config Overlay System, Auth Middleware  
**Added:** Sprint 3

Every API request resolves profile_id from X-Auth-Token. News, feedback, conversations, scenarios, files, UI state all scoped to profile_id. Cross-profile bleed fixed Sprint 15.

---

### Auth Middleware
**Files:** `backend/server.py`, `backend/auth.py`  
**Related:** Profile Isolation, Session Token  
**Added:** Sprint 3

Validates X-Auth-Token for all /api/* paths (except proxy and AUTH_EXEMPT). Resolves profile_id. Also accepts token as query param for downloads.

---

### Gzip Cache
**Files:** `backend/server.py`  
**Related:** CORSHandler  
**Added:** Sprint 12

In-memory cache for compressed static files keyed by filepath + mtime. Avoids re-compressing on every request.

---

### StratOS Class
**Files:** `backend/main.py`  
**Related:** Dispatch Dict, Background Scheduler  
**Added:** Sprint 1

Main orchestrator in main.py (1,678 lines). Owns config, database, scanner, scorer, SSE manager. Methods: run_scan(), run_market_refresh(), _call_ollama().

---

### Background Scheduler
**Files:** `backend/main.py`  
**Related:** StratOS Class  
**Added:** Sprint 2

Optional background thread for periodic news+market scans. Controlled by schedule.background_enabled and schedule.background_interval_minutes.


## 🎭 Personas

### Persona
**Files:** `backend/routes/personas.py`, `backend/routes/persona_prompts.py`  
**Related:** Context Packing, Persona Context Isolation  
**Added:** Sprint 5K

Specialized AI personality. Six: Intelligence (news), Market (finance), Scholarly (academic/YouTube), Gaming (RPG), Anime (entertainment), TCG (trading cards). Each has distinct prompt, tools, context.

---

### Persona Context Isolation
**Files:** `backend/routes/personas.py`, `frontend/agent-suggestions.js`  
**Related:** Persona, Context Packing  
**Added:** Sprint 6

Each persona gets own system prompt, conversation history, entity memory, file space. persona_context table stores custom instructions. Suggestions isolated too.

---

### Persona Guide
**Files:** `frontend/agent.js`  
**Related:** Persona  
**Added:** Sprint 13

V3 grid popup — 6 persona cards with star parallax, frosted glass. Click card to switch, click example to send as prompt. World Import banner for Gaming.

---

### Context Builder
**Files:** `backend/routes/personas.py`  
**Related:** Context Packing, Selective Loading  
**Added:** Sprint 5K

Per-persona functions assembling system prompt. Each persona branch in _pack_context() adds domain-specific data.

---

### Dynamic Suggestions
**Files:** `frontend/agent-suggestions.js`, `backend/routes/agent.py`  
**Related:** Persona, Response Chips  
**Added:** Sprint 8

Per-persona follow-up suggestions after each response. Intelligence→news queries, Market→ticker analysis, Gaming→immersive actions. SSE-delivered, rendered as chips.

---

### Persona Prompt Builder
**Files:** `backend/routes/persona_prompts.py`  
**Related:** Persona, Context Builder  
**Added:** Sprint 9

persona_prompts.py — extracted prompt definitions for all 6 personas plus dispatcher function. 188 lines.


## 🤖 Agent System

### Agent Chat
**Files:** `backend/routes/agent.py`, `frontend/agent.js`  
**Related:** Tool Dispatch, Persona  
**Added:** Sprint 5K

POST /api/agent-chat — streaming SSE response from Ollama with tool-call support. All 6 personas with persona-specific prompts and tool access.

---

### Agent Tools
**Files:** `backend/routes/agent_tools.py`  
**Related:** Tool Dispatch, execute_tool  
**Added:** Sprint 5K

11 Ollama function-calling tools: web_search, manage_watchlist, manage_categories, search_feed, search_files, read_document, search_insights, list_channels, get_video_summary, search_narrations, analyze_image, import_canon_world.

---

### Tool Dispatch
**Files:** `backend/routes/agent_tools.py`  
**Related:** Agent Tools  
**Added:** Sprint 5K

execute_tool() in agent_tools.py — routes tool calls to implementations. parse_text_tool_calls() handles LLMs emitting JSON in text.

---

### Response Chips
**Files:** `frontend/agent.js`  
**Related:** Dynamic Suggestions  
**Added:** Sprint 12

Clickable action buttons below agent responses (Track, Summarize, Search). Disabled for non-Intelligence personas.

---

### Conversation Management
**Files:** `backend/routes/persona_data.py`, `frontend/agent.js`  
**Related:** Persona Context Isolation  
**Added:** Sprint 7

DB-backed chat history in conversations table. Per-persona, per-profile. Create, switch, delete, archive via is_active flag.

---

### Free Length Toggle
**Files:** `frontend/agent.js`  
**Related:** Agent Chat  
**Added:** Sprint 12

Short/Long mode. Short: num_predict 1500 + brevity. Long: num_predict 8000 + structured formatting. Persisted to localStorage.

---

### All Scans Toggle
**Files:** `frontend/agent.js`  
**Related:** Agent Chat  
**Added:** Sprint 12

Current/All Scans toggle. 'All' loads historical scan data + 30 articles. 'Current' loads latest 10.

---

### Fullscreen Agent
**Files:** `frontend/agent.js`, `frontend/agent-customizer.js`  
**Related:** Conversation Management  
**Added:** Sprint 9

Expanded agent view filling main content. Sidebar with conversation list, resizable. Customizer panel (FROZEN).

---

### Thinking Indicator
**Files:** `backend/routes/helpers.py`, `frontend/agent.js`  
**Related:** Agent Chat  
**Added:** Sprint 5K

Strip <think>...</think> blocks from Ollama responses. Shows 'thinking...' pulse during streaming. strip_think_blocks() in helpers.py.

---

### Ask Agent Button
**Files:** `frontend/feed.js`, `frontend/agent.js`  
**Related:** Agent Chat  
**Added:** Sprint 15

Button on feed cards opens fullscreen agent, auto-sends article for analysis. Switches to Scholarly persona.


## 🎮 Gaming

### Scenario
**Files:** `backend/processors/scenario_templates.py`, `backend/routes/persona_data.py`  
**Related:** Entity, Selective Loading, Canon Import  
**Added:** Sprint 7

Game world as directory under data/users/{id}/context/gaming/scenarios/{name}/. Contains world/, characters/, scenes/, items/. DB entry in scenarios table.

---

### Entity
**Files:** `backend/routes/agent.py`  
**Related:** Scenario, Auto-Memory  
**Added:** Sprint 6

Character/NPC tracked in persona_entities table. Has identity_md, personality_md, speaking_style_md, memory_md, etc. Persona-generic.

---

### NPC Roster
**Files:** `backend/routes/agent.py`  
**Related:** Entity, Scenario Auto-Updater  
**Added:** Sprint 7

Active NPCs in a scenario stored in persona_entities with scenario_name. Agent auto-detects speaking NPC and updates memory.

---

### Immersive RP Mode
**Files:** `frontend/agent.js`, `backend/routes/personas.py`  
**Related:** GM Mode  
**Added:** Sprint 6

AI stays in character — no meta-commentary, first/third-person narration. Toggled via stratos-agent-rp-mode localStorage.

---

### GM Mode
**Files:** `backend/routes/personas.py`  
**Related:** Immersive RP Mode, Scenario  
**Added:** Sprint 7

Game Master mode — AI manages world, NPCs, quests, stats. Generates scenes, NPC dialogue, combat.

---

### Scenario Auto-Updater
**Files:** `backend/processors/scenario_updater.py`  
**Related:** Scenario, Entity  
**Added:** Sprint 7

Daemon thread after each Gaming response. Updates: current scene, NPC memory, stats, inventory, quest state.

---

### Auto-Memory
**Files:** `backend/routes/agent.py`  
**Related:** Entity, NPC Roster  
**Added:** Sprint 6

After each agent response, _update_entity_memory() extracts facts about active NPC and appends to memory_md.

---

### Selective Loading
**Files:** `backend/routes/personas.py`  
**Related:** Context Packing, Scenario  
**Added:** Sprint 7

Only scenario files relevant to current interaction loaded into context. Keyword-triggered, 4,000 token budget.

---

### Canon Import
**Files:** `backend/processors/canon_import.py`, `backend/routes/agent_tools.py`  
**Related:** Scenario, Fandom Wiki Integration  
**Added:** Sprint 13

import_canon_world tool — imports franchise from Fandom wiki. 5-pass pipeline. FandomFetcher with batch wikitext. 50+ franchise aliases.

---

### Fandom Wiki Integration
**Files:** `backend/processors/canon_import.py`  
**Related:** Canon Import  
**Added:** Sprint 13

FandomFetcher — Fandom revisions API (batch 10 pages/request). Alias map ('SAO'→'swordartonline'). Serper fallback for unknown franchises.

---

### Scenario Generator
**Files:** `backend/processors/scenario_generator.py`  
**Related:** Scenario  
**Added:** Sprint 7

4-pass LLM generation: world→characters→items→opening scene. Creates directory structure with markdown. Progress via SSE.

---

### Scenario Templates
**Files:** `backend/processors/scenario_templates.py`  
**Related:** Scenario  
**Added:** Sprint 7

Directory structure templates. create_scenario_skeleton() makes world/, characters/, scenes/, items/. NPC and item markdown templates.


## 📊 Scoring Pipeline

### AdaptiveScorer
**Files:** `backend/processors/scorer_adaptive.py`  
**Related:** ScorerBase, KeywordIndex, Noise Patterns  
**Added:** Sprint B3

Sole scorer since B3.3. Extends ScorerBase. Dynamically builds relevance rules from user's categories, keywords, role. Hybrid: rule-based noise + Ollama LLM. ~1,200 lines.

---

### ScorerBase
**Files:** `backend/processors/scorer_base.py`  
**Related:** AdaptiveScorer, ScoringMemory  
**Added:** Sprint 1

Base class: Ollama client, calibration tables, score classification, noise patterns, language filtering, ScoringMemory. ~850 lines.

---

### ScoringMemory
**Files:** `backend/processors/scorer_base.py`  
**Related:** Incremental Scanning  
**Added:** Sprint 2

Per-profile scoring cache. record() saves URL→score, should_reuse() checks if scored, is_fresh() checks staleness. Enables incremental scanning.

---

### KeywordIndex
**Files:** `backend/processors/scorer_adaptive.py`  
**Related:** AdaptiveScorer  
**Added:** Sprint B3

Fast keyword lookup with frequency tracking. Built from user's categories and custom keywords. Rule-based pre-scoring before LLM.

---

### SCORE:X.X|REASON: Format
**Files:** `backend/processors/scorer_base.py`, `backend/export_training.py`  
**Related:** Training/Inference Alignment  
**Added:** Sprint 1

Output format from scoring model. First line: SCORE:X.X|REASON:explanation. Parsed by _parse_score(). Must match between training and inference.

---

### Forbidden 5.0
**Files:** `backend/processors/scorer_base.py`  
**Related:** AdaptiveScorer  
**Added:** Sprint 2

Score 5.0 = model indecision. Config scoring.forbidden_score forces re-score or fallback. Training data excludes 5.0.

---

### DoRA Fine-Tuning
**Files:** `backend/train_lora.py`  
**Related:** LoRA Training Pipeline  
**Added:** Sprint V2

Weight-Decomposed Low-Rank Adaptation. V2 pipeline. Dual weighting: WeightedRandomSampler + per-sample loss weighting.

---

### LoRA Training Pipeline
**Files:** `backend/train_lora.py`  
**Related:** DoRA Fine-Tuning, Distillation  
**Added:** Sprint 1

train_lora.py (1,344 lines). Auto-selects model tier by VRAM. Unsloth (CUDA) + PEFT (ROCm). Pipeline: train→merge→GGUF→Ollama register.

---

### Distillation
**Files:** `backend/distill.py`  
**Related:** LoRA Training Pipeline, Shadow Scores  
**Added:** Sprint 1

Teacher-student via Claude Opus. Re-scores items, saves disagreements (≥2.0 delta) as corrections in user_feedback.

---

### Shadow Scores
**Files:** `backend/processors/scorer_base.py`  
**Related:** Distillation  
**Added:** Sprint 2

Side-by-side comparisons in shadow_scores table. Tracks model drift across training runs.

---

### Noise Patterns
**Files:** `backend/processors/scorer_base.py`  
**Related:** AdaptiveScorer  
**Added:** Sprint 1

Rule-based pre-filters: NOISE_EXACT, NOISE_PATTERNS, STALE_PATTERNS, GARBAGE_DOMAINS. Matching articles skip LLM entirely.

---

### Incremental Scanning
**Files:** `backend/main.py`  
**Related:** ScoringMemory  
**Added:** Sprint 3

_reuse_snapshot_scores() — URL→score lookup from previous output. Already-scored skip LLM. Skipped if context hash changed.


## 📺 YouTube Pipeline

### Lens
**Files:** `backend/processors/lenses.py`  
**Related:** Lens Extraction, YouTube Processor  
**Added:** Sprint YouTube

Knowledge extraction template for transcripts. 7 lenses: transcript, summary, eloquence, narrations, history, spiritual, politics. Defined in LENS_PROMPTS.

---

### Lens Extraction
**Files:** `backend/processors/lenses.py`, `backend/routes/youtube_endpoints.py`  
**Related:** Lens, Merge Dedup Keys  
**Added:** Sprint YouTube

On-demand via POST /api/youtube/extract-lens. Daemon thread, stored in video_insights. Merge modes: 'new', 'replace', 'merge' (append + dedup).

---

### Three-Tier Transcript
**Files:** `backend/processors/youtube.py`  
**Related:** YouTube Processor  
**Added:** Sprint YouTube

Tier 1: youtube-transcript-api (fast, free). Tier 2: Supadata API (paid). Tier 3: faster-whisper offline (download + transcribe).

---

### YouTube Processor
**Files:** `backend/processors/youtube.py`  
**Related:** Three-Tier Transcript, Lens  
**Added:** Sprint YouTube

YouTubeProcessor class (859 lines). Transcript acquisition, channel video fetching, ID resolution. Three-tier fallback.

---

### Bilingual Storage
**Files:** `backend/migrations.py`, `frontend/youtube.js`  
**Related:** CJK Support  
**Added:** Sprint YouTube

video_insights.language column (migration 023). Multiple languages per lens. youtube_videos.transcript_language tracks detected language. UI toggle.

---

### CJK Support
**Files:** `backend/processors/lenses.py`, `frontend/youtube.js`  
**Related:** Bilingual Storage  
**Added:** Sprint YouTube

Chinese/Japanese/Korean transcript and extraction. Language auto-detection, target_language selector (Auto/EN/AR/JA/KO/ZH/FR/DE/ES/RU).

---

### Narration Source Verification
**Files:** `backend/routes/youtube_endpoints.py`  
**Related:** Narration Badges  
**Added:** Sprint YouTube

3-stage: pattern matching→Serper search→confidence scoring. Cached in narration_sources table. Religion-agnostic.

---

### Narration Badges
**Files:** `frontend/youtube.js`  
**Related:** Narration Source Verification  
**Added:** Sprint YouTube

UI badges showing verification status and clickable source links (sunnah.com, Google Scholar, etc.).

---

### Merge Dedup Keys
**Files:** `backend/routes/youtube_endpoints.py`  
**Related:** Lens Extraction  
**Added:** Sprint YouTube

'merge' mode dedup keys: eloquence→term, narrations→narration_text, history→event, spiritual→lesson, politics→topic.

---

### Eloquence Rarity
**Files:** `backend/processors/lenses.py`, `frontend/youtube.js`  
**Related:** Lens  
**Added:** Sprint YouTube

Rarity classification (uncommon/rare) for vocabulary terms. Client-side filter bar (All/Uncommon/Rare).

---

### Insights Modal
**Files:** `frontend/youtube.js`  
**Related:** Lens, Size Modes  
**Added:** Sprint YouTube

Theme-aware modal with 7 lens tabs. Size modes (sm/normal/lg). CSS variables for theming. Polling updates every 3s.

---

### Extract All
**Files:** `backend/routes/youtube_endpoints.py`  
**Related:** Lens Extraction  
**Added:** Sprint YouTube

POST /api/youtube/extract-all/:channel_id — bulk extraction for all videos. Background thread, skips existing.


## 💾 Database

### WAL Mode
**Files:** `backend/database.py`  
**Related:** Database Class  
**Added:** Sprint 1

Write-Ahead Logging on SQLite. Allows concurrent reads during writes. Set in database.py _make_conn().

---

### Database Class
**Files:** `backend/database.py`  
**Related:** WAL Mode, schema_version  
**Added:** Sprint 1

Singleton SQLite manager (645 lines). Thread-safe via threading.local(). _commit() uses Python write lock + busy_timeout 10s.

---

### schema_version
**Files:** `backend/migrations.py`  
**Related:** Migrations System  
**Added:** Sprint 1

Single-row table tracking migration count. 25 migrations total. Checked on startup.

---

### Migrations System
**Files:** `backend/migrations.py`  
**Related:** schema_version  
**Added:** Sprint 1

@migration decorator appends to MIGRATIONS list. run_migrations() applies pending. Each runs exactly once.

---

### news_items
**Files:** `backend/database.py`  
**Related:** user_feedback  
**Added:** Migration 001

Core article storage. id (URL hash), title, url, score, score_reason, profile_id. Indexed by profile+fetched+score.

---

### user_feedback
**Files:** `backend/database.py`  
**Related:** news_items, Distillation  
**Added:** Migration 001

User ratings and dismiss actions. Used for training export. ai_score, user_score, action, profile_id.

---

### conversations
**Files:** `backend/migrations.py`  
**Related:** Conversation Management  
**Added:** Migration 020

DB-backed chat history. profile_id, persona, title, messages (JSON), is_active, archived. Migration 020.

---

### scenarios
**Files:** `backend/migrations.py`  
**Related:** Scenario  
**Added:** Migration 021

Gaming scenario metadata. profile_id, name, genre, state_md, world_md, characters_json, is_active. Migration 021.

---

### persona_entities
**Files:** `backend/migrations.py`  
**Related:** Entity, Auto-Memory  
**Added:** Migration 022

Persona-generic entity persistence. identity_md, personality_md, speaking_style_md, memory_md, etc. Migration 022.

---

### video_insights
**Files:** `backend/migrations.py`  
**Related:** Lens Extraction  
**Added:** Migration 016

Lens extraction results. video_id, lens_name, language, content (JSON). Migration 016 + language in 023.

---

### narration_sources
**Files:** `backend/migrations.py`  
**Related:** Narration Source Verification  
**Added:** Migration 024

Cached resolved source URLs. narration_hash, resolved_url, resolution_method, confidence. Migration 024.

---

### sprint_log
**Files:** `backend/migrations.py`  
**Related:** Sprint Prompt Builder  
**Added:** Migration 025

Sprint prompt builder history. sprint_number, sprint_name, owner, generated_prompt, status. Migration 025.

---

### prompt_templates
**Files:** `backend/migrations.py`  
**Related:** Sprint Prompt Builder  
**Added:** Migration 025

Saved prompt builder form templates. profile_id, name, template_json. UNIQUE(profile_id, name). Migration 025.

---

### profiles
**Files:** `backend/database.py`  
**Related:** Config Overlay System  
**Added:** Migration 002

User profiles. config_overlay (JSON), ui_state (JSON with nested ui_settings). Migration 002.

---

### sessions
**Files:** `backend/auth.py`  
**Related:** Auth Middleware  
**Added:** Migration 002

Auth session tokens. token, user_id, profile_id, device_id, expires_at. Migration 002.


## 🖥️ Frontend Modules

### agent.js
**Files:** `frontend/agent.js`  
**Related:** Agent Chat, Fullscreen Agent  
**Added:** Sprint 5K

Agent chat core (~1,877 lines). Streaming, tool calls, conversation tabs, persona switching, TTS, fullscreen. Largest frontend module.

---

### ui-sync.js
**Files:** `frontend/ui-sync.js`  
**Related:** UI State Persistence  
**Added:** Sprint 9

localStorage↔DB sync. Patches setItem/removeItem to debounce-sync to /api/ui-state. Initial migration, sendBeacon flush, _uiSyncSuppressed flag.

---

### youtube.js
**Files:** `frontend/youtube.js`  
**Related:** Insights Modal, Lens  
**Added:** Sprint YouTube

YouTube UI (~1,327 lines). Channels, videos, 7-tab insights modal, polling, narration badges, size modes, language toggle, export.

---

### wizard.js
**Files:** `frontend/wizard.js`  
**Related:** wizard-data.js  
**Added:** Sprint 10

7-stage onboarding (2,172 lines). Priorities→details→feed→suggestions→generation→review→complete. Ollama-powered.

---

### wizard-data.js
**Files:** `frontend/wizard-data.js`  
**Related:** wizard.js  
**Added:** Sprint 11

Wizard static data (487 lines). CATS, ROLE_KEYWORDS, DOMAIN_PILLS, PANELS. Split from wizard.js Sprint 11.

---

### file-browser.js
**Files:** `frontend/file-browser.js`  
**Related:** Persona Context Isolation  
**Added:** Sprint 8

Nautilus-style modal file explorer (1,012 lines). Browse, edit, create, delete. Markdown toolbar. Per-persona isolation.

---

### theme-editor.js
**Files:** `frontend/theme-editor.js`  
**Related:** Theme System  
**Added:** Sprint 9

Real-time CSS variable tweaker (1,068 lines). Color pickers, presets. stratos-theme-custom-{theme} storage.

---

### games-ui.js
**Files:** `frontend/games-ui.js`  
**Related:** Scenario  
**Added:** Sprint 7

Gaming scenario UI. Scenario selector, genre picker, generation progress, entity management.

---

### fullscreen-chart.js
**Files:** `frontend/fullscreen-chart.js`  
**Related:** market.js  
**Added:** Sprint 9

Full-page chart viewer (2,432 lines). Largest file. TradingView charts, drawing tools, Fibonacci, auto-refresh.

---

### prompt-builder.js
**Files:** `frontend/prompt-builder.js`  
**Related:** Sprint Prompt Builder  
**Added:** Sprint Codex

Sprint Prompt Builder UI (559 lines). Build/History/Templates tabs. Auto-populates from /api/dev/context.

---

### sw.js
**Files:** `frontend/sw.js`  
**Added:** Sprint 1

Service Worker for PWA caching. Network-first for API. Cache version v13.


## 🔌 API Endpoints

### POST /api/agent-chat
**Files:** `backend/routes/agent.py`  
**Related:** Agent Chat  
**Added:** Sprint 5K

Stream chat response from Ollama. SSE format. Tool calls, persona switching, conversation persistence.

---

### GET /api/events
**Files:** `backend/server.py`, `backend/sse.py`  
**Related:** SSE Event System  
**Added:** Sprint 1

SSE stream. Events: scan_progress, scan_complete, briefing_ready, lens_extracted, youtube_status. Profile-scoped.

---

### GET /api/health
**Files:** `backend/routes/data_endpoints.py`  
**Added:** Sprint 1

Server health. Returns uptime, ollama_status, scorer_model, db_size, last_scan, memory_usage.

---

### POST /api/tts
**Files:** `backend/routes/media.py`  
**Related:** Dual TTS  
**Added:** Sprint 8

Text-to-speech. Kokoro for 8 langs, Edge-TTS for Arabic. Returns WAV/MP3. Max 5,000 chars.

---

### POST /api/stt
**Files:** `backend/routes/media.py`  
**Related:** STT  
**Added:** Sprint 8

Speech-to-text. Raw audio body. Returns {text, language}. faster-whisper CPU int8.

---

### GET /api/dev/context
**Files:** `backend/routes/dev_endpoints.py`  
**Related:** Sprint Prompt Builder  
**Added:** Sprint Codex

Live project state for Sprint Prompt Builder. Git log, file structure, DB tables, safety branches.

---

### POST /api/prompt-builder/generate
**Files:** `backend/routes/dev_endpoints.py`  
**Related:** Sprint Prompt Builder  
**Added:** Sprint Codex

Assemble sprint prompt from form data + live context. Template engine with 9 section builders.

---

### POST /api/youtube/extract-lens
**Files:** `backend/routes/youtube_endpoints.py`  
**Related:** Lens Extraction  
**Added:** Sprint YouTube

Async lens extraction. Daemon thread, polling 3s + SSE lens_extracted.

---

### POST /api/generate-profile
**Files:** `backend/routes/generate.py`  
**Added:** Sprint 3

AI generates categories + tickers from role/location. Ollama think=False for speed.


## ⚙️ Infrastructure

### Ollama
**Files:** `backend/main.py`  
**Related:** qwen3.5:9b, VRAM Budget  
**Added:** Sprint 1

Local LLM server. Hosts scorer (stratos-scorer-v2.2, ~9.5GB) + inference (qwen3.5:9b, ~6.6GB). OLLAMA_MAX_LOADED_MODELS=2.

---

### qwen3.5:9b
**Files:** `backend/main.py`  
**Related:** Ollama  
**Added:** Sprint 1

Primary inference model (~6.6GB). Agent chat, wizard, briefings, profile gen. All non-scoring tasks.

---

### VRAM Budget
**Files:** `backend/main.py`  
**Related:** Ollama, Kokoro  
**Added:** Sprint 1

24GB (AMD 7900 XTX). Scorer (~9.5GB) + qwen3.5:9b (~6.6GB) = ~16.1GB. Kokoro forced to CPU.

---

### ROCm
**Files:** `backend/train_lora.py`  
**Related:** VRAM Budget  
**Added:** Sprint 1

AMD GPU compute. AOTRITON_ENABLE_EXPERIMENTAL must NOT be set (NaN gradients). ROCR_VISIBLE_DEVICES=0 prevents iGPU.

---

### Kokoro
**Files:** `backend/processors/tts.py`  
**Related:** Dual TTS, Edge-TTS  
**Added:** Sprint 8

Kokoro-82M local TTS. 54 voices, 8 languages. Forced to CPU (20× faster than ROCm MIOpen). LRU cache max 3.

---

### Edge-TTS
**Files:** `backend/processors/tts.py`  
**Related:** Dual TTS, Kokoro  
**Added:** Sprint 8

Microsoft neural cloud TTS. 26 Arabic dialect voices. Auto-routed when text is Arabic.

---

### Dual TTS
**Files:** `backend/processors/tts.py`  
**Related:** Kokoro, Edge-TTS  
**Added:** Sprint 8

Two-engine: Kokoro (local, 8 langs) + Edge-TTS (cloud, Arabic). TTSProcessor auto-routes by language. 15s timeout, 5K char max.

---

### STT
**Files:** `backend/processors/stt.py`, `frontend/stt.js`  
**Related:** Dual TTS  
**Added:** Sprint 8

faster-whisper large-v3-turbo, CPU int8, lazy-loaded (~10s, ~1.6GB RAM). PyAV for audio decoding.

---

### Serper
**Files:** `backend/fetchers/serper_search.py`  
**Related:** SearXNG  
**Added:** Sprint 1

Serper.dev Google search API. Primary search provider. Agent web_search, narration verification, franchise resolution.

---

### SearXNG
**Files:** `backend/fetchers/searxng_search.py`  
**Related:** Serper  
**Added:** Sprint 2

Self-hosted metasearch engine. Alternative search provider.

---

### Session Token
**Files:** `backend/auth.py`, `backend/server.py`  
**Related:** Auth Middleware  
**Added:** Sprint 3

SHA-256 token in sessions table. X-Auth-Token header or ?token= query param. Validated on every API call.

---

### Autopilot
**Files:** `backend/autopilot.py`  
**Related:** Distillation  
**Added:** Sprint 2

Autonomous loop (959 lines). Cycles profiles, generates context via Opus, scans, distills, trains. Budget management.


## 🔧 Config Keys

### scoring.model
**Files:** `backend/main.py`, `backend/config.yaml`  
**Related:** scoring.inference_model  
**Added:** Sprint 1

Ollama scoring model name (e.g. 'stratos-scorer-v2.2'). Fine-tuned LoRA.

---

### scoring.inference_model
**Files:** `backend/main.py`, `backend/config.yaml`  
**Related:** scoring.model  
**Added:** Sprint 1

Ollama model for non-scoring tasks. Default: qwen3.5:9b.

---

### scoring.ollama_host
**Files:** `backend/main.py`  
**Related:** Ollama  
**Added:** Sprint 1

Ollama server URL. Default: http://localhost:11434.

---

### scoring.critical_min / high_min / medium_min
**Files:** `backend/processors/scorer_base.py`  
**Related:** Forbidden 5.0  
**Added:** Sprint 1

Score thresholds: critical ≥9.0, high ≥7.0, medium ≥5.0. Below = noise.

---

### search.provider
**Files:** `backend/config.yaml`, `frontend/settings.js`  
**Related:** Serper, SearXNG  
**Added:** Sprint 1

Search API: 'serper', 'google', 'searxng', or 'duckduckgo'.

---

### schedule.background_enabled
**Files:** `backend/main.py`, `backend/config.yaml`  
**Related:** Background Scheduler  
**Added:** Sprint 2

Enable background scan scheduler. Default: false.

---

### news.timelimit
**Files:** `backend/config.yaml`  
**Added:** Sprint 1

Search time window: 'd' (day), 'w' (week), 'm' (month).

---

### scoring.retain_high_scores
**Files:** `backend/main.py`, `backend/config.yaml`  
**Related:** Incremental Scanning  
**Added:** Sprint 3

Persist articles ≥ threshold across scans for 24h. Max 20 items.


## 🎨 UI Systems

### Theme System
**Files:** `frontend/themes.css`, `frontend/theme-editor.js`  
**Related:** Theme Editor, Perf Mode  
**Added:** Sprint 1

8 themes × 4 modes. CSS variables in themes.css, editable via theme-editor.js.

---

### Theme Editor
**Files:** `frontend/theme-editor.js`  
**Related:** Theme System  
**Added:** Sprint 9

Real-time CSS variable tweaker. Color pickers, presets. Storage: stratos-theme-custom-{theme}.

---

### Perf Mode
**Files:** `frontend/styles.css`  
**Related:** Theme System  
**Added:** Sprint 4

body.perf-mode disables: backdrop-filter, animations, transitions, box-shadow. Essential animations preserved.

---

### UI State Persistence
**Files:** `frontend/ui-sync.js`, `backend/routes/data_endpoints.py`  
**Related:** profiles  
**Added:** Sprint 9

ui-sync.js patches localStorage to auto-sync ~65+ keys to DB. Deep merge, sendBeacon flush, suppress flag.

---

### Star Engine
**Files:** `frontend/ui.js`  
**Related:** Theme System  
**Added:** Sprint 4

Animated starfield background. Generation counter prevents memory leaks. Theme-colored via --accent.

---

### Accent Glow Scrollbar
**Files:** `frontend/styles.css`  
**Related:** Theme System  
**Added:** Sprint 10

Global scrollbar using color-mix(var(--accent)). 5px, theme-adaptive. Hover 65%, active 80%.

---

### Onboarding Tour
**Files:** `frontend/tour.js`  
**Added:** Sprint 4

Interactive guided tour (830 lines). Step overlay highlighting UI elements. stratos_tour_never to skip.

---

### Mobile Gestures
**Files:** `frontend/mobile.js`  
**Added:** Sprint Mobile

Swipe sidebar, swipe cards, pull-to-refresh, bottom nav, PWA install. 860 lines.

---

### Size Modes
**Files:** `frontend/youtube.js`
**Related:** Insights Modal
**Added:** Sprint YouTube

YouTube insights modal: sm/normal/lg. CSS classes yi-size-sm/yi-size-lg. localStorage stratos_yi_size.

---

### Intelligence Hue
**Files:** `backend/routes/behavioral.py`, `frontend/hue.js`, `frontend/sibyl.js`
**Related:** AdaptiveScorer, Sibyl Theme
**Added:** Sprint Sibyl

Behavioral analysis system. Backend /api/hue endpoint computes a composite intelligence "hue" from scored news signals. hue.js renders a collapsible sidebar widget. sibyl.js provides the fullscreen Sibyl panel with expanded analytics.

---

### Sibyl Login Theme
**Files:** `frontend/auth-star-canvas.js`
**Related:** Theme System, Star Engine
**Added:** Sprint Sibyl

Brain neural network animation on the login page. Pre-computed outline vertices with animated pulse connections. Renders on the auth-star-canvas behind the login form.

---

### Sibyl In-App Theme
**Files:** `frontend/ui.js`, `frontend/theme-editor.js`
**Related:** Theme System, Theme Editor, Star Engine
**Added:** Sprint Sibyl

Anatomical brain visual as theme element in ui.js. Theme editor exposes position, scale, opacity, and glow sliders (glow is Sibyl-only). Default: cx=0.33, cy=0.06, scale=0.3, glow=3.0, opacity=1.0.

---

### Per-Theme Element Defaults
**Files:** `frontend/ui.js`, `frontend/theme-editor.js`
**Related:** Theme System, Theme Editor
**Added:** Sprint Defaults

Lookup table (_themeElementDefaults) maps each of 9 themes to optimal cx/cy/scale/glow/opacity defaults. Replaces hardcoded 0.5/0.35 fallbacks. Reset button snaps to per-theme defaults instead of center.

---

### New User Defaults
**Files:** `frontend/app.js`, `frontend/theme-editor.js`, `frontend/ui.js`, `frontend/nav.js`, `frontend/styles.css`
**Related:** Theme Editor, Star Engine, Font Size System
**Added:** Sprint Defaults

Font sizes renamed to 3-tier (Compact/Small/Medium). Old Extra Large = new Medium (default). Theme editor defaults: card opacity 0.45, panel opacity 0.2, border radius 25px, blur 2px. Stars enabled by default with density 2.5x and brightness 1.8x. Categories collapsed by default with glowing chevron indicator. Legacy migration via stratos_fontsize_v2 flag.


## 📖 Glossary

### Token Budget
**Files:** `backend/routes/personas.py`  
**Related:** Context Packing, Selective Loading  
**Added:** Sprint 5K

Max token count for context: 16K for persona (word×1.3), 4K for selective scenario loading.

---

### Scan Cycle
**Files:** `backend/main.py`  
**Related:** Background Scheduler  
**Added:** Sprint 1

Full news + market refresh. Fetch→score→store→broadcast scan_complete. Manual (/api/refresh) or scheduled.

---

### Preference Signal
**Files:** `backend/migrations.py`  
**Added:** Migration 018

User interaction signals in user_preference_signals. persona_source, signal_type, signal_key, weight. Cross-persona personalization.

---

### Extraction Confidence
**Files:** `backend/routes/youtube_endpoints.py`  
**Related:** Narration Source Verification  
**Added:** Sprint YouTube

0.0–1.0 for narration verification. Pattern match: 0.9, search: 0.7, none: 0.0.

---

### Sprint Prompt Builder
**Files:** `frontend/prompt-builder.js`, `backend/routes/dev_endpoints.py`  
**Related:** sprint_log, prompt_templates  
**Added:** Sprint Codex

Dev tool assembling sprint prompts with live filesystem context. Form in prompt-builder.js, engine in dev_endpoints.py.

---

### Holdout Evaluation
**Files:** `backend/evaluate_scorer.py`  
**Related:** AdaptiveScorer  
**Added:** Sprint 2

Reserved test set for scorer accuracy. evaluate_scorer.py runs predictions and reports MAE.

---

### Training/Inference Alignment
**Files:** `backend/export_training.py`, `backend/processors/scorer_adaptive.py`  
**Related:** SCORE:X.X|REASON: Format  
**Added:** Sprint 1

Critical: export_training.py must exactly match scorer_adaptive.py format. Misalignment = model failure.

---

### Context Hash
**Files:** `backend/main.py`  
**Related:** Profile Isolation  
**Added:** Sprint 3

SHA-256 of role|context|location. Profile-scoped retention. Changed role = new hash = rescore everything.

---

### Modelfile.v22
**Files:** `backend/Modelfile.v22`  
**Related:** AdaptiveScorer  
**Added:** Sprint 1

Ollama model definition for scorer. FROM, PARAMETER, SYSTEM prompt. DO NOT TOUCH.

---

### num_predict
**Files:** `backend/routes/agent.py`  
**Related:** Free Length Toggle  
**Added:** Sprint 1

Ollama max output tokens. Scoring: ~200. Short mode: 1500. Long mode: 8000. Per-call.

