# StratOS Codebase Codex

> Generated: 2026-03-13 | Version: 1.0

**14 categories, 151 terms**

## Table of Contents
- [🏗️ Architecture](#architecture) (11 terms)
- [🎭 Personas](#personas) (9 terms)
- [🤖 Agent System](#agent-system) (10 terms)
- [🎮 Gaming](#gaming) (10 terms)
- [🎯 Scoring Pipeline](#scoring-pipeline) (11 terms)
- [🎥 YouTube Pipeline](#youtube-pipeline) (9 terms)
- [🗄️ Database](#database) (15 terms)
- [💻 Frontend Modules](#frontend-modules) (16 terms)
- [🔌 API Endpoints](#api-endpoints) (10 terms)
- [⚙️ Infrastructure](#infrastructure) (13 terms)
- [🔧 Config Keys](#config-keys) (9 terms)
- [📖 Glossary](#glossary) (16 terms)
- [📡 Data Fetchers](#data-fetchers) (5 terms)
- [🧠 Training Pipeline](#training-pipeline) (7 terms)

---

## 🏗️ Architecture

### Dispatch Dict

Sequential route dispatch in server.py. CORSHandler's do_GET/do_POST call handle_get()/handle_post() on each route module in order; the first module to return True wins. Replaced a 2,860-line if/elif chain in Sprint 5K.

**Files:** `backend/server.py`
**Related:** Route Module, CORSHandler
**Added in:** Sprint 5K
**Type:** concept

### Route Module

A Python file in backend/routes/ that exports handle_get(), handle_post(), and optionally handle_delete(). Each module owns a domain (feeds, media, controls, etc.) and returns True if it handled the request, False to pass to the next module.

**Files:** `backend/routes/`
**Related:** Dispatch Dict, CORSHandler
**Added in:** Sprint 5K
**Type:** pattern

### CORSHandler

The main HTTP request handler class created inside server.py's create_handler() closure. Inherits SimpleHTTPRequestHandler, adds CORS headers, auth enforcement, rate limiting, profile isolation, SSE streaming, and gzip compression. Binds to strat (StratOS instance) and auth (AuthManager) via closure.

**Files:** `backend/server.py`
**Related:** ThreadedHTTPServer, Route Module, Auth Middleware
**Added in:** Sprint 4
**Type:** class

### ThreadedHTTPServer

ThreadingMixIn + HTTPServer with daemon_threads=True. Every incoming request gets its own thread, enabling concurrent SSE streams and long-running API calls without blocking.

**Files:** `backend/server.py`
**Related:** CORSHandler
**Added in:** Sprint 4
**Type:** class

### StratOS Class

The central orchestrator in main.py. Manages config loading, scan pipeline, background scheduler, market fetcher, scorer, database, SSE broadcasting, profile caching, and output file writing. Instantiated once at startup.

**Files:** `backend/main.py`
**Related:** Database, AdaptiveScorer, SSEManager
**Added in:** Sprint 1
**Type:** class

### SSE Event System

Server-Sent Events via /api/events. SSEManager tracks connected clients with optional profile_id scoping. Broadcasts scan progress, briefing_ready, youtube_processing, lens_extracted, extract_all_complete, and shutdown events in real-time.

**Files:** `backend/sse.py`, `backend/server.py`
**Related:** SSEManager, Profile Isolation
**Added in:** Sprint 4
**Type:** concept

### SSEManager

Thread-safe SSE client registry in sse.py. register()/unregister() track (wfile, profile_id) tuples. broadcast() sends JSON events to matching clients, auto-removing dead connections.

**Files:** `backend/sse.py`
**Related:** SSE Event System
**Added in:** Sprint 4
**Type:** class

### Auth Middleware

Inline auth enforcement in CORSHandler's do_GET/do_POST. Checks X-Auth-Token header against both file-based sessions (.sessions.json) and DB-backed sessions (sessions table). Resolves profile_id per-request for thread-safe data isolation.

**Files:** `backend/server.py`, `backend/auth.py`
**Related:** AuthManager, Profile Isolation, Session Token
**Added in:** Sprint 4
**Type:** concept

### Config Overlay

Per-profile configuration stored as JSON in profiles.config_overlay. Merged on top of the base config.yaml at login. Allows each user to have different tickers, categories, and feed toggles without modifying the shared config file.

**Files:** `backend/auth.py`, `backend/routes/config.py`
**Related:** Profile Isolation, AuthManager
**Added in:** Sprint 8
**Type:** concept

### Gzip Cache

In-memory gzip cache for static file serving in server.py. Keyed by filepath, stores (mtime, compressed_bytes). Only compresses .js/.css/.html/.json/.svg/.txt/.xml files when the client sends Accept-Encoding: gzip.

**Files:** `backend/server.py`
**Related:** CORSHandler
**Added in:** Sprint 5K
**Type:** optimization

### Graceful Shutdown

SIGINT/SIGTERM handler in start_server() that cancels in-progress scans, waits up to 10s for completion, checkpoints WAL, notifies SSE clients, and shuts down the HTTP server. Has a 15s watchdog timer to force exit if shutdown hangs.

**Files:** `backend/server.py`
**Related:** WAL Mode, SSE Event System
**Added in:** Sprint 5K
**Type:** concept

---

## 🎭 Personas

### Persona

A named agent mode (intelligence, market, scholarly, gaming, anime, tcg) that defines a system prompt, tool set, context builder, and greeting. Adding a persona requires only a config entry and context builder function in personas.py.

**Files:** `backend/routes/personas.py`, `backend/routes/persona_prompts.py`
**Related:** Context Packing, Persona Tools, Persona Prompt
**Added in:** Sprint 5K
**Type:** concept

### Intelligence Persona

Default persona. News-focused agent with tools: web_search, search_feed, manage_categories, search_files, read_document. Context includes scored news feed, briefing alerts, category stats, and scan history.

**Files:** `backend/routes/persona_prompts.py`, `backend/routes/personas.py`
**Related:** Persona, Context Packing
**Added in:** Sprint 5K
**Type:** persona

### Market Persona

Finance-focused agent with tools: manage_watchlist, search_feed, web_search. Context includes full market data across all timeframes, finance-tagged news, and market briefing summary.

**Files:** `backend/routes/persona_prompts.py`, `backend/routes/personas.py`
**Related:** Persona, Watchlist
**Added in:** Sprint 5K
**Type:** persona

### Scholarly Persona

Research assistant with YouTube lecture insights. Tools: search_insights, list_channels, get_video_summary, search_narrations, search_files, read_document, read_url, web_search. Context includes tracked channels, recent insights, and user context.

**Files:** `backend/routes/persona_prompts.py`, `backend/routes/personas.py`
**Related:** Persona, YouTube Pipeline, Narration Verification
**Added in:** Sprint 5K
**Type:** persona

### Gaming Persona

Game Master engine with GM mode (third-person narration, stat blocks, choices) and Immersive mode (in-character RP). Tools: search_files, read_document, import_canon_world. Uses selective file-based context loading.

**Files:** `backend/routes/persona_prompts.py`, `backend/routes/personas.py`
**Related:** Persona, Selective Context Loading, Scenario, Canon Import, Immersive RP Mode
**Added in:** Sprint 5K
**Type:** persona

### Context Packing

_pack_context() in personas.py builds per-persona context within a 16K token budget (word count x 1.3 estimation). Universal sections (profile, custom instructions, preferences) always included. Persona-specific sections fill remaining budget. Truncation at section boundaries.

**Files:** `backend/routes/personas.py`
**Related:** Persona, Token Budget
**Added in:** Sprint 5K Phase 4
**Type:** algorithm

### Persona Tools

PERSONA_TOOLS dict in personas.py maps each persona name to a list of allowed tool names. Only listed tools are sent to Ollama's tool definitions for that persona. E.g., gaming gets search_files + import_canon_world but not web_search.

**Files:** `backend/routes/personas.py`
**Related:** Persona, Agent Tools
**Added in:** Sprint 5K
**Type:** config

### Persona Context Manager

PersonaContextManager in processors/persona_context.py. Manages per-persona, per-profile context directory and database entries. Supports file CRUD, versioning, and system_context editing. Directory: data/users/{profile_id}/context/{persona_name}/

**Files:** `backend/processors/persona_context.py`
**Related:** Persona, File Browser
**Added in:** Sprint 6
**Type:** class

### Preference Signals

user_preference_signals table stores cross-persona feedback loops. Signal types include topic interests, language preferences, and content style. Auto-generated from user interactions or manually set. Injected into context packing.

**Files:** `backend/routes/persona_data.py`, `backend/routes/personas.py`
**Related:** Context Packing, Persona
**Added in:** Sprint 6
**Type:** concept

---

## 🤖 Agent System

### Agent Chat

Streaming chat endpoint at POST /api/agent-chat. Sends messages to Ollama /api/chat with tool definitions, streams SSE response tokens, handles tool calls inline, and supports cancellation. Uses qwen3.5:9b inference model.

**Files:** `backend/routes/agent.py`
**Related:** Agent Tools, Persona, Streaming SSE
**Added in:** Sprint 3
**Type:** endpoint

### Agent Tools

AGENT_TOOLS list in agent_tools.py defines 11 Ollama-format tool definitions. execute_tool() dispatches by name. Tools: web_search, search_feed, manage_watchlist, manage_categories, search_files, read_document, search_insights, list_channels, get_video_summary, search_narrations, read_url, import_canon_world.

**Files:** `backend/routes/agent_tools.py`
**Related:** Agent Chat, Persona Tools
**Added in:** Sprint 5K
**Type:** module

### execute_tool

Dispatcher function in agent_tools.py that receives a tool name, args dict, strat instance, profile_id, and persona. Routes to the appropriate implementation. Returns a string result that gets injected back into the LLM conversation as a tool response.

**Files:** `backend/routes/agent_tools.py`
**Related:** Agent Tools
**Added in:** Sprint 5K
**Type:** function

### parse_text_tool_calls

Fallback parser in agent_tools.py that extracts tool calls from plain text when the LLM doesn't use the structured tool_calls format. Regex-based, handles various formatting styles.

**Files:** `backend/routes/agent_tools.py`
**Related:** Agent Tools, execute_tool
**Added in:** Sprint 5K
**Type:** function

### strip_think_blocks

Utility in helpers.py that removes <think>...</think> blocks from LLM output. Also handles the case where the opening tag is missing but </think> is present. Called as a safety net on all inference output.

**Files:** `backend/routes/helpers.py`
**Related:** strip_reasoning_preamble
**Added in:** Sprint 5K
**Type:** function

### strip_reasoning_preamble

Aggressive reasoning detector in helpers.py. Identifies untagged internal reasoning (model narrating thought process as plain text) using regex patterns for reasoning starts, mid-response reasoning, and transition phrases. Returns cleaned content or empty string to trigger retry.

**Files:** `backend/routes/helpers.py`
**Related:** strip_think_blocks
**Added in:** Sprint 9
**Type:** function

### Suggest Context

POST /api/suggest-context endpoint. Takes role + location, calls Ollama to generate a suggested tracking context string and tickers. Used by the onboarding wizard and autopilot.

**Files:** `backend/routes/agent.py`
**Related:** Profile Generator
**Added in:** Sprint 3
**Type:** endpoint

### Agent Warmup

POST /api/agent-warmup sends a minimal prompt to Ollama to pre-load the inference model into VRAM. Rate limited to 2 calls per 60 seconds. Prevents cold-start latency on first real agent chat.

**Files:** `backend/routes/controls.py`
**Related:** VRAM Budget
**Added in:** Sprint 8
**Type:** endpoint

### Dynamic Suggestions

Persona-aware suggestion chips generated by agent-suggestions.js. Each persona has context-appropriate starter prompts that adapt to available data. Displayed below the chat input area.

**Files:** `frontend/agent-suggestions.js`
**Related:** Persona, Agent Chat
**Added in:** Sprint 8
**Type:** feature

### Conversation Persistence

Chat history stored in conversations table. Each conversation has profile_id, persona, title, messages (JSON array), is_active flag. Max 10 per persona per profile; oldest auto-archived.

**Files:** `backend/routes/persona_data.py`
**Related:** Agent Chat
**Added in:** Sprint 7
**Type:** feature

---

## 🎮 Gaming

### Scenario

A game world stored as both a DB row (scenarios table) and a file-system directory tree. Contains world/, characters/, scenes/ directories with markdown files for setting, rules, NPCs, player stats, and current scene.

**Files:** `backend/processors/scenarios.py`, `backend/processors/scenario_templates.py`
**Related:** Scenario Generator, Selective Context Loading, Gaming Persona
**Added in:** Sprint 7
**Type:** concept

### Scenario Generator

LLM-powered content generation in scenario_generator.py. Runs 4 passes against Ollama: setting + rules, NPCs, starting scene, then index generation. Runs in background thread with progress_callback for real-time status.

**Files:** `backend/processors/scenario_generator.py`
**Related:** Scenario, Scenario Templates
**Added in:** Sprint 7
**Type:** module

### Scenario Templates

Defines the canonical folder structure for scenarios: _index.json, world/setting.md, world/rules.md, characters/_roster.json, characters/player/stats.md, scenes/current.md, plus NPC subdirectories. create_scenario_skeleton() builds this tree.

**Files:** `backend/processors/scenario_templates.py`
**Related:** Scenario, Scenario Generator
**Added in:** Sprint 7
**Type:** module

### Scenario Updater

Auto-updates scenario files after each AI response. Runs in background thread after response streaming completes. Updates current.md scene state, player stats, NPC memory, inventory. Never blocks user's next message.

**Files:** `backend/processors/scenario_updater.py`
**Related:** Scenario, Agent Chat
**Added in:** Sprint 7
**Type:** module

### Selective Context Loading

_pack_gaming_context_selective() in personas.py. Instead of loading one massive world.md, scans the user's message for keywords and loads only relevant files (inventory, quests, combat rules, mentioned NPCs). Budget: 4000 tokens.

**Files:** `backend/routes/personas.py`
**Related:** Context Packing, Scenario
**Added in:** Sprint 7
**Type:** algorithm

### Canon Import

Fetches franchise data from Fandom wikis to auto-populate scenario files. Resolves franchise names to wiki subdomains (50+ aliases: SAO, Naruto, Witcher, etc.), scrapes characters, locations, lore, power systems.

**Files:** `backend/processors/canon_import.py`
**Related:** Scenario, Gaming Persona
**Added in:** Sprint 8
**Type:** module

### Immersive RP Mode

Alternative gaming mode where the AI responds AS characters rather than as a Game Master. No stat boxes or numbered choices. Uses character personality, memory, and relationship data. Active NPC set via agent UI.

**Files:** `backend/routes/persona_prompts.py`, `backend/routes/personas.py`
**Related:** Gaming Persona, Persona Entities
**Added in:** Sprint 7
**Type:** mode

### Persona Entities

persona_entities table stores characters, NPCs, and figures per persona/scenario. Fields: identity_md, personality_md, speaking_style_md, relationship_md, memory_md, knowledge_md. Used for both gaming NPCs and scholarly figures.

**Files:** `backend/routes/persona_data.py`
**Related:** Scenario, Immersive RP Mode
**Added in:** Sprint 7
**Type:** table

### NPC Roster

characters/_roster.json in a scenario directory. Lists all NPCs with id, name, keywords (for mention detection), and active_in_scene flag. Used by selective context loading to auto-load NPC profiles when the user mentions their name.

**Files:** `backend/routes/personas.py`, `backend/processors/scenario_templates.py`
**Related:** Selective Context Loading, Scenario
**Added in:** Sprint 7
**Type:** file

### Keyword-Triggered Loading

Part of selective context loading. A map of keywords to file paths: 'inventory'/'backpack' loads player/inventory.md, 'quest'/'mission' loads player/quests.md, 'fight'/'attack' loads mechanics/combat.md + player/equipment.md.

**Files:** `backend/routes/personas.py`
**Related:** Selective Context Loading
**Added in:** Sprint 7
**Type:** algorithm

---

## 🎯 Scoring Pipeline

### AdaptiveScorer

The sole scorer since B3.3. Profile-adaptive: dynamically builds relevance rules from user's categories, keywords, role, and context. Hybrid: rule-based noise filters first, then Ollama LLM scoring for items that pass. 1,482 lines.

**Files:** `backend/processors/scorer_adaptive.py`
**Related:** ScorerBase, Scoring Memory, Forbidden 5.0
**Added in:** Sprint 2
**Type:** class

### ScorerBase

Base class with shared scoring infrastructure: Ollama client, calibration tables, noise pattern lists (500+ patterns), language filtering, ScoringMemory. 891 lines of shared constants and utilities.

**Files:** `backend/processors/scorer_base.py`
**Related:** AdaptiveScorer
**Added in:** Sprint 2
**Type:** class

### Scoring Memory

ScoringMemory class in scorer_base.py. Stores high-scoring article examples as few-shot context for the scorer's prompt. Cleared between autopilot cycles to prevent cross-profile contamination.

**Files:** `backend/processors/scorer_base.py`
**Related:** AdaptiveScorer, Autopilot
**Added in:** Sprint 3
**Type:** class

### SCORE:X.X|REASON: Format

The mandatory output format for the scoring LLM. Every response must contain SCORE:X.X|REASON:text. Parsed by regex. Training data format must match inference format character-for-character.

**Files:** `backend/processors/scorer_adaptive.py`, `backend/export_training.py`
**Related:** Prompt Version Pinning
**Added in:** Sprint 2
**Type:** format

### Forbidden 5.0

Score of exactly 5.0 is forbidden in scoring. The scorer is trained to give binary signal: above 5 = relevant, below 5 = noise. A score of 5.0 is ambiguous and indicates the model failed to commit.

**Files:** `backend/processors/scorer_adaptive.py`
**Related:** AdaptiveScorer
**Added in:** Sprint 3
**Type:** rule

### Shadow Scores

shadow_scores table stores side-by-side comparisons between the primary scorer and a shadow scorer. Used for validation: tracks delta between scorers, scan_id, title, category.

**Files:** `backend/database.py`
**Related:** AdaptiveScorer
**Added in:** Sprint 3
**Type:** table

### Distillation

Teacher-student distillation using Claude Opus. Sends locally-scored items to Opus for re-scoring. Disagreements (delta >= 2.0) saved as corrections in user_feedback. Drives the self-improvement loop.

**Files:** `backend/distill.py`
**Related:** Export Training, LoRA Training
**Added in:** Sprint 2
**Type:** process

### Export Training

export_training.py converts corrections + user feedback into ChatML JSONL training data. Critical: format must be character-for-character identical to inference format in scorer_adaptive.py.

**Files:** `backend/export_training.py`
**Related:** Distillation, LoRA Training, Prompt Version Pinning
**Added in:** Sprint 2
**Type:** script

### LoRA Training

train_lora.py implements LoRA/DoRA fine-tuning. Auto-selects model tier by VRAM. Supports Unsloth (CUDA) and PEFT (ROCm). Pipeline: train -> merge -> GGUF export -> Ollama register -> config update.

**Files:** `backend/train_lora.py`
**Related:** Distillation, Export Training, Model Manager
**Added in:** Sprint 2
**Type:** script

### Prompt Version Pinning

prompt_version.py hashes the scoring prompt template at both training and inference time. Logs WARNING if they drift. The #1 recurring failure mode: misaligned prompts cause trained models to fail at parsing.

**Files:** `backend/prompt_version.py`
**Related:** Export Training, AdaptiveScorer
**Added in:** Sprint 3
**Type:** module

### Incremental Scanning

_reuse_snapshot_scores() builds URL->score lookup from previous output. Already-scored articles skip LLM scoring. Articles not re-fetched are carried forward. Skipped if context hash changed (different role = rescore all).

**Files:** `backend/main.py`
**Related:** AdaptiveScorer, Context Hash
**Added in:** Sprint 4
**Type:** optimization

---

## 🎥 YouTube Pipeline

### YouTube Processor

YouTubeProcessor class in processors/youtube.py. Manages channel tracking, video discovery via RSS, transcript acquisition, and lens extraction orchestration. 859 lines.

**Files:** `backend/processors/youtube.py`
**Related:** YouTube Worker, Lens, Three-Tier Transcript
**Added in:** Sprint 6
**Type:** class

### Lens

A focused analysis prompt that extracts specific insight types from transcripts. 7 lenses: summary, eloquence, narrations, history, spiritual, politics, transcript. Each runs individually against qwen3.5:9b.

**Files:** `backend/processors/lenses.py`
**Related:** YouTube Processor, Lens Extraction
**Added in:** Sprint 6
**Type:** concept

### Three-Tier Transcript

Transcript acquisition cascade: Tier 1: youtube-transcript-api (free, instant, captioned videos). Tier 2: Supadata API (paid, wider coverage). Tier 3: faster-whisper local transcription (CPU, any video).

**Files:** `backend/processors/youtube.py`
**Related:** YouTube Processor
**Added in:** Sprint 6
**Type:** algorithm

### Bilingual Extraction

Migration 023 added language column to video_insights. Lenses can extract in the transcript's original language AND English. CJK support for Japanese, Chinese, Korean transcripts.

**Files:** `backend/processors/lenses.py`, `backend/migrations.py`
**Related:** Lens
**Added in:** Sprint 10
**Type:** feature

### YouTube Worker

Background daemon thread in youtube_worker.py. Processes video queue: transcription then lens extraction, one video at a time (Whisper CPU uses all cores). Started by start_server().

**Files:** `backend/processors/youtube_worker.py`
**Related:** YouTube Processor
**Added in:** Sprint 6
**Type:** module

### Narration Source Resolution

Two-stage cascade in source_resolver.py. Stage A: regex pattern matching against known hadith API endpoints (sync, free). Stage B: Serper web search with authority domain scoring (async, ~$0.001/query).

**Files:** `backend/processors/source_resolver.py`
**Related:** Narration Verification, narration_sources Table
**Added in:** Sprint 10
**Type:** module

### Narration Verification

Anti-hallucination RAG in verification.py. Verifies hadith, historical narrations, and scholarly citations detected by the narrations lens against real databases. The LLM detects narrations; this module verifies them.

**Files:** `backend/processors/verification.py`
**Related:** Narration Source Resolution, Scholarly Persona
**Added in:** Sprint 8
**Type:** module

### Lens Merge

_merge_lens_content() in youtube_endpoints.py. When re-extracting a lens, deduplicates items by lens-specific keys: eloquence by term, narrations by narration_text, history by event. Summary lens always full-replaces.

**Files:** `backend/routes/youtube_endpoints.py`
**Related:** Lens
**Added in:** Sprint 10
**Type:** function

### Extract All

POST /api/youtube/extract-all/:channel_id queues lens extraction for all transcribed videos in a channel. Runs in background, broadcasts extract_all_complete via SSE on completion.

**Files:** `backend/routes/youtube_endpoints.py`
**Related:** Lens, YouTube Worker
**Added in:** Sprint 10
**Type:** endpoint

---

## 🗄️ Database

### Database Class

SQLite manager in database.py. Per-thread connections via threading.local(). WAL mode, busy_timeout 10s, foreign keys ON. Singleton via get_database(). All table creation via migration framework.

**Files:** `backend/database.py`
**Related:** WAL Mode, Migration Framework
**Added in:** Sprint 1
**Type:** class

### WAL Mode

Write-Ahead Logging. Set via PRAGMA journal_mode=WAL. Enables concurrent reads during writes. Checkpointed on shutdown via PRAGMA wal_checkpoint(TRUNCATE).

**Files:** `backend/database.py`, `backend/server.py`
**Related:** Database Class
**Added in:** Sprint 1
**Type:** config

### Migration Framework

Numbered migrations in migrations.py. @migration decorator registers functions. schema_version table tracks progress. Safe to call on every startup. 25 migrations covering all tables.

**Files:** `backend/migrations.py`
**Related:** Database Class, schema_version
**Added in:** Sprint 4
**Type:** module

### news_items

Core table for scored news articles. Columns: id, title, url, summary, source, root, category, score, score_reason, timestamp, fetched_at, profile_id. UNIQUE(url, profile_id). Migration 001+008.

**Files:** `backend/migrations.py`, `backend/database.py`
**Related:** Profile Isolation
**Added in:** Migration 001
**Type:** table

### user_feedback

Training data table. Actions: click, dismiss, rate, save, thumbs_up, thumbs_down. Stores ai_score, user_score, profile provenance (role, location, context). Never cleaned -- permanent training data.

**Files:** `backend/migrations.py`, `backend/database.py`
**Related:** Distillation, Export Training
**Added in:** Migration 002
**Type:** table

### youtube_channels

Tracked YouTube channels per profile. Columns: channel_id, channel_name, channel_url, lenses (JSON array), last_checked. UNIQUE(profile_id, channel_id). Migration 016.

**Files:** `backend/migrations.py`
**Related:** YouTube Processor
**Added in:** Migration 016
**Type:** table

### youtube_videos

Processed videos. Columns: video_id, title, transcript_text, transcript_method, transcript_language, status (pending/transcribed/complete/failed). UNIQUE(profile_id, video_id). Migration 016+023.

**Files:** `backend/migrations.py`
**Related:** YouTube Processor, Three-Tier Transcript
**Added in:** Migration 016
**Type:** table

### video_insights

Extracted lens results per video. Columns: video_id, profile_id, lens_name, content (JSON), language. Indexed by (profile_id, lens_name) and (video_id, lens_name, language). Migration 016+023.

**Files:** `backend/migrations.py`
**Related:** Lens, Bilingual Extraction
**Added in:** Migration 016
**Type:** table

### conversations

DB-backed chat history. Columns: profile_id, persona, title, messages (JSON array), is_active, archived. Max 10 per persona per profile. Migration 020.

**Files:** `backend/migrations.py`
**Related:** Agent Chat, Conversation Persistence
**Added in:** Migration 020
**Type:** table

### scenarios

Game scenario storage. Columns: profile_id, name, description, state_md, world_md, characters_json, genre, is_active. UNIQUE(profile_id, name). Migration 021.

**Files:** `backend/migrations.py`
**Related:** Scenario
**Added in:** Migration 021
**Type:** table

### persona_entities

Persona-generic entity persistence. 13 markdown fields: identity, personality, speaking_style, relationship, memory, knowledge, extra. UNIQUE(profile_id, persona, scenario_name, name). Migration 022.

**Files:** `backend/migrations.py`
**Related:** Persona Entities, Immersive RP Mode
**Added in:** Migration 022
**Type:** table

### narration_sources

Cache for resolved source URLs from narration verification. Columns: video_id, narration_hash, source_claimed, resolved_url, resolution_method, confidence. UNIQUE(video_id, profile_id, narration_hash). Migration 024.

**Files:** `backend/migrations.py`
**Related:** Narration Source Resolution
**Added in:** Migration 024
**Type:** table

### sprint_log

Sprint Prompt Builder history. Columns: sprint_number, sprint_name, owner, generated_prompt, status. Migration 025.

**Files:** `backend/migrations.py`
**Related:** Prompt Builder
**Added in:** Migration 025
**Type:** table

### prompt_templates

Saved prompt templates for the Sprint Prompt Builder. Columns: profile_id, name, template_json. UNIQUE(profile_id, name). Migration 025.

**Files:** `backend/migrations.py`
**Related:** Prompt Builder
**Added in:** Migration 025
**Type:** table

### Profile Isolation

profile_id column on news_items, scan_log, user_feedback, shadow_scores, briefings (migration 008). Every query filters by profile_id. Sessions table links tokens to profile_id. Prevents data leakage between users.

**Files:** `backend/migrations.py`, `backend/server.py`
**Related:** Auth Middleware, Config Overlay
**Added in:** Migration 008
**Type:** concept

---

## 💻 Frontend Modules

### app.js

Main application state and rendering. 2,137 lines. Manages data/marketData/newsData state, renders news feed cards, market widgets, sidebar navigation, SSE connection, settings integration. Entry point for dashboard.

**Files:** `frontend/app.js`
**Related:** feed.js, ui.js
**Added in:** Sprint 1
**Type:** module

### agent.js

AI chat interface. 1,877 lines. Manages agentHistory, streaming responses, tool call rendering, persona switching, conversation tabs, fullscreen mode, TTS playback via speakMessage(). Global _currentTTSAudio tracks playback.

**Files:** `frontend/agent.js`
**Related:** agent-suggestions.js, agent-tickers.js, agent-customizer.js
**Added in:** Sprint 3
**Type:** module

### youtube.js

YouTube channel management and insights viewer. 1,327 lines. Channel list, video list, insights modal with lens tabs, narration badges, language selector, extract buttons, export functionality.

**Files:** `frontend/youtube.js`
**Related:** YouTube Pipeline
**Added in:** Sprint 6
**Type:** module

### wizard.js

Onboarding wizard. 2,172 lines. 7-stage flow: welcome, role/location, context questionnaire, AI-powered category generation, ticker selection, feed configuration, summary. Theme-adaptive CSS.

**Files:** `frontend/wizard.js`, `frontend/wizard-data.js`
**Related:** wizard-data.js
**Added in:** Sprint 4
**Type:** module

### settings.js

Configuration management UI. 1,770 lines. Profile editing, category management, ticker setup, feed toggles, API key inputs, TTS settings, theme selection. Tracks unsaved changes via _settingsDirty flag.

**Files:** `frontend/settings.js`
**Related:** settings-categories.js, settings-sources.js, settings-tickers.js
**Added in:** Sprint 2
**Type:** module

### ui.js

Theme system and UI utilities. 1,618 lines. 8 themes (midnight, noir, coffee, rose, cosmos, nebula, aurora, sakura), theme switching, performance mode, CSS variable management, nebula canvas.

**Files:** `frontend/ui.js`
**Related:** theme-editor.js, Performance Mode
**Added in:** Sprint 1
**Type:** module

### ui-sync.js

UI settings persistence to DB. 382 lines. Saves localStorage settings to /api/ui-state, reloads after login clears localStorage. Solves the profile-switch settings wipe problem.

**Files:** `frontend/ui-sync.js`
**Related:** settings.js
**Added in:** Sprint 9
**Type:** module

### file-browser.js

Nautilus-style modal file browser and editor. 1,012 lines. Directory tree, file content viewing/editing, create file/folder, delete. Uses /api/persona-files endpoints.

**Files:** `frontend/file-browser.js`
**Related:** Persona Context Manager
**Added in:** Sprint 7
**Type:** module

### games-ui.js

Scenario selector UI for Games persona. 440 lines. Lists scenarios, create/activate/delete, generation status polling, scenario detail display.

**Files:** `frontend/games-ui.js`
**Related:** Scenario, Gaming Persona
**Added in:** Sprint 7
**Type:** module

### prompt-builder.js

Sprint Prompt Builder UI. 559 lines. Form for configuring sprint parameters, phases, file ownership, rules. Calls /api/prompt-builder/generate, saves to sprint_log. Session B's module.

**Files:** `frontend/prompt-builder.js`
**Related:** dev_endpoints.py, sprint_log
**Added in:** Sprint 10
**Type:** module

### stt.js

Voice input via faster-whisper. 344 lines. Injects mic button into agent input via DOM. MediaRecorder captures audio, POSTs to /api/stt. Does NOT modify agent.js.

**Files:** `frontend/stt.js`
**Related:** STT Processor
**Added in:** Sprint 8
**Type:** module

### tts-settings.js

Voice picker, speed control, persona voice overrides, preview. 173 lines. Renders voice list from /api/tts/voices, handles preview playback, stores preferences.

**Files:** `frontend/tts-settings.js`
**Related:** TTS Processor
**Added in:** Sprint 8
**Type:** module

### theme-editor.js

Real-time CSS variable tweaker. 1,068 lines. Floating panel for customizing theme colors with live preview. Saves custom overrides to localStorage per base theme.

**Files:** `frontend/theme-editor.js`
**Related:** ui.js
**Added in:** Sprint 9
**Type:** module

### fullscreen-chart.js

Fullscreen chart mode using TradingView Lightweight Charts. 2,432 lines. Works on all devices. Multiple timeframes, auto-refresh via /api/market-tick, technical indicators.

**Files:** `frontend/fullscreen-chart.js`, `frontend/fullscreen-chart-utils.js`
**Related:** market.js
**Added in:** Sprint 5
**Type:** module

### auth.js

Landing/login/register UI. 839 lines. Email-based auth with verification codes, OTP login, password reset. Color theme rotation on page load. Star canvas background.

**Files:** `frontend/auth.js`, `frontend/auth-star-canvas.js`, `frontend/auth-styles.js`
**Related:** Auth Middleware
**Added in:** Sprint 4
**Type:** module

### tour.js

Guided tour system. 830 lines. Spotlight overlay with SVG mask cutout + positioned tooltip cards. Works across all 8 themes x 3 modes (24 combinations).

**Files:** `frontend/tour.js`
**Added in:** Sprint 9
**Type:** module

---

## 🔌 API Endpoints

### /api/agent-chat

POST. Streaming AI chat. Body: {message, persona, conversation_id, rp_mode, active_npc, use_all_scans}. SSE response: tokens, tool_call, tool_result, done events. Auth required.

**Files:** `backend/routes/agent.py`
**Related:** Agent Chat
**Added in:** Sprint 3
**Type:** endpoint

### /api/data

GET. Full dashboard data (~2MB). Returns news, market, briefing, timestamps, scan status. Gzip compressed. Profile-scoped via auth token. The main data endpoint.

**Files:** `backend/routes/data_endpoints.py`
**Added in:** Sprint 1
**Type:** endpoint

### /api/events

GET. SSE stream. Long-lived connection for real-time updates. Events: status (scan progress), briefing_ready, youtube_processing, lens_extracted, extract_all_complete, shutdown. Profile-scoped.

**Files:** `backend/server.py`
**Related:** SSE Event System
**Added in:** Sprint 4
**Type:** endpoint

### /api/refresh

GET. Triggers full scan (news + market + scoring + briefing) in background thread. Returns immediately with {status: refresh_triggered}. Auth exempt but rate limited (2/60s).

**Files:** `backend/routes/controls.py`
**Added in:** Sprint 1
**Type:** endpoint

### /api/tts

POST. Text-to-speech synthesis. Body: {text, voice, language, speed, persona}. Returns audio/wav (Kokoro) or audio/mpeg (Edge-TTS). Max 5,000 characters. Auto-routes Arabic to Edge-TTS.

**Files:** `backend/routes/media.py`
**Related:** TTS Processor
**Added in:** Sprint 8
**Type:** endpoint

### /api/stt

POST. Speech-to-text. Raw audio body (WebM/WAV/OGG). Optional X-Language-Hint header. Returns {text, language, duration, processing_time}. Uses faster-whisper large-v3-turbo on CPU.

**Files:** `backend/routes/media.py`
**Related:** STT Processor
**Added in:** Sprint 8
**Type:** endpoint

### /api/youtube/channels

GET: list tracked channels. POST: add channel (or single video). DELETE /api/youtube/channels/:id: remove channel. Handles both channel URLs and video URLs.

**Files:** `backend/routes/youtube_endpoints.py`
**Related:** YouTube Processor
**Added in:** Sprint 6
**Type:** endpoint

### /api/youtube/extract-lens

POST. On-demand lens extraction for a single video. Body: {video_id, lens, language, mode}. Modes: new, replace, merge. Runs in background thread, broadcasts lens_extracted via SSE.

**Files:** `backend/routes/youtube_endpoints.py`
**Related:** Lens
**Added in:** Sprint 10
**Type:** endpoint

### /api/dev/context

GET. Returns live project context for Sprint Prompt Builder: git log, file structure, DB tables, safety branches, pending items. Session B's endpoint.

**Files:** `backend/routes/dev_endpoints.py`
**Related:** Prompt Builder
**Added in:** Sprint 10
**Type:** endpoint

### /api/prompt-builder/generate

POST. Assembles a sprint prompt from form data + live filesystem context. Returns {prompt: string}. Template engine is string assembly, no LLM involved.

**Files:** `backend/routes/dev_endpoints.py`
**Related:** Prompt Builder
**Added in:** Sprint 10
**Type:** endpoint

---

## ⚙️ Infrastructure

### Ollama

Local LLM server. Hosts two models: scoring model (~9.5GB) + qwen3.5:9b (~6.6GB). Both fit in 24GB VRAM with OLLAMA_MAX_LOADED_MODELS=2. ROCm backend for AMD 7900 XTX.

**Files:** `backend/main.py`
**Related:** qwen3.5:9b, VRAM Budget
**Added in:** Sprint 1
**Type:** service

### qwen3.5:9b

The inference model used for agent chat, market analysis, wizard, briefings, profile generation, and lens extraction. ~6.6GB VRAM. Set via config.yaml scoring.inference_model.

**Files:** `backend/main.py`
**Related:** Ollama, VRAM Budget
**Added in:** Sprint 3
**Type:** model

### VRAM Budget

24GB total (AMD 7900 XTX). Scorer (~9.5GB) + qwen3.5:9b (~6.6GB) = ~16.1GB. Kokoro-82M TTS adds ~500MB on demand. OLLAMA_MAX_LOADED_MODELS=2 prevents model swapping.

**Related:** Ollama, qwen3.5:9b
**Added in:** Sprint 1
**Type:** constraint

### TTS Processor

Dual-engine TTS in processors/tts.py. KokoroEngine: 54 voices, 8 languages, GPU-accelerated, WAV output. EdgeTTSEngine: 26 Arabic dialect voices, Microsoft cloud, MP3 output. LRU pipeline cache (max 3), 15s timeout.

**Files:** `backend/processors/tts.py`
**Related:** Kokoro, Edge-TTS
**Added in:** Sprint 8
**Type:** class

### Kokoro

Kokoro-82M local TTS engine. 54 voices across en/ja/zh/fr/ko/hi/it/pt. ~500MB VRAM. WAV output. LRU pipeline cache keeps 3 most-recent language pipelines loaded.

**Files:** `backend/processors/tts.py`
**Related:** TTS Processor
**Added in:** Sprint 8
**Type:** engine

### Edge-TTS

Microsoft Neural TTS cloud service. 26 Arabic dialect voices. Zero VRAM. MP3 output. Used automatically for Arabic text via Unicode range detection.

**Files:** `backend/processors/tts.py`
**Related:** TTS Processor
**Added in:** Sprint 8
**Type:** engine

### STT Processor

Speech-to-text via faster-whisper large-v3-turbo on CPU (int8). ~1.6GB RAM. Lazy-loads on first request (~10s). Uses PyAV for audio decoding. Max 25MB audio.

**Files:** `backend/processors/stt.py`
**Added in:** Sprint 8
**Type:** class

### SearXNG

Self-hosted metasearch engine. Optional search provider configured via config.yaml search.searxng_host. Alternative to Serper/Google for web search tool.

**Files:** `backend/fetchers/searxng_search.py`
**Related:** Serper
**Added in:** Sprint 2
**Type:** service

### Serper

Google search API (serper.dev). Primary search provider for web_search tool and news fetching. Configured via config.yaml search.serper_api_key. ~$0.001 per query.

**Files:** `backend/fetchers/serper_search.py`
**Related:** SearXNG
**Added in:** Sprint 2
**Type:** service

### PaddleOCR

PaddleOCR-VL 0.9B via Ollama for image text extraction. ~1GB VRAM, loaded on demand. Used by file upload to extract text from images and PDFs.

**Files:** `backend/processors/ocr.py`
**Added in:** Sprint 7
**Type:** service

### Autopilot

Fully autonomous self-improvement loop. Picks diverse professional profiles, generates context via Opus, scans, distills, trains. Manages budget, persistent state, profile diversity guards. 17 profile templates.

**Files:** `backend/autopilot.py`
**Related:** Distillation, LoRA Training
**Added in:** Sprint 3
**Type:** script

### Model Manager

CLI tool for safe model switching with rollback. Commands: status, switch, rollback, history, validate, register. All changes logged to data/model_history.json.

**Files:** `backend/model_manager.py`
**Related:** LoRA Training
**Added in:** Sprint 4
**Type:** script

### Email Service

SMTP-based email delivery for verification codes, OTP login codes, and password resets. Gracefully degrades when SMTP is not configured. HTML-formatted emails.

**Files:** `backend/email_service.py`
**Related:** Auth Middleware
**Added in:** Sprint 8
**Type:** class

---

## 🔧 Config Keys

### scoring.model

Name of the Ollama scoring model. Default: stratos-scorer-v2.2. Changed by model_manager.py switch or train_lora.py after successful training.

**Files:** `backend/config.yaml`
**Related:** AdaptiveScorer, Model Manager
**Added in:** Sprint 1
**Type:** config

### scoring.inference_model

Model used for all non-scoring inference: agent chat, wizard, briefings, profile generation, lens extraction. Default: qwen3.5:9b.

**Files:** `backend/config.yaml`
**Related:** qwen3.5:9b
**Added in:** Sprint 3
**Type:** config

### scoring.ollama_host

URL of the Ollama server. Default: http://localhost:11434. All LLM calls go through this host.

**Files:** `backend/config.yaml`
**Related:** Ollama
**Added in:** Sprint 1
**Type:** config

### scoring.critical_min

Minimum score for 'critical' tier. Default: 9.0. Articles scoring >= this trigger critical alerts in briefings.

**Files:** `backend/config.yaml`
**Related:** AdaptiveScorer
**Added in:** Sprint 1
**Type:** config

### scoring.filter_below

Articles scoring below this are filtered from the feed. Default: 5.0. The noise floor.

**Files:** `backend/config.yaml`
**Related:** AdaptiveScorer, Forbidden 5.0
**Added in:** Sprint 1
**Type:** config

### scoring.retain_high_scores

Whether to carry forward high-scoring articles from previous scans. Default: false. When true, articles above retention_threshold are preserved across scans.

**Files:** `backend/config.yaml`
**Related:** Incremental Scanning
**Added in:** Sprint 4
**Type:** config

### search.provider

Search API provider. Values: 'serper', 'google', 'searxng'. Default: serper. Controls which backend web_search tool uses.

**Files:** `backend/config.yaml`
**Related:** Serper, SearXNG
**Added in:** Sprint 2
**Type:** config

### schedule.background_enabled

Enable automatic background scanning on an interval. Default: false. When true, runs scans every background_interval_minutes.

**Files:** `backend/config.yaml`
**Added in:** Sprint 3
**Type:** config

### dynamic_categories

Array of user-defined news categories, each with id, label, icon, items (keywords), scorer_type, root. Generated by AI via /api/generate-profile or manually configured.

**Files:** `backend/config.yaml`
**Related:** Profile Generator
**Added in:** Sprint 2
**Type:** config

---

## 📖 Glossary

### Scan

A complete intelligence cycle: fetch news from all sources, score each article via the scorer, generate a briefing, write output JSON, and broadcast completion via SSE. Triggered by /api/refresh or the background scheduler.

**Files:** `backend/main.py`
**Related:** AdaptiveScorer, Briefing Generator
**Added in:** Sprint 1
**Type:** concept

### Briefing

LLM-generated intelligence summary produced after each scan. Contains critical_alerts, high_priority items, market_summary, and trend analysis. Generated in background thread; SSE briefing_ready event notifies frontend.

**Files:** `backend/processors/briefing.py`
**Related:** Scan
**Added in:** Sprint 2
**Type:** concept

### Root

Geographic scope of a news category: 'kuwait', 'regional', or 'global'. Affects which search queries and sources are used during news fetching.

**Files:** `backend/config.yaml`, `backend/fetchers/news.py`
**Related:** dynamic_categories
**Added in:** Sprint 1
**Type:** concept

### Performance Mode

body.perf-mode CSS class. Disables backdrop-filter, animations, transitions, box-shadow, cosmos canvas. Essential animations preserved (spinners, progress bars). Persists via localStorage stratos_perf_mode.

**Files:** `frontend/ui.js`, `frontend/styles.css`
**Added in:** Sprint 9
**Type:** feature

### Context Hash

SHA-256 hash of role|context|location used to tag retained articles. Switching roles produces a new hash, so retained articles don't bleed across contexts. Trivial edits normalized before hashing.

**Files:** `backend/main.py`
**Related:** Incremental Scanning
**Added in:** Sprint 4
**Type:** concept

### Think Mode

Ollama parameter. For JSON-only endpoints (wizard, generate-profile), set think=False to skip reasoning -- all num_predict tokens go to output, 3-5x faster. For reasoning endpoints (agent chat), leave thinking enabled.

**Files:** `backend/routes/wizard.py`, `backend/routes/generate.py`
**Related:** strip_think_blocks
**Added in:** Sprint 5K
**Type:** concept

### Token Budget

16,000 tokens for context packing. Qwen3.5:9b has 32K context. Tool definitions ~2K + conversation history ~3K + response ~2K = 7K overhead. 16K for injected context is conservative and safe.

**Files:** `backend/routes/personas.py`
**Related:** Context Packing
**Added in:** Sprint 5K
**Type:** constraint

### AuthManager

Profile-based auth in auth.py. Hashed PINs (SHA-256), session tokens (hex, 7-day TTL), rate limiting per endpoint, API key masking, blacklist-preserve profile switching.

**Files:** `backend/auth.py`
**Related:** Auth Middleware, Session Token
**Added in:** Sprint 4
**Type:** class

### Session Token

64-character hex string generated by secrets.token_hex(32). Stored in both .sessions.json (legacy PIN auth) and sessions table (email auth). 7-day TTL. Sent via X-Auth-Token header.

**Files:** `backend/auth.py`
**Related:** AuthManager
**Added in:** Sprint 4
**Type:** concept

### Blacklist-Preserve

Profile switching strategy in auth.py. SYSTEM_KEYS (scoring, search, system, discovery, cache, email, schedule) are preserved across switches. Everything else is nuked and replaced with the new profile's config.

**Files:** `backend/auth.py`
**Related:** AuthManager, Config Overlay
**Added in:** Sprint 8
**Type:** pattern

### Briefing Generator

processors/briefing.py generates LLM-powered intelligence briefings. 573 lines. Runs in background thread after scan. Checks _scan_cancelled to abandon work if a new scan starts.

**Files:** `backend/processors/briefing.py`
**Related:** Briefing, Scan
**Added in:** Sprint 2
**Type:** class

### Profile Generator

processors/profile_generator.py. AI-powered category generation pipeline: LLM Generate -> Sanitize -> Enrichment -> Dedup -> Merge Tiny -> Career Opt-out. 421 lines.

**Files:** `backend/processors/profile_generator.py`
**Related:** dynamic_categories
**Added in:** Sprint 3
**Type:** class

### Context Compression

processors/context_compression.py. Auto-generated state.md files per persona. Logs conversations to daily JSONL. Summarizes older conversations to save context budget.

**Files:** `backend/processors/context_compression.py`
**Related:** Context Packing
**Added in:** Sprint 7
**Type:** module

### User Data

Per-user data directory management in user_data.py. Provides structured JSONL/JSON exports alongside SQLite: data/users/{user_id}/ with scan logs, feedback, briefings.

**Files:** `backend/user_data.py`
**Related:** Database Class
**Added in:** Sprint 8
**Type:** module

### Prompt Builder

Sprint Prompt Builder in dev_endpoints.py + prompt-builder.js. Form UI that assembles sprint prompts from templates + live filesystem state. Template engine is pure string assembly, no LLM.

**Files:** `backend/routes/dev_endpoints.py`, `frontend/prompt-builder.js`
**Related:** sprint_log, prompt_templates
**Added in:** Sprint 10
**Type:** feature

### Service Worker

sw.js implements network-first caching with offline shell. Cache name: stratos-v13. Pre-caches app shell assets for instant offline loading.

**Files:** `frontend/sw.js`
**Added in:** Sprint 5
**Type:** module

---

## 📡 Data Fetchers

### NewsFetcher

Multi-source news fetcher in fetchers/news.py. Sources: DuckDuckGo, Serper, RSS feeds. Uses ThreadPoolExecutor for parallel fetching. Deduplicates by URL.

**Files:** `backend/fetchers/news.py`
**Related:** Scan, Serper
**Added in:** Sprint 1
**Type:** class

### MarketFetcher

Yahoo Finance market data via yfinance in fetchers/market.py. Handles multiple intervals (1m, 5m, 1d, 1wk). fetch_single() for per-ticker live updates. Parallel batch fetching.

**Files:** `backend/fetchers/market.py`
**Related:** Watchlist
**Added in:** Sprint 1
**Type:** class

### EntityDiscovery

Tracks keyword frequency changes to detect rising topics in fetchers/discovery.py. Compares mention counts against a 7-day baseline. Auto-tracks when rising_multiplier exceeded.

**Files:** `backend/fetchers/discovery.py`
**Added in:** Sprint 2
**Type:** class

### Extra Feeds

RSS feed URL registry in fetchers/extra_feeds.py. Curated lists of finance, politics, and jobs RSS feeds. User-toggleable via config. get_catalog() returns the full feed list for the Settings UI.

**Files:** `backend/fetchers/extra_feeds.py`
**Added in:** Sprint 3
**Type:** module

### Watchlist

User's tracked market tickers stored in config.yaml market.tickers. Each entry: {symbol, name}. Managed via manage_watchlist agent tool or Settings UI.

**Files:** `backend/config.yaml`
**Related:** MarketFetcher, Market Persona
**Added in:** Sprint 1
**Type:** concept

---

## 🧠 Training Pipeline

### Learn Cycle

Manual self-improvement: distill.py (Opus re-scores) -> export_training.py (JSONL) -> train_lora.py (fine-tune). Automated by learn_cycle.sh or autopilot.py.

**Files:** `backend/distill.py`, `backend/export_training.py`, `backend/train_lora.py`
**Related:** Distillation, Export Training, LoRA Training
**Added in:** Sprint 2
**Type:** process

### DoRA

Weight-Decomposed Low-Rank Adaptation. Advanced LoRA variant used in V2 training pipeline. Combines decomposed weight matrices for better fine-tuning.

**Files:** `backend/train_lora.py`
**Related:** LoRA Training
**Added in:** Sprint 4
**Type:** technique

### V2 Pipeline

data/v2_pipeline/ directory. 30 training profiles, 4-stage process: profile definitions -> article collection -> Opus batch scoring -> training data preparation with dual weighting.

**Files:** `backend/data/v2_pipeline/`
**Related:** LoRA Training
**Added in:** Sprint 4
**Type:** module

### MAE

Mean Absolute Error between model predictions and ground truth (Opus) scores. Primary evaluation metric. MAE > 0.6 indicates the model needs retraining.

**Files:** `backend/evaluate_scorer.py`, `backend/model_manager.py`
**Related:** LoRA Training
**Added in:** Sprint 3
**Type:** metric

### Direction Accuracy

Percentage of items where the model's score and Opus agree on above/below 5.0 threshold. Below 95% triggers a review recommendation.

**Files:** `backend/evaluate_scorer.py`, `backend/model_manager.py`
**Related:** MAE
**Added in:** Sprint 3
**Type:** metric

### Diversity Guard

In autopilot.py, prevents catastrophic forgetting during incremental training. If training data covers fewer than 4 profiles, forces full retrain from base model instead of incremental update.

**Files:** `backend/autopilot.py`
**Related:** Autopilot, LoRA Training
**Added in:** Sprint 4
**Type:** safeguard

### CostTracker

Budget tracking for autopilot Opus API calls. Persists to data/cost_tracker.json. Tracks total_spent, budget, remaining. Stops autopilot when budget exhausted.

**Files:** `backend/autopilot.py`
**Related:** Autopilot
**Added in:** Sprint 3
**Type:** class
