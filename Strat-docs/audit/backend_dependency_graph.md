# Backend Dependency Graph — StratOS (2026-03-18)

## Architecture Overview
- **Total**: ~25,635 lines across 29 core files
- **Entry**: main.py (1,795 lines) → server.py (821 lines) dispatches to routes/
- **DB**: SQLite WAL, 47+ tables, singleton via database.py
- **SSE**: sse.py broadcasts to registered clients

## Core Files

### main.py
- **Imports**: database, fetchers/*, processors/scorer_adaptive, processors/briefing, sse, user_data
- **Shared State**: `_config_lock` (RLock), `_scan_cancelled` (Event), `scan_status`, `_profile_configs`
- **SSE Events**: status, scan_progress, article_added, briefing_ready, shutdown, critical_signal, pass1_complete, scan_cancelled, scan_error, complete
- **DB Tables**: news_items, market_snapshots, entities, entity_mentions, scan_log, briefings

### server.py
- **Imports**: ALL route modules, auth.py, email_service
- **Endpoints**: /api/events (SSE), /api/auth (POST), /api/register (POST), /api/logout (POST), /api/config (POST)
- **Shared State**: `_gzip_cache`, `_gzip_cache_lock`, `_device_profile_cache`
- **Dispatches**: handle_get/post/delete to all route modules sequentially

### database.py (1,049 lines)
- **Thread Safety**: Per-thread connections via threading.local(), `_commit_lock`
- **Column Whitelists**: `_RP_MESSAGE_COLUMNS`, `_CHARACTER_CARD_COLUMNS`, `_GENERATED_IMAGE_COLUMNS`
- **Key Methods**: insert_news_item, search_news_history, insert_rp_message, get_full_branch_conversation, upsert_rp_context, get_ui_state

### auth.py
- **Shared State**: `_active_sessions`, `_rate_buckets`, `_rate_lock`
- **Rate Limits**: /api/refresh (2/60s), /api/agent-chat (20/60s), /api/auth/login (10/300s)

### sse.py
- **Shared State**: `_clients` list, `_lock`
- **Method**: broadcast(event_type, data, profile_id)

## Route Modules

### routes/agent.py (~500 lines)
- **Endpoints**: POST /api/agent-chat, GET /api/agent-status, POST /api/suggest-context, POST /api/ask, POST /api/file-assist
- **DB Reads**: persona_entities, news_items
- **SSE Events**: suggestions, error
- **Depends On**: personas.py (build_persona_prompt, build_persona_context), agent_tools.py, persona_prompts.py

### routes/personas.py (1,011 lines)
- **Functions**: build_persona_context, _build_news_context, _build_historical_context, _build_scholarly_context, _build_games_context, _pack_context
- **DB Reads**: news_items, user_feedback, user_preference_signals, persona_entities
- **Context Budget**: 16K tokens (word_count × 1.3)

### routes/persona_prompts.py (271 lines)
- **Functions**: _intelligence_prompt, _market_prompt, _scholarly_prompt, _games_prompt, _games_immersive_prompt
- **No DB access, no SSE events**

### routes/rp_chat.py (1,257 lines)
- **Endpoints**: POST /api/rp/chat, /api/rp/regenerate, /api/rp/edit, /api/rp/branch, GET /api/rp/branches/{sid}, POST /api/rp/director-note, GET /api/rp/history/{sid}, POST /api/rp/feedback
- **DB Tables**: rp_messages, rp_edits, rp_suggestions, rp_feedback, rp_context
- **SSE Events**: token, error

### routes/data_endpoints.py
- **GET Endpoints**: /api/data, /api/briefing, /api/status, /api/ui-state, /api/saved-signals, /api/config, /api/profiles, /api/export, /api/feedback-stats, /api/search-all-contexts, /api/shadow-scores, /api/agent-status, /api/health, /api/top-movers, /api/scan-log, /api/search-status
- **POST Endpoints**: /api/feedback, /api/unsave-signal, /api/ui-state, /api/search
- **DB Tables**: news_items, user_feedback, profiles, scan_log

### routes/feeds.py
- **Endpoints**: GET /api/finance-news, /api/politics-news, /api/jobs-news, /api/custom-news, /api/feed-catalog/{type}, POST /api/discover-rss
- **SSRF Protection**: URL validation before + after redirects

### routes/controls.py (238 lines)
- **Endpoints**: GET /api/refresh, /api/refresh-market, /api/refresh-news, /api/market-tick, /api/scan/status, POST /api/scan/cancel, /api/agent-warmup, GET/POST /api/ticker-presets

### routes/config.py
- **Endpoints**: POST /api/config
- **Writes**: config.yaml (system-level), profile YAML (profile-level)
- **Keys Handled**: profile, market, news, search, scoring, tts, dynamic_categories, extra_feeds_*, custom_feeds*, ui_preferences

### routes/media.py (426 lines)
- **Endpoints**: GET /api/proxy, /api/tts/voices, /api/tts/status, /api/persona-files, POST /api/files/upload, /api/tts, /api/tts/preview, /api/stt, DELETE /api/files/*

### routes/youtube_endpoints.py (950 lines)
- **Endpoints**: Channel CRUD, video transcription, insight extraction, retranscribe, translate, cancel
- **DB Tables**: youtube_channels, youtube_videos, video_insights
- **SSE Events**: youtube_processing, lens_extracted

### routes/character_cards.py
- **Endpoints**: POST /api/cards, GET/PUT/DELETE /api/cards/{id}, POST /api/cards/import/tavern, GET /api/cards/my, /api/cards/browse, /api/cards/trending, /api/cards/search
- **DB Tables**: character_cards, character_card_stats, character_card_ratings

### routes/image_gen.py (332 lines)
- **Endpoints**: POST /api/image/generate, /api/image/character-portrait, GET /api/image/{id}, /api/image/gallery, DELETE /api/image/{id}
- **DB Tables**: generated_images

### routes/persona_data.py (737 lines)
- **Endpoints**: Conversations CRUD, scenarios CRUD, preferences, workspace export/import, persona context
- **DB Tables**: conversations (via persona_data), persona_context, persona_entities, user_preference_signals

### routes/wizard.py (434 lines)
- **Endpoints**: POST /api/wizard-preselect, /api/wizard-tab-suggest, /api/wizard-rv-items

### routes/generate.py
- **Endpoints**: POST /api/generate-profile

### routes/rp_memory.py (404 lines)
- **Tiered Memory**: Tier 1 (facts), Tier 2 (recent), Tier 3 (arc summaries)
- **DB Table**: rp_context
- **Functions**: extract_facts_immediate, extract_facts, detect_scene_transition, extract_arc_summary

### routes/helpers.py
- **Utilities**: json_response, error_response, read_json_body, sse_event, start_sse, strip_think_blocks

### routes/url_validation.py (113 lines)
- **Function**: validate_url() — SSRF protection
