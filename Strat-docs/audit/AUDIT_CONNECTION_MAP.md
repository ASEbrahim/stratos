# Audit Connection Map

## Module Dependency Graph

### server.py (Hub — HTTP Dispatcher)
- **Imports**: routes/{agent, auth, generate, wizard, helpers, config, feeds, media, data_endpoints, controls, youtube_endpoints, persona_data, dev_endpoints}, email_service.EmailService
- **Creates**: CORSHandler (HTTP handler), ThreadedHTTPServer
- **Binds via closure**: strat (StratOS), auth (AuthManager), frontend_dir, output_dir
- **SSE**: Manages `/api/events` long-lived connections; heartbeat every 15s
- **Static files**: Gzip-compressed serving with thread-safe cache

### main.py (Core Orchestrator — StratOS class)
- **Imports**: database, fetchers/{market, news, discovery}, processors/{scorer_adaptive, briefing}, sse
- **Key methods**: run_scan(), run_market_refresh(), run_news_refresh(), ensure_profile()
- **Threading**: _scan_lock, _output_lock, _config_lock, _ollama_lock, _scan_cancelled event
- **Profile isolation**: _profile_configs cache, active_profile tracking
- **SSE**: Delegates to SSEManager (self.sse)

### auth.py (AuthManager)
- **Session types**: In-memory dict (_active_sessions) + DB-backed (sessions table)
- **Rate limiting**: Global per-path (not per-client — see F001)
- **Profile loading**: Blacklist-preserve pattern for config isolation

### database.py (SQLite Manager)
- **Threading**: Per-thread connections via threading.local()
- **WAL mode**: busy_timeout=10s, NORMAL synchronous
- **Singleton**: Thread-safe via double-checked locking (C008 fix)
- **Tables**: 25 tables across 25 migrations

### sse.py (SSEManager)
- **Profile-scoped**: Broadcasts can target specific profile_ids
- **Thread-safe**: Internal lock on _clients list
- **Dead client cleanup**: Inline during broadcast

---

## Route Module Map

| Module | GET Paths | POST Paths | DELETE Paths |
|--------|-----------|------------|--------------|
| controls.py | /api/refresh, /api/refresh-market, /api/market-tick, /api/refresh-news, /api/ticker-presets, /api/scan/status | /api/scan/cancel, /api/agent-warmup, /api/ticker-presets | — |
| feeds.py | /api/finance-news, /api/politics-news, /api/jobs-news, /api/custom-news, /api/feed-catalog/* | /api/discover-rss | — |
| youtube_endpoints.py | /api/youtube/channels, /api/youtube/status, /api/youtube/videos/:id, /api/youtube/insights/:id, /api/youtube/export/:id | /api/youtube/channels, /api/youtube/process/:id, /api/youtube/extract-all/:id, /api/youtube/extract-lens, /api/youtube/retranscribe, /api/youtube/channels/:id/lenses | /api/youtube/channels/:id |
| data_endpoints.py | /api/data, /api/briefing, /api/status, /api/config, /api/profiles, /api/export, /api/feedback-stats, /api/search-all-contexts, /api/shadow-scores, /api/health, /api/top-movers, /api/ui-state, /api/scan-log, /api/agent-status, /api/agent-personas | /api/feedback, /api/search, /api/scan-log, /api/ui-state | — |
| persona_data.py | /api/persona-context, /api/conversations, /api/scenarios, /api/personas/:p/entities, /api/persona-state, /api/profile/workspace-stats, /api/preference-signals | /api/persona-context, /api/conversations, /api/scenarios/*, /api/personas/:p/entities, /api/conversation-log, /api/update-state, /api/profile/export, /api/profile/import, /api/preference-signals | /api/persona-context, /api/conversations/:id, /api/personas/:p/entities/:name, /api/scenarios, /api/preference-signals/:id |
| media.py | /api/proxy, /api/tts/voices, /api/tts/status, /api/persona-files | /api/files/upload, /api/files/list, /api/tts, /api/tts/preview, /api/tts/voices/custom, /api/stt, /api/persona-files/write, /api/persona-files/mkdir | /api/files/:id, /api/persona-files |
| dev_endpoints.py | /api/dev/context, /api/dev/sprint-log, /api/dev/templates | /api/prompt-builder/generate, /api/dev/sprint-log, /api/dev/templates | — |
| auth (routes) | /api/auth/registration-status, /api/auth/check, /api/profiles, /api/admin/users | /api/auth/register, /api/auth/verify, /api/auth/login, /api/auth/logout, /api/auth/change-password, /api/auth/forgot-password, /api/auth/reset-password, /api/auth/otp-request, /api/auth/otp-verify, /api/auth/delete-account, /api/profiles, /api/profiles/:id/activate, /api/admin/invite, /api/admin/reset-user, /api/admin/delete-user | /api/profiles/:id |

---

## Processor Dependencies

| Processor | Uses | Used By |
|-----------|------|---------|
| briefing.py | Ollama (inference_model), routes/helpers | main.py (deferred briefing thread) |
| lenses.py | Ollama (inference_model) | youtube.py, youtube_endpoints.py |
| youtube.py | youtube-transcript-api, Supadata API, faster-whisper, requests | youtube_endpoints.py, youtube_worker.py |
| youtube_worker.py | youtube.py, lenses.py | server.py (daemon thread) |
| source_resolver.py | fetchers/serper_search | youtube_endpoints.py |
| tts.py | kokoro, edge_tts | media.py |
| stt.py | faster_whisper, PyAV | media.py |
| persona_context.py | database | persona_data.py, media.py |
| context_compression.py | Ollama, filesystem | persona_data.py |
| scenarios.py | database | persona_data.py |
| scenario_generator.py | Ollama | persona_data.py (background thread) |
| workspace.py | database, filesystem | persona_data.py |
| file_handler.py | database, filesystem, OCR | media.py |

## Frontend Module Map

| Module | Key Responsibilities |
|--------|---------------------|
| app.js | State management, nav sections, market view, SSE event handling |
| agent.js | Chat UI, conversation management, streaming, persona switching |
| feed.js | News feed rendering, filtering, search |
| settings.js | Config load/save, settings panels |
| youtube.js | YouTube panel, insights modal, lens viewer |
| ui.js | Theme system, chart theming, stars, perf mode |
| auth.js | Login/register forms, session management |
| market.js | TradingView chart wrapper, OHLC data resolution, timeframe switching |
| markets-panel.js | Multi-chart grid, heatmap overview, ticker intel sidebar |
| fullscreen-chart.js | Binance-style fullscreen chart with drawing tools, intel panel, embedded agent |
| fullscreen-chart-utils.js | SMA calculation, OHLC data extraction, price formatting |
| nav.js | Sidebar navigation, section collapse, tab switching, resize handle |
| mobile.js | Touch gestures (swipe sidebar, card swipe, pull-to-refresh), bottom nav, PWA install |
| stt.js | Speech-to-text mic button, hold-to-record, MediaRecorder API |
| tts-settings.js | Voice picker (Kokoro + Edge-TTS), speed control, per-persona voice overrides |
| codex.js | Codex browser modal, category/term navigation, search |
| games-ui.js | Scenario selector, entity management, RP mode toggle, generation polling |
| file-browser.js | Nautilus-style file browser/editor, persona-scoped directories, drag/resize |
| persona-context.js | Per-persona system context editor, version history, revert |
| scan-history.js | Scan history panel, export (CSV/JSON) |
| wizard.js | Onboarding wizard, category generation, AI suggestions |
| wizard-data.js | Wizard data constants (categories, interests, role keywords) |
| settings-categories.js | Dynamic category rendering, add/remove/toggle |
| settings-sources.js | News source catalog, custom RSS feeds, RSS auto-discovery |
| settings-tickers.js | Ticker management, drag-drop reorder, presets |
| theme-editor.js | Real-time CSS variable tweaker, per-theme presets |
| ui-dialogs.js | Toast notifications, styled prompt/confirm modals |
| ui-sync.js | localStorage-to-DB sync for UI settings, debounced persistence |
| workspace.js | Profile export/import, preference signals, workspace stats |
| agent-customizer.js | Fullscreen agent appearance settings (opacity, blur, fonts, presets) |
| agent-suggestions.js | Persona-aware suggestion chips, response follow-up chips |
| agent-tickers.js | Watchlist widget, ticker/category commands via agent chat |
| prompt-builder.js | Sprint prompt builder form, templates, history |
| auth-star-canvas.js | Interactive star field on auth screen |
| auth-styles.js | Auth overlay CSS injection |
| tour.js | Guided tour system (basic + explore), spotlight overlay, step navigation |
| sw.js | Service worker — network-first caching, offline shell |

## Auth Token Key Usage (Session 2 Discovery)

The canonical auth token key is `stratos_auth_token` (defined in `auth.js` as `AUTH_TOKEN_KEY`).

| Pattern | Files | Status |
|---------|-------|--------|
| `getAuthToken()` | Most files | Correct |
| `localStorage.getItem('auth_token')` | app.js, feed.js, settings-sources.js | **Fixed in C012** |
| `localStorage.getItem('stratos_token')` | settings-tickers.js | **Fixed in C013** |
| `localStorage.getItem('stratos_session_token')` | prompt-builder.js, agent.js | **Fixed in C014** (agent.js: F010) |
| `Authorization: Bearer` header | settings-tickers.js | **Fixed in C013** (backend only reads X-Auth-Token) |
