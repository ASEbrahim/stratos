# Cross-Boundary Dependency Graph — StratOS (2026-03-18)

## 1. Endpoint Cross-Reference

| Endpoint | Method | Backend File | Frontend Consumer(s) |
|----------|--------|-------------|---------------------|
| /api/config | GET | data_endpoints.py | app.js, settings.js, agent.js, markets-panel.js, feed.js (20+) |
| /api/config | POST | config.py | settings.js, tts-settings.js, feed.js, agent-tickers.js, app.js |
| /api/data | GET | data_endpoints.py | app.js (main dashboard load) |
| /api/briefing | GET | data_endpoints.py | app.js (briefing panel) |
| /api/status | GET | data_endpoints.py | app.js, markets-panel.js |
| /api/agent-chat | POST | agent.py | agent.js, markets-panel.js, fullscreen-chart.js |
| /api/agent-status | GET | data_endpoints.py | agent.js, stt.js |
| /api/saved-signals | GET | data_endpoints.py | app.js |
| /api/unsave-signal | POST | data_endpoints.py | app.js |
| /api/feedback | POST | data_endpoints.py | app.js, ui.js |
| /api/ui-state | GET/POST | data_endpoints.py | ui-sync.js |
| /api/market-tick | GET | controls.py | markets-panel.js, fullscreen-chart.js |
| /api/refresh | GET | controls.py | app.js |
| /api/refresh-market | GET | controls.py | app.js |
| /api/refresh-news | GET | controls.py | app.js |
| /api/scan/status | GET | controls.py | app.js |
| /api/scan/cancel | POST | controls.py | app.js |
| /api/agent-warmup | POST | controls.py | app.js |
| /api/ticker-presets | GET/POST | controls.py | settings-tickers.js |
| /api/youtube/channels | GET/POST | youtube_endpoints.py | youtube.js, youtube-kb.js |
| /api/youtube/videos/{id} | GET | youtube_endpoints.py | youtube-kb.js |
| /api/youtube/insights/{id} | GET | youtube_endpoints.py | youtube.js, youtube-kb.js |
| /api/youtube/extract-lens | POST | youtube_endpoints.py | youtube.js, youtube-kb.js |
| /api/youtube/retranscribe | POST | youtube_endpoints.py | youtube.js, youtube-kb.js |
| /api/youtube/translate-transcript | POST | youtube_endpoints.py | youtube.js, youtube-kb.js |
| /api/feed-catalog/{type} | GET | feeds.py | feed.js, settings-sources.js |
| /api/finance-news | GET | feeds.py | app.js (extra feeds) |
| /api/politics-news | GET | feeds.py | app.js (extra feeds) |
| /api/jobs-news | GET | feeds.py | app.js (extra feeds) |
| /api/custom-news | GET | feeds.py | app.js (extra feeds) |
| /api/discover-rss | POST | feeds.py | settings-sources.js |
| /api/conversations | GET/POST/PUT/DELETE | persona_data.py | agent.js |
| /api/persona-context | GET/POST | persona_data.py | persona-context.js |
| /api/scenarios | GET/POST | persona_data.py | games-ui.js |
| /api/preference-signals | GET/DELETE | persona_data.py | workspace.js |
| /api/generate-profile | POST | generate.py | wizard.js, settings.js |
| /api/wizard-preselect | POST | wizard.py | wizard.js |
| /api/wizard-tab-suggest | POST | wizard.py | wizard.js |
| /api/wizard-rv-items | POST | wizard.py | wizard.js |
| /api/rp/chat | POST | rp_chat.py | agent.js (gaming mode), mobile chat.ts |
| /api/rp/regenerate | POST | rp_chat.py | mobile rp.ts |
| /api/rp/edit | POST | rp_chat.py | agent.js, mobile rp.ts |
| /api/rp/history/{sid} | GET | rp_chat.py | mobile rp.ts |
| /api/cards | POST/GET/PUT/DELETE | character_cards.py | mobile (card library) |
| /api/cards/import/tavern | POST | character_cards.py | mobile (import) |
| /api/image/generate | POST | image_gen.py | image-gen.js, mobile rp.ts |
| /api/image/gallery | GET | image_gen.py | image-gen.js |
| /api/tts | POST | media.py | agent.js (TTS playback) |
| /api/tts/voices | GET | media.py | tts-settings.js |
| /api/tts/status | GET | media.py | tts-settings.js |
| /api/stt | POST | media.py | stt.js |
| /api/proxy | GET | media.py | feed.js (image proxying) |
| /api/persona-files | GET | media.py | file-browser.js |
| /api/files/upload | POST | media.py | file-browser.js, agent.js |
| /api/profiles | GET/POST/DELETE | auth.py | settings.js, auth.js |
| /api/auth/* | POST | auth.py | auth.js |
| /api/events | GET | server.py | app.js (SSE connection) |

## 2. SSE Event Wiring

| Event Name | Backend Emitter | Frontend Handler |
|------------|-----------------|------------------|
| scan | main.py | app.js → _handleSSEScan() |
| complete | main.py | app.js → _handleSSEComplete() |
| scan_cancelled | main.py | app.js → _handleSSECancelled() |
| scan_error | main.py | app.js → _handleSSEError() |
| status | main.py | app.js → _handleSSEStatus() |
| pass1_complete | main.py | app.js → _handleSSEPass1Complete() |
| briefing_ready | main.py | app.js → fetches /api/briefing |
| critical_signal | main.py | app.js → _handleCriticalSignal() |
| youtube_processing | youtube_endpoints.py | app.js → _handleYouTubeSSE() |
| lens_extracted | youtube_endpoints.py | app.js → _handleLensExtracted() → youtube.js + youtube-kb.js |
| narration_resolved | source_resolver.py | app.js → _handleNarrationResolved() → youtube.js |

## 3. Shared State Flows

| Flow | Writer | Reader | Notes |
|------|--------|--------|-------|
| Config (role, tickers, categories) | settings.js → POST /api/config → config.yaml | ALL frontend via GET /api/config | Config lock serializes |
| News data | Scanner (main.py) → news_items table | app.js via GET /api/data | JSON export per-profile |
| Market data | MarketFetcher → market_snapshots | markets-panel.js, fullscreen-chart.js via GET /api/market-tick | 60s cache TTL |
| Saved signals | app.js → POST /api/feedback (save) | app.js → GET /api/saved-signals | Per-profile scoped |
| UI state | ui-sync.js → POST /api/ui-state | ui-sync.js → GET /api/ui-state | 80+ localStorage keys synced |
| Chat history | agent.js → POST /api/agent-chat | agent.js → GET /api/conversations/{id} | DB-backed |
| YouTube insights | youtube_endpoints.py → video_insights table | youtube.js, youtube-kb.js → GET /api/youtube/insights | SSE triggers refresh |

## 4. Sprint Item Parallelization Analysis

### SAFE TO RUN IN PARALLEL (no shared files/endpoints)

**Group A: Markets** (markets-panel.js, market.js, fullscreen-chart.js)
- BUG-05: Markets duplicate chart → markets-panel.js
- BUG-06: Market auto-refresh immediate trigger → app.js (_setAutoRefresh)
- BUG-07: Markets summary sync → markets-panel.js
- UX-10: Market refresh bar toggleable → markets-panel.js

**Group B: Theme Editor** (theme-editor.js only)
- BUG-03: Reset button → theme-editor.js
- BUG-04: Theme save position/scale/blur → theme-editor.js
- UX-11: Undo button → theme-editor.js

**Group C: YouTube** (youtube.js, youtube-kb.js)
- UX-06: Show retranscription progress → youtube.js, youtube-kb.js
- UX-07: Save area for YouTube → youtube.js, youtube-kb.js

**Group D: Agent Chat** (agent.js, persona_prompts.py, personas.py)
- UX-01: Signal hyperlinks → agent.js
- UX-04: Edit own messages → agent.js
- UX-05: Edit AI messages as learning → agent.js, agent.py

**Group E: Gaming** (games-ui.js, persona_data.py, rp_chat.py)
- BUG-12: Character reveal progression → games-ui.js, personas.py (suggestion gen)
- GAME-01 through GAME-06 → games-ui.js, new files

**Group F: Feed** (feed.js, app.js)
- UX-08: More context input for sources → settings-sources.js
- UX-09: Post-wizard scan guidance → wizard.js

**Group G: Error Messages** (cross-cutting)
- BUG-09: Loading indicators → multiple files (SEQUENTIAL)
- BUG-10: Vague error messages → multiple files (SEQUENTIAL)

### MUST BE SEQUENTIAL (shared state or cross-file dependencies)
- BUG-09 (loading indicators) touches agent.js, youtube.js, markets-panel.js, feed.js
- BUG-10 (error messages) touches multiple catch blocks in frontend + backend
- UX-02 (article transcriber) needs new processor + endpoint + frontend wiring
- UX-03 (context-driven suggestions) touches agent-suggestions.js + agent.py (already modified)

### RECOMMENDED PARALLEL GROUPS

| Agent | Items | Files Touched |
|-------|-------|---------------|
| Agent 1: Markets | BUG-05, BUG-07, UX-10 | markets-panel.js, market.js |
| Agent 2: Theme | BUG-03, BUG-04, UX-11 | theme-editor.js |
| Agent 3: Agent Chat UX | UX-01, UX-04, UX-05 | agent.js |
| Agent 4: Gaming | BUG-12, GAME-01, GAME-04 | games-ui.js, gaming-import-wizard.js (new), personas.py |
| Agent 5: YouTube | UX-06, UX-07 | youtube.js, youtube-kb.js |
| **Sequential**: BUG-09, BUG-10, BUG-06 | Cross-cutting | Multiple files |

### FILE OWNERSHIP MAP (conflict avoidance)

| File | Owned By | DO NOT Touch From Other Agents |
|------|----------|-------------------------------|
| agent.js | Agent 3 (Chat UX) | — |
| markets-panel.js | Agent 1 (Markets) | — |
| theme-editor.js | Agent 2 (Theme) | — |
| games-ui.js | Agent 4 (Gaming) | — |
| youtube.js, youtube-kb.js | Agent 5 (YouTube) | — |
| app.js | Sequential only | Multiple agents need it → run last |
| settings.js | Sequential only | Shared config UI |
| persona_prompts.py | ALREADY DONE (AGENT-02/03) | — |
| personas.py | ALREADY DONE (AGENT-05/06) + Agent 4 needs it for BUG-12 | Coordinate |
| feed.js | Sequential (UX-08, UX-09 related) | — |
