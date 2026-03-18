# Frontend Dependency Graph — StratOS (2026-03-18)

## Script Load Order (index.html)
1. ui-dialogs.js — Toast & modal system
2. auth-styles.js — Auth CSS injection
3. auth-star-canvas.js — Auth canvas effects
4. ui-sync.js — UI state persistence
5. auth.js — Login/register, fetch interceptor
6. tts-settings.js — TTS voice configuration
7. app.js — Main initialization (2208 lines)
8. nav.js — Sidebar + navigation
9. settings.js — Config management (1793 lines)
10. settings-tickers.js — Ticker management
11. settings-sources.js — News source catalog
12. settings-categories.js — Dynamic categories
13. wizard-data.js — Wizard constants
14. wizard.js — Onboarding wizard (2172 lines)
15. market.js — Market data visualization
16. markets-panel.js — Markets panel
17. feed.js — News feed rendering (1520 lines)
18. persona-context.js — Persona context editing
19. file-browser.js — File browser (1012 lines)
20. youtube.js — YouTube settings integration (1467 lines)
21. youtube-kb.js — YouTube knowledge base (1250 lines)
22. games-ui.js — Gaming scenarios
23. workspace.js — Profile workspace
24. agent-suggestions.js — Suggestion rendering
25. agent-customizer.js — Agent customization (FROZEN)
26. agent-tickers.js — Agent ticker management
27. stt.js — Speech-to-text
28. agent.js — AI chat agent (2041 lines)
29. image-gen.js — Image generation
30. ui.js — Main UI utilities (1618 lines)
31. prompt-builder.js — Prompt building tools
32. codex.js — Codex system
33. scan-history.js — Scan history
34. theme-editor.js — Theme editor (1068 lines)
35. tour.js — User tour (830 lines)
36. fullscreen-chart-utils.js — Chart utilities
37. fullscreen-chart.js — Fullscreen chart (2432 lines)
38. mobile.js — Mobile adaptations (860 lines)

## File Details

### auth.js (842 lines)
- **API Calls**: /api/auth/register, /api/auth/login, /api/auth/login-pin, /api/auth/reset-password, /api/auth/profiles, /api/auth/status
- **Exports**: getAuthToken(), setAuthToken(), clearAuthToken(), getActiveProfile(), setActiveProfile(), getDeviceId(), checkAuthAndInit()
- **Intercepts**: ALL /api/ fetches — injects X-Auth-Token + X-Device-Id headers
- **localStorage**: stratos_auth_token, stratos_active_profile, stratos_device_id, stratos_auth_theme_idx

### app.js (2208 lines)
- **API Calls**: GET /api/config, GET /api/saved-signals, POST /api/unsave-signal, POST /api/feedback, POST /api/agent-warmup
- **SSE Listeners**: scan, complete, scan_cancelled, scan_error, status, pass1_complete, briefing_ready, critical_signal, youtube_processing, lens_extracted, narration_resolved
- **Exports**: init(), getSavedSignals(), toggleSaveSignal(), _setAutoRefresh(), _applyDensity(), _applyFontSize(), _setDefaultChartType(), _persistUIPref(), syncSavedSignals()
- **Global State**: newsData, financeNewsData, politicsNewsData, jobsNewsData, customNewsData, configData, activeRoot

### agent.js (2041 lines)
- **API Calls**: GET/POST/PUT/DELETE /api/conversations, POST /api/agent-chat (SSE stream)
- **Exports**: sendAgentMessage(), switchPersona(), newAgentChat(), _toggleFreeLength(), _toggleAllScans(), _toggleSignalInjection(), appendAgentMessage(), formatAgentText()
- **Global State**: agentHistory, currentPersona, _agentActiveConvId, _agentFreeLength, _agentAllScans, _agentInjectSignals, agentStreaming
- **DOM**: #agent-panel, #agent-messages, #agent-input, #agent-conv-tabs

### feed.js (1520 lines)
- **API Calls**: GET /api/feed-catalog/{type}, POST /api/config
- **Exports**: renderFeed(), renderBriefing(), deduplicateNews(), matchesKuwait(), matchesBanks(), matchesRegional(), matchesMarket(), timeAgo()
- **Global State**: scoreFilters, newsData (from app.js), activeRoot (from app.js)
- **DOM**: #news-feed, #feed-search, #feed-controls, #briefing-panel

### settings.js (1793 lines)
- **API Calls**: GET /api/config, POST /api/config, POST /api/save-serper-key, GET /api/serper-credits, GET /api/search-status
- **Exports**: loadConfig(), saveConfig(), loadPresets(), esc()
- **Global State**: configData, window._settingsDirty

### youtube-kb.js (1250 lines)
- **API Calls**: GET /api/youtube/channels, GET /api/youtube/videos/{id}, GET /api/youtube/insights/{id}, POST /api/youtube/extract-all/{id}, POST /api/youtube/process/{id}
- **Exports**: initYouTubeKB()
- **SSE**: Wraps _handleLensExtracted to invalidate insight cache
- **Global State**: _ykbChannels, _ykbVideos, _ykbInsightsByLang, _ykbInsightsLoaded

### youtube.js (1467 lines)
- **API Calls**: GET /api/youtube/channels, POST /api/youtube/channels, DELETE /api/youtube/channels/{id}, POST /api/youtube/extract-lens, POST /api/youtube/retranscribe
- **Exports**: _handleLensExtracted(), _handleNarrationResolved(), initYouTube()
- **SSE handlers**: lens_extracted (original), narration_resolved

### markets-panel.js
- **API Calls**: GET /api/market-tick, POST /api/agent-chat
- **DOM**: #markets-panel, market data cards, refresh bar

### fullscreen-chart.js (2432 lines)
- **API Calls**: GET /api/market-tick, POST /api/agent-chat
- **Exports**: openFullscreenChart(), setChartType(), addToWatchlist()
- **Global State**: _tvChart, _selectedTicker, _watchlist

### theme-editor.js (1068 lines)
- **Exports**: initThemeEditor(), openThemeEditor(), setThemeProperty(), saveThemePreset(), loadThemePreset()
- **localStorage**: Theme presets, CSS variable state

### games-ui.js
- **API Calls**: GET /api/scenarios, GET /api/scenarios/active, POST /api/scenarios/create, GET /api/scenarios/generate-status, POST /api/scenarios/activate
- **Exports**: initGamesUI(), _gamesGetState(), _gamesAutoDetectNpc()

### wizard.js (2172 lines)
- **API Calls**: POST /api/generate-profile, GET /api/wizard-preselect, GET /api/wizard-tab-suggest, GET /api/wizard-rv-items
- **Exports**: openWizard(), closeWizard(), _wiz (global object)

### ui-sync.js (389 lines)
- **API Calls**: GET /api/ui-state, POST /api/ui-state, sendBeacon on unload
- **Patches**: localStorage.setItem/removeItem to auto-detect changes
- **Syncs**: 80+ localStorage keys to/from server

### nav.js (274 lines)
- **Exports**: toggleSidebar(), renderNav(), setActive()
- **DOM**: #sidebar, #nav-menu, #main-content, panel visibility

### file-browser.js (1012 lines)
- **API Calls**: GET /api/persona-files, GET /api/persona-files/read, POST /api/persona-files/write, POST /api/file-assist

### workspace.js
- **API Calls**: GET /api/profile/workspace-stats, GET /api/profile/export, POST /api/profile/import, GET/DELETE /api/preference-signals

### persona-context.js
- **API Calls**: GET/POST /api/persona-context, GET /api/persona-context/versions, POST /api/persona-context/revert

## Global Functions Used Cross-File
- `showToast(msg, type)` — ui-dialogs.js (30+ call sites)
- `esc(str)` — settings.js (20+ files)
- `stratosPrompt()`, `stratosConfirm()` — ui-dialogs.js (20+ files)
- `getAuthToken()`, `getActiveProfile()` — auth.js (15+ files)
- `lucide.createIcons()` — external library (25+ files)
- `renderFeed()`, `renderNav()` — feed.js, nav.js (called from app.js, agent.js, etc.)
