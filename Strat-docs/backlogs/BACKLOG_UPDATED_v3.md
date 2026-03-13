# StratOS Backlog — March 2026

---

## Bugs & Fixes

| # | Issue | Severity | Notes |
|---|-------|----------|-------|
| B1 | Cross-profile data bleed + SQLite `unable to open database file` crashes | Critical | Two profiles polling SSE simultaneously exhaust SQLite file handles. Profile switching spam every 5s from concurrent tabs. Fix: scope SSE to `profile_id`, add connection pooling. |
| B2 | Market refresh happens automatically even when background refresh is disabled | High | `schedule.background_enabled: false` in config.yaml not respected. Market refresh fires regardless. |
| B3 | Presets broken in Settings | High | Preset loading/saving not functioning. |
| B4 | Free chat mode doesn't work — responses have no spaces/formatting | High | Output: `"Heythere!Since"toppicks"canmeanafewthings..."` — text concatenated without whitespace. Streaming response handler or prompt format issue. |
| B5 | Cannot minimize Strat Agent after expanding | Medium | No way to collapse agent back from fullscreen. Missing minimize/restore toggle. |
| B6 | Agent chat history disappears on page reload | High | Chat messages not persisted. Save conversation to localStorage or DB and restore on load. |
| B7 | Fake market refresh timeout — appears to complete but didn't fetch | Medium | Progress indicator completes without real data fetch. Related to `_scan_pid` NameError in `main.py:837`. |
| B8 | Fetching phase takes longer than scoring phase | Medium | News scraping is the bottleneck (205 articles serially). Consider parallel scraping with ThreadPoolExecutor. |
| B9 | Expanded agent mode doesn't use web search and other tools | High | Tool use (web search, market lookup) disabled in fullscreen/expanded mode. Agent system prompt may differ between modes. |
| B10 | Hover states broken | Low | Identify which hover states are broken — agent responses, feed cards, or nav items. |
| B11 | `_scan_pid` NameError in `main.py:837` during market refresh | Medium | `run_market_refresh` references `_scan_pid` not defined in scope. Market refresh fails silently. |
| B12 | RSS discovery endpoint doesn't route through CF Worker proxy for blocked domains | Low | `discover-rss` fetches directly — fails for ISP-blocked sites. Should check `blocked_domains` and route through worker. |

---

## New Features

| # | Feature | Priority | Notes |
|---|---------|----------|-------|
| F1 | Location filter in wizard for location-strict categories | Medium | During category setup, add location constraint toggle. "Kuwait Careers" = location-locked, "Global Tech" = not. |
| F2 | Update wizard hardcoded categories | Medium | Wizard category presets are outdated. Refresh with current industry categories and subcategory options. |
| F3 | When clicking "Continue with AI", go to Strat Agent tab, focus it, and send the message | High | Navigate to agent tab with context from Ask AI. |
| F4 | Save scroll position per feed tab | Low | Remember scroll position. When returning to a tab, restore instead of jumping to top. |
| F5 | When no subcategories selected in wizard, unhighlight parent category | Low | Visual feedback — parent category badge should dim when all subcategories deselected. |
| F6 | Make Arcane the default theme | Low | Change default from Cosmos to Arcane for new accounts. |
| F7 | Remove Deep AI feature (A35b model removed) | Medium | Deep AI toggle references a model no longer deployed. Remove feature flag and UI toggle. |
| F8 | Make hyperlinks clickable in agent responses | Medium | Parse URLs in agent response rendering and wrap in `<a>` tags. |
| F9 | Add Google OAuth authentication | High | Google sign-in as alternative to PIN/email auth. Requires OAuth client ID, callback handler, session mapping. |
| F10 | Twitch live status detection (requires Twitch API) | Medium | RSS bridges only return VODs. Needs Twitch API client ID + OAuth token + polling. |
| F11 | Pixiv integration via RSSHub (requires refresh token) | Low | RSSHub Pixiv route needs `PIXIV_REFRESHTOKEN`. Requires extracting token from Pixiv login. |
| F12 | RSSHub as permanent systemd service | Medium | Currently `docker run -d`. Needs systemd unit for persistence across reboots. Also needs WARP connected for blocked routes. |
| F13 | Rich media feed remaining work | Medium | Non-YouTube video embeds, Twitch live badges, S/M/L toggle for articles section (currently images only). |
| F14 | Image/PDF upload with OCR → Agent pipeline | High | PaddleOCR-VL 1.5 (0.9B, ~1GB VRAM, on Ollama). Upload image/PDF → OCR extracts text → passes to Strat Agent for summarization/analysis. `ollama pull MedAIBase/PaddleOCR-VL:0.9b`. Set `OLLAMA_MAX_LOADED_MODELS=3`. See SESSION_CONTEXT §14 for full details. |
| F15 | Audio upload with Whisper transcription → Agent pipeline | Medium | Whisper.cpp on CPU (0 VRAM). Upload audio → transcribe → pass to Agent. Same pattern as OCR. |
| F16 | TCG card photo → identify + price lookup (Collection Mode) | Medium | PaddleOCR-VL extracts card name + set number from photo → query YGOPRODeck/Scryfall/PokémonTCG API → return price + metadata. Killer feature for Collection Mode. |

---

## UX / UI Polish

| # | Change | Priority | Notes |
|---|--------|----------|-------|
| U1 | Articles section not affected by S/M/L grid size toggle | Low | Toggle only applies to Images. Articles should also respond to size changes. |

---

## Completed This Session

| # | Item | Status |
|---|------|--------|
| I1 | Cloudflare Worker proxy for ISP bypass | ✅ `stratos-proxy.stratintos.workers.dev` |
| I2 | `/api/proxy` endpoint (auth-exempt) for frontend image loading | ✅ |
| I3 | Media type detection in `/api/custom-news` (video/stream/image/manga/article) | ✅ |
| I4 | Rich media view — Videos, Streams, Images (grouped, S/M/L), Articles (grouped), Lightbox | ✅ |
| I5 | YouTube channel URL auto-convert (`@handle` → RSS) via YouTube internal API | ✅ |
| I6 | Booru high-res image URLs (yande.re/konachan full res, danbooru 360px limit) | ✅ Partial |
| I7 | RSSHub self-hosted via Docker | ✅ |
| I8 | Cloudflare WARP for Docker ISP bypass | ✅ Conflicts with cloudflared — toggle as needed |
| I9 | Media feed suggestions tab in Settings | ✅ |
| I10 | Feed cleanup script — tested all feeds, kept 16 working | ✅ |
| I11 | `blocked_domains` config (19 domains) | ✅ |

---

## Summary by Priority

**Critical:** ~~B1~~ ✅

**High:** B2, B3, ~~B4~~ ✅, ~~B6~~ ✅, B9, ~~F3~~ ✅, F9, ~~F14~~ ✅

**Medium:** ~~B5~~ ✅, B7, ~~B8~~ NOT A BUG, B11, F1, F2, ~~F7~~ ALREADY REMOVED, ~~F8~~ ✅, F10, F12, F13, F15, F16

**Low:** ~~B10~~ NOT A BUG, ~~B12~~ ✅, F4, F5, ~~F6~~ ✅, F11, U1
