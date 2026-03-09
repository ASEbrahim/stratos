# StratOS Backlog — March 2026

---

## Bugs & Fixes

| # | Issue | Severity | Notes |
|---|-------|----------|-------|
| B1 | Cross-profile data bleed + SQLite `unable to open database file` crashes | Critical | Two profiles polling SSE simultaneously exhaust SQLite file handles. Profile switching spam every 5s from concurrent tabs. Fix: scope SSE to `profile_id`, add connection pooling. |
| B2 | Market refresh happens automatically even when background refresh is disabled | High | `schedule.background_enabled: false` in config.yaml not respected. Market refresh fires regardless. |
| B3 | Presets broken in Settings | High | Preset loading/saving not functioning. |
| B4 | Free chat mode doesn't work well — responses have no spaces/formatting | High | Output: `"Heythere!Since"toppicks"canmeanafewthings..."` — text concatenated without whitespace. Streaming response handler or prompt format issue. |
| B5 | Cannot minimize Strat Agent after expanding (the top bar disappears once the chat gets long enough) | Medium | No way to collapse agent back from fullscreen. Missing minimize/restore toggle. |
| B6 | Agent chat history disappears on page reload | High | Chat messages not persisted. Save conversation to localStorage or DB and restore on load. |
| B7 | Fake market refresh timeout — the market actually refreshes but gives this error anyway | Medium | Progress indicator completes without real data fetch. Related to `_scan_pid` NameError in `main.py:837`. |
| B8 | Fetching phase takes longer than scoring phase | Medium | News scraping is the bottleneck (205 articles serially) make sure NOT to trade fidelity when considering fixes.. Consider parallel scraping with ThreadPoolExecutor. |
| B9 | Expanded agent mode doesn't use web search and other tools | High | Tool use (web search, market lookup) disabled in fullscreen/expanded mode. Agent system prompt may differ between modes. |
| B10 | Hover states broken (badly positioned in wizard) | Low | Identify which hover states are broken — agent responses, feed cards, or nav items. |
| B11 | `_scan_pid` NameError in `main.py:837` during market refresh | Medium | `run_market_refresh` references `_scan_pid` not defined in scope. Market refresh fails silently. |
| B12 | RSS discovery endpoint doesn't route through CF Worker proxy for blocked domains (added a wrap using cloudflare but must add a toggle button for it)| Low | `discover-rss` fetches directly — fails for ISP-blocked sites. Should check `blocked_domains` and route through worker. |

---

## New Features

| # | Feature | Priority | Notes |
|---|---------|----------|-------|
| F1 | Location filter in wizard for location-strict categories | Medium | During category setup, add location constraint toggle. "Kuwait Careers" = location-locked, "Global Tech" = not. |
| F2 | Update wizard hardcoded categories and selection process | High | Wizard category presets are outdated. Refresh with current industry categories and subcategory options. |
| F3 | When clicking "Continue with AI", go to Strat Agent tab, focus it, and send the message | High | Navigate to agent tab with context from Ask AI. |
| F4 | Save scroll position per feed tab | Low | Remember scroll position. When returning to a tab, restore instead of jumping to top. |
| F5 | When no subcategories selected in wizard, unhighlight parent category | Low | Visual feedback — parent category badge should dim when all subcategories deselected. |
| F6 | Make Arcane the default theme | Low | Change default from Cosmos to Arcane for new accounts. |
| F7 | Remove Deep AI feature in wizard (Qwen3-A35b model removed) | Medium | Deep AI toggle references a model no longer deployed. Remove feature flag and UI toggle. |
| F8 | Make hyperlinks clickable in agent responses | Medium | Parse URLs in agent response rendering and wrap in `<a>` tags. |
| F9 | Add Google OAuth authentication | High | Google sign-in as alternative to PIN/email auth. Requires OAuth client ID, callback handler, session mapping. |
| F10 | Twitch live status detection (requires Twitch API) | Medium | RSS bridges only return VODs. Needs Twitch API client ID + OAuth token + polling. |
| F11 | Pixiv integration via RSSHub (requires refresh token) | Low | RSSHub Pixiv route needs `PIXIV_REFRESHTOKEN`. Requires extracting token from Pixiv login. | (FOR TWITCH AND PIXIV AND ANY OTHER WEBSITE THAT REQUIRES OR SUGGESTS A LOGIN, PROMPT THE USER TO LOG IN USING THEIR ACC)
| F12 | RSSHub as permanent systemd service | Medium | Currently `docker run -d`. Needs systemd unit for persistence across reboots. Also needs WARP connected for blocked routes. |
| F13 | Rich media feed remaining work | Medium | Fixing the preview and how everything looks (polish) |

---

## UX / UI Polish

| # | Change | Priority | Notes |
|---|--------|----------|-------|
| U1 | Custom RSS area does not look good | Low | Size toggle should apply to everything. Articles should also respond to size changes. |

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

Any additional recommended features by YOU are welcome to suggest.
