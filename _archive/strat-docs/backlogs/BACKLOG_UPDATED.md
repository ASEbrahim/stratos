# StratOS Backlog — March 2026 (Updated)

---

## Bugs & Fixes

| # | Issue | Severity | Notes |
|---|-------|----------|-------|
| B1 | Ctrl+C / Ctrl+V intercepted by chart shortcuts inside Strat Agent textarea | High | Keyboard shortcuts should be suppressed when focus is in agent input. |
| B2 | Scan loading bar appears for both profiles when only one is scanning | High | SSE scan progress events not scoped to profile — frontend shows bar globally instead of per-profile. |
| B3 | Cross-profile data bleed (Ahmad→Kirissie) + SQLite `unable to open database file` crashes | Critical | Two profiles polling SSE simultaneously exhaust SQLite file handles. Root cause: profile switching spam every 5s from concurrent tabs. Fix: scope SSE to `profile_id`, add connection pooling or WAL mode hardening. |
| B4 | Tutorial popup doesn't deploy after creating a new account | Medium | Tour trigger not firing post-registration. Check if `stratos_tour_never` is set prematurely or tour init races with profile creation. |
| B5 | "Expand price scale" in summary chart doesn't work well | Medium | Price scale auto-fit or manual expand misbehaving in markets-panel summary view. |
| B6 | Strat Agent gives "Max tool rounds exceeded" sometimes | Medium | Agent hitting the tool round cap in `agent.py`. Either increase max rounds, optimize tool call chains, or add a retry/continue mechanism. |
| B9 | Custom RSS tab doesn't appear until page refresh after saving the first feed | Medium | Feed tab not dynamically added to DOM on first custom RSS save — requires nav rebuild or reactive tab insertion. |
| B10 | Market refresh happens automatically even when background refresh is disabled | High | `schedule.background_enabled: false` in config.yaml is not being respected. Market refresh fires regardless of the toggle state. |
| B11 | Presets broken in Settings | High | Preset loading/saving not functioning. Needs investigation in settings.js preset handling. |
| B12 | Free chat mode doesn't work — responses have no spaces/formatting | High | Free chat toggle in agent produces unformatted output: `"Heythere!Since"toppicks"canmeanafewthings..."` — text concatenated without whitespace. Likely a streaming response handler or prompt format issue. |
| B13 | Cannot minimize Strat Agent after expanding | Medium | Once agent panel is expanded to fullscreen, no way to collapse it back. Missing minimize/restore toggle. |
| B14 | Agent chat history disappears on page reload | High | Chat messages not persisted. Need to save agent conversation to localStorage or DB and restore on load. |
| B15 | Fake market refresh timeout — refresh appears to complete but didn't actually fetch | Medium | Market refresh progress indicator completes without real data fetch. May be related to `_scan_pid` NameError in `main.py:837`. |
| B16 | Fetching phase takes longer than the scoring phase | Medium | News scraping is the bottleneck (205 articles scraped serially). Consider parallel scraping with ThreadPoolExecutor or increase concurrent connections. |
| B17 | Expanded agent mode doesn't use web search and other tools | High | When agent is in fullscreen/expanded mode, tool use (web search, market lookup, etc.) is disabled or not triggered. Agent system prompt may differ between modes. |
| B18 | Hover states broken (unspecified elements) | Low | Need to identify which hover states are broken — likely agent responses, feed cards, or nav items. |
| B19 | `_scan_pid` NameError in `main.py:837` during market refresh | Medium | `run_market_refresh` references `_scan_pid` which is not defined in that scope. Causes market refresh to fail silently. |
| B20 | RSS discovery endpoint doesn't route through CF Worker proxy for blocked domains | Low | `discover-rss` fetches directly — fails for ISP-blocked sites like yande.re, danbooru. Should check `blocked_domains` and route through worker. Only affects auto-detect; manual URL entry works fine. |

---

## New Features

| # | Feature | Priority | Notes |
|---|---------|----------|-------|
| F1 | Interactive map with add-ons in Politics section + optional live feed | High | Geopolitical map widget — show politics signals on a world/regional map. Could use Leaflet or Mapbox. |
| F2 | Suggestion responses from Strat Agent / Market replies — press tab to apply | High | Actionable suggestion chips (e.g. "Add to watchlist", "Track this category"). One-tap to execute. |
| F3 | Full-screen mode for Strat Agent (desktop) | Medium | Add desktop expand/fullscreen toggle for the agent panel. |
| F4 | Create a Jobs category in feeds with Indeed, LinkedIn Jobs, Bayt, GulfTalent | Medium | Add job board scrapers/RSS as feed sources for career-relevant profiles. |
| F5 | Toggle between free chat and structured format in Strat Agent | Medium | Free mode = no system prompt constraints, just direct LLM chat. Currently broken (B12). |
| F6 | Replace auto trend lines with Fibonacci retracement or custom calculation | High | Auto trend lines aren't useful. Replace with fib retracement + written analysis. |
| F7 | Convert links to RSS in custom RSS feeds | Low | Auto-detect and convert regular URLs to RSS feed URLs. YouTube channel URL auto-convert is now implemented (✅). |
| F8 | Custom RSS: rich media feed with videos, images, streams | Medium | **Partially implemented** — media view with Videos/Images/Streams/Articles sections, lightbox, S/M/L grid toggle. Still needs: video embed for non-YouTube sources, Twitch live status via API. |
| F9 | RSS options relevant to each feed tab (Finance, Politics, Custom) | Medium | Contextual RSS source suggestions per tab. Media suggestions tab added (✅). |
| F10 | Continue conversation with Strat Agent from Ask AI feature | High | "Open in Agent" button on Ask AI responses. Should navigate to agent tab, focus it, and send the message with context. |
| F11 | Collapsible sections for Markets and Asset Analysis in summary/feeds | Low | Add expand/collapse toggles to reduce visual clutter. |
| F12 | Location filter in wizard for location-strict categories | Medium | During category setup in wizard, add a location constraint toggle. Categories like "Kuwait Careers" should be location-locked, while "Global Tech" should not. Wizard should present this choice. |
| F13 | Update wizard hardcoded categories | Medium | Wizard category presets are outdated. Refresh with current industry categories and subcategory options. |
| F14 | Save position of unique item (scroll position persistence) | Low | Remember scroll position per feed tab. When switching between tabs and returning, restore the previous scroll position instead of jumping to top. |
| F15 | When no subcategories are selected in wizard, unhighlight parent category | Low | Visual feedback fix — parent category badge should dim when all its subcategories are deselected. |
| F16 | Make Arcane the default theme | Low | Change default theme from Cosmos to Arcane for new accounts. |
| F17 | Remove Deep AI feature (A35b model removed) | Medium | Deep AI toggle references a model that's no longer deployed. Remove the feature flag and UI toggle. |
| F18 | Make hyperlinks clickable in agent responses | Medium | Agent responses should render URLs as clickable links. Currently displayed as plain text. Parse URLs in agent response rendering and wrap in `<a>` tags. |
| F19 | Add Google OAuth authentication | High | Add Google sign-in as an alternative to PIN/email auth. Requires Google OAuth client ID, callback handler, and session mapping. |
| F20 | Twitch live status detection (requires Twitch API) | Medium | RSS bridges only return VODs. Live/offline status needs Twitch API client ID + OAuth token + polling endpoint. |
| F21 | Pixiv integration via RSSHub (requires refresh token) | Low | RSSHub Pixiv route needs `PIXIV_REFRESHTOKEN` environment variable. Requires extracting token from Pixiv login session. |
| F22 | Self-hosted RSSHub (Docker) as permanent service | Medium | RSSHub is running via Docker (`docker run -d --name rsshub --network host diygod/rsshub`) but needs to be a systemd service for persistence across reboots. Also needs WARP connected for ISP-blocked routes to work. |

---

## UX / UI Polish

| # | Change | Priority | Notes |
|---|--------|----------|-------|
| U1 | Star and normal buttons should blink | Medium | Add subtle blink/pulse animation to star buttons and standard action buttons. |
| U2 | Show article publish date instead of current time next to signals | High | Signal timestamps currently show fetch/display time. Replace with article's actual publication date from RSS metadata. |
| U3 | Ask AI needs user context (role + location at minimum) | High | `handle_ask` in agent.py sends article context but no user profile context. Add role, location, and key categories to the Ask AI system prompt. |
| U4 | Customize button should blink/pulse | Low | Attention-drawing animation on theme customize button. |
| U5 | Bell icon should glow and blink | Low | Notification bell gets persistent glow + blink when unread notifications exist. |
| U6 | Change StratOS logo — current eye design doesn't look good | Medium | New logo design needed. Try blending all themes together for prototypes. |
| U7 | Coffee theme: invert to primary colors of Cosmos | Medium | Coffee should become warm/light inverted counterpart of Cosmos. |
| U8 | Cosmos solar system: make orbits slightly clearer | Low | Increase orbit line opacity/width. |
| U9 | Refine categories after scoring (concept only — don't implement yet) | Medium | Post-scan category refinement — let users merge/split/rename categories based on scoring results. |
| U10 | Custom feed articles section not affected by S/M/L grid size toggle | Low | S/M/L toggle only applies to Images section. Articles section should also respond to size changes. |

---

## Infrastructure / Completed This Session

| # | Item | Status | Notes |
|---|------|--------|-------|
| I1 | Cloudflare Worker proxy for ISP bypass | ✅ Done | `stratos-proxy.stratintos.workers.dev` — proxies blocked domains (boorus, manga sites). Config-driven allowlist. |
| I2 | `/api/proxy` endpoint for frontend image loading | ✅ Done | Exempt from auth. Routes blocked thumbnails through CF Worker. |
| I3 | Media type detection in `/api/custom-news` | ✅ Done | Detects video/stream/image/manga/article from URL patterns. Extracts YouTube embed IDs, Twitch channel names. |
| I4 | Rich media view (feed.js) | ✅ Done | Videos (horizontal cards, click-to-play YouTube), Streams (Twitch embeds), Images (grouped by source, S/M/L grid), Articles (grouped by source). Lightbox with keyboard navigation. |
| I5 | YouTube channel URL auto-convert | ✅ Done | Paste `youtube.com/@handle` in Add Feed → auto-resolves channel ID via YouTube internal API → adds RSS feed. |
| I6 | Booru high-res image URLs | ✅ Partial | Yande.re/Konachan `/preview/` → `/image/` (full res). Danbooru limited to 360px (paid API required for originals). |
| I7 | RSSHub self-hosted via Docker | ✅ Done | `docker run -d --name rsshub --network host diygod/rsshub`. Requires WARP connected for blocked sites. |
| I8 | Cloudflare WARP for Docker ISP bypass | ✅ Done | `warp-cli connect/disconnect`. Conflicts with cloudflared tunnels — must disconnect WARP before running tunnel. |
| I9 | Media feed suggestions in Settings | ✅ Done | "Media" tab in RSS suggestions with YouTube channels, booru feeds, manga sources. |
| I10 | Config cleanup script | ✅ Done | Tested all feeds, removed broken ones, kept 16 working feeds for Strat profile. |
| I11 | `blocked_domains` in config.yaml | ✅ Done | 19 domains including boorus, manga, CDNs. YAML fixed after sed corruption. |

---

## Summary by Priority

**Critical:**
- B3 — Profile switching spam → SQLite crash + data bleed

**High (next sprint):**
- B10 — Auto market refresh ignoring disabled flag
- B11 — Presets broken
- B12 — Free chat mode output unformatted
- B14 — Agent chat lost on reload
- B17 — Expanded agent missing tools
- F6 — Replace auto trend lines
- F10 — Continue Ask AI → Agent with context
- F19 — Google OAuth
- U2 — Show article publish date
- U3 — Ask AI needs user context

**Medium:**
- B4 — Tutorial not deploying
- B5 — Price scale expand
- B6 — Max tool rounds
- B13 — Can't minimize expanded agent
- B15 — Fake market refresh timeout
- B16 — Fetching slower than scoring
- B19 — `_scan_pid` NameError
- F3 — Agent fullscreen desktop
- F8 — Rich media feed (remaining work)
- F12 — Location filter in wizard
- F13 — Update wizard hardcoded categories
- F17 — Remove Deep AI feature
- F18 — Hyperlinks in agent responses
- F22 — RSSHub as systemd service

**Low:**
- B18 — Hover states
- B20 — Discovery endpoint proxy
- F7 — Link to RSS conversion (YouTube done)
- F11 — Collapsible sections
- F14 — Save scroll position
- F15 — Wizard subcategory highlighting
- F16 — Default theme to Arcane
- U10 — S/M/L toggle for articles
