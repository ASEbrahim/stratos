
# StratOS Backlog — March 2026

---

## Bugs & Fixes & Additions

| # | Issue | Severity | Notes |
|---|-------|----------|-------|
| B1 | Ctrl+C & CTRL+V do not work in Strat Agent and instead triggers chart type toggle instead of copy/paste | High | Keyboard shortcut conflict — chart shortcuts intercepting inside agent textarea. Shortcuts should be suppressed when focus is in agent input. |
| B2 | Scan loading bar appears for both profiles when only one is scanning | High | SSE scan progress events not scoped to profile — frontend shows bar globally instead of per-profile. |
| B3 | Scan on Ahmad's instance showed results in Kirissie's instance (this only happens on Ahmads account and did not happen when I created a new account. If Ahmads account is broken just remove it entirely from the db and I'll recreate the account) | Critical | Cross-instance data bleed. Two StratOS instances sharing state — likely same browser session or shared auth token/localStorage. Related to previous D025–D026 fixes but across browser tabs/instances. |
| B4 | Tutorial popup didn't deploy after creating a new account | Medium | Tour trigger not firing post-registration. Auto-created default profile on registration should trigger tour — check if `stratos_tour_never` is being set prematurely or tour init races with profile creation. |
| B5 | "Expand price scale" in summary chart doesn't work well | Medium | Price scale auto-fit or manual expand misbehaving in markets-panel summary view. |
| B6 | Strat Agent gives "Max tool rounds exceeded" sometimes | Medium | Agent hitting the tool round cap in `agent.py`. Either increase max rounds, optimize tool call chains, or add a retry/continue mechanism. |
| B7 | Skip
| B8 | Skip
| B9 | Custom RSS tab doesn't appear until page refresh after saving the first feed | Medium | Feed tab not dynamically added to DOM on first custom RSS save — requires nav rebuild or reactive tab insertion. |

---

## New Features

| # | Feature | Priority | Notes |
|---|---------|----------|-------|
| F1 | Interactive map with add-ons in Politics section + optional live feed | High | Geopolitical map widget — show politics signals on a world/regional map. Could use Leaflet or Mapbox. Live feed overlay for real-time political events. |
| F2 | Suggestion responses from Strat Agent / Market replies — press tab to apply | High | After agent or market analysis replies, show actionable suggestion chips (e.g. "Add to watchlist", "Track this category", "Set alert"). One-tap to execute. |
| F3 | Full-screen mode for Strat Agent | Medium | Desktop already has mobile full-screen agent. Add a desktop expand/fullscreen toggle for the agent panel (similar to chart focus mode). |
| F4 | Create a Jobs category in the feeds and include Indeed and other job listing sources | Medium | Add job board scrapers/RSS (Indeed, LinkedIn Jobs, Bayt, GulfTalent) as feed sources, especially for career-relevant profiles. |
| F5 | Toggle between free chat and structured format in Strat Agent | Medium | Let users switch between freeform conversation and the current structured tool-use format. Free mode = no system prompt constraints, just direct LLM chat. |
| F6 | Replace auto trend lines with Fibonacci retracement or custom calculation | High | Auto trend lines in summary view aren't useful. Replace with fib retracement or a novel analytical calculation + written analysis. |
| F7 | Convert links to RSS in custom RSS feeds | Low | Auto-detect and convert regular URLs to RSS feed URLs (find `<link rel="alternate" type="application/rss+xml">` or use feed discovery). |
| F8 | Custom RSS: support live feeds, videos, images, or bullet-point toggle | Medium | Extend custom RSS rendering to handle media-rich feeds — show embedded video/images, or toggle to compact bullet-point view like the built-in feeds. |
| F9 | RSS options relevant to each feed tab (Finance, Politics, Custom) | Medium | Show contextual RSS source suggestions in a right-side panel when a feed tab is active. Finance tab → financial RSS options, Politics tab → political RSS options, etc. |
| F10 | Continue conversation with Strat Agent from Ask AI feature | High | "Open in Agent" button on Ask AI responses. Agent loads the full Ask AI conversation context so the user can keep going without re-explaining. |
| F11 | Collapsible sections for Markets and Asset Analysis in summary/feeds | Low | Add expand/collapse toggles to Markets panel and Asset Analysis sections on summary and feed pages to reduce visual clutter. |

---

## UX / UI Polish

| # | Change | Priority | Notes |
|---|--------|----------|-------|
| U1 | Star and normal buttons should blink | Medium | Add subtle blink/pulse animation to star (favorite) buttons and standard action buttons for visual feedback. |
| U2 | Show article publish date instead of current time next to signals | High | Signal timestamps currently show fetch/display time. Replace with the article's actual publication date from the RSS/scraper metadata. |
| U3 | Ask AI needs user context (role + location at minimum) | High | `handle_ask` in agent.py sends article context but no user profile context. Add role, location, and key categories to the Ask AI system prompt. |
| U4 | Customize button should blink/pulse | Low | Add attention-drawing animation to the theme customize button (subtle glow pulse). |
| U5 | Bell icon should glow and blink | Low | Notification bell gets a persistent glow + blink animation when there are unread notifications or new scan results. |
| U6 | Change StratOS logo (the eye thing doesn't look good) maybe try blending all the themes together and see what comes out with multiple prototypes?| Medium | New logo design needed — current logo to be replaced. |
| U7 | Coffee theme: invert to primary colors of Cosmos | Medium | Redesign Coffee theme as the color-inverted counterpart of Cosmos. Cosmos is dark blue/gold — Coffee should become warm/light with inverted accent palette. |
| U8 | Cosmos solar system: make orbits slightly clearer | Low | Increase orbit line opacity or width in the Cosmos theme's solar system animation for better visibility. |
| U9 | Refine categories after scoring | Medium | Post-scan category refinement — let users (or auto-suggest) merge, split, or rename categories based on what the scoring revealed. | (Dont apply yet, just a concept)!!!!!

---

## Summary by Priority

**Critical (fix immediately):**
- B3 — Cross-instance data bleed between Kirissie and Ahmad (or delete Ahmad entirely)

**High (next sprint):**
- B1 — Ctrl+C & CTRL+V conflict in agent
- B2 — Loading bar shows for both profiles
- F1 — Politics interactive map
- F2 — Suggestion responses (press to apply)
- F6 — Replace auto trend lines
- F10 — Continue Ask AI conversation in Agent
- U2 — Show article publish date on signals
- U3 — Ask AI needs user context

**Medium (upcoming):**
- B4 — Tutorial popup not deploying
- B5 — Price scale expand broken
- B6 — Max tool rounds exceeded
- B7 — SKIP
- B9 — Custom RSS tab refresh needed
- F3 — Agent fullscreen (desktop)
- F4 — Job listings (Indeed etc.)
- F5 — Agent free chat toggle
- F8 — Custom RSS media support
- F9 — Contextual RSS suggestions per tab
- U1 — Button blink animations
- U6 — New logo
- U7 — Coffee theme redesign
- U9 — Post-scoring category refinement

**Low (backlog):**
- B8 — SKIP
- F7 — Auto-convert links to RSS
- F11 — Collapsible markets/analysis sections
- U4 — Customize button blink
- U5 — Bell glow/blink
- U8 — Cosmos orbit visibility
