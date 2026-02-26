# StratOS — Claude Code Session Prompt

Use this file as the starting prompt when beginning a new Claude Code session for StratOS development. It provides complete context about the project, its current state, architecture, and conventions.

---

## Project Overview

StratOS is a self-hosted strategic intelligence dashboard. It fetches news from multiple sources (RSS, DuckDuckGo, Serper/Google), scores them for relevance using a local Ollama-hosted LLM, and presents a prioritized feed via a web UI. The system continuously improves its scoring model through a distillation loop: Claude Opus re-scores articles, disagreements become training data, and LoRA fine-tuning updates the local model.

**Scale:** 45,000+ lines across 40+ modules. Solo project.
**Working directory:** `/home/ahmad/Downloads/StratOS/StratOS1/backend`
**Frontend:** `../frontend/`

## How to Run

```bash
# Start the full server (dashboard at http://localhost:8080)
cd /home/ahmad/Downloads/StratOS/StratOS1/backend
python3 main.py --serve --background

# Or use the launcher script
bash stratos.sh

# Kill + restart
fuser -k 8080/tcp; rm -f strat_os.db-shm strat_os.db-wal; sleep 1; python3 main.py --serve --background
```

**Prerequisites**: Ollama must be running (`ollama serve`).

## Key Documentation Files

| File | Purpose |
|------|---------|
| `backend/CLAUDE.md` | Development guidance — read this FIRST |
| `backend/STRATOS_FRS_v8.md` | Complete Functional Requirements Specification (v9, ~1,800 lines) |
| `README.md` (root) | Project README with architecture, setup, features, roadmap |
| `backend/CLAUDE_CODE_SESSION_PROMPT.md` | This file |
| `backend/STRATOS_THEME_REFERENCE.md` | Theme system CSS reference (if present) |

## Technology Stack

- **Backend:** Python 3.12, built-in threaded HTTP server (no Flask/FastAPI)
- **Frontend:** Vanilla JavaScript SPA + Tailwind CSS + TradingView Lightweight Charts
- **AI:** Ollama (local), Qwen3-8B fine-tuned with DoRA, Claude Opus (teacher for distillation)
- **Database:** SQLite with WAL mode
- **Market Data:** Yahoo Finance via yfinance
- **Search:** DuckDuckGo, Serper API, Google Custom Search, RSS (30+ feeds)
- **Mobile:** Progressive Web App (PWA) with touch gestures

## Architecture Overview

```
News Sources → NewsFetcher → AI Scorer → SQLite DB → JSON API → Frontend
                                ↑                        ↓
                         Feedback Loop              User Feedback
                                ↑                        ↓
                    LoRA Training ← Export ← Distillation (Opus)
```

### Backend Structure
```
backend/
├── main.py              # StratOS class: scan pipeline, config, scheduler, entry point (1,052 lines)
├── server.py            # HTTP server, CORSHandler, route dispatch (1,436 lines)
├── database.py          # SQLite with WAL mode, singleton (602 lines)
├── auth.py              # AuthManager: profiles, sessions, rate limiting (267 lines)
├── sse.py               # SSEManager: client tracking, event broadcasting (55 lines)
├── config.yaml          # Central config (262 lines)
├── fetchers/
│   ├── news.py          # Multi-source news fetcher (922 lines)
│   ├── market.py        # Yahoo Finance via yfinance (459 lines)
│   ├── discovery.py     # Entity discovery (314 lines)
│   ├── extra_feeds.py   # RSS feed URL registry (274 lines)
│   ├── serper_search.py # Serper API client (332 lines)
│   └── google_search.py # Google Custom Search (300 lines)
├── processors/
│   ├── scorer_base.py   # Shared scorer infrastructure (847 lines)
│   ├── scorer_adaptive.py # Profile-adaptive scorer — sole active scorer (1,238 lines)
│   ├── briefing.py      # LLM briefing generator (461 lines)
│   └── profile_generator.py # AI category generation (406 lines)
├── routes/
│   ├── agent.py         # Chat agent with Ollama tool-use (829 lines)
│   ├── config.py        # /api/config save handler (193 lines)
│   ├── generate.py      # /api/generate-profile (282 lines)
│   ├── wizard.py        # Wizard backend routes (382 lines)
│   └── helpers.py       # JSON response, SSE, gzip utilities
├── distill.py           # Claude Opus distillation (601 lines)
├── export_training.py   # Training data exporter (539 lines)
├── train_lora.py        # LoRA fine-tuning pipeline (1,344 lines)
└── autopilot.py         # Autonomous self-improvement loop (959 lines)
```

### Frontend Structure
```
frontend/
├── index.html           # Main dashboard SPA (1,329 lines)
├── styles.css           # 8 themes × 2 modes + stars + mobile (1,666 lines)
├── ui.js                # Theme system, stars, dark mode, parallax, toasts (445 lines)
├── nav.js               # Sidebar navigation + resize logic (251 lines)
├── app.js               # Core: data loading, SSE, scan control (1,956 lines)
├── feed.js              # News feed rendering (784 lines)
├── market.js            # Market data display (1,053 lines)
├── markets-panel.js     # TradingView charts panel (1,877 lines)
├── agent.js             # Chat agent UI (937 lines)
├── settings.js          # Settings panel — 4 tabs (2,745 lines)
├── wizard.js            # Onboarding wizard (2,583 lines)
├── mobile.js            # Mobile gestures, bottom nav, PWA (2,154 lines)
├── auth.js              # Authentication UI (429 lines)
├── theme-editor.js      # Live theme customization (356 lines)
├── scan-history.js      # Scan log viewer (209 lines)
└── sw.js                # Service worker (92 lines)
```

## Current Theme System (8 Themes)

Themes: Midnight (default), Noir, Coffee, Rose, Cosmos, Nebula, Aurora, Sakura.
Removed: Arctic, Terminal, Latte.

Each theme has:
- Normal variant: `[data-theme="name"]` CSS selector
- Dark variant: `[data-theme="name"][data-dark="true"]` compound selector
- 20+ CSS custom properties (--bg-primary, --accent, --text-primary, --chart-line, etc.)

**"Deeper" toggle** (`data-dark` attribute): Pushes all themes to ultra-dark OLED-black. Stored in `localStorage('stratos-dark')`.

**Stars toggle**: Renders 80 animated particles via `#star-canvas`. For starry themes (Cosmos, Nebula, Aurora, Sakura), custom `--star-color-*` vars are used. For regular themes, generic white/silver. Sakura has petal-shaped particles. Star parallax: `translateY(scrollY * -0.08)`. Stored in `localStorage('stratos-stars')`.

**Theme picker**: Collapsible panel in sidebar bottom section. 2 rows of theme labels + toggles row (Deeper, Stars). Text-only labels (no colored circles).

**Font sizes**: small (14px), medium (16px), large (18px), xlarge (20px). `--ui-scale` CSS variable for icon scaling.

## Sidebar Behavior

- **Collapse button** (`toggleSidebar()`): Sets `.sidebar-collapsed` class, width=0, overflow=hidden
- **Drag resize** (handle at right edge): Min 160px, max 400px. When dragged below 200px, adds `.sidebar-narrow` class which hides `.sidebar-bottom-section` (themes, profile, status)
- **Width persisted** via `localStorage('sidebarWidth')`

## Header Toolbar Layout

```
[Last Sync]  [Refresh label]
             [Market ⟳] [News 📡] | [History] [Settings] [Help]
```

All buttons are uniform `p-1.5` icon-only squares. Market refresh uses `refresh-cw` icon, News scan uses `radar` icon (green tint). Scan states: idle (green), scanning (red, square stop icon), stopping (grey, disabled).

## Chart Focus Mode

- **Main chart**: Static "Focus Mode" text button with expand icon positioned ABOVE the chart (lines 409-415 in index.html). NOT an overlay icon.
- **Mini compare charts**: Overlay icon buttons added dynamically by `_addFullscreenBtn()` in mobile.js
- Function: `_openFullscreenChart(container, title)` — CSS `position: fixed` on existing element (no DOM movement)

## Chart Toolbar

Located above the chart in markets panel. Uses `px-2.5 py-1.5` padding and `w-4 h-4` icons (16px). Buttons: Line/Candle toggle, Crosshair, Auto Trend, Draw mode, Export. Drawing color picker with `w-5 h-5` circles.

## Settings Panel (4 Tabs)

Organized into Profile, Sources, Market, and System tabs. Supports Simple and Advanced modes.
- **Profile tab**: Role, location, context, Generate/Wizard/Suggest/Save buttons, dynamic categories
- **Sources tab**: RSS feed toggles (Finance, Politics catalogs), custom feeds, search provider config
- **Market tab**: Ticker watchlist management, ticker presets (save/load), alert thresholds
- **System tab**: Display settings (density, font size, chart type, auto-refresh), Ollama model config

## Ollama Models

| Model | Purpose | Config Key |
|-------|---------|------------|
| `stratos-scorer-v2` | Fine-tuned scoring (8.7GB Q8_0) | `scoring.model` |
| `qwen3:30b-a3b` | Agent chat, suggestions, briefings | `scoring.inference_model` |
| `qwen3:14b` | Wizard (lighter, faster) | `scoring.wizard_model` |

## Key Design Decisions

1. **Qwen3 think blocks**: NEVER set `"think": False` — Qwen3 ignores it. Omit the parameter entirely. Always strip `<think>...</think>` from content as safety net.
2. **Training/inference alignment**: `export_training.py` must produce messages character-for-character identical to `scorer_adaptive.py` inference format.
3. **PEFT meta device fix**: After `get_peft_model()`, delete `hf_device_map` and force `.to("cuda:0")`.
4. **AOTRITON**: Do NOT set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` — causes NaN gradients on 7900 XTX.
5. **Streaming Ollama**: All scoring calls use `stream=True` with `cancel_check` every 10 chunks for responsive cancellation.
6. **No web framework**: Pure stdlib `http.server` with ThreadingMixIn. All routing/middleware inline.

## Secrets

All API keys in `backend/.env` (gitignored):
```
ANTHROPIC_API_KEY=sk-ant-...
SERPER_API_KEY=...
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
```

## Hardware

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 7 7800X3D |
| GPU | AMD Radeon 7900 XTX (24GB VRAM) |
| RAM | 32GB DDR5-6000 CL30 |
| OS | Ubuntu 24 (dual-boot) |
| GPU Compute | AMD ROCm 6.2 |

## Ollama Environment Variables

```bash
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_KEEP_ALIVE=10m
OLLAMA_NUM_PARALLEL=1
OLLAMA_FLASH_ATTENTION=0
ROCR_VISIBLE_DEVICES=0
```

## API Endpoints

Key endpoints: `/api/news`, `/api/market`, `/api/refresh` (trigger scan), `/api/scan/status`, `/api/scan/cancel`, `/api/status`, `/api/feedback`, `/api/agent-chat`, `/api/suggest-context`, `/api/generate-profile`, `/api/config`, `/api/events` (SSE), `/api/briefing`, `/api/wizard-preselect`, `/api/wizard-tab-suggest`, `/api/wizard-rv-items`, `/api/update-profile`, `/api/export`.

## Recent Changes (v6.0 / FRS v9)

1. **Theme overhaul**: Removed Arctic, Terminal, Latte. Added Cosmos, Nebula, Aurora, Sakura. Added dark mode ("Deeper") toggle, star animations with parallax, sakura petal particles.
2. **Settings reorganization**: 4-tab layout (Profile, Sources, Market, System). Ticker presets.
3. **Header toolbar alignment**: Icon-only refresh buttons (Market + News scan) aligned with History/Settings/Help in one uniform row.
4. **Focus Mode**: Text button with expand icon above main chart (not overlay icon).
5. **Chart toolbar**: Enlarged buttons (px-2.5 py-1.5, w-4 h-4 icons).
6. **Sidebar narrow collapse**: Bottom section (themes, profile) hides when sidebar dragged below 200px.
7. **Extra Large font**: 20px option in System settings, scales everything including icons.
8. **Collapsible theme picker**: Chevron indicator, toggle on/off in sidebar.

## Planned Features (Roadmap)

- Email-based authentication (replace PIN auth)
- Cross-device profile sync
- Multi-user household support
- Email notifications (critical alerts, daily/weekly digest)
- Cloud backup & restore
- V3 scorer (50-profile pipeline)
- Additional data sources (LinkedIn, government portals)

## Common Tasks

### Restart the server
```bash
cd /home/ahmad/Downloads/StratOS/StratOS1/backend
fuser -k 8080/tcp; rm -f strat_os.db-shm strat_os.db-wal; sleep 1; python3 main.py --serve --background
```

### Run a learning cycle
```bash
bash learn_cycle.sh
```

### Check the database
```bash
sqlite3 strat_os.db ".tables"
sqlite3 strat_os.db "SELECT COUNT(*) FROM news_items;"
```

---

*This prompt provides sufficient context for a new Claude Code session to continue development on StratOS without prior conversation history. For the complete specification, refer to STRATOS_FRS_v8.md (v9).*
