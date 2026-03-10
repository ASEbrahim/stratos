# STRAT_OS — Strategic Intelligence Operating System

A self-hosted intelligence dashboard that hunts for actionable signals across markets, careers, and emerging tech — with a self-improving AI scorer that learns from teacher models (Gemini Flash + Claude Opus) and your own feedback.

> Built as a solo project by a Computer Engineering student in Kuwait. 61,000+ lines across 40+ modules. Runs entirely on free, local tools — with a $6 training pipeline (Gemini Flash) that permanently improves the local model. Three scorer generations, each cheaper and better. Installable as a Progressive Web App on any device.

---

## What It Does

STRAT_OS scores everything it finds on a 0–10 scale of **"how actionable is this for my future?"** and throws away the rest.

**Five verticals, one dashboard:**

- **Markets** — Real-time candlestick/line/area charts with 20 drawing tools, Fibonacci retracement, pattern detection, collapsible asset analysis, and side-by-side comparison across stocks, crypto, and commodities. Fullscreen focus mode on desktop and mobile.
- **Career & Certifications** — Targeted job postings, certification requirements, and fresh graduate openings at specific companies. Dedicated Jobs tab with career-focused RSS feeds.
- **Emerging Tech** — Tracks breakthrough technologies scored by investment and career relevance.
- **Financial Advantage** — Bank deals, student offers, investment opportunities.
- **Regional Industry** — GCC-specific developments, government projects.
- **Rich Media** — Video/stream/image/manga detection with grid view, RSS auto-discovery, contextual feed suggestions, and Cloudflare Worker proxy for media sources.

Everything is profile-aware. The scoring adapts to who you are. Trained on 45 profiles across 20+ countries.

---

## Architecture

```
┌─────────────────────────────────────┐     ┌───────────────────────────────┐
│         BACKEND (Python)            │     │   FRONTEND (Vanilla JS)        │
│                                     │     │                                │
│  Fetchers ──► Scorer ──► JSON ──SSE─┼────►  Dashboard (desktop + mobile)  │
│  ├─ Market (yfinance)               │     │  ├─ Executive Summary          │
│  ├─ News (DDG + Serper + RSS)       │     │  ├─ Markets Panel + Focus Mode │
│  ├─ Discovery (entity detect)       │     │  ├─ Strat Agent Chat           │
│  ├─ CF Worker proxy (media/RSS)     │     │  ├─ Wizard (4-step onboarding) │
│  └─ Kuwait scrapers                 │     │  ├─ Settings / 24 Theme Vars   │
│                                     │     │  ├─ Rich Media (video/manga)   │
│  Ollama (local LLM scoring)  ◄──┐   │     │  └─ Mobile (gestures, PWA)     │
│  SQLite (dedup + feedback)      │   │     │                                │
│  DB-Auth (email verify)         │   │     │  TradingView Charts            │
│  Stop Scan (graceful cancel)    │   │     │  Vanilla CSS (24 variants)     │
│                                 │   │     │  Service Worker (offline)       │
│  ┌─ SELF-IMPROVING PIPELINE ────┘   │     └───────────────────────────────┘
│  │  User feedback (save/dismiss) ───┤
│  │  Gemini Flash / Opus distill ────┤
│  │  DoRA fine-tuning (ROCm/CUDA) ──►│ stratos-scorer-v2.2.gguf
│  └──────────────────────────────────┘
│
└──── Cloudflare Tunnel (public access)
```

---

## Self-Improving Scorer

This is the core differentiator. The scoring model gets smarter over time through two feedback channels and three training generations:

### V2.2 Scorer Performance (Production)

The current V2.2 scorer (Qwen3.5-9B + DoRA) was trained on 11,785 Gemini Flash-scored examples across 45 profiles:

| Metric | V1 | V2 | V2.2 (Production) |
|--------|-----|-----|-------------------|
| Direction Accuracy | 90.7% | 91.7% | **94.5%** |
| MAE (Opus holdout) | 1.553 | 1.544 | **0.914** |
| Within 1 Point | — | 53.4% | **75.5%** |
| Spearman ρ | — | 0.555 | **0.691** |
| Parse Failures | — | 5.0% | **0.0%** |
| Training Cost | ~$52 (Opus) | ~$52 (Opus) | **$6 (Gemini Flash)** |

### Tier 1: Implicit Feedback Loop (always active)
Every interaction teaches the system: **save** = positive signal, **dismiss** = negative signal, **rate** = explicit correction, **click** = implicit interest. These accumulate in SQLite and get injected into the LLM scoring prompt as personalized few-shot examples.

### Tier 2: Model Distillation (manual or scheduled)
Dual teacher pipeline: **Gemini Flash** ($0.00012/article) for bulk training data, **Claude Opus** for high-precision holdout evaluation. Disagreements become corrections that permanently improve the local model's weights via DoRA fine-tuning.

```
                         ┌─────────────┐
  User interactions ────►│  Feedback DB │◄──── Gemini/Opus corrections
  (save/dismiss/rate)    └──────┬──────┘      (distill.py / v22_expansion.py)
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
           Prompt injection          Training data export
           (next scan cycle)         (export_training.py)
                                            │
                                            ▼
                                    DoRA fine-tuning
                                    (train_lora.py / data/v2_pipeline/)
                                            │
                                            ▼
                                    stratos-scorer-v2.2
                                    (Ollama GGUF model)
```

---

## Quickstart

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/download)

### Setup

```bash
# Windows:
setup.bat

# Linux/Mac:
chmod +x setup.sh && ./setup.sh
```

### Run

```bash
# Windows:
start.bat

# Linux/Mac:
chmod +x start.sh && ./start.sh

# Or manually:
cd backend && python main.py --serve
```

Open `http://localhost:8080` — register with email + password on first visit (or PIN auth for legacy profiles).

### Install as App (PWA)

STRAT_OS is a Progressive Web App. On your phone or desktop:
- **Chrome/Edge:** Click the install icon in the address bar, or the "Install" banner that appears
- **iOS Safari:** Tap Share → Add to Home Screen
- **Android Chrome:** Tap the "Install STRAT_OS" banner

The app works offline for cached data and provides a native app experience (no browser chrome).

### Public Access (Cloudflare Tunnel)

```bash
cloudflared tunnel --url http://127.0.0.1:8080
# Use 127.0.0.1, not localhost (avoids IPv6 issues on Windows)
```

---

## Distillation & Training Pipeline

### One-Time Setup (Ubuntu + AMD GPU)

```bash
cd backend
chmod +x setup_rocm_training.sh
./setup_rocm_training.sh      # Installs ROCm, PyTorch, PEFT, llama.cpp

# Also install Ollama on Linux:
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3.5:9b
```

### Running a Learning Cycle

```bash
cd backend

# Step 1: Distillation — Claude Opus re-scores local model's work (~$0.40)
python3 distill.py --dry-run        # Preview cost first
python3 distill.py                   # Run for real

# Step 2: Export corrections as training data
python3 export_training.py

# Step 3: DoRA fine-tune (auto-selects best model for your GPU)
python3 train_lora.py

# Step 4: Update config.yaml
# scoring:
#   model: stratos-scorer-v2
```

Or just run `./learn_cycle.sh` (Linux) or `learn_cycle.bat` (Windows) to do all steps automatically.

### Training Hardware

V2.2 training uses Qwen3.5-9B with DoRA adapters. Requires 20+ GB VRAM:

| Component | Spec |
|-----------|------|
| GPU | AMD RX 7900 XTX (24GB VRAM) |
| Base Model | Qwen3.5-9B |
| Fine-Tuning | DoRA, rank 16, alpha 32 |
| Precision | bf16 with gradient checkpointing |
| VRAM Usage | 20-21GB peak |
| Training Time | ~7 hours (11,785 examples) |
| Compute | ROCm 6.2 (PyTorch + PEFT) |

### API Key Setup

Get a key from [platform.claude.com](https://platform.claude.com), then:

```bash
# Create backend/.env with:
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
```

### V2.2 Pipeline (Large-Scale Training)

For full-scale training across many profiles, the V2.2 pipeline in `data/v2_pipeline/` runs five phases:

1. **Extract** (`v22_expansion.py --phase 1`) — Pull 2,000 articles from production DB
2. **Score** (`v22_expansion.py --phase 2`) — Score with Gemini 3 Flash (~9,500/day, auto-resumes)
3. **Calibrate** (`v22_expansion.py --phase 3`) — Apply isotonic calibration to Gemini scores
4. **Prepare** (`v22_expansion.py --phase 5`) — Merge with existing Opus scores, build training JSONL
5. **Train** (`train_v2.py`) — DoRA fine-tune with weighted sampling

**V2.2 training stats:**
- 11,785 scored examples across 45 profiles
- Dual teacher: Gemini Flash (bulk) + Claude Opus (holdout evaluation)
- Cost: **$6** for the full Gemini scoring pipeline (10x cheaper than V2's $52 Opus pipeline)
- Train time: 7 hours on AMD Radeon 7900 XTX (24GB VRAM)
- Result: 40% MAE reduction over V2, zero parse failures

---

## Project Structure

```
StratOS/
├── backend/
│   ├── main.py                    # HTTP server, SSE, auth, orchestration
│   ├── server.py                  # Route dispatch, CORSHandler
│   ├── database.py                # SQLite — dedup, history, feedback
│   ├── auth.py                    # Session management, PIN auth, rate limiting
│   ├── email_service.py           # SMTP delivery, verification/reset emails
│   ├── user_data.py               # Per-user data directories (JSONL exports)
│   ├── sse.py                     # Server-Sent Events broadcasting
│   ├── migrations.py              # DB schema migrations (version 9)
│   ├── config.yaml                # Global configuration
│   ├── distill.py                 # Claude Opus distillation pipeline
│   ├── export_training.py         # Corrections → LoRA training data
│   ├── train_lora.py              # LoRA fine-tuning (ROCm + CUDA)
│   ├── autopilot.py               # Autonomous self-improvement loop
│   ├── learn_cycle.sh             # Linux: automated learning cycle
│   ├── stratos_overnight.sh       # Overnight autonomous training
│   ├── setup_rocm_training.sh     # One-time AMD GPU training setup
│   ├── model_manager.py             # Ollama model lifecycle
│   ├── evaluate_scorer.py           # Dual-temp model evaluation
│   ├── prompt_version.py            # Prompt hash drift detection
│   ├── .env                       # API keys (not committed)
│   ├── fetchers/
│   │   ├── market.py              # yfinance — stocks, crypto, commodities
│   │   ├── news.py                # DuckDuckGo + Serper + RSS aggregation
│   │   ├── discovery.py           # Dynamic entity detection
│   │   ├── extra_feeds.py         # Finance/politics RSS feed catalog
│   │   ├── kuwait_scrapers.py     # Kuwait-specific sources
│   │   ├── google_search.py       # Google Custom Search client
│   │   └── serper_search.py       # Serper API integration
│   ├── processors/
│   │   ├── scorer.py              # Hardcoded Kuwait scorer (legacy)
│   │   ├── scorer_base.py         # Shared scorer infrastructure
│   │   ├── scorer_adaptive.py     # Profile-adaptive scorer (primary)
│   │   ├── briefing.py            # AI briefing generator
│   │   └── profile_generator.py   # AI category generation pipeline
│   ├── routes/
│   │   ├── agent.py               # Strat Agent chat with tool-use
│   │   ├── auth.py                # DB-auth routes (register, verify, login, profiles)
│   │   ├── config.py              # Settings save API
│   │   ├── generate.py            # Profile/category generation
│   │   ├── wizard.py              # Onboarding wizard API
│   │   └── helpers.py             # JSON response, LLM output cleaning
│   ├── data/v2_pipeline/          # V2 large-scale training pipeline
│   ├── data/users/{id}/           # Per-user JSONL exports
│   └── profiles/                  # Legacy per-user YAML configs
├── frontend/
│   ├── index.html                 # Single-page application shell (~1,329 lines)
│   ├── app.js                     # Core: data loading, SSE, scan control (~1,956 lines)
│   ├── auth.js                    # Device-scoped auth + registration
│   ├── nav.js                     # Sidebar navigation + resize logic (251 lines)
│   ├── feed.js                    # News feed + grid/list toggle + feedback
│   ├── market.js                  # Main chart (TradingView Lightweight) (1,053 lines)
│   ├── markets-panel.js           # Markets tab — multi-chart, Fibonacci (1,877 lines)
│   ├── agent.js                   # Strat Agent chat interface
│   ├── settings.js                # Settings panel — 4 tabs (Profile, Sources, Market, System) (~2,745 lines)
│   ├── wizard.js                  # Onboarding wizard (Quick/Deep modes) (2,583 lines)
│   ├── mobile.js                  # Mobile gestures, bottom nav, PWA install (~2,154 lines)
│   ├── theme-editor.js            # Live theme customization per-variable
│   ├── scan-history.js            # Scan log viewer
│   ├── tour.js                     # Welcome tour + feature exploration (tooltips, modals)
│   ├── ui.js                      # 8 themes, dark/brighter modes, stars, parallax, cross-device sync
│   ├── sw.js                      # Service worker (offline + caching)
│   ├── styles.css                 # 8 themes × 3 modes (24 variants) + stars + mobile
│   ├── manifest.json              # PWA manifest
│   └── icon-192.png / icon-512.png  # PWA icons
└── requirements.txt
```

---

## Features

### Scoring Engine
Every news item gets a Strategic Importance Score (0.0–10.0):
- **9.0–10.0** (Critical) — Direct career matches, paradigm-shifting tech, significant market moves
- **7.0–8.9** (High) — Relevant skills, regional growth, notable trends
- **5.0–6.9** (Moderate) — Tangentially useful, worth knowing
- **Below 5.0** (Noise) — Filtered from the main view

### Markets Panel
Candlestick, line, and area charts via TradingView Lightweight Charts. Five timeframes, drawing tools, Fibonacci retracement with pattern detection, auto-trend lines, side-by-side comparison, draggable cards, PNG export, keyboard shortcuts. **Fullscreen focus mode** expands any chart to full viewport (desktop and mobile).

### Chart Drawing Tools (20 Tools)
Full drawing tool suite for chart annotation in fullscreen focus mode: Trend Line, Ray, Horizontal Line, H-Ray, Vertical Line, Parallel Channel, Fibonacci Retracement, Rectangle, Long Position, Short Position, Measure, Text, Magnet, Eraser, Trash All, Lock, Hide/Show, and Screenshot. Desktop tooltips appear on 400ms hover and display the tool name, description, and keyboard shortcut badge.

### Keyboard Shortcuts
Complete shortcut set for chart tools, active in fullscreen chart mode: T = Trend Line, H = Horizontal Line, F = Fibonacci Retracement, R = Rectangle, V = Vertical Line, and more. Tooltips on every tool button show the name, a brief description, and a key badge so shortcuts are always discoverable.

### Two-Tier Mobile Toolbar
An always-visible 48px hotbar with 6 quick-access drawing tools and a grip handle sits at the bottom of the fullscreen chart on mobile. Drag or tap the grip to reveal the full panel — a 4-column grid containing all 20 drawing tools. The panel locks open and does not auto-dismiss, so you can switch tools without re-opening it.

### Site-Wide Interactivity
Hover effects on all buttons, navigation items, cards, and dropdowns across every page. Accent-colored focus glow on inputs and form elements. Active press feedback provides tactile response on every interactive control.

### Strat Agent
Context-aware AI chat with ticker commands (`$NVDA`, `$BTC`), full portfolio context, conversation export/import, and streaming responses via Ollama. Tools: web search, feed search, watchlist management, category management. Fullscreen desktop mode, free/structured chat toggle, suggestion chips after responses, and "Continue in Agent" from any Ask AI answer. Dedicated full-screen agent view on mobile.

### Profile System
AI-generated interest categories, per-profile watchlists and RSS feeds, email-based authentication with verification codes, and theme customization with 8 themes × 3 modes = 24 variants. Cross-device sync for theme, mode, stars, and avatar via DB-backed `ui_state`. 45 trained profiles across 20+ countries.

### Stop Scan
Graceful cancellation of in-progress scans with streaming Ollama check. Partial results are preserved -- you keep everything scored before the cancel hit.

### Onboarding Wizard
4-step AI-guided profile setup with Quick/Deep modes. Walks new users through role, location, interests, and market watchlist. Generates tailored categories and keywords via the inference model.

### Multi-Profile Authentication
Email-based registration with verification codes, bcrypt password hashing, and 7-day sliding sessions. Per-request profile isolation ensures concurrent users never see each other's data — even during long-running agent chat or scan operations. Legacy PIN auth still supported for YAML profiles.

### Mobile Experience
Full mobile-native experience with:
- **Touch gestures:** Swipe sidebar open from left edge, swipe cards to save/dismiss, pull-to-refresh
- **Bottom navigation:** 5-tab bar (Home, Markets, Agent, Saved, Settings) with active state highlighting
- **Compact layout:** 2-column card grid, hidden sidebar/agent panels, streamlined header
- **Mobile agent:** Dedicated full-screen agent chat view
- **Fullscreen charts:** Expand any chart to full viewport with back-button support
- **Two-tier drawing toolbar:** Always-visible hotbar with 6 quick tools + lockable full panel (4-column grid, all 20 tools) in fullscreen chart focus mode
- **20 drawing tools:** Full chart annotation suite available in mobile fullscreen focus mode
- **Grid/list toggle:** Switch between compact grid and detailed list in Saved view
- **User profile settings** accessible from mobile settings tab
- **PWA installable:** Add to home screen for native app feel with offline support

### Theme System
8 color themes: Midnight (default), Noir, Coffee, Rose, Cosmos, Nebula, Aurora, Sakura. Each has 20+ CSS custom properties in Normal, Deeper, and Brighter modes (24 variants). Twinkling star animations with scroll parallax, Sakura petal particles, collapsible theme picker, and a live theme editor for per-variable customization. Extra Large font size option (20px) that scales everything including icons. Theme, mode, and star preferences sync across devices automatically.

### Scan History
Audit trail of all fetch operations with per-scan breakdowns: sources hit, items scored, duration, and error counts. Accessible from the dashboard sidebar.

### User Profile Settings
Account management section in Settings: change display name, account email, avatar upload, and PIN (with Forgot PIN recovery). Email is persisted to the backend profile YAML. Backend `/api/update-profile` endpoint with secure PIN verification.

---

## Tech Stack

| Layer | Technology | Cost |
|-------|-----------|------|
| AI Scoring | Ollama (Qwen3.5-9B / stratos-scorer-v2.2) | Free (local) |
| Distillation | Gemini Flash + Claude Opus API | ~$6/generation |
| DoRA Training | PyTorch + PEFT + DoRA adapters (ROCm/CUDA) | Free (local) |
| Market Data | yfinance | Free |
| News Search | DuckDuckGo + RSS + Serper (with automatic fallback) | Free (or cheap) |
| Charts | TradingView Lightweight Charts | Free |
| Database | SQLite | Free |
| Frontend | Vanilla JS + Vanilla CSS (24 theme variants) | Free |
| Mobile | PWA + Touch gestures (vanilla JS) | Free |
| Hosting | Cloudflare Tunnel | Free |

---

## Configuration

### Auto-Distillation

Add to `config.yaml` to run distillation automatically every N scans:

```yaml
distillation:
  auto_every: 20        # Run every 20 scans (~weekly)
  hours: 168            # Look back 7 days
  limit: 200            # Max items per cycle
  threshold: 2.0        # Disagreement sensitivity
```

### Changing the Scoring Model

```yaml
scoring:
  model: stratos-scorer-v2.2   # V2.2 DoRA fine-tuned (Qwen3.5-9B)
  inference_model: qwen3.5:9b  # Agent chat, market analysis
  wizard_model: qwen3.5:9b     # Wizard, briefings, profile generation
  ollama_host: http://localhost:11434
```

### Ollama Environment Variables

```bash
OLLAMA_MAX_LOADED_MODELS=2    # Scorer + qwen3.5:9b coexist (~16GB)
OLLAMA_KEEP_ALIVE=10m         # Auto-unload after idle
OLLAMA_NUM_PARALLEL=1         # Single-user (parallel is 6% slower)
OLLAMA_FLASH_ATTENTION=0      # ROCm 6.2 — unreliable
ROCR_VISIBLE_DEVICES=0        # Prevent iGPU interference
```

---

## Known Issues

- **Profile isolation** — SSE events and `is_url_seen()` not fully scoped to profile_id (designed, not yet built)
- **Chat history** — Agent conversation history not persisting across sessions
- **Expanded agent mode** — Fullscreen agent mode missing tool-use capabilities
- **innerHTML XSS** — Several frontend innerHTML assignments need sanitization

---

## Roadmap

- [x] Self-improving scoring via implicit feedback loops
- [x] Claude Opus distillation pipeline
- [x] LoRA fine-tuning with auto GPU detection (ROCm + CUDA)
- [x] Auto-distillation scheduling in main server
- [x] V2.2 scorer with 45-profile training (Gemini Flash, $6, 40% MAE reduction)
- [x] Profile-aware scoring across 20+ countries
- [x] Onboarding wizard (4-step AI-guided profile setup)
- [x] Stop Scan with graceful cancellation
- [x] Multi-user auth with email verification + bcrypt passwords
- [x] Mobile UI overhaul (gestures, bottom nav, compact layout)
- [x] Mobile agent page (dedicated full-screen chat)
- [x] Progressive Web App (installable, offline support)
- [x] Fullscreen chart focus mode (desktop + mobile)
- [x] 24 theme variants (8 themes × 3 modes: Normal, Deeper, Brighter) with live theme editor
- [x] Saved view grid/list toggle
- [x] User profile settings page (name, avatar, PIN change)
- [x] DuckDuckGo fallback when Serper credits exhausted
- [x] 20 chart drawing tools with desktop keyboard shortcuts
- [x] Chart tool tooltips (name + description + key badge)
- [x] Two-tier mobile drawing toolbar (hotbar + lockable panel)
- [x] Account email field + forgot PIN
- [x] Site-wide hover interactivity
- [x] Who Are You button reorder with Setup Wizard binding
- [x] Settings reorganization (4 tabs: Profile, Sources, Market, System)
- [x] Ticker presets (save/load watchlists)
- [x] Focus Mode text button above chart (replaces overlay icon)
- [x] Header toolbar alignment (icon-only refresh buttons aligned with History/Settings/Help)
- [x] Sidebar narrow collapse (themes/profile hide when sidebar dragged < 200px)
- [x] Extra Large font size (20px) with proportional icon scaling

- [x] Email-based authentication with verification codes and password reset
- [x] Multi-user support with per-request profile isolation
- [x] Cross-device theme/avatar sync via DB-backed ui_state
- [x] Per-user data directories with JSONL exports
- [x] Guided Tours panel in Settings for mobile tour access
- [x] Deferred non-blocking briefing (background thread, SSE notification)
- [x] Incremental scanning with snapshot score reuse
- [x] Profile-scoped article retention via context hash
- [x] Rich media feeds (video/stream/image/manga detection + grid view)
- [x] RSS auto-discovery + contextual feed suggestions
- [x] Jobs feed tab with career-focused RSS feeds
- [x] Fibonacci retracement (replaced auto trend lines)
- [x] Agent: fullscreen mode, suggestion chips, free/structured toggle, feed search tool
- [x] Collapsible Markets widget and Asset Analysis sections
- [x] Button animations (save pulse, customize glow, bell blink)
- [x] Cloudflare Worker proxy for RSS/media sources

#### In Progress
- [ ] V2.2 expansion scoring (2,000 articles × 25 profiles via Gemini Flash, ~27% complete)
- [ ] Feed density implementation
- [ ] Chart initial zoom optimization

#### Planned — Phase 0 (Stability)
- [ ] Profile isolation (database schema redesign, per-profile SSE scoping)
- [ ] Security audit fixes (7 P0 issues from full codebase audit)
- [ ] Named Cloudflare tunnel (permanent URL)
- [ ] Kill wasted Serper queries (~7-8 per scan)

#### Planned — Phase 1 (Multi-User)
- [ ] FastAPI + uvicorn migration
- [ ] SQLite → PostgreSQL
- [ ] Google OAuth
- [ ] Dual Ollama instances (scorer + inference)

#### Future — Platform Vision
- [ ] Anime/Manga mode (AniList-inspired, 60% UI already built)
- [ ] Collection/TCG mode (TCGPlayer-inspired, 70% infra reuse)
- [ ] Gaming mode (Steam-inspired)

### Development History

| Milestone | Date | Details |
|-----------|------|---------|
| Mobile Gestures + UI | Feb 25, 2026 | Fullscreen charts, profile settings, DDG fallback, multi-chart mobile layout |
| Chart Tools + Interactivity | Feb 26, 2026 | 20 chart drawing tools, tooltips with keyboard shortcuts, two-tier mobile toolbar |
| Theme Overhaul + Settings | Feb 26, 2026 | 8 themes with dark mode + stars, settings 4-tab layout, ticker presets |
| Pipeline + Retention | Feb 26-27, 2026 | Incremental scanning, deferred briefing, profile-scoped retention, model routing |
| Auth Pipeline | Feb 28, 2026 | Email verification, SMTP, per-user data directories, profile data isolation |
| UI Sync + Profile Isolation | Mar 1, 2026 | Cross-device theme/avatar sync, guided tours, per-request profile_id isolation |
| V2.2 Scorer Deployed | Mar 8, 2026 | Qwen3.5-9B DoRA, 40% MAE reduction, $6 Gemini training, zero parse failures |
| Rich Media + Agent | Mar 8-10, 2026 | Jobs tab, RSS discovery, media grid, Fibonacci charts, agent fullscreen + chips |
| Codebase Audit | Mar 4, 2026 | 102 issues found (7 P0, 43 P1, 52 P2), threading fixes, security analysis |

---

## License

Private project. Not licensed for redistribution.
Fine-tuned model weights are derivative works under Apache 2.0 (Qwen3 license).

---

*Built with stubbornness and an unreasonable number of late nights.*
