# StratOS — Session Context: Scaling, AI Costs, Scoring Audit & Feature Planning

**Date:** March 9, 2026
**Context:** Comprehensive planning session covering scaling strategy, AI inference costs, scoring quality audit, infrastructure decisions, and feature roadmap. This document captures every decision, analysis, and finding from the session. It is intended as input for future Claude Code sessions.

---

## Table of Contents

1. [Feature Ideas — Categorized & Analyzed](#1-feature-ideas)
2. [Scaling Plan — 4 Phases](#2-scaling-plan)
3. [AI Inference Cost Analysis](#3-ai-inference-cost-analysis)
4. [GPU & Inference Architecture](#4-gpu--inference-architecture)
5. [Ollama vs vLLM — When to Use Which](#5-ollama-vs-vllm)
6. [Search Infrastructure — Serper Alternatives & Tiers](#6-search-infrastructure)
7. [Scoring Quality Audit — Real Data Analysis](#7-scoring-quality-audit)
8. [Scan Pipeline Performance Analysis](#8-scan-pipeline-performance)
9. [Security & IP Protection](#9-security--ip-protection)
10. [Development Tooling — Playwright, Claude Code on Linux](#10-development-tooling)
11. [Updated Backlog — Bugs, Features, UX](#11-updated-backlog)
12. [Infrastructure Completed This Session (Rich Media Feed)](#12-infrastructure-completed-this-session)
13. [Key Decisions & Action Items](#13-key-decisions--action-items)

---

## 1. Feature Ideas — Categorized & Analyzed

Ahmad proposed 26 feature ideas. Each was analyzed for feasibility, priority, and whether it belongs in StratOS or is a separate project.

### Category A: Strong StratOS Features (Build These)

| # | Feature | Priority | Effort | Analysis |
|---|---------|----------|--------|----------|
| 1 | Email summary of fetched signals after each scan | High | 4-6h | SMTP infrastructure likely exists or is trivial to add. Trigger after scan completion, format top signals per category, send via email service. High-value, low-effort. |
| 2 | Top 3 articles per category summarized by best available model (Gemini 2.5 Flash recommended) | High | 4-6h | Gemini Flash at $0.01/1M tokens is the best cost/quality ratio. Better summarization than local models. Could replace or augment the current briefing generator. |
| 3 | Better URL reader/summarizer — agent can fetch and summarize any URL | Medium | 3-4h | Extend Ask AI to accept URLs. The article scraper infrastructure already exists in `fetchers/news.py`. Pipe scraped content to LLM for summary. |
| 22 | YouTube video transcript summarizer | Medium | 3-4h | Use `youtube-transcript-api` pip package to fetch transcripts, pass to LLM. Clean single-session feature. No new infrastructure needed. |
| 26 | Edit RSS URL after adding it in custom feeds | Low | 30min | Simple UI fix — add an edit button next to the × on each feed pill in `settings.js`. Minimal effort. |
| 10 | Localization features (weather, restaurants, shops, malls) | Medium | 8-16h | Weather API is easy (OpenWeatherMap free tier). Restaurants/shops need Google Places API. Scope tightly — weather widget first, places later. Good for portfolio. |
| 11 | Interactive calendar with reminders, priorities, snooze, StratOS integration | High | 20-40h | Biggest feature on the list. Makes StratOS a daily driver instead of just a news reader. Use FullCalendar.js for the frontend. Backend needs a `calendar_events` table with profile_id, timestamp, priority, snooze state. Connect to scan results ("New KIPIC job posting" → auto-create calendar reminder). |
| 12 | Raindrop-style bookmarks (tag + organize saved articles) | Medium | 8-12h | The "Saved" tab already exists. Extend with: tags (user-defined), folders/collections, full-text search across saved articles. This is bookmark management, not a new feature — it's improving an existing one. |
| 6-7 | Voice note transcription / TTS for agent | Low | 8-12h | Whisper.cpp is the best local transcription option. Piper TTS for text-to-speech. Both run on CPU. Cool feature but a rabbit hole — park until after graduation unless it becomes a senior design requirement. |

### Category B: Development Practices (Adopt Immediately)

| # | Feature | Priority | Analysis |
|---|---------|----------|----------|
| 18 | Use sub-agents with Claude Code | High | Workflow improvement, not a code feature. Claude Code supports task decomposition — break large tasks into sub-agents that handle specific files/modules. Learn this now; it makes every future sprint faster. |
| 13 | Comprehensive Claude Code session prompt (v10) | High | The session prompt (`claude_code_session_prompt_v9.md`) needs updating with: current model stack (Qwen3.5-9B, not 30B-A3B), OLLAMA_MAX_LOADED_MODELS=2, new infrastructure (RSSHub Docker, CF Worker proxy, WARP), scoring audit findings, and this document's contents. |
| 17 | Write everything to files for context; use QMD for semantic search; compact manually before system does it automatically | High | STATE.md append-only is already this pattern. Extend with a DECISIONS.md for architectural choices. The QMD (quarto markdown) suggestion is about making docs searchable — relevant if integrating with Obsidian later. |

### Category C: Research Tasks (Investigate, Don't Build)

| # | Feature | Analysis |
|---|---------|----------|
| 4 | Phone → Claude Code terminal communication | Already possible: Termux + SSH (`ssh ahmad@local-ip`) or JuiceSSH app. No building needed. |
| 5 | Investigate Codex vs Claude Code | Codex is OpenAI's equivalent. Claude Code is better for StratOS because all project context is already built around it. Not worth switching. |
| 8 | Whisper model investigation for transcription | Whisper.cpp is currently the best local option. Faster-whisper is a ctranslate2 variant with 4x speedup. No action needed until voice features are prioritized. |
| 9 | Consider switching from SQLite to Obsidian | **No. Obsidian is a note-taking app, not a database.** You'd lose auth, profiles, scoring history, migrations — everything. If you want Obsidian integration, export notes TO Obsidian, don't replace the DB. The correct DB migration path is SQLite → PostgreSQL (see Scaling Plan §2). |
| 14 | Semantic search for Obsidian + markdown | Interesting for personal knowledge management but not StratOS-related. Park entirely. |
| 15 | Excalidraw MCP integration | Niche. Only useful for visual planning in agent. Low ROI for the effort. |
| 16 | Check Clawdiverse for useful features | Worth a 30-minute browse for inspiration. Don't commit to building anything from it without evaluating against the existing backlog. |
| 21 | Investigate Manus usage ROI for StratOS | Manus is an AI agent platform. Your local stack already handles the same workflows. Low priority unless Manus offers something your architecture can't do. |

### Category D: Separate Projects (Do NOT Mix with StratOS)

| # | Feature | Analysis |
|---|---------|----------|
| 23 | TCG/Collectible/game implementation | Zero overlap with StratOS's intelligence platform mission. Build as a separate project if interested. Mixing it in dilutes the product story for employers. |
| 25 | Unrestricted chatbot with roleplay capability and job assigning | Also a separate project. StratOS's agent is purpose-built for intelligence analysis. A general chatbot with roleplay is a different product entirely. |

### Category E: Security (Do Before Going Public)

| # | Feature | Priority | Analysis |
|---|---------|----------|----------|
| 19-20 | Security audit + AI-hacker simulation | Critical | Must complete before any public demo. Have Claude Code simulate attacks: SQL injection on all endpoints, XSS via RSS feed content injection, SSRF via `/api/proxy` (the biggest attack surface — it's auth-exempt), auth bypass attempts, rate limit testing. See §9 for full details. |

### Uncategorized Session References

| # | Item | Context |
|---|------|---------|
| 24 | `claude --resume 285667bb-69c8-40bb-934a-3648eca3b82e` | A specific Claude Code session to resume. Save this ID for reference. |

---

## 2. Scaling Plan — 4 Phases

### Current Architecture Limits

| Component | Current | Breaks at | Why |
|-----------|---------|-----------|-----|
| HTTP Server | Python `http.server` (threaded) | ~20 concurrent users | No async, GIL-bound, one thread per request |
| Database | SQLite WAL mode | ~50 concurrent writers | File-level locking, already crashing with 2 profiles |
| AI Inference | Single Ollama on 1 GPU | 1 request at a time | Sequential processing; model swapping if using different models |
| Search API | Serper (1,555 credits remaining) | ~25 full scans | Each scan burns 30-60 queries |
| Memory | 32GB DDR5 | ~3 concurrent scans | Each scan + Ollama + scraping peaks at ~10GB |
| Frontend | Vanilla JS SPA | Never (for rendering) | No framework overhead. But no SSR, no SEO |
| Networking | Cloudflare quick tunnel | Each restart = new URL | No stable domain |

### Phase 0: Stability (Now → April 2026)

**Goal:** Fix what's broken before adding anything.

| Task | Effort | Impact |
|------|--------|--------|
| Fix profile isolation / SSE scoping (B1) | 4-8h | Eliminates SQLite crashes and cross-profile data bleed |
| Fix `_scan_pid` NameError in `main.py:837` (B11) | 30min | Market refresh stops failing silently |
| Security audit on `/api/proxy` — it's auth-exempt and unlimited | 2-4h | Prevents abuse of CF Worker proxy as open relay |
| Rate-limit `/api/proxy` (10 req/s per IP) | 1-2h | Defense in depth |
| Named Cloudflare Tunnel (stable URL like `stratos.yourdomain.com`) | 1h | Permanent access URL for demos |
| RSSHub as systemd service | 30min | Survives reboots |
| WARP + cloudflared conflict resolution script | 2h | Toggle WARP only when Docker RSSHub needs it |
| Set `OLLAMA_MAX_LOADED_MODELS=2` and `OLLAMA_KEEP_ALIVE=-1` | 5min | **Both scorer and inference model stay warm in VRAM simultaneously. Eliminates 5-15 second model swap penalty. Both models fit (~14-15GB total) with ~9GB headroom.** |

### Phase 1: Multi-User Ready (April → June 2026)

**Goal:** 5-10 users can use StratOS simultaneously.

| Task | Effort | Impact |
|------|--------|--------|
| Switch to FastAPI + uvicorn | 16-24h | Async I/O, WebSocket SSE, auto-docs, proper middleware |
| SQLite → PostgreSQL (or SQLite with connection pooling) | 8-16h | Concurrent writes, proper user isolation |
| Per-profile scan queuing (FIFO, no preemption) | 4-8h | Scans queue with position indicator instead of fighting |
| Session-scoped SSE with heartbeat | 4h | Each client gets its own event stream, no cross-talk |
| Google OAuth (F9) | 8-12h | Real auth, replaces PIN codes |
| Gemini Flash API integration for summarization | 4-6h | $0.01/1M tokens, better summaries than local models |
| Email digest service | 4-6h | Morning email with top signals per category |
| SearXNG self-hosted for search | 2-4h | Free, unlimited search queries |

### Phase 2: Cloud-Hybrid (August → December 2026)

**Goal:** Offload inference to cloud when local GPU is busy. Support 20-50 users.

| Task | Effort | Impact |
|------|--------|--------|
| Docker Compose for full stack | 4-6h | One `docker-compose up` for everything |
| Redis for caching + session store | 4-8h | Replaces in-memory caches, survives restarts |
| Background job queue (Celery or ARQ) | 8-12h | Scans, emails, briefings as background workers |
| Ollama → vLLM for production scorer serving | 4-8h | Continuous batching: 8-10 concurrent scoring requests per GPU |
| Cloud GPU fallback (Vast.ai/RunPod) | 4-8h | Route to cloud when local GPU busy (>30s response time) |
| CDN for frontend (Cloudflare Pages) | 2h | Static files served from edge |
| PostgreSQL migration | 8-16h | Replace SQLite entirely |

### Phase 3: SaaS-Ready (Spring 2027, aligns with Senior Design)

**Goal:** Public beta. Paid tiers. 100+ users.

| Task | Effort |
|------|--------|
| Multi-tenant architecture with data partitioning | 20-40h |
| Stripe payment integration (Free + Pro tiers) | 8-12h |
| Landing page + documentation site | 8-16h |
| Usage metering + rate limiting per user | 4-8h |
| Mobile app wrapper (Capacitor around PWA) | 8-12h |

### Timeline (Aligned with Academics)

```
Mar 2026 ─── Phase 0: Stability + Security Audit
Apr 2026 ─── Phase 1 Start: FastAPI, dual-model Ollama, OAuth, email digest
May 2026 ─── Phase 1 Complete. STOP new features — exam prep.
Jun-Jul 2026 ── Summer semester (compressed Signals & Systems + OS). Maintenance only.
Aug 2026 ─── Phase 2 Start: Docker Compose, PostgreSQL, Redis, cloud GPU fallback
Sep-Dec 2026 ── Fall (17 credits, hardware labs, Senior Design I). StratOS IS the capstone.
Jan-May 2027 ── Spring (18-20 credits + capstone). Phase 3 if time. Portfolio polish. Job apps.
```

---

## 3. AI Inference Cost Analysis

### What One User Costs Per Day

**1 Scan Cycle:**
| Task | LLM Calls | Total Tokens |
|------|-----------|-------------|
| Score 300 articles (batches of 4) | ~75 | ~18,750 |
| Re-score 12 uncertain items | 12 | ~4,560 |
| Briefing generation | 1 | ~3,500 |
| **Scan total** | **~88** | **~26,810** |

**Daily Agent Usage (estimates):**
| Task | Calls | Total Tokens |
|------|-------|-------------|
| Agent chat (10 messages) | 10 | ~8,000 |
| Ask AI (5 articles) | 5 | ~6,000 |
| Market analysis (3 queries) | 3 | ~2,700 |
| Wizard/profile (1x setup) | 3 | ~3,000 |
| **Agent total** | **~21** | **~19,700** |

**Total: ~80K tokens/day per active user (2 scans + agent usage)**

### Cloud API Costs Per User

| Provider | Model | Cost per 80K tokens | 10 users/day | 100 users/day |
|----------|-------|---------------------|--------------|---------------|
| Google | Gemini 2.5 Flash | **$0.02** | $0.20 | $2.00 |
| Google | Gemini 2.5 Pro | $0.35 | $3.50 | $35.00 |
| Anthropic | Claude Haiku 4.5 | $0.15 | $1.50 | $15.00 |
| OpenAI | GPT-4o-mini | $0.02 | $0.20 | $2.00 |
| Groq | Llama 3.3 70B | $0.04 | $0.40 | $4.00 |
| Together.ai | Qwen3-8B | **$0.01** | $0.10 | $1.00 |

**Key finding: Gemini Flash costs $0.02/day per user = $0.60/month per user.**

### The Hybrid Approach (Recommended)

```
SCORING (the fine-tuned model — StratOS's competitive moat):
  ├── Ahmad's usage: Free (local 7900 XTX)
  ├── Pro tier users: Together.ai Qwen3-8B ($0.003/scan) or cloud vLLM GPU
  ├── Free tier users: Gemini Flash with prompt-based scoring (~80-85% quality)
  └── When scaling: Deploy merged GGUF to Vast.ai ($108/mo for 50-100 users via vLLM)

AGENT/CHAT (general inference):
  ├── Ahmad: Free (local Ollama)
  ├── Other users: Gemini 2.5 Flash ($0.02/day/user)
  └── Complex queries fallback: Claude Haiku ($0.15/day)

SUMMARIZATION (briefings, email digests):
  └── Gemini 2.5 Flash for everyone ($0.01 per briefing)

TRAINING (distillation + fine-tune):
  ├── Claude Batch API: $17-80 per training cycle
  ├── Local GPU for actual training: Free
  └── Frequency: Once per 2-3 months
```

### Cost Projections — Hybrid Approach

| Users | Scoring | Agent | Summaries | Total/mo | Per-user/mo |
|-------|---------|-------|-----------|----------|-------------|
| 1 (Ahmad) | $0 local | $0 local | $0 local | **$0** | $0 |
| 10 | $0 local | $6 | $2 | **$8** | $0.80 |
| 20 | $0 local | $12 | $4 | **$16** | $0.80 |
| 30 | $108 cloud GPU | $18 | $6 | **$132** | $4.40 |
| 50 | $108 | $30 | $10 | **$148** | $2.96 |
| 100 | $216 (2 GPUs) | $60 | $20 | **$296** | $2.96 |

**The jump happens at 30 users** — that's when the local GPU can't keep up and cloud GPU is needed. Before that, Ahmad's PC handles everything.

### Break-Even Analysis

| Expense | Monthly |
|---------|---------|
| VPS (Hetzner) | $5 |
| Domain | $1 |
| Gemini API (50 free-tier users) | $25 |
| Cloud GPU for scorer (20 pro users) | $108 |
| Serper API (or $0 with SearXNG) | $0-50 |
| **Total** | **$139-189** |

**Break-even at 14-19 Pro users at $10/month.**

### Competitive Pricing Context

| Product | Price | What they do |
|---------|-------|-------------|
| Feedly Pro+ | $18/mo | RSS reader with AI summaries |
| Perplexity Pro | $20/mo | AI search |
| Seeking Alpha Premium | $239/yr | Financial intelligence |
| Bloomberg Terminal | $24,000/yr | Market data + news |

StratOS at $10/mo is genuinely competitive.

---

## 4. GPU & Inference Architecture

### Current Hardware

| Component | Spec |
|-----------|------|
| GPU | AMD Radeon 7900 XTX — 24GB VRAM |
| CPU | AMD Ryzen 7800X3D |
| RAM | 32GB DDR5-6000 CL30 |
| Spare GPU | RTX 2070 Super 8GB — **NOT usable** (mixed AMD/NVIDIA driver stacks can't coexist in same Ollama instance; dual-GPU idea is dead) |

### VRAM Budget — Inference (What Actually Runs)

| Model | Quantization | VRAM | Role |
|-------|-------------|------|------|
| stratos-scorer-v2.2 (Qwen3-8B DoRA) | Q8_0 | ~8.7GB | Relevance scoring |
| qwen3.5:9b | Q4_K_M | ~5.5-6GB | Agent, briefing, wizard |
| KV cache (both models) | — | ~1-2GB | Context windows |
| **Total** | | **~15-17GB** | **~7-9GB headroom** |

**CRITICAL FINDING: Both models fit simultaneously in 24GB VRAM.** The model swapping that caused 5-15 second delays was because Ollama defaults to unloading after 5 minutes of inactivity. Fix:

```bash
# Add to start script or systemd service
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=-1
```

This was available the entire time. Both models stay warm, zero swap penalty.

### VRAM Budget — Training (Why 23.3GB Was Reported)

Training Qwen3.5-9B with DoRA requires ~23GB because of optimizer states and gradient checkpointing. This is training-only and has nothing to do with inference:

| Component | VRAM |
|-----------|------|
| Base weights (BF16) | ~18GB |
| DoRA adapters | ~0.2GB |
| Optimizer states | ~1.5GB |
| Activations + grad checkpoint | ~3-5GB |
| **Total** | **~23GB** |

Training is 4x more memory-intensive than inference. They are completely different VRAM regimes.

### Multi-User Inference — How It Actually Works

**Ollama (current, single-user dev):**
- Serves one request at a time, sequential
- 2 concurrent scans = each takes 2x longer (linear degradation)
- Does not crash — has internal request queue
- Good for: development, single user, model hot-swapping

**vLLM (production, multi-user):**
- Continuous batching — processes 8-10 requests simultaneously on one GPU
- Instead of 10 × 3s = 30s sequential → 10 requests in ~5s parallel
- RTX 3090 ($108/mo) handles 50 users' scoring via vLLM
- Requires safetensors format (not GGUF) — conversion is lossless
- Good for: serving one model to many users

**Conversion from Ollama/GGUF to vLLM (lossless, no retraining):**

```python
# The merged safetensors likely already exist from training pipeline
# If not, re-merge:
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model = PeftModel.from_pretrained(base, "./stratos-scorer-v2.2-adapter")
merged = model.merge_and_unload()
merged.save_pretrained("./stratos-scorer-v2.2-merged")  # safetensors format
# This folder works directly with vLLM
```

**Production deployment command:**

```bash
pip install vllm
vllm serve /path/to/stratos-scorer-v2.2-merged \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 8192
```

### Scan Queuing Architecture (For 10+ Users)

```
User triggers scan → Enters FIFO queue → Position shown in UI
                      ("Position 2 in queue, ~8 minutes estimated")

Queue rules:
  - First in, first out. No preemption.
  - Running scan is NEVER interrupted.
  - Daily cap: 10 scans per profile.
  - If all 10 users scan once morning + once evening = 20 scans/day
  - Average queue depth: 0-1 (scans rarely overlap)
  - Worst case (3 simultaneous): 2-3 minute wait

Implementation:
  - Python queue.Queue in front of existing scan thread
  - SSE event: {"type": "queue_position", "position": 2, "eta_seconds": 480}
  - Existing _scan_cancelled flag and thread lock already prevent concurrent scans
```

### The First-Pass Filtering Idea

Use a cheap fast model (Gemini Flash) to pre-filter before the fine-tuned scorer:

```
300 fetched articles
    ↓
Gemini Flash: "Here are 300 headlines. Return IDs of 50-80
              that could possibly be relevant to [profile]."
    ↓ (1 API call, ~2 seconds, $0.001)
50-80 candidate articles
    ↓
Fine-tuned scorer: Full profile-aware scoring
    ↓ (50-80 LLM calls instead of 300)
Scored feed
```

**Result:** Same quality output, 4x fewer scorer calls, faster scans.

**Status:** Needs testing. Run both pipelines on same 300 articles, compare outputs. The fine-tuned model IS better at scoring (trained on 5,679 profile-specific examples). Gemini is better at fast bulk filtering. They complement each other. Estimated experiment effort: 2 hours.

---

## 5. Ollama vs vLLM — When to Use Which

| Criterion | Ollama | vLLM |
|-----------|--------|------|
| Use case | Single-user dev, model exploration | Multi-user production serving |
| Format | GGUF (quantized, compact) | Safetensors (HuggingFace native) |
| Model swapping | Hot-swap between models (5-15s) | Single model per instance |
| Batching | None — sequential requests | Continuous batching (8-10x throughput) |
| API | Simple REST, similar to OpenAI | OpenAI-compatible REST |
| Memory management | Automatic VRAM management | Manual `--gpu-memory-utilization` |
| LoRA support | Baked into GGUF at conversion | Native runtime LoRA adapter loading |
| Ease of setup | `ollama run model` | `pip install vllm && vllm serve` |

**Decision: Keep Ollama for all development and personal use (Phases 0-1). Switch to vLLM only when serving the scorer to paying users (Phase 2+).**

---

## 6. Search Infrastructure — Serper Alternatives & Tiers

### Current State

- Serper: 1,555 credits remaining (~25 full scans)
- Each scan: 30-60 search queries at $0.02/query
- DuckDuckGo: Free fallback, rate-limited, lower quality
- **Search cost is the actual scaling bottleneck, not AI inference**

### Alternatives

| Provider | Cost | Quality | Queries/day |
|----------|------|---------|-------------|
| **SearXNG** (self-hosted) | **Free** | Good (aggregates Google, Bing, DDG) | **Unlimited** |
| Brave Search API | $3/1000 queries after 2000 free/mo | Good | Unlimited (paid) |
| DuckDuckGo | Free | Lower | Rate-limited |
| Google Custom Search | $5/1000 after 100 free/day | Best | Unlimited (paid) |
| Serper | $50/2500 credits | Very good | Credit-limited |

### Recommended: User-Selectable Search Tiers

```
Free tier:    SearXNG self-hosted (default, unlimited, good quality)
Fallback:     DuckDuckGo (rate-limited, lower quality)
Premium:      Serper ($0.02/query, best quality)
Alternative:  Google Custom Search (if user provides their own API key)
```

**Implementation:** Add `searxng` as an option to the existing `search.provider` config key. SearXNG returns JSON — adapter code is minimal. Let users select in Settings → Sources.

**SearXNG deployment:**
```bash
docker run -d --name searxng -p 8888:8080 searxng/searxng
```

**Important:** Do NOT automate Serper account creation for free credits. That's TOS violation and they'll IP-ban. SearXNG is the legitimate free alternative.

---

## 7. Scoring Quality Audit — Real Data Analysis

### Dataset

Analyzed export from `stratos_export_2026-03-09-14-52.json`:
- **Profile:** Game Dev in Kuwait
- **Items:** 100 (filtered view of full scan)
- **Categories:** 6 (kuwait_gamedev_employers, global_game_tech, indie_game_trends, regional_gaming_news, gamedev_learning, mid_career_development)

### Score Distribution

| Range | Count | % |
|-------|-------|---|
| 9+ Critical | 0 | 0% |
| 7-8.9 High | 18 | 18% |
| 5-6.9 Medium | 12 | 12% |
| <5 Noise | 70 | 70% |

### Category Breakdown

| Category | Items | Avg Score | Max Score |
|----------|-------|-----------|-----------|
| global_game_tech | 8 | 5.4 | 8.0 |
| indie_game_trends | 19 | 5.6 | 8.0 |
| regional_gaming_news | 36 | 4.3 | 7.5 |
| gamedev_learning | 16 | 4.5 | 7.0 |
| mid_career_development | 5 | 5.4 | 7.0 |
| kuwait_gamedev_employers | 16 | 3.5 | 3.5 |

### 5 Concrete Problems Found

#### Problem 1: LLM Timeouts Defaulting to Generous Rule Scores (13 items)

When the LLM doesn't respond in time, articles get a default rule-based score. The rule engine sees regional keywords ("Middle East", "Kuwait") and assigns 7.0, which is too generous for non-profile-relevant content.

**Mis-scored examples:**
- `[7.0]` "UAE warns enemies amid Middle East tension" → Geopolitics, zero relevance to game dev
- `[7.0]` "Middle East Lifestyle News - AGBI" → Lifestyle magazine, not gaming
- `[6.5]` "2026 AdvaMed® CEO Summit" → Medical device conference
- `[4.5]` "Iran Sustains Strikes on Arab States" → War news, not game dev
- `[4.5]` "Women's Day 2026: How to build right mutual fund portfolio" → Finance advice

**Root cause:** Rule-based fallback score of 7.0 for regional keyword match is too high.
**Fix:** Lower the regional keyword rule-match fallback to 4.0-5.0. "Middle East" alone should not elevate above medium without game/tech keywords co-occurring. Implement in `scorer_base.py` or `scorer_adaptive.py` rule-scoring section.

#### Problem 2: Kuwait Category Full of Garbage (All at 3.5)

All 16 `kuwait_gamedev_employers` items scored exactly 3.5. Every one is irrelevant:
- "Liverpool in 'close contact' over move to sign former Premier" — Football transfers
- "A Look At The Salaries Of US Legacy Carrier Pilots" — Airline industry
- "TOC TEST ON LIVE SITE" — A broken test page
- "Ex-NBA big man admitted bootleg Percocet addiction" — Sports/drugs

**Root cause:** The wizard hallucinated entities that don't exist. `"KIPIC Games"`, `"GDC Kuwait"`, `"Kuwait Interactive Entertainment Association"` — all returned 0 Serper results. The fallback generic queries (`"local game studios Kuwait"`) pulled in random career/hiring content.
**Fix:** This is a wizard entity generation problem, not a scorer problem. The scorer correctly gave these 3.5 (low). The search queries need to be validated — if an entity returns 0 results, discard it and use broader real queries instead. Fix in `routes/wizard.py` → `_call_ollama` for rv-items, or add a post-generation validation step.

#### Problem 3: SEO Spam Scored 7.5-8.0

The scorer doesn't distinguish source quality. Spam blogs and legitimate sources get equal treatment:
- `[8.0]` "The Toby Foliage Engine - Free Download - Dev Asset Collection" — spam download site
- `[7.5]` "Robuxus Com: Pioneering the Future of In-Game Monetization" — SEO spam
- `[7.5]` "Your Ultimate Guide to the Digital Gaming Revolution - bitsandblogs" — clickbait
- `[7.5]` "Animation Id Roblox: Unlocking Endless Creativity Through Motion" — SEO content

**Root cause:** No source quality signal in the scoring pipeline.
**Fix options:**
1. Maintain a `spam_domains.txt` blocklist in the fetcher — auto-reject articles from known spam domains before scoring. Low effort, high impact.
2. Add source quality as a scoring signal — pass domain reputation or a "is this a known quality source" flag to the LLM scorer prompt. Medium effort.
3. Post-scoring filter — flag articles with clickbait title patterns ("Ultimate Guide", "Unlock", "Pioneering", "Free Download") and demote by 1-2 points. Low effort.

Recommendation: Options 1 + 3 combined. Blocklist + clickbait demotion. No model retraining needed.

#### Problem 4: Category Misassignment

"Middle East Lifestyle News - AGBI" is in `regional_gaming_news` at 7.0. A lifestyle magazine has nothing to do with gaming. The category name includes "gaming" but the search queries that populated it didn't filter for it.

**Root cause:** Dynamic category search queries are too broad. The category is `regional_gaming_news` but the search query was something like `"Middle East esports events" regional gaming tech...` which matched a lifestyle article that mentions "tech".
**Fix:** Tighten search queries to require game/gaming/esport co-occurrence. In `fetchers/kuwait_scrapers.py`, the query builder for each category should AND the category's core keywords, not just OR them with generic terms.

#### Problem 5: Zero Critical Signals

For "Game Dev in Kuwait", items like "Unity announces new pricing model", "Kuwait gaming expo dates announced", or "GDC scholarships for MENA developers" should be 9+. The fact that zero items hit critical means:
1. The search queries aren't finding the most relevant content (wizard entity problem)
2. OR the scorer's threshold for 9+ is too conservative for this profile

**Diagnosis:** Likely problem #1. The search queries are fetching generic content that tops out at 8.0. Truly critical signals (direct employer news, direct tool announcements) aren't being found because the wizard-generated entities don't exist.
**Fix:** Improve wizard entity generation (see Problem 2). Also: add real entities to the Game Dev profile manually for testing — "Unity Technologies", "Epic Games", "Steam", "Kuwait Esports Association" (if it exists), "GDC", "itch.io". Then rescan and compare.

### Overall Scoring Assessment

**The scorer itself is working correctly on the content it receives.** It's assigning reasonable scores to the articles that arrive. The problem is upstream:
1. The wizard generates hallucinated entities → search queries return garbage
2. The search queries are too broad → irrelevant articles enter the pipeline
3. No source quality filtering → SEO spam gets through
4. LLM timeout fallback score is too generous at 7.0

**Priority fixes (no retraining needed):**
1. Wizard entity validation — discard entities with 0 search results
2. Spam domain blocklist
3. Clickbait title pattern demotion (-1 to -2 points)
4. Lower rule-based regional keyword fallback from 7.0 to 4.5
5. Tighter search query construction (require category-core keywords)

### Test Profiles for Further Validation

These profiles should be run through wizard quick setup → scan → analyze results to validate scoring across diverse roles:

| # | Role | Location | Why This Profile |
|---|------|----------|-----------------|
| 1 | Petroleum Engineer | Kuwait City, Kuwait | Ahmad's original target market — should score well |
| 2 | Frontend Developer | Dubai, UAE | Tech role, different GCC country |
| 3 | Medical Resident (Internal Medicine) | Riyadh, Saudi Arabia | Non-tech, tests category diversity |
| 4 | Marine Biologist | Cape Town, South Africa | Niche + unusual location, stress test |
| 5 | Cryptocurrency Trader | Singapore | Finance-heavy, tests market integration |
| 6 | High School Math Teacher | Amman, Jordan | Non-tech, education sector |
| 7 | Supply Chain Manager | Rotterdam, Netherlands | Industry role, European location |
| 8 | Indie Game Developer | Seoul, South Korea | Similar to kirissie but different region |
| 9 | Civil Rights Lawyer | Washington DC, USA | Completely different domain |
| 10 | Agricultural Engineer | Nairobi, Kenya | Niche, tests African regional content |

**Testing protocol:** Pick 3-4. Run wizard quick setup without adjusting anything. Run scan. Export results. Analyze: score distribution, category quality, false positives, missing critical signals.

---

## 8. Scan Pipeline Performance Analysis

### Real Timing Data (From Server Logs)

```
03:27:35  [1/4] Fetching news starts
03:28:33  First 25/108 articles scraped (58s elapsed — search queries + RSS + first scrape batch)
03:28:40  50/108 scraped
03:28:45  75/108 scraped
03:28:49  100/108 scraped (16s for scraping phase)
03:30:07  Scoring complete: 153 items, 84 rule + 69 LLM
03:30:07  News refresh complete in 152.2 seconds
03:32:22  Deferred briefing complete (2m15s after scan)
```

### Phase Breakdown

| Phase | Duration | % of Total | Bottleneck? |
|-------|----------|-----------|-------------|
| Search queries (Serper + DDG + RSS) | ~58s | 38% | Yes — sequential API calls |
| Article scraping (108 articles) | ~16s | 11% | No — already parallelized |
| LLM scoring (69 calls) | ~69s | 45% | Moderate — limited by GPU throughput |
| Rule scoring + overhead | ~9s | 6% | No |
| **Total scan** | **152s** | 100% | |
| Briefing (deferred) | ~135s | Separate | Background, doesn't block user |

### Key Observations

1. **Article scraping is NOT the bottleneck.** 108 articles in 16 seconds = already parallelized with ThreadPoolExecutor. Earlier assumption was wrong.
2. **Search queries are the biggest single chunk (38%).** 60+ sequential Serper/DDG queries. Parallelizing search queries across providers would cut this significantly.
3. **LLM scoring (45%) is the GPU-bound phase.** 69 LLM calls at ~1s each = ~69s. OLLAMA_MAX_LOADED_MODELS=2 ensures scorer is already warm (no swap penalty). Further optimization: batch more items per LLM call (currently groups of 4).
4. **Total scan: ~150s (2.5 minutes).** Ahmad says recent optimizations have brought this to 120-280s range, mostly dependent on fetching speed.
5. **Scoring quality matters more than scoring speed.** 30 seconds for scoring is acceptable. The real problem is what arrives at the scorer (see §7).

### Optimization Priorities

| Optimization | Expected Gain | Effort |
|-------------|--------------|--------|
| Parallelize search queries (async Serper + DDG simultaneously) | -20-30s | 4h |
| Increase LLM batch size from 4 to 8 items | -15-20s | 2h |
| SearXNG (eliminates Serper credit concern, enables unlimited parallel queries) | Cost savings | 2-4h |
| First-pass Gemini Flash filtering (skip obvious noise before LLM scoring) | -30-40s | 4-6h |
| **Combined** | **~80-100s total (down from 150s)** | |

---

## 9. Security & IP Protection

### Threat Model

Two concerns: someone cloning the product, and someone attacking infrastructure.

### Preventing AI Cloning

**Honest assessment:** You cannot stop someone from building a similar product. Defenses:

1. **The fine-tuned scorer IS the moat.** Model weights, training data, and distillation pipeline are invisible to users. 19+ training iterations to reach current quality. Someone would need months to reproduce.
2. **Obfuscate frontend JS in production** (Terser/uglify). Not foolproof but raises effort.
3. **Don't open-source the scorer model or training data.** Platform code can be open-source; model weights stay private.
4. **Speed is the real defense.** By the time someone copies v1, you're on v3.

### Infrastructure Security

**Already have (via Cloudflare Tunnel):**
- DDoS protection (automatic, free tier)
- SSL termination
- Basic bot detection

**Must add before going public:**

#### Phase 0 Security (Do Now)

| Task | Why | Effort |
|------|-----|--------|
| Rate-limit `/api/proxy` (10 req/s per IP) | Auth-exempt endpoint, can be abused as open relay | 1-2h |
| SSRF protection on `/api/proxy` | Block requests to internal IPs (127.0.0.1, 10.x, 192.168.x) and non-HTTP protocols | 1h |
| Validate proxy target URLs against allowlist | Only allow proxying to domains in `blocked_domains` config | 30min |
| CORS whitelist (replace `*` with actual domain) | Currently accepts requests from any origin | 30min |
| Input sanitization on RSS feed URLs | Prevent injection via malicious feed URLs | 1h |
| XSS prevention in feed content rendering | RSS content can contain malicious HTML/JS — sanitize before rendering | 2h |
| Rate-limit `/api/scan` (10/day per profile) | Prevent one user from hammering scans | 30min |

#### Phase 1 Security

| Task | Effort |
|------|--------|
| Google OAuth (replaces PIN auth) | 8-12h |
| CSRF tokens on all POST endpoints | 4h |
| Content Security Policy headers | 2h |
| Audit all raw SQL for injection | 2-4h |

#### Phase 2 Security

| Task | Effort |
|------|--------|
| JWT tokens (replace session table) | 4-8h |
| Role-based access (admin, user, viewer) | 4-8h |
| Audit logging (who did what when) | 4h |
| Encrypted secrets (not plaintext in config.yaml) | 2h |
| Penetration test checklist | 4-8h |

### Claude Code Security Audit Prompt

For a Claude Code session focused on security:

```
Simulate a hacker targeting StratOS. For each endpoint in server.py, attempt:
1. SQL injection via all user-controlled parameters
2. XSS via RSS feed content (inject <script> tags in feed items)
3. SSRF via /api/proxy (try fetching internal IPs, localhost, file:// URLs)
4. Auth bypass (forge tokens, session hijacking, timing attacks)
5. Rate limit testing (automated scan triggering)
6. Path traversal via any file-serving endpoints

For each vulnerability found:
- Demonstrate the exploit
- Assess severity (Critical/High/Medium/Low)
- Implement the fix
- Add a test proving the fix works
- Do NOT break any existing functionality

Start by reading server.py, auth.py, and database.py.
```

---

## 10. Development Tooling — Playwright, Claude Code on Linux

### Browser Automation with Playwright

For Claude Code to interact with the StratOS UI as a real user (clicking, scrolling, filling forms, taking screenshots):

```bash
# Install
pip install playwright --break-system-packages
playwright install chromium
```

**Claude Code can then drive a real browser:**

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # visible browser window
    page = browser.new_page()
    page.goto("http://localhost:8080")
    page.screenshot(path="/tmp/stratos-home.png")
    page.click("text=Custom")
    page.wait_for_timeout(2000)
    page.screenshot(path="/tmp/stratos-custom.png")
    browser.close()
```

**With Brave browser specifically:**
```python
browser = p.chromium.launch(
    headless=False,
    executable_path="/usr/bin/brave-browser"
)
```

**Why this matters:** Programmatic testing misses UX/design bugs. Having Claude Code act as a user reveals layout issues, broken interactions, missing hover states, and flow problems that JSON analysis never catches.

**Claude Code session prompt for UI testing:**
```
Launch Playwright with headless=False (so I can watch). Navigate to localhost:8080.
Log in as the Strat profile. Go through every tab (Summary, Markets, Custom, Settings).
Take screenshots at each step. Report any UI issues: broken layouts, missing elements,
non-functional buttons, visual glitches, accessibility problems. Use Brave browser.
```

### Alternative: Terminal Output Analysis

For non-UI debugging, Claude Code can read StratOS output via tmux:

```bash
# Start StratOS in tmux
tmux new -s stratos
python3 main.py --serve
# Ctrl+B, D to detach

# Claude Code reads output:
tmux capture-pane -t stratos -p
```

### Scoring Pipeline Analysis (Programmatic)

For scoring-specific debugging without a browser:

```bash
claude "Read backend/output/news_data_*.json. Show score distribution.
Identify articles that seem obviously mis-scored based on the profile context.
Check for: LLM timeout fallbacks at 7.0, SEO spam in high scores,
regional news without profile relevance, miscategorized items."
```

---

## 11. Updated Backlog — Bugs, Features, UX

### Bugs & Fixes

| # | Issue | Severity | Notes |
|---|-------|----------|-------|
| B1 | Cross-profile data bleed + SQLite crashes | Critical | SSE not scoped to profile_id. 2 profiles polling = file handle exhaustion. |
| B2 | Market refresh happens with background refresh disabled | High | `schedule.background_enabled: false` not respected. |
| B3 | Presets broken in Settings | High | Preset loading/saving not functioning. |
| B4 | Free chat mode — responses have no spaces/formatting | High | Streaming handler or prompt format concatenating without whitespace. |
| B5 | Cannot minimize Strat Agent after expanding | Medium | Missing minimize/restore toggle. |
| B6 | Agent chat history lost on page reload | High | Not persisted to localStorage or DB. |
| B7 | Fake market refresh timeout | Medium | Progress completes without data fetch. Related to `_scan_pid` NameError. |
| B8 | Fetching phase slower than scoring | Medium | Search queries are sequential (38% of scan time). Parallelize. |
| B9 | Expanded agent doesn't use web search/tools | High | Tool use disabled in fullscreen mode. |
| B10 | Hover states broken | Low | Need to identify which elements. |
| B11 | `_scan_pid` NameError in `main.py:837` | Medium | Market refresh fails silently. |
| B12 | RSS discovery doesn't use CF Worker proxy | Low | Blocked domains fail in auto-detect. |

### New Features

| # | Feature | Priority | Effort |
|---|---------|----------|--------|
| F1 | Location filter in wizard for location-strict categories | Medium | 4-6h |
| F2 | Update wizard hardcoded categories | Medium | 4-6h |
| F3 | "Continue with AI" → navigate to Agent tab, focus, send message | High | 4h |
| F4 | Save scroll position per feed tab | Low | 2h |
| F5 | Unhighlight parent category when no subcategories selected (wizard) | Low | 1h |
| F6 | Make Arcane default theme | Low | 30min |
| F7 | Remove Deep AI feature (A35b model removed) | Medium | 2h |
| F8 | Clickable hyperlinks in agent responses | Medium | 2h |
| F9 | Google OAuth authentication | High | 8-12h |
| F10 | Twitch live status detection (requires API) | Medium | 4-6h |
| F11 | Pixiv integration via RSSHub (needs refresh token) | Low | 2h |
| F12 | RSSHub as permanent systemd service | Medium | 1h |
| F13 | Rich media feed remaining work (video embeds, S/M/L for articles) | Medium | 8h |
| F14 | Email summary of fetched signals | High | 4-6h |
| F15 | Top 3 articles per category via Gemini Flash summarizer | High | 4-6h |
| F16 | URL reader/summarizer in agent | Medium | 3-4h |
| F17 | YouTube video transcript summarizer | Medium | 3-4h |
| F18 | Edit RSS URL after adding in custom | Low | 30min |
| F19 | Interactive calendar with reminders + StratOS integration | High | 20-40h |
| F20 | Raindrop-style bookmarks (tags, folders, search for saved articles) | Medium | 8-12h |
| F21 | SearXNG as self-hosted search provider | Medium | 2-4h |
| F22 | Scan queue with position indicator for multi-user | Medium | 4-6h |
| F23 | Spam domain blocklist + clickbait title demotion in scorer | Medium | 2-3h |
| F24 | Wizard entity validation (discard 0-result entities) | High | 3-4h |
| F25 | Lower rule-based regional keyword fallback from 7.0 to 4.5 | High | 30min |

### UX / UI Polish

| # | Change | Priority |
|---|--------|----------|
| U1 | Articles section not affected by S/M/L grid toggle | Low |

### Priority Summary

**Critical:** B1

**High (next sprint):** B2, B3, B4, B6, B9, F3, F9, F14, F15, F24, F25

**Medium:** B5, B7, B8, B11, F1, F2, F7, F8, F10, F12, F13, F16, F17, F20, F21, F22, F23

**Low:** B10, B12, F4, F5, F6, F11, F18, U1

---

## 12. Infrastructure Completed This Session (Rich Media Feed)

This section documents infrastructure that was built during this session (March 9, 2026) and is now deployed.

| # | Item | Status | Files Modified |
|---|------|--------|---------------|
| I1 | Cloudflare Worker proxy for ISP bypass | ✅ | `cloudflare-proxy/src/index.js`, `wrangler.toml` |
| I2 | `/api/proxy` endpoint (auth-exempt) | ✅ | `backend/server.py` |
| I3 | Media type detection in `/api/custom-news` | ✅ | `backend/server.py` |
| I4 | Rich media view (Videos, Streams, Images, Articles, Lightbox, S/M/L toggle) | ✅ | `frontend/feed.js` |
| I5 | YouTube channel URL auto-convert via YouTube internal resolve API | ✅ | `backend/server.py` (discover-rss endpoint) |
| I6 | Booru high-res image URLs (yande.re/konachan full, danbooru 360px limit) | ✅ Partial | `backend/server.py` |
| I7 | RSSHub self-hosted via Docker (`docker run -d --name rsshub --network host diygod/rsshub`) | ✅ | Not a code change — Docker container |
| I8 | Cloudflare WARP for Docker ISP bypass | ✅ | System-level, conflicts with cloudflared tunnels |
| I9 | Media feed suggestions tab in Settings | ✅ | `frontend/settings.js` |
| I10 | Feed cleanup script (tested all feeds, removed broken, kept 16 working) | ✅ | Database updated |
| I11 | `blocked_domains` in config.yaml (19 domains) | ✅ | `backend/config.yaml` |

### Config State After Session

```yaml
# backend/config.yaml (proxy section)
proxy:
  cloudflare_worker: 'https://stratos-proxy.stratintos.workers.dev'
  blocked_domains:
    - yande.re
    - danbooru.donmai.us
    - gelbooru.com
    - konachan.com
    - safebooru.org
    - f95zone.to
    - files.yande.re
    - cdn.donmai.us
    - raikou1.donmai.us
    - raikou2.donmai.us
    - img1.gelbooru.com
    - img2.gelbooru.com
    - img3.gelbooru.com
    - mangadex.org
    - uploads.mangadex.org
    - api.mangadex.org
    - chan.sankakucomplex.com
    - rule34.xxx
    - kemono.su
```

### Services Running

```
Port 8080:  StratOS (Python HTTP server)
Port 1200:  RSSHub Docker (requires WARP connected for blocked sites)
Port 11434: Ollama (scorer + inference models)
WARP:       Cloudflare WARP (toggle on/off — conflicts with cloudflared tunnel)
CF Worker:  stratos-proxy.stratintos.workers.dev (ISP bypass proxy)
```

### Known Operational Notes

1. **WARP vs cloudflared conflict:** Cannot run both simultaneously. Disconnect WARP (`warp-cli disconnect`) before starting cloudflared tunnel. Reconnect WARP (`warp-cli connect`) when RSSHub needs to fetch from blocked sites.
2. **RSSHub routes that work:** nhentai, ehentai, kemono. Routes that don't: iwara (needs API auth), pixiv (needs PIXIV_REFRESHTOKEN env var), rule34 (route doesn't exist in RSSHub).
3. **Danbooru images limited to 360px:** Their `/sample/` and `/original/` endpoints require paid Gold account. GGUF thumbnails are all we get for free.
4. **Browser downloads not saving to ~/Downloads:** Ahmad's browser saves files unpredictably. For future sessions: use inline Python patches or `cat > file << 'EOF'` instead of file downloads when possible.

---

## 13. Key Decisions & Action Items

### Decisions Made This Session

| Decision | Chosen | Rejected | Why |
|----------|--------|----------|-----|
| AI scaling approach | Hybrid (local + Gemini Flash + cloud GPU at 30+ users) | Pure cloud API, pure self-hosted | Best cost/quality at every scale level |
| Search scaling | SearXNG self-hosted (free, unlimited) as primary | More Serper credits, automated account creation | Legitimate, zero cost, good quality |
| Production inference server | vLLM when serving scorer to users | Ollama for production | Continuous batching = 8-10x throughput |
| Model format conversion | Lossless safetensors export (no retraining) | Retraining for vLLM | Weights are identical, just different file format |
| Frontend framework | Keep vanilla JS | Migrate to React/Svelte | No team to justify framework overhead |
| Database migration path | SQLite → PostgreSQL at Phase 1-2 | SQLite forever, Obsidian | Already hitting SQLite limits with 2 profiles |
| Scoring fix approach | Upstream fixes (search queries, wizard entities, spam filter) | Retrain scorer model | Scorer works correctly — it's receiving garbage input |
| Auth approach | Google OAuth at Phase 1 | Keep PIN auth, Firebase, Clerk | Industry standard, free, well-documented |
| Monetization | Freemium: Free (prompt-based scoring) + Pro $10/mo (fine-tuned scorer) | Free forever, Enterprise only | Break-even at 14-19 users |
| TCG/chatbot/roleplay features | Separate projects | Include in StratOS | Dilutes product story for employers |

### Immediate Action Items (Do Before Next Sprint)

```bash
# 1. Set Ollama to keep both models loaded (5 seconds, massive impact)
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_KEEP_ALIVE=-1
# Add to ~/.bashrc or systemd service file

# 2. Verify both models fit
curl -s http://localhost:11434/api/ps | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f\"{m['name']}: {m['size_vram']//1024//1024}MB\") for m in d.get('models',[])]"

# 3. Make RSSHub survive reboots
sudo bash -c 'cat > /etc/systemd/system/rsshub.service << EOF
[Unit]
Description=RSSHub
After=docker.service
Requires=docker.service

[Service]
ExecStart=/usr/bin/docker start -a rsshub
ExecStop=/usr/bin/docker stop rsshub
Restart=always

[Install]
WantedBy=multi-user.target
EOF'
sudo systemctl enable rsshub
```

### Next 3 Claude Code Sessions (Recommended Order)

**Session 1: Security Audit (Phase 0)**
- Use the security audit prompt from §9
- Focus on `/api/proxy` SSRF, XSS in RSS content, SQL injection
- Add rate limiting, CORS whitelist, input validation
- Estimated: 4-6 hours

**Session 2: Scoring Quality Fixes (Phase 0)**
- Fix F24: Wizard entity validation (discard 0-result entities)
- Fix F25: Lower regional keyword fallback from 7.0 to 4.5
- Fix F23: Add spam domain blocklist + clickbait demotion
- Tighten search query construction in kuwait_scrapers.py
- Run 3-4 test profiles, compare before/after
- Estimated: 4-6 hours

**Session 3: Email Digest + Gemini Integration (Phase 1)**
- Implement F14: Email summary after scan
- Implement F15: Gemini Flash summarizer for top articles
- Add Gemini API key to `.env`
- Estimated: 6-8 hours

---

*This document captures the complete planning session of March 9, 2026. It should be provided as context to future Claude Code sessions alongside the existing session prompt (v9) and the updated BACKLOG. For the scoring audit specifically, the raw export data is in `stratos_export_2026-03-09-14-52.json`.*
