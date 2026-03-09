# StratOS Scaling Plan

**Date:** March 2026
**Current state:** Single-machine, single-user (with multi-profile), 45K+ LOC, SQLite, local Ollama inference
**Target:** Production-grade multi-user platform suitable for portfolio demos, then potential SaaS

---

## 1. Current Architecture Limits

| Component | Current | Breaks at | Why |
|-----------|---------|-----------|-----|
| HTTP Server | Python `http.server` (threaded) | ~20 concurrent users | No async, GIL-bound, one thread per request |
| Database | SQLite WAL mode | ~50 concurrent writers | File-level locking, already crashing with 2 profiles |
| AI Inference | Single Ollama on 1 GPU | 1 scan at a time | Model swapping (scorer↔inference) adds 5-15s per switch |
| Search API | Serper (2500 credits) | ~25 full scans | Each scan burns 30-60 queries |
| Memory | 32GB DDR5 | ~3 concurrent scans | Each scan + Ollama + scraping peaks at ~10GB |
| Storage | Local SSD | Years | DB is 187MB after months of use, growth is slow |
| Frontend | Vanilla JS SPA | Never (for rendering) | No framework overhead, fast. But no SSR, no SEO |
| Networking | Cloudflare quick tunnel | Each restart = new URL | No stable domain, no HTTPS cert control |

---

## 2. Scaling Phases

### Phase 0: Stability (Now → April 2026)
**Goal:** Fix what's broken before adding anything.

| Task | Effort | Impact |
|------|--------|--------|
| Fix profile isolation / SSE scoping (B1) | 4-8h | Eliminates SQLite crashes, data bleed |
| Fix `_scan_pid` NameError (B11) | 30min | Market refresh stops failing silently |
| Security audit on `/api/proxy` (open endpoint) | 2-4h | Prevents abuse of your CF Worker proxy |
| Rate-limit `/api/proxy` (10 req/s per IP) | 1-2h | Prevents proxy being used as open relay |
| Named Cloudflare Tunnel (stable URL) | 1h | Permanent `stratos.yourdomain.com` |
| RSSHub as systemd service | 30min | Survives reboots |
| WARP + cloudflared conflict resolution | 2h | Script to toggle WARP only when Docker needs it |

### Phase 1: Multi-User Ready (April → June 2026)
**Goal:** 5-10 users can use StratOS simultaneously without crashes.

| Task | Effort | Impact |
|------|--------|--------|
| Switch to FastAPI + uvicorn | 16-24h | Async I/O, proper CORS, WebSocket SSE, auto-docs |
| SQLite → PostgreSQL (or SQLite with WAL2 + connection pool) | 8-16h | Concurrent writes, proper isolation |
| Per-profile scan queuing | 4-8h | Scans queue instead of fighting for Ollama |
| Separate scorer and inference Ollama instances | 2h | Two ports: 11434 (inference), 11435 (scorer). Eliminates model swapping |
| Session-scoped SSE with heartbeat | 4h | Each client gets its own event stream, no cross-talk |
| Google OAuth (F9) | 8-12h | Real auth instead of PIN codes |
| Frontend: add React or keep vanilla? | Decision | Vanilla is fine for now. React migration only if team grows |

### Phase 2: Cloud-Hybrid (June → September 2026)
**Goal:** Offload inference to cloud when local GPU is busy. Support 20-50 users.

| Task | Effort | Impact |
|------|--------|--------|
| Ollama load balancer (local + RunPod/Vast.ai fallback) | 8-12h | If local GPU is busy, route to cloud GPU ($0.20/hr) |
| Gemini Flash API for summarization (items 1-2) | 4-6h | Cheap ($0.01/1M tokens), fast, better summaries than local |
| Redis for caching + session store | 4-8h | Replaces in-memory caches, survives restarts |
| Background job queue (Celery or ARQ) | 8-12h | Scans, emails, briefings run as background workers |
| Email digest service (item 1) | 4-6h | Morning email with top signals per category |
| CDN for frontend (Cloudflare Pages) | 2h | Static files served from edge, not your machine |
| Docker Compose for full stack | 4-6h | One `docker-compose up` for StratOS + Ollama + RSSHub + Redis + Postgres |

### Phase 3: SaaS-Ready (September 2026 → Spring 2027)
**Goal:** Public beta. Paid tiers. 100+ users.

| Task | Effort | Impact |
|------|--------|--------|
| Multi-tenant architecture | 20-40h | Proper user isolation, data partitioning, admin panel |
| Stripe/payment integration | 8-12h | Free tier (3 categories, 1 scan/day) + Pro tier ($10/mo, unlimited) |
| Custom domain per user or subdomain routing | 4-8h | `ahmad.stratos.app` or `stratos.app/ahmad` |
| Scorer model as shared service | 4-8h | One scorer instance serves all users, queued |
| Usage metering + rate limiting | 4-8h | Track API calls, scan counts, storage per user |
| Landing page + docs site | 8-16h | Marketing page, onboarding guide, API docs |
| Mobile app (Capacitor wrapper around PWA) | 8-12h | App Store presence, push notifications |

---

## 3. AI Model Scaling

### Current Model Stack
| Role | Model | Size | Speed |
|------|-------|------|-------|
| Scorer | Qwen3-8B DoRA (stratos-scorer-v2.2) | 8.7GB Q8_0 | ~15 t/s |
| Inference/Agent | Qwen3.5-9B | ~6GB Q4 | ~34 t/s |
| Briefing/Wizard | Qwen3.5-9B | shared | shared |
| Teacher (distillation) | Claude Sonnet 4.6 Batch API | cloud | batch |

### Scaling Path

**Phase 1 — Dual GPU (when riser cable acquired)**
```
GPU 1 (7900 XTX 24GB): Inference model full-time
GPU 2 (2070 Super 8GB): Scorer model full-time (Q4_K_M ~4.5GB)
Result: Zero model swapping, 2x throughput
```

**Phase 2 — Cloud Hybrid**
```
Local GPU: Scorer (always warm, no latency)
Cloud fallback: Gemini Flash for summarization ($0.01/1M tokens)
Cloud fallback: Claude Haiku for agent when local is busy ($0.25/1M tokens)
Trigger: If Ollama response > 30s, route to cloud
```

**Phase 3 — Dedicated Inference Server**
```
RunPod/Vast.ai: Persistent A100 ($1-2/hr) or L4 ($0.30/hr)
Deploy: vLLM or TGI with the scorer model
Serve: All users hit one inference endpoint
Local: Development and training only
```

### Training Scaling
| Phase | Dataset | Method | Cost |
|-------|---------|--------|------|
| Current (v2.2) | 5,679 CoT examples | DoRA on Qwen3-8B, local ROCm | ~$17 distillation |
| Next (v3) | 15,000+ examples via pipeline | 50 AI profiles → batch score → train | ~$50-80 distillation |
| Future | 50,000+ examples, multi-language | Full fine-tune or larger base model | ~$200-400 |

---

## 4. Data Scaling

### Storage Growth Projection
| Timeframe | DB Size | News Items | Profiles |
|-----------|---------|------------|----------|
| Now | 187MB | ~10K scored | 15 |
| 6 months | ~500MB | ~50K scored | 50 |
| 1 year | ~2GB | ~200K scored | 200 |
| 2 years | ~10GB | ~1M scored | 1000+ |

### When to Migrate from SQLite
**Trigger:** Any of these:
- More than 5 concurrent writers (you're already hitting this with 2 profiles)
- DB file > 5GB (WAL performance degrades)
- Need full-text search across all articles

**Migration path:** SQLite → PostgreSQL
- Use `pgloader` for one-shot migration
- Replace `sqlite3` calls with `asyncpg` or `psycopg2`
- Add connection pooling (PgBouncer or built-in)
- Effort: 8-16h depending on how many raw SQL queries exist

### Search Scaling
| Phase | Provider | Cost | Queries/day |
|-------|----------|------|-------------|
| Now | Serper (1600 credits left) | $50/2500 | ~50-100 |
| Phase 1 | Serper + DuckDuckGo fallback | $50/mo | ~200-500 |
| Phase 2 | SearXNG self-hosted | Free | Unlimited |
| Phase 3 | Google Custom Search API + SearXNG | $5/1000 queries | Unlimited |

---

## 5. Infrastructure Scaling

### Current (Single Machine)
```
Ahmad's PC (Ryzen 7800X3D, 32GB, 7900 XTX)
├── Python HTTP Server (port 8080)
├── Ollama (port 11434)
├── RSSHub Docker (port 1200)
├── Cloudflare Tunnel (random URL)
├── Cloudflare WARP (toggle for Docker)
└── SQLite (file-based)
```

### Phase 1 (Stable Single Machine)
```
Ahmad's PC
├── FastAPI + uvicorn (port 8080)
├── Ollama #1 — inference (port 11434, 7900 XTX)
├── Ollama #2 — scorer (port 11435, 2070 Super)
├── RSSHub Docker (port 1200, systemd)
├── Cloudflare Named Tunnel (stratos.yourdomain.com)
├── Redis (session cache, job queue)
└── SQLite WAL2 with connection pool
```

### Phase 2 (Docker Compose)
```yaml
# docker-compose.yml
services:
  stratos:
    build: ./backend
    ports: ["8080:8080"]
    depends_on: [redis, postgres]
    environment:
      - DATABASE_URL=postgresql://stratos:pass@postgres/stratos
      - REDIS_URL=redis://redis:6379
      - OLLAMA_HOST=http://host.docker.internal:11434

  frontend:
    image: nginx:alpine
    volumes: ["./frontend:/usr/share/nginx/html"]
    ports: ["80:80"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: stratos
      POSTGRES_PASSWORD: pass
    volumes: ["pgdata:/var/lib/postgresql/data"]

  rsshub:
    image: diygod/rsshub
    network_mode: host

volumes:
  pgdata:
```

### Phase 3 (Cloud Deploy)
```
Cloudflare Pages (frontend)
  ↓
Cloudflare Tunnel → VPS (Hetzner €4/mo)
  ├── FastAPI (Docker)
  ├── PostgreSQL
  ├── Redis
  └── Background workers (Celery)

Ollama stays local (or RunPod for heavy users)
CF Worker proxy (already deployed)
```

---

## 6. Cost Scaling

| Phase | Monthly Cost | What You Get |
|-------|-------------|--------------|
| Phase 0 (now) | $0 | Single user, local everything |
| Phase 1 | $0-5 | Cloudflare free tier, domain ($12/yr) |
| Phase 2 | $10-30 | Gemini API ($5), VPS ($4-10), domain |
| Phase 3 | $50-100 | VPS ($20), cloud GPU fallback ($30), APIs ($20), domain |
| Phase 3 + revenue | Break-even at 10 Pro users ($10/mo each) | |

---

## 7. Security Scaling

### Phase 0 (Do Now)
- [ ] Rate-limit `/api/proxy` — currently auth-exempt, open to abuse
- [ ] Add SSRF protection to proxy — validate target URLs, block internal IPs
- [ ] Audit all `eval()`, `exec()`, raw SQL for injection
- [ ] HTTPS enforcement (Cloudflare tunnel handles this)
- [ ] CORS whitelist (currently `*`)
- [ ] API key rotation mechanism

### Phase 1
- [ ] Google OAuth (replaces PIN auth)
- [ ] CSRF tokens on all POST endpoints
- [ ] Input sanitization on all user-facing fields
- [ ] XSS prevention in feed content rendering (HTML in RSS)
- [ ] Content Security Policy headers

### Phase 2
- [ ] JWT tokens (replace session table)
- [ ] Role-based access (admin, user, viewer)
- [ ] Audit logging (who did what when)
- [ ] Encrypted secrets (not plaintext in config.yaml)
- [ ] Penetration testing checklist

---

## 8. Feature Scaling Categories

### Category A: Intelligence Engine (Core Product)
The thing that makes StratOS StratOS. Highest priority.

| Feature | Phase | Effort |
|---------|-------|--------|
| Email digest of top signals | 1 | 4-6h |
| Gemini Flash summarizer for top articles | 1 | 4-6h |
| URL reader + summarizer in agent | 1 | 3-4h |
| YouTube video transcript summarizer | 1 | 3-4h |
| Semantic search across scored articles | 2 | 8-12h |
| Multi-language scoring (Arabic + English) | 2 | 12-20h |
| Custom alert rules (price thresholds, keyword triggers) | 2 | 8-12h |

### Category B: User Experience
What makes people come back daily.

| Feature | Phase | Effort |
|---------|-------|--------|
| Agent chat persistence across reloads | 0 | 4-6h |
| Interactive calendar with reminders | 2 | 20-40h |
| Raindrop-style bookmarks (tag + organize saved articles) | 1 | 8-12h |
| Hyperlinks in agent responses | 1 | 2h |
| Rich media feed improvements (video embeds, live status) | 1 | 8-12h |
| Weather + local info widget | 1 | 4-6h |
| PWA push notifications | 2 | 4-6h |

### Category C: Developer Experience
Makes building faster.

| Feature | Phase | Effort |
|---------|-------|--------|
| Claude Code sub-agents workflow | 0 | 2h learning |
| Comprehensive session prompt (v10) | 0 | 2h |
| Docker Compose for local dev | 1 | 4-6h |
| CI/CD pipeline (GitHub Actions) | 2 | 4-8h |
| Automated testing (pytest + frontend) | 1 | 8-12h |
| STATE.md + DECISIONS.md discipline | 0 | Ongoing |

### Category D: Portfolio & Career
What gets you hired.

| Feature | Phase | Effort |
|---------|-------|--------|
| Landing page with demo video | 2 | 8-16h |
| Architecture diagram (Excalidraw) | 1 | 2-4h |
| Performance benchmarks document | 1 | 2-4h |
| Security audit report | 1 | 4-8h |
| Senior Design capstone integration | 3 | Ongoing |
| Live demo with stable URL | 1 | 1h (named tunnel) |

---

## 9. Timeline (Aligned with Academics)

```
Mar 2026 ─── Phase 0: Stability + Security Audit
              ├── Fix B1 (profile isolation)
              ├── Security audit (/api/proxy, XSS, SSRF)
              ├── Named Cloudflare Tunnel
              └── RSSHub systemd + WARP toggle script

Apr 2026 ─── Phase 1 Start: Multi-User Foundation
              ├── FastAPI migration (biggest lift)
              ├── Dual GPU setup (riser cable)
              ├── Google OAuth
              └── Email digest + Gemini summarizer

May 2026 ─── Phase 1 Complete + Academic Prep
              ├── Agent fixes (chat persistence, free chat, expanded mode)
              ├── Bookmarks system
              └── STOP new features — exam prep

Jun-Jul 2026 ── Summer Semester (compressed Signals + OS)
              ├── Minimal StratOS work — maintenance only
              └── Phase 2 planning during breaks

Aug 2026 ─── Phase 2 Start: Cloud-Hybrid
              ├── Docker Compose
              ├── PostgreSQL migration
              ├── Redis + job queue
              └── Cloud GPU fallback

Sep-Dec 2026 ── Fall Semester (17 credits, hardware labs, Senior Design I)
              ├── StratOS IS the Senior Design project
              ├── Phase 2 complete
              ├── Landing page + demo
              └── V3 scorer training (large dataset pipeline)

Jan-May 2027 ── Spring Semester (18-20 credits + capstone)
              ├── Phase 3: SaaS features if time allows
              ├── Senior Design II presentation
              ├── Portfolio polish
              └── Job applications with live demo
```

---

## 10. Key Decisions to Make

| Decision | Options | Recommendation | When |
|----------|---------|----------------|------|
| Framework migration | FastAPI vs Flask vs keep http.server | FastAPI — async, WebSocket, auto-docs | Phase 1 (April) |
| Database | Keep SQLite vs PostgreSQL vs both | PostgreSQL for prod, SQLite for dev | Phase 1-2 |
| Frontend framework | Keep vanilla JS vs React vs Svelte | Keep vanilla until team grows | Defer |
| Cloud provider | Hetzner vs DigitalOcean vs Oracle Free | Hetzner (€4/mo, EU data) | Phase 2 |
| Summarization API | Gemini Flash vs Claude Haiku vs local | Gemini Flash ($0.01/1M tokens, fastest) | Phase 1 |
| Auth | Keep custom vs Firebase Auth vs Clerk | Google OAuth + custom JWT | Phase 1 |
| Monetization | Free forever vs freemium vs enterprise | Freemium (prove value first) | Phase 3 |

---

## 11. Metrics to Track

Start measuring these now so you have data for scaling decisions:

| Metric | How | Why |
|--------|-----|-----|
| Scan duration (total, fetch, score) | Log timestamps | Know when fetching > scoring (B8) |
| Ollama response time (p50, p95) | Log per-call timing | Know when to add cloud fallback |
| Serper credits burn rate | Already tracked | Know when to switch to SearXNG |
| Active profiles per day | DB query | Know when to scale auth/sessions |
| SQLite WAL file size | `ls -la strat_os.db-wal` | Know when to migrate to Postgres |
| Custom feed count per user | DB query | Know RSS proxy load |
| CF Worker requests/day | Cloudflare dashboard | Know if hitting 100K free limit |
| Error rate (500s, timeouts) | Log grep | General health |

---

## 12. What NOT to Scale (Keep Simple)

- **Don't add React/Vue** unless you have a second frontend developer
- **Don't microservice** until you have 1000+ users — monolith is fine
- **Don't multi-region** — single server with Cloudflare CDN handles global latency
- **Don't build a mobile app** — PWA is sufficient until App Store presence matters
- **Don't build admin panel** — use direct DB queries until Phase 3
- **Don't premature optimize** — profile before you refactor
