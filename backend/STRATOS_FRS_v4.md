# Functional Requirements Specification (FRS)

**Project:** StratOS — Strategic Intelligence Operating System  
**Version:** 4.0 (Multi-Profile Universal Platform)  
**Date:** February 17, 2026  
**Status:** Active Development (v2 Architecture)

---

## 1. Project Overview

StratOS is a self-hosted, profile-driven strategic intelligence platform. It aggregates news from multiple sources, tracks real-time financial markets, scores content relevance using a locally fine-tuned AI model, and delivers personalized briefings — all running on consumer-grade hardware with zero cloud dependency.

**Core Mission:** Tell the system who you are, and it tells you what matters today.

The system serves any professional profile — not just the original CPEG student use case. A user defines their identity (role, location, industry, tracked companies, interests) in a YAML configuration file, and the entire pipeline reconfigures automatically: what gets scraped, how it gets scored, what surfaces on the dashboard.

### 1.1 Intelligence Verticals

1. **Career Intel:** Job requirements, certifications, and hiring programs at tracked companies.
2. **Financial Advantage:** Banking deals, allowance transfers, investment-relevant offers from tracked institutions.
3. **Future Tech Trends:** Emerging technology developments relevant to the user's field and interests.
4. **Market Intelligence:** Real-time tracking of stocks, crypto, and precious metals with interactive charting.

---

## 2. System Architecture

Decoupled, headless architecture with profile-driven configuration.

### 2.1 Backend: The Intelligence Engine

- **Role:** Autonomous agent responsible for scraping, parsing, scoring, and serving data.
- **Schedule:** Runs periodically (configurable, typically every 15-60 minutes).
- **Language:** Python 3.12
- **AI Models:**
  - **Scorer:** Custom fine-tuned Qwen3-8B (DoRA, served via Ollama) — assigns relevance scores based on user profile.
  - **Agent:** General-purpose local model (Qwen3-12B or similar) — handles search query generation, briefing creation, and category generation.
- **Key Files:**
  - `scorer_adaptive.py` — Profile-aware scoring engine with tracked fields injection.
  - `export_training.py` — Training data export with profile metadata.
  - `distill.py` — Claude Opus teacher-student distillation pipeline.
  - `train_lora.py` — DoRA fine-tuning with best-checkpoint recovery.
  - `config.yaml` — System configuration and profile definitions.

### 2.2 Frontend: The Dashboard

- **Role:** Visualization and interaction layer.
- **Technology:** Single-Page Application (HTML, JavaScript, TailwindCSS).
- **Charting:** TradingView Lightweight Charts library for professional market visualization.
- **Access:** Served via Cloudflare Tunnel for secure remote access.

### 2.3 Authentication & Multi-Profile

- Device-isolated user profiles with per-device authentication.
- Profile selection drives all downstream behavior (scoring prompts, scraper queries, dashboard filters, market watchlists).
- Profiles defined in YAML — no code changes required to add new users.

---

## 3. Module A: Data Ingestion

### 3.1 Market Data Fetcher

**Source:** yfinance (Yahoo Finance API wrapper — free, no API key required).

**Default Watchlist (configurable per profile):**

| Category | Assets |
|----------|--------|
| Tech Leaders | NVDA (Nvidia), AMD (Advanced Micro Devices) |
| Precious Metals | GC=F (Gold), SI=F (Silver), HG=F (Copper), PL=F (Platinum) |
| Crypto | BTC-USD (Bitcoin), XRP-USD (XRP) |

**Data Resolution:** For each asset, fetch three timeframes:

1. **1-Minute:** Last 24 hours
2. **5-Minute:** Last 5 days
3. **15-Minute:** Last 5 days

**Visualization:** Interactive TradingView Lightweight Charts with visible data nodes at each interval for value inspection.

### 3.2 News Intelligence Fetcher

**Sources:** DuckDuckGo search API (free, no key required) + RSS feeds.

**Query Generation:** The agent model generates search queries dynamically based on the active user profile. Queries adapt to the user's tracked companies, institutions, interests, and location.

**Default Verticals (CPEG Kuwait profile):**

**Vertical 1 — Career & Certifications:**
- Tracked companies: Equate, SLB (Schlumberger), Halliburton, KOC, KNPC
- Goal: Fresh graduate openings, required certifications, job requirements

**Vertical 2 — Financial Benefits:**
- Tracked institutions: Warba Bank, Boubyan Bank, NBK
- Goal: Student/employee deals, allowance transfer bonuses, banking offers

**Vertical 3 — Future Tech:**
- Tracked interests: AI, quantum computing, semiconductors, next-gen battery, 6G
- Goal: Emerging trends, breakthrough developments, investment signals

### 3.3 Dynamic Category Generation

The agent model can generate new content categories based on user interests without manual configuration. If a topic gains relevance (e.g., a new bank launches a student campaign), the system detects and surfaces it automatically.

---

## 4. Module B: AI Processing

### 4.1 Scoring Model (stratos-scorer)

**Architecture:** Qwen3-8B base model, fine-tuned with DoRA (Weight-Decomposed Low-Rank Adaptation), rank 16, served via Ollama.

**Output Format:**
```
SCORE: X.X | REASON: [explanation of relevance to user profile]
```

**Score Range:** 0.0 – 10.0

**Scoring is profile-relative.** The same article receives different scores for different users based on their role, location, tracked entities, and interests. The scorer reads the full user profile from the system prompt at inference time.

### 4.2 Scoring Rubric

| Score Range | Classification | Examples |
|-------------|---------------|----------|
| 9.0 – 10.0 | Critical/Actionable | Direct career match (tracked company hiring for user's role), paradigm-shifting technology verified, free money/banking offer from tracked institution, market crash/spike >5% |
| 7.0 – 8.9 | High Importance | Relevant skills/certifications trending, regional industry growth, tracked company news (not directly actionable) |
| 5.0 – 6.9 | Moderate | Tangentially related industry news, general technology updates in user's field |
| 0.0 – 4.9 | Noise (filtered) | Generic ads, sports, celebrity news, political fluff, "Top 10" listicles |

**The Forbidden 5.0:** The scorer must commit — positively actionable (6.0+) or noise (4.0-). No fence-sitting at exactly 5.0.

### 4.3 Model Training Pipeline

1. **Data Export** (`export_training.py`): Extracts scored examples from the corrections database with full profile metadata (tracked companies, institutions, interests).
2. **Distillation** (`distill.py`): Claude Opus reviews local model scores, provides corrections, and generates labeled training examples. Supports multi-profile contrastive training — the same article scored by 8 different synthetic profiles.
3. **Training** (`train_lora.py`): DoRA fine-tuning on merged dataset (legacy corrections + Opus-labeled multi-profile examples). Includes best-checkpoint recovery for multi-epoch runs.
4. **Deployment:** Automatic LoRA merge → GGUF quantization (Q8_0) → Ollama model registration.

**Training Profiles (v2):**

| Profile | Location | Industry |
|---------|----------|----------|
| kuwait_cpeg | Kuwait | Computer Engineering / Oil & Gas |
| texas_nurse | Texas, USA | Healthcare / Nursing |
| london_finance | London, UK | Investment Banking |
| munich_mecheng | Munich, Germany | Automotive / Mechanical Engineering |
| bangalore_ds | Bangalore, India | Data Science / IT |
| dc_cybersec | Washington DC, USA | Cybersecurity / Government |
| dubai_founder | Dubai, UAE | Startup / E-commerce |
| toronto_teacher | Toronto, Canada | Education / EdTech |

**Automated Improvement Cycle:** The system identifies scoring disagreements between the local model and Opus corrections, collects them, retrains, and redeploys — improving accuracy over time with minimal manual intervention. Each distillation cycle costs approximately $0.40.

### 4.4 Validation Metrics (Phase 3)

| Metric | Target | Description |
|--------|--------|-------------|
| Profile Sensitivity Rate (PSR) | > 80% | Same article scored differently for different profiles |
| Mean Absolute Error (MAE) | < 1.5 | Average score deviation from Opus reference |
| Spearman ρ | > 0.80 | Rank-order correlation with Opus scoring |

---

## 5. Module C: Output & Display

### 5.1 News Display

**Visual Scoring Indicators:**

| Score | Border Style |
|-------|-------------|
| 9.0+ | Green pulsing border (Critical/Actionable) |
| 8.0+ | Blue border (High Importance) |
| 5.0 – 7.9 | Yellow border (Moderate) |
| < 5.0 | Red border (Noise — hidden from Executive Summary, visible in sub-menus) |

**News Item Fields:**
- `title`: Article headline
- `url`: Source link
- `summary`: AI-generated summary
- `timestamp`: Publication time (ISO 8601)
- `source`: Publisher name
- `root`: Classification — `global`, `regional`, `kuwait`, or `ai`
- `score`: Relevance score (0.0 – 10.0)
- `reason`: AI explanation of score

### 5.2 Market Display

Interactive TradingView charts for each watched asset with three timeframe tabs (1m, 5m, 15m). Charts display price, percentage change, and historical data with interactive crosshair for value inspection.

### 5.3 Briefings

AI-generated daily/periodic briefings summarizing the highest-scored items, organized by category, with actionable recommendations highlighted.

---

## 6. Infrastructure

### 6.1 Hardware Requirements (Reference Setup)

| Component | Specification |
|-----------|--------------|
| GPU | AMD Radeon 7900 XTX (24GB VRAM) |
| CPU | AMD Ryzen 7 5800X |
| RAM | 32GB+ |
| Storage | NVMe SSD (200GB+ free for models and training) |
| OS | Ubuntu 24.04 (dual-boot capable) |
| GPU Framework | AMD ROCm |

### 6.2 Software Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.12 |
| Frontend | HTML, JavaScript, TailwindCSS |
| AI Serving | Ollama |
| Training | PyTorch (ROCm), TRL, PEFT |
| Model Format | GGUF (Q8_0 quantization) |
| Search | DuckDuckGo API |
| Market Data | yfinance |
| News Feeds | RSS |
| Remote Access | Cloudflare Tunnel |
| Model Conversion | llama.cpp, cmake |

### 6.3 Cost Structure

| Item | Cost |
|------|------|
| Hardware | One-time (consumer GPU) |
| Search API | Free (DuckDuckGo) |
| Market Data | Free (yfinance) |
| News Feeds | Free (RSS) |
| AI Inference | Free (local Ollama) |
| Distillation Cycle | ~$0.40 per cycle (Claude API) |
| Ongoing Operation | $0.00 |

---

## 7. File Structure

```
~/Downloads/StratOS/StratOS1/backend/
├── config.yaml              # System + profile configuration
├── scorer_adaptive.py        # Profile-aware scoring engine
├── export_training.py        # Training data export
├── distill.py                # Opus distillation pipeline
├── train_lora.py             # DoRA fine-tuning + deployment
├── data/
│   ├── training_merged.jsonl # Combined training dataset
│   ├── models/
│   │   └── v15/              # Current model version
│   │       ├── lora_adapter/ # Trained LoRA weights
│   │       └── *.gguf        # Quantized model file
│   └── corrections.db        # Scoring corrections database
└── frontend/
    └── index.html            # Dashboard SPA
```

---

## 8. Roadmap

### Completed
- [x] Multi-source news aggregation (DuckDuckGo, RSS)
- [x] Real-time market tracking with TradingView charts
- [x] Local AI scoring with Ollama
- [x] Profile-driven architecture (YAML-based)
- [x] Device-isolated authentication
- [x] Model distillation pipeline (Claude Opus teacher)
- [x] Multi-profile contrastive training (8 profiles, 7,000+ examples)
- [x] DoRA fine-tuning with automated deployment
- [x] Dynamic category generation
- [x] Cloudflare Tunnel remote access

### In Progress
- [ ] v15 scorer validation (PSR, MAE, Spearman ρ)
- [ ] Automated continuous improvement cycle

### Future
- [ ] Agent model fine-tuning (search queries, briefings)
- [ ] Multi-profile LoRA (separate adapters per profile type)
- [ ] Additional data sources (LinkedIn, government portals)
- [ ] Mobile-optimized dashboard
- [ ] Air-gapped deployment documentation (for enterprise/government use)
