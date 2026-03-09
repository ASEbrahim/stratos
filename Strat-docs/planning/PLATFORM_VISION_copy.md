# StratOS Platform Vision — Multi-Mode Architecture

**Date:** March 9, 2026
**Status:** Vision document — not yet in development
**Prerequisite:** Intelligence Mode (current StratOS) must reach Phase 1 stability first

---

## 1. The Core Idea

StratOS today: *"Tell the system who you are, and it tells you what matters today"* — for professionals.

StratOS as a platform: The same sentence, but for **anyone with a passion they want to track.** The intelligence engine doesn't care whether it's scoring job postings, TCG price drops, anime episode releases, or Steam deals. It scores relevance against a profile. The profile defines what matters.

**Not one dashboard with different feeds.** Four purpose-built experiences, each designed to feel native to its audience, each drawing inspiration from the best products in that space. Shared engine underneath, completely separate surfaces on top.

---

## 2. Platform Architecture

### 2.1 What's Shared (The Engine)

```
┌──────────────────────────────────────────────────────────────┐
│                    STRATOS CORE ENGINE                        │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Auth & Users │  │ Profile Mgmt │  │ Scoring Pipeline    │ │
│  │ OAuth/PIN    │  │ Mode-aware   │  │ Rules → LLM → Sort  │ │
│  │ Sessions     │  │ Categories   │  │ Feedback loop       │ │
│  │ Permissions  │  │ Preferences  │  │ Distillation/train  │ │
│  └─────────────┘  └──────────────┘  └─────────────────────┘ │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Data Fetch   │  │ Database     │  │ Agent / Chat        │ │
│  │ RSS/Scrape   │  │ SQLite/PG    │  │ LLM inference       │ │
│  │ APIs         │  │ Migrations   │  │ Tool-use            │ │
│  │ Search       │  │ Feedback     │  │ Context-aware       │ │
│  └─────────────┘  └──────────────┘  └─────────────────────┘ │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ SSE / Events│  │ Notifications│  │ Price Tracker        │ │
│  │ Real-time   │  │ Email digest │  │ yfinance / custom    │ │
│  │ Scan status │  │ Push (PWA)   │  │ TradingView charts   │ │
│  └─────────────┘  └──────────────┘  └─────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Theme System: 8 themes × dark mode × stars × per-mode   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ OCR Engine   │  │ Audio Engine  │  │ File Upload         │ │
│  │ PaddleOCR-VL │  │ Whisper.cpp  │  │ Image/PDF/Audio     │ │
│  │ 1.5 (0.9B)  │  │ (CPU only)   │  │ → text → Agent      │ │
│  │ ~1GB VRAM   │  │ 0 VRAM       │  │                     │ │
│  └─────────────┘  └──────────────┘  └─────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

Everything in this box already exists or is already planned. Every mode reuses it.

### 2.2 What's Per-Mode (The Surfaces)

Each mode provides:

| Layer | What it does | Examples |
|-------|-------------|---------|
| **Page Layout** | Unique HTML structure, navigation, sections | Grid of card covers vs. dense signal list |
| **UI Components** | Domain-specific widgets | Airing calendar, collection grid, price chart |
| **Feed Sources** | Default RSS/API feeds for that domain | MAL RSS, TCGPlayer API, Steam news |
| **Wizard Presets** | Domain-specific onboarding categories | "What genres do you watch?" vs. "What TCGs do you collect?" |
| **Scorer Context** | How the LLM evaluates relevance | "Is this reprint relevant to a vintage Yu-Gi-Oh collector?" |
| **Agent Persona** | Domain-tuned chat behavior | "Recommend anime similar to..." vs. "What's the market for this card?" |
| **Data Models** | Domain-specific DB tables | `collection_items`, `watch_progress`, `game_backlog` |
| **OCR Use Cases** | Domain-specific image upload flows | Card photo → price lookup vs. Screenshot → article summary |

### 2.3 Routing Architecture

```
stratos.app/                    → Mode selector (or default mode based on profile)
stratos.app/intelligence        → Intelligence Mode (current StratOS)
stratos.app/anime               → Anime/Manga Mode
stratos.app/collection          → Collection/TCG Mode
stratos.app/gaming              → Gaming Mode

# OR subdomain approach:
intel.stratos.app
anime.stratos.app
collect.stratos.app
games.stratos.app
```

The frontend can be structured as:

```
frontend/
├── shared/                     # Core components used by all modes
│   ├── auth.js
│   ├── agent.js                # Chat agent (persona differs per mode)
│   ├── settings.js             # Shared settings framework
│   ├── ui.js                   # Themes, dark mode, stars
│   ├── nav.js                  # Sidebar (layout differs per mode)
│   ├── mobile.js               # PWA, gestures
│   └── charts.js               # TradingView (reused for price tracking)
│
├── intelligence/               # Current StratOS
│   ├── index.html
│   ├── feed.js
│   ├── markets-panel.js
│   ├── wizard.js
│   └── styles.css
│
├── anime/                      # Anime/Manga Mode
│   ├── index.html
│   ├── library.js              # Watch/read list management
│   ├── schedule.js             # Airing calendar
│   ├── discover.js             # Seasonal charts, recommendations
│   ├── media-feed.js           # Rich media grid (reuses feed.js patterns)
│   ├── wizard.js               # "What genres? What's your MAL?"
│   └── styles.css
│
├── collection/                 # TCG/Collectible Mode
│   ├── index.html
│   ├── inventory.js            # Collection manager
│   ├── market.js               # Price tracker, alerts
│   ├── browse.js               # Set browser, card search
│   ├── trades.js               # Want list, trade list
│   ├── wizard.js               # "What TCGs? Budget? Collecting goals?"
│   └── styles.css
│
├── gaming/                     # Gaming Mode
│   ├── index.html
│   ├── backlog.js              # Game library, play status
│   ├── deals.js                # Price drops, free games
│   ├── releases.js             # Upcoming release calendar
│   ├── updates.js              # Patch notes aggregator
│   ├── wizard.js               # "What platforms? Genres? Budget?"
│   └── styles.css
```

Backend stays monolithic — mode is just a profile attribute:

```python
# profiles table gets a 'mode' column
# mode: 'intelligence' | 'anime' | 'collection' | 'gaming'
# The scorer prompt adapts based on mode
# The wizard loads different presets based on mode
# The agent system prompt changes based on mode
```

---

## 3. Mode Breakdown — Intelligence (Current)

**Inspiration:** Bloomberg Terminal, Feedly Pro+, Morning Brew, The Daily Brief

### What Exists Today
- Dense signal feed scored 0-10
- Market charts (TradingView) with drawing tools
- AI agent for analysis
- Custom RSS feeds with rich media
- 4-step wizard for profile setup
- Email digest (planned F14)
- 8 themes with dark mode

### What It Feels Like
Professional. Data-dense. Dark UI. Information hierarchy: critical → high → medium → noise. Charts, numbers, timestamps. The user opens it at 7am with coffee and gets a briefing.

### Target Audience
Professionals, job seekers, investors, researchers, students tracking their field.

### Unique Components
- Executive Summary briefing
- Market panel with candlestick/line/area charts
- Career/employer intelligence
- Regional news tracking
- Financial advantage signals

### Status: **Active development. 80% complete.**

---

## 4. Mode Breakdown — Anime/Manga

**Inspiration from each competitor (study these):**

### AniList (anilist.co)
- Clean card-based grid layout with cover art
- Activity feed (social — who watched what)
- Airing schedule as a weekly calendar grid
- Statistics page (genres watched, time spent, score distribution)
- Manga/anime toggle
- Status categories: Watching, Completed, Paused, Dropped, Planning
- **What to steal:** The card grid layout, the status management, the weekly schedule view, the statistics dashboard

### MyAnimeList (myanimelist.net)
- More list-based than grid-based
- Detailed metadata (studios, staff, characters, reviews)
- Seasonal charts (all anime airing this season, sortable)
- Forums and recommendations
- **What to steal:** The seasonal chart concept, the depth of metadata, recommendation engine logic

### MangaDex (mangadex.org)
- Reader-focused — cover grid → chapter list → built-in reader
- Follow list with chapter update notifications
- Group/scanlation tracking
- Tag-based discovery
- **What to steal:** The follow + notification system for chapter updates

### Crunchyroll / Funimation
- Episode-centric — "new episodes today"
- Simulcast calendar
- Genre-based browsing
- **What to steal:** The "new today" urgency, simulcast calendar layout

### What StratOS Anime Mode Would Look Like

```
┌──────────────────────────────────────────────────────┐
│  StratOS Anime                                [User] │
├──────────┬───────────────────────────────────────────┤
│          │                                           │
│  Library │  AIRING THIS WEEK                         │
│  --------│  ┌─Mon──┬─Tue──┬─Wed──┬─Thu──┬─Fri──┐    │
│  Watching│  │Solo  │Demon │One   │MHA   │Chainsw│   │
│  Reading │  │Lev.  │Slay. │Piece │S8    │Man P2│   │
│  Planned │  │Ep 12 │Ep 45 │Ep... │Ep 3  │Ep 8  │   │
│  Dropped │  └──────┴──────┴──────┴──────┴──────┘    │
│  Compltd │                                           │
│  --------│  YOUR FEED (scored by StratOS engine)     │
│  Discover│  ┌─────────────────────────────────┐      │
│  Schedule│  │ [9.2] New Chainsaw Man OST leak │      │
│  Stats   │  │ [8.7] Blue Lock S2 announced    │      │
│  --------│  │ [8.1] MangaDex reader update    │      │
│  Agent 🤖│  │ [7.5] Spring 2026 seasonal chart│     │
│          │  └─────────────────────────────────┘      │
│  Settings│                                           │
│          │  LATEST CHAPTERS                          │
│          │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐     │
│          │  │Solo│ │JJK │ │CSM │ │OP  │ │BL  │     │
│          │  │Ch. │ │Ch. │ │Ch. │ │Ch. │ │Ch. │     │
│          │  │238 │ │271 │ │189 │ │1132│ │298 │     │
│          │  └────┘ └────┘ └────┘ └────┘ └────┘     │
│          │                                           │
│          │  MEDIA (from booru feeds — already built!) │
│          │  ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐      │
│          │  │  ││  ││  ││  ││  ││  ││  ││  │      │
│          │  └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘      │
└──────────┴───────────────────────────────────────────┘
```

### Data Sources

| Source | Type | What it provides | API/RSS |
|--------|------|-----------------|---------|
| AniList API | GraphQL API | Anime/manga metadata, airing schedule, user lists | `graphql.anilist.co` (free, no key) |
| Jikan API | REST API | MyAnimeList data (unofficial) | `api.jikan.moe` (free, rate-limited) |
| MangaDex API | REST API | Manga chapters, updates, covers | `api.mangadex.org` (free) |
| Livechart.me | RSS | Airing schedule, seasonal charts | RSS available |
| ANN | RSS | Anime news | `animenewsnetwork.com/all/rss.xml` |
| Crunchyroll | RSS | Episode releases | `crunchyroll.com/newsrss` |
| MAL RSS | RSS | User list updates | `myanimelist.net/rss.php?type=rw&u=USERNAME` |
| Booru feeds | RSS/Atom | Fan art (already built!) | yande.re, danbooru, konachan — already working |
| RSSHub | Various | nhentai, ehentai, pixiv (already working via Docker) | localhost:1200 |
| YouTube | RSS | Anime YouTubers, PVs, OSTs (already built!) | YouTube channel RSS |

### Unique Data Models

```sql
-- User's anime/manga library
CREATE TABLE anime_library (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER REFERENCES profiles(id),
    mal_id INTEGER,                    -- MyAnimeList ID (universal identifier)
    anilist_id INTEGER,                -- AniList ID
    title TEXT NOT NULL,
    type TEXT CHECK(type IN ('anime', 'manga')),
    status TEXT CHECK(status IN ('watching', 'reading', 'completed', 'paused', 'dropped', 'planned')),
    progress INTEGER DEFAULT 0,        -- Episodes watched / chapters read
    total INTEGER,                     -- Total episodes / chapters
    score REAL,                        -- User's personal score (1-10)
    cover_url TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chapter/episode update tracking
CREATE TABLE media_updates (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER REFERENCES profiles(id),
    mal_id INTEGER,
    update_type TEXT CHECK(update_type IN ('episode', 'chapter')),
    number INTEGER,                    -- Episode/chapter number
    title TEXT,
    source_url TEXT,                   -- Link to watch/read
    released_at TIMESTAMP,
    seen BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Airing schedule cache
CREATE TABLE airing_schedule (
    id INTEGER PRIMARY KEY,
    mal_id INTEGER,
    title TEXT,
    episode INTEGER,
    airing_at TIMESTAMP,
    day_of_week INTEGER,               -- 0=Mon, 6=Sun
    cover_url TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Wizard Flow (Anime Mode)

```
Step 1: "What do you watch/read?"
  → Genre multi-select: Action, Romance, Isekai, Seinen, Shounen, Slice of Life,
    Horror, Mecha, Sports, Psychological, Comedy, Fantasy, Sci-Fi
  → Format: Anime only, Manga only, Both

Step 2: "Import your list" (optional)
  → Paste MyAnimeList username → fetch via Jikan API
  → Or: Paste AniList username → fetch via GraphQL
  → Or: Skip and build manually

Step 3: "What matters to you?"
  → Categories (multi-select):
    - New episode alerts (airing shows I follow)
    - Chapter updates (manga I'm reading)
    - Seasonal preview (what's coming next season)
    - Industry news (studio announcements, staff changes)
    - Fan art feed (from booru sources)
    - Merch/figure deals
    - Convention/event news

Step 4: AI generates personalized categories
  → Same wizard pipeline as Intelligence mode
  → Example output for someone who watches seinen + psychological:
    "Dark Seinen Releases", "Studio MAPPA News", "Psychological Thriller Manga Updates"
```

### Scorer Adaptation

The scorer prompt changes based on mode. For anime:

```
You are scoring content relevance for an anime/manga enthusiast.

Profile: Watches seinen and psychological anime. Reads manga weekly.
Currently following: Chainsaw Man, Solo Leveling, Blue Lock, Frieren.
Preferred studios: MAPPA, Wit Studio, Ufotable.
Interests: Animation quality analysis, OST discussion, manga vs anime comparison.

Score this item 0-10 based on how likely this person would want to see it TODAY.
9-10: Direct update to a show/manga they follow, or major announcement from preferred studios
7-8: Highly relevant genre news, seasonal chart for genres they watch
5-6: General anime industry news, tangentially related content
3-4: Different genre/format than their preferences
0-2: Completely irrelevant (sports anime news for someone who only watches seinen)
```

### What's Already Built That Transfers

| Component | Status | Reuse |
|-----------|--------|-------|
| Booru image feeds (yande.re, danbooru, konachan) | ✅ Working | Direct reuse — fan art grid |
| Rich media grid with S/M/L toggle | ✅ Working | Direct reuse — cover art display |
| Image lightbox with keyboard navigation | ✅ Working | Direct reuse |
| RSSHub Docker (nhentai, ehentai, kemono) | ✅ Working | Direct reuse — manga sources |
| YouTube channel RSS auto-detect | ✅ Working | Direct reuse — anime YouTubers |
| CF Worker proxy for ISP-blocked sites | ✅ Working | Direct reuse — MangaDex is blocked |
| Scoring pipeline | ✅ Working | Reuse with adapted prompts |
| Agent chat | ✅ Working | Reuse with anime persona |
| Theme system | ✅ Working | Direct reuse — Sakura theme is literally made for this |
| PaddleOCR-VL 1.5 | ✅ Available | Upload manga page screenshot → OCR → translate. Upload screenshot → extract anime title from subtitles/UI |

**Anime mode has the highest code reuse of any mode. ~60% of the UI is already built.**

---

## 5. Mode Breakdown — Collection/TCG

**Inspiration from each competitor (study these):**

### TCGPlayer (tcgplayer.com)
- Price-centric: every card shows market price, low, mid, high
- Price history chart (candlestick/line — you already have TradingView!)
- Set browser with card grid
- Cart/buylist integration
- **What to steal:** Price chart per item (TradingView reuse), set browser layout, price alert concept

### Scryfall (scryfall.com) — Magic: The Gathering
- Powerful search syntax (color, type, rarity, price, format)
- Card image grid with hover for details
- Deck builder
- Price sourcing from multiple vendors
- **What to steal:** The search syntax concept, card grid with hover detail, multi-vendor price comparison

### Cardmarket (cardmarket.com)
- European marketplace — price trends across sellers
- Want list / offer matching
- Price guide with trend arrows
- **What to steal:** Want list matching, price trend visualization

### PriceCharting (pricecharting.com) — Video Games + Cards
- Price history for any collectible
- Collection value tracker (total portfolio value over time)
- Grading tier pricing (PSA 10 vs raw)
- **What to steal:** Portfolio value chart (reuse TradingView), grading-aware pricing

### Discogs (discogs.com) — Vinyl Records
- Collection management + marketplace
- Want list with notifications
- Release timeline
- **What to steal:** Collection management UX, release timeline, want list notifications

### What StratOS Collection Mode Would Look Like

```
┌──────────────────────────────────────────────────────┐
│  StratOS Collect                              [User] │
├──────────┬───────────────────────────────────────────┤
│          │                                           │
│  My Cards│  PORTFOLIO VALUE          $2,847 (+3.2%)  │
│  --------│  ┌─────────────────────────────────────┐  │
│  Yu-Gi-Oh│  │  📈 TradingView chart of total     │  │
│  Pokemon │  │     collection value over time      │  │
│  MTG     │  └─────────────────────────────────────┘  │
│  --------│                                           │
│  Browse  │  PRICE ALERTS                             │
│  Search  │  ┌─────────────────────────────────────┐  │
│  --------│  │ [🔴] Blue-Eyes Alt Art dropped 15%  │  │
│  Want    │  │ [🟢] Charizard ex hit target $45    │  │
│  Trade   │  │ [🟡] New Pokemon set presale open   │  │
│  --------│  └─────────────────────────────────────┘  │
│  Alerts  │                                           │
│  Market  │  YOUR FEED (scored by StratOS engine)     │
│  --------│  [9.1] Yu-Gi-Oh banlist update March 2026 │
│  Agent 🤖│  [8.5] Pokemon 151 reprint confirmed     │
│          │  [7.8] MTG Thunder Junction price crash   │
│  Settings│                                           │
│          │  COLLECTION GRID                          │
│          │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐     │
│          │  │card│ │card│ │card│ │card│ │card│     │
│          │  │$45 │ │$12 │ │$89 │ │$23 │ │$67 │     │
│          │  │ ▲3%│ │ ▼1%│ │ ▲8%│ │ ─  │ │ ▲2%│     │
│          │  └────┘ └────┘ └────┘ └────┘ └────┘     │
└──────────┴───────────────────────────────────────────┘
```

### Data Sources

| Source | Type | What it provides |
|--------|------|-----------------|
| TCGPlayer API | REST | Card prices, set data, market values |
| Scryfall API | REST | MTG card data, images, prices, rulings (free, no key) |
| PokémonTCG API | REST | Pokemon card data + images (free, key optional) |
| YGOPRODeck API | REST | Yu-Gi-Oh card data, prices, banlist (free) |
| PriceCharting API | REST | Cross-TCG price history |
| eBay API | REST | Recent sold listings for market price validation |
| Reddit RSS | RSS | r/yugioh, r/PokemonTCG, r/magicTCG — news + deals |
| YouTube | RSS | TCG YouTubers (market analysis, openings) — already built |
| Google News RSS | RSS | "[TCG name] reprint OR banlist OR set release" — already working pattern |

### Unique Data Models

```sql
-- User's collection
CREATE TABLE collection_items (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER REFERENCES profiles(id),
    tcg TEXT CHECK(tcg IN ('yugioh', 'pokemon', 'mtg', 'onepiece', 'digimon', 'other')),
    card_id TEXT,                       -- External ID (scryfall_id, ygoprodeck_id, etc.)
    name TEXT NOT NULL,
    set_name TEXT,
    set_code TEXT,
    rarity TEXT,
    condition TEXT CHECK(condition IN ('NM', 'LP', 'MP', 'HP', 'DMG')),
    graded BOOLEAN DEFAULT FALSE,
    grade_company TEXT,                 -- PSA, BGS, CGC
    grade_value REAL,                   -- 10, 9.5, 9, etc.
    quantity INTEGER DEFAULT 1,
    purchase_price REAL,               -- What you paid
    current_price REAL,                -- Latest market price
    image_url TEXT,
    status TEXT CHECK(status IN ('collection', 'trade', 'want', 'sold')),
    notes TEXT,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    price_updated_at TIMESTAMP
);

-- Price history (like market tickers but for cards)
CREATE TABLE card_price_history (
    id INTEGER PRIMARY KEY,
    card_id TEXT,
    tcg TEXT,
    price REAL,
    price_type TEXT CHECK(price_type IN ('market', 'low', 'mid', 'high')),
    source TEXT,                        -- tcgplayer, cardmarket, ebay
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Price alerts
CREATE TABLE price_alerts (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER REFERENCES profiles(id),
    card_id TEXT,
    name TEXT,
    condition TEXT CHECK(condition IN ('above', 'below', 'change_pct')),
    threshold REAL,                    -- Price threshold or % change
    triggered BOOLEAN DEFAULT FALSE,
    triggered_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Want list
CREATE TABLE want_list (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER REFERENCES profiles(id),
    card_id TEXT,
    name TEXT,
    tcg TEXT,
    max_price REAL,                    -- Maximum willing to pay
    priority TEXT CHECK(priority IN ('high', 'medium', 'low')),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### What Transfers From Existing Infrastructure

| Component | Reuse |
|-----------|-------|
| TradingView charts | **Direct reuse** — card price history = stock price history. Same chart component, different data source. This is the killer reuse. |
| Price alerts | Same architecture as market ticker alerts. Different threshold logic. |
| Scoring pipeline | Reuse with TCG-specific prompts ("Is this banlist update relevant to a Yu-Gi-Oh player?") |
| Agent chat | "What's the current market for PSA 10 Charizard?" — same agent, TCG persona |
| RSS feeds | Reddit TCG subs, YouTube TCG channels, Google News — all use existing RSS infrastructure |
| Image grid (S/M/L) | Card images in the same grid as booru art. Already built. |
| CF Worker proxy | If any TCG API is blocked by ISP |
| **PaddleOCR-VL 1.5** | **Killer feature for Collection mode.** Photo of card → OCR extracts name + set number → API price lookup → auto-add to collection. Also: photo of binder page → batch extract multiple cards. See §14 in Session Context doc. |

---

## 6. Mode Breakdown — Gaming

**Inspiration from each competitor:**

### Steam (store.steampowered.com)
- Game cards with cover art, tags, price, reviews
- Wishlist with price drop notifications
- Discovery queue (personalized recommendations)
- Activity feed (friends' purchases, achievements)
- **What to steal:** Wishlist + price alert, tag-based discovery, game card layout

### IGDB (igdb.com) / HowLongToBeat (howlongtobeat.com)
- Game database with metadata (platforms, genres, release dates)
- Playtime estimates (main story, completionist, average)
- Backlog management
- **What to steal:** Playtime estimates, backlog status tracking

### IsThereAnyDeal (isthereanydeal.com)
- Cross-store price comparison (Steam, Epic, GOG, Humble)
- Historical price charts
- Waitlist with price threshold alerts
- **What to steal:** Cross-store price tracking, historical low alerts

### Deku Deals (dekudeals.com) — Nintendo
- Clean card grid with current price + historical low
- Price drop alerts
- DLC tracking
- **What to steal:** The simplicity. Card + price + alert. Nothing else.

### What StratOS Gaming Mode Would Look Like

```
┌──────────────────────────────────────────────────────┐
│  StratOS Games                                [User] │
├──────────┬───────────────────────────────────────────┤
│          │                                           │
│  Backlog │  DEALS TODAY                              │
│  --------│  ┌────────────────────────────────────┐   │
│  Playing │  │ Elden Ring DLC    $29 → $19 (-34%) │   │
│  Finished│  │ Balatro           $14 → $9  (-36%) │   │
│  Dropped │  │ Hades II          $29 → $22 (-24%) │   │
│  Wishlist│  └────────────────────────────────────┘   │
│  --------│                                           │
│  Releases│  UPCOMING (from your wishlist)            │
│  Deals   │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  Updates │  │GTA 6│ │Hollo│ │Silk.│ │MH   │       │
│  --------│  │TBD  │ │Mar25│ │Apr 8│ │Jun  │       │
│  Agent 🤖│  │ PS5 │ │ PC  │ │ All │ │ All │       │
│          │  └─────┘ └─────┘ └─────┘ └─────┘       │
│  Settings│                                           │
│          │  YOUR FEED (scored by StratOS engine)     │
│          │  [9.5] Hollow Knight Silksong release date │
│          │  [8.2] Steam Spring Sale starts March 15  │
│          │  [7.8] Elden Ring DLC patch notes         │
│          │                                           │
│          │  PATCH NOTES                              │
│          │  ┌──────────────────────────────────────┐ │
│          │  │ Helldivers 2 — v1.4.2 balance patch  │ │
│          │  │ Lethal Company — v50 new moons       │ │
│          │  └──────────────────────────────────────┘ │
└──────────┴───────────────────────────────────────────┘
```

### Data Sources

| Source | Type | What it provides |
|--------|------|-----------------|
| Steam API | REST | Game data, prices, player counts (free, key required) |
| IGDB API (Twitch) | REST | Game metadata, release dates, covers (free, Twitch OAuth) |
| IsThereAnyDeal API | REST | Cross-store prices, historical lows (free key) |
| HowLongToBeat | Scrape | Playtime estimates |
| SteamDB RSS | RSS | Price changes, updates, new releases |
| itch.io | RSS | Indie games — already working in current feeds |
| GitHub releases | Atom | Open source game updates — already working (LAZER feed) |
| Reddit RSS | RSS | r/GameDeals, r/patientgamers, r/NintendoSwitch |
| YouTube | RSS | Game review channels — already built |
| Patch notes RSS | Various | Per-game subreddits, official blogs |

### Unique Data Models

```sql
-- User's game library / backlog
CREATE TABLE game_library (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER REFERENCES profiles(id),
    igdb_id INTEGER,
    steam_id INTEGER,
    title TEXT NOT NULL,
    platform TEXT,                      -- PC, PS5, Switch, Xbox, Mobile
    status TEXT CHECK(status IN ('playing', 'finished', 'dropped', 'wishlist', 'backlog')),
    playtime_hours REAL,
    completion_pct REAL,
    personal_score REAL,               -- 1-10
    cover_url TEXT,
    purchase_price REAL,
    store TEXT,                         -- steam, epic, gog, itch, physical
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Price tracking (cross-store)
CREATE TABLE game_prices (
    id INTEGER PRIMARY KEY,
    igdb_id INTEGER,
    store TEXT,
    price REAL,
    original_price REAL,
    discount_pct REAL,
    url TEXT,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Wishlist alerts
CREATE TABLE game_alerts (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER REFERENCES profiles(id),
    igdb_id INTEGER,
    title TEXT,
    alert_type TEXT CHECK(alert_type IN ('price_drop', 'release', 'update', 'free')),
    price_threshold REAL,              -- For price_drop alerts
    triggered BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 7. Implementation Strategy

### Order of Development

```
Phase 0-1 (Now → June 2026):
  └── Intelligence Mode to completion (current StratOS)
      This is the portfolio piece. The senior design project.
      Everything else waits until this is solid.

Phase 2 (Aug → Dec 2026):
  └── Anime/Manga Mode (FIRST expansion)
      Why first:
        - 60% of UI already built (media grid, booru feeds, lightbox, RSSHub)
        - AniList API is free, no key needed, excellent documentation
        - Sakura theme is literally designed for this audience
        - MangaDex already in your blocked_domains + CF Worker proxy
        - Ahmad already uses these services (dogfooding)
        - Smallest gap between "what exists" and "what's needed"

Phase 3 (Spring 2027):
  └── Collection/TCG Mode (SECOND expansion)
      Why second:
        - TradingView price charts transfer directly (biggest code reuse)
        - Multiple free card APIs with no auth
        - Clear monetization path (collectors track real money)
        - More complex data models but simpler UI than anime

  └── Gaming Mode (THIRD, or parallel with TCG)
      Why third:
        - Most API dependencies (Steam, IGDB, ITAD all need keys)
        - Highest competition (Steam itself is the incumbent)
        - But also highest audience (every gamer wants deal alerts)
```

### How to Build Each Mode

**Step 1: Data Sources (1-2 days)**
- Identify 3-5 free APIs for the domain
- Build fetcher adapters (same pattern as `fetchers/news.py`)
- Verify RSS feeds work through existing infrastructure
- Add any blocked domains to CF Worker allowlist

**Step 2: Data Models (1 day)**
- Add domain-specific tables via migration framework (already exists)
- Build basic CRUD endpoints in server.py (or FastAPI if migrated by then)
- Test with manual inserts

**Step 3: Wizard Presets (1 day)**
- Create domain-specific category presets
- Adapt wizard flow for the domain ("What genres?" instead of "What industry?")
- Test AI category generation with domain-appropriate prompts

**Step 4: Scorer Adaptation (half day)**
- Write domain-specific scorer prompt template
- Test on 50 sample articles from the domain
- Tune thresholds (what's 9+ for an anime fan vs. a card collector?)

**Step 5: Frontend (3-5 days)**
- Create mode-specific `index.html` and JS files
- Reuse shared components (auth, agent, settings, themes, charts)
- Build domain-specific widgets (calendar, collection grid, backlog tracker)
- Mobile-responsive from the start

**Step 6: Agent Persona (half day)**
- Write domain-specific system prompt for the agent
- Test with 20 representative queries
- Ensure tool-use works (search, fetch, analyze)

**Total per mode: ~8-12 days of focused development.**

### Shared Component Extraction

Before building the second mode, extract these from Intelligence mode into shared modules:

| Component | Current Location | Extract To | Used By |
|-----------|-----------------|------------|---------|
| Auth flow | `auth.js`, `auth.py` | `shared/auth.js` | All modes |
| Agent chat | `agent.js`, `routes/agent.py` | `shared/agent.js` | All modes (persona differs) |
| Theme system | `ui.js`, `styles.css` | `shared/ui.js` | All modes |
| Image grid + lightbox | `feed.js` | `shared/media-grid.js` | Anime, Collection |
| Price charts | `markets-panel.js` | `shared/charts.js` | Intelligence, Collection, Gaming |
| Settings framework | `settings.js` | `shared/settings.js` | All modes (tabs differ) |
| Wizard framework | `wizard.js` | `shared/wizard.js` | All modes (presets differ) |
| RSS feed management | `settings.js`, `feed.js` | `shared/feeds.js` | All modes |
| Mobile gestures + PWA | `mobile.js` | `shared/mobile.js` | All modes |

This extraction is a prerequisite for the second mode. It's also the natural outcome of the FastAPI migration in Phase 1 — you'd restructure the frontend at the same time.

---

## 8. Monetization Per Mode

| Mode | Free Tier | Pro Tier ($10/mo) |
|------|-----------|-------------------|
| Intelligence | 3 categories, 1 scan/day, prompt-based scoring | Unlimited categories, unlimited scans, fine-tuned scorer, email digest |
| Anime | 10 tracked shows, basic feed | Unlimited library, chapter alerts, import from MAL/AniList, rich media feed |
| Collection | 50 cards tracked, basic prices | Unlimited collection, price history charts, alerts, want list matching, portfolio value tracker |
| Gaming | 20 wishlist games, basic deals | Unlimited backlog, cross-store price tracking, historical lows, patch note aggregation |

**Bundle: $25/mo for all modes.** This is the power user tier — someone who's a professional, anime fan, card collector, and gamer. They exist (they're your AUK classmates).

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Scope creep kills Intelligence mode | High | Fatal | **Hard rule: Intelligence mode Phase 1 complete before ANY other mode starts** |
| API rate limits on free tiers | Medium | Annoying | Cache aggressively, SearXNG for search, respect rate limits |
| Too many modes, none polished | High | Bad product | Start with 2 modes maximum. Intelligence + Anime. Add more only if both are solid. |
| Senior Design scope too large | Medium | Academic risk | Scope the capstone to Intelligence mode + one expansion. The platform architecture is the innovation, not the number of modes. |
| Burnout from 18-credit semesters + StratOS | High | Health risk | Maintenance-only during compressed semesters. No new modes during finals. |

---

## 10. What This Means for the Session Prompt (v10)

The next version of `claude_code_session_prompt` should include:

1. **Mode concept:** Profile has a `mode` field. Scorer prompt, wizard presets, agent persona, and frontend layout all vary by mode.
2. **Shared vs. per-mode file structure:** Document which files are shared and which are mode-specific.
3. **Current model stack update:** Qwen3.5-9B (not 30B-A3B), OLLAMA_MAX_LOADED_MODELS=2.
4. **New infrastructure:** RSSHub Docker, CF Worker proxy, WARP, blocked_domains.
5. **Platform vision reference:** Link to this document for any Claude Code session that touches multi-mode architecture.

---

## 11. Summary

**What StratOS becomes:**

```
Today:    A news intelligence dashboard for professionals
Tomorrow: A personal intelligence PLATFORM that adapts to any passion

The engine is the same: "Tell me who you are, I'll tell you what matters."
The surfaces are different: Bloomberg for professionals, AniList for anime fans,
                           TCGPlayer for collectors, Steam for gamers.
```

**What to build, in what order:**

```
1. Intelligence Mode → COMPLETE IT (Phase 0-1, now → June 2026)
2. Anime/Manga Mode → FIRST EXPANSION (Phase 2, Aug 2026)
3. Collection/TCG Mode → SECOND (Phase 3, early 2027)
4. Gaming Mode → THIRD (Phase 3, if time permits)
```

**Why this works:**
- Each mode has a clear real-world competitor to learn from
- 60-80% of infrastructure is shared across all modes
- Each mode has its own monetization path
- The platform story is stronger than any single mode for employers
- "I built a platform that serves multiple audiences with shared AI infrastructure" is a better senior design pitch than "I built a news reader"
