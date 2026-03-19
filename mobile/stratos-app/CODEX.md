# StratOS Mobile App — Codex

> Complete technical reference for the StratOS mobile application.
> Theme: **5 themes** — Arcane (default), Sakura, Nebula, Cosmos, Noir

---

## Architecture Overview

```
stratos-app/
├── app/                    # Expo Router — file-based routing (15 screens)
│   ├── _layout.tsx         # Root: auth check, error boundary, offline banner
│   ├── index.tsx           # Entry redirect: → discover
│   ├── (auth)/             # Auth flow (no tabs)
│   │   ├── login.tsx       # Email/password + StarParallax background
│   │   └── register.tsx    # Name/email/password + StarParallax background
│   ├── (tabs)/             # Main app (bottom tab navigator)
│   │   ├── _layout.tsx     # Tab bar: Discover, Chats, Library, Create, Profile
│   │   ├── discover.tsx    # Character browsing: trending, genres, grid, search, theme picker
│   │   ├── chats.tsx       # Chat session history: search, resume, delete
│   │   ├── library.tsx     # My Characters + Saved + Chat History tabs
│   │   ├── create.tsx      # Character editor (Simple + Advanced modes)
│   │   └── profile.tsx     # Stats, theme selector, NSFW filter, settings
│   ├── character/[id].tsx  # Character detail: rate, save, edit, clone, generate image
│   ├── chat/[id].tsx       # RP chat: SSE streaming, branches, swipes, edits, director notes
│   ├── imagegen/index.tsx  # Image generation: CHROMA model, steps slider, gallery
│   └── gaming/
│       ├── scenarios.tsx   # Gaming scenario browser
│       └── [id].tsx        # Gaming session: stat bar, option buttons, GM chat
├── components/             # Reusable UI (37 components)
│   ├── cards/ (6)          # Character card rendering
│   │   ├── CharacterCard.tsx      # Grid + horizontal + featured variants
│   │   ├── CharacterDetail.tsx    # Full detail: rate, save, edit, clone, share
│   │   ├── CharacterDepthSection.tsx  # Collapsible quality elements
│   │   ├── QualityScore.tsx       # 6-element quality badge
│   │   ├── SimilarCharacters.tsx  # Genre-matched carousel
│   │   └── TagPills.tsx           # Genre tag pills
│   ├── chat/ (13)          # Chat interface
│   │   ├── MessageBubble.tsx      # Markdown formatting (*bold*, *italic*, quotes)
│   │   ├── ChatInput.tsx          # Auto-expanding textarea + send
│   │   ├── ChatInputSection.tsx   # Composite: input + suggestions + director note
│   │   ├── ChatMessageList.tsx    # FlatList with edit/regen/feedback buttons
│   │   ├── SessionHeader.tsx      # Avatar, name, typing indicator, menu
│   │   ├── SuggestionChips.tsx    # Staggered animated action chips
│   │   ├── TypingIndicator.tsx    # Three-dot bob animation
│   │   ├── BranchSelector.tsx     # Branch dropdown for conversation timelines
│   │   ├── DirectorNoteBar.tsx    # Collapsible director note input
│   │   ├── EditSheet.tsx          # Bottom sheet message editor with reason tags
│   │   ├── FeedbackButtons.tsx    # Thumbs up/down for training data
│   │   ├── SwipeIndicator.tsx     # Swipe navigation (1/3 with prev/next)
│   │   └── TrainingOptIn.tsx      # Privacy opt-in modal
│   ├── creator/ (8)        # Character creation wizard
│   │   ├── CardEditor.tsx         # Orchestrator: mode toggle + progress bar
│   │   ├── SimpleEditor.tsx       # 3 sections: Identity, Dynamics, Story
│   │   ├── AdvancedEditor.tsx     # 5 sections: + Style, Depth
│   │   ├── CategoryCard.tsx       # Animated collapsible card + wizard popup modal
│   │   ├── PillSelector.tsx       # Single/multi-select pills + 11 preset arrays
│   │   ├── EditorActions.tsx      # Import, Generate Image, Content Rating, Save
│   │   ├── AvatarPicker.tsx       # Camera/gallery picker with preview
│   │   └── GuidedFields.tsx       # Numbered quality element fields
│   ├── gaming/ (3)         # Gaming mode
│   │   ├── ScenarioCard.tsx       # Scenario browser card
│   │   ├── StatBar.tsx            # HP/stats display with icons
│   │   └── OptionButtons.tsx      # Staggered numbered choice buttons
│   └── shared/ (8)         # App-wide
│       ├── StarParallax.tsx       # Particle system (theme-aware colors)
│       ├── LoadingScreen.tsx      # Centered spinner
│       ├── EmptyState.tsx         # Animated empty state with icon
│       ├── Header.tsx             # Back button + title + optional right action
│       ├── ErrorBoundary.tsx      # Class-based error catch
│       ├── OfflineBanner.tsx      # Animated offline/online banner (web)
│       ├── Skeleton.tsx           # Shimmer skeleton loaders (card, row, full)
│       └── ThemedAlert.tsx        # Custom themed alert modal + useThemedAlert hook
├── lib/                    # Business logic (13 modules)
│   ├── types.ts            # All interfaces + AuthError/ApiError + quality scoring
│   ├── api.ts              # Fetch wrapper: auth headers, 401 handling, GET retry
│   ├── auth.ts             # Login, register, profile, logout
│   ├── chat.ts             # SSE streaming, cancel, suggestions, message IDs
│   ├── characters.ts       # CRUD, search, trending, pagination
│   ├── gaming.ts           # Scenarios, session start, option parsing
│   ├── rp.ts               # RP expansion: branches, swipes, edits, feedback, image gen
│   ├── sse.ts              # SSE parser: timeout, stall detection, abort support
│   ├── storage.ts          # AsyncStorage: sessions, cards, stats
│   ├── mappers.ts          # BackendCard → CharacterCard type mapping
│   ├── tavern-import.ts    # TavernCard V2 PNG parser + JSON mapper
│   ├── mock.ts             # 10 characters + 3 scenarios + mock data
│   └── utils.ts            # safeParse, buildUrl, reportError
├── stores/                 # Zustand v5 state management (4 stores)
│   ├── authStore.ts        # User session + concurrent login guard
│   ├── chatStore.ts        # Session, messages, streaming, persist(flush)
│   ├── characterStore.ts   # Lists, search, pagination, save/delete
│   └── themeStore.ts       # 5 themes + NSFW filter + AsyncStorage persist
├── constants/              # Configuration (5 files)
│   ├── config.ts           # API_BASE, USE_MOCKS, timeouts, limits
│   ├── theme.ts            # Static fallback colors, spacing, typography, borderRadius
│   ├── themes.ts           # 5 theme definitions (Arcane, Sakura, Nebula, Cosmos, Noir)
│   ├── genres.ts           # 7 genre definitions + color mapping
│   └── fonts.ts            # Quicksand + Poppins font family mapping
└── assets/                 # Images, fonts
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Framework | React Native + Expo 54 | Cross-platform mobile |
| Language | TypeScript (strict, noImplicitReturns) | Type safety |
| Routing | Expo Router v6 | File-based navigation |
| State | Zustand v5 | Lightweight global state |
| Animation | Reanimated v4 | 60fps UI-thread animations |
| Icons | Lucide React Native | Consistent icon set (20+ icons used) |
| Images | expo-image | Cached image loading |
| Auth Storage | expo-secure-store | Encrypted token storage |
| Haptics | expo-haptics | Tactile feedback |
| Slider | @react-native-community/slider | Image gen steps control |

---

## Themes (5 available)

| Theme | Accent | Description |
|-------|--------|-------------|
| **Arcane** (default) | `#4fa8d4` hextech blue | League of Legends Arcane — deep indigo, brass warmth |
| **Sakura** | `#f0a0b8` pink | Cherry blossom aesthetic — soft pinks, lavender |
| **Nebula** | `#38bdf8` cyan | Deep space — violet + cyan gradients |
| **Cosmos** | `#e8b931` gold | Golden starlight — warm blue + gold |
| **Noir** | `#8b5cf6` violet | Pure black + electric violet |

Each theme defines: bg (4), text (5), accent (12 including genre colors), border (2), status (3), quality (4), star/petal RGB values, glow gradients.

---

## Data Flow

### Auth Flow
```
App Launch → checkAuth() → getToken() from SecureStore
  ├─ Token exists → getProfile() → isAuthenticated = true → Discover
  └─ No token → isAuthenticated = false → Discover (anonymous browsing)
Login → POST /api/auth/login → setToken() → navigate to Discover
```

### Chat Flow (RP)
```
Start Session → startSession(card) → create sessionId + first_message
Send Message → add user msg → persistSession(flush) → streamMessage()
  ├─ SSE chunks → update streamingContent (live typing effect)
  ├─ Done → add assistant msg → persistSession() → loadSuggestions()
  └─ Navigate away → persistSession(flush) preserves messages
Resume → loadRecentSessions() → resumeSession(session)
```

### Character Creation Flow
```
Create Tab → CardEditor (Simple/Advanced mode toggle)
  ├─ Simple: 3 sections (Identity, Dynamics, Story) — 8 CategoryCards
  ├─ Advanced: 5 sections (+ Style, Depth) — 19 CategoryCards
  ├─ Pill-only fields → CategoryPopup (wizard-style modal, wrapping pills)
  ├─ Text fields → CategoryCard (inline collapsible accordion)
  └─ Progress bar tracks filled fields / total
Save → POST /api/cards → incrementStat → navigate back
```

### Image Generation Flow
```
Generate Portrait → ensure_comfyui() → GPU swap from Ollama
  ├─ VRAM verified via rocm-smi before starting ComfyUI
  ├─ CHROMA model: 4-32 steps (slider), 3 styles, 3 sizes
  ├─ Poll ComfyUI /history until complete
  └─ Save to gallery + optional set-as-avatar
```

---

## API Endpoints (40+ endpoints across 9 backend routes)

### Auth
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/auth/login` | Authenticate user |
| POST | `/api/auth/register` | Create account |
| GET | `/api/auth/me` | Fetch user profile |

### Character Cards
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/cards/trending` | Top trending cards |
| GET | `/api/cards/browse` | Browse with pagination |
| GET | `/api/cards/search` | Search by name/genre |
| GET | `/api/cards/{id}` | Single card detail |
| POST | `/api/cards` | Create new card |
| PUT | `/api/cards/{id}` | Update card |
| DELETE | `/api/cards/{id}` | Delete card (creator only) |
| GET | `/api/cards/my` | User's created cards |
| POST | `/api/cards/{id}/rate` | Rate 1-5 |
| POST | `/api/cards/{id}/publish` | Publish card |
| POST | `/api/cards/{id}/save` | Save to library |

### RP Chat
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/rp/chat` | Send message (SSE stream) |
| POST | `/api/rp/regenerate` | Swipe: regenerate last AI message |
| POST | `/api/rp/swipe` | Select swipe variant |
| POST | `/api/rp/edit` | Edit message (DPO pair) |
| POST | `/api/rp/branch` | Branch conversation (SSE) |
| POST | `/api/rp/director-note` | Set director's note |
| POST | `/api/rp/feedback` | Thumbs up/down |
| GET | `/api/rp/history/{sid}` | Conversation + branches |
| GET | `/api/rp/branches/{sid}` | List branches |
| POST | `/api/rp/opt-in` | Training data opt-in |

### Image Generation
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/image/generate` | Text-to-image (CHROMA) |
| POST | `/api/image/character-portrait` | Portrait from card |
| GET | `/api/image/gallery` | Generation history |
| GET | `/api/image/{id}` | Serve image |
| DELETE | `/api/image/{id}` | Delete image |

### Gaming
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/scenarios` | List scenarios |
| GET | `/api/scenarios/{id}` | Scenario detail |
| POST | `/api/scenarios/{id}/start` | Start gaming session |

---

## Character Creation — Wizard UI

### CategoryPopup (pill-only fields — opens as wizard modal)
- Gender, Age Range, Archetype, Relationship, Personality Tags
- POV, Response Length, NSFW Comfort
- Full-screen centered panel with wrapping pills, themed icon header, Done button

### CategoryCard (text fields — inline collapsible accordion)
- Description, Physical Description, Personality, Speech Pattern
- First Message, Scenario, all Depth fields
- Animated height + chevron rotation, accent strip + completion indicator

### Icon + Color Assignments
| Category | Icon | Color |
|----------|------|-------|
| Gender | User | accent.primary |
| Age Range | Clock | accent.primary |
| Archetype | Crown | accent.secondary |
| Relationship | Heart | accent.romance |
| Personality Tags | Sparkles | accent.anime |
| Description | ScrollText | accent.fantasy |
| Personality | Brain | accent.modern |
| Speech Pattern | MessageCircle | accent.historical |
| First Message | Feather | accent.fantasy |
| Scenario | Globe | accent.horror |
| Narration POV | BookOpen | accent.modern |
| NSFW Comfort | Shield | nsfw |

---

## RP Chat Features

- **Swipe** — regenerate last AI message only (same branch)
- **Branch** — create new timeline at earlier message
- **Director's Note** — one-shot steering instruction for next AI response
- **Edit** — modify AI/user messages (creates DPO pair for training)
- **Feedback** — thumbs up/down on messages
- **Session Context** — persistent context injected every turn
- **3-tier memory** — facts (regex, every turn), conversation (sliding window), arcs (LLM, every 10 turns)

---

## Character Card Quality System

6 elements scored:
1. **Physical Description** — appearance + one unique detail
2. **Speech Pattern** — how they talk (formal, slang, poetic)
3. **Emotional Trigger** — what provokes strong reaction
4. **Defensive Mechanism** — how they protect themselves emotionally
5. **Vulnerability** — the crack in the armor
6. **Specific Detail** — one concrete grounding detail

| Elements Filled | Level | Color |
|----------------|-------|-------|
| 0-2 | Basic | `#5a6080` (dim steel) |
| 3-4 | Good | `#d4a044` (Piltover gold) |
| 5 | Great | `#7cc4e8` (bright hextech) |
| 6 + first_message + scenario | Exceptional | `#c468e0` (Shimmer violet) |

---

## Glossary

| Term | Definition |
|------|-----------|
| **Branch** | New conversation timeline created from an earlier message |
| **CategoryCard** | Animated collapsible card wrapper for character editor fields |
| **CategoryPopup** | Wizard-style modal for pill-only fields (wrapping layout) |
| **CHROMA** | Image generation model (8.9B, Flux.1-schnell based, natively uncensored) |
| **Director's Note** | One-shot instruction that steers the AI's next response |
| **DPO** | Direct Preference Optimization — training from edited message pairs |
| **GPU Manager** | Auto-swaps Ollama/ComfyUI with VRAM verification via rocm-smi |
| **Pill** | Tappable selection chip for categorical character attributes |
| **Quality Score** | 0-6 rating based on filled character card elements |
| **Reanimated** | React Native animation library running on UI thread |
| **RP (Roleplay)** | Free-form conversation where the AI plays a character |
| **SectionHeader** | Group title with progress counter (e.g., "2/4") |
| **SSE** | Server-Sent Events — streaming protocol for chat |
| **Swipe** | Regenerate only the last AI message (same branch) |
| **TavernCard V2** | Character card format in PNG metadata — SillyTavern compatible |
| **Zustand** | Lightweight React state management |
