# StratOS Mobile App — Codex

> Complete technical reference for the StratOS mobile application.
> Theme: **Arcane** (Hextech blue + Piltover gold + Zaun green)

---

## Architecture Overview

```
stratos-app/
├── app/                    # Expo Router — file-based routing (13 screens)
│   ├── _layout.tsx         # Root: auth check, error boundary, status bar
│   ├── index.tsx           # Entry redirect: → login or → discover
│   ├── (auth)/             # Auth flow (no tabs)
│   │   ├── login.tsx       # Email/password + StarParallax background
│   │   └── register.tsx    # Name/email/password + StarParallax background
│   ├── (tabs)/             # Main app (bottom tab navigator)
│   │   ├── _layout.tsx     # Tab bar config: Discover, Library, Create, Profile
│   │   ├── discover.tsx    # Character browsing: trending, genres, grid, search
│   │   ├── library.tsx     # My Characters + Saved Characters
│   │   ├── create.tsx      # Character card creator (Quick + Advanced)
│   │   └── profile.tsx     # User info, stats, settings, logout
│   ├── character/[id].tsx  # Character detail: full card + Start Conversation
│   ├── chat/[id].tsx       # RP chat: SSE streaming, suggestions, haptics
│   └── gaming/
│       ├── scenarios.tsx   # Gaming scenario browser
│       └── [id].tsx        # Gaming session: stat bar, option buttons, GM chat
├── components/             # Reusable UI (16 components)
│   ├── cards/              # Character card rendering
│   │   ├── CharacterCard.tsx    # Grid + horizontal variants
│   │   ├── CharacterDetail.tsx  # Full character sheet + CTA
│   │   ├── QualityScore.tsx     # 6-element quality indicator
│   │   └── TagPills.tsx         # Genre tag pills
│   ├── chat/               # Chat interface
│   │   ├── MessageBubble.tsx    # User/assistant/system bubbles + *action* italics
│   │   ├── ChatInput.tsx        # Auto-expanding textarea + send button
│   │   ├── SuggestionChips.tsx  # Tappable action suggestions
│   │   ├── SessionHeader.tsx    # Character avatar + name + back
│   │   └── TypingIndicator.tsx  # Animated 3-dot indicator
│   ├── creator/            # Character creation
│   │   ├── CardEditor.tsx       # Quick/Advanced mode form
│   │   └── GuidedFields.tsx     # 6-element quality guide
│   ├── gaming/             # Gaming mode
│   │   ├── ScenarioCard.tsx     # Scenario browser card
│   │   ├── StatBar.tsx          # HP/stats display
│   │   └── OptionButtons.tsx    # Numbered choice buttons
│   └── shared/             # App-wide
│       ├── StarParallax.tsx     # Particle system (stars + hex motes + shooters)
│       ├── LoadingScreen.tsx    # Centered spinner
│       ├── EmptyState.tsx       # Title + subtitle for empty lists
│       ├── Header.tsx           # Back button + title
│       └── ErrorBoundary.tsx    # Class-based error catch
├── lib/                    # Business logic (7 modules)
│   ├── types.ts            # All TypeScript interfaces + quality scoring
│   ├── api.ts              # Fetch wrapper: auth headers, 401 handling
│   ├── auth.ts             # Login, register, profile, logout
│   ├── chat.ts             # SSE streaming, suggestions, message IDs
│   ├── characters.ts       # CRUD, search, trending
│   ├── gaming.ts           # Scenarios, option parsing
│   └── mock.ts             # 6 character cards + 3 scenarios + mock data
├── stores/                 # Zustand state management (3 stores)
│   ├── authStore.ts        # User session
│   ├── chatStore.ts        # Active conversation + streaming state
│   └── characterStore.ts   # Character lists + search + genre filter
├── constants/              # Configuration (3 files)
│   ├── config.ts           # USE_MOCKS flag, API_BASE URL
│   ├── theme.ts            # Arcane color palette, spacing, typography
│   └── genres.ts           # Genre definitions + color mapping
└── assets/                 # Images, fonts
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Framework | React Native + Expo 54 | Cross-platform mobile |
| Language | TypeScript (strict mode) | Type safety |
| Routing | Expo Router v6 | File-based navigation |
| State | Zustand v5 | Lightweight global state |
| Animation | Reanimated v4 | 60fps UI-thread animations |
| Icons | Lucide React Native | Consistent icon set |
| Images | expo-image | Cached image loading |
| Auth Storage | expo-secure-store | Encrypted token storage |
| Haptics | expo-haptics | Tactile feedback |
| Lists | @shopify/flash-list | Virtualized list performance |

---

## Theme: Arcane

### Color Palette

| Token | Hex | Purpose |
|-------|-----|---------|
| `bg.primary` | `#06070e` | Deep indigo-black (main background) |
| `bg.secondary` | `#0c0e1a` | Zaun dark (cards, panels) |
| `bg.tertiary` | `#151828` | Panel dark blue (inputs) |
| `bg.elevated` | `#1e2340` | Elevated surfaces |
| `accent.primary` | `#4fa8d4` | **Hextech blue** (main accent) |
| `accent.secondary` | `#d4a044` | Piltover gold (ratings, highlights) |
| `accent.zaun` | `#44d480` | Zaun toxic green (success, gaming) |
| `accent.shimmer` | `#c468e0` | Shimmer violet (exceptional quality) |
| `text.primary` | `#e4e8f4` | Cool white |
| `text.secondary` | `#8e94b0` | Muted lavender |
| `text.muted` | `#5a6080` | Dim steel |

### Star Parallax Particles

- **Stars**: Hextech blue, Piltover gold, Zaun green — 40 total (mobile-optimized)
- **Hex Motes**: Diamond-shaped energy fragments — 18 total, pulsing glow
- **Shooting Stars**: Blue-tinted gradient trails, ~6-10s interval
- **Touch Parallax**: PanResponder — drag to shift field, spring-back release

---

## Data Flow

### Auth Flow
```
App Launch → checkAuth() → getToken() from SecureStore
  ├─ Token exists → getProfile() → isAuthenticated = true → Discover
  └─ No token → isAuthenticated = false → Login screen
Login → POST /api/login → setToken() → navigate to Discover
```

### Chat Flow
```
Start Session → startSession(card) → create sessionId + first message
Send Message → add user msg → set isStreaming → streamMessage()
  ├─ SSE chunks → update streamingContent (live typing effect)
  └─ Done → add assistant msg → loadSuggestions()
Suggestion Tap → sendMessage(suggestion.prompt) → same flow
```

### Mock Layer
```
USE_MOCKS = true (constants/config.ts)
  └─ All lib/ functions check USE_MOCKS first
     ├─ true → return mock data with simulated delay
     └─ false → make real API call via apiFetch()
Toggle: set USE_MOCKS = false and update API_BASE
```

---

## API Endpoints (Backend)

### Existing (StratOS backend)
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/login` | Authenticate user |
| POST | `/api/register` | Create account |
| POST | `/api/agent/chat` | SSE streaming chat |
| GET | `/api/agent/history` | Conversation history |
| POST | `/api/agent/suggest` | Suggestion chips |

### New (needed for mobile)
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/characters` | List public cards |
| GET | `/api/characters/:id` | Single card |
| POST | `/api/characters` | Create card |
| GET | `/api/characters/trending` | By session count |
| GET | `/api/characters/search` | By query/genre |
| POST | `/api/characters/:id/rate` | Star rating |
| GET | `/api/scenarios` | List scenarios |
| GET | `/api/scenarios/:id` | Scenario + entities |
| POST | `/api/scenarios/:id/start` | Start gaming session |

---

## Component API Reference

### StarParallax
```tsx
<StarParallax>         // Full interactive (40 stars + 18 motes + touch parallax)
  {children}           // Content rendered on top
</StarParallax>

<StarParallaxBg />     // Lightweight ambient (20 stars + 6 motes, no interaction)
                       // Use as absolute-positioned background behind ScrollView
```

### CharacterCardComponent
```tsx
<CharacterCardComponent
  card={CharacterCard}           // Required: card data
  variant="grid" | "horizontal"  // Default: "grid" (2-column), "horizontal" (carousel)
/>
```

### MessageBubble
```tsx
<MessageBubble
  message={ChatMessage}    // Required: { role, content, ... }
  accentColor="#4fa8d4"    // Optional: override accent (genre-based)
/>
// Formats: *action text* → italic, paragraph breaks preserved
```

### SuggestionChips
```tsx
<SuggestionChips
  suggestions={Suggestion[]}     // { label, prompt } pairs
  onSelect={(prompt) => void}    // Sends expanded prompt, not label
  accentColor="#4fa8d4"
/>
```

### QualityScore
```tsx
<QualityScore
  card={CharacterCard}
  showElements={false}    // Show ✓/✗ per element
  size="small" | "large"  // Badge vs full display
/>
// Levels: Basic (0-2), Good (3-4), Great (5), Exceptional (6 + msg + scenario)
```

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

## Improvements vs Previous Build

### Code Quality
1. **Removed NativeWind/Tailwind** — unused, added 2 deps for no benefit. Pure StyleSheet is faster.
2. **Dropped react-native-gesture-handler dependency** for StarParallax — PanResponder works without extra native module.
3. **Removed expo-file-system** — TavernCard import uses fetch + FileReader (works everywhere).
4. **Consolidated shooting star state** — useRef instead of mutable useMemo (React warning).
5. **Simplified mock data** — shorter descriptions in mock.ts, same quality coverage.

### Performance
1. **StarParallax uses single SharedValue** — all 58+ elements animate from one `t` value via `useFrameCallback`.
2. **Hex motes use diamond-rotated View** instead of complex border-radius petal shapes — fewer reanimated style props.
3. **StarParallaxBg uses 26 elements** (20 stars + 6 motes) vs full 60 — lighter for scroll backgrounds.

### Architecture
1. **No `colors.accent.default`** — renamed to `colors.accent.primary` for clarity.
2. **Added `colors.accent.secondary`** (Piltover gold) for ratings/highlights.
3. **Arcane theme adds `zaun` and `shimmer`** accents for richer visual vocabulary.
4. **Star parallax particles match theme** — hex diamonds vs sakura petals.

---

## Glossary

| Term | Definition |
|------|-----------|
| **Arcane** | Theme inspired by the Arcane TV show — deep indigo-blacks, hextech blue accents, Piltover brass warmth, Zaun toxic green |
| **Character Card** | A structured description of an RP character with 6 quality elements, used as context for AI conversations |
| **Expo Router** | File-based routing for React Native — files in `app/` map directly to routes |
| **First Message** | The character's opening message; sets tone for every conversation; highest-impact quality field |
| **Gaming Mode** | Structured interactive fiction with numbered options, stat tracking, and a GM system prompt |
| **Hex Mote** | Diamond-shaped animated energy particle in the Arcane star parallax — replaces sakura petals |
| **Hextech** | Primary accent color (#4fa8d4, blue) — used for interactive elements, buttons, active states |
| **Mock Layer** | Togglable fake API layer (`USE_MOCKS=true`) for development without backend |
| **Piltover Gold** | Secondary accent (#d4a044) — used for ratings, highlights, fantasy genre |
| **Quality Score** | 0-6 rating based on filled character card elements; displayed as ★ badges |
| **Reanimated** | React Native animation library running on UI thread — drives StarParallax at 60fps |
| **RP (Roleplay)** | Free-form conversation where the AI plays a character defined by a character card |
| **SharedValue** | Reanimated primitive — value that lives on UI thread, drives animations without JS bridge |
| **SSE (Server-Sent Events)** | Streaming protocol for chat — server sends chunks, client displays as typing effect |
| **StarParallax** | Background particle system with stars, hex motes, shooting stars, and touch-responsive parallax |
| **Suggestion Chips** | Tappable action buttons after AI responses — label is short, sent prompt is immersive |
| **TavernCard V2** | Character card format embedded in PNG metadata (tEXt chunk) — compatible with SillyTavern |
| **Zaun Green** | Accent color (#44d480) — used for success states, horror genre, gaming stats |
| **Zustand** | Lightweight React state management — stores for auth, chat, and characters |
