# StratOS Mobile

A standalone character roleplay and interactive fiction app built with React Native + Expo. Part of the StratOS intelligence platform ecosystem — shares the same backend, database, training pipeline, and AI models.

## Features

### Character Roleplay
- **SSE streaming chat** with real-time token-by-token AI responses
- **10 built-in characters** across 7 genres (Fantasy, Sci-Fi, Romance, Horror, Modern, Anime, Historical)
- **6-element quality system**: physical description, speech pattern, emotional trigger, defensive mechanism, vulnerability, specific detail
- **Rich prose formatting**: bold, italic, quotes, action text
- **Director's Note**: steer the AI's next response with invisible instructions

### Interaction Layer
- **Edit AI messages** with optional reason tags (voice, length, accuracy, tone, agency) — creates DPO training pairs
- **Swipe regeneration** on last message — browse alternative responses with ← 1/3 → navigation
- **Conversation branching** — edit earlier messages to create alternate timelines
- **Thumbs up/down feedback** on every AI response
- **Branch selector** for navigating conversation trees

### Character Community
- **Create characters** in Simple or Advanced mode
- **TavernCard V2 import** from PNG files
- **Browse & discover** published characters with genre filters and search
- **Rate characters** with 5-star ratings
- **Save to library** for offline access

### Image Generation
- **Text-to-image** with FLUX.1-schnell (SFW) and Pony V7 (NSFW)
- **Character portraits** auto-generated from card descriptions
- Style selector: Anime / Realistic / Illustration
- Size selector: Portrait / Square / Landscape

### Gaming Mode
- **Interactive scenarios** with stat bars and branching choices
- 3 built-in scenarios: fantasy siege, sci-fi mystery, horror mansion
- Numbered option buttons with animated entry

### Design
- **4 atmospheric themes**: Nebula (default), Sakura, Cosmos, Noir
- **Custom typography**: Quicksand (headings) + Poppins (body)
- **Star parallax** background with twinkling stars, floating orbs, shooting stars
- **Haptic feedback** on all interactions
- **Offline-first** with AsyncStorage persistence

## Tech Stack

- **Frontend**: React Native, Expo 54, TypeScript, Expo Router
- **State**: Zustand stores
- **Animation**: React Native Reanimated
- **Fonts**: @expo-google-fonts/quicksand, @expo-google-fonts/poppins
- **Backend**: Python HTTP server, SQLite, Ollama LLM inference
- **Image Gen**: ComfyUI (FLUX + Pony V7)
- **Training**: DoRA fine-tuning, DPO from user edits/swipes

## Getting Started

### Prerequisites
- Node.js 18+
- Expo CLI (`npm install -g expo-cli`)
- Backend running at `http://192.168.0.148:8080` (or set in `constants/config.ts`)

### Install & Run
```bash
cd mobile/stratos-app
npm install
npx expo start --web --port 8081
```

### Development Mode
The app ships with `USE_MOCKS = true` in `constants/config.ts`. All features work with mock data — no backend required. Set to `false` to connect to the real backend.

### Backend Setup
```bash
cd backend
python3 main.py --serve --background  # Starts on port 8080
```

Requires Ollama running with:
- `stratos-rp-q8` — RP chat model
- `qwen3.5:9b` — gaming/analysis model

## Project Structure

```
mobile/stratos-app/
├── app/                    # Expo Router screens
│   ├── (auth)/            # Login/register
│   ├── (tabs)/            # Main tabs: Discover, Chats, Library, Create, Profile
│   ├── chat/[id].tsx      # Chat screen with RP expansion
│   ├── character/[id].tsx # Character detail
│   ├── gaming/            # Gaming scenarios
│   └── imagegen/          # Image generation
├── components/
│   ├── cards/             # CharacterCard, CharacterDetail, QualityScore
│   ├── chat/              # MessageBubble, ChatInput, FeedbackButtons,
│   │                        SwipeIndicator, DirectorNoteBar, EditSheet,
│   │                        BranchSelector, TrainingOptIn, SuggestionChips
│   ├── creator/           # CardEditor, AvatarPicker, GuidedFields
│   ├── gaming/            # ScenarioCard, StatBar, OptionButtons
│   └── shared/            # Header, LoadingScreen, StarParallax, ErrorBoundary
├── constants/
│   ├── config.ts          # API_BASE, USE_MOCKS, feature flags
│   ├── fonts.ts           # Font family constants
│   ├── genres.ts          # Genre definitions and colors
│   ├── theme.ts           # Spacing, typography, border radius
│   └── themes.ts          # 4 theme color definitions
├── lib/
│   ├── api.ts             # HTTP client with auth token handling
│   ├── auth.ts            # Login/register/profile
│   ├── characters.ts      # Character CRUD (wired to /api/cards/*)
│   ├── chat.ts            # SSE streaming chat + suggestions
│   ├── gaming.ts          # Scenario fetching
│   ├── mappers.ts         # Backend→mobile field normalization
│   ├── mock.ts            # Mock data (10 characters, 3 scenarios)
│   ├── rp.ts              # RP expansion: edit, swipe, branch, feedback,
│   │                        director's note, image gen, card actions
│   ├── storage.ts         # AsyncStorage persistence
│   └── types.ts           # TypeScript interfaces
└── stores/
    ├── authStore.ts       # Auth state (Zustand)
    ├── characterStore.ts  # Character state
    ├── chatStore.ts       # Chat state
    └── themeStore.ts      # Theme state
```

## Backend API Endpoints

### Auth
- `POST /api/auth/login` — Login with email/password
- `POST /api/auth/register` — Register new user
- `GET /api/auth/me` — Get current user profile

### Character Cards
- `GET /api/cards/browse` — Browse published cards
- `GET /api/cards/trending` — Top cards
- `GET /api/cards/search?q=` — Search cards
- `GET /api/cards/<id>` — Get single card
- `POST /api/cards` — Create card
- `PUT /api/cards/<id>` — Update card
- `POST /api/cards/<id>/rate` — Rate (1-5)
- `POST /api/cards/<id>/publish` — Publish
- `POST /api/cards/<id>/save` — Save to library

### RP Chat
- `POST /api/rp/chat` — SSE streaming chat
- `POST /api/rp/regenerate` — Swipe (last message only)
- `POST /api/rp/swipe` — Select swipe alternative
- `POST /api/rp/edit` — Edit AI message
- `POST /api/rp/branch` — Create conversation branch
- `POST /api/rp/director-note` — Set steering note
- `POST /api/rp/feedback` — Thumbs up/down
- `GET /api/rp/history/<sid>` — Full conversation
- `GET /api/rp/branches/<sid>` — List branches

### Image Generation
- `POST /api/image/generate` — Free-form text-to-image
- `POST /api/image/character-portrait` — From card fields
- `GET /api/image/<id>` — Serve generated image
- `GET /api/image/gallery` — User's generation history

### Gaming
- `GET /api/scenarios` — List scenarios
- `GET /api/scenarios/<id>` — Get scenario

## Data Pipeline

The app feeds a continuous improvement loop:

1. **User edits** AI messages → stored as DPO training pairs (original vs edited)
2. **Swipe selections** → preference pairs (chosen vs rejected)
3. **Thumbs up/down** → quality signals
4. **Nightly scorer** processes conversations → quality scores per session
5. **Monthly aggregation** exports SFT + DPO JSONL for fine-tuning
6. **A/B testing** compares model versions via deterministic session-based split

Only data from users who opted in is included in training.

## License

Private — StratOS Platform
