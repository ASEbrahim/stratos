# StratOS Mobile — Functional Requirements Specification

## Document Info
- **Version**: 2.0
- **Date**: March 16, 2026
- **Last session**: CHROMA image gen, tiered memory, chat quality fixes
- **Platform**: React Native + Expo 54 + TypeScript
- **Backend**: Python HTTP server (port 8080), SQLite, Ollama LLM

---

## 1. Authentication

### 1.1 Login
- **Endpoint**: `POST /api/auth/login`
- User provides email + password
- Receives JWT token stored in device secure storage
- Auto-login on app restart if token valid

### 1.2 Registration
- **Endpoint**: `POST /api/auth/register`
- Name, email, password fields
- Auto-login after registration

### 1.3 Session Management
- Token included in all API requests via `Authorization: Bearer` header
- 401 response clears token and redirects to login
- Token persisted in Expo SecureStore (native) / localStorage (web)

---

## 2. Character Discovery

### 2.1 Browse Characters
- **Endpoint**: `GET /api/cards/browse`, `GET /api/cards/trending`, `GET /api/cards/search`
- Grid layout showing character cards (6 visible on screen)
- Genre filter chips (Anime, Fantasy, Sci-Fi, Romance, Horror, Modern, Historical)
- Sort by: Popular, Newest, Top Rated, Gaming
- Search by name/tags
- Random shuffle button

### 2.2 Character Card Display
- Name + genre tag on single row
- Avatar image or letter placeholder
- "Chat" button at card bottom
- Double-tap to bookmark (heart animation + haptic)
- Press animation (scale 0.98)

### 2.3 Character Detail
- Full card view with all 6 quality elements
- Opening line preview
- Star rating (interactive 1-5)
- Start Conversation / Continue Conversation buttons
- Generate Portrait button (links to image generation)
- Save to Library / Share buttons
- Similar characters section
- Report button

---

## 3. Chat (Roleplay & Gaming)

### 3.1 Core Chat
- **Endpoint**: `POST /api/rp/chat` (SSE streaming)
- Real-time token-by-token response display
- Message bubbles with rich formatting (bold, italic, quotes)
- Typing indicator during generation
- Auto-scroll on new messages and keyboard open
- Session persistence (auto-save on screen exit)
- Suggestion chips after each response
- "Seen ✓" indicator on user messages

### 3.2 Swipe (Regenerate)
- **Endpoint**: `POST /api/rp/regenerate`
- Only available on the last assistant message
- Creates alternative responses with same `swipe_group_id`
- Left/right arrows to navigate alternatives (← 1/3 →)
- Regenerate button disabled during streaming (race condition prevention)

### 3.3 Edit AI Messages
- **Endpoint**: `POST /api/rp/edit`
- Pencil icon on every assistant message
- Opens bottom sheet modal with editable textarea
- Optional reason dropdown: voice, length, accuracy, tone, agency, other
- Auto-categorizes edit if no reason selected
- Creates DPO training pair (original → edited)

### 3.4 Branching
- **Endpoint**: `POST /api/rp/branch`
- Edit own message → creates new branch from that point
- Branch selector dropdown at top of chat (visible when branches > 1)
- Branch-by-reference: parent messages not copied, reconstructed via chain
- Maximum depth: 15 levels

### 3.5 Director's Note
- **Endpoint**: `POST /api/rp/director-note` (storage), sent inline via `/api/rp/chat` `director_note` field
- Collapsible bar above chat input (DirectorNoteBar component)
- Guides AI's next response ("keep it short", "make her hesitant", "make her angry")
- Shows "Will apply to your next message" hint when note is active
- After use: note shown grayed out with "Reuse" button (not silently cleared)
- Single-use by default, reusable on tap
- Injected as system message: `DIRECTOR'S NOTE (this turn only): {note}`

### 3.6 Feedback
- **Endpoint**: `POST /api/rp/feedback`
- Thumbs up/down on every assistant message
- One feedback per message per user (idempotent)
- Visual confirmation (icon fills in)
- Feeds into nightly quality scoring pipeline

### 3.7 Gaming Mode
- **Endpoint**: `GET /api/scenarios`, `GET /api/scenarios/<id>`
- Scenario selection from discover tab
- Stat bars (HP, attributes)
- Numbered option buttons for choices
- Turn-based narrative with branching outcomes

---

## 4. Character Creation

### 4.1 Simple Mode
- Avatar + Name (side-by-side compact row)
- Genre selection
- Description, Personality, First Message, Scenario
- Content rating (SFW/NSFW)

### 4.2 Advanced Mode
- Organized into 3 sections: Identity, Behavior, Depth
- Identity: Description + Physical Description
- Behavior: Personality, Speech Pattern, First Message, Scenario
- Depth: Emotional Trigger, Defensive Mechanism, Vulnerability, Specific Detail
- Quality elements count (0-6)

### 4.3 TavernCard V2 Import
- Upload PNG with embedded character data
- Auto-populates all fields
- Switches to Advanced mode on import

### 4.4 Character Card API
- **Create**: `POST /api/cards`
- **Update**: `PUT /api/cards/<id>`
- **Delete**: `DELETE /api/cards/<id>`
- **Publish**: `POST /api/cards/<id>/publish`
- **Rate**: `POST /api/cards/<id>/rate`
- **Save**: `POST /api/cards/<id>/save`

---

## 5. Image Generation

### 5.1 Free-form Text-to-Image
- **Endpoint**: `POST /api/image/generate` (accepts `prompt`, `width`, `height`, `steps`, `seed`, `negative_prompt`)
- Prompt text input with style prefix (anime/realistic/illustration prepended automatically)
- Style selector: Anime / Realistic / Illustration
- Quality selector: Fast (8 steps ~30s) / Balanced (12 steps ~45s) / Quality (28 steps ~90s)
- Size selector: Portrait (768×1024) / Square (1024×1024) / Landscape (1024×768)
- Dynamic model badge shows selected quality + estimated time
- Seed control under Advanced toggle (random seed button)
- Loading state with estimated generation time

### 5.2 Character Portrait
- Both character mode and free-form use `POST /api/image/generate` (prompt built client-side)
- Style prefix + physical_description + character name assembled into prompt
- Quality modes apply to both character portraits and free-form
- "Set as Avatar" button: `PUT /api/cards/:id` with `{ avatar_image_path: imageId }`
- Only shows for characters with `card_id` param present

### 5.3 Image Gallery
- **Endpoint**: `GET /api/image/gallery`
- 3-column grid of recent generations (20 max)
- Tap to preview, long-press to delete
- Reloads every time gallery section is opened

### 5.4 Save & Delete
- **Save**: Share API on web, expo-media-library on native
- **Delete**: `DELETE /api/image/:id` with existence check
- **Endpoint**: `GET /api/image/:id` for serving images

### 5.5 Backend: CHROMA Model
- Single model: **CHROMA** (chroma-unlocked-v50.safetensors, 17GB fp16)
- Natively uncensored — handles SFW + NSFW without model switching
- ComfyUI workflow: UNETLoader + KSampler, cfg 4.0, beta scheduler, euler sampler
- Text encoders: CLIP-L + T5-XXL (Q8 GGUF)
- ~26s per image (cached), ~90s cold start
- Cannot coexist with Ollama — gpu_manager auto-swaps via API (no sudo)
- Old FLUX + Pony V7 models deleted

---

## 6. Tiered Memory System

### 6.1 Architecture
Three-tier persistent memory for cross-session RP chat context:
- **Tier 1 — Facts**: permanent key-value pairs (name, traits, items, injuries). Regex extraction on every message (instant). LLM extraction deferred until NUM_PARALLEL > 1.
- **Tier 2 — Recent Conversation**: sliding window from `rp_messages` table. Fills remaining token budget. No summarization.
- **Tier 3 — Arc Summaries**: relationship state checkpoints (not plot summaries). Triggered on scene transitions or every 25 messages. Deferred until NUM_PARALLEL > 1.

### 6.2 Context Injection
- `build_rp_context(session_id, db, character_card_id, token_budget=4000)` assembles all tiers
- Injected into system prompt before conversation history
- Returns debug dict: `{"tier1": N, "tier3": N, "tier2": N, "total": N}`
- Cross-session: facts from previous sessions with same character carry over

### 6.3 Database
- Table: `rp_session_context` (migration 028)
- Columns: `id, session_id, tier, category, key, value, turn_number, updated_at`
- Methods: `upsert_rp_context`, `get_rp_context`, `get_rp_context_for_character`

### 6.4 Auto-fill Formatting
- Cards without `speech_pattern` get OOC formatting hint at chat time
- New cards auto-fill `speech_pattern` via `fillCardDefaults()` in `lib/characters.ts`
- Includes: asterisk actions, quote dialogue, proportional length, anti-echo

---

## 7. Privacy & Consent

### 7.1 Training Data Opt-In
- One-time popup on first RP chat access
- Tracked via AsyncStorage (`training_opt_in_prompted`)
- Accept → anonymous data included in training pipeline
- Decline → data excluded from aggregation
- Changeable in Settings

### 7.2 Data Pipeline
- Nightly quality scorer (3 AM cron)
- Monthly SFT/DPO data export
- Only opted-in users included
- DPO pairs from edits and swipe selections

---

## 7. Themes & Customization

### 7.1 Themes
- 4 themes: Nebula (default), Sakura, Cosmos, Noir
- Theme dots on discover page (2 left of logo, 2 right)
- Active theme glows, inactive dimmed
- Persistent via AsyncStorage

### 7.2 Typography
- Quicksand: headings, logos, buttons
- Poppins: body text, labels, captions
- Loaded via expo-font in root layout

### 7.3 Visual Effects
- Star parallax background with twinkling stars, floating orbs, shooting stars
- Denser star cluster around StratOS logo
- All animations via Reanimated native driver

---

## 8. Offline Support

- Chat sessions persisted in AsyncStorage (max 50)
- Saved characters stored locally
- Graceful degradation when backend unavailable
- Mock mode (`USE_MOCKS = true`) for development

---

## 9. Non-Functional Requirements

### 9.1 Performance
- 60fps animations (Reanimated native driver)
- Virtualized lists (FlatList) for chat and character grids
- Font loading screen prevents layout flash

### 9.2 Accessibility
- All interactive elements have accessibilityLabel and accessibilityRole
- Content descriptions on images
- Haptic feedback on user actions

### 9.3 Security
- JWT tokens in secure storage
- No hardcoded API keys
- NSFW content filter toggle
- Input validation on all forms
