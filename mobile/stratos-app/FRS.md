# StratOS Mobile — Functional Requirements Specification

## Document Info
- **Version**: 1.0
- **Date**: March 15, 2026
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
- **Endpoint**: `POST /api/rp/director-note`
- Collapsible bar above chat input
- Guides AI's next response ("keep it short", "make her hesitant")
- After use: note shown grayed out with "Reuse" button (not silently cleared)
- Single-use by default, reusable on tap

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
- **Endpoint**: `POST /api/image/generate`
- Prompt text input
- Style selector: Anime / Realistic / Illustration
- Size selector: Portrait (768×1024) / Square (1024×1024) / Landscape (1024×768)
- Loading state with estimated time (~5-30 seconds)
- Generated image display

### 5.2 Character Portrait
- **Endpoint**: `POST /api/image/character-portrait`
- Auto-constructs prompt from character card fields
- One-tap from character detail page
- Model routing: SFW → FLUX, NSFW → Pony V7

### 5.3 Backend Requirements
- ComfyUI in API mode on port 8188
- FLUX.1-schnell Q8 GGUF (~12GB)
- Pony Diffusion V7 for NSFW (~6.5GB)
- Cannot coexist with Ollama in 24GB VRAM

---

## 6. Privacy & Consent

### 6.1 Training Data Opt-In
- One-time popup on first RP chat access
- Tracked via AsyncStorage (`training_opt_in_prompted`)
- Accept → anonymous data included in training pipeline
- Decline → data excluded from aggregation
- Changeable in Settings

### 6.2 Data Pipeline
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
