# StratOS Changes — March 4, 2026

## Files Modified (11 files)

### 1. Card Opacity + Effects Sliders in Theme Editor
**Files:** `theme-editor.js`, `styles.css`

- Added new **Effects** group to theme editor with 5 range sliders:
  - **Card Opacity** (0.1–1.0) — controls glass panel background transparency
  - **Panel Opacity** (0.1–1.0) — controls sidebar transparency
  - **Border Radius** (0–24px) — adjusts corner rounding globally
  - **Glow Intensity** (0–1.0) — controls hover glow strength
  - **Glass Blur** (0–32px) — controls backdrop-filter blur amount
- All sliders save per-theme and auto-apply on load
- Added range slider CSS with accent-colored thumb, hover scale effect
- Updated `.glass-panel` to use CSS variables for blur, radius, and glow
- `clearAllOverrides()` now also clears effects-derived variables

### 2. Account Email Pre-populated on Registration
**File:** `server.py`

- `/api/status` endpoint now looks up the user's email from the `users` table via `profiles.user_id` join for DB-auth users (who have no YAML file)
- This means when you create an account via email registration, your email automatically appears in Settings → Profile → Account Email

### 3. Context Box Moved to Sources Tab + Hover Tooltips
**File:** `index.html`

- **Moved** the Context & Interests textarea, Generate, Suggest, Wizard, and Save buttons from the Profile tab (`data-stab="profile"`) to the **Sources** tab (`data-stab="sources"`)
- Profile tab now only contains role and location (identity)
- Sources tab now starts with Context & Interests (what to track), followed by categories and feeds
- Added `strat-tip` hover tooltips to:
  - "How fresh?" section header
  - All existing profile/context/generate/suggest/save elements
- Added **visible tool badges** to the Agent panel header showing available capabilities (Search, Watchlist, Categories) with hover explanations

### 4. Movers Button Stops Spinning
**File:** `styles.css`

- Added missing `.animate-spin` CSS class that was referenced throughout the codebase but never defined
- Also added `.animate-pulse` keyframe animation
- The `addTopMoversTickers()` function in settings.js already had correct `classList.remove('animate-spin')` in its `finally` block — the issue was the missing CSS class definition

### 5. Per-Card Refresh Button on Market Ticker Charts
**File:** `markets-panel.js`

- Added a refresh ↻ button to each chart card header (next to the close × button)
- New `_mpRefreshSingle(id)` function fetches fresh data for just that ticker via `/api/market-tick`
- Visual feedback: refresh icon spins during fetch, stops when done
- Shows toast on success/failure
- Updates `marketData` in memory and re-renders the specific chart

### 6. Progress Bar Timing Fix
**File:** `app.js`

- Fixed polling-based progress bar missing the `news_done` stage
- Previously: poll only recognized `starting`, `market`, `news` as fetch phases — missed `news_done`
- The bar would jump from 75% to 0% briefly before scoring started
- Now correctly shows 100% for `news_done` before transitioning to score phase

### 7. Agent Pre-initialization on Launch
**File:** `app.js`

- Added `_warmupAgent()` function called 5 seconds after `init()` completes (low priority, after everything else loads)
- Checks `/api/agent-status` for model availability
- Sends a minimal warmup request to Ollama (`num_predict: 1`) to force the model into GPU/memory
- Completely silent — no UI indication, catches all errors
- First real chat message should now respond significantly faster

### 8. Username Login Support
**File:** `server.py`

- When `/api/auth/login` receives an identifier without `@`, it now looks up the user's email from `users.display_name` (case-insensitive)
- If found, transparently swaps the username for the email before passing to the auth handler
- Falls through gracefully if not found — shows normal "user not found" error

### 9. "Don't Show Again" for Tour Popup
**File:** `auth.js`

- Fixed `_clearProfileLocalStorage()` which was wiping `stratos_tour_never` on every login/profile switch
- The key is now preserved alongside auth tokens, device ID, and theme
- "Don't show again" now persists across sessions and profile switches as intended

### 10. Wizard Stylization Overhaul
**File:** `wizard.js`

Inspired by the portfolio reference HTML, applied polish across the wizard:

- **Backdrop:** Added `backdrop-filter: blur(8px)` for glass effect, smoother transition
- **Modal:** Increased border-radius to 24px, added noise texture overlay (`::after` pseudo-element), accent-tinted border glow, deeper shadow with accent color bleed
- **Header:** Increased padding, bolder brand text (weight 800), backdrop-filter, better badge typography
- **Cards (`.gcard`):** Added radial gradient glow on hover (`::after`), increased lift to -4px, improved border-radius to 16px, smoother cubic-bezier transitions
- **Build Button:** Stronger glow shadow, wider shimmer highlight, accent color box-shadow on hover
- **Rail Sections:** Better border-radius (16px), subtle accent glow when expanded
- **Detail Accordion:** Improved padding, accent glow when open, deeper shadow
- **Quick Setup:** Stronger hover glow with accent light bleed
- **Loading:** Better typography (800 weight), bar inner glow, spinner glow filter
- **Close Button:** Added scale-up on hover (1.05×)
- **AI Suggestions:** Added box-shadow, better padding/radius, bolder header
- **Scrollbars:** Added custom scrollbar colors in rail and main panels
