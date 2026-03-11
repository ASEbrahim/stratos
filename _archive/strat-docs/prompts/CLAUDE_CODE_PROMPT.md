# StratOS UI/UX Feature + Fix Implementation

You are implementing 10 UI/UX changes to the StratOS codebase. Read this entire document before writing any code.

## Pre-Flight: Audit Coordination

An independent security audit landed 5 commits (181 insertions, 96 deletions, 17 files). These commits MUST be in your working tree before you start:

```
06d7ad9 — Phase 1: XSS escaping, score clamping, placeholder guards
ba5eeef — Phase 2: Error masking, security headers, saveSerperKey fix
61a7d6b — Phase 2 continued
b84e981 — Phase 3: Auth rate limits, wizard rate limits, rate limiting moved before route dispatch
46a18ae — Phase 4: SSE profile scoping, score injection prevention, profile-scoped ScoringMemory
d08652a — Phase 5: CORS allowlist, SRI integrity hashes on CDN scripts
```

**Critical audit touchpoints that overlap with your work:**
- `server.py`: Phase 3 moved rate limiting BEFORE route dispatch. Phase 2 masked error strings. Your username-lookup (Fix 1) and email-populate (Fix 8) changes must merge cleanly with these.
- `auth.js`: Phase 1 added XSS escaping. Your tour fix (Fix 2) is in `_clearProfileLocalStorage()` which is separate — but verify no conflicts.
- `wizard.js`: Phase 3 added wizard rate limits. Your CSS-only changes (Style 10) modify the `WIZ_CSS` string, not the rate limit logic — but verify.
- `index.html`: Phase 5 added SRI hashes to CDN `<script>` tags. Your DOM restructuring (Fix 9) must not remove or alter those.
- `settings.js`: Phase 1 added XSS escaping in `showToast()` and preset rendering. You don't touch settings.js, but verify after the context-box move that settings.js still finds its elements.

### Setup
```bash
git stash  # if needed
git pull
git log --oneline -10  # verify audit commits are present
git checkout -b feature/ui-fixes-and-polish
```

---

## Changes (implement and commit each independently)

---

### Commit 1: `fix: resolve display_name to email for username login`
**File:** `server.py`

**Root cause:** `/api/auth/login` receives `{ email: identifier, password }` from the frontend. When the user types their display_name (username) instead of email, `handle_auth_routes()` queries `users.email` and finds nothing.

**What to do:** In `do_POST()`, the block that handles `/api/auth/` routes (currently around line 860-868 — may have shifted from audit). After `data` is parsed from POST body (line ~864) but BEFORE `handle_auth_routes()` is called (line ~867), add:

```python
# Username → email resolution for login
if self.path == "/api/auth/login" and data.get("email") and "@" not in data["email"]:
    try:
        cursor = strat.db.conn.cursor()
        cursor.execute(
            "SELECT email FROM users WHERE display_name = ? COLLATE NOCASE LIMIT 1",
            (data["email"],)
        )
        row = cursor.fetchone()
        if row:
            data["email"] = row[0]
    except Exception:
        pass  # Fall through — auth will show "user not found"
```

**DB schema** (`migrations.py:199`): `users` table has `email TEXT UNIQUE NOT NULL`, `display_name TEXT NOT NULL`.

**Edge cases:**
- If the audit moved rate limiting before auth dispatch: the username lookup still needs to be between `data = json.loads(...)` and `handle_auth_routes(...)`, regardless of where rate limiting sits.
- The `COLLATE NOCASE` handles users typing "John" vs "john".
- If no `@` is found but it's not a valid username either, `handle_auth_routes` handles the error naturally.

---

### Commit 2: `fix: preserve tour dismissal across login/profile switch`
**File:** `auth.js`

**Root cause:** `_clearProfileLocalStorage()` (line 1029) iterates all localStorage keys and removes everything except auth tokens, device ID, and theme. This nukes `stratos_tour_never` on every login.

**What to do:** Line 1037 currently reads:
```javascript
if (k === AUTH_TOKEN_KEY || k === AUTH_PROFILE_KEY || k === DEVICE_ID_KEY || k === AUTH_THEME_KEY) continue;
```
Change to:
```javascript
if (k === AUTH_TOKEN_KEY || k === AUTH_PROFILE_KEY || k === DEVICE_ID_KEY || k === AUTH_THEME_KEY || k === 'stratos_tour_never') continue;
```
Also update the comment on line 1036 from "intentionally NOT preserved" to "preserved — user's tour preference should persist across sessions".

---

### Commit 3: `fix: add missing .animate-spin CSS class`
**File:** `styles.css`

**Root cause:** The codebase uses `.animate-spin` in 20+ places across `agent.js`, `app.js`, `feed.js`, `settings.js`, `index.html`, and `markets-panel.js`. The `@keyframes spin` exists (around line 488) but no class applies it. This is why the Movers button, agent spinner, and all refresh icons never actually spin.

**What to do:** After the existing `@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }` block, add:
```css
.animate-spin {
    animation: spin 1s linear infinite;
}
.animate-pulse {
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
```

**Verification:** `grep -rn 'animate-spin' --include='*.js' --include='*.html' --include='*.css'` — confirm no duplicate definition exists.

---

### Commit 4: `fix: progress bar polling missing news_done stage`
**File:** `app.js`

**Root cause:** The SSE handler `_handleSSEScan()` (line ~1187) correctly lists `['starting', 'market', 'news', 'news_done']` as fetch phases. But the polling fallback `checkStatus()` (line ~1512) has `['starting', 'market', 'news']` — missing `'news_done'`. When the poll catches `news_done`, `pollPct` stays 0 and the bar drops from 75% to 0% briefly.

**What to do:** At line ~1512, change:
```javascript
var pollIsFetch = ['starting', 'market', 'news'].includes(st);
```
to:
```javascript
var pollIsFetch = ['starting', 'market', 'news', 'news_done'].includes(st);
```

And inside the `if (pollIsFetch)` block (after `else if (st === 'news') pollPct = 75;`), add:
```javascript
else if (st === 'news_done') pollPct = 100;
```

Also add `'news_done'` to the poll title section (~line 1536):
```javascript
else if (st === 'news_done') pollTitle = 'Articles fetched!';
```

---

### Commit 5: `feat: theme editor effects sliders`
**Files:** `theme-editor.js`, `styles.css`

**Context:** `theme-editor.js` has `EDITOR_GROUPS` array (line 9) with 4 groups (Backgrounds, Accent, Text, Chart & Borders). Each group has `vars` array with `{ key, label, type: 'hex' }` entries. The `buildPanel()` function (line ~220) creates color picker inputs. `computeDerived()` (line 56) auto-computes dependent CSS variables. `clearAllOverrides()` (line 130) removes custom vars.

**Add to `EDITOR_GROUPS`** (after the Chart & Borders group, before the closing `];`):
```javascript
{
    label: 'Effects',
    icon: 'sliders',
    vars: [
        { key: '--te-card-opacity',   label: 'Card Opacity',   type: 'range', min: 0.1, max: 1, step: 0.05, default: 0.82 },
        { key: '--te-panel-opacity',  label: 'Panel Opacity',  type: 'range', min: 0.1, max: 1, step: 0.05, default: 0.96 },
        { key: '--te-border-radius',  label: 'Border Radius',  type: 'range', min: 0, max: 24, step: 1, default: 12, unit: 'px' },
        { key: '--te-glow-intensity', label: 'Glow Intensity', type: 'range', min: 0, max: 1, step: 0.05, default: 0.15 },
        { key: '--te-blur',           label: 'Glass Blur',     type: 'range', min: 0, max: 32, step: 1, default: 12, unit: 'px' },
    ]
},
```

**Update `buildPanel()`:** Inside the `group.vars.forEach(v => { ... })` loop, check `v.type`. If `'hex'`, render the existing color picker. If `'range'`, render:
```html
<div class="te-range-wrap">
    <label class="te-color-label">{v.label}</label>
    <div style="display:flex;align-items:center;gap:8px;">
        <input type="range" class="te-range-slider" data-var="{v.key}" min="{v.min}" max="{v.max}" step="{v.step}" value="{v.default}" />
        <span class="te-range-val">{v.default}{v.unit||''}</span>
    </div>
</div>
```
Wire up `input` event on range sliders to call `applyAndSave(varName, value)` and update the span.

**Update `computeDerived()`:** Add after the existing derived computations:
```javascript
// Effects sliders → CSS variables
if (overrides['--te-card-opacity'] !== undefined) {
    const op = parseFloat(overrides['--te-card-opacity']);
    if (overrides['--bg-panel-solid']) {
        const { r, g, b } = hexToRgb(overrides['--bg-panel-solid']);
        derived['--bg-panel'] = `rgba(${r},${g},${b},${op})`;
    }
    derived['--card-opacity'] = String(op);
}
if (overrides['--te-panel-opacity'] !== undefined) {
    const op = parseFloat(overrides['--te-panel-opacity']);
    if (overrides['--sidebar-bg']) {
        const { r, g, b } = hexToRgb(overrides['--sidebar-bg']);
        derived['--sidebar-bg'] = `rgba(${r},${g},${b},${op})`;
    }
}
if (overrides['--te-border-radius'] !== undefined) {
    const px = overrides['--te-border-radius'] + 'px';
    derived['--radius-lg'] = px;
    derived['--radius-xl'] = (parseFloat(overrides['--te-border-radius']) + 4) + 'px';
}
if (overrides['--te-glow-intensity'] !== undefined) {
    derived['--glow-intensity'] = String(overrides['--te-glow-intensity']);
}
if (overrides['--te-blur'] !== undefined) {
    derived['--glass-blur'] = overrides['--te-blur'] + 'px';
}
```

**Update `clearAllOverrides()`:** Add these to the derived cleanup array:
```javascript
'--card-opacity', '--radius-lg', '--radius-xl', '--glow-intensity', '--glass-blur'
```

**Update `syncPickersToCurrentTheme()`:** After syncing color pickers, also sync range sliders from loaded overrides.

**In `styles.css`:** Add range slider styling:
```css
.te-range-wrap { padding: 4px 0; }
.te-range-slider {
    -webkit-appearance: none; appearance: none;
    width: 100%; height: 4px; border-radius: 2px;
    background: rgba(255,255,255,0.1); outline: none;
    cursor: pointer;
}
.te-range-slider::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--accent, #10b981); border: 2px solid rgba(255,255,255,0.2);
    cursor: pointer; transition: transform 0.15s;
}
.te-range-slider::-webkit-slider-thumb:hover { transform: scale(1.2); }
.te-range-val {
    font-size: 10px; color: var(--text-muted, #64748b);
    min-width: 32px; text-align: right; font-variant-numeric: tabular-nums;
}
```

Update `.glass-panel` to use the new variables:
```css
backdrop-filter: blur(var(--glass-blur, 10px));
-webkit-backdrop-filter: blur(var(--glass-blur, 10px));
border-radius: var(--radius-lg, 12px);
```

---

### Commit 6: `feat: per-card refresh button on market charts`
**File:** `markets-panel.js`

**Context:** `mpAddChart()` builds chart card HTML via string concatenation. Each card header (`row1`) has timeframe buttons and a close (×) button.

**What to do:**
1. In the `row1` string, before the close button, add a refresh button:
```html
<button onclick="_mpRefreshSingle('{id}')" class="mp-refresh-btn" title="Refresh this ticker" style="color:var(--text-muted);cursor:pointer;background:none;border:none;padding:2px;">
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>
        <path d="M21 3v5h-5"/>
        <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>
        <path d="M3 21v-5h5"/>
    </svg>
</button>
```

2. Add new function:
```javascript
function _mpRefreshSingle(id) {
    var entry = _mpCharts.find(function(c) { return c.id === id; });
    if (!entry) return;
    var btn = document.querySelector('#mp-card-' + id + ' .mp-refresh-btn svg');
    if (btn) btn.classList.add('animate-spin');

    var sym = entry.symbol || id;
    var interval = entry.interval || '1d';
    fetch('/api/market-tick?symbol=' + encodeURIComponent(sym) + '&interval=' + encodeURIComponent(interval))
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data && data.data) {
                marketData[sym] = data.data;
                _mpUpdateChart(entry);
                if (typeof showToast === 'function') showToast(sym + ' refreshed', 'success');
            }
        })
        .catch(function(err) {
            console.warn('[MP] Refresh failed for', sym, err);
            if (typeof showToast === 'function') showToast('Failed to refresh ' + sym, 'error');
        })
        .finally(function() {
            if (btn) btn.classList.remove('animate-spin');
        });
}
```
Make sure `_mpRefreshSingle` is accessible globally (attach to `window` or declare at module scope).

---

### Commit 7: `feat: agent model warmup on page load`
**Files:** `app.js`, `server.py`

**Problem:** First agent chat message takes 10-30s because Ollama needs to load the model into VRAM. Pre-loading it silently on page load eliminates this delay.

**IMPORTANT — CORS consideration:** The frontend can't call `http://localhost:11434` (Ollama) directly due to cross-origin restrictions. Route through the backend instead.

**In `server.py`**, add a new endpoint in `do_POST()` (after the agent-status handling or in an appropriate location):
```python
if self.path == "/api/agent-warmup":
    try:
        scoring_cfg = strat.config.get("scoring", {})
        ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
        model = scoring_cfg.get("inference_model", "qwen3:30b-a3b")
        import requests as req
        req.post(f"{ollama_host}/api/generate",
                 json={"model": model, "prompt": "hi", "stream": False,
                        "options": {"num_predict": 1}},
                 timeout=30)
    except Exception:
        pass
    _send_json(self, {"ok": True})
    return
```

Note: `requests` is already imported at the top of server.py. Move the import to use whatever alias is already used (check if it's `import requests` or `import requests as req` etc). Also make sure this endpoint goes through rate limiting just like other POST endpoints.

**In `app.js`**, at the end of `init()` (line ~470, after the `maybeStartTour` setTimeout), add:
```javascript
// Deferred agent warmup — pre-load model into VRAM
setTimeout(function() {
    fetch('/api/agent-warmup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
        .then(function() { console.log('[Agent] Model pre-loaded'); })
        .catch(function() { /* silent — warmup is best-effort */ });
}, 5000);
```

**Why 5 seconds:** By then the page, charts, and initial data load are complete. The warmup is low priority.

---

### Commit 8: `feat: auto-populate account email for DB-auth users`
**File:** `server.py`

**Root cause:** The `/api/status` endpoint (line ~288) reads email from YAML profile files (line ~311). DB-auth users (registered via email) have no YAML file, so their email never populates in Settings → Profile → Account Email.

**What to do:** After the existing DB fallback block for avatars (around line ~317-323, where it reads `ui_state` for DB-auth avatar fallback), add:
```python
# DB fallback for email (DB-auth users have no YAML)
if "email" not in status:
    try:
        cursor = strat.db.conn.cursor()
        cursor.execute("""
            SELECT u.email FROM users u
            JOIN profiles p ON p.user_id = u.id
            WHERE p.id = ? LIMIT 1
        """, (_status_pid,))
        email_row = cursor.fetchone()
        if email_row and email_row[0]:
            status["email"] = email_row[0]
    except Exception:
        pass
```

Place this BEFORE `self.wfile.write(json.dumps(status).encode())` (line ~324).

**Verify:** The frontend reads `status.email` in `app.js` (search for `emailEl` references) and auto-populates the account email input.

---

### Commit 9: `refactor: move context box to Sources tab, add tooltips`
**File:** `index.html`

**Context:** Settings uses `data-stab` attributes for tab switching. `switchSettingsTab()` in `settings.js` shows/hides panels by matching `data-stab` to the active tab. Elements exist in the DOM always — they're just hidden via CSS when their tab isn't selected.

**What to do:**

**Split the "Quick Profile" panel** (around line ~875, `data-stab="profile"`):

**Panel 1 — stays in Profile tab** (`data-stab="profile"`):
```html
<div class="glass-panel rounded-xl p-6" data-stab="profile">
    <h3 class="text-sm font-bold text-slate-200 mb-1 flex items-center gap-2">
        <i data-lucide="user" class="w-4 h-4" style="color:var(--accent)"></i> Who are you?
        <span class="strat-tip" data-tip="Your role and location shape how the AI scores and prioritizes news for you."><i data-lucide="info" class="w-3 h-3" style="color:var(--text-muted);"></i></span>
    </h3>
    <p class="text-xs text-slate-500 mb-4">Tell StratOS who you are so it can personalize your feed.</p>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
            <label class="text-xs text-slate-400 block mb-1.5 font-medium strat-tip" data-tip="Your profession or role — shapes which topics the AI considers high-priority for you.">Your role</label>
            <input type="text" id="simple-role" data-tour="role-field" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-200 focus:border-emerald-500 focus:outline-none" placeholder="e.g., Computer Engineering Student">
        </div>
        <div>
            <label class="text-xs text-slate-400 block mb-1.5 font-medium strat-tip" data-tip="Your location — enables region-specific news sources and local market tracking.">Location</label>
            <input type="text" id="simple-location" data-tour="location-field" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-200 focus:border-emerald-500 focus:outline-none" placeholder="e.g., Kuwait">
        </div>
    </div>
</div>
```

**Panel 2 — new, in Sources tab** (`data-stab="sources"`) — place this BEFORE the `<div id="simple-categories-container">`:
Move the following elements here: context textarea (`#simple-context`), profile questions toggle, Generate/Wizard/Suggest/Save/Undo/Redo buttons, generate status span. New title: "Context & Interests" with `<i data-lucide="brain">` icon.

Add `strat-tip` tooltips to:
- Context section header: "Describe what you care about — the AI uses this to score relevance, generate categories, and write briefings."
- Generate button: `data-tip="Uses AI to generate tracking categories, keywords, and tickers based on your role and context."`
- Suggest button: `data-tip="AI writes a context paragraph from your role — you can edit it after."`
- Save button: `data-tip="Saves your context without regenerating categories."`
- Wizard button: `data-tip="Step-by-step guided setup wizard."`

**Add "How fresh?" tooltip:** Find the "How fresh?" `<h3>` and add:
```html
<span class="strat-tip" data-tip="Controls the maximum age of articles fetched during scans."><i data-lucide="info" class="w-3 h-3" style="color:var(--text-muted);"></i></span>
```

**Add agent tool badges:** In the agent panel header (search for `"Search the web, manage your feed, analyze signals"`), after that `<p>` tag, add:
```html
<div class="flex items-center gap-1 mt-1">
    <span class="text-[8px] font-mono px-1.5 py-0.5 rounded-full strat-tip" style="background:rgba(99,102,241,0.1); color:#818cf8; border:1px solid rgba(99,102,241,0.2);" data-tip="Agent can search the web for real-time info.">🔍 Search</span>
    <span class="text-[8px] font-mono px-1.5 py-0.5 rounded-full strat-tip" style="background:rgba(16,185,129,0.1); color:#34d399; border:1px solid rgba(16,185,129,0.2);" data-tip="Agent can add/remove market tickers.">📊 Watchlist</span>
    <span class="text-[8px] font-mono px-1.5 py-0.5 rounded-full strat-tip" style="background:rgba(251,191,36,0.1); color:#fbbf24; border:1px solid rgba(251,191,36,0.2);" data-tip="Agent can manage tracking categories.">🏷️ Categories</span>
</div>
```

**CRITICAL:** The `.strat-tip` CSS already exists in `styles.css` (around line 669). Do NOT recreate it.

**CRITICAL:** SRI hashes on `<script>` tags from the Phase 5 audit — do NOT touch those. Only modify the settings panel `<div>` structure.

---

### Commit 10: `style: wizard glassmorphism and visual polish`
**File:** `wizard.js` — specifically the `WIZ_CSS` template string (starts around line 668)

This is CSS-only — no logic changes. Apply these refinements to the wizard's embedded stylesheet:

**`.wiz-bk` (backdrop):**
- Add `backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);`
- Change background opacity from .55 → .6
- Smoother transition: `cubic-bezier(.4,0,.2,1)`

**`.wiz-modal`:**
- Border-radius: 20px → 24px
- Add noise texture via `::after` pseudo-element: SVG feTurbulence at opacity 0.03
- Accent-tinted border: `color-mix(in srgb, var(--accent) 15%, var(--brd))`
- Deeper shadow with accent bleed: `box-shadow:0 32px 100px rgba(0,0,0,.5),0 0 0 1px rgba(255,255,255,.04) inset,0 0 80px -20px color-mix(in srgb, var(--accent) 8%, transparent)`
- Smoother spring: `cubic-bezier(.16,1.11,.36,1.02)`
- Set `.wiz-modal > * { position:relative;z-index:1; }` so noise doesn't cover content

**`.wiz-hdr`:**
- Font weight 800, font-size 18px for `.wiz-brand`
- Add `backdrop-filter:blur(12px)`
- Padding: 14px 24px → 16px 28px

**`.gcard` (priority cards):**
- Border-radius: `var(--radius)` → 16px
- Add `::after` radial gradient glow on hover
- Hover lift: -3px → -4px
- Smoother transitions: `cubic-bezier(.4,0,.2,1)`

**`.wiz-build-btn`:**
- Border-radius → 16px
- Stronger hover shadow with accent bleed
- Wider shimmer (rgba .1 → .12)

**`.rail-sec`:**
- Border-radius → 16px
- Add subtle accent glow when expanded: `box-shadow:0 2px 12px color-mix(in srgb, var(--accent) 6%, transparent)`

**`.det-section`:**
- Border-radius → 16px
- Header padding: 14px 18px → 16px 20px
- Better expanded state shadow

**`.wiz-close`:**
- Add `transform:scale(1.05)` on hover

**`.ld` (loading):**
- Title font-weight → 800
- Progress bar: add `box-shadow:inset 0 1px 3px rgba(0,0,0,.2)` to bar, `0 0 8px var(--accent-dim)` to fill
- Spinner: add `filter:drop-shadow(0 0 6px var(--accent-dim))`

**`.ai-sug`:**
- Border-radius → 16px
- Add `box-shadow:0 4px 16px color-mix(in srgb, var(--accent) 5%, transparent)`
- Header font-weight → 700

**Scrollbars:** Add `scrollbar-color:var(--accent-dim) transparent` to `.wiz-rail-scroll` and `.wiz-main`

---

## Refinement Pass 1 (after all 10 commits)

Create a new commit: `chore: refinement pass 1`

Run these checks and fix anything found:

```bash
# 1. Verify all element IDs still findable after DOM restructuring
for id in simple-context simple-role simple-location simple-generate-btn suggest-context-btn save-context-btn simple-generate-status profile-qs-toggle profile-qs-panel setup-wizard-btn; do
    count=$(grep -c "id=\"$id\"" frontend/index.html 2>/dev/null || grep -c "id=\"$id\"" index.html 2>/dev/null)
    echo "$id: $count occurrence(s)"
done
# Each should be exactly 1

# 2. No duplicate .animate-spin definition
grep -rn 'animate-spin' --include='*.css' --include='*.html' | grep -v 'class.*animate-spin' | grep -v 'classList'

# 3. Verify SRI hashes untouched (Phase 5 audit)
grep -n 'integrity=' frontend/index.html 2>/dev/null || grep -n 'integrity=' index.html 2>/dev/null

# 4. No syntax errors in Python
python3 -c "import py_compile; py_compile.compile('server.py', doraise=True)" 2>/dev/null || \
python3 -c "import py_compile; py_compile.compile('backend/server.py', doraise=True)"

# 5. No merge conflict markers
grep -rn '<<<<<<\|>>>>>>' --include='*.py' --include='*.js' --include='*.html' --include='*.css'

# 6. Settings.js still references correct element IDs
grep -n 'simple-context\|simple-role\|simple-location\|simple-generate' frontend/settings.js 2>/dev/null || \
grep -n 'simple-context\|simple-role\|simple-location\|simple-generate' settings.js 2>/dev/null | head -20

# 7. Warmup endpoint doesn't bypass rate limiting accidentally
grep -n 'agent-warmup' server.py backend/server.py 2>/dev/null

# 8. Theme editor range sliders persist correctly
# Manual: Open theme editor → adjust Card Opacity → close → reopen → check slider position
```

Fix any issues found, then `git add -A && git commit -m "chore: refinement pass 1"`.

---

## Refinement Pass 2 (functional verification)

Create commit: `chore: refinement pass 2`

```bash
# 1. Check the full settings tab structure is valid HTML
python3 -c "
from html.parser import HTMLParser
with open('frontend/index.html' if __import__('os').path.exists('frontend/index.html') else 'index.html') as f:
    HTMLParser().feed(f.read())
print('HTML parsed OK')
"

# 2. Verify context box appears in Sources tab (data-stab="sources")
grep -B2 'id="simple-context"' frontend/index.html 2>/dev/null || grep -B2 'id="simple-context"' index.html 2>/dev/null
# Should show a parent with data-stab="sources"

# 3. Verify role/location stay in Profile tab
grep -B2 'id="simple-role"' frontend/index.html 2>/dev/null || grep -B2 'id="simple-role"' index.html 2>/dev/null
# Should show a parent with data-stab="profile"

# 4. Verify tour key is preserved
grep 'stratos_tour_never' frontend/auth.js 2>/dev/null || grep 'stratos_tour_never' auth.js 2>/dev/null

# 5. Verify username lookup in server.py
grep -A5 'display_name.*COLLATE NOCASE' server.py backend/server.py 2>/dev/null

# 6. Verify no console.log/debugger left (except intentional ones)
grep -rn 'console\.log\|debugger' --include='*.js' | grep -v node_modules | grep -v '// console' | grep -v 'console.warn\|console.error\|console.log.*Agent.*pre' | head -10

# 7. Count total changes
git diff main --stat
```

Fix any issues, then `git add -A && git commit -m "chore: refinement pass 2"`.

---

## Final Verification

```bash
# Full git log review
git log --oneline main..HEAD

# Expected: ~12 commits (10 features + 2 refinements)
# Each independently revertable

# Verify no regressions to audit work
git diff main -- server.py | grep -c 'rate_limited\|security.*header\|X-Content-Type\|X-Frame-Options'
# Should be 0 removed lines for those patterns

# Merge
git checkout main
git merge feature/ui-fixes-and-polish
git log --oneline -15
```

---

## File Reference Summary

| File | Changes |
|------|---------|
| `server.py` | Username→email lookup (Fix 1), email DB fallback (Feat 8), agent-warmup endpoint (Feat 7) |
| `auth.js` | Preserve `stratos_tour_never` (Fix 2) |
| `styles.css` | `.animate-spin` class (Fix 3), range slider CSS (Feat 5), `.glass-panel` vars (Feat 5) |
| `app.js` | Progress bar `news_done` (Fix 4), agent warmup call (Feat 7) |
| `theme-editor.js` | Effects group, range inputs, derived vars, sync/clear (Feat 5) |
| `markets-panel.js` | Per-card refresh button + `_mpRefreshSingle()` (Feat 6) |
| `index.html` | Split profile panel, tooltips, agent badges (Refactor 9) |
| `wizard.js` | CSS polish in `WIZ_CSS` string (Style 10) |
