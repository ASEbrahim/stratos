# Session Report — 2026-02-28

> Auth Pipeline + Profile Isolation + Per-User Data Directories

**Commits:** 10 (d047344 → 1e9f92f)
**Files changed:** 12+ (backend + frontend submodule)
**New files:** `user_data.py`, `data/archive/*`, `data/users/*`

---

## Overview

This session completed a full auth system overhaul and per-user data export pipeline. The work spans three areas:

1. **Auth pipeline** — email verification replaces invite codes, pending registrations staging, SMTP delivery, password UX
2. **Profile data isolation** — prevent cross-user data bleed via blank config/data for profileless users and localStorage whitelist wipe
3. **Per-user data directories** — structured JSONL/JSON exports for developer analysis, legacy profile archive

---

## 1. Commits

| Hash | Type | Summary |
|------|------|---------|
| `d047344` | feat | Email verification replaces invite codes, 5-digit code with auto-login |
| `7a6d0ad` | fix | Users only created after email verification, block unverified login |
| `72851ca` | feat | SMTP via .env, delete account route, env-flexible email provider |
| `26aaef2` | feat | Password show/hide eye toggle on all auth fields |
| `ac42ec6` | fix | Prevent profile data bleed to new/profileless users |
| `e2c6733` | fix | Profile data bleed + tour "Don't show again" |
| `161a6cd` | feat | Tour "Don't show again" + preserve preference across logins |
| `24939a4` | feat | Per-user data directories with JSONL exports + legacy profile archive |
| `1e9f92f` | chore | Update STATE.md with auth pipeline, data isolation, per-user data dirs |

All commits are independent and rollback-safe via `git revert`.

---

## 2. Auth Pipeline

### 2.1 Registration Flow (Before → After)

**Before:**
```
Register → INSERT INTO users → send verification email → user verifies later
Problem: unverified users exist in users table and can login
```

**After:**
```
Register → INSERT INTO pending_registrations → send verification email
Verify   → INSERT INTO users (verified=true) → DELETE from pending → auto-login
```

### 2.2 Migration 009 — pending_registrations Table

```sql
CREATE TABLE IF NOT EXISTS pending_registrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    verification_code_hash TEXT NOT NULL,
    verification_expires DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_pending_email ON pending_registrations(email);
```

- Verification code: 5-digit (10000-99999), SHA-256 hashed, 15-minute expiry
- Re-registration: `DELETE FROM pending_registrations WHERE email = ?` before insert (upsert behavior)
- First user auto-promoted to admin

### 2.3 SMTP Configuration

Credentials in `backend/.env` (gitignored):
```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=StratintOS@gmail.com
SMTP_PASSWORD=<app-password>
```

`main.py` loads `.env` on startup, overrides config.yaml `email` section. Works with any SMTP provider (Gmail, Fastmail, Outlook, etc.). If SMTP not configured, verification codes logged to console for dev/testing.

### 2.4 Login Enhancements

- **Login by email or username:** `WHERE email = ?` OR `WHERE LOWER(display_name) = ?`
- **Block unverified:** `if not email_verified: → 403 "Email not verified"`
- **Password upgrade:** Legacy SHA-256/PIN hashes auto-upgraded to bcrypt on successful login

### 2.5 Account Deletion

- Route: `DELETE /api/auth/delete-account`
- Requires password confirmation
- Cascading delete: sessions → profiles → user
- Confirmation dialog in frontend with password prompt

### 2.6 Password Eye Toggle

`_togglePw(id)` toggles input between `type="password"` and `type="text"`. SVG eye icon (open/closed) rendered via `_pwEye(id)`. Applied to all 7 password/PIN fields:
- Login password
- Register password + confirm
- Email register password + confirm
- Password reset (new password + confirm)

CSS: `.pw-input-wrap` (relative positioning), `.pw-eye-btn` (absolute right, translucent)

---

## 3. Profile Data Isolation

### 3.1 Problem

When a new user logged in, they saw:
- Previous user's role, location, context in profile settings
- Previous user's categories and signals in the dashboard
- Previous user's market tickers
- Tour didn't trigger (tour flags from previous user persisted)

### 3.2 Root Causes

| # | Cause | Location |
|---|-------|----------|
| 1 | `/api/config` returns `strat.config` (singleton) regardless of who's asking | `server.py` |
| 2 | `/api/data` returns previous user's data when no profile is set | `server.py` |
| 3 | `_clearProfileLocalStorage()` only cleared 7 hardcoded keys | `auth.js` |
| 4 | In-memory `configData` variable not cleared on login | `auth.js` |

### 3.3 Fixes

**Server-side (server.py):**
```python
_cfg_token = self.headers.get('X-Auth-Token', '')
_cfg_profile = auth.get_session_profile(_cfg_token) if _cfg_token else ''
cfg = strat.config if _cfg_profile else {
    "profile": {}, "market": {"tickers": []}, "news": {"timelimit": "w"},
    "search": strat.config.get("search", {}),
    "dynamic_categories": [], "scoring": strat.config.get("scoring", {}),
}
```

- Profileless users get blank config (no role/categories/tickers)
- Search keys preserved (needed for profile generation wizard)
- Scoring config preserved (for consistency)

**Client-side (auth.js) — whitelist wipe:**
```javascript
function _clearProfileLocalStorage() {
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (!k) continue;
        if (k === AUTH_TOKEN_KEY || k === AUTH_PROFILE_KEY ||
            k === DEVICE_ID_KEY || k === AUTH_THEME_KEY ||
            k === 'stratos_tour_never') continue;
        keysToRemove.push(k);
    }
    keysToRemove.forEach(k => localStorage.removeItem(k));
    // Clear in-memory variables
    if (typeof configData !== 'undefined') { configData = null; }
    if (typeof newsData !== 'undefined') { newsData = []; }
    if (typeof marketData !== 'undefined') { marketData = {}; }
    // ... other data arrays
}
```

Called on: login success, email verification success, password reset, logout.

### 3.4 Tour "Don't Show Again"

- `stratos_tour_never` flag in localStorage (global, not per-profile)
- Whitelisted in `_clearProfileLocalStorage()` — survives login wipes
- "Don't show again" button on both welcome modal and step tooltips
- CSS: `.tour-never` and `.tour-welcome-never` button styles

---

## 4. Per-User Data Directories

### 4.1 Directory Structure

```
backend/data/users/{user_id}/
├── profile.json              # Config snapshot (updated on every config save)
├── scan_log.jsonl            # One line per scan (metrics, score distribution)
├── feedback.jsonl            # User feedback (ratings, saves, dismissals)
├── briefings/
│   └── YYYY-MM-DD.json       # Daily briefings (full JSON document)
└── scans/
    └── YYYY-MM-DD.jsonl      # Articles scored per day (one line per article)
```

### 4.2 Utility Module — user_data.py

```python
_BASE = Path(__file__).parent / "data" / "users"

def get_path(user_id: int) -> Path:
    return _BASE / str(user_id)

def ensure_dir(user_id: int) -> Path:
    p = get_path(user_id)
    (p / "scans").mkdir(parents=True, exist_ok=True)
    (p / "briefings").mkdir(parents=True, exist_ok=True)
    return p

def append_jsonl(user_id: int, filename: str, data: dict):
    # Appends one JSON line. No-op for user_id <= 0. Creates dir if missing.

def write_json(user_id: int, filename: str, data: dict):
    # Writes/overwrites JSON file. No-op for user_id <= 0.

def get_user_id_for_profile(db, profile_id: int) -> int:
    # Resolves profile_id → user_id via DB. Returns 0 for legacy profiles.
```

### 4.3 Write Hooks

| Trigger | DB Method | Export File | Fields |
|---------|-----------|-------------|--------|
| Scan completion | `save_scan_log()` | `scan_log.jsonl` | scan_id, timestamp, elapsed_secs, items_fetched, items_scored, critical, high, medium, noise, rule_scored, llm_scored, retained |
| User feedback | `save_feedback()` | `feedback.jsonl` | timestamp, action, title, url, category, ai_score, user_score, note |
| Briefing generation | `save_briefing()` | `briefings/YYYY-MM-DD.json` | Full briefing document |
| Output write | `_write_output()` | `scans/YYYY-MM-DD.jsonl` | title, url, score, score_reason, category, source, timestamp |
| Config save | `handle_config_save()` | `profile.json` | profile, market.tickers, dynamic_categories, scoring, feeds, updated_at |

### 4.4 Directory Creation Points

- **Email verification** (`routes/auth.py`, verify route): `user_data.ensure_dir(user_id)` after INSERT INTO users
- **Profile activation** (`routes/auth.py`, activate route): `user_data.ensure_dir(user_id)` after session update

### 4.5 Config Overlay DB Sync

`_sync_to_db_profile()` in `routes/config.py` — called on every POST /api/config:

1. Resolve session token → profile_id + user_id
2. Build overlay dict (user-facing fields only, no API keys/SMTP)
3. `UPDATE profiles SET config_overlay = ? WHERE id = ?`
4. `user_data.write_json(user_id, "profile.json", snapshot)`

Overlay fields: profile (role/location/context), market.tickers, news.timelimit, dynamic_categories, extra_feeds_finance/politics, custom_feeds, custom_tab_name, scoring (retention settings).

### 4.6 _write_output() Call Sites

All 8 call sites in main.py updated to pass `profile_id`:

| Line | Context | profile_id source |
|------|---------|-------------------|
| 427 | Pass 1 partial output (full scan) | `_scan_pid` |
| 442 | Cancelled after pass 1 (full scan) | `_scan_pid` |
| 535 | Final output (full scan) | `_scan_pid` |
| 764 | Market-only refresh | 0 (no articles) |
| 904 | Pass 1 partial output (news refresh) | `_scan_pid` |
| 919 | Cancelled after pass 1 (news refresh) | `_scan_pid` |
| 1009 | Final output (news refresh) | `_scan_pid` |
| 1233 | Deferred briefing patches output | `profile_id` (closure) |

---

## 5. Legacy Profile Archive

### 5.1 What Was Archived

| File | Destination | Size |
|------|-------------|------|
| `profiles/Ahmad.yaml` | `data/archive/Ahmad.yaml` | 4 KB |
| `profiles/Kyu.yaml` | `data/archive/Kyu.yaml` | 8 KB |
| `output/news_data_Ahmad.json` | `data/archive/news_data_Ahmad.json` | 5.4 MB |

### 5.2 What Was Deleted

- `profiles/Ahmad.yaml`, `profiles/Kyu.yaml`
- `profiles/Ahmad/`, `profiles/Kyu/` (empty preset dirs)
- `output/news_data_Ahmad.json`
- `output/bleed_test_*.json` (diagnostic artifacts)
- `output/news_data.json` (stale legacy default)

### 5.3 Why

Legacy YAML profiles served no purpose for DB-auth users. The `profiles/` folder is now empty. Config is stored in `profiles.config_overlay` DB column. The archive preserves Ahmad's curated 100-article output and both profile configs for reference.

---

## 6. Current DB-Auth Users

| user_id | Email | Display Name | Profiles |
|---------|-------|-------------|----------|
| 3 | ahmad@test.com | Ahmad | Developer_KW (id=4), Chef_Tokyo (id=5) |
| 4 | bob@test.com | Bob | Bob_Profile (id=6) |
| 5 | newuser@test.com | NewUser | (none) |
| 10 | kirissie@gmail.com | Kirissie | (none) |

---

## 7. Decisions Made

| ID | Decision | Rationale | Rejected |
|----|----------|-----------|----------|
| D023 | pending_registrations staging table | Prevent half-created accounts | email_verified flag (row exists before verify) |
| D024 | SMTP creds in .env | Security, single config point | config.yaml (VCS exposure), per-user SMTP |
| D025 | localStorage whitelist wipe | Future-proof, complete cleanup | Blacklist (always incomplete) |
| D026 | Blank config for profileless users | Prevent data bleed | Global config clear (breaks other sessions) |
| D027 | Per-user dir keyed by user_id | Consolidate across profiles | Per-profile dirs (fragmented), per-username (mutable) |
| D028 | JSONL for timeseries, JSON for snapshots | Grep-friendly, atomic append | All JSON (can't append), SQLite export (duplicates DB) |
| D029 | Triple config sync (YAML + DB + JSON) | Each serves different purpose | DB-only or YAML-only |

---

## 8. Failures Identified & Fixed

| ID | Failure | Fix |
|----|---------|-----|
| F014 | `/api/config` returns previous user's in-memory config | Check active profile, return blank for profileless |
| F015 | localStorage blacklist misses new keys | Whitelist approach: wipe everything except auth keys |
| F016 | Users stored in DB before email verification | Staging table (pending_registrations) |

---

## 9. What's Not Yet Done

- **Config overlay loading on profile switch:** activating a profile should load its `config_overlay` into `strat.config`
- **Agent chat logging:** per-user JSONL for agent conversations needs DB table + frontend wiring
- **Admin panel:** user management UI, session viewer, system health dashboard
- **Frontend SSE for briefing_ready:** backend broadcasts event but frontend doesn't listen yet

---

## 10. Verification Results

| Test | Result |
|------|--------|
| Server starts clean | PASS — no errors, all imports OK |
| `get_user_id_for_profile()` resolves correctly | PASS — profile_id=5 → user_id=3 |
| `ensure_dir()` creates correct structure | PASS — scans/ and briefings/ subdirs created |
| `append_jsonl()` writes and appends | PASS — test entry written and verified |
| Config save → DB overlay update | PASS — `config_overlay` column updated |
| Config save → profile.json snapshot | PASS — profile.json written with correct keys |
| Legacy profiles archived | PASS — `data/archive/` contains 3 files |
| `profiles/` dir empty | PASS — no YAML files remain |
| Migration 009 applied | PASS — `pending_registrations` table exists |
