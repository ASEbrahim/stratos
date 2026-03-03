# StratOS Full Codebase Audit — 2026-03-04

> Automated scan of backend Python, frontend JS, config, tests, dependencies, DB schema, and git health.
> 119 commits analyzed. ~45,000 lines scanned across 40+ files.

---

## Executive Summary

| Area | P0 (Critical) | P1 (Important) | P2 (Minor) | Total |
|------|:---:|:---:|:---:|:---:|
| Backend Python | 3 | 14 | 19 | 36 |
| Frontend JS | 1 | 19 | 23 | 43 |
| Config/Tests/Deps/Git | 3 | 10 | 10 | 23 |
| **Grand Total** | **7** | **43** | **52** | **102** |

**Top 7 Critical Issues (P0):**
1. Plaintext API keys in git-tracked `config.yaml` (Google, Serper, Gmail SMTP)
2. `scan_status` dict mutated from multiple threads without synchronization
3. SQLite connection shared across threads — only commits are locked, not cursor ops
4. SHA-256 PIN hashing without salt — entire PIN keyspace crackable in seconds
5. Hardcoded PIN "5080" in client-side JS to unlock Serper API key field
6. Untracked `data/users/` directory with PII not in `.gitignore`
7. Alternate DB file `stratos.db` not in `.gitignore`

---

## BACKEND PYTHON FINDINGS

### Bugs

| ID | Severity | File | Description |
|----|----------|------|-------------|
| BUG-01 | **P0** | main.py:114-125,319-326,726-728 | `scan_status` dict mutated from scan thread, market refresh thread, and HTTP handler threads without any lock. Can cause data corruption. |
| BUG-02 | **P0** | database.py:33 | `check_same_thread=False` but only `_commit()` holds the lock. `cursor.execute()` + `cursor.fetchone()` are unprotected — two threads can interleave queries. |
| BUG-03 | P1 | database.py:668-685 | `save_ui_state` uses `with self.conn:` (auto-commit) bypassing the `_commit()` lock — second unprotected commit path. |
| BUG-04 | P1 | auth.py:242-259 | `_validate_db_session` calls `get_database()` which may create a second DB instance under edge conditions. Also DELETEs expired sessions during auth check (side effect). |
| BUG-05 | P1 | main.py:716-794 | `run_market_refresh` does NOT acquire `_scan_lock`. Sets `is_scanning=True` while a concurrent scan may be running. Both write `scan_status` and output file simultaneously. |
| BUG-06 | P1 | server.py:1468-1504 | Profile preset load does `strat.config.update(preset)` — if preset contains an API key, it overwrites the global config, leaking to other users. |
| BUG-08 | P1 | main.py:1226-1262 | Deferred briefing patches output file without holding any lock. Concurrent scan can overwrite new output with stale data. |

### Security

| ID | Severity | File | Description |
|----|----------|------|-------------|
| SEC-01 | **P0** | auth.py:79-82 | PIN hashed with plain SHA-256, no salt. 4-6 digit PINs crackable instantly with rainbow tables. |
| SEC-02 | P1 | server.py:81 | `Access-Control-Allow-Origin: *` on ALL endpoints including authenticated ones. Any website can make cross-origin requests. |
| SEC-03 | P1 | auth.py:60-65 | Session tokens stored in plain-text `.sessions.json`. File read access = session hijacking. |
| SEC-04 | P1 | server.py:1170-1178 | Path traversal potential in ticker/profile preset names. Sanitization allows spaces/hyphens. |
| SEC-05 | P1 | server.py:1127-1138 | `/api/save-serper-key` writes API key to git-tracked `config.yaml` via `yaml.dump`. |
| SEC-06 | P2 | server.py:247-251,490 | Error messages leak internal paths/stack traces to API clients. |
| SEC-07 | P1 | server.py:850-857 | `/api/scan/cancel` has no auth check — any unauthenticated user can cancel scans. |

### Performance

| ID | Severity | File | Description |
|----|----------|------|-------------|
| PERF-03 | P1 | server.py:656-717 | `yfinance.download()` in `/api/top-movers` blocks handler thread for 5-15s. |
| PERF-04 | P1 | main.py:431-432 | Per-item DB commit in `save_news_item` loop. 200+ items = 200+ disk syncs. Should batch. |
| PERF-01 | P2 | server.py:162-197 | `/api/data` reads entire JSON + computes MD5 hash on every request. Should cache. |
| PERF-02 | P2 | scorer_adaptive.py:247-359 | `_universal_noise_check` runs 8 lists of regex per article. Could pre-compile/combine. |

### Missing Error Handling

| ID | Severity | File | Description |
|----|----------|------|-------------|
| ERR-01 | P1 | server.py:130-142 | Profile ID resolution silently swallows all exceptions. Falls back to `profile_id=0`. |
| ERR-03 | P1 | routes/config.py:163-164 | `yaml.dump` failure after `open("w")` truncates config file. Should write-then-rename. |
| ERR-04 | P2 | main.py:233-242 | `.env` parser doesn't handle quoted values or `export` prefix. |
| ERR-06 | P2 | routes/config.py:244-261 | Profile YAML sync failures logged at DEBUG — invisible in normal operation. |
| ERR-07 | P2 | user_data.py:43-44,59-60 | Per-user data write failures swallowed at DEBUG level. |

### Consistency

| ID | Severity | File | Description |
|----|----------|------|-------------|
| CON-01 | P1 | auth.py + routes/auth.py | Two parallel auth systems: legacy PIN (SHA-256, file-backed) vs email (bcrypt, DB-backed). Dual path creates confusion. |
| CON-05 | P1 | main.py:284-621,796-1081 | Two-pass scoring pipeline duplicated between `_run_scan_impl` and `_run_news_refresh_impl`. |
| CON-02 | P2 | server.py | Three different JSON response patterns (`_send_json`, `json_response`, manual). |
| CON-04 | P2 | main.py + docs | Score 5.0 boundary: code uses `5.0 < score` (exclusive) but docs say "5.0-6.9 = Medium". |

### Dead Code

| ID | Severity | File | Description |
|----|----------|------|-------------|
| DEAD-01 | P2 | scorer_adaptive.py:746-785 | `_build_llm_prompt` (v1 format) never called. `_llm_score` uses `_build_llm_prompt_v2`. |
| DEAD-03 | P2 | server.py:43-50 | `_send_json` duplicates `json_response` from helpers (without gzip). |

---

## FRONTEND JAVASCRIPT FINDINGS

### Security

| ID | Severity | File | Description |
|----|----------|------|-------------|
| S1 | **P0** | settings.js:~143 | Hardcoded PIN "5080" in client-side JS to unlock Serper API key field. Visible in DevTools. |
| S2 | P1 | ui.js:770 | `showToast()` uses `innerHTML` with message — XSS if message contains user data. |
| S3 | P1 | ui.js:672 | `submitAI()` injects `data.error` via innerHTML — server error messages rendered as HTML. |
| S4 | P1 | ui.js:737 | `submitRating()` injects `data.error` via innerHTML. |
| S5 | P1 | agent.js, markets-panel.js, mobile.js | `formatAgentText()` converts markdown-like text to HTML via regex. Crafted AI responses could inject HTML/scripts. |
| S6 | P1 | auth.js | PIN sent in plaintext over HTTP. Credential exposure without HTTPS. |
| S7 | P2 | sw.js:54-66 | Service Worker caches API responses including potentially sensitive data (`/api/config`). |
| S8 | P2 | auth.js, app.js | Auth tokens in `localStorage` — accessible to any XSS on same origin. |

### Bugs

| ID | Severity | File | Description |
|----|----------|------|-------------|
| B4 | P1 | market.js:1110-1111 | `toggleManualSearch()` and `applyManualSearchState()` are empty stubs. UI buttons call them but nothing happens. |
| B5 | P1 | app.js:1560-1606 | Race condition: `checkStatus()` called from `setInterval`. Multiple concurrent executions race on `isAutoLoading` flag. |
| B6 | P1 | market.js:1295-1299 | `_mainChartFit()` uses `fitContent()` — contradicts project convention to use `setVisibleLogicalRange()`. |
| B7 | P1 | theme-editor.js | `updateChart()` called without checking if chart is initialized. Fails if theme editor opened before market data loads. |
| B8 | P2 | scan-history.js:~124 | Uses deprecated implicit `window.event` global. Fails in strict mode. |

### Performance

| ID | Severity | File | Description |
|----|----------|------|-------------|
| FP1 | P1 | Throughout (app.js, settings.js, ui.js, etc.) | `lucide.createIcons()` called 15+ times across files. Scans entire DOM each time. Should use targeted `{nodes: [container]}`. |
| FP2 | P1 | feed.js | `renderFeed()` rebuilds entire innerHTML on every call. Destroys and recreates all DOM nodes for 80+ items. |
| FP3 | P1 | nav.js | `getTabCount()` recomputes `deduplicateNews()` + filters on every `renderNav()` call. Should memoize. |
| FP4 | P2 | ui.js:197-483 | Star canvas animation `requestAnimationFrame` loop runs continuously even when occluded by other views. |

### Mobile

| ID | Severity | File | Description |
|----|----------|------|-------------|
| M1 | P1 | mobile.js | Sidebar swipe and card swipe may conflict near left screen edge. |
| M2 | P1 | mobile.js:1037-1053 | Fullscreen chart hotbar doesn't account for `env(safe-area-inset-bottom)` on gesture-nav devices. |
| M3 | P1 | markets-panel.js:623-639 | Agent resize handle only has `mousedown` — no touch event handler for mobile. |

### Consistency

| ID | Severity | File | Description |
|----|----------|------|-------------|
| C1 | P1 | All JS files | Mixed `var`/`let`/`const` across files. `markets-panel.js` uses `var` extensively; others use `let`/`const`. |
| C2 | P1 | feed.js, markets-panel.js, mobile.js | Three duplicate `esc()` functions: `esc()`, `_mpEsc()`, `_cfsEsc()` — identical logic, different names. |
| C3 | P2 | agent.js, markets-panel.js, mobile.js | Three separate chat implementations with similar streaming/history/UI patterns. Should share a common module. |
| C6 | P2 | market.js, agent.js, markets-panel.js, mobile.js | Four different price formatting functions (`fp`, `_mpFp`, `_fsFmtPrice`). |

### Dead Code

| ID | Severity | File | Description |
|----|----------|------|-------------|
| D1 | P2 | market.js:1109-1111 | Empty `toggleManualSearch()` and `applyManualSearchState()` stubs. |
| D2 | P2 | tour.js:811-814 | `_dismissOldWizard()` is a documented no-op for backward compat. |
| D4 | P2 | markets-panel.js:652-734 | `mpImportMarketAgent()` and `mpHandleAgentImport()` contain near-identical import logic. |
| D5 | P2 | settings.js:620-686 | `legacyFields = []` — always empty, `forEach` loops are no-ops. |

### Accessibility

| ID | Severity | File | Description |
|----|----------|------|-------------|
| A1 | P1 | index.html, all JS | No `aria-label` on icon-only buttons. Screen readers announce empty buttons. |
| A2 | P1 | feed.js | Feed cards are `<div>` with `onclick` — not focusable, no keyboard nav. |
| A3 | P1 | market.js, feed.js | Price changes communicated solely through color (red/green). No text indicators for color-blind users. |
| A5 | P2 | ui.js:749-785 | Toast notifications lack `role="alert"` or `aria-live`. |
| A6 | P2 | wizard.js, tour.js | Modal overlays don't trap keyboard focus. |

---

## CONFIG / TESTS / DEPS / GIT FINDINGS

### Config

| ID | Severity | File | Description |
|----|----------|------|-------------|
| CFG-01 | **P0** | config.yaml (lines 38-53) | Plaintext API keys for Google (`AIzaSy...`), Serper (`bc26c29...`), and Gmail SMTP (`REDACTED_SMTP_PASSWORD`) in git-tracked file. |
| CFG-02 | P1 | config.yaml:42 | `serper_credits: 0` — stale value that could mislead. |
| CFG-03 | P2 | config.yaml | `schedule.background_enabled: false` but interval configured. Potential user confusion. |

### Test Coverage

| Missing Coverage | Severity |
|-----------------|----------|
| Auth flow (login, register, session, OTP) | **P1** |
| Scan pipeline (main.py `run_scan`) | **P1** |
| Database operations (database.py, 700 lines) | **P1** |
| Migration framework (10 migrations) | **P1** |
| Profile switching / isolation | **P1** |
| News fetching, market fetching | P2 |
| Briefing generation, API routes, SSE | P2 |
| Email service, config save, agent chat | P2 |

**Current coverage:** 1 test file (635 lines, 47 tests) covering format alignment, forbidden 5.0, score parsing, language filtering, reasoning stripping. Everything else is untested.

### Dependencies

| ID | Severity | Description |
|----|----------|-------------|
| DEP-01 | P1 | `anthropic` package imported by distill.py but missing from requirements.txt |
| DEP-02 | P1 | `pytest` used in tests but missing from requirements.txt |
| DEP-03 | P2 | No pip lockfile. Builds not reproducible. |

### Database Schema

| ID | Severity | Description |
|----|----------|-------------|
| DB-01 | P1 | `is_url_seen()` ignores `profile_id` — returns True for ALL profiles if URL exists for any one. |
| DB-02 | P1 | No index on `user_feedback.url`. `was_dismissed()` scans full table. |
| DB-03 | P2 | `briefings` table lacks composite unique constraint for one-per-profile-per-day. |
| DB-04 | P2 | `entities.mention_count` column never updated — vestigial. |

### Git Health

| ID | Severity | Description |
|----|----------|-------------|
| GIT-01 | **P0** | `backend/data/users/` directory with PII not in `.gitignore`. Could be committed with `git add -A`. |
| GIT-02 | **P0** | `backend/stratos.db` not in `.gitignore` (only `strat_os.db` is ignored). |
| GIT-03 | P1 | `backend/data/archive/news_data_Ahmad.json` (5.2MB) tracked in git. |
| GIT-04 | P1 | 44 unpushed commits to remote. |
| GIT-05 | P1 | Multi-MB JSONL files in `data/v2_pipeline/` tracked in git. |

### Documentation Discrepancies

| ID | Severity | Description |
|----|----------|-------------|
| DOC-01 | P1 | CLAUDE.md says "secrets live in `.env`, NOT in `config.yaml`" — but `config.yaml` has plaintext keys. |
| DOC-02 | P1 | CLAUDE.md doesn't mention `routes/auth.py` (email auth) — only references legacy `auth.py`. |
| DOC-03 | P2 | CLAUDE.md database section lists 7 tables but 12+ exist (missing: `users`, `profiles`, `sessions`, etc.). |
| DOC-04 | P2 | CLAUDE.md doesn't mention `email_service.py` or `user_data.py`. |

---

## Recommended Fix Priority

### Immediate (P0 — do now)
1. **Remove plaintext secrets from config.yaml** — replace with `${VAR}` placeholders
2. **Add `data/users/` and `stratos.db` to .gitignore**
3. **Add thread lock around scan_status reads/writes**
4. **Wrap SQLite cursor operations in the same lock as commits**
5. **Remove hardcoded PIN from frontend settings.js**
6. **Salt the legacy PIN hashes** (or deprecate PIN auth entirely)

### Next Sprint (P1 — this week)
7. Fix deferred briefing file write race condition
8. Fix market refresh not acquiring scan lock
9. Sanitize innerHTML assignments (XSS vectors)
10. Add `safe-area-inset-bottom` to mobile hotbar
11. Batch DB commits in scan pipeline
12. Add missing `.gitignore` entries + untrack large files
13. Add auth check to `/api/scan/cancel`
14. Write tests for auth, scan pipeline, and database operations
15. Restrict CORS to known origins
16. Use targeted `lucide.createIcons({nodes: [container]})`
17. Fix `is_url_seen()` to filter by profile_id

### Backlog (P2 — when convenient)
18. Unify JSON response patterns in server.py
19. Unify duplicate esc()/formatPrice functions across JS files
20. Extract shared chat module from 3 agent implementations
21. Add ARIA labels to icon-only buttons
22. Memoize nav tab counts
23. Remove dead code (unused prompt builder, empty stubs, legacy arrays)
24. Add pip lockfile
25. Update CLAUDE.md with current architecture

---

*Audit generated by Claude Code — 2026-03-04. 3 parallel agents scanned backend, frontend, and infrastructure.*
