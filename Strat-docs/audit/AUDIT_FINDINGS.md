# Audit Findings (Not Fixed)

Issues discovered but NOT safe to fix automatically.

---

### F001: Rate limiter is global per-path, not per-client
- **File**: `backend/auth.py`, `rate_limited()` at line 304
- **Issue**: Rate limiting uses path as the only key. One user hitting a limit blocks ALL users on that endpoint.
- **Recommended fix**: Key by `(path, client_ip)` or `(path, auth_token)`. Not fixed because it changes the rate limiting API contract and behavior.

### F002: `cgi.FieldStorage` deprecated in Python 3.13+
- **File**: `backend/routes/persona_data.py`, line 572
- **Issue**: `cgi.FieldStorage` for multipart form parsing is deprecated since Python 3.11 and removed in 3.13.
- **Recommended fix**: Use `multipart` or `email` stdlib parsers. Not fixed because it requires testing the import flow.

### F003: Profile config is shared mutable global state
- **File**: `backend/main.py`, `strat.config`
- **Issue**: `strat.config` is a single mutable dict shared across all request threads. While `_config_lock` protects profile switches, individual dict reads/writes during request handling are unsynchronized. Two concurrent requests from different profiles could read stale config.
- **Recommended fix**: Use per-request config snapshots (pass config as parameter rather than using `strat.config`). Not fixed because it's an architectural change.

### F004: `_sessions_file` persistence has no file locking
- **File**: `backend/auth.py`, `_save_sessions()` at line 73
- **Issue**: `.sessions.json` is written without file locking. Concurrent saves could corrupt it.
- **Recommended fix**: Use atomic write (write to temp file, then rename). Not fixed — sessions are backed by DB for new auth; this is legacy.

### F005: Empty `except` blocks suppress real errors
- **Files**: Multiple (server.py lines 170, 475; auth.py line 89; data_endpoints.py multiple)
- **Issue**: Bare `except Exception: pass` blocks hide errors that should be logged. Some are appropriate (optional feature detection), but many would benefit from at least debug-level logging.
- **Recommended fix**: Add `logger.debug()` calls. Not fixed en masse because some `pass` blocks are intentional.

### F006: Persona entities POST body may be read twice
- **File**: `backend/routes/persona_data.py`, lines 458-462
- **Issue**: If `len(parts) >= 5`, the body is read again at line 462 even though it may have already been consumed by a handler above in the same function (e.g., conversations POST at line 252). The stashed body pattern could cause issues.
- **Recommended fix**: Read body once at the top of `handle_post()`. Not fixed because the path prefix check prevents overlapping handlers.

### F007: `save_ui_state` uses connection context manager but `_commit` uses its own lock
- **File**: `backend/database.py`, `save_ui_state()` at line 601
- **Issue**: `with self.conn:` auto-commits on success, but the database also has `_commit()` with its own lock. The two commit mechanisms could conflict.
- **Recommended fix**: Use consistent commit strategy. Currently harmless because `with self.conn:` is isolated to `save_ui_state`.

### F008: `search_insights` in youtube.py uses LIKE '%query%' without full-text indexing
- **File**: `backend/processors/youtube.py`, `search_insights()` at line 823
- **Issue**: `LIKE '%query%'` forces a full table scan on the `content` column of `video_insights`. For large datasets, this will be slow.
- **Recommended fix**: Add SQLite FTS5 virtual table for insight content. Not fixed — performance optimization, not a bug.

### F009: `_tier3_whisper` loads model on every call (no caching)
- **File**: `backend/processors/youtube.py`, `_tier3_whisper()` at line 357
- **Issue**: `WhisperModel(model_name, ...)` is instantiated per-call. Model loading takes ~10s and ~1.6GB RAM.
- **Recommended fix**: Cache the model instance. Not fixed because Whisper is used infrequently and caching a 1.6GB model indefinitely may not be desirable.

---

## Session 2 Findings (Frontend Deep Audit)

### F010: agent.js uses wrong token key `stratos_session_token` for file upload
- **File**: `frontend/agent.js`, line 1152
- **Issue**: File upload uses `localStorage.getItem('stratos_session_token')` which doesn't exist. The correct key is `stratos_auth_token`.
- **Recommended fix**: Change to `getAuthToken()`. Not fixed because agent.js is in the do-not-touch list.

### F011: `_escForAttr` in games-ui.js doesn't escape backslashes
- **File**: `frontend/games-ui.js`, line 126
- **Issue**: `_escForAttr(s)` only escapes single quotes and double quotes, not backslashes. A scenario name containing `\` followed by `'` could break out of onclick attribute strings.
- **Recommended fix**: Add `.replace(/\\/g, '\\\\')` before the quote replacements. Not fixed — scenario names are user-created via a controlled prompt and the risk is self-XSS only.

### F012: Fullscreen chart IIFE leaks document-level event listeners
- **File**: `frontend/fullscreen-chart.js`, lines 605-615
- **Issue**: The intel panel drag-resize registers `mousemove`, `mouseup`, `touchmove`, `touchend` on `document` inside an IIFE. These listeners are never removed when the fullscreen chart closes. Each open/close cycle adds 4 new listeners.
- **Recommended fix**: Store handler references and remove them in `_fsClose()`. Not fixed because the handlers check a scoped `_ipDragging` flag so they're functionally inert, but they do leak memory over many open/close cycles.

### F013: `exportDashboard` relies on implicit `event` global
- **File**: `frontend/scan-history.js`, line 124
- **Issue**: `const btn = event?.target?.closest('.sh-export-btn')` uses the deprecated implicit `event` global. This works when called from inline `onclick` but would fail if called programmatically.
- **Recommended fix**: Pass `event` explicitly: `exportDashboard(format, event)`. Not fixed because all current call sites are inline onclick handlers where `event` is implicitly available.

### F014: prompt-builder.js uses native prompt/confirm/alert
- **File**: `frontend/prompt-builder.js`, lines 288, 352, 354, 360, 478, 488, 538
- **Issue**: Uses native `prompt()`, `confirm()`, `alert()` instead of the styled `stratosPrompt()`, `stratosConfirm()`, `showToast()` used everywhere else. Causes jarring UX inconsistency.
- **Recommended fix**: Replace with styled equivalents. Not fixed because it's a UX issue, not a functional bug, and the prompt-builder is a dev tool.

### F015: Service worker caches non-auth API responses
- **File**: `frontend/sw.js`, lines 56-73
- **Issue**: API responses not in `NO_CACHE_API` list (e.g., `/api/tts/voices`, `/api/feed-catalog/*`, `/api/agent-status`) are cached and served from cache on network failure. These responses may be profile-specific but the cache key doesn't include the auth token, so a profile switch could serve stale data.
- **Recommended fix**: Either add more endpoints to `NO_CACHE_API` or include a profile identifier in the cache key. Not fixed because the SW is network-first (cache only on network failure) and the impact is limited to offline scenarios.

### F016: `_gamesEntities` comparison mismatch in entity bar
- **File**: `frontend/games-ui.js`, line 368
- **Issue**: `_gamesActiveNpc.toLowerCase().replace(/ /g, '_') === e.name` compares the NPC display name (spaces→underscores) against the entity `name` field. But in `_gamesSelectEntity()` (line 383), `_gamesActiveNpc` is set to `displayName` (with spaces). The `_gamesSetNpc` and `_gamesAutoDetectNpc` functions also use display names. So the comparison in the entity bar highlight is inconsistent — it may fail to highlight the active character chip if the display name contains spaces that don't match the `name` field.
- **Recommended fix**: Compare against `e.display_name` instead of `e.name` (after the transform). Not fixed because the comparison happens to work when names are single words (no spaces), which is the common case.
