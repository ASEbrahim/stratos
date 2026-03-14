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
