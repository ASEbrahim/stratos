# Audit Session Changelog

## Session: 2026-03-14

Each entry documents a code change with before/after context.

---

### C001: `read_json_body` missing JSONDecodeError handling
- **File**: `backend/routes/helpers.py`
- **Function**: `read_json_body()` at line 202
- **Before**: `json.loads(text)` with no try/except — raises unhandled JSONDecodeError on malformed POST bodies
- **After**: Wrapped in `try/except (json.JSONDecodeError, UnicodeDecodeError)`, returns `{}` on invalid input
- **Severity**: Medium (500 error on malformed requests)

### C002: `_notify` mutates event dict with `.pop()`
- **File**: `backend/processors/youtube_worker.py`
- **Function**: `_notify()` at line 166
- **Before**: `event.pop('type')` mutates the caller's dict — could cause missing `type` key on subsequent accesses
- **After**: `event.get('type')` + dict comprehension to create a new payload without `type`
- **Severity**: Low (subtle data corruption)

### C003: `do_DELETE` missing `_session_profile` initialization
- **File**: `backend/server.py`
- **Function**: `CORSHandler.do_DELETE()` at line 534
- **Before**: `_session_profile` never set, `ensure_profile()` never called — DELETE requests could operate on wrong profile's config
- **After**: Mirrors do_GET/do_POST pattern — initializes `_session_profile`, calls `ensure_profile()`
- **Severity**: High (cross-profile data leak on DELETE operations)

### C004: `_gzip_cache` thread safety
- **File**: `backend/server.py`
- **Function**: `CORSHandler.do_GET()` static file gzip section
- **Before**: `_gzip_cache` dict read/written from multiple threads without synchronization — race condition
- **After**: Added `_gzip_cache_lock` (threading.Lock) protecting all reads and writes
- **Severity**: Low (rare corruption, mostly cosmetic)

### C005: Health endpoint filtered scan log to profile_id=0
- **File**: `backend/routes/data_endpoints.py`
- **Function**: `handle_get()` `/api/health` at line 411
- **Before**: `get_scan_log(1)` defaults to `profile_id=0`, only returning legacy-profile scans
- **After**: `get_scan_log(1, profile_id=None)` returns most recent scan regardless of profile
- **Severity**: Medium (stale health data)

### C006: Market refresh read-modify-write race condition
- **File**: `backend/main.py`
- **Function**: `run_market_refresh()` at line 814
- **Before**: Output JSON read happened outside `_output_lock` — race with deferred briefing writer could lose briefing data
- **After**: Entire read-modify-write sequence runs under `_output_lock`
- **Severity**: High (data loss — briefing could be overwritten)

### C007: Double Content-Length evaluation in youtube_endpoints
- **File**: `backend/routes/youtube_endpoints.py`
- **Function**: `handle_post()` at line 313
- **Before**: `int(handler.headers.get('Content-Length', 0))` evaluated twice in same expression
- **After**: Evaluate once into `_cl` variable
- **Severity**: Low (wasteful, no actual bug)

### C008: Database singleton not thread-safe
- **File**: `backend/database.py`
- **Function**: `get_database()` at line 640
- **Before**: Race condition — two threads could simultaneously create Database instances
- **After**: Double-checked locking pattern with `_db_instance_lock`
- **Severity**: Medium (potential duplicate DB instances on startup)

### C009: Incomplete user data deletion on account removal
- **File**: `backend/routes/auth.py`
- **Function**: `_delete_user_data()` at line 897
- **Before**: Only cleaned 5 original tables — missed 10 newer profile-scoped tables
- **After**: Iterates over all profile-scoped tables with try/except for schema compatibility
- **Severity**: High (orphaned user data after account deletion)

### C010: Wrong SSE manager attribute name in YouTube operations
- **File**: `backend/routes/youtube_endpoints.py` + `backend/server.py`
- **Before**: Referenced `strat.sse_manager` which doesn't exist — attribute is `strat.sse`. All `hasattr` checks silently returned False, disabling SSE broadcasts
- **After**: Corrected all 9 references to `strat.sse`
- **Severity**: High (all YouTube SSE notifications silently broken)
