# StratOS Comprehensive Change Plan

**Generated from:** Full codebase audit — every DB method, every caller, every endpoint, every data flow traced end-to-end.

**Guiding principles:**
- Zero schema migrations (all fixes are application-layer)
- Zero architectural changes (all fixes preserve existing patterns)
- Backward compatible (defaults match current behavior for legacy data)
- Ordered by impact × blast-radius (highest impact, lowest risk first)

---

## TIER 1 — Profile Isolation Bugs (Active Data Contamination)

### 1.1 `get_feedback_for_scoring()` — NO profile_id filter [CRITICAL]

**What:** `database.py:595-657` — All three queries (positive, negative, corrections) select from `user_feedback` with NO `WHERE profile_id = ?`. This feeds into `ScoringMemory._get_feedback()` → `format_feedback_for_prompt()` → injected into every LLM scoring prompt.

**Impact:** Profile A's thumbs-up/dismiss/rate signals contaminate Profile B's AI scoring. A petroleum engineer's "useful" articles boost scores for a game developer, and vice versa. This is *behavioral bleed* — it silently degrades scoring precision without any visible data leak.

**Call chain:**
```
scorer_adaptive.py:773,813,973  →  ScoringMemory.format_feedback_for_prompt()
  → ScoringMemory._get_feedback()  (scorer_base.py:595-612)
    → db.get_feedback_for_scoring(days=30, limit=15)  ← NO profile_id
```

**Fix — 4 files, ~15 lines:**

**(a) database.py** — Add `profile_id` parameter to method, add `AND profile_id = ?` to all three queries:
```python
def get_feedback_for_scoring(self, days=30, limit=20, profile_id=0):
```
Line 613: `WHERE created_at > ? AND profile_id = ? AND (` with params `(since, profile_id, limit)`
Line 629: same pattern
Line 644: same pattern

**(b) scorer_base.py** — Add `profile_id` to `ScoringMemory.__init__` and thread through:
```python
def __init__(self, memory_path=None, max_examples=10, min_score=8.5, db=None, profile_id=0):
    ...
    self.profile_id = profile_id
```
Line 607: `self.db.get_feedback_for_scoring(days=30, limit=15, profile_id=self.profile_id)`

ScoringMemory instantiation (line 739):
```python
self.memory = ScoringMemory(
    max_examples=..., min_score=..., db=db, profile_id=profile_id
)
```

ScorerBase.__init__ signature (line 727):
```python
def __init__(self, config, db=None, profile_id=0):
```

**(c) scorer_adaptive.py** — Thread `profile_id` through AdaptiveScorer:
```python
def __init__(self, config, db=None, profile_id=0):
    super().__init__(config, db=db, profile_id=profile_id)
```

**(d) main.py** — Pass `profile_id` to `_create_scorer`:
```python
def _create_scorer(config, db=None, profile_id=0):
    return AdaptiveScorer(config, db=db, profile_id=profile_id)
```
Line 99:  `self.scorer = _create_scorer(self.config, db=self.db, profile_id=0)`
Line 347: `self.scorer = _create_scorer(self.config, db=self.db, profile_id=_scan_pid)`  ← critical
Line 847: `self.scorer = _create_scorer(self.config, db=self.db, profile_id=_scan_pid)`

**Why this is the #1 fix:** Feedback is the personalization engine. Cross-profile feedback makes the AI score for a blended phantom user that matches nobody. Fixing this single bug may noticeably improve scoring quality for every multi-profile instance.

---

### 1.2 `get_feedback_stats()` — NO profile_id filter [MEDIUM]

**What:** `database.py:556-593` — All three queries (total count, by-action breakdown, score disagreements) select from `user_feedback` with NO `WHERE profile_id = ?`.

**Impact:** Any user hitting `/api/feedback-stats` (server.py:500, 1256) sees aggregate feedback stats from ALL profiles. Leaks: total feedback count, action breakdown, and the 20 worst score disagreements (including article titles and scores from other profiles).

**Fix — 2 files, ~8 lines:**

**(a) database.py:**
```python
def get_feedback_stats(self, profile_id=0):
```
Line 560: `SELECT COUNT(*) FROM user_feedback WHERE profile_id = ?` with `(profile_id,)`
Line 563: `SELECT action, COUNT(*) FROM user_feedback WHERE profile_id = ? GROUP BY action`
Line 567-570: Add `AND profile_id = ?` to the rated-items query
Line 577-582: Add `AND profile_id = ?` to the disagreements query

**(b) server.py:**
Line 500: `stats = strat.db.get_feedback_stats(profile_id=self._profile_id)`
Line 1256: `stats = strat.db.get_feedback_stats(profile_id=self._profile_id)`

---

### 1.3 `get_shadow_scores()` — NO profile_id filter [LOW]

**What:** `database.py:430-438` — Query returns shadow score comparisons from ALL profiles.

**Impact:** `/api/shadow-scores` (server.py:729) returns article titles, scores, and deltas from every profile. This is a diagnostic endpoint (not user-facing in the standard UI), but still a data leak.

**Fix — 2 files, ~4 lines:**

**(a) database.py:**
```python
def get_shadow_scores(self, limit=200, min_delta=0, profile_id=0):
    ...
    WHERE ABS(delta) >= ? AND profile_id = ?
    ORDER BY id DESC LIMIT ?
    """, (min_delta, profile_id, limit))
```

**(b) server.py:**
Line 729: `scores = strat.db.get_shadow_scores(limit=limit, min_delta=min_delta, profile_id=self._profile_id)`

---

### 1.4 `get_scan_log()` — falsy-zero profile_id check [LOW]

**What:** `database.py:390` — `if profile_id:` evaluates to `False` when `profile_id=0`, causing the unfiltered branch to execute. Profile_id=0 is the sentinel for legacy/unset profiles.

**Impact:** Any request with profile_id=0 (legacy profiles, unauthenticated background calls) returns scan logs from ALL profiles. Also affects the health endpoint (server.py:786) which calls `get_scan_log(1)` with no profile_id argument.

**Fix — 1 file, 1 line:**

**database.py line 390:**
```python
if profile_id is not None:     # was: if profile_id:
```

---

## TIER 2 — Concurrency & Thread Safety

### 2.1 Background scheduler uses last-logged-in profile [MEDIUM]

**What:** `main.py:1497-1500` — The background scheduler calls `self.run_scan()` with no `profile_id` argument. `run_scan` falls back to `self.active_profile_id` (line 320), which is set by whatever HTTP request was last processed (server.py:136, 1048). So scheduled scans always run for the last-logged-in user.

**Impact:** In a multi-user deployment, User A might get no scheduled scans while User B gets all of them, depending on who was active most recently. If no one has logged in since restart, `active_profile_id` stays at 0 (legacy).

**Fix — main.py, ~12 lines:**

Option A (simple, correct): Run background scans only if a specific profile is active. If multiple profiles need background scans, iterate over all profile_ids:

```python
def scheduler_loop():
    while not self._stop_scheduler.is_set():
        try:
            # Run for the explicitly-set profile, or skip if none
            pid = self.active_profile_id
            if pid:
                self.run_scan(profile_id=pid)
            else:
                logger.info("Background scan skipped — no active profile")
        except Exception as e:
            logger.error(f"Scheduled scan failed: {e}")
        self._stop_scheduler.wait(timeout=interval_min * 60)
```

Option B (multi-profile scheduler — only if needed):
```python
# Get all active profile IDs from DB, run sequential scans for each
cursor = self.db.conn.cursor()
cursor.execute("SELECT id FROM profiles WHERE is_active = 1")
for row in cursor.fetchall():
    self.run_scan(profile_id=row[0])
```

**Recommendation:** Option A for now. Option B is a feature request, not a bug fix.

---

### 2.2 `strat.active_profile_id` global mutation race [LOW]

**What:** `server.py:136,1048` — Every authenticated GET and POST request sets `strat.active_profile_id = _pid_row[0]`. This is a shared mutable on the singleton `strat` object, modified by every HTTP thread.

**Impact:** Low in practice because all scan/refresh entry points pass explicit `profile_id` from the HTTP handler. The only consumer of the fallback is the background scheduler (fixed in 2.1). However, it's architecturally unsound.

**Fix — server.py, 2 lines:**

Remove the global mutation. The per-request `self._profile_id` already handles data isolation. The `strat.active_profile_id` assignment exists only for backward compatibility with the background scheduler, which should be fixed independently (2.1).

```python
# Line 136 and 1048: DELETE these lines
# strat.active_profile_id = _pid_row[0]  # backward compat
```

After 2.1 is applied, nothing reads `active_profile_id` from the global anymore. Confirm by grep before removing.

---

## TIER 3 — Cleanup & DB Growth

### 3.1 `cleanup_old_data()` doesn't clean scan_log, shadow_scores, or user_feedback [MEDIUM]

**What:** `database.py:663-679` — Only cleans `news_items`, `briefings`, `market_snapshots`, and `entity_mentions`. Three tables grow unbounded:
- `scan_log` — one row per scan, ~2-3 per hour with background scheduler
- `shadow_scores` — ~10 rows per scan (sampled)
- `user_feedback` — one row per click/dismiss/rate/save

**Impact:** After months of operation, these tables accumulate tens of thousands of rows. Not a crisis, but unnecessarily bloats the DB and slows queries that lack LIMIT (like `get_feedback_stats`).

**Fix — database.py, ~6 lines added to `cleanup_old_data()`:**

```python
# After existing cleanup logic, add:
cursor.execute("DELETE FROM scan_log WHERE started_at < ?", (cutoff,))
cursor.execute("DELETE FROM shadow_scores WHERE created_at < ?", (cutoff,))
# Keep feedback longer (90 days) since it's training data
feedback_cutoff = (datetime.now() - timedelta(days=max(days, 90))).isoformat()
cursor.execute("DELETE FROM user_feedback WHERE created_at < ?", (feedback_cutoff,))
```

Note: scan_log and shadow_scores cleanup should NOT be profile-scoped (they're diagnostic data), but the feedback cleanup could be. For simplicity, just clean by age globally.

---

### 3.2 `cleanup()` on shutdown doesn't pass profile_id [LOW]

**What:** `main.py:1538` — `self.db.cleanup_old_data(days=30)` passes no `profile_id`, so the `if profile_id is not None:` branch is False and it cleans ALL profiles' news_items and briefings. This is actually correct behavior for shutdown cleanup (clean everything old), but inconsistent with the per-profile cleanup in the scan loop (line 602).

**Fix:** No change needed. The shutdown cleanup should be global. Add a comment for clarity:

```python
# Global cleanup on shutdown (not profile-scoped — clean all old data)
self.db.cleanup_old_data(days=30)
```

---

## TIER 4 — Dead Code Removal

### 4.1 Remove 10 dead DB methods

The following methods have ZERO callers outside database.py:

| Method | Lines | Why dead |
|--------|-------|----------|
| `deactivate_entity` | 277-281 | No endpoint or pipeline calls it |
| `dismiss_news` | 139-143 | Feedback uses `was_dismissed` via user_feedback table instead |
| `get_briefing_by_date` | 320-334 | No caller — briefings fetched by recency only |
| `get_latest_market` | 204-213 | Market data served from output JSON, never from DB |
| `get_market_trend` | 193-202 | Same — market trends come from yfinance, not DB |
| `get_news_by_id` | 132-137 | No endpoint calls it |
| `get_recent_mentions` | 243-252 | Entity discovery only uses `get_entity_baseline` |
| `get_recent_news` | 121-130 | News served from output JSON, never DB query |
| `is_url_seen` | 154-158 | Deduplication done via INSERT OR IGNORE |
| `mark_shown` | 160-167 | shown_to_user flag never read anywhere |
| `update_news_score` | 113-119 | Scores set via save_news_item INSERT OR IGNORE |

**Fix:** Delete all 10 methods. This removes ~90 lines of untested, unmaintained code that could accumulate profile isolation bugs if anyone ever calls them.

**Risk:** Zero — no callers. Confirm with grep before deleting.

---

## TIER 5 — Robustness & Edge Cases

### 5.1 `distill.py:save_corrections()` doesn't set profile_id [LOW]

**What:** Lines 153-195 — Inserts into `user_feedback` without a `profile_id` column in the INSERT statement. Relies on the column's DEFAULT 0 value.

**Impact:** Distillation corrections go into profile_id=0, which means:
- After fix 1.1, they won't be loaded for any non-legacy profile's scoring
- This is actually arguably correct (distillation corrections should apply to the specific profile they were generated for)

**Fix — distill.py, ~4 lines:**

Add `profile_id` parameter to `save_corrections()` and include it in the INSERT:
```python
def save_corrections(db_path, corrections, profile=None, profile_id=0):
    ...
    cursor.execute("""
        INSERT INTO user_feedback
        (news_id, title, url, ..., profile_id)
        VALUES (?, ?, ?, ..., ?)
    """, (..., profile_id))
```

Update caller (main.py `_run_auto_distillation`) to pass `profile_id`.

---

### 5.2 `export_training.py:get_corrections()` queries all profiles [LOW]

**What:** Lines 198-290 — Queries `user_feedback` without profile_id filter. All profiles' feedback is mixed into one training set.

**Impact:** Training data from different profiles (different roles/contexts) gets combined. The JOIN `ON f.profile_id = n.profile_id` (line 219, 251) prevents cross-profile news_items from joining, but feedback rows themselves are still unfiltered.

**Fix:** This is intentional for training — you want all available data. But add an optional `profile_id` parameter for profile-specific exports:
```python
def get_corrections(db_path, min_delta=1.5, after=None, profile_id=None):
    ...
    if profile_id is not None:
        time_filter += " AND f.profile_id = ?"
        time_params += (profile_id,)
```

---

### 5.3 Health endpoint leaks last scan from any profile [VERY LOW]

**What:** `server.py:786` — `scans = strat.db.get_scan_log(1)` — The health check fetches the most recent scan with no profile filter.

**Impact:** Extremely low — health endpoint only shows timing/counts, not article content. But technically exposes that *a scan happened* and how long it took.

**Fix:** Pass profile_id=0 explicitly so the falsy-zero fix (1.4) handles it consistently:
```python
scans = strat.db.get_scan_log(1, profile_id=0)  # System-level health check
```

Or leave as-is — health is a system diagnostic, not user-facing.

---

## TIER 6 — Entity Discovery Profile Scoping (DEFERRED)

### 6.1 Entity tables have no profile_id column

**What:** `entities` and `entity_mentions` tables have no `profile_id`. `get_tracked_entities()` returns ALL entities globally. `record_entity_mention()` writes without profile scoping.

**Impact:** Entity discovery cross-contaminates:
- Profile A's petroleum entities inflate entity baselines for Profile B (game developer)
- `find_rising_entities()` rising detection uses global baselines, potentially suppressing real rises for individual profiles

**Why DEFERRED:**
- Zero API surface for entities — no server.py endpoint exposes entity data to users
- Entity discoveries only feed into briefings, which ARE profile-scoped
- Adding `profile_id` to entities would require: migration (ALTER TABLE), schema change, updating `EntityDiscovery` to accept and pass profile_id, ensuring `discover()` receives it from `main.py`
- This is a feature enhancement, not a bug fix

**When to implement:** When entity discovery gets a frontend panel or API endpoint.

---

## TIER 7 — Optimizations (No Behavior Change)

### 7.1 Add missing composite index on `user_feedback(profile_id, created_at)` [OPTIMIZATION]

**What:** After fix 1.1, all three queries in `get_feedback_for_scoring()` will filter by `profile_id` AND `created_at`. The existing index `idx_feedback_profile` covers `profile_id` only. A composite index would avoid the secondary sort.

**Fix — new migration:**
```python
@migration
def migration_011(cursor):
    """Add composite index for profile-scoped feedback queries."""
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_profile_date "
        "ON user_feedback(profile_id, created_at)"
    )
```

**Impact:** Faster feedback queries during scoring runs. Negligible for small datasets, meaningful at scale.

---

### 7.2 ScoringMemory feedback cache doesn't invalidate on profile switch [OPTIMIZATION]

**What:** `scorer_base.py:595-612` — The 10-minute feedback cache (`_feedback_cache`) is keyed only by time, not by profile_id. If the scorer's profile_id changes mid-cache-window, stale feedback from the previous profile is served.

**Impact:** Very low — the scorer is rebuilt from scratch on every scan (main.py:347), which resets the cache. But if the scorer were ever reused across profile switches, this would cause stale feedback bleed.

**Fix:** After fix 1.1, the `profile_id` is baked into `ScoringMemory.__init__`, and the scorer is reconstructed per-scan with the correct profile_id. The cache naturally resets. No additional fix needed, but add a comment:

```python
# Cache resets naturally because ScoringMemory is rebuilt per-scan
# with the correct profile_id. No cross-profile stale cache possible.
```

---

### 7.3 `get_feedback_stats()` queries should use profile-scoped index [OPTIMIZATION]

**What:** After fix 1.2, the stats queries will filter by `profile_id`. The composite index from 7.1 will benefit these queries too.

**Fix:** Already covered by 7.1 — the same composite index serves both methods.

---

## Implementation Order

| Order | Fix | Files Changed | Lines Changed | Risk |
|-------|-----|--------------|---------------|------|
| 1 | 1.1 — Feedback scorer isolation | database.py, scorer_base.py, scorer_adaptive.py, main.py | ~15 | Low |
| 2 | 1.2 — Feedback stats isolation | database.py, server.py | ~8 | Very Low |
| 3 | 1.4 — Falsy-zero scan_log | database.py | 1 | Zero |
| 4 | 1.3 — Shadow scores isolation | database.py, server.py | ~4 | Very Low |
| 5 | 3.1 — Cleanup unbounded tables | database.py | ~6 | Very Low |
| 6 | 4.1 — Dead code removal | database.py | ~90 (deletion) | Zero |
| 7 | 2.1 — Background scheduler profile | main.py | ~12 | Low |
| 8 | 2.2 — Remove global mutation | server.py | 2 (deletion) | Low* |
| 9 | 7.1 — Composite index migration | migrations.py | ~5 | Zero |
| 10 | 5.1 — Distill profile_id | distill.py | ~4 | Very Low |
| 11 | 5.2 — Export training filter | export_training.py | ~4 | Very Low |

*2.2 depends on 2.1 being applied first.

**Total: ~150 lines changed across 8 files. Zero schema migrations. Zero architectural changes.**

---

## What NOT To Change (Audit Confirmed Correct)

These were flagged by the original audit but are verified non-issues:

| Item | Why it's fine |
|------|--------------|
| `market_snapshots` has no `profile_id` | Market data is objective public data. Profile isolation happens at config layer (MarketFetcher only fetches active profile's tickers). `/api/data` serves from profile-scoped output JSON. `get_market_trend` and `get_latest_market` are dead code. |
| `entities`/`entity_mentions` have no `profile_id` | No API surface. Only consumed by `discovery.py` pipeline. Output goes into briefings which ARE profile-scoped. |
| No FK from data tables to profiles | SQLite doesn't support `ALTER TABLE ADD CONSTRAINT`. Would require full table rebuilds. `profile_id` comes from authenticated session resolution, never from user input. Application-layer validation is sufficient. |
| No DB-level session guard | Already handled by per-request `self._profile_id` pattern (D030). SQLite triggers can't reference other tables effectively. |
| `export_training.py` mixes profiles | Intentional for training — you want all available corrections with their `profile_role`/`profile_location` context columns for multi-profile training sets. |
| `get_recent_news()` has no profile filter | Dead code — zero callers. |
| `update_news_score()` has no profile guard | Dead code — zero callers. |
| `is_url_seen()` has no profile filter | Dead code — zero callers. |
