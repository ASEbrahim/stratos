"""
STRAT_OS - Database Manager
Handles all persistent storage using SQLite.
"""

import sqlite3
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

from migrations import run_migrations
import user_data

logger = logging.getLogger(__name__)

# Column whitelists for dynamic SQL — prevents injection via dict keys
_RP_MESSAGE_COLUMNS = frozenset({
    'model_version', 'character_card_id', 'persona', 'response_tokens',
    'response_ms', 'swipe_group_id', 'was_selected', 'director_note',
    'parent_branch_id', 'branch_point_turn', 'is_active',
})
_CHARACTER_CARD_COLUMNS = frozenset({
    'physical_description', 'speech_pattern', 'emotional_trigger',
    'defensive_mechanism', 'vulnerability', 'specific_detail',
    'personality', 'scenario', 'first_message', 'example_dialogues',
    'genre_tags', 'content_rating', 'avatar_image_path', 'is_published',
    'quality_elements_count', 'imported_from', 'tavern_card_raw',
})
_CHARACTER_CARD_UPDATE_COLUMNS = _CHARACTER_CARD_COLUMNS | frozenset({'name'})
_GENERATED_IMAGE_COLUMNS = frozenset({
    'negative_prompt', 'seed', 'steps', 'character_card_id', 'session_id',
})


class Database:
    """SQLite database manager for STRAT_OS.

    Uses per-thread connections via threading.local() to eliminate cursor
    interleaving across threads. WAL mode handles concurrent read/write
    at the SQLite level; busy_timeout (5s) handles write contention.
    """

    def __init__(self, db_path: str = "strat_os.db"):
        """Initialize database with per-thread connection pool."""
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._all_conns = []           # Track connections for cleanup
        self._conn_track_lock = threading.Lock()
        self.lock = threading.Lock()   # Kept for backward compat (_commit callers)
        self._create_tables()

    def _make_conn(self) -> sqlite3.Connection:
        """Create a new SQLite connection with standard pragmas."""
        c = sqlite3.connect(str(self.db_path))
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        c.execute("PRAGMA journal_mode = WAL")
        c.execute("PRAGMA busy_timeout = 10000")
        c.execute("PRAGMA synchronous = NORMAL")
        c.execute("PRAGMA cache_size = -8000")
        c.execute("PRAGMA temp_store = MEMORY")
        with self._conn_track_lock:
            self._all_conns.append(c)
        return c

    @property
    def conn(self) -> sqlite3.Connection:
        """Per-thread connection (backward-compatible property).

        External code accessing db.conn transparently gets the calling
        thread's own connection — no cursor interleaving possible.
        """
        c = getattr(self._local, 'conn', None)
        if c is None:
            c = self._make_conn()
            self._local.conn = c
        return c

    def _commit(self):
        """Commit the current thread's transaction.

        WAL mode + busy_timeout handle cross-thread write serialization
        at the SQLite level. The Python lock serializes commits from the
        application side to prevent concurrent write contention.
        """
        with self.lock:
            self.conn.commit()
    
    def _create_tables(self):
        """Run the migration framework to create/update all tables."""
        run_migrations(self.conn)
        logger.info("Database tables initialized")
    
    # =========================================================================
    # NEWS ITEMS
    # =========================================================================
    
    def save_news_item(self, item: Dict[str, Union[str, float, int, None]], profile_id: int = 0) -> bool:
        """
        Save a news item to the database.
        Returns True if new item, False if duplicate.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO news_items
                (id, title, url, summary, source, root, category, score, score_reason, timestamp, fetched_at, profile_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.get('id'),
                item.get('title'),
                item.get('url'),
                item.get('summary'),
                item.get('source'),
                item.get('root'),
                item.get('category'),
                item.get('score', 0.0),
                item.get('score_reason'),
                item.get('timestamp'),
                datetime.now().isoformat(),
                profile_id,
            ))
            self._commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to save news item: {e}")
            return False
    
    def was_dismissed(self, url: str, profile_id: int = 0) -> bool:
        # Note: Dismissal is tracked via user_feedback (action='dismiss'), NOT the
        # news_items.dismissed column. That column and its writer (dismiss_news) are
        # both dead — no live code path reads or writes news_items.dismissed.
        """Check if a URL was dismissed by the user (via user_feedback table)."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT 1 FROM user_feedback WHERE url = ? AND action = 'dismiss' AND profile_id = ? LIMIT 1",
            (url, profile_id)
        )
        return cursor.fetchone() is not None

    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    def save_market_snapshot(self, symbol: str, name: str, interval: str, data: Dict):
        """Save a market data snapshot."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO market_snapshots 
            (symbol, name, interval, price, change_percent, high, low, history_json, snapshot_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            name,
            interval,
            data.get('price'),
            data.get('change'),
            data.get('high'),
            data.get('low'),
            json.dumps(data.get('history', [])),
            datetime.now().isoformat()
        ))
        self._commit()
    
    # =========================================================================
    # ENTITY DISCOVERY
    # =========================================================================
    
    def record_entity_mention(self, entity_name: str, count: int = 1):
        """Record mentions of an entity for discovery tracking."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO entity_mentions (entity_name, mention_count, recorded_at)
            VALUES (?, ?, ?)
        """, (entity_name, count, datetime.now().isoformat()))
        self._commit()
    
    def get_entity_baseline(self, entity_name: str, hours: int = 168) -> float:
        """Get average mentions per day for an entity over baseline period."""
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        cursor.execute("""
            SELECT AVG(daily_total) as avg_mentions FROM (
                SELECT DATE(recorded_at) as day, SUM(mention_count) as daily_total
                FROM entity_mentions
                WHERE entity_name = ? AND recorded_at > ?
                GROUP BY DATE(recorded_at)
            )
        """, (entity_name, since))
        row = cursor.fetchone()
        return row['avg_mentions'] if row and row['avg_mentions'] else 0.0
    
    def add_tracked_entity(self, name: str, category: str, is_discovered: bool = False):
        """Add a new entity to track."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO entities (name, category, is_discovered, discovered_at)
                VALUES (?, ?, ?, ?)
            """, (name, category, 1 if is_discovered else 0, datetime.now().isoformat() if is_discovered else None))
            self._commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to add entity: {e}")
            return False
    
    def get_tracked_entities(self, active_only: bool = True) -> List[Dict]:
        """Get all tracked entities."""
        cursor = self.conn.cursor()
        if active_only:
            cursor.execute("SELECT * FROM entities WHERE is_active = 1")
        else:
            cursor.execute("SELECT * FROM entities")
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # BRIEFINGS
    # =========================================================================
    
    def save_briefing(self, briefing: Dict, profile_id: int = 0):
        """Save a generated briefing."""
        cursor = self.conn.cursor()
        now = datetime.now()
        cursor.execute("""
            INSERT INTO briefings (content_json, generated_at, profile_id)
            VALUES (?, ?, ?)
        """, (json.dumps(briefing), now.isoformat(), profile_id))
        self._commit()
        briefing_id = cursor.lastrowid

        # Per-user daily briefing export
        uid = user_data.get_user_id_for_profile(self, profile_id)
        if uid > 0:
            user_data.write_json(uid, f"briefings/{now.strftime('%Y-%m-%d')}.json", briefing)

        return briefing_id
    
    def get_recent_briefings(self, limit: int = 10, profile_id: int = 0) -> List[Dict]:
        """Get recent briefings for a specific profile."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM briefings WHERE profile_id = ? ORDER BY generated_at DESC LIMIT ?
        """, (profile_id, limit))
        rows = cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d['content'] = json.loads(d['content_json'])
            del d['content_json']
            result.append(d)
        return result

    # =========================================================================
    # SCAN LOG
    # =========================================================================
    
    def save_scan_log(self, entry: Dict[str, Union[str, int, float, None]], profile_id: int = 0) -> int:
        """Save a scan completion log entry. Returns the scan log row ID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO scan_log
            (started_at, elapsed_secs, items_fetched, items_scored,
             critical, high, medium, noise, rule_scored, llm_scored, error, truncated, retained, profile_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.get('started_at', datetime.now().isoformat()),
            entry.get('elapsed_secs', 0),
            entry.get('items_fetched', 0),
            entry.get('items_scored', 0),
            entry.get('critical', 0),
            entry.get('high', 0),
            entry.get('medium', 0),
            entry.get('noise', 0),
            entry.get('rule_scored', 0),
            entry.get('llm_scored', 0),
            entry.get('error'),
            entry.get('truncated', 0),
            entry.get('retained', 0),
            profile_id,
        ))
        self._commit()
        scan_id = cursor.lastrowid

        # Per-user JSONL export
        uid = user_data.get_user_id_for_profile(self, profile_id)
        if uid > 0:
            user_data.append_jsonl(uid, "scan_log.jsonl", {
                "scan_id": scan_id,
                "timestamp": entry.get('started_at', datetime.now().isoformat()),
                "elapsed_secs": entry.get('elapsed_secs', 0),
                "items_fetched": entry.get('items_fetched', 0),
                "items_scored": entry.get('items_scored', 0),
                "critical": entry.get('critical', 0),
                "high": entry.get('high', 0),
                "medium": entry.get('medium', 0),
                "noise": entry.get('noise', 0),
                "rule_scored": entry.get('rule_scored', 0),
                "llm_scored": entry.get('llm_scored', 0),
                "retained": entry.get('retained', 0),
            })

        return scan_id

    def get_scan_log(self, limit: int = 50, profile_id: int = 0) -> List[Dict]:
        """Get recent scan log entries, optionally filtered by profile_id."""
        cursor = self.conn.cursor()
        if profile_id is not None:
            cursor.execute("""
                SELECT * FROM scan_log WHERE profile_id = ? ORDER BY id DESC LIMIT ?
            """, (profile_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM scan_log ORDER BY id DESC LIMIT ?
            """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # SHADOW SCORES
    # =========================================================================

    def save_shadow_score(self, entry: Dict[str, Union[str, int, float, None]], profile_id: int = 0):
        """Save a shadow scoring comparison entry."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO shadow_scores
                (scan_id, news_id, title, category, primary_scorer, primary_score,
                 shadow_scorer, shadow_score, delta, created_at, profile_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.get('scan_id'),
                entry.get('news_id', ''),
                entry.get('title', ''),
                entry.get('category', ''),
                entry.get('primary_scorer', ''),
                entry.get('primary_score', 0),
                entry.get('shadow_scorer', ''),
                entry.get('shadow_score', 0),
                entry.get('delta', 0),
                datetime.now().isoformat(),
                profile_id,
            ))
            self._commit()
        except Exception as e:
            logger.error(f"Failed to save shadow score: {e}")

    def get_shadow_scores(self, limit: int = 200, min_delta: float = 0, profile_id: int = 0) -> List[Dict]:
        """Get recent shadow score comparisons for a specific profile."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM shadow_scores
            WHERE ABS(delta) >= ? AND profile_id = ?
            ORDER BY id DESC LIMIT ?
        """, (min_delta, profile_id, limit))
        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # HISTORICAL QUERIES (for Agent)
    # =========================================================================
    
    def get_top_signals(self, days: int = 7, min_score: float = 7.0, limit: int = 20, profile_id: int = 0) -> List[Dict]:
        """Get top-scoring signals from the past N days."""
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute("""
            SELECT title, url, score, score_reason, category, source, fetched_at
            FROM news_items
            WHERE fetched_at > ? AND score >= ? AND dismissed = 0 AND profile_id = ?
            ORDER BY score DESC, fetched_at DESC
            LIMIT ?
        """, (since, min_score, profile_id, limit))
        return [dict(row) for row in cursor.fetchall()]

    def get_category_stats(self, days: int = 7, profile_id: int = 0) -> List[Dict]:
        """Get scoring stats per category over the past N days."""
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute("""
            SELECT category,
                   COUNT(*) as total,
                   ROUND(AVG(score), 1) as avg_score,
                   SUM(CASE WHEN score >= 9.0 THEN 1 ELSE 0 END) as critical,
                   SUM(CASE WHEN score >= 7.0 AND score < 9.0 THEN 1 ELSE 0 END) as high,
                   MAX(score) as best_score
            FROM news_items
            WHERE fetched_at > ? AND dismissed = 0 AND profile_id = ?
            GROUP BY category
            ORDER BY avg_score DESC
        """, (since, profile_id))
        return [dict(row) for row in cursor.fetchall()]

    def search_news_history(self, keyword: str, days: int = 14, limit: int = 15, profile_id: int = 0) -> List[Dict]:
        """Search past news items by keyword in title or summary."""
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(days=days)).isoformat()
        like = f"%{keyword}%"
        cursor.execute("""
            SELECT title, url, score, score_reason, category, source, fetched_at
            FROM news_items
            WHERE fetched_at > ? AND (title LIKE ? OR summary LIKE ?) AND dismissed = 0 AND profile_id = ?
            ORDER BY score DESC, fetched_at DESC
            LIMIT ?
        """, (since, like, like, profile_id, limit))
        return [dict(row) for row in cursor.fetchall()]

    def get_daily_signal_counts(self, days: int = 7, profile_id: int = 0) -> List[Dict]:
        """Get daily counts of signals by score tier."""
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute("""
            SELECT DATE(fetched_at) as day,
                   COUNT(*) as total,
                   SUM(CASE WHEN score >= 9.0 THEN 1 ELSE 0 END) as critical,
                   SUM(CASE WHEN score >= 7.0 AND score < 9.0 THEN 1 ELSE 0 END) as high,
                   SUM(CASE WHEN score > 5.0 AND score < 7.0 THEN 1 ELSE 0 END) as medium
            FROM news_items
            WHERE fetched_at > ? AND dismissed = 0 AND profile_id = ?
            GROUP BY DATE(fetched_at)
            ORDER BY day DESC
        """, (since, profile_id))
        return [dict(row) for row in cursor.fetchall()]

    def save_feedback(self, feedback: Dict[str, Union[str, int, float, None]], profile_id: int = 0) -> bool:
        """Save user feedback (click, dismiss, rate, save, thumbs) for a news item.

        Profile columns (profile_role, profile_location) are stored when available
        so training data can be paired with the correct system prompt later.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO user_feedback
                (news_id, title, url, root, category, ai_score, user_score, note, action, created_at,
                 profile_role, profile_location, profile_context, profile_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.get('news_id', ''),
                feedback.get('title', ''),
                feedback.get('url', ''),
                feedback.get('root', ''),
                feedback.get('category', ''),
                feedback.get('ai_score'),
                feedback.get('user_score'),
                feedback.get('note', ''),
                feedback.get('action', 'click'),
                datetime.now().isoformat(),
                feedback.get('profile_role', ''),
                feedback.get('profile_location', ''),
                feedback.get('profile_context', ''),
                profile_id,
            ))
            self._commit()

            # Per-user JSONL export
            uid = user_data.get_user_id_for_profile(self, profile_id)
            if uid > 0:
                user_data.append_jsonl(uid, "feedback.jsonl", {
                    "timestamp": datetime.now().isoformat(),
                    "action": feedback.get('action', 'click'),
                    "title": feedback.get('title', ''),
                    "url": feedback.get('url', ''),
                    "category": feedback.get('category', ''),
                    "ai_score": feedback.get('ai_score'),
                    "user_score": feedback.get('user_score'),
                    "note": feedback.get('note', ''),
                })

            return True
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return False
    
    def get_feedback_stats(self, profile_id: int = 0) -> Dict[str, Union[int, float, dict, list, None]]:
        """Get summary stats on user feedback for a specific profile."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM user_feedback WHERE profile_id = ?", (profile_id,))
        total = cursor.fetchone()[0]

        cursor.execute("SELECT action, COUNT(*) FROM user_feedback WHERE profile_id = ? GROUP BY action", (profile_id,))
        by_action = {row[0]: row[1] for row in cursor.fetchall()}

        # Average user score vs AI score for rated items
        cursor.execute("""
            SELECT AVG(ai_score), AVG(user_score), COUNT(*)
            FROM user_feedback WHERE profile_id = ? AND user_score IS NOT NULL
        """, (profile_id,))
        row = cursor.fetchone()
        avg_ai = round(row[0], 1) if row[0] else None
        avg_user = round(row[1], 1) if row[1] else None
        rated_count = row[2]

        # Score disagreements — items where user and AI differ by >2 points
        cursor.execute("""
            SELECT title, ai_score, user_score, category
            FROM user_feedback
            WHERE profile_id = ? AND user_score IS NOT NULL AND ABS(ai_score - user_score) > 2.0
            ORDER BY ABS(ai_score - user_score) DESC
            LIMIT 20
        """, (profile_id,))
        disagreements = [dict(row) for row in cursor.fetchall()]

        return {
            'total_feedback': total,
            'by_action': by_action,
            'rated_count': rated_count,
            'avg_ai_score': avg_ai,
            'avg_user_score': avg_user,
            'top_disagreements': disagreements
        }

    def get_feedback_for_scoring(self, days: int = 30, limit: int = 20, profile_id: int = 0) -> Dict[str, list]:
        """
        Retrieve user feedback organized for scorer prompt injection.
        
        Returns:
            {
                'positive': [items user engaged with positively],
                'negative': [items user dismissed or rated poorly],
                'corrections': [items where user and AI disagreed significantly]
            }
        """
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Positive signals: saved items + highly rated items + clicks on high-scored + thumbs up
        cursor.execute("""
            SELECT DISTINCT title, ai_score, user_score, category, root, action
            FROM user_feedback
            WHERE created_at > ? AND profile_id = ?
              AND (
                  action = 'save'
                  OR action = 'thumbs_up'
                  OR (action = 'rate' AND user_score >= 7.0)
                  OR (action = 'click' AND ai_score >= 6.0)
              )
            ORDER BY created_at DESC
            LIMIT ?
        """, (since, profile_id, limit))
        positive = [dict(row) for row in cursor.fetchall()]
        
        # Negative signals: dismissed items + low-rated items + thumbs down
        cursor.execute("""
            SELECT DISTINCT title, ai_score, user_score, category, root, action
            FROM user_feedback
            WHERE created_at > ? AND profile_id = ?
              AND (
                  action = 'dismiss'
                  OR action = 'thumbs_down'
                  OR (action = 'rate' AND user_score IS NOT NULL AND user_score <= 4.0)
              )
            ORDER BY created_at DESC
            LIMIT ?
        """, (since, profile_id, limit))
        negative = [dict(row) for row in cursor.fetchall()]
        
        # Corrections: significant disagreements (user score differs from AI by ≥2)
        cursor.execute("""
            SELECT title, ai_score, user_score, category, root,
                   (user_score - ai_score) as delta
            FROM user_feedback
            WHERE created_at > ? AND profile_id = ?
              AND user_score IS NOT NULL
              AND ABS(ai_score - user_score) >= 2.0
            ORDER BY ABS(ai_score - user_score) DESC
            LIMIT ?
        """, (since, profile_id, limit))
        corrections = [dict(row) for row in cursor.fetchall()]
        
        return {
            'positive': positive,
            'negative': negative,
            'corrections': corrections
        }
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def cleanup_old_data(self, days: int = 30, profile_id: int = None):
        """Remove data older than specified days. If profile_id given, only that profile's data."""
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        if profile_id is not None:
            cursor.execute("DELETE FROM news_items WHERE fetched_at < ? AND profile_id = ?", (cutoff, profile_id))
            cursor.execute("DELETE FROM briefings WHERE generated_at < ? AND profile_id = ?", (cutoff, profile_id))
        else:
            cursor.execute("DELETE FROM news_items WHERE fetched_at < ?", (cutoff,))
            cursor.execute("DELETE FROM briefings WHERE generated_at < ?", (cutoff,))
        # Market snapshots and entity mentions are global — always clean by age
        cursor.execute("DELETE FROM market_snapshots WHERE snapshot_at < ?", (cutoff,))
        cursor.execute("DELETE FROM entity_mentions WHERE recorded_at < ?", (cutoff,))

        # Scan log and shadow scores — keep at least 90 days for diagnostic history
        diag_cutoff = (datetime.now() - timedelta(days=max(days, 90))).isoformat()
        cursor.execute("DELETE FROM scan_log WHERE started_at < ?", (diag_cutoff,))
        cursor.execute("DELETE FROM shadow_scores WHERE created_at < ?", (diag_cutoff,))
        # Note: user_feedback is NOT cleaned — it's training data that doesn't expire

        self._commit()
        logger.info(f"Cleaned up data older than {days} days (scan_log/shadow_scores: {max(days, 90)} days)")
    
    def get_ui_state(self, profile_id: int) -> dict:
        """Return parsed ui_state dict for a profile, or {} if missing/invalid."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT ui_state FROM profiles WHERE id = ?", (profile_id,))
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to read ui_state for profile {profile_id}: {e}")
        return {}

    def save_ui_state(self, profile_id: int, partial_state: dict):
        """Atomic merge-update: reads existing ui_state, deep-merges new keys, writes back."""
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute("SELECT ui_state FROM profiles WHERE id = ?", (profile_id,))
                row = cursor.fetchone()
                existing = {}
                if row and row[0]:
                    try:
                        existing = json.loads(row[0])
                    except json.JSONDecodeError:
                        pass
                # Deep merge: for dict-valued keys (like ui_settings), merge nested dicts
                for k, v in partial_state.items():
                    if isinstance(v, dict) and isinstance(existing.get(k), dict):
                        existing[k].update(v)
                    else:
                        existing[k] = v
                cursor.execute("UPDATE profiles SET ui_state = ? WHERE id = ?",
                               (json.dumps(existing), profile_id))
        except Exception as e:
            logger.error(f"Failed to save ui_state for profile {profile_id}: {e}")

    # =========================================================================
    # RP MESSAGES & BRANCHING
    # =========================================================================

    MAX_BRANCH_DEPTH = 15

    def insert_rp_message(self, session_id: str, profile_id: int, branch_id: str,
                          turn_number: int, role: str, content: str, **kwargs) -> int:
        """Insert an RP message and return its ID. Retries on DB lock."""
        cols = ['session_id', 'profile_id', 'branch_id', 'turn_number', 'role', 'content']
        vals: list = [session_id, profile_id, branch_id, turn_number, role, content]
        for k in _RP_MESSAGE_COLUMNS:
            if k in kwargs:
                cols.append(k)
                vals.append(kwargs[k])
        placeholders = ','.join(['?'] * len(vals))
        col_str = ','.join(cols)
        import time
        for attempt in range(3):
            try:
                cursor = self.conn.execute(
                    f"INSERT INTO rp_messages ({col_str}) VALUES ({placeholders})", vals
                )
                self._commit()
                return cursor.lastrowid
            except Exception as e:
                if 'locked' in str(e) and attempt < 2:
                    time.sleep(0.5)
                    continue
                raise

    def get_full_branch_conversation(self, session_id: str, branch_id: str = 'main') -> list:
        """Reconstruct full conversation by walking the parent branch chain.

        Branches store only their new messages after the branch point.
        Parent messages are fetched recursively up the chain.
        Enforces MAX_BRANCH_DEPTH to prevent runaway queries.
        """
        messages = []
        current_branch = branch_id
        depth = 0

        while current_branch and depth < self.MAX_BRANCH_DEPTH:
            # Get branch metadata
            branch_meta = self.conn.execute(
                "SELECT DISTINCT parent_branch_id, branch_point_turn FROM rp_messages "
                "WHERE session_id = ? AND branch_id = ? LIMIT 1",
                (session_id, current_branch)
            ).fetchone()

            if current_branch == 'main' or not branch_meta or not branch_meta['parent_branch_id']:
                # Root branch — get all messages
                rows = self.conn.execute(
                    "SELECT * FROM rp_messages WHERE session_id = ? AND branch_id = ? "
                    "AND was_selected = TRUE ORDER BY turn_number",
                    (session_id, current_branch)
                ).fetchall()
                messages = [dict(r) for r in rows] + messages
                break
            else:
                # Child branch — get messages after branch point
                branch_point = branch_meta['branch_point_turn'] or 0
                rows = self.conn.execute(
                    "SELECT * FROM rp_messages WHERE session_id = ? AND branch_id = ? "
                    "AND turn_number > ? AND was_selected = TRUE ORDER BY turn_number",
                    (session_id, current_branch, branch_point)
                ).fetchall()
                messages = [dict(r) for r in rows] + messages
                current_branch = branch_meta['parent_branch_id']
                depth += 1

        return messages

    def get_rp_branches(self, session_id: str) -> list:
        """List all branches for a session with metadata."""
        rows = self.conn.execute("""
            SELECT branch_id, parent_branch_id, branch_point_turn,
                   COUNT(*) as turn_count,
                   MIN(created_at) as created_at,
                   MAX(is_active) as is_active
            FROM rp_messages
            WHERE session_id = ? AND was_selected = TRUE
            GROUP BY branch_id
            ORDER BY MIN(created_at)
        """, (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def create_branch(self, session_id: str, from_branch: str, at_turn: int,
                      new_branch_id: str) -> dict:
        """Create a new branch by reference. Does NOT copy parent messages.

        Returns: {"branch_id": str} or {"error": str}
        """
        # Check depth
        depth = 0
        check_branch = from_branch
        while check_branch and check_branch != 'main' and depth < self.MAX_BRANCH_DEPTH + 1:
            row = self.conn.execute(
                "SELECT DISTINCT parent_branch_id FROM rp_messages "
                "WHERE session_id = ? AND branch_id = ? LIMIT 1",
                (session_id, check_branch)
            ).fetchone()
            if not row or not row['parent_branch_id']:
                break
            check_branch = row['parent_branch_id']
            depth += 1

        if depth >= self.MAX_BRANCH_DEPTH:
            return {"error": f"Maximum branch depth ({self.MAX_BRANCH_DEPTH}) exceeded"}

        return {"branch_id": new_branch_id}

    # =========================================================================
    # RP EDITS
    # =========================================================================

    def insert_rp_edit(self, message_id: int, session_id: str, original: str,
                       edited: str, category: str = None, reason: str = None,
                       card_id: str = None):
        """Record an edit on an AI response (DPO training pair)."""
        self.conn.execute(
            "INSERT INTO rp_edits (message_id, session_id, original_content, edited_content, "
            "edit_delta_category, edit_reason, character_card_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (message_id, session_id, original, edited, category, reason, card_id)
        )
        # Update the message content to the edited version
        self.conn.execute(
            "UPDATE rp_messages SET content = ? WHERE id = ?", (edited, message_id)
        )
        self._commit()

    def get_edits_for_session(self, session_id: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM rp_edits WHERE session_id = ? ORDER BY created_at", (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # =========================================================================
    # RP FEEDBACK
    # =========================================================================

    def insert_rp_feedback(self, message_id: int, profile_id: int, feedback_type: str):
        """Record thumbs up/down. Idempotent (UNIQUE constraint)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO rp_feedback (message_id, profile_id, feedback_type) "
            "VALUES (?, ?, ?)", (message_id, profile_id, feedback_type)
        )
        self._commit()

    def get_feedback_for_session(self, session_id: str) -> dict:
        """Returns: {"thumbs_up": int, "thumbs_down": int}"""
        rows = self.conn.execute("""
            SELECT f.feedback_type, COUNT(*) as cnt
            FROM rp_feedback f JOIN rp_messages m ON f.message_id = m.id
            WHERE m.session_id = ?
            GROUP BY f.feedback_type
        """, (session_id,)).fetchall()
        result = {"thumbs_up": 0, "thumbs_down": 0}
        for r in rows:
            if r['feedback_type'] in result:
                result[r['feedback_type']] = r['cnt']
        return result

    # =========================================================================
    # CHARACTER CARDS
    # =========================================================================

    def insert_character_card(self, card_id: str, creator_id: int, name: str, **fields):
        """Create a new character card. Retries on DB lock."""
        cols = ['id', 'creator_profile_id', 'name']
        vals: list = [card_id, creator_id, name]
        for k in _CHARACTER_CARD_COLUMNS:
            if k in fields:
                cols.append(k)
                vals.append(fields[k])
        placeholders = ','.join(['?'] * len(vals))
        col_str = ','.join(cols)
        import time as _time
        for _attempt in range(3):
            try:
                self.conn.execute(f"INSERT INTO character_cards ({col_str}) VALUES ({placeholders})", vals)
                self._commit()
                return
            except Exception as e:
                if 'locked' in str(e) and _attempt < 2:
                    _time.sleep(0.5)
                    continue
                raise

    def get_character_card(self, card_id: str) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM character_cards WHERE id = ?", (card_id,)).fetchone()
        return dict(row) if row else None

    def get_published_cards(self, genre: str = None, sort: str = 'trending',
                            limit: int = 20, offset: int = 0) -> list:
        query = "SELECT c.*, COALESCE(s.total_sessions, 0) as sessions, COALESCE(s.avg_rating, 0) as avg_rating_val, " \
                "COALESCE(p.name, u.display_name, 'Unknown') as creator_name " \
                "FROM character_cards c LEFT JOIN character_card_stats s ON c.id = s.card_id " \
                "LEFT JOIN profiles p ON c.creator_profile_id = p.id " \
                "LEFT JOIN users u ON p.user_id = u.id " \
                "WHERE c.is_published = TRUE"
        params = []
        if genre:
            query += " AND c.genre_tags LIKE ?"
            params.append(f'%{genre}%')
        if sort == 'trending':
            query += " ORDER BY COALESCE(s.total_sessions, 0) DESC"
        elif sort == 'newest':
            query += " ORDER BY c.created_at DESC"
        elif sort == 'top_rated':
            query += " ORDER BY COALESCE(s.avg_rating, 0) DESC"
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def update_character_card(self, card_id: str, **fields):
        if not fields:
            return
        sets = []
        vals: list = []
        for k, v in fields.items():
            if k not in _CHARACTER_CARD_UPDATE_COLUMNS:
                logger.warning(f"update_character_card: ignoring unknown column '{k}'")
                continue
            sets.append(f"{k} = ?")
            vals.append(v)
        if not sets:
            return
        sets.append("updated_at = CURRENT_TIMESTAMP")
        vals.append(card_id)
        self.conn.execute(f"UPDATE character_cards SET {','.join(sets)} WHERE id = ?", vals)
        self._commit()

    def publish_character_card(self, card_id: str):
        self.conn.execute("UPDATE character_cards SET is_published = TRUE, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (card_id,))
        self._commit()

    def rate_character_card(self, card_id: str, profile_id: int, rating: int):
        self.conn.execute(
            "INSERT OR REPLACE INTO character_card_ratings (card_id, profile_id, rating) VALUES (?, ?, ?)",
            (card_id, profile_id, rating)
        )
        # Update avg rating in stats
        self.conn.execute("""
            INSERT INTO character_card_stats (card_id, total_ratings, avg_rating)
            VALUES (?, 1, ?)
            ON CONFLICT(card_id) DO UPDATE SET
                total_ratings = (SELECT COUNT(*) FROM character_card_ratings WHERE card_id = ?),
                avg_rating = (SELECT AVG(rating) FROM character_card_ratings WHERE card_id = ?),
                updated_at = CURRENT_TIMESTAMP
        """, (card_id, rating, card_id, card_id))
        self._commit()

    # =========================================================================
    # GENERATED IMAGES
    # =========================================================================

    def insert_generated_image(self, image_id: str, profile_id: int, prompt: str,
                               model: str, width: int, height: int, filename: str,
                               filepath: str, **kwargs):
        cols = ['id', 'profile_id', 'prompt', 'model', 'width', 'height', 'filename', 'file_path']
        vals: list = [image_id, profile_id, prompt, model, width, height, filename, filepath]
        for k in _GENERATED_IMAGE_COLUMNS:
            if k in kwargs:
                cols.append(k)
                vals.append(kwargs[k])
        placeholders = ','.join(['?'] * len(vals))
        col_str = ','.join(cols)
        self.conn.execute(f"INSERT INTO generated_images ({col_str}) VALUES ({placeholders})", vals)
        self._commit()

    def get_user_images(self, profile_id: int, limit: int = 20) -> list:
        rows = self.conn.execute(
            "SELECT * FROM generated_images WHERE profile_id = ? ORDER BY created_at DESC LIMIT ?",
            (profile_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    # =========================================================================
    # RP SESSION CONTEXT (Tiered Memory)
    # =========================================================================

    def upsert_rp_context(self, session_id: str, tier: int, category: str,
                          key: str, value: str, turn_number: int = 0):
        """Insert or update a context entry. Upserts on (session_id, tier, category, key)."""
        existing = self.conn.execute(
            "SELECT id FROM rp_session_context WHERE session_id = ? AND tier = ? AND category = ? AND key = ?",
            (session_id, tier, category, key)
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE rp_session_context SET value = ?, turn_number = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (value, turn_number, existing['id'])
            )
        else:
            self.conn.execute(
                "INSERT INTO rp_session_context (session_id, tier, category, key, value, turn_number) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, tier, category, key, value, turn_number)
            )
        self._commit()

    def insert_rp_context(self, session_id: str, tier: int, category: str,
                          key: str, value: str, turn_number: int = 0):
        """Insert a new context entry (no upsert)."""
        self.conn.execute(
            "INSERT INTO rp_session_context (session_id, tier, category, key, value, turn_number) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, tier, category, key, value, turn_number)
        )
        self._commit()

    def get_rp_context(self, session_id: str, tier: int = None,
                       category: str = None, limit: int = 50) -> list:
        """Get context entries for a session, optionally filtered by tier/category."""
        query = "SELECT * FROM rp_session_context WHERE session_id = ?"
        params: list = [session_id]
        if tier is not None:
            query += " AND tier = ?"
            params.append(tier)
        if category is not None:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(query, params).fetchall()]

    def delete_rp_context(self, session_id: str, tier: int = None):
        """Delete context entries for a session, optionally filtered by tier."""
        if tier is not None:
            self.conn.execute(
                "DELETE FROM rp_session_context WHERE session_id = ? AND tier = ?",
                (session_id, tier)
            )
        else:
            self.conn.execute(
                "DELETE FROM rp_session_context WHERE session_id = ?", (session_id,)
            )
        self._commit()

    def get_rp_context_for_character(self, character_card_id: str, tier: int = 1,
                                      limit: int = 30) -> list:
        """Get context across ALL sessions for a character (cross-session memory)."""
        rows = self.conn.execute(
            """SELECT rc.* FROM rp_session_context rc
               JOIN rp_messages rm ON rc.session_id = rm.session_id
               WHERE rm.character_card_id = ? AND rc.tier = ?
               GROUP BY rc.category, rc.key
               ORDER BY rc.updated_at DESC LIMIT ?""",
            (character_card_id, tier, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    # =========================================================================
    # PRIVACY / CONSENT
    # =========================================================================

    def set_training_opt_in(self, profile_id: int, opted_in: bool):
        self.conn.execute(
            "UPDATE profiles SET training_data_opt_in = ? WHERE id = ?",
            (opted_in, profile_id)
        )
        self._commit()

    def get_training_opt_in(self, profile_id: int) -> bool:
        row = self.conn.execute(
            "SELECT training_data_opt_in FROM profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        return bool(row['training_data_opt_in']) if row else False

    def close(self):
        """Close all per-thread database connections."""
        with self._conn_track_lock:
            for c in self._all_conns:
                try:
                    c.close()
                except Exception as e:
                    logger.error(f"Error closing database connection: {e}")
            self._all_conns.clear()
        self._local.conn = None

    def cleanup(self):
        """Alias for close() — ensures all tracked connections are released."""
        self.close()

    def __del__(self):
        """Ensure connections are closed when the Database object is garbage collected."""
        try:
            self.close()
        except Exception:
            pass  # __del__ must never raise


# Singleton instance
_db_instance = None
_db_instance_lock = threading.Lock()

def get_database(db_path: str = "strat_os.db") -> Database:
    """Get or create database instance (thread-safe singleton)."""
    global _db_instance
    if _db_instance is None:
        with _db_instance_lock:
            if _db_instance is None:
                _db_instance = Database(db_path)
    return _db_instance
