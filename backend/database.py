"""
STRAT_OS - Database Manager
Handles all persistent storage using SQLite.
"""

import sqlite3
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from migrations import run_migrations
import user_data

logger = logging.getLogger(__name__)


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
    
    def save_news_item(self, item: Dict[str, Any], profile_id: int = 0) -> bool:
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
    
    def save_scan_log(self, entry: Dict[str, Any], profile_id: int = 0) -> int:
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

    def save_shadow_score(self, entry: Dict[str, Any], profile_id: int = 0):
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

    def save_feedback(self, feedback: Dict[str, Any], profile_id: int = 0) -> bool:
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
    
    def get_feedback_stats(self, profile_id: int = 0) -> Dict[str, Any]:
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

    def close(self):
        """Close all per-thread database connections."""
        with self._conn_track_lock:
            for c in self._all_conns:
                try:
                    c.close()
                except Exception:
                    pass
            self._all_conns.clear()
        self._local.conn = None


# Singleton instance
_db_instance = None

def get_database(db_path: str = "strat_os.db") -> Database:
    """Get or create database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path)
    return _db_instance
