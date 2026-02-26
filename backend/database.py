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

logger = logging.getLogger(__name__)


class Database:
    """SQLite database manager for STRAT_OS."""
    
    def __init__(self, db_path: str = "strat_os.db"):
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self.conn = None
        self.lock = threading.Lock()
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Performance pragmas
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")      # Write-Ahead Logging — faster concurrent reads
        self.conn.execute("PRAGMA busy_timeout = 5000")      # Wait up to 5s instead of failing immediately
        self.conn.execute("PRAGMA synchronous = NORMAL")     # Good balance of safety + speed
        self.conn.execute("PRAGMA cache_size = -8000")       # 8MB cache (default is 2MB)
        self.conn.execute("PRAGMA temp_store = MEMORY")      # Temp tables in RAM
    
    def _commit(self):
        """Thread-safe commit — prevents 'database is locked' during overlapping scans."""
        with self.lock:
            self.conn.commit()
    
    def _create_tables(self):
        """Run the migration framework to create/update all tables."""
        run_migrations(self.conn)
        logger.info("Database tables initialized")
    
    # =========================================================================
    # NEWS ITEMS
    # =========================================================================
    
    def save_news_item(self, item: Dict[str, Any]) -> bool:
        """
        Save a news item to the database.
        Returns True if new item, False if duplicate.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO news_items 
                (id, title, url, summary, source, root, category, score, score_reason, timestamp, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                datetime.now().isoformat()
            ))
            self._commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to save news item: {e}")
            return False
    
    def update_news_score(self, item_id: str, score: float, reason: str = None):
        """Update the score for a news item."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE news_items SET score = ?, score_reason = ? WHERE id = ?
        """, (score, reason, item_id))
        self._commit()
    
    def get_recent_news(self, hours: int = 24, min_score: float = 0.0) -> List[Dict]:
        """Get recent news items above minimum score."""
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        cursor.execute("""
            SELECT * FROM news_items 
            WHERE fetched_at > ? AND score >= ? AND dismissed = 0
            ORDER BY score DESC, fetched_at DESC
        """, (since, min_score))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_news_by_id(self, item_id: str) -> Optional[Dict]:
        """Get a specific news item by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM news_items WHERE id = ?", (item_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def dismiss_news(self, item_id: str):
        """Mark a news item as dismissed."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE news_items SET dismissed = 1 WHERE id = ?", (item_id,))
        self._commit()

    def was_dismissed(self, url: str) -> bool:
        """Check if a URL was dismissed by the user (via user_feedback table)."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT 1 FROM user_feedback WHERE url = ? AND action = 'dismiss' LIMIT 1",
            (url,)
        )
        return cursor.fetchone() is not None

    def is_url_seen(self, url: str) -> bool:
        """Check if we've already fetched this URL."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM news_items WHERE url = ?", (url,))
        return cursor.fetchone() is not None
    
    def mark_shown(self, item_ids: List[str]):
        """Mark items as shown to user."""
        cursor = self.conn.cursor()
        cursor.executemany(
            "UPDATE news_items SET shown_to_user = 1 WHERE id = ?",
            [(id,) for id in item_ids]
        )
        self._commit()
    
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
    
    def get_market_trend(self, symbol: str, days: int = 5) -> List[Dict]:
        """Get historical price snapshots for trend analysis."""
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute("""
            SELECT * FROM market_snapshots 
            WHERE symbol = ? AND interval = '5m' AND snapshot_at > ?
            ORDER BY snapshot_at ASC
        """, (symbol, since))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_latest_market(self, symbol: str) -> Optional[Dict]:
        """Get the most recent market snapshot for a symbol."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM market_snapshots 
            WHERE symbol = ? 
            ORDER BY snapshot_at DESC LIMIT 1
        """, (symbol,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
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
    
    def get_recent_mentions(self, entity_name: str, hours: int = 24) -> int:
        """Get total mentions in recent period."""
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        cursor.execute("""
            SELECT SUM(mention_count) as total FROM entity_mentions
            WHERE entity_name = ? AND recorded_at > ?
        """, (entity_name, since))
        row = cursor.fetchone()
        return row['total'] if row and row['total'] else 0
    
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
    
    def deactivate_entity(self, name: str):
        """Stop tracking an entity."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE entities SET is_active = 0 WHERE name = ?", (name,))
        self._commit()
    
    # =========================================================================
    # BRIEFINGS
    # =========================================================================
    
    def save_briefing(self, briefing: Dict):
        """Save a generated briefing."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO briefings (content_json, generated_at)
            VALUES (?, ?)
        """, (json.dumps(briefing), datetime.now().isoformat()))
        self._commit()
        return cursor.lastrowid
    
    def get_recent_briefings(self, limit: int = 10) -> List[Dict]:
        """Get recent briefings."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM briefings ORDER BY generated_at DESC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d['content'] = json.loads(d['content_json'])
            del d['content_json']
            result.append(d)
        return result
    
    def get_briefing_by_date(self, date_str: str) -> Optional[Dict]:
        """Get briefing for a specific date."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM briefings 
            WHERE DATE(generated_at) = DATE(?)
            ORDER BY generated_at DESC LIMIT 1
        """, (date_str,))
        row = cursor.fetchone()
        if row:
            d = dict(row)
            d['content'] = json.loads(d['content_json'])
            del d['content_json']
            return d
        return None
    
    # =========================================================================
    # SCAN LOG
    # =========================================================================
    
    def save_scan_log(self, entry: Dict[str, Any]) -> int:
        """Save a scan completion log entry. Returns the scan log row ID."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO scan_log
            (started_at, elapsed_secs, items_fetched, items_scored,
             critical, high, medium, noise, rule_scored, llm_scored, error, truncated, retained)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ))
        self._commit()
        return cursor.lastrowid
    
    def get_scan_log(self, limit: int = 50) -> List[Dict]:
        """Get recent scan log entries."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM scan_log ORDER BY id DESC LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # SHADOW SCORES
    # =========================================================================

    def save_shadow_score(self, entry: Dict[str, Any]):
        """Save a shadow scoring comparison entry."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO shadow_scores
                (scan_id, news_id, title, category, primary_scorer, primary_score,
                 shadow_scorer, shadow_score, delta, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ))
            self._commit()
        except Exception as e:
            logger.error(f"Failed to save shadow score: {e}")

    def get_shadow_scores(self, limit: int = 200, min_delta: float = 0) -> List[Dict]:
        """Get recent shadow score comparisons."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM shadow_scores
            WHERE ABS(delta) >= ?
            ORDER BY id DESC LIMIT ?
        """, (min_delta, limit))
        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # HISTORICAL QUERIES (for Agent)
    # =========================================================================
    
    def get_top_signals(self, days: int = 7, min_score: float = 7.0, limit: int = 20) -> List[Dict]:
        """Get top-scoring signals from the past N days."""
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(days=days)).isoformat()
        cursor.execute("""
            SELECT title, url, score, score_reason, category, source, fetched_at
            FROM news_items
            WHERE fetched_at > ? AND score >= ? AND dismissed = 0
            ORDER BY score DESC, fetched_at DESC
            LIMIT ?
        """, (since, min_score, limit))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_category_stats(self, days: int = 7) -> List[Dict]:
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
            WHERE fetched_at > ? AND dismissed = 0
            GROUP BY category
            ORDER BY avg_score DESC
        """, (since,))
        return [dict(row) for row in cursor.fetchall()]
    
    def search_news_history(self, keyword: str, days: int = 14, limit: int = 15) -> List[Dict]:
        """Search past news items by keyword in title or summary."""
        cursor = self.conn.cursor()
        since = (datetime.now() - timedelta(days=days)).isoformat()
        like = f"%{keyword}%"
        cursor.execute("""
            SELECT title, url, score, score_reason, category, source, fetched_at
            FROM news_items
            WHERE fetched_at > ? AND (title LIKE ? OR summary LIKE ?) AND dismissed = 0
            ORDER BY score DESC, fetched_at DESC
            LIMIT ?
        """, (since, like, like, limit))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_daily_signal_counts(self, days: int = 7) -> List[Dict]:
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
            WHERE fetched_at > ? AND dismissed = 0
            GROUP BY DATE(fetched_at)
            ORDER BY day DESC
        """, (since,))
        return [dict(row) for row in cursor.fetchall()]
    
    def save_feedback(self, feedback: Dict[str, Any]) -> bool:
        """Save user feedback (click, dismiss, rate, save, thumbs) for a news item.
        
        Profile columns (profile_role, profile_location) are stored when available
        so training data can be paired with the correct system prompt later.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO user_feedback 
                (news_id, title, url, root, category, ai_score, user_score, note, action, created_at,
                 profile_role, profile_location, profile_context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ))
            self._commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            return False
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get summary stats on user feedback."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM user_feedback")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT action, COUNT(*) FROM user_feedback GROUP BY action")
        by_action = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Average user score vs AI score for rated items
        cursor.execute("""
            SELECT AVG(ai_score), AVG(user_score), COUNT(*)
            FROM user_feedback WHERE user_score IS NOT NULL
        """)
        row = cursor.fetchone()
        avg_ai = round(row[0], 1) if row[0] else None
        avg_user = round(row[1], 1) if row[1] else None
        rated_count = row[2]
        
        # Score disagreements — items where user and AI differ by >2 points
        cursor.execute("""
            SELECT title, ai_score, user_score, category
            FROM user_feedback 
            WHERE user_score IS NOT NULL AND ABS(ai_score - user_score) > 2.0
            ORDER BY ABS(ai_score - user_score) DESC
            LIMIT 20
        """)
        disagreements = [dict(row) for row in cursor.fetchall()]
        
        return {
            'total_feedback': total,
            'by_action': by_action,
            'rated_count': rated_count,
            'avg_ai_score': avg_ai,
            'avg_user_score': avg_user,
            'top_disagreements': disagreements
        }

    def get_feedback_for_scoring(self, days: int = 30, limit: int = 20) -> Dict[str, list]:
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
            WHERE created_at > ?
              AND (
                  action = 'save'
                  OR action = 'thumbs_up'
                  OR (action = 'rate' AND user_score >= 7.0)
                  OR (action = 'click' AND ai_score >= 6.0)
              )
            ORDER BY created_at DESC
            LIMIT ?
        """, (since, limit))
        positive = [dict(row) for row in cursor.fetchall()]
        
        # Negative signals: dismissed items + low-rated items + thumbs down
        cursor.execute("""
            SELECT DISTINCT title, ai_score, user_score, category, root, action
            FROM user_feedback
            WHERE created_at > ?
              AND (
                  action = 'dismiss'
                  OR action = 'thumbs_down'
                  OR (action = 'rate' AND user_score IS NOT NULL AND user_score <= 4.0)
              )
            ORDER BY created_at DESC
            LIMIT ?
        """, (since, limit))
        negative = [dict(row) for row in cursor.fetchall()]
        
        # Corrections: significant disagreements (user score differs from AI by ≥2)
        cursor.execute("""
            SELECT title, ai_score, user_score, category, root,
                   (user_score - ai_score) as delta
            FROM user_feedback
            WHERE created_at > ?
              AND user_score IS NOT NULL
              AND ABS(ai_score - user_score) >= 2.0
            ORDER BY ABS(ai_score - user_score) DESC
            LIMIT ?
        """, (since, limit))
        corrections = [dict(row) for row in cursor.fetchall()]
        
        return {
            'positive': positive,
            'negative': negative,
            'corrections': corrections
        }
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def cleanup_old_data(self, days: int = 30):
        """Remove data older than specified days."""
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("DELETE FROM news_items WHERE fetched_at < ?", (cutoff,))
        cursor.execute("DELETE FROM market_snapshots WHERE snapshot_at < ?", (cutoff,))
        cursor.execute("DELETE FROM entity_mentions WHERE recorded_at < ?", (cutoff,))
        cursor.execute("DELETE FROM briefings WHERE generated_at < ?", (cutoff,))
        
        self._commit()
        logger.info(f"Cleaned up data older than {days} days")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# Singleton instance
_db_instance = None

def get_database(db_path: str = "strat_os.db") -> Database:
    """Get or create database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path)
    return _db_instance
