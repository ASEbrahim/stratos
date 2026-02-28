"""
STRAT_OS — Lightweight DB Migration Framework (A1.4)

Numbered migrations replace scattered ALTER TABLE try/excepts.
Each migration runs exactly once; progress tracked in schema_version table.
"""

import sqlite3
import logging

logger = logging.getLogger(__name__)

# =========================================================================
# Migration registry — append new migrations at the end, never reorder
# =========================================================================

MIGRATIONS = []


def migration(func):
    """Decorator that registers a migration function."""
    MIGRATIONS.append(func)
    return func


# -- Migration 001: baseline tables --
@migration
def migration_001(cursor):
    """Create core tables (news_items, market_snapshots, entities, entity_mentions, scan_log, briefings)."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_items (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT UNIQUE NOT NULL,
            summary TEXT,
            source TEXT,
            root TEXT,
            category TEXT,
            score REAL DEFAULT 0.0,
            score_reason TEXT,
            timestamp TEXT,
            fetched_at TEXT NOT NULL,
            shown_to_user INTEGER DEFAULT 0,
            dismissed INTEGER DEFAULT 0
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            name TEXT,
            interval TEXT NOT NULL,
            price REAL,
            change_percent REAL,
            high REAL,
            low REAL,
            history_json TEXT,
            snapshot_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            category TEXT,
            is_discovered INTEGER DEFAULT 0,
            discovered_at TEXT,
            is_active INTEGER DEFAULT 1,
            mention_count INTEGER DEFAULT 0
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entity_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT NOT NULL,
            mention_count INTEGER DEFAULT 1,
            recorded_at TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            elapsed_secs REAL,
            items_fetched INTEGER DEFAULT 0,
            items_scored INTEGER DEFAULT 0,
            critical INTEGER DEFAULT 0,
            high INTEGER DEFAULT 0,
            medium INTEGER DEFAULT 0,
            noise INTEGER DEFAULT 0,
            rule_scored INTEGER DEFAULT 0,
            llm_scored INTEGER DEFAULT 0,
            error TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS briefings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_json TEXT NOT NULL,
            generated_at TEXT NOT NULL
        )
    """)


# -- Migration 002: user_feedback table --
@migration
def migration_002(cursor):
    """Create user_feedback table with note column."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_id TEXT NOT NULL,
            title TEXT,
            url TEXT,
            root TEXT,
            category TEXT,
            ai_score REAL,
            user_score REAL,
            note TEXT,
            action TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)


# -- Migration 003: indexes --
@migration
def migration_003(cursor):
    """Create performance indexes."""
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_fetched ON news_items(fetched_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_score ON news_items(score)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_symbol ON market_snapshots(symbol, snapshot_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mentions_date ON entity_mentions(recorded_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_news ON user_feedback(news_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_date ON user_feedback(created_at)")


# -- Migration 004: user_feedback profile columns --
@migration
def migration_004(cursor):
    """Add profile columns to user_feedback for training data provenance."""
    for col in ["profile_role TEXT", "profile_location TEXT", "profile_context TEXT"]:
        try:
            cursor.execute(f"ALTER TABLE user_feedback ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # Column already exists


# -- Migration 005: scan_log truncated column --
@migration
def migration_005(cursor):
    """Add truncated column to scan_log for Ollama stream error tracking."""
    try:
        cursor.execute("ALTER TABLE scan_log ADD COLUMN truncated INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists


# -- Migration 006: shadow_scores table --
@migration
def migration_006(cursor):
    """Create shadow_scores table for scorer validation."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS shadow_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER,
            news_id TEXT,
            title TEXT,
            category TEXT,
            primary_scorer TEXT,
            primary_score REAL,
            shadow_scorer TEXT,
            shadow_score REAL,
            delta REAL,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_shadow_scan ON shadow_scores(scan_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_shadow_delta ON shadow_scores(delta)")


# -- Migration 007: scan_log retained column --
@migration
def migration_007(cursor):
    """Add retained column to scan_log for high-score article retention tracking."""
    try:
        cursor.execute("ALTER TABLE scan_log ADD COLUMN retained INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists


# -- Migration 008: Profile isolation + email auth tables --
@migration
def migration_008(cursor):
    """Add profile isolation (profile_id columns) and email auth tables."""

    # --- Auth tables ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            is_admin BOOLEAN DEFAULT FALSE,
            email_verified BOOLEAN DEFAULT FALSE,
            verification_code_hash TEXT,
            verification_expires DATETIME,
            reset_code_hash TEXT,
            reset_code_expires DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            config_overlay TEXT NOT NULL DEFAULT '{}',
            ui_state TEXT NOT NULL DEFAULT '{}',
            is_default BOOLEAN DEFAULT FALSE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_active DATETIME,
            UNIQUE(user_id, name)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            profile_id INTEGER REFERENCES profiles(id) ON DELETE SET NULL,
            device_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            expires_at DATETIME NOT NULL,
            last_active DATETIME
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS invite_codes (
            code TEXT PRIMARY KEY,
            created_by INTEGER REFERENCES users(id),
            used_by INTEGER REFERENCES users(id),
            expires_at DATETIME NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --- profile_id on existing tables ---
    # Add profile_id column to 5 existing tables (DEFAULT 0 = legacy sentinel)
    for table in ['news_items', 'scan_log', 'user_feedback', 'shadow_scores', 'briefings']:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN profile_id INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Rebuild news_items to change UNIQUE(url) to UNIQUE(url, profile_id)
    # SQLite doesn't support DROP CONSTRAINT, so we rebuild the table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_items_new (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            summary TEXT,
            source TEXT,
            root TEXT,
            category TEXT,
            score REAL DEFAULT 0.0,
            score_reason TEXT,
            timestamp TEXT,
            fetched_at TEXT NOT NULL,
            shown_to_user INTEGER DEFAULT 0,
            dismissed INTEGER DEFAULT 0,
            profile_id INTEGER NOT NULL DEFAULT 0,
            UNIQUE(url, profile_id)
        )
    """)
    cursor.execute("""
        INSERT OR IGNORE INTO news_items_new
        SELECT id, title, url, summary, source, root, category, score,
               score_reason, timestamp, fetched_at, shown_to_user, dismissed, profile_id
        FROM news_items
    """)
    cursor.execute("DROP TABLE news_items")
    cursor.execute("ALTER TABLE news_items_new RENAME TO news_items")

    # --- Indexes ---
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_profiles_user ON profiles(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_profile ON news_items(profile_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_profile ON user_feedback(profile_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_shadow_profile ON shadow_scores(profile_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scanlog_profile ON scan_log(profile_id)")
    # Recreate indexes that were on the old news_items table
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_fetched ON news_items(fetched_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_score ON news_items(score)")


# =========================================================================
# Migration runner
# =========================================================================

def _ensure_schema_version_table(cursor):
    """Create the schema_version tracking table if it doesn't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT
        )
    """)
    cursor.execute("""
        INSERT OR IGNORE INTO schema_version (id, version) VALUES (1, 0)
    """)


def _get_version(cursor):
    """Get current schema version."""
    cursor.execute("SELECT version FROM schema_version WHERE id = 1")
    row = cursor.fetchone()
    return row[0] if row else 0


def _set_version(cursor, version):
    """Update schema version."""
    from datetime import datetime
    cursor.execute(
        "UPDATE schema_version SET version = ?, updated_at = ? WHERE id = 1",
        (version, datetime.now().isoformat())
    )


def run_migrations(conn):
    """
    Run all pending migrations against the given connection.

    Safe to call on every startup — already-applied migrations are skipped.
    Returns the number of migrations applied this run.
    """
    cursor = conn.cursor()
    _ensure_schema_version_table(cursor)

    current = _get_version(cursor)
    total = len(MIGRATIONS)
    applied = 0

    if current >= total:
        logger.info(f"Database schema up to date (version {current})")
        return 0

    for i in range(current, total):
        migration_fn = MIGRATIONS[i]
        version = i + 1
        name = migration_fn.__doc__ or migration_fn.__name__
        name = name.strip().split('\n')[0]  # first line only
        logger.info(f"Running migration {version}/{total}: {name}")
        try:
            migration_fn(cursor)
            _set_version(cursor, version)
            conn.commit()
            applied += 1
        except Exception as e:
            conn.rollback()
            logger.error(f"Migration {version} failed: {e}")
            raise

    logger.info(f"Applied {applied} migration(s), schema now at version {total}")
    return applied
