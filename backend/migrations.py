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
# NOTE: f-strings in ALTER TABLE are safe — column names are hardcoded literals,
# not user input. SQLite DDL does not support parameterized column names.
# FINDING-010: Acknowledged. All DDL f-strings use developer-controlled literals only.
# =========================================================================

# Allowlists for DDL validation (defense-in-depth — catches typos too)
_VALID_TABLES = frozenset([
    'news_items', 'market_snapshots', 'entities', 'entity_mentions', 'scan_log',
    'briefings', 'user_feedback', 'shadow_scores', 'users', 'profiles', 'sessions',
    'pending_registrations', 'invite_codes', 'conversations', 'scenarios',
    'persona_entities', 'persona_context', 'user_files', 'user_preference_signals',
    'youtube_channels', 'youtube_videos', 'video_insights', 'narration_sources',
    'prompt_templates', 'sprint_log', 'schema_version',
    'rp_messages', 'rp_edits', 'rp_suggestions', 'rp_feedback',
    'rp_conversation_scores', 'rp_session_context',
    'character_cards', 'character_card_stats', 'character_card_ratings',
    'generated_images',
])

def _safe_table(name: str) -> str:
    """Validate table name against allowlist before DDL interpolation."""
    if name not in _VALID_TABLES:
        raise ValueError(f"Unknown table in migration: {name}")
    return name

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
            cursor.execute(f"ALTER TABLE {_safe_table(table)} ADD COLUMN profile_id INTEGER NOT NULL DEFAULT 0")
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


# -- Migration 009: pending_registrations table --
@migration
def migration_009(cursor):
    """Create pending_registrations table — users aren't created until email is verified."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pending_registrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            is_admin BOOLEAN DEFAULT FALSE,
            verification_code_hash TEXT NOT NULL,
            verification_expires DATETIME NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pending_email ON pending_registrations(email)")


# -- Migration 010: OTP login columns --
@migration
def migration_010(cursor):
    """Add OTP login code columns to users table."""
    for col in ["otp_code_hash TEXT", "otp_code_expires DATETIME"]:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # Column already exists


# -- Migration 011: Composite index for profile-scoped feedback queries --
@migration
def migration_011(cursor):
    """Add composite index for profile-scoped feedback queries (fixes 1.1/1.2)."""
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_profile_date "
        "ON user_feedback(profile_id, created_at)"
    )


# -- Migration 012: Brute-force protection for verification codes --
@migration
def migration_012(cursor):
    """Add verify_attempts counter to pending_registrations."""
    try:
        cursor.execute("ALTER TABLE pending_registrations ADD COLUMN verify_attempts INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # Column already exists


# -- Migration 013: Composite index for was_dismissed() query --
@migration
def migration_013(cursor):
    """Add composite index for was_dismissed() query (url, action, profile_id)."""
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_url_action_profile "
        "ON user_feedback(url, action, profile_id)"
    )


# -- Migration 014: Composite index for profile-scoped news queries --
@migration
def migration_014(cursor):
    """Add composite index for profile-scoped news filtering (get_top_signals, get_category_stats)."""
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_news_profile_fetched_score "
        "ON news_items(profile_id, fetched_at, score)"
    )


@migration
def migration_015(cursor):
    """Create user_files table for per-user document storage."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            content_text TEXT DEFAULT '',
            uploaded_at TEXT NOT NULL,
            file_path TEXT NOT NULL
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_files_profile "
        "ON user_files(profile_id)"
    )


# -- Migration 016: YouTube knowledge extraction tables --
@migration
def migration_016(cursor):
    """Create YouTube channel tracking, video processing, and insight extraction tables."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS youtube_channels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER NOT NULL,
            channel_id TEXT NOT NULL,
            channel_name TEXT,
            channel_url TEXT,
            lenses TEXT,
            added_at TEXT DEFAULT (datetime('now')),
            last_checked TEXT,
            UNIQUE(profile_id, channel_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS youtube_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id INTEGER REFERENCES youtube_channels(id),
            profile_id INTEGER NOT NULL,
            video_id TEXT NOT NULL,
            title TEXT,
            published_at TEXT,
            duration_seconds INTEGER,
            transcript_text TEXT,
            transcript_method TEXT,
            status TEXT DEFAULT 'pending',
            error_message TEXT,
            processed_at TEXT,
            UNIQUE(profile_id, video_id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER REFERENCES youtube_videos(id),
            profile_id INTEGER NOT NULL,
            lens_name TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_yt_channels_profile ON youtube_channels(profile_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_yt_videos_profile ON youtube_videos(profile_id, status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_yt_insights_profile_lens ON video_insights(profile_id, lens_name)")


# -- Migration 017: Persona context isolation table --
@migration
def migration_017(cursor):
    """Create persona_context table for per-persona, per-profile context storage."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persona_context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER NOT NULL,
            persona_name TEXT NOT NULL,
            context_key TEXT NOT NULL,
            content TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now')),
            UNIQUE(profile_id, persona_name, context_key)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_persona_context ON persona_context(profile_id, persona_name)")


# -- Migration 018: User preference signals for personalization feedback --
@migration
def migration_018(cursor):
    """Create user_preference_signals table for cross-persona feedback loop."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preference_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER NOT NULL,
            persona_source TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            signal_key TEXT NOT NULL,
            signal_weight REAL DEFAULT 1.0,
            auto_generated BOOLEAN DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            expires_at TEXT,
            UNIQUE(profile_id, persona_source, signal_type, signal_key)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pref_signals ON user_preference_signals(profile_id, signal_type)")


# -- Migration 019: persona column on user_files --
@migration
def migration_019(cursor):
    """Add persona column to user_files for persona-scoped file isolation."""
    try:
        cursor.execute("ALTER TABLE user_files ADD COLUMN persona TEXT NOT NULL DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # Column already exists
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_files_persona "
        "ON user_files(profile_id, persona)"
    )


# -- Migration 020: conversations table --
@migration
def migration_020(cursor):
    """Create conversations table for DB-backed chat history."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER NOT NULL,
            persona TEXT NOT NULL,
            title TEXT DEFAULT '',
            messages TEXT NOT NULL DEFAULT '[]',
            is_active INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            archived INTEGER DEFAULT 0
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_profile_persona ON conversations(profile_id, persona, archived)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_active ON conversations(profile_id, persona, is_active)")


# -- Migration 021: scenarios table --
@migration
def migration_021(cursor):
    """Create scenarios table for DB-backed game scenario storage."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER NOT NULL,
            persona TEXT NOT NULL DEFAULT 'gaming',
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            state_md TEXT DEFAULT '',
            world_md TEXT DEFAULT '',
            characters_json TEXT DEFAULT '[]',
            genre TEXT DEFAULT '',
            is_active INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(profile_id, name)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scenarios_profile ON scenarios(profile_id, persona, is_active)")


# -- Migration 022: persona entities table --
@migration
def migration_022(cursor):
    """Create persona_entities table for persona-generic entity persistence (characters, figures, etc.)."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persona_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER NOT NULL,
            persona TEXT NOT NULL,
            scenario_name TEXT DEFAULT '',
            name TEXT NOT NULL,
            display_name TEXT NOT NULL,
            entity_type TEXT DEFAULT 'character',
            identity_md TEXT DEFAULT '',
            personality_md TEXT DEFAULT '',
            speaking_style_md TEXT DEFAULT '',
            relationship_md TEXT DEFAULT '',
            memory_md TEXT DEFAULT '',
            knowledge_md TEXT DEFAULT '',
            extra_md TEXT DEFAULT '',
            auto_save INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(profile_id, persona, scenario_name, name)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_profile ON persona_entities(profile_id, persona, scenario_name)")


# -- Migration 023: Bilingual lens extraction --
@migration
def migration_023(cursor):
    """Add language column to video_insights and transcript_language to youtube_videos for bilingual extraction."""
    # Add language column to video_insights (default 'en' for existing rows)
    try:
        cursor.execute("ALTER TABLE video_insights ADD COLUMN language TEXT NOT NULL DEFAULT 'en'")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add transcript_language to youtube_videos
    try:
        cursor.execute("ALTER TABLE youtube_videos ADD COLUMN transcript_language TEXT DEFAULT 'en'")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Add index for language-scoped queries
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_yt_insights_lang "
        "ON video_insights(video_id, lens_name, language)"
    )


@migration
def migration_024(cursor):
    """Add narration_sources table for caching resolved source URLs."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS narration_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            profile_id INTEGER NOT NULL,
            narration_hash TEXT NOT NULL,
            source_claimed TEXT,
            source_reference TEXT,
            resolved_url TEXT,
            resolution_method TEXT,
            confidence REAL DEFAULT 0.0,
            resolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(video_id, profile_id, narration_hash)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_narr_src_video "
        "ON narration_sources(video_id, profile_id)"
    )


# -- Migration 025: Sprint prompt builder tables --
@migration
def migration_025(cursor):
    """Create sprint_log and prompt_templates tables for the Sprint Prompt Builder."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sprint_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sprint_number INTEGER,
            sprint_name TEXT NOT NULL,
            owner TEXT,
            generated_prompt TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'generated'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompt_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            profile_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            template_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(profile_id, name)
        )
    """)


@migration
def migration_026(cursor):
    """Add brute-force attempt counters for password reset and OTP verification."""
    for col in ["reset_attempts INTEGER DEFAULT 0", "otp_attempts INTEGER DEFAULT 0"]:
        try:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # Column already exists


# -- Migration 027: RP expansion — messages, edits, feedback, cards, images, scores --
@migration
def migration_027(cursor):
    """Create tables for RP interaction layer, character cards, image gen, and data pipeline."""

    # ── RP Message System ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rp_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            profile_id INTEGER NOT NULL,
            branch_id TEXT NOT NULL DEFAULT 'main',
            parent_branch_id TEXT DEFAULT NULL,
            branch_point_turn INTEGER DEFAULT NULL,
            turn_number INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
            content TEXT NOT NULL,
            model_version TEXT,
            character_card_id TEXT,
            persona TEXT DEFAULT 'roleplay',
            response_tokens INTEGER,
            response_ms INTEGER,
            is_active BOOLEAN DEFAULT TRUE,
            swipe_group_id TEXT DEFAULT NULL,
            was_selected BOOLEAN DEFAULT TRUE,
            director_note TEXT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rp_msg_session ON rp_messages(session_id, branch_id, turn_number)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rp_msg_profile ON rp_messages(profile_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rp_msg_swipe ON rp_messages(swipe_group_id)")

    # ── User Edits on AI Responses (DPO training pairs) ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rp_edits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL REFERENCES rp_messages(id),
            session_id TEXT NOT NULL,
            original_content TEXT NOT NULL,
            edited_content TEXT NOT NULL,
            edit_delta_category TEXT DEFAULT NULL,
            edit_reason TEXT DEFAULT NULL,
            character_card_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rp_edits_session ON rp_edits(session_id)")

    # ── Director's Note / Steering Suggestions ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rp_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL REFERENCES rp_messages(id),
            session_id TEXT NOT NULL,
            suggestion_text TEXT NOT NULL,
            was_followed BOOLEAN DEFAULT NULL,
            user_continued BOOLEAN DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── User Feedback (thumbs up/down) ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rp_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL REFERENCES rp_messages(id),
            profile_id INTEGER NOT NULL,
            feedback_type TEXT NOT NULL CHECK(feedback_type IN ('thumbs_up', 'thumbs_down')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(message_id, profile_id)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rp_feedback_message ON rp_feedback(message_id)")

    # ── Character Cards ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS character_cards (
            id TEXT PRIMARY KEY,
            creator_profile_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            physical_description TEXT DEFAULT '',
            speech_pattern TEXT DEFAULT '',
            emotional_trigger TEXT DEFAULT '',
            defensive_mechanism TEXT DEFAULT '',
            vulnerability TEXT DEFAULT '',
            specific_detail TEXT DEFAULT '',
            personality TEXT DEFAULT '',
            scenario TEXT DEFAULT '',
            first_message TEXT DEFAULT '',
            example_dialogues TEXT DEFAULT '',
            genre_tags TEXT DEFAULT '[]',
            content_rating TEXT DEFAULT 'sfw' CHECK(content_rating IN ('sfw', 'nsfw')),
            avatar_image_path TEXT DEFAULT NULL,
            is_published BOOLEAN DEFAULT FALSE,
            quality_elements_count INTEGER DEFAULT 0,
            imported_from TEXT DEFAULT NULL,
            tavern_card_raw TEXT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_creator ON character_cards(creator_profile_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_published ON character_cards(is_published, quality_elements_count DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cards_genre ON character_cards(genre_tags)")

    # ── Character Card Community Stats ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS character_card_stats (
            card_id TEXT PRIMARY KEY REFERENCES character_cards(id),
            total_sessions INTEGER DEFAULT 0,
            avg_session_turns REAL DEFAULT 0,
            avg_session_duration_s REAL DEFAULT 0,
            total_ratings INTEGER DEFAULT 0,
            avg_rating REAL DEFAULT 0,
            thumbs_up_count INTEGER DEFAULT 0,
            thumbs_down_count INTEGER DEFAULT 0,
            regeneration_rate REAL DEFAULT 0,
            edit_rate REAL DEFAULT 0,
            abandonment_rate REAL DEFAULT 0,
            quality_elements_filled INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Character Card Ratings ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS character_card_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            card_id TEXT NOT NULL REFERENCES character_cards(id),
            profile_id INTEGER NOT NULL,
            rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(card_id, profile_id)
        )
    """)

    # ── Generated Images ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS generated_images (
            id TEXT PRIMARY KEY,
            profile_id INTEGER NOT NULL,
            prompt TEXT NOT NULL,
            negative_prompt TEXT DEFAULT '',
            model TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            seed INTEGER,
            steps INTEGER,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            character_card_id TEXT DEFAULT NULL,
            session_id TEXT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_profile ON generated_images(profile_id)")

    # ── Conversation Quality Scores (nightly batch) ──
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rp_conversation_scores (
            session_id TEXT PRIMARY KEY,
            profile_id INTEGER,
            total_turns INTEGER,
            session_duration_s REAL,
            quality_score REAL DEFAULT 0.5,
            avg_length_ratio REAL,
            godmod_detected_count INTEGER DEFAULT 0,
            register_mismatch_count INTEGER DEFAULT 0,
            repetition_avg REAL DEFAULT 0,
            correction_count INTEGER DEFAULT 0,
            thumbs_up_count INTEGER DEFAULT 0,
            thumbs_down_count INTEGER DEFAULT 0,
            regeneration_count INTEGER DEFAULT 0,
            edit_count INTEGER DEFAULT 0,
            character_card_id TEXT,
            model_version TEXT,
            persona TEXT,
            scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Privacy opt-in on profiles ──
    try:
        cursor.execute("ALTER TABLE profiles ADD COLUMN training_data_opt_in BOOLEAN DEFAULT FALSE")
    except sqlite3.OperationalError:
        pass  # Column already exists


@migration
def migration_028(cursor):
    """Create rp_session_context table for tiered memory persistence."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rp_session_context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            tier INTEGER NOT NULL,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            turn_number INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rp_ctx_session ON rp_session_context(session_id, tier)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rp_ctx_updated ON rp_session_context(session_id, updated_at DESC)")


# -- Migration 029: High-traffic query indexes --
@migration
def migration_029(cursor):
    """Add missing indexes for high-traffic query patterns."""
    # news_items(profile_id, url) — used by duplicate checks and lookups
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_news_profile_url "
        "ON news_items(profile_id, url)"
    )
    # rp_messages(session_id, branch_id) — used by get_full_branch_conversation
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_rp_msg_session_branch "
        "ON rp_messages(session_id, branch_id)"
    )
    # sessions(token) — already PK, but add index on user_id + expires for cleanup queries
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_token_expires "
        "ON sessions(token, expires_at)"
    )
    # character_cards(creator_profile_id) — already idx_cards_creator, verify it exists
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_cards_creator "
        "ON character_cards(creator_profile_id)"
    )


@migration
def migration_030(cursor):
    """Add pill fields to character_cards (gender, archetype, POV, relationship, NSFW comfort)."""
    _safe_table('character_cards')
    for col, default in [
        ('gender', None),                    # 'female', 'male', 'nonbinary', NULL
        ('archetype_override', None),        # 'shy', 'confident', 'tough', 'clinical', 'sweet', 'submissive', NULL
        ('narration_pov', None),             # 'third', 'first', 'mixed', NULL (default: third)
        ('relationship_to_user', None),      # 'stranger', 'friend', 'rival', 'love_interest', 'mentor', 'servant', NULL
        ('nsfw_comfort', None),              # 'fade', 'suggestive', 'explicit', NULL
        ('response_length_pref', None),      # 'brief', 'normal', 'detailed', NULL
        ('age_range', None),                 # 'teen', 'young_adult', 'adult', 'middle_aged', 'elderly', NULL
    ]:
        try:
            cursor.execute(f"ALTER TABLE character_cards ADD COLUMN {col} TEXT DEFAULT NULL")
        except sqlite3.OperationalError:
            pass  # column already exists


@migration
def migration_031(cursor):
    """Add narration_style to character_cards (full/cinematic/script)."""
    _safe_table('character_cards')
    try:
        cursor.execute("ALTER TABLE character_cards ADD COLUMN narration_style TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # column already exists


@migration
def migration_032(cursor):
    """Add personality_tags JSON column to character_cards for mix-and-match tags."""
    _safe_table('character_cards')
    try:
        cursor.execute("ALTER TABLE character_cards ADD COLUMN personality_tags TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # column already exists


@migration
def migration_033(cursor):
    """Add pinned column to youtube_videos for persistent video pinning."""
    _safe_table('youtube_videos')
    try:
        cursor.execute("ALTER TABLE youtube_videos ADD COLUMN pinned INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists


@migration
def migration_034(cursor):
    """Add Google OAuth columns to users table."""
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN google_id TEXT UNIQUE")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN auth_method TEXT DEFAULT 'password'")
    except sqlite3.OperationalError:
        pass
    # Index for fast Google ID lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id)")


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
        conn.commit()  # commit the schema_version check to release write lock
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
