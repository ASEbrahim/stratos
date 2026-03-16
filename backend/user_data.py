"""
Per-user data directory management for StratOS.

Provides structured, grep-friendly data exports alongside SQLite storage.
Each DB-auth user gets: data/users/{user_id}/ with scan logs, feedback,
briefings, and article exports as JSONL/JSON files.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("STRAT_OS")

_BASE = Path(__file__).parent / "data" / "users"


def _safe_filename(filename: str) -> str:
    """Validate filename to prevent path traversal.

    Rejects any filename containing path separators, parent-directory
    references, or null bytes.  Returns the basename only.
    """
    if not filename:
        raise ValueError("Empty filename")
    # Strip directory components — only the leaf name is allowed
    name = Path(filename).name
    if name != filename:
        raise ValueError(f"Path traversal attempt blocked: {filename!r}")
    if '..' in name or '\x00' in name:
        raise ValueError(f"Invalid filename: {filename!r}")
    return name


def get_path(user_id: int) -> Path:
    """Get the data directory path for a user."""
    return _BASE / str(user_id)


def ensure_dir(user_id: int) -> Path:
    """Create the per-user data directory structure. Returns the path."""
    p = get_path(user_id)
    (p / "scans").mkdir(parents=True, exist_ok=True)
    (p / "briefings").mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(user_id: int, filename: str, data: dict):
    """Append a single JSON line to a user's JSONL file."""
    if user_id <= 0:
        return
    try:
        safe_name = _safe_filename(filename)
        p = get_path(user_id)
        if not p.exists():
            ensure_dir(user_id)
        filepath = p / safe_name
        # Final safeguard: resolved path must be under _BASE
        if not filepath.resolve().is_relative_to(_BASE.resolve()):
            raise ValueError(f"Path escape blocked: {filepath}")
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, default=str) + "\n")
    except ValueError as e:
        logger.warning(f"user_data append_jsonl({user_id}, {filename}): {e}")
    except Exception as e:
        logger.error(f"user_data append_jsonl({user_id}, {filename}): {e}")


def write_json(user_id: int, filename: str, data: dict):
    """Write/overwrite a JSON file in a user's directory."""
    if user_id <= 0:
        return
    try:
        safe_name = _safe_filename(filename)
        p = get_path(user_id)
        if not p.exists():
            ensure_dir(user_id)
        filepath = p / safe_name
        # Final safeguard: resolved path must be under _BASE
        if not filepath.resolve().is_relative_to(_BASE.resolve()):
            raise ValueError(f"Path escape blocked: {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except ValueError as e:
        logger.warning(f"user_data write_json({user_id}, {filename}): {e}")
    except Exception as e:
        logger.error(f"user_data write_json({user_id}, {filename}): {e}")


def get_user_id_for_profile(db, profile_id: int) -> int:
    """Resolve a DB profile_id to user_id. Returns 0 for legacy profiles."""
    if profile_id <= 0:
        return 0
    try:
        cursor = db.conn.cursor()
        cursor.execute("SELECT user_id FROM profiles WHERE id = ?", (profile_id,))
        row = cursor.fetchone()
        return row[0] if row else 0
    except Exception as e:
        logger.error(f"get_user_id_for_profile({profile_id}): {e}")
        return 0
