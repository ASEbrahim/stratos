"""
Scenario Manager — Multi-scenario management for the Games persona.

Storage: SQLite `scenarios` table (migration 021).
Lazy-migrates legacy file-based scenarios on first access per profile.
"""

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def _validate_profile_id(profile_id):
    """Validate profile_id is a positive integer."""
    if not isinstance(profile_id, int) or profile_id <= 0:
        raise ValueError(f"Invalid profile_id: {profile_id}")


def _validate_scenario_name(name):
    """Validate scenario name to prevent path traversal and injection."""
    if not name or not isinstance(name, str):
        raise ValueError("Scenario name must be a non-empty string")
    if '..' in name or '/' in name or '\\' in name:
        raise ValueError(f"Invalid scenario name (path traversal attempt): {name}")
    if len(name) > 200:
        raise ValueError("Scenario name too long (max 200 chars)")


class ScenarioManager:
    """Manages multiple saved game/roleplay scenarios per profile via DB."""

    def __init__(self, config: dict, db=None):
        self.base_dir = Path(config.get("system", {}).get("data_dir", "data")) / "users"
        self.db = db

    # ── Legacy file migration ──────────────────────────────────────

    def _legacy_dir(self, profile_id: int) -> Path:
        return self.base_dir / str(profile_id) / "context" / "gaming" / "scenarios"

    def _migrate_file_scenarios(self, profile_id: int):
        """One-time migration: move file-based scenarios to DB rows."""
        legacy = self._legacy_dir(profile_id)
        if not legacy.exists():
            return
        cursor = self.db.conn.cursor()
        # Check if we already migrated (any DB scenarios exist for this profile)
        cursor.execute("SELECT COUNT(*) FROM scenarios WHERE profile_id = ?", (profile_id,))
        if cursor.fetchone()[0] > 0:
            return  # Already migrated

        active_file = legacy / "active_scenario.txt"
        active_name = active_file.read_text().strip() if active_file.exists() else None
        migrated = 0

        for d in sorted(legacy.iterdir()):
            if not d.is_dir() or d.name == '__pycache__':
                continue
            name = d.name
            state_md = ''
            world_md = ''
            characters_json = '[]'

            sf = d / "state.md"
            if sf.exists():
                state_md = sf.read_text(errors='replace')
            wf = d / "world.md"
            if wf.exists():
                world_md = wf.read_text(errors='replace')
            cf = d / "characters.json"
            if cf.exists():
                try:
                    characters_json = cf.read_text()
                    json.loads(characters_json)  # validate
                except json.JSONDecodeError as e:
                    logger.warning(f"Migration: corrupt characters.json in '{name}': {e}")
                    characters_json = '[]'
                except Exception as e:
                    logger.warning(f"Migration: failed to read characters.json in '{name}': {e}")
                    characters_json = '[]'

            try:
                cursor.execute(
                    "INSERT OR IGNORE INTO scenarios "
                    "(profile_id, persona, name, state_md, world_md, characters_json, is_active, description) "
                    "VALUES (?, 'gaming', ?, ?, ?, ?, ?, ?)",
                    (profile_id, name, state_md, world_md, characters_json,
                     1 if name == active_name else 0,
                     world_md[:100] if world_md else '')
                )
                migrated += 1
            except Exception as e:
                logger.warning(f"Failed to migrate scenario '{name}': {e}")

        if migrated:
            self.db._commit()
            logger.info(f"Migrated {migrated} file-based scenarios to DB for profile {profile_id}")

    # ── DB operations ──────────────────────────────────────────────

    def get_active_scenario(self, profile_id: int) -> Optional[str]:
        """Get the name of the currently active scenario."""
        self._migrate_file_scenarios(profile_id)
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT name FROM scenarios WHERE profile_id = ? AND is_active = 1 LIMIT 1",
            (profile_id,)
        )
        row = cursor.fetchone()
        return row['name'] if row else None

    def set_active_scenario(self, profile_id: int, scenario_name: str) -> bool:
        """Set the active scenario. Returns False if scenario doesn't exist."""
        self._migrate_file_scenarios(profile_id)
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id FROM scenarios WHERE profile_id = ? AND name = ?",
                       (profile_id, scenario_name))
        if not cursor.fetchone():
            return False
        # Deactivate all, then activate target
        cursor.execute("UPDATE scenarios SET is_active = 0 WHERE profile_id = ?", (profile_id,))
        cursor.execute("UPDATE scenarios SET is_active = 1 WHERE profile_id = ? AND name = ?",
                       (profile_id, scenario_name))
        self.db._commit()

        # Also update persona_context active_scenario key
        try:
            scenario = self.get_scenario(profile_id, scenario_name)
            content = ''
            if scenario:
                parts = []
                if scenario.get('world'):
                    parts.append(scenario['world'][:2000])
                if scenario.get('state'):
                    parts.append(scenario['state'][:2000])
                content = '\n\n'.join(parts)
            cursor.execute(
                "INSERT INTO persona_context (profile_id, persona_name, context_key, content, updated_at) "
                "VALUES (?, 'gaming', 'active_scenario', ?, ?) "
                "ON CONFLICT(profile_id, persona_name, context_key) "
                "DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
                (profile_id, content, datetime.now().isoformat())
            )
            self.db._commit()
        except Exception as e:
            logger.warning(f"Failed to update active_scenario in DB: {e}")

        return True

    def list_scenarios(self, profile_id: int) -> List[Dict[str, Any]]:
        """List all scenarios with metadata."""
        self._migrate_file_scenarios(profile_id)
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT id, name, description, genre, is_active, world_md, characters_json, "
            "created_at, updated_at FROM scenarios "
            "WHERE profile_id = ? ORDER BY name",
            (profile_id,)
        )
        result = []
        for row in cursor.fetchall():
            d = dict(row)
            chars = []
            try:
                chars = json.loads(d.get('characters_json', '[]'))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Corrupt characters_json for scenario '{d['name']}': {e}")
                chars = []
            info = {
                "name": d['name'],
                "is_active": bool(d['is_active']),
                "has_state": True,
                "has_characters": len(chars) > 0,
                "has_world": bool(d.get('world_md')),
                "world_description": (d.get('world_md') or '')[:100],
                "character_count": len(chars) if isinstance(chars, list) else 0,
                "created": d.get('created_at', ''),
                "modified": d.get('updated_at', ''),
            }
            result.append(info)
        return result

    def create_scenario(self, profile_id: int, scenario_name: str,
                        world_md: str = "", characters: List[Dict] = None) -> Dict[str, Any]:
        """Create a new scenario."""
        _validate_profile_id(profile_id)
        self._migrate_file_scenarios(profile_id)
        safe_name = "".join(c for c in scenario_name if c.isalnum() or c in ('_', '-', ' ')).strip()
        safe_name = safe_name.replace(' ', '_')
        if not safe_name:
            return {"error": "Invalid scenario name"}
        if len(safe_name) > 200:
            return {"error": "Scenario name too long (max 200 characters)"}

        cursor = self.db.conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO scenarios (profile_id, persona, name, description, state_md, world_md, characters_json) "
                "VALUES (?, 'gaming', ?, ?, ?, ?, ?)",
                (profile_id, safe_name, (world_md or '')[:100],
                 f"# {scenario_name} — State\n\nScenario just created. No events yet.",
                 world_md or f"# {scenario_name}\n\nDescribe your world here.",
                 json.dumps(characters or []))
            )
            self.db._commit()
        except Exception as e:
            if 'UNIQUE' in str(e):
                return {"error": f"Scenario '{safe_name}' already exists"}
            logger.error(f"Failed to create scenario '{safe_name}': {e}")
            return {"error": "Failed to create scenario"}

        # Auto-activate if it's the only scenario
        cursor.execute("SELECT COUNT(*) FROM scenarios WHERE profile_id = ?", (profile_id,))
        if cursor.fetchone()[0] == 1:
            self.set_active_scenario(profile_id, safe_name)

        return {"ok": True, "name": safe_name}

    def delete_scenario(self, profile_id: int, scenario_name: str) -> bool:
        """Delete a scenario from DB and remove its file folder."""
        _validate_profile_id(profile_id)
        _validate_scenario_name(scenario_name)

        cursor = self.db.conn.cursor()
        cursor.execute("DELETE FROM scenarios WHERE profile_id = ? AND name = ?",
                       (profile_id, scenario_name))
        self.db._commit()
        deleted = cursor.rowcount > 0

        # Also remove file-based scenario folder to prevent re-migration
        scenario_dir = self._legacy_dir(profile_id) / scenario_name
        # Verify resolved path stays within legacy dir (prevent traversal)
        legacy_base = self._legacy_dir(profile_id).resolve()
        resolved_dir = scenario_dir.resolve()
        if not str(resolved_dir).startswith(str(legacy_base) + '/') and resolved_dir != legacy_base:
            logger.warning(f"Path traversal blocked in delete_scenario: {scenario_name}")
            return deleted

        if scenario_dir.exists() and scenario_dir.is_dir():
            try:
                shutil.rmtree(scenario_dir)
                logger.info(f"Removed scenario folder: {scenario_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove scenario folder {scenario_dir}: {e}")

        return deleted

    def get_scenario(self, profile_id: int, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Get full scenario data."""
        self._migrate_file_scenarios(profile_id)
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT name, state_md, world_md, characters_json FROM scenarios "
            "WHERE profile_id = ? AND name = ?",
            (profile_id, scenario_name)
        )
        row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        result = {"name": d['name']}
        if d.get('state_md'):
            result['state'] = d['state_md']
        if d.get('world_md'):
            result['world'] = d['world_md']
        try:
            result['characters'] = json.loads(d.get('characters_json', '[]'))
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Corrupt characters_json for scenario '{scenario_name}': {e}")
            result['characters'] = []
        return result

    def save_scenario(self, profile_id: int, scenario_name: str,
                      state: str = None, world: str = None,
                      characters: List[Dict] = None) -> bool:
        """Update scenario data."""
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id FROM scenarios WHERE profile_id = ? AND name = ?",
                       (profile_id, scenario_name))
        if not cursor.fetchone():
            return False

        if state is not None:
            cursor.execute("UPDATE scenarios SET state_md = ?, updated_at = ? WHERE profile_id = ? AND name = ?",
                           (state, datetime.now().isoformat(), profile_id, scenario_name))
        if world is not None:
            cursor.execute("UPDATE scenarios SET world_md = ?, updated_at = ? WHERE profile_id = ? AND name = ?",
                           (world, datetime.now().isoformat(), profile_id, scenario_name))
        if characters is not None:
            cursor.execute("UPDATE scenarios SET characters_json = ?, updated_at = ? WHERE profile_id = ? AND name = ?",
                           (json.dumps(characters), datetime.now().isoformat(), profile_id, scenario_name))
        self.db._commit()
        return True
