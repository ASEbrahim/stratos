"""
Scenario Manager — Multi-scenario management for the Games persona.

Each scenario is a save slot:
  data/users/{profile_id}/context/gaming/scenarios/{scenario_name}/
    ├── state.md           # Current state of this scenario
    ├── characters.json    # Character roster
    ├── world.md           # World bible
    └── conversation_log/  # Conversation history for this scenario
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ScenarioManager:
    """Manages multiple saved game/roleplay scenarios per profile."""

    def __init__(self, config: dict, db=None):
        self.base_dir = Path(config.get("system", {}).get("data_dir", "data")) / "users"
        self.db = db

    def _scenarios_dir(self, profile_id: int) -> Path:
        d = self.base_dir / str(profile_id) / "context" / "gaming" / "scenarios"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _active_file(self, profile_id: int) -> Path:
        return self._scenarios_dir(profile_id) / "active_scenario.txt"

    def get_active_scenario(self, profile_id: int) -> Optional[str]:
        """Get the name of the currently active scenario."""
        af = self._active_file(profile_id)
        if af.exists():
            name = af.read_text().strip()
            # Verify it still exists
            if (self._scenarios_dir(profile_id) / name).is_dir():
                return name
        return None

    def set_active_scenario(self, profile_id: int, scenario_name: str) -> bool:
        """Set the active scenario. Returns False if scenario doesn't exist."""
        scenario_dir = self._scenarios_dir(profile_id) / scenario_name
        if not scenario_dir.is_dir():
            return False
        self._active_file(profile_id).write_text(scenario_name)

        # Also update persona_context active_scenario key in DB
        if self.db:
            try:
                cursor = self.db.conn.cursor()
                cursor.execute(
                    "INSERT INTO persona_context (profile_id, persona_name, context_key, content, updated_at) "
                    "VALUES (?, 'gaming', 'active_scenario', ?, ?) "
                    "ON CONFLICT(profile_id, persona_name, context_key) "
                    "DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
                    (profile_id, self._load_scenario_state(profile_id, scenario_name),
                     datetime.now().isoformat())
                )
                self.db._commit()
            except Exception as e:
                logger.warning(f"Failed to update active_scenario in DB: {e}")

        return True

    def list_scenarios(self, profile_id: int) -> List[Dict[str, Any]]:
        """List all scenarios with metadata."""
        scenarios_dir = self._scenarios_dir(profile_id)
        active = self.get_active_scenario(profile_id)
        result = []

        for d in sorted(scenarios_dir.iterdir()):
            if d.is_dir() and d.name != '__pycache__':
                state_file = d / "state.md"
                chars_file = d / "characters.json"
                world_file = d / "world.md"

                info = {
                    "name": d.name,
                    "is_active": d.name == active,
                    "has_state": state_file.exists(),
                    "has_characters": chars_file.exists(),
                    "has_world": world_file.exists(),
                    "created": datetime.fromtimestamp(d.stat().st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(d.stat().st_mtime).isoformat(),
                }

                # Count characters
                if chars_file.exists():
                    try:
                        chars = json.loads(chars_file.read_text())
                        info["character_count"] = len(chars) if isinstance(chars, list) else len(chars.get("characters", []))
                    except Exception:
                        info["character_count"] = 0

                result.append(info)

        return result

    def create_scenario(self, profile_id: int, scenario_name: str,
                        world_md: str = "", characters: List[Dict] = None) -> Dict[str, Any]:
        """Create a new scenario folder with optional initial content."""
        # Sanitize name
        safe_name = "".join(c for c in scenario_name if c.isalnum() or c in ('_', '-', ' ')).strip()
        safe_name = safe_name.replace(' ', '_')
        if not safe_name:
            return {"error": "Invalid scenario name"}

        scenario_dir = self._scenarios_dir(profile_id) / safe_name
        if scenario_dir.exists():
            return {"error": f"Scenario '{safe_name}' already exists"}

        scenario_dir.mkdir(parents=True)
        (scenario_dir / "conversation_log").mkdir()

        # Write initial content
        if world_md:
            (scenario_dir / "world.md").write_text(world_md)
        else:
            (scenario_dir / "world.md").write_text(f"# {scenario_name}\n\nDescribe your world here.")

        (scenario_dir / "state.md").write_text(f"# {scenario_name} — State\n\nScenario just created. No events yet.")

        if characters:
            (scenario_dir / "characters.json").write_text(json.dumps(characters, indent=2))
        else:
            (scenario_dir / "characters.json").write_text(json.dumps([], indent=2))

        # Auto-activate if it's the only scenario
        scenarios = self.list_scenarios(profile_id)
        if len(scenarios) == 1:
            self.set_active_scenario(profile_id, safe_name)

        return {"ok": True, "name": safe_name}

    def delete_scenario(self, profile_id: int, scenario_name: str) -> bool:
        """Delete a scenario and all its data."""
        scenario_dir = self._scenarios_dir(profile_id) / scenario_name
        if not scenario_dir.is_dir():
            return False

        # Clear active if this is the active one
        if self.get_active_scenario(profile_id) == scenario_name:
            af = self._active_file(profile_id)
            if af.exists():
                af.unlink()

        shutil.rmtree(scenario_dir)
        return True

    def get_scenario(self, profile_id: int, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Get full scenario data."""
        scenario_dir = self._scenarios_dir(profile_id) / scenario_name
        if not scenario_dir.is_dir():
            return None

        result = {"name": scenario_name}

        state_file = scenario_dir / "state.md"
        if state_file.exists():
            result["state"] = state_file.read_text(errors='replace')

        world_file = scenario_dir / "world.md"
        if world_file.exists():
            result["world"] = world_file.read_text(errors='replace')

        chars_file = scenario_dir / "characters.json"
        if chars_file.exists():
            try:
                result["characters"] = json.loads(chars_file.read_text())
            except Exception:
                result["characters"] = []

        return result

    def save_scenario(self, profile_id: int, scenario_name: str,
                      state: str = None, world: str = None,
                      characters: List[Dict] = None) -> bool:
        """Update scenario files."""
        scenario_dir = self._scenarios_dir(profile_id) / scenario_name
        if not scenario_dir.is_dir():
            return False

        if state is not None:
            (scenario_dir / "state.md").write_text(state)
        if world is not None:
            (scenario_dir / "world.md").write_text(world)
        if characters is not None:
            (scenario_dir / "characters.json").write_text(json.dumps(characters, indent=2))

        return True

    def _load_scenario_state(self, profile_id: int, scenario_name: str) -> str:
        """Load scenario state for DB sync."""
        scenario_dir = self._scenarios_dir(profile_id) / scenario_name
        parts = []

        world_file = scenario_dir / "world.md"
        if world_file.exists():
            parts.append(world_file.read_text(errors='replace')[:2000])

        state_file = scenario_dir / "state.md"
        if state_file.exists():
            parts.append(state_file.read_text(errors='replace')[:2000])

        return "\n\n".join(parts)
