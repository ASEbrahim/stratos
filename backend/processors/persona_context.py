"""
Persona Context Manager — Per-persona, per-profile context isolation.

Each persona has its own context directory and database entries:
  data/users/{profile_id}/context/{persona_name}/
    ├── system_context.md     # User-editable persona instructions
    ├── preferences.json      # Learned preferences
    └── uploads/              # Persona-scoped file uploads

Context is stored in both the database (persona_context table) and
filesystem (for uploaded files). The database is the source of truth
for system_context and preferences. The filesystem handles uploads.

Version history: last 5 versions of system_context.md are kept.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

MAX_VERSIONS = 5

# Default context templates per persona
DEFAULT_CONTEXTS = {
    'intelligence': "# Intelligence Persona Context\n\nAdd notes about your career focus, industry interests, and what signals matter most to you.",
    'market': "# Market Persona Context\n\nAdd notes about your investment thesis, sectors you track, and any price alerts or patterns you care about.",
    'scholarly': "# Scholarly Persona Context\n\nAdd notes about your research interests, channels you follow, and scholarly topics you're studying.",
    'gaming': "# Games Persona Context\n\nDescribe your characters, world settings, and roleplay preferences here. This context is injected every turn to maintain consistency.",
    'anime': "# Anime Persona Context\n\nAdd your watchlist, favorite genres, and discussion preferences.",
    'tcg': "# TCG Persona Context\n\nAdd your collection focus, format preferences, and budget/trade notes.",
}


class PersonaContextManager:
    """Manages isolated context storage for each persona per profile."""

    def __init__(self, config: dict, db=None):
        self.base_dir = Path(config.get("system", {}).get("data_dir", "data")) / "users"
        self.db = db

    def _context_dir(self, profile_id: int, persona_name: str) -> Path:
        """Get or create the context directory for a persona."""
        # Validate persona_name to prevent path traversal
        if not persona_name or '..' in persona_name or '/' in persona_name or '\\' in persona_name:
            raise ValueError(f"Invalid persona name: {persona_name!r}")
        d = self.base_dir / str(profile_id) / "context" / persona_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _uploads_dir(self, profile_id: int, persona_name: str) -> Path:
        """Get or create the uploads directory for a persona."""
        d = self._context_dir(profile_id, persona_name) / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Context CRUD ──────────────────────────────────────

    def get_context(self, profile_id: int, persona_name: str,
                    context_key: str = 'system_context') -> Optional[str]:
        """Get a context value for a persona."""
        if not self.db:
            return None
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "SELECT content FROM persona_context "
                "WHERE profile_id = ? AND persona_name = ? AND context_key = ?",
                (profile_id, persona_name, context_key)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)['content']
            # Return default if no custom context exists
            if context_key == 'system_context':
                return DEFAULT_CONTEXTS.get(persona_name, '')
            return None
        except Exception as e:
            logger.error(f"Get context error: {e}")
            return None

    def save_context(self, profile_id: int, persona_name: str,
                     context_key: str, content: str) -> bool:
        """Save a context value, with version history for system_context."""
        if not self.db:
            return False
        try:
            # Version history for system_context
            if context_key == 'system_context':
                self._save_version(profile_id, persona_name, content)

            cursor = self.db.conn.cursor()
            cursor.execute(
                """INSERT INTO persona_context (profile_id, persona_name, context_key, content, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(profile_id, persona_name, context_key)
                   DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at""",
                (profile_id, persona_name, context_key, content, datetime.now().isoformat())
            )
            self.db._commit()
            return True
        except Exception as e:
            logger.error(f"Save context error: {e}")
            return False

    def delete_context(self, profile_id: int, persona_name: str,
                       context_key: str) -> bool:
        """Reset a context key to default (deletes custom value)."""
        if not self.db:
            return False
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "DELETE FROM persona_context "
                "WHERE profile_id = ? AND persona_name = ? AND context_key = ?",
                (profile_id, persona_name, context_key)
            )
            self.db._commit()
            return True
        except Exception as e:
            logger.error(f"Delete context error: {e}")
            return False

    def list_context_keys(self, profile_id: int, persona_name: str) -> List[Dict[str, Any]]:
        """List all context keys for a persona."""
        if not self.db:
            return []
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "SELECT context_key, LENGTH(content) as size, updated_at "
                "FROM persona_context WHERE profile_id = ? AND persona_name = ? "
                "ORDER BY context_key",
                (profile_id, persona_name)
            )
            return [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            logger.error(f"List context keys error: {e}")
            return []

    # ── Version History ────────────────────────────────────

    def _save_version(self, profile_id: int, persona_name: str, new_content: str):
        """Save a version of system_context before overwriting."""
        ctx_dir = self._context_dir(profile_id, persona_name)
        versions_dir = ctx_dir / "_versions"
        versions_dir.mkdir(exist_ok=True)

        # Get current content
        current = self.get_context(profile_id, persona_name, 'system_context')
        if current and current.strip() and current != new_content:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_file = versions_dir / f"system_context_{ts}.md"
            version_file.write_text(current, encoding='utf-8')

            # Prune old versions
            versions = sorted(versions_dir.glob("system_context_*.md"), reverse=True)
            for old in versions[MAX_VERSIONS:]:
                old.unlink()

    def get_versions(self, profile_id: int, persona_name: str) -> List[Dict[str, str]]:
        """List version history for system_context."""
        ctx_dir = self._context_dir(profile_id, persona_name)
        versions_dir = ctx_dir / "_versions"
        if not versions_dir.exists():
            return []

        versions = []
        for f in sorted(versions_dir.glob("system_context_*.md"), reverse=True):
            versions.append({
                'filename': f.name,
                'timestamp': f.name.replace('system_context_', '').replace('.md', ''),
                'size': f.stat().st_size,
            })
        return versions[:MAX_VERSIONS]

    def revert_to_version(self, profile_id: int, persona_name: str,
                          version_filename: str) -> bool:
        """Revert system_context to a specific version."""
        # Validate filename to prevent path traversal
        if not version_filename or '/' in version_filename or '\\' in version_filename or '..' in version_filename:
            logger.warning(f"Invalid version filename rejected: {version_filename!r}")
            return False
        ctx_dir = self._context_dir(profile_id, persona_name)
        version_path = ctx_dir / "_versions" / version_filename
        if not version_path.exists():
            return False

        content = version_path.read_text(encoding='utf-8')
        return self.save_context(profile_id, persona_name, 'system_context', content)

    # ── Persona File Browser ───────────────────────────────

    def list_files(self, profile_id: int, persona_name: str,
                   subpath: str = '/') -> List[Dict[str, Any]]:
        """List files in a persona's context directory."""
        ctx_dir = self._context_dir(profile_id, persona_name)
        target = (ctx_dir / subpath.lstrip('/')).resolve()

        # Security: ensure target is under ctx_dir
        if not str(target).startswith(str(ctx_dir.resolve())):
            return []

        if not target.is_dir():
            return []

        entries = []
        for item in sorted(target.iterdir()):
            if item.name.startswith('.'):
                continue
            rel = str(item.relative_to(ctx_dir.resolve()))
            entries.append({
                'name': item.name,
                'path': '/' + rel,
                'type': 'directory' if item.is_dir() else 'file',
                'size': item.stat().st_size if item.is_file() else 0,
                'modified': datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
            })
        return entries

    def read_file(self, profile_id: int, persona_name: str,
                  filepath: str) -> Optional[str]:
        """Read a file from persona's context directory."""
        ctx_dir = self._context_dir(profile_id, persona_name)
        target = (ctx_dir / filepath.lstrip('/')).resolve()

        if not str(target).startswith(str(ctx_dir.resolve())):
            return None
        if not target.is_file():
            return None

        try:
            return target.read_text(encoding='utf-8', errors='replace')[:50000]
        except Exception as e:
            logger.error(f"Read file error ({target}): {e}")
            return None

    def write_file(self, profile_id: int, persona_name: str,
                   filepath: str, content: str) -> bool:
        """Write a file in persona's context directory."""
        ctx_dir = self._context_dir(profile_id, persona_name)
        target = (ctx_dir / filepath.lstrip('/')).resolve()

        if not str(target).startswith(str(ctx_dir.resolve())):
            return False

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
            return True
        except Exception as e:
            logger.error(f"Write file error: {e}")
            return False

    def delete_file(self, profile_id: int, persona_name: str,
                    filepath: str) -> bool:
        """Delete a file or directory from persona's context directory."""
        import shutil
        ctx_dir = self._context_dir(profile_id, persona_name)
        target = (ctx_dir / filepath.lstrip('/')).resolve()

        if not str(target).startswith(str(ctx_dir.resolve())):
            return False
        # Never delete the context root itself
        if target == ctx_dir.resolve():
            return False

        try:
            if target.is_file():
                target.unlink()
                return True
            elif target.is_dir():
                shutil.rmtree(target)
                return True
        except Exception as e:
            logger.error(f"Delete file error: {e}")
        return False

    def make_dir(self, profile_id: int, persona_name: str, dirpath: str) -> bool:
        """Create a directory in persona's context directory."""
        ctx_dir = self._context_dir(profile_id, persona_name)
        target = (ctx_dir / dirpath.lstrip('/')).resolve()

        if not str(target).startswith(str(ctx_dir.resolve())):
            return False

        try:
            target.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Make dir error: {e}")
            return False

    # ── Cross-persona search ───────────────────────────────

    def search_all_contexts(self, profile_id: int, query: str) -> List[Dict[str, Any]]:
        """Search across all persona contexts for a profile (UI-only, not agent tool)."""
        if not self.db:
            return []
        if not query or not query.strip():
            return []
        try:
            cursor = self.db.conn.cursor()
            # Escape LIKE wildcards in user input to prevent unintended pattern matching
            safe_query = query.replace('%', '\\%').replace('_', '\\_')
            cursor.execute(
                """SELECT persona_name, context_key, content, updated_at
                   FROM persona_context
                   WHERE profile_id = ? AND content LIKE ? ESCAPE '\\'
                   ORDER BY updated_at DESC""",
                (profile_id, f'%{safe_query}%')
            )
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                # Extract snippet around match
                content = d.get('content', '')
                idx = content.lower().find(query.lower())
                if idx >= 0:
                    start = max(0, idx - 100)
                    d['snippet'] = content[start:start + 300]
                results.append(d)
            return results
        except Exception as e:
            logger.error(f"Cross-persona search error: {e}")
            return []
