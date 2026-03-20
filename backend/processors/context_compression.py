"""
Context Compression — Auto-generated state.md and conversation log management.

Every persona accumulates context over time. This module:
  1. Logs conversations to daily JSONL files
  2. Generates state.md from recent conversations + user context
  3. Compresses older conversations into weekly summaries

state.md is the primary context the agent loads, NOT full raw history.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ContextCompressor:
    """Manages conversation logging and context compression for all personas."""

    def __init__(self, config: dict, db=None):
        self.base_dir = Path(config.get("system", {}).get("data_dir", "data")) / "users"
        self.db = db
        self.ollama_host = config.get("scoring", {}).get("ollama_host", "http://localhost:11434")
        self.model = config.get("scoring", {}).get("inference_model", "qwen3.5:9b")

    def _persona_dir(self, profile_id: int, persona_name: str) -> Path:
        d = self.base_dir / str(profile_id) / "context" / persona_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log_conversation(self, profile_id: int, persona_name: str,
                         messages: List[Dict[str, str]]) -> None:
        """Append conversation messages to today's JSONL log."""
        log_dir = self._persona_dir(profile_id, persona_name) / "conversation_log"
        log_dir.mkdir(exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{today}.jsonl"

        with open(log_file, 'a') as f:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "messages": messages,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def get_recent_conversations(self, profile_id: int, persona_name: str,
                                 days: int = 3) -> List[Dict]:
        """Load recent conversation entries from the last N days."""
        log_dir = self._persona_dir(profile_id, persona_name) / "conversation_log"
        if not log_dir.exists():
            return []

        entries = []
        for i in range(days):
            day = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            log_file = log_dir / f"{day}.jsonl"
            if log_file.exists():
                try:
                    with open(log_file) as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                entries.append(json.loads(line))
                except Exception as e:
                    logger.warning(f"Error reading conversation log {log_file}: {e}")

        return entries

    def get_state(self, profile_id: int, persona_name: str) -> str:
        """Read the current state.md for a persona. Returns empty string if none."""
        state_file = self._persona_dir(profile_id, persona_name) / "state.md"
        if state_file.exists():
            return state_file.read_text(errors='replace')
        return ""

    def save_state(self, profile_id: int, persona_name: str, content: str) -> None:
        """Write state.md for a persona."""
        state_file = self._persona_dir(profile_id, persona_name) / "state.md"
        state_file.write_text(content)

    def update_state(self, profile_id: int, persona_name: str,
                     conversation: List[Dict[str, str]]) -> str:
        """
        Update state.md after a conversation session.

        Uses LLM to summarize what changed and update the state document.
        Returns the updated state content.
        """
        import requests

        current_state = self.get_state(profile_id, persona_name)

        # Build conversation summary for the LLM
        conv_text = ""
        for msg in conversation[-10:]:  # Last 10 messages max
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                conv_text += f"{role.upper()}: {content[:500]}\n"

        if not conv_text.strip():
            return current_state

        # Get user context
        user_ctx = ""
        try:
            cursor = self.db.conn.cursor()
            cursor.execute(
                "SELECT content FROM persona_context WHERE profile_id = ? AND persona_name = ? AND context_key = 'system_context'",
                (profile_id, persona_name)
            )
            row = cursor.fetchone()
            if row:
                user_ctx = dict(row).get('content', '')[:500]
        except Exception as e:
            logger.warning(f"Failed to read user context for state update: {e}")

        system = f"""You are a context summarizer. Update the STATE document for the {persona_name} persona.

CURRENT STATE:
{current_state[:2000] if current_state else '(empty — this is the first state)'}

USER CONTEXT:
{user_ctx[:500] if user_ctx else '(none)'}

Write a concise state.md that captures:
- Key topics discussed
- Any decisions or conclusions reached
- Ongoing questions or threads
- For games: character states, scene, plot events
- For scholarly: research topics, findings, open questions

Keep it under 500 words. Use markdown headers. Be factual, not creative."""

        try:
            r = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"Update state based on this conversation:\n\n{conv_text}"}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 1024, "num_ctx": 8192},
                    "think": False,
                },
                timeout=60
            )
            if r.status_code == 200:
                result = r.json().get("message", {}).get("content", "")
                if result.strip():
                    self.save_state(profile_id, persona_name, result.strip())
                    return result.strip()
        except Exception as e:
            logger.warning(f"State update LLM call failed: {e}")

        return current_state

    def compress_week(self, profile_id: int, persona_name: str,
                      week_start: str) -> Optional[str]:
        """
        Compress a week's conversation logs into a weekly summary.

        week_start: ISO date string like "2026-03-03" (Monday of the week)
        """
        import requests

        log_dir = self._persona_dir(profile_id, persona_name) / "conversation_log"
        summaries_dir = self._persona_dir(profile_id, persona_name) / "summaries"
        summaries_dir.mkdir(exist_ok=True)

        summary_file = summaries_dir / f"week_{week_start}.md"
        if summary_file.exists():
            return summary_file.read_text()

        # Collect the week's conversations
        start_date = datetime.fromisoformat(week_start)
        all_text = []
        for i in range(7):
            day = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            log_file = log_dir / f"{day}.jsonl"
            if log_file.exists():
                try:
                    with open(log_file) as f:
                        for line in f:
                            entry = json.loads(line.strip())
                            for msg in entry.get("messages", []):
                                if msg.get("role") in ("user", "assistant"):
                                    all_text.append(f"{msg['role'].upper()}: {msg.get('content', '')[:300]}")
                except Exception as e:
                    logger.warning(f"Error reading conversation log {log_file} during compression: {e}")
                    continue

        if not all_text:
            return None

        # Truncate for LLM context
        combined = "\n".join(all_text)[:6000]

        try:
            r = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": f"Summarize this week's {persona_name} conversations into a concise weekly digest. Include key topics, decisions, and outcomes. Under 300 words. Use markdown."},
                        {"role": "user", "content": combined}
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 512, "num_ctx": 8192},
                    "think": False,
                },
                timeout=60
            )
            if r.status_code == 200:
                summary = r.json().get("message", {}).get("content", "")
                if summary.strip():
                    header = f"# Week of {week_start}\n\n"
                    full = header + summary.strip()
                    summary_file.write_text(full)
                    return full
        except Exception as e:
            logger.warning(f"Week compression failed: {e}")

        return None

    def get_summaries(self, profile_id: int, persona_name: str) -> List[Dict[str, str]]:
        """List available weekly summaries."""
        summaries_dir = self._persona_dir(profile_id, persona_name) / "summaries"
        if not summaries_dir.exists():
            return []

        result = []
        for f in sorted(summaries_dir.glob("week_*.md"), reverse=True):
            result.append({
                "week": f.stem.replace("week_", ""),
                "path": str(f),
                "size": f.stat().st_size,
            })
        return result
