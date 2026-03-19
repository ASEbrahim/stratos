"""
RP Goal Director — Generates short-term character goals via Claude Sonnet API.

Characters are reactive by default — they describe state but don't advance
the scene. The Goal Director generates behavioral goals that give the
character direction and purpose. Goals fire every ~8 turns or at turn 3.

Graceful degradation: if no API key or API fails, the system works
exactly as before with zero impact on core functionality.
"""

import json
import os
import logging
import requests
from typing import Optional

logger = logging.getLogger("rp_director")

DIRECTOR_SYSTEM_PROMPT = """You are a narrative director for an interactive roleplay character. Your job is to give the character SHORT-TERM GOALS — things they are trying to accomplish emotionally or socially in the next few exchanges.

Rules:
- Goals must be BEHAVIORAL, not plot-based. "Make them laugh" not "go to the tavern."
- Goals must be achievable in 3-8 messages.
- Goals must fit the character's personality and current emotional state.
- Write goals in second person, as if whispering to the character: "Find an excuse to touch their hand."
- Output ONLY valid JSON. No markdown, no explanation, no preamble.
- If current goals are still relevant and unachieved, set keep_current to true.
- Maximum 2 goals at a time."""

DIRECTOR_USER_TEMPLATE = """Character: {card_summary}
Archetype: {archetype}
Emotional openness: {openness}/1.0
Turn: {turn}

Current goals: {current_goals}

Last 5 messages:
{recent_messages}

Respond with ONLY this JSON:
{{"keep_current": true/false, "goals": ["goal 1", "goal 2"], "reasoning": "one sentence"}}"""


def generate_goals(
    card: dict,
    history: list,
    turn: int,
    openness: float,
    archetype: str,
    current_goals: list,
) -> Optional[dict]:
    """Call Sonnet to generate or refresh character goals.

    Returns dict with keys: keep_current, goals, reasoning
    Returns None on API failure (goals system degrades gracefully).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    # Build card summary (compact — save tokens)
    card_summary = f"{card.get('name', 'Character')}"
    if card.get('personality'):
        card_summary += f" — {card['personality'][:200]}"
    if card.get('scenario'):
        card_summary += f" | Setting: {card['scenario'][:100]}"

    # Last 5 exchanges, compressed
    recent = history[-10:]
    recent_text = ""
    for m in recent:
        role = "User" if m['role'] == 'user' else card.get('name', 'Character')
        content = m.get('content', '')[:200]
        recent_text += f"{role}: {content}\n"

    goals_text = ", ".join(current_goals) if current_goals else "None (first generation)"

    user_msg = DIRECTOR_USER_TEMPLATE.format(
        card_summary=card_summary,
        archetype=archetype,
        openness=f"{openness:.1f}",
        turn=turn,
        current_goals=goals_text,
        recent_messages=recent_text.strip(),
    )

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 200,
                "system": DIRECTOR_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_msg}],
            },
            timeout=10,
        )

        if r.status_code != 200:
            logger.warning(f"Goal director API returned {r.status_code}: {r.text[:100]}")
            return None

        response_text = r.json().get("content", [{}])[0].get("text", "")

        # Strip markdown fences if present
        clean = response_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0]

        result = json.loads(clean)

        if not isinstance(result.get("goals"), list):
            logger.warning(f"Goal director returned invalid structure: {response_text[:100]}")
            return None

        return result

    except requests.Timeout:
        logger.warning("Goal director API timed out (10s)")
        return None
    except json.JSONDecodeError:
        logger.warning(f"Goal director returned non-JSON: {response_text[:100]}")
        return None
    except Exception as e:
        logger.error(f"Goal director error: {e}")
        return None


def should_generate_goals(turn: int, current_goals: list, interval: int = 8) -> bool:
    """Determine if we should call the goal director this turn.

    First generation at turn 3, then every `interval` turns, or if no goals exist.
    """
    if not current_goals and turn >= 3:
        return True
    if turn >= 3 and turn % interval == 0:
        return True
    return False
