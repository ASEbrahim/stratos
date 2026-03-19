"""
RP Meta-Controller — Haiku-powered narrative direction and parameter tuning.

Replaces formula-based injection with intelligent, context-aware parameters.
Returns both generation parameters AND behavioral goals in a single API call.
Single compressed line output (~50 tokens) keeps cost at ~$0.001/call.

Graceful degradation: if API key is missing or API fails, all systems
fall back to existing formula behavior with zero impact on core RP.
"""

import os
import re
import json
import logging
import requests
from typing import Optional

logger = logging.getLogger("rp_meta")

# ═══════════════════════════════════════════════════════════
# Compressed output schema
# ═══════════════════════════════════════════════════════════

PARAM_KEYS = ["openness", "initiative", "format", "length", "tone",
              "temperature", "speech", "pacing"]

# Continuous params blend smoothly; discrete params use API value directly
_CONTINUOUS_PARAMS = {"openness", "temperature"}

META_SYSTEM_PROMPT = """You are a narrative director for interactive roleplay. You control HOW a character responds by setting generation parameters. You do NOT write the response — a local AI model does that. You set the dials.

Respond with ONLY one line of space-separated values, then | then up to 2 short behavioral goals:
[openness 0.0-1.0] [initiative 0/1] [format 1=dialogue 2=action 3=narration] [length 1=brief 2=normal 3=detailed] [tone 1=guarded 2=warm 3=playful 4=intense 5=tender 6=cold] [temperature 0.72-0.92] [speech_enforcement 1=none 2=gentle 3=strict] [pacing 1=slow 2=natural 3=accelerate]|goal1|goal2

Rules:
- openness: how emotionally vulnerable the character is RIGHT NOW. 0.0 = stone wall. 1.0 = completely open. Earn it slowly — don't jump from 0.2 to 0.8 in one turn.
- initiative: 1 ONLY if the user's last message contained physical action AND the relationship is established. Otherwise 0.
- format: match the scene energy. Dialogue for conversation, action for physical scenes, narration for introspective moments.
- length: match the user's energy. Short user input = brief response. Emotional outpouring = detailed allowed.
- tone: the emotional coloring of the response. Not the character's personality — the tone of THIS specific moment.
- temperature: 0.78 is default. Lower for controlled/precise characters. Higher for chaotic/emotional moments.
- speech_enforcement: 3 for characters with distinctive speech patterns (archaic, terse). 1 for casual/normal speech.
- pacing: 1=slow the scene down (linger on moments). 3=advance the scene (something should change).
- Goals: whisper behavioral directions to the character. "Make them laugh." "Show vulnerability." "Test their patience.\""""

META_USER_TEMPLATE = """Character: {name} ({archetype})
Speech: {speech_excerpt}
Turn: {turn}
Previous params: {last_params}

Last 3 exchanges:
{compressed_history}

User's new message: {user_msg}"""


# ═══════════════════════════════════════════════════════════
# Gate — decide whether to call the API
# ═══════════════════════════════════════════════════════════

def should_call_meta(user_msg: str, last_assistant_msg: str, turn: int) -> bool:
    """Decide if this turn needs a Haiku meta-controller call."""
    # Always call on first real exchanges
    if turn <= 3:
        return True
    # Emotional keywords in user message
    if re.search(r'\b(love|hate|trust|scared|afraid|hurt|sorry|forgive|miss|promise|die|kill)\b', user_msg, re.I):
        return True
    # Physical action (asterisks with verbs)
    if re.search(r'\*.*\b(kiss|touch|grab|push|pull|hold|slap|hug|stroke|press)\b', user_msg, re.I):
        return True
    # Short message responding to a question or request
    if len(user_msg.split()) <= 5 and last_assistant_msg and (
        '?' in last_assistant_msg or
        re.search(r'\b(want|shall|come|let me|may I|ready|tell me)\b', last_assistant_msg, re.I)):
        return True
    # Periodic recalibration
    if turn % 8 == 0:
        return True
    return False


# ═══════════════════════════════════════════════════════════
# Parser — compressed output → structured dict
# ═══════════════════════════════════════════════════════════

def parse_meta_params(raw_line: str) -> Optional[dict]:
    """Parse compressed meta-controller output into a structured dict.

    Input: "0.2 0 1 1 1 0.78 2 1|Study them carefully|Test their resolve"
    Output: {"openness": 0.2, "initiative": 0, "format": 1, "length": 1,
             "tone": 1, "temperature": 0.78, "speech": 2, "pacing": 1,
             "goals": ["Study them carefully", "Test their resolve"]}
    """
    if not raw_line:
        return None

    parts = raw_line.split('|')
    param_str = parts[0].strip()
    goals = [g.strip() for g in parts[1:] if g.strip()]

    values = param_str.split()
    if len(values) < 8:
        return None

    try:
        params = {
            "openness": max(0.0, min(1.0, float(values[0]))),
            "initiative": int(float(values[1])),
            "format": max(1, min(3, int(float(values[2])))),
            "length": max(1, min(3, int(float(values[3])))),
            "tone": max(1, min(6, int(float(values[4])))),
            "temperature": max(0.72, min(0.92, float(values[5]))),
            "speech": max(1, min(3, int(float(values[6])))),
            "pacing": max(1, min(3, int(float(values[7])))),
        }
        params["goals"] = goals[:2]
        return params
    except (ValueError, IndexError):
        return None


# ═══════════════════════════════════════════════════════════
# Blending — merge API params with formula fallback
# ═══════════════════════════════════════════════════════════

def blend_params(api_params: Optional[dict], formula_params: dict,
                 api_weight: float = 0.6) -> dict:
    """Blend API params with formula fallback. API=None → 100% formula.

    Continuous params (openness, temperature) blend smoothly.
    Discrete params (format, tone, etc.) use API value directly.
    """
    if api_params is None:
        return formula_params
    blended = {}
    for key in formula_params:
        if key not in api_params:
            blended[key] = formula_params[key]
        elif key in _CONTINUOUS_PARAMS and isinstance(api_params[key], (int, float)):
            blended[key] = api_weight * api_params[key] + (1 - api_weight) * formula_params[key]
        else:
            blended[key] = api_params[key]
    return blended


# ═══════════════════════════════════════════════════════════
# Meta-controller API call
# ═══════════════════════════════════════════════════════════

def call_meta_controller(
    card: dict,
    history: list,
    user_msg: str,
    turn: int,
    archetype: str,
    session_id: str,
    db,
) -> Optional[dict]:
    """Call Haiku to get meta-controller parameters + goals.

    Returns parsed params dict or None on failure.
    Stores params and goals in rp_session_context for persistence.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    name = card.get("name", "Character") if card else "Character"
    speech = card.get("speech_pattern", "")[:100] if card else ""

    # Get last params from DB
    stored = db.get_rp_context(session_id, tier=2, category="meta_params", limit=1)
    last_params = stored[0]["value"] if stored else "none (first turn)"

    # Compress last 3 exchanges
    recent = history[-6:]
    compressed = ""
    for m in recent:
        role = "User" if m["role"] == "user" else name
        compressed += f"{role}: {m.get('content', '')[:150]}\n"

    user_prompt = META_USER_TEMPLATE.format(
        name=name,
        archetype=archetype,
        speech_excerpt=speech or "natural",
        turn=turn,
        last_params=last_params,
        compressed_history=compressed.strip() or "(new conversation)",
        user_msg=user_msg[:300],
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
                "max_tokens": 100,
                "system": META_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=8,
        )

        if r.status_code != 200:
            logger.warning(f"Meta controller API returned {r.status_code}: {r.text[:100]}")
            return None

        response_text = r.json().get("content", [{}])[0].get("text", "").strip()
        logger.info(f"Meta controller raw: {response_text}")

        params = parse_meta_params(response_text)
        if not params:
            logger.warning(f"Meta controller parse failed: {response_text[:100]}")
            return None

        # Store params in DB
        param_line = (f"{params['openness']:.2f} {params['initiative']} {params['format']} "
                      f"{params['length']} {params['tone']} {params['temperature']:.2f} "
                      f"{params['speech']} {params['pacing']}")
        db.upsert_rp_context(session_id, tier=2, category="meta_params",
                             key="current", value=param_line, turn_number=turn)

        # Store goals (reuses existing director_goal key)
        if params.get("goals"):
            db.upsert_rp_context(session_id, tier=2, category="director_goal",
                                 key="current_goals", value=json.dumps(params["goals"]),
                                 turn_number=turn)
            logger.info(f"Meta controller goals: {params['goals']}")

        logger.info(f"Meta controller: {param_line}")
        return params

    except requests.Timeout:
        logger.warning("Meta controller timed out (8s)")
        return None
    except Exception as e:
        logger.error(f"Meta controller error: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# DB reader — load stored meta params
# ═══════════════════════════════════════════════════════════

def get_stored_meta_params(session_id: str, db) -> Optional[dict]:
    """Load the most recent meta params from DB."""
    stored = db.get_rp_context(session_id, tier=2, category="meta_params", limit=1)
    if not stored:
        return None
    return parse_meta_params(stored[0]["value"])


# ═══════════════════════════════════════════════════════════
# Generic Haiku helper — reusable by rp_memory, character_cards
# ═══════════════════════════════════════════════════════════

def call_haiku(prompt: str, max_tokens: int = 200, system: str = None) -> Optional[str]:
    """Call Haiku API. Returns response text or None on failure.

    Used by rp_memory.py (fact extraction, arc summaries) and
    character_cards.py (enrichment, example dialogues) to replace
    local Qwen calls with Haiku for better quality.

    Graceful degradation: returns None if API key is missing or API fails,
    allowing callers to fall back to Ollama.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        payload = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system

        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        if r.status_code != 200:
            logger.warning(f"Haiku API returned {r.status_code}")
            return None
        return r.json().get("content", [{}])[0].get("text", "").strip()
    except requests.Timeout:
        logger.warning("Haiku API timed out (15s)")
        return None
    except Exception as e:
        logger.warning(f"Haiku API error: {e}")
        return None
