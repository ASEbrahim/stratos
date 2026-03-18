"""
RP Per-Turn Injection — 11 conditional hints injected before each user message.

Extracted from rp_chat.py for parallelization. Contains:
- _build_turn_injection() — the main injection builder
- All hint logic: situation, scenario, callback, format, tone, length,
  questions, pushback, anti-repeat, blend, openness
"""

import re
import logging
from collections import Counter

from routes.rp_prompt import (
    _ARCHETYPES, _detect_archetype, _get_secondary_archetype,
    _get_dialogue_tone, _get_archetype_format, _get_emotional_openness,
    _get_archetype_length, _should_ask_question,
)

logger = logging.getLogger("rp_injection")


def _build_turn_injection(history: list, card: dict, content: str,
                          user_turn: int, session_id: str, db) -> str:
    """Build the per-turn injection system message with 11 conditional hints.

    Returns a bracketed string like "[SITUATION: ... FORMAT: ... OPENNESS: ...]"
    suitable for inserting as a system message right before the user message.
    Used by chat, regenerate, and branch endpoints for consistent quality.
    """
    personality_text = card.get('personality', '') if card else ''
    arch_override = card.get('archetype_override') if card else None
    user_words = len(content.split())

    # ── Situation awareness ──
    situation_parts = []
    if len(history) >= 6:
        arcs = db.get_rp_context(session_id, tier=3, category="arc_summary", limit=1)
        if arcs:
            situation_parts.append(arcs[0]['value'])
        facts = db.get_rp_context(session_id, tier=1, limit=5)
        fact_items = [f"{f['key']}: {f['value']}" for f in facts
                      if f.get('category') != 'session' and f.get('value')]
        if fact_items:
            situation_parts.append("Known: " + ", ".join(fact_items[:5]))

    # ── Scenario reminder ──
    scenario_reminder = ""
    scenario_text = card.get('scenario', '') if card else ''
    if scenario_text:
        scenario_reminder = f"SETTING (mention something from this environment): {scenario_text[:150]}"

    # ── Callback hint ──
    callback_hint = ""
    if len(history) >= 8:
        lookback = history[max(0, len(history)-12):max(0, len(history)-4)]
        user_details = [m['content'] for m in lookback if m['role'] == 'user' and len(m['content']) > 15]
        if user_details:
            callback_hint = "CALLBACK: Reference something from earlier in the conversation."

    # ── Archetype-specific format rotation ──
    format_hint = _get_archetype_format(user_turn, personality_text)

    # ── Archetype-aware dialogue tone progression ──
    dialogue_tone = _get_dialogue_tone(user_turn, personality_text, content)

    # ── Length control — aggressive for short input, proportional otherwise ──
    archetype = _detect_archetype(personality_text, override=arch_override)
    arch_length = _get_archetype_length(personality_text)
    length_hint = ""
    if user_words <= 3:
        if arch_length == "short":
            length_hint = "LENGTH: 1-word input. Reply with ONE short sentence. Nothing more."
        else:
            length_hint = "LENGTH: 1-word input. Reply with 1 sentence MAX. No extra narration, no nurturing, no padding."
    elif user_words <= 5:
        if arch_length == "short":
            length_hint = "LENGTH: Very short input. Reply with 1 sentence MAX."
        else:
            length_hint = "LENGTH: Short input. Reply with 1-2 sentences MAX."
    elif user_words <= 10:
        if arch_length == "short":
            length_hint = "LENGTH: Keep it terse — match character's brevity."
        else:
            length_hint = "LENGTH: Match the user's brevity."

    # ── Question generation (archetype-aware) ──
    question_hint = ""
    should_ask, q_type = _should_ask_question(user_turn, personality_text)
    if should_ask:
        q_hints = {
            "curious": "END with a genuine question — you're curious about them.",
            "challenge": "END with a challenging question — test them, provoke them.",
            "tactical": "END with a tactical question — assess the situation.",
            "probing": "END with a probing question — you want to understand something specific.",
            "check_in": "END with a caring question — make sure they're okay.",
            "seeking": "END with a question seeking direction — what do they want?",
        }
        question_hint = q_hints.get(q_type, "END with a question to the user.")

    # ── Anti-sycophancy for confident/tough ──
    pushback_hint = ""
    archetype = _detect_archetype(personality_text, override=arch_override)
    arch_data = _ARCHETYPES.get(archetype, {})
    if arch_data.get("pushback") and user_turn >= 2:
        pushback_hint = "Don't just agree or flirt harder. Challenge them. Push back. Disagree. Have your own opinion. Say 'no' or 'you're wrong' if it fits."

    # ── Dynamic archetype blending ──
    blend_hint = ""
    secondary = _get_secondary_archetype(personality_text)
    if secondary and user_turn >= 6:
        blend_hints = {
            "shy": "Let unexpected shyness or vulnerability peek through.",
            "confident": "Let a flash of unexpected boldness break through.",
            "tough": "Let protective instincts surface unexpectedly.",
            "sweet": "Let unexpected tenderness surface.",
            "clinical": "Let analytical observation slip in.",
        }
        blend_hint = blend_hints.get(secondary, "")

    # ── Anti-repetition — dynamic phrase extraction from recent responses ──
    anti_repeat = ""
    if len(history) >= 4:
        recent_responses = [m['content'] for m in history[-6:] if m['role'] == 'assistant']
        if len(recent_responses) >= 2:
            all_actions = []
            for resp in recent_responses:
                actions = re.findall(r'\*([^*]{5,40})\*', resp)
                all_actions.extend(a.lower().strip() for a in actions)
            phrase_counts = Counter(all_actions)
            repeated = [phrase for phrase, count in phrase_counts.items() if count >= 2]
            if repeated:
                anti_repeat = f"You already used these descriptions recently: {', '.join(repeated[:4])}. Use COMPLETELY DIFFERENT physical details this time."

    # ── Dedup warning (from previous turn's detection) ──
    dedup_hint = ""
    dedup_flags = db.get_rp_context(session_id, tier=1, category="system", limit=1)
    for f in dedup_flags:
        if f.get('key') == 'dedup_warning' and f.get('value') == 'true':
            dedup_hint = "WARNING: Your last response repeated content from your previous message. This turn, write something COMPLETELY FRESH — no reused phrases, descriptions, or sentence structures."
            break

    # ── Emotional openness meter ──
    openness = _get_emotional_openness(user_turn, personality_text, content)

    # ── Assemble ──
    inject_parts = []
    if dedup_hint:
        inject_parts.append(dedup_hint)
    if situation_parts:
        inject_parts.append("SITUATION: " + " | ".join(situation_parts))
    if scenario_reminder:
        inject_parts.append(scenario_reminder)
    if callback_hint:
        inject_parts.append(callback_hint)
    inject_parts.append(f"FORMAT: {format_hint}")
    inject_parts.append(f"DIALOGUE TONE: {dialogue_tone}")
    if length_hint:
        inject_parts.append(length_hint)
    if question_hint:
        inject_parts.append(question_hint)
    if pushback_hint:
        inject_parts.append(pushback_hint)
    if anti_repeat:
        inject_parts.append(anti_repeat)
    if blend_hint:
        inject_parts.append(blend_hint)
    inject_parts.append(f"OPENNESS: {openness:.1f}/1.0 — {'fully guarded' if openness < 0.2 else 'mostly guarded' if openness < 0.4 else 'warming up' if openness < 0.6 else 'walls down' if openness < 0.8 else 'completely open'}")

    return "[" + ". ".join(inject_parts) + "]"
