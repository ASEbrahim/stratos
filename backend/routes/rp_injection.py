"""
RP Per-Turn Injection — 11 conditional hints injected before each user message.

Extracted from rp_chat.py for parallelization. Contains:
- _build_turn_injection() — the main injection builder
- All hint logic: situation, scenario, callback, format, tone, length,
  questions, pushback, anti-repeat, blend, openness
"""

import json
import re
import logging
from collections import Counter

from routes.rp_prompt import (
    _ARCHETYPES, _detect_archetype, _get_secondary_archetype,
    _get_dialogue_tone, _get_archetype_format, _get_emotional_openness,
    _get_archetype_length, _should_ask_question,
    _ERP_PATTERNS, _HIGH_ENERGY_PATTERNS,
)
from routes.rp_meta import parse_meta_params

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

    # ── Load meta params (set by rp_meta.py meta-controller) ──
    meta_params = None
    stored_meta = db.get_rp_context(session_id, tier=2, category="meta_params", limit=1)
    if stored_meta:
        meta_params = parse_meta_params(stored_meta[0]['value'])

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

    # ── Format rotation (meta-controller or archetype formula) ──
    if meta_params:
        _fmt = meta_params.get('format', 1)
        if _fmt == 3:
            format_hint = 'DO NOT start with * or ". Start with a plain sentence — a thought, observation, or feeling.'
        elif _fmt == 2:
            format_hint = "Start your response with *action in asterisks*"
        else:
            format_hint = 'Start your response with "dialogue in quotes"'
    else:
        format_hint = _get_archetype_format(user_turn, personality_text)

    # ── Narration style constraint (separate from narration_pov which is about person) ──
    narration_style = card.get('narration_style') if card else None
    if narration_style == 'cinematic':
        format_hint += ' NARRATION RULE: *actions* and "dialogue" ONLY. No internal monologue, no narrator voice, no plain prose.'
    elif narration_style == 'script':
        format_hint += ' NARRATION RULE: Dialogue is primary. Minimal *action* beats. No internal monologue. What is SAID and DONE, not thought.'

    # ── Dialogue tone (meta-controller or archetype formula) ──
    if meta_params:
        _tone = meta_params.get('tone', 2)
        _TONE_HINTS = {
            1: "Keep your guard up. Measured responses, nothing freely given.",
            2: "Warm but not overly so. Let genuine interest show.",
            3: "Light, teasing energy. Let playfulness guide this moment.",
            4: "Raw, unfiltered intensity. Don't hold back.",
            5: "Tender, gentle, close. This is a quiet, vulnerable moment.",
            6: "Cold, distant, controlled. Hold them at arm's length.",
        }
        dialogue_tone = _TONE_HINTS.get(_tone, _TONE_HINTS[2])
    else:
        dialogue_tone = _get_dialogue_tone(user_turn, personality_text, content)

    # ── Length control (meta-controller or formula) ──
    archetype = _detect_archetype(personality_text, override=arch_override)
    length_hint = ""
    if meta_params:
        _len = meta_params.get('length', 2)
        if _len == 1:
            length_hint = "LENGTH: Keep it brief — 1-2 sentences max."
        elif _len == 3:
            length_hint = "LENGTH: Take your time — detailed, immersive response."
        # _len == 2 → no constraint (normal)
    else:
        arch_length = _get_archetype_length(personality_text)
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

    # ── Gagged/silenced speech detection ──
    gag_hint = ""
    gag_patterns = re.compile(r'\b(cover(?:s|ed)? (?:her|his|their) mouth|gag(?:s|ged)?|muffle|shut (?:her|his|their) mouth|hand over (?:her|his|their) mouth|clamp(?:s|ed)? (?:her|his|their) mouth)\b', re.I)
    if gag_patterns.search(content):
        gag_hint = "CHARACTER CANNOT SPEAK clearly. Only muffled sounds ('mmph', 'nngh'), body language, and internal thoughts. NO clear dialogue this turn."

    # ── Intensity length ceiling (only when no meta — meta already handles intensity) ──
    if not length_hint and not meta_params:
        is_erp = bool(_ERP_PATTERNS.search(content))
        is_high = bool(_HIGH_ENERGY_PATTERNS.search(content))
        if is_erp or is_high:
            length_hint = "LENGTH: Intense scene — be vivid and specific, but stay under 150 words. Intensity = sharper detail, NOT more words."

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

    # ── Anti-sycophancy (meta tone=cold or archetype pushback) ──
    pushback_hint = ""
    if meta_params:
        if meta_params.get('tone') == 6 and user_turn >= 2:
            pushback_hint = "Don't just agree or flirt harder. Challenge them. Push back. Disagree. Have your own opinion."
    else:
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

    # ── Anti-repetition — expanded n-gram phrase extraction ──
    anti_repeat = ""
    if len(history) >= 4:
        recent_responses = [m['content'] for m in history[-6:] if m['role'] == 'assistant']
        if len(recent_responses) >= 2:
            all_phrases = []
            for resp in recent_responses:
                clean = resp.replace('*', '').replace('"', '').lower()
                words = clean.split()
                for i in range(len(words) - 3):
                    phrase = ' '.join(words[i:i+4])
                    if len(phrase) > 15:  # skip trivial phrases
                        all_phrases.append(phrase)
            phrase_counts = Counter(all_phrases)
            repeated = [phrase for phrase, count in phrase_counts.items() if count >= 2]
            if repeated:
                top_repeated = sorted(repeated, key=lambda p: phrase_counts[p], reverse=True)[:6]
                anti_repeat = f"BANNED PHRASES (do NOT use these — find completely different words): {', '.join(top_repeated)}"

    # ── Dedup warning (from previous turn's detection) ──
    dedup_hint = ""
    dedup_flags = db.get_rp_context(session_id, tier=1, category="system", limit=1)
    for f in dedup_flags:
        if f.get('key') == 'dedup_warning' and f.get('value') == 'true':
            dedup_hint = "WARNING: Your last response repeated content from your previous message. This turn, write something COMPLETELY FRESH — no reused phrases, descriptions, or sentence structures."
            break

    # ── Goal Director (loaded from rp_context, generated async by rp_director.py) ──
    goal_hint = ""
    stored_goals = db.get_rp_context(session_id, tier=2, category="director_goal", limit=1)
    if stored_goals:
        try:
            current_goals = json.loads(stored_goals[0].get('value', '[]'))
            if current_goals:
                goal_hint = f"GOAL: {current_goals[0]}"
                if len(current_goals) > 1:
                    goal_hint += f" Secondary: {current_goals[1]}"
        except (json.JSONDecodeError, IndexError):
            pass

    # ── Emotional openness (meta-controller or formula) ──
    if meta_params:
        openness = meta_params.get('openness', 0.1)
    else:
        openness = _get_emotional_openness(user_turn, personality_text, content, history=history)

    # ── Pacing hint (meta only) ──
    pacing_hint = ""
    if meta_params:
        _pacing = meta_params.get('pacing', 2)
        if _pacing == 1:
            pacing_hint = "PACING: Slow this moment down. Linger on details, sensations, micro-expressions."
        elif _pacing == 3:
            pacing_hint = "PACING: Advance the scene. Something should change or happen."

    # ── Speech enforcement hint (meta only) ──
    speech_hint = ""
    if meta_params:
        _speech = meta_params.get('speech', 1)
        if _speech == 3:
            speech_hint = "SPEECH: Enforce this character's distinctive speech pattern strictly. Every line must sound like them."
        elif _speech == 2:
            speech_hint = "SPEECH: Maintain the character's speech pattern. Light enforcement."

    # ── Assemble ──
    inject_parts = []
    if dedup_hint:
        inject_parts.append(dedup_hint)
    if gag_hint:
        inject_parts.append(gag_hint)
    if goal_hint:
        inject_parts.append(goal_hint)
    if situation_parts:
        inject_parts.append("SITUATION: " + " | ".join(situation_parts))
    if scenario_reminder:
        inject_parts.append(scenario_reminder)
    if callback_hint:
        inject_parts.append(callback_hint)
    format_preamble = '*asterisks* for ALL actions/narration. "quotes" for ALL speech. Never use asterisks for dialogue. Never use bare prose for actions.'
    inject_parts.append(f"FORMAT: {format_preamble} This turn: {format_hint}")
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
    if pacing_hint:
        inject_parts.append(pacing_hint)
    if speech_hint:
        inject_parts.append(speech_hint)
    inject_parts.append(f"OPENNESS: {openness:.1f}/1.0 — {'fully guarded' if openness < 0.2 else 'mostly guarded' if openness < 0.4 else 'warming up' if openness < 0.6 else 'walls down' if openness < 0.8 else 'completely open'}")

    return "[" + ". ".join(inject_parts) + "]"
