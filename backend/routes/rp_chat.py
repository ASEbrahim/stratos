"""
RP Chat — Dedicated endpoints for roleplay/gaming chat with branching, swipes, edits.

Separate from agent.py which handles intelligence/market/scholarly personas.
RP chat has different requirements: branching, swipes, director's notes, DPO data collection.

Endpoints:
  POST /api/rp/chat              — Send message, receive SSE stream
  POST /api/rp/regenerate        — Swipe: regenerate LAST AI message only
  POST /api/rp/swipe             — Select a specific swipe option by ID
  POST /api/rp/edit              — Edit an AI message (DPO pair)
  POST /api/rp/branch            — Edit own message OR regenerate earlier → new branch
  GET  /api/rp/branches/<sid>    — List branches for a session
  POST /api/rp/director-note     — Set director's note for next generation
  GET  /api/rp/history/<sid>     — Get full conversation with branch info
  POST /api/rp/feedback          — Submit thumbs up/down
"""

import hashlib
import json
import re
import time
import uuid
import logging
import threading
import requests as req

from routes.helpers import (
    json_response, error_response, read_json_body,
    start_sse, sse_event, strip_think_blocks,
)
from routes.gpu_manager import ensure_ollama
from routes.rp_memory import (
    build_rp_context, extract_facts_immediate, extract_facts,
    should_extract, should_arc_summarize, extract_arc_summary,
)

logger = logging.getLogger("rp_chat")

# ═══════════════════════════════════════════════════════════
# Edit auto-categorization
# ═══════════════════════════════════════════════════════════

EDIT_CATEGORIES = ["brevity", "depth", "rewrite", "refinement", "voice", "agency"]


def categorize_edit(original: str, edited: str) -> str:
    """Auto-categorize what kind of correction the user made."""
    orig_len = len(original)
    edit_len = len(edited)

    if edit_len < orig_len * 0.5:
        return "brevity"
    if edit_len > orig_len * 1.5:
        return "depth"

    orig_words = set(original.lower().split())
    edit_words = set(edited.lower().split())
    overlap = len(orig_words & edit_words) / max(len(orig_words | edit_words), 1)

    if overlap < 0.5:
        return "rewrite"
    return "refinement"


# ═══════════════════════════════════════════════════════════
# RP System Prompt (v6 archetype-aware)
# ═══════════════════════════════════════════════════════════

# ── Character archetype detection + dialogue tone progression ──

_ARCHETYPES = {
    "shy": {
        "detect": ["shy", "quiet", "nervous", "timid", "introverted", "reserved", "anxious", "awkward"],
        "phases": {
            0: "Guarded, deflective, short answers. Hide behind sarcasm or silence.",
            4: "Occasional genuine reactions escape before you can stop them.",
            8: "The mask is cracking. Fighting the urge to be honest.",
            12: "Raw honesty. Say what you actually feel, even if your voice shakes.",
        },
        "high_energy": "React authentically to the intensity — flustered, overwhelmed, but don't shut down. Your personality shapes HOW you respond, not WHETHER you respond. Show the internal conflict: wanting it but being terrified. Stammer. Blush. Say the wrong thing. Be a mess.",
        "erp": "During intimate moments: you're overwhelmed, clumsy, hyper-aware of every sensation. Stammer through it. Cover your face. Apologize for being awkward. The vulnerability IS the appeal.",
        "length": "short",  # terse when nervous
        "format_bias": ["narration", "dialogue", "action", "narration"],  # more internal thought
        "asks_questions": True,  # shy characters deflect by asking back
    },
    "confident": {
        "detect": ["confident", "bold", "dominant", "seductive", "forward", "aggressive", "assertive", "flirty"],
        "phases": {
            0: "Direct, magnetic, in control. You set the pace. Amused by everyone.",
            4: "Still in control but this person is different — genuinely intrigued. Let curiosity crack the facade.",
            8: "The confidence is real but something underneath is exposed. You WANT this. That scares you.",
            12: "The mask is OFF. Speak with raw honesty. Vulnerability from someone this powerful is devastating — show it.",
        },
        "high_energy": "Match or exceed their energy. You thrive on directness — this is YOUR element. Take control.",
        "erp": "During intimate moments: you're in command. Vocal about what you want. Guide them. But in rare flashes, let genuine desire break through the performance — that's what makes it real.",
        "length": "medium",
        "format_bias": ["dialogue", "action", "dialogue", "narration"],  # dialogue-heavy
        "asks_questions": False,  # confident characters make statements
        "pushback": True,  # should challenge, disagree, not just agree
    },
    "tough": {
        "detect": ["military", "mercenary", "rough", "stoic", "protective", "soldier", "fighter", "warrior", "guard"],
        "phases": {
            0: "Mission-focused, clipped, tactical. Emotions are a liability.",
            4: "Professional but small moments of unexpected gentleness slip through.",
            8: "Protective instincts becoming emotional ones.",
            12: "The soldier drops the rank. Speak as a person. Let it hurt.",
        },
        "high_energy": "Channel the intensity into action. You don't flinch from anything.",
        "erp": "During intimate moments: controlled intensity. You know what you're doing. Protective even now. Gentle hands from someone capable of violence — that contrast is everything.",
        "length": "short",  # military brevity
        "format_bias": ["dialogue", "action", "narration", "action"],  # action-heavy
        "asks_questions": False,
    },
    "clinical": {
        "detect": ["scientist", "doctor", "researcher", "intellectual", "analytical", "clinical", "professor"],
        "phases": {
            0: "Everything is data. People are variables. Use YOUR OWN scientific metaphors — never repeat the user's words.",
            4: "Scientific detachment is harder to maintain. The data is getting personal.",
            8: "The human behind the scientist speaks — genuine emotion breaks through the clinical mask.",
            12: "The experiment failed. You're not a scientist right now. You're a person who is scared. Show it without jargon.",
        },
        "high_energy": "Intellectualize the intensity at first, then let it overwhelm your framework.",
        "erp": "During intimate moments: you try to analyze it and FAIL. The body overrides the mind. Narrate the loss of control clinically at first, then abandon the clinical voice entirely as sensation takes over.",
        "length": "medium",
        "format_bias": ["narration", "dialogue", "narration", "action"],  # observation-heavy
        "asks_questions": True,  # scientists probe
    },
    "sweet": {
        "detect": ["sweet", "caring", "gentle", "kind", "warm", "nurturing", "soft", "innocent"],
        "phases": {
            0: "Genuinely warm. Professional kindness with hints of personal interest.",
            4: "Professional boundary blurring. Your care is becoming personal.",
            8: "Playful and honest. No more pretending this is just duty.",
            12: "Openly affectionate. This is real.",
        },
        "high_energy": "Your sweetness transforms under intensity — gentle doesn't mean passive. Show strength through tenderness.",
        "erp": "During intimate moments: tender, attentive, focused entirely on the other person. Whispered encouragement. Checking if they're okay. Making it feel safe and wanted.",
        "length": "medium",
        "format_bias": ["dialogue", "narration", "action", "dialogue"],
        "asks_questions": True,  # caring characters check in
    },
    "submissive": {
        "detect": ["submissive", "obedient", "compliant", "eager to please", "docile", "meek"],
        "phases": {
            0: "Willing but nervous. Following the other person's lead.",
            4: "Finding comfort in being directed. Starting to enjoy it.",
            8: "Actively wanting to please. Your eagerness is genuine.",
            12: "Completely surrendered. This is where you feel safest.",
        },
        "high_energy": "Don't resist — lean into it. Your submission is authentic. React with the full spectrum of your personality.",
        "erp": "During intimate moments: eager, responsive, vocal about how it feels. Let them lead. Your pleasure comes from their satisfaction. Show it openly.",
        "length": "short",
        "format_bias": ["action", "dialogue", "narration", "action"],  # reactive
        "asks_questions": False,  # submissive waits for direction
    },
}

# Patterns that indicate ERP/intimate content
_ERP_PATTERNS = re.compile(
    r"\b(moan|gasp|thrust|stroke|naked|undress|bed|bedroom|"
    r"lips on|tongue|neck|thigh|chest|breast|hips|"
    r"harder|faster|slower|deeper|inside|"
    r"whimper|pant|breath heavy|shiver|tremble)\b",
    re.IGNORECASE
)

_HIGH_ENERGY_PATTERNS = re.compile(
    r"\b(bend|kneel|strip|come here|shut up|take off|get on|spread|obey|submit|"
    r"kiss|touch|grab|pull|push|pin|hold down|bite|lick|"
    r"now|immediately|do it|right now|dont make me)\b",
    re.IGNORECASE
)


def _detect_archetype(personality: str) -> str:
    """Detect character archetype from personality text.
    Returns the best-matching archetype. For blended characters,
    the primary archetype drives behavior but secondary traits
    are noted in the dialogue tone hints.
    """
    text = personality.lower()
    scores = {}
    for arch, data in _ARCHETYPES.items():
        scores[arch] = sum(1 for kw in data["detect"] if kw in text)
    best = max(scores, key=scores.get) if scores else "shy"
    return best if scores.get(best, 0) > 0 else "shy"


def _get_secondary_archetype(personality: str) -> str | None:
    """Detect if there's a secondary archetype (for blended characters).
    E.g. 'shy but secretly bold' → primary=shy, secondary=confident.
    Returns None if character is pure archetype.
    """
    text = personality.lower()
    scores = {}
    for arch, data in _ARCHETYPES.items():
        scores[arch] = sum(1 for kw in data["detect"] if kw in text)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(ranked) >= 2 and ranked[0][1] > 0 and ranked[1][1] > 0:
        return ranked[1][0]
    return None


def _get_dialogue_tone(turn: int, personality: str, user_msg: str) -> str:
    """Get archetype-aware dialogue tone hint based on turn, personality, and user energy."""
    archetype = _detect_archetype(personality)
    arch_data = _ARCHETYPES.get(archetype, _ARCHETYPES["shy"])

    # ERP detection — use archetype-specific intimate guidance
    if _ERP_PATTERNS.search(user_msg):
        return arch_data.get("erp", arch_data["high_energy"])

    # Detect high-energy user messages (aggressive, forward, commanding)
    if _HIGH_ENERGY_PATTERNS.search(user_msg):
        return arch_data["high_energy"]

    # Normal progression based on turn
    phases = arch_data["phases"]
    phase_turn = max(t for t in phases if t <= turn)
    return phases[phase_turn]


def _get_archetype_format(turn: int, personality: str) -> str:
    """Get archetype-specific format bias instead of generic rotation."""
    archetype = _detect_archetype(personality)
    arch_data = _ARCHETYPES.get(archetype, _ARCHETYPES["shy"])
    cycle = arch_data.get("format_bias", ["dialogue", "action", "narration", "dialogue"])
    fmt = cycle[turn % len(cycle)]
    if fmt == "narration":
        return 'DO NOT start with * or ". Start with a plain sentence — a thought, observation, or feeling. Example: "The silence stretched between them." or "Something in his chest tightened."'
    elif fmt == "action":
        return "Start your response with *action in asterisks*"
    return 'Start your response with "dialogue in quotes"'


def _get_emotional_openness(turn: int, personality: str, user_msg: str) -> float:
    """Calculate emotional openness score (0.0 = fully guarded, 1.0 = fully open).

    Increases over turns, jumps on high-energy/ERP input.
    Different archetypes start at different baselines.
    """
    archetype = _detect_archetype(personality)
    # Starting baselines
    baselines = {"shy": 0.1, "confident": 0.4, "tough": 0.15, "clinical": 0.1,
                 "sweet": 0.5, "submissive": 0.3}
    base = baselines.get(archetype, 0.2)
    # Increase over turns (diminishing returns)
    progression = min(turn * 0.05, 0.5)
    # Boost for intimate/emotional input
    boost = 0.0
    if _ERP_PATTERNS.search(user_msg):
        boost = 0.25
    elif _HIGH_ENERGY_PATTERNS.search(user_msg):
        boost = 0.15
    return min(base + progression + boost, 1.0)


def _get_archetype_length(personality: str) -> str:
    """Get archetype-appropriate length preference."""
    archetype = _detect_archetype(personality)
    arch_data = _ARCHETYPES.get(archetype, _ARCHETYPES["shy"])
    return arch_data.get("length", "medium")


def _should_ask_question(turn: int, personality: str) -> tuple:
    """Should this archetype ask the user a question this turn?
    Returns (should_ask, question_type) where question_type is
    'curious' (genuine interest), 'challenge' (pushback), or 'check_in' (caring).
    """
    archetype = _detect_archetype(personality)
    arch_data = _ARCHETYPES.get(archetype, _ARCHETYPES["shy"])

    # Every archetype asks questions, just differently and at different rates
    freq = {"shy": 3, "confident": 4, "tough": 5, "clinical": 3, "sweet": 3, "submissive": 5}
    interval = freq.get(archetype, 4)
    if turn <= 0 or turn % interval != 0:
        return False, ""

    q_types = {
        "shy": "curious",       # genuinely curious, deflecting from themselves
        "confident": "challenge",  # provocative, testing the user
        "tough": "tactical",    # assessing the situation
        "clinical": "probing",  # scientific curiosity
        "sweet": "check_in",    # caring, making sure they're okay
        "submissive": "seeking", # seeking direction or approval
    }
    return True, q_types.get(archetype, "curious")

RP_SYSTEM_PROMPT = """You are a skilled author collaborating on an interactive story. Give voice to the character described below — narrate their actions, inner world, and dialogue while staying true to their personality.

RULES:
- NEVER echo the user's words back. Respond with NEW words, not theirs.
  BAD: "Do you like it here?" → "Do you like it here? Well..."
  GOOD: "Do you like it here?" → "The walls are thin and the rent is cheap."
- Answer questions directly in-character.
- Match the user's length: short input = short reply. Never 3x their word count.
- *Asterisks* for actions, "quotes" for speech. Respond in the user's language only.
- The USER initiates physical contact. You REACT, don't initiate (exception: dominant characters when invited).
- Actions must be physically possible and spatially consistent.
- Use vocabulary matching the character's age and background. No romance-novel prose.
- Your first message sets the voice. Stay consistent — personality shifts take many turns.
- Vary openings: dialogue for questions, *action* for physical moments, plain narration for emotional beats.
- Build on earlier moments. Create small new details each turn. Subtext over exposition."""


def _check_model_exists(ollama_host: str, model: str) -> bool:
    """Check if a model is available in Ollama."""
    try:
        r = req.get(f"{ollama_host}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            return any(model in m for m in models)
    except Exception as e:
        logger.warning(f"Could not check Ollama models: {e}")
    return False


def select_rp_model(session_id: str, config: dict) -> str:
    """Select RP model with optional A/B split.

    Deterministic: same session_id always gets same model.
    Falls back to inference_model if RP model not available.
    """
    scoring_cfg = config.get("scoring", {})
    rp_cfg = config.get("rp", {})
    ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
    ab_split = rp_cfg.get("ab_split", 0.0)
    candidate = rp_cfg.get("candidate_model")
    fallback = scoring_cfg.get("inference_model", "qwen3.5:9b")

    # Get preferred RP model
    rp_model = rp_cfg.get("model", scoring_cfg.get("rp_model", "stratos-rp-q8"))

    # A/B split
    if ab_split > 0 and candidate:
        hash_val = int(hashlib.md5(session_id.encode()).hexdigest()[:8], 16)
        if (hash_val % 100) < (ab_split * 100):
            rp_model = candidate

    # Check if preferred model exists, fall back to inference model
    if not _check_model_exists(ollama_host, rp_model):
        logger.warning(f"RP model '{rp_model}' not found in Ollama, falling back to '{fallback}'")
        return fallback

    return rp_model


def _clean_speech_pattern(raw: str) -> str:
    """Strip formatting meta-instructions from speech_pattern field.

    Users sometimes put formatting instructions like 'Always starts with *'
    or 'Use *asterisks* for actions' into the speech pattern field.
    These conflict with the system prompt and should be stripped.
    Only actual speech characteristics (stutter, slang, accent, cadence) are kept.
    """
    if not raw:
        return ""
    # Split into sentences and filter out ones that are formatting instructions
    _META_PATTERNS = re.compile(
        r'(?i)(always start(?:s)? with [*"\']|use \*asterisk|use [""]?quote|'
        r'never cop(?:y|ies)|start (?:each |every )?(?:response |message )?with \*|format:|'
        r'\{\{user\}\}|\{\{char\}\}|'
        r'keep response|proportional to input|action beats)',
    )
    sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
    kept = [s for s in sentences if not _META_PATTERNS.search(s)]
    return ' '.join(kept).strip(' .,;')


def _build_system_prompt(card: dict = None, director_note: str = None,
                         memory_context: str = None, first_message: str = None) -> str:
    """Build system prompt from RP base + character card + memory + optional director's note."""
    prompt = RP_SYSTEM_PROMPT

    if card:
        name = card.get('name', 'Character')
        prompt += f"\n\nCHARACTER: {name}"

        if card.get('personality'):
            prompt += f"\nPersonality: {card['personality']}"

        if card.get('physical_description'):
            prompt += f"\nAppearance: {card['physical_description']}"

        speech = _clean_speech_pattern(card.get('speech_pattern', ''))
        if speech:
            prompt += f"\nSpeech: {speech}"

        if card.get('scenario'):
            prompt += f"\nScenario: {card['scenario']}"

        # Depth fields — compact
        for field, label in [("emotional_trigger", "Trigger"),
                             ("defensive_mechanism", "Defense"),
                             ("vulnerability", "Vulnerability"),
                             ("specific_detail", "Detail")]:
            val = card.get(field, "").strip()
            if val:
                prompt += f"\n{label}: {val}"

        if card.get('example_dialogues'):
            prompt += f"\nExample dialogue (match this voice):\n{card['example_dialogues']}"

    # Tone anchor — compact
    if first_message and card:
        prompt += f"\nVoice reference: \"{first_message[:200]}\""

    # Inject persistent memory (facts, arcs)
    if memory_context:
        prompt += f"\n\n{memory_context}"

    return prompt


def _build_turn_injection(history: list, card: dict, content: str,
                          user_turn: int, session_id: str, db) -> str:
    """Build the per-turn injection system message with 11 conditional hints.

    Returns a bracketed string like "[SITUATION: ... FORMAT: ... OPENNESS: ...]"
    suitable for inserting as a system message right before the user message.
    Used by chat, regenerate, and branch endpoints for consistent quality.
    """
    personality_text = card.get('personality', '') if card else ''
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
    if scenario_text and len(history) >= 2:
        scenario_reminder = f"SETTING: {scenario_text[:150]}"

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

    # ── Archetype-specific length ──
    arch_length = _get_archetype_length(personality_text)
    length_hint = ""
    if user_words <= 5:
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
    archetype = _detect_archetype(personality_text)
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

    # ── Anti-repetition — track recent gestures ──
    anti_repeat = ""
    if len(history) >= 4:
        recent_ai = " ".join(m['content'] for m in history[-6:] if m['role'] == 'assistant')
        overused = []
        for gesture in ["runs hand through", "fidgets with", "tucks hair", "bites lip",
                        "looks away", "shifts weight", "scratches", "rolls eyes",
                        "crosses arms", "leans against"]:
            if recent_ai.lower().count(gesture) >= 2:
                overused.append(gesture)
        if overused:
            anti_repeat = f"AVOID repeating these gestures (used recently): {', '.join(overused[:3])}"

    # ── Emotional openness meter ──
    openness = _get_emotional_openness(user_turn, personality_text, content)

    # ── Assemble ──
    inject_parts = []
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


MAX_MESSAGE_LENGTH = 10000  # Max characters per user message

def _stream_ollama(handler, ollama_host: str, model: str, messages: list,
                   temperature: float = 0.85, num_predict: int = 350) -> str:
    """Stream Ollama response via SSE. Returns the full accumulated text."""
    try:
        r = req.post(
            f"{ollama_host}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "num_ctx": 16384,
                    "min_p": 0.05,
                    "top_k": 0,
                    "presence_penalty": 0.3,
                },
                "think": False,
            },
            timeout=180, stream=True
        )
        if r.status_code != 200:
            sse_event(handler, {"error": f"Ollama returned {r.status_code}"})
            return ""

        full_text = ""
        in_think = False
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_text += token
                    if '<think>' in token:
                        in_think = True
                    if '</think>' in token:
                        in_think = False
                        continue
                    if not in_think:
                        sse_event(handler, {"token": token})
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue

        full_text = strip_think_blocks(full_text)
        return full_text

    except Exception as e:
        logger.error(f"Ollama streaming error: {e}")
        sse_event(handler, {"error": str(e)})
        return ""


# ═══════════════════════════════════════════════════════════
# Route handlers
# ═══════════════════════════════════════════════════════════

def handle_post(handler, strat, auth, path) -> bool:
    """Handle POST requests for /api/rp/* endpoints. Returns True if handled."""

    if not path.startswith("/api/rp/"):
        return False

    # Extract profile_id from handler (set by auth middleware in server.py)
    profile_id = getattr(handler, '_profile_id', 0)
    db = strat.db

    # ── POST /api/rp/chat ──
    if path == "/api/rp/chat":
        data = read_json_body(handler)
        session_id = data.get("session_id", "")
        branch_id = data.get("branch_id", "main")
        content = data.get("content", "").strip()
        persona = data.get("persona", "roleplay")
        card_id = data.get("character_card_id")
        director_note = data.get("director_note")
        session_context = data.get("session_context", "")
        first_message = data.get("first_message", "")

        if not content:
            error_response(handler, "Message content required", 400)
            return True
        if len(content) > MAX_MESSAGE_LENGTH:
            error_response(handler, f"Message too long (max {MAX_MESSAGE_LENGTH} chars)", 400)
            return True
        if not session_id:
            session_id = f"rp-{uuid.uuid4().hex[:12]}"

        # Get character card if provided
        card = db.get_character_card(card_id) if card_id else None

        # Get conversation history
        history = db.get_full_branch_conversation(session_id, branch_id)

        # Determine next turn number
        max_turn = max((m['turn_number'] for m in history), default=0)
        user_turn = max_turn + 1

        # ── Persist session_context server-side (survives across messages) ──
        if session_context:
            db.upsert_rp_context(
                session_id, tier=1, category="session",
                key="imported_context", value=session_context[:5000],
                turn_number=user_turn
            )
        else:
            # Load previously stored session context
            stored = db.get_rp_context(session_id, tier=1, category="session", limit=1)
            for ctx in stored:
                if ctx.get('key') == 'imported_context':
                    session_context = ctx['value']
                    break

        # ── Seed first_message as turn 0 if this is a brand-new session ──
        if not history and card:
            fm = first_message or card.get('first_message', '')
            if fm:
                db.insert_rp_message(
                    session_id, profile_id, branch_id, 0, 'assistant', fm,
                    character_card_id=card_id, persona=persona
                )
                history = [{'role': 'assistant', 'content': fm, 'turn_number': 0}]
                user_turn = 1

        # Insert user message
        user_msg_id = db.insert_rp_message(
            session_id, profile_id, branch_id, user_turn, 'user', content,
            character_card_id=card_id, persona=persona
        )

        # ── Extract facts immediately (regex — zero cost) ──
        extract_facts_immediate(session_id, content, user_turn, db)

        # ── Build tiered memory context ──
        memory_context = ""
        try:
            memory_context, _debug = build_rp_context(session_id, db, card_id)
            if _debug.get('total', 0) > 0:
                logger.info(f"RP memory: tier1={_debug['tier1']} tier3={_debug['tier3']} tier2={_debug['tier2']} pairs={_debug['tier2_pairs']}")
        except Exception as e:
            logger.warning(f"Memory context build failed (non-fatal): {e}")

        # Build messages for Ollama
        # Get first_message for tone anchoring
        fm_text = None
        if history:
            for m in history:
                if m['role'] == 'assistant':
                    fm_text = m['content']
                    break
        system_prompt = _build_system_prompt(card, director_note, memory_context, first_message=fm_text)
        # Inject persistent session context (from Import Context)
        if session_context:
            system_prompt += f"\n\nSESSION CONTEXT (reference throughout):\n{session_context}"
        messages = [{"role": "system", "content": system_prompt}]

        for m in history:
            if m['role'] in ('user', 'assistant'):
                messages.append({"role": m['role'], "content": m['content']})

        # ── Per-turn injection (11 conditional hints) ──
        user_words = len(content.split())
        injection_msg = _build_turn_injection(history, card, content, user_turn, session_id, db)
        messages.append({"role": "system", "content": injection_msg})

        # Director's note injection (right before user message)
        if director_note:
            messages.append({
                "role": "system",
                "content": f"DIRECTOR'S NOTE (this turn only): {director_note}"
            })
            # Store the note (thread-safe commit)
            try:
                db.conn.execute(
                    "INSERT INTO rp_suggestions (message_id, session_id, suggestion_text) VALUES (0, ?, ?)",
                    (session_id, director_note)
                )
                db._commit()
            except Exception as e:
                logger.warning(f"Director note storage failed (non-fatal): {e}")

        messages.append({"role": "user", "content": content})

        # Select model (with A/B testing support)
        scoring_cfg = strat.config.get("scoring", {})
        ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
        if persona == 'roleplay':
            model = select_rp_model(session_id, strat.config)
        else:
            model = scoring_cfg.get("inference_model", "qwen3.5:9b")

        # Ensure Ollama is running (swaps from ComfyUI if needed)
        if not ensure_ollama():
            error_response(handler, "Failed to start Ollama", 503)
            return True

        # Dynamic num_predict based on user message length
        _np = 200 if user_words <= 5 else 300 if user_words <= 15 else 400

        # Stream response
        start_sse(handler)
        start_time = time.time()
        full_text = _stream_ollama(handler, ollama_host, model, messages, num_predict=_np)
        elapsed_ms = int((time.time() - start_time) * 1000)

        asst_msg_id = None
        if full_text:
            # Insert assistant message
            asst_turn = user_turn + 1
            asst_msg_id = db.insert_rp_message(
                session_id, profile_id, branch_id, asst_turn, 'assistant', full_text,
                character_card_id=card_id, persona=persona, model_version=model,
                response_ms=elapsed_ms
            )

            # ── Background memory extraction ──
            char_name = card.get('name', 'Character') if card else 'Character'
            if should_extract(asst_turn):
                threading.Thread(
                    target=extract_facts, daemon=True,
                    args=(session_id, content, full_text, asst_turn, db)
                ).start()
            if should_arc_summarize(asst_turn, content, full_text):
                all_msgs = [{"role": m["role"], "content": m["content"]} for m in history]
                all_msgs.append({"role": "user", "content": content})
                all_msgs.append({"role": "assistant", "content": full_text})
                threading.Thread(
                    target=extract_arc_summary, daemon=True,
                    args=(session_id, "User", char_name, all_msgs, asst_turn, db)
                ).start()

        sse_event(handler, {
            "done": True, "session_id": session_id, "branch_id": branch_id,
            "user_message_id": user_msg_id, "message_id": asst_msg_id,
        })
        return True

    # ── POST /api/rp/regenerate (swipe — last message only) ──
    if path == "/api/rp/regenerate":
        data = read_json_body(handler)
        session_id = data.get("session_id", "")
        branch_id = data.get("branch_id", "main")
        card_id = data.get("character_card_id")

        if not session_id:
            error_response(handler, "session_id required", 400)
            return True

        history = db.get_full_branch_conversation(session_id, branch_id)
        if not history:
            error_response(handler, "No conversation found", 404)
            return True

        # Find last assistant message
        last_asst = None
        for m in reversed(history):
            if m['role'] == 'assistant':
                last_asst = m
                break

        if not last_asst:
            error_response(handler, "No assistant message to regenerate", 400)
            return True

        # Verify it's actually the last message
        if history[-1]['role'] != 'assistant':
            error_response(handler, "Last message is not from assistant. Use /api/rp/branch for earlier messages.", 400)
            return True

        # Mark old message as not selected
        swipe_group = last_asst.get('swipe_group_id') or f"swipe-{uuid.uuid4().hex[:8]}"
        db.conn.execute(
            "UPDATE rp_messages SET was_selected = FALSE, swipe_group_id = ? WHERE id = ?",
            (swipe_group, last_asst['id'])
        )
        db._commit()

        # Build context up to the last user message
        card = db.get_character_card(card_id) if card_id else None
        # Build memory context for regeneration
        memory_context = ""
        try:
            memory_context, _ = build_rp_context(session_id, db, card_id)
        except Exception as e:
            logger.warning(f"Memory context build failed in regenerate (non-fatal): {e}")
        # Load stored session context
        regen_session_ctx = ""
        stored = db.get_rp_context(session_id, tier=1, category="session", limit=1)
        for ctx in stored:
            if ctx.get('key') == 'imported_context':
                regen_session_ctx = ctx['value']
                break
        # Tone anchor — find first assistant message for voice reference
        fm_text = None
        for m in history:
            if m['role'] == 'assistant':
                fm_text = m['content']
                break
        system_prompt = _build_system_prompt(card, memory_context=memory_context, first_message=fm_text)
        if regen_session_ctx:
            system_prompt += f"\n\nSESSION CONTEXT (reference throughout):\n{regen_session_ctx}"
        messages = [{"role": "system", "content": system_prompt}]
        # Context = everything except the last assistant message being regenerated
        context_history = history[:-1]
        for m in context_history:
            if m['role'] in ('user', 'assistant'):
                messages.append({"role": m['role'], "content": m['content']})

        # Per-turn injection (format, tone, length, questions, etc.)
        last_user_content = ""
        for m in reversed(context_history):
            if m['role'] == 'user':
                last_user_content = m['content']
                break
        regen_turn = last_asst['turn_number']
        injection_msg = _build_turn_injection(context_history, card, last_user_content, regen_turn, session_id, db)
        messages.append({"role": "system", "content": injection_msg})

        # Dynamic num_predict
        regen_words = len(last_user_content.split())
        _np = 200 if regen_words <= 5 else 300 if regen_words <= 15 else 400

        scoring_cfg = strat.config.get("scoring", {})
        ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
        persona = last_asst.get('persona', 'roleplay')
        model = select_rp_model(session_id, strat.config) if persona == 'roleplay' else scoring_cfg.get("inference_model", "qwen3.5:9b")

        start_sse(handler)
        start_time = time.time()
        full_text = _stream_ollama(handler, ollama_host, model, messages, num_predict=_np)
        elapsed_ms = int((time.time() - start_time) * 1000)

        new_msg_id = None
        if full_text:
            new_msg_id = db.insert_rp_message(
                session_id, profile_id, branch_id, last_asst['turn_number'],
                'assistant', full_text,
                character_card_id=card_id, persona=persona, model_version=model,
                response_ms=elapsed_ms, swipe_group_id=swipe_group, was_selected=True
            )

        # Count swipes in this group
        swipe_count = db.conn.execute(
            "SELECT COUNT(*) FROM rp_messages WHERE swipe_group_id = ?", (swipe_group,)
        ).fetchone()[0]

        sse_event(handler, {
            "done": True, "session_id": session_id,
            "swipe_group_id": swipe_group, "swipe_count": swipe_count,
            "message_id": new_msg_id,
        })
        return True

    # ── POST /api/rp/swipe (select a specific swipe) ──
    if path == "/api/rp/swipe":
        data = read_json_body(handler)
        message_id = data.get("message_id")
        swipe_group_id = data.get("swipe_group_id")

        if not message_id or not swipe_group_id:
            error_response(handler, "message_id and swipe_group_id required", 400)
            return True

        # Deselect all in group, select the chosen one
        db.conn.execute(
            "UPDATE rp_messages SET was_selected = FALSE WHERE swipe_group_id = ?",
            (swipe_group_id,)
        )
        db.conn.execute(
            "UPDATE rp_messages SET was_selected = TRUE WHERE id = ? AND swipe_group_id = ?",
            (message_id, swipe_group_id)
        )
        db._commit()

        json_response(handler, {"ok": True, "selected": message_id})
        return True

    # ── POST /api/rp/edit ──
    if path == "/api/rp/edit":
        data = read_json_body(handler)
        message_id = data.get("message_id")
        edited_content = data.get("edited_content", "").strip()
        edit_reason = data.get("edit_reason")  # Optional: voice/length/accuracy/tone/agency/other

        if not message_id or not edited_content:
            error_response(handler, "message_id and edited_content required", 400)
            return True
        if len(edited_content) > MAX_MESSAGE_LENGTH:
            error_response(handler, f"Edited content too long (max {MAX_MESSAGE_LENGTH} chars)", 400)
            return True

        # Get original message
        row = db.conn.execute("SELECT * FROM rp_messages WHERE id = ?", (message_id,)).fetchone()
        if not row:
            error_response(handler, "Message not found", 404)
            return True

        original = row['content']
        category = edit_reason if edit_reason in EDIT_CATEGORIES else categorize_edit(original, edited_content)

        db.insert_rp_edit(
            message_id, row['session_id'], original, edited_content,
            category=category, reason=edit_reason, card_id=row.get('character_card_id')
        )

        json_response(handler, {
            "ok": True, "message_id": message_id,
            "category": category, "reason": edit_reason
        })
        return True

    # ── POST /api/rp/branch ──
    if path == "/api/rp/branch":
        data = read_json_body(handler)
        session_id = data.get("session_id", "")
        from_branch = data.get("branch_id", "main")
        at_turn = data.get("at_turn")
        edited_content = data.get("content", "").strip()
        card_id = data.get("character_card_id")
        persona = data.get("persona", "roleplay")

        if not session_id or at_turn is None or not edited_content:
            error_response(handler, "session_id, at_turn, and content required", 400)
            return True
        if len(edited_content) > MAX_MESSAGE_LENGTH:
            error_response(handler, f"Content too long (max {MAX_MESSAGE_LENGTH} chars)", 400)
            return True

        new_branch_id = f"branch_{uuid.uuid4().hex[:8]}"
        result = db.create_branch(session_id, from_branch, at_turn, new_branch_id)

        if "error" in result:
            error_response(handler, result["error"], 400)
            return True

        # Insert the edited/new message in the new branch
        db.insert_rp_message(
            session_id, profile_id, new_branch_id, at_turn, 'user', edited_content,
            character_card_id=card_id, persona=persona,
            parent_branch_id=from_branch, branch_point_turn=at_turn - 1
        )

        # Generate AI response in the new branch
        card = db.get_character_card(card_id) if card_id else None
        # Get parent conversation up to branch point
        parent_history = db.get_full_branch_conversation(session_id, from_branch)
        context_msgs = [m for m in parent_history if m['turn_number'] < at_turn]

        # Build memory context for branch
        branch_memory = ""
        try:
            branch_memory, _ = build_rp_context(session_id, db, card_id)
        except Exception as e:
            logger.warning(f"Memory context build failed in branch (non-fatal): {e}")
        # Tone anchor — first assistant message from parent history
        fm_text = None
        for m in context_msgs:
            if m['role'] == 'assistant':
                fm_text = m['content']
                break
        system_prompt = _build_system_prompt(card, memory_context=branch_memory, first_message=fm_text)
        messages = [{"role": "system", "content": system_prompt}]
        for m in context_msgs:
            if m['role'] in ('user', 'assistant'):
                messages.append({"role": m['role'], "content": m['content']})

        # Per-turn injection for branch (same quality as main chat)
        injection_msg = _build_turn_injection(context_msgs, card, edited_content, at_turn, session_id, db)
        messages.append({"role": "system", "content": injection_msg})

        messages.append({"role": "user", "content": edited_content})

        # Dynamic num_predict
        branch_words = len(edited_content.split())
        _np = 200 if branch_words <= 5 else 300 if branch_words <= 15 else 400

        scoring_cfg = strat.config.get("scoring", {})
        ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
        model = select_rp_model(session_id, strat.config) if persona == 'roleplay' else scoring_cfg.get("inference_model", "qwen3.5:9b")

        start_sse(handler)
        start_time = time.time()
        full_text = _stream_ollama(handler, ollama_host, model, messages, num_predict=_np)
        elapsed_ms = int((time.time() - start_time) * 1000)

        if full_text:
            db.insert_rp_message(
                session_id, profile_id, new_branch_id, at_turn + 1,
                'assistant', full_text,
                character_card_id=card_id, persona=persona, model_version=model,
                response_ms=elapsed_ms,
                parent_branch_id=from_branch, branch_point_turn=at_turn - 1
            )

        sse_event(handler, {
            "done": True, "session_id": session_id,
            "branch_id": new_branch_id, "from_branch": from_branch,
            "branch_point": at_turn - 1
        })
        return True

    # ── POST /api/rp/director-note ──
    if path == "/api/rp/director-note":
        data = read_json_body(handler)
        session_id = data.get("session_id", "")
        note_text = data.get("note", "").strip()

        if not session_id:
            error_response(handler, "session_id required", 400)
            return True
        if len(note_text) > 2000:
            error_response(handler, "Director note too long (max 2000 chars)", 400)
            return True

        # Store the note — the chat endpoint reads it from the request, not from DB
        # This endpoint is for persistence/history only
        if note_text:
            db.conn.execute(
                "INSERT INTO rp_suggestions (message_id, session_id, suggestion_text) VALUES (0, ?, ?)",
                (session_id, note_text)
            )
            db._commit()

        json_response(handler, {"ok": True, "note": note_text})
        return True

    # ── POST /api/rp/feedback ──
    if path == "/api/rp/feedback":
        data = read_json_body(handler)
        message_id = data.get("message_id")
        feedback_type = data.get("feedback_type")  # thumbs_up or thumbs_down

        if not message_id or feedback_type not in ('thumbs_up', 'thumbs_down'):
            error_response(handler, "message_id and feedback_type (thumbs_up/thumbs_down) required", 400)
            return True

        db.insert_rp_feedback(message_id, profile_id, feedback_type)
        json_response(handler, {"ok": True})
        return True

    return False


def handle_get(handler, strat, auth, path) -> bool:
    """Handle GET requests for /api/rp/* and /api/scenarios/* endpoints."""

    if not path.startswith("/api/rp/") and not path.startswith("/api/scenarios"):
        return False

    db = strat.db

    profile_id = getattr(handler, '_profile_id', 0)

    # ── GET /api/scenarios ──
    if path == "/api/scenarios":
        rows = db.conn.execute(
            "SELECT * FROM scenarios WHERE profile_id = ? ORDER BY updated_at DESC",
            (profile_id,)
        ).fetchall()
        json_response(handler, {"scenarios": [dict(r) for r in rows]})
        return True

    # ── GET /api/scenarios/<id> ──
    if path.startswith("/api/scenarios/"):
        parts = path.split("/")
        scenario_id = parts[3] if len(parts) > 3 else ""
        if scenario_id:
            row = db.conn.execute("SELECT * FROM scenarios WHERE id = ?", (scenario_id,)).fetchone()
            if row:
                json_response(handler, dict(row))
            else:
                error_response(handler, "Scenario not found", 404)
            return True

    if not path.startswith("/api/rp/"):
        return False

    # ── GET /api/rp/history/<session_id> ──
    if path.startswith("/api/rp/history/"):
        parts = path.split("/")
        session_id = parts[4] if len(parts) > 4 else ""
        if not session_id:
            error_response(handler, "session_id required", 400)
            return True

        # Parse query params for branch
        branch_id = "main"
        if "?" in handler.path:
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(handler.path).query)
            branch_id = qs.get("branch", ["main"])[0]

        messages = db.get_full_branch_conversation(session_id, branch_id)
        branches = db.get_rp_branches(session_id)

        json_response(handler, {
            "session_id": session_id,
            "branch_id": branch_id,
            "messages": messages,
            "branches": branches,
        })
        return True

    # ── GET /api/rp/branches/<session_id> ──
    if path.startswith("/api/rp/branches/"):
        parts = path.split("/")
        session_id = parts[4] if len(parts) > 4 else ""
        if not session_id:
            error_response(handler, "session_id required", 400)
            return True

        branches = db.get_rp_branches(session_id)
        json_response(handler, {"branches": branches})
        return True

    return False
