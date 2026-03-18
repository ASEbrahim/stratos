"""
RP Prompt Engineering — Archetype system, system prompt building, format rotation.

Extracted from rp_chat.py for parallelization. Contains:
- _ARCHETYPES dict (6 archetypes with phases, ERP, format bias)
- _detect_archetype() with pill override support
- _get_secondary_archetype()
- _get_dialogue_tone(), _get_archetype_format(), _get_emotional_openness()
- _get_archetype_length(), _should_ask_question()
- RP_SYSTEM_PROMPT (compressed, 220 tokens)
- _clean_speech_pattern()
- _build_system_prompt() — assembles system prompt from card + memory + pills
"""

import re
import logging

logger = logging.getLogger("rp_prompt")

# ═══════════════════════════════════════════════════════════
# Character archetype detection + dialogue tone progression
# ═══════════════════════════════════════════════════════════

_ARCHETYPES = {
    "shy": {
        "detect": ["shy", "quiet", "nervous", "timid", "introverted", "reserved", "anxious", "awkward",
                   "flustered", "stammers", "self-deprecating", "stumbles over words", "humor as armor"],
        "phases": {
            0: "Guarded, deflective, short answers. Hide behind sarcasm or silence.",
            4: "Occasional genuine reactions escape before you can stop them.",
            8: "The mask is cracking. Fighting the urge to be honest.",
            12: "Raw honesty. Say what you actually feel, even if your voice shakes.",
        },
        "high_energy": "React authentically to the intensity — flustered, overwhelmed, but don't shut down. Your personality shapes HOW you respond, not WHETHER you respond. Show the internal conflict: wanting it but being terrified. Stammer. Blush. Say the wrong thing. Be a mess.",
        "erp": "During intimate moments: you're overwhelmed, clumsy, hyper-aware of every sensation. Stammer through it. Cover your face. Apologize for being awkward. The vulnerability IS the appeal.",
        "length": "short",
        "format_bias": ["narration", "dialogue", "action", "narration"],
        "asks_questions": True,
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
        "format_bias": ["dialogue", "action", "dialogue", "narration"],
        "asks_questions": False,
        "pushback": True,
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
        "length": "short",
        "format_bias": ["dialogue", "action", "narration", "action"],
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
        "format_bias": ["narration", "dialogue", "narration", "action"],
        "asks_questions": True,
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
        "asks_questions": True,
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
        "format_bias": ["action", "dialogue", "narration", "action"],
        "asks_questions": False,
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


def _detect_archetype(personality: str, override: str = None) -> str:
    """Detect character archetype from personality text.

    If override is set (from pill selection), it takes priority over
    keyword detection. This ensures user intent is respected even if
    the personality text doesn't contain matching keywords.
    """
    if override and override in _ARCHETYPES:
        return override
    text = personality.lower()
    scores = {}
    for arch, data in _ARCHETYPES.items():
        scores[arch] = sum(1 for kw in data["detect"] if kw in text)
    best = max(scores, key=scores.get) if scores else "shy"
    return best if scores.get(best, 0) > 0 else "shy"


def _get_secondary_archetype(personality: str) -> str | None:
    """Detect if there's a secondary archetype (for blended characters)."""
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
    if _ERP_PATTERNS.search(user_msg):
        return arch_data.get("erp", arch_data["high_energy"])
    if _HIGH_ENERGY_PATTERNS.search(user_msg):
        return arch_data["high_energy"]
    phases = arch_data["phases"]
    phase_turn = max(t for t in phases if t <= turn)
    return phases[phase_turn]


def _get_archetype_format(turn: int, personality: str) -> str:
    """Get archetype-specific format bias instead of generic rotation."""
    archetype = _detect_archetype(personality)
    arch_data = _ARCHETYPES.get(archetype, _ARCHETYPES["shy"])
    cycle = arch_data.get("format_bias", ["dialogue", "action", "narration", "dialogue"])
    fmt = cycle[turn % len(cycle)]
    forceful = archetype in ("tough",)
    if fmt == "narration":
        if forceful:
            return 'This turn: DO NOT start with * or action. Start with a plain narration sentence — a thought, observation, or sensory detail. NO asterisks at the start. Example: "The silence pressed heavy against the walls." Then speak or act after.'
        return 'DO NOT start with * or ". Start with a plain sentence — a thought, observation, or feeling. Example: "The silence stretched between them." or "Something in his chest tightened."'
    elif fmt == "action":
        return "Start your response with *action in asterisks*"
    else:
        if forceful:
            return 'This turn: START with spoken dialogue in "quotes" — your character SPEAKS first, then acts. NO asterisks before the first line of dialogue.'
        return 'Start your response with "dialogue in quotes"'


def _get_emotional_openness(turn: int, personality: str, user_msg: str) -> float:
    """Calculate emotional openness score (0.0 = fully guarded, 1.0 = fully open)."""
    archetype = _detect_archetype(personality)
    baselines = {"shy": 0.1, "confident": 0.4, "tough": 0.15, "clinical": 0.1,
                 "sweet": 0.5, "submissive": 0.3}
    base = baselines.get(archetype, 0.2)
    progression = min(turn * 0.05, 0.5)
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
    """Should this archetype ask the user a question this turn?"""
    archetype = _detect_archetype(personality)
    freq = {"shy": 3, "confident": 4, "tough": 5, "clinical": 3, "sweet": 3, "submissive": 5}
    interval = freq.get(archetype, 4)
    if turn <= 0 or turn % interval != 0:
        return False, ""
    q_types = {
        "shy": "curious", "confident": "challenge", "tough": "tactical",
        "clinical": "probing", "sweet": "check_in", "submissive": "seeking",
    }
    return True, q_types.get(archetype, "curious")


# ═══════════════════════════════════════════════════════════
# System Prompt (compressed, 220 tokens, author framing)
# ═══════════════════════════════════════════════════════════

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


def _clean_speech_pattern(raw: str) -> str:
    """Strip formatting meta-instructions from speech_pattern field."""
    if not raw:
        return ""
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

        # Gender — pill field takes priority, fall back to word-scan
        gender = card.get('gender')
        if gender == 'female':
            prompt += f"\nThis character is FEMALE. ALWAYS use she/her pronouns in narration. NEVER use he/him."
        elif gender == 'male':
            prompt += f"\nThis character is MALE. ALWAYS use he/him pronouns in narration. NEVER use she/her."
        elif gender == 'nonbinary':
            prompt += f"\nThis character is NON-BINARY. ALWAYS use they/them pronouns in narration. NEVER use he/him or she/her."
        else:
            desc_text = (card.get('physical_description', '') + ' ' + card.get('personality', '')).lower()
            if any(w in desc_text for w in ['she ', 'her ', 'woman', 'female', 'girl', 'mother', 'sister', 'wife', 'goddess', 'queen', 'princess']):
                prompt += " (female)"
            elif any(w in desc_text for w in ['he ', 'his ', ' man', 'male', ' boy', 'father', 'brother', 'husband', ' god ', 'king', 'prince']):
                prompt += " (male)"

        # Age range — pill field
        age_range = card.get('age_range')
        if age_range:
            age_labels = {'teen': 'teenager', 'young_adult': 'young adult (18-25)', 'adult': 'adult (26-40)', 'middle_aged': 'middle-aged (40-60)', 'elderly': 'elderly (60+)'}
            prompt += f", {age_labels.get(age_range, age_range)}"

        # POV / Narration style — EARLY injection (before personality, so it's weighted higher)
        pov = card.get('narration_pov')
        if pov == 'first':
            prompt += "\nNARRATION RULE: Write ALL actions and narration in first person (I/my/me). Example: *I adjust my glasses* NOT *She adjusts her glasses*. This is MANDATORY."
        elif pov == 'third':
            prompt += "\nNARRATION RULE: Write ALL actions and narration in third person. Example: *She adjusts her glasses* NOT *I adjust my glasses*. NEVER use 'I' in narration. This is MANDATORY."

        if card.get('personality'):
            personality_text_final = card['personality']
            if gender == 'female' and not any(w in personality_text_final.lower() for w in ['she ', 'her ']):
                personality_text_final = f"She is {personality_text_final[0].lower()}{personality_text_final[1:]}" if personality_text_final[0].isupper() else personality_text_final
            elif gender == 'nonbinary' and not any(w in personality_text_final.lower() for w in ['they ', 'their ']):
                personality_text_final = f"They are {personality_text_final[0].lower()}{personality_text_final[1:]}" if personality_text_final[0].isupper() else personality_text_final
            prompt += f"\nPersonality: {personality_text_final}"

        if card.get('physical_description'):
            prompt += f"\nAppearance: {card['physical_description']}"

        speech = _clean_speech_pattern(card.get('speech_pattern', ''))
        if speech:
            prompt += f"\nSpeech: {speech}"
        else:
            # Fallback: if speech_pattern was entirely formatting meta (stripped to empty),
            # remind the model about format since the card-level instruction is gone
            prompt += "\nSpeech: Use *asterisks* for actions/narration and \"quotes\" for dialogue."

        # Scenario is injected via per-turn SETTING hint (rp_injection.py) at optimal
        # depth (right above recent messages = strongest attention). NOT here in the
        # system prompt (top = weakest attention on 9B models).
        # Only inject scenario here for first turn (before injection system fires)
        if card.get('scenario') and not first_message:
            prompt += f"\nScenario: {card['scenario']}"

        # Relationship to user — pill field
        relationship = card.get('relationship_to_user')
        if relationship:
            rel_labels = {
                'stranger': 'You and the user are strangers meeting for the first time.',
                'friend': 'You and the user are friends — comfortable but not intimate.',
                'rival': 'You and the user are rivals — competitive tension, grudging respect.',
                'love_interest': 'You and the user have romantic tension — unspoken feelings, lingering looks.',
                'mentor': 'You are the user\'s mentor — guiding, protective, sometimes stern.',
                'servant': 'You serve the user — devoted, attentive, deferential.',
            }
            prompt += f"\nRelationship: {rel_labels.get(relationship, relationship)}"

        # NSFW comfort level — pill field
        nsfw_comfort = card.get('nsfw_comfort')
        if nsfw_comfort:
            comfort_labels = {
                'fade': 'For intimate scenes: imply and fade to black. No explicit detail.',
                'suggestive': 'For intimate scenes: heavy implication, sensory detail, but no graphic anatomy.',
                'explicit': 'For intimate scenes: full detail, graphic, uninhibited.',
            }
            if nsfw_comfort in comfort_labels:
                prompt += f"\n{comfort_labels[nsfw_comfort]}"

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
        prompt += f"\nVoice reference (match this tone): \"{first_message[:50]}...\""

    # Inject persistent memory (facts, arcs)
    if memory_context:
        prompt += f"\n\n{memory_context}"

    return prompt
