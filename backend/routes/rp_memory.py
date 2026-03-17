"""
RP Tiered Memory System.

Three-tier persistent memory for RP conversations:
  Tier 1 — Facts: permanent key-value pairs (name, traits, items)
  Tier 2 — Recent conversation: sliding window from rp_messages (no storage here)
  Tier 3 — Arc summaries: relationship state checkpoints (not plot summaries)

Called from rp_chat.py. Non-blocking extraction after responses.
"""

import re
import json
import logging
import requests
from datetime import datetime

logger = logging.getLogger("rp_memory")

OLLAMA_HOST = "http://localhost:11434"

# =========================================================================
# Tier 1 — Fact Extraction (regex + LLM)
# =========================================================================

FACT_PATTERNS = [
    # Name detection
    (r"(?:my name is|I'm|call me|I am)\s+([A-Z][a-z]+)", "user_fact", "name"),
    # Origin/location
    (r"(?:I'm from|I live in|I grew up in|I was born in)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", "user_fact", "origin"),
    # Physical traits (scars, tattoos, marks)
    (r"I (?:have|got) (?:a |an )?(?:scar|tattoo|mark|piercing|birthmark)\s+(.{5,60}?)(?:\.|,|!|$)", "user_fact", "physical_trait"),
    # Injuries/events (scratched, broke, hurt, burned, cut, lost)
    (r"I (?:scratched|broke|hurt|injured|burned|cut|sprained|twisted|bruised)\s+(?:my\s+)?(.{3,50}?)(?:\.|,|!|$)", "user_fact", "injury"),
    (r"I (?:lost|found|broke|dropped)\s+(?:my\s+)?(.{3,50}?)(?:\.|,|!|$)", "user_fact", "event"),
    # Items/possessions
    (r"I (?:always |usually )?(?:carry|wear|have)\s+(?:a |an |my )?(.{5,50}?)(?:\.|,|!|$)", "user_fact", "item"),
    # Preferences
    (r"(?:my favorite|I love|I prefer|I enjoy)\s+(.{3,40}?)(?:\.|,|!|$)", "user_fact", "preference"),
    # Relationships (people + pets)
    (r"(?:my |I have a )((?:wife|husband|partner|sister|brother|mother|father|daughter|son|friend|dog|cat|pet)\s*.{0,30}?)(?:\.|,|!|$)", "user_fact", "relationship"),
    # Age/birthday
    (r"I(?:'m| am)\s+(\d{1,3})\s+years?\s+old", "user_fact", "age"),
    (r"(?:my birthday is|I was born)\s+(?:on\s+)?(.{3,30}?)(?:\.|,|!|$)", "user_fact", "birthday"),
]

TIER1_LLM_PROMPT = """Extract NEW facts from this RP exchange as JSON. Only include information not previously established.
Return ONLY valid JSON, nothing else.

User message: {user_msg}
Character response: {ai_response}

Format:
{{"user_facts": ["name: X", "trait: Y"], "npc_changes": [{{"name": "Z", "change": "..."}}], "scene": {{"location": "...", "mood": "..."}}}}

If nothing new, return: {{"user_facts": [], "npc_changes": [], "scene": {{}}}}"""


def extract_facts_regex(user_msg: str) -> list:
    """Extract facts from user message using regex. Fast, reliable for simple patterns."""
    facts = []
    for pattern, category, key in FACT_PATTERNS:
        matches = re.findall(pattern, user_msg, re.IGNORECASE)
        for match in matches:
            value = match.strip().rstrip('.,!?')
            if len(value) > 2:
                facts.append((category, key, value))
    return facts


def extract_facts_llm(user_msg: str, ai_response: str) -> list:
    """Extract complex facts using LLM. Slower, for relationship/mood changes."""
    prompt = TIER1_LLM_PROMPT.format(user_msg=user_msg[:500], ai_response=ai_response[:500])
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": "qwen3.5:9b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200},
            "think": False,
        }, timeout=30)
        if r.status_code != 200:
            return []
        text = r.json().get("response", "")
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            return []
        data = json.loads(json_match.group())
        facts = []
        for uf in data.get("user_facts", []):
            if ":" in uf:
                key, value = uf.split(":", 1)
                facts.append(("user_fact", key.strip().lower(), value.strip()))
        for npc in data.get("npc_changes", []):
            if npc.get("name") and npc.get("change"):
                facts.append(("npc_state", npc["name"], npc["change"]))
        scene = data.get("scene", {})
        if scene.get("location"):
            facts.append(("scene", "location", scene["location"]))
        if scene.get("mood"):
            facts.append(("scene", "mood", scene["mood"]))
        return facts
    except (json.JSONDecodeError, requests.exceptions.RequestException) as e:
        logger.warning(f"LLM fact extraction failed: {e}")
        return []


def extract_facts_immediate(session_id: str, user_msg: str, turn_number: int, db):
    """Instant regex extraction — runs on EVERY user message (zero cost).
    Called synchronously before the LLM response, not in a thread."""
    facts = extract_facts_regex(user_msg)
    for category, key, value in facts:
        db.upsert_rp_context(session_id, tier=1, category=category,
                             key=key, value=value, turn_number=turn_number)
        logger.info(f"Tier 1 immediate: [{category}] {key}={value}")


def extract_facts(session_id: str, user_msg: str, ai_response: str,
                  turn_number: int, db):
    """Deep extraction — LLM analyzes last 5 message pairs for subtle facts.
    Runs every 5th message pair in a background thread."""
    # Gather last 5 user+assistant pairs for LLM analysis
    try:
        recent = db.conn.execute(
            """SELECT role, content FROM rp_messages
               WHERE session_id = ? AND role IN ('user', 'assistant')
               ORDER BY turn_number DESC LIMIT 10""",
            (session_id,)
        ).fetchall()
        # Build combined text of recent exchanges
        combined_user = " ".join(m["content"] for m in recent if m["role"] == "user")
        combined_ai = " ".join(m["content"][:200] for m in recent if m["role"] == "assistant")
    except Exception as e:
        logger.warning(f"Failed to fetch recent messages for LLM extraction: {e}")
        combined_user = user_msg
        combined_ai = ai_response

    # Always run LLM extraction (not as fallback — catches subtle facts regex misses)
    facts = extract_facts_llm(combined_user, combined_ai)

    # Deduplicate by (category, key)
    seen = set()
    unique_facts = []
    for f in facts:
        k = (f[0], f[1])
        if k not in seen:
            seen.add(k)
            unique_facts.append(f)

    for category, key, value in unique_facts:
        db.upsert_rp_context(session_id, tier=1, category=category,
                             key=key, value=value, turn_number=turn_number)
        logger.info(f"Tier 1 extracted: [{category}] {key}={value} (turn {turn_number})")


def should_extract(turn_number: int) -> bool:
    """Gate: run LLM fact extraction every 3rd turn (regex runs every turn regardless)."""
    return turn_number > 0 and turn_number % 3 == 0


# =========================================================================
# Tier 3 — Arc Summaries (relationship state checkpoints)
# =========================================================================

TIER3_PROMPT = """You are reading an RP conversation between {user_name} and {character_name}.
Describe the CURRENT relationship state in 2-3 sentences.

Rules:
- Describe WHERE things stand NOW, not what happened
- Include: trust level, emotional tension, unresolved feelings, who has power
- Use specific details from the conversation, not generic phrases
- BAD: "They grew closer and shared feelings"
- GOOD: "She trusts him enough to show vulnerability but flinches when he reaches for her hand — the physical intimacy is ahead of the emotional safety"

Recent conversation:
{recent_text}

Current state:"""

SCENE_TRANSITION_PATTERNS = [
    r"(?:walked|moved|went|headed|traveled|arrived|entered|stepped)\s+(?:to|into|towards)\s+",
    r"(?:they|we|she|he)\s+(?:were|are)\s+(?:now\s+)?(?:in|at)\s+",
]

TIME_SKIP_PATTERNS = [
    r"(?:next|the following)\s+(?:morning|day|evening|night|week)",
    r"(?:hours?|days?|weeks?)\s+later",
    r"(?:the next|that)\s+(?:morning|evening|night)",
]


def detect_scene_transition(user_msg: str, ai_response: str) -> bool:
    """Detect location changes, time skips, or new NPCs."""
    combined = f"{user_msg} {ai_response}"
    for pattern in SCENE_TRANSITION_PATTERNS + TIME_SKIP_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return True
    return False


def should_arc_summarize(turn_number: int, user_msg: str, ai_response: str) -> bool:
    """Gate: generate arc summary on scene transitions or every 10 messages."""
    if turn_number > 0 and turn_number % 10 == 0:
        return True
    return detect_scene_transition(user_msg, ai_response)


def extract_arc_summary(session_id: str, user_name: str, char_name: str,
                        recent_messages: list, turn_number: int, db):
    """Generate a relationship state checkpoint using LLM."""
    # Build recent text from messages
    recent_text = ""
    for msg in recent_messages[-10:]:  # last 10 messages
        role = "User" if msg.get("role") == "user" else char_name
        recent_text += f"{role}: {msg.get('content', '')[:200]}\n"

    if not recent_text.strip():
        return

    prompt = TIER3_PROMPT.format(
        user_name=user_name or "User",
        character_name=char_name or "Character",
        recent_text=recent_text[:1500],
    )

    try:
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json={
            "model": "qwen3.5:9b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 200},
            "think": False,
        }, timeout=45)
        if r.status_code != 200:
            logger.warning(f"Arc summary LLM call failed: {r.status_code}")
            return
        summary = r.json().get("response", "").strip()

        # Quality gate: 30-200 words
        word_count = len(summary.split())
        if word_count < 10 or word_count > 200:
            logger.warning(f"Arc summary quality gate failed: {word_count} words, retrying")
            # Retry once
            r = requests.post(f"{OLLAMA_HOST}/api/generate", json={
                "model": "qwen3.5:9b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 200},
                "think": False,
            }, timeout=45)
            if r.status_code == 200:
                summary = r.json().get("response", "").strip()
                word_count = len(summary.split())
                if word_count < 10 or word_count > 200:
                    logger.warning(f"Arc summary retry also failed: {word_count} words, discarding")
                    return

        # Remove think tags if present
        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()

        # Store as arc summary
        arc_key = f"arc_{turn_number}"
        db.insert_rp_context(session_id, tier=3, category="arc_summary",
                             key=arc_key, value=summary, turn_number=turn_number)
        logger.info(f"Tier 3 arc summary stored at turn {turn_number}: {summary[:80]}...")

        # Compress old arcs if total exceeds 800 tokens (~600 words)
        _compress_old_arcs(session_id, db)

    except requests.exceptions.RequestException as e:
        logger.warning(f"Arc summary extraction failed: {e}")


def _compress_old_arcs(session_id: str, db):
    """If total arc summaries exceed ~800 tokens, merge the two oldest into one."""
    arcs = db.get_rp_context(session_id, tier=3, category="arc_summary", limit=100)
    if not arcs:
        return
    total_words = sum(len(a["value"].split()) for a in arcs)
    if total_words <= 600:  # ~800 tokens
        return

    # Sort by turn_number ascending (oldest first)
    arcs.sort(key=lambda a: a.get("turn_number", 0))

    # Merge the two oldest
    if len(arcs) >= 2:
        oldest_two = arcs[:2]
        merged = f"{oldest_two[0]['value']} {oldest_two[1]['value']}"
        # Trim to ~150 words
        words = merged.split()
        if len(words) > 150:
            merged = " ".join(words[:150])

        # Delete the two oldest, insert merged
        for arc in oldest_two:
            db.conn.execute("DELETE FROM rp_session_context WHERE id = ?", (arc["id"],))
        db.conn.commit()
        db.insert_rp_context(session_id, tier=3, category="arc_summary",
                             key=f"arc_merged_{oldest_two[0].get('turn_number', 0)}",
                             value=merged, turn_number=oldest_two[0].get("turn_number", 0))
        logger.info(f"Compressed 2 oldest arcs into 1 ({len(merged.split())} words)")


# =========================================================================
# Context Assembly — build_rp_context()
# =========================================================================

def _estimate_tokens(text: str) -> int:
    """Rough token estimation: word_count * 1.3"""
    return int(len(text.split()) * 1.3)


def build_rp_context(session_id: str, db, character_card_id: str = None,
                     token_budget: int = 4000) -> tuple:
    """
    Assemble tiered memory into a prompt string.

    Returns: (context_string, debug_dict)
    debug_dict: {"tier1": token_count, "tier3": token_count, "tier2": token_count,
                 "tier2_pairs": N, "total": token_count}
    """
    debug = {"tier1": 0, "tier3": 0, "tier2": 0, "tier2_pairs": 0, "total": 0}
    parts = []

    # ── Tier 1: Facts (always, capped at 200 tokens) ──
    tier1_facts = db.get_rp_context(session_id, tier=1, limit=50)
    # Also load cross-session facts if we have a character card
    if character_card_id and not tier1_facts:
        tier1_facts = db.get_rp_context_for_character(character_card_id, tier=1, limit=30)

    if tier1_facts:
        fact_lines = []
        tier1_tokens = 0
        for fact in tier1_facts:
            line = f"- {fact['category']}/{fact['key']}: {fact['value']}"
            line_tokens = _estimate_tokens(line)
            if tier1_tokens + line_tokens > 200:
                break
            fact_lines.append(line)
            tier1_tokens += line_tokens

        if fact_lines:
            tier1_text = "[Memory — Known Facts]\n" + "\n".join(fact_lines)
            parts.append(tier1_text)
            debug["tier1"] = _estimate_tokens(tier1_text)

    # ── Tier 3: Arc Summaries (always, capped at 800 tokens) ──
    tier3_arcs = db.get_rp_context(session_id, tier=3, category="arc_summary", limit=20)
    # Also load cross-session arcs
    if character_card_id and not tier3_arcs:
        tier3_arcs = db.get_rp_context_for_character(character_card_id, tier=3, limit=10)

    if tier3_arcs:
        arc_lines = []
        tier3_tokens = 0
        # Sort by turn_number ascending for chronological order
        tier3_arcs.sort(key=lambda a: a.get("turn_number", 0))
        for arc in tier3_arcs:
            arc_tokens = _estimate_tokens(arc["value"])
            if tier3_tokens + arc_tokens > 800:
                break
            arc_lines.append(arc["value"])
            tier3_tokens += arc_tokens

        if arc_lines:
            tier3_text = "[Memory — Relationship Arc]\n" + "\n---\n".join(arc_lines)
            parts.append(tier3_text)
            debug["tier3"] = _estimate_tokens(tier3_text)

    # ── Tier 2: Recent Messages (fill remaining budget) ──
    remaining_budget = token_budget - debug["tier1"] - debug["tier3"] - 100  # 100 token buffer
    if remaining_budget > 0:
        messages = db.conn.execute(
            """SELECT role, content FROM rp_messages
               WHERE session_id = ? AND branch_id = 'main' AND is_active = TRUE
               ORDER BY turn_number DESC""",
            (session_id,)
        ).fetchall()

        tier2_lines = []
        tier2_tokens = 0
        pair_count = 0
        for msg in messages:
            line = f"{msg['role']}: {msg['content']}"
            line_tokens = _estimate_tokens(line)
            if tier2_tokens + line_tokens > remaining_budget:
                break
            tier2_lines.append(line)
            tier2_tokens += line_tokens
            if msg["role"] == "user":
                pair_count += 1

        # Reverse to chronological order
        tier2_lines.reverse()
        debug["tier2"] = tier2_tokens
        debug["tier2_pairs"] = pair_count

    # ── Assemble ──
    context = "\n\n".join(parts) if parts else ""
    debug["total"] = debug["tier1"] + debug["tier3"] + debug["tier2"]

    return context, debug
