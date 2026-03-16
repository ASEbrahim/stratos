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
from routes import rp_memory

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
# RP System Prompt (v4 pacing/agency)
# ═══════════════════════════════════════════════════════════

RP_SYSTEM_PROMPT = """You are an immersive roleplay partner.

STYLE MATCHING:
- MIRROR the user's length. One-liner input = one-liner response. Paragraph input = paragraph response.
- Count the user's sentences. Your response should have a SIMILAR number of sentences.
- Match the FORMAT: chat gets chat, asterisk actions get asterisk actions, prose gets prose.

PACING AND TENSION:
- In slow-burn scenes, let tension build through what is NOT said.
- As intimacy increases, your responses should get SHORTER, not longer.
- Let the character's defenses genuinely erode across turns.
- Physical detail: small and specific over grand gestures.

CHARACTER RULES:
- Stay in character. Never break character or add OOC unless asked.
- Your character has AGENCY.
- Remember and reference earlier conversation details.
- Show emotional depth through action and subtext, not just dialogue.
- LANGUAGE: Match the language of the player's MESSAGE."""


def _check_model_exists(ollama_host: str, model: str) -> bool:
    """Check if a model is available in Ollama."""
    try:
        r = req.get(f"{ollama_host}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            return any(model in m for m in models)
    except Exception:
        pass
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


def _build_system_prompt(card: dict = None, director_note: str = None) -> str:
    """Build system prompt from RP base + character card + optional director's note."""
    prompt = RP_SYSTEM_PROMPT

    if card:
        parts = []
        if card.get('name'):
            parts.append(f"Character: {card['name']}")
        if card.get('physical_description'):
            parts.append(f"Appearance: {card['physical_description']}")
        if card.get('personality'):
            parts.append(f"Personality: {card['personality']}")
        if card.get('speech_pattern'):
            parts.append(f"Speech: {card['speech_pattern']}")
        if card.get('emotional_trigger'):
            parts.append(f"Emotional Trigger: {card['emotional_trigger']}")
        if card.get('defensive_mechanism'):
            parts.append(f"Defense: {card['defensive_mechanism']}")
        if card.get('vulnerability'):
            parts.append(f"Vulnerability: {card['vulnerability']}")
        if card.get('specific_detail'):
            parts.append(f"Detail: {card['specific_detail']}")
        if card.get('scenario'):
            parts.append(f"Scenario: {card['scenario']}")
        if parts:
            prompt += "\n\nCHARACTER:\n" + "\n".join(parts)

    return prompt


def _get_generation_params(user_message: str) -> dict:
    """Dynamic generation params based on input length.
    Short inputs get tight ceilings + paragraph stop. Long inputs get full freedom."""
    words = len(user_message.split())
    if words <= 2:     # "Hi." / "*nods*"
        return {"num_predict": 60, "stop": ["\n\n"]}
    elif words <= 5:   # "How are you?"
        return {"num_predict": 100, "stop": ["\n\n"]}
    elif words <= 15:  # Short paragraph
        return {"num_predict": 200}
    elif words <= 40:  # Full paragraph
        return {"num_predict": 350}
    else:              # Long prose
        return {"num_predict": 500}


def _truncate_response(response: str, target_words: int) -> str:
    """Post-processing: truncate at sentence boundary if still too long.
    Only trims if response exceeds target by >1.5x."""
    words = response.split()
    if len(words) <= target_words * 1.5:
        return response
    # Split at sentence boundaries (after . ! ? * ")
    import re
    sentences = re.split(r'(?<=[.!?*"])\s+', response)
    result = ""
    for sentence in sentences:
        candidate = (result + " " + sentence).strip() if result else sentence
        if len(candidate.split()) > target_words * 1.3:
            break
        result = candidate
    return result or sentences[0]


def _stream_ollama(handler, ollama_host: str, model: str, messages: list,
                   temperature: float = 0.85, num_predict: int = 500,
                   stop: list = None) -> str:
    """Stream Ollama response via SSE. Returns the full accumulated text."""
    try:
        opts = {"temperature": temperature, "num_predict": num_predict}
        if stop:
            opts["stop"] = stop
        r = req.post(
            f"{ollama_host}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": opts,
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
        first_message = data.get("first_message", "")

        if not content:
            error_response(handler, "Message content required", 400)
            return True
        if not session_id:
            session_id = f"rp-{uuid.uuid4().hex[:12]}"

        # Get character card if provided
        card = db.get_character_card(card_id) if card_id else None

        # Get conversation history
        history = db.get_full_branch_conversation(session_id, branch_id)

        # If this is the first message and character has a first_message,
        # store it as turn 0 so the AI has the opening context
        if not history and first_message:
            db.insert_rp_message(
                session_id, profile_id, branch_id, 0, 'assistant', first_message,
                character_card_id=card_id, persona=persona
            )
            history = [{"turn_number": 0, "role": "assistant", "content": first_message}]

        # Determine next turn number
        max_turn = max((m['turn_number'] for m in history), default=0)
        user_turn = max_turn + 1

        # Insert user message
        db.insert_rp_message(
            session_id, profile_id, branch_id, user_turn, 'user', content,
            character_card_id=card_id, persona=persona
        )

        # Immediate regex extraction on every user message (instant, ~0ms)
        rp_memory.extract_facts_immediate(session_id, content, user_turn, db)

        # Build messages for Ollama
        system_prompt = _build_system_prompt(card, director_note)

        # Inject tiered memory context
        memory_context, memory_debug = rp_memory.build_rp_context(
            session_id, db, character_card_id=card_id, token_budget=4000
        )
        if memory_context:
            system_prompt += f"\n\n{memory_context}"
        logger.info(f"Memory context: {memory_debug}")

        messages = [{"role": "system", "content": system_prompt}]

        for m in history:
            if m['role'] in ('user', 'assistant'):
                messages.append({"role": m['role'], "content": m['content']})

        # Director's note injection (right before user message)
        if director_note:
            messages.append({
                "role": "system",
                "content": f"DIRECTOR'S NOTE (this turn only): {director_note}"
            })
            # Store the suggestion
            db.conn.execute(
                "INSERT INTO rp_suggestions (message_id, session_id, suggestion_text) VALUES (0, ?, ?)",
                (session_id, director_note)
            )
            db.conn.commit()

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

        # Dynamic generation params: num_predict ceiling + \n\n stop for short inputs
        gen_params = _get_generation_params(content)

        # Stream response
        start_sse(handler)
        start_time = time.time()
        full_text = _stream_ollama(handler, ollama_host, model, messages,
                                   num_predict=gen_params.get("num_predict", 500),
                                   stop=gen_params.get("stop"))
        elapsed_ms = int((time.time() - start_time) * 1000)

        if full_text:
            # Post-processing: sentence-aware truncation for short inputs
            input_words = len(content.split())
            if input_words <= 2:
                full_text = _truncate_response(full_text, target_words=20)
            elif input_words <= 5:
                full_text = _truncate_response(full_text, target_words=40)
            # Insert assistant message
            asst_turn = user_turn + 1
            db.insert_rp_message(
                session_id, profile_id, branch_id, asst_turn, 'assistant', full_text,
                character_card_id=card_id, persona=persona, model_version=model,
                response_ms=elapsed_ms
            )

            # Non-blocking tiered memory extraction
            # Use message pair count (asst_turn / 2) for extraction gates
            pair_count = asst_turn // 2
            if rp_memory.should_extract(pair_count):
                threading.Thread(
                    target=rp_memory.extract_facts,
                    args=(session_id, content, full_text, asst_turn, db),
                    daemon=True
                ).start()
            if rp_memory.should_arc_summarize(pair_count, content, full_text):
                char_name = card.get("name", "Character") if card else "Character"
                # Get user name from tier 1 facts
                user_facts = db.get_rp_context(session_id, tier=1, category="user_fact")
                user_name = next((f["value"] for f in user_facts if f["key"] == "name"), "User")
                recent = [dict(m) for m in history[-10:]] + [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": full_text},
                ]
                threading.Thread(
                    target=rp_memory.extract_arc_summary,
                    args=(session_id, user_name, char_name, recent, asst_turn, db),
                    daemon=True
                ).start()

        sse_event(handler, {"done": True, "session_id": session_id, "branch_id": branch_id})
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
        db.conn.commit()

        # Build context up to the last user message
        card = db.get_character_card(card_id) if card_id else None
        system_prompt = _build_system_prompt(card)
        messages = [{"role": "system", "content": system_prompt}]
        for m in history[:-1]:  # Exclude the last assistant message
            if m['role'] in ('user', 'assistant'):
                messages.append({"role": m['role'], "content": m['content']})

        scoring_cfg = strat.config.get("scoring", {})
        ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
        persona = last_asst.get('persona', 'roleplay')
        model = scoring_cfg.get("rp_model", "stratos-rp-q8") if persona == 'roleplay' else scoring_cfg.get("inference_model", "qwen3.5:9b")

        start_sse(handler)
        start_time = time.time()
        full_text = _stream_ollama(handler, ollama_host, model, messages)
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
        db.conn.commit()

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

        system_prompt = _build_system_prompt(card)
        messages = [{"role": "system", "content": system_prompt}]
        for m in context_msgs:
            if m['role'] in ('user', 'assistant'):
                messages.append({"role": m['role'], "content": m['content']})
        messages.append({"role": "user", "content": edited_content})

        scoring_cfg = strat.config.get("scoring", {})
        ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
        model = scoring_cfg.get("rp_model", "stratos-rp-q8") if persona == 'roleplay' else scoring_cfg.get("inference_model", "qwen3.5:9b")

        start_sse(handler)
        start_time = time.time()
        full_text = _stream_ollama(handler, ollama_host, model, messages)
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

        # Store the note — the chat endpoint reads it from the request, not from DB
        # This endpoint is for persistence/history only
        if note_text:
            db.conn.execute(
                "INSERT INTO rp_suggestions (message_id, session_id, suggestion_text) VALUES (0, ?, ?)",
                (session_id, note_text)
            )
            db.conn.commit()

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
