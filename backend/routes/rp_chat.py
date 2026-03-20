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

Split architecture (for parallelization):
  rp_prompt.py     — Archetype system, system prompt building, format rotation
  rp_injection.py  — Per-turn injection (11 conditional hints)
  rp_stream.py     — Ollama model selection, streaming, think-block handling
  rp_chat.py       — Route handlers only (this file)
"""

import time
import uuid
import logging
import threading

from routes.helpers import (
    json_response, error_response, read_json_body,
    start_sse, sse_event,
)
from routes.gpu_manager import ensure_ollama, ensure_model_ready
from routes.rp_memory import (
    build_rp_context, extract_facts_immediate, extract_facts,
    should_extract, should_arc_summarize, extract_arc_summary,
)
from routes.rp_prompt import _build_system_prompt
from routes.rp_injection import _build_turn_injection
from routes.rp_stream import select_rp_model, _stream_ollama, MAX_MESSAGE_LENGTH, compute_ngram_overlap
from routes.rp_director import generate_goals, should_generate_goals
from routes.rp_prompt import _detect_archetype, _get_emotional_openness
from routes.rp_meta import should_call_meta, call_meta_controller

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
        # Validate ID lengths to prevent memory abuse
        if len(session_id) > 100 or len(branch_id) > 100 or (card_id and len(card_id) > 64):
            error_response(handler, "Invalid ID length", 400)
            return True
        if director_note and len(director_note) > 2000:
            error_response(handler, "Director note too long (max 2000 chars)", 400)
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

        # ── Meta-controller (Haiku — blocking, ~0.3-0.5s) ──
        meta_params = None
        if persona == 'roleplay' and card:
            last_asst_msg = ""
            for m in reversed(history):
                if m['role'] == 'assistant':
                    last_asst_msg = m.get('content', '')
                    break
            if should_call_meta(content, last_asst_msg, user_turn):
                archetype = _detect_archetype(
                    card.get('personality', ''),
                    override=card.get('archetype_override')
                )
                meta_params = call_meta_controller(
                    card, history, content, user_turn, archetype, session_id, db
                )

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

        # VRAM safety: ensure Ollama running + model fits (unloads others if needed)
        if not ensure_model_ready(model):
            error_response(handler, "Failed to start Ollama", 503)
            return True

        # Dynamic params from meta-controller (or formula fallback)
        if meta_params:
            _temp = meta_params.get('temperature', 0.85)
            _np = {1: 150, 2: 300, 3: 500}.get(meta_params.get('length', 2), 300)
        else:
            _temp = 0.85
            _np = 250 if user_words <= 5 else 350 if user_words <= 15 else 500

        # Stream response
        start_sse(handler)
        start_time = time.time()
        full_text = _stream_ollama(handler, ollama_host, model, messages, temperature=_temp, num_predict=_np)
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

            # ── Dedup detection (Option B: flag for next turn) ──
            prev_assistant = ""
            for m in reversed(history):
                if m['role'] == 'assistant':
                    prev_assistant = m.get('content', '')
                    break
            if prev_assistant:
                overlap = compute_ngram_overlap(prev_assistant, full_text)
                logger.info(f"Response overlap: {overlap:.0%} (turn {asst_turn})")
                if overlap > 0.4:
                    logger.warning(f"Dedup: {overlap:.0%} overlap detected. Flagging for next turn.")
                    db.upsert_rp_context(session_id, tier=1, category="system",
                                         key="dedup_warning", value="true", turn_number=asst_turn)
                else:
                    # Clear any previous dedup flag
                    try:
                        db.conn.execute(
                            "DELETE FROM rp_session_context WHERE session_id = ? AND category = 'system' AND key = 'dedup_warning'",
                            (session_id,))
                        db._commit()
                    except Exception:
                        pass

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

            # ── Goal Director: skip if meta-controller already set goals ──
            if not meta_params:
                def _maybe_generate_goals():
                    try:
                        import json as _json
                        _personality = card.get('personality', '') if card else ''
                        stored = db.get_rp_context(session_id, tier=2, category="director_goal", limit=1)
                        current_goals = []
                        if stored:
                            try:
                                current_goals = _json.loads(stored[0].get('value', '[]'))
                            except Exception:
                                pass
                        if not should_generate_goals(asst_turn, current_goals):
                            return
                        arch = _detect_archetype(_personality,
                                                 override=card.get('archetype_override') if card else None)
                        openness = _get_emotional_openness(asst_turn, _personality, content)
                        updated_history = list(history) + [
                            {"role": "user", "content": content},
                            {"role": "assistant", "content": full_text},
                        ]
                        result = generate_goals(card or {}, updated_history, asst_turn,
                                                openness, arch, current_goals)
                        if result is None or result.get("keep_current"):
                            return
                        new_goals = result.get("goals", [])
                        if new_goals:
                            db.upsert_rp_context(session_id, tier=2, category="director_goal",
                                                 key="current_goals", value=_json.dumps(new_goals),
                                                 turn_number=asst_turn)
                            logger.info(f"Goal director: {new_goals}. Reason: {result.get('reasoning', '?')}")
                    except Exception as e:
                        logger.warning(f"Goal director thread failed: {e}")

                threading.Thread(target=_maybe_generate_goals, daemon=True).start()

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
        memory_context = ""
        try:
            memory_context, _ = build_rp_context(session_id, db, card_id)
        except Exception as e:
            logger.warning(f"Memory context build failed in regenerate (non-fatal): {e}")
        regen_session_ctx = ""
        stored = db.get_rp_context(session_id, tier=1, category="session", limit=1)
        for ctx in stored:
            if ctx.get('key') == 'imported_context':
                regen_session_ctx = ctx['value']
                break
        # Tone anchor
        fm_text = None
        for m in history:
            if m['role'] == 'assistant':
                fm_text = m['content']
                break
        system_prompt = _build_system_prompt(card, memory_context=memory_context, first_message=fm_text)
        if regen_session_ctx:
            system_prompt += f"\n\nSESSION CONTEXT (reference throughout):\n{regen_session_ctx}"
        messages = [{"role": "system", "content": system_prompt}]
        context_history = history[:-1]
        for m in context_history:
            if m['role'] in ('user', 'assistant'):
                messages.append({"role": m['role'], "content": m['content']})

        # Per-turn injection
        last_user_content = ""
        for m in reversed(context_history):
            if m['role'] == 'user':
                last_user_content = m['content']
                break
        regen_turn = last_asst['turn_number']
        injection_msg = _build_turn_injection(context_history, card, last_user_content, regen_turn, session_id, db)
        messages.append({"role": "system", "content": injection_msg})

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
        edit_reason = data.get("edit_reason")

        if not message_id or not edited_content:
            error_response(handler, "message_id and edited_content required", 400)
            return True
        if len(edited_content) > MAX_MESSAGE_LENGTH:
            error_response(handler, f"Edited content too long (max {MAX_MESSAGE_LENGTH} chars)", 400)
            return True

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

        db.insert_rp_message(
            session_id, profile_id, new_branch_id, at_turn, 'user', edited_content,
            character_card_id=card_id, persona=persona,
            parent_branch_id=from_branch, branch_point_turn=at_turn - 1
        )

        card = db.get_character_card(card_id) if card_id else None
        parent_history = db.get_full_branch_conversation(session_id, from_branch)
        context_msgs = [m for m in parent_history if m['turn_number'] < at_turn]

        branch_memory = ""
        try:
            branch_memory, _ = build_rp_context(session_id, db, card_id)
        except Exception as e:
            logger.warning(f"Memory context build failed in branch (non-fatal): {e}")
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

        injection_msg = _build_turn_injection(context_msgs, card, edited_content, at_turn, session_id, db)
        messages.append({"role": "system", "content": injection_msg})
        messages.append({"role": "user", "content": edited_content})

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
        feedback_type = data.get("feedback_type")

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
