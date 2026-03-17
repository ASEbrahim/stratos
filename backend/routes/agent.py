"""
Agent routes — /api/agent-chat, /api/suggest-context, /api/ask, /api/agent-status

Uses Ollama /api/chat with tool definitions so the LLM can naturally invoke:
  - web_search: real-time Google search via Serper API
  - manage_watchlist: add/remove/list market tickers
  - manage_categories: add/remove keywords, list/toggle categories
"""

import json
import re
import logging
import threading
import requests as req
from pathlib import Path

from routes.helpers import (json_response, error_response, read_json_body, sse_event, start_sse,
                           strip_think_blocks, strip_reasoning_preamble)
from routes.agent_tools import AGENT_TOOLS, execute_tool, parse_text_tool_calls

logger = logging.getLogger("STRAT_OS")


# ═══════════════════════════════════════════════════════════
# AGENT STATUS
# ═══════════════════════════════════════════════════════════

def handle_agent_status(handler, strat):
    scoring_cfg = strat.config.get("scoring", {})
    ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
    model = scoring_cfg.get("inference_model", "qwen3.5:9b")
    available = False
    try:
        r = req.get(f"{ollama_host}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            available = any(model.split(":")[0] in m for m in models)
    except Exception as e:
        logger.debug(f"Agent status: Ollama check failed: {e}")
    from processors.stt import STTProcessor
    stt_ok, stt_msg = STTProcessor.is_available()
    json_response(handler, {"available": available, "model": model, "host": ollama_host,
                             "stt": {"available": stt_ok, "message": stt_msg}})


# ═══════════════════════════════════════════════════════════
# SUGGESTION GENERATION
# ═══════════════════════════════════════════════════════════

def _generate_suggestions(handler, ollama_host, model, user_msg, agent_response,
                          persona_name, rp_mode='', active_scenario='', active_npc=''):
    """Generate 3 contextual follow-up suggestions via a lightweight LLM call.
    Sends a 'suggestions' SSE event. Falls back silently on failure.
    Gaming persona returns rich suggestions: {label, prompt} pairs."""
    try:
        # Gaming persona: rich suggestions with label + immersive prompt
        if persona_name == 'gaming' and active_scenario:
            _generate_gaming_suggestions(handler, ollama_host, model, user_msg, agent_response,
                                         rp_mode, active_scenario, active_npc)
            return

        context_hint = ''
        prompt = (
            f"Based on this conversation, suggest 3 short follow-up actions (3-8 words each). "
            f"They should feel like natural continuations.\n\n"
            f"User: {user_msg[:200]}\n"
            f"Assistant: {agent_response[:500]}\n"
            f"Persona: {persona_name}\n"
            f"{context_hint}\n\n"
            f"Return ONLY a JSON array of 3 strings. No explanation.\n"
            f'Example: ["Explore the dark corridor", "Ask about the ancient relic", "Check my character stats"]'
        )
        r = req.post(
            f"{ollama_host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 120},
                "think": False,
            },
            timeout=8,
        )
        if r.status_code == 200:
            raw = r.json().get("message", {}).get("content", "").strip()
            raw = strip_think_blocks(raw)
            bracket_start = raw.find("[")
            bracket_end = raw.rfind("]")
            if bracket_start >= 0 and bracket_end > bracket_start:
                arr = json.loads(raw[bracket_start:bracket_end + 1])
                if isinstance(arr, list) and len(arr) >= 1:
                    suggestions = [s.strip() for s in arr if isinstance(s, str) and 2 < len(s.strip()) < 80][:3]
                    if suggestions:
                        sse_event(handler, {"suggestions": suggestions})
                        return
    except Exception as e:
        logger.debug(f"Suggestion generation failed: {e}")


def _generate_gaming_suggestions(handler, ollama_host, model, user_msg, agent_response,
                                  rp_mode, active_scenario, active_npc):
    """Generate rich gaming suggestions with label + immersive prompt."""
    try:
        mode_desc = "Game Master (third-person narration)" if rp_mode == 'gm' else f"Immersive RP (talking to {active_npc or 'a character'})"
        prompt = (
            f"Based on this game exchange, suggest 3-4 next actions.\n\n"
            f"Mode: {mode_desc}\n"
            f"Scenario: {active_scenario}\n"
            f"Last player action: {user_msg[:200]}\n"
            f"Last game response: {agent_response[:500]}\n\n"
            f'Return a JSON array of suggestion objects:\n'
            f'[{{"label": "Talk to Klein", "prompt": "I walk over to Klein. \'Hey, got any tips?\'"}}]\n\n'
            f"Rules:\n"
            f'- "label": 3-6 words, shown on the button\n'
            f'- "prompt": 1-2 immersive sentences, first person as the player\n'
            f"- Vary types: exploration, social, combat, investigation\n"
            f"- IMPORTANT: Write labels and prompts in the SAME language as the player's last message. If the player wrote in Japanese, suggestions must be in Japanese. If Arabic, in Arabic.\n"
            f"Return ONLY the JSON array."
        )
        r = req.post(
            f"{ollama_host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 300},
                "think": False,
            },
            timeout=10,
        )
        if r.status_code == 200:
            raw = r.json().get("message", {}).get("content", "").strip()
            raw = strip_think_blocks(raw)
            bracket_start = raw.find("[")
            bracket_end = raw.rfind("]")
            if bracket_start >= 0 and bracket_end > bracket_start:
                arr = json.loads(raw[bracket_start:bracket_end + 1])
                if isinstance(arr, list) and len(arr) >= 1:
                    suggestions = []
                    for s in arr[:4]:
                        if isinstance(s, dict) and s.get('label') and s.get('prompt'):
                            suggestions.append({"label": s['label'][:50], "prompt": s['prompt'][:200]})
                        elif isinstance(s, str) and 2 < len(s.strip()) < 80:
                            suggestions.append(s.strip())
                    if suggestions:
                        sse_event(handler, {"suggestions": suggestions})
                        return
    except Exception as e:
        logger.debug(f"Gaming suggestion generation failed: {e}")


# ═══════════════════════════════════════════════════════════
# ENTITY MEMORY AUTO-UPDATE (background, non-blocking)
# ═══════════════════════════════════════════════════════════

def _update_entity_memory(db, ollama_host, model, profile_id, persona, scenario, entity_name, user_msg, ai_response):
    """Background LLM call to update an entity's memory after an interaction.
    Called in a daemon thread — must not block the response stream.

    Thread safety note: SQLite cursor creation from db.conn is serialized by the GIL
    and db._commit() uses an internal lock. This is safe for WAL-mode SQLite as long as
    only short-lived cursors are used (no long-held cursor references across awaits).
    """
    try:
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT * FROM persona_entities WHERE profile_id = ? AND persona = ? AND scenario_name = ? AND name = ?",
            (profile_id, persona, scenario, entity_name))
        row = cursor.fetchone()
        if not row:
            return
        entity = dict(row)
        if not entity.get('auto_save', 1):
            return

        prompt = f"""Based on this roleplay exchange, update {entity['display_name']}'s memory.

Player said: {user_msg[:500]}
{entity['display_name']} responded: {ai_response[:800]}

Current relationship: {entity.get('relationship_md', 'None')[:300]}
Current knowledge: {entity.get('knowledge_md', 'None')[:300]}

Return ONLY a JSON object:
{{"interaction_summary": "One sentence summary of what happened",
"relationship_change": "unchanged" or "description of change",
"new_knowledge": ["list of new things they learned"] or []}}"""

        r = req.post(
            f"{ollama_host}/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "stream": False, "options": {"temperature": 0.3, "num_predict": 200}, "think": False},
            timeout=15)

        if r.status_code != 200:
            return

        raw = strip_think_blocks(r.json().get("message", {}).get("content", "").strip())
        brace_start = raw.find("{")
        brace_end = raw.rfind("}")
        if brace_start < 0 or brace_end <= brace_start:
            return

        update = json.loads(raw[brace_start:brace_end + 1])
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Append to memory
        memory = entity.get('memory_md', '') or ''
        summary = update.get('interaction_summary', '')
        if summary:
            memory += f"\n- [{now}] {summary}"

        # Update relationship if changed
        relationship = entity.get('relationship_md', '') or ''
        rel_change = update.get('relationship_change', 'unchanged')
        if rel_change and rel_change != 'unchanged':
            relationship += f"\n- [{now}] {rel_change}"

        # Update knowledge
        knowledge = entity.get('knowledge_md', '') or ''
        new_knowledge = update.get('new_knowledge', [])
        if new_knowledge and isinstance(new_knowledge, list):
            for item in new_knowledge:
                if isinstance(item, str) and item.strip():
                    knowledge += f"\n- {item.strip()}"

        cursor.execute(
            "UPDATE persona_entities SET memory_md = ?, relationship_md = ?, knowledge_md = ?, updated_at = ? "
            "WHERE profile_id = ? AND persona = ? AND scenario_name = ? AND name = ?",
            (memory.strip(), relationship.strip(), knowledge.strip(), datetime.now().isoformat(),
             profile_id, persona, scenario, entity_name))
        db._commit()
        logger.info(f"Entity memory updated: {entity_name} ({persona}/{scenario})")

    except Exception as e:
        logger.debug(f"Entity memory update failed: {e}")


def _post_response_tasks(handler, strat, ollama_host, model, user_msg, ai_response,
                         persona_name, profile_id, rp_mode='gm', active_npc='', scenario=''):
    """Run post-response tasks: suggestions + entity memory update + scenario auto-update."""
    _generate_suggestions(handler, ollama_host, model, user_msg, ai_response,
                          persona_name, rp_mode=rp_mode, active_scenario=scenario, active_npc=active_npc)

    # Background entity memory update for immersive RP mode
    if rp_mode == 'immersive' and active_npc and persona_name in ('gaming', 'scholarly'):
        entity_name = active_npc.strip().lower().replace(' ', '_')
        threading.Thread(
            target=_update_entity_memory,
            args=(strat.db, ollama_host, model, profile_id, persona_name,
                  scenario, entity_name, user_msg, ai_response),
            daemon=True
        ).start()

    # Background scenario auto-update for gaming persona (Sprint 7)
    # Skip auto-update if the response mentions an import/generation in progress
    _skip_auto_update = any(phrase in ai_response.lower() for phrase in
                            ['importing', 'canon import', 'import runs in the background',
                             'being generated', 'check the scenario panel'])
    if persona_name == 'gaming' and scenario and not _skip_auto_update:
        try:
            from routes.personas import _get_scenario_path
            scenario_path = _get_scenario_path(strat, profile_id)
            if scenario_path:
                def _run_scenario_update():
                    try:
                        from processors.scenario_updater import post_response_update
                        post_response_update(
                            ollama_host, model, scenario_path,
                            user_msg, ai_response,
                            mode=rp_mode, active_npc=active_npc
                        )
                    except Exception as e:
                        logger.debug(f"Scenario auto-update failed: {e}")
                threading.Thread(target=_run_scenario_update, daemon=True).start()
        except Exception as e:
            logger.debug(f"Scenario path lookup failed: {e}")


# ═══════════════════════════════════════════════════════════
# MAIN AGENT CHAT HANDLER
# ═══════════════════════════════════════════════════════════

def handle_agent_chat(handler, strat, output_file, profile_id=0):
    """POST /api/agent-chat — Streaming agent conversation with tool use."""
    try:
        from routes.personas import (
            build_persona_prompt, build_persona_context, get_persona_config
        )

        body = read_json_body(handler)
        user_msg = body.get("message", "").strip()
        history = body.get("history", [])
        free_mode = body.get("mode") == "free"
        rp_mode = body.get("rp_mode", "gm")  # 'gm' or 'immersive'
        active_npc = body.get("active_npc", "")
        active_scenario = body.get("active_scenario", "")
        npc_personality = body.get("npc_personality", "")
        npc_memory = body.get("npc_memory", "")
        free_length = body.get("free_length", False)
        use_all_scans = body.get("use_all_scans", False)
        # Support single persona or multi-persona querying
        personas_param = body.get("personas", body.get("persona", "intelligence"))
        if isinstance(personas_param, list) and len(personas_param) > 1:
            multi_personas = personas_param[:3]  # Max 3 for context budget
            persona_name = multi_personas[0]  # Primary persona
        else:
            multi_personas = None
            persona_name = personas_param if isinstance(personas_param, str) else personas_param[0] if isinstance(personas_param, list) else "intelligence"
        if not user_msg:
            raise ValueError("Empty message")

        # ── Input validation: enforce size limits ──
        MAX_MSG_LEN = 50_000  # ~12K tokens
        MAX_HISTORY_TURNS = 50
        if len(user_msg) > MAX_MSG_LEN:
            user_msg = user_msg[:MAX_MSG_LEN]
            logger.warning(f"Agent chat: user message truncated to {MAX_MSG_LEN} chars")
        if len(history) > MAX_HISTORY_TURNS:
            history = history[-MAX_HISTORY_TURNS:]

        scoring_cfg = strat.config.get("scoring", {})
        ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
        model = scoring_cfg.get("inference_model", "qwen3.5:9b")

        # ── Model swap routing: roleplay persona uses dedicated RP model ──
        _is_rp = persona_name == 'roleplay'
        if _is_rp:
            model = strat.config.get("scoring", {}).get("rp_model", "stratos-rp-q8")

        profile = strat.config.get("profile", {})
        role = profile.get("role", "user")
        location = profile.get("location", "")

        serper_available = bool(strat.config.get("search", {}).get("serper_api_key", ""))
        search_note = "You have a web_search tool for real-time Google search." if serper_available else "Web search not configured — feed data only."

        tickers = [t.get("symbol", "") for t in strat.config.get("market", {}).get("tickers", [])]
        cats = strat.config.get("dynamic_categories", [])
        cat_summary = ", ".join(f"{c.get('label','')} ({len(c.get('items',[]))} kw)" for c in cats[:10])

        # Build persona-specific prompt and context
        persona_config = get_persona_config(persona_name)
        if multi_personas:
            # Multi-persona mode: merge contexts and tools from all selected personas
            base_prompt = build_persona_prompt(
                persona_name, role, location, tickers, cat_summary, search_note
            )
            # Merge contexts from all personas
            context_parts = []
            merged_tools = set()
            for p in multi_personas:
                p_config = get_persona_config(p)
                merged_tools.update(p_config['tools'])
                ctx = build_persona_context(p, strat, output_file, profile_id,
                                            user_message=user_msg, rp_mode=rp_mode,
                                            active_npc=active_npc,
                                            use_all_scans=use_all_scans)
                if ctx:
                    context_parts.append(f"[{p.upper()} DATA]\n{ctx}")
            persona_config = {**persona_config, 'tools': list(merged_tools)}
            # Add cross-persona note to prompt
            other_names = [p for p in multi_personas if p != persona_name]
            system_prompt = base_prompt + f"\n\nYou also have context from: {', '.join(other_names)}. Use all available data to answer."
            if context_parts:
                system_prompt += "\n\n" + "\n\n".join(context_parts)
        else:
            # Load entity data from DB
            if persona_name in ('gaming', 'scholarly', 'roleplay') and active_scenario and strat.db:
                try:
                    cursor = strat.db.conn.cursor()
                    if rp_mode == 'immersive' and active_npc:
                        # Immersive: load single active NPC's full data
                        entity_name = active_npc.strip().lower().replace(' ', '_')
                        cursor.execute(
                            "SELECT * FROM persona_entities WHERE profile_id = ? AND persona = ? AND scenario_name = ? AND name = ?",
                            (profile_id, persona_name, active_scenario, entity_name))
                        row = cursor.fetchone()
                        if row:
                            e = dict(row)
                            npc_personality = '\n'.join(filter(None, [
                                f"## Identity\n{e['identity_md']}" if e.get('identity_md') else '',
                                f"## Personality\n{e['personality_md']}" if e.get('personality_md') else '',
                                f"## Speaking Style\n{e['speaking_style_md']}" if e.get('speaking_style_md') else '',
                            ]))
                            npc_memory = '\n'.join(filter(None, [
                                f"## Relationship with Player\n{e['relationship_md']}" if e.get('relationship_md') else '',
                                f"## Interaction Memory\n{e['memory_md']}" if e.get('memory_md') else '',
                                f"## Knowledge\n{e['knowledge_md']}" if e.get('knowledge_md') else '',
                            ]))
                    elif rp_mode == 'gm':
                        # GM mode: load full entity roster so GM knows the cast
                        cursor.execute(
                            "SELECT display_name, identity_md, personality_md, speaking_style_md "
                            "FROM persona_entities WHERE profile_id = ? AND persona = ? AND scenario_name = ?",
                            (profile_id, persona_name, active_scenario))
                        rows = [dict(r) for r in cursor.fetchall()]
                        if rows:
                            roster = ["## Character Roster"]
                            for e in rows[:10]:
                                name = e.get('display_name', '???')
                                parts = []
                                if e.get('identity_md'): parts.append(e['identity_md'][:150])
                                if e.get('personality_md'): parts.append(e['personality_md'][:150])
                                if e.get('speaking_style_md'): parts.append(f"Voice: {e['speaking_style_md'][:80]}")
                                roster.append(f"**{name}**: {' | '.join(parts)}" if parts else f"**{name}**")
                            npc_personality = '\n'.join(roster)
                except Exception as ex:
                    logger.debug(f"Entity load failed: {ex}")

            base_prompt = build_persona_prompt(
                persona_name, role, location, tickers, cat_summary, search_note,
                rp_mode=rp_mode, active_npc=active_npc,
                npc_personality=npc_personality, npc_memory=npc_memory
            )
            persona_context = build_persona_context(
                persona_name, strat, output_file, profile_id,
                user_message=user_msg, rp_mode=rp_mode, active_npc=active_npc,
                use_all_scans=use_all_scans
            )
            system_prompt = base_prompt
            if persona_context:
                system_prompt += f"\n\n{persona_context}"

        # Keyword-triggered history search
        _triggers = ['history','last week','past','before','trend','used to','earlier','previously','been','lately']
        msg_lower = user_msg.lower()
        if any(t in msg_lower for t in _triggers):
            _stop = {'the','a','an','is','are','was','were','has','have','had','do','does',
                     'did','will','would','could','should','can','may','might','about','from',
                     'with','what','when','where','how','why','who','any','been','there',
                     'this','that','these','those','show','tell','give','find','look','get',
                     'me','my','i','you','we','our','last','week','past','history','trend',
                     'lately','recently','before','earlier','previously','much','many','more'}
            words = [w.strip('?.,!') for w in user_msg.split() if len(w) > 2]
            keywords = [w for w in words if w.lower() not in _stop]
            if keywords:
                search_results = []
                for kw in keywords[:3]:
                    try:
                        results = strat.db.search_news_history(kw, days=14, limit=5, profile_id=profile_id)
                        search_results.extend(results)
                    except Exception as e:
                        logger.debug(f"Agent: history search failed for '{kw}': {e}")
                if search_results:
                    seen, unique = set(), []
                    for r in search_results:
                        t = r.get('title', '')
                        if t not in seen:
                            seen.add(t)
                            unique.append(r)
                    lines = [f"  [{r.get('score',0):.1f}] {r.get('title','')[:80]} ({r.get('category','')}, {r.get('fetched_at','')[:10]})" for r in unique[:8]]
                    if lines:
                        system_prompt += f"\n\nDB SEARCH '{' '.join(keywords[:3])}':\n" + "\n".join(lines)

        # ── Response length mode ──
        _num_predict = 8000 if free_length else 1500
        if free_length:
            system_prompt += "\n\nLENGTH MODE: The user has enabled extended responses. You may produce longer, more detailed output. Structure long responses with **headers**, bullet points, and clear sections. Use markdown formatting for readability."
        else:
            system_prompt += "\n\nBREVITY: Keep responses short and conversational — 2-4 sentences max unless the user asks for detail. No headers, no bullet lists, no markdown formatting unless specifically asked. Reply like a knowledgeable friend in a chat app."

        # ── Free chat mode: no tools, simple system prompt ──
        # TODO: consolidate with scorer_base._call_ollama — agent.py has its own streaming impl
        if free_mode:
            free_system = f"You are a helpful AI assistant. The user is a {role} in {location}. Be conversational, helpful, and concise."
            messages = [{"role": "system", "content": free_system}]
            for h in history[:-1]:
                if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
                    messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": user_msg})

            start_sse(handler)
            try:
                r = req.post(
                    f"{ollama_host}/api/chat",
                    json={"model": model, "messages": messages, "stream": True,
                          "options": {"temperature": 0.7, "num_predict": _num_predict}},
                    timeout=180, stream=True
                )
                if r.status_code != 200:
                    sse_event(handler, {"error": f"Ollama returned {r.status_code}"})
                    return
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
                            # Track <think> blocks without stripping whitespace from tokens
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
                full_text = strip_reasoning_preamble(full_text)
                _post_response_tasks(handler, strat, ollama_host, model, user_msg, full_text,
                                     persona_name, profile_id, rp_mode, active_npc, active_scenario)
                sse_event(handler, {"done": True})
            except Exception as e:
                sse_event(handler, {"error": str(e)})
            return

        # Build messages for /api/chat
        messages = [{"role": "system", "content": system_prompt}]
        for h in history[:-1]:
            if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_msg})

        start_sse(handler)

        # Tools to send — filtered by persona config
        allowed_tools = persona_config['tools']
        if not serper_available and 'web_search' in allowed_tools:
            allowed_tools = [t for t in allowed_tools if t != 'web_search']
        tools = [t for t in AGENT_TOOLS if t["function"]["name"] in allowed_tools]

        # ── No-tools persona: stream response directly (like free mode but with full prompt) ──
        _stream_temp = 0.85 if _is_rp else 0.5
        if not tools:
            try:
                r = req.post(
                    f"{ollama_host}/api/chat",
                    json={"model": model, "messages": messages, "stream": True,
                          "options": {"temperature": _stream_temp, "num_predict": _num_predict},
                          "think": False},
                    timeout=180, stream=True
                )
                if r.status_code != 200:
                    sse_event(handler, {"error": f"Ollama returned {r.status_code}"})
                    return
                in_think = False
                full_text = ""
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
                _post_response_tasks(handler, strat, ollama_host, model, user_msg, full_text,
                                     persona_name, profile_id, rp_mode, active_npc, active_scenario)
                sse_event(handler, {"done": True})
            except Exception as e:
                sse_event(handler, {"error": str(e)})
            return

        # ── Tool-call loop (max 8 rounds) ──
        _tool_temp = 0.85 if _is_rp else 0.4
        for round_num in range(8):
            try:
                _tool_payload = {
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "stream": False,
                    "options": {
                        "temperature": _tool_temp,
                        "num_predict": _num_predict,
                    },
                }
                if _is_rp:
                    _tool_payload["think"] = False  # Prevent think-block leakage on RP model
                r = req.post(
                    f"{ollama_host}/api/chat",
                    json=_tool_payload,
                    timeout=180
                )
            except Exception as e:
                sse_event(handler, {"error": f"Ollama connection failed: {e}"})
                return

            if r.status_code != 200:
                sse_event(handler, {"error": f"Ollama returned {r.status_code}: {r.text[:200]}"})
                return

            resp = r.json()
            assistant_msg = resp.get("message", {})
            tool_calls = assistant_msg.get("tool_calls", [])
            content = assistant_msg.get("content", "")

            # ── Fallback: parse text-based tool calls ──
            # Some models output tool calls as text instead of
            # structured tool_calls. Detect and extract them.
            if not tool_calls and content:
                parsed = parse_text_tool_calls(content)
                if parsed:
                    tool_calls = parsed
                    # Remove the text tool call from content so it doesn't
                    # leak to the user
                    content = ""
                    assistant_msg = {"role": "assistant", "content": "", "tool_calls": tool_calls}

            if not tool_calls:
                # Final response — clean up LLM output
                text = strip_think_blocks(content)
                text = strip_reasoning_preamble(text)
                if not text and round_num < 7:
                    # Empty or reasoning-only response. Retry with a direct nudge.
                    messages.append({"role": "assistant", "content": ""})
                    messages.append({"role": "user", "content": "(Respond directly to the user. No internal reasoning or thought process. Just the answer.)"})
                    logger.warning(f"Agent: empty/reasoning-only response on round {round_num}, retrying with nudge")
                    continue
                for char in text:
                    sse_event(handler, {"token": char})
                _post_response_tasks(handler, strat, ollama_host, model, user_msg, text,
                                     persona_name, profile_id, rp_mode, active_npc, active_scenario)
                sse_event(handler, {"done": True})
                return

            # Execute tool calls
            messages.append(assistant_msg)
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                # Handle string arguments (some models stringify JSON)
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                logger.info(f"Agent tool: {name}({json.dumps(args)[:200]})")
                sse_event(handler, {"status": f"🔍 {name}..." if name == "web_search" else f"⚙️ {name}..."})
                result = execute_tool(name, args, strat, profile_id=profile_id, persona=persona_name)
                messages.append({"role": "tool", "content": result})

        # If we exhausted all rounds, send whatever we have as a final response
        logger.warning("Agent: max tool rounds reached, forcing final response")
        messages.append({"role": "user", "content": "(Summarize what you found so far and respond to the user now.)"})
        try:
            r = req.post(
                f"{ollama_host}/api/chat",
                json={"model": model, "messages": messages, "stream": False,
                      "options": {"temperature": 0.4, "num_predict": 2000}},
                timeout=120)
            if r.status_code == 200:
                text = strip_think_blocks(r.json().get("message", {}).get("content", ""))
                text = strip_reasoning_preamble(text)
                if text:
                    for char in text:
                        sse_event(handler, {"token": char})
                    _post_response_tasks(handler, strat, ollama_host, model, user_msg, text,
                                         persona_name, profile_id, rp_mode, active_npc, active_scenario)
                    sse_event(handler, {"done": True})
                    return
        except Exception as e:
            logger.warning(f"Agent: final forced response failed: {e}")
        sse_event(handler, {"error": "Max tool rounds exceeded — try a simpler question"})

    except ValueError as e:
        logger.warning(f"Agent chat validation error: {e}")
        error_response(handler, str(e), 400)
    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        error_response(handler, "Internal server error", 500)


# ═══════════════════════════════════════════════════════════
# SINGLE-SHOT ASK
# ═══════════════════════════════════════════════════════════

def handle_ask(handler, strat, output_dir):
    try:
        body = read_json_body(handler)
        question = body.get("question", "").strip()
        item_title = body.get("title", "")
        item_summary = body.get("summary", "")
        item_url = body.get("url", "")
        item_score = body.get("score", "")
        item_reason = body.get("reason", "")
        item_content = body.get("content", "")
        item_category = body.get("category", "")
        if not question:
            raise ValueError("No question")
        if len(question) > 10_000:
            question = question[:10_000]

        article_ctx = f"Title: {item_title}\nURL: {item_url}\nCategory: {item_category}\nScore: {item_score} — {item_reason}\nSummary: {item_summary}\nContent: {(item_content or item_summary)[:2000]}"
        prompt = f"News signal:\n\n{article_ctx}\n\nQuestion: {question}"

        scorer = strat.scorer
        # Build system prompt with user context
        profile = strat.config.get("profile", {})
        role = profile.get("role", "").strip()
        location = profile.get("location", "").strip()
        user_ctx = ""
        if role or location:
            parts = []
            if role:
                parts.append(f"role: {role}")
            if location:
                parts.append(f"location: {location}")
            user_ctx = f" The user is a professional ({', '.join(parts)}). Tailor your analysis to their perspective."

        # Check Ollama + inference model directly
        try:
            _r = req.get(f"{scorer.host}/api/tags", timeout=5)
            if _r.status_code != 200:
                raise RuntimeError("Ollama unavailable")
        except req.exceptions.ConnectionError:
            raise RuntimeError("Ollama unavailable")

        response = req.post(
            f"{scorer.host}/api/chat",
            json={"model": scorer.inference_model,
                  "messages": [
                      {"role": "system", "content": f"Concise analyst. Answer in 2-4 sentences max. Lead with the actionable insight. Be honest if info is insufficient. Respond directly.{user_ctx}"},
                      {"role": "user", "content": prompt},
                  ],
                  "stream": False, "think": False,
                  "options": {"temperature": 0.5, "num_predict": 500, "num_ctx": 4096}},
            timeout=60)
        answer = ""
        if response.status_code == 200:
            answer = response.json().get("message", {}).get("content", "").strip()
            answer = strip_think_blocks(answer)
            answer = strip_reasoning_preamble(answer)
        json_response(handler, {"answer": answer})
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        error_response(handler, "Internal server error", 500)


# ═══════════════════════════════════════════════════════════
# FILE ASSIST — lightweight LLM for file editor operations
# ═══════════════════════════════════════════════════════════

def handle_file_assist(handler, strat):
    """POST /api/file-assist — Fast LLM call for file editing (revise, continue, grammar, etc.)"""
    try:
        body = read_json_body(handler)
        action = body.get("action", "").strip()
        content = body.get("content", "").strip()
        filename = body.get("filename", "file")
        instruction = body.get("instruction", "").strip()
        if not content:
            raise ValueError("No content")
        MAX_FILE_ASSIST_LEN = 100_000
        if len(content) > MAX_FILE_ASSIST_LEN:
            content = content[:MAX_FILE_ASSIST_LEN]
            logger.warning(f"file-assist: content truncated to {MAX_FILE_ASSIST_LEN} chars")

        # Action-specific system prompts and user prompts
        word_count = len(content.split())
        if action == "continue":
            system = ("You are a creative writer continuing a story/document. "
                       "Read the existing text carefully — match the tone, style, characters, and setting exactly. "
                       "Continue naturally from where the text ends. Write 2-4 paragraphs. "
                       "Output ONLY the continuation text, nothing else.")
            user_msg = content
            max_tokens = min(max(word_count, 200), 1000)
            temperature = 0.7
        elif action == "revise":
            system = ("You are an editor. Improve the writing quality: better word choice, flow, clarity. "
                       "Keep the same meaning, characters, setting, and structure. "
                       "Output ONLY the revised text, nothing else.")
            user_msg = content
            max_tokens = min(max(word_count * 2, 200), 2000)
            temperature = 0.5
        elif action == "grammar":
            system = ("You are a proofreader. Fix grammar, spelling, and punctuation errors ONLY. "
                       "Do NOT change meaning, style, or word choice. "
                       "Output ONLY the corrected text, nothing else.")
            user_msg = content
            max_tokens = min(max(word_count * 2, 100), 2000)
            temperature = 0.2
        elif action == "summarize":
            system = "Summarize the given text in 2-3 concise sentences. Output ONLY the summary."
            user_msg = content
            max_tokens = 150
            temperature = 0.3
        elif action == "custom":
            if not instruction:
                raise ValueError("No instruction provided")
            system = ("You are a writing assistant. Follow the user's instruction precisely. "
                       "Apply it to the provided text. Output ONLY the result, no explanations.")
            user_msg = f"Instruction: {instruction}\n\n---\n\n{content}"
            max_tokens = min(max(word_count * 2, 200), 2000)
            temperature = 0.5
        else:
            raise ValueError(f"Unknown action: {action}")

        # Estimate context size needed — content tokens + output tokens + overhead
        est_input_tokens = int(word_count * 1.3) + 100
        num_ctx = min(max(est_input_tokens + max_tokens + 256, 1024), 4096)

        scorer = strat.scorer
        response = req.post(
            f"{scorer.host}/api/chat",
            json={
                "model": scorer.inference_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx},
            },
            timeout=90,
        )
        answer = ""
        if response.status_code == 200:
            answer = response.json().get("message", {}).get("content", "").strip()
            answer = strip_think_blocks(answer)
            answer = strip_reasoning_preamble(answer)
        json_response(handler, {"result": answer, "action": action})
    except Exception as e:
        logger.error(f"file-assist error: {e}")
        error_response(handler, str(e), 500)


# ═══════════════════════════════════════════════════════════
# SUGGEST CONTEXT
# ═══════════════════════════════════════════════════════════

def handle_suggest_context(handler, strat):
    try:
        data = read_json_body(handler)
        role = data.get("role", "").strip()
        location = data.get("location", "").strip()
        # Fall back to profile config if role not in request body
        if not role:
            profile = strat.config.get("profile", {})
            role = profile.get("role", "").strip()
            location = location or profile.get("location", "").strip()
        if not role:
            raise ValueError("Role required")

        scorer = strat.scorer
        # Check Ollama + inference model directly (not scorer model)
        try:
            _r = req.get(f"{scorer.host}/api/tags", timeout=5)
            if _r.status_code != 200:
                raise RuntimeError("Ollama unavailable")
            _models = [m.get("name","") for m in _r.json().get("models",[])]
            _bases = [n.split(":")[0] for n in _models]
            _inf = getattr(scorer, 'inference_model', None) or scorer.model
            _inf_base = _inf.split(":")[0]
            if not (_inf in _models or _inf_base in _bases or f"{_inf}:latest" in _models):
                raise RuntimeError(f"Inference model '{_inf}' not found in Ollama")
        except req.exceptions.ConnectionError:
            raise RuntimeError("Ollama unavailable")

        system_msg = """You are a concise assistant that suggests tracking interests for a professional using a strategic intelligence dashboard.

The dashboard monitors: career opportunities, financial deals, technology trends, and market movements.

CRITICAL — ROLE-SPECIFIC RELEVANCE:
Your suggestions MUST be tightly relevant to the user's ACTUAL profession. Do NOT default to generic tech topics.
- A geophysicist tracks: seismic methods, reservoir characterization, exploration tech, subsurface imaging — NOT semiconductors or ICT certifications.
- A software engineer tracks: programming frameworks, cloud platforms, SaaS companies — NOT oil drilling or seismic processing.
- A civil engineer tracks: construction companies, infrastructure mega-projects, BIM software — NOT quantum computing.
- A banker tracks: financial regulation, interest rates, fintech — NOT petrochemicals.
ONLY suggest topics that someone in this exact role would read about daily.

KUWAIT DOMAIN KNOWLEDGE:
- K-Sector (oil/gas employers): KOC, KNPC, KIPIC, KPC, Equate, PIC, KUFPEC
- International oil/gas in Kuwait: SLB (Schlumberger), Halliburton, Baker Hughes, CGG, WesternGeco, PetroVista
- Tech/Telecom: Zain, Ooredoo, STC Kuwait, KNET, Alghanim
- Banks: NBK, Boubyan, KFH, Warba, Burgan, Gulf Bank

Rules:
- Output TWO paragraphs:
  PARAGRAPH 1 (TRACK): What to track and why. Be SPECIFIC to their exact role — companies, technologies, sectors, certifications relevant to THEIR field. Write in second person. 3-4 sentences.
  PARAGRAPH 2 (IGNORE): What to DEPRIORITIZE. Start with "Deprioritize:" followed by specifics. 1-2 sentences.
- Do NOT use bullet points, lists, or numbering.
- Do NOT include Chinese text.
- Do NOT start with "As a..."
- Keep under 120 words.

After paragraphs, on a NEW LINE:
TICKERS: 3-8 Yahoo Finance symbols relevant to this role.
- Exact format: AAPL, GC=F, BTC-USD, ^GSPC
- Match their SPECIFIC industry (oil/gas person → CL=F, BZ=F, energy stocks; tech person → NVDA, MSFT, etc.)."""

        paren_notes = re.findall(r'\(([^)]+)\)', role)
        notes_line = "\nIMPORTANT NOTES: " + "; ".join(paren_notes) if paren_notes else ""

        prompt = f"Role: {role}\nLocation: {location or 'Not specified'}{notes_line}\n\nWrite tracking profile + tickers:"

        response = req.post(
            f"{scorer.host}/api/chat",
            json={"model": _inf,
                  "messages": [
                      {"role": "system", "content": system_msg},
                      {"role": "user", "content": prompt},
                  ],
                  "stream": False, "think": False,
                  "options": {"temperature": 0.5, "num_predict": 500, "num_ctx": 2048}},
            timeout=60)

        suggestion, tickers = "", []
        if response.status_code == 200:
            raw = response.json().get("message", {}).get("content", "").strip().strip("\"'")
            raw = strip_think_blocks(raw)
            raw = strip_reasoning_preamble(raw)
            raw = re.sub(r'[^\x00-\x7F]+', '', raw).strip()

            lines = raw.split('\n')
            text_lines, ticker_line = [], ""
            for line in lines:
                s = line.strip()
                if s.upper().startswith('TICKERS:') or s.upper().startswith('TICKER:'):
                    ticker_line = s.split(':', 1)[1].strip()
                else:
                    text_lines.append(line)

            if ticker_line:
                candidates = [t.strip().strip('.,') for t in ticker_line.replace(';', ',').split(',')]
                valid_re = re.compile(r'^[A-Z0-9\^][A-Z0-9.\-=]{0,12}$')
                tickers = [t for t in candidates if t and valid_re.match(t)]

            raw = '\n'.join(text_lines).strip()
            for prefix in ['As a ', 'For a ', 'Here are ', 'Sure, ', 'Sure: ', "Here's "]:
                if raw.lower().startswith(prefix.lower()):
                    raw = raw[len(prefix):].strip()

            sentences = re.split(r'(?<=[.!?])\s+', raw)
            result, wc = [], 0
            for s in sentences:
                w = len(s.split())
                if wc + w > 150: break
                result.append(s)
                wc += w
            suggestion = ' '.join(result).strip()
            if suggestion and suggestion[-1] not in '.!?':
                suggestion += '.'

        json_response(handler, {"suggestion": suggestion, "tickers": tickers})
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        error_response(handler, "Internal server error", 500)


# ═══════════════════════════════════════════════════════════
# CONTEXT BUILDERS
# ═══════════════════════════════════════════════════════════

def _build_historical_context(strat, profile_id=0):
    parts = []
    db = strat.db
    try:
        scans = db.get_scan_log(5, profile_id=profile_id)
        if scans:
            lines = [f"  {s.get('started_at','')[:16].replace('T',' ')}: {s.get('items_scored',0)} items → {s.get('critical',0)} crit, {s.get('high',0)} high" if not s.get('error') else f"  {s.get('started_at','')[:16]}: FAILED" for s in scans]
            parts.append("RECENT SCANS:\n" + "\n".join(lines))
    except Exception as e:
        logger.debug(f"Agent context: failed to load scan log: {e}")
    _pid = profile_id
    try:
        stats = db.get_category_stats(days=7, profile_id=_pid)
        if stats:
            lines = [f"  {c.get('category','?')}: {c.get('total',0)} items, avg {c.get('avg_score',0)}, {c.get('critical',0)} crit, {c.get('high',0)} high" for c in stats[:10]]
            parts.append("CATEGORY PERFORMANCE (7d):\n" + "\n".join(lines))
    except Exception as e:
        logger.debug(f"Agent context: failed to load category stats: {e}")
    try:
        top = db.get_top_signals(days=7, min_score=7.5, limit=10, profile_id=_pid)
        if top:
            lines = [f"  [{t.get('score',0):.1f}] {t.get('title','')[:80]} ({t.get('category','')}, {t.get('fetched_at','')[:10]})" for t in top]
            parts.append("TOP SIGNALS (7d):\n" + "\n".join(lines))
    except Exception as e:
        logger.debug(f"Agent context: failed to load top signals: {e}")
    try:
        daily = db.get_daily_signal_counts(days=7, profile_id=_pid)
        if daily:
            lines = [f"  {d.get('day','?')}: {d.get('total',0)} total, {d.get('critical',0)} crit, {d.get('high',0)} high" for d in daily]
            parts.append("DAILY TREND:\n" + "\n".join(lines))
    except Exception as e:
        logger.debug(f"Agent context: failed to load daily counts: {e}")
    return "\n\n".join(parts)


def _build_agent_context(strat, output_file):
    """Build agent context from the profile-specific output file."""
    news_context = ""
    output_path = Path(output_file)
    if not output_path.exists():
        return news_context
    try:
        with open(output_path) as f:
            scraped = json.loads(f.read())

        news_items = scraped.get("news", [])
        top = sorted([x for x in news_items if isinstance(x, dict)], key=lambda x: x.get("score", 0), reverse=True)[:30]
        lines = []
        for it in top:
            try:
                lines.append(f"[{float(it.get('score',0)):.1f}] {it.get('title','')} ({it.get('source','')}, {it.get('category',it.get('root',''))}) — {str(it.get('summary',''))[:200]}")
            except Exception as e:
                logger.debug(f"Agent context: skipping malformed news item: {e}")
                continue
        news_context = "\n".join(lines)

        market = scraped.get("market", {})
        mlines = []
        for sym, md in market.items():
            try:
                if not isinstance(md, dict): continue
                name = md.get("name", sym)
                data_dict = md.get("data", {}) if isinstance(md.get("data"), dict) else {}
                # Collect multiple timeframes for richer context
                parts = []
                for k in ["1m", "5m", "1d_1mo", "1d_1y", "1wk"]:
                    d = data_dict.get(k)
                    if isinstance(d, dict) and "price" in d:
                        p = float(d.get("price", 0))
                        c = float(d.get("change", 0))
                        high = d.get("high")
                        low = d.get("low")
                        hl = f", H/L: ${float(high):.2f}/${float(low):.2f}" if high and low else ""
                        hist = d.get("history", [])
                        trend = ""
                        if isinstance(hist, list) and len(hist) >= 5:
                            r5 = hist[-5:]
                            if all(isinstance(x, (int, float)) for x in r5):
                                trend = f", trend: {'rising' if r5[-1]>r5[0] else 'falling' if r5[-1]<r5[0] else 'flat'}"
                        parts.append(f"  {k}: ${p:.2f} ({c:+.2f}%){hl}{trend}")
                if parts:
                    mlines.append(f"{name} ({sym}):\n" + "\n".join(parts[:3]))  # Top 3 timeframes
            except Exception as e:
                logger.debug(f"Agent context: skipping malformed market entry: {e}")
                continue
        if mlines:
            ts = scraped.get("timestamps", {}).get("market", "")
            ts_label = f" (as of {ts[:16].replace('T',' ')})" if ts else ""
            news_context += f"\n\nMARKET DATA{ts_label}:\n" + "\n".join(mlines)

        briefing = scraped.get("briefing", {})
        if isinstance(briefing, dict) and briefing:
            bp = []
            alerts = briefing.get("critical_alerts", [])
            if isinstance(alerts, list) and alerts:
                al = [f"- {a.get('headline',a.get('title',''))} ({a.get('score',0)}): {str(a.get('analysis',''))[:150]}" for a in alerts[:5] if isinstance(a, dict)]
                if al: bp.append("CRITICAL ALERTS:\n" + "\n".join(al))
            picks = briefing.get("high_priority", [])
            if isinstance(picks, list) and picks:
                pl = [f"- {p.get('title','')} ({p.get('score',0)})" for p in picks[:5] if isinstance(p, dict)]
                if pl: bp.append("TOP PICKS:\n" + "\n".join(pl))
            if briefing.get("market_summary"):
                bp.append("MARKET: " + str(briefing["market_summary"]))
            if bp:
                news_context = "\n\n".join(bp) + "\n\n" + news_context
    except Exception as e:
        logger.warning(f"Agent context error: {e}")
    return news_context
