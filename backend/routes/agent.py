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
    except Exception:
        pass
    json_response(handler, {"available": available, "model": model, "host": ollama_host})


# ═══════════════════════════════════════════════════════════
# SUGGESTION GENERATION
# ═══════════════════════════════════════════════════════════

def _generate_suggestions(handler, ollama_host, model, user_msg, agent_response, persona_name):
    """Generate 3 contextual follow-up suggestions via a lightweight LLM call.
    Sends a 'suggestions' SSE event. Falls back silently on failure."""
    try:
        prompt = (
            f"Based on this conversation, suggest 3 short follow-up actions (3-8 words each). "
            f"They should feel like natural continuations.\n\n"
            f"User: {user_msg[:200]}\n"
            f"Assistant: {agent_response[:500]}\n"
            f"Persona: {persona_name}\n\n"
            f"Return ONLY a JSON array of 3 strings. No explanation.\n"
            f'Example: ["Start the Misty Peaks quest", "Describe my character stats", "Tell me more about X"]'
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
            # Extract JSON array from response
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

        scoring_cfg = strat.config.get("scoring", {})
        ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
        model = scoring_cfg.get("inference_model", "qwen3.5:9b")
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
                ctx = build_persona_context(p, strat, output_file, profile_id)
                if ctx:
                    context_parts.append(f"[{p.upper()} DATA]\n{ctx}")
            persona_config = {**persona_config, 'tools': list(merged_tools)}
            # Add cross-persona note to prompt
            other_names = [p for p in multi_personas if p != persona_name]
            system_prompt = base_prompt + f"\n\nYou also have context from: {', '.join(other_names)}. Use all available data to answer."
            if context_parts:
                system_prompt += "\n\n" + "\n\n".join(context_parts)
        else:
            base_prompt = build_persona_prompt(
                persona_name, role, location, tickers, cat_summary, search_note
            )
            persona_context = build_persona_context(
                persona_name, strat, output_file, profile_id
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
                    except Exception:
                        pass
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

        # ── Free chat mode: no tools, simple system prompt ──
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
                          "options": {"temperature": 0.7, "num_predict": 3000}},
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
                _generate_suggestions(handler, ollama_host, model, user_msg, full_text, persona_name)
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
        if not tools:
            try:
                r = req.post(
                    f"{ollama_host}/api/chat",
                    json={"model": model, "messages": messages, "stream": True,
                          "options": {"temperature": 0.5, "num_predict": 3000}},
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
                _generate_suggestions(handler, ollama_host, model, user_msg, full_text, persona_name)
                sse_event(handler, {"done": True})
            except Exception as e:
                sse_event(handler, {"error": str(e)})
            return

        # ── Tool-call loop (max 8 rounds) ──
        for round_num in range(8):
            try:
                r = req.post(
                    f"{ollama_host}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "stream": False,
                        "options": {
                            "temperature": 0.4,
                            "num_predict": 3000,
                        },
                        # Qwen3.5 separates reasoning into thinking field automatically.
                    },
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
                _generate_suggestions(handler, ollama_host, model, user_msg, text, persona_name)
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
                    _generate_suggestions(handler, ollama_host, model, user_msg, text, persona_name)
                    sse_event(handler, {"done": True})
                    return
        except Exception:
            pass
        sse_event(handler, {"error": "Max tool rounds exceeded — try a simpler question"})

    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        error_response(handler, "Internal server error")


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
# SUGGEST CONTEXT
# ═══════════════════════════════════════════════════════════

def handle_suggest_context(handler, strat):
    try:
        data = read_json_body(handler)
        role = data.get("role", "").strip()
        location = data.get("location", "").strip()
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
            json={"model": scorer.inference_model,
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
            except Exception: continue
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
            except Exception: continue
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
