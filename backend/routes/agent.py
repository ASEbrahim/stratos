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
import yaml
import requests as req
from pathlib import Path
from typing import Dict, Any, List, Optional

from routes.helpers import (json_response, error_response, read_json_body, sse_event, start_sse,
                           strip_think_blocks, strip_reasoning_preamble)

logger = logging.getLogger("STRAT_OS")


# ═══════════════════════════════════════════════════════════
# TOOL DEFINITIONS (sent to Ollama so the LLM knows what's available)
# ═══════════════════════════════════════════════════════════

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using Google for real-time information. Use this when the user asks to look something up, search for something, find current news, or when your existing feed data doesn't have what they need. Always prefer this for questions about current events, job postings, bank offers, or anything that needs live data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific — include company names, locations, dates."
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["web", "news"],
                        "description": "Use 'news' for recent news articles, 'web' for general. Default: web"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_watchlist",
            "description": "View, add, or remove stock/crypto/commodity tickers from the user's market watchlist. Use when the user asks about their tickers, wants to track a new asset, or remove one. Understand natural language: 'track Tesla' = add TSLA, 'stop following gold' = remove GC=F, 'what am I watching' = list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "add", "remove"],
                        "description": "list = show watchlist, add = add tickers, remove = remove tickers"
                    },
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ticker symbols (e.g. ['TSLA','ETH-USD','SI=F']). Not needed for list."
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_categories",
            "description": "View, modify, or toggle the user's news feed categories and their search keywords. Use when the user asks about categories, wants to add/remove search terms, or enable/disable a category. Understand natural language: 'what topics am I tracking' = list, 'add Ethereum to crypto' = add_keyword, 'turn off oil and gas' = disable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "show_keywords", "add_keyword", "remove_keyword", "enable", "disable"],
                        "description": "list = all categories, show_keywords = keywords in one category, add/remove_keyword = modify keywords, enable/disable = toggle"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category name (fuzzy matched). Required for all except list."
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords to add/remove. Only for add_keyword/remove_keyword."
                    }
                },
                "required": ["action"]
            }
        }
    }
]


# ═══════════════════════════════════════════════════════════
# TOOL EXECUTION
# ═══════════════════════════════════════════════════════════

def _execute_tool(tool_name, args, strat):
    """Execute a tool call and return the result string."""
    try:
        if tool_name == "web_search":
            return _tool_web_search(args, strat)
        elif tool_name == "manage_watchlist":
            return _tool_manage_watchlist(args, strat)
        elif tool_name == "manage_categories":
            return _tool_manage_categories(args, strat)
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        logger.error(f"Tool error ({tool_name}): {e}")
        return f"Tool error: {e}"






def _parse_text_tool_calls(content: str) -> list:
    """Parse tool calls that Qwen3 outputs as text instead of structured JSON.

    Handles patterns like:
      <function-call>{"name":"web_search","arguments":{...}}</function-call>
      <tool_call>{"name":"web_search","arguments":{...}}</tool_call>
      {"name":"web_search","arguments":{...}}   (bare JSON at start)

    Returns list of tool_call dicts compatible with Ollama format, or [].
    """
    tool_calls = []
    text = content.strip()

    # Pattern 1: <function-call> ... </function-call>  or  <tool_call> ... </tool_call>
    tag_patterns = [
        r'<function-call>\s*(.*?)\s*</function-call>',
        r'<tool_call>\s*(.*?)\s*</tool_call>',
        r'<function_call>\s*(.*?)\s*</function_call>',
    ]
    for pattern in tag_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            parsed = _try_parse_tool_json(match.strip())
            if parsed:
                tool_calls.append(parsed)

    if tool_calls:
        return tool_calls

    # Pattern 2: bare JSON that looks like a tool call
    # Only if the entire content looks like a JSON tool call (not mixed with text)
    if text.startswith('{') and '"name"' in text:
        # Try to extract just the JSON portion
        try:
            # Find matching closing brace
            brace_depth = 0
            end_idx = 0
            for i, ch in enumerate(text):
                if ch == '{':
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        end_idx = i + 1
                        break
            if end_idx > 0:
                parsed = _try_parse_tool_json(text[:end_idx])
                if parsed:
                    tool_calls.append(parsed)
        except Exception:
            pass

    return tool_calls


def _try_parse_tool_json(text: str) -> dict:
    """Try to parse a JSON string as a tool call. Returns Ollama-format dict or None."""
    try:
        data = json.loads(text)
        name = data.get("name", "")
        arguments = data.get("arguments", data.get("parameters", {}))
        if name and name in ("web_search", "manage_watchlist", "manage_categories"):
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            return {
                "function": {
                    "name": name,
                    "arguments": arguments
                }
            }
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def _tool_web_search(args, strat):
    from fetchers.serper_search import get_serper_client
    query = args.get("query", "").strip()
    search_type = args.get("search_type", "web")
    if not query:
        return "Error: No search query."

    client = get_serper_client(strat.config)
    if not client:
        return "Web search unavailable — no Serper API key. User can add one in Settings."

    status = client.get_status()
    if status.get("limit_reached"):
        return "Serper API limit reached."

    try:
        if search_type == "news":
            results = client.search_news(query, num_results=8, time_period="w")
        else:
            results = client.search(query, num_results=8)
        if not results:
            return f"No results for: {query}"

        lines = []
        for i, r in enumerate(results[:8], 1):
            title = r.get("title", "")
            snippet = r.get("snippet", "")[:200]
            url = r.get("url", "")
            source = r.get("source", "")
            date_str = r.get("date", "")
            meta = f" ({source}, {date_str})" if source or date_str else ""
            lines.append(f"{i}. {title}{meta}\n   {snippet}\n   {url}")
        remaining = status.get("remaining", "?")
        return f"Search: '{query}' ({len(results)} results, {remaining} left):\n\n" + "\n\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"


def _tool_manage_watchlist(args, strat):
    action = args.get("action", "list")
    symbols = [s.strip().upper() for s in args.get("symbols", []) if s.strip()]
    tickers = [t.get("symbol", "") for t in strat.config.get("market", {}).get("tickers", [])]

    if action == "list":
        return f"Watchlist ({len(tickers)}): {', '.join(tickers)}" if tickers else "Watchlist is empty."

    if action == "add":
        if not symbols:
            return "No symbols to add."
        adding = [s for s in symbols if s not in tickers]
        already = [s for s in symbols if s in tickers]
        tickers = tickers + adding
        _save_tickers(strat, tickers)
        parts = []
        if adding: parts.append(f"Added: {', '.join(adding)}")
        if already: parts.append(f"Already there: {', '.join(already)}")
        parts.append(f"Watchlist: {', '.join(tickers)} ({len(tickers)})")
        return ". ".join(parts)

    if action == "remove":
        if not symbols:
            return "No symbols to remove."
        removing = [s for s in symbols if s in tickers]
        not_found = [s for s in symbols if s not in tickers]
        tickers = [t for t in tickers if t not in symbols]
        _save_tickers(strat, tickers)
        parts = []
        if removing: parts.append(f"Removed: {', '.join(removing)}")
        if not_found: parts.append(f"Not found: {', '.join(not_found)}")
        parts.append(f"Watchlist: {', '.join(tickers) or '(empty)'} ({len(tickers)})")
        return ". ".join(parts)

    return f"Unknown action: {action}"


def _tool_manage_categories(args, strat):
    action = args.get("action", "list")
    cat_query = args.get("category", "").strip()
    keywords = [k.strip() for k in args.get("keywords", []) if k.strip()]
    categories = strat.config.get("dynamic_categories", [])

    if action == "list":
        if not categories:
            return "No categories configured."
        lines = [f"- {c.get('label','?')} ({'on' if c.get('enabled',True) else 'off'}, {len(c.get('items',[]))} kw)" for c in categories]
        return f"Categories ({len(categories)}):\n" + "\n".join(lines)

    cat = _fuzzy_find_category(categories, cat_query)
    if not cat:
        names = ", ".join(c.get("label", "?") for c in categories)
        return f"No category matching '{cat_query}'. Available: {names}"

    if action == "show_keywords":
        items = cat.get("items", [])
        return f"{cat['label']} ({len(items)} kw): {', '.join(items)}" if items else f"{cat['label']} has no keywords."

    if action == "add_keyword":
        if not keywords:
            return "No keywords to add."
        existing = [i.lower() for i in cat.get("items", [])]
        adding = [k for k in keywords if k.lower() not in existing]
        cat["items"] = cat.get("items", []) + adding
        _save_categories(strat, categories)
        return f"Added to {cat['label']}: {', '.join(adding)}. Now {len(cat['items'])} keywords." if adding else "All keywords already exist."

    if action == "remove_keyword":
        if not keywords:
            return "No keywords to remove."
        kw_lower = [k.lower() for k in keywords]
        cat["items"] = [i for i in cat.get("items", []) if i.lower() not in kw_lower]
        _save_categories(strat, categories)
        return f"Removed from {cat['label']}. Now {len(cat['items'])} keywords."

    if action in ("enable", "disable"):
        cat["enabled"] = (action == "enable")
        _save_categories(strat, categories)
        return f"{cat['label']} is now {'enabled' if cat['enabled'] else 'disabled'}."

    return f"Unknown action: {action}"


def _fuzzy_find_category(categories, query):
    q = query.lower().strip()
    if not q:
        return None
    for c in categories:
        if c.get("id", "").lower() == q or c.get("label", "").lower() == q:
            return c
    for c in categories:
        if q in c.get("label", "").lower() or q in c.get("id", "").lower():
            return c
        if c.get("label", "").lower() in q or c.get("id", "").lower() in q:
            return c
    return None


def _save_tickers(strat, symbols):
    from routes.config import _parse_ticker_objects
    objs = _parse_ticker_objects(symbols)
    if "market" not in strat.config:
        strat.config["market"] = {}
    strat.config["market"]["tickers"] = objs
    _write_config(strat)
    try:
        from fetchers.market import MarketFetcher
        strat.market_fetcher = MarketFetcher(strat.config.get("market", {}))
    except Exception:
        pass


def _save_categories(strat, categories):
    strat.config["dynamic_categories"] = categories
    _write_config(strat)


def _write_config(strat):
    try:
        with open(strat.config_path, "w") as f:
            yaml.dump(strat.config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    except Exception as e:
        logger.error(f"Config write failed: {e}")


# ═══════════════════════════════════════════════════════════
# MAIN AGENT CHAT HANDLER
# ═══════════════════════════════════════════════════════════

def handle_agent_status(handler, strat):
    scoring_cfg = strat.config.get("scoring", {})
    ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
    model = scoring_cfg.get("inference_model", "qwen3:30b-a3b")
    available = False
    try:
        r = req.get(f"{ollama_host}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            available = any(model.split(":")[0] in m for m in models)
    except Exception:
        pass
    json_response(handler, {"available": available, "model": model, "host": ollama_host})


def handle_agent_chat(handler, strat, output_dir, profile_id=0):
    """POST /api/agent-chat — Streaming agent conversation with tool use."""
    try:
        body = read_json_body(handler)
        user_msg = body.get("message", "").strip()
        history = body.get("history", [])
        if not user_msg:
            raise ValueError("Empty message")

        news_context = _build_agent_context(strat, output_dir)
        historical_context = _build_historical_context(strat, profile_id)

        scoring_cfg = strat.config.get("scoring", {})
        ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
        model = scoring_cfg.get("inference_model", "qwen3:30b-a3b")  # Use inference model, not scorer
        profile = strat.config.get("profile", {})
        role = profile.get("role", "user")
        location = profile.get("location", "")

        serper_available = bool(strat.config.get("search", {}).get("serper_api_key", ""))
        search_note = "You have a web_search tool for real-time Google search." if serper_available else "Web search not configured — feed data only."

        tickers = [t.get("symbol", "") for t in strat.config.get("market", {}).get("tickers", [])]
        cats = strat.config.get("dynamic_categories", [])
        cat_summary = ", ".join(f"{c.get('label','')} ({len(c.get('items',[]))} kw)" for c in cats[:10])

        system_prompt = f"""You are STRAT AGENT, an AI assistant in a strategic intelligence dashboard (StratOS).

USER PROFILE: {role} in {location}
WATCHLIST: {', '.join(tickers) if tickers else '(empty)'}
CATEGORIES: {cat_summary or '(none)'}

TOOLS:
1. {search_note}
2. manage_watchlist — add/remove/list tickers. Understand natural language: "track Tesla" = add TSLA, "stop following gold" = remove GC=F, "what's on my watchlist" = list.
3. manage_categories — add/remove keywords, list/toggle categories. "add Ethereum to crypto" = add_keyword, "what topics am I tracking" = list, "show me what's in hardware" = show_keywords.

RULES:
- Be concise. Lead with the most actionable insight. Use bullet points for multiple items. Keep total response under 200 words unless the user explicitly asks for detail.
- Default: 3-5 bullet points or 2-3 short paragraphs max. No padding, no filler.
- For questions about current events, jobs, bank offers, or anything NOT in the feed — use web_search tool. DO NOT fabricate information.
- When the user wants to modify tickers or categories (even in casual language), use the appropriate tool.
- Use **bold** for key terms. Numbered lists only for 3+ items.
- When using web_search, synthesize results — connect them to the user's interests and feed data.
- If data is not available and search returns nothing, say so honestly.
- Match the user's tone. Be direct and natural.
- NEVER output raw JSON, XML tags, or function call syntax in your response. Always respond in natural language.
- When you have nothing useful to add, say so briefly rather than giving an empty response.
- CRITICAL: Respond DIRECTLY to the user. Do NOT narrate your thought process ("Okay, the user asked...", "Let me check...", "I need to..."). Just answer.

CURRENT FEED DATA:
{news_context[:5000]}

HISTORICAL DATA:
{historical_context[:3000]}"""

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

        # Build messages for /api/chat
        messages = [{"role": "system", "content": system_prompt}]
        for h in history[:-1]:
            if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_msg})

        start_sse(handler)

        # Tools to send
        tools = AGENT_TOOLS if serper_available else [t for t in AGENT_TOOLS if t["function"]["name"] != "web_search"]

        # ── Tool-call loop (max 3 rounds) ──
        for round_num in range(3):
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
                        # Let Qwen3 think naturally — thinking goes to separate field.
                        # think:false causes reasoning to leak into content as plain text.
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
            # Qwen3 MoE sometimes outputs tool calls as text instead of
            # structured tool_calls. Detect and extract them.
            if not tool_calls and content:
                parsed = _parse_text_tool_calls(content)
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
                if not text and round_num < 2:
                    # Empty or reasoning-only response. Retry with a direct nudge.
                    messages.append({"role": "assistant", "content": ""})
                    messages.append({"role": "user", "content": "(Respond directly to the user. No internal reasoning or thought process. Just the answer.)"})
                    logger.warning(f"Agent: empty/reasoning-only response on round {round_num}, retrying with nudge")
                    continue
                for char in text:
                    sse_event(handler, {"token": char})
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
                result = _execute_tool(name, args, strat)
                messages.append({"role": "tool", "content": result})

        sse_event(handler, {"error": "Max tool rounds exceeded"})

    except Exception as e:
        logger.error(f"Agent chat error: {e}")
        error_response(handler, str(e))


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
                      {"role": "system", "content": "Concise analyst. Answer in 2-4 sentences max. Lead with the actionable insight. Be honest if info is insufficient. No reasoning, no think tags. Respond directly."},
                      {"role": "user", "content": prompt},
                  ],
                  "stream": False, "options": {"temperature": 0.5, "num_predict": 2000, "num_ctx": 4096}},
            timeout=90)
        answer = ""
        if response.status_code == 200:
            answer = response.json().get("message", {}).get("content", "").strip()
            answer = strip_think_blocks(answer)
            answer = strip_reasoning_preamble(answer)
        json_response(handler, {"answer": answer})
    except Exception as e:
        error_response(handler, str(e), 500)


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
                  "stream": False, "options": {"temperature": 0.5, "num_predict": 2500, "num_ctx": 2048}},
            timeout=120)  # 120s — model swap can take 30-40s on first call

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
        error_response(handler, str(e), 500)


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
    except Exception: pass
    _pid = profile_id
    try:
        stats = db.get_category_stats(days=7, profile_id=_pid)
        if stats:
            lines = [f"  {c.get('category','?')}: {c.get('total',0)} items, avg {c.get('avg_score',0)}, {c.get('critical',0)} crit, {c.get('high',0)} high" for c in stats[:10]]
            parts.append("CATEGORY PERFORMANCE (7d):\n" + "\n".join(lines))
    except Exception: pass
    try:
        top = db.get_top_signals(days=7, min_score=7.5, limit=10, profile_id=_pid)
        if top:
            lines = [f"  [{t.get('score',0):.1f}] {t.get('title','')[:80]} ({t.get('category','')}, {t.get('fetched_at','')[:10]})" for t in top]
            parts.append("TOP SIGNALS (7d):\n" + "\n".join(lines))
    except Exception: pass
    try:
        daily = db.get_daily_signal_counts(days=7, profile_id=_pid)
        if daily:
            lines = [f"  {d.get('day','?')}: {d.get('total',0)} total, {d.get('critical',0)} crit, {d.get('high',0)} high" for d in daily]
            parts.append("DAILY TREND:\n" + "\n".join(lines))
    except Exception: pass
    return "\n\n".join(parts)


def _build_agent_context(strat, output_dir):
    news_context = ""
    output_path = Path(output_dir) / "news_data.json"
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
                db = md.get("data", {}) if isinstance(md.get("data"), dict) else {}
                tf = None
                for k in ["1m","5m","1d_1mo","1d_1y","1wk"]:
                    d = db.get(k) or md.get(k)
                    if isinstance(d, dict) and "price" in d:
                        tf = d; break
                if tf:
                    p, c = float(tf.get("price",0)), float(tf.get("change",0))
                    line = f"{name} ({sym}): ${p:.2f} ({c:+.2f}%)"
                    hist = tf.get("history", [])
                    if isinstance(hist, list) and len(hist) >= 5:
                        r5 = hist[-5:]
                        if all(isinstance(x,(int,float)) for x in r5):
                            line += f" | {'rising' if r5[-1]>r5[0] else 'falling' if r5[-1]<r5[0] else 'flat'}"
                    mlines.append(line)
            except Exception: continue
        if mlines:
            news_context += "\n\nMARKET DATA:\n" + "\n".join(mlines)

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
