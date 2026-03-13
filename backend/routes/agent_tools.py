"""
Agent tool definitions and implementations.
Extracted from agent.py (Sprint 5K Phase 2).

Exports:
  - AGENT_TOOLS: list of Ollama tool definitions
  - execute_tool(name, args, strat, profile_id, persona) → str
  - parse_text_tool_calls(content) → list of tool_call dicts
"""

import json
import re
import logging
import yaml
from typing import Dict, Any, List

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
            "name": "search_feed",
            "description": "Search the user's scored news feed history. Use when the user asks about past articles, what scored highest, what they missed, or wants to find articles about a specific topic. Examples: 'what scored highest this week', 'articles about NVIDIA', 'what did I miss yesterday', 'show me dismissed articles about oil'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "top_signals", "daily_summary"],
                        "description": "search = keyword search in past articles, top_signals = highest-scoring articles, daily_summary = signal counts by day"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search keyword(s) for 'search' action. Not needed for top_signals/daily_summary."
                    },
                    "days": {
                        "type": "integer",
                        "description": "How many days back to search. Default: 7"
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum score filter for top_signals. Default: 7.0"
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
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search across the user's uploaded documents (PDFs, text files, images with OCR). Use when the user asks to find something in their files, look up a term in their documents, or reference uploaded content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term to find in uploaded documents."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read the full extracted text content of an uploaded document by its ID. Use after search_files returns results and the user wants to see the full content of a specific document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "integer",
                        "description": "The file ID returned by search_files or list."
                    }
                },
                "required": ["file_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_insights",
            "description": "Search across extracted YouTube video insights. Use when the user asks about what a channel or lecture discussed, or wants to find specific topics from processed videos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term (topic, name, event, etc.)"
                    },
                    "lens": {
                        "type": "string",
                        "enum": ["summary", "eloquence", "history", "spiritual", "politics", "narrations"],
                        "description": "Filter by lens type. Omit to search all lenses."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_channels",
            "description": "List tracked YouTube channels with video counts and configured lenses. Use when the user asks what channels they're tracking or wants to see processing status.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_video_summary",
            "description": "Get all extracted insights for a specific YouTube video. Use when the user asks about a specific video or lecture.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_title": {
                        "type": "string",
                        "description": "Video title or partial title to search for."
                    }
                },
                "required": ["video_title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_narrations",
            "description": "Search for verified or unverified scholarly narrations (hadith, historical citations) from processed videos. Use when the user asks about specific narrations or scholarly citations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term (narrator, topic, hadith text, etc.)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": "Fetch and read the text content of a web page or article. Use when the user pastes a URL and wants you to summarize it, or when you need to read a source found via web_search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to read (must start with http:// or https://)."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "import_canon_world",
            "description": "Import a canon world from an anime, game, show, or franchise into the active scenario. Fetches characters, locations, lore, and power systems from Fandom wikis and populates all scenario files. Use when the user says 'build the SAO world', 'import Witcher', 'create a Naruto scenario', 'implement the Attack on Titan universe', or any variant asking to build/import/create a world based on an existing franchise.",
            "parameters": {
                "type": "object",
                "properties": {
                    "franchise": {
                        "type": "string",
                        "description": "The franchise name (e.g., 'Sword Art Online', 'The Witcher', 'Naruto', 'Avatar')"
                    },
                    "scenario_name": {
                        "type": "string",
                        "description": "Name for the scenario. If not specified, derived from franchise name."
                    }
                },
                "required": ["franchise"]
            }
        }
    }
]


# ═══════════════════════════════════════════════════════════
# TOOL EXECUTION DISPATCHER
# ═══════════════════════════════════════════════════════════

def execute_tool(tool_name, args, strat, profile_id=0, persona=''):
    """Execute a tool call and return the result string."""
    try:
        if tool_name == "web_search":
            return _tool_web_search(args, strat)
        elif tool_name == "search_feed":
            return _tool_search_feed(args, strat, profile_id=profile_id)
        elif tool_name == "manage_watchlist":
            return _tool_manage_watchlist(args, strat)
        elif tool_name == "manage_categories":
            return _tool_manage_categories(args, strat)
        elif tool_name == "search_files":
            return _tool_search_files(args, strat, profile_id=profile_id, persona=persona)
        elif tool_name == "read_document":
            return _tool_read_document(args, strat, profile_id=profile_id)
        elif tool_name == "search_insights":
            return _tool_search_insights(args, strat, profile_id=profile_id)
        elif tool_name == "list_channels":
            return _tool_list_channels(args, strat, profile_id=profile_id)
        elif tool_name == "get_video_summary":
            return _tool_get_video_summary(args, strat, profile_id=profile_id)
        elif tool_name == "search_narrations":
            return _tool_search_narrations(args, strat, profile_id=profile_id)
        elif tool_name == "read_url":
            return _tool_read_url(args, strat)
        elif tool_name == "import_canon_world":
            return _tool_import_canon_world(args, strat, profile_id=profile_id)
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        logger.error(f"Tool error ({tool_name}): {e}")
        return f"Tool error: {e}"


# ═══════════════════════════════════════════════════════════
# TEXT TOOL CALL PARSER (for models that output tools as text)
# ═══════════════════════════════════════════════════════════

def parse_text_tool_calls(content: str) -> list:
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
        if name and name in ("web_search", "search_feed", "manage_watchlist", "manage_categories", "import_canon_world"):
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


# ═══════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════

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


def _tool_search_feed(args, strat, profile_id=0):
    action = args.get("action", "search")
    days = int(args.get("days", 7))
    db = strat.db

    if action == "search":
        query = args.get("query", "").strip()
        if not query:
            return "Error: No search query provided."
        results = db.search_news_history(query, days=days, limit=15, profile_id=profile_id)
        if not results:
            return f"No articles matching '{query}' in the past {days} days."
        lines = [f"{i}. [{r.get('score',0):.1f}] {r.get('title','')[:80]} ({r.get('category','')}, {r.get('fetched_at','')[:10]})\n   {r.get('score_reason','')[:120]}" for i, r in enumerate(results, 1)]
        return f"Feed search '{query}' ({len(results)} results, past {days}d):\n\n" + "\n\n".join(lines)

    if action == "top_signals":
        min_score = float(args.get("min_score", 7.0))
        results = db.get_top_signals(days=days, min_score=min_score, limit=15, profile_id=profile_id)
        if not results:
            return f"No signals scoring above {min_score} in the past {days} days."
        lines = [f"{i}. [{r.get('score',0):.1f}] {r.get('title','')[:80]} ({r.get('category','')}, {r.get('fetched_at','')[:10]})\n   {r.get('score_reason','')[:120]}" for i, r in enumerate(results, 1)]
        return f"Top signals (>={min_score}, past {days}d, {len(results)} results):\n\n" + "\n\n".join(lines)

    if action == "daily_summary":
        results = db.get_daily_signal_counts(days=days, profile_id=profile_id)
        if not results:
            return f"No scan data in the past {days} days."
        lines = [f"  {d.get('day','?')}: {d.get('total',0)} total, {d.get('critical',0)} critical, {d.get('high',0)} high, {d.get('medium',0)} medium" for d in results]
        return f"Daily signal summary (past {days}d):\n" + "\n".join(lines)

    return f"Unknown action: {action}"


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


def _tool_search_files(args, strat, profile_id=0, persona=''):
    """Search user's uploaded documents, scoped to active persona."""
    query = args.get("query", "").strip()
    if not query:
        return "No search query provided."
    try:
        from processors.file_handler import FileHandler
        fh = FileHandler(strat.config, db=strat.db)
        results = fh.search_files(profile_id, query, limit=10, persona=persona)
        if not results:
            return f"No documents found matching '{query}'."
        lines = []
        for r in results:
            lines.append(f"[ID:{r['id']}] {r['filename']} ({r['file_type']}, {r['uploaded_at'][:10]})")
            if r.get('snippet'):
                lines.append(f"  ...{r['snippet'].strip()[:200]}...")
        return f"Found {len(results)} document(s) matching '{query}':\n" + "\n".join(lines)
    except Exception as e:
        return f"File search error: {e}"


def _tool_read_document(args, strat, profile_id=0):
    """Read full text content of an uploaded document."""
    file_id = args.get("file_id")
    if file_id is None:
        return "No file_id provided."
    try:
        from processors.file_handler import FileHandler
        fh = FileHandler(strat.config, db=strat.db)
        content = fh.get_file_content(profile_id, int(file_id))
        if content is None:
            return f"Document {file_id} not found or not accessible."
        if not content:
            return f"Document {file_id} exists but has no extracted text content."
        return f"Document content (file_id={file_id}):\n{content[:8000]}"
    except Exception as e:
        return f"Read document error: {e}"


def _tool_search_insights(args, strat, profile_id=0):
    """Search extracted YouTube video insights."""
    query = args.get("query", "").strip()
    if not query:
        return "No search query provided."
    try:
        from processors.youtube import YouTubeProcessor
        yt = YouTubeProcessor(strat.config, db=strat.db)
        lens = args.get("lens")
        results = yt.search_insights(profile_id, query, lens_name=lens, limit=10)
        if not results:
            return f"No insights found matching '{query}'."
        lines = []
        for r in results:
            content = r.get('content', {})
            preview = json.dumps(content, ensure_ascii=False)[:200] if isinstance(content, (dict, list)) else str(content)[:200]
            lines.append(
                f"[{r.get('lens_name','')}] Video: {r.get('video_title','')} "
                f"(Channel: {r.get('channel_name','')})\n  {preview}"
            )
        return f"Found {len(results)} insight(s) for '{query}':\n" + "\n".join(lines)
    except Exception as e:
        return f"Insight search error: {e}"


def _tool_list_channels(args, strat, profile_id=0):
    """List tracked YouTube channels."""
    try:
        from processors.youtube import YouTubeProcessor
        yt = YouTubeProcessor(strat.config, db=strat.db)
        channels = yt.list_channels(profile_id)
        if not channels:
            return "No YouTube channels tracked. Add one with the YouTube channel management API."
        lines = []
        for ch in channels:
            lenses = json.loads(ch.get('lenses', '[]')) if isinstance(ch.get('lenses'), str) else ch.get('lenses', [])
            lines.append(
                f"• {ch.get('channel_name', ch.get('channel_id',''))} — "
                f"{ch.get('video_count', 0)} videos ({ch.get('completed_count', 0)} processed), "
                f"Lenses: {', '.join(lenses)}"
            )
        return "Tracked channels:\n" + "\n".join(lines)
    except Exception as e:
        return f"Channel list error: {e}"


def _tool_get_video_summary(args, strat, profile_id=0):
    """Get all insights for a video by title search."""
    title = args.get("video_title", "").strip()
    if not title:
        return "No video title provided."
    try:
        cursor = strat.db.conn.cursor()
        cursor.execute(
            "SELECT id, title, video_id, status FROM youtube_videos "
            "WHERE profile_id = ? AND title LIKE ? ORDER BY published_at DESC LIMIT 1",
            (profile_id, f'%{title}%')
        )
        row = cursor.fetchone()
        if not row:
            return f"No video found matching '{title}'."
        video = dict(row)
        if video['status'] != 'complete':
            return f"Video '{video['title']}' is {video['status']} — not yet processed."

        from processors.youtube import YouTubeProcessor
        yt = YouTubeProcessor(strat.config, db=strat.db)
        insights = yt.get_video_insights(video['id'], profile_id)
        if not insights:
            return f"Video '{video['title']}' has no extracted insights."

        parts = [f"Insights for: {video['title']}"]
        for ins in insights:
            content = ins.get('content', {})
            content_str = json.dumps(content, ensure_ascii=False, indent=2) if isinstance(content, (dict, list)) else str(content)
            parts.append(f"\n--- {ins['lens_name'].upper()} ---\n{content_str[:1500]}")
        return "\n".join(parts)
    except Exception as e:
        return f"Video summary error: {e}"


def _tool_search_narrations(args, strat, profile_id=0):
    """Search narrations from processed videos."""
    query = args.get("query", "").strip()
    if not query:
        return "No search query provided."
    try:
        from processors.youtube import YouTubeProcessor
        yt = YouTubeProcessor(strat.config, db=strat.db)
        results = yt.search_insights(profile_id, query, lens_name='narrations', limit=10)
        if not results:
            return f"No narrations found matching '{query}'."
        lines = []
        for r in results:
            content = r.get('content', [])
            if isinstance(content, list):
                for narr in content:
                    if isinstance(narr, dict) and query.lower() in json.dumps(narr, ensure_ascii=False).lower():
                        verified = narr.get('verification', {}).get('verified', False)
                        status = "✓ Verified" if verified else "⚠ Unverified"
                        lines.append(
                            f"[{status}] {narr.get('narration_text', '')[:200]}\n"
                            f"  Attribution: {narr.get('speaker_attribution', 'N/A')}\n"
                            f"  Source: {narr.get('source_claimed', 'N/A')}\n"
                            f"  Video: {r.get('video_title', '')}"
                        )
            if not lines:
                lines.append(f"Video: {r.get('video_title', '')} — narrations lens result found but no text match")
        return f"Narration search for '{query}':\n" + "\n".join(lines) if lines else f"No specific narrations matching '{query}'."
    except Exception as e:
        return f"Narration search error: {e}"


def _tool_read_url(args, strat):
    """Fetch and extract text content from a URL."""
    url = args.get("url", "").strip()
    if not url or not url.startswith("http"):
        return "Invalid URL. Must start with http:// or https://."
    try:
        from fetchers.news import NewsFetcher
        fetcher = NewsFetcher(strat.config)
        text = fetcher.scrape_article(url)
        if not text:
            return f"Could not extract text from {url}. The page may be blocked, require JavaScript, or not contain readable text."
        # Truncate for context budget
        if len(text) > 4000:
            text = text[:4000] + "\n\n[...truncated — article continues...]"
        return f"Content from {url}:\n\n{text}"
    except Exception as e:
        return f"Error reading URL: {e}"


def _tool_import_canon_world(args, strat, profile_id=0):
    """Import a canon world from a franchise via Fandom wikis."""
    import threading
    import re as _re
    franchise_name = args.get("franchise", "").strip()
    if not franchise_name:
        return "Error: No franchise name provided."

    # Resolve franchise
    serper_fn = None
    try:
        from fetchers.serper_search import get_serper_client
        client = get_serper_client(strat.config)
        if client:
            serper_fn = lambda q, **kw: client.search(q, num_results=kw.get('num_results', 3))
    except Exception:
        pass

    from processors.canon_import import resolve_franchise
    info = resolve_franchise(franchise_name, serper_search_fn=serper_fn)
    if not info:
        return f"Could not find a Fandom wiki for '{franchise_name}'. Try the exact franchise name (e.g., 'Sword Art Online' instead of 'SAO') or create the scenario manually."

    # Create scenario
    scenario_name = args.get("scenario_name", "").strip()
    if not scenario_name:
        scenario_name = _re.sub(r'[^a-z0-9_]', '_', info['full_name'].lower())[:40]

    from processors.scenario_templates import get_scenario_base_path, create_scenario_skeleton
    data_dir = strat.config.get("system", {}).get("data_dir", "data")
    base_path = get_scenario_base_path(data_dir, profile_id)
    scenario_path = create_scenario_skeleton(base_path, scenario_name)

    # Save to DB
    sm = None
    try:
        from processors.scenarios import ScenarioManager
        sm = ScenarioManager(strat.config, db=strat.db)
        sm.create_scenario(profile_id, scenario_name,
                           world_md=f"Canon import: {info['full_name']}")
    except Exception as e:
        logger.warning(f"DB save failed: {e}")

    # Track generation status (same mechanism as regular scenario generation)
    if not hasattr(strat, '_scenario_gen_status'):
        strat._scenario_gen_status = {}
    status_key = f"{profile_id}:{scenario_name}"
    strat._scenario_gen_status[status_key] = {
        "status": "generating",
        "source": "canon_import",
        "franchise": info['full_name'],
        "passes": {}
    }

    scoring_cfg = strat.config.get("scoring", {})
    ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
    model = scoring_cfg.get("inference_model", "qwen3.5:9b")

    def _progress_cb(pass_num, pass_name, status):
        strat._scenario_gen_status[status_key]["passes"][str(pass_num)] = {
            "name": pass_name, "status": status
        }
        if pass_num == 4 and status == "done":
            strat._scenario_gen_status[status_key]["status"] = "complete"
            # Update DB with generated setting
            try:
                import os
                setting_path = os.path.join(scenario_path, 'world', 'setting.md')
                if sm and os.path.exists(setting_path):
                    with open(setting_path) as f:
                        setting = f.read()
                    sm.save_scenario(profile_id, scenario_name, world_md=setting)
            except Exception:
                pass

    def _import_in_background():
        try:
            from processors.canon_import import run_canon_import
            run_canon_import(ollama_host, model, scenario_path, info, progress_callback=_progress_cb)
        except Exception as e:
            logger.error(f"Canon import failed: {e}")
            strat._scenario_gen_status[status_key]["status"] = "failed"

    threading.Thread(target=_import_in_background, daemon=True).start()

    return (f"Importing **{info['full_name']}** from {info['wiki']}.fandom.com into scenario '{scenario_name}'.\n\n"
            f"This will fetch characters, locations, world lore, and power systems from the wiki.\n"
            f"Check the scenario panel for progress. Genre: {info.get('genre', 'RPG')}.\n\n"
            f"The import runs in the background — you can continue chatting while it works.")
