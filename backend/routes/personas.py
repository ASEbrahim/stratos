"""
Persona Registry — Config-driven multi-persona agent architecture.

Each persona defines:
  - system_prompt: instruction set (UNDER 300 words — 9B model constraint)
  - tools: list of tool names this persona can use
  - context_builder: assembles relevant data for this persona
  - greeting: what the agent says when first activated

Adding a new persona = adding a config entry + context builder function.
No core agent logic changes needed.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("STRAT_OS")


# ═══════════════════════════════════════════════════════════
# CONTEXT BUILDERS
# ═══════════════════════════════════════════════════════════

def _build_news_context(strat, output_file: str) -> str:
    """Build news + market + briefing context from profile output."""
    news_context = ""
    output_path = Path(output_file)
    if not output_path.exists():
        return news_context
    try:
        with open(output_path) as f:
            scraped = json.loads(f.read())

        news_items = scraped.get("news", [])
        top = sorted(
            [x for x in news_items if isinstance(x, dict)],
            key=lambda x: x.get("score", 0), reverse=True
        )[:30]
        lines = []
        for it in top:
            try:
                lines.append(
                    f"[{float(it.get('score',0)):.1f}] {it.get('title','')} "
                    f"({it.get('source','')}, {it.get('category',it.get('root',''))}) "
                    f"— {str(it.get('summary',''))[:200]}"
                )
            except Exception:
                continue
        news_context = "\n".join(lines)

        # Market data
        market = scraped.get("market", {})
        mlines = []
        for sym, md in market.items():
            try:
                if not isinstance(md, dict):
                    continue
                name = md.get("name", sym)
                data_dict = md.get("data", {}) if isinstance(md.get("data"), dict) else {}
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
                    mlines.append(f"{name} ({sym}):\n" + "\n".join(parts[:3]))
            except Exception:
                continue
        if mlines:
            ts = scraped.get("timestamps", {}).get("market", "")
            ts_label = f" (as of {ts[:16].replace('T',' ')})" if ts else ""
            news_context += f"\n\nMARKET DATA{ts_label}:\n" + "\n".join(mlines)

        # Briefing
        briefing = scraped.get("briefing", {})
        if isinstance(briefing, dict) and briefing:
            bp = []
            alerts = briefing.get("critical_alerts", [])
            if isinstance(alerts, list) and alerts:
                al = [
                    f"- {a.get('headline',a.get('title',''))} ({a.get('score',0)}): "
                    f"{str(a.get('analysis',''))[:150]}"
                    for a in alerts[:5] if isinstance(a, dict)
                ]
                if al:
                    bp.append("CRITICAL ALERTS:\n" + "\n".join(al))
            picks = briefing.get("high_priority", [])
            if isinstance(picks, list) and picks:
                pl = [f"- {p.get('title','')} ({p.get('score',0)})" for p in picks[:5] if isinstance(p, dict)]
                if pl:
                    bp.append("TOP PICKS:\n" + "\n".join(pl))
            if briefing.get("market_summary"):
                bp.append("MARKET: " + str(briefing["market_summary"]))
            if bp:
                news_context = "\n\n".join(bp) + "\n\n" + news_context
    except Exception as e:
        logger.warning(f"News context error: {e}")
    return news_context


def _build_historical_context(strat, profile_id: int = 0) -> str:
    """Build scan history, category stats, top signals, daily trend."""
    parts = []
    db = strat.db
    try:
        scans = db.get_scan_log(5, profile_id=profile_id)
        if scans:
            lines = [
                f"  {s.get('started_at','')[:16].replace('T',' ')}: "
                f"{s.get('items_scored',0)} items → {s.get('critical',0)} crit, {s.get('high',0)} high"
                if not s.get('error') else f"  {s.get('started_at','')[:16]}: FAILED"
                for s in scans
            ]
            parts.append("RECENT SCANS:\n" + "\n".join(lines))
    except Exception as e:
        logger.debug(f"Context: scan log failed: {e}")

    try:
        stats = db.get_category_stats(days=7, profile_id=profile_id)
        if stats:
            lines = [
                f"  {c.get('category','?')}: {c.get('total',0)} items, "
                f"avg {c.get('avg_score',0)}, {c.get('critical',0)} crit, {c.get('high',0)} high"
                for c in stats[:10]
            ]
            parts.append("CATEGORY PERFORMANCE (7d):\n" + "\n".join(lines))
    except Exception as e:
        logger.debug(f"Context: category stats failed: {e}")

    try:
        top = db.get_top_signals(days=7, min_score=7.5, limit=10, profile_id=profile_id)
        if top:
            lines = [
                f"  [{t.get('score',0):.1f}] {t.get('title','')[:80]} "
                f"({t.get('category','')}, {t.get('fetched_at','')[:10]})"
                for t in top
            ]
            parts.append("TOP SIGNALS (7d):\n" + "\n".join(lines))
    except Exception as e:
        logger.debug(f"Context: top signals failed: {e}")

    try:
        daily = db.get_daily_signal_counts(days=7, profile_id=profile_id)
        if daily:
            lines = [
                f"  {d.get('day','?')}: {d.get('total',0)} total, "
                f"{d.get('critical',0)} crit, {d.get('high',0)} high"
                for d in daily
            ]
            parts.append("DAILY TREND:\n" + "\n".join(lines))
    except Exception as e:
        logger.debug(f"Context: daily counts failed: {e}")

    return "\n\n".join(parts)


def _build_scholarly_context(strat, profile_id: int = 0) -> str:
    """Build scholarly context: tracked channels, recent insights, user context."""
    parts = []
    db = strat.db
    if not db:
        return ""

    try:
        cursor = db.conn.cursor()

        # Channel summary
        cursor.execute(
            """SELECT c.channel_name, COUNT(v.id) as total,
                      SUM(CASE WHEN v.status = 'complete' THEN 1 ELSE 0 END) as done
               FROM youtube_channels c
               LEFT JOIN youtube_videos v ON v.channel_id = c.id AND v.profile_id = c.profile_id
               WHERE c.profile_id = ?
               GROUP BY c.id""",
            (profile_id,)
        )
        channels = [dict(r) for r in cursor.fetchall()]
        if channels:
            lines = [f"  {c['channel_name']}: {c['done']}/{c['total']} videos processed" for c in channels]
            parts.append("TRACKED CHANNELS:\n" + "\n".join(lines))

        # Recent insights (last 5 processed videos)
        cursor.execute(
            """SELECT v.title, v.video_id, c.channel_name, v.processed_at
               FROM youtube_videos v
               JOIN youtube_channels c ON v.channel_id = c.id
               WHERE v.profile_id = ? AND v.status = 'complete'
               ORDER BY v.processed_at DESC LIMIT 5""",
            (profile_id,)
        )
        recent = [dict(r) for r in cursor.fetchall()]
        if recent:
            lines = [f"  {r['title']} ({r['channel_name']}, {r.get('processed_at','')[:10]})" for r in recent]
            parts.append("RECENT PROCESSED VIDEOS:\n" + "\n".join(lines))

        # User-editable persona context
        cursor.execute(
            "SELECT content FROM persona_context WHERE profile_id = ? AND persona_name = 'scholarly' AND context_key = 'system_context'",
            (profile_id,)
        )
        row = cursor.fetchone()
        if row:
            user_ctx = dict(row).get('content', '')
            if user_ctx.strip():
                parts.append(f"USER CONTEXT:\n{user_ctx[:1000]}")

    except Exception as e:
        logger.warning(f"Scholarly context error: {e}")

    return "\n\n".join(parts)


def _build_games_context(strat, profile_id: int = 0) -> str:
    """Build games/roleplay context: world bible + active scenario state.
    Legacy fallback — used only if no file-based scenario exists."""
    parts = []
    db = strat.db
    if not db:
        return ""

    try:
        cursor = db.conn.cursor()

        # User-editable system context (world bible)
        cursor.execute(
            "SELECT content FROM persona_context WHERE profile_id = ? AND persona_name = 'gaming' AND context_key = 'system_context'",
            (profile_id,)
        )
        row = cursor.fetchone()
        if row:
            world_bible = dict(row).get('content', '')
            if world_bible.strip():
                parts.append(f"WORLD BIBLE:\n{world_bible[:3000]}")

        # Active scenario state
        cursor.execute(
            "SELECT content FROM persona_context WHERE profile_id = ? AND persona_name = 'gaming' AND context_key = 'active_scenario'",
            (profile_id,)
        )
        row = cursor.fetchone()
        if row:
            scenario = dict(row).get('content', '')
            if scenario.strip():
                parts.append(f"CURRENT SCENARIO STATE:\n{scenario[:2000]}")

    except Exception as e:
        logger.warning(f"Games context error: {e}")

    if not parts:
        parts.append("No scenario active. Help the user set up a new roleplay scenario.")

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════
# SELECTIVE CONTEXT LOADING (Sprint 7)
# ═══════════════════════════════════════════════════════════

def _get_scenario_path(strat, profile_id: int) -> Optional[str]:
    """Get the file-system path for the active scenario, or None."""
    try:
        data_dir = strat.config.get("system", {}).get("data_dir", "data")
        base = os.path.join(data_dir, "users", str(profile_id), "context", "gaming", "scenarios")
        if not os.path.isdir(base):
            return None

        # Check DB for active scenario name
        cursor = strat.db.conn.cursor()
        cursor.execute(
            "SELECT name FROM scenarios WHERE profile_id = ? AND is_active = 1 LIMIT 1",
            (profile_id,))
        row = cursor.fetchone()
        if not row:
            return None
        scenario_name = row['name']

        scenario_path = os.path.join(base, scenario_name)
        # Auto-migrate old format if needed
        if os.path.isdir(scenario_path) and not os.path.isfile(os.path.join(scenario_path, '_index.json')):
            try:
                from processors.scenario_templates import migrate_old_scenario
                migrate_old_scenario(scenario_path)
            except Exception as e:
                logger.debug(f"Migration failed: {e}")
        # Only use file-based loading if the new structure exists
        if os.path.isfile(os.path.join(scenario_path, '_index.json')):
            return scenario_path
    except Exception as e:
        logger.debug(f"Scenario path lookup failed: {e}")
    return None


def _load_scenario_file(scenario_path: str, relative_path: str) -> Optional[str]:
    """Load a text file from the scenario, return None if missing."""
    try:
        full_path = os.path.join(scenario_path, relative_path)
        with open(full_path, 'r') as f:
            return f.read().strip()
    except (FileNotFoundError, OSError):
        return None


def _load_scenario_json(scenario_path: str, relative_path: str) -> Optional[dict]:
    """Load a JSON file from the scenario, return None if missing."""
    try:
        full_path = os.path.join(scenario_path, relative_path)
        with open(full_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _enforce_context_budget(parts: list, budget_tokens: int) -> str:
    """Enforce token budget. Core parts are never dropped. Conditional parts trimmed."""
    def est_tokens(text):
        return int(len(text.split()) * 1.3)

    core = [(text, label) for text, label in parts if label == 'core']
    conditional = [(text, label) for text, label in parts if label != 'core']

    core_text = '\n\n'.join(t for t, _ in core)
    core_tokens = est_tokens(core_text)
    remaining = budget_tokens - core_tokens

    included = [core_text] if core_text.strip() else []
    for text, label in conditional:
        tokens = est_tokens(text)
        if tokens <= remaining:
            included.append(text)
            remaining -= tokens
        else:
            words = text.split()
            max_words = int(remaining / 1.3)
            if max_words > 20:
                included.append(' '.join(words[:max_words]) + '\n...(truncated)')
            break

    return '\n\n---\n\n'.join(included)


def _pack_gaming_context_selective(strat, profile_id: int, scenario_path: str,
                                   mode: str = 'gm', active_npc: str = '',
                                   user_message: str = '') -> str:
    """Build gaming context by selectively loading only relevant files.

    Instead of loading one massive world.md, this scans the user's message
    for keywords and loads only the files that matter for this interaction.

    Budget: ~4000 tokens for gaming context.
    """
    parts = []

    # ──────────────────────────────────────────────
    # ALWAYS LOADED (core state, ~400 tokens total)
    # ──────────────────────────────────────────────
    setting = _load_scenario_file(scenario_path, 'world/setting.md')
    if setting:
        parts.append(("## World\n" + setting, 'core'))

    current_scene = _load_scenario_file(scenario_path, 'scenes/current.md')
    if current_scene:
        parts.append((current_scene, 'core'))

    player_stats = _load_scenario_file(scenario_path, 'characters/player/stats.md')
    if player_stats:
        parts.append(("## Player Stats\n" + player_stats, 'core'))

    # ──────────────────────────────────────────────
    # MODE-SPECIFIC LOADING
    # ──────────────────────────────────────────────
    if mode == 'gm':
        rules = _load_scenario_file(scenario_path, 'world/rules.md')
        if rules:
            parts.append(("## Rules\n" + rules, 'gm'))

        index = _load_scenario_json(scenario_path, '_index.json')
        current_loc = index.get('current_location', '') if index else ''
        if current_loc:
            loc_file = _load_scenario_file(scenario_path, f'world/locations/{current_loc}.md')
            if loc_file:
                parts.append((loc_file, 'location'))

        active_npcs = index.get('active_npcs_in_scene', []) if index else []
        for npc_id in active_npcs[:5]:
            profile = _load_scenario_file(scenario_path, f'characters/npcs/{npc_id}/profile.md')
            if profile:
                summary = '\n'.join(profile.split('\n')[:5])
                parts.append((summary, 'npc_summary'))

    elif mode == 'immersive':
        if active_npc:
            npc_id = active_npc.strip().lower().replace(' ', '_')
            for npc_file in ['profile.md', 'memory.md', 'knowledge.md', 'dialogue.md']:
                content = _load_scenario_file(scenario_path, f'characters/npcs/{npc_id}/{npc_file}')
                if content:
                    parts.append((content, 'active_npc'))

    # ──────────────────────────────────────────────
    # KEYWORD-TRIGGERED LOADING
    # ──────────────────────────────────────────────
    msg = user_message.lower()

    keyword_file_map = [
        (['inventory', 'backpack', 'items', 'what do i have', 'carrying', 'bag'],
         'characters/player/inventory.md', 'inventory'),
        (['quest', 'mission', 'objective', 'what should i do', 'task', 'quest board'],
         'characters/player/quests.md', 'quests'),
        (['skill', 'ability', 'spell', 'what can i do', 'learn', 'technique'],
         'characters/player/skills.md', 'skills'),
        (['equipment', 'gear', 'weapon', 'armor', 'equip', 'wearing'],
         'characters/player/equipment.md', 'equipment'),
        (['buy', 'sell', 'shop', 'price', 'trade', 'merchant', 'store', 'gold', 'coin', 'money'],
         'world/mechanics/economy.md', 'economy'),
        (['craft', 'forge', 'upgrade', 'smith', 'create', 'build', 'enchant'],
         'world/mechanics/crafting.md', 'crafting'),
        (['fight', 'attack', 'combat', 'battle', 'swing', 'cast', 'defend', 'dodge', 'block', 'hit'],
         'world/mechanics/combat.md', 'combat'),
        (['guild', 'join', 'faction', 'alliance', 'clan', 'group', 'team up'],
         None, 'factions'),
        (['journal', 'notes', 'discoveries', 'what do i know', 'remember'],
         'characters/player/journal.md', 'journal'),
    ]

    for keywords, filepath, label in keyword_file_map:
        if any(kw in msg for kw in keywords):
            if filepath:
                content = _load_scenario_file(scenario_path, filepath)
                if content:
                    parts.append((content, label))
            elif label == 'factions':
                factions_dir = os.path.join(scenario_path, 'world', 'factions')
                if os.path.isdir(factions_dir):
                    for fname in os.listdir(factions_dir):
                        if fname.endswith('.md'):
                            content = _load_scenario_file(scenario_path, f'world/factions/{fname}')
                            if content:
                                parts.append((content, 'faction'))

    # COMBAT TRIGGER: also load player equipment
    if any(kw in msg for kw in ['fight', 'attack', 'combat', 'battle', 'swing']):
        equip = _load_scenario_file(scenario_path, 'characters/player/equipment.md')
        if equip:
            parts.append((equip, 'combat_equip'))

    # NPC NAME DETECTION: if user mentions an NPC, load their profile
    roster = _load_scenario_json(scenario_path, 'characters/_roster.json')
    if roster:
        for npc in roster.get('npcs', []):
            npc_name = npc.get('name', '').lower()
            npc_keywords = [k.lower() for k in npc.get('keywords', [])]

            if npc_name in msg or any(kw in msg for kw in npc_keywords):
                npc_id = npc['id']
                profile = _load_scenario_file(scenario_path, f'characters/npcs/{npc_id}/profile.md')
                if profile:
                    parts.append((profile, f'npc_{npc_id}'))
                if mode == 'immersive':
                    memory = _load_scenario_file(scenario_path, f'characters/npcs/{npc_id}/memory.md')
                    if memory:
                        parts.append((memory, f'npc_{npc_id}_memory'))

    # ──────────────────────────────────────────────
    # BUDGET ENFORCEMENT (4000 tokens)
    # ──────────────────────────────────────────────
    return _enforce_context_budget(parts, 4000)


def _build_market_context(strat, output_file: str) -> str:
    """Build market-focused context: prices, movers, finance news."""
    output_path = Path(output_file)
    if not output_path.exists():
        return ""

    parts = []
    try:
        with open(output_path) as f:
            scraped = json.loads(f.read())

        # Full market data (all timeframes)
        market = scraped.get("market", {})
        mlines = []
        for sym, md in market.items():
            try:
                if not isinstance(md, dict):
                    continue
                name = md.get("name", sym)
                data_dict = md.get("data", {}) if isinstance(md.get("data"), dict) else {}
                sym_parts = []
                for k in ["1m", "5m", "1d_1mo", "1d_1y", "1wk"]:
                    d = data_dict.get(k)
                    if isinstance(d, dict) and "price" in d:
                        p = float(d.get("price", 0))
                        c = float(d.get("change", 0))
                        high, low = d.get("high"), d.get("low")
                        hl = f", H/L: ${float(high):.2f}/${float(low):.2f}" if high and low else ""
                        hist = d.get("history", [])
                        trend = ""
                        if isinstance(hist, list) and len(hist) >= 5:
                            r5 = hist[-5:]
                            if all(isinstance(x, (int, float)) for x in r5):
                                trend = f", trend: {'rising' if r5[-1]>r5[0] else 'falling' if r5[-1]<r5[0] else 'flat'}"
                        sym_parts.append(f"  {k}: ${p:.2f} ({c:+.2f}%){hl}{trend}")
                if sym_parts:
                    mlines.append(f"{name} ({sym}):\n" + "\n".join(sym_parts))
            except Exception:
                continue
        if mlines:
            ts = scraped.get("timestamps", {}).get("market", "")
            ts_label = f" (as of {ts[:16].replace('T',' ')})" if ts else ""
            parts.append(f"MARKET DATA{ts_label}:\n" + "\n".join(mlines))

        # Finance-tagged news items
        news_items = scraped.get("news", [])
        finance_items = [
            x for x in news_items if isinstance(x, dict)
            and x.get('category', '').lower() in ('finance', 'banks', 'market', 'economy')
        ]
        if finance_items:
            top_fin = sorted(finance_items, key=lambda x: x.get("score", 0), reverse=True)[:15]
            lines = [
                f"[{float(it.get('score',0)):.1f}] {it.get('title','')} ({it.get('source','')})"
                for it in top_fin
            ]
            parts.append("FINANCE NEWS:\n" + "\n".join(lines))

        # Briefing market summary
        briefing = scraped.get("briefing", {})
        if isinstance(briefing, dict) and briefing.get("market_summary"):
            parts.append("MARKET BRIEFING: " + str(briefing["market_summary"]))
    except Exception as e:
        logger.warning(f"Market context error: {e}")

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════
# PERSONA PROMPT BUILDERS (extracted to persona_prompts.py)
# ═══════════════════════════════════════════════════════════

from routes.persona_prompts import build_persona_prompt  # noqa: E402


# ═══════════════════════════════════════════════════════════
# PERSONA REGISTRY
# ═══════════════════════════════════════════════════════════

# Tool names available per persona
PERSONA_TOOLS = {
    'intelligence': ['web_search', 'search_feed', 'manage_watchlist', 'manage_categories', 'search_files', 'read_document'],
    'market': ['manage_watchlist', 'search_feed', 'web_search'],
    'scholarly': ['search_insights', 'list_channels', 'get_video_summary', 'search_narrations', 'search_files', 'read_document', 'web_search', 'read_url'],
    'gaming': ['search_files', 'read_document'],
    'anime': [],
    'tcg': [],
}

PERSONA_GREETINGS = {
    'intelligence': "How can I help you today? I have your latest feed data and can search the web or your news history.",
    'market': "Market analyst ready. I can see your watchlist and current prices. What would you like to analyze?",
    'scholarly': "Welcome. I'm ready to discuss history, language, philosophy, or academic topics. What's on your mind?",
    'anime': "Anime mode is coming soon! For now, I can chat about anime and manga.",
    'tcg': "TCG mode is coming soon! For now, I can chat about trading card games.",
    'gaming': "Ready for adventure. Describe your world or load a scenario to begin.",
}


def get_persona_config(persona: str) -> Dict[str, Any]:
    """Get configuration for a persona."""
    return {
        'name': persona,
        'tools': PERSONA_TOOLS.get(persona, []),
        'greeting': PERSONA_GREETINGS.get(persona, "How can I help?"),
    }


# ═══════════════════════════════════════════════════════════
# SMART CONTEXT PACKING (Sprint 5K Phase 4)
# ═══════════════════════════════════════════════════════════

def _estimate_tokens(text: str) -> int:
    """Estimate token count: word count * 1.3."""
    return int(len(text.split()) * 1.3) if text else 0


def _get_profile_context(strat, profile_id: int) -> str:
    """Role, location, user_context from profile config. ~200 tokens."""
    profile = strat.config.get("profile", {})
    role = profile.get("role", "")
    location = profile.get("location", "")
    context = profile.get("context", "")
    if not any([role, location, context]):
        return ""
    lines = ["## Your User Profile"]
    if role:
        lines.append(f"Role: {role}")
    if location:
        lines.append(f"Location: {location}")
    if context:
        lines.append(f"Context: {context}")
    return "\n".join(lines)


def _get_persona_custom_context(strat, profile_id: int, persona: str) -> str:
    """User-edited system_context from persona_context table."""
    try:
        cursor = strat.db.conn.cursor()
        cursor.execute(
            "SELECT content FROM persona_context WHERE profile_id = ? AND persona_name = ? AND context_key = 'system_context'",
            (profile_id, persona)
        )
        row = cursor.fetchone()
        if row:
            content = dict(row).get('content', '')
            if content.strip():
                return f"## Custom Instructions\n{content.strip()}"
    except Exception:
        pass
    return ""


def _get_preference_signals(strat, profile_id: int, persona: str) -> str:
    """Read user_preference_signals for this persona. ~200 tokens."""
    try:
        cursor = strat.db.conn.cursor()
        cursor.execute(
            "SELECT signal_type, signal_key, signal_weight FROM user_preference_signals "
            "WHERE profile_id = ? AND (persona_source = ? OR persona_source = 'global') "
            "ORDER BY signal_weight DESC LIMIT 20",
            (profile_id, persona)
        )
        signals = [dict(r) for r in cursor.fetchall()]
        if not signals:
            return ""
        lines = ["## Your Preferences"]
        for s in signals:
            lines.append(f"- {s['signal_type']}: {s['signal_key']} (importance: {s['signal_weight']:.1f})")
        return "\n".join(lines)
    except Exception:
        return ""


def _get_recent_feed(strat, profile_id: int, limit: int = 10) -> str:
    """Top scored articles from last 24h. ~1000 tokens."""
    try:
        cursor = strat.db.conn.cursor()
        cursor.execute(
            "SELECT title, score, source, category FROM news_items "
            "WHERE profile_id = ? AND fetched_at > datetime('now', '-1 day') "
            "ORDER BY score DESC LIMIT ?",
            (profile_id, limit)
        )
        articles = [dict(r) for r in cursor.fetchall()]
        if not articles:
            return ""
        lines = ["## Recent Feed Intelligence (last 24h)"]
        for a in articles:
            lines.append(f"- [{a['score']:.1f}] {a['title']} ({a['source']}, {a['category']})")
        return "\n".join(lines)
    except Exception:
        return ""


def _get_feedback_summary(strat, profile_id: int) -> str:
    """What the user has saved/rated highly. ~300 tokens."""
    try:
        cursor = strat.db.conn.cursor()
        cursor.execute(
            "SELECT title, action, user_score FROM user_feedback "
            "WHERE profile_id = ? AND action IN ('save', 'rate') AND (user_score IS NULL OR user_score >= 7) "
            "ORDER BY created_at DESC LIMIT 8",
            (profile_id,)
        )
        rows = [dict(r) for r in cursor.fetchall()]
        if not rows:
            return ""
        lines = ["## Your Saved/Highly-Rated Articles"]
        for r in rows:
            score_str = f" (rated {r['user_score']:.0f})" if r.get('user_score') else ""
            lines.append(f"- {r['title']}{score_str}")
        return "\n".join(lines)
    except Exception:
        return ""


def _get_file_summaries(strat, profile_id: int, persona: str) -> str:
    """List of uploaded files with content preview. ~500 tokens."""
    try:
        cursor = strat.db.conn.cursor()
        cursor.execute(
            "SELECT filename, file_type, SUBSTR(content_text, 1, 100) as preview FROM user_files "
            "WHERE profile_id = ? AND (persona = ? OR persona = '') ORDER BY uploaded_at DESC LIMIT 10",
            (profile_id, persona)
        )
        files = [dict(r) for r in cursor.fetchall()]
        if not files:
            return ""
        lines = ["## Your Document Library"]
        for f in files:
            preview = f['preview'].replace('\n', ' ')[:80] if f.get('preview') else ''
            suffix = f": {preview}..." if preview else ""
            lines.append(f"- {f['filename']} ({f['file_type']}){suffix}")
        return "\n".join(lines)
    except Exception:
        return ""


def _get_youtube_knowledge(strat, profile_id: int) -> str:
    """Tracked channels and recent insights. ~1000 tokens."""
    try:
        cursor = strat.db.conn.cursor()
        cursor.execute(
            "SELECT id, channel_name FROM youtube_channels WHERE profile_id = ?",
            (profile_id,)
        )
        channels = [dict(r) for r in cursor.fetchall()]
        if not channels:
            return ""
        lines = ["## YouTube Knowledge Base"]
        for ch in channels:
            lines.append(f"\n### {ch['channel_name']}")
            cursor.execute(
                "SELECT id, title FROM youtube_videos WHERE channel_id = ? AND status = 'complete' "
                "ORDER BY processed_at DESC LIMIT 3",
                (ch['id'],)
            )
            videos = [dict(r) for r in cursor.fetchall()]
            for v in videos:
                lines.append(f"- {v['title']}")
                # Get summary lens if available
                cursor.execute(
                    "SELECT content FROM video_insights WHERE video_id = ? AND lens_name = 'summary' LIMIT 1",
                    (v['id'],)
                )
                summary_row = cursor.fetchone()
                if summary_row:
                    content = dict(summary_row).get('content', '')
                    if content:
                        # Parse JSON content if needed
                        try:
                            parsed = json.loads(content) if isinstance(content, str) and content.startswith('{') else content
                            if isinstance(parsed, dict):
                                content = parsed.get('summary', str(parsed))[:200]
                            else:
                                content = str(content)[:200]
                        except Exception:
                            content = str(content)[:200]
                        lines.append(f"  Summary: {content}...")
        return "\n".join(lines)
    except Exception as e:
        logger.debug(f"YouTube knowledge context error: {e}")
        return ""


def _get_active_scenario(strat, profile_id: int) -> str:
    """Full gaming scenario: state + world + characters. ~2000 tokens."""
    try:
        cursor = strat.db.conn.cursor()
        cursor.execute(
            "SELECT name, description, state_md, world_md, characters_json, genre "
            "FROM scenarios WHERE profile_id = ? AND is_active = 1 LIMIT 1",
            (profile_id,)
        )
        row = cursor.fetchone()
        if not row:
            return ""
        s = dict(row)
        parts = [f"## Active Scenario: {s['name']}"]
        if s.get('genre'):
            parts.append(f"Genre: {s['genre']}")
        if s.get('description'):
            parts.append(f"Description: {s['description']}")
        if s.get('world_md'):
            parts.append(f"\n### World\n{s['world_md'][:1500]}")
        if s.get('characters_json'):
            try:
                chars = json.loads(s['characters_json']) if isinstance(s['characters_json'], str) else s['characters_json']
                if isinstance(chars, list) and chars:
                    parts.append("\n### Characters")
                    for c in chars[:5]:
                        if isinstance(c, dict):
                            parts.append(f"- {c.get('name', '???')}: {c.get('description', '')[:100]}")
                        else:
                            parts.append(f"- {str(c)[:100]}")
            except Exception:
                pass
        if s.get('state_md'):
            parts.append(f"\n### Current State\n{s['state_md'][:1000]}")
        return "\n".join(parts)
    except Exception:
        return ""


def _get_market_articles(strat, profile_id: int, limit: int = 5) -> str:
    """Top finance/market category articles. ~500 tokens."""
    try:
        cursor = strat.db.conn.cursor()
        cursor.execute(
            "SELECT title, score, source FROM news_items "
            "WHERE profile_id = ? AND LOWER(category) IN ('finance', 'markets', 'economy', 'crypto', 'banks') "
            "AND fetched_at > datetime('now', '-2 days') ORDER BY score DESC LIMIT ?",
            (profile_id, limit)
        )
        articles = [dict(r) for r in cursor.fetchall()]
        if not articles:
            return ""
        lines = ["## Recent Market News"]
        for a in articles:
            lines.append(f"- [{a['score']:.1f}] {a['title']} ({a['source']})")
        return "\n".join(lines)
    except Exception:
        return ""


def _pack_context(persona_name: str, strat, output_file: str,
                  profile_id: int, max_tokens: int = 16000,
                  user_message: str = '', rp_mode: str = 'gm',
                  active_npc: str = '') -> str:
    """Build the richest possible context that fits in max_tokens.

    Token estimation: word count * 1.3
    Budget: 16,000 tokens. Qwen3.5:9b has 32K context.
    Tool definitions ~2K + conversation history ~3K + response ~2K = 7K used.
    Leaves 25K, so 16K for context is conservative and safe.
    """
    sections = []

    # Gaming persona: skip real-world profile, use selective file loading
    if persona_name == 'gaming':
        scenario_path = _get_scenario_path(strat, profile_id)
        if scenario_path:
            # New file-based selective loading
            gaming_ctx = _pack_gaming_context_selective(
                strat, profile_id, scenario_path,
                mode=rp_mode, active_npc=active_npc,
                user_message=user_message
            )
            sections.append(('gaming_selective', gaming_ctx))
            sections.append(('files', _get_file_summaries(strat, profile_id, 'gaming')))
        else:
            # Fallback to DB-based context for old scenarios
            sections.append(('profile', _get_profile_context(strat, profile_id)))
            sections.append(('custom_context', _get_persona_custom_context(strat, profile_id, persona_name)))
            games = _build_games_context(strat, profile_id)
            sections.append(('games_data', games))
            sections.append(('scenario', _get_active_scenario(strat, profile_id)))
            sections.append(('files', _get_file_summaries(strat, profile_id, 'gaming')))
    else:
        # Universal: always include (highest priority) — non-gaming personas
        sections.append(('profile', _get_profile_context(strat, profile_id)))
        sections.append(('custom_context', _get_persona_custom_context(strat, profile_id, persona_name)))
        sections.append(('preferences', _get_preference_signals(strat, profile_id, persona_name)))

    # Persona-specific data (fills remaining budget)
    if persona_name == 'intelligence':
        # Existing rich context builders
        news = _build_news_context(strat, output_file)
        hist = _build_historical_context(strat, profile_id)
        sections.append(('feed_data', f"CURRENT FEED DATA:\n{news}" if news else ''))
        sections.append(('historical', f"HISTORICAL DATA:\n{hist}" if hist else ''))
        sections.append(('recent_feed', _get_recent_feed(strat, profile_id, limit=10)))
        sections.append(('feedback', _get_feedback_summary(strat, profile_id)))

    elif persona_name == 'scholarly':
        scholarly = _build_scholarly_context(strat, profile_id)
        sections.append(('scholarly_data', scholarly))
        sections.append(('files', _get_file_summaries(strat, profile_id, 'scholarly')))
        sections.append(('youtube', _get_youtube_knowledge(strat, profile_id)))

    elif persona_name == 'market':
        market = _build_market_context(strat, output_file)
        sections.append(('market_data', market))
        sections.append(('market_news', _get_market_articles(strat, profile_id, limit=5)))

    # Pack into prompt, respecting token budget
    prompt_parts = []
    token_count = 0
    for label, content in sections:
        if not content:
            continue
        section_tokens = _estimate_tokens(content)
        if token_count + section_tokens > max_tokens:
            # Try to fit a truncated version
            remaining_tokens = max_tokens - token_count
            if remaining_tokens > 100:
                # Rough truncation: ~0.77 words per token, ~5 chars per word
                max_chars = int(remaining_tokens / 1.3 * 5)
                truncated = content[:max_chars].rsplit('\n', 1)[0]
                if truncated:
                    prompt_parts.append(truncated + "\n[...truncated...]")
            break
        prompt_parts.append(content)
        token_count += section_tokens

    return '\n\n'.join(prompt_parts)


def _load_state_md(strat, profile_id: int, persona_name: str) -> str:
    """Load state.md for a persona if it exists."""
    try:
        from processors.context_compression import ContextCompressor
        cc = ContextCompressor(strat.config, db=strat.db)
        state = cc.get_state(profile_id, persona_name)
        if state.strip():
            return f"PERSONA STATE:\n{state[:2000]}"
    except Exception:
        pass
    return ""


def build_persona_context(persona: str, strat, output_file: str,
                          profile_id: int = 0, user_message: str = '',
                          rp_mode: str = 'gm', active_npc: str = '') -> str:
    """Build context data for the given persona using smart context packing."""
    state = _load_state_md(strat, profile_id, persona)
    ctx = _pack_context(persona, strat, output_file, profile_id,
                        user_message=user_message, rp_mode=rp_mode,
                        active_npc=active_npc)

    if state:
        ctx = f"{state}\n\n{ctx}" if ctx else state
    return ctx


def list_personas() -> List[Dict[str, str]]:
    """List all available personas with their greetings."""
    return [
        {'name': name, 'greeting': PERSONA_GREETINGS.get(name, '')}
        for name in PERSONA_TOOLS.keys()
    ]
