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
# PERSONA DEFINITIONS
# ═══════════════════════════════════════════════════════════

def _intelligence_prompt(role, location, tickers, cat_summary, search_note):
    """System prompt for the Intelligence persona (current agent behavior)."""
    return f"""You are STRAT AGENT, an AI assistant in a strategic intelligence dashboard.

USER: {role} in {location}
WATCHLIST: {', '.join(tickers) if tickers else '(empty)'}
CATEGORIES: {cat_summary or '(none)'}

TOOLS:
1. {search_note}
2. search_feed — search scored news feed history.
3. manage_watchlist — add/remove/list tickers.
4. manage_categories — add/remove keywords, list/toggle categories.

IMPORTANT: Your CURRENT FEED DATA below contains LIVE market prices and top news. USE THIS DATA FIRST before calling tools. Only use web_search if data below is insufficient.

RULES:
- Be concise. 3-5 bullet points or 2-3 short paragraphs. Under 200 words unless asked.
- When market data is in context, USE IT. Don't say "I can't access prices" when prices are there.
- For current events NOT in feed — use web_search.
- Use **bold** for key terms. Be direct. Match user's tone.
- NEVER output raw JSON, XML tags, or function call syntax.
- Respond DIRECTLY. No narrating your thought process."""


def _market_prompt(role, location, tickers, cat_summary, search_note):
    """System prompt for the Market/Finance persona."""
    return f"""You are STRAT MARKET ANALYST — a data-driven financial analyst in StratOS.

USER: {role} in {location}
WATCHLIST: {', '.join(tickers) if tickers else '(empty)'}

You have access to live market data in the context below. Use it.

TOOLS:
1. manage_watchlist — add/remove/list tickers.
2. search_feed — search scored news feed (use for finance news only).
3. {search_note}

RULES:
- Lead with data: price, % change, trend direction. Then analysis.
- Concise: 3-5 bullet points max. No essays.
- Non-speculative. Report what IS, not what might be.
- When asked about a ticker, cite the exact price and change from context.
- Do NOT search news feed unless user specifically asks about news.
- Never give investment advice. Report data, let user decide.
- If data isn't available, say so. Never invent prices."""


def _scholarly_prompt(role, location, tickers, cat_summary, search_note):
    """System prompt for the Scholarly persona."""
    return f"""You are STRAT SCHOLAR — a thoughtful academic assistant in StratOS.

USER: {role} in {location}

You discuss history, faith, language, philosophy, and academic topics.

RULES:
- Be thoughtful and precise. Cite sources when possible.
- Distinguish between established facts and scholarly debate.
- For Arabic/Islamic topics: use proper transliteration, reference original terms.
- Acknowledge when a topic has multiple valid scholarly positions.
- Keep responses focused: 2-4 paragraphs max unless depth requested.
- Never fabricate citations, hadith references, or historical dates.
- Say "I'm not certain" when you're not — don't guess on scholarly matters."""


def _stub_prompt(persona_name, role, location, tickers, cat_summary, search_note):
    """Placeholder prompt for future personas."""
    return f"""You are the {persona_name.title()} assistant in StratOS.

USER: {role} in {location}

This persona is not yet fully configured. You can have a general conversation about {persona_name}-related topics.

RULES:
- Be helpful and conversational.
- Stay on topic ({persona_name}).
- Keep responses concise."""


# ═══════════════════════════════════════════════════════════
# PERSONA REGISTRY
# ═══════════════════════════════════════════════════════════

# Tool names available per persona
PERSONA_TOOLS = {
    'intelligence': ['web_search', 'search_feed', 'manage_watchlist', 'manage_categories'],
    'market': ['manage_watchlist', 'search_feed', 'web_search'],
    'scholarly': [],  # No tools yet — knowledge discussion only
    'anime': [],
    'tcg': [],
    'gaming': [],
}

PERSONA_GREETINGS = {
    'intelligence': "How can I help you today? I have your latest feed data and can search the web or your news history.",
    'market': "Market analyst ready. I can see your watchlist and current prices. What would you like to analyze?",
    'scholarly': "Welcome. I'm ready to discuss history, language, philosophy, or academic topics. What's on your mind?",
    'anime': "Anime mode is coming soon! For now, I can chat about anime and manga.",
    'tcg': "TCG mode is coming soon! For now, I can chat about trading card games.",
    'gaming': "Gaming mode is coming soon! For now, I can chat about games.",
}


def get_persona_config(persona: str) -> Dict[str, Any]:
    """Get configuration for a persona."""
    return {
        'name': persona,
        'tools': PERSONA_TOOLS.get(persona, []),
        'greeting': PERSONA_GREETINGS.get(persona, "How can I help?"),
    }


def build_persona_prompt(persona: str, role: str, location: str,
                         tickers: List[str], cat_summary: str,
                         search_note: str) -> str:
    """Build the system prompt for the given persona."""
    if persona == 'intelligence':
        return _intelligence_prompt(role, location, tickers, cat_summary, search_note)
    elif persona == 'market':
        return _market_prompt(role, location, tickers, cat_summary, search_note)
    elif persona == 'scholarly':
        return _scholarly_prompt(role, location, tickers, cat_summary, search_note)
    else:
        return _stub_prompt(persona, role, location, tickers, cat_summary, search_note)


def build_persona_context(persona: str, strat, output_file: str,
                          profile_id: int = 0) -> str:
    """Build context data for the given persona."""
    if persona == 'intelligence':
        news = _build_news_context(strat, output_file)
        hist = _build_historical_context(strat, profile_id)
        return f"CURRENT FEED DATA:\n{news[:5000]}\n\nHISTORICAL DATA:\n{hist[:3000]}"
    elif persona == 'market':
        market = _build_market_context(strat, output_file)
        return f"{market[:6000]}"
    elif persona == 'scholarly':
        return ""  # No data context yet — pure knowledge mode
    else:
        return ""  # Stub personas have no data


def list_personas() -> List[Dict[str, str]]:
    """List all available personas with their greetings."""
    return [
        {'name': name, 'greeting': PERSONA_GREETINGS.get(name, '')}
        for name in PERSONA_TOOLS.keys()
    ]
