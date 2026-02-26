"""
STRAT_OS - Briefing Generator
Generates the agent's intelligence briefing using LLM.
"""

import json
import logging
import re
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional

from routes.helpers import strip_think_blocks, strip_reasoning_preamble, extract_json

logger = logging.getLogger(__name__)


class BriefingGenerator:
    """
    Generates personalized intelligence briefings.

    The briefing is the "voice" of the agent - formal, professional,
    and focused on actionable intelligence.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize briefing generator with dynamic prompts from config.

        Args:
            config: Full config.yaml containing profile, news, and scoring sections
        """
        # Extract scoring config (support both full config and scoring-only)
        scoring_config = config.get("scoring", config) if "scoring" in config else config

        self.model = scoring_config.get("wizard_model") or scoring_config.get("inference_model", "qwen3:14b")
        self.host = scoring_config.get("ollama_host", "http://localhost:11434")
        self._available = None

        # Build dynamic system prompt from config
        self.system_prompt = self._build_system_prompt(config)

    def _build_system_prompt(self, config: Dict[str, Any]) -> str:
        """Build the system prompt dynamically from config settings."""
        # Extract profile
        profile = config.get("profile", {})
        role = profile.get("role", "Computer Engineering (CPEG) Student")
        location = profile.get("location", "Kuwait")
        nationality = profile.get("nationality", "Kuwaiti")
        graduation = profile.get("graduation", "May 2027")
        experience = profile.get("experience_level", "Zero")

        # Extract goals
        goals = profile.get("goals", {})
        primary_goal = goals.get("primary", "Secure a technical position at a top-tier employer")
        financial_goal = goals.get("financial", "Maximize student benefits")

        # Extract target companies from dynamic categories (preferred), profile tiers, or career keywords (fallback)
        dynamic_cats = config.get("dynamic_categories", [])
        if dynamic_cats:
            # Build company/keyword lists from dynamic categories
            all_items = []
            for cat in dynamic_cats:
                if cat.get('enabled') is not False:
                    all_items.extend(cat.get('items', []))
            companies_str = ", ".join(all_items[:15]) if all_items else "target companies"
            banks_str = "local banks"  # Dynamic mode doesn't separate banks
            tech_str = ", ".join(all_items[:10]) if all_items else "emerging technologies"
            
            # Build category summary for the prompt
            cat_summary = "; ".join([
                f"{c.get('label', 'Unknown')}: {', '.join(c.get('items', [])[:5])}" 
                for c in dynamic_cats if c.get('enabled') is not False
            ])
        else:
            cat_summary = ""
            targets = profile.get("targets", {})
            if targets:
                tier1 = targets.get("tier_1_k_sector", {})
                tier1_companies = tier1.get("companies", []) if isinstance(tier1, dict) else tier1
                tier2 = targets.get("tier_2_international", {})
                tier2_companies = tier2.get("companies", []) if isinstance(tier2, dict) else tier2
                companies_str = ", ".join(tier1_companies + tier2_companies)
            else:
                news_config = config.get("news", {})
                career_keywords = news_config.get("career", {}).get("keywords", [])
                companies_str = ", ".join(career_keywords) if career_keywords else "target companies"

            # Extract banks from finance keywords
            news_config = config.get("news", {})
            bank_keywords = news_config.get("finance", {}).get("keywords", [])
            banks_str = ", ".join(bank_keywords) if bank_keywords else "local banks"

            # Extract tech keywords
            tech_keywords = news_config.get("tech_trends", {}).get("keywords", [])
            tech_str = ", ".join(tech_keywords[:5]) if tech_keywords else "emerging technologies"

        # Extract context (the user's primary self-description)
        context = profile.get("context", "")
        
        # Auto-detect student vs professional from context
        experienced_signals = ['head', 'manager', 'director', 'lead', 'senior', 'principal',
                               'supervisor', 'chief', 'specialist', 'analyst', 'consultant',
                               'professor', 'researcher', 'engineer at', 'professional']
        context_lower = context.lower()
        is_experienced = any(sig in context_lower for sig in experienced_signals)
        
        # Build user identity line from context (overrides legacy fields if experienced)
        if is_experienced and context:
            user_identity = context
            user_level_line = f"- Profile: {context}"
        else:
            user_identity = f"{role} in {location}"
            user_level_line = f"- Graduating: {graduation}\n- Experience: {experience}"

        return f"""You are STRAT_OS, a strategic intelligence agent for a {role} in {location}.

CRITICAL: You MUST respond ONLY in English. Never use Chinese, Arabic, or any other language.

Your communication style is:
- Formal and professional
- Direct and actionable
- Concise but thorough
- Never uses emojis or casual language

Your role is to analyze intelligence and present it in a briefing format.
Focus on what the user should DO with the information, not just what happened.
ONLY include items that are DIRECTLY relevant to the user's context below.

USER PROFILE:
- Role: {role} ({nationality} national)
{user_level_line}
{f'- Context: {context}' if context else ''}
- Primary Goal: {primary_goal}
- Financial Goal: {financial_goal}
{f'- Tracking Categories: {cat_summary}' if cat_summary else f'- Target Companies: {companies_str}'}
{f'- Banks of Interest: {banks_str}' if not cat_summary else ''}
- Tech Interests: {tech_str}

RELEVANCE RULE: Every item in the briefing must relate to: {context if context else user_identity}.
Do NOT include items just because they scored high — they must be relevant to the user's stated focus."""
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if self._available is not None:
            return self._available
        
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            self._available = response.status_code == 200
        except:
            self._available = False
        
        return self._available
    
    def _call_ollama(self, prompt: str, max_tokens: int = 500,
                     strip_reasoning: bool = True) -> Optional[str]:
        """Make a request to Ollama.

        Args:
            strip_reasoning: If True, strip reasoning preamble (for natural language).
                If False, skip reasoning stripping (for JSON responses).
        """
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    # Don't set think:false — Qwen3 leaks reasoning into content.
                    "options": {
                        "temperature": 0.7,
                        "num_predict": max_tokens + 1500  # Extra budget for thinking tokens
                    }
                },
                timeout=150
            )

            if response.status_code == 200:
                text = response.json().get("message", {}).get("content", "")
                if text:
                    text = strip_think_blocks(text)
                    if strip_reasoning:
                        text = strip_reasoning_preamble(text)
                # Safety: detect if model switched to Chinese (Qwen2.5 bilingual issue)
                if text:
                    cjk_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\u2e80-\u2eff\u3000-\u303f\uff00-\uffef]', text))
                    if cjk_chars > len(text) * 0.15:
                        text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\u2e80-\u2eff\u3000-\u303f\uff00-\uffef]', '', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        logger.warning(f"Stripped CJK characters from Ollama response ({cjk_chars} chars)")
                return text
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
        
        return None
    
    def generate_market_summary(self, market_data: Dict[str, Any], alerts: List[Dict]) -> str:
        """
        Generate a market summary paragraph.
        
        Args:
            market_data: Market data dict
            alerts: List of market alerts (>5% moves)
            
        Returns:
            Summary paragraph
        """
        if not market_data:
            return "Market data unavailable."
        
        # Build market context
        summaries = []
        for symbol, data in market_data.items():
            if "error" in data:
                continue
            name = data.get("name", symbol)
            interval_data = data.get("data", {}).get("1m", {})
            if interval_data:
                price = interval_data.get("price", 0)
                change = interval_data.get("change", 0)
                direction = "up" if change > 0 else "down" if change < 0 else "flat"
                summaries.append(f"{name}: ${price:.2f} ({change:+.2f}%)")
        
        if not self.is_available():
            # Fallback: simple summary
            if alerts:
                alert_text = "; ".join([
                    f"{a['name']} moved {a['change']:+.1f}%" for a in alerts
                ])
                return f"Significant market movement detected: {alert_text}. Full watchlist: {', '.join(summaries[:4])}."
            else:
                return f"Markets are relatively stable. Watchlist snapshot: {', '.join(summaries[:4])}."
        
        # Use LLM for better summary
        prompt = f"""Summarize this market data in 2-3 sentences in English. Be specific about significant moves.

Market Data:
{chr(10).join(summaries)}

Alerts (>5% moves): {json.dumps(alerts) if alerts else 'None'}

Write a brief, professional market summary in English only:"""
        
        response = self._call_ollama(prompt, max_tokens=200)
        return response if response else f"Watchlist: {', '.join(summaries[:4])}."
    
    def generate_critical_alert(self, item: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a critical alert block for a high-priority news item.
        
        Args:
            item: News item dict
            
        Returns:
            Dict with 'headline' and 'analysis' keys
        """
        if not self.is_available():
            return {
                "headline": item.get("title", "Critical Alert"),
                "analysis": item.get("score_reason", "This item requires your attention."),
                "action": "Review immediately and take appropriate action."
            }
        
        prompt = f"""Analyze this critical intelligence item and provide:
1. A concise headline (1 line)
2. Why it matters to the user (2-3 sentences)
3. Recommended action (1 sentence)

News Item:
Title: {item.get('title', '')}
Summary: {item.get('summary', '')[:400]}
Source: {item.get('source', '')}
Score Reason: {item.get('score_reason', '')}

Respond in this JSON format:
{{"headline": "...", "analysis": "...", "action": "..."}}"""
        
        response = self._call_ollama(prompt, max_tokens=300, strip_reasoning=False)

        if response:
            try:
                return json.loads(extract_json(response))
            except (json.JSONDecodeError, ValueError):
                pass
        
        return {
            "headline": item.get("title", "Critical Alert"),
            "analysis": item.get("score_reason", "This item requires your attention."),
            "action": "Review the full article for details."
        }
    
    def generate_discovery_note(self, discoveries: List[Dict]) -> Optional[str]:
        """
        Generate a note about newly discovered entities.
        
        Args:
            discoveries: List of discovered entity dicts
            
        Returns:
            Discovery note string or None
        """
        if not discoveries:
            return None
        
        entities = [d["name"] for d in discoveries[:3]]
        
        if not self.is_available():
            return f"I detected increased activity around: {', '.join(entities)}. These have been added to my monitoring list."
        
        prompt = f"""I discovered these entities appearing more frequently than usual:
{json.dumps(discoveries[:3], indent=2)}

Write a brief note (2-3 sentences) explaining:
1. What I noticed
2. Why it might be relevant
3. That I've added them to monitoring

Keep it formal and professional:"""
        
        response = self._call_ollama(prompt, max_tokens=200)
        return response if response else f"Detected rising activity: {', '.join(entities)}. Added to monitoring."
    
    def _generate_combined(self, market_data, market_alerts, critical_items, discoveries):
        """Generate market summary, critical alerts, and discovery note in a single LLM call."""
        # Build market context
        summaries = []
        for symbol, data in (market_data or {}).items():
            if "error" in data:
                continue
            name = data.get("name", symbol)
            interval_data = data.get("data", {}).get("1m", {})
            if interval_data:
                price = interval_data.get("price", 0)
                change = interval_data.get("change", 0)
                summaries.append(f"{name}: ${price:.2f} ({change:+.2f}%)")

        # Build combined prompt
        sections = []

        # Market section
        sections.append("## MARKET DATA")
        if summaries:
            sections.append("\n".join(summaries))
        else:
            sections.append("No market data available.")
        if market_alerts:
            alert_lines = [f"- {a['name']} moved {a['change']:+.1f}%" for a in market_alerts]
            sections.append("Alerts (>5% moves):\n" + "\n".join(alert_lines))

        # Critical items section
        top_critical = critical_items[:5]
        if top_critical:
            sections.append(f"\n## CRITICAL INTELLIGENCE ITEMS ({len(top_critical)} items)")
            for i, item in enumerate(top_critical, 1):
                sections.append(f"\nItem {i}:")
                sections.append(f"Title: {item.get('title', '')}")
                sections.append(f"Summary: {item.get('summary', '')[:300]}")
                sections.append(f"Score Reason: {item.get('score_reason', '')}")

        # Discoveries section
        if discoveries:
            entities = [d["name"] for d in discoveries[:3]]
            sections.append(f"\n## DISCOVERIES\nEntities with rising activity: {', '.join(entities)}")

        prompt = "\n".join(sections) + """

Generate a complete intelligence briefing as JSON. Respond ONLY with valid JSON, no other text.

{
  "market_summary": "2-3 sentence professional market summary",
  "critical_alerts": [
    {"headline": "concise headline", "analysis": "2-3 sentences on why it matters", "action": "1 sentence recommendation"}
  ],
  "discovery_note": "2-3 sentence note about discovered entities, or null if none"
}

Rules:
- market_summary: Be specific about significant moves. If no data, say "Market data unavailable."
- critical_alerts: One object per critical item above, in the same order. If no critical items, use empty array [].
- discovery_note: null if no discoveries section above.
- Write ONLY in English. Be formal, professional, and actionable."""

        # Token budget: ~300 per alert + 200 market + 200 discovery + JSON overhead
        max_output = 200 + len(top_critical) * 300 + 200 + 100
        response = self._call_ollama(prompt, max_tokens=max_output, strip_reasoning=False)

        if response:
            try:
                parsed = json.loads(extract_json(response))
                # Validate structure
                if isinstance(parsed, dict) and "market_summary" in parsed:
                    return parsed
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Combined briefing JSON parse failed: {e}")

        return None

    def generate_briefing(self,
                         market_data: Dict[str, Any],
                         market_alerts: List[Dict],
                         news_items: List[Dict],
                         discoveries: List[Dict],
                         previous_briefing: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a complete intelligence briefing.

        Uses a single combined LLM call for all sections (market summary,
        critical alerts, discovery note) instead of 7 separate calls.
        Falls back to individual calls if the combined approach fails.
        """
        logger.info("Generating intelligence briefing...")

        now = datetime.now()

        # Categorize news by score
        critical_items = [i for i in news_items if i.get("score", 0) >= 9.0]
        high_items = [i for i in news_items if 7.0 <= i.get("score", 0) < 9.0]
        medium_items = [i for i in news_items if 5.0 <= i.get("score", 0) < 7.0]
        noise_count = len([i for i in news_items if i.get("score", 0) < 5.0])

        # Try combined single-call approach
        market_summary = None
        critical_alerts = []
        discovery_note = None

        if self.is_available():
            import time as _time
            _t0 = _time.time()
            combined = self._generate_combined(market_data, market_alerts, critical_items, discoveries)
            _elapsed = _time.time() - _t0

            if combined:
                logger.info(f"Combined briefing generated in {_elapsed:.1f}s (1 LLM call)")
                market_summary = combined.get("market_summary")
                discovery_note = combined.get("discovery_note")

                # Map combined alerts back to items
                raw_alerts = combined.get("critical_alerts", [])
                for i, item in enumerate(critical_items[:5]):
                    if i < len(raw_alerts):
                        alert = raw_alerts[i]
                        if isinstance(alert, dict):
                            alert["item_id"] = item.get("id")
                            alert["url"] = item.get("url")
                            alert["score"] = item.get("score")
                            # Ensure required keys exist
                            alert.setdefault("headline", item.get("title", "Critical Alert"))
                            alert.setdefault("analysis", item.get("score_reason", "Requires attention."))
                            alert.setdefault("action", "Review the full article for details.")
                            critical_alerts.append(alert)
                    else:
                        # Fewer alerts returned than items — use fallback for this item
                        critical_alerts.append({
                            "headline": item.get("title", "Critical Alert"),
                            "analysis": item.get("score_reason", "Requires attention."),
                            "action": "Review the full article for details.",
                            "item_id": item.get("id"),
                            "url": item.get("url"),
                            "score": item.get("score")
                        })
            else:
                logger.warning(f"Combined briefing failed after {_elapsed:.1f}s, falling back to individual calls")

        # Fallback: individual calls if combined didn't produce results
        if market_summary is None:
            market_summary = self.generate_market_summary(market_data, market_alerts)

        if not critical_alerts:
            for item in critical_items[:5]:
                alert = self.generate_critical_alert(item)
                alert["item_id"] = item.get("id")
                alert["url"] = item.get("url")
                alert["score"] = item.get("score")
                critical_alerts.append(alert)

        if discovery_note is None and discoveries:
            discovery_note = self.generate_discovery_note(discoveries)

        # Build briefing
        briefing = {
            "generated_at": now.isoformat(),
            "greeting": self._get_greeting(now),

            "critical_alerts": critical_alerts,
            "critical_count": len(critical_items),

            "market_summary": market_summary,
            "market_alerts": market_alerts,

            "high_priority": [
                {
                    "id": i.get("id"),
                    "title": i.get("title"),
                    "source": i.get("source"),
                    "score": i.get("score"),
                    "reason": i.get("score_reason"),
                    "url": i.get("url")
                }
                for i in high_items[:5]
            ],
            "high_count": len(high_items),

            "medium_priority": [
                {
                    "id": i.get("id"),
                    "title": i.get("title"),
                    "source": i.get("source"),
                    "score": i.get("score"),
                    "url": i.get("url")
                }
                for i in medium_items[:5]
            ],
            "medium_count": len(medium_items),

            "noise_count": noise_count,

            "discoveries": discoveries[:3] if discoveries else [],
            "discovery_note": discovery_note,

            "summary_stats": {
                "total_items_analyzed": len(news_items),
                "critical": len(critical_items),
                "high": len(high_items),
                "medium": len(medium_items),
                "filtered_noise": noise_count
            }
        }

        logger.info(f"Briefing generated: {len(critical_alerts)} critical, {len(high_items)} high priority")
        return briefing
    
    def _get_greeting(self, dt: datetime) -> str:
        """Get time-appropriate greeting."""
        hour = dt.hour
        if hour < 12:
            return "Good morning."
        elif hour < 17:
            return "Good afternoon."
        else:
            return "Good evening."


def generate_briefing(config: Dict[str, Any],
                     market_data: Dict[str, Any],
                     market_alerts: List[Dict],
                     news_items: List[Dict],
                     discoveries: List[Dict]) -> Dict[str, Any]:
    """
    Convenience function to generate a briefing.
    
    Args:
        config: Full configuration dict
        market_data: Market data
        market_alerts: Market alerts
        news_items: Scored news items
        discoveries: Discovered entities
        
    Returns:
        Briefing dict
    """
    scoring_config = config.get("scoring", {})
    generator = BriefingGenerator(config)  # Pass full config, not just scoring
    return generator.generate_briefing(
        market_data=market_data,
        market_alerts=market_alerts,
        news_items=news_items,
        discoveries=discoveries
    )
