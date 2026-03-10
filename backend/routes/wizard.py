"""
Wizard routes — /api/wizard-preselect, /api/wizard-tab-suggest, /api/wizard-rv-items
Lightweight LLM calls for wizard step guidance.
"""

import json
import re
import logging
import requests as req
from concurrent.futures import ThreadPoolExecutor, as_completed

from routes.helpers import (json_response, error_response, read_json_body,
                           strip_think_blocks, extract_json)

logger = logging.getLogger("STRAT_OS")


PRESELECT_SYSTEM = """You are a configuration assistant that personalizes a news dashboard.
The user will give you their EXACT professional role and location. Based on THAT SPECIFIC role and location, pick which categories and sub-categories are most relevant.
CRITICAL: Respond with ONLY valid JSON — no markdown, no explanation, no backticks, no reasoning.
Schema: {"selected_categories": ["id1","id2",...], "selected_subs": {"id1": ["sub1","sub2"], "id2": ["sub3"]}}
Rules:
- THINK about what this specific role does daily — a doctor has different needs than a software engineer
- Select 2-4 categories that a person with this EXACT role + location would check daily
- For each selected category, pick the most relevant sub-categories (1-3 each)
- The user's ROLE is the primary signal — match categories to their professional domain
- The user's LOCATION affects which sub-categories matter (e.g., local markets, local companies)
- Career & Jobs is relevant for MOST professionals — it covers job market, hiring trends, and professional growth. Only skip for C-suite executives or retirees
- Markets & Investing only if the role involves finance or the user mentioned investing
- Deals & Offers is relevant for students and budget-conscious professionals
- Interests & Trends (id: interests) has no subs — just include the id if relevant
- Do NOT just pick the same defaults for everyone — personalize based on role"""


TAB_SUGGEST_SYSTEM = """You are a keyword suggestion engine. Given a user's role, location, career stage, and category context, suggest 5-8 specific keywords or entities to track.
CRITICAL: Respond with ONLY a JSON array of strings — no markdown, no explanation, no backticks, no reasoning.
Example: ["keyword1", "keyword2", "keyword3"]
Rules:
- Suggest entities relevant to the user's EXACT role + category + career stage
- CAREER STAGE is the MOST IMPORTANT signal after role. Read the "User's current selections" line carefully:
  * Student → internship programs, university career fairs, entry-level bootcamps, student competitions, beginner certifications, campus recruiters
  * Fresh Graduate → graduate programs, associate roles, junior positions, early-career accelerators, mentorship programs
  * Mid-career → professional certifications, industry conferences, specialist roles, upskilling platforms
  * Senior → C-suite networking, executive leadership summits, board advisory, strategic consulting firms, industry thought leadership
- OPPORTUNITY TYPE further refines suggestions:
  * Internships → intern programs at specific companies, summer internship deadlines, internship aggregators
  * Full-time Jobs → hiring companies, professional certifications, recruiter platforms, salary benchmarks
  * Freelancing → freelance platforms, portfolio tools, contract marketplaces
- Results for "Student looking for Internships" must be COMPLETELY DIFFERENT from "Senior looking for Full-time Jobs" — zero overlap
- Be specific: company names, certification names, technology names, platform names
- Avoid generic terms like "news", "updates", "trends", "latest"
- For location-specific roles, include relevant local entities (companies, institutions, programs)"""


def _call_ollama(host, model, system, prompt, max_tokens=500, temperature=0.2,
                  timeout=60):
    """Make an Ollama chat call with thinking disabled. Returns raw text or None.

    These endpoints only need structured JSON — no reasoning required.
    ``think: false`` skips internal reasoning so all ``num_predict`` tokens
    go directly to the JSON output, making responses 3-5x faster.
    """
    try:
        num_ctx = max(4096, max_tokens + 2048)
        resp = req.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": num_ctx,
                }
            },
            timeout=timeout
        )
        if resp.status_code != 200:
            logger.warning(f"Wizard: Ollama returned {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        raw = data.get("message", {}).get("content", "")
        raw = strip_think_blocks(raw)  # safety net
        logger.info(f"Wizard: model={model}, tokens={max_tokens}, raw_len={len(raw)}")
        return raw
    except (req.exceptions.ConnectionError, req.exceptions.ReadTimeout):
        logger.warning("Wizard: Ollama not reachable or timed out")
        return None
    except Exception as e:
        logger.warning(f"Wizard: Ollama call failed: {e}")
        return None


def handle_wizard_preselect(handler, strat):
    """POST /api/wizard-preselect — AI picks relevant categories for Step 1."""
    try:
        data = read_json_body(handler)
        role = data.get("role", "").strip()
        location = data.get("location", "").strip()
        available = data.get("available_categories", [])

        if not role:
            error_response(handler, "Role is required", 400)
            return

        scorer = strat.scorer
        host = scorer.host
        model = strat.config.get('scoring', {}).get('wizard_model') or getattr(scorer, 'inference_model', None) or scorer.model

        # Build a compact list for the LLM
        cat_desc = []
        for cat in available:
            subs = ", ".join(s.get("id", "") for s in cat.get("subs", []))
            cat_desc.append(f'  {cat["id"]}: {cat["label"]} (subs: {subs})')
        cat_list = "\n".join(cat_desc)

        prompt = f"""This person's role: {role}
This person's location: {location or 'Not specified'}

Given this SPECIFIC role and location, which of these categories and sub-categories should they track?

Available categories:
{cat_list}

Pick the 2-4 categories and sub-categories most relevant for a "{role}" in "{location or 'anywhere'}"."""

        logger.info(f"Wizard preselect: role='{role}', location='{location}', model='{model}'")
        raw = _call_ollama(host, model, PRESELECT_SYSTEM, prompt, max_tokens=500, temperature=0.2)
        logger.info(f"Wizard preselect raw response: {raw[:300] if raw else 'None'}")

        if not raw:
            # Fallback: empty selections — let frontend handle defaults
            json_response(handler, {"selected_categories": [], "selected_subs": {}})
            return

        # Parse JSON from response (strip reasoning wrapping if present)
        try:
            result = json.loads(extract_json(raw))
            cats = result.get("selected_categories", [])
            subs = result.get("selected_subs", {})
            if not isinstance(cats, list):
                cats = []
            if not isinstance(subs, dict):
                subs = {}
            json_response(handler, {"selected_categories": cats, "selected_subs": subs})
            return
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback on parse failure
        logger.warning(f"Wizard preselect: Could not parse LLM response: {raw[:200]}")
        json_response(handler, {"selected_categories": [], "selected_subs": {}})

    except Exception as e:
        logger.error(f"Wizard preselect error: {e}")
        error_response(handler, "Internal server error", 500)


def handle_wizard_tab_suggest(handler, strat):
    """POST /api/wizard-tab-suggest — AI suggests keywords for a Step 2 tab."""
    try:
        data = read_json_body(handler)
        role = data.get("role", "").strip()
        location = data.get("location", "").strip()
        category_id = data.get("category_id", "")
        category_label = data.get("category_label", "")
        existing_items = data.get("existing_items", [])
        selections_context = data.get("selections_context", "").strip()
        selections = data.get("selections", {})  # structured dict: {label: [values]}
        exclude_selected = data.get("exclude_selected", [])  # previously-added suggestions

        # Merge explicitly excluded items (previously-added suggestions)
        if exclude_selected:
            existing_items = list(set(existing_items + [str(e) for e in exclude_selected]))

        # Build context from structured selections if provided (overrides flat string)
        if selections and isinstance(selections, dict):
            parts = []
            for label, values in selections.items():
                if isinstance(values, list) and values:
                    parts.append(f"{label}: {', '.join(str(v) for v in values)}")
                elif isinstance(values, str) and values:
                    parts.append(f"{label}: {values}")
            if parts:
                selections_context = '; '.join(parts)

        if not role or not category_label:
            error_response(handler, "Role and category_label are required", 400)
            return

        scorer = strat.scorer
        host = scorer.host
        model = strat.config.get('scoring', {}).get('wizard_model') or getattr(scorer, 'inference_model', None) or scorer.model

        existing_str = ", ".join(existing_items[:20]) if existing_items else "none yet"

        # Build context block — make stage/opportunity prominent with anti-examples
        context_block = ""
        if selections_context:
            # Parse stage for anti-example guidance
            stage_hints = ""
            sc_lower = selections_context.lower()
            if "student" in sc_lower or "internship" in sc_lower:
                stage_hints = "\nFOCUS ON: internship programs, student competitions, campus career fairs, beginner courses, entry-level bootcamps, university clubs.\nAVOID: executive roles, C-suite networking, senior leadership, board positions, management consulting."
            elif "senior" in sc_lower or "executive" in sc_lower:
                stage_hints = "\nFOCUS ON: executive search firms, C-suite networking events, board advisory roles, leadership summits, strategic consulting, industry keynotes.\nAVOID: internships, student programs, entry-level, campus fairs, beginner courses, bootcamps."
            elif "fresh graduate" in sc_lower:
                stage_hints = "\nFOCUS ON: graduate programs, trainee positions, associate roles, early-career accelerators, professional mentorship, entry certifications.\nAVOID: executive roles, C-suite, board positions, senior management, leadership summits."

            context_block = f"\n\n=== CRITICAL CONTEXT ===\n{selections_context}{stage_hints}\n=== END ==="

        prompt = f"""Role: {role}
Location: {location or 'Not specified'}
Category: {category_label}{context_block}
Already tracking: {existing_str}

Suggest 5-8 specific entities or keywords for the "{category_label}" category. Every suggestion MUST match the stage and goals above. Do not repeat items already being tracked."""

        raw = _call_ollama(host, model, TAB_SUGGEST_SYSTEM, prompt, max_tokens=500, temperature=0.3)

        if not raw:
            # Fallback: empty suggestions
            json_response(handler, {"suggestions": []})
            return

        # Parse JSON array from response (strip reasoning wrapping if present)
        try:
            suggestions = json.loads(extract_json(raw))
            if isinstance(suggestions, list):
                suggestions = [s for s in suggestions if isinstance(s, str)][:8]
                json_response(handler, {"suggestions": suggestions})
                return
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback on parse failure
        logger.warning(f"Wizard tab suggest: Could not parse LLM response: {raw[:200]}")
        json_response(handler, {"suggestions": []})

    except Exception as e:
        logger.error(f"Wizard tab suggest error: {e}")
        error_response(handler, "Internal server error", 500)


RV_ITEMS_SYSTEM = """You are a personalization engine for a professional news dashboard.
Given a user's EXACT role, location, and selected dashboard sections, generate role-appropriate entities for each section.
CRITICAL: Respond with ONLY valid JSON — no markdown, no explanation, no backticks, no reasoning.

Schema:
{
  "sections": {
    "<section_id>": {"label": "short label", "items": ["entity1", "entity2", ...], "tags": {"entity1": "tag1", "entity2": "tag2"}},
    ...
  },
  "discover": [
    {"name": "entity", "tag": "category", "target": "section_id"},
    ...
  ]
}

Rules:
- For EACH section, suggest 3-6 specific entities (companies, platforms, certifications, organizations) that a person with this EXACT role in this EXACT location would use or track
- "label" is a short category word like "Employers", "Certifications", "Platforms", "Companies", "Organizations"
- "tags" maps each entity to a short 1-2 word category tag for display (e.g. "Jobs", "Cloud", "Finance")
- "discover" has 4-6 additional entities the user might want, spread across different sections
- Be HIGHLY specific to the role and location:
  * A "Pastry Chef in Tokyo" should see: Japanese culinary schools, Tsuji Culinary Institute, Japanese Patisserie Association — NOT AWS, Google, CCNA
  * A "Marine Biologist in Italy" should see: ISPRA, CNR, Mediterranean Science Commission — NOT LinkedIn Jobs, Indeed, Glassdoor
  * A "Nuclear Engineer in Germany" should see: Framatome, TÜV, Bundesamt für Strahlenschutz — NOT Apple, Meta, Microsoft
- For job/career sections: suggest role-specific job platforms, industry recruiters, and local employers
- For certification sections: suggest certifications relevant to THIS role, not generic IT certs
- For industry sections: suggest companies and organizations in THIS person's actual industry
- For events sections: suggest conferences and events for THIS profession
- NEVER suggest generic tech entities (AWS, Google, Microsoft, CCNA, CompTIA) unless the role is specifically in tech/IT
- Location matters: include local companies, institutions, job portals for the user's country/city"""


def _validate_entities_ddg(items: list, max_workers: int = 4, timeout: float = 5.0) -> list:
    """Validate entity names via DDG search. Returns only items with >0 results."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return items  # Can't validate without DDG — pass through

    def _check(item):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(item, max_results=1))
                return item, len(results) > 0
        except Exception:
            return item, True  # On error, keep the item

    valid = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_check, item): item for item in items}
        for f in as_completed(futures, timeout=timeout * len(items)):
            try:
                item, has_results = f.result(timeout=timeout)
                if has_results:
                    valid.append(item)
                else:
                    logger.info(f"Wizard entity validation: dropped '{item}' (0 search results)")
            except Exception:
                valid.append(futures[f])  # Keep on timeout
    return valid


def handle_wizard_rv_items(handler, strat):
    """POST /api/wizard-rv-items — AI generates role-aware entities for Step 3 review."""
    try:
        data = read_json_body(handler)
        role = data.get("role", "").strip()
        location = data.get("location", "").strip()
        sections = data.get("sections", [])  # [{id, name, category}]

        if not role or not sections:
            error_response(handler, "Role and sections are required", 400)
            return

        scorer = strat.scorer
        host = scorer.host
        model = strat.config.get('scoring', {}).get('wizard_model') or getattr(scorer, 'inference_model', None) or scorer.model

        # Build compact section list for the LLM
        sec_lines = []
        for sec in sections:
            sec_lines.append(f'  {sec["id"]}: {sec["name"]} (category: {sec.get("category", "general")})')
        sec_list = "\n".join(sec_lines)

        prompt = f"""Role: {role}
Location: {location or 'Not specified'}

Generate role-appropriate entities for each of these dashboard sections:

{sec_list}

Remember: every entity MUST be relevant to a "{role}" in "{location or 'anywhere'}". Do NOT use generic tech defaults."""

        logger.info(f"Wizard rv-items: role='{role}', location='{location}', sections={len(sections)}")
        raw = _call_ollama(host, model, RV_ITEMS_SYSTEM, prompt, max_tokens=1500, temperature=0.3)
        logger.info(f"Wizard rv-items raw response: {raw[:300] if raw else 'None'}")

        if not raw:
            json_response(handler, {"sections": {}, "discover": []})
            return

        # Parse JSON from response (strip reasoning wrapping if present)
        try:
            result = json.loads(extract_json(raw))
            secs = result.get("sections", {})
            discover = result.get("discover", [])
            if not isinstance(secs, dict):
                secs = {}
            validated = {}
            for sid, sdata in secs.items():
                if isinstance(sdata, dict):
                    items = sdata.get("items", [])
                    if isinstance(items, list):
                        items = [i for i in items if isinstance(i, str)][:8]
                    else:
                        items = []
                    # Validate entities exist via DDG search
                    if items:
                        items = _validate_entities_ddg(items)
                    label = sdata.get("label", "Tracking")
                    tags = sdata.get("tags", {})
                    if not isinstance(tags, dict):
                        tags = {}
                    validated[sid] = {"label": label, "items": items, "tags": tags}
            if not isinstance(discover, list):
                discover = []
            discover = [d for d in discover if isinstance(d, dict) and "name" in d][:6]
            json_response(handler, {"sections": validated, "discover": discover})
            return
        except (json.JSONDecodeError, ValueError):
            pass

        logger.warning(f"Wizard rv-items: Could not parse LLM response: {raw[:200]}")
        json_response(handler, {"sections": {}, "discover": []})

    except Exception as e:
        logger.error(f"Wizard rv-items error: {e}")
        error_response(handler, "Internal server error", 500)
