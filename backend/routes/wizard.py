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


PRESELECT_SYSTEM = """You are a configuration assistant that personalizes a professional news dashboard.
Given a user's EXACT role and location, select which categories and sub-categories they should track.
CRITICAL: Respond with ONLY valid JSON — no markdown, no explanation, no backticks.
Schema: {"selected_categories": ["id1","id2",...], "selected_subs": {"id1": ["sub1","sub2"], "id2": ["sub3"]}}
IMPORTANT: selected_subs keys MUST be category IDs (career, industry, learning, markets, deals) — NOT sub-category IDs. Every selected category MUST have at least 1 sub listed.

SELECTION STRATEGY:
1. ROLE DOMAIN determines which categories and industry subs matter:
   - Software/web/app developers → ind_tech, learning:courses,open_source
   - DevOps/SRE/cloud → ind_tech, learning:certs,hands_on (cloud certs matter)
   - Data/ML/AI roles → ind_tech, learning:academic,courses,competitions
   - Cybersecurity → ind_tech, learning:certs,hands_on,competitions
   - Finance roles → markets is essential with specific subs (stocks/bonds/forex)
   - Healthcare (doctor/nurse) → ind_health, learning:certs,academic
   - Pharma → ind_pharma, learning:certs,academic
   - Creative (designer/artist) → career:creative_jobs, ind_media, learning:workshops
   - Education (teacher/professor) → career:teaching_pos, ind_edu, learning:academic
   - Non-software engineering → pick correct industry (ind_construct, ind_energy, ind_auto, ind_aero)
   - Legal → ind_legal, learning:certs,confevents
   - Journalism/media → ind_media, learning:workshops
   - Product/project management → ind_tech + career:jobhunt + learning:workshops,confevents
   - Architecture (buildings) → ind_construct, learning:certs,workshops
   - Translator/interpreter/linguist → ind_edu, career:freelance,remote, learning:courses,workshops (NOT ind_legal, NOT ind_tech)
   - Hospitality (chef/hotel) → career:jobhunt, ind_food, learning:workshops,certs

2. SUB-CATEGORY PRECISION — differentiate similar roles:
   - Frontend Dev → courses, open_source (visual/UX focus)
   - Backend Engineer → courses, hands_on (system design)
   - DevOps → certs, hands_on (AWS/GCP/K8s certs)
   - Data Scientist → academic, courses, competitions
   - ML Engineer → courses, open_source, competitions
   - QA Engineer → certs, hands_on, courses
   - Product Manager → workshops, confevents (NOT certs)
   - UX Designer → workshops, courses (NOT academic)
   - Translator → courses, workshops (language certs like JLPT/DELF, NOT IT certs)

3. LOCATION influences subs:
   - Gulf/Middle East → ind_oilgas ONLY if role is engineering, energy, or oil-related. Software devs in Kuwait do NOT need ind_oilgas
   - Tech hubs (SF, Austin, Seattle) → career:startup_jobs
   - Financial hubs (NYC, London, HK) → markets subs matter more
   - Do NOT add industry subs just because of location — the ROLE is always the primary signal

4. RULES:
   - Select 2-4 categories, 1-3 subs each
   - EVERY selected category MUST have subs listed (except interests which has none)
   - career for most professionals (skip for C-suite/retirees)
   - markets ONLY for finance/investing/real-estate roles
   - deals for students (studisc) and early-career
   - interests (no subs) for trend-following roles
   - Each role must produce a UNIQUE sub combination"""


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


MAX_INPUT_LEN = 200  # Max length for role, location, context fields
MAX_CONTEXT_LEN = 1000  # Max length for context/selection strings


def handle_wizard_preselect(handler, strat):
    """POST /api/wizard-preselect — AI picks relevant categories for Step 1."""
    try:
        data = read_json_body(handler)
        role = data.get("role", "").strip()[:MAX_INPUT_LEN]
        location = data.get("location", "").strip()[:MAX_INPUT_LEN]
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
        role = data.get("role", "").strip()[:MAX_INPUT_LEN]
        location = data.get("location", "").strip()[:MAX_INPUT_LEN]
        category_id = data.get("category_id", "")[:MAX_INPUT_LEN]
        category_label = data.get("category_label", "")[:MAX_INPUT_LEN]
        existing_items = data.get("existing_items", [])
        selections_context = data.get("selections_context", "").strip()[:MAX_CONTEXT_LEN]
        selections = data.get("selections", {})  # structured dict: {label: [values]}
        exclude_selected = data.get("exclude_selected", [])  # previously-added suggestions
        is_refresh = data.get("is_refresh", False)

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

        logger.info(f"Wizard tab-suggest: category_id='{category_id}', category_label='{category_label}', role='{role}'")

        if not role or not category_label:
            error_response(handler, "Role and category_label are required", 400)
            return

        scorer = strat.scorer
        host = scorer.host
        model = strat.config.get('scoring', {}).get('wizard_model') or getattr(scorer, 'inference_model', None) or scorer.model

        existing_str = ", ".join(existing_items[:40]) if existing_items else "none yet"

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

        raw = _call_ollama(host, model, TAB_SUGGEST_SYSTEM, prompt, max_tokens=500, temperature=0.7 if is_refresh else 0.3)

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
    "<section_id>": {"label": "short label", "items": ["entity1", "entity2", ...]},
    ...
  },
  "discover": ["extra entity 1", "extra entity 2", ...]
}

Rules:
- For EACH section, suggest 3-5 specific entities (companies, platforms, certifications, organizations) relevant to this EXACT role and location
- "label" is a short category word like "Employers", "Certifications", "Platforms", "Companies"
- "discover" has 4-6 additional entities the user might want
- Be HIGHLY specific to the role and location:
  * "Pastry Chef in Tokyo" → Tsuji Culinary Institute, Japanese Patisserie Association — NOT AWS, Google
  * "Marine Biologist in Italy" → ISPRA, CNR, Mediterranean Science Commission — NOT LinkedIn, Indeed
- NEVER suggest generic tech entities unless the role is specifically in tech/IT
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
        except Exception as e:
            logger.debug(f"Wizard entity validation error for '{item}': {e}")
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
            except Exception as e:
                logger.debug(f"Wizard entity validation future error: {e}")
                valid.append(futures[f])  # Keep on timeout
    return valid


def handle_wizard_rv_items(handler, strat):
    """POST /api/wizard-rv-items — AI generates role-aware entities for Step 3 review."""
    try:
        data = read_json_body(handler)
        role = data.get("role", "").strip()[:MAX_INPUT_LEN]
        location = data.get("location", "").strip()[:MAX_INPUT_LEN]
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

        # Scale tokens by section count — each section needs ~100-150 tokens
        max_tok = min(3000, max(800, len(sections) * 200))
        logger.info(f"Wizard rv-items: role='{role}', location='{location}', sections={len(sections)}, max_tokens={max_tok}")
        raw = _call_ollama(host, model, RV_ITEMS_SYSTEM, prompt, max_tokens=max_tok, temperature=0.3)
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
                    # DDG validation removed — LLM entities are contextual enough,
                    # and validation added 10-30s latency for minimal benefit
                    label = sdata.get("label", "Tracking")
                    tags = sdata.get("tags", {})
                    if not isinstance(tags, dict):
                        tags = {}
                    validated[sid] = {"label": label, "items": items, "tags": tags}
            if not isinstance(discover, list):
                discover = []
            # Support both formats: list of strings or list of dicts
            cleaned_discover = []
            for d in discover[:6]:
                if isinstance(d, str):
                    cleaned_discover.append({"name": d, "tag": "", "target": ""})
                elif isinstance(d, dict) and "name" in d:
                    cleaned_discover.append(d)
            discover = cleaned_discover
            json_response(handler, {"sections": validated, "discover": discover})
            return
        except (json.JSONDecodeError, ValueError):
            pass

        logger.warning(f"Wizard rv-items: Could not parse LLM response: {raw[:200]}")
        json_response(handler, {"sections": {}, "discover": []})

    except Exception as e:
        logger.error(f"Wizard rv-items error: {e}")
        error_response(handler, "Internal server error", 500)
