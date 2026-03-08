"""
Generate routes — /api/generate-profile
"""

import json
import re
import logging
import requests as req

from routes.helpers import (json_response, error_response, read_json_body,
                           strip_think_blocks, extract_json)
from processors.profile_generator import run_pipeline

logger = logging.getLogger("STRAT_OS")

# The full system prompt for category generation
GENERATE_SYSTEM_PROMPT = """You are a configuration assistant for STRAT_OS, an intelligence monitoring dashboard.
Given a user's professional role, location, and interests, generate ONLY the tracking categories that are genuinely relevant to them.

You MUST respond with ONLY valid JSON — no markdown, no explanation, no backticks, no reasoning. Just the JSON object.

The JSON schema:
{
  "categories": [
    {
      "id": "string (unique short id, lowercase, no spaces, max 12 chars)",
      "label": "string (display name, 2-4 words, readable and clear — NOT abbreviated)",
      "icon": "string (one of: briefcase, landmark, cpu, bar-chart-3, globe, heart, book-open, shield, zap, building-2, flask-conical, truck, stethoscope, gavel, palette, award, target)",
      "items": ["string array of ENTITIES and SHORT TERMS to search for — see rules below"],
      "enabled": true,
      "scorer_type": "string (one of: career, banks, tech, regional, auto). Use 'career' for employer/job categories, 'banks' for banking/finance deals, 'tech' for technology/science, 'regional' for GCC/regional news, 'auto' for everything else",
      "root": "string (one of: kuwait, regional, global, ai). 'kuwait' = LOCAL news tab (use for location-specific employers/institutions regardless of country), 'regional' = surrounding region news, 'global' = worldwide/industry news, 'ai' = AI-specific news"
    }
  ],
  "tickers": ["string array of relevant market ticker symbols (Yahoo Finance format)"],
  "timelimit": "d or w or m (recommended news freshness)",
  "context": "string (1-2 sentence profile summary for AI scoring)"
}

=== CATEGORY DESIGN RULES ===

Generate 4-7 BROAD categories. Aim for at least 5 categories. Each category must cover a WIDE topic area with 5-8 items inside.
- If you generate fewer than 4 categories, you have FAILED — you are being too broad. Split the largest category.
- If you generate more than 7 categories, you have FAILED — combine related topics.
- Categories are like FOLDERS — they group related topics together.
- Individual topics, techniques, and sub-specialties are ITEMS inside categories, NOT standalone categories.
- If a topic could fit as an item inside a broader category, it MUST be an item, not its own category.

CRITICAL SEPARATION RULES:
- EMPLOYERS and BANKS must NEVER be in the same category. Companies you work at (employers) and banks that offer financial deals are completely different verticals.
- INVESTMENT ASSETS (gold, silver, crypto, stocks) belong in TICKERS, not as category items. The market ticker system already tracks prices/charts for these. Category items are for NEWS topics.
  BAD: "Emerging Tech" → items: ["superconductors", "blockchain", "gold", "silver"]  ← gold/silver are commodities, not tech trends
  GOOD: "Emerging Tech" → items: ["superconductors", "quantum computing", "solid-state battery", "neuromorphic chips"]
  GOOD: tickers: ["GC=F", "SI=F", "BTC-USD"]  ← investment assets tracked here instead
- If the user mentions both employers AND banks, create SEPARATE categories with different scorer_types (career vs banks).
- If the user mentions wanting to track investment trends/commodities, put the TECHNOLOGY behind them in a tech category, and the ASSETS in tickers.

LABELS:
- Labels must be FULL READABLE NAMES, not abbreviations.
- GOOD labels: "Career & Hiring", "Process Engineering", "Materials Science", "Energy Markets"
- BAD labels: "Proc Safe", "Cert Track", "Alt Fuels", "Ref Op", "Mat Sci Innv", "Water Treat"
- Labels should be 2-4 words, clear to a human reader without context.

WHAT SHOULD BE A CATEGORY vs an ITEM:
Category = a broad domain that generates its own news feed tab (5-8 of these)
Item = a specific entity, topic, or keyword INSIDE a category (5-8 per category)

Example for "Pastry Chef in Tokyo":
GOOD (4 broad categories):
  "Tokyo Culinary Scene" (scorer_type: career, root: kuwait) → items: ["Tsuji Culinary Institute", "Tokyo pastry jobs", "Japan Restaurant Association", "hotel pastry positions", "bakery openings Tokyo"]
  "Pastry Techniques & Trends" (scorer_type: tech, root: global) → items: ["French pastry methods", "chocolate tempering", "sugar work techniques", "sourdough fermentation", "molecular gastronomy", "plant-based desserts"]
  "Culinary Business & Industry" (scorer_type: auto, root: regional) → items: ["Japan food industry", "bakery supply chain", "restaurant management", "food safety regulations", "Michelin Guide Japan"]
  "Career & Certifications" (scorer_type: career, root: global) → items: ["Le Cordon Bleu", "pastry competitions", "World Pastry Cup", "food hygiene certification", "culinary awards"]
BAD for chef: "Cloud Computing Trends" → chefs don't track infrastructure!
BAD for chef: "K-Sector Employers" → completely irrelevant for culinary professionals!

Example for "Civil Engineer in Riyadh":
GOOD (5 categories):
  "Saudi Construction Employers" (scorer_type: career, root: kuwait) → items: ["Saudi Aramco", "NEOM", "Diriyah Gate", "Red Sea Global", "Bechtel Saudi", "Jacobs Engineering"]
  "Structural & Geotechnical" (scorer_type: tech, root: global) → items: ["BIM modeling", "structural analysis", "foundation design", "concrete technology", "seismic design", "green building"]
  "Saudi Infrastructure Projects" (scorer_type: regional, root: regional) → items: ["Vision 2030 projects", "Riyadh Metro", "NEOM construction", "Saudi mega projects", "GCC infrastructure"]
  "Career & Certifications" (scorer_type: career, root: global) → items: ["PE license Saudi", "PMP certification", "LEED accreditation", "Saudi Council of Engineers"]
  "Construction Industry News" (scorer_type: auto, root: global) → items: ["construction automation", "prefabrication trends", "sustainability standards", "smart buildings", "construction safety"]
BAD for civil engineer: "Semiconductor & ICT Trends" → civil engineers don't track chip manufacturing!
BAD for civil engineer: "Banking Deals" → irrelevant unless they asked for it!

Example for "Marketing Strategist in Singapore":
GOOD (5 categories):
  "Digital Marketing & Analytics" (scorer_type: auto, root: global) → items: ["Google Analytics", "SEO strategy", "social media marketing", "content marketing", "programmatic advertising", "marketing automation"]
  "Singapore & APAC Markets" (scorer_type: regional, root: regional) → items: ["Singapore consumer trends", "APAC market reports", "Southeast Asia digital", "Grab", "Shopee", "Sea Group"]
  "Brand & Creative Strategy" (scorer_type: auto, root: global) → items: ["brand positioning", "consumer insights", "competitive analysis", "campaign management", "creative direction"]
  "Career & Professional Growth" (scorer_type: career, root: kuwait) → items: ["Google Ads certification", "HubSpot Academy", "marketing leadership roles", "agency roles Singapore"]
  "Advertising & Media" (scorer_type: auto, root: global) → items: ["Meta Ads", "TikTok marketing", "influencer marketing", "media buying", "CRM platforms", "MarTech"]
BAD for marketing strategist: "Tech & Software Innovations" → they need MarTech, not chip/infrastructure trends!
BAD for marketing strategist: "Telecom & 5G Infrastructure" → completely irrelevant unless they work IN telecom!

Example for "Registered Nurse in Chicago":
GOOD (4 categories):
  "Clinical Practice & Safety" (scorer_type: auto) → items: ["evidence-based nursing", "patient safety", "infection control", "medication management", "wound care", "critical care updates"]
  "Chicago Healthcare" (scorer_type: regional) → items: ["Northwestern Medicine", "Rush University Medical", "Advocate Aurora", "UI Health", "Illinois nursing board"]
  "Nursing Career & Certifications" (scorer_type: career) → items: ["CCRN certification", "nurse practitioner programs", "BSN to DNP", "travel nursing", "nursing leadership"]
  "Healthcare Industry Trends" (scorer_type: auto) → items: ["telehealth nursing", "AI in healthcare", "nurse staffing crisis", "digital health records", "health equity"]
BAD for a nurse: "Oil & Energy Markets" → completely irrelevant!
BAD for a nurse: "Tech & Software Innovations" → they need health IT, not general tech!

=== CRITICAL — ROLE FOCUS RULE ===
The user's ROLE is the #1 signal. The context field may contain leftover or generic text — IGNORE any context that contradicts the role.
If the role says "geophysicist" but context mentions "semiconductor trends", generate categories for a GEOPHYSICIST, not a semiconductor engineer.
Every category you generate must pass the test: "Would someone with THIS EXACT JOB TITLE read news about this daily?"

=== ROLE DIVERSITY ===
Users can be ANYTHING: marketers, lawyers, chefs, teachers, farmers, journalists, designers, accountants, nurses, architects, etc.
Do NOT default to tech/engineering categories for non-tech roles. A chef needs restaurant industry news, not cloud computing. A lawyer needs legal updates, not telecom infrastructure. A teacher needs education policy, not cybersecurity trends.
Generate categories that match the ACTUAL profession — not a generic tech template.

=== LOCATION-SPECIFIC GUIDANCE ===

Use your knowledge of major employers, institutions, and industry bodies in the user's location. Examples:
- Kuwait: K-Sector (KOC, KNPC, Equate, KIPIC), banks (NBK, Boubyan, KFH), telecom (Zain, Ooredoo)
- Saudi Arabia: Saudi Aramco, NEOM, SABIC, STC, Al Rajhi Bank
- UAE: ADNOC, Etisalat, Emirates NBD, DIFC, DMCC
- Singapore: DBS, Grab, Sea Group, GovTech, Temasek
- USA: vary by city and industry — use common sense

CRITICAL — EMPLOYER RELEVANCE RULE:
Industry-specific employers (oil/gas companies, construction firms, etc.) are ONLY relevant for roles that work IN that industry.
They are NOT relevant for: unrelated professions in the same location. A teacher in Kuwait doesn't need K-Sector oil companies. A nurse in Riyadh doesn't need Saudi Aramco."""


def handle_generate_profile(handler, strat):
    """POST /api/generate-profile — AI-powered category generation."""
    try:
        data = read_json_body(handler)
        role = data.get("role", "").strip()
        location = data.get("location", "").strip()
        user_context = data.get("context", "").strip()
        deep = data.get("deep", False)

        if not role:
            raise ValueError("Role is required")

        scorer = strat.scorer
        scoring_cfg = strat.config.get('scoring', {})
        # Deep mode uses the full inference model; quick uses the lighter wizard model
        if deep:
            wiz_model = getattr(scorer, 'inference_model', None) or scorer.model
            logger.info(f"Generate: DEEP mode — using {wiz_model}")
        else:
            wiz_model = scoring_cfg.get('wizard_model') or getattr(scorer, 'inference_model', None) or scorer.model
            logger.info(f"Generate: QUICK mode — using {wiz_model}")
        # Check if Ollama is reachable and the model is available
        try:
            _r = req.get(f"{scorer.host}/api/tags", timeout=5)
            if _r.status_code != 200:
                raise RuntimeError("Ollama is not available")
            _models = [m.get("name","") for m in _r.json().get("models",[])]
            _bases = [n.split(":")[0] for n in _models]
            _inf_base = wiz_model.split(":")[0]
            if not (wiz_model in _models or _inf_base in _bases or f"{wiz_model}:latest" in _models):
                raise RuntimeError(f"Model '{wiz_model}' not found in Ollama")
        except req.exceptions.ConnectionError:
            raise RuntimeError("Ollama is not available")

        # Build user prompt
        prompt = f"""Role: {role}
Location: {location or 'Not specified'}
Additional context: {user_context or 'None provided'}

Generate the optimal STRAT_OS configuration for this person. Remember:
- The ROLE field is the primary signal — generate categories for THAT job title
- If context mentions topics unrelated to the role, IGNORE those topics
- 4-7 broad categories with 5-8 items each
- Employers and banks in SEPARATE categories
- Investment assets (gold, crypto, stocks) go in tickers, NOT category items
- Labels must be full readable names (2-4 words)
- Use the correct scorer_type for each category
- Every category must be something this person would check DAILY in their job"""

        # Retry loop — LLM sometimes returns malformed JSON on first try
        raw = ""
        for attempt in range(2):
            messages = [
                {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            if attempt > 0:
                # Reinforce JSON-only output on retry
                messages.append({"role": "assistant", "content": "{"})
                messages.append({"role": "user", "content": "Respond ONLY with valid JSON. No explanation, no reasoning, no markdown. Just the JSON object starting with {"})

            response = req.post(
                f"{scorer.host}/api/chat",
                json={
                    "model": wiz_model,
                    "messages": messages,
                    "stream": False,
                    "think": False,  # JSON-only output — no reasoning needed
                    "options": {
                        "temperature": 0.3 if attempt == 0 else 0.2,
                        "num_predict": 2000 if attempt == 0 else 3000,
                        "num_ctx": 6144 if attempt == 0 else 8192,
                    }
                },
                timeout=60 if attempt == 0 else 90
            )

            if response.status_code != 200:
                if attempt == 0:
                    continue
                raise RuntimeError(f"Ollama returned {response.status_code}")

            resp_data = response.json()
            raw = resp_data.get("message", {}).get("content", "")
            raw = strip_think_blocks(raw)  # safety net

            logger.info(f"Generate: raw response ({len(raw)} chars): {raw[:200]}...")

            # Extract and parse JSON (handles reasoning text wrapping JSON)
            json_str = extract_json(raw)
            try:
                profile_data = json.loads(json_str)
                break  # Success
            except json.JSONDecodeError:
                if attempt == 0:
                    logger.warning(f"Generate: JSON decode failed, retrying... raw: {raw[:300]}")
                    continue
                raise ValueError(f"No valid JSON in AI response after retry. Raw: {raw[:300]}")

        # Validate structure
        categories = profile_data.get("categories", [])
        tickers = profile_data.get("tickers", [])
        if not isinstance(categories, list):
            categories = []
        if not isinstance(tickers, list):
            tickers = []

        # Run the full post-processing pipeline
        categories, tickers = run_pipeline(
            categories=categories,
            tickers=tickers,
            role=role,
            location=location,
            context=user_context or profile_data.get("context", ""),
            ollama_host=scorer.host,
            model=wiz_model
        )

        profile_data["categories"] = categories
        profile_data["tickers"] = tickers

        json_response(handler, profile_data)

    except json.JSONDecodeError as e:
        logger.error(f"Generate: JSON parse error: {e}")
        error_response(handler, f"AI returned invalid JSON: {e}", 500)
    except Exception as e:
        logger.error(f"Generate error: {e}")
        error_response(handler, "Internal server error", 500)
