#!/usr/bin/env python3
"""
DuckDuckGo Multi-Profile Test Loop
===================================
Tests the full fetch→score pipeline across diverse profiles using DuckDuckGo
as the search provider. Measures language filtering, score distribution,
and overall quality per profile.

Usage:
    python3 test_ddg_profiles.py
"""

import copy
import json
import logging
import os
import sys
import time
import yaml
from collections import Counter
from pathlib import Path

# --- Setup ---

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DDG_TEST")
logger.setLevel(logging.INFO)

# Suppress noisy libraries
for lib in ['urllib3', 'requests', 'duckduckgo_search', 'yfinance', 'peewee']:
    logging.getLogger(lib).setLevel(logging.ERROR)

# --- Test Profiles ---

TEST_PROFILES = [
    {
        "name": "Petroleum Engineer — Kuwait",
        "profile": {
            "role": "Petroleum Engineering Consultant",
            "location": "Kuwait",
            "context": "Oil and gas industry intelligence, upstream projects, K-Sector employers.",
        },
        "dynamic_categories": [
            {"id": "ksector", "label": "K-Sector Employers", "enabled": True,
             "scorer_type": "career", "root": "kuwait",
             "items": ["KOC", "KNPC", "KIPIC", "KPC", "Equate", "SLB", "Halliburton", "Baker Hughes"]},
            {"id": "pettech", "label": "Petroleum Technology", "enabled": True,
             "scorer_type": "tech", "root": "global",
             "items": ["reservoir simulation", "seismic interpretation", "well logging",
                       "drilling technology", "digital oilfield", "AI in E&P"]},
            {"id": "kwreg", "label": "Kuwait Industry News", "enabled": True,
             "scorer_type": "regional", "root": "kuwait",
             "items": ["Kuwait oil output", "KOC projects", "KNPC expansions",
                       "Kuwait energy policy", "GCC upstream developments"]},
        ],
        "news_categories": {
            "career": {"root": "kuwait", "keywords": ["KOC", "KNPC", "SLB", "Halliburton"], "queries": []},
            "tech_trends": {"root": "global", "keywords": ["reservoir simulation", "seismic interpretation", "digital oilfield"], "queries": []},
            "regional": {"root": "regional", "keywords": ["Kuwait oil output", "KOC projects", "GCC upstream"], "queries": []},
            "finance": {"root": "kuwait", "keywords": [], "queries": []},
        },
    },
    {
        "name": "Software Engineer — Berlin",
        "profile": {
            "role": "Senior Software Engineer",
            "location": "Berlin, Germany",
            "context": "Full-stack developer focused on cloud infrastructure, Kubernetes, and AI/ML platforms.",
        },
        "dynamic_categories": [
            {"id": "techco", "label": "Tech Employers", "enabled": True,
             "scorer_type": "career", "root": "global",
             "items": ["Google", "Amazon", "SAP", "Zalando", "Delivery Hero", "N26", "Siemens Digital"]},
            {"id": "devtech", "label": "Development Technologies", "enabled": True,
             "scorer_type": "tech", "root": "global",
             "items": ["Kubernetes", "Docker", "Terraform", "AWS", "TypeScript", "React", "Go", "Rust"]},
            {"id": "aiml", "label": "AI & Machine Learning", "enabled": True,
             "scorer_type": "tech", "root": "global",
             "items": ["LLM fine-tuning", "RAG pipelines", "MLOps", "PyTorch", "transformer models"]},
        ],
        "news_categories": {
            "career": {"root": "global", "keywords": ["Google", "Amazon", "SAP", "Zalando"], "queries": []},
            "tech_trends": {"root": "global", "keywords": ["Kubernetes", "LLM", "RAG", "MLOps", "transformer"], "queries": []},
            "regional": {"root": "regional", "keywords": ["Berlin tech scene", "German startup ecosystem"], "queries": []},
            "finance": {"root": "global", "keywords": [], "queries": []},
        },
    },
    {
        "name": "Financial Analyst — Dubai",
        "profile": {
            "role": "Financial Analyst at Investment Bank",
            "location": "Dubai, UAE",
            "context": "Equity research, GCC markets, sovereign wealth funds, real estate investment trusts.",
        },
        "dynamic_categories": [
            {"id": "gccfin", "label": "GCC Financial Institutions", "enabled": True,
             "scorer_type": "career", "root": "kuwait",
             "items": ["Emirates NBD", "ADCB", "FAB", "Mashreq", "DIB", "ADIA", "Mubadala"]},
            {"id": "eqres", "label": "Equity Research Topics", "enabled": True,
             "scorer_type": "tech", "root": "global",
             "items": ["IPO analysis", "equity valuation", "DCF modeling", "earnings forecast",
                       "sector rotation", "market microstructure"]},
            {"id": "uaereg", "label": "UAE Market Regulation", "enabled": True,
             "scorer_type": "regional", "root": "regional",
             "items": ["DFM regulations", "DIFC policy", "UAE central bank", "ADX listings",
                       "GCC sovereign wealth funds"]},
        ],
        "news_categories": {
            "career": {"root": "global", "keywords": ["Emirates NBD", "ADIA", "Mubadala", "FAB"], "queries": []},
            "tech_trends": {"root": "global", "keywords": ["equity valuation", "DCF modeling", "IPO"], "queries": []},
            "regional": {"root": "regional", "keywords": ["Dubai financial market", "UAE central bank", "GCC sovereign wealth"], "queries": []},
            "finance": {"root": "global", "keywords": [], "queries": []},
        },
    },
    {
        "name": "Doctor — Riyadh",
        "profile": {
            "role": "Internal Medicine Physician",
            "location": "Riyadh, Saudi Arabia",
            "context": "Hospital-based physician tracking medical research, Saudi health policy, and career advancement.",
        },
        "dynamic_categories": [
            {"id": "ksamed", "label": "Saudi Healthcare Organizations", "enabled": True,
             "scorer_type": "career", "root": "regional",
             "items": ["King Faisal Specialist Hospital", "KAUST Health", "Saudi MOH",
                       "NGHA", "Saudi German Hospital", "King Abdulaziz Medical City"]},
            {"id": "medres", "label": "Medical Research", "enabled": True,
             "scorer_type": "tech", "root": "global",
             "items": ["clinical trials", "internal medicine guidelines", "cardiology research",
                       "diabetes management", "evidence-based medicine"]},
            {"id": "ksapol", "label": "Saudi Health Policy", "enabled": True,
             "scorer_type": "regional", "root": "regional",
             "items": ["Saudi Vision 2030 health", "Saudi MOH policy",
                       "Saudi medical licensing", "SCFHS", "Saudization healthcare"]},
        ],
        "news_categories": {
            "career": {"root": "regional", "keywords": ["King Faisal Specialist Hospital", "Saudi MOH", "NGHA"], "queries": []},
            "tech_trends": {"root": "global", "keywords": ["clinical trials", "internal medicine", "cardiology"], "queries": []},
            "regional": {"root": "regional", "keywords": ["Saudi Vision 2030 health", "Saudi MOH policy", "SCFHS"], "queries": []},
            "finance": {"root": "global", "keywords": [], "queries": []},
        },
    },
    {
        "name": "Nuclear Engineer — Germany",
        "profile": {
            "role": "Nuclear Engineer",
            "location": "Germany",
            "context": "Nuclear engineer in Germany focused on career development, government jobs, and professional certifications.",
        },
        "dynamic_categories": [
            {"id": "nuceng", "label": "Nuclear Employers", "enabled": True,
             "scorer_type": "career", "root": "global",
             "items": ["AREVA", "Siemens Energy", "Framatome", "E.ON", "ENBW", "Westinghouse"]},
            {"id": "nucres", "label": "Nuclear R&D", "enabled": True,
             "scorer_type": "tech", "root": "global",
             "items": ["fusion energy", "reactor design", "radiation safety",
                       "small modular reactors", "nuclear waste disposal"]},
            {"id": "nucpol", "label": "Nuclear Policy", "enabled": True,
             "scorer_type": "regional", "root": "regional",
             "items": ["German nuclear phase-out", "EU nuclear policy", "IAEA guidelines",
                       "nuclear licensing", "radiation protection laws"]},
        ],
        "news_categories": {
            "career": {"root": "global", "keywords": ["AREVA", "Siemens Energy", "Framatome"], "queries": []},
            "tech_trends": {"root": "global", "keywords": ["fusion energy", "reactor design", "small modular reactors"], "queries": []},
            "regional": {"root": "regional", "keywords": ["German nuclear phase-out", "EU nuclear policy", "IAEA"], "queries": []},
            "finance": {"root": "global", "keywords": [], "queries": []},
        },
    },
    {
        "name": "Marketing Manager — Tokyo",
        "profile": {
            "role": "Marketing Manager",
            "location": "Tokyo, Japan",
            "context": "Digital marketing for consumer electronics, social media strategy, Japanese market trends.",
        },
        "dynamic_categories": [
            {"id": "jpco", "label": "Japanese Tech Companies", "enabled": True,
             "scorer_type": "career", "root": "regional",
             "items": ["Sony", "Panasonic", "Toyota", "Nintendo", "Rakuten", "LINE", "SoftBank"]},
            {"id": "digimark", "label": "Digital Marketing", "enabled": True,
             "scorer_type": "tech", "root": "global",
             "items": ["social media marketing", "SEO strategy", "content marketing",
                       "influencer marketing", "programmatic advertising"]},
            {"id": "jpmarket", "label": "Japan Market Trends", "enabled": True,
             "scorer_type": "regional", "root": "regional",
             "items": ["Japan consumer spending", "Tokyo retail market",
                       "Japanese e-commerce", "Japan digital advertising"]},
        ],
        "news_categories": {
            "career": {"root": "regional", "keywords": ["Sony", "Panasonic", "Rakuten", "SoftBank"], "queries": []},
            "tech_trends": {"root": "global", "keywords": ["social media marketing", "SEO", "influencer marketing"], "queries": []},
            "regional": {"root": "regional", "keywords": ["Japan consumer spending", "Japanese e-commerce"], "queries": []},
            "finance": {"root": "global", "keywords": [], "queries": []},
        },
    },
]


def build_config(profile_spec: dict) -> dict:
    """Build a full StratOS config dict from a test profile spec."""
    return {
        "profile": profile_spec["profile"],
        "scoring": {
            "model": "stratos-scorer-v1",
            "inference_model": "qwen3:30b-a3b",
            "ollama_host": "http://localhost:11434",
            "forbidden_score": 5.0,
            "critical_min": 9.0,
            "high_min": 7.0,
            "medium_min": 5.0,
        },
        "search": {
            "provider": "duckduckgo",
        },
        "news": {
            "timelimit": "w",
            **profile_spec["news_categories"],
            "rss_feeds": [],
        },
        "dynamic_categories": profile_spec["dynamic_categories"],
        "market": {"tickers": []},
        "cache": {"news_ttl_seconds": 0},  # No cache — always fresh
        "system": {"max_news_items": 100, "database_file": "strat_os.db"},
        "discovery": {"enabled": False},
        "extra_feeds_finance": {},
        "extra_feeds_politics": {},
        "custom_feeds": [],
    }


def detect_language(text: str) -> str:
    """Simple heuristic to detect dominant script in text."""
    if not text:
        return "empty"
    counts = {"latin": 0, "arabic": 0, "cjk": 0, "cyrillic": 0, "other": 0}
    for ch in text:
        cp = ord(ch)
        if (0x0041 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
            counts["latin"] += 1
        elif 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFDFF:
            counts["arabic"] += 1
        elif 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0x3400 <= cp <= 0x4DBF:
            counts["cjk"] += 1
        elif 0x0400 <= cp <= 0x04FF:
            counts["cyrillic"] += 1
        elif cp > 127:
            counts["other"] += 1

    total = sum(counts.values())
    if total == 0:
        return "empty"
    # Return the dominant non-empty script
    dominant = max(counts, key=counts.get)
    pct = counts[dominant] / total * 100 if total else 0
    if counts[dominant] == 0:
        return "empty"
    return f"{dominant}({pct:.0f}%)"


def run_profile_test(profile_spec: dict) -> dict:
    """Run the full fetch→score pipeline for a single profile and return metrics."""
    from fetchers.kuwait_scrapers import KuwaitIntelligenceFetcher
    from fetchers.news import NewsItem
    from processors.scorer_adaptive import AdaptiveScorer
    from processors.scorer_base import location_to_lang

    config = build_config(profile_spec)
    profile_name = profile_spec["name"]
    location = config["profile"]["location"]

    # Language info
    allowed_scripts, ddg_region, lang_label = location_to_lang(location)
    logger.info(f"  Language config: allowed={allowed_scripts}, region={ddg_region}, label={lang_label}")

    # Phase 1: Fetch — use KuwaitIntelligenceFetcher directly (tests DDG queries)
    t0 = time.time()
    try:
        fetcher = KuwaitIntelligenceFetcher(config)
        raw_items = fetcher.fetch_all()
        # Convert dicts to NewsItem-like dicts (they're already dicts from kuwait_scrapers)
        items = raw_items
    except Exception as e:
        logger.error(f"  Fetch failed: {e}")
        return {"name": profile_name, "error": str(e), "fetched": 0, "scored": 0}

    fetch_time = time.time() - t0
    # KuwaitIntelligenceFetcher returns dicts directly (not NewsItem objects)
    item_dicts = items
    logger.info(f"  Fetched {len(item_dicts)} items in {fetch_time:.1f}s")

    if not item_dicts:
        return {"name": profile_name, "fetched": 0, "scored": 0, "fetch_time": fetch_time}

    # Phase 2: Score
    t1 = time.time()
    try:
        scorer = AdaptiveScorer(config)
        scored = scorer.score_items(item_dicts, max_workers=1)
    except Exception as e:
        logger.error(f"  Scoring failed: {e}")
        return {"name": profile_name, "error": str(e), "fetched": len(item_dicts), "scored": 0}

    score_time = time.time() - t1

    # Phase 3: Analyze
    scores = [it.get("score", 0) for it in scored]
    pre_filtered = [it for it in scored if it.get("pre_filter_score") is not None]

    # Language analysis
    lang_dist = Counter()
    non_english_high = []
    for it in scored:
        lang = detect_language(it.get("title", ""))
        lang_dist[lang] += 1
        if "cjk" in lang or "cyrillic" in lang or "arabic" in lang:
            if it.get("score", 0) >= 5.0:
                non_english_high.append({
                    "title": it["title"][:80],
                    "score": it["score"],
                    "lang": lang,
                })

    # Score distribution
    critical = sum(1 for s in scores if s >= 9.0)
    high = sum(1 for s in scores if 7.0 <= s < 9.0)
    medium = sum(1 for s in scores if 5.0 <= s < 7.0)
    noise = sum(1 for s in scores if s < 5.0)
    avg_score = sum(scores) / len(scores) if scores else 0

    # Top items
    top_items = sorted(scored, key=lambda x: x.get("score", 0), reverse=True)[:5]
    top_summaries = []
    for it in top_items:
        top_summaries.append({
            "title": it.get("title", "")[:100],
            "score": it.get("score", 0),
            "source": it.get("source", "?"),
            "lang": detect_language(it.get("title", "")),
        })

    # Bottom items (lowest non-zero)
    bottom_items = sorted([it for it in scored if it.get("score", 0) > 0],
                          key=lambda x: x.get("score", 0))[:3]
    bottom_summaries = []
    for it in bottom_items:
        bottom_summaries.append({
            "title": it.get("title", "")[:100],
            "score": it.get("score", 0),
            "reason": it.get("score_reason", "")[:80],
            "lang": detect_language(it.get("title", "")),
        })

    return {
        "name": profile_name,
        "location": location,
        "lang_config": {"allowed_scripts": list(allowed_scripts), "ddg_region": ddg_region, "lang_label": lang_label},
        "fetched": len(item_dicts),
        "pre_filtered": len(pre_filtered),
        "scored": len(scored),
        "fetch_time": round(fetch_time, 1),
        "score_time": round(score_time, 1),
        "avg_score": round(avg_score, 2),
        "distribution": {"critical": critical, "high": high, "medium": medium, "noise": noise},
        "language_dist": dict(lang_dist.most_common()),
        "non_target_lang_high_scores": non_english_high,
        "top_5": top_summaries,
        "bottom_3": bottom_summaries,
    }


def print_report(results: list):
    """Print a formatted summary report."""
    print("\n" + "=" * 90)
    print("DDG MULTI-PROFILE TEST REPORT")
    print("=" * 90)

    for r in results:
        print(f"\n{'─' * 90}")
        print(f"PROFILE: {r['name']}")
        if r.get("error"):
            print(f"  ERROR: {r['error']}")
            continue

        print(f"  Location: {r.get('location', '?')} | Lang: {r.get('lang_config', {}).get('lang_label', '?')} | DDG Region: {r.get('lang_config', {}).get('ddg_region', '?')}")
        print(f"  Fetched: {r['fetched']} | Pre-filtered: {r.get('pre_filtered', 0)} | Scored: {r['scored']}")
        print(f"  Fetch time: {r.get('fetch_time', 0)}s | Score time: {r.get('score_time', 0)}s")
        print(f"  Avg score: {r.get('avg_score', 0)}")

        dist = r.get("distribution", {})
        total = r.get("scored", 1) or 1
        print(f"  Distribution: Critical={dist.get('critical',0)} | High={dist.get('high',0)} | Medium={dist.get('medium',0)} | Noise={dist.get('noise',0)}")
        relevant_pct = (dist.get('critical', 0) + dist.get('high', 0) + dist.get('medium', 0)) / total * 100
        print(f"  Relevance rate: {relevant_pct:.1f}% (score >= 5.0)")

        lang = r.get("language_dist", {})
        print(f"  Languages: {lang}")

        if r.get("non_target_lang_high_scores"):
            print(f"  !! NON-TARGET LANGUAGE SCORED HIGH ({len(r['non_target_lang_high_scores'])} items):")
            for it in r["non_target_lang_high_scores"][:3]:
                print(f"     [{it['score']:.1f}] {it['lang']} — {it['title']}")

        print(f"\n  TOP 5:")
        for i, it in enumerate(r.get("top_5", []), 1):
            print(f"    {i}. [{it['score']:.1f}] ({it['source']}) {it['lang']} — {it['title']}")

        print(f"\n  BOTTOM 3:")
        for i, it in enumerate(r.get("bottom_3", []), 1):
            print(f"    {i}. [{it['score']:.1f}] {it['lang']} — {it['title']}")
            if it.get('reason'):
                print(f"       Reason: {it['reason']}")

    # Summary table
    print(f"\n{'=' * 90}")
    print("SUMMARY TABLE")
    print(f"{'=' * 90}")
    print(f"{'Profile':<35} {'Fetched':>7} {'Rel%':>6} {'Crit':>5} {'High':>5} {'Med':>5} {'Noise':>6} {'Avg':>6} {'NonTgt':>7}")
    print(f"{'─' * 35} {'─' * 7} {'─' * 6} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 6} {'─' * 6} {'─' * 7}")
    for r in results:
        if r.get("error"):
            print(f"{r['name']:<35} ERROR: {r['error'][:40]}")
            continue
        d = r.get("distribution", {})
        total = r.get("scored", 1) or 1
        rel_pct = (d.get('critical', 0) + d.get('high', 0) + d.get('medium', 0)) / total * 100
        non_tgt = len(r.get("non_target_lang_high_scores", []))
        print(f"{r['name']:<35} {r['fetched']:>7} {rel_pct:>5.1f}% {d.get('critical',0):>5} {d.get('high',0):>5} {d.get('medium',0):>5} {d.get('noise',0):>6} {r.get('avg_score',0):>6.2f} {non_tgt:>7}")

    print(f"\n{'=' * 90}")


def main():
    print("DuckDuckGo Multi-Profile Test Loop")
    print(f"Profiles: {len(TEST_PROFILES)}")
    print(f"Search provider: DuckDuckGo only")
    print()

    # Check Ollama is running
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            print("ERROR: Ollama not responding")
            sys.exit(1)
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"Ollama models: {', '.join(models[:5])}")
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama: {e}")
        sys.exit(1)

    results = []
    total_start = time.time()

    for i, profile in enumerate(TEST_PROFILES, 1):
        print(f"\n{'=' * 90}")
        logger.info(f"[{i}/{len(TEST_PROFILES)}] Testing: {profile['name']}")
        print(f"[{i}/{len(TEST_PROFILES)}] Testing: {profile['name']}")

        try:
            result = run_profile_test(profile)
            results.append(result)
        except Exception as e:
            logger.error(f"Profile test failed: {e}", exc_info=True)
            results.append({"name": profile["name"], "error": str(e), "fetched": 0, "scored": 0})

        # DDG rate limits aggressively with .news() — need longer pauses
        if i < len(TEST_PROFILES):
            logger.info("  Pausing 60s before next profile (DDG rate limit cooldown)...")
            time.sleep(60)

    total_time = time.time() - total_start
    print_report(results)
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}m)")

    # Save raw JSON results
    out_path = Path("test_ddg_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Raw results saved to: {out_path}")


if __name__ == "__main__":
    main()
