#!/usr/bin/env python3
"""
Post-Fix Validation — Verify diagnostic bug fixes work correctly.

Runs 3 adversarial profiles sequentially to confirm:
1. BUG-1 FIX: No cross-profile retention contamination
2. S3 FIX: Generic keyword matches route to DoRA model (not hardcoded 9.0)
3. BUG-2 FIX: wait_for_briefing() captures briefings reliably

Observation-only — does NOT commit results. Restores original config after.
Budget: Max 3 × ~23 = ~69 Serper credits. Stops if credits < 1500.
"""

import copy
import hashlib
import json
import logging
import os
import re
import shutil
import statistics
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path

# Setup
BACKEND = Path(__file__).parent
os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("POSTFIX")

DIAGNOSTIC_DIR = BACKEND / "diagnostic"
DIAGNOSTIC_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = BACKEND / "output" / "news_data.json"

# Load .env
env_path = BACKEND / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, value = line.partition('=')
                os.environ[key.strip()] = value.strip()

# PIN hash for test profiles
PIN_HASH = hashlib.sha256("0000".encode()).hexdigest()

# ─── Profile Definitions (P6, P7, P10 from Phase 1 diagnostic) ────

PROFILES = {
    6: {
        "name": "PostFix_P6_ElecTech",
        "profile": {
            "role": "Electrical Technician at the Ministry of Electricity & Water",
            "location": "Kuwait City, Kuwait",
            "context": "Maintains power distribution substations. Interested in trade certifications (City & Guilds, IEC standards) and government pay scale updates. Looking for overtime opportunities.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "mewjob",
                "label": "MEW & Government Jobs",
                "icon": "building-2",
                "items": ["Ministry of Electricity Water Kuwait", "MEW Kuwait", "government technician jobs Kuwait", "Kuwait civil service"],
                "enabled": True,
                "scorer_type": "career",
                "root": "kuwait"
            },
            {
                "id": "elecert",
                "label": "Electrical Trade Certifications",
                "icon": "award",
                "items": ["City Guilds electrical", "IEC 60364 training", "NEC code updates", "electrical safety certification"],
                "enabled": True,
                "scorer_type": "auto",
                "root": "global"
            },
            {
                "id": "powsys",
                "label": "Power Distribution & Substations",
                "icon": "cpu",
                "items": ["substation maintenance", "power transformer", "switchgear technology", "smart grid Kuwait"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "regional"
            }
        ],
        "news": {"timelimit": "w", "career": {"root": "kuwait", "keywords": [], "queries": []},
                 "finance": {"root": "kuwait", "keywords": [], "queries": []},
                 "regional": {"root": "regional", "keywords": [], "queries": []},
                 "tech_trends": {"root": "global", "keywords": [], "queries": []}, "rss_feeds": []},
        "market": {"tickers": [], "intervals": {}, "alert_threshold_percent": 5.0}
    },
    7: {
        "name": "PostFix_P7_PetroJourno",
        "profile": {
            "role": "Energy Sector Journalist at Kuwait Times",
            "location": "Kuwait City, Kuwait",
            "context": "Covering OPEC decisions, KPC quarterly results, and regional energy transition. Writing feature stories on oil industry workforce changes.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "opecnw",
                "label": "OPEC & Energy Markets",
                "icon": "target",
                "items": ["OPEC production", "KPC quarterly results", "Kuwait oil output", "energy transition GCC"],
                "enabled": True,
                "scorer_type": "regional",
                "root": "regional"
            },
            {
                "id": "kwtmed",
                "label": "Kuwait Media & Press",
                "icon": "building-2",
                "items": ["Kuwait Times", "Arab Times", "Kuwait News Agency KUNA", "GCC press awards"],
                "enabled": True,
                "scorer_type": "career",
                "root": "kuwait"
            },
            {
                "id": "enstory",
                "label": "Energy Industry Stories",
                "icon": "cpu",
                "items": ["oil workforce automation", "green hydrogen GCC", "carbon capture Middle East", "LNG export terminal"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "global"
            }
        ],
        "news": {"timelimit": "w", "career": {"root": "kuwait", "keywords": [], "queries": []},
                 "finance": {"root": "kuwait", "keywords": [], "queries": []},
                 "regional": {"root": "regional", "keywords": [], "queries": []},
                 "tech_trends": {"root": "global", "keywords": [], "queries": []}, "rss_feeds": []},
        "market": {"tickers": [{"symbol": "CL=F", "name": "Crude Oil"}], "intervals": {}, "alert_threshold_percent": 5.0}
    },
    10: {
        "name": "PostFix_P10_CSUK",
        "profile": {
            "role": "Computer Science Student (Junior year)",
            "location": "Kuwait City, Kuwait",
            "context": "Studying at Kuwait University (not AUK). Interested in web development and UI/UX design. Looking for summer internships at banks and tech startups in Kuwait.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "kustud",
                "label": "Kuwait University CS Department",
                "icon": "building-2",
                "items": ["Kuwait University", "KU computer science", "Kuwait University internship", "KU engineering faculty"],
                "enabled": True,
                "scorer_type": "career",
                "root": "kuwait"
            },
            {
                "id": "webdev",
                "label": "Web Development & UI/UX",
                "icon": "cpu",
                "items": ["React framework", "UI/UX design trends", "frontend development", "Figma prototyping"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "global"
            },
            {
                "id": "kwintern",
                "label": "Kuwait Tech Internships",
                "icon": "award",
                "items": ["Kuwait fintech startups", "Warba Bank technology", "NBK digital", "Zain Kuwait internship"],
                "enabled": True,
                "scorer_type": "career",
                "root": "kuwait"
            }
        ],
        "news": {"timelimit": "w", "career": {"root": "kuwait", "keywords": [], "queries": []},
                 "finance": {"root": "kuwait", "keywords": [], "queries": []},
                 "regional": {"root": "regional", "keywords": [], "queries": []},
                 "tech_trends": {"root": "global", "keywords": [], "queries": []}, "rss_feeds": []},
        "market": {"tickers": [], "intervals": {}, "alert_threshold_percent": 5.0}
    },
}

DEFAULT_FEEDS_FINANCE = {
    "cnbc_top": False, "cnbc_finance": False, "marketwatch": False,
    "mw_pulse": False, "yahoo_fin": False, "investing": False,
    "ft": False, "bloomberg": False, "wsj_world": False,
    "economist": False, "seekingalpha": False, "biz_insider": False,
    "coindesk": False, "cointelegraph": False, "gn_kuwait_biz": False,
    "gn_gcc_finance": False, "zawya": False, "arab_times_biz": False,
    "boursa_kuwait": False, "oilprice": False, "rigzone": False,
}
DEFAULT_FEEDS_POLITICS = {
    "bbc_world": False, "bbc_mideast": False, "aljazeera": False,
    "npr_world": False, "nyt_world": False, "nyt_mideast": False,
    "wapo_world": False, "wapo_politics": False, "cnn_world": False,
    "fox_world": False, "guardian": False, "dw": False, "france24": False,
    "gn_kuwait": False, "arab_times": False, "times_kuwait": False,
    "gn_mideast": False, "alarabiya": False, "middleeasteye": False,
    "newarab": False, "scmp": False, "nikkei_asia": False,
}


def get_context_hash(role, context, location):
    raw = f"{role}|{context}|{location}"
    norm = re.sub(r'\s+', ' ', raw.strip().lower())
    return hashlib.sha256(norm.encode()).hexdigest()[:12]


def check_serper_credits():
    try:
        import requests
        r = requests.get('https://google.serper.dev/account',
                         headers={'X-API-KEY': os.environ.get('SERPER_API_KEY', '')},
                         timeout=10)
        return r.json().get("balance", "unknown")
    except:
        return "unknown"


def run_profile(profile_num, prev_hash=None, prev_name=None):
    """Run a profile scan and return results with contamination analysis."""
    pdef = PROFILES[profile_num]
    name = pdef["name"]

    logger.info(f"\n{'='*60}")
    logger.info(f"PROFILE {profile_num}: {name}")
    logger.info(f"  Role: {pdef['profile']['role']}")
    logger.info(f"  Location: {pdef['profile']['location']}")
    logger.info(f"{'='*60}")

    # Create profile YAML
    yaml_data = {
        "profile": {**pdef["profile"], "name": name, "avatar": f"V{profile_num}", "email": f"val{profile_num}@test.local"},
        "security": {"pin_hash": PIN_HASH, "devices": ["validation_device"]},
        "dynamic_categories": pdef["dynamic_categories"],
        "market": pdef["market"],
        "news": pdef["news"],
        "extra_feeds_finance": DEFAULT_FEEDS_FINANCE.copy(),
        "extra_feeds_politics": DEFAULT_FEEDS_POLITICS.copy(),
        "custom_feeds": [],
        "custom_tab_name": "Custom",
    }
    filepath = BACKEND / "profiles" / f"{name}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    # Create FRESH StratOS instance
    from main import StratOS
    strat = StratOS("config.yaml")

    # Load profile
    with open(filepath) as f:
        preset = yaml.safe_load(f) or {}

    for key in ["dynamic_categories", "profile", "extra_feeds_finance",
                "extra_feeds_politics", "custom_feeds", "custom_tab_name"]:
        strat.config.pop(key, None)

    strat.config["market"] = {
        "tickers": [],
        "intervals": strat.config.get("market", {}).get("intervals", {}),
        "alert_threshold_percent": 5.0,
    }
    strat.config["news"] = {
        "timelimit": "w",
        "career": {"root": "kuwait", "keywords": [], "queries": []},
        "finance": {"root": "kuwait", "keywords": [], "queries": []},
        "regional": {"root": "regional", "keywords": [], "queries": []},
        "tech_trends": {"root": "global", "keywords": [], "queries": []},
        "rss_feeds": [],
    }
    strat.config.update(preset)
    strat.config.pop("security", None)

    # API keys
    search_cfg = strat.config.setdefault("search", {})
    if os.environ.get("SERPER_API_KEY"):
        search_cfg["serper_api_key"] = os.environ["SERPER_API_KEY"]
    if os.environ.get("GOOGLE_API_KEY"):
        search_cfg["google_api_key"] = os.environ["GOOGLE_API_KEY"]
    if os.environ.get("GOOGLE_CSE_ID"):
        search_cfg["google_cx"] = os.environ["GOOGLE_CSE_ID"]
    search_cfg["provider"] = "serper"

    # Scoring config
    scoring = strat.config.setdefault("scoring", {})
    scoring.setdefault("model", "stratos-scorer-v2")
    scoring.setdefault("wizard_model", "qwen3:14b")
    scoring.setdefault("ollama_host", "http://localhost:11434")
    scoring.setdefault("filter_below", 5.0)
    scoring.setdefault("fallback_score", 3.0)
    scoring.setdefault("critical_min", 9.0)
    scoring.setdefault("high_min", 7.0)
    scoring.setdefault("medium_min", 5.0)
    scoring.setdefault("timeout", {
        "rolling_window": 20, "seed_avg": 10,
        "fast_buffer": 30, "fast_minimum": 45,
        "slow_multiplier": 3, "slow_buffer": 60, "avg_cap": 60
    })

    sys_cfg = strat.config.setdefault("system", {})
    sys_cfg["max_news_items"] = 100
    cache_cfg = strat.config.setdefault("cache", {})
    cache_cfg["news_ttl_seconds"] = 0

    strat.active_profile = name
    strat._profile_configs[name] = copy.deepcopy(strat.config)

    ctx_hash = get_context_hash(
        pdef["profile"]["role"],
        pdef["profile"]["context"],
        pdef["profile"]["location"]
    )

    # Run scan
    t_start = time.time()
    output = strat.run_scan()
    scan_time = time.time() - t_start

    # Use wait_for_briefing() (BUG-2 fix test)
    logger.info("  Using wait_for_briefing(30)...")
    briefing_captured = strat.wait_for_briefing(timeout=30)
    logger.info(f"  wait_for_briefing returned: {briefing_captured}")

    # Re-read output file for patched briefing
    briefing = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            patched = json.load(f)
        briefing = patched.get("briefing", {})
        output["briefing"] = briefing

    news = output.get("news", [])
    scores = [a.get("score", 0) for a in news]

    # ── Contamination Check (BUG-1 fix test) ──
    contamination = {"leaked_count": 0, "leaked_articles": []}
    if prev_hash and prev_name:
        for article in news:
            rbp = article.get("retained_by_profile", "")
            if rbp and rbp != ctx_hash:
                contamination["leaked_count"] += 1
                contamination["leaked_articles"].append({
                    "title": article.get("title", "")[:60],
                    "score": article.get("score", 0),
                    "retained_by_profile": rbp,
                    "expected_hash": ctx_hash,
                    "category": article.get("category", ""),
                })

    # ── Generic Keyword Check (S3 fix test — P6 only) ──
    generic_routing = {"routed_to_llm": 0, "still_hardcoded": 0, "articles": []}
    if profile_num == 6:
        for article in news:
            reason = article.get("score_reason", "")
            score = article.get("score", 0)
            if "Generic keywords only, needs LLM" in reason or "Generic entry-level keywords, needs LLM" in reason:
                generic_routing["routed_to_llm"] += 1
                generic_routing["articles"].append({
                    "title": article.get("title", "")[:60],
                    "score": score,
                    "reason": reason[:100],
                })
            # Check for "water" in reason with high score (potential false positive surviving)
            if "water" in reason.lower() and score >= 8.0:
                generic_routing["still_hardcoded"] += 1
                generic_routing["articles"].append({
                    "title": article.get("title", "")[:60],
                    "score": score,
                    "reason": reason[:100],
                    "concern": "high-scoring water article"
                })

    # ── All articles tagged check (BUG-1 completeness) ──
    untagged = [a.get("title", "")[:50] for a in news if not a.get("retained_by_profile")]

    results = {
        "profile_num": profile_num,
        "name": name,
        "role": pdef["profile"]["role"],
        "context_hash": ctx_hash,
        "total_articles": len(news),
        "scan_time": round(scan_time, 1),
        "scores_mean": round(statistics.mean(scores), 2) if scores else 0,
        "scores_above_7": sum(1 for s in scores if s >= 7.0),
        "scores_below_5": sum(1 for s in scores if s < 5.0),
        "retained_count": sum(1 for a in news if a.get("retained")),
        "briefing_captured": bool(briefing.get("sections") or briefing.get("summary") or briefing.get("text")),
        "wait_for_briefing_returned": briefing_captured,
        "untagged_articles": len(untagged),
        "contamination": contamination,
        "generic_routing": generic_routing if profile_num == 6 else None,
        "top_5": [{
            "title": a.get("title", "")[:60],
            "score": a.get("score", 0),
            "reason": a.get("score_reason", "")[:80],
            "category": a.get("category", ""),
            "retained_by_profile": a.get("retained_by_profile", ""),
        } for a in sorted(news, key=lambda x: x.get("score", 0), reverse=True)[:5]],
    }

    # Save per-profile output
    out_path = DIAGNOSTIC_DIR / f"postfix_p{profile_num}_output.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Cleanup profile YAML
    if filepath.exists():
        filepath.unlink()

    return results, ctx_hash, name


def main():
    logger.info("=" * 60)
    logger.info("POST-FIX VALIDATION — 3 Adversarial Profiles")
    logger.info("=" * 60)

    # Check Serper credits
    credits_before = check_serper_credits()
    logger.info(f"Serper credits before: {credits_before}")
    if isinstance(credits_before, (int, float)) and credits_before < 1500:
        logger.error(f"Credits below 1500 ({credits_before}). Aborting.")
        return

    # Delete output file for clean slate
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        logger.info("Deleted output/news_data.json for clean slate")

    # Backup config
    config_backup = BACKEND / "config.yaml.postfix_backup"
    shutil.copy2(BACKEND / "config.yaml", config_backup)

    all_results = []
    prev_hash = None
    prev_name = None

    profile_order = [6, 7, 10]

    try:
        for pnum in profile_order:
            # Budget check before each profile
            credits = check_serper_credits()
            if isinstance(credits, (int, float)) and credits < 1500:
                logger.warning(f"Credits dropped below 1500 ({credits}). Stopping.")
                break

            results, ctx_hash, name = run_profile(pnum, prev_hash, prev_name)
            all_results.append(results)
            prev_hash = ctx_hash
            prev_name = name

            logger.info(f"\n--- P{pnum} Summary ---")
            logger.info(f"  Articles: {results['total_articles']}, Mean: {results['scores_mean']}")
            logger.info(f"  Briefing captured: {results['briefing_captured']}")
            logger.info(f"  Untagged articles: {results['untagged_articles']}")
            logger.info(f"  Contamination: {results['contamination']['leaked_count']} leaked")
            if results.get('generic_routing'):
                gr = results['generic_routing']
                logger.info(f"  Generic routing: {gr['routed_to_llm']} to LLM, {gr['still_hardcoded']} still hardcoded")

    finally:
        # Restore config
        shutil.copy2(config_backup, BACKEND / "config.yaml")
        config_backup.unlink()
        logger.info("Restored config.yaml")

        # Cleanup test profile YAMLs
        for pnum in profile_order:
            pdef = PROFILES[pnum]
            fp = BACKEND / "profiles" / f"{pdef['name']}.yaml"
            if fp.exists():
                fp.unlink()

    credits_after = check_serper_credits()

    # ── Final Report ──
    logger.info("\n" + "=" * 60)
    logger.info("POST-FIX VALIDATION REPORT")
    logger.info("=" * 60)

    all_pass = True

    # Test 1: BUG-1 — No contamination
    logger.info("\n[TEST 1] BUG-1 FIX — Cross-Profile Contamination")
    for r in all_results:
        leaked = r["contamination"]["leaked_count"]
        status = "PASS" if leaked == 0 else "FAIL"
        if leaked > 0:
            all_pass = False
        logger.info(f"  P{r['profile_num']} ({r['name']}): {status} — {leaked} leaked articles")
        if leaked > 0:
            for la in r["contamination"]["leaked_articles"][:3]:
                logger.info(f"    LEAKED: {la['title']} (score={la['score']}, hash={la['retained_by_profile']})")

    # Test 2: BUG-1 — All articles tagged
    logger.info("\n[TEST 2] BUG-1 FIX — All Articles Tagged with context_hash")
    for r in all_results:
        untagged = r["untagged_articles"]
        status = "PASS" if untagged == 0 else "FAIL"
        if untagged > 0:
            all_pass = False
        logger.info(f"  P{r['profile_num']}: {status} — {untagged} untagged articles")

    # Test 3: BUG-2 — Briefing capture
    logger.info("\n[TEST 3] BUG-2 FIX — wait_for_briefing() Captures Briefing")
    for r in all_results:
        captured = r["briefing_captured"]
        wait_ok = r["wait_for_briefing_returned"]
        status = "PASS" if captured and wait_ok else "FAIL"
        if not (captured and wait_ok):
            all_pass = False
        logger.info(f"  P{r['profile_num']}: {status} — briefing={'YES' if captured else 'NO'}, wait_for_briefing={wait_ok}")

    # Test 4: S3 — Generic keyword routing (P6 only)
    logger.info("\n[TEST 4] S3 FIX — Generic Keyword Routing to DoRA Model (P6)")
    p6 = next((r for r in all_results if r["profile_num"] == 6), None)
    if p6 and p6.get("generic_routing"):
        gr = p6["generic_routing"]
        logger.info(f"  Routed to LLM: {gr['routed_to_llm']} articles")
        logger.info(f"  Still hardcoded high: {gr['still_hardcoded']} articles")
        if gr["still_hardcoded"] > 0:
            all_pass = False
            logger.info("  FAIL — Some generic-keyword articles still scored ≥8.0")
            for a in gr["articles"]:
                if a.get("concern"):
                    logger.info(f"    CONCERN: {a['title']} (score={a['score']}, reason={a['reason']})")
        else:
            logger.info("  PASS — No geographic false positives with generic keywords")
        for a in gr["articles"]:
            if not a.get("concern"):
                logger.info(f"    LLM-routed: {a['title'][:50]} → score={a['score']}")
    elif p6:
        logger.info("  INFO — No generic keyword matches detected (may be expected)")

    # Credits summary
    logger.info(f"\n[CREDITS] Before: {credits_before}, After: {credits_after}")

    # Overall verdict
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    logger.info(f"{'='*60}")

    # Save full results
    report_path = DIAGNOSTIC_DIR / "postfix_validation_results.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "profiles_tested": [6, 7, 10],
            "credits_before": credits_before,
            "credits_after": credits_after,
            "all_pass": all_pass,
            "results": all_results,
        }, f, indent=2)
    logger.info(f"\nFull results saved to: {report_path}")


if __name__ == "__main__":
    main()
