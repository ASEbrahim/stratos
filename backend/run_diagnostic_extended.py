#!/usr/bin/env python3
"""
StratOS Extended Pipeline Diagnostic — Phase 2
Targets: briefing capture, RSS feeds, pipeline edge cases, non-English content,
student vs senior discrimination, and isolated (no-retention) scoring quality.

KEY IMPROVEMENT: Creates a FRESH StratOS instance per profile to eliminate
cross-profile retention contamination (BUG-1 from Phase 1).
Also waits for deferred briefing thread completion.
"""

import copy
import hashlib
import json
import logging
import os
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
logger = logging.getLogger("DIAG-EXT")

DIAGNOSTIC_DIR = BACKEND / "diagnostic"
DIAGNOSTIC_DIR.mkdir(exist_ok=True)

# Load .env once
env_path = BACKEND / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, value = line.partition('=')
                os.environ[key.strip()] = value.strip()


# ─── Profile Definitions ────────────────────────────────────────────────

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
    "fox_world": False, "guardian": False, "dw": False,
    "france24": False, "gn_kuwait": False, "arab_times": False,
    "times_kuwait": False, "gn_mideast": False, "alarabiya": False,
    "middleeasteye": False, "newarab": False, "scmp": False,
    "nikkei_asia": False,
}
PIN_HASH = hashlib.sha256("0000".encode()).hexdigest()


PROFILES = {
    # P11: RSS-Heavy Energy Profile — Tests RSS feed integration
    # Enable multiple energy/finance RSS feeds to see if RSS items get scored properly
    11: {
        "name": "Diag_P11_RSSEnergy",
        "profile": {
            "role": "Energy Market Analyst",
            "location": "Houston, Texas, USA",
            "context": "Monitoring crude oil markets, refinery operations, and LNG exports. Tracking energy policy changes and OPEC+ decisions.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "oilmkt",
                "label": "Crude Oil & LNG Markets",
                "icon": "target",
                "items": ["crude oil price", "LNG export terminal", "refinery capacity", "Brent crude"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "global"
            },
            {
                "id": "enpol",
                "label": "Energy Policy & Regulation",
                "icon": "gavel",
                "items": ["FERC regulation", "EPA emissions", "DOE energy policy", "carbon tax"],
                "enabled": True,
                "scorer_type": "regional",
                "root": "global"
            },
        ],
        "news": {
            "timelimit": "w",
            "career": {"root": "global", "keywords": [], "queries": []},
            "finance": {"root": "global", "keywords": [], "queries": []},
            "regional": {"root": "global", "keywords": [], "queries": []},
            "tech_trends": {"root": "global", "keywords": [], "queries": []},
            "rss_feeds": []
        },
        "market": {"tickers": [{"symbol": "CL=F", "name": "Crude Oil"}, {"symbol": "NG=F", "name": "Natural Gas"}], "intervals": {}, "alert_threshold_percent": 5.0},
        # Enable energy-related RSS feeds
        "extra_feeds_finance_overrides": {
            "oilprice": True, "rigzone": True, "bloomberg": True, "cnbc_top": True
        },
        "extra_feeds_politics_overrides": {
            "bbc_world": True
        },
    },

    # P12: Arabic/Kuwait Minimal Profile — Tests non-English content handling
    # Kuwait-based with Arabic-leaning keywords, minimal categories
    12: {
        "name": "Diag_P12_ArabicKW",
        "profile": {
            "role": "Government Relations Specialist",
            "location": "Kuwait City, Kuwait",
            "context": "Managing government affairs for a Kuwaiti conglomerate. Tracking National Assembly debates, GCC summits, and public sector procurement tenders. Primary news consumption in Arabic.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "kwgov",
                "label": "Kuwait Government & Parliament",
                "icon": "gavel",
                "items": ["Kuwait National Assembly", "مجلس الأمة الكويتي", "Kuwait government tenders", "Kuwait municipal elections"],
                "enabled": True,
                "scorer_type": "regional",
                "root": "kuwait"
            },
            {
                "id": "gccrel",
                "label": "GCC Diplomatic Relations",
                "icon": "target",
                "items": ["GCC summit 2026", "Kuwait-Saudi relations", "Gulf Cooperation Council", "Kuwait foreign ministry"],
                "enabled": True,
                "scorer_type": "regional",
                "root": "regional"
            },
        ],
        "news": {
            "timelimit": "w",
            "career": {"root": "kuwait", "keywords": [], "queries": []},
            "finance": {"root": "kuwait", "keywords": [], "queries": []},
            "regional": {"root": "regional", "keywords": [], "queries": []},
            "tech_trends": {"root": "global", "keywords": [], "queries": []},
            "rss_feeds": []
        },
        "market": {"tickers": [], "intervals": {}, "alert_threshold_percent": 5.0},
        "extra_feeds_finance_overrides": {},
        "extra_feeds_politics_overrides": {
            "gn_kuwait": True, "arab_times": True, "alarabiya": True,
            "gn_mideast": True
        },
    },

    # P13: Ultra-Niche Profile — Tests extreme score distribution (narrow field)
    # Very specific academic field that will produce few relevant articles
    13: {
        "name": "Diag_P13_Niche",
        "profile": {
            "role": "Papyrologist specializing in Ptolemaic demotic scripts",
            "location": "Oxford, United Kingdom",
            "context": "Researching decipherment of Ptolemaic-era Egyptian demotic papyri. Looking for museum digitization projects, paleography conferences, and Coptic language analysis tools.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "papyr",
                "label": "Papyrology & Ancient Scripts",
                "icon": "cpu",
                "items": ["papyrus digitization", "demotic script analysis", "Ptolemaic Egypt research", "ancient manuscript OCR"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "global"
            },
            {
                "id": "museum",
                "label": "Museum & Archive Projects",
                "icon": "building-2",
                "items": ["British Museum papyrus", "Bodleian Library manuscripts", "Egyptian Museum Cairo", "papyrology postdoc"],
                "enabled": True,
                "scorer_type": "career",
                "root": "global"
            },
        ],
        "news": {
            "timelimit": "w",
            "career": {"root": "global", "keywords": [], "queries": []},
            "finance": {"root": "global", "keywords": [], "queries": []},
            "regional": {"root": "global", "keywords": [], "queries": []},
            "tech_trends": {"root": "global", "keywords": [], "queries": []},
            "rss_feeds": []
        },
        "market": {"tickers": [], "intervals": {}, "alert_threshold_percent": 5.0},
        "extra_feeds_finance_overrides": {},
        "extra_feeds_politics_overrides": {},
    },

    # P14: MechEng Student — Direct comparison to P9 MechSABIC (student vs senior)
    # Same field (mechanical engineering) but student perspective
    14: {
        "name": "Diag_P14_MechStudent",
        "profile": {
            "role": "Mechanical Engineering Student (Senior year)",
            "location": "Jubail, Saudi Arabia",
            "context": "Completing final year at Jubail Industrial College. Interested in co-op positions at SABIC or Saudi Aramco. Studying for FE exam.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "sabicj",
                "label": "SABIC & Saudi Petrochemicals",
                "icon": "building-2",
                "items": ["SABIC", "Saudi Aramco downstream", "Jubail Industrial City", "Saudi Vision 2030 industry"],
                "enabled": True,
                "scorer_type": "career",
                "root": "regional"
            },
            {
                "id": "mecheng",
                "label": "Mechanical Engineering Tech",
                "icon": "cpu",
                "items": ["heat exchanger design", "plant maintenance optimization", "predictive maintenance", "process piping"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "global"
            },
            {
                "id": "mechcrt",
                "label": "MechEng Student Certifications",
                "icon": "award",
                "items": ["FE exam mechanical", "EIT certification", "SABIC internship program", "Saudi engineering co-op"],
                "enabled": True,
                "scorer_type": "career",
                "root": "regional"
            }
        ],
        "news": {
            "timelimit": "w",
            "career": {"root": "regional", "keywords": [], "queries": []},
            "finance": {"root": "regional", "keywords": [], "queries": []},
            "regional": {"root": "regional", "keywords": [], "queries": []},
            "tech_trends": {"root": "global", "keywords": [], "queries": []},
            "rss_feeds": []
        },
        "market": {"tickers": [], "intervals": {}, "alert_threshold_percent": 5.0},
        "extra_feeds_finance_overrides": {},
        "extra_feeds_politics_overrides": {},
    },

    # P15: Single-Item Categories — Tests pipeline with edge-case categories
    # One category with a single item, one completely empty, and one normal
    15: {
        "name": "Diag_P15_EdgeCase",
        "profile": {
            "role": "Freelance Graphic Designer",
            "location": "Lisbon, Portugal",
            "context": "Working on branding projects for tech startups. Looking for design tool updates and freelance contract opportunities.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "destools",
                "label": "Design Tools & Software",
                "icon": "cpu",
                "items": ["Figma updates", "Adobe Creative Cloud", "Canva pro features", "Affinity Designer"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "global"
            },
            {
                "id": "singlecat",
                "label": "Single Item Category",
                "icon": "target",
                "items": ["Lisbon design meetup"],
                "enabled": True,
                "scorer_type": "auto",
                "root": "global"
            },
            {
                "id": "emptycat",
                "label": "Empty Category",
                "icon": "shield",
                "items": [],
                "enabled": True,
                "scorer_type": "auto",
                "root": "global"
            },
        ],
        "news": {
            "timelimit": "w",
            "career": {"root": "global", "keywords": [], "queries": []},
            "finance": {"root": "global", "keywords": [], "queries": []},
            "regional": {"root": "global", "keywords": [], "queries": []},
            "tech_trends": {"root": "global", "keywords": [], "queries": []},
            "rss_feeds": []
        },
        "market": {"tickers": [], "intervals": {}, "alert_threshold_percent": 5.0},
        "extra_feeds_finance_overrides": {},
        "extra_feeds_politics_overrides": {},
    },

    # P16: Broad Generalist — Tests score discrimination with overlapping interests
    # Interests so broad they could match anything — tests whether scorer gives meaningful scores
    16: {
        "name": "Diag_P16_Generalist",
        "profile": {
            "role": "CEO of a mid-size technology consultancy",
            "location": "Singapore",
            "context": "Managing a 200-person firm across APAC. Interested in AI trends, business strategy, talent acquisition, regional politics, and investment opportunities. Needs to stay informed on everything.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "aiml",
                "label": "AI & Machine Learning",
                "icon": "cpu",
                "items": ["artificial intelligence business", "ChatGPT enterprise", "AI regulation", "machine learning trends"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "global"
            },
            {
                "id": "bizstrat",
                "label": "Business Strategy & Investment",
                "icon": "target",
                "items": ["venture capital APAC", "startup funding Singapore", "management consulting trends", "M&A technology"],
                "enabled": True,
                "scorer_type": "auto",
                "root": "global"
            },
            {
                "id": "apacpol",
                "label": "APAC Politics & Economy",
                "icon": "gavel",
                "items": ["Singapore economy", "ASEAN trade", "China tech policy", "India digital regulation"],
                "enabled": True,
                "scorer_type": "regional",
                "root": "global"
            },
        ],
        "news": {
            "timelimit": "w",
            "career": {"root": "global", "keywords": [], "queries": []},
            "finance": {"root": "global", "keywords": [], "queries": []},
            "regional": {"root": "global", "keywords": [], "queries": []},
            "tech_trends": {"root": "global", "keywords": [], "queries": []},
            "rss_feeds": []
        },
        "market": {"tickers": [{"symbol": "^STI", "name": "STI Singapore"}], "intervals": {}, "alert_threshold_percent": 5.0},
        "extra_feeds_finance_overrides": {
            "bloomberg": True, "cnbc_top": True
        },
        "extra_feeds_politics_overrides": {
            "scmp": True, "nikkei_asia": True
        },
    },
}


def get_context_hash(role, context, location):
    """Compute the context hash (same as StratOS._get_context_hash)."""
    import re
    raw = f"{role}|{context}|{location}"
    norm = re.sub(r'\s+', ' ', raw.strip().lower())
    return hashlib.sha256(norm.encode()).hexdigest()[:12]


def create_profile_yaml(profile_num):
    """Create a profile YAML file for the given profile definition."""
    pdef = PROFILES[profile_num]
    name = pdef["name"]

    # Build feed overrides
    feeds_fin = DEFAULT_FEEDS_FINANCE.copy()
    feeds_fin.update(pdef.get("extra_feeds_finance_overrides", {}))
    feeds_pol = DEFAULT_FEEDS_POLITICS.copy()
    feeds_pol.update(pdef.get("extra_feeds_politics_overrides", {}))

    yaml_data = {
        "profile": {**pdef["profile"], "name": name, "avatar": f"P{profile_num}", "email": f"diag{profile_num}@test.local"},
        "security": {"pin_hash": PIN_HASH, "devices": ["diagnostic_device"]},
        "dynamic_categories": pdef["dynamic_categories"],
        "market": pdef["market"],
        "news": pdef["news"],
        "extra_feeds_finance": feeds_fin,
        "extra_feeds_politics": feeds_pol,
        "custom_feeds": [],
        "custom_tab_name": "Custom",
    }

    filepath = BACKEND / "profiles" / f"{name}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Created profile YAML: {filepath}")
    return filepath


def check_serper_credits():
    """Check remaining Serper credits."""
    try:
        import requests
        r = requests.get('https://google.serper.dev/account',
                         headers={'X-API-KEY': os.environ.get('SERPER_API_KEY', '')},
                         timeout=10)
        return r.json().get("balance", "unknown")
    except Exception as e:
        logger.warning(f"Could not check Serper credits: {e}")
    return "unknown"


def run_isolated_profile_scan(profile_num):
    """
    Run a profile scan in a FRESH StratOS instance.
    This eliminates cross-profile retention contamination.
    Also waits for deferred briefing completion.
    """
    import threading
    pdef = PROFILES[profile_num]
    name = pdef["name"]

    logger.info(f"\n{'='*60}")
    logger.info(f"PROFILE {profile_num}: {name}")
    logger.info(f"  Role: {pdef['profile']['role']}")
    logger.info(f"  Location: {pdef['profile']['location']}")
    logger.info(f"  RSS Finance enabled: {[k for k,v in pdef.get('extra_feeds_finance_overrides',{}).items() if v]}")
    logger.info(f"  RSS Politics enabled: {[k for k,v in pdef.get('extra_feeds_politics_overrides',{}).items() if v]}")
    logger.info(f"{'='*60}")

    # Create profile YAML
    filepath = create_profile_yaml(profile_num)

    # Create a FRESH StratOS instance (no retention from previous profiles)
    from main import StratOS
    strat = StratOS("config.yaml")

    # Load the profile into StratOS
    with open(filepath) as f:
        preset = yaml.safe_load(f) or {}

    # Clear profile-specific fields (same as auth.load_profile_config)
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

    # Apply profile config
    strat.config.update(preset)
    strat.config.pop("security", None)

    # Preserve search API keys
    search_cfg = strat.config.setdefault("search", {})
    if os.environ.get("SERPER_API_KEY"):
        search_cfg["serper_api_key"] = os.environ["SERPER_API_KEY"]
    if os.environ.get("GOOGLE_API_KEY"):
        search_cfg["google_api_key"] = os.environ["GOOGLE_API_KEY"]
    if os.environ.get("GOOGLE_CSE_ID"):
        search_cfg["google_cx"] = os.environ["GOOGLE_CSE_ID"]
    search_cfg["provider"] = "serper"

    # Ensure scoring config
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

    # System config
    sys_cfg = strat.config.setdefault("system", {})
    sys_cfg["max_news_items"] = 100
    sys_cfg["output_file"] = "output/news_data.json"
    sys_cfg["database_file"] = "strat_os.db"

    # Cache config — force fresh fetch
    cache_cfg = strat.config.setdefault("cache", {})
    cache_cfg["news_ttl_seconds"] = 0
    cache_cfg["market_ttl_seconds"] = 60

    # Set active profile
    strat.active_profile = name
    strat._profile_configs[name] = copy.deepcopy(strat.config)

    # Compute context hash
    ctx_hash = get_context_hash(
        pdef["profile"]["role"],
        pdef["profile"]["context"],
        pdef["profile"]["location"]
    )

    # Run the scan
    t_start = time.time()
    pipeline_events = {
        "errors": [],
        "warnings": [],
        "rss_items_count": 0,
        "search_items_count": 0,
        "deferred_items": 0,
        "pass1_scored": 0,
        "pass2_scored": 0,
        "retained_count": 0,
    }

    try:
        output = strat.run_scan()
        t_scan_end = time.time()
        scan_elapsed = t_scan_end - t_start

        # Capture pipeline metrics from scan status
        pipeline_events["pass1_scored"] = strat.scan_status.get("scored", 0)
        pipeline_events["retained_count"] = getattr(strat, '_last_retained_count', 0)

        # Save initial output (pre-briefing)
        output_path = DIAGNOSTIC_DIR / f"profile_{profile_num}_output.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        initial_briefing = output.get("briefing", {})
        logger.info(f"  Initial briefing captured: {'NON-EMPTY' if initial_briefing else 'EMPTY'}")

        # Wait for deferred briefing thread
        logger.info(f"  Waiting 30s for deferred briefing thread...")
        time.sleep(30)

        # Re-read the output file to capture the patched briefing
        briefing_captured = {}
        try:
            output_file = BACKEND / "output" / "news_data.json"
            if output_file.exists():
                with open(output_file) as f:
                    patched_output = json.load(f)
                briefing_captured = patched_output.get("briefing", {})
                logger.info(f"  Post-wait briefing: {'NON-EMPTY' if briefing_captured else 'STILL EMPTY'}")
                if briefing_captured:
                    # Update the saved output with the briefing
                    output["briefing"] = briefing_captured
                    with open(output_path, "w") as f:
                        json.dump(output, f, indent=2)
        except Exception as e:
            logger.warning(f"  Could not re-read output for briefing: {e}")

        t_end = time.time()

        # Save profile YAML copy
        profile_copy = DIAGNOSTIC_DIR / f"profile_{profile_num}_config.yaml"
        shutil.copy2(filepath, profile_copy)

        # Analyze results
        news = output.get("news", [])
        scores = [item.get("score", 0) for item in news if item.get("score") is not None]

        # Check for retained articles (shouldn't be any with fresh instance)
        retained_articles = [a for a in news if a.get("retained")]
        if retained_articles:
            pipeline_events["warnings"].append(
                f"Found {len(retained_articles)} retained articles in isolated instance!"
            )

        # Check for source distribution (RSS vs search)
        rss_articles = [a for a in news if a.get("source", "").lower() in
                        ["rss", "oilprice", "rigzone", "bloomberg", "cnbc", "bbc",
                         "arab times", "alarabiya", "scmp", "nikkei"]]
        pipeline_events["rss_items_count"] = len(rss_articles)

        # Analyze briefing quality
        briefing_analysis = {}
        if briefing_captured:
            briefing_analysis = {
                "has_content": True,
                "critical_count": briefing_captured.get("critical_count", 0),
                "high_count": briefing_captured.get("high_count", 0),
                "sections": list(briefing_captured.keys()),
                "text_length": len(json.dumps(briefing_captured)),
            }
            # Check if briefing references correct role
            briefing_text = json.dumps(briefing_captured).lower()
            role_words = pdef["profile"]["role"].lower().split()
            role_match = sum(1 for w in role_words if w in briefing_text and len(w) > 3)
            briefing_analysis["role_word_matches"] = role_match
            briefing_analysis["role_total_words"] = len([w for w in role_words if len(w) > 3])
        else:
            briefing_analysis = {"has_content": False, "reason": "Briefing still empty after 30s wait"}

        # Build results
        results = {
            "profile_num": profile_num,
            "name": name,
            "role": pdef["profile"]["role"],
            "location": pdef["profile"]["location"],
            "context": pdef["profile"]["context"],
            "context_hash": ctx_hash,
            "categories": [{"id": c["id"], "label": c["label"], "items": c["items"]} for c in pdef["dynamic_categories"]],
            "total_articles": len(news),
            "scan_time_seconds": round(scan_elapsed, 1),
            "total_time_with_briefing_wait": round(t_end - t_start, 1),
            "scores": scores,
            "score_stats": {},
            "score_distribution": {},
            "category_counts": {},
            "source_distribution": {},
            "top_10": [],
            "bottom_10": [],
            "pipeline_events": pipeline_events,
            "briefing_analysis": briefing_analysis,
            "retained_count": len(retained_articles),
            "rss_feeds_enabled": {
                "finance": [k for k, v in pdef.get("extra_feeds_finance_overrides", {}).items() if v],
                "politics": [k for k, v in pdef.get("extra_feeds_politics_overrides", {}).items() if v],
            },
            "meta": output.get("meta", {}),
        }

        if scores:
            results["score_stats"] = {
                "mean": round(statistics.mean(scores), 2),
                "median": round(statistics.median(scores), 2),
                "stdev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0,
                "min": round(min(scores), 1),
                "max": round(max(scores), 1),
                "above_9": sum(1 for s in scores if s >= 9.0),
                "above_7": sum(1 for s in scores if s >= 7.0),
                "between_5_7": sum(1 for s in scores if 5.0 <= s < 7.0),
                "below_5": sum(1 for s in scores if s < 5.0),
                "pct_above_7": round(sum(1 for s in scores if s >= 7.0) / len(scores) * 100, 1),
                "pct_below_5": round(sum(1 for s in scores if s < 5.0) / len(scores) * 100, 1),
            }

            # Score distribution (1-point buckets)
            for s in scores:
                bucket = f"{int(s)}.0-{int(s)}.9"
                results["score_distribution"][bucket] = results["score_distribution"].get(bucket, 0) + 1

        # Category counts
        for item in news:
            cat = item.get("category", "unknown")
            results["category_counts"][cat] = results["category_counts"].get(cat, 0) + 1

        # Source distribution
        for item in news:
            src = item.get("source", "unknown")
            results["source_distribution"][src] = results["source_distribution"].get(src, 0) + 1

        # Top 10 and bottom 10
        sorted_news = sorted(news, key=lambda x: x.get("score", 0), reverse=True)
        results["top_10"] = [{
            "title": a.get("title", ""),
            "score": a.get("score", 0),
            "reason": a.get("score_reason", ""),
            "category": a.get("category", ""),
            "url": a.get("url", ""),
            "source": a.get("source", ""),
            "retained": a.get("retained", False),
        } for a in sorted_news[:10]]

        above_threshold = [a for a in sorted_news if a.get("score", 0) >= 5.0]
        results["bottom_10"] = [{
            "title": a.get("title", ""),
            "score": a.get("score", 0),
            "reason": a.get("score_reason", ""),
            "category": a.get("category", ""),
            "url": a.get("url", ""),
            "source": a.get("source", ""),
        } for a in above_threshold[-10:]] if above_threshold else []

        logger.info(f"Profile {profile_num} complete: {len(news)} articles, "
                    f"mean={results['score_stats'].get('mean', 0)}, "
                    f"time={scan_elapsed:.1f}s, "
                    f"briefing={'YES' if briefing_analysis.get('has_content') else 'NO'}")

        return results

    except Exception as e:
        t_end = time.time()
        logger.error(f"Profile {profile_num} scan FAILED: {e}", exc_info=True)
        pipeline_events["errors"].append(str(e))
        return {
            "profile_num": profile_num,
            "name": name,
            "role": pdef["profile"]["role"],
            "location": pdef["profile"]["location"],
            "error": str(e),
            "scan_time_seconds": round(t_end - t_start, 1),
            "total_articles": 0,
            "scores": [],
            "score_stats": {},
            "score_distribution": {},
            "category_counts": {},
            "source_distribution": {},
            "top_10": [],
            "bottom_10": [],
            "pipeline_events": pipeline_events,
            "briefing_analysis": {"has_content": False, "reason": f"Scan failed: {e}"},
            "retained_count": 0,
        }


def compare_student_vs_senior(all_results):
    """Compare P14 (MechEng Student) with P9 data (if available)."""
    p14 = all_results.get(14, {})

    # Load P9 results from Phase 1
    p9_path = DIAGNOSTIC_DIR / "profile_9_output.json"
    p9_articles = []
    if p9_path.exists():
        with open(p9_path) as f:
            p9_data = json.load(f)
            p9_articles = p9_data.get("news", [])

    if not p9_articles or not p14.get("top_10"):
        return {"status": "incomplete", "reason": "Missing P9 or P14 data"}

    # Find shared articles
    p14_urls = {a.get("url", "") for a in p14.get("articles", []) if a.get("url")}
    p9_url_scores = {a.get("url", ""): {"score": a.get("score", 0), "reason": a.get("score_reason", "")}
                     for a in p9_articles if a.get("url")}

    shared = []
    for url in p14_urls:
        if url in p9_url_scores:
            p14_article = next((a for a in p14.get("articles", []) if a.get("url") == url), {})
            shared.append({
                "url": url,
                "title": p14_article.get("title", ""),
                "p14_score": p14_article.get("score", 0),
                "p14_reason": p14_article.get("score_reason", ""),
                "p9_score": p9_url_scores[url]["score"],
                "p9_reason": p9_url_scores[url]["reason"],
                "delta": round(p14_article.get("score", 0) - p9_url_scores[url]["score"], 1),
            })

    shared.sort(key=lambda x: abs(x["delta"]), reverse=True)

    # Check if score reasons mention "student" for P14
    p14_student_mentions = 0
    for a in p14.get("top_10", []):
        reason = a.get("reason", "").lower()
        if "student" in reason or "intern" in reason or "co-op" in reason:
            p14_student_mentions += 1

    return {
        "shared_articles": shared[:15],
        "shared_count": len(shared),
        "p14_student_mentions_in_top10": p14_student_mentions,
        "p14_total": p14.get("total_articles", 0),
        "p9_total": len(p9_articles),
    }


def main():
    """Run the extended diagnostic."""
    logger.info("=" * 70)
    logger.info("STRATOS EXTENDED PIPELINE DIAGNOSTIC — PHASE 2")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("KEY: Fresh StratOS per profile, briefing capture, RSS, edge cases")
    logger.info("=" * 70)

    initial_credits = check_serper_credits()
    logger.info(f"Serper credits at start: {initial_credits}")

    all_results = {}

    for pnum in sorted(PROFILES.keys()):
        # Check credits
        credits = check_serper_credits()
        logger.info(f"\nSerper credits before profile {pnum}: {credits}")

        try:
            credits_int = int(credits) if str(credits).replace('.', '').isdigit() else 9999
        except (ValueError, TypeError):
            credits_int = 9999

        if credits_int < 1500:
            logger.warning(f"Serper credits below 1500 ({credits}). Stopping.")
            break

        result = run_isolated_profile_scan(pnum)
        all_results[pnum] = result

        # Brief pause between profiles
        time.sleep(3)

    final_credits = check_serper_credits()
    logger.info(f"\nSerper credits at end: {final_credits}")

    # Student vs Senior comparison
    student_vs_senior = compare_student_vs_senior(all_results)

    # Save results (without full articles for summary file)
    results_path = DIAGNOSTIC_DIR / "extended_results.json"
    serializable = {}
    for pnum, result in all_results.items():
        r = {k: v for k, v in result.items() if k != "articles"}
        serializable[str(pnum)] = r

    with open(results_path, "w") as f:
        json.dump({
            "profiles": serializable,
            "student_vs_senior": student_vs_senior,
            "initial_credits": str(initial_credits),
            "final_credits": str(final_credits),
            "credits_used": int(initial_credits) - int(final_credits) if str(initial_credits).isdigit() and str(final_credits).isdigit() else "unknown",
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Clean up test profile YAMLs
    for pnum in PROFILES:
        pname = PROFILES[pnum]["name"]
        yaml_path = BACKEND / "profiles" / f"{pname}.yaml"
        if yaml_path.exists():
            yaml_path.unlink()
            logger.info(f"Cleaned up: {yaml_path}")

    # Restore config.yaml from git
    os.system("cd /home/ahmad/Downloads/StratOS/StratOS1 && git checkout backend/config.yaml 2>/dev/null")

    logger.info(f"\n{'='*70}")
    logger.info("EXTENDED DIAGNOSTIC COMPLETE")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Profiles tested: {len(all_results)}")
    logger.info(f"Credits used: {initial_credits} -> {final_credits}")
    logger.info(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    main()
