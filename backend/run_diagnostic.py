#!/usr/bin/env python3
"""
StratOS Pipeline Diagnostic Loop — Multi-Profile Stress Test
Runs 10 test profiles (5 diverse + 5 adversarial) to identify scoring,
fetching, and categorization issues across different user contexts.

DIAGNOSTIC ONLY — no code changes.
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
logger = logging.getLogger("DIAGNOSTIC")

DIAGNOSTIC_DIR = BACKEND / "diagnostic"
DIAGNOSTIC_DIR.mkdir(exist_ok=True)

# ─── Profile Definitions ────────────────────────────────────────────────

PROFILES = {
    # ── Part A: Diverse Profiles ──
    1: {
        "name": "Diag_P1_Nurse",
        "profile": {
            "role": "Registered Nurse",
            "location": "Lubbock, Texas, USA",
            "context": "Looking for travel nursing contracts and continuing education. Interested in rural healthcare policy.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "travnurs",
                "label": "Travel Nursing Contracts",
                "icon": "heart",
                "items": ["travel nursing agencies", "Aya Healthcare", "Cross Country Nurses", "AMN Healthcare"],
                "enabled": True,
                "scorer_type": "career",
                "root": "global"
            },
            {
                "id": "rurhlth",
                "label": "Rural Healthcare Policy",
                "icon": "shield",
                "items": ["rural hospital funding", "telehealth expansion", "nurse practitioner scope", "Medicare rural"],
                "enabled": True,
                "scorer_type": "regional",
                "root": "global"
            },
            {
                "id": "nursced",
                "label": "Nursing Continuing Education",
                "icon": "award",
                "items": ["ANCC certification", "nursing CEU", "clinical nurse specialist", "BSN to DNP programs"],
                "enabled": True,
                "scorer_type": "auto",
                "root": "global"
            }
        ],
        "news": {"timelimit": "w", "career": {"root": "global", "keywords": [], "queries": []},
                 "finance": {"root": "global", "keywords": [], "queries": []},
                 "regional": {"root": "global", "keywords": [], "queries": []},
                 "tech_trends": {"root": "global", "keywords": [], "queries": []}, "rss_feeds": []},
        "market": {"tickers": [{"symbol": "HCA", "name": "HCA Healthcare"}], "intervals": {}, "alert_threshold_percent": 5.0}
    },
    2: {
        "name": "Diag_P2_Quant",
        "profile": {
            "role": "Quantitative Analyst at a hedge fund",
            "location": "London, United Kingdom",
            "context": "Developing algorithmic trading strategies. Tracking regulatory changes (MiFID II, Basel). Interested in alternative data sources.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "algotr",
                "label": "Algorithmic Trading Strategies",
                "icon": "cpu",
                "items": ["high-frequency trading", "statistical arbitrage", "market microstructure", "order flow analysis"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "global"
            },
            {
                "id": "finreg",
                "label": "Financial Regulation",
                "icon": "gavel",
                "items": ["MiFID II", "Basel III", "FCA regulation", "ESMA guidelines"],
                "enabled": True,
                "scorer_type": "regional",
                "root": "global"
            },
            {
                "id": "altdata",
                "label": "Alternative Data Sources",
                "icon": "target",
                "items": ["satellite imagery finance", "NLP sentiment trading", "web scraping alpha", "quant research"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "global"
            }
        ],
        "news": {"timelimit": "w", "career": {"root": "global", "keywords": [], "queries": []},
                 "finance": {"root": "global", "keywords": [], "queries": []},
                 "regional": {"root": "global", "keywords": [], "queries": []},
                 "tech_trends": {"root": "global", "keywords": [], "queries": []}, "rss_feeds": []},
        "market": {"tickers": [{"symbol": "^FTSE", "name": "FTSE 100"}, {"symbol": "GS", "name": "Goldman Sachs"}], "intervals": {}, "alert_threshold_percent": 5.0}
    },
    3: {
        "name": "Diag_P3_MarineBio",
        "profile": {
            "role": "PhD student in Marine Biology",
            "location": "Tokyo, Japan",
            "context": "Researching coral reef bleaching and ocean acidification. Looking for postdoc positions and conference CFPs.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "corres",
                "label": "Coral Reef & Ocean Research",
                "icon": "cpu",
                "items": ["coral bleaching", "ocean acidification", "marine biodiversity", "reef restoration"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "global"
            },
            {
                "id": "maricar",
                "label": "Marine Science Careers",
                "icon": "building-2",
                "items": ["postdoc marine biology", "JAMSTEC", "NOAA research", "marine science fellowship"],
                "enabled": True,
                "scorer_type": "career",
                "root": "global"
            },
            {
                "id": "confcfp",
                "label": "Conferences & CFPs",
                "icon": "target",
                "items": ["ICRS coral symposium", "ocean sciences meeting", "marine biology conference 2026"],
                "enabled": True,
                "scorer_type": "auto",
                "root": "global"
            }
        ],
        "news": {"timelimit": "w", "career": {"root": "global", "keywords": [], "queries": []},
                 "finance": {"root": "global", "keywords": [], "queries": []},
                 "regional": {"root": "global", "keywords": [], "queries": []},
                 "tech_trends": {"root": "global", "keywords": [], "queries": []}, "rss_feeds": []},
        "market": {"tickers": [], "intervals": {}, "alert_threshold_percent": 5.0}
    },
    4: {
        "name": "Diag_P4_CyberDubai",
        "profile": {
            "role": "Senior Cybersecurity Consultant",
            "location": "Dubai, UAE",
            "context": "Penetration testing and SOC operations for government clients. Tracking regional threat landscape and compliance (NESA, CSA).",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "cybthr",
                "label": "Cybersecurity Threats & Defense",
                "icon": "shield",
                "items": ["zero-day exploits", "ransomware defense", "APT detection", "SOC automation"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "global"
            },
            {
                "id": "gcccyb",
                "label": "GCC Cybersecurity Compliance",
                "icon": "gavel",
                "items": ["NESA UAE", "CSA cybersecurity", "Dubai DIFC data protection", "Saudi NCA standards"],
                "enabled": True,
                "scorer_type": "regional",
                "root": "regional"
            },
            {
                "id": "cybcar",
                "label": "Cybersecurity Careers UAE",
                "icon": "award",
                "items": ["CISSP certification", "CEH training", "cybersecurity jobs Dubai", "GIAC certification"],
                "enabled": True,
                "scorer_type": "career",
                "root": "regional"
            }
        ],
        "news": {"timelimit": "w", "career": {"root": "regional", "keywords": [], "queries": []},
                 "finance": {"root": "regional", "keywords": [], "queries": []},
                 "regional": {"root": "regional", "keywords": [], "queries": []},
                 "tech_trends": {"root": "global", "keywords": [], "queries": []}, "rss_feeds": []},
        "market": {"tickers": [], "intervals": {}, "alert_threshold_percent": 5.0}
    },
    5: {
        "name": "Diag_P5_Teacher",
        "profile": {
            "role": "High school Mathematics Teacher",
            "location": "Sao Paulo, Brazil",
            "context": "Preparing students for university entrance exams. Interested in EdTech tools and curriculum development.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "edtech",
                "label": "EdTech Tools & Platforms",
                "icon": "cpu",
                "items": ["Khan Academy", "Desmos math", "GeoGebra", "AI tutoring platforms"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "global"
            },
            {
                "id": "mathcur",
                "label": "Mathematics Curriculum",
                "icon": "target",
                "items": ["ENEM exam preparation", "math olympiad Brazil", "curriculum BNCC", "vestibular math"],
                "enabled": True,
                "scorer_type": "auto",
                "root": "global"
            },
            {
                "id": "teachpd",
                "label": "Teacher Professional Development",
                "icon": "award",
                "items": ["teacher training workshops", "education conferences Brazil", "STEM teaching methods"],
                "enabled": True,
                "scorer_type": "career",
                "root": "global"
            }
        ],
        "news": {"timelimit": "w", "career": {"root": "global", "keywords": [], "queries": []},
                 "finance": {"root": "global", "keywords": [], "queries": []},
                 "regional": {"root": "global", "keywords": [], "queries": []},
                 "tech_trends": {"root": "global", "keywords": [], "queries": []}, "rss_feeds": []},
        "market": {"tickers": [], "intervals": {}, "alert_threshold_percent": 5.0}
    },

    # ── Part B: Adversarial "Near-Miss" Profiles ──
    6: {
        "name": "Diag_P6_ElecTech",
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
        "name": "Diag_P7_PetroJourno",
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
    8: {
        "name": "Diag_P8_KOCIT",
        "profile": {
            "role": "IT Support Specialist at Kuwait Oil Company",
            "location": "Ahmadi, Kuwait",
            "context": "Managing helpdesk tickets, deploying endpoint security, maintaining SAP access for field teams. Studying for CompTIA Security+ certification.",
            "interests": []
        },
        "dynamic_categories": [
            {
                "id": "kocit",
                "label": "KOC IT & Digital",
                "icon": "building-2",
                "items": ["KOC", "Kuwait Oil Company digital", "KOC IT department", "KOC SAP"],
                "enabled": True,
                "scorer_type": "career",
                "root": "kuwait"
            },
            {
                "id": "itsup",
                "label": "IT Support & Security",
                "icon": "shield",
                "items": ["endpoint security", "helpdesk automation", "ITIL best practices", "Microsoft 365 admin"],
                "enabled": True,
                "scorer_type": "tech",
                "root": "global"
            },
            {
                "id": "itcert",
                "label": "IT Certifications",
                "icon": "award",
                "items": ["CompTIA Security+", "CompTIA A+", "CCNA certification", "Azure fundamentals"],
                "enabled": True,
                "scorer_type": "auto",
                "root": "global"
            }
        ],
        "news": {"timelimit": "w", "career": {"root": "kuwait", "keywords": [], "queries": []},
                 "finance": {"root": "kuwait", "keywords": [], "queries": []},
                 "regional": {"root": "regional", "keywords": [], "queries": []},
                 "tech_trends": {"root": "global", "keywords": [], "queries": []}, "rss_feeds": []},
        "market": {"tickers": [], "intervals": {}, "alert_threshold_percent": 5.0}
    },
    9: {
        "name": "Diag_P9_MechSABIC",
        "profile": {
            "role": "Mechanical Engineer at SABIC",
            "location": "Jubail, Saudi Arabia",
            "context": "Working on heat exchanger design and plant maintenance optimization. Interested in ASME certifications and Saudi Vision 2030 industrial projects.",
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
                "label": "Mechanical Certifications",
                "icon": "award",
                "items": ["ASME certification", "PE mechanical exam", "API 510 inspector", "Six Sigma manufacturing"],
                "enabled": True,
                "scorer_type": "auto",
                "root": "global"
            }
        ],
        "news": {"timelimit": "w", "career": {"root": "regional", "keywords": [], "queries": []},
                 "finance": {"root": "regional", "keywords": [], "queries": []},
                 "regional": {"root": "regional", "keywords": [], "queries": []},
                 "tech_trends": {"root": "global", "keywords": [], "queries": []}, "rss_feeds": []},
        "market": {"tickers": [], "intervals": {}, "alert_threshold_percent": 5.0}
    },
    10: {
        "name": "Diag_P10_CSUK",
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

# Default feed settings (all off to minimize noise)
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


def create_profile_yaml(profile_num: int) -> Path:
    """Create a profile YAML file for the given profile definition."""
    pdef = PROFILES[profile_num]
    name = pdef["name"]

    yaml_data = {
        "profile": {**pdef["profile"], "name": name, "avatar": f"P{profile_num}", "email": f"diag{profile_num}@test.local"},
        "security": {"pin_hash": PIN_HASH, "devices": ["diagnostic_device"]},
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

    logger.info(f"Created profile YAML: {filepath}")
    return filepath


def get_context_hash(role, context, location):
    """Compute the context hash (same as StratOS._get_context_hash)."""
    import re
    raw = f"{role}|{context}|{location}"
    norm = re.sub(r'\s+', ' ', raw.strip().lower())
    return hashlib.sha256(norm.encode()).hexdigest()[:12]


def run_profile_scan(profile_num: int, strat) -> dict:
    """Load a profile into StratOS and run a scan. Returns results dict."""
    pdef = PROFILES[profile_num]
    name = pdef["name"]

    logger.info(f"\n{'='*60}")
    logger.info(f"PROFILE {profile_num}: {name}")
    logger.info(f"  Role: {pdef['profile']['role']}")
    logger.info(f"  Location: {pdef['profile']['location']}")
    logger.info(f"{'='*60}")

    # Load the profile into StratOS by manipulating config directly
    filepath = BACKEND / "profiles" / f"{name}.yaml"
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

    # Preserve search API keys (load .env manually like main.py does)
    env_path = BACKEND / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    os.environ[key.strip()] = value.strip()

    search_cfg = strat.config.setdefault("search", {})
    if os.environ.get("SERPER_API_KEY"):
        search_cfg["serper_api_key"] = os.environ["SERPER_API_KEY"]
    if os.environ.get("GOOGLE_API_KEY"):
        search_cfg["google_api_key"] = os.environ["GOOGLE_API_KEY"]
    if os.environ.get("GOOGLE_CSE_ID"):
        search_cfg["google_cx"] = os.environ["GOOGLE_CSE_ID"]
    search_cfg["provider"] = "serper"

    # Ensure scoring config is present
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

    # Cache config
    cache_cfg = strat.config.setdefault("cache", {})
    cache_cfg["news_ttl_seconds"] = 0  # Force fresh fetch
    cache_cfg["market_ttl_seconds"] = 60

    # Set active profile
    strat.active_profile = name
    strat._profile_configs[name] = copy.deepcopy(strat.config)

    # Clear any cached news from previous profile
    if hasattr(strat, 'news_fetcher') and hasattr(strat.news_fetcher, '_cache'):
        strat.news_fetcher._cache = None
        strat.news_fetcher._cache_time = None

    # Clear previous snapshot to force full scoring
    if hasattr(strat, '_prev_snapshot'):
        strat._prev_snapshot = None
    if hasattr(strat, '_retained_snapshot'):
        strat._retained_snapshot = None
    if hasattr(strat, '_snapshot_context_hash'):
        strat._snapshot_context_hash = None

    # Run the scan
    t_start = time.time()
    try:
        output = strat.run_scan()
        t_end = time.time()
        elapsed = t_end - t_start

        # Save output
        output_path = DIAGNOSTIC_DIR / f"profile_{profile_num}_output.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        # Save profile YAML copy
        profile_copy = DIAGNOSTIC_DIR / f"profile_{profile_num}_config.yaml"
        shutil.copy2(filepath, profile_copy)

        # Compute context hash
        ctx_hash = get_context_hash(
            pdef["profile"]["role"],
            pdef["profile"]["context"],
            pdef["profile"]["location"]
        )

        # Analyze results
        news = output.get("news", [])
        scores = [item.get("score", 0) for item in news if item.get("score") is not None]

        results = {
            "profile_num": profile_num,
            "name": name,
            "role": pdef["profile"]["role"],
            "location": pdef["profile"]["location"],
            "context": pdef["profile"]["context"],
            "context_hash": ctx_hash,
            "categories": [{"id": c["id"], "label": c["label"], "items": c["items"]} for c in pdef["dynamic_categories"]],
            "total_articles": len(news),
            "scan_time_seconds": round(elapsed, 1),
            "scores": scores,
            "score_stats": {},
            "category_counts": {},
            "source_distribution": {},
            "top_10": [],
            "bottom_10": [],
            "articles": news,
            "briefing": output.get("briefing", {}),
            "meta": output.get("meta", {}),
        }

        if scores:
            results["score_stats"] = {
                "mean": round(statistics.mean(scores), 2),
                "median": round(statistics.median(scores), 2),
                "stdev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0,
                "min": round(min(scores), 1),
                "max": round(max(scores), 1),
                "above_7": sum(1 for s in scores if s >= 7.0),
                "below_5": sum(1 for s in scores if s < 5.0),
                "pct_above_7": round(sum(1 for s in scores if s >= 7.0) / len(scores) * 100, 1),
                "pct_below_5": round(sum(1 for s in scores if s < 5.0) / len(scores) * 100, 1),
            }

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
        } for a in sorted_news[:10]]

        above_threshold = [a for a in sorted_news if a.get("score", 0) >= 5.0]
        results["bottom_10"] = [{
            "title": a.get("title", ""),
            "score": a.get("score", 0),
            "reason": a.get("score_reason", ""),
            "category": a.get("category", ""),
            "url": a.get("url", ""),
        } for a in above_threshold[-10:]] if above_threshold else []

        logger.info(f"Profile {profile_num} complete: {len(news)} articles, "
                    f"mean={results['score_stats'].get('mean', 0)}, "
                    f"time={elapsed:.1f}s")

        return results

    except Exception as e:
        t_end = time.time()
        logger.error(f"Profile {profile_num} scan FAILED: {e}", exc_info=True)
        return {
            "profile_num": profile_num,
            "name": name,
            "role": pdef["profile"]["role"],
            "location": pdef["profile"]["location"],
            "error": str(e),
            "scan_time_seconds": round(t_end - t_start, 1),
            "total_articles": 0,
            "articles": [],
            "scores": [],
            "score_stats": {},
            "category_counts": {},
            "source_distribution": {},
            "top_10": [],
            "bottom_10": [],
        }


def check_serper_credits():
    """Check remaining Serper credits."""
    try:
        # Load .env manually
        env_path = BACKEND / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, _, value = line.partition('=')
                        os.environ[key.strip()] = value.strip()

        from fetchers.serper_search import SerperQueryTracker
        tracker = SerperQueryTracker()
        status = tracker.get_status()
        return status.get("remaining", "unknown")
    except Exception as e:
        logger.warning(f"Could not check Serper credits: {e}")
    return "unknown"


def analyze_kuwait_contamination(all_results):
    """Check for Kuwait/Ahmad-specific content appearing in non-Kuwait profiles."""
    kuwait_keywords = ["kuwait", "koc", "knpc", "kipic", "equate", "kpc",
                       "auk", "american university", "petroleum engineer",
                       "seisnetics", "kuwait university"]

    contamination = {}
    for pnum, result in all_results.items():
        if pnum >= 6:  # Adversarial profiles handle Kuwait differently
            continue
        if "kuwait" in result.get("location", "").lower():
            continue  # Skip Kuwait-based profiles

        contaminated = []
        for article in result.get("articles", []):
            title_lower = (article.get("title", "") or "").lower()
            for kw in kuwait_keywords:
                if kw in title_lower and article.get("score", 0) >= 5.0:
                    contaminated.append({
                        "title": article.get("title", ""),
                        "score": article.get("score", 0),
                        "keyword": kw,
                    })
                    break

        if contaminated:
            contamination[pnum] = contaminated

    return contamination


def analyze_adversarial_discrimination(all_results):
    """Cross-compare adversarial profiles 6-10 for discrimination quality."""
    # Collect all unique article URLs from adversarial profiles
    url_map = {}  # url -> {profile_num: {score, title}}

    for pnum in range(6, 11):
        result = all_results.get(pnum, {})
        for article in result.get("articles", []):
            url = article.get("url", "")
            if not url:
                continue
            if url not in url_map:
                url_map[url] = {"title": article.get("title", "")}
            url_map[url][pnum] = {
                "score": article.get("score", 0),
                "reason": article.get("score_reason", ""),
            }

    # Find articles that appear in multiple adversarial profiles
    shared_articles = []
    for url, data in url_map.items():
        profile_scores = {k: v for k, v in data.items() if isinstance(k, int)}
        if len(profile_scores) >= 2:
            high_count = sum(1 for v in profile_scores.values() if v["score"] >= 7.0)
            shared_articles.append({
                "title": data["title"],
                "url": url,
                "profiles": profile_scores,
                "high_across_multiple": high_count >= 2,
            })

    # Sort by number of profiles (most shared first)
    shared_articles.sort(key=lambda x: len(x["profiles"]), reverse=True)

    return shared_articles[:20]  # Top 20 most shared


def main():
    """Run the full diagnostic loop."""
    logger.info("=" * 70)
    logger.info("STRATOS PIPELINE DIAGNOSTIC LOOP")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 70)

    # Check Serper credits
    initial_credits = check_serper_credits()
    logger.info(f"Serper credits at start: {initial_credits}")

    # Create all profile YAMLs first
    logger.info("\n--- Creating test profile YAMLs ---")
    for pnum in PROFILES:
        create_profile_yaml(pnum)

    # Initialize StratOS
    logger.info("\n--- Initializing StratOS ---")
    from main import StratOS
    strat = StratOS("config.yaml")

    # Run all profiles
    all_results = {}
    created_profiles = []

    for pnum in sorted(PROFILES.keys()):
        # Check Serper credits before each profile
        credits = check_serper_credits()
        logger.info(f"\nSerper credits before profile {pnum}: {credits}")

        try:
            credits_int = int(credits) if str(credits).isdigit() else 9999
        except (ValueError, TypeError):
            credits_int = 9999

        if credits_int < 1600:
            logger.warning(f"Serper credits below 1800 ({credits}). Stopping diagnostic to preserve credits.")
            break

        result = run_profile_scan(pnum, strat)
        all_results[pnum] = result
        created_profiles.append(PROFILES[pnum]["name"])

        # Brief pause between profiles to avoid rate limiting
        time.sleep(2)

    # Final credits
    final_credits = check_serper_credits()
    logger.info(f"\nSerper credits at end: {final_credits}")

    # Save all results
    results_path = DIAGNOSTIC_DIR / "all_results.json"
    # Create a serializable version (remove full articles to save space)
    serializable = {}
    for pnum, result in all_results.items():
        r = {k: v for k, v in result.items() if k != "articles"}
        serializable[str(pnum)] = r

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    # Run cross-profile analysis
    contamination = analyze_kuwait_contamination(all_results)
    shared_adversarial = analyze_adversarial_discrimination(all_results)

    # Save analysis
    analysis_path = DIAGNOSTIC_DIR / "cross_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump({
            "kuwait_contamination": {str(k): v for k, v in contamination.items()},
            "adversarial_shared_articles": shared_adversarial,
            "initial_credits": str(initial_credits),
            "final_credits": str(final_credits),
        }, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info(f"Results saved to: {DIAGNOSTIC_DIR}")
    logger.info(f"Profiles tested: {len(all_results)}")
    logger.info(f"Credits used: {initial_credits} -> {final_credits}")
    logger.info(f"{'='*70}")

    return all_results, contamination, shared_adversarial


if __name__ == "__main__":
    all_results, contamination, shared = main()
