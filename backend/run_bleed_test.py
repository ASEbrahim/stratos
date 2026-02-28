#!/usr/bin/env python3
"""
StratOS Profile Isolation Bleed Test (D027)
============================================

Tests cross-profile data contamination by running two maximally distinct profiles
and checking for data bleed across 6 contamination vectors:

  1. Context hash isolation — different profiles produce different hashes
  2. Score reuse protection — context hash change forces full rescore
  3. Article retention filter — retained articles don't leak across profiles
  4. Config switching — profile-specific fields properly cleared/replaced
  5. Keyword/category isolation — different profiles produce different article sets
  6. Database contamination — DB writes from Profile A don't appear in Profile B output

Profiles:
  A: Marine Biologist in Reykjavik (Arctic ecosystems, fishery management)
  B: Fashion Designer in Milan (sustainable textiles, luxury innovation)

Usage:
  python3 run_bleed_test.py              # Full test (requires Ollama + internet)
  python3 run_bleed_test.py --dry-run    # Config/hash isolation only (no scan)
"""

import argparse
import copy
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from textwrap import indent

# ── Setup ────────────────────────────────────────────────────────────────────

os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.WARNING,  # Suppress StratOS debug noise
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
# Let the test's own logger be visible
test_logger = logging.getLogger("BLEED_TEST")
test_logger.setLevel(logging.INFO)

# ── Test Profile Definitions ────────────────────────────────────────────────

PROFILE_A = {
    "profile": {
        "role": "Marine Biologist",
        "location": "Reykjavik",
        "context": "Marine conservation researcher in Iceland focused on Arctic ecosystems, "
                   "fishery management, ocean acidification, and climate impacts on marine biodiversity. "
                   "Tracks North Atlantic fish stock surveys, ICES scientific advisory reports, "
                   "and Icelandic Marine Research Institute publications.",
        "interests": [],
        "name": "BleedTestA",
        "avatar": "MB",
    },
    "security": {"pin_hash": hashlib.sha256(b"0000").hexdigest()},
    "market": {
        "tickers": [
            {"symbol": "AKER.OL", "name": "Aker ASA", "category": "custom"},
            {"symbol": "MHG.OL", "name": "Mowi (salmon)", "category": "custom"},
        ],
        "intervals": {},
        "alert_threshold_percent": 5.0,
    },
    "news": {
        "timelimit": "w",
        "career": {"root": "global", "keywords": ["marine biology jobs", "oceanography positions"], "queries": []},
        "finance": {"root": "global", "keywords": [], "queries": []},
        "regional": {"root": "regional", "keywords": ["Iceland fisheries", "Arctic policy"], "queries": []},
        "tech_trends": {"root": "global", "keywords": ["ocean monitoring tech"], "queries": []},
        "rss_feeds": [],
    },
    "dynamic_categories": [
        {
            "id": "arctic_research",
            "label": "Arctic Marine Research",
            "icon": "anchor",
            "items": [
                "marine biology", "Arctic ecosystems", "oceanography", "fish populations",
                "ocean acidification", "deep sea research", "whale migration",
                "phytoplankton", "coral reef", "marine conservation",
                "fishery management", "ICES advisory", "North Atlantic",
            ],
            "enabled": True,
            "scorer_type": "tech",
            "root": "global",
        },
        {
            "id": "iceland_env",
            "label": "Iceland Environmental Policy",
            "icon": "globe",
            "items": [
                "Iceland fishing quota", "Icelandic Marine Research Institute",
                "Arctic Council", "North Atlantic Salmon Conservation",
                "Iceland climate policy", "geothermal energy Iceland",
            ],
            "enabled": True,
            "scorer_type": "regional",
            "root": "regional",
        },
    ],
    "extra_feeds_finance": {},
    "extra_feeds_politics": {},
    "custom_feeds": [],
    "custom_tab_name": "Custom",
}

PROFILE_B = {
    "profile": {
        "role": "Fashion Designer",
        "location": "Milan",
        "context": "Fashion designer in Milan specializing in sustainable textiles, luxury fashion "
                   "innovation, and haute couture. Tracks Milan Fashion Week, Vogue Business reports, "
                   "emerging designer showcases, and Italian textile manufacturing trends.",
        "interests": [],
        "name": "BleedTestB",
        "avatar": "FD",
    },
    "security": {"pin_hash": hashlib.sha256(b"0000").hexdigest()},
    "market": {
        "tickers": [
            {"symbol": "KER.PA", "name": "Kering (Gucci)", "category": "custom"},
            {"symbol": "MC.PA", "name": "LVMH", "category": "custom"},
        ],
        "intervals": {},
        "alert_threshold_percent": 5.0,
    },
    "news": {
        "timelimit": "w",
        "career": {"root": "global", "keywords": ["fashion designer jobs", "textile design roles"], "queries": []},
        "finance": {"root": "global", "keywords": [], "queries": []},
        "regional": {"root": "regional", "keywords": ["Milan fashion", "Italian luxury"], "queries": []},
        "tech_trends": {"root": "global", "keywords": ["sustainable textiles"], "queries": []},
        "rss_feeds": [],
    },
    "dynamic_categories": [
        {
            "id": "fashion_trends",
            "label": "Fashion & Design Trends",
            "icon": "palette",
            "items": [
                "sustainable fashion", "luxury brands", "textile innovation",
                "designer collections", "Milan Fashion Week", "haute couture",
                "fast fashion", "fashion technology", "wearable tech fashion",
                "Vogue", "fashion sustainability", "circular fashion",
            ],
            "enabled": True,
            "scorer_type": "tech",
            "root": "global",
        },
        {
            "id": "italian_luxury",
            "label": "Italian Luxury & Textiles",
            "icon": "star",
            "items": [
                "Italian textile manufacturing", "Prada", "Gucci", "Armani",
                "Versace", "Dolce Gabbana", "Valentino", "Fendi",
                "Italian fashion industry", "Milan design week",
            ],
            "enabled": True,
            "scorer_type": "regional",
            "root": "regional",
        },
    ],
    "extra_feeds_finance": {},
    "extra_feeds_politics": {},
    "custom_feeds": [],
    "custom_tab_name": "Custom",
}

# Keywords that are maximally distinct — used for contamination detection
PROFILE_A_MARKERS = {
    "marine", "ocean", "arctic", "fish", "whale", "coral", "oceanography",
    "phytoplankton", "fishery", "iceland", "reykjavik", "salmon", "ices",
    "sea", "aquatic", "biodiversity", "ecosystem",
}
PROFILE_B_MARKERS = {
    "fashion", "textile", "luxury", "couture", "designer", "vogue", "milan",
    "prada", "gucci", "armani", "versace", "fendi", "valentino", "runway",
    "apparel", "garment", "fabric",
}


# ── Utility Functions ────────────────────────────────────────────────────────

def compute_context_hash(profile_data):
    """Replicate StratOS._get_context_hash() for verification."""
    p = profile_data.get("profile", {})
    parts = []
    for text in [p.get("role", ""), p.get("context", ""), p.get("location", "")]:
        normalized = " ".join(text.lower().split()) if text else ""
        parts.append(normalized)
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:12]


def marker_score(text, markers):
    """Count how many marker keywords appear in a text string."""
    if not text:
        return 0
    text_lower = text.lower()
    return sum(1 for m in markers if m in text_lower)


def classify_article(article):
    """Classify an article as belonging to Profile A, B, both, or neither."""
    title = article.get("title", "")
    summary = article.get("summary", "")
    combined = f"{title} {summary}"
    a_score = marker_score(combined, PROFILE_A_MARKERS)
    b_score = marker_score(combined, PROFILE_B_MARKERS)
    if a_score > 0 and b_score == 0:
        return "A"
    elif b_score > 0 and a_score == 0:
        return "B"
    elif a_score > 0 and b_score > 0:
        return "BOTH"
    return "NEUTRAL"


# ── Test Result Tracking ─────────────────────────────────────────────────────

class TestResults:
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def check(self, name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        self.tests.append({"name": name, "status": status, "detail": detail})
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        icon = "\033[92m[PASS]\033[0m" if passed else "\033[91m[FAIL]\033[0m"
        print(f"  {icon} {name}")
        if detail:
            print(f"         {detail}")

    def warn(self, name, detail=""):
        self.tests.append({"name": name, "status": "WARN", "detail": detail})
        self.warnings += 1
        print(f"  \033[93m[WARN]\033[0m {name}")
        if detail:
            print(f"         {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        if self.failed == 0:
            print(f"\033[92mAll {total} tests passed\033[0m ({self.warnings} warnings)")
        else:
            print(f"\033[91m{self.failed}/{total} tests FAILED\033[0m ({self.warnings} warnings)")
        print(f"{'='*60}")
        return self.failed == 0


# ── Test Phases ──────────────────────────────────────────────────────────────

def phase_0_setup(results):
    """Create test profiles and verify they exist."""
    print("\n\033[1m── Phase 0: Setup ──\033[0m")

    profiles_dir = Path("profiles")
    profiles_dir.mkdir(exist_ok=True)

    # Write test profiles
    for name, data in [("BleedTestA", PROFILE_A), ("BleedTestB", PROFILE_B)]:
        filepath = profiles_dir / f"{name}.yaml"
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        results.check(f"Profile {name} created", filepath.exists())

    # Verify context hashes are different
    hash_a = compute_context_hash(PROFILE_A)
    hash_b = compute_context_hash(PROFILE_B)
    results.check(
        "Context hashes are distinct",
        hash_a != hash_b,
        f"A={hash_a}, B={hash_b}"
    )
    return hash_a, hash_b


def phase_1_config_isolation(results):
    """Test that profile switching properly replaces config."""
    print("\n\033[1m── Phase 1: Config Isolation ──\033[0m")

    from main import StratOS
    from auth import AuthManager

    # Use test database and output
    test_db = Path("test_bleed.db")
    test_output = Path("output/bleed_test_a.json")

    # Temporarily override config
    strat = StratOS(config_path="config.yaml")
    strat.db = __import__('database').Database(str(test_db))
    strat.output_file = test_output
    test_output.parent.mkdir(parents=True, exist_ok=True)

    auth = AuthManager("config.yaml")

    # Load Profile A
    ok_a = auth.load_profile_config("BleedTestA", strat)
    results.check("Profile A loads successfully", ok_a)

    # Capture Profile A config
    config_a = copy.deepcopy(strat.config)
    role_a = config_a.get("profile", {}).get("role", "")
    location_a = config_a.get("profile", {}).get("location", "")
    cats_a = [c["id"] for c in config_a.get("dynamic_categories", [])]
    tickers_a = [t["symbol"] for t in config_a.get("market", {}).get("tickers", [])]

    results.check("Profile A role is 'Marine Biologist'", role_a == "Marine Biologist", f"got: {role_a}")
    results.check("Profile A location is 'Reykjavik'", location_a == "Reykjavik", f"got: {location_a}")
    results.check("Profile A has arctic categories", "arctic_research" in cats_a, f"categories: {cats_a}")
    results.check("Profile A has marine tickers", "AKER.OL" in tickers_a, f"tickers: {tickers_a}")

    # Switch to Profile B
    ok_b = auth.load_profile_config("BleedTestB", strat)
    results.check("Profile B loads successfully", ok_b)

    config_b = copy.deepcopy(strat.config)
    role_b = config_b.get("profile", {}).get("role", "")
    location_b = config_b.get("profile", {}).get("location", "")
    cats_b = [c["id"] for c in config_b.get("dynamic_categories", [])]
    tickers_b = [t["symbol"] for t in config_b.get("market", {}).get("tickers", [])]

    results.check("Profile B role is 'Fashion Designer'", role_b == "Fashion Designer", f"got: {role_b}")
    results.check("Profile B location is 'Milan'", location_b == "Milan", f"got: {location_b}")
    results.check("Profile B has fashion categories", "fashion_trends" in cats_b, f"categories: {cats_b}")
    results.check("Profile B has luxury tickers", "KER.PA" in tickers_b, f"tickers: {tickers_b}")

    # Verify no Profile A contamination in Profile B
    results.check(
        "No arctic categories in Profile B",
        "arctic_research" not in cats_b,
        f"B categories: {cats_b}"
    )
    results.check(
        "No marine tickers in Profile B",
        "AKER.OL" not in tickers_b,
        f"B tickers: {tickers_b}"
    )
    results.check(
        "Profile B role != Profile A role",
        role_b != role_a,
        f"A={role_a}, B={role_b}"
    )

    # Verify ensure_profile works correctly
    strat.ensure_profile("BleedTestA")
    results.check(
        "ensure_profile switches back to A",
        strat.config.get("profile", {}).get("role") == "Marine Biologist",
        f"got: {strat.config.get('profile', {}).get('role')}"
    )

    strat.ensure_profile("BleedTestB")
    results.check(
        "ensure_profile switches to B",
        strat.config.get("profile", {}).get("role") == "Fashion Designer",
        f"got: {strat.config.get('profile', {}).get('role')}"
    )

    # Verify API keys preserved across switch
    search = strat.config.get("search", {})
    has_serper = bool(search.get("serper_api_key", ""))
    has_google = bool(search.get("google_api_key", ""))
    results.check(
        "API keys preserved after profile switch",
        has_serper or has_google,
        f"serper={'yes' if has_serper else 'no'}, google={'yes' if has_google else 'no'}"
    )

    # Clean up test DB
    strat.db.conn.close()
    if test_db.exists():
        test_db.unlink()

    return strat


def phase_2_scan_and_compare(results, hash_a, hash_b):
    """Run actual scans with both profiles and compare outputs."""
    print("\n\033[1m── Phase 2: Full Scan Comparison ──\033[0m")
    print("  (This requires Ollama + internet and may take several minutes per scan)")

    from main import StratOS
    from auth import AuthManager
    import database as db_module

    test_db_path = Path("test_bleed.db")
    output_a_path = Path("output/bleed_test_a.json")
    output_b_path = Path("output/bleed_test_b.json")

    # Patch the database singleton to use test DB (avoids locking production DB)
    old_instance = getattr(db_module, '_db_instance', None)
    test_db = db_module.Database(str(test_db_path))
    db_module._db_instance = test_db

    # Create StratOS — it will get our test DB via get_database()
    strat = StratOS(config_path="config.yaml")
    strat.output_file = output_a_path
    strat.output_file.parent.mkdir(parents=True, exist_ok=True)

    auth = AuthManager("config.yaml")

    # ── Scan A: Marine Biologist ──
    print("\n  \033[1mScan A: Marine Biologist (Reykjavik)\033[0m")
    auth.load_profile_config("BleedTestA", strat)
    strat.ensure_profile("BleedTestA")
    strat.output_file = output_a_path
    strat.output_file.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        output_a = strat.run_scan()
        elapsed_a = time.time() - t0
        results.check(f"Scan A completed in {elapsed_a:.0f}s", True)
    except Exception as e:
        results.check(f"Scan A completed", False, f"Error: {e}")
        return None, None

    # Also save a copy of the output to a separate file
    with open(output_a_path, "w") as f:
        json.dump(output_a, f, indent=2)

    articles_a = output_a.get("news", [])
    meta_a = output_a.get("meta", {})
    results.check(
        f"Scan A produced articles",
        len(articles_a) > 0,
        f"{len(articles_a)} articles"
    )
    results.check(
        "Scan A context hash matches expected",
        meta_a.get("context_hash") == hash_a,
        f"expected={hash_a}, got={meta_a.get('context_hash')}"
    )

    # ── Scan B: Fashion Designer ──
    print(f"\n  \033[1mScan B: Fashion Designer (Milan)\033[0m")
    auth.load_profile_config("BleedTestB", strat)
    strat.ensure_profile("BleedTestB")
    strat.output_file = output_b_path

    t0 = time.time()
    try:
        output_b = strat.run_scan()
        elapsed_b = time.time() - t0
        results.check(f"Scan B completed in {elapsed_b:.0f}s", True)
    except Exception as e:
        results.check(f"Scan B completed", False, f"Error: {e}")
        return output_a, None

    with open(output_b_path, "w") as f:
        json.dump(output_b, f, indent=2)

    articles_b = output_b.get("news", [])
    meta_b = output_b.get("meta", {})
    results.check(
        f"Scan B produced articles",
        len(articles_b) > 0,
        f"{len(articles_b)} articles"
    )
    results.check(
        "Scan B context hash matches expected",
        meta_b.get("context_hash") == hash_b,
        f"expected={hash_b}, got={meta_b.get('context_hash')}"
    )

    # Clean up — restore DB singleton
    test_db.conn.close()
    db_module._db_instance = old_instance

    return output_a, output_b


def phase_3_contamination_analysis(results, output_a, output_b, hash_a, hash_b):
    """Analyze scan outputs for cross-profile contamination."""
    print("\n\033[1m── Phase 3: Contamination Analysis ──\033[0m")

    articles_a = output_a.get("news", [])
    articles_b = output_b.get("news", [])

    # ── 3.1: Context hash isolation ──
    results.check(
        "Context hashes differ between scans",
        output_a["meta"]["context_hash"] != output_b["meta"]["context_hash"],
    )

    # ── 3.2: retained_by_profile tagging ──
    a_tags = set(a.get("retained_by_profile", "") for a in articles_a)
    b_tags = set(a.get("retained_by_profile", "") for a in articles_b)

    results.check(
        "All Scan A articles tagged with hash_a",
        all(a.get("retained_by_profile") == hash_a for a in articles_a),
        f"tags found: {a_tags}"
    )
    results.check(
        "All Scan B articles tagged with hash_b",
        all(a.get("retained_by_profile") == hash_b for a in articles_b),
        f"tags found: {b_tags}"
    )
    results.check(
        "No hash_a tags in Scan B output",
        hash_a not in b_tags,
        f"B tags: {b_tags}"
    )

    # ── 3.3: Retained article leak ──
    retained_in_b = [a for a in articles_b if a.get("retained")]
    leaked_retained = [
        a for a in retained_in_b
        if a.get("retained_by_profile") == hash_a
    ]
    results.check(
        "No retained articles from Profile A in Profile B",
        len(leaked_retained) == 0,
        f"{len(leaked_retained)} leaked retained articles"
    )

    # ── 3.4: Content overlap analysis ──
    urls_a = set(a["url"] for a in articles_a if a.get("url"))
    urls_b = set(a["url"] for a in articles_b if a.get("url"))
    overlap = urls_a & urls_b
    overlap_pct = len(overlap) / max(len(urls_a | urls_b), 1) * 100

    if overlap_pct > 50:
        results.warn(
            f"High URL overlap: {len(overlap)}/{len(urls_a | urls_b)} ({overlap_pct:.0f}%)",
            "Profiles share too many articles — categories may not be distinct enough"
        )
    else:
        results.check(
            f"Reasonable URL overlap: {len(overlap)} shared ({overlap_pct:.0f}%)",
            True,
            f"A={len(urls_a)}, B={len(urls_b)}, shared={len(overlap)}"
        )

    # ── 3.5: Keyword contamination ──
    # Check if Profile B output contains marine biology content scored highly
    b_high_score = [a for a in articles_b if a.get("score", 0) >= 7.0]
    a_content_in_b_high = []
    for article in b_high_score:
        cls = classify_article(article)
        if cls == "A":
            a_content_in_b_high.append(article)

    if a_content_in_b_high:
        results.warn(
            f"Profile A content scored high in Profile B: {len(a_content_in_b_high)} articles",
            f"Articles: {[a['title'][:60] for a in a_content_in_b_high[:3]]}"
        )
    else:
        results.check(
            "No Profile A content scored highly in Profile B",
            True,
            f"Checked {len(b_high_score)} high-scoring articles in B"
        )

    # Reverse check: Profile A content in Profile B low-scored
    a_high_score = [a for a in articles_a if a.get("score", 0) >= 7.0]
    b_content_in_a_high = []
    for article in a_high_score:
        cls = classify_article(article)
        if cls == "B":
            b_content_in_a_high.append(article)

    if b_content_in_a_high:
        results.warn(
            f"Profile B content scored high in Profile A: {len(b_content_in_a_high)} articles",
            f"Articles: {[a['title'][:60] for a in b_content_in_a_high[:3]]}"
        )
    else:
        results.check(
            "No Profile B content scored highly in Profile A",
            True,
            f"Checked {len(a_high_score)} high-scoring articles in A"
        )

    # ── 3.6: Score distribution comparison ──
    def score_dist(articles):
        critical = sum(1 for a in articles if a.get("score", 0) >= 9.0)
        high = sum(1 for a in articles if 7.0 <= a.get("score", 0) < 9.0)
        medium = sum(1 for a in articles if 5.0 < a.get("score", 0) < 7.0)
        noise = sum(1 for a in articles if a.get("score", 0) <= 5.0)
        return {"critical": critical, "high": high, "medium": medium, "noise": noise}

    dist_a = score_dist(articles_a)
    dist_b = score_dist(articles_b)
    print(f"\n  Score distributions:")
    print(f"    Profile A: {dist_a}")
    print(f"    Profile B: {dist_b}")

    # ── 3.7: Category distribution ──
    cats_a = {}
    for a in articles_a:
        cat = a.get("category", "unknown")
        cats_a[cat] = cats_a.get(cat, 0) + 1

    cats_b = {}
    for a in articles_b:
        cat = a.get("category", "unknown")
        cats_b[cat] = cats_b.get(cat, 0) + 1

    print(f"\n  Category distributions:")
    print(f"    Profile A: {dict(sorted(cats_a.items(), key=lambda x: -x[1]))}")
    print(f"    Profile B: {dict(sorted(cats_b.items(), key=lambda x: -x[1]))}")

    # Check that Profile A has marine categories and Profile B has fashion categories
    a_has_marine = any(k in ["arctic_research", "iceland_env"] for k in cats_a)
    b_has_fashion = any(k in ["fashion_trends", "italian_luxury"] for k in cats_b)

    results.check(
        "Profile A articles categorized as marine/arctic",
        a_has_marine,
        f"A categories: {list(cats_a.keys())}"
    )
    results.check(
        "Profile B articles categorized as fashion/luxury",
        b_has_fashion,
        f"B categories: {list(cats_b.keys())}"
    )

    # ── 3.8: Score reuse protection (context hash gate) ──
    # After Scan A wrote output, Scan B should NOT have reused any scores
    # (because context_hash changed). We verify by checking that
    # B articles have their own score_reason (not copied from A).
    if overlap:
        shared_articles = []
        a_by_url = {a["url"]: a for a in articles_a}
        for article in articles_b:
            url = article.get("url")
            if url in a_by_url:
                a_article = a_by_url[url]
                shared_articles.append({
                    "url": url,
                    "title": article.get("title", "")[:60],
                    "score_a": a_article.get("score", 0),
                    "score_b": article.get("score", 0),
                    "reason_a": a_article.get("score_reason", "")[:80],
                    "reason_b": article.get("score_reason", "")[:80],
                })

        # If scores are identical for shared articles, that's suspicious
        identical_scores = sum(1 for s in shared_articles if s["score_a"] == s["score_b"])
        total_shared = len(shared_articles)

        if total_shared > 0:
            pct_identical = identical_scores / total_shared * 100
            if pct_identical > 80:
                results.warn(
                    f"Suspicious: {identical_scores}/{total_shared} shared articles "
                    f"have identical scores ({pct_identical:.0f}%)",
                    "May indicate score reuse despite context hash change"
                )
            else:
                results.check(
                    f"Shared articles scored differently: {total_shared - identical_scores}/{total_shared} differ",
                    True,
                    f"{pct_identical:.0f}% identical (expected low for distinct profiles)"
                )

            # Print sample of shared articles with different scores
            print(f"\n  Shared article samples (first 5):")
            for s in shared_articles[:5]:
                delta = abs(s["score_a"] - s["score_b"])
                print(f"    {s['title']}")
                print(f"      A={s['score_a']:.1f}, B={s['score_b']:.1f} (delta={delta:.1f})")


def phase_4_db_contamination(results):
    """Check the test database for contamination patterns."""
    print("\n\033[1m── Phase 4: Database Contamination Check ──\033[0m")

    import sqlite3

    test_db_path = Path("test_bleed.db")
    if not test_db_path.exists():
        results.warn("Test database not found — skipping DB checks")
        return

    conn = sqlite3.connect(str(test_db_path))
    conn.row_factory = sqlite3.Row

    # Check news_items — all articles share one global table
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM news_items")
    total_items = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(DISTINCT url) as cnt FROM news_items")
    unique_urls = cursor.fetchone()["cnt"]

    results.check(
        f"DB has {total_items} news items ({unique_urls} unique URLs)",
        total_items > 0,
    )

    # Key finding: DB has NO profile_id column — this IS the contamination
    cursor.execute("PRAGMA table_info(news_items)")
    columns = [row["name"] for row in cursor.fetchall()]
    has_profile_id = "profile_id" in columns

    if has_profile_id:
        results.check("news_items has profile_id column", True)
    else:
        results.warn(
            "news_items lacks profile_id column",
            "EXPECTED: This is the contamination vector that profile isolation will fix. "
            "Both profiles write to the same table with no ownership marker."
        )

    # Check scan_log — should have 2 entries
    cursor.execute("SELECT COUNT(*) as cnt FROM scan_log")
    scan_count = cursor.fetchone()["cnt"]
    results.check(
        f"scan_log has {scan_count} entries (expected 2)",
        scan_count >= 2,
    )

    conn.close()


def phase_5_cleanup(results, keep_artifacts=False):
    """Remove test profiles and artifacts."""
    print("\n\033[1m── Phase 5: Cleanup ──\033[0m")

    profiles_dir = Path("profiles")
    cleaned = []

    for name in ["BleedTestA", "BleedTestB"]:
        filepath = profiles_dir / f"{name}.yaml"
        if filepath.exists():
            filepath.unlink()
            cleaned.append(name)

    if not keep_artifacts:
        for path in [Path("test_bleed.db"), Path("test_bleed.db-wal"), Path("test_bleed.db-shm")]:
            if path.exists():
                path.unlink()
                cleaned.append(str(path))

        for path in [Path("output/bleed_test_a.json"), Path("output/bleed_test_b.json")]:
            if path.exists():
                # Don't delete — keep for manual inspection
                pass

    results.check(f"Cleaned up: {', '.join(cleaned)}", len(cleaned) > 0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="StratOS Profile Isolation Bleed Test")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test config/hash isolation only (no Ollama/scan)")
    parser.add_argument("--keep", action="store_true",
                        help="Keep test artifacts for inspection")
    args = parser.parse_args()

    print("=" * 60)
    print("  StratOS Profile Isolation Bleed Test (D027)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        print("  MODE: Dry run (config/hash isolation only)")
    else:
        print("  MODE: Full scan (requires Ollama + internet)")
    print("=" * 60)

    results = TestResults()

    # Phase 0: Setup
    hash_a, hash_b = phase_0_setup(results)

    # Phase 1: Config isolation
    phase_1_config_isolation(results)

    if not args.dry_run:
        # Phase 2: Full scan comparison
        output_a, output_b = phase_2_scan_and_compare(results, hash_a, hash_b)

        if output_a and output_b:
            # Phase 3: Contamination analysis
            phase_3_contamination_analysis(results, output_a, output_b, hash_a, hash_b)

            # Phase 4: DB contamination
            phase_4_db_contamination(results)
        else:
            print("\n  Skipping contamination analysis — scans did not complete")
    else:
        print("\n  \033[90m(Skipping Phase 2-4 in dry-run mode)\033[0m")

    # Phase 5: Cleanup
    phase_5_cleanup(results, keep_artifacts=args.keep)

    # Summary
    all_passed = results.summary()

    # Write results to JSON for later analysis
    report_path = Path("output/bleed_test_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "mode": "dry_run" if args.dry_run else "full",
            "hash_a": hash_a,
            "hash_b": hash_b,
            "passed": results.passed,
            "failed": results.failed,
            "warnings": results.warnings,
            "tests": results.tests,
        }, f, indent=2)
    print(f"\n  Report saved to: {report_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
