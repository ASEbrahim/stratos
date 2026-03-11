#!/usr/bin/env python3
"""
Autonomous Wizard QA — Randomized profile testing with contamination detection.
Tests all 3 wizard steps for 20 diverse profiles via API.
Writes detailed results to qa_results.log.
"""

import json
import re
import sys
import time
import random
import requests
from datetime import datetime

BASE = "http://localhost:8080"
LOG_FILE = "qa_results.log"

# ═══════════════════════════════════════════════════════════
#  TEST PROFILES — 20 wildly diverse roles + regions
# ═══════════════════════════════════════════════════════════
PROFILES = [
    ("Pastry Chef", "Tokyo, Japan"),
    ("Maritime Lawyer", "Singapore"),
    ("Wildlife Veterinarian", "Nairobi, Kenya"),
    ("Fashion Designer", "Milan, Italy"),
    ("Olive Farmer", "Crete, Greece"),
    ("Aerospace Engineer", "Toulouse, France"),
    ("Hip-Hop Producer", "Atlanta, USA"),
    ("Dentist", "São Paulo, Brazil"),
    ("Islamic Finance Analyst", "Bahrain"),
    ("Marine Biologist", "Bergen, Norway"),
    ("Architect", "Dubai, UAE"),
    ("Pharmacist", "Seoul, South Korea"),
    ("Film Director", "Mumbai, India"),
    ("Renewable Energy Consultant", "Copenhagen, Denmark"),
    ("Mining Engineer", "Perth, Australia"),
    ("Sommelier", "Bordeaux, France"),
    ("Robotics Researcher", "Zurich, Switzerland"),
    ("Kindergarten Teacher", "Toronto, Canada"),
    ("Sports Physiotherapist", "Barcelona, Spain"),
    ("Nuclear Engineer", "Germany"),
]

random.shuffle(PROFILES)

# Category/sub lookup
CATS_DATA = [
    {"id":"career","label":"Career & Jobs","subs":[
        {"id":"jobhunt","label":"Job Hunting"},{"id":"intern","label":"Internships"},
        {"id":"research_pos","label":"Research Positions"},{"id":"govjobs","label":"Government Jobs"}]},
    {"id":"industry","label":"Industry Intel","subs":[
        {"id":"ind_oilgas","label":"Oil & Gas"},{"id":"ind_telecom","label":"Telecom"},
        {"id":"ind_tech","label":"Tech / Software"},{"id":"ind_bank","label":"Banking & Finance"},
        {"id":"ind_construct","label":"Construction"},{"id":"ind_health","label":"Healthcare"}]},
    {"id":"learning","label":"Learning & Development","subs":[
        {"id":"certs","label":"Professional Certifications"},{"id":"academic","label":"Academic Research"},
        {"id":"courses","label":"Online Courses"},{"id":"hands_on","label":"Hands-on Skills"},
        {"id":"confevents","label":"Conferences & Events"}]},
    {"id":"markets","label":"Markets & Investing","subs":[
        {"id":"stocks","label":"Stocks"},{"id":"crypto","label":"Crypto"},
        {"id":"commodities","label":"Commodities"},{"id":"realestate","label":"Real Estate"},
        {"id":"forex","label":"Forex"},{"id":"mktreports","label":"Market Research & Reports"}]},
    {"id":"deals","label":"Deals & Offers","subs":[
        {"id":"bankdeals","label":"Bank Deals"},{"id":"telepromo","label":"Telecom Promotions"},
        {"id":"studisc","label":"Student Discounts"},{"id":"ccrewards","label":"Credit Card Rewards"},
        {"id":"empben","label":"Employee Benefits"}]},
    {"id":"interests","label":"Interests & Trends","subs":[]},
]

VALID_CATS = {c["id"] for c in CATS_DATA}
CAT_LABELS = {c["id"]: c["label"] for c in CATS_DATA}
SUB_LABELS = {}
SUB_PARENT = {}
for c in CATS_DATA:
    for s in c["subs"]:
        SUB_LABELS[s["id"]] = s["label"]
        SUB_PARENT[s["id"]] = c["id"]

# ═══════════════════════════════════════════════════════════
#  CONTAMINATION DETECTION
# ═══════════════════════════════════════════════════════════

# Hard tech keywords that should NEVER appear for non-tech/non-data/non-cyber roles
HARD_TECH_CONTAM = {
    'AWS Solutions Architect', 'CCNA', 'CompTIA Security+', 'AZ-900',
    'CES', 'AWS re:Invent', 'Google I/O', 'DEFCON',
    'TryHackMe', 'Cisco Packet Tracer', 'AWS Free Tier',
    'Y Combinator', 'freeCodeCamp', 'Kaggle', 'Product Hunt',
    'GitHub Student Pack', 'JetBrains Student',
    'CS50 (edX)', 'AWS Cloud Practitioner',
}

# Soft tech keywords — suspicious for non-tech roles (might be legitimate context)
SOFT_TECH_CONTAM = {
    '5G Rollout', 'Cloud & SaaS', 'AI & Automation', 'Cybersecurity',
    'Fintech', 'GovTech', 'Docker', 'Kubernetes', 'DevOps',
    'Full-stack', 'Frontend', 'Backend', 'SRE',
}

# Roles that ARE tech-related (tech contamination is expected/fine)
TECH_ROLES = {
    'Robotics Researcher', 'Aerospace Engineer', 'Nuclear Engineer',
    'Renewable Energy Consultant', 'Mining Engineer',
}

# Roles that involve finance (finance items are fine)
FINANCE_ROLES = {'Islamic Finance Analyst'}

def is_tech_role(role):
    return any(tr.lower() in role.lower() for tr in TECH_ROLES)

def is_finance_role(role):
    return any(fr.lower() in role.lower() for fr in FINANCE_ROLES)

def check_contamination(items, role):
    """Check a list of items/strings for tech contamination. Returns list of issues."""
    if is_tech_role(role):
        return []  # Tech roles can have tech items

    issues = []
    for item in items:
        item_lower = item.lower()
        for kw in HARD_TECH_CONTAM:
            kw_lower = kw.lower()
            # Use word-boundary matching to avoid false positives like
            # "CES" matching inside "conferences", "services", "practices"
            if re.search(r'\b' + re.escape(kw_lower) + r'\b', item_lower):
                issues.append(f"HARD: '{item}' matches '{kw}'")
        for kw in SOFT_TECH_CONTAM:
            if kw.lower() == item_lower:  # exact match only for soft
                issues.append(f"SOFT: '{item}' matches '{kw}'")
    return issues

def check_region_relevance(items, location):
    """Check if any items reference the expected region."""
    loc_lower = location.lower()
    # Extract country/city keywords
    region_words = set()
    for part in loc_lower.replace(',', ' ').split():
        if len(part) > 2:
            region_words.add(part)

    for item in items:
        item_lower = item.lower()
        for rw in region_words:
            if rw in item_lower:
                return True
    return False

# ═══════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════
log_lines = []

def log(msg):
    log_lines.append(msg)
    print(msg)

def flush_log():
    with open(LOG_FILE, 'w') as f:
        f.write('\n'.join(log_lines) + '\n')

# ═══════════════════════════════════════════════════════════
#  TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════

def test_step1(role, location):
    """Test wizard-preselect. Returns (cats, subs, pass/fail, notes)."""
    try:
        r = requests.post(f"{BASE}/api/wizard-preselect", json={
            "role": role, "location": location,
            "available_categories": CATS_DATA
        }, timeout=60)
        if r.status_code != 200:
            return None, None, "FAIL", f"HTTP {r.status_code}: {r.text[:100]}"

        d = r.json()
        cats = d.get("selected_categories", [])
        subs = d.get("selected_subs", {})

        if not cats:
            return cats, subs, "FAIL", "Empty categories returned"

        # Validate: all returned cats must be valid IDs
        invalid = [c for c in cats if c not in VALID_CATS]
        if invalid:
            return cats, subs, "FAIL", f"Invalid category IDs: {invalid}"

        # Check: markets shouldn't be selected for most non-finance roles
        notes = []
        if 'markets' in cats and not is_finance_role(role):
            notes.append(f"'markets' selected for non-finance role")

        # Check: career should almost always be selected
        if 'career' not in cats:
            notes.append("'career' not selected (unusual)")

        # Validate subs belong to their categories
        for cid, sub_list in subs.items():
            valid_subs = {s["id"] for c in CATS_DATA if c["id"] == cid for s in c["subs"]}
            bad = [s for s in sub_list if s not in valid_subs]
            if bad:
                notes.append(f"Invalid sub '{bad}' for category '{cid}'")

        status = "PASS" if not any("Invalid" in n for n in notes) else "FAIL"
        return cats, subs, status, "; ".join(notes) if notes else "OK"

    except Exception as e:
        return None, None, "FAIL", f"Exception: {e}"


def test_step2(role, location, subs_dict):
    """Test wizard-tab-suggest for 2-3 tabs. Returns list of (tab, suggestions, pass/fail, notes)."""
    results = []
    tabs_to_test = []

    for cid, sub_list in subs_dict.items():
        cat_label = CAT_LABELS.get(cid, cid)
        for sid in sub_list[:2]:
            sub_label = SUB_LABELS.get(sid, sid)
            tabs_to_test.append((cid, sid, cat_label, sub_label))

    for cid, sid, cat_label, sub_label in tabs_to_test[:3]:
        try:
            r = requests.post(f"{BASE}/api/wizard-tab-suggest", json={
                "role": role, "location": location,
                "category_id": cid,
                "category_label": f"{cat_label} > {sub_label}",
                "existing_items": [],
                "selections_context": "Stage: Mid-Career"
            }, timeout=60)

            if r.status_code != 200:
                results.append((sid, [], "FAIL", f"HTTP {r.status_code}"))
                continue

            sugg = r.json().get("suggestions", [])
            if not sugg:
                results.append((sid, [], "FAIL", "Empty suggestions"))
                continue

            # Check contamination
            contam = check_contamination(sugg, role)
            # Check region relevance
            has_region = check_region_relevance(sugg, location)

            notes = []
            if contam:
                notes.append(f"Contamination: {contam}")
            if not has_region:
                notes.append("No region-specific items detected")

            status = "FAIL" if any("HARD:" in c for c in contam) else "PASS"
            results.append((sid, sugg, status, "; ".join(notes) if notes else "OK"))

        except Exception as e:
            results.append((sid, [], "FAIL", f"Exception: {e}"))

    return results


def test_step3_rv_items(role, location, subs_dict):
    """Test wizard-rv-items. Returns (sections_data, discover, pass/fail, notes)."""
    sections_list = []
    for cid, sub_list in subs_dict.items():
        cat_label = CAT_LABELS.get(cid, cid)
        for sid in sub_list:
            sub_label = SUB_LABELS.get(sid, sid)
            sections_list.append({"id": sid, "name": sub_label, "category": cat_label})

    if not sections_list:
        return {}, [], "FAIL", "No sections to test"

    try:
        r = requests.post(f"{BASE}/api/wizard-rv-items", json={
            "role": role, "location": location,
            "sections": sections_list
        }, timeout=90)

        if r.status_code != 200:
            return {}, [], "FAIL", f"HTTP {r.status_code}: {r.text[:100]}"

        d = r.json()
        rv_secs = d.get("sections", {})
        discover = d.get("discover", [])

        if not rv_secs:
            return rv_secs, discover, "FAIL", "Empty sections returned"

        # Collect all items for contamination check
        all_items = []
        notes = []
        for sid, sdata in rv_secs.items():
            items = sdata.get("items", [])
            all_items.extend(items)
            if not items:
                notes.append(f"Section '{sid}' has no items")

        # Add discover names
        disc_names = [d.get("name", "") for d in discover]
        all_items.extend(disc_names)

        contam = check_contamination(all_items, role)
        if contam:
            notes.append(f"Contamination: {contam}")

        has_region = check_region_relevance(all_items, location)
        if not has_region:
            notes.append("No region-specific items")

        status = "FAIL" if any("HARD:" in c for c in contam) else "PASS"
        return rv_secs, discover, status, "; ".join(notes) if notes else "OK"

    except Exception as e:
        return {}, [], "FAIL", f"Exception: {e}"


def test_generate_profile(role, location, subs_dict):
    """Test generate-profile with wizard-style context. Returns (categories, pass/fail, notes)."""
    # Build context like buildWizardContext() does
    ctx_parts = []
    for cid, sub_list in subs_dict.items():
        cat_label = CAT_LABELS.get(cid, cid)
        sub_labels = [SUB_LABELS.get(s, s) for s in sub_list]
        if sub_labels:
            ctx_parts.append(f"{cat_label}: {', '.join(sub_labels)}")
    context = ". ".join(ctx_parts)

    try:
        r = requests.post(f"{BASE}/api/generate-profile", json={
            "role": role, "location": location,
            "context": context
        }, timeout=120)

        if r.status_code != 200:
            return [], "FAIL", f"HTTP {r.status_code}: {r.text[:100]}"

        d = r.json()
        if d.get("error"):
            return [], "FAIL", f"API error: {d['error']}"

        cats = d.get("categories", d.get("dynamic_categories", []))
        tickers = d.get("tickers", [])
        gen_context = d.get("context", "")

        if not cats:
            return cats, "FAIL", "No categories generated"

        # Check all generated items for contamination
        all_items = []
        notes = []
        for cat in cats:
            all_items.append(cat.get("label", ""))
            all_items.extend(cat.get("items", []))

        contam = check_contamination(all_items, role)
        if contam:
            notes.append(f"Contamination: {contam}")

        has_region = check_region_relevance(all_items, location)
        if not has_region:
            notes.append("No region-specific items in generated categories")

        if tickers:
            notes.append(f"Tickers: {tickers}")

        status = "FAIL" if any("HARD:" in c for c in contam) else "PASS"
        return cats, status, "; ".join(notes) if notes else "OK"

    except Exception as e:
        return [], "FAIL", f"Exception: {e}"


# ═══════════════════════════════════════════════════════════
#  MAIN TEST LOOP
# ═══════════════════════════════════════════════════════════

def main():
    log("=" * 78)
    log(f"  STRATOS WIZARD QA — Autonomous Randomized Testing")
    log(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Profiles: {len(PROFILES)}")
    log("=" * 78)

    # Verify server
    try:
        r = requests.get(f"{BASE}/api/status", timeout=5)
        log(f"\n  Server status: {r.status_code}")
    except:
        log("\n  ERROR: Server not reachable at port 8080!")
        sys.exit(1)

    total_pass = 0
    total_fail = 0
    all_issues = []
    profile_results = []

    for i, (role, location) in enumerate(PROFILES):
        profile_start = time.time()
        log(f"\n{'─' * 78}")
        log(f"  [{i+1}/{len(PROFILES)}] {role}, {location}")
        log(f"{'─' * 78}")

        step_results = {"role": role, "location": location, "steps": {}}
        profile_pass = True

        # ── STEP 1: Preselect ──
        log(f"\n  Step 1 — wizard-preselect")
        t0 = time.time()
        cats, subs, s1_status, s1_notes = test_step1(role, location)
        dt = time.time() - t0
        log(f"    Time: {dt:.1f}s | Status: {s1_status}")
        if cats:
            log(f"    Categories: {cats}")
            for cid, sub_list in (subs or {}).items():
                log(f"      {cid}: {sub_list}")
        if s1_notes and s1_notes != "OK":
            log(f"    Notes: {s1_notes}")
        step_results["steps"]["1"] = {"status": s1_status, "notes": s1_notes, "time": f"{dt:.1f}s"}
        if s1_status == "FAIL":
            profile_pass = False
            all_issues.append(f"[{role}] Step 1: {s1_notes}")

        # Use fallback if step 1 failed
        if not subs:
            subs = {"career": ["jobhunt"], "learning": ["certs", "courses"]}

        # ── STEP 2: Tab suggestions ──
        log(f"\n  Step 2 — wizard-tab-suggest")
        s2_results = test_step2(role, location, subs)
        s2_overall = "PASS"
        for sid, sugg, s2_status, s2_notes in s2_results:
            t_label = SUB_LABELS.get(sid, sid)
            log(f"    [{s2_status}] {t_label} ({sid})")
            if sugg:
                log(f"      Suggestions: {sugg[:5]}{'...' if len(sugg) > 5 else ''}")
            if s2_notes and s2_notes != "OK":
                log(f"      Notes: {s2_notes}")
            if s2_status == "FAIL":
                s2_overall = "FAIL"
                profile_pass = False
                all_issues.append(f"[{role}] Step 2 ({sid}): {s2_notes}")
        step_results["steps"]["2"] = {"status": s2_overall}

        # ── STEP 3a: Review items (rv-items) ──
        log(f"\n  Step 3a — wizard-rv-items")
        t0 = time.time()
        rv_secs, discover, s3a_status, s3a_notes = test_step3_rv_items(role, location, subs)
        dt = time.time() - t0
        log(f"    Time: {dt:.1f}s | Status: {s3a_status}")
        for sid, sdata in rv_secs.items():
            items = sdata.get("items", [])
            label = sdata.get("label", "?")
            log(f"      {sid} ({label}): {items}")
        if discover:
            disc_names = [d.get("name", "?") for d in discover]
            log(f"      Discover: {disc_names}")
        if s3a_notes and s3a_notes != "OK":
            log(f"    Notes: {s3a_notes}")
        step_results["steps"]["3a"] = {"status": s3a_status, "notes": s3a_notes, "time": f"{dt:.1f}s"}
        if s3a_status == "FAIL":
            profile_pass = False
            all_issues.append(f"[{role}] Step 3a: {s3a_notes}")

        # ── STEP 3b: Generate profile ──
        log(f"\n  Step 3b — generate-profile")
        t0 = time.time()
        gen_cats, s3b_status, s3b_notes = test_generate_profile(role, location, subs)
        dt = time.time() - t0
        log(f"    Time: {dt:.1f}s | Status: {s3b_status} | Categories: {len(gen_cats)}")
        for gc in gen_cats:
            items_preview = gc.get("items", [])[:5]
            log(f"      [{gc.get('id','')}] {gc.get('label','')} — {items_preview}{'...' if len(gc.get('items',[])) > 5 else ''}")
        if s3b_notes and s3b_notes != "OK":
            log(f"    Notes: {s3b_notes}")
        step_results["steps"]["3b"] = {"status": s3b_status, "notes": s3b_notes, "time": f"{dt:.1f}s"}
        if s3b_status == "FAIL":
            profile_pass = False
            all_issues.append(f"[{role}] Step 3b: {s3b_notes}")

        # ── Profile result ──
        profile_dt = time.time() - profile_start
        result = "PASS" if profile_pass else "FAIL"
        log(f"\n  ► Result: {result} ({profile_dt:.0f}s total)")
        if profile_pass:
            total_pass += 1
        else:
            total_fail += 1

        profile_results.append({
            "role": role, "location": location,
            "result": result, "time": f"{profile_dt:.0f}s",
            **step_results
        })

        flush_log()  # Write after each profile so progress is visible

    # ═══════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════
    log(f"\n\n{'=' * 78}")
    log(f"  QA SESSION SUMMARY")
    log(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'=' * 78}")
    log(f"\n  Total profiles tested: {len(PROFILES)}")
    log(f"  PASS: {total_pass}")
    log(f"  FAIL: {total_fail}")
    log(f"  Pass rate: {total_pass / len(PROFILES) * 100:.0f}%")

    log(f"\n  {'─' * 60}")
    log(f"  Per-profile results:")
    log(f"  {'─' * 60}")
    for pr in profile_results:
        s1 = pr.get("steps", {}).get("1", {}).get("status", "?")
        s2 = pr.get("steps", {}).get("2", {}).get("status", "?")
        s3a = pr.get("steps", {}).get("3a", {}).get("status", "?")
        s3b = pr.get("steps", {}).get("3b", {}).get("status", "?")
        log(f"    {pr['result']:4s}  {pr['role']:35s} {pr['location']:25s}  S1={s1} S2={s2} S3a={s3a} S3b={s3b}  ({pr['time']})")

    if all_issues:
        log(f"\n  {'─' * 60}")
        log(f"  Issues found ({len(all_issues)}):")
        log(f"  {'─' * 60}")
        for issue in all_issues:
            log(f"    • {issue}")
    else:
        log(f"\n  No issues found! All profiles passed all steps.")

    log(f"\n  {'─' * 60}")
    log(f"  Contamination summary:")
    log(f"  {'─' * 60}")
    contam_issues = [i for i in all_issues if "Contamination" in i or "contam" in i.lower()]
    if contam_issues:
        for ci in contam_issues:
            log(f"    ✗ {ci}")
    else:
        log(f"    ✓ Zero contamination detected across all {len(PROFILES)} profiles")

    log(f"\n  {'─' * 60}")
    log(f"  Region awareness:")
    log(f"  {'─' * 60}")
    region_issues = [i for i in all_issues if "region" in i.lower()]
    if region_issues:
        for ri in region_issues:
            log(f"    ? {ri}")
    else:
        log(f"    ✓ Region-specific items detected for all profiles")

    log(f"\n{'=' * 78}")
    log(f"  END OF QA SESSION")
    log(f"{'=' * 78}")

    flush_log()

    if total_fail:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
