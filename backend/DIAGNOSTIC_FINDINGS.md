# StratOS Diagnostic Findings — Consolidated Report

**Date:** 2026-02-27
**Scope:** Phase 1: 10-profile stress test (5 diverse + 5 adversarial). Phase 2: 6-profile extended test (RSS, niche, edge cases, student vs senior, briefing capture)
**Total profiles tested:** 16 | **Total Serper credits used:** 310 (1916 → 1579) | **Bugs found:** 3

---

## 1. Critical Bugs

### BUG-1: Retention System Cross-Profile Article Leak
**Severity:** CRITICAL | **Location:** `main.py:1267-1268`
**Impact:** Up to 20 high-scoring articles from Profile N contaminate Profile N+1's output with wrong scores, wrong reasons, and wrong context.

**Root Cause:** `_merge_retained_articles()` uses this filter:
```python
article_profile = article.get("retained_by_profile", "")
if article_profile and context_hash and article_profile != context_hash:
    continue
```
Freshly scored articles never have `retained_by_profile` set (only retained articles get it in `_build_output()` lines 1327-1329). Empty string is falsy → the guard condition evaluates to `False` → article passes through unfiltered.

**Evidence:** Observed in 9 of 10 profiles. Contamination chain:
- P1 Nurse → P2 Quant: 20 nursing articles (9.5 scores) in quant feed
- P4 Cybersec → P5 Teacher: 18 Cisco zero-day articles in math teacher feed
- P8 KOC IT → P9 MechSABIC: 19 IT helpdesk articles in mechanical engineer feed

Score reasons from the wrong profile are preserved verbatim (e.g., "applicable to a power distribution technician's role" appears in the journalist's feed).

**Fix:** Either:
1. Always set `retained_by_profile` on ALL articles in `_build_output()`, not just retained ones
2. Change filter: `if not article_profile or (context_hash and article_profile != context_hash): continue`

---

### BUG-2: Deferred Briefing Not Captured by Programmatic Consumers
**Severity:** MODERATE | **Location:** `main.py:1144-1177`
**Impact:** Any code that reads the `run_scan()` return value or copies the output file immediately after scan completion gets EMPTY briefings.

**Root Cause:** `_spawn_deferred_briefing()` runs in a daemon thread. `run_scan()` returns output BEFORE the briefing thread completes. The briefing patches `output/news_data.json` 10-22s later. There's no `thread.join()` or completion event to wait on.

**Evidence:** All 10 diagnostic profile outputs have `"briefing": {}`. Log shows all 10 briefings completed successfully 10-22s after scan returned.

**Impact for users:** Dashboard users see empty briefing initially, then it appears on next poll (or via SSE `briefing_ready` event — but frontend doesn't listen for it yet per STATE.md). API consumers (export, programmatic access) may get stale/empty briefing.

---

## 2. Scorer Findings

### FINDING-S1: V2 Scorer Role-Awareness is Strong on Fresh Articles
When scoring articles for the first time (not from retention), the scorer correctly:
- References the profile's specific role in score reasons ("PhD research on coral reef bleaching", "helpdesk automation", "quantitative analyst")
- Differentiates student vs professional (student detection working for P3 Marine Bio, P10 CS Student)
- Applies geographic context (DDG region set correctly per profile location)
- Uses category-specific scoring paths (career, tech, regional, auto)

**Metric:** Rule-based scoring handles 60-80% of articles. LLM called for 15-42 ambiguous items per scan. Phase 3 re-scoring (4.5-6.5 uncertain zone) rescores 3-8 items per profile with meaningful changes.

### FINDING-S2: Score Distribution is Bimodal
Most profiles show a clear two-peak distribution:
- Peak 1: Scores 7.0-9.5 (relevant articles)
- Peak 2: Scores 2.0-3.5 (noise)
- Very few articles in the 4.0-6.9 "medium" range

This indicates the rule-based scorer is confidently binary. The LLM uncertain zone (4.5-6.5) captures only 3-8 items per scan. This isn't necessarily bad — it means the rule engine is doing its job — but it means the "medium" band shown in the UI will be nearly empty.

### FINDING-S3: Keyword "water" Creates Geographic False Positives
Profile 6 (Electrical Technician at Ministry of Electricity & **Water**) surfaced Uganda and Rwanda water sector job boards scored 9.0. The keyword "water" from the employer name matched against African recruitment content. The scorer's rule-based phase matched on keyword without verifying geographic relevance.

### FINDING-S4: Cross-Entity Combination Queries Are Ineffective
Queries like `"ANCC certification" "nursing CEU" deal OR partnership 2026` returned 0 results across most profiles. These highly specific multi-entity queries are too narrow for weekly news search. ~30% of all Serper queries returned 0 results.

### FINDING-S5: Config Bleed from Base Config
Profile 1 (Nurse, Texas) had 3 articles in the `eletrends` category (electrical engineering) from the base `config.yaml`. The `_reclassify_dynamic()` function may be matching against categories not belonging to the loaded profile.

---

## 3. Pipeline Findings

### FINDING-P1: Scan Performance is Consistent
| Metric | Min | Max | Avg |
|--------|-----|-----|-----|
| Total scan time | 40.7s | 84.7s | 52.9s |
| Articles fetched | 59 | 169 | 107 |
| Articles in output | 68 | 100 | 94 |
| LLM scoring calls | 15 | 42 | 26 |
| Serper credits/profile | ~15 | ~30 | ~23 |

Profile 8 (KOC IT) was the slowest at 84.7s — it had the most ambiguous items (42 LLM calls) due to IT+oil sector keyword overlap creating scoring uncertainty.

### FINDING-P2: Pre-Filter Kills 20-50% of Fetched Articles
The noise pre-filter in `news.py` removes 20-50% of fetched articles before scoring:
- P1 Nurse: 55/113 killed (49%)
- P3 Marine Bio: 42/119 killed (35%)
- P6 ElecTech: 22/59 killed (37%)
- P7 PetroJourno: 57/169 killed (34%)

This is healthy — it saves LLM scoring time on obvious garbage.

### FINDING-P3: Article Content Cache Carries Between Profiles
Logs show "Loaded 82-100 cached article contents from previous run" on each profile. The deep scraping cache persists between profiles, reusing previously scraped HTML. This is benign (saves network time) but means profiles can share article content cache. It's NOT the cause of the scoring contamination (that's BUG-1).

### FINDING-P4: K-Sector Evergreen Queries Are Correctly Gated
The Kuwait-specific career portal queries (KOC, KNPC, KIPIC) correctly:
- Ran for Kuwait-based profiles WITH oil sector entities (P7, P8)
- Skipped for Kuwait-based profiles WITHOUT oil sector entities (P6 ElecTech — no oil entities)
- Skipped for all non-Kuwait profiles (P1-P5, P9)

### FINDING-P5: No Deferred/Timed-Out Items in Any Profile
All 10 profiles completed Pass 1 scoring with 0 deferred items. No items needed Pass 2 retry. This indicates the scorer is fast enough with the current timeout configuration (45s minimum). The V2 model scores each article in ~1-3s.

---

## 4. Briefing Findings

### FINDING-B1: Briefings Generate Successfully But Are Uncapturable
All 10 briefings generated in 10-22s via the deferred background thread, using `qwen3:14b`. The briefings patch the output file successfully. But:
- The `run_scan()` return value has empty briefing
- The output file is overwritten by the NEXT profile's scan before the current briefing can be captured
- No API to wait for briefing completion (no `thread.join()`, no event/flag)

### FINDING-B2: Briefing Content Contaminated by Retention Bug
Since the briefing generator receives `scored_items` which include retained articles from the previous profile, the briefing content reflects the wrong profile's context. A math teacher's briefing would contain cybersecurity threat intelligence.

### FINDING-B3: Briefing Timing
| Profile | Briefing Time | Critical Signals | High Signals |
|---------|--------------|-----------------|-------------|
| P1 Nurse | 21.2s | 5 | 34 |
| P2 Quant | 10.0s | 0 | 58 |
| P3 Marine Bio | 15.6s | 5 | 43 |
| P4 Cyber Dubai | 15.2s | 5 | 20 |
| P5 Teacher | 15.8s | 2 | 12 |
| P6 ElecTech | 21.9s | 3 | 39 |
| P7 PetroJourno | *not captured* | | |
| P8 KOC IT | 16.5s | 5 | 57 |
| P9 MechSABIC | 9.9s | 2 | 26 |
| P10 CS Student | *running at script exit* | | |

---

## 5. Search/Fetching Findings

### FINDING-F1: Serper Credit Efficiency
228 credits for 10 profiles = ~23 credits/profile. With ~1688 remaining, there's headroom for ~73 more profile scans. The dynamic query builder generates 10-30 queries per category depending on type (career generates more than tech).

### FINDING-F2: Source Distribution Varies by Profile
- Tech/science profiles: 80-99% Serper/Google
- Kuwait profiles: 56-91% Serper, 7-14% Kuwait News
- All profiles: Instagram 3-16% (low value), LinkedIn 3-15% (often login-walled)

### FINDING-F3: Instagram and LinkedIn Sources Are Noise
Instagram articles almost always score < 3.0 (event photos, promotions). LinkedIn articles are often behind login walls — scraped content is just "Sign in to view" boilerplate. Both sources inflate fetch counts without adding signal.

---

## 6. Summary of Issues by Priority

| ID | Severity | Category | Issue | Fix Effort |
|----|----------|----------|-------|-----------|
| BUG-1 | CRITICAL | Retention | Cross-profile article leak | 1-line fix |
| BUG-2 | MODERATE | Briefing | Deferred briefing uncapturable | Add join/event |
| S3 | MODERATE | Scorer | "water" keyword geographic false positive | Add location validation |
| S5 | LOW | Config | Base config categories bleed into profiles | Check _reclassify_dynamic |
| S4 | LOW | Fetcher | Cross-entity queries return 0 results | Simplify/remove |
| F3 | LOW | Fetcher | Instagram/LinkedIn noise | Pre-filter by domain |
| B2 | MODERATE | Briefing | Briefing contaminated by retention bug | Fixed by BUG-1 fix |

---

## 7. Phase 2 Extended Diagnostic Results (Profiles 11-16)

Phase 2 ran 6 profiles with critical methodology improvements:
- **Fresh StratOS instance per profile** (eliminates in-memory retention)
- **30s post-scan wait** for briefing thread completion
- **RSS/extra feed toggles** enabled on P11, P12, P16
- **Edge cases**: empty categories, ultra-niche, broad generalist, student vs senior

### Phase 2 Profile Summary

| Profile | Role | Location | Articles | Scan Time | Mean Score | ≥7.0 | <5.0 | Briefing |
|---------|------|----------|----------|-----------|------------|------|------|----------|
| P11 | Energy Market Analyst | Houston, USA | 74 | 55.5s | 6.08 | 46 (62%) | 28 (38%) | YES (5284ch) |
| P12 | Gov Relations Specialist | Kuwait City | 73 | 28.9s | 6.13 | 45 (62%) | 28 (38%) | YES (4577ch) |
| P13 | Papyrologist (Niche) | Oxford, UK | 31 | 21.3s | 6.27 | 19 (61%) | 12 (39%) | YES (1088ch) |
| P14 | MechEng Student | Jubail, Saudi | 17 | 21.5s | 3.16 | 2 (12%) | 15 (88%) | YES (1227ch) |
| P15 | Graphic Designer (Edge) | Lisbon, Portugal | 45 | 25.7s | 4.81 | 20 (44%) | 25 (56%) | YES (2094ch) |
| P16 | CEO/Generalist | Singapore | 81 | 49.1s | 5.88 | 47 (58%) | 33 (41%) | YES (4756ch) |

**Credits used:** 82 (1661 → 1579) = ~14 credits/profile (lower than Phase 1's ~23/profile due to smaller category sets)

---

### BUG-3: Extra Feeds Not Integrated into Scan Pipeline
**Severity:** MODERATE | **Location:** Architecture gap between `extra_feeds.py` and `main.py`
**Impact:** Feed toggle settings (`extra_feeds_finance`, `extra_feeds_politics`) have NO effect on scan output. Users enabling oilprice/bloomberg/bbc/etc toggles expect those articles to appear in their scored feed — they don't.

**Root Cause:** `fetch_extra_feeds()` is only called from `server.py:415-416` via on-demand API endpoints (`/api/finance-news`, `/api/politics-news`). The scan pipeline in `main.py:run_scan()` never calls it. The `NewsFetcher.fetch_all()` only uses `news.rss_feeds` (custom user-defined RSS URLs).

**Evidence:** P11 configured 5 feeds (oilprice, rigzone, bloomberg, cnbc_top, bbc_world). Log shows: `Fetched 0 items from 0/0 RSS feeds`. Zero RSS-sourced articles in output. Same for P12 (4 Arabic feeds) and P16 (4 feeds).

**Fix:** Either:
1. Integrate `fetch_extra_feeds()` into `NewsFetcher.fetch_all()` so enabled feeds flow into the scoring pipeline
2. Document that feed toggles are browse-only (separate tabs) and do not affect scored feed

---

### FINDING-E1: BUG-1 Confirmed — Contamination Persists Even With Fresh Instances
**Severity:** CRITICAL confirmation

Despite creating a **fresh StratOS instance per profile**, the retention contamination persists because `_snapshot_previous_articles()` reads from the shared `output/news_data.json` file. Each profile's high-scoring articles leak into the next profile through the file system.

**Contamination chain (Phase 2):**
- P11 → P12: 20 retained articles (oilmkt category, scored for "Energy Market Analyst")
- P12 → P13: 17 retained articles (gccrel category, scored for "Government Relations Specialist") — **a Papyrologist in Oxford now has 17 GCC diplomatic articles**
- P13 → P14: 1 retained article (museum category, scored for "Papyrologist")
- P14 → P15: 1 retained article (mechcrt category, scored for "MechEng Student")
- P15 → P16: 1 retained article (destools category, scored for "Graphic Designer")

**Key evidence:** P13's gccrel articles have `retained_by_profile: 2728d4eb0af4` — that's P12's context hash, NOT P13's. The wrong profile's hash is stamped on retained articles, proving the filter at `main.py:1267-1268` is bypassed.

**Additional insight:** The count decreases down the chain (20→17→1→1→1) because each successive profile has fewer high-scoring articles that survive the threshold filter. But even 1 leaked article in the wrong feed is a data integrity violation.

---

### FINDING-E2: Briefing Capture 100% Successful (BUG-2 Workaround Validated)
All 6 briefings captured after 30s wait. Confirms the deferred briefing thread completes within 10-22s reliably.

| Profile | Briefing Length | Critical | High | Role-Word Match |
|---------|----------------|----------|------|-----------------|
| P11 Energy | 5284 chars | 9 | 34 | 3/3 |
| P12 GovRel | 4577 chars | 4 | 21 | 3/3 |
| P13 Niche | 1088 chars | 0 | 2 | 0/5 |
| P14 Student | 1227 chars | 1 | 0 | 2/5 |
| P15 Edge | 2094 chars | 0 | 19 | 3/3 |
| P16 Generalist | 4756 chars | 3 | 43 | 3/3 |

**Observations:**
- Briefing length correlates with article volume and relevance (P13 niche gets shortest)
- Role-word matching: 4/6 profiles score 100% match (role keywords appear in briefing text)
- P13 (Papyrologist) scores 0/5 — briefing can't meaningfully summarize ultra-niche academic content
- P14 (Student) briefing correctly identified "EIT certification" as critical for the student's career path
- Fix for BUG-2 is validated: `thread.join(timeout=30)` or an event flag will reliably capture briefings

---

### FINDING-E3: Bimodal Score Distribution is Universal
Phase 2 confirms Phase 1's finding: virtually ZERO articles in the 5.0-6.9 "medium" range across ALL profile types.

| Profile | ≥9.0 | 7.0-8.9 | 5.0-6.9 | <5.0 |
|---------|------|---------|---------|------|
| P11 Energy | 9 | 37 | 0 | 28 |
| P12 GovRel | 13 | 32 | 0 | 28 |
| P13 Niche | 4 | 15 | 0 | 12 |
| P14 Student | 1 | 1 | 0 | 15 |
| P15 Edge | 1 | 19 | 0 | 25 |
| P16 Generalist | 3 | 44 | 1 | 33 |

**Impact:** The UI "Medium" filter tab will be empty for most users. Consider removing the medium band from the UI or adjusting the rule-based scorer to produce more graduated scores.

---

### FINDING-E4: Student vs Senior Discrimination
P14 (MechEng Student, Jubail) vs P9 (MechEng Senior at SABIC, Jubail):

| Metric | P14 Student | P9 Senior |
|--------|-------------|-----------|
| Total articles | 17 | ~90 |
| Mean score | 3.16 | ~6.5 |
| Articles ≥7.0 | 2 (12%) | ~55 (60%) |
| Articles <5.0 | 15 (88%) | ~35 (40%) |
| Shared URLs | 0 | 0 |

**Key findings:**
- Zero shared articles between student and senior — completely different query generation
- Student profile correctly surfaced FE exam and internship content, not advanced engineering
- Senior profile surfaced industry news, SABIC developments, ASME certifications
- The scorer produced dramatically different scores: student mean 3.16 vs senior ~6.5
- Student profile briefing correctly identified EIT certification as critical action item

**Conclusion:** V2 scorer demonstrates strong student/professional differentiation through both query generation (different searches) and scoring (different thresholds).

---

### FINDING-E5: Ultra-Niche Profile Handled Gracefully (With Caveats)
P13 (Papyrologist in Ptolemaic demotic scripts) produced only 31 articles. However:
- **17 articles (55%) are contaminated** from P12's GCC diplomatic content via retention leak
- **Only 14 articles are genuine papyrology results** (4 scored ≥9.0, 10 scored for relevant museums/archives)
- The scorer correctly identified "Leuven Database of Ancient Books" (8.0) and "Ancient Egyptian race controversy" (7.0) as relevant
- No crashes, no errors — pipeline gracefully handles ultra-niche with low article yield
- Briefing was shortest (1088 chars) reflecting scarcity of content

---

### FINDING-E6: Empty/Minimal Categories Handled Without Errors
P15 had three categories: 4 items (normal), 1 item (single keyword), 0 items (empty). Results:
- **Zero errors or crashes** from empty category
- Single-item category ("Lisbon design meetup") generated valid searches
- 44/45 articles correctly assigned to `destools` category
- 1 article in `mechcrt` category (contamination from P14 via retention leak)
- Pipeline resilient to edge-case category configurations

---

### FINDING-E7: Broad Generalist Gets Fewer Top Scores
P16 (CEO, Singapore, 3 broad categories covering AI/business/politics) vs specialist profiles:

| | Top Scores (≥9.0) | Mean | Articles |
|---|---|---|---|
| P11 Specialist (Energy) | 12% | 6.08 | 74 |
| P16 Generalist (CEO) | 4% | 5.88 | 81 |

The scorer correctly avoids inflating scores for broad, non-specific interests. A generalist gets more moderate scores (more 7.0-8.0) and fewer critical alerts (3 vs 9). This is appropriate behavior — broad interests should produce signal, not critical urgency.

---

### FINDING-E8: Cross-Entity Combination Queries Still Ineffective (Phase 1 S4 Confirmed)
Phase 2 profiles confirm: multi-entity combination queries return 0 results consistently.
- P16: `"venture capital APAC" "startup funding Singapore" deal OR partnership 2026` → 0 results
- P16: `"Singapore economy" "China tech policy" deal OR partnership...` → 0 results
- These queries are too narrow for weekly news. ~30% of Serper queries wasted across all profiles.

---

## 8. Updated Summary of Issues by Priority

| ID | Severity | Category | Issue | Fix Effort | Phase |
|----|----------|----------|-------|-----------|-------|
| BUG-1 | CRITICAL | Retention | Cross-profile article leak via `_merge_retained_articles()` | 1-line fix | 1+2 |
| BUG-2 | MODERATE | Briefing | Deferred briefing uncapturable by programmatic consumers | Add `thread.join()` | 1 |
| BUG-3 | MODERATE | RSS/Feeds | Extra feeds not integrated into scan pipeline | Integrate or document | 2 |
| S3 | MODERATE | Scorer | "water" keyword geographic false positive | Add location validation | 1 |
| E3 | LOW | Scorer | Bimodal scores — empty medium band in UI | UI/scorer adjustment | 2 |
| S5 | LOW | Config | Base config categories bleed into profiles | Check `_reclassify_dynamic` | 1 |
| S4/E8 | LOW | Fetcher | Cross-entity combo queries return 0 results (~30% waste) | Simplify/remove combos | 1+2 |
| F3 | LOW | Fetcher | Instagram/LinkedIn noise sources | Pre-filter by domain | 1 |
| B2 | MODERATE | Briefing | Briefing contaminated by retention bug | Fixed by BUG-1 fix | 1 |

---

## 9. Recommendations

### Critical (Fix Before Production)
1. **Fix BUG-1** — One-line fix in `_build_output()` at `main.py:1327`: always set `retained_by_profile` on ALL output articles, not just retained ones. OR change filter at line 1267-1268 to: `if not article_profile or (context_hash and article_profile != context_hash): continue`
2. **Fix BUG-2** — Add `thread.join(timeout=30)` or a threading Event flag in `_spawn_deferred_briefing()`, expose a `wait_for_briefing()` method

### Important (Fix Soon)
3. **Address BUG-3** — Either integrate `fetch_extra_feeds()` into the scan pipeline so feed toggles affect scored output, or clearly document that feed tabs are browse-only
4. **Remove cross-entity combination queries** — ~30% of Serper queries return 0 results. Remove or simplify the `_build_cross_entity_queries()` logic in `kuwait_scrapers.py`
5. **Add geographic validation** — For keyword matches from employer/entity names (e.g., "water"), validate that the article's geographic context matches the profile's location before assigning high scores

### Minor (Backlog)
6. **Pre-filter Instagram/LinkedIn** — These sources consistently score <3.0. Consider domain-level pre-filtering
7. **Adjust medium band** — Either widen the LLM uncertain zone (e.g., 3.5-7.5 instead of 4.5-6.5) or remove the "medium" filter from the UI
8. **Ultra-niche query expansion** — For profiles with very specific domains, consider broadening the query generation to capture adjacent topics
