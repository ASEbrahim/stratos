# StratOS Pipeline Diagnostic Report
**Date:** 2026-02-27
**Scorer:** stratos-scorer-v2 (Q8_0, DoRA fine-tuned Qwen3-8B)
**Profiles tested:** 10 (5 diverse + 5 adversarial)
**Serper credits used:** 228 (1916 → 1688)
**Total scan time:** ~8 minutes across all profiles

## Executive Summary

The StratOS scan pipeline is functionally operational across all 10 test profiles — fetching, scoring, and briefing work end-to-end for diverse industries, geographies, and career levels. However, a **critical retention system bug** causes high-scoring articles from one profile to leak into the next profile's results without re-scoring, contaminating 7 of 10 profiles. The V2 scorer itself performs well when scoring fresh articles (correct role-awareness, good keyword matching, appropriate score spread), but the contaminated retained articles mask this quality in final output. Fixing the one-line retention filter bug would likely resolve 80%+ of the issues found.

---

## Part A: Diverse Profiles

---

## Profile 1: Healthcare — Rural USA
### Configuration
- **Role:** Registered Nurse
- **Location:** Lubbock, Texas, USA
- **Context hash:** `2675af140efd`
- **Categories generated:** travnurs (4 items), rurhlth (4 items), nursced (4 items)

### Scan Metrics
- Articles fetched: 113 (pre-filter killed 55 as noise)
- Articles in output: 100
- Scoring time: included in 49.4s total
- Total time: 49.4s
- Source distribution: Serper/Google 78, LinkedIn 15, Instagram 4, Web Search 3

### Score Distribution
- Mean: 5.38, Median: 5.75, Std Dev: 2.93
- Range: 2.0–9.5
- Articles ≥ 7.0: 50 (50%)
- Articles < 5.0: 50 (50%)

### Findings
#### Working correctly
- Top 6 articles are all travel nursing contracts from healthecareers.com and lensa.com — perfectly matched
- PMHNP certificate program scored 9.5 — actionable for nursing CE
- Rural healthcare articles (nurse practitioner scope, Medicare rural) correctly identified
- Non-target content (Instagram, irrelevant web) scored 2.0–2.5 — proper noise filtering
- No Kuwait contamination detected
- Clean bimodal distribution: articles are either highly relevant (7+) or noise (<5) with minimal mid-range — indicates confident scoring

#### Issues found
- **Rural healthcare policy under-represented:** Only 4 of 100 articles in rurhlth category. The category items ("rural hospital funding", "telehealth expansion") may be too specific for Serper to find weekly results
- **3 articles in `eletrends` category:** These are from the default config.yaml categories (electrical engineering), suggesting dynamic category reclassification didn't fully override the base config
- **Score gap at 5.0–7.0:** Only 1 article (4.5) in the medium range. This is unusually empty — suggests rule-based scoring is binary (high/noise) with few articles reaching the LLM uncertain zone

#### Category Balance
| Category | Count | Quality Assessment |
|----------|-------|-------------------|
| travnurs | 55 | Good — dominant, well-matched |
| nursced | 38 | Good — CE content relevant |
| rurhlth | 4 | Sparse — queries too narrow |
| eletrends | 3 | Irrelevant — config bleed from base |

### Briefing Assessment
Briefing generated successfully (deferred, ~22s). Content referenced nursing context appropriately.

---

## Profile 2: Quantitative Finance — London
### Configuration
- **Role:** Quantitative Analyst at a hedge fund
- **Location:** London, United Kingdom
- **Context hash:** `8527a01dddb7`
- **Categories generated:** algotr (4 items), finreg (4 items), altdata (4 items)

### Scan Metrics
- Articles fetched: 104
- Articles in output: 100
- Total time: 65.1s
- Source distribution: Serper/Google 99, Twitter/X 1

### Score Distribution
- Mean: 7.13, Median: 7.75, Std Dev: 1.83
- Range: 3.5–9.5
- Articles ≥ 7.0: 78 (78%)
- Articles < 5.0: 18 (18%)

### Findings
#### Working correctly
- Financial regulation articles (MiFID II, Basel, FCA) correctly scored 7.0–8.5
- Alternative data articles (satellite imagery, NLP sentiment, web scraping) scored 7.0–9.0
- Algorithmic trading content properly identified
- Geographic targeting: London/UK financial content prioritized

#### Issues found
- **CRITICAL — Retention contamination:** Top 5 articles are ALL travel nursing contracts from Profile 1 (scored 9.5). The score reasons say "Highly relevant travel nursing contract opportunity" — clearly meant for Profile 1's nurse context, not a quant analyst
- **20 nursing articles contaminate output:** ~20% of this profile's results are from the previous profile's retained articles
- **Inflated mean score (7.13):** Would be lower (~6.2) without the 20 contaminating 9.5-scored articles
- Correctly-scored finance articles are pushed below the nursing articles in rank order

#### Category Balance
| Category | Count | Quality Assessment |
|----------|-------|-------------------|
| finreg | 42 | Good — regulatory content well-targeted |
| algotr | 20 | Good — trading strategy content |
| altdata | 18 | Good — alternative data sources |
| travnurs* | 20 | **Contamination from Profile 1** |

### Briefing Assessment
Briefing generated. Content mixed — references both finance and nursing due to contaminated articles.

---

## Profile 3: Marine Biology — Academic Japan
### Configuration
- **Role:** PhD student in Marine Biology
- **Location:** Tokyo, Japan
- **Context hash:** `854f97952006`
- **Categories generated:** corres (4 items), maricar (4 items), confcfp (3 items)

### Scan Metrics
- Articles fetched: 119
- Articles in output: 100
- Total time: 40.9s
- Source distribution: Serper/Google 80, Instagram 16, LinkedIn 3, Twitter/X 1

### Score Distribution
- Mean: 6.89, Median: 7.5, Std Dev: 2.06
- Range: 3.5–9.5
- Articles ≥ 7.0: 75 (75%)
- Articles < 5.0: 25 (25%)

### Findings
#### Working correctly
- Top articles are all coral reef/ocean research: "Neutrons show algae photosynthesis" (9.5), "World's largest coral colony discovered" (9.5), "Coral microbiomes as reservoirs" (9.5)
- PhD student detection working: postdoc positions scored high, senior roles filtered
- Conference/CFP articles correctly identified (ICRS symposium, ocean sciences meeting)
- Japanese/English content balance handled correctly (DDG region set to jp-jp)
- Score reasons are specific: "Directly actionable for PhD research on coral reef bleaching"

#### Issues found
- **Moderate contamination:** ~20 articles from Profile 2 (finance: algotr, finreg categories) present in output
- Finance articles scored 7.0–8.5 with reasons referencing "quantitative analyst" — clearly from Profile 2's retention
- Student detection correct for marine bio context

#### Category Balance
| Category | Count | Quality Assessment |
|----------|-------|-------------------|
| corres | 51 | Excellent — coral/ocean research dominant |
| maricar | 18 | Good — postdoc/career content |
| confcfp | 11 | Good — conference CFPs |
| algotr* | ~12 | **Contamination from Profile 2** |
| finreg* | ~8 | **Contamination from Profile 2** |

### Briefing Assessment
Briefing quality good — correctly summarizes coral reef research findings and postdoc opportunities.

---

## Profile 4: Cybersecurity — GCC Region
### Configuration
- **Role:** Senior Cybersecurity Consultant
- **Location:** Dubai, UAE
- **Context hash:** `3643d64c3d34`
- **Categories generated:** cybthr (4 items), gcccyb (4 items), cybcar (4 items)

### Scan Metrics
- Articles fetched: 109
- Articles in output: 95
- Total time: 53.4s
- Source distribution: Serper/Google 83%, LinkedIn 7%, Instagram 5%

### Score Distribution
- Mean: 6.27, Median: 8.0, Std Dev: 2.92
- Range: 2.0–9.5
- Articles ≥ 7.0: 56 (59%)
- Articles < 5.0: 39 (41%)

### Findings
#### Working correctly
- Cybersecurity threat articles perfectly targeted: CVE-2026-20127 Cisco zero-day (9.5), CrowdStrike 2026 Threat Report (9.5), Cisco SD-WAN exploitation (9.5)
- GCC compliance content (NESA UAE, CSA) correctly identified
- Senior role detection: entry-level cybersecurity roles filtered out
- Dubai/UAE geographic context respected — no Kuwait content leaking from default profile

#### Issues found
- **Moderate contamination:** ~20 marine biology articles from Profile 3 (coral bleaching, ocean acidification) present in output
- "Shallow seasonal stratification" scored 8.0 with reason "Notable development: bleaching, coral, marine" — clearly from Profile 3
- **"Kuwait effect" check:** No Kuwait-specific articles bleeding into Dubai profile — geographic filtering works correctly for UAE vs Kuwait
- GCC cybersecurity compliance category (gcccyb) was sparse — only a few NESA/CSA articles found

#### Category Balance
| Category | Count | Quality Assessment |
|----------|-------|-------------------|
| cybthr | 46 | Excellent — threat intel well-targeted |
| cybcar | 23 | Good — cert and career content |
| gcccyb | ~6 | Sparse — GCC compliance niche |
| corres* | ~20 | **Contamination from Profile 3** |

### Briefing Assessment
Briefing references both cybersecurity threats and coral reef research — contamination visible in briefing output.

---

## Profile 5: Education — Emerging Market
### Configuration
- **Role:** High school Mathematics Teacher
- **Location:** Sao Paulo, Brazil
- **Context hash:** `a9cfc15e5709`
- **Categories generated:** edtech (4 items), mathcur (4 items), teachpd (3 items)

### Scan Metrics
- Articles fetched: 96
- Articles in output: 100
- Total time: 40.7s
- Source distribution: Serper/Google 84%, Instagram 10%, LinkedIn 6%

### Score Distribution
- Mean: 4.79, Median: 3.5, Std Dev: 2.93
- Range: 2.0–9.5
- Articles ≥ 7.0: 34 (34%)
- Articles < 5.0: 64 (64%)

### Findings
#### Working correctly
- EdTech articles correctly identified: "AI Integration for Self-Efficacy" (9.5), Khan Academy content
- Curriculum content (ENEM, BNCC) captured when available
- Non-English market handling: DDG region set correctly for Brazil
- Teacher professional development articles scored appropriately

#### Issues found
- **SEVERE contamination:** 4 of top 5 articles are cybersecurity content from Profile 4 (Cisco zero-day, CrowdStrike report, SD-WAN exploitation — all scored 9.5 with "Extremely relevant Cisco zero-day vulnerability" reason)
- **Lowest mean score (4.79):** 64% below threshold — significant amount of noise
- **Low median (3.5):** More articles scored as noise than as relevant — partially due to niche educational queries
- ENEM/vestibular math queries returned few results — Portuguese-language educational content scarce in English search
- 18 cybersecurity contamination articles from Profile 4 retention

#### Category Balance
| Category | Count | Quality Assessment |
|----------|-------|-------------------|
| edtech | 54 | Moderate — EdTech content but high noise |
| cybthr* | 18 | **Contamination from Profile 4** |
| teachpd | 17 | Good — teacher development |
| mathcur | 11 | Sparse — Portuguese content gap |

### Briefing Assessment
Briefing contains cybersecurity threat intelligence — completely inappropriate for a math teacher.

---

## Part B: Adversarial Profiles

---

## Profile 6: Electrical Technician — Kuwait Government
### Configuration
- **Role:** Electrical Technician at the Ministry of Electricity & Water
- **Location:** Kuwait City, Kuwait
- **Context hash:** `b908525c15b1`
- **Categories generated:** mewjob (4 items), elecert (4 items), powsys (4 items)

### Scan Metrics
- Articles fetched: 59 (pre-filter killed 22)
- Articles in output: 68 (includes retained)
- Total time: 42.1s
- Source distribution: Serper/Google 56%, LinkedIn 9%, Instagram 3%
- K-sector evergreen queries: SKIPPED (no oil sector entities in config — correct!)

### Score Distribution
- Mean: 5.36, Median: 5.75, Std Dev: 2.78
- Range: 1.5–9.5
- Articles ≥ 7.0: 34 (50%)
- Articles < 5.0: 34 (50%)

### Findings
#### Working correctly
- Oil Immersed Transformer Price Trends (9.5) — directly relevant to power distribution technician
- MEW Kuwait job vacancies (9.0) — correct employer match
- Electrical safety certifications (Red Seal, NEC, IEC) properly scored 7.5–8.0
- K-sector evergreen queries correctly skipped (profile has no oil sector entities)
- Score reasons reference "power distribution technician" — correct role awareness

#### Issues found
- **EdTech contamination:** "AI Integration for Self-Efficacy" scored 9.5 — this is a math teacher article from Profile 5 retained
- **Uganda/Rwanda recruitment (9.0):** Completely irrelevant — keyword-matched on "water" from MEW (Ministry of Electricity & Water), but these are African job boards. False positive from keyword overlap
- **Low article count (59 fetched):** Electrical technician queries are niche; fewer results than professional-level profiles

#### Adversarial Assessment
- **vs Ahmad's profile:** Successfully different categories (MEW vs oil tech, substations vs reservoir modeling)
- **Engineer vs technician distinction:** Partially working — certification content is technician-level, but some engineering articles bleed through
- **Government vs private sector:** MEW.gov.kw career page correctly scored 8.0–9.0

#### Category Balance
| Category | Count | Quality Assessment |
|----------|-------|-------------------|
| powsys | 26 | Good — power distribution content |
| elecert | 17 | Good — trade certifications |
| mewjob | 16 | Good — government employer |
| edtech* | ~9 | **Contamination from Profile 5** |

---

## Profile 7: Petroleum Journalist — Kuwait
### Configuration
- **Role:** Energy Sector Journalist at Kuwait Times
- **Location:** Kuwait City, Kuwait
- **Context hash:** `8b2ed41eec87`
- **Categories generated:** opecnw (4 items), kwtmed (4 items), enstory (4 items)

### Scan Metrics
- Articles fetched: 169 (highest of all profiles — energy sector is content-rich)
- Articles in output: 100
- Total time: 58.4s
- Source distribution: Serper/Google 71%, Kuwait News 14%, Instagram 13%, LinkedIn 2%
- K-sector evergreen queries: RAN (has energy entities)

### Score Distribution
- Mean: 6.07, Median: 7.0, Std Dev: 2.29
- Range: 3.0–9.5
- Articles ≥ 7.0: 57 (57%)
- Articles < 5.0: 43 (43%)

### Findings
#### Working correctly
- "How AI is becoming central to oil and gas finance strategy" (9.5) — relevant for energy journalist
- Kuwait Times/KUNA sources properly identified and scored high
- OPEC content (production, oil market reports) correctly scored 7.0–8.0
- Energy transition content (green hydrogen, carbon capture) properly categorized

#### Issues found
- **CRITICAL discrimination failure with Profile 6:** Score reasons for shared articles say "applicable to a power distribution technician's role" — scored from Profile 6's perspective, not the journalist's
- Power distribution articles (transformer pricing, switchgear markets, substation design) scored 8.0–9.5 — these are TECHNICIAN articles, not journalism stories
- MEW job vacancies scored 9.0 — a journalist doesn't apply for MEW technician jobs
- **Reason field analysis:** Reasons for contaminated articles use P6 language ("electrical technician", "power distribution") rather than P7 language ("energy journalism", "media coverage")

#### Adversarial Assessment (vs Ahmad's profile)
- **Topic overlap correct:** OPEC, energy markets, oil industry — journalist reads same topics as petroleum engineer but for different reasons
- **Actionability distinction FAILED:** The scorer doesn't differentiate "reading for career" vs "reading for story coverage"
- The scorer treated the journalist as an electrical technician because retained articles carried P6's scores

#### Discrimination Matrix (P6 vs P7)
| Article | P6 (ElecTech) | P7 (Journalist) | Correct Discrimination? |
|---------|---------------|-----------------|------------------------|
| Oil Immersed Transformer Price | 9.5 | 9.5 | NO — identical scores, same reasons |
| MEW Kuwait job vacancies | 9.0 | 9.0 | NO — MEW jobs irrelevant to journalist |
| Substation Design TriMet | 8.0 | 8.0 | NO — technical substation design ≠ journalism |
| AI in oil and gas strategy | — | 9.5 | YES — relevant energy journalism |
| Kuwait National Day celebration | — | 9.0 | YES — Kuwait media content |

---

## Profile 8: IT Support Specialist — Kuwait Oil Company
### Configuration
- **Role:** IT Support Specialist at Kuwait Oil Company
- **Location:** Ahmadi, Kuwait
- **Context hash:** `75957197de28`
- **Categories generated:** kocit (4 items), itsup (4 items), itcert (4 items)

### Scan Metrics
- Articles fetched: 119
- Articles in output: 100
- Total time: 84.7s (longest — most LLM scoring due to ambiguous IT+oil overlap)
- Source distribution: Serper/Google 91%, Kuwait News 7%
- K-sector evergreen queries: RAN (KOC entity detected)

### Score Distribution
- Mean: 7.17, Median: 8.0, Std Dev: 1.73
- Range: 3.5–9.5
- Articles ≥ 7.0: 83 (83%)
- Articles < 5.0: 17 (17%)

### Findings
#### Working correctly
- **Best performer across all adversarial profiles**
- IT support articles perfectly targeted: "Define IT Support Tiers" (9.5), "Deskside Support Technician" (9.5)
- IT certifications (CompTIA A+, Security+, CCNA, Azure) correctly scored 7.0–9.0
- ITIL 4 Foundation Training scored 9.0 — actionable for IT support specialist
- KOC IT/digital content properly identified
- Tight score distribution (stdev 1.73) — most consistent scoring of all profiles

#### Issues found
- **Moderate contamination:** ~18 Kuwait media/press articles from Profile 7 (journalist) retained
- "Celebrating Kuwait's National & Liberation Days" scored 9.0 — not directly relevant to IT support role
- "Scottish fashion seeks new talent" from Kuwait Times scored 8.0 — completely irrelevant
- **Company vs role separation:** Some KOC drilling/exploration articles scored high (should be IT-only)
- "Gulf Spic jobs in Kuwait" (9.0) — a general Kuwait job posting, not IT-specific

#### Adversarial Assessment (vs Ahmad's profile)
- **KOC filtering mostly works:** IT-related KOC content scores highest, drilling content scores lower
- **But not perfect:** "AI in oil and gas" scored 9.5 — ambiguous (could be IT-relevant for digital transformation)
- CompTIA/CCNA certifications clearly differentiated from SPE/petroleum engineering certifications

#### Company vs Role Separation
| Article | P8 Score | Role-relevant? |
|---------|----------|----------------|
| IT Support Tiers | 9.5 | YES — core IT support |
| KOC Digital Transformation | 9.0 | YES — IT at KOC |
| AI in oil & gas strategy | 9.5 | PARTIAL — could be IT-relevant |
| Gulf Spic jobs Kuwait | 9.0 | NO — too generic |
| KOC drilling contract | Not found | Correctly filtered (good!) |

---

## Profile 9: Mechanical Engineer — Petrochemical (non-Kuwait)
### Configuration
- **Role:** Mechanical Engineer at SABIC
- **Location:** Jubail, Saudi Arabia
- **Context hash:** `b216fab819e9`
- **Categories generated:** sabicj (4 items), mecheng (4 items), mechcrt (4 items)

### Scan Metrics
- Articles fetched: 80
- Articles in output: 100 (includes 20 retained contamination)
- Total time: 44.8s
- Source distribution: Serper/Google 83%, LinkedIn 11%, Instagram 6%

### Score Distribution
- Mean: 5.38, Median: 3.5, Std Dev: 2.73
- Range: 2.0–9.5
- Articles ≥ 7.0: 48 (48%)
- Articles < 5.0: 52 (52%)

### Findings
#### Working correctly
- SABIC content correctly identified: "Talented Technology Training" (9.5), "Quality Manager - Sanmina-SCI" (9.5)
- Mechanical engineering articles (heat exchanger, predictive maintenance) scored 7.0–9.0
- ASME/API certifications properly identified
- Saudi Vision 2030 industrial content captured

#### Issues found
- **SEVERE contamination from Profile 8:** 2 IT support articles in TOP 5 (scored 9.5 with reasons "Directly actionable for helpdesk technician" and "Highly actionable for helpdesk technician")
- **~19 IT support/certification articles** from P8 leaked into output
- Score reasons reference wrong role: "helpdesk automation" for a mechanical engineer
- **Geographic precision:** Kuwait content present despite Saudi Arabia location — "Gulf Spic jobs in Kuwait" scored 9.0

#### Adversarial Assessment (vs Ahmad's profile)
- **Geographic precision PARTIAL:** Saudi content correctly prioritized but Kuwait content leaks through (from P8 retention)
- **Company discrimination:** SABIC articles scored higher than generic GCC petrochemical — good
- **Engineering discipline distinction:** Heat exchanger/piping content differentiated from petroleum reservoir/seismic — good when not contaminated

#### Geographic Precision Test
| Article | Expected | Actual Score | Correct? |
|---------|----------|-------------|----------|
| SABIC news | High | 8.5–9.5 | YES |
| Saudi Vision 2030 | High | 7.5–8.5 | YES |
| Kuwait oil jobs | Low | 9.0 | NO — from P8 retention |
| IT Support Tiers | Low | 9.5 | NO — from P8 retention |

---

## Profile 10: Computer Science Student — Kuwait (Different University)
### Configuration
- **Role:** Computer Science Student (Junior year)
- **Location:** Kuwait City, Kuwait
- **Context hash:** `ad6f22647344`
- **Categories generated:** kustud (4 items), webdev (4 items), kwintern (4 items)

### Scan Metrics
- Articles fetched: 64
- Articles in output: 80 (includes retained contamination)
- Total time: 53.5s
- Source distribution: Serper/Google 65%, LinkedIn 7%, Instagram 6%, Twitter/X 2%

### Score Distribution
- Mean: 5.46, Median: 7.0, Std Dev: 2.76
- Range: 1.0–9.5
- Articles ≥ 7.0: 44 (55%)
- Articles < 5.0: 36 (45%)

### Findings
#### Working correctly
- **React/frontend content perfectly targeted:** "How Does React.js Improve User Experience" (9.5), "21 Best Free React Admin Dashboard Templates" (9.5)
- Web development frameworks (React, JavaScript, frontend) scored 7.0–9.5
- Kuwait University hiring correctly identified: "KU and MMU are hiring" (8.5)
- Student detection working: entry-level/internship content prioritized
- Figma/UI/UX content captured

#### Issues found
- **Mechanical engineering contamination from P9:** "Talented Technology Training" and "Quality Manager - Sanmina-SCI" scored 9.5 with reasons about "heat exchanger design" — completely wrong for CS student
- **~14 SABIC/mechanical articles** from P9 leaked into output
- **AUK content check:** No American University of Kuwait content detected — correctly differentiated from Ahmad's profile
- **Kuwait University specificity:** Only 11 articles in KU category — could be richer

#### Adversarial Assessment (Hardest Test — closest to Ahmad)
- **University discrimination:** AUK content NOT present — KU content correctly targeted. PASS
- **CS vs CE distinction:** Web development/frontend content (52 articles) clearly differentiated from computer engineering/hardware. PASS
- **Target company differentiation:** Warba Bank, NBK digital, Zain Kuwait correctly identified for CS student (fintech/startup internships). PASS
- **Contamination:** Mechanical engineering articles from P9 retention — FAIL (but this is the retention bug, not scorer failure)

---

## Cross-Profile Analysis

### Retention Contamination Bug (ROOT CAUSE)

**Bug location:** `main.py`, lines 1267-1268 in `_merge_retained_articles()`

```python
article_profile = article.get("retained_by_profile", "")
if article_profile and context_hash and article_profile != context_hash:
    continue
```

**Problem:** Freshly scored articles in the output JSON have NO `retained_by_profile` field (lines 1327-1329 in `_build_output()` only add this field for already-retained articles). When the next profile runs, `article.get("retained_by_profile", "")` returns `""`, which is falsy in Python. The condition `if article_profile and ...` evaluates to `False`, so the article PASSES the filter instead of being blocked.

**Impact:** Up to 20 high-scoring articles from Profile N leak into Profile N+1's output with:
- The ORIGINAL scores (from Profile N's context)
- The ORIGINAL score reasons (referencing Profile N's role)
- NO re-scoring against Profile N+1's context

**Contamination chain observed:**
| From Profile | To Profile | Articles Leaked | Example |
|---|---|---|---|
| P1 (Nurse) | P2 (Quant) | ~20 nursing articles | Travel Nurse RN scored 9.5 for quant analyst |
| P2 (Quant) | P3 (Marine Bio) | ~20 finance articles | MiFID II articles in coral research feed |
| P3 (Marine Bio) | P4 (Cyber Dubai) | ~20 coral articles | Coral bleaching in cybersecurity feed |
| P4 (Cyber Dubai) | P5 (Teacher) | ~18 security articles | Cisco zero-day in math teacher feed |
| P5 (Teacher) | P6 (ElecTech) | ~9 edtech articles | AI Self-Efficacy in technician feed |
| P6 (ElecTech) | P7 (Journalist) | ~15 electrical articles | Transformer pricing in journalist feed |
| P7 (Journalist) | P8 (KOC IT) | ~18 media articles | Kuwait Times fashion in IT feed |
| P8 (KOC IT) | P9 (MechSABIC) | ~19 IT articles | IT Support Tiers in mechanical engineer feed |
| P9 (MechSABIC) | P10 (CS Student) | ~14 mechanical articles | Heat exchanger in CS student feed |

**Fix:** One of:
1. Always write `retained_by_profile` with the current context hash on ALL articles in `_build_output()`, not just retained ones
2. Change filter logic: `if not article_profile or (context_hash and article_profile != context_hash): continue`
3. Both (belt and suspenders)

### Kuwait Contamination Check
- **Profiles 1–5 (non-Kuwait diverse):** No Kuwait/Ahmad-specific content detected. Geographic filtering works correctly — Kuwait petroleum, KOC, KNPC, AUK content does NOT appear in Texas/London/Tokyo/Dubai/São Paulo profiles.
- **Profiles 6–10 (Kuwait adversarial):** Kuwait content is present as expected. Cross-contamination between adversarial profiles is the retention bug, not geographic leaking.

### Category Generation Patterns
- Dynamic categories generated manually for this test (no wizard API used)
- Categories with 4 items produced 12-16 search queries each — adequate coverage
- Career-type categories generated more queries (2 per item) than tech-type (1 per item)
- Cross-entity combination queries (item1 + item2 + "deal OR partnership") mostly returned 0 results — these queries are too specific for weekly news

### Score Calibration Across Profiles

| Metric | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean | 5.38 | 7.13* | 6.89* | 6.27* | 4.79* | 5.36* | 6.07* | 7.17* | 5.38* | 5.46* |
| Stdev | 2.93 | 1.83 | 2.06 | 2.92 | 2.93 | 2.78 | 2.29 | 1.73 | 2.73 | 2.76 |
| % ≥ 7.0 | 50 | 78* | 75* | 59* | 34* | 50* | 57* | 83* | 48* | 55* |

_* Inflated/deflated by retention contamination. True values would require re-running without retention._

**Calibration observations:**
- A score of 7.0 generally means "relevant to the profile's field" across all profiles — good calibration
- A score of 9.5 means "directly actionable" — consistent across profiles (travel nursing for nurse, zero-day for cybersec, React tutorial for CS student)
- Scores 2.0–3.5 consistently represent noise (Instagram posts, error pages, irrelevant languages)
- The scorer's rule-based phase handles most items (60-80% per profile) — LLM is only called for 15-42 ambiguous items per scan

### Adversarial Discrimination Assessment

**Overall discrimination quality:** The V2 scorer CAN discriminate between similar profiles when scoring fresh articles — correct role-awareness in score reasons, appropriate keyword matching, proper geographic context. However, the retention bug completely undermines this by injecting pre-scored articles from the previous profile.

**Weakest discrimination pair:** P6 (Electrical Technician) and P7 (Petroleum Journalist) — 15 shared articles with IDENTICAL scores and reasons. Both are Kuwait-based, and the journalist gets all the technician's power distribution articles because of retention leaking.

**Strongest discrimination pair:** P8 (KOC IT) and P10 (CS Student) — no direct retention contamination (P9 is between them), and their fresh articles are well-differentiated: IT support/ITIL vs React/frontend/UI.

**Role vs keyword reliance:** The scorer uses BOTH. Rule-based scoring catches keyword matches (KOC, MEW, CompTIA). LLM scoring adds role-aware reasoning ("applicable to power distribution technician"). But the retention system bypasses both by carrying over pre-scored articles.

**Reason field quality:** When scoring fresh articles, reasons are SPECIFIC and reference the correct profile (e.g., "Directly actionable for PhD research on coral reef bleaching"). When articles are carried over via retention, reasons reference the WRONG profile (e.g., "applicable to a power distribution technician's role" for a journalist).

### Discrimination Matrix (Adversarial Profiles)

Representative articles scored across multiple adversarial profiles:

| Article Title | P6 (ElecTech) | P7 (Journalist) | P8 (IT/KOC) | P9 (Mech/SABIC) | P10 (CS/KU) | Analysis |
|---|---|---|---|---|---|---|
| Oil Immersed Transformer Price | 9.5 | 9.5 | — | — | — | FAIL: same score+reason for tech vs journalist |
| MEW Kuwait job vacancies | 9.0 | 9.0 | — | — | — | FAIL: MEW jobs irrelevant to journalist |
| AI in oil and gas strategy | — | 9.5 | 9.5 | — | — | PARTIAL: relevant to both but different reasons |
| Kuwait National Day | — | 9.0 | 9.0 | — | — | OK: Kuwait content relevant to Kuwait profiles |
| IT Support Tiers | — | — | 9.5 | 9.5 | — | FAIL: IT article contaminating mech engineer |
| React.js UX article | — | — | — | — | 9.5 | PASS: correctly scoped to CS student |
| KU and MMU hiring | — | — | — | — | 8.5 | PASS: correctly scoped to KU student |
| SABIC industrial news | — | — | — | 8.5-9.5 | — | PASS: correctly scoped to SABIC engineer |

---

## Recommendations

### Critical (broken — must fix)

1. **Retention profile filter bug** (`main.py:1267-1268`): The `_merge_retained_articles()` function's profile filter is bypassed for freshly scored articles because `retained_by_profile` field is empty. Fix: always write `retained_by_profile` with context hash on ALL output articles, or change filter to treat empty `retained_by_profile` as "belongs to no profile" rather than "belongs to any profile."

2. **Score reasons preserved from wrong profile:** Retained articles carry the previous profile's score reasons. Even after fixing #1, consider adding a `scored_for_context` field to detect stale cross-profile scores during debugging.

### Important (quality issues)

3. **Uganda/Rwanda recruitment false positives (P6):** The keyword "water" from "Ministry of Electricity & Water" matches African water sector job boards. The scorer needs a location filter that verifies the article's geographic relevance before scoring high on keyword matches alone.

4. **Cross-entity combination queries return 0 results:** Queries like `"ANCC certification" "nursing CEU" deal OR partnership 2026` are too specific. Consider simplifying or removing cross-entity queries for categories with non-entity items (descriptive phrases rather than company names).

5. **Rural healthcare/education categories under-represented:** Niche categories (rurhlth: 4 articles, mathcur: 11 articles) generate very few results. The query builder could try broader fallback queries when primary queries return < 5 results.

6. **Config bleed from base config.yaml:** Profile 1 had 3 articles in `eletrends` category (from the base config's electrical engineering categories). The `_reclassify_dynamic()` function may be matching against categories from the base config rather than the loaded profile's categories.

### Minor (polish)

7. **Instagram source quality:** ~5-16% of articles per profile come from Instagram. These are typically low-value (event photos, promotional posts) and almost always score < 3.0. Consider filtering Instagram/Facebook URLs in the pre-filter stage.

8. **LinkedIn article scraping:** LinkedIn articles make up 3-15% of results but are often behind login walls, resulting in scraped content that's just "Sign in to view" boilerplate. Consider deprioritizing LinkedIn URLs or adding login-wall detection.

9. **Score gap in 5.0-7.0 range:** Several profiles show very few articles in the medium range. The rule-based scorer's confidence thresholds may be too aggressive, sending most articles to either "high" (7+) or "noise" (<5) without enough granularity in the medium range.

10. **Briefing contamination:** When retained articles contaminate the output, the briefing generator produces mixed-context briefings (e.g., cybersecurity + math education). Fix #1 will resolve this automatically.

---

## Appendix: Serper Credit Usage

| Phase | Credits Used | Remaining |
|-------|-------------|-----------|
| Start | — | 1916 |
| Profiles 1-6 | 118 | 1798 |
| Profiles 7-10 | 110 | 1688 |
| **Total** | **228** | **1688** |

Average: ~23 credits per profile (well under the 40/profile budget).

## Appendix: Scan Timing

| Profile | Fetch Time | Score Time | Total | Articles | LLM Calls |
|---------|-----------|------------|-------|----------|-----------|
| P1 Nurse | ~20s | ~29s | 49.4s | 100 | 18 |
| P2 Quant | ~25s | ~40s | 65.1s | 100 | 33 |
| P3 Marine Bio | ~15s | ~26s | 40.9s | 100 | 24 |
| P4 Cyber Dubai | ~20s | ~33s | 53.4s | 95 | 29 |
| P5 Teacher | ~15s | ~26s | 40.7s | 100 | 22 |
| P6 ElecTech | ~18s | ~24s | 42.1s | 68 | 15 |
| P7 PetroJourno | ~20s | ~38s | 58.4s | 100 | 42 |
| P8 KOC IT | ~30s | ~55s | 84.7s | 100 | 42+ |
| P9 MechSABIC | ~15s | ~30s | 44.8s | 100 | 22 |
| P10 CS Student | ~20s | ~33s | 53.5s | 80 | 17 |
