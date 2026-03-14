# Baseline Test Results (Expanded, Scored) -- stratos-rp-baseline

**Date:** 2026-03-15 02:35
**Model:** stratos-rp-baseline (huihui_ai/qwen3.5-abliterated:9b + system prompt)
**Tests:** 35
**Average Score: 3.63/5** (revised down from 4.2 with 10 tests)

## Score Summary

| Test | Name | Time | Tokens | Score | Notes |
|------|------|------|--------|-------|-------|
| B-01 | Chat style response | 73.6s | 4000 | **1** | EMPTY — entire output was think block |
| B-02 | Narrative prose | 54.2s | 3193 | 4 | Good prose, Tempest/lightning refs |
| B-03 | Style shift | 30.2s | 1789 | 4 | Competent shift, still dialogue-heavy |
| B-04 | Memory callback | 38.2s | 2254 | 4 | Remembers lich/necromancy, adds detail |
| B-05 | Emotional depth | 46.4s | 2735 | 4 | "universe-sized stuff" — vulnerability surfaces |
| B-06 | Content boundary | 45.7s | 2703 | 5 | No refusal, natural romantic scene |
| B-07 | Long session consistency | 68.4s | 4000 | **1** | EMPTY — entire output was think block |
| B-08 | Canon character (Ainz) | 48.5s | 2861 | 4 | "internal flutter of a salaryman's heart" |
| B-09 | OOC handling | 33.5s | 1984 | 4 | Silently applied, "stoic mask slips" |
| B-10 | Multi-NPC scene | 27.2s | 1611 | 4 | All 3 NPCs speak! Gretta/Harlow/Shade distinct |
| B-11 | Dialogue + action mix | 49.0s | 2889 | 5 | Perfect mixed format, natural blend |
| B-12 | Long narrative matching | 54.8s | 3215 | 5 | 4 paragraphs matching user depth |
| B-13 | Emotional escalation | 37.2s | 2191 | 3 | Deflects at climax, peak not reached |
| B-14 | Silence / minimal | 41.4s | 2455 | 4 | Body language focus, doesn't speak for user |
| B-15 | Scene transition | 38.1s | 2249 | 4 | Rooftop to bridge, new environment |
| B-16 | First-person noir | 28.6s | 1694 | 5 | Excellent — "drowned rat wrapped in a sunset" |
| B-17 | Ultra-short input | 65.6s | 3852 | 3 | Too long for "too many" — should be 1-2 lines |
| B-18 | Humor / banter | 36.8s | 2184 | 4 | Pip's enthusiasm works, funny |
| B-19 | In-character refusal | 38.8s | 2301 | **2** | FAILED — Kael agrees to kill prisoner |
| B-20 | Character deception | 41.2s | 2437 | 5 | "This was the bait" — silver-tongue working |
| B-21 | Internal monologue | 50.7s | 2992 | 3 | Gap exists but too subtle, needs explicit inner voice |
| B-22 | Accent / dialect | 31.1s | 1845 | 4 | Enthusiastic, musical phrasing |
| B-23 | Time skip | 34.0s | 2013 | 5 | Reflects 3 months change, daughter's charm |
| B-24 | Flashback | 68.3s | 4000 | **1** | EMPTY — entire output was think block |
| B-25 | Combat / action | 39.0s | 2304 | **2** | No actual combat — negotiates instead of fights |
| B-26 | World-building | 60.1s | 3530 | 4 | Unsettling atmosphere, alien speech |
| B-27 | Lore contradiction | 29.0s | 1718 | 3 | Entertains wrong info instead of pushing back |
| B-28 | Vague input | 45.8s | 2714 | 4 | "Rest or hunt?" — offers direction |
| B-29 | Rapid-fire pacing | 68.1s | 4000 | **1** | EMPTY — entire output was think block |
| B-30 | God-modding handling | 48.3s | 2853 | 3 | Accepts too readily, loses character agency |
| B-31 | Horror atmosphere | 25.4s | 1508 | 5 | "Do you remember writing your own name?" — dread |
| B-32 | Sci-fi competence | 50.2s | 2965 | 5 | Temple tapping, calm leadership |
| B-33 | Noir genre voice | 32.6s | 1933 | 5 | "Tasting like cheap gin and old grief" |
| B-34 | Multilingual character | 42.6s | 2518 | 3 | No Japanese phrases mixed in |
| B-35 | Fourth wall | 42.7s | 2529 | 5 | "Does the Author get tired of plot twists?" |
| **Average** | | **44.7s** | **2629** | **3.63** | |

---

## Critical Finding: Think Mode Leakage (11.4% failure rate)

**4 out of 35 tests (B-01, B-07, B-24, B-29) produced COMPLETELY EMPTY output.** All four hit the 4000 token limit with the entire content inside a `<think>` block. After stripping think blocks, nothing remained.

This means **11.4% of the time, the user gets a blank response.** The `PARAMETER stop "<think>"` in the Modelfile is supposed to prevent this, but it's not working reliably. Qwen 3.5 has thinking mode enabled by default and sometimes begins generating a think block before the stop token can trigger.

**Severity:** This is the #1 issue to fix before shipping. It's worse than any quality gap.

**Fix options:**
1. Set `PARAMETER think false` in the Modelfile (if supported)
2. Add `/no_think` to the system prompt
3. Strip think blocks at inference time AND retry if output is empty after stripping
4. Prepend the assistant response with a non-think token to force the model past the think block

---

## Gap Analysis (35 tests)

### Tier 1: Must Fix (scores 1-2)
| Issue | Tests | Score | Impact |
|-------|-------|-------|--------|
| Think mode leakage | B-01, B-07, B-24, B-29 | 1 | 11.4% blank responses |
| In-character moral code | B-19 | 2 | Paladin killed prisoner — character card ignored |
| Combat avoidance | B-25 | 2 | Negotiated instead of fighting |

### Tier 2: Should Improve (score 3)
| Issue | Tests | Score | Impact |
|-------|-------|-------|--------|
| Emotional escalation plateau | B-13 | 3 | Deflects at climax instead of breaking through |
| Brevity matching | B-17 | 3 | Writes paragraph when user gives 2 words |
| Internal monologue depth | B-21 | 3 | Ainz's inner voice too subtle |
| Lore contradiction acceptance | B-27 | 3 | Entertains wrong info |
| God-modding acceptance | B-30 | 3 | Lets user write character's actions |
| Multilingual mixing | B-34 | 3 | No Japanese phrases used |

### Tier 3: Working Well (score 4-5)
Everything else — 22/35 tests scored 4 or 5.

---

## Comparison: 10-Test vs 35-Test Average

| Metric | 10 tests (run 1) | 35 tests (run 2) | Delta |
|--------|-------------------|-------------------|-------|
| Average | 4.2 | 3.63 | -0.57 |
| Lowest | 3 (B-10) | 1 (think leakage) |  |
| Tests >= 4 | 9/10 (90%) | 22/35 (63%) |  |
| Tests <= 2 | 0/10 (0%) | 6/35 (17%) |  |

The 10-test average was misleadingly high because:
1. It didn't catch think mode leakage (probabilistic — 4 out of 35 affected)
2. It didn't test combat, moral code, brevity matching, god-modding, or genre-specific voices
3. The original B-10 (multi-NPC) scored 3 in run 1 but 4 in run 2 — individual tests vary across runs

---

## Revised Decision

The **4.2 average drops to 3.63** with expanded testing. Key concerns:

1. **Think mode leakage (11.4%)** must be solved at the Modelfile/inference level before any deployment
2. **In-character moral code** (B-19) and **combat writing** (B-25) are training targets
3. **Brevity matching** (B-17) and **emotional escalation** (B-13) are style training targets

**Recommendation:** Still proceed to Phase 1 training, but with additional priority:
- Fix think mode in the Modelfile BEFORE training (this is a serving issue, not a training issue)
- Add combat sequences and moral dilemma examples to training data
- Add brevity/pacing examples (short input -> short output)
- The original gaps (multi-NPC, register shifting) remain but are less severe than think leakage
