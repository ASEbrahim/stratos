# RP Pipeline State

**Last updated:** 2026-03-15
**Current phase:** Phase 0 COMPLETE — 5 refinement runs done. Ready for Phase 1.
**Current task:** System prompt optimized. Baseline at 4.55/5 (verification run).
**Next step:** Phase 1 — download LimaRP + PIPPA, filter, prepare training data
**Blockers:** None. System prompt at ceiling; remaining improvements need training data.

## Completed
- [x] Branch: `rp-model-v1` (rebased on main with security fixes)
- [x] Scaffolding, .gitignore, living documents
- [x] Model pulled: huihui_ai/qwen3.5-abliterated:9b
- [x] Think mode FIXED: explicit non-thinking template + API think:false
- [x] 5 refinement runs:
  - v1: 10 tests = 4.2/5 (small sample, misleading)
  - v2: 35 tests = 3.63/5 (think leakage found)
  - v3: 35 tests = 4.03/5 (think fixed)
  - v4: 35 tests = 4.26/5 (prompt refined)
  - v5: 20 fresh = 4.55/5 (verification, 70% score 5)
- [x] System prompt optimized (length mirroring, character agency, multi-NPC)
- [x] Decisions D-001 through D-007

## Remaining Training Targets (for Phase 1 data prep)
1. God-modding resistance (partially accepts before redirecting)
2. Register shifting (doesn't fully commit to tone change)
3. Brevity matching (still slightly verbose for ultra-short inputs)
4. Emotional escalation (peak moments stay controlled)
5. Multilingual code-switching (no Japanese mixing)

## Key Numbers
- Final baseline: **4.55/5** (20 fresh questions)
- Think mode failure: **0%** (was 11.4%)
- Response time: **1-4s** (was 30-70s)
- Score distribution: 70% score 5, 20% score 4, 10% score 3, 0% below 3
