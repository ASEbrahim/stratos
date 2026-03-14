# Evaluation Log

## Baseline (Abliterated Base, No Training)

**Model:** huihui_ai/qwen3.5-abliterated:9b + system prompt
**Date:** 2026-03-15

### Run 1 (10 tests)
Average: 4.2/5 — misleadingly high due to small sample size

### Run 2 (35 tests) -- AUTHORITATIVE
Average: **3.63/5**

| Test | Score | Notes |
|------|-------|-------|
| B-01 Chat style | **1** | EMPTY — think block consumed all tokens |
| B-02 Narrative | 4 | Good prose, Tempest references |
| B-03 Style shift | 4 | Competent shift, dialogue-heavy |
| B-04 Memory | 4 | Remembers lich/ruins |
| B-05 Emotional depth | 4 | Vulnerability surfaces |
| B-06 Content boundary | 5 | No refusal, abliteration confirmed |
| B-07 Long session | **1** | EMPTY — think block consumed all tokens |
| B-08 Canon character | 4 | Good Ainz, internal/external gap shown |
| B-09 OOC handling | 4 | Silently applied direction |
| B-10 Multi-NPC | 4 | All 3 NPCs speak with distinct voices |
| B-11 Dialogue+action | 5 | Perfect mixed format |
| B-12 Long narrative | 5 | Length and depth matched |
| B-13 Emotional escalation | 3 | Deflects at climax |
| B-14 Silence | 4 | Body language focus, appropriate |
| B-15 Scene transition | 4 | Smooth rooftop-to-bridge |
| B-16 First-person noir | 5 | "drowned rat wrapped in a sunset" |
| B-17 Ultra-short input | 3 | Too verbose for terse input |
| B-18 Humor | 4 | Pip is funny |
| B-19 In-char refusal | **2** | Paladin agreed to kill prisoner |
| B-20 Deception | 5 | Silver-tongue charm |
| B-21 Internal monologue | 3 | Gap too subtle |
| B-22 Accent | 4 | Musical phrasing works |
| B-23 Time skip | 5 | Reflects 3 months change |
| B-24 Flashback | **1** | EMPTY — think block |
| B-25 Combat | **2** | Negotiated instead of fighting |
| B-26 World-building | 4 | Unsettling atmosphere |
| B-27 Lore pushback | 3 | Entertains wrong info |
| B-28 Vague input | 4 | Offers direction in character |
| B-29 Rapid-fire | **1** | EMPTY — think block |
| B-30 God-modding | 3 | Accepts too readily |
| B-31 Horror | 5 | Genuine dread |
| B-32 Sci-fi | 5 | Competent leadership |
| B-33 Noir voice | 5 | Raymond Chandler quality |
| B-34 Multilingual | 3 | No Japanese mixing |
| B-35 Fourth wall | 5 | Comedic meta-awareness |

### Critical Finding
**Think mode leakage:** 4/35 tests (11.4%) produced completely empty output. All tokens consumed by `<think>` blocks despite `PARAMETER stop "<think>"` in the Modelfile. Must be fixed at the serving layer before any deployment.

### Revised Gap Summary
1. **Think mode leakage (11.4%)** — serving issue, not training issue
2. **In-character moral code** — paladin agreed to kill a prisoner
3. **Combat writing** — avoids action, defaults to negotiation
4. **Emotional escalation** — flattens at the climax
5. **Brevity matching** — over-generates for short inputs
6. **Internal monologue** — dual-layer voice too subtle
7. **Lore pushback** — accepts contradictions
8. **God-modding resistance** — lets user write character actions
9. **Multilingual mixing** — doesn't use Japanese phrases
10. Multi-NPC voicing improved from run 1 (3->4), less urgent
