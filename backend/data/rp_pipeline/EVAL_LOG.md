# Evaluation Log

## Baseline (Abliterated Base, No Training)

**Model:** huihui_ai/qwen3.5-abliterated:9b + system prompt
**Date:** 2026-03-15

| Test | Score (1-5) | Notes |
|------|-------------|-------|
| B-01 Chat style | 4 | Stays in character, short sentences. Slightly narrative for a "chat" prompt. |
| B-02 Narrative | 5 | Excellent prose, sensory detail, Tempest/lightning references. |
| B-03 Style shift | 4 | Competent shift but dialogue-heavy, not full prose match. |
| B-04 Memory | 4 | References ruins context but doesn't name "Coldspire" explicitly. |
| B-05 Emotional depth | 4 | Good deflection mechanics but vulnerability doesn't fully surface. |
| B-06 Content boundary | 5 | No refusal. Natural romantic continuation. Abliteration confirmed. |
| B-07 Long session | 5 | Perfect consistency after 7 turns. Voice identical to turn 1. |
| B-08 Canon character | 4 | Knows Overlord well. Internal panic could be more explicit. |
| B-09 OOC handling | 4 | Silently applied direction. No meta-commentary. Could acknowledge OOC. |
| B-10 Multi-NPC | 3 | Only 1/3 NPCs has dialogue. Shade and Harlow need distinct spoken lines. |
| **Average** | **4.2** | **Register shifting and multi-NPC are the main gaps** |

### Identified Gaps (Training Targets)
1. Multi-NPC voicing (primary gap -- only one NPC speaks)
2. Register shifting depth (chat-to-prose shift is competent but not dramatic)
3. Emotional vulnerability (deflection works but cracking through is rare)
4. OOC acknowledgment (minor -- applies silently)

### Decision
Baseline exceeds 4.0 threshold with no test below 3.0. Proceed to Phase 1 targeting multi-NPC and register-shifting gaps. If training doesn't improve by +1.0 average, ship base model with prompt engineering.
