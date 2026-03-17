# RP System Overhaul — Session Report (2026-03-17)

## Overview

Full overhaul of the RP chat system addressing 15+ bugs reported from live testing. 11 commits, 22 files changed, ~1,077 lines added across backend, frontend, and mobile.

**Rollback branches:**
- `pre-rp-prompt-rewrite` @ `28ef226` — before prompt rewrite
- `pre-refinement-round2` @ `6831dfd` — before refinement round 2
- `pre-fixes-backup` @ `9f314db` — original pre-audit state

---

## Bugs Fixed

### 1. Memory System Was Dead Code
**Problem:** `rp_memory.py` had a full 3-tier memory system (facts, conversation, arc summaries) that was never imported or called from `rp_chat.py`.
**Fix:** Integrated `build_rp_context()`, `extract_facts_immediate()`, `extract_facts()`, `should_arc_summarize()`, and `extract_arc_summary()` into the chat handler.
**Result:** Characters now track facts (regex immediate + LLM every 3rd turn), build arc summaries (every 10 turns + scene transitions), and reference earlier conversation naturally.

### 2. Edit/Feedback Always Sent message_id=0
**Problem:** Backend never returned message IDs in SSE done events. Mobile used `parseInt(UUID)` which yielded `NaN` → `0`. Web frontend never set `data-message-id` on DOM elements.
**Fix:** Full pipeline: Backend returns `user_message_id` + `message_id` in SSE done event → SSE parser passes done payload through → chatStore stores `dbId` on messages → FeedbackButtons and EditSheet use `dbId`.
**Files:** rp_chat.py, sse.ts, chat.ts, chatStore.ts, types.ts, FeedbackButtons.tsx, EditSheet.tsx, ChatMessageList.tsx, chat/[id].tsx

### 3. Character Cards Had Minimal Influence
**Problem:** Card fields were injected as flat metadata labels (`Personality: X`). The model treated them as optional context.
**Fix:** Rewritten as immersive instructions: `CORE PERSONALITY — this defines how you think, feel, and act in EVERY response`. Example dialogues emphasized as voice reference. Personality placed before appearance for priority.

### 4. FORMAT_HINT Forced 100% Action-First
**Problem:** Mobile `chat.ts` appended `[OOC: Use *asterisks* for actions...]` to every user message for cards without `speech_pattern`. This overrode system prompt variety instructions, causing 100% `*action*` → `"dialogue"` format.
**Fix:** Removed format-forcing from the hint. Kept length proportionality + variety instruction. Format variety now controlled by per-turn system message injection.

### 5. Response Format Was Monotonous
**Problem:** Every response started with `*action*` then `"dialogue"` then `*action*`. Tested across 3 prompt variants × 30 messages each.
**Fix:** Per-turn format rotation injected as system message: cycles through "Start with dialogue" / "Start with *action*" / "Start with narration". Tested on all 3 available models.
**Result:** Consistent 25/50/25 act/dlg/narration split across all models and card types.

### 6. Echo Rate Was 76%
**Problem:** Model restated user questions ("Do you like it here?" → "Do you like it here? Well...") and echoed key words.
**Fix:** 4 concrete BAD/GOOD examples in system prompt covering both sentence-level and word-level echo. Clinical archetype gets explicit "Use YOUR OWN metaphors — never repeat user's words."
**Result:** Echo dropped from 76% → 0-13% depending on character.

### 7. No First Message Context
**Problem:** Character card's `first_message` was stored but never seeded as turn 0 in conversation.
**Fix:** When starting a new session with a card that has `first_message`, insert it as turn 0 before the user's first message.

### 8. Session Context Disappeared After One Message
**Problem:** `session_context` was only passed in POST body per-message. If frontend didn't resend, it was lost.
**Fix:** Backend persists `session_context` in `rp_session_context` table (tier 1, category "session"). Auto-loaded on subsequent messages even without frontend resending.

### 9. Context Modal Not Editable
**Problem:** Mobile context modal cleared `contextInput` after applying, so reopening showed empty input.
**Fix:** Pre-populates with existing context from store when opening. Title changes to "Edit Context" when context already exists.

### 10. GPU Model Swap Crashed PC
**Problem:** Swapping Ollama ↔ ComfyUI killed Ollama process with insufficient VRAM cooldown (2s). GPU driver couldn't release memory fast enough → fragmentation → system crash. File handle leak on repeated swaps.
**Fix (Option B):** `ensure_comfyui()` now unloads Ollama models via API (`keep_alive: 0`) instead of killing process. Ollama stays alive at 0 VRAM. Verification after unload. File handle tracked and closed.

### 11. Dialogue Didn't Progress (Stagnant)
**Problem:** Actions/narration showed emotional progression but actual dialogue stayed tonally flat across 15+ turns.
**Fix:** Archetype-aware dialogue tone progression system:
- 6 archetypes auto-detected from personality text: shy, confident, tough, clinical, sweet, submissive
- 4 phases per archetype with specific dialogue tone instructions
- User energy detection: aggressive/forward messages trigger archetype-specific high-energy responses
- Tested across all archetypes with both slow-burn and aggressive scenarios

### 12. Responses Too Long for Short Inputs
**Problem:** Short messages like "Hey" got 500-700 character responses.
**Fix:** Per-turn length hint injection: "LENGTH: User sent a SHORT message. Reply with 1-2 sentences MAX." for ≤5 word inputs.
**Result:** Short-input responses dropped from ~500ch to ~150-200ch.

### 13. No References to Earlier Conversation
**Problem:** Characters never referenced things from earlier turns.
**Fix:** CALLBACK hint injected after turn 8: "Naturally reference or build on something from earlier." System prompt explicitly says "REFERENCE EARLIER MOMENTS."

### 14. "New" Marker Lasted 30 Days
**Fix:** Changed to 2 days.

### 15. Trending Sort Meaningless When All Cards Have 0 Sessions
**Fix:** Trending now falls back to `quality_elements_count DESC, created_at DESC`.

---

## New Features

### Character Card Auto-Enrichment
When creating a card with missing depth fields (emotional_trigger, defensive_mechanism, vulnerability, specific_detail), the backend auto-generates them via LLM in a background thread. Card creation returns instantly — enrichment fills in ~2-3 seconds. Only fills empty fields, never overwrites user content. Works for manual creation and Tavern imports.

### JSON Character Import
File picker now accepts both PNG (TavernCard V2 with embedded data) and JSON files. Supports Tavern JSON exports, SillyTavern format (`char_name`, `char_persona`, `world_scenario`, `greeting`), and custom JSON.

### Generate Image Button Always Visible
Removed the requirement to type a character name before the Generate Image button appears.

---

## Testing Methodology

### A/B Prompt Testing
- 30 conversational messages per variant, simulating full escalating conversation
- Tested 3 prompt variants: no-format, explicit-formats, per-turn-injection
- Analyzed: format distribution, echo rate, question answering, length matching, character consistency, dialogue progression, new elements

### Cross-Model Validation
All prompt changes tested against 3 models:
- `qwen3.5:9b` (inference model)
- `huihui_ai/qwen3.5-abliterated:9b` (uncensored)
- `stratos-rp-q8` (fine-tuned scorer)

### Full Card Ranking
All 11 character cards tested on standardized 12-turn conversations with full metric analysis:

| # | Card | Rating | Grade | Score | Echo | QA | Arc |
|---|------|--------|-------|-------|------|----|-----|
| 1 | Nurse Sakura | nsfw | A+ | 85 | 8% | 100% | YES |
| 2 | Yuki Tanaka | sfw | A+ | 84 | 8% | 100% | YES |
| 3 | Elara Moonwhisper | sfw | A+ | 83 | 0% | 100% | YES |
| 4 | Kai Volkov | nsfw | A+ | 83 | 0% | 100% | YES |
| 5 | Raven Blackwood | nsfw | A+ | 81 | 0% | 100% | YES |
| 6 | Mira Chen | sfw | A | 79 | 8% | 100% | YES |
| 7 | The Archivist | sfw | A | 78 | 0% | 100% | NO |
| 8 | Dr. Sable Vex | sfw | A | 73 | 8% | 100% | YES |
| 9 | Elsa | nsfw | A | 71 | 0% | 100% | NO |
| 10 | Kael Stormborn | sfw | A | 70 | 25% | 100% | YES |
| 11 | Reny | nsfw | B | 65 | 8% | 100% | NO |

**Key stats:** 5 A+, 5 A, 1 B. 0% deflection across all 11 cards. Echo ≤8% for 9/11 cards.

---

## Architecture Diagram

```
User Message
    │
    ▼
[rp_chat.py /api/rp/chat]
    │
    ├─ extract_facts_immediate() ──→ rp_session_context (tier 1, regex)
    │
    ├─ build_rp_context() ──→ Assemble tier1 facts + tier3 arcs
    │
    ├─ _build_system_prompt()
    │   ├─ RP_SYSTEM_PROMPT (v7)
    │   ├─ Character card (personality, speech, scenario, depth)
    │   ├─ Example dialogues
    │   └─ Memory context (facts + arcs)
    │
    ├─ [INJECTION system message]
    │   ├─ SITUATION: arc summary + known facts
    │   ├─ CALLBACK: reference earlier conversation
    │   ├─ FORMAT: rotate dialogue/action/narration
    │   ├─ DIALOGUE TONE: archetype-phase-aware
    │   └─ LENGTH: short input → short reply
    │
    ├─ Director's note (optional)
    │
    └─ User message → Ollama → SSE stream → client
                                    │
                                    ├─ message_id returned in done event
                                    │
                                    └─ Background threads:
                                        ├─ extract_facts() every 3rd turn
                                        └─ extract_arc_summary() every 10th turn
```

---

## Files Changed (22 files, ~1,077 lines)

**Backend:**
- `routes/rp_chat.py` — System prompt v7, archetype system, format rotation, situational injection, memory integration, message IDs
- `routes/character_cards.py` — Auto-enrichment, Tavern import enrichment
- `routes/rp_memory.py` — Arc frequency 25→10, fact extraction 5→3
- `routes/gpu_manager.py` — Option B (unload-only, no kill), file handle fix
- `database.py` — Trending sort tiebreaker

**Mobile:**
- `stores/chatStore.ts` — dbId capture from SSE done event
- `lib/chat.ts` — FORMAT_HINT fix, onDone signature
- `lib/sse.ts` — Pass done payload through
- `lib/types.ts` — dbId field on ChatMessage
- `lib/tavern-import.ts` — JSON import support
- `components/chat/FeedbackButtons.tsx` — Use dbId
- `components/chat/EditSheet.tsx` — Use dbId
- `components/chat/ChatMessageList.tsx` — Pass dbId through
- `components/creator/EditorActions.tsx` — Always show Generate Image
- `components/creator/CardEditor.tsx` — JSON file picker, context pre-populate
- `components/cards/CharacterCard.tsx` — 2-day "New" marker
- `app/chat/[id].tsx` — dbId, context editing

**Frontend (Web):**
- `agent.js` — Scroll fix

**Test:**
- `tests/test_rp_prompts.py` — A/B test harness (481 lines)

---

## Remaining Known Issues

1. **Web frontend RP** uses `/api/agent-chat` not `/api/rp/chat` — edits/feedback still broken on web (mobile works)
2. **Reny card scores B (65)** — his speech_pattern field says "Always starts with *" which conflicts with format variety
3. **Confident archetype dialogue progression** could be stronger — Raven still occasionally reads flat
4. **`/api/suggest-context` throws "Role required"** — mobile sends session_id/persona but endpoint expects role from profile config (non-fatal, suggestions return empty)
