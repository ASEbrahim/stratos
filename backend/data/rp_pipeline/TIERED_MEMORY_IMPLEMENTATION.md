# Tiered Memory Implementation Plan

## Overview
Three-tier persistent memory for RP chat. Cross-session, cross-platform (web + mobile).
SQLite as source of truth. No flat files.

## Architecture

### Tier 1 — Facts (Permanent, ~100-200 tokens)
- Extracted key-value pairs: name, traits, backstory, relationship milestones
- Stored in `rp_session_context` table with `tier=1`
- **Never summarized, never dropped**
- Extraction: hybrid regex + LLM every 5th message
- Regex patterns: `my name is (\w+)`, `I'm from (\w+)`, `I have a (\w+)`, `call me (\w+)`
- LLM fallback for complex facts (relationship changes, emotional revelations)
- Token budget: ~100-200 tokens, capped. Sort by `updated_at` DESC, take top N that fit
- Categories: `user_fact`, `npc_state`, `scene`, `relationship`

### Tier 2 — Recent Conversation (Verbatim, Rolling Window)
- Last N message pairs from `rp_messages` table (already exists)
- No summarization — just a sliding window
- Fills remaining token budget after Tier 1 + Tier 3
- Typically ~2000-3000 tokens = ~10-15 message pairs
- This is what gives emotional continuity and natural flow

### Tier 3 — Arc Summaries (Periodic State Checkpoints)
- NOT plot summaries — relationship state snapshots
- Triggered on: location change, new NPC, content rating shift, or every 25 messages fallback
- Stored in `rp_session_context` with `tier=3`, `category='arc_summary'`
- Each ~100-150 tokens, stacks over time
- Oldest arcs compress into each other if total exceeds ~800 tokens
- Quality gate: discard if <30 tokens or >200 tokens, retry once

### Tier 3 Prompt (CRITICAL — do not modify without testing)
```
You are reading an RP conversation between {user_name} and {character_name}.
Describe the CURRENT relationship state in 2-3 sentences.

Rules:
- Describe WHERE things stand NOW, not what happened
- Include: trust level, emotional tension, unresolved feelings, who has power
- Use specific details from the conversation, not generic phrases
- BAD: "They grew closer and shared feelings"
- GOOD: "She trusts him enough to show vulnerability but flinches when he reaches for her hand — the physical intimacy is ahead of the emotional safety"

Current state:
```

## Database Schema

### New table: `rp_session_context`
```sql
CREATE TABLE rp_session_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    tier INTEGER NOT NULL,           -- 1=facts, 2=unused (messages table), 3=arc_summary
    category TEXT NOT NULL,          -- 'user_fact', 'npc_state', 'scene', 'relationship', 'arc_summary'
    key TEXT NOT NULL,               -- 'name', 'location', 'mood', NPC name, arc index
    value TEXT NOT NULL,             -- the extracted fact or summary text
    turn_number INTEGER DEFAULT 0,   -- which message triggered this extraction
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_rp_context_session ON rp_session_context(session_id, tier);
CREATE INDEX idx_rp_context_updated ON rp_session_context(session_id, updated_at DESC);
```

### Migration number: 028

## Token Budget Per Turn
```
System prompt + character card:  ~300-400 tokens
Tier 1 facts:                    ~100-200 tokens (capped at 200)
Tier 3 arc summaries:            ~400-800 tokens (oldest compress)
Tier 2 recent messages:          ~2000-3000 tokens (fills remaining)
Steering instructions:           ~50-100 tokens
─────────────────────────────────────────────
Total:                           ~3000-4500 tokens out of 32K
Headroom:                        ~27K tokens
```

## Files to Create/Modify

### NEW: `backend/routes/rp_memory.py` (~300 lines)
Core memory system. Functions:

1. `extract_facts(session_id, user_msg, ai_response, turn_number, db)`
   - Runs every 5th message (check `turn_number % 5 == 0`)
   - Phase 1: regex extraction for simple patterns
   - Phase 2: LLM extraction for complex facts (only if regex found nothing)
   - Writes to `rp_session_context` with `tier=1`
   - Upserts: if key already exists for this session, update value + updated_at

2. `extract_arc_summary(session_id, user_name, char_name, recent_messages, turn_number, db)`
   - Triggered by: scene transition detected OR turn_number % 25 == 0
   - Uses Tier 3 prompt (above)
   - Quality gate: 30 < len(summary.split()) < 200, else retry once
   - Writes to `rp_session_context` with `tier=3`, `category='arc_summary'`
   - Compress oldest arcs if total tier 3 tokens > 800

3. `detect_scene_transition(user_msg, ai_response) -> bool`
   - Regex for location keywords: "walked to", "arrived at", "entered the", "moved to"
   - Regex for time skips: "next morning", "hours later", "the following day"
   - Regex for new NPCs: character names not seen in previous messages
   - Returns True if any detected

4. `build_rp_context(session_id, db, token_budget=4000) -> (str, dict)`
   - Returns: (assembled_prompt_string, debug_dict)
   - debug_dict: `{"tier1": token_count, "tier3": token_count, "tier2": token_count, "tier2_pairs": N, "total": token_count}`
   - Assembly order:
     a. Load all Tier 1 facts for session, sort by updated_at DESC, take top N within 200 token cap
     b. Load all Tier 3 arc summaries for session, all of them (within 800 token cap)
     c. Calculate remaining budget: token_budget - tier1_tokens - tier3_tokens - 100 (steering buffer)
     d. Load Tier 2 messages from rp_messages, newest first, fill remaining budget
   - Format Tier 1 as: `[Memory - Known Facts]\n- Name: Marcus\n- Scar: left hand, childhood\n...`
   - Format Tier 3 as: `[Memory - Relationship Arc]\n{arc1}\n{arc2}\n...`
   - Token estimation: word_count * 1.3

5. `should_extract(turn_number) -> bool`
   - Returns True if turn_number % 5 == 0

6. `should_arc_summarize(turn_number, user_msg, ai_response) -> bool`
   - Returns True if detect_scene_transition() OR turn_number % 25 == 0

### MODIFY: `backend/routes/rp_chat.py`
Minimal changes:

1. Import `rp_memory`
2. In chat handler, BEFORE building prompt:
   ```python
   context_str, debug = rp_memory.build_rp_context(session_id, db, token_budget=4000)
   # Inject context_str into system prompt
   ```
3. AFTER streaming response complete (non-blocking):
   ```python
   if rp_memory.should_extract(turn_number):
       threading.Thread(target=rp_memory.extract_facts, args=(...)).start()
   if rp_memory.should_arc_summarize(turn_number, user_msg, ai_response):
       threading.Thread(target=rp_memory.extract_arc_summary, args=(...)).start()
   ```

### MODIFY: `backend/database.py`
Add migration 028 for `rp_session_context` table.
Add helper methods:
- `insert_rp_context(session_id, tier, category, key, value, turn_number)`
- `upsert_rp_context(session_id, tier, category, key, value, turn_number)`
- `get_rp_context(session_id, tier=None, category=None, limit=50)`
- `delete_rp_context(session_id)` (for cleanup)

## Regex Patterns for Tier 1 Extraction
```python
FACT_PATTERNS = [
    (r"(?:my name is|I'm|call me|I am)\s+([A-Z][a-z]+)", "user_fact", "name"),
    (r"(?:I'm from|I live in|I grew up in)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", "user_fact", "origin"),
    (r"I (?:have|got) (?:a |an )?(.{5,50}?)(?:\.|,|$)", "user_fact", "trait"),
    (r"(?:my favorite|I love|I prefer)\s+(.{3,30}?)(?:\.|,|!|$)", "user_fact", "preference"),
]
```

## LLM Extraction Prompt (Tier 1, every 5th message)
```
Extract NEW facts from this exchange as JSON. Only include information not previously known.
{"user_facts": ["name: X", "trait: Y"], "npc_changes": [{"name": "Z", "change": "..."}], "scene": {"location": "...", "mood": "..."}}
If nothing new, return {"user_facts": [], "npc_changes": [], "scene": {}}
```

## Scene Transition Detection Patterns
```python
LOCATION_PATTERNS = [
    r"(?:walked|moved|went|headed|traveled|arrived|entered|stepped)\s+(?:to|into|towards)\s+",
    r"(?:they|we|she|he)\s+(?:were|are)\s+(?:now\s+)?(?:in|at)\s+",
]
TIME_PATTERNS = [
    r"(?:next|the following)\s+(?:morning|day|evening|night|week)",
    r"(?:hours?|days?|weeks?)\s+later",
    r"(?:the next|that)\s+(?:morning|evening|night)",
]
```

## Testing Plan
1. Create new RP session with Elsa
2. Send: "My name is Viktor and I was born in Prague. I have a silver ring on my right pinky."
3. Send 5 filler messages (trigger extraction)
4. Verify Tier 1: query `rp_session_context WHERE session_id=X AND tier=1` — should have name, origin, trait
5. Send 20 more filler messages (trigger arc summary at msg 25)
6. Verify Tier 3: query `rp_session_context WHERE tier=3` — should have 1 arc summary
7. Start NEW session with same character
8. Verify `build_rp_context()` injects facts from previous session
9. Check debug dict: `{"tier1": ~100, "tier3": ~150, "tier2": ~2500, "total": ~2750}`
10. Verify scorer hash unchanged: `1435531a3e145d1c6547e97a28392f9a`

## Sprint Breakdown

### Sprint 1: Foundation
- [ ] Safety branch
- [ ] Migration 028: `rp_session_context` table
- [ ] Database helper methods (insert, upsert, get, delete)
- [ ] Commit checkpoint

### Sprint 2: Tier 1 — Fact Extraction
- [ ] Regex extraction patterns
- [ ] LLM extraction with JSON parsing + fallback
- [ ] `extract_facts()` function
- [ ] `should_extract()` gate (every 5th message)
- [ ] Test: send messages with facts, verify stored in DB
- [ ] Commit checkpoint

### Sprint 3: Tier 3 — Arc Summaries
- [ ] Scene transition detection (regex)
- [ ] Arc summary LLM prompt (use exact prompt from above)
- [ ] Quality gate (30-200 token range, retry once)
- [ ] `extract_arc_summary()` function
- [ ] `should_arc_summarize()` gate
- [ ] Arc compression (oldest arcs merge if >800 tokens total)
- [ ] Test: send 25+ messages, verify arc summary generated
- [ ] Commit checkpoint

### Sprint 4: Context Assembly
- [ ] `build_rp_context()` function
- [ ] Token budget management (Tier 1 cap 200, Tier 3 cap 800, Tier 2 fills rest)
- [ ] Debug dict output with per-tier token counts
- [ ] Format output string with [Memory] headers
- [ ] Test: verify assembled context contains facts + arcs + recent messages
- [ ] Commit checkpoint

### Sprint 5: Wire into rp_chat.py
- [ ] Import rp_memory
- [ ] Inject context before prompt building
- [ ] Trigger extraction after response (non-blocking thread)
- [ ] Trigger arc summary on transitions (non-blocking thread)
- [ ] Test: full 20-message conversation with memory persistence
- [ ] Test: cross-session memory (new session recalls old facts)
- [ ] Verify scorer hash unchanged
- [ ] Commit checkpoint

### Sprint 6: Validation & Polish
- [ ] Run 50-message stress test
- [ ] Verify token budgets in debug logs
- [ ] Test edge cases: empty messages, very long messages, rapid fire
- [ ] Test cross-platform: verify web app still works (no regressions)
- [ ] Final commit

## Hard Constraints
- DO NOT touch scorer, agent, or any non-RP routes
- DO NOT change SSE streaming interface
- DO NOT modify mobile app frontend (context injection is backend-only)
- Scorer hash MUST remain: 1435531a3e145d1c6547e97a28392f9a
- Extraction threads must be non-blocking (user never waits for extraction)
- All LLM calls use existing Ollama connection (no new dependencies)
