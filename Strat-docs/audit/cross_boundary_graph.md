# Cross-Boundary Dependency Graph & Isolation Groups
## Generated: 2026-03-18 (Session 2 — post-pill-implementation)

---

## SCOPE BOUNDARY (CRITICAL)

**StratOS Web (110K lines) = OFF LIMITS.** Do not touch scorer, intelligence, web frontend, wizard, briefings, market panel, or any non-RP backend routes.

**StratOS Mobile (~20K lines) + RP Backend Routes = SAFE TO MODIFY.**

---

## ISOLATION GROUPS

### Group A: RP Chat System (SAFE)
```
backend/routes/rp_chat.py        (900+ lines — GOD FILE)
backend/routes/rp_memory.py      (405 lines)
backend/routes/gpu_manager.py    (240 lines)

DB: rp_messages, rp_edits, rp_suggestions, rp_feedback, rp_session_context
Ollama: YES (RP model — GPU exclusive)
SSE: {token}, {done, message_id, user_message_id}

Parallel with: C (mobile UI), D (tests — but not during test execution)
NOT with: B (shared character_cards reads), D (during Ollama test runs)
```

### Group B: Character Cards (SAFE)
```
backend/routes/character_cards.py  (420 lines)
backend/database.py                (ONLY card helpers, lines 810-900)

DB: character_cards, character_card_stats, character_card_ratings
Ollama: YES (auto-enrichment background thread)

Parallel with: C, D
NOT with: A (rp_chat reads character_cards)
CAUTION: database.py shared — only touch _CHARACTER_CARD_COLUMNS or card methods
```

### Group C: Mobile App (SAFE)
```
mobile/stratos-app/**  (entire mobile codebase)

DB: None (API only)
Ollama: None (API only)
SSE: Consumes {token, done} from /api/rp/chat

Parallel with: A, B, D (always safe — no backend file overlap)
CAUTION: SSE format changes in Group A break Group C
```

### Group D: Test Infrastructure (SAFE)
```
backend/tests/rp_regression_test.py
backend/tests/rp_baseline.json

DB: Reads character_cards (SELECT only)
Ollama: YES — EXCLUSIVE during test runs (~8 min)

Parallel with: C (code changes only, not test execution)
RULE: No Ollama work while tests run
```

### Group E: Image Gen (SAFE)
```
backend/routes/image_gen.py
mobile/stratos-app/app/imagegen/

GPU: ComfyUI (cannot coexist with Ollama)
NOT with: A, B, D (GPU contention)
```

### Group F: Migrations (SEQUENTIAL)
```
backend/migrations.py
backend/database.py (column whitelists)

RULE: Run FIRST in any session. ONE migration. Commit + restart before other work.
```

### Group X: StratOS Web (OFF LIMITS)
```
main.py, server.py, auth.py, sse.py
processors/*, fetchers/*, routes/agent*.py, routes/feeds.py,
routes/data_endpoints.py, routes/controls.py, routes/youtube*.py,
routes/persona_data.py, routes/media.py, routes/generate.py,
routes/wizard.py, routes/config.py, routes/auth.py, routes/dev*.py
frontend/*.js, frontend/*.css, frontend/*.html
```

### Group Y: Scorer/Training (OFF LIMITS)
```
processors/scorer_*.py, train_*.py, distill.py, export_training.py,
evaluate_scorer.py, data/v2_pipeline/*
```

---

## API CONTRACT (Mobile → Backend)

| Mobile File | Endpoint | Response | Critical Fields |
|-------------|----------|----------|-----------------|
| lib/chat.ts | POST /api/rp/chat | SSE stream | token, done, message_id, user_message_id |
| lib/chat.ts | POST /api/rp/regenerate | SSE stream | token, done, swipe_group_id, message_id |
| lib/chat.ts | POST /api/suggest-context | JSON | suggestions:[] OR suggestion:"" |
| lib/rp.ts | POST /api/rp/feedback | JSON | ok |
| lib/rp.ts | POST /api/rp/edit | JSON | ok, message_id, category |
| lib/rp.ts | POST /api/rp/branch | SSE stream | done, branch_id, from_branch |
| lib/rp.ts | POST /api/rp/director-note | JSON | ok, note |
| lib/characters.ts | POST /api/cards | JSON | ok, card_id |
| lib/characters.ts | PUT /api/cards/{id} | JSON | ok |
| lib/rp.ts | POST /api/image/generate | JSON | ok, image_id |

**RULE:** Changing any response field in this table requires updating the corresponding mobile file.

---

## GPU CONTENTION

```
Ollama (NUM_PARALLEL=1):
  RP model (Q8, 17GB) — rp_chat, rp_memory, regression tests
  qwen3.5:9b (15GB) — agent chat, wizard, enrichment, memory extraction
  scorer-v2.2 (9.5GB) — scan pipeline

  MAX_LOADED_MODELS=2: scorer + qwen3.5 coexist.
  RP model evicts BOTH when loaded (17GB alone).

ComfyUI (CHROMA, 22GB peak):
  Cannot coexist with ANY Ollama model.
  gpu_manager.py handles swap.

RULE: Only ONE GPU consumer at a time during development/testing.
```

---

## SPRINT TASK PARALLELISM

```
Phase A: Max 2 agents
  Agent 1: A6 → A2 → A3 (sequential — all touch rp_chat.py)
  Agent 2: A5 (test infra — independent code changes)
  A1: AFTER Agent 1 (needs final baseline)

Phase B remaining: Max 2 agents
  Agent 1: B1 (migration) → B2 (backend)
  Agent 2: B3 (auto-enrichment — after B1 commits)

Phase C: Max 2 agents
  Agent 1: C1 backend (SSE metadata) + C4 (arc frequency)
  Agent 2: C1 frontend (mock data first, wire after backend)
```

---

## GOD FILE: rp_chat.py (900+ lines)

Recommended split for future parallelism:
1. `rp_prompt.py` — system prompt, archetype detection, format rotation
2. `rp_injection.py` — _build_turn_injection() + 11 hint functions
3. `rp_stream.py` — _stream_ollama(), think block handling
4. `rp_chat.py` — route handlers only

This unlocks 2 agents working on prompt engineering + route logic simultaneously.

---

## NO CIRCULAR DEPENDENCIES

All imports are unidirectional. Clean dependency tree:
```
server.py → routes/* → helpers.py
rp_chat.py → rp_memory.py → (no further local imports)
rp_chat.py → gpu_manager.py → (no further local imports)
character_cards.py → helpers.py → (no further local imports)
```
