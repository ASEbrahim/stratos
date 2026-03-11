# Sprint 4: Agent UX Overhaul Report

## Date: 2026-03-11
## Safety branch: `pre-sprint4`

---

## Completed Phases

### Phase 1: Dynamic LLM-Generated Suggestion Chips
**Commit:** `13fad4a`
- Backend: After each agent response, lightweight LLM call (think:false, 120 tokens) generates 3 contextual follow-up suggestions
- Sent as `suggestions` SSE event before `done`
- Frontend: Prefers dynamic suggestions, falls back to rule-based chips
- Works in all three response paths: free mode, no-tools, tool-call loop

### Phase 1.5: Clickable Numbered Options
**Commit:** `d5b5442`
- Detects numbered option lists preceded by question context (2+ consecutive items)
- Renders as persona-themed clickable buttons with number badges
- Click auto-sends the option text, grays out alternatives
- Smart detection avoids false positives on regular numbered lists

### Phase 2: Conversation Management
**Commit:** `ecb7327`
- New Chat button in header
- Conversation tabs bar with horizontal scroll
- Per-persona conversation history (max 10 each)
- Auto-titles from first user message
- Delete conversations via hover X
- Migrates old single-history localStorage format
- Persona switching loads that persona's conversations

### Phase 4: Mode Toggle Removal
**Commit:** `55c6200`
- Removed structured/free mode toggle (was confusing, free disabled all tools)
- Always uses structured mode with persona-filtered tools

### Phase 4.5: Agent Fullscreen Overhaul
**Commit:** `356a5a9`
- CSS cubic-bezier transitions for smooth enter/exit
- Centered max-width 800px layout (like Claude/ChatGPT)
- Auto-focus input on enter, Escape key to exit
- Proper message area sizing with generous padding

### Phase 6: Games Scenario UI Redesign
**Commit:** `052b0b8`
- Replaced dropdown with horizontal scrollable scenario cards
- Cards show name, description preview, active indicator
- Empty state: "Create Your First World" with genre quick-start buttons
- Action buttons for world files and delete below cards

### Phase 7: Persona Context Clarity
**Commit:** `7018d52`
- Persona switch shows brief animated transition (icon + name)
- Context and file browser button tooltips update with persona name

### Phase 8: Market Focus Mode Auto-Update
**Commits:** `783dfe4`, `740d5fc`
- Extended auto-refresh to ALL timeframes (was only 1m/5m)
- Immediate fresh data fetch on entering focus mode
- Restarts refresh interval on ticker switch
- 30s for 1m, 60s for all other timeframes

### Phase 9: Streaming Cancel Button
**Commit:** `b9c6e67`
- Send button becomes stop button (square icon) during streaming
- Uses AbortController to cancel the fetch
- Restores to send arrow when done

### Video Insights UI Redesign
**Commit:** `e23127c`
- Purple scholarly theme with lens-specific icons
- Video title in header, lens count meta text
- "Ask Agent" button to query Strat Agent about the video
- Animated loading state, better empty state
- Wider panel (580px) for readability

### Mobile CSS
**Commit:** `1c1f4d2`
- 44px min-height touch targets for option buttons
- Hidden scrollbars for chips and tabs
- fadeIn keyframe animation

---

## Remaining (Sprint 5)

1. **Phase 3: Message editing & timeline branching** — advanced feature, lowest priority
2. **Phase 10: Playwright tests** for all Sprint 4 features
3. **TTS (Piper)**: Backend processor exists, binary installed. Need API endpoint + agent "speak" button.
4. **preference_signals table**: Exists but not read during scoring — wire up in scoring pipeline.

---

## Commit Log

```
13fad4a feat: dynamic LLM-generated suggestion chips (Phase 1)
d5b5442 feat: clickable numbered options in agent responses (Phase 1.5)
55c6200 fix: remove confusing structured/free mode toggle (Phase 4)
356a5a9 feat: agent fullscreen overhaul with smooth transitions (Phase 4.5)
783dfe4 fix: focus mode auto-refresh for all timeframes (Phase 8)
ecb7327 feat: conversation management with per-persona history (Phase 2)
740d5fc fix: restart focus mode auto-refresh on ticker switch
1c1f4d2 style: mobile-safe CSS for Sprint 4 features
052b0b8 feat: scenario UI redesign with cards and genre presets (Phase 6)
7018d52 feat: persona switch transition and context clarity (Phase 7)
b9c6e67 feat: streaming cancel button on agent send (Phase 9)
e23127c feat: video insights UI redesign with agent integration
```
