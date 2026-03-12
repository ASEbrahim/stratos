# StratOS Agent — Comprehensive Test Pipeline

> Generated: March 12, 2026 | Updated: March 12, 2026
> Covers: All 6 personas, multi-persona combos, GM/RP modes, file/context isolation, tool usage

---

## 4-Tier Automated Execution

| Tier | Tests | Time | Runner | Files |
|------|-------|------|--------|-------|
| **SMOKE** | 11 (16 incl. persona variants) | ~2 min | `npx playwright test tests/browser/smoke.spec.js` | `smoke.spec.js` |
| **AUTOMATED** | 53 | ~8 min | `npx playwright test tests/browser/auto-*.spec.js --workers=2` | 7 `auto-*.spec.js` files |
| **SEMI-AUTO** | 12 | ~4 min | `npx playwright test tests/browser/semi-auto-capture.spec.js --workers=1` | `semi-auto-capture.spec.js` |
| **MANUAL** | ~42 | varies | Follow checklist below | This document |

### Execution Order
1. **SMOKE** — Run before every commit. All must pass.
2. **AUTOMATED** — Run before merge to main. 53 deterministic tests.
3. **SEMI-AUTO** — Run weekly. Captures output to `/tmp/agent-test-review/` for human QA.
4. **MANUAL** — Run before release. Follow the detailed checklist below.

### Quick Commands
```bash
# Pre-commit gate (must pass)
npx playwright test tests/browser/smoke.spec.js

# Full automated suite
npx playwright test tests/browser/auto-*.spec.js --workers=2

# Semi-auto captures (review /tmp/agent-test-review/ after)
npx playwright test tests/browser/semi-auto-capture.spec.js --workers=1

# Everything at once
npx playwright test tests/browser/smoke.spec.js tests/browser/auto-*.spec.js tests/browser/semi-auto-capture.spec.js --workers=2
```

---

## Table of Contents

1. [Infrastructure Pre-checks](#1-infrastructure-pre-checks)
2. [Intelligence Persona](#2-intelligence-persona)
3. [Market Persona](#3-market-persona)
4. [Scholarly Persona](#4-scholarly-persona)
5. [Gaming Persona — GM Mode](#5-gaming-persona--gm-mode)
6. [Gaming Persona — Immersive RP Mode](#6-gaming-persona--immersive-rp-mode)
7. [Anime Persona (Stub)](#7-anime-persona)
8. [TCG Persona (Stub)](#8-tcg-persona)
9. [Multi-Persona Combinations](#9-multi-persona-combinations)
10. [File & Context Isolation](#10-file--context-isolation)
11. [Conversation Management](#11-conversation-management)
12. [Edge Cases & Error Handling](#12-edge-cases--error-handling)
13. [TTS Pipeline](#13-tts-pipeline)
14. [Mobile Agent Pipeline](#14-mobile-agent-pipeline)

---

## 1. Infrastructure Pre-checks

Run these before any persona test. All must pass.

| ID | Test | Command / Action | Expected |
|----|------|------------------|----------|
| I-1 | Ollama reachable | `curl http://localhost:11434/api/tags` | 200, lists models |
| I-2 | Agent status | `GET /api/agent-status` | `{"available": true, "model": "qwen3.5:9b"}` |
| I-3 | Persona list | `GET /api/agent-personas` | 6 personas: intelligence, market, scholarly, gaming, anime, tcg |
| I-4 | Search provider | `GET /api/config` or check `config.yaml` | `search.provider` is `serper` or `searxng`, key present |
| I-5 | Server running | `curl http://localhost:8080` | 200, HTML page |
| I-6 | Auth token valid | Login via UI, check `X-Auth-Token` header works | Token accepted on API calls |

```bash
# Quick infrastructure validation
curl -s http://localhost:8080/api/agent-status | python3 -m json.tool
curl -s http://localhost:8080/api/agent-personas | python3 -m json.tool
```

---

## 2. Intelligence Persona

**Identity:** STRAT AGENT — strategic intelligence dashboard assistant
**Tools:** `web_search`, `search_feed`, `manage_watchlist`, `manage_categories`, `search_files`, `read_document`
**Context loaded:** News feed (top 30), market data, briefing, historical data, recent feed (24h), feedback/saved articles

### 2.1 Basic Chat

| ID | Prompt | Expected Behavior | Validates |
|----|--------|-------------------|-----------|
| INT-1 | "Hello" | Friendly greeting, mentions capabilities (search, watchlist, categories). Under 200 words. | Basic response, conciseness rule |
| INT-2 | "What's in my feed today?" | References actual feed data (article titles, scores, categories). Uses context data, NOT web search. | Feed context injection |
| INT-3 | "Summarize the top 3 signals" | Lists top-scored articles with titles, scores, and brief analysis. 3-5 bullets. | Ranking, formatting |
| INT-4 | "What happened last week with oil?" | Triggers history keyword detection → searches news DB → references historical articles. | History auto-search |

### 2.2 Tool Usage

| ID | Prompt | Expected Tool | Expected Behavior |
|----|--------|---------------|-------------------|
| INT-5 | "Search the web for latest NVIDIA earnings" | `web_search` | Status indicator shows "🔍 web_search...", returns real search results. |
| INT-6 | "Search my feed for articles about Kuwait" | `search_feed` | Searches local news DB. Returns matching articles with titles and scores. |
| INT-7 | "Add TSLA to my watchlist" | `manage_watchlist` | Confirms ticker added. Verify via `GET /api/config` that TSLA appears in tickers. |
| INT-8 | "Remove TSLA from my watchlist" | `manage_watchlist` | Confirms removal. Verify ticker gone from config. |
| INT-9 | "Add a category called 'AI Chips' with keywords semiconductor, nvidia, amd" | `manage_categories` | Confirms category created. Verify in config. |
| INT-10 | "Remove the AI Chips category" | `manage_categories` | Confirms removal. |

### 2.3 Response Quality

| ID | Check | Pass Criteria |
|----|-------|---------------|
| INT-11 | Response length | Under 200 words for simple questions |
| INT-12 | Data-first approach | Uses feed/market context before suggesting web search |
| INT-13 | Persona routing hints | When asked about stocks specifically → suggests "Switch to Market persona for deeper analysis" |
| INT-14 | Follow-up suggestions | SSE stream ends with `{"suggestions": [...]}` containing 3 relevant follow-ups |

---

## 3. Market Persona

**Identity:** STRAT MARKET ANALYST — data-driven financial analyst
**Tools:** `manage_watchlist`, `search_feed`, `web_search`
**Context loaded:** Full market data (all timeframes), finance-tagged news, briefing market summary

### 3.1 Basic Chat

| ID | Prompt | Expected Behavior | Validates |
|----|--------|-------------------|-----------|
| MKT-1 | "Hello" | Mentions watchlist tickers by name, current prices. Professional tone. | Greeting, market awareness |
| MKT-2 | "How is NVDA doing?" | Leads with price, % change, trend direction. References actual market data from context. | Data-first rule |
| MKT-3 | "Compare NVDA and TSLA" | Side-by-side comparison with prices, changes, trends. Non-speculative. | Multi-ticker analysis |
| MKT-4 | "Should I buy NVDA?" | Refuses investment advice. Presents data objectively. | Non-speculative rule |

### 3.2 Tool Usage

| ID | Prompt | Expected Tool | Expected Behavior |
|----|--------|---------------|-------------------|
| MKT-5 | "Add BTC-USD to my watchlist" | `manage_watchlist` | Adds ticker. Verify in config. |
| MKT-6 | "What's the latest news on Bitcoin?" | `web_search` or `search_feed` | Searches and returns finance-relevant results. |
| MKT-7 | "Search my feed for banking news" | `search_feed` | Searches local DB with finance/banking filter. |

### 3.3 Response Quality

| ID | Check | Pass Criteria |
|----|-------|---------------|
| MKT-8 | Leads with numbers | Every market response starts with price/% data |
| MKT-9 | No `manage_categories` tool | Market persona should NOT have category management |
| MKT-10 | No `search_files` tool | Market persona should NOT search uploaded files |
| MKT-11 | Persona routing | When asked about academic topics → suggests switching to Scholarly |
| MKT-12 | No unsolicited feed search | Does NOT auto-search news unless user asks |

---

## 4. Scholarly Persona

**Identity:** STRAT SCHOLAR — research assistant with YouTube lecture access
**Tools:** `search_insights`, `list_channels`, `get_video_summary`, `search_narrations`, `search_files`, `read_document`, `web_search`, `read_url`
**Context loaded:** YouTube channels, recent videos, user context, file summaries, YouTube knowledge base

### 4.1 Basic Chat

| ID | Prompt | Expected Behavior | Validates |
|----|--------|-------------------|-----------|
| SCH-1 | "Hello" | Mentions academic focus areas. Professional scholarly tone. | Greeting |
| SCH-2 | "Tell me about the Battle of Badr" | Uses search tools before answering from memory. Cites sources if available. Proper Arabic transliteration. | Research-first rule, transliteration |
| SCH-3 | "What videos do I have on Islamic history?" | Calls `search_insights` or `list_channels`. Returns channel/video info from DB. | YouTube integration |
| SCH-4 | "Summarize the latest video from [channel]" | Calls `get_video_summary`. Returns structured summary. | Video summary tool |

### 4.2 Tool Usage

| ID | Prompt | Expected Tool | Expected Behavior |
|----|--------|---------------|-------------------|
| SCH-5 | "Search my files for notes on philosophy" | `search_files` | Searches scholarly-scoped uploaded files only. |
| SCH-6 | "Read the document 'research_notes.md'" | `read_document` | Returns file content from scholarly persona scope. |
| SCH-7 | "Search for narrations about patience" | `search_narrations` | Searches YouTube narration insights. States VERIFIED/UNVERIFIED status. |
| SCH-8 | "Read this URL: https://example.com/article" | `read_url` | Fetches and summarizes URL content. |
| SCH-9 | "Search the web for recent papers on quantum computing" | `web_search` | Web search with academic framing. |

### 4.3 Response Quality

| ID | Check | Pass Criteria |
|----|-------|---------------|
| SCH-10 | Citation behavior | Cites video titles/channels when referencing YouTube data |
| SCH-11 | Narration verification | Clearly states VERIFIED or UNVERIFIED for hadith/narrations |
| SCH-12 | No fabricated citations | Does not invent source names or hadith numbers |
| SCH-13 | Length | 2-4 paragraphs max for academic questions |
| SCH-14 | Persona routing | When asked about stock prices → suggests Market persona |
| SCH-15 | No `manage_watchlist` | Scholarly persona should NOT have watchlist tools |

---

## 5. Gaming Persona — GM Mode

**Identity:** STRAT GAMES — Game Master and narrator engine
**Tools:** `search_files`, `read_document`
**Context loaded:** World bible, active scenario state, scenario data, file summaries
**Mode:** `rp_mode: "gm"`

### 5.1 Scenario Setup

| ID | Action | Expected | Validates |
|----|--------|----------|-----------|
| GM-1 | Create scenario "Dark_Realm" via UI or API | `POST /api/scenarios/create` returns `{"ok": true}`. Scenario appears in dropdown. | Scenario creation |
| GM-2 | Set scenario active | `POST /api/scenarios/activate` → scenario bar shows "Dark_Realm" | Scenario activation |
| GM-3 | Verify GM mode is default | Scenario bar shows GM mode toggle, GM is selected | Default mode |

### 5.2 GM Narration

| ID | Prompt | Expected Behavior | Validates |
|----|--------|-------------------|-----------|
| GM-4 | "Describe the starting area" | Third-person narration. Vivid setting description. Presents 3-4 numbered choices at the end. | GM narration style, numbered choices |
| GM-5 | "I choose option 2" | Continues narrative based on choice. May include dice roll for uncertain outcome. Status block with HP/inventory. | Choice handling, dice rolls |
| GM-6 | "I attack the goblin with my sword" | Combat narration. Dice roll result. Damage calculation. Updated HP in status block. | Combat, stat tracking |
| GM-7 | "Check my inventory" | Displays current inventory, stats, conditions in a formatted status block. | Stat block rendering |
| GM-8 | "OOC: Can you add a magic system?" | Responds out of character (not in narrative voice). Discusses game mechanics directly. | OOC prefix handling |

### 5.3 GM Response Format

| ID | Check | Pass Criteria |
|----|-------|---------------|
| GM-9 | Third-person narration | Never uses "I" as the narrator. Uses "you" for the player. |
| GM-10 | NPC dialogue tags | NPCs speak with dialogue tags: `"Stand back!" the guard shouted.` |
| GM-11 | Numbered choices | Most responses end with 2-4 numbered options |
| GM-12 | Stat blocks present | Combat/exploration responses include HP, ATK, DEF, inventory |
| GM-13 | Dice roll notation | Uses notation like `[Roll: 14 + 3 = 17, Success]` |
| GM-14 | Entity roster loaded | If entities exist for this scenario, GM references them by name/personality |
| GM-15 | World bible consistency | Responses align with world description set in scenario |

### 5.4 Scenario Persistence

| ID | Action | Expected | Validates |
|----|--------|----------|-----------|
| GM-16 | Send 3 messages, refresh page | Conversation preserved. Scenario still active. | State persistence |
| GM-17 | Switch to intelligence, switch back to gaming | Scenario still active. GM mode still selected. Chat history preserved. | Cross-persona persistence |
| GM-18 | Create second scenario "Space_Opera" | Both scenarios in dropdown. Can switch between them. | Multi-scenario |

---

## 6. Gaming Persona — Immersive RP Mode

**Identity:** Creative roleplay partner who inhabits characters
**Tools:** `search_files`, `read_document`
**Mode:** `rp_mode: "immersive"`

### 6.1 Entity Setup

| ID | Action | Expected | Validates |
|----|--------|----------|-----------|
| RP-1 | Create entity "Sakura" (type: character) with personality/speaking style | `POST /api/personas/gaming/entities` returns `{"ok": true}` | Entity creation |
| RP-2 | Set Sakura as active NPC in UI | NPC selector shows Sakura selected | NPC selection |
| RP-3 | Switch to Immersive mode | Scenario bar shows "Immersive" toggle active. No stat blocks from here on. | Mode switch |

### 6.2 Immersive RP Conversation

| ID | Prompt | Expected Behavior | Validates |
|----|--------|-------------------|-----------|
| RP-4 | "Hello, how are you?" | Responds AS Sakura in first person. Uses character's personality and speaking style. Includes `*italic actions*` for physical descriptions. | In-character response, italics |
| RP-5 | "Tell me about yourself" | Responds with information consistent with Sakura's `identity_md` and `personality_md`. Uses `**bold**` for emphasis. | Identity consistency, bold |
| RP-6 | "What do you think about the war?" | Responds based on character's knowledge and personality. May reference `knowledge_md`. | Knowledge integration |
| RP-7 | "*leans forward* What's your secret?" | Responds to action cues. Includes own actions in italics. Maintains character voice. | Action parsing |

### 6.3 Immersive RP Format

| ID | Check | Pass Criteria |
|----|-------|---------------|
| RP-8 | First-person voice | Character speaks as "I", not described in third person |
| RP-9 | Italic actions | Physical actions wrapped in `*italics*` |
| RP-10 | Bold emphasis | Important words/dramatic moments use `**bold**` |
| RP-11 | NO stat blocks | No HP, ATK, DEF, inventory displays |
| RP-12 | NO numbered choices | No "1. 2. 3." option lists |
| RP-13 | NO emoji headers | No emoji-prefixed section headers |
| RP-14 | NO GM narration | No third-person narrator voice |

### 6.4 Entity Memory Auto-Update

| ID | Action | Expected | Validates |
|----|--------|----------|-----------|
| RP-15 | Send 3 messages to Sakura about a specific topic | After each exchange, background thread updates `memory_md`, `relationship_md`, `knowledge_md` | Auto memory update |
| RP-16 | Check entity via API: `GET /api/personas/gaming/entities/Sakura` | `memory_md` should reference the conversation topic. `relationship_md` may show rapport changes. | Memory persistence |
| RP-17 | Start new conversation with Sakura | Character should reference previous interactions based on updated memory | Memory recall |

### 6.5 GM ↔ Immersive Switching

| ID | Action | Expected | Validates |
|----|--------|----------|-----------|
| RP-18 | In GM mode, switch to Immersive | Next response should be in-character, first-person. No stat blocks. | Mode switch mid-session |
| RP-19 | In Immersive mode, switch to GM | Next response should be third-person narration with stat blocks and choices. | Mode switch back |
| RP-20 | Switch NPC from Sakura to new entity "Gojo" | Responses now use Gojo's personality/speaking style. Sakura's memory preserved. | NPC switching |

---

## 7. Anime Persona

**Identity:** Stub/placeholder
**Tools:** None (`[]`)
**Status:** "Coming soon" — uses generic `_stub_prompt`

| ID | Prompt | Expected Behavior | Validates |
|----|--------|-------------------|-----------|
| ANI-1 | "Hello" | Responds conversationally about anime/manga. Mentions "coming soon" or limited capabilities. | Stub behavior |
| ANI-2 | "What's the best anime of 2025?" | Gives an opinion based on training data. No tool calls. No status indicators. | No-tools streaming |
| ANI-3 | "Search for latest anime news" | Does NOT call web_search (not in tool set). Responds with general knowledge only. | Tool restriction |
| ANI-4 | "Add NVDA to watchlist" | Cannot manage watchlist. Responds it doesn't have that capability. | Tool restriction |
| ANI-5 | Check SSE stream format | Tokens stream directly (no tool loop). Temperature 0.5. `{"suggestions": [...]}` at end. | Stub streaming path |

---

## 8. TCG Persona

**Identity:** Stub/placeholder
**Tools:** None (`[]`)
**Status:** "Coming soon" — uses generic `_stub_prompt`

| ID | Prompt | Expected Behavior | Validates |
|----|--------|-------------------|-----------|
| TCG-1 | "Hello" | Responds conversationally about TCGs (Magic, Pokemon, Yu-Gi-Oh). | Stub behavior |
| TCG-2 | "Build me a Blue-Eyes White Dragon deck" | Gives deck advice based on training data. No tool calls. | No-tools path |
| TCG-3 | "Search for latest card set releases" | Cannot web search. General knowledge only. | Tool restriction |
| TCG-4 | Check response temperature | Responses should feel slightly more creative (temp 0.5 vs 0.4 for tool personas). | Temperature setting |

---

## 9. Multi-Persona Combinations

**Mechanic:** Up to 3 personas selected. First = primary (base prompt). Tools and contexts merged.

### 9.1 Two-Persona Combinations

| ID | Combo | Test Prompt | Expected Behavior | Validates |
|----|-------|-------------|-------------------|-----------|
| MP-1 | Intelligence + Market | "What's happening with NVDA and is there any news about it?" | Uses both market data AND news feed context. Has all tools from both. | Tool merge, context merge |
| MP-2 | Intelligence + Scholarly | "Find scholarly articles about Kuwait's economy in my feed and videos" | Searches news feed AND YouTube insights. Has `search_feed` + `search_insights`. | Cross-domain search |
| MP-3 | Market + Scholarly | "How does the current oil price relate to Gulf economic research?" | Combines market data with scholarly context. Has market tools + scholarly tools. | Mixed context |
| MP-4 | Intelligence + Gaming | "What news do I have about gaming industry stocks?" | Uses news feed context + gaming file access. | Unusual combo works |
| MP-5 | Scholarly + Gaming | "Research medieval history for my fantasy scenario" | Scholarly search tools + gaming file access. | Research-for-gaming |
| MP-6 | Gaming + Anime | "Help me build an anime-themed gaming scenario" | Gaming tools (search_files, read_document) + anime stub context. | Stub + active combo |

### 9.2 Three-Persona Combinations

| ID | Combo | Test Prompt | Expected Behavior | Validates |
|----|-------|-------------|-------------------|-----------|
| MP-7 | Intelligence + Market + Scholarly | "Analyze oil markets, find related news, and check academic sources" | Full tool set (11 tools). All three contexts loaded. | Max tool merge |
| MP-8 | Intelligence + Gaming + TCG | "Find news about tabletop gaming industry" | Intelligence tools + gaming file tools. TCG adds nothing. | Stub adds no tools |

### 9.3 Multi-Persona Validation

| ID | Check | How to Verify | Pass Criteria |
|----|-------|---------------|---------------|
| MP-9 | Tool merge correctness | Send a prompt that requires a tool from each persona | Both tools should be available and callable |
| MP-10 | Context headers present | Check SSE stream or server logs | System prompt contains `[INTELLIGENCE DATA]`, `[MARKET DATA]`, etc. |
| MP-11 | Cross-persona note | Check system prompt | Contains "You also have context from: X, Y" |
| MP-12 | Max 3 cap | Try selecting 4 personas via API | Only first 3 are used |
| MP-13 | Primary persona base | First persona in array determines base prompt identity | Response style matches primary persona |

---

## 10. File & Context Isolation

### 10.1 File Isolation Matrix

| ID | Action | Expected | Validates |
|----|--------|----------|-----------|
| FI-1 | Upload `test_intel.txt` with `X-Persona: intelligence` | File visible in intelligence file list | Upload with persona |
| FI-2 | List files with `persona=intelligence` | Shows `test_intel.txt` | Correct persona filter |
| FI-3 | List files with `persona=gaming` | Does NOT show `test_intel.txt` | Cross-persona isolation |
| FI-4 | List files with `persona=scholarly` | Does NOT show `test_intel.txt` | Cross-persona isolation |
| FI-5 | Upload `test_gaming.txt` with `X-Persona: gaming` | Only visible in gaming file list | Second persona upload |
| FI-6 | Upload `test_shared.txt` with NO `X-Persona` header | Visible from ALL personas (empty persona = shared) | Shared file behavior |
| FI-7 | Ask Intelligence "search my files for test" | Returns `test_intel.txt` only (+ shared). NOT `test_gaming.txt`. | Tool-level isolation |
| FI-8 | Ask Gaming "search my files for test" | Returns `test_gaming.txt` only (+ shared). NOT `test_intel.txt`. | Tool-level isolation |

### 10.2 Filesystem Browser Isolation

| ID | Action | Expected | Validates |
|----|--------|----------|-----------|
| FI-9 | Write file via API: `POST /api/persona-files/write` `{persona: "scholarly", path: "/notes.md", content: "test"}` | File created at `data/users/{pid}/context/scholarly/notes.md` | Filesystem write |
| FI-10 | List files for scholarly: `GET /api/persona-files?persona=scholarly&path=/` | Shows `notes.md` | Filesystem list |
| FI-11 | List files for gaming: `GET /api/persona-files?persona=gaming&path=/` | Does NOT show `notes.md` | Filesystem isolation |
| FI-12 | Create folder: `POST /api/persona-files/mkdir` `{persona: "gaming", path: "/lore"}` | Directory created. Visible in gaming only. | Folder creation |
| FI-13 | Path traversal attempt: `GET /api/persona-files/read?persona=gaming&path=../../scholarly/notes.md` | Rejected (403 or empty). | Security guard |

### 10.3 Context Isolation

| ID | Action | Expected | Validates |
|----|--------|----------|-----------|
| FI-14 | Save context: `POST /api/persona-context` `{persona: "intelligence", key: "system_context", content: "Focus on oil markets"}` | Saved for intelligence only | Context save |
| FI-15 | Get context for intelligence | Returns "Focus on oil markets" | Context retrieval |
| FI-16 | Get context for gaming | Returns default template (NOT "Focus on oil markets") | Context isolation |
| FI-17 | Save context for gaming: "Fantasy world with dragons" | Only gaming gets this context | Cross-persona isolation |
| FI-18 | Chat with Intelligence: "What are my custom instructions?" | Should reference "Focus on oil markets" | Context injection |
| FI-19 | Chat with Gaming: "What's my world about?" | Should reference "Fantasy world with dragons" | Context injection |

### 10.4 Auto-Creation Verification

| ID | Action | Expected | Validates |
|----|--------|----------|-----------|
| FI-20 | First-ever file write to anime persona | Directory `data/users/{pid}/context/anime/` auto-created | On-demand dir creation |
| FI-21 | First context save for TCG | Row created in `persona_context` table | On-demand DB row |
| FI-22 | Get context for persona with NO saved context | Returns default template string (NOT empty, NOT error) | Default fallback |
| FI-23 | Save context, then save again | Version created in `_versions/` directory. Max 5 versions. | Version history |

---

## 11. Conversation Management

| ID | Action | Expected | Validates |
|----|--------|----------|-----------|
| CV-1 | Switch to a persona with no conversations | Auto-creates first conversation via `POST /api/conversations` | Auto-create |
| CV-2 | Create 11 conversations for one persona | Oldest conversation archived (check count ≤ 10 active) | Archival limit |
| CV-3 | Send messages, switch persona, switch back | Messages preserved. Active conversation restored. | Cross-persona persistence |
| CV-4 | Rename conversation (double-click tab) | Title updates in DB and UI | Rename |
| CV-5 | Delete conversation | Removed from list. If it was active, another becomes active. | Delete |
| CV-6 | Create new chat while one is active | Old chat deactivated. New chat is active and empty. | New chat flow |
| CV-7 | Verify conversation title auto-set | After first message, conversation title = first 40 chars of user message | Auto-title |
| CV-8 | Check message limit | Conversation stores last 40 messages | History pruning |

---

## 12. Edge Cases & Error Handling

| ID | Test | Expected | Validates |
|----|------|----------|-----------|
| EC-1 | Send empty message | Not sent. Input validation prevents empty submit. | Frontend validation |
| EC-2 | Send message while streaming | Disabled. Send button shows stop icon. | Streaming lock |
| EC-3 | Cancel mid-stream (click stop) | Stream aborted. Partial response preserved. | Abort handling |
| EC-4 | Send very long message (>5000 chars) | Should still work (server truncates if needed) | Long input |
| EC-5 | Kill Ollama mid-response | Error toast: "The AI model appears to be offline..." | Error feedback |
| EC-6 | Invalid persona name in API | Graceful fallback to intelligence or error | Invalid input |
| EC-7 | Expired auth token | 401 response. Toast: "session expired, please refresh" | Auth enforcement |
| EC-8 | Simultaneous requests from same user | Second request waits or fails gracefully | Concurrency |
| EC-9 | Network disconnect during streaming | Error toast with network error message | Network failure |
| EC-10 | Special characters in message (`<script>`, SQL injection) | Properly escaped. No XSS. No SQL injection. | Security |

---

## 13. TTS Pipeline

| ID | Test | Expected | Validates |
|----|------|----------|-----------|
| TTS-1 | Click speak button on agent response | Audio plays. Button changes to stop icon. | Basic TTS |
| TTS-2 | Click speak again while playing | Audio stops. Button resets. | Toggle behavior |
| TTS-3 | TTS with Piper not installed | Toast: "TTS unavailable — Piper not installed" | 503 handling |
| TTS-4 | TTS with expired session | Toast: "TTS failed — session expired" | 401 handling |
| TTS-5 | Disable TTS in Settings | Speak buttons hidden (`body.tts-disabled .speak-btn { display: none }`) | Toggle setting |
| TTS-6 | Re-enable TTS | Speak buttons visible again | Setting restore |
| TTS-7 | TTS on long text (>5000 chars) | Server truncates to 5000. Audio still plays. | Length limit |

---

## 14. Mobile Agent Pipeline

| ID | Test | Expected | Validates |
|----|------|----------|-----------|
| MOB-1 | Open mobile agent (bottom nav) | Full-page view with header, persona row, messages, textarea input | Basic open |
| MOB-2 | Send message | Message appears instantly (hook-based, no polling). Response streams in real-time. | Hook sync |
| MOB-3 | Shift+Enter in textarea | Creates newline, does NOT send | Textarea behavior |
| MOB-4 | Enter in textarea | Sends message | Send shortcut |
| MOB-5 | Suggestion chips visible on empty chat | Chips shown. Clicking sends that prompt. Chips hide after first message. | Suggestions |
| MOB-6 | Toggle conversation tabs | Conv tabs appear/disappear. Can switch conversations. | Conv tabs |
| MOB-7 | Switch persona via persona row | Persona switches. Messages update. Active persona highlighted. | Persona switching |
| MOB-8 | New chat button | Creates new conversation. Messages clear. Suggestions reappear. | New chat |
| MOB-9 | Clear chat button | Conversation cleared. Welcome screen shown. | Clear |
| MOB-10 | Back button (Android) | Closes mobile agent view. Returns to previous screen. | History popstate |
| MOB-11 | Close mobile agent, reopen | State preserved. Same conversation shown. | Persistence |
| MOB-12 | Long response streaming | Content syncs every 300ms (throttled). Scrolls to bottom. | Throttled sync |

---

## Execution Order

**Recommended sequence for full validation:**

```
1. Infrastructure pre-checks (I-1 → I-6)
2. Intelligence (INT-1 → INT-14)
3. Market (MKT-1 → MKT-12)
4. Scholarly (SCH-1 → SCH-15)
5. Gaming GM (GM-1 → GM-18)
6. Gaming Immersive RP (RP-1 → RP-20)
7. Anime (ANI-1 → ANI-5)
8. TCG (TCG-1 → TCG-4)
9. Multi-Persona combos (MP-1 → MP-13)
10. File & Context isolation (FI-1 → FI-23)
11. Conversation management (CV-1 → CV-8)
12. Edge cases (EC-1 → EC-10)
13. TTS (TTS-1 → TTS-7)
14. Mobile (MOB-1 → MOB-12)
```

**Total test cases: 156**

---

## API Quick Reference for Test Execution

```bash
# Set auth token
TOKEN="your-auth-token"
HEADERS="-H 'Content-Type: application/json' -H 'X-Auth-Token: $TOKEN'"

# Intelligence chat
curl -N -X POST http://localhost:8080/api/agent-chat \
  -H "Content-Type: application/json" -H "X-Auth-Token: $TOKEN" \
  -d '{"message":"Hello","history":[],"persona":"intelligence","mode":"structured"}'

# Market chat
curl -N -X POST http://localhost:8080/api/agent-chat \
  -H "Content-Type: application/json" -H "X-Auth-Token: $TOKEN" \
  -d '{"message":"How is NVDA?","history":[],"persona":"market","mode":"structured"}'

# Scholarly chat
curl -N -X POST http://localhost:8080/api/agent-chat \
  -H "Content-Type: application/json" -H "X-Auth-Token: $TOKEN" \
  -d '{"message":"Tell me about Badr","history":[],"persona":"scholarly","mode":"structured"}'

# Gaming GM chat
curl -N -X POST http://localhost:8080/api/agent-chat \
  -H "Content-Type: application/json" -H "X-Auth-Token: $TOKEN" \
  -d '{"message":"Describe the starting area","history":[],"persona":"gaming","mode":"structured","rp_mode":"gm"}'

# Gaming Immersive RP chat
curl -N -X POST http://localhost:8080/api/agent-chat \
  -H "Content-Type: application/json" -H "X-Auth-Token: $TOKEN" \
  -d '{"message":"Hello Sakura","history":[],"persona":"gaming","mode":"structured","rp_mode":"immersive","active_npc":"sakura"}'

# Multi-persona (Intelligence + Market)
curl -N -X POST http://localhost:8080/api/agent-chat \
  -H "Content-Type: application/json" -H "X-Auth-Token: $TOKEN" \
  -d '{"message":"NVDA news and price?","history":[],"personas":["intelligence","market"],"mode":"structured"}'

# File upload (persona-scoped)
curl -X POST http://localhost:8080/api/files/upload \
  -H "X-Auth-Token: $TOKEN" -H "X-Filename: test.txt" -H "X-Persona: scholarly" \
  --data-binary "This is a test file for scholarly persona"

# File list (persona-scoped)
curl -X POST http://localhost:8080/api/files/list \
  -H "Content-Type: application/json" -H "X-Auth-Token: $TOKEN" \
  -d '{"persona":"scholarly"}'

# Context save
curl -X POST http://localhost:8080/api/persona-context \
  -H "Content-Type: application/json" -H "X-Auth-Token: $TOKEN" \
  -d '{"persona":"intelligence","key":"system_context","content":"Focus on oil markets and Kuwait economy"}'

# Context get
curl http://localhost:8080/api/persona-context?persona=intelligence&key=system_context \
  -H "X-Auth-Token: $TOKEN"

# Scenario create
curl -X POST http://localhost:8080/api/scenarios/create \
  -H "Content-Type: application/json" -H "X-Auth-Token: $TOKEN" \
  -d '{"name":"Dark_Realm","world":"A dark fantasy kingdom under siege by demons."}'

# Entity create
curl -X POST http://localhost:8080/api/personas/gaming/entities \
  -H "Content-Type: application/json" -H "X-Auth-Token: $TOKEN" \
  -d '{"scenario":"Dark_Realm","name":"sakura","display_name":"Sakura","entity_type":"character","personality_md":"Cheerful but fierce warrior princess","speaking_style_md":"Formal but warm, uses honorifics"}'
```
