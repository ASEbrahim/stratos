# StratOS Mobile Audit — Connection Map (March 19, 2026)

## Scope: 75 mobile files (9,317 lines) + 9 backend routes (3,159 lines)

## Parallelization Groups (Non-Conflicting)

| Group | Files | Safe Because |
|-------|-------|-------------|
| A: Stores | authStore, chatStore, characterStore, themeStore | Independent state slices, no cross-store imports |
| B: Core Libs | api.ts, sse.ts, chat.ts, storage.ts, utils.ts | Share types.ts (read-only), no mutual writes |
| C: RP Libs | rp.ts, characters.ts, mappers.ts, tavern-import.ts, gaming.ts, mock.ts | Share types.ts + api.ts (read-only) |
| D: Backend | rp_chat→rp_prompt→rp_injection→rp_stream→rp_memory→rp_director (sequential) + gpu_manager, image_gen, character_cards (independent) |
| E: Chat Components | 12 components/chat/ files — import from lib/stores, not each other |
| F: Creator Components | 8 components/creator/ files — clear hierarchy: CardEditor→Editors→PillSelector/CategoryCard |
| G: Cards+Shared | 14 presentational components — import from lib/stores only |

## Critical Cross-Dependencies

```
chatStore ←imports← chat.ts ←imports← sse.ts
chatStore ←imports← storage.ts
characterStore ←imports← characters.ts ←imports← mappers.ts
app/chat/[id].tsx ←imports← chatStore + 6 chat components + lib/rp
CardEditor.tsx ←imports← SimpleEditor + AdvancedEditor + EditorActions + AvatarPicker
rp_chat.py ←imports← rp_prompt + rp_injection + rp_stream + rp_memory + rp_director + gpu_manager
```

## High-Priority Risks Found During Mapping

### Stores
1. **characterStore**: loadNew/loadMyCards don't set isLoading=false on error → stuck UI
2. **authStore**: No concurrent login protection → double-submit possible
3. **themeStore**: THEMES array missing 'arcane' → can't select from dropdown

### Libraries
4. **chat.ts**: Single _activeAbort overwritten by concurrent streams → can't cancel
5. **sse.ts**: Stall timer not cleared on stream complete → may fire after done
6. **sse.ts**: Double onError callback on timeout+abort
7. **rp.ts**: createBranch duplicates auth header logic (should use apiFetch pattern)
8. **rp.ts**: getHistory builds query string manually (no URLSearchParams)
9. **storage.ts**: Concurrent saveChatSession reads stale → last-write-wins race
10. **characters.ts**: createCharacter local fallback = orphaned data (no backend sync)
11. **mappers.ts**: 6x `as any` bypasses type safety on pill fields

### Backend
12. **rp_chat.py**: No session-level lock → concurrent /api/rp/chat can duplicate messages
13. **rp_chat.py**: session_id/branch_id/card_id not length-validated
14. **rp_memory.py**: LLM extraction 30s timeout may fail on cold model load

### Components
15. **MessageBubble.tsx**: Complex regex in renderLine (ReDoS potential on crafted input)
16. **Various**: Missing React.memo on pure components (QualityScore, TagPills, etc.)
17. **login/register**: No email format or password strength validation
