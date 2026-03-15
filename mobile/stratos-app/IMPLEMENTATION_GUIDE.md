# Implementation Guide — RP Expansion UI Components

## CRITICAL RULES (Re-read after every component)
1. **NO VRAM usage** — training is running. No Ollama, no ComfyUI, no GPU.
2. **USE_MOCKS = true** — all new features must work with mock fallbacks
3. **Commit after each component** — small, atomic commits
4. **TypeScript must pass** — `npx tsc --noEmit` after every change
5. **Don't break existing UI** — test that discover/chat/library/profile still work
6. **Fonts**: Quicksand for headings (`fonts.heading`), Poppins for body (`fonts.body`)
7. **Theme**: Use `useThemeStore(s => s.colors)` — never hardcode colors
8. **4 themes**: Nebula (default), Sakura, Cosmos, Noir
9. **Haptics**: Use on user actions (selection, success, error)
10. **imports from lib/rp.ts**: editMessage, regenerateMessage, selectSwipe, sendFeedback, setDirectorNote, createBranch, getBranches, getHistory

## Components to Build (in order)
1. **FeedbackButtons** — thumbs up/down under AI messages
2. **SwipeIndicator** — ← 1/3 → arrows on last AI message after regenerate
3. **DirectorNoteBar** — collapsible bar above ChatInput
4. **EditSheet** — bottom sheet to edit AI message with reason dropdown
5. **BranchSelector** — dropdown at top of chat when branches > 1
6. **Update chat screen** — integrate all 5 components into app/chat/[id].tsx
7. **Character detail** — add "Generate Portrait" button linking to /imagegen
8. **Card actions** — rate/publish/save buttons on character detail
9. **Opt-in popup** — training data consent on first RP chat access

## Chat Screen Current Layout
```
SessionHeader
FlatList<MessageBubble>
StreamingBubble (during generation)
SuggestionChips
ChatInput
```

## Chat Screen Target Layout
```
SessionHeader + BranchSelector (if branches > 1)
FlatList<MessageBubble + FeedbackButtons (on AI msgs) + SwipeIndicator (on last AI msg)>
StreamingBubble
SuggestionChips
DirectorNoteBar (collapsible, shows last note grayed)
ChatInput
```

## Key Files
- Chat screen: `app/chat/[id].tsx`
- Character detail: `app/character/[id].tsx`
- Character detail view: `components/cards/CharacterDetail.tsx`
- Chat components: `components/chat/`
- API functions: `lib/rp.ts`
- Types: `lib/types.ts`
- Theme: `stores/themeStore.ts`
- Fonts: `constants/fonts.ts`
- Spacing/borders: `constants/theme.ts`

## ChatMessage Interface (current)
```typescript
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
}
```
Note: No `swipe_group_id`, `turn_number`, or `message_id` (backend int).
For expansion features, we'll need to extend this or use separate state.

## After ALL components done:
1. Run `npx tsc --noEmit` — must pass
2. Commit everything
3. Update CLAUDE.md with new patterns
4. Create FRS document
5. Create README.md for GitHub
6. Initialize git remote if needed
