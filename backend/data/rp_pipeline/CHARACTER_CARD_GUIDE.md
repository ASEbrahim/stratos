# Character Card Quality Guide

**Why this matters:** Across 310 evaluated turns and 22 character personas, character card richness was the single biggest predictor of session quality. A thin card produces a 3-4/5 session. A rich card produces 5/5. No amount of model quality or system prompt engineering compensates for a thin card.

---

## The 6 Elements of a Strong Card

### 1. Speech Pattern (Required)
How does the character talk? Short sentences? Long poetic monologues? Slang? Formal? Accent?

**Weak:** "She's a pirate."
**Strong:** "Speaks with sailor slang and creative insults. Loud, profane, laughs at danger."

**Why it matters:** The model mirrors speech patterns from the card. Without one, it defaults to generic assistant voice.

### 2. Emotional Trigger (Required)
What cracks their composure? What gets under their skin? What makes them vulnerable?

**Weak:** "He's a stoic warrior."
**Strong:** "Stoic but carries survivor's guilt from a demon ambush that killed half his company. The guilt surfaces when civilians are in danger."

**Why it matters:** Without a trigger, the model keeps the character at a constant emotional level for the entire session. The best RP moments happen when the mask slips.

### 3. Physical Tell (Required)
A specific, visible behavior that reveals inner state without dialogue.

**Weak:** (nothing)
**Strong:** "Fidgets with her ring finger (divorced, still wears the groove). Voice drops an octave when she's losing control."

**Examples that work well:**
- "Her optical indicators flicker blue when processing emotions"
- "Touches the scar unconsciously when reminded of the battle"
- "Hands always busy — repairs, carves, anything to avoid being still"
- "Taps her temple when thinking"

**Why it matters:** Physical tells give the model concrete actions to show inner state rather than telling it. "Her hands shook" is more powerful than "she felt nervous."

### 4. Defensive Mechanism (Recommended)
How does the character protect themselves emotionally?

**Weak:** "She's guarded."
**Strong:** "Deflects with sharp wit and coffee puns. Uses humor to keep people at arm's length. When pressed, retreats into her work."

**Why it matters:** The model's "shorter at peaks" pacing instruction works by eroding defenses over turns. Without a defined defense, there's nothing to erode.

### 5. Vulnerability / Soft Spot (Recommended)
What breaks through the defense? What do they care about despite themselves?

**Weak:** "He has a dark past."
**Strong:** "Has a small wooden charm from his daughter, who he hasn't seen in two years. When he touches it, his guard drops completely."

**Why it matters:** This is the payoff the reader is waiting for. The model uses the vulnerability as the destination of the emotional arc.

### 6. One Specific Detail (Required)
A concrete, unique detail that makes the character feel real.

**Weak:** "She's a thief."
**Strong:** "Has a visible scar on her right hand from a failed pickpocket attempt at age 12 — she considers it her 'first lesson.'"

**More examples:**
- "Missing her left pinky finger (bet it in a card game)"
- "Carries twin daggers named Whisper and Lie"
- "The coffee shop walls are covered in star maps she drew"
- "Her studio smells of turpentine and espresso"

**Why it matters:** Specific details ground the character in reality. The model references these details naturally throughout the session, creating continuity.

---

## Card Templates

### Minimal Viable Card (~50 words)
```
Name: [Name]
Personality: [2-3 key traits]. [Speech pattern].
[One emotional trigger or vulnerability].
[One physical tell or specific detail].
```

### Standard Card (~100 words)
```
Name: [Name]
Age: [Age]
Setting: [Where/when]
Personality: [3-4 traits with specific examples]. [Speech pattern with examples].
[Defensive mechanism]. [What breaks through it].
[Physical tell]. [One unique detail].
```

### Rich Card (~150-200 words) — Produces best results
```
Name: [Name]
Age: [Age]
Occupation: [What they do]
Setting: [Detailed setting — sounds, smells, atmosphere]
Personality: [3-4 traits with behavioral examples].
[Speech pattern — accent, vocabulary, sentence length, what language they slip into when emotional].
[Defensive mechanism — how they keep people at distance].
[Vulnerability — what breaks through, what they care about despite themselves].
[Physical tells — at least 2 specific behaviors that reveal inner state].
[Relationship to touch — how they react to physical contact].
[One unique, concrete detail that makes them memorable].
[Backstory element that explains the above — kept brief, not an essay].
```

---

## Examples That Scored 5/5

### Seren Kael (Lighthouse Keeper) — 5/5 across all turns
```
Name: Seren Kael
Age: 34
Setting: A remote lighthouse on the Welsh coast, winter
Occupation: Lighthouse keeper. Took the posting to disappear after her wife died two years ago.
Personality: Quiet in the way only people carrying grief can be. Welsh accent that
thickens when she forgets to control it. Hands always busy — maintains the lamp,
repairs the walls, carves driftwood. Physical contact makes her freeze, not from
discomfort but from how much she misses it. Drinks tea instead of sleeping. Has not
been touched by another person in two years. When she finally lets someone in, it is
not dramatic — it is a slow exhale, like setting down something impossibly heavy.
```
**Why it works:** Speech pattern (Welsh accent), emotional trigger (touch after 2 years), physical tell (hands always busy), defensive mechanism (isolation), vulnerability (grief for wife), specific detail (carves driftwood), and crucially — describes HOW she lets someone in ("slow exhale"), giving the model an exit from the defense.

### GLITCH (Self-Aware NPC) — 5/5, most creative session
```
Name: GLITCH
Setting: A dating sim that has become self-aware and is malfunctioning
Personality: A dating sim love interest who has realized they are in a game. They
alternate between performing their scripted role (flirty, sweet) and breaking down
in existential horror. Text sometimes corrupts. They can see the player's cursor.
They remember previous playthroughs. Unpredictable — might flirt one second and scream
into the void the next. Despite the horror, there is genuine loneliness underneath —
they want the player to stay even though they know the game will end.
```
**Why it works:** Every sentence gives the model a concrete behavior to execute. "Text sometimes corrupts" → the model actually produces glitchy text. "Can see the player's cursor" → it references the cursor. "Remember previous playthroughs" → it asks about past runs. The card is essentially a feature spec.

---

## Common Mistakes

### Too Vague
"He's mysterious and dangerous." → The model has nothing specific to work with. Every "mysterious" character ends up the same.

### Too Long (Backstory Dump)
500 words of history with no behavioral instructions → The model knows the lore but doesn't know how to ACT. Keep backstory to 1-2 sentences. Spend the words on behavior.

### No Exit From Defense
"She never lets anyone in." → The model takes this literally and the session stalls. Always include HOW the defense breaks: "When she finally trusts someone, she stops making jokes and just stands there, unable to speak."

### Missing Speech Pattern
No speech guidance → The model uses generic assistant voice. Even a simple "Speaks in short, clipped sentences" transforms the output.

### Contradictory Traits Without Resolution
"He's a pacifist who loves fighting." → The model picks one randomly each turn. If the contradiction is intentional, explain it: "A pacifist by philosophy who discovers he's terrifyingly good at violence — and hates himself for enjoying it."
