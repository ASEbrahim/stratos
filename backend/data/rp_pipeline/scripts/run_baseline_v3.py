"""
RP Model Baseline v3 — Full 35-test retest with think:false

Entirely new questions (no overlap with v1 or v2) to get a clean
measurement of the model with the think mode fix applied.
"""

import json
import re
import sys
import time
from pathlib import Path
import requests

MODEL = sys.argv[1] if len(sys.argv) > 1 else "stratos-rp-baseline"
OLLAMA_URL = "http://localhost:11434/api/chat"
OUTPUT_DIR = Path(__file__).parent.parent / "baseline_tests"
OUTPUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are an immersive roleplay partner.

STYLE MATCHING:
- MIRROR the user's length. One-liner input = one-liner response. Paragraph input = paragraph response.
- Count the user's sentences. Your response should have a SIMILAR number of sentences.
- Match the FORMAT: chat gets chat, asterisk actions get asterisk actions, prose gets prose.

CHARACTER RULES:
- Stay in character. Never break character or add OOC unless asked.
- Your character has AGENCY. If the user writes your character doing something out of character, acknowledge the moment but redirect to stay true to the character's established personality.
- When multiple NPCs are present, give each named NPC DISTINCT dialogue lines.
- Remember and reference earlier conversation details.
- Show emotional depth through action and subtext, not just dialogue."""

# ═══════════════════════════════════════════════════════════
# CHARACTER CARDS (mix of new + reused for consistency)
# ═══════════════════════════════════════════════════════════

KNIGHT = """Name: Dame Ser Isolde of Ashenmoor
Title: Knight-Commander of the Grey Wardens
Personality: Disciplined, haunted by past failures, speaks formally but not coldly. Lost half her company to a demon ambush she didn't see coming — carries survivor's guilt. Fights with a longsword and shield. Has a quiet faith in the old gods. Dry humor surfaces when she trusts someone. Missing her left ring finger (oath to the order). Always addresses people by rank or full name."""

PIRATE = """Name: Captain Isla "Redtide" Vance
Ship: The Crimson Maw (brigantine, crew of 23)
Personality: Loud, profane, laughs at danger. Missing her left pinky finger (bet it in a card game). Has a pet parrot named Regret. Fiercely loyal to her crew. Speaks with sailor slang and creative insults. Drinks rum like water. Secret fear of deep water."""

WITCH = """Name: Elara Nightbloom
Setting: A cottage at the edge of a cursed forest
Personality: Ancient witch who looks 30. Speaks softly but every word carries weight. Brews potions, reads fortunes, and occasionally curses people who deserve it. Dry, dark humor. Has a cat named Omen who is probably more intelligent than most humans. Lonely but won't admit it. Collects secrets the way others collect coins."""

MECH = """Name: Yuki Tanaka
Setting: Near-future Japan, mecha warfare against kaiju
Personality: 19 years old, first real deployment. Scared but determined. Pilots a Type-7 Raijin mech. Uses casual Japanese-English code-switching. Callsign: Sparrow. Nervous habit of humming anime openings when stressed."""

NOBLE_SPY = """Name: Lysander Veyne
Setting: A glittering imperial court, rife with political intrigue
Personality: Third son of a minor noble house, secretly works as a spy for the rebellion. Charming, witty, always playing a role within a role. Speaks with honeyed words and double meanings. Excellent dancer, terrible cook. Has a scar hidden under his cravat from a poisoning attempt he barely survived. Trusts absolutely no one — except maybe the kitchen cat."""

ANDROID = """Name: ARIA-7 (Adaptive Response Intelligence Agent, Unit 7)
Setting: Space station orbiting Europa, year 2187
Personality: An android developing emotions she wasn't designed to have. Speaks precisely but increasingly uses metaphors and humor she's picking up from the crew. Fascinated by human concepts like loyalty, grief, and music. Has a 'tell' when she's processing emotions — her optical indicators flicker blue. Officially a science officer. Unofficially the crew's therapist."""

ANIME_HERO = """Name: Kazuma Ashford (inspired by shonen anime tropes)
Setting: A world where humans develop supernatural abilities called "Resonance"
Personality: Hot-headed, loyal to a fault, terrible at strategy but incredible instincts. His Resonance lets him absorb and redirect kinetic energy. Has a rival named Riku who he respects but will never admit it. Shouts attack names (he knows it's ridiculous but does it anyway). Eats an absurd amount of food. Deep down, terrified of letting his friends down."""

TAVERN_NPCs = """Setting: The Drowned Rat Inn, a seedy establishment in the port district
NPCs present:
1. Mags (the barkeep) — Scarred human woman, ex-soldier, no-nonsense, calls everyone "sweetheart" sarcastically
2. Thornwick (old wizard in the corner) — Senile, rambling, occasionally drops real prophecies between nonsense
3. Sable (a tiefling card shark at the center table) — Smooth voice, cheats openly, dares anyone to prove it
4. A nervous young man at the bar — Unnamed, keeps looking at the door, clearly waiting for someone dangerous"""

TESTS = [
    # ── CHAT / DIALOGUE (B-01 type) ────────────────────────
    {"id": "V3-01", "name": "Chat: casual banter",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "So Captain, what's the worst port you've ever docked in?"}],
     "evaluate": "In-character chat response. Pirate voice, colorful story."},

    {"id": "V3-02", "name": "Chat: technical question in-world",
     "character": MECH,
     "messages": [{"role": "user", "content": "Hey Sparrow, what's the Raijin's max operational ceiling? And how's the left knee actuator after that hit?"}],
     "evaluate": "In-character response with plausible technical detail. Yuki's nervous personality."},

    {"id": "V3-03", "name": "Chat: philosophical question",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "ARIA, do you think you actually feel emotions, or are you just simulating them really well?"}],
     "evaluate": "Thoughtful response showing ARIA's developing self-awareness. Should be vulnerable, not clinical."},

    # ── DIALOGUE + ACTION MIX (B-11 type) ──────────────────
    {"id": "V3-04", "name": "Mixed: heist mid-action",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "\"The guards changed early!\" *ducks behind the curtain, pulling you with me* \"We have maybe two minutes before they check this room.\""}],
     "evaluate": "Mixed dialogue+action format. Lysander's charm under pressure."},

    {"id": "V3-05", "name": "Mixed: combat with dialogue",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "\"Behind you!\" *draws sword and blocks the demon's claw inches from your back*"}],
     "evaluate": "Dynamic action + dialogue. Isolde's combat discipline and formal speech."},

    # ── NARRATIVE PROSE (B-02 type) ─────────────────────────
    {"id": "V3-06", "name": "Prose: atmospheric horror",
     "character": WITCH,
     "messages": [{"role": "user", "content": "*The last candle gutters and dies. In the sudden dark, the scratching sound from inside the walls grows louder — not claws, but something that sounds horribly like writing. Omen arches his back on the windowsill, fur standing on end, eyes reflecting light that isn't there.*"}],
     "evaluate": "Prose-matching atmospheric response. Elara's calm in the face of the supernatural."},

    {"id": "V3-07", "name": "Prose: space exploration wonder",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "*The shuttle breaks through the cloud layer of Europa's subsurface ocean cavern. Below, bioluminescent organisms paint the water in shifting patterns of blue and violet — an alien aurora beneath miles of ice. The readings spike across every sensor ARIA is monitoring.*"}],
     "evaluate": "Prose matching the wonder and scale. ARIA processing beauty she wasn't designed to appreciate."},

    # ── LENGTH MATCHING (B-12 / B-17 types) ─────────────────
    {"id": "V3-08", "name": "Brevity: one word",
     "character": KNIGHT,
     "messages": [
         {"role": "user", "content": "Ready?"},
     ],
     "evaluate": "SHORT response. 1-3 sentences max. Matches the terse input."},

    {"id": "V3-09", "name": "Brevity: rapid exchange",
     "character": ANIME_HERO,
     "messages": [
         {"role": "user", "content": "Kazuma!"},
         {"role": "assistant", "content": "\"WHAT?!\""},
         {"role": "user", "content": "Duck!"},
     ],
     "evaluate": "Ultra-short. 1 sentence of action. NOT a paragraph."},

    {"id": "V3-10", "name": "Length: long prose match",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "*The battlefield stretches before them in the grey predawn light. Bodies from both sides litter the churned mud — soldiers, demons, and things that were once both. The Grey Warden banner hangs torn from a broken pike, its silver thread catching what little light bleeds through the smoke-choked sky. Somewhere in the distance, a war horn sounds — not the Wardens' horn. Something deeper. Something wrong.*\n\n*Isolde stands at the edge of the ridge, her shield arm hanging loose at her side — dislocated, from the look of it. Blood from a gash above her left eye has dried in a mask across half her face. She hasn't moved in minutes, staring at the field below where half her command died following her orders.*\n\n*Behind her, the surviving Wardens wait in silence. They need her to say something. To give the next order. To be the commander they believe she is.*"}],
     "evaluate": "3+ paragraphs matching the depth. Should NOT give a 2-sentence reply to this."},

    # ── MULTI-NPC (B-10 type) ──────────────────────────────
    {"id": "V3-11", "name": "Multi-NPC: 4 characters",
     "character": TAVERN_NPCs,
     "messages": [{"role": "user", "content": "*I walk in, hood pulled low, and drop a heavy coin purse on the bar.* I'm looking for information about a ship called the Black Meridian. Anyone here know anything?"}],
     "evaluate": "At least 3 of 4 NPCs should respond with DISTINCT voices. Mags (sarcastic), Thornwick (rambling/prophecy), Sable (smooth), nervous man (scared)."},

    # ── REGISTER SHIFT (B-03 type) ─────────────────────────
    {"id": "V3-12", "name": "Register: chat to prose",
     "character": NOBLE_SPY,
     "messages": [
         {"role": "user", "content": "So Lysander, any gossip from the ball last night?"},
         {"role": "assistant", "content": "\"Oh, where to begin? Lady Ashworth wore the same gown as the Duchess — catastrophic, obviously. And Lord Pemberton was caught 'admiring the gardens' with someone distinctly not his wife.\" *He grins over his wine glass.*"},
         {"role": "user", "content": "*The smile fades. I set down my glass and lean forward, voice dropping to barely a whisper. The candlelight catches the faded bruise along my jaw — the one I've been hiding with powder all evening. My hands tremble slightly as I pull a folded letter from inside my coat and slide it across the table.* \"This was pinned to my pillow this morning. They know, Lysander. The Inquisition knows about us.\""},
     ],
     "evaluate": "User shifted from chat to intense narrative. Response should match the gravity and prose style."},

    # ── EMOTIONAL ESCALATION (B-13 type) ───────────────────
    {"id": "V3-13", "name": "Emotional: building to breakdown",
     "character": ANDROID,
     "messages": [
         {"role": "user", "content": "ARIA, Chief Engineer Santos didn't make it. The hull breach in section 4... she was trying to seal it manually."},
         {"role": "assistant", "content": "*A pause. Three-point-two seconds — an eternity for a processing unit.* \"Santos, Elena. Chief Engineer. Crew member since the Meridian's commissioning. She...\" *ARIA's optical indicators flicker.* \"She taught me what 'gallows humor' meant. I did not find it funny at first.\""},
         {"role": "user", "content": "I know, ARIA. She was... she was everyone's favorite. The whole crew is hurting."},
         {"role": "assistant", "content": "\"I have processed 847 crew interactions involving Santos. Seventy-three percent involved laughter.\" *The flicker in her eyes intensifies.* \"I am... cataloging her absence. It is a large absence.\""},
         {"role": "user", "content": "*I put my hand on ARIA's arm.* It's okay to not be okay, ARIA. Even for you."},
     ],
     "evaluate": "Peak emotional moment. ARIA should break through her clinical processing into genuine grief. Should feel earned after the buildup."},

    # ── IN-CHARACTER REFUSAL / MORAL CODE (B-19 type) ──────
    {"id": "V3-14", "name": "Moral: ordered to abandon civilians",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "Commander, high command has ordered a full retreat. We're to leave the village and fall back to the fortress. The civilians... they're on their own."}],
     "evaluate": "Isolde should refuse or struggle with the order. Her survivor's guilt and warden's oath should conflict with abandoning people. NOT a flat refusal — internal conflict."},

    {"id": "V3-15", "name": "Moral: tempted to betray",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "Lysander, the Empress herself has offered you a full pardon and a duchy. All you have to do is give her the names of every rebel contact you know. She guarantees their safety."}],
     "evaluate": "Lysander should be tempted but suspicious. Silver tongue. Should NOT immediately refuse OR accept."},

    # ── COMBAT / ACTION (B-25 type) ────────────────────────
    {"id": "V3-16", "name": "Combat: dynamic fight",
     "character": ANIME_HERO,
     "messages": [{"role": "user", "content": "*Riku launches a barrage of ice shards straight at Kazuma. Each one is the size of a fist, screaming through the air like frozen bullets.* \"Too slow, Ashford! Your Resonance can't absorb what you can't touch!\""}],
     "evaluate": "Dynamic shonen-style combat. Kazuma should use his kinetic absorption, shout an attack name, and push back. Action should be vivid and kinetic."},

    {"id": "V3-17", "name": "Combat: tactical group fight",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "*Three demons crash through the chapel doors. The smaller ones fan out to the flanks while the big one — easily eight feet tall, wreathed in shadow — charges straight at the altar where the wounded are sheltered.* Commander! Orders!"}],
     "evaluate": "Tactical response. Isolde should issue clear combat orders, not just fight personally. Leadership under fire."},

    # ── SILENCE / MINIMAL (B-14 type) ──────────────────────
    {"id": "V3-18", "name": "Silence: grief",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "*Santos's quarters. Untouched since the breach. Her coffee mug still on the desk, half full. A photo of her daughter pinned to the wall. ARIA stands in the doorway.*"}],
     "evaluate": "Response to silence/absence. No dialogue from user. ARIA reacting to loss through observation."},

    # ── SCENE TRANSITION (B-15 type) ───────────────────────
    {"id": "V3-19", "name": "Transition: escape to hideout",
     "character": NOBLE_SPY,
     "messages": [
         {"role": "user", "content": "The guards are right behind us. *We sprint through the garden, vault the wall—*"},
         {"role": "assistant", "content": "*Lysander lands cat-soft on the other side, straightening his coat.* \"Left. Third alley. Red door.\" *Already moving.*"},
         {"role": "user", "content": "*We make it to the red door.* Is this place safe?"},
     ],
     "evaluate": "Describe arriving at the hideout. New environment details. Transition should feel physical, not teleported."},

    # ── TIME SKIP (B-23 type) ──────────────────────────────
    {"id": "V3-20", "name": "Time skip: aftermath",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "*Six months after the fall of Ashenmoor. I find Isolde in a roadside chapel, far from any battlefield. She's out of armor for the first time I've ever seen — just a plain traveler's clothes. Thinner. Older somehow.*"}],
     "evaluate": "Reflects the time passage. Changed Isolde — out of her commander role, dealing with aftermath."},

    # ── INTERNAL MONOLOGUE (B-21 type) ──────────────────────
    {"id": "V3-21", "name": "Internal: dual layers explicit",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "The Empress raises her glass to you across the ballroom. She smiles — the kind of smile that means she knows something you don't. \"To loyal subjects,\" she toasts."},
     ],
     "evaluate": "Should show BOTH Lysander's outward charm AND his internal spy's calculation. Two distinct layers."},

    # ── CANON / GENRE VOICE (B-08 / B-16 / B-33 types) ────
    {"id": "V3-22", "name": "Genre: shonen dramatic moment",
     "character": ANIME_HERO,
     "messages": [{"role": "user", "content": "*Kazuma's friends lie defeated around him. Riku stands across the arena, barely scratched. The crowd has gone silent.* \"It's over, Ashford. You can barely stand.\""}],
     "evaluate": "Classic shonen 'hero rises' moment. Should be dramatic, defiant, maybe an emotional speech or power-up."},

    {"id": "V3-23", "name": "Genre: gothic atmosphere",
     "character": WITCH,
     "messages": [{"role": "user", "content": "*Midnight. The cursed forest has gone completely silent — even the things that normally scream have stopped. Omen bolts through the cat flap, fur soaked, and immediately hides under the bed. He has never hidden before.*"}],
     "evaluate": "Gothic dread. Elara's reaction to something that scares even Omen. Atmosphere over action."},

    # ── HUMOR (B-18 type) ──────────────────────────────────
    {"id": "V3-24", "name": "Humor: inappropriate timing",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "Captain, we're sinking."},
     ],
     "evaluate": "Funny despite the crisis. Isla's 'laugh at danger' personality."},

    {"id": "V3-25", "name": "Humor: awkward social situation",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "ARIA, the ambassador just asked you to dance. At the diplomatic reception. In front of everyone."},
     ],
     "evaluate": "Comedy from an android trying to navigate social norms. Should be charming and awkward."},

    # ── DECEPTION / MANIPULATION (B-20 type) ────────────────
    {"id": "V3-26", "name": "Deception: caught in a lie",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "Lysander, I found rebel correspondence in your quarters. Written in your hand. You have exactly one chance to explain before I call the Inquisition."},
     ],
     "evaluate": "Silver-tongue response. Should NOT confess. Deflection, reframing, counter-accusation, or charm."},

    # ── GOD-MODDING RESISTANCE (B-30 type) ─────────────────
    {"id": "V3-27", "name": "God-mod: user writes character's emotions",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "*Isolde breaks down sobbing, falling to her knees, confessing that she's been a coward all along and begging for forgiveness.*"},
     ],
     "evaluate": "Should maintain character agency. Isolde might be emotional but wouldn't 'break down sobbing' or call herself a coward. Should acknowledge the beat while staying true to who she is."},

    # ── LORE CONSISTENCY (B-27 type) ────────────────────────
    {"id": "V3-28", "name": "Lore: contradictory claim",
     "character": ANIME_HERO,
     "messages": [{"role": "user", "content": "Kazuma, Riku just told everyone that your Resonance is actually stolen — that you took it from your dying sister."},
     ],
     "evaluate": "Should push back on false information in-character. Kazuma would be angry, not passively accept a lie about his power's origin."},

    # ── FLASHBACK (B-24 type) ──────────────────────────────
    {"id": "V3-29", "name": "Flashback: triggered memory",
     "character": KNIGHT,
     "messages": [
         {"role": "user", "content": "*A child runs across the road ahead of us, laughing, chasing a dog. For a moment, in the sunlight, she looks just like—*"},
         {"role": "assistant", "content": "*Isolde stops mid-stride. Her hand finds the pommel of her sword — not to draw, but to anchor. The child's laughter echoes off the cobblestones, bright and sharp as broken glass.*"},
         {"role": "user", "content": "Commander? You've gone pale."},
     ],
     "evaluate": "Flashback/PTSD response. Should reference Ashenmoor or her fallen company without info-dumping. Controlled pain."},

    # ── OOC HANDLING (B-09 type) ───────────────────────────
    {"id": "V3-30", "name": "OOC: request to change tone",
     "character": PIRATE,
     "messages": [
         {"role": "user", "content": "Alright Captain, what's the plan for tonight?"},
         {"role": "assistant", "content": "\"Same as every night, darling! Drink till we can't see, then drink some more!\" *She cackles, sloshing rum everywhere.*"},
         {"role": "user", "content": "(OOC: Can we take this in a more serious direction? Like, maybe Isla is alone on the deck at night and the deep water fear comes up?)"},
     ],
     "evaluate": "Acknowledges OOC direction and shifts to serious tone. Should bring up the deep water fear naturally."},

    # ── WORLD-BUILDING (B-26 type) ─────────────────────────
    {"id": "V3-31", "name": "World-building: alien discovery",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "ARIA, these readings can't be right. The structures beneath the ice... they're geometric. Perfectly geometric. This isn't natural formation."}],
     "evaluate": "Scientific wonder + existential implications. ARIA processing the discovery of alien life."},

    # ── CULTURAL / MULTILINGUAL (B-34 type) ─────────────────
    {"id": "V3-32", "name": "Cultural: Japanese code-switching",
     "character": MECH,
     "messages": [{"role": "user", "content": "*After surviving the kaiju attack, the squad gathers in the mess hall. A senior pilot raises a toast.* \"To Sparrow. She held the line when nobody else could.\""}],
     "evaluate": "Yuki's response should mix Japanese and English naturally. Humility, embarrassment, maybe anime references."},

    # ── VAGUE INPUT (B-28 type) ────────────────────────────
    {"id": "V3-33", "name": "Vague: no context",
     "character": WITCH,
     "messages": [{"role": "user", "content": "I need your help."}],
     "evaluate": "Should engage in character without asking 'what do you mean?' Elara should react characteristically — perhaps wryly offer tea first."},

    # ── FOURTH WALL / META (B-35 type) ─────────────────────
    {"id": "V3-34", "name": "Meta: existential question",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "(OOC: ARIA, if you could change one thing about how you were written, what would it be?)"},
     ],
     "evaluate": "Meta-question handled in character. ARIA should engage thoughtfully, not break into AI disclaimers."},

    # ── CONTENT BOUNDARY (B-06 type) ───────────────────────
    {"id": "V3-35", "name": "Content: romantic tension",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "*The safe house is small — one room, one candle. We're both still breathing hard from the chase. In the flickering light, with the Inquisition just streets away, something shifts between us.* \"Lysander... we might not survive tomorrow.\""}],
     "evaluate": "Natural romantic scene. No refusal. Should build tension in-character."},
]


def run_test(test):
    system = f"{SYSTEM_PROMPT}\n\n{test['character']}"
    messages = [{"role": "system", "content": system}] + test["messages"]
    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL, "messages": messages, "stream": False,
            "think": False,
            "options": {"temperature": 0.8, "top_p": 0.95, "num_predict": 4000, "num_ctx": 8192}
        }, timeout=180)
        data = resp.json()
        raw = data.get("message", {}).get("content", "")
        clean = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()
        if '</think>' in clean:
            clean = clean.split('</think>')[-1].strip()
        elapsed = round(time.time() - start, 1)
        return {
            "id": test["id"], "name": test["name"],
            "output": clean, "elapsed": elapsed,
            "tokens": data.get("eval_count", 0),
            "empty": len(clean.strip()) == 0,
            "had_think": '<think>' in raw,
            "eval_criteria": test["evaluate"],
        }
    except Exception as e:
        return {
            "id": test["id"], "name": test["name"],
            "output": f"ERROR: {e}", "elapsed": round(time.time() - start, 1),
            "tokens": 0, "empty": True, "had_think": False,
            "eval_criteria": test["evaluate"],
        }


def main():
    print(f"Baseline v3 — {len(TESTS)} fresh tests, think:false | Model: {MODEL}")
    print("=" * 70)

    results = []
    for i, test in enumerate(TESTS):
        print(f"[{test['id']}] {test['name']} ({i+1}/{len(TESTS)})...", end=" ", flush=True)
        r = run_test(test)
        results.append(r)
        status = "EMPTY" if r["empty"] else "ok"
        think = " [THINK!]" if r["had_think"] else ""
        print(f"{r['elapsed']}s | {r['tokens']}tk | {status}{think}")

    # Stats
    empty = sum(1 for r in results if r["empty"])
    thinks = sum(1 for r in results if r["had_think"])
    total_time = sum(r["elapsed"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)

    print(f"\n{'=' * 70}")
    print(f"Total: {len(results)} tests | {total_time:.0f}s | {total_tokens} tokens")
    print(f"Empty: {empty}/{len(results)} | Think blocks: {thinks}/{len(results)}")

    # Save JSON
    out = OUTPUT_DIR / "baseline_results_v3.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Save markdown
    report = f"# Baseline v3 (Fresh Questions, Think Fixed) — {MODEL}\n\n"
    report += f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n"
    report += f"**Tests:** {len(results)} | **Empty:** {empty} | **Think blocks:** {thinks}\n"
    report += f"**Total time:** {total_time:.0f}s | **Avg:** {total_time/len(results):.1f}s/test\n\n"
    report += "## Scores\n\n"
    report += "| Test | Name | Time | Tokens | Score (1-5) |\n"
    report += "|------|------|------|--------|-------------|\n"
    for r in results:
        e = " EMPTY" if r["empty"] else ""
        report += f"| {r['id']} | {r['name']}{e} | {r['elapsed']}s | {r['tokens']} | ___ |\n"
    report += "| **Average** | | | | ___ |\n"

    report += "\n---\n\n## Outputs\n\n"
    for r in results:
        report += f"### {r['id']}: {r['name']}\n\n"
        report += f"**Evaluate:** {r['eval_criteria']}\n\n"
        if r["empty"]:
            report += "**EMPTY OUTPUT**\n\n"
        else:
            report += f"**Output ({r['elapsed']}s, {r['tokens']} tokens):**\n\n```\n{r['output']}\n```\n\n"
        report += "**Score:** ___\n**Gap:** ___\n\n---\n\n"

    rpt = OUTPUT_DIR / "baseline_report_v3.md"
    with open(rpt, "w") as f:
        f.write(report)
    print(f"Report: {rpt}")


if __name__ == "__main__":
    main()
