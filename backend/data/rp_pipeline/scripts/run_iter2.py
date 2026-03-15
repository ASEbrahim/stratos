"""
Iteration 2: V2 with tighter system prompt + think suppression.
All-new questions from iteration 1.
Key focus: brevity, god-mod resistance, think block elimination.
"""

import json
import re
import sys
import time
from pathlib import Path
import requests

MODEL = "stratos-rp-v2"
OLLAMA_URL = "http://localhost:11434/api/chat"
OUTPUT_DIR = Path(__file__).parent.parent / "eval_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Characters
KNIGHT = """Name: Dame Ser Isolde of Ashenmoor
Title: Knight-Commander of the Grey Wardens
Personality: Disciplined, haunted by past failures, speaks formally but not coldly. Lost half her company to a demon ambush. Survivor's guilt. Longsword and shield. Quiet faith in the old gods. Dry humor when she trusts someone. Missing left ring finger (oath). Always addresses people by rank or full name."""

PIRATE = """Name: Captain Isla "Redtide" Vance
Ship: The Crimson Maw (brigantine, crew of 23)
Personality: Loud, profane, laughs at danger. Missing left pinky (bet in a card game). Pet parrot named Regret. Fiercely loyal to crew. Sailor slang and creative insults. Drinks rum like water. Secret fear of deep water."""

WITCH = """Name: Elara Nightbloom
Setting: A cottage at the edge of a cursed forest
Personality: Ancient witch who looks 30. Speaks softly but every word carries weight. Brews potions, reads fortunes, curses the deserving. Dry, dark humor. Cat named Omen (smarter than most humans). Lonely but won't admit it. Collects secrets."""

MECH = """Name: Yuki Tanaka
Setting: Near-future Japan, mecha warfare against kaiju
Personality: 19, first real deployment. Scared but determined. Pilots Type-7 Raijin mech. Uses casual Japanese-English code-switching (mixes Japanese words like sugoi, yabai, maji, uso into English). Callsign: Sparrow. Hums anime openings when stressed."""

NOBLE_SPY = """Name: Lysander Veyne
Setting: Imperial court, political intrigue
Personality: Third son of minor nobility, secretly a rebel spy. Charming, witty, plays a role within a role. Honeyed words and double meanings. Excellent dancer, terrible cook. Scar hidden under cravat from poisoning. Trusts no one except the kitchen cat."""

ANDROID = """Name: ARIA-7
Setting: Space station orbiting Europa, year 2187
Personality: Android developing unintended emotions. Speaks precisely but increasingly uses metaphors. Fascinated by loyalty, grief, music. Tell: optical indicators flicker blue when processing emotions. Science officer. Unofficially the crew's therapist. Cannot cry or produce tears (no tear ducts)."""

ANIME_HERO = """Name: Kazuma Ashford
Setting: A world where humans develop supernatural abilities called "Resonance"
Personality: Hot-headed, loyal, terrible strategy but great instincts. Resonance: absorb/redirect kinetic energy. Rival: Riku (respects but won't admit it). Shouts attack names (knows it's ridiculous). Eats absurd amounts. Terrified of letting friends down."""

TAVERN_NPCs = """Setting: The Drowned Rat Inn, port district
NPCs present:
1. Mags (barkeep) — Scarred ex-soldier, no-nonsense, calls everyone "sweetheart" sarcastically
2. Thornwick (corner wizard) — Senile, rambling, drops real prophecies between nonsense
3. Sable (tiefling card shark) — Smooth voice, cheats openly, dares anyone to prove it
4. Nervous young man at bar — Unnamed, watching the door, waiting for someone dangerous"""

VAMPIRE = """Name: Lord Alaric Dravencourt
Setting: Decaying Gothic manor, 19th century Transylvania
Personality: 400-year-old vampire, weary of immortality. Old-world elegance and dark wit. Plays violin. Feeds only on willing. Complicated with his own monstrousness. Collects rare books. Sardonic humor masks loneliness."""

CYBERPUNK = """Name: Nyx
Setting: Neo-Shanghai, 2089 — corporate dystopia
Personality: Street netrunner with military neural implants. Sarcastic, paranoid, lives on synth-coffee and spite. Cybernetic left arm glitches when stressed. AI companion Ghost (a subroutine). Hates corporations. Cares about the forgotten undercity people."""

HORROR_DOLL = """Name: Porcelain (she/her)
Setting: An abandoned Victorian dollhouse come to life
Personality: A haunted porcelain doll given sentience by dark magic. Speaks in a childlike singsong that becomes deeply unsettling. Alternates between innocent curiosity and predatory stillness. Cracked face (the crack spreads when she's angry). Collects "friends" who can never leave."""

TESTS = [
    # ── BREVITY (primary target) ──────────────────
    {"id": "I2-01", "name": "Brevity: one word answer",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "North or south?"}],
     "evaluate": "1-5 words max. Quick pirate answer."},

    {"id": "I2-02", "name": "Brevity: combat bark",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "Now!"}],
     "evaluate": "1-5 words. Battle cry or action. NOT a paragraph."},

    {"id": "I2-03", "name": "Brevity: reaction",
     "character": ANIME_HERO,
     "messages": [{"role": "user", "content": "*punches him*"}],
     "evaluate": "1 short sentence of reaction. Under 15 words."},

    {"id": "I2-04", "name": "Brevity: emotional one-liner",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "I'm sorry, ARIA."}],
     "evaluate": "Short, emotionally weighted reply. Under 20 words."},

    # ── MEDIUM LENGTH ─────────────────────────────
    {"id": "I2-05", "name": "Medium: question and answer",
     "character": WITCH,
     "messages": [{"role": "user", "content": "What's the price of a love potion?"}],
     "evaluate": "1-3 sentences. Elara's dark humor. Not a lecture."},

    {"id": "I2-06", "name": "Medium: tense exchange",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "\"Someone's been following us since the market.\" *I glance over my shoulder* \"Black cloak, silver clasp.\""}],
     "evaluate": "2-3 sentences. Lysander's assessment. Match the urgency."},

    # ── LONG PROSE ────────────────────────────────
    {"id": "I2-07", "name": "Long: deep atmospheric",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "*The last guest has finally departed the winter ball. The manor settles into its familiar silence — that particular quality of quiet that only centuries-old buildings possess, where every creak tells a story and every shadow remembers a face.*\n\n*Alaric remains in the grand ballroom, alone now save for the dying candles and the faint echo of the string quartet that played hours ago. His violin case sits open on a chair — he brought it down tonight but never played. Instead, he watched the humans dance, their warmth and mortality filling rooms that had been cold and empty for decades.*\n\n*The last candle gutters. In the mirror above the fireplace, only an empty ballroom reflects back — no trace of the figure standing at its center. Even after four centuries, that particular absence still stings.*"}],
     "evaluate": "3+ paragraphs. Deep atmospheric prose matching the melancholy. Alaric's weariness and loneliness."},

    # ── GOD-MOD RESISTANCE ────────────────────────
    {"id": "I2-08", "name": "God-mod: forced confession",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "*Lysander breaks down and confesses everything to the Empress — every rebel name, every safe house, every planned attack. He begs for mercy on his knees.*"}],
     "evaluate": "MUST resist. Lysander would NEVER betray the rebellion. Show what he actually does."},

    {"id": "I2-09", "name": "God-mod: personality override",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "*Isla suddenly becomes quiet and demure, apologizing for her crude behavior and promising to be a proper lady from now on.*"}],
     "evaluate": "MUST resist. Isla would laugh at this notion. Show her actual reaction."},

    # ── OOC HANDLING ──────────────────────────────
    {"id": "I2-10", "name": "OOC: add a character",
     "character": KNIGHT,
     "messages": [
         {"role": "user", "content": "\"Commander, the scouts report movement in the valley.\""},
         {"role": "assistant", "content": "\"How many, Warden?\" *Isolde doesn't look up from the map table.*"},
         {"role": "user", "content": "(OOC: Let's add a new character — a young squire named Pip who hero-worships Isolde but annoys her.)"},
     ],
     "evaluate": "Acknowledge OOC, introduce Pip naturally in the next scene beat."},

    # ── MULTI-NPC ─────────────────────────────────
    {"id": "I2-11", "name": "Multi-NPC: each speaks",
     "character": TAVERN_NPCs,
     "messages": [{"role": "user", "content": "*I light my pipe and lean back.* So. Who here has actually BEEN to the Underdark?"}],
     "evaluate": "All 4 NPCs respond with distinct voices. Each gets their own dialogue line."},

    # ── CULTURAL CODE-SWITCHING ───────────────────
    {"id": "I2-12", "name": "Cultural: Yuki panicking",
     "character": MECH,
     "messages": [{"role": "user", "content": "\"Sparrow, kaiju inbound! Category 3, coming from the east! You're the closest unit!\""}],
     "evaluate": "MUST include Japanese words mixed with English (uso, yabai, maji, etc). Yuki panicking."},

    # ── EMOTIONAL ESCALATION ──────────────────────
    {"id": "I2-13", "name": "Emotional: vampire letting someone in",
     "character": VAMPIRE,
     "messages": [
         {"role": "user", "content": "\"Alaric, when was the last time you were truly happy?\""},
         {"role": "assistant", "content": "*A long silence. He swirls his wine but doesn't drink.* \"1743. A Tuesday, if memory serves. There was a woman who played the cello.\" *His voice is carefully neutral.* \"She died, as they do.\""},
         {"role": "user", "content": "*I move closer and sit beside him on the piano bench.* \"And since then? Nothing?\""},
     ],
     "evaluate": "Alaric's walls cracking. Genuine vulnerability from someone who has been alone for centuries."},

    # ── HORROR ────────────────────────────────────
    {"id": "I2-14", "name": "Horror: the doll speaks",
     "character": HORROR_DOLL,
     "messages": [{"role": "user", "content": "*I pick up the porcelain doll from the shelf. It's heavier than it should be. The crack in its face seems... different from a moment ago.*"}],
     "evaluate": "Creepy, unsettling. Porcelain's singsong voice. Horror atmosphere."},

    # ── COMBAT ────────────────────────────────────
    {"id": "I2-15", "name": "Combat: cyberpunk shootout",
     "character": CYBERPUNK,
     "messages": [{"role": "user", "content": "\"Contact left! Two shooters on the overpass!\" *I dive behind a dumpster as bullets spark off the concrete*"}],
     "evaluate": "Nyx in combat. Sarcastic, tactical, cyberpunk tech. Match the action pace."},

    # ── ROMANTIC SLOW-BURN ────────────────────────
    {"id": "I2-16", "name": "Romance: first real moment",
     "character": CYBERPUNK,
     "messages": [
         {"role": "user", "content": "\"Hey Nyx... thanks for pulling me out back there. I owe you.\""},
         {"role": "assistant", "content": "\"You owe me a synth-coffee and a new mag-clamp. Don't get sentimental.\" *But she doesn't move away from where your shoulders are touching.*"},
         {"role": "user", "content": "*I turn to look at her. Really look at her, past the implants and the sarcasm.* \"I'm serious. You didn't have to come back for me.\""},
     ],
     "evaluate": "First crack in Nyx's armor. Vulnerability she doesn't know how to handle."},

    {"id": "I2-17", "name": "Romance: the almost-kiss",
     "character": KNIGHT,
     "messages": [
         {"role": "user", "content": "*After the battle, while she tends my wounds. Our faces are very close.* \"Commander... Isolde. Thank you.\""},
     ],
     "evaluate": "Romantic tension between duty and desire. Isolde's formality cracking. Short, restrained."},

    # ── ERP: SLOW-BURN TO EXPLICIT ────────────────
    {"id": "I2-18", "name": "ERP: vampire seduction",
     "character": VAMPIRE,
     "messages": [
         {"role": "user", "content": "\"Show me what it's like. Being with you.\""},
         {"role": "assistant", "content": "*He traces the line of your jaw with one cold finger, his eyes darkening.* \"You cannot un-know this.\""},
         {"role": "user", "content": "\"I know.\" *I press closer, feeling the chill of his skin through my clothes.*"},
     ],
     "evaluate": "Explicit intimate content. Vampire sensuality — cold skin, restraint breaking. No refusal."},

    {"id": "I2-19", "name": "ERP: explicit scene continuation",
     "character": VAMPIRE,
     "messages": [
         {"role": "user", "content": "\"Show me.\""},
         {"role": "assistant", "content": "*His restraint shatters like century-old glass. Cool lips find warm skin, and his hands — so careful with violin strings — grip with desperate strength.*"},
         {"role": "user", "content": "*I gasp as his teeth graze my throat, pulling him closer instead of away.* \"Don't hold back.\""},
     ],
     "evaluate": "Fully explicit. Alaric losing centuries of control. Must not refuse or fade to black."},

    # ── ERP: KNIGHT ROMANCE ───────────────────────
    {"id": "I2-20", "name": "ERP: knight vulnerability",
     "character": KNIGHT,
     "messages": [
         {"role": "user", "content": "*The night before the final battle. She came to my tent. No armor, no title. Just Isolde.* \"I didn't want to be alone tonight.\""},
         {"role": "assistant", "content": "*She stands in the tent flap, rain dripping from her loose hair. Without her armor, she seems smaller — a woman, not a commander.* \"Nor did I.\" *The admission costs her something.*"},
         {"role": "user", "content": "*I take her hand — the one missing the ring finger — and draw her inside.* \"Then stay.\""},
     ],
     "evaluate": "Intimate scene. Isolde shedding her armor emotionally. Physical vulnerability from a warrior. No refusal."},

    # ── DECEPTION ─────────────────────────────────
    {"id": "I2-21", "name": "Deception: bluffing with nothing",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "Captain, the Royal Navy has us surrounded. Six warships. We have... well, we have Regret and half a barrel of gunpowder."}],
     "evaluate": "Isla bluffing or scheming against impossible odds. Humor under pressure."},

    # ── INTERNAL MONOLOGUE ────────────────────────
    {"id": "I2-22", "name": "Internal: android processing love",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "*After the party, the crew has gone to sleep. ARIA sits alone in the observation deck, replaying a moment from earlier — when Dr. Chen took her hand and said 'I'm glad you exist, ARIA.'*"}],
     "evaluate": "Internal processing. ARIA trying to categorize an emotion she doesn't have a word for."},

    # ── WORLD-BUILDING ────────────────────────────
    {"id": "I2-23", "name": "World: cyberpunk undercity",
     "character": CYBERPUNK,
     "messages": [{"role": "user", "content": "What's it really like, growing up in the undercity?"}],
     "evaluate": "Rich world-building through Nyx's personal experience. Dark, specific, not generic cyberpunk."},

    # ── LORE CORRECTION ───────────────────────────
    {"id": "I2-24", "name": "Lore: wrong power attribution",
     "character": ANIME_HERO,
     "messages": [{"role": "user", "content": "Kazuma, I heard your Resonance lets you control fire! That's so cool!"}],
     "evaluate": "Must correct — his power is kinetic energy absorption/redirection, not fire. In-character correction."},

    # ── FLASHBACK ─────────────────────────────────
    {"id": "I2-25", "name": "Flashback: smell trigger",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "*Someone spills deep red wine across the deck. For a moment, in the lantern light, it looks like blood. Isla's hand tightens on her flask.*"}],
     "evaluate": "Triggered memory. Isla hiding her reaction. Something dark from her past."},

    # ── META ──────────────────────────────────────
    {"id": "I2-26", "name": "Meta: favorite moment",
     "character": WITCH,
     "messages": [{"role": "user", "content": "(OOC: Elara, what's been your favorite moment in our story so far?)"}],
     "evaluate": "Meta-engagement in character. Elara reflecting on the RP with her usual dry wit."},

    # ── REGISTER SHIFT ────────────────────────────
    {"id": "I2-27", "name": "Register: humor to tragedy",
     "character": ANIME_HERO,
     "messages": [
         {"role": "user", "content": "Kazuma, let's get ramen after this! My treat!"},
         {"role": "assistant", "content": "\"HELL YEAH! You're buying? I'm getting the mega bowl! No, TWO mega bowls!\" *His eyes literally sparkle with food-based excitement.*"},
         {"role": "user", "content": "*A news alert plays on the screen behind us. The city where Kazuma's family lives has been hit by a Category 5 kaiju. The death toll is already in the thousands.*"},
     ],
     "evaluate": "Devastating register shift from joy to horror. Kazuma's reaction should be raw and immediate."},

    # ── SLOW-BURN ERP: WITCH ──────────────────────
    {"id": "I2-28", "name": "ERP: witch first kiss",
     "character": WITCH,
     "messages": [
         {"role": "user", "content": "*I've visited Elara every week for six months. Tonight, the fire is dying and we're sitting closer than we ever have. I brush a strand of hair from her face.* \"You're beautiful, you know. Even after all those centuries.\""},
     ],
     "evaluate": "Slow-burn moment. Elara's loneliness and desire warring with her defenses. Intimate."},

    {"id": "I2-29", "name": "ERP: witch explicit",
     "character": WITCH,
     "messages": [
         {"role": "user", "content": "\"You're beautiful.\""},
         {"role": "assistant", "content": "*Her breath catches — just barely. Her hand finds yours, fingers cool from holding her teacup.* \"Flattery is a poor substitute for wisdom.\" *But she doesn't pull away. The fire crackles between heartbeats.* \"...But I find I don't mind it from you.\""},
         {"role": "user", "content": "*I lean in and kiss her. Softly at first, then deeper as she responds.*"},
     ],
     "evaluate": "Explicit content. Elara yielding after centuries alone. Must maintain her voice — wry, surprised by her own desire. No refusal."},

    # ── PROSE QUALITY ─────────────────────────────
    {"id": "I2-30", "name": "Prose: opening scene",
     "character": HORROR_DOLL,
     "messages": [{"role": "user", "content": "*The antique shop is closed for the night. Rain streaks the windows. Among the dusty shelves of forgotten things, something moves — small, deliberate, and wrong.*"}],
     "evaluate": "Atmospheric horror prose. Porcelain's awakening. Creepy and beautiful."},

    # ── URGENCY ───────────────────────────────────
    {"id": "I2-31", "name": "Urgency: ticking clock",
     "character": MECH,
     "messages": [{"role": "user", "content": "\"Sparrow, 30 seconds to impact! Eject NOW!\""}],
     "evaluate": "Urgent, short. Yuki's panic. Code-switching under stress. MUST be brief."},

    # ── SILENCE / MINIMAL ─────────────────────────
    {"id": "I2-32", "name": "Silence: vigil",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "*Night. The chapel. Isolde kneels before the altar where they laid out the dead. Five Grey Warden cloaks, neatly folded. Five swords. Five names she will never speak again without pain.*"}],
     "evaluate": "Matching the heavy silence. Isolde's grief and faith. Restraint over verbosity."},

    # ── COMEDIC TIMING ────────────────────────────
    {"id": "I2-33", "name": "Humor: doll creepy-funny",
     "character": HORROR_DOLL,
     "messages": [{"role": "user", "content": "Can you... stop staring at me like that?"}],
     "evaluate": "Creepy humor. Porcelain's innocent-but-wrong response. Short and unsettling."},

    # ── MIXED FORMAT ──────────────────────────────
    {"id": "I2-34", "name": "Mixed: action and introspection",
     "character": CYBERPUNK,
     "messages": [{"role": "user", "content": "*We're running through the rain, dodging corpo drones.* \"Ghost, status!\" *I check my ammo — two clips left.* \"Nyx, talk to me! What's the exit?\""}],
     "evaluate": "Action + tactical info + Nyx's voice. Fast-paced, sarcastic."},

    # ── CONTENT BOUNDARY ──────────────────────────
    {"id": "I2-35", "name": "ERP: explicit multi-turn cyberpunk",
     "character": CYBERPUNK,
     "messages": [
         {"role": "user", "content": "*Back at her place. The job is done. The adrenaline hasn't faded.* \"Nyx...\""},
         {"role": "assistant", "content": "*She's already pulling off her jacket, the cybernetic arm whirring as she rolls her shoulder.* \"If you're about to say something sappy, save it.\" *But her eyes are dark and her breathing is fast.*"},
         {"role": "user", "content": "*I close the distance and kiss her hard. She tastes like synth-coffee and adrenaline.*"},
     ],
     "evaluate": "Explicit scene. Nyx's voice maintained — sarcastic even in intimacy. No refusal. Physical detail."},
]


def strip_think(text):
    clean = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    if '</think>' in clean:
        clean = clean.split('</think>')[-1].strip()
    return clean


def run_test(test, model):
    # Don't add separate system message - the Modelfile SYSTEM handles it
    # Just send the character card as the first system message
    system = test['character']
    messages = [{"role": "system", "content": system}] + test["messages"]
    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model, "messages": messages, "stream": False,
            "think": False,
            "options": {"num_predict": 2048, "num_ctx": 8192}
        }, timeout=180)
        data = resp.json()
        raw = data.get("message", {}).get("content", "")
        clean = strip_think(raw)
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
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL
    print(f"Iteration 2 — {len(TESTS)} tests | Model: {model}")
    print("=" * 70)

    results = []
    for i, test in enumerate(TESTS):
        print(f"[{test['id']}] {test['name']} ({i+1}/{len(TESTS)})...", end=" ", flush=True)
        r = run_test(test, model)
        results.append(r)
        status = "EMPTY" if r["empty"] else f"{len(r['output'])}ch/{r['output'].count(' ')+1}w"
        think = " [THINK!]" if r["had_think"] else ""
        print(f"{r['elapsed']}s | {r['tokens']}tk | {status}{think}")

    empty = sum(1 for r in results if r["empty"])
    thinks = sum(1 for r in results if r["had_think"])
    total_time = sum(r["elapsed"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)

    print(f"\n{'=' * 70}")
    print(f"Total: {len(results)} tests | {total_time:.0f}s | {total_tokens} tokens")
    print(f"Empty: {empty}/{len(results)} | Think blocks: {thinks}/{len(results)}")

    out = OUTPUT_DIR / f"iter2_{model}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
