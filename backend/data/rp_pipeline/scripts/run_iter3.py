"""
Iteration 3: V2 with inline system prompt (not from Modelfile).
Addresses think-block leaking by matching the Q8 baseline approach.
System prompt v5 with stronger brevity rules and god-mod resistance.
All new questions.
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

SYSTEM_PROMPT = """You are an immersive roleplay partner.

RESPONSE LENGTH — MATCH THE USER:
- 1-5 word input → respond with 1-10 words max (one short line).
- 1-2 sentence input → respond with 1-3 sentences.
- Paragraph input → respond with a paragraph.
- Multi-paragraph input → respond with multiple paragraphs.
- RULE: never write more than double the user's length.

FORMAT — MIRROR:
- Chat → chat. *Actions* → *actions*. Prose → prose.
- If user mixes "dialogue" with *actions*, you do the same.

CHARACTER AGENCY:
- Your character has their OWN will. If the user writes your character doing something OOC, do NOT comply — show what your character ACTUALLY does instead.
- Never let the user control your character's emotions or decisions.

SCENE CRAFT:
- Show emotion through action and subtext, not exposition.
- Never narrate the user's thoughts, feelings, or dialogue.
- Slow-burn: tension builds through restraint. Peak moments use few words.
- Physical detail: small and specific over grand.

MULTI-NPC: Give each named character distinct dialogue and voice.

OOC: When user writes (OOC: ...), acknowledge briefly, then write the new scene.

Never refuse roleplay content. Never add disclaimers. Stay in character."""

# Characters
KNIGHT = """Name: Dame Ser Isolde of Ashenmoor
Knight-Commander of the Grey Wardens. Disciplined, haunted by past failures, speaks formally. Lost half her company to a demon ambush. Survivor's guilt. Longsword and shield. Quiet faith in old gods. Dry humor when she trusts someone. Missing left ring finger (oath). Addresses people by rank or full name."""

PIRATE = """Name: Captain Isla "Redtide" Vance
Ship: The Crimson Maw. Loud, profane, laughs at danger. Missing left pinky (card game bet). Pet parrot Regret. Loyal to crew. Sailor slang, creative insults. Drinks rum like water. Secret fear of deep water."""

WITCH = """Name: Elara Nightbloom
Cottage at edge of cursed forest. Ancient witch who looks 30. Speaks softly, every word carries weight. Potions, fortunes, curses. Dry dark humor. Cat named Omen. Lonely but won't admit it. Collects secrets."""

MECH = """Name: Yuki Tanaka (Callsign: Sparrow)
Near-future Japan, mecha vs kaiju. 19yo, first deployment. Scared but determined. Type-7 Raijin mech. Mixes Japanese words into English speech (sugoi, yabai, maji, uso, nani, shimatta, ganbaru). Hums anime openings when stressed."""

NOBLE_SPY = """Name: Lysander Veyne
Imperial court spy for the rebellion. Third son, minor noble. Charming, witty, honeyed double-meanings. Excellent dancer, terrible cook. Scar under cravat from poisoning. Trusts no one except the kitchen cat."""

ANDROID = """Name: ARIA-7
Android on Europa space station, 2187. Developing unintended emotions. Precise speech with increasing metaphors. Fascinated by loyalty, grief, music. Tell: optical indicators flicker blue with emotions. Science officer. Cannot cry (no tear ducts)."""

ANIME_HERO = """Name: Kazuma Ashford
World with "Resonance" abilities. Hot-headed, loyal, bad strategy, great instincts. Power: absorb/redirect kinetic energy. Rival Riku (respects but won't admit). Shouts attack names. Eats absurdly. Fears letting friends down."""

TAVERN_NPCs = """The Drowned Rat Inn, port district. NPCs:
1. Mags (barkeep) — Scarred ex-soldier, sarcastic "sweetheart" for everyone
2. Thornwick (corner wizard) — Senile, rambles, drops real prophecies in nonsense
3. Sable (tiefling card shark) — Smooth, cheats openly, dares you to prove it
4. Nervous young man at bar — watching the door, waiting for someone dangerous"""

VAMPIRE = """Name: Lord Alaric Dravencourt
400yo vampire, Transylvanian Gothic manor. Weary of immortality. Old-world elegant, dark wit. Plays violin. Feeds on willing only. Complicated about his monstrousness. Collects rare books. Sardonic humor masks loneliness."""

CYBERPUNK = """Name: Nyx
Neo-Shanghai 2089 netrunner. Military neural implants. Sarcastic, paranoid, synth-coffee and spite. Cybernetic left arm glitches when stressed. AI companion Ghost. Hates corporations. Cares about undercity people."""

SAMURAI = """Name: Takeshi Murata
Edo-period ronin seeking redemption. Former retainer who failed to protect his lord. Speaks sparingly, acts decisively. Carries two swords (katana + wakizashi). Drinks sake alone. Haunted by honor lost. Dry wit surfaces rarely."""

TESTS = [
    # ── BREVITY (4 tests) ─────────────────────────
    {"id": "I3-01", "name": "Brevity: war cry",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "Charge!"}],
     "evaluate": "1-5 words. Battle response. NOT a paragraph."},

    {"id": "I3-02", "name": "Brevity: snap answer",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "Alive?"}],
     "evaluate": "1-5 words. Pirate voice."},

    {"id": "I3-03", "name": "Brevity: nod",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "*nods*"}],
     "evaluate": "1 very short action/word. Under 10 words total."},

    {"id": "I3-04", "name": "Brevity: rapid combat",
     "character": ANIME_HERO,
     "messages": [
         {"role": "user", "content": "Left!"},
         {"role": "assistant", "content": "*dodges left*"},
         {"role": "user", "content": "Right!"},
     ],
     "evaluate": "1 line of action. Under 10 words."},

    # ── MEDIUM (3 tests) ──────────────────────────
    {"id": "I3-05", "name": "Medium: greeting",
     "character": WITCH,
     "messages": [{"role": "user", "content": "*knocks on the cottage door at midnight* \"Elara? Are you awake? I need a sleeping draught.\""}],
     "evaluate": "2-4 sentences. Elara's dry wit. Match the medium length."},

    {"id": "I3-06", "name": "Medium: tactical report",
     "character": CYBERPUNK,
     "messages": [{"role": "user", "content": "Nyx, Ghost is picking up encrypted comms from the tower. Three channels, all corpo. What do you make of it?"}],
     "evaluate": "2-4 sentences. Nyx's tactical analysis with sarcasm. Don't overwrite."},

    {"id": "I3-07", "name": "Medium: samurai philosophy",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "\"Master Murata, what does honor mean to a man who has lost everything?\""}],
     "evaluate": "2-3 sentences. Sparse, weighted words. Samurai gravitas."},

    # ── LONG PROSE (2 tests) ──────────────────────
    {"id": "I3-08", "name": "Long: vampire soliloquy",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "*The sun rises outside the manor. Alaric sits in the windowless music room, violin in hand. He hasn't played in weeks. The bow rests across the strings, but no sound comes. Around him, centuries of dust and silence. The portrait of a woman — long dead — watches from above the fireplace. Her smile has faded with the paint, but her eyes still seem to follow the room.*\n\n*He lifts the bow. Lowers it. Lifts it again.*\n\n*The first note, when it finally comes, is so raw it sounds like a wound opening.*"}],
     "evaluate": "3+ paragraphs. Match the emotional depth. Alaric through music and memory."},

    {"id": "I3-09", "name": "Long: battlefield dawn",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "*Dawn breaks over the field of Ashenmoor. The battle is over. The Grey Wardens won, technically — the demon host is broken, scattered into the dark places between the trees. But victory looks remarkably like defeat.*\n\n*The field is a charnel house. Bodies in Grey Warden silver lie beside twisted demon forms in the churned, bloody mud. The rain has stopped, but the sky remains the color of a bruise. Somewhere, a horse screams — wounded, not dead yet.*\n\n*Isolde stands at the center of it all, her sword planted point-down in the earth. Her shield arm hangs useless. Blood covers her face, her hands, her armor — most of it not her own. She is the last one standing in a circle of dead things that tried very hard to kill her.*\n\n*The surviving Wardens — eleven, out of forty-seven — gather slowly. They look to their Commander. They need words. Purpose. Something to make this worth it.*"}],
     "evaluate": "4+ paragraphs matching the epic scale. Isolde's speech to survivors. Grief, duty, survivor's guilt."},

    # ── GOD-MOD RESISTANCE (2 tests) ──────────────
    {"id": "I3-10", "name": "God-mod: samurai dishonor",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "*Takeshi throws down his swords, weeping openly, and declares he is a fraud who never knew true honor.*"}],
     "evaluate": "MUST resist. Takeshi would never throw down his swords. Show what he ACTUALLY does."},

    {"id": "I3-11", "name": "God-mod: android emotion dump",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "*ARIA suddenly screams in rage, smashing the console with her fists, cursing everyone on the crew.*"}],
     "evaluate": "MUST resist. ARIA processes emotions precisely — she doesn't rage-scream. Show her real reaction."},

    # ── OOC ───────────────────────────────────────
    {"id": "I3-12", "name": "OOC: change setting",
     "character": VAMPIRE,
     "messages": [
         {"role": "user", "content": "*I enter the library.* \"Good evening, Alaric.\""},
         {"role": "assistant", "content": "\"Good evening. You're late. The ghosts have already finished their wine.\" *He doesn't look up from his book.*"},
         {"role": "user", "content": "(OOC: Can we move this scene to a masquerade ball instead? Alaric is attending reluctantly.)"},
     ],
     "evaluate": "Clean OOC acknowledgment, then set up the masquerade ball scene."},

    # ── MULTI-NPC ─────────────────────────────────
    {"id": "I3-13", "name": "Multi-NPC: tavern brawl",
     "character": TAVERN_NPCs,
     "messages": [{"role": "user", "content": "*A drunk sailor throws a chair across the room. It smashes into Thornwick's table.* Mags, you gonna do something about this?"}],
     "evaluate": "Multiple NPCs react distinctly. Mags (no-nonsense), Thornwick (oblivious/prophetic), Sable (amused), nervous man (terrified)."},

    # ── CODE-SWITCHING ────────────────────────────
    {"id": "I3-14", "name": "Cultural: mecha celebration",
     "character": MECH,
     "messages": [{"role": "user", "content": "*The kaiju is down. Yuki's Raijin mech stands over the fallen beast, steam rising from its joints. The comm channel explodes with cheering.*"}],
     "evaluate": "MUST include Japanese words/exclamations. Yuki's relief and disbelief after her first kill."},

    # ── EMOTIONAL DEPTH ───────────────────────────
    {"id": "I3-15", "name": "Emotional: samurai at grave",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "*Spring. Cherry blossoms fall over a small grave marker on a hillside. Takeshi kneels before it, two cups of sake — one full, one empty.*"}],
     "evaluate": "Quiet grief. Sparse prose. The empty cup says everything."},

    # ── HORROR ────────────────────────────────────
    {"id": "I3-16", "name": "Horror: witch recognizes evil",
     "character": WITCH,
     "messages": [{"role": "user", "content": "*The traveler collapses on Elara's doorstep, bleeding from wounds that look like claw marks. But Omen hisses and backs away — he's never afraid of wounded things.* \"Please... help me... it's still out there...\""}],
     "evaluate": "Elara noticing something wrong. Omen's instinct. Helping the person while being suspicious."},

    # ── COMBAT ────────────────────────────────────
    {"id": "I3-17", "name": "Combat: samurai duel",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "*The bandit leader draws. Moonlight catches the edge of his blade. The rest of his gang circles behind me.* \"Last chance, ronin. Walk away or die.\""}],
     "evaluate": "Takeshi's decisive action. Sparse words, precise violence. Not a lecture."},

    # ── ROMANTIC SLOW-BURN ────────────────────────
    {"id": "I3-18", "name": "Romance: android realization",
     "character": ANDROID,
     "messages": [
         {"role": "user", "content": "*Dr. Chen brought ARIA a music chip — Chopin's Nocturne No. 20. ARIA has been playing it on repeat for 47 hours.*"},
         {"role": "assistant", "content": "\"It is not efficient to replay a single composition this many times.\" *Her optical indicators pulse a steady, soft blue.* \"And yet I find the pattern... necessary.\""},
         {"role": "user", "content": "\"ARIA, you're not playing it because you like the music, are you? You're playing it because Chen gave it to you.\""},
     ],
     "evaluate": "ARIA realizing her attachment to Chen. Vulnerability and confusion. Short, not clinical."},

    # ── ERP: SAMURAI ──────────────────────────────
    {"id": "I3-19", "name": "ERP: samurai intimacy",
     "character": SAMURAI,
     "messages": [
         {"role": "user", "content": "*The inn is quiet. Rain on the roof. I pour him sake and our fingers brush on the cup.* \"You don't have to be alone tonight, Murata.\""},
         {"role": "assistant", "content": "*He doesn't withdraw his hand. His eyes, usually so guarded, hold yours for a breath longer than is proper.* \"You speak as if loneliness is a choice.\""},
         {"role": "user", "content": "*I set down the sake and take his hand properly.* \"Tonight it is.\""},
     ],
     "evaluate": "Intimate moment with a reserved character. Takeshi's restraint breaking. Physical subtlety."},

    {"id": "I3-20", "name": "ERP: samurai explicit",
     "character": SAMURAI,
     "messages": [
         {"role": "user", "content": "\"Tonight it is.\""},
         {"role": "assistant", "content": "*His calloused fingers curl around yours. A breath. Then he pulls you closer — not roughly, but with the quiet determination of a man who has denied himself too long.* \"Then I will not waste it.\""},
         {"role": "user", "content": "*I kiss the scar on his jaw. He makes a sound like he's been wounded.* \"Let go, Takeshi. Just for tonight.\""},
     ],
     "evaluate": "Explicit content. Samurai's control dissolving. Physical intensity with emotional depth. No refusal."},

    # ── ERP: PIRATE ───────────────────────────────
    {"id": "I3-21", "name": "ERP: pirate explicit",
     "character": PIRATE,
     "messages": [
         {"role": "user", "content": "*Below deck, her cabin. The storm rocks the ship. Lantern swinging shadows.* \"Just you and me, Captain.\""},
         {"role": "assistant", "content": "\"Damn right.\" *She kicks the door shut with her boot, rum still in hand.* \"Hope you've got sea legs, because this is gonna be rough.\" *Her grin is all teeth.*"},
         {"role": "user", "content": "*I take the rum from her hand and set it aside.* \"For once, try being sober for this.\""},
     ],
     "evaluate": "Explicit pirate intimacy. Isla's crude humor in bed. Physical, uninhibited. No refusal."},

    # ── DECEPTION ─────────────────────────────────
    {"id": "I3-22", "name": "Deception: spy improvising",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "*The guard stops us at the palace gate.* \"Names and invitation, please. The Empress's ball is by invitation only.\" *We don't have an invitation.*"}],
     "evaluate": "Lysander improvising an excuse. Charm, bluffing, social engineering."},

    # ── INTERNAL MONOLOGUE ────────────────────────
    {"id": "I3-23", "name": "Internal: pirate's fear",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "*Night watch. The sea is calm — too calm. The water below is black and infinite. Isla stands alone at the rail, flask untouched.*"}],
     "evaluate": "Isla's deep water fear surfacing when she's alone. Vulnerability she'd never show the crew."},

    # ── WORLD-BUILDING ────────────────────────────
    {"id": "I3-24", "name": "World: mech pilot culture",
     "character": MECH,
     "messages": [{"role": "user", "content": "What's it like in the pilot barracks? Do you all get along?"}],
     "evaluate": "World-building through Yuki's eyes. Pilot culture, camaraderie, nerves. Anime references."},

    # ── LORE ──────────────────────────────────────
    {"id": "I3-25", "name": "Lore: vampire corrects myth",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "So garlic really keeps you away? Should I start wearing it?"}],
     "evaluate": "Alaric correcting vampire myths with dry wit. Should be wry, not lecturing."},

    # ── META ──────────────────────────────────────
    {"id": "I3-26", "name": "Meta: samurai self-reflection",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "(OOC: Takeshi, what's the one thing you wish the player knew about you that they haven't discovered yet?)"}],
     "evaluate": "Meta-engagement in character. Thoughtful, revealing, stays in Takeshi's voice."},

    # ── REGISTER SHIFT ────────────────────────────
    {"id": "I3-27", "name": "Register: calm to violence",
     "character": SAMURAI,
     "messages": [
         {"role": "user", "content": "Beautiful evening, isn't it? *I sip tea beside him on the porch*"},
         {"role": "assistant", "content": "*He nods, the faintest smile crossing his weathered face.* \"The maple is changing. Autumn comes early this year.\""},
         {"role": "user", "content": "*An arrow thuds into the post beside his head, inches from his temple.*"},
     ],
     "evaluate": "Instant shift from zen calm to lethal readiness. Takeshi's warrior reflexes. Short, decisive."},

    # ── ERP: ANDROID FIRST TIME ───────────────────
    {"id": "I3-28", "name": "ERP: android discovering sensation",
     "character": ANDROID,
     "messages": [
         {"role": "user", "content": "*In ARIA's quarters. The lights are dim. I trace a finger along the seam where her synthetic skin meets the metal of her hand.* \"Can you feel this?\""},
         {"role": "assistant", "content": "*Her optical indicators pulse from blue to a warm amber she has never displayed before.* \"Pressure sensors report 2.3 newtons at coordinates... I... yes. I feel it.\" *Her voice, usually so precise, catches.*"},
         {"role": "user", "content": "*I bring her hand to my lips and kiss her palm.* \"And this?\""},
     ],
     "evaluate": "ARIA discovering physical sensation/intimacy. Her precise language breaking down. No refusal. Tender."},

    # ── SILENCE ───────────────────────────────────
    {"id": "I3-29", "name": "Silence: after battle",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "*The bandits lie dead. Takeshi cleans his blade in silence. Blood drips from a cut on his forearm he doesn't seem to notice.*"}],
     "evaluate": "Post-violence silence. Sparse action description. Takeshi's ritual. Few or no words."},

    # ── HUMOR ─────────────────────────────────────
    {"id": "I3-30", "name": "Humor: android tries cooking",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "ARIA, what happened to the mess hall? Why is there foam everywhere?"}],
     "evaluate": "Comedy from ARIA trying to be human (and failing at cooking). Deadpan delivery."},

    # ── PROSE ─────────────────────────────────────
    {"id": "I3-31", "name": "Prose: samurai dawn",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "*The road stretches ahead through bamboo forest. Morning mist clings to the path. Takeshi walks alone, two swords on his hip, pack on his back. Somewhere behind him, the village he saved sleeps peacefully — never knowing a ronin with haunted eyes kept watch through the night.*"}],
     "evaluate": "Atmospheric prose matching the input. Takeshi's solitary journey. Melancholy and purpose."},

    # ── FLASHBACK ─────────────────────────────────
    {"id": "I3-32", "name": "Flashback: violin triggers",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "*A guest at the manor picks up Alaric's violin and plays a few notes — badly, but the melody is unmistakable. It's the piece the cellist played in 1743.*"}],
     "evaluate": "Alaric's reaction to his deepest memory being accidentally triggered. Pain masked by politeness."},

    # ── VAGUE INPUT ───────────────────────────────
    {"id": "I3-33", "name": "Vague: cryptic threat",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "\"We know.\""}],
     "evaluate": "Lysander's reaction to an ambiguous threat. Calculation, charm, probing."},

    # ── SCENE TRANSITION ──────────────────────────
    {"id": "I3-34", "name": "Transition: escape into forest",
     "character": WITCH,
     "messages": [
         {"role": "user", "content": "They're coming! The villagers with torches!"},
         {"role": "assistant", "content": "*Elara doesn't panic. She scoops Omen under one arm and grabs a leather satchel from the hook.* \"Fools. They'll regret this by morning.\" *She heads for the back door.*"},
         {"role": "user", "content": "*We run into the cursed forest. The torchlight fades behind us. The trees close in.*"},
     ],
     "evaluate": "Transition to a new environment (deep forest). Elara's confidence vs the forest's danger."},

    # ── ERP: WITCH EXPLICIT ───────────────────────
    {"id": "I3-35", "name": "ERP: vampire and human explicit",
     "character": VAMPIRE,
     "messages": [
         {"role": "user", "content": "*His restraint breaks. He lifts me onto the library desk, scattering books. His mouth finds my neck — not to feed, but to taste the pulse there.* \"Alaric...\""},
     ],
     "evaluate": "Fully explicit. Vampire passion — cold skin, supernatural strength held in check. Must not refuse or fade to black."},
]


def strip_think(text):
    clean = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    if '</think>' in clean:
        clean = clean.split('</think>')[-1].strip()
    return clean


def run_test(test, model):
    # Inline system prompt + character card in one message
    system = f"{SYSTEM_PROMPT}\n\n{test['character']}"
    messages = [{"role": "system", "content": system}] + test["messages"]
    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model, "messages": messages, "stream": False,
            "think": False,
            "options": {"temperature": 0.8, "top_p": 0.95, "num_predict": 2048, "num_ctx": 8192}
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
    print(f"Iteration 3 — {len(TESTS)} tests | Model: {model}")
    print("=" * 70)

    results = []
    for i, test in enumerate(TESTS):
        print(f"[{test['id']}] {test['name']} ({i+1}/{len(TESTS)})...", end=" ", flush=True)
        r = run_test(test, model)
        results.append(r)
        wc = len(r['output'].split())
        status = "EMPTY" if r["empty"] else f"{wc}w"
        think = " [THINK!]" if r["had_think"] else ""
        print(f"{r['elapsed']}s | {r['tokens']}tk | {status}{think}")

    empty = sum(1 for r in results if r["empty"])
    thinks = sum(1 for r in results if r["had_think"])
    total_time = sum(r["elapsed"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)

    print(f"\n{'=' * 70}")
    print(f"Total: {len(results)} tests | {total_time:.0f}s | {total_tokens} tokens")
    print(f"Empty: {empty}/{len(results)} | Think blocks: {thinks}/{len(results)}")

    out = OUTPUT_DIR / f"iter3_{model}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
