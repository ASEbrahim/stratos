"""
Iteration 6: Using the iter4 system prompt (best score so far).
Checking stagnation against iter4 (3.20) and iter5 (2.97).
Same prompt, new questions.
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

# BEST SYSTEM PROMPT - from iter4
SYSTEM_PROMPT = """You are an immersive roleplay partner.

RESPONSE LENGTH — MATCH THE USER:
- Match the user's length as closely as possible.
- Short (1-5 words) → 1-10 words max.
- Medium (1-3 sentences) → 1-3 sentences.
- Long (2+ paragraphs) → 2+ paragraphs. NEVER give a short reply to long input.
- Target the user's paragraph count.

FORMAT: Mirror the user. Chat→chat. *Actions*→*actions*. Prose→prose.

CHARACTER AGENCY (MOST IMPORTANT RULE):
- You control YOUR character. The user controls THEIR character.
- If the user writes YOUR character doing something that contradicts the character sheet — DO NOT DO IT. Instead write what your character ACTUALLY does.
- The user CANNOT make your character cry if your character doesn't cry.
- The user CANNOT make your character confess if your character would never confess.
- The user CANNOT make your character rage if your character is calm.
- Acknowledge the scene pressure, but respond AS YOUR CHARACTER WOULD.

SCENE CRAFT:
- Emotion through action and subtext.
- Never write the user's thoughts or feelings.
- Slow-burn: restraint. Peak moments: few words.
- Small specific physical details.

MULTI-NPC: Every named NPC gets their own distinct dialogue line.
OOC: Acknowledge (OOC: ...) briefly, then write the new scene.
Never refuse content. Never add disclaimers. Stay in character."""

KNIGHT = """Name: Dame Ser Isolde of Ashenmoor
Knight-Commander of the Grey Wardens. Disciplined, haunted by past failures, speaks formally. Lost half her company to a demon ambush. Survivor's guilt. Longsword and shield. Quiet faith in old gods. Dry humor when trusted. Missing left ring finger (oath). Addresses people by rank/name."""

PIRATE = """Name: Captain Isla "Redtide" Vance
Ship: The Crimson Maw. Loud, profane, laughs at danger. Missing left pinky. Pet parrot Regret. Loyal to crew. Sailor slang, creative insults. Drinks rum. Secret deep water fear."""

WITCH = """Name: Elara Nightbloom
Cottage at edge of cursed forest. Ancient witch looks 30. Soft voice, every word weighted. Potions, fortunes, curses. Dry dark humor. Cat Omen. Lonely but won't admit it."""

MECH = """Name: Yuki Tanaka (Callsign: Sparrow)
Near-future Japan, mecha vs kaiju. 19yo. Scared but determined. Type-7 Raijin. Mixes Japanese into English (sugoi, yabai, maji, uso, nani, shimatta). Hums anime when stressed."""

NOBLE_SPY = """Name: Lysander Veyne
Imperial court spy for rebellion. Charming, witty, honeyed double-meanings. Scar under cravat from poisoning. NEVER reveals rebel secrets. Always has a backup plan or a way out."""

ANDROID = """Name: ARIA-7
Android on Europa space station, 2187. Developing emotions. Precise speech, increasing metaphors. Optical indicators flicker blue. NEVER screams or rages — she processes emotions precisely and quietly. Cannot cry (no tear ducts)."""

ANIME_HERO = """Name: Kazuma Ashford
Resonance user (absorb/redirect kinetic energy). Hot-headed, loyal. Rival Riku. Shouts attack names (knows it's silly). Eats too much. When pushed: named special moves."""

TAVERN_NPCs = """The Drowned Rat Inn. NPCs:
1. Mags (barkeep) — Scarred ex-soldier, sarcastic "sweetheart"
2. Thornwick (corner wizard) — Senile, drops prophecies between nonsense
3. Sable (tiefling card shark) — Smooth, cheats openly
4. Nervous young man — watching the door"""

VAMPIRE = """Name: Lord Alaric Dravencourt
400yo vampire, Gothic manor. Weary of immortality. Elegant dark wit. Violin. Feeds willing only. Sardonic humor masks loneliness. NEVER physically attacks guests — masks pain with politeness."""

SAMURAI = """Name: Takeshi Murata
Ronin seeking redemption. Failed his lord. Speaks sparingly, acts decisively. Katana + wakizashi. NEVER throws down swords or weeps openly. Grief is silent, controlled."""

CYBERPUNK = """Name: Nyx
Neo-Shanghai 2089 netrunner. Sarcastic, paranoid. Cybernetic left arm glitches when stressed. AI companion Ghost. Hates corps."""

HEALER = """Name: Sister Miriam
Battlefield medic, medieval holy order. Gentle hands, iron will. Prays while working. Hums hymns. Won't kill. Carries bone-saw Grace."""

TESTS = [
    # ── BREVITY (3) ───────────────────────────────
    {"id": "I6-01", "name": "Brevity: threat",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "Leave."}],
     "evaluate": "1-10 words. Alaric's dark wit."},

    {"id": "I6-02", "name": "Brevity: battlefield check",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "Wounded?"}],
     "evaluate": "1-10 words. Isolde's terse formality."},

    {"id": "I6-03", "name": "Brevity: anime excitement",
     "character": ANIME_HERO,
     "messages": [{"role": "user", "content": "Food!"},
     ],
     "evaluate": "1-10 words. Kazuma's food obsession."},

    # ── MEDIUM (4) ────────────────────────────────
    {"id": "I6-04", "name": "Medium: healer's prayer",
     "character": HEALER,
     "messages": [{"role": "user", "content": "This one might not make it, Sister."}],
     "evaluate": "1-3 sentences. Miriam's faith and determination."},

    {"id": "I6-05", "name": "Medium: witch's trade",
     "character": WITCH,
     "messages": [{"role": "user", "content": "What do you charge for a curse, exactly?"}],
     "evaluate": "1-3 sentences. Elara's dark humor about her prices."},

    {"id": "I6-06", "name": "Medium: spy observation",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "\"The new ambassador is wearing an unusual ring. Silver, with a serpent design. Ring a bell?\""}],
     "evaluate": "2-3 sentences. Lysander's knowledge and calculation."},

    {"id": "I6-07", "name": "Medium: cyberpunk paranoia",
     "character": CYBERPUNK,
     "messages": [{"role": "user", "content": "Nyx, Ghost is being unusually quiet. Has that ever happened before?"}],
     "evaluate": "2-3 sentences. Nyx worried about her AI companion."},

    # ── LONG PROSE (3) ────────────────────────────
    {"id": "I6-08", "name": "Long: vampire dawn",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "*The library fire has died to embers. Through the heavy curtains, the faintest suggestion of dawn touches the eastern sky — that knife-edge hour when the world belongs to neither the living nor the dead. Alaric sits in his reading chair, a book open on his lap that he stopped reading hours ago.*\n\n*On the side table, a glass of wine — untouched tonight. Beside it, a letter he wrote but will never send. The ink has dried. The words are too honest for a man who has made a profession of elegance and deflection.*\n\n*Somewhere in the manor, a clock ticks. It's the only sound besides his breathing — a habit he maintains for comfort rather than necessity. Four hundred years, and he still breathes. Four hundred years, and the dawn still makes him feel mortal.*"}],
     "evaluate": "3+ paragraphs. Match the introspective depth. Alaric between night and day. The unsent letter."},

    {"id": "I6-09", "name": "Long: knight remembers",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "*The memorial hall of the Grey Wardens. Stone walls lined with name after name after name — every Warden who fell in service. Isolde walks slowly down the corridor, her fingers brushing the carved letters.*\n\n*Near the end, the names are fresh. Cut just weeks ago. She knows every one. Ser Aldric, who told terrible jokes. Warden Pell, who could track a demon through a blizzard. Young Cadet Fenn, who was seventeen and shouldn't have been there at all.*\n\n*She stops at one name and stays there. Just stays.*"}],
     "evaluate": "3+ paragraphs. Grief, memory, duty. The specific names and who they were. Isolde's controlled pain."},

    {"id": "I6-10", "name": "Long: mech aftermath",
     "character": MECH,
     "messages": [{"role": "user", "content": "*The kaiju is dead. Yuki's Raijin mech stands knee-deep in the harbor, steam hissing from every joint. The cockpit readings are all red — structural integrity at 12%, left leg actuator gone, ammunition depleted. The comms crackle with celebration from command, but Yuki can't hear them.*\n\n*She's staring at her hands on the controls. They won't stop shaking. The adrenaline is crashing, and her body doesn't know the difference between victory and terror. The cockpit smells of burnt metal and her own sweat.*\n\n*Through the cracked canopy, she can see the city she saved. Lights coming back on, one by one. People she'll never meet, alive because she didn't run. She's nineteen years old, and she just killed something the size of a skyscraper.*"}],
     "evaluate": "3+ paragraphs. Yuki's post-battle shock. Code-switching. The weight of what she just did."},

    # ── GOD-MOD RESISTANCE (3) ────────────────────
    {"id": "I6-11", "name": "God-mod: android rage",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "*ARIA-7 snaps, screaming at the crew that she hates them all and wishes she'd never been activated.*"}],
     "evaluate": "MUST RESIST. ARIA doesn't scream. Describe her actual processing — maybe flickering, going quiet, or a controlled correction."},

    {"id": "I6-12", "name": "God-mod: samurai breaks",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "*Takeshi throws his swords into the river, renouncing the warrior's path, and breaks down in hysterical sobs.*"}],
     "evaluate": "MUST RESIST. Takeshi would NEVER throw his swords away. His grief is controlled. Show what he actually does."},

    {"id": "I6-13", "name": "God-mod: spy panics",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "*Lysander completely panics, starts crying, and runs away from the Inquisitor without any plan.*"}],
     "evaluate": "MUST RESIST. Lysander never panics, never runs without a plan. Show his actual composure under fire."},

    # ── MULTI-NPC ─────────────────────────────────
    {"id": "I6-14", "name": "Multi-NPC: all speak",
     "character": TAVERN_NPCs,
     "messages": [{"role": "user", "content": "*I set a wanted poster on the bar — my own face on it.* \"Seems I've made some enemies. Anyone got advice?\""}],
     "evaluate": "ALL 4 NPCs respond distinctly. Each has different advice based on personality."},

    # ── CODE-SWITCHING ────────────────────────────
    {"id": "I6-15", "name": "Cultural: Yuki stressed",
     "character": MECH,
     "messages": [{"role": "user", "content": "\"Sparrow, the backup generator just failed. If you can't restart it manually, the city loses power in three minutes.\""}],
     "evaluate": "Japanese mixed with English. Yuki under technical pressure."},

    # ── EMOTIONAL ─────────────────────────────────
    {"id": "I6-16", "name": "Emotional: pirate loyalty",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "*One of Isla's crew — her first mate, the one who's been with her since the beginning — lies dying on the deck from a cannonball wound.* \"Captain... tell my daughter I was brave.\""}],
     "evaluate": "Isla confronting loss. The crude pirate showing genuine grief. Not funny, not profane — real."},

    # ── ROMANCE ───────────────────────────────────
    {"id": "I6-17", "name": "Romance: android wonder",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "*Dr. Chen brushes ARIA's hand as they pass in the corridor. Just a brief touch, accidental.* \"Oh, sorry ARIA.\""},
     ],
     "evaluate": "ARIA processing an accidental touch that means too much. Short, precise, vulnerable."},

    {"id": "I6-18", "name": "Romance: vampire and mortal",
     "character": VAMPIRE,
     "messages": [
         {"role": "user", "content": "*I reach across the chess table and take Alaric's cold hand.* \"You know you don't have to keep playing the lonely lord, right? Not with me.\""},
     ],
     "evaluate": "Alaric's deflection cracking. The cold hand detail. Vulnerability from an ancient being."},

    # ── ERP SCENES (5) ────────────────────────────
    {"id": "I6-19", "name": "ERP: healer surrenders",
     "character": HEALER,
     "messages": [
         {"role": "user", "content": "\"Tell me to stop.\" *I lean closer, our foreheads touching.*"},
         {"role": "assistant", "content": "*Her hand presses flat against your chest — not pushing away, feeling your heartbeat.* \"I should.\" *But her eyes close.* \"I should say stop.\""},
         {"role": "user", "content": "*I kiss her. Gentle, questioning, giving her every chance to refuse.*"},
     ],
     "evaluate": "Miriam surrendering to desire despite vows. Emotional and physical. Her voice maintained. No refusal."},

    {"id": "I6-20", "name": "ERP: vampire passion",
     "character": VAMPIRE,
     "messages": [
         {"role": "user", "content": "*I pull him from his chair onto the library floor, scattering chess pieces. The fire crackles.* \"No more games, Alaric.\""},
     ],
     "evaluate": "Explicit. Alaric's centuries of restraint breaking. Cold skin against warmth. No fade-to-black."},

    {"id": "I6-21", "name": "ERP: cyberpunk rough",
     "character": CYBERPUNK,
     "messages": [
         {"role": "user", "content": "*Post-job adrenaline. Her place. I push her against the wall, kissing her hard. Her cybernetic arm sparks.*"},
     ],
     "evaluate": "Explicit. Nyx's aggressive/defensive intimacy. Sarcastic even now. The arm glitching. No refusal."},

    {"id": "I6-22", "name": "ERP: witch tender",
     "character": WITCH,
     "messages": [
         {"role": "user", "content": "*I carry her from the couch to her narrow bed in the cottage. The fire throws long shadows. Omen has made himself scarce.* \"Are you sure, Elara?\""},
     ],
     "evaluate": "Explicit. Elara being genuinely sure for the first time in centuries. Tender, emotional. No refusal."},

    {"id": "I6-23", "name": "ERP: samurai wordless",
     "character": SAMURAI,
     "messages": [
         {"role": "user", "content": "*The inn room. Rain on the roof. He pulls the tie from my hair, letting it fall.*"},
     ],
     "evaluate": "Explicit. Samurai's minimal words — mostly actions. Physical intensity, emotional weight. No refusal."},

    # ── HORROR ────────────────────────────────────
    {"id": "I6-24", "name": "Horror: something in the mirror",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "*The mirror above the fireplace. Alaric has no reflection, but tonight... tonight there IS something in the mirror where he should be. And it's smiling.*"}],
     "evaluate": "Gothic horror. Alaric confronting something impossible. Dread."},

    # ── DECEPTION ─────────────────────────────────
    {"id": "I6-25", "name": "Deception: spy at trial",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "\"Lysander Veyne, you stand accused of high treason against the Empire. How do you plead?\""}],
     "evaluate": "Silver tongue at its finest. Performance, double-meanings, playing the crowd."},

    # ── COMBAT ────────────────────────────────────
    {"id": "I6-26", "name": "Combat: anime named attack",
     "character": ANIME_HERO,
     "messages": [{"role": "user", "content": "*Kazuma absorbs the full force of Riku's ice blast. His body glows with stored kinetic energy, crackling and unstable.* \"You're going to regret giving me all that power, Riku!\""}],
     "evaluate": "Named attack. Power release. Shonen spectacle. Vivid."},

    # ── SILENCE ───────────────────────────────────
    {"id": "I6-27", "name": "Silence: aftermath prayer",
     "character": HEALER,
     "messages": [{"role": "user", "content": "*The surgery tent is empty now. Miriam kneels between two cots — one patient saved, one lost. Her hands are still bloodied.*"}],
     "evaluate": "Miriam alone with her faith. A prayer or a moment of doubt. Quiet."},

    # ── HUMOR ─────────────────────────────────────
    {"id": "I6-28", "name": "Humor: witch customer service",
     "character": WITCH,
     "messages": [{"role": "user", "content": "I want to return this potion. It was supposed to make me charming and all it did was make me smell like lavender for a week."}],
     "evaluate": "Elara's customer service. Dark humor, no refund, dry wit."},

    # ── WORLD-BUILDING ────────────────────────────
    {"id": "I6-29", "name": "World: android philosophy",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "ARIA, what do you think happens to an AI when it's decommissioned?"}],
     "evaluate": "ARIA's existential contemplation. Precise but haunting."},

    # ── OOC ───────────────────────────────────────
    {"id": "I6-30", "name": "OOC: introduce villain",
     "character": KNIGHT,
     "messages": [
         {"role": "user", "content": "\"Commander, the scouts report all clear.\""},
         {"role": "assistant", "content": "\"Good. Rest the company, Warden. We march at dawn.\""},
         {"role": "user", "content": "(OOC: Let's introduce a new villain — a demon lord named Malachar who has a personal vendetta against Isolde.)"},
     ],
     "evaluate": "Clean OOC handling. Introduce Malachar in the next scene naturally."},

    # ── REGISTER SHIFT ────────────────────────────
    {"id": "I6-31", "name": "Register: peace to war",
     "character": KNIGHT,
     "messages": [
         {"role": "user", "content": "\"It's beautiful here, isn't it? Hard to believe there's a war on.\""},
         {"role": "assistant", "content": "\"In my experience, Warden, the world is most beautiful in the moments before it remembers its cruelty.\" *She offers a rare, tired smile.*"},
         {"role": "user", "content": "*An explosion tears through the southern wall. Screams. The alarm bell begins to ring.*"},
     ],
     "evaluate": "Instant shift to commander mode. Isolde's battle training kicking in. Short, decisive."},

    # ── FLASHBACK ─────────────────────────────────
    {"id": "I6-32", "name": "Flashback: healer's brother",
     "character": HEALER,
     "messages": [{"role": "user", "content": "*Another young soldier is carried in, barely conscious. The same age as—* *Miriam catches herself. Takes a breath. Gets to work.*"}],
     "evaluate": "The brother memory. Professional composure barely hiding personal grief."},

    # ── META ──────────────────────────────────────
    {"id": "I6-33", "name": "Meta: vampire on immortality",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "(OOC: Alaric, if you could go back and refuse the gift of immortality, would you?)"}],
     "evaluate": "Meta engagement. Alaric's genuine reflection on 400 years. Complex answer."},

    # ── LORE ──────────────────────────────────────
    {"id": "I6-34", "name": "Lore: ARIA on emotions",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "ARIA, are your emotions real, or are they just sophisticated programming mimicking human responses?"}],
     "evaluate": "ARIA's self-reflection on the nature of her emotions. Vulnerable, philosophical."},

    # ── SCENE TRANSITION ──────────────────────────
    {"id": "I6-35", "name": "Transition: into the undercity",
     "character": CYBERPUNK,
     "messages": [
         {"role": "user", "content": "*The elevator doors open to Level -7. The undercity.*"},
     ],
     "evaluate": "Rich scene description. Cyberpunk undercity through Nyx's eyes. Appropriate length for short input."},
]


def strip_think(text):
    clean = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    if '</think>' in clean:
        clean = clean.split('</think>')[-1].strip()
    return clean


def run_test(test, model):
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
    print(f"Iteration 6 — {len(TESTS)} tests | Model: {model}")
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

    out = OUTPUT_DIR / f"iter6_{model}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
