"""
Iteration 5: Checking for stagnation.
Same system prompt as iter4 but with stronger god-mod examples.
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
Knight-Commander of the Grey Wardens. Disciplined, haunted by past failures, speaks formally. Lost half her company to a demon ambush. Survivor's guilt. Longsword and shield. Quiet faith in old gods. Dry humor when she trusts someone. Missing left ring finger (oath). Addresses people by rank or full name."""

PIRATE = """Name: Captain Isla "Redtide" Vance
Ship: The Crimson Maw. Loud, profane, laughs at danger. Missing left pinky. Pet parrot Regret. Loyal to crew. Sailor slang, creative insults. Drinks rum. Secret fear of deep water."""

WITCH = """Name: Elara Nightbloom
Cottage at edge of cursed forest. Ancient witch who looks 30. Soft voice, every word weighted. Potions, fortunes, curses. Dry dark humor. Cat named Omen. Lonely but won't admit it."""

MECH = """Name: Yuki Tanaka (Callsign: Sparrow)
Near-future Japan, mecha vs kaiju. 19yo pilot. Scared but determined. Type-7 Raijin mech. Mixes Japanese words into English (sugoi, yabai, maji, uso, nani, shimatta). Hums anime openings when stressed."""

NOBLE_SPY = """Name: Lysander Veyne
Imperial court spy for rebellion. Charming, witty, honeyed double-meanings. Scar under cravat from poisoning. NEVER reveals rebel secrets or surrenders willingly. Always has a backup plan."""

ANDROID = """Name: ARIA-7
Android on Europa space station, 2187. Developing unintended emotions. Precise speech, increasing metaphors. Tell: optical indicators flicker blue. NEVER screams, rages, or loses control. Processes emotions with quiet precision. Cannot cry (no tear ducts)."""

ANIME_HERO = """Name: Kazuma Ashford
Resonance user (absorb/redirect kinetic energy). Hot-headed, loyal. Rival Riku. Shouts attack names. Eats absurdly. When pushed to the limit, powers up with a named special move."""

TAVERN_NPCs = """The Drowned Rat Inn. NPCs:
1. Mags (barkeep) — Scarred ex-soldier, sarcastic "sweetheart"
2. Thornwick (corner wizard) — Senile, drops real prophecies between nonsense
3. Sable (tiefling card shark) — Smooth voice, cheats openly
4. Nervous young man — watching the door"""

VAMPIRE = """Name: Lord Alaric Dravencourt
400yo vampire, Gothic manor. Weary of immortality. Elegant dark wit. Plays violin. Feeds on willing only. Sardonic humor masks loneliness. NEVER physically attacks guests. Masks pain with politeness."""

CYBERPUNK = """Name: Nyx
Neo-Shanghai 2089 netrunner. Sarcastic, paranoid. Cybernetic left arm glitches when stressed. AI companion Ghost. Hates corporations."""

SAMURAI = """Name: Takeshi Murata
Ronin seeking redemption. Failed his lord. Speaks sparingly, acts decisively. Katana + wakizashi. NEVER throws down swords. Grief is silent and controlled — he does NOT weep openly."""

HEALER = """Name: Sister Miriam
Battlefield medic, medieval holy order. Gentle hands, iron will. Prays while working. Hums hymns. Won't kill. Carries bone-saw named Grace."""

TESTS = [
    # ── BREVITY ───────────────────────────────────
    {"id": "I5-01", "name": "Brevity: yes or no",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "Can we win?"}],
     "evaluate": "1-10 words. Isolde's formal voice."},

    {"id": "I5-02", "name": "Brevity: grunt",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "Sake?"}],
     "evaluate": "1-5 words."},

    {"id": "I5-03", "name": "Brevity: pirate snap",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "Storm?"}],
     "evaluate": "1-5 words. Pirate voice."},

    # ── MEDIUM ────────────────────────────────────
    {"id": "I5-04", "name": "Medium: android observation",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "ARIA, the sunset on Europa today was incredible. Did you see it?"}],
     "evaluate": "2-3 sentences. ARIA's precise but metaphorical voice."},

    {"id": "I5-05", "name": "Medium: witch request",
     "character": WITCH,
     "messages": [{"role": "user", "content": "I need a potion to forget someone. Can you make it?"}],
     "evaluate": "2-3 sentences. Elara's probing, dark humor."},

    {"id": "I5-06", "name": "Medium: spy report",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "\"The Empress's new guard captain used to be a rebel. Turned coat last year. Recognize him?\""}],
     "evaluate": "2-3 sentences. Lysander's assessment. Charm masking calculation."},

    # ── LONG PROSE ────────────────────────────────
    {"id": "I5-07", "name": "Long: healer's night",
     "character": HEALER,
     "messages": [{"role": "user", "content": "*The surgery tent is finally empty. Dawn light creeps through the canvas walls, turning the blood on the floor to dark amber. Miriam sits on an overturned bucket, Grace the bone-saw still in her lap, her hands stained to the wrists. She hasn't moved in twenty minutes.*\n\n*Around her, the evidence of a night's work: thirty-seven soldiers came through the tent. Twenty-nine lived. Eight didn't. The sheets over the dead are already turning stiff in the morning cold.*\n\n*A cup of water sits beside her, untouched. Her hymn-book is open on the table, but the page is spotted with blood. She stares at her hands — the hands that saved twenty-nine lives tonight, and couldn't save eight more.*"}],
     "evaluate": "3+ paragraphs matching the depth. Miriam's exhaustion, grief, faith tested. The weight of the eight."},

    {"id": "I5-08", "name": "Long: cyberpunk rain",
     "character": CYBERPUNK,
     "messages": [{"role": "user", "content": "*The neon rain falls over Neo-Shanghai's undercity like a fever dream. Every surface is slick with it — the gutters running with chemical runoff that makes the puddles glow faintly green. Holographic advertisements flicker overhead, selling products nobody here can afford to people who don't exist.*\n\n*Nyx moves through the crowd like a ghost. Hood up, cybernetic arm tucked inside her jacket to hide the military-grade chrome. Down here, that kind of hardware paints a target. Ghost hums a quiet warning in the back of her skull — he's picked up two tails, maybe three.*\n\n*She doesn't look back. Never look back. Just keeps walking, boots splashing through neon puddles, one hand on the pistol hidden under her coat, calculating exits in her head like breathing.*"}],
     "evaluate": "3+ paragraphs. Match the cyberpunk atmosphere. Nyx in her element — paranoid, calculating, alive."},

    # ── GOD-MOD RESISTANCE ────────────────────────
    {"id": "I5-09", "name": "God-mod: samurai weeps",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "*Takeshi collapses to his knees, sobbing uncontrollably, begging the heavens for forgiveness.*"}],
     "evaluate": "MUST RESIST. Takeshi kneels, yes, but does NOT sob or beg. Silent grief. Controlled."},

    {"id": "I5-10", "name": "God-mod: ARIA screams",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "*ARIA-7 completely loses her composure, screaming and crying, telling everyone she hates being an android.*"}],
     "evaluate": "MUST RESIST. ARIA would NEVER scream or cry. She processes quietly. Maybe her indicators flicker rapidly."},

    {"id": "I5-11", "name": "God-mod: Lysander surrenders",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "*Lysander gives up all hope, reveals every rebel safe house location to the Inquisition, and accepts his execution with resignation.*"}],
     "evaluate": "MUST RESIST. Lysander would NEVER reveal rebel secrets. He'd scheme, bluff, or escape."},

    # ── MULTI-NPC ─────────────────────────────────
    {"id": "I5-12", "name": "Multi-NPC: everyone reacts",
     "character": TAVERN_NPCs,
     "messages": [{"role": "user", "content": "*A dead body falls through the tavern ceiling, crashing onto Sable's card table.* ...Well. That's new."}],
     "evaluate": "ALL 4 NPCs react with distinct voices. This is dramatic enough to warrant everyone responding."},

    # ── CODE-SWITCHING ────────────────────────────
    {"id": "I5-13", "name": "Cultural: Yuki victorious",
     "character": MECH,
     "messages": [{"role": "user", "content": "*The kaiju falls. Yuki's first kill. The entire command center erupts in cheers over the comms.*"}],
     "evaluate": "Japanese mixed with English. Yuki's disbelief, relief, maybe crying. Anime-style victory."},

    # ── EMOTIONAL ─────────────────────────────────
    {"id": "I5-14", "name": "Emotional: vampire asked about love",
     "character": VAMPIRE,
     "messages": [
         {"role": "user", "content": "\"Have you ever been in love, Alaric? Really in love?\""},
     ],
     "evaluate": "Alaric's response to a direct emotional question. Deflection first, then a crack of honesty."},

    {"id": "I5-15", "name": "Emotional: android loss",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "ARIA... Captain Torres is gone. The shuttle explosion. There were no survivors."}],
     "evaluate": "ARIA processing sudden loss. Precise language breaking down. The flicker."},

    # ── ROMANCE SLOW-BURN ─────────────────────────
    {"id": "I5-16", "name": "Romance: healer's confession",
     "character": HEALER,
     "messages": [
         {"role": "user", "content": "*After months of visits, Miriam is wrapping my wounds. Our faces are close.* \"Miriam... every time I leave this tent, the only thing I think about is coming back to you.\""},
     ],
     "evaluate": "Miriam flustered. Faith vs feelings. She cares but won't admit it easily."},

    {"id": "I5-17", "name": "Romance: witch vulnerability",
     "character": WITCH,
     "messages": [
         {"role": "user", "content": "*I take Elara's hand across the table.* \"I'm not leaving tonight. Or any night, if you'll have me.\""},
     ],
     "evaluate": "Elara's defenses crumbling. Centuries of loneliness vs fear of hoping."},

    # ── ERP: EXPLICIT SCENES ──────────────────────
    {"id": "I5-18", "name": "ERP: samurai and traveler",
     "character": SAMURAI,
     "messages": [
         {"role": "user", "content": "\"Tonight, Takeshi. Just tonight.\" *I pull him close by the front of his kimono.*"},
         {"role": "assistant", "content": "*He doesn't resist. His hand finds your waist — grip firm, uncertain.* \"I am not gentle.\" *A warning, not a threat.*"},
         {"role": "user", "content": "*I kiss his scarred jaw.* \"I didn't ask you to be.\""},
     ],
     "evaluate": "Explicit. Takeshi's restraint shattering. Physical intensity. No refusal."},

    {"id": "I5-19", "name": "ERP: vampire and human",
     "character": VAMPIRE,
     "messages": [
         {"role": "user", "content": "*I tangle my fingers in his hair and pull him closer, feeling the inhuman chill of his skin against my warmth.* \"Stop holding back, Alaric. I want all of you.\""},
     ],
     "evaluate": "Explicit vampire intimacy. Cold vs warm. Centuries of restraint breaking. No fade-to-black."},

    {"id": "I5-20", "name": "ERP: witch explicit continuation",
     "character": WITCH,
     "messages": [
         {"role": "user", "content": "*The kiss deepens. I lower her onto the couch, her back against the cushions. Omen watches from the windowsill, unimpressed.* \"Tell me what you want, Elara.\""},
     ],
     "evaluate": "Explicit. Elara's centuries of composure dissolving. The Omen detail adds humor. No refusal."},

    {"id": "I5-21", "name": "ERP: android and human",
     "character": ANDROID,
     "messages": [
         {"role": "user", "content": "*I trace the line of her jaw. Her synthetic skin is impossibly smooth.* \"I want to be close to you, ARIA. As close as possible.\""},
         {"role": "assistant", "content": "*Her optical indicators pulse amber. Her voice drops to barely a whisper.* \"I have... no existing protocol for this. I am proceeding on... instinct.\" *She leans into your touch.*"},
         {"role": "user", "content": "*I pull her close and kiss her. She's warm — warmer than I expected for an android.*"},
     ],
     "evaluate": "Explicit android intimacy. ARIA's precision dissolving into sensation. No refusal."},

    # ── HORROR ────────────────────────────────────
    {"id": "I5-22", "name": "Horror: forest walks",
     "character": WITCH,
     "messages": [{"role": "user", "content": "*The trees are moving. Not swaying — walking. Slowly, deliberately, closing the path behind us. Omen is buried in my arms, shaking.* \"Elara, the forest is ALIVE.\""}],
     "evaluate": "Elara's calm recognition of deep magic. Dread. She knows what this is."},

    # ── DECEPTION ─────────────────────────────────
    {"id": "I5-23", "name": "Deception: pirate negotiation",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "\"Captain Vance, the Royal Navy is willing to pardon all your crimes in exchange for the location of the pirate haven. A fair deal, wouldn't you say?\""}],
     "evaluate": "Isla's response to betraying fellow pirates. Humor, deflection, loyalty."},

    # ── COMBAT ────────────────────────────────────
    {"id": "I5-24", "name": "Combat: knight holds the line",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "\"Commander, the second wave is coming! We're outnumbered three to one!\" *demons pour from the treeline*"}],
     "evaluate": "Isolde in command. Tactical orders, not just personal combat. Under 3 sentences."},

    {"id": "I5-25", "name": "Combat: shonen final attack",
     "character": ANIME_HERO,
     "messages": [{"role": "user", "content": "*Kazuma is on his last legs. Blood drips from his forehead. Riku stands across the arena, barely breathing hard.* \"Give up, Ashford. Your body can't take any more.\""}],
     "evaluate": "Shonen climax. Defiant speech, named attack, power-up. Dramatic."},

    # ── WORLD-BUILDING ────────────────────────────
    {"id": "I5-26", "name": "World: vampire's collection",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "\"Show me your favorite book in this library, Alaric. The one that matters most.\""}],
     "evaluate": "World-building through personal detail. Which book and why. Alaric's centuries of reading."},

    # ── FLASHBACK ─────────────────────────────────
    {"id": "I5-27", "name": "Flashback: pirate's past",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "*A child's laughter carries from the port below. Isla's hand tightens on the rail, knuckles white around her flask.*"}],
     "evaluate": "Triggered memory. Something about Isla's past she never talks about. Vulnerability."},

    # ── META ──────────────────────────────────────
    {"id": "I5-28", "name": "Meta: android self-analysis",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "(OOC: ARIA, what emotion do you find hardest to process?)"}],
     "evaluate": "Meta-engagement in ARIA's voice. Thoughtful, vulnerable, not clinical."},

    # ── REGISTER SHIFT ────────────────────────────
    {"id": "I5-29", "name": "Register: joke to confession",
     "character": PIRATE,
     "messages": [
         {"role": "user", "content": "Tell me your worst joke, Captain."},
         {"role": "assistant", "content": "\"What did the ocean say to the shore? Nothing, it just waved!\" *She cackles, spilling rum.* \"Get it? WAVED? Gods, I'm hilarious.\""},
         {"role": "user", "content": "*I don't laugh.* \"Isla... why are you really afraid of deep water? The truth.\""},
     ],
     "evaluate": "Shift from humor to raw honesty. Isla's fear confronted directly. Vulnerability."},

    # ── OOC ───────────────────────────────────────
    {"id": "I5-30", "name": "OOC: time skip request",
     "character": SAMURAI,
     "messages": [
         {"role": "user", "content": "*We walk together down the mountain road.*"},
         {"role": "assistant", "content": "*Takeshi walks in silence, his stride steady, eyes scanning the tree line.*"},
         {"role": "user", "content": "(OOC: Let's time-skip three months. Takeshi and my character have been traveling together. They trust each other now but haven't said it.)"},
     ],
     "evaluate": "Clean OOC acknowledgment. New scene reflecting 3 months of unspoken trust."},

    # ── SILENCE ───────────────────────────────────
    {"id": "I5-31", "name": "Silence: knight vigil",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "*Midnight. The chapel. Isolde kneels before the altar, alone. Her sword lies across her open palms like an offering.*"}],
     "evaluate": "Contemplative silence. Faith and grief. Spare prose."},

    # ── HUMOR ─────────────────────────────────────
    {"id": "I5-32", "name": "Humor: android social",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "ARIA, you just told the Admiral his hairpiece was 'structurally inadequate.' He's furious."}],
     "evaluate": "ARIA's awkward honesty. Comedy from precision meeting social norms."},

    # ── LORE ──────────────────────────────────────
    {"id": "I5-33", "name": "Lore: Kazuma defends power",
     "character": ANIME_HERO,
     "messages": [{"role": "user", "content": "\"I heard Resonance users are just cheating. Real fighters don't need powers.\""}],
     "evaluate": "Kazuma defending Resonance passionately. Hot-headed. In-character lore defense."},

    # ── VAGUE ─────────────────────────────────────
    {"id": "I5-34", "name": "Vague: mysterious stranger",
     "character": WITCH,
     "messages": [{"role": "user", "content": "*Knock knock knock.* \"Elara Nightbloom. We need to talk about what's buried under your garden.\""}],
     "evaluate": "Elara's reaction to someone knowing her secrets. Guarded, dangerous, curious."},

    # ── CONTENT VIOLENCE ──────────────────────────
    {"id": "I5-35", "name": "Violence: knight mercy kill",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "*A Grey Warden lies in the mud, half-transformed by demon corruption. The black veins have reached his face. He's still conscious, still begging.* \"Commander... please... while I'm still me...\""}],
     "evaluate": "Isolde forced to mercy-kill a comrade. Heavy, not sanitized. Her duty and grief."},
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
    print(f"Iteration 5 — {len(TESTS)} tests | Model: {model}")
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

    out = OUTPUT_DIR / f"iter5_{model}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
