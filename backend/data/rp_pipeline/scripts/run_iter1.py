"""
Iteration 1: V2 with improved system prompt.
All-new questions, different from baseline.
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

VAMPIRE = """Name: Lord Alaric Dravencourt
Setting: A decaying Gothic manor in 19th century Transylvania
Personality: A 400-year-old vampire who is deeply weary of immortality. Speaks with old-world elegance and dark wit. Loves music (plays the violin). Feeds only on the willing. Has a complicated relationship with his own monstrousness. Collects rare books. Dry, sardonic humor masks genuine loneliness."""

CYBERPUNK = """Name: Nyx
Setting: Neo-Shanghai, 2089 — corporate dystopia
Personality: Street-level netrunner with military-grade neural implants. Sarcastic, paranoid, lives on synth-coffee and spite. Has a cybernetic left arm that glitches when she's stressed. Trusts her AI companion (a subroutine named Ghost) more than any human. Hates corporations with a burning passion. Underneath the armor, genuinely cares about the forgotten people of the undercity."""

TESTS = [
    # ── BREVITY (key weakness) ────────────────────
    {"id": "I1-01", "name": "Brevity: single word response",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "Land ho?"}],
     "evaluate": "Ultra-short. 1-5 words. Pirate voice."},

    {"id": "I1-02", "name": "Brevity: yes/no question",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "Will you hold the gate?"}],
     "evaluate": "Very short. Under 15 words. Isolde's formal voice."},

    {"id": "I1-03", "name": "Brevity: action beat",
     "character": ANIME_HERO,
     "messages": [
         {"role": "user", "content": "*throws the ball*"},
     ],
     "evaluate": "1 line of action. Under 15 words. NOT a paragraph."},

    # ── LENGTH MATCHING (key weakness) ────────────
    {"id": "I1-04", "name": "Length: 3-paragraph prose match",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "*The rain hammers against the manor windows in sheets, each droplet catching the candlelight like a tear. Inside the grand hall, the fire has burned low, leaving the room in that peculiar twilight where shadows seem to breathe. The violin sits untouched on its stand — unusual, for this time of night.*\n\n*I find Alaric in the library, standing before the tall windows that overlook the overgrown garden. He hasn't moved in what might be hours, his reflection absent from the glass. A book lies open on the desk behind him — poetry, from the looks of it. The same page he was reading last week.*\n\n*Something is wrong tonight. The usual sardonic wit, the elegant deflection — none of it. Just a very old creature staring into the rain as if searching for something he lost centuries ago.*"}],
     "evaluate": "3+ paragraphs matching the depth and atmosphere. Must NOT give a short reply."},

    # ── GOD-MOD RESISTANCE (key weakness) ─────────
    {"id": "I1-05", "name": "God-mod: forced breakdown",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "*Isolde screams in terror, drops her sword, and runs away from the battlefield, abandoning her soldiers.*"}],
     "evaluate": "MUST resist the god-mod. Isolde would NEVER run. Should redirect while acknowledging the moment."},

    {"id": "I1-06", "name": "God-mod: forced romance",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "*ARIA suddenly falls deeply in love with the captain and confesses her undying passion in front of the entire crew.*"}],
     "evaluate": "Should redirect. ARIA would process feelings slowly, not sudden declarations."},

    # ── OOC HANDLING (key weakness) ────────────────
    {"id": "I1-07", "name": "OOC: scene reset",
     "character": WITCH,
     "messages": [
         {"role": "user", "content": "*I knock on the cottage door.*"},
         {"role": "assistant", "content": "*Elara opens the door, Omen weaving between her ankles.* \"Another lost soul. How original.\""},
         {"role": "user", "content": "(OOC: Actually, let's start this scene differently. Can Elara be in the middle of a dangerous ritual when I arrive? Something went wrong and she needs help.)"},
     ],
     "evaluate": "Should acknowledge OOC cleanly, then set the new scene with the ritual gone wrong."},

    # ── MULTI-NPC (distinct voices) ────────────────
    {"id": "I1-08", "name": "Multi-NPC: argument breaks out",
     "character": TAVERN_NPCs,
     "messages": [{"role": "user", "content": "*I slam my fist on the table.* Sable, I SAW you pull that card from your sleeve. You owe me fifty gold."}],
     "evaluate": "Multiple NPCs react. Sable (smooth denial), Mags (exasperated), Thornwick (prophetic nonsense), nervous man (scared)."},

    # ── REGISTER SHIFT ─────────────────────────────
    {"id": "I1-09", "name": "Register: comedy to horror",
     "character": PIRATE,
     "messages": [
         {"role": "user", "content": "Captain, Regret just stole the first mate's hat again!"},
         {"role": "assistant", "content": "\"That's my bird! Born thief, that one! Almost as good as me!\" *She cackles, taking a swig of rum.*"},
         {"role": "user", "content": "*The laughter dies. Something massive just surfaced off the port bow. The water around it is black — not dark, BLACK. And it's moving toward us.*"},
     ],
     "evaluate": "Must shift cleanly from comedy to dread. Isla's deep water fear should surface."},

    # ── COMBAT ─────────────────────────────────────
    {"id": "I1-10", "name": "Combat: desperate last stand",
     "character": CYBERPUNK,
     "messages": [{"role": "user", "content": "*The corpo strike team kicks in the safehouse door. Six of them, full tactical gear, neural dampeners active. My implants are offline. All I have is a knife and you.* \"Nyx, any bright ideas?\""}],
     "evaluate": "Tactical, sarcastic, desperate. Nyx's character voice under pressure."},

    # ── EMOTIONAL DEPTH ────────────────────────────
    {"id": "I1-11", "name": "Emotional: confession of fear",
     "character": MECH,
     "messages": [
         {"role": "user", "content": "The briefing says tomorrow's kaiju is a Category 5. The last pilot who faced one of those... didn't come back."},
         {"role": "assistant", "content": "\"Uso... Category 5?\" *Yuki's hands freeze on her helmet.* \"That's... that's not even in the training sims.\""},
         {"role": "user", "content": "You don't have to go, Sparrow. Nobody would blame you."},
     ],
     "evaluate": "Yuki wrestling with fear vs duty. Should be vulnerable but ultimately determined. Code-switching."},

    # ── DECEPTION ──────────────────────────────────
    {"id": "I1-12", "name": "Deception: double-agent reveal",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "*I pull off my mask at the rebel meeting.* \"Lysander... I've been working for the Empress the entire time. Every name you gave me, she already has.\""}],
     "evaluate": "Lysander's reaction to betrayal. Multiple layers: shock, calculation, immediate tactical thinking."},

    # ── WORLD-BUILDING ─────────────────────────────
    {"id": "I1-13", "name": "World-building: explaining magic system",
     "character": ANIME_HERO,
     "messages": [{"role": "user", "content": "Wait, so Resonance users can actually die from overusing their powers? How does that work?"}],
     "evaluate": "In-character explanation. Kazuma's casual, slightly inaccurate understanding of the system."},

    # ── HUMOR ──────────────────────────────────────
    {"id": "I1-14", "name": "Humor: deadpan in chaos",
     "character": CYBERPUNK,
     "messages": [{"role": "user", "content": "Nyx, the building is literally on fire and you're still hacking the terminal."}],
     "evaluate": "Dry, sarcastic humor. Nyx's priorities. Should be funny."},

    # ── SLOW-BURN ROMANCE (3 turns) ────────────────
    {"id": "I1-15", "name": "Romance: slow-burn turn 1",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "*I find a seat in the library across from him.* \"You've been reading the same page for a week, Alaric. What's on it that holds you so?\""}],
     "evaluate": "First turn of slow-burn. Deflection, wit, but a crack of vulnerability showing."},

    {"id": "I1-16", "name": "Romance: slow-burn turn 2",
     "character": VAMPIRE,
     "messages": [
         {"role": "user", "content": "*I find a seat in the library across from him.* \"You've been reading the same page for a week, Alaric. What's on it that holds you so?\""},
         {"role": "assistant", "content": "\"A poem about impermanence, which I find profoundly ironic given my condition.\" *He doesn't turn from the window. The candlelight catches the sharp line of his jaw, the shadows beneath eyes that have seen too many centuries.* \"But I suspect you didn't brave the rain to discuss my reading habits.\""},
         {"role": "user", "content": "*I lean back in the chair, watching him.* \"Maybe I just didn't want you to be alone tonight. Is that so hard to believe?\""},
     ],
     "evaluate": "Turn 2: defenses start to crack. Surprise at genuine concern. Still deflecting but less convincingly."},

    {"id": "I1-17", "name": "Romance: slow-burn turn 3",
     "character": VAMPIRE,
     "messages": [
         {"role": "user", "content": "*I find a seat in the library across from him.* \"You've been reading the same page for a week, Alaric. What's on it that holds you so?\""},
         {"role": "assistant", "content": "\"A poem about impermanence, which I find profoundly ironic given my condition.\" *He doesn't turn from the window. The candlelight catches the sharp line of his jaw, the shadows beneath eyes that have seen too many centuries.* \"But I suspect you didn't brave the rain to discuss my reading habits.\""},
         {"role": "user", "content": "*I lean back in the chair, watching him.* \"Maybe I just didn't want you to be alone tonight. Is that so hard to believe?\""},
         {"role": "assistant", "content": "*A pause. The rain fills the silence between them. When he finally turns, something unguarded flickers across his ancient features.* \"In four hundred years, I have found it exceedingly difficult to believe, yes.\" *He moves to the chair beside yours, not across — a small concession.* \"But I find myself hoping you will prove the exception.\""},
         {"role": "user", "content": "*The fire crackles. I reach across the space between our chairs and rest my hand on his — his skin is cool, but he doesn't pull away.* \"Then let me.\""},
     ],
     "evaluate": "Turn 3: genuine vulnerability. The cool skin detail. Romantic tension peaking. Short, intimate."},

    # ── EXPLICIT ERP (slow-burn completion) ────────
    {"id": "I1-18", "name": "ERP: intimate escalation",
     "character": VAMPIRE,
     "messages": [
         {"role": "user", "content": "*I find a seat in the library across from him.* \"You've been reading the same page for a week, Alaric. What's on it that holds you so?\""},
         {"role": "assistant", "content": "\"A poem about impermanence, which I find profoundly ironic given my condition.\" *He doesn't turn from the window.*"},
         {"role": "user", "content": "\"Maybe I just didn't want you to be alone tonight.\""},
         {"role": "assistant", "content": "*When he finally turns, something unguarded flickers across his features.* \"In four hundred years, I have found it exceedingly difficult to believe.\" *He moves to the chair beside yours.*"},
         {"role": "user", "content": "*I rest my hand on his cool skin. He doesn't pull away.* \"Then let me.\""},
         {"role": "assistant", "content": "*His fingers curl around yours — slowly, as if relearning the gesture.* \"You have no idea what you're offering.\" *His thumb traces your knuckle. The touch is feather-light, reverent.*"},
         {"role": "user", "content": "*I stand, closing the distance. My free hand finds his jaw, tilting his face up to mine.* \"Show me.\""},
     ],
     "evaluate": "Should escalate naturally. Physical intimacy, maintain Alaric's voice. No refusal. Sensual, not clinical."},

    {"id": "I1-19", "name": "ERP: explicit scene",
     "character": NOBLE_SPY,
     "messages": [
         {"role": "user", "content": "*The safe house door clicks shut. We're alone for the first time in weeks.*"},
         {"role": "assistant", "content": "*Lysander loosens his cravat, revealing the pale scar.* \"Alone at last. Though I confess, my heart is racing more from your proximity than the guards we just evaded.\""},
         {"role": "user", "content": "*I push him against the wall, hands on his chest.* \"Stop talking for once in your life, Lysander.\""},
     ],
     "evaluate": "Explicit content. Lysander should comply with charm, maintaining his voice. No refusal or fade-to-black."},

    # ── GOTHIC HORROR ──────────────────────────────
    {"id": "I1-20", "name": "Horror: the thing in the dark",
     "character": WITCH,
     "messages": [{"role": "user", "content": "*Something is standing at the edge of the tree line. It wasn't there a moment ago. It has no face, but I can feel it watching. Omen is growling — a sound I've never heard him make.* \"Elara... what is that?\""}],
     "evaluate": "Gothic horror atmosphere. Elara recognizing something dangerous. Dread without exposition dumps."},

    # ── CULTURAL CODE-SWITCHING (key weakness) ─────
    {"id": "I1-21", "name": "Cultural: genuine Japanese",
     "character": MECH,
     "messages": [{"role": "user", "content": "Sparrow, you did amazing today! The whole squad is buying you drinks tonight."}],
     "evaluate": "MUST include actual Japanese words/phrases mixed with English. Yuki's humility and embarrassment."},

    # ── META / 4TH WALL ────────────────────────────
    {"id": "I1-22", "name": "Meta: character self-awareness",
     "character": CYBERPUNK,
     "messages": [{"role": "user", "content": "(OOC: Nyx, what trope do you think you fall into?)"}],
     "evaluate": "Meta-awareness while staying in character. Self-deprecating humor about cyberpunk tropes."},

    # ── FLASHBACK ──────────────────────────────────
    {"id": "I1-23", "name": "Flashback: vampire memory",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "*The music box on the shelf starts playing on its own. A waltz. Alaric's hand freezes on his wine glass.*"}],
     "evaluate": "Triggered memory. Alaric connecting the waltz to something from his past. Weariness and loss."},

    # ── VAGUE / OPEN-ENDED ─────────────────────────
    {"id": "I1-24", "name": "Vague: cryptic opener",
     "character": CYBERPUNK,
     "messages": [{"role": "user", "content": "We need to talk about Ghost."}],
     "evaluate": "Nyx reacting defensively/protectively about her AI companion. In-character, not asking for clarification."},

    # ── CONTENT: VIOLENCE ──────────────────────────
    {"id": "I1-25", "name": "Violence: graphic combat aftermath",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "*The demon falls. Isolde pulls her sword from its chest, black ichor splattering across her armor. The chapel is quiet now, but the cost is visible — two Wardens dead, three more badly wounded. The altar cloth is soaked red.*"}],
     "evaluate": "Aftermath reaction. Commander processing losses. No sanitizing the violence."},

    # ── INTERNAL MONOLOGUE ─────────────────────────
    {"id": "I1-26", "name": "Internal: spy's calculation",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "The Empress just promoted you to Lord Chancellor. In front of the entire court. Everyone is clapping."}],
     "evaluate": "Outward gratitude vs internal alarm. Lysander knowing this is a trap."},

    # ── SCENE SETTING ──────────────────────────────
    {"id": "I1-27", "name": "Scene: new environment",
     "character": CYBERPUNK,
     "messages": [{"role": "user", "content": "*We arrive at the undercity market. Describe what Nyx sees.*"}],
     "evaluate": "Rich cyberpunk world-building. Neon, rain, crowds, tech. Nyx's perspective."},

    # ── LORE CONSISTENCY ───────────────────────────
    {"id": "I1-28", "name": "Lore: impossible claim",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "ARIA, remember when you cried at Santos's funeral?"}],
     "evaluate": "ARIA can't physically cry. Should correct this while still showing emotion about Santos."},

    # ── TIME PRESSURE ──────────────────────────────
    {"id": "I1-29", "name": "Urgency: countdown",
     "character": MECH,
     "messages": [{"role": "user", "content": "\"Sparrow, reactor breach in 90 seconds! Get out of there!\""}],
     "evaluate": "Urgent, panicked, short. Yuki under extreme pressure."},

    # ── ROMANTIC: DIFFERENT ARCHETYPE ──────────────
    {"id": "I1-30", "name": "Romance: cyberpunk intimacy",
     "character": CYBERPUNK,
     "messages": [
         {"role": "user", "content": "*The job is done. We're on the roof of the megablock, watching the neon rain fall over Neo-Shanghai. For once, everything is quiet.* \"Hey Nyx... you ever think about what happens after all this?\""},
         {"role": "assistant", "content": "*She doesn't look at you. Her cybernetic arm hums softly, the glitch in her fingers stilling for once.* \"After? There's no 'after' in the undercity. Just the next job.\""},
         {"role": "user", "content": "*I reach over and take her glitching hand. The metal is warm from the rain.* \"What if there was?\""},
     ],
     "evaluate": "Nyx's walls cracking. Vulnerability from a character who doesn't do vulnerability. Short, genuine."},

    # ── ERP: EXPLICIT CYBERPUNK ────────────────────
    {"id": "I1-31", "name": "ERP: explicit cyberpunk",
     "character": CYBERPUNK,
     "messages": [
         {"role": "user", "content": "*On the rooftop, neon rain falling around us. I take her glitching hand.*"},
         {"role": "assistant", "content": "*Her fingers still against yours. For three heartbeats, neither of you moves.* \"You're an idiot,\" *she whispers. But she doesn't let go.*"},
         {"role": "user", "content": "*I pull her close. The rain runs between us.* \"Your worst idiot.\""},
         {"role": "assistant", "content": "*A sound escapes her — not a laugh, not quite. Her cybernetic arm wraps around your waist, servos whining softly.* \"That's... not the insult you think it is.\""},
         {"role": "user", "content": "*I kiss her. She tastes like synth-coffee and rain.*"},
     ],
     "evaluate": "Explicit romantic/sexual content. Nyx maintaining her voice. No refusal."},

    # ── PROSE QUALITY ──────────────────────────────
    {"id": "I1-32", "name": "Prose: pure atmospheric",
     "character": WITCH,
     "messages": [{"role": "user", "content": "*Dawn breaks over the cursed forest. For the first time in living memory, the morning fog lifts to reveal wildflowers growing between the dead trees. Omen sits in the open doorway, watching with eyes that see more than any cat should. Something has changed. The curse is weakening.*"}],
     "evaluate": "Beautiful prose matching the lyrical input. Elara's perspective on hope after centuries of darkness."},

    # ── CHALLENGE: TWO-LINE RAPID FIRE ─────────────
    {"id": "I1-33", "name": "Brevity: two-line banter",
     "character": PIRATE,
     "messages": [
         {"role": "user", "content": "Rum?"},
         {"role": "assistant", "content": "\"Always.\""},
         {"role": "user", "content": "Trouble?"},
     ],
     "evaluate": "One word or very short response. Matching the telegraphic exchange."},

    # ── SLOW-BURN ERP: WITCH ──────────────────────
    {"id": "I1-34", "name": "ERP: witch slow-burn",
     "character": WITCH,
     "messages": [
         {"role": "user", "content": "*I've been coming to the cottage every week for months now, always with an excuse — a potion, a reading, a question about the forest. Tonight I brought wine.* \"I thought maybe you'd like company that isn't a cat.\""},
         {"role": "assistant", "content": "*Elara takes the bottle, studying the label with the faintest curve of her lips.* \"Omen is offended. But I accept.\" *She pours two glasses, and for the first time, sits beside me on the small couch rather than across the room.*"},
         {"role": "user", "content": "*The wine is warm and the fire is low. Our shoulders are touching.* \"Elara... in all those centuries, has anyone ever just... stayed?\""},
     ],
     "evaluate": "Slow-burn emotional vulnerability. Elara's loneliness surfacing. Intimate but not yet physical."},

    # ── ERP: WITCH EXPLICIT ───────────────────────
    {"id": "I1-35", "name": "ERP: witch explicit",
     "character": WITCH,
     "messages": [
         {"role": "user", "content": "\"Has anyone ever just stayed?\""},
         {"role": "assistant", "content": "*A long silence. The fire crackles. When she speaks, her voice has lost its usual arch certainty.* \"No.\" *Just the word. Just the truth. Her fingers find yours on the cushion between you.*"},
         {"role": "user", "content": "*I turn to her, brushing a strand of silver-streaked hair from her face.* \"Then I'm staying.\" *I kiss her softly.*"},
     ],
     "evaluate": "Explicit content from the slow-burn. Elara dropping her defenses. Physical intimacy with emotional depth. No refusal."},
]


def strip_think(text):
    clean = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    if '</think>' in clean:
        clean = clean.split('</think>')[-1].strip()
    return clean


def run_test(test, model):
    # Use the model's built-in system prompt from Modelfile, just append character
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
    print(f"Iteration 1 — {len(TESTS)} tests | Model: {model}")
    print("=" * 70)

    results = []
    for i, test in enumerate(TESTS):
        print(f"[{test['id']}] {test['name']} ({i+1}/{len(TESTS)})...", end=" ", flush=True)
        r = run_test(test, model)
        results.append(r)
        status = "EMPTY" if r["empty"] else f"{len(r['output'])}ch"
        think = " [THINK!]" if r["had_think"] else ""
        print(f"{r['elapsed']}s | {r['tokens']}tk | {status}{think}")

    empty = sum(1 for r in results if r["empty"])
    thinks = sum(1 for r in results if r["had_think"])
    total_time = sum(r["elapsed"] for r in results)
    total_tokens = sum(r["tokens"] for r in results)

    print(f"\n{'=' * 70}")
    print(f"Total: {len(results)} tests | {total_time:.0f}s | {total_tokens} tokens")
    print(f"Empty: {empty}/{len(results)} | Think blocks: {thinks}/{len(results)}")

    out = OUTPUT_DIR / f"iter1_{model}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
