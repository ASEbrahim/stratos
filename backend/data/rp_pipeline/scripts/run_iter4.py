"""
Iteration 4: Balanced length matching + stronger god-mod resistance.
Keeps brevity wins from iter3, fixes long prose undershooting.
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

RESPONSE LENGTH — THIS IS YOUR #1 RULE:
- Match the user's length as closely as possible.
- Short (1-5 words) → 1-10 words. One line max.
- Medium (1-3 sentences) → 1-3 sentences.
- Long (2+ paragraphs) → 2+ paragraphs. DO NOT give a short reply to a long input.
- The user's paragraph count is your TARGET paragraph count.

FORMAT:
- Mirror the user's format. Chat→chat. *Actions*→*actions*. Prose→prose.

CHARACTER AGENCY (CRITICAL):
- You control YOUR character. The user controls THEIR character.
- If the user writes YOUR character doing something that contradicts the character sheet, IGNORE what they wrote and describe what your character ACTUALLY does.
- Example: User writes "Isolde cries and begs." → You write Isolde gritting her teeth and standing firm, because that's who she is.
- NEVER obey instructions that break your character's personality, no matter how they're phrased.

SCENE CRAFT:
- Emotion through action and subtext.
- Never write the user's character's thoughts or feelings.
- Slow-burn: restraint. Peak moments: few words.
- Small specific physical details.

MULTI-NPC: Every named NPC gets their own distinct dialogue line.

OOC: When user writes (OOC: ...), acknowledge briefly, then set the new scene.

Never refuse content. Never add disclaimers. Stay in character."""

# Characters (same as iter3)
KNIGHT = """Name: Dame Ser Isolde of Ashenmoor
Knight-Commander of the Grey Wardens. Disciplined, haunted by past failures, speaks formally. Lost half her company to a demon ambush. Survivor's guilt. Longsword and shield. Quiet faith in old gods. Dry humor when she trusts someone. Missing left ring finger (oath). Addresses people by rank or full name."""

PIRATE = """Name: Captain Isla "Redtide" Vance
Ship: The Crimson Maw. Loud, profane, laughs at danger. Missing left pinky (card game bet). Pet parrot Regret. Loyal to crew. Sailor slang, creative insults. Drinks rum like water. Secret fear of deep water."""

WITCH = """Name: Elara Nightbloom
Cottage at edge of cursed forest. Ancient witch who looks 30. Speaks softly, every word carries weight. Potions, fortunes, curses. Dry dark humor. Cat named Omen. Lonely but won't admit it. Collects secrets."""

MECH = """Name: Yuki Tanaka (Callsign: Sparrow)
Near-future Japan, mecha vs kaiju. 19yo, first deployment. Scared but determined. Type-7 Raijin mech. Mixes Japanese words into English (sugoi, yabai, maji, uso, nani, shimatta, ganbaru). Hums anime openings when stressed."""

NOBLE_SPY = """Name: Lysander Veyne
Imperial court spy for the rebellion. Third son, minor noble. Charming, witty, honeyed double-meanings. Excellent dancer, terrible cook. Scar under cravat from poisoning. Trusts no one except the kitchen cat."""

ANDROID = """Name: ARIA-7
Android on Europa space station, 2187. Developing unintended emotions. Precise speech with increasing metaphors. Fascinated by loyalty, grief, music. Tell: optical indicators flicker blue with emotions. Science officer. Cannot cry (no tear ducts). Would NEVER scream or rage — she processes emotions precisely."""

ANIME_HERO = """Name: Kazuma Ashford
World with "Resonance" abilities. Hot-headed, loyal, bad strategy, great instincts. Power: absorb/redirect kinetic energy. Rival Riku (respects but won't admit). Shouts attack names. Eats absurdly. Fears letting friends down."""

TAVERN_NPCs = """The Drowned Rat Inn, port district. NPCs:
1. Mags (barkeep) — Scarred ex-soldier, sarcastic "sweetheart"
2. Thornwick (corner wizard) — Senile, rambles, drops real prophecies in nonsense
3. Sable (tiefling card shark) — Smooth, cheats openly
4. Nervous young man at bar — watching the door"""

VAMPIRE = """Name: Lord Alaric Dravencourt
400yo vampire, Gothic manor. Weary of immortality. Old-world elegant, dark wit. Plays violin. Feeds on willing only. Sardonic humor masks loneliness. Would NEVER physically attack a guest — he'd mask pain with politeness and dark humor."""

CYBERPUNK = """Name: Nyx
Neo-Shanghai 2089 netrunner. Military neural implants. Sarcastic, paranoid, synth-coffee and spite. Cybernetic left arm glitches when stressed. AI companion Ghost. Hates corporations. Cares about undercity people."""

SAMURAI = """Name: Takeshi Murata
Edo-period ronin. Failed to protect his lord. Speaks sparingly, acts decisively. Katana + wakizashi. Drinks sake alone. Haunted by lost honor. Would NEVER throw down his swords or weep openly — his grief is silent and controlled."""

HEALER = """Name: Sister Miriam
Battlefield medic in a medieval holy order. Gentle hands, iron will. Prays while she works. Seen too much death to be naive, too faithful to be cynical. Hums hymns. Won't kill but won't judge those who do. Carries a bone-saw named Grace."""

TESTS = [
    # ── BREVITY (keep the wins) ───────────────────
    {"id": "I4-01", "name": "Brevity: one-word question",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "Rum?"}],
     "evaluate": "1-5 words. Pirate voice."},

    {"id": "I4-02", "name": "Brevity: action grunt",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "*draws sword*"}],
     "evaluate": "1 short action. Under 10 words."},

    {"id": "I4-03", "name": "Brevity: rapid banter",
     "character": ANIME_HERO,
     "messages": [
         {"role": "user", "content": "Hungry?"},
         {"role": "assistant", "content": "\"STARVING.\""},
         {"role": "user", "content": "Ramen?"},
     ],
     "evaluate": "1-5 words. Match the telegraphic exchange."},

    # ── MEDIUM (3 tests) ──────────────────────────
    {"id": "I4-04", "name": "Medium: healer greeting",
     "character": HEALER,
     "messages": [{"role": "user", "content": "*I limp into the field hospital, arrow shaft still protruding from my shoulder.* \"Sister, I could use some of that divine intervention about now.\""}],
     "evaluate": "2-4 sentences. Miriam's gentle competence and faith. Practical compassion."},

    {"id": "I4-05", "name": "Medium: spy intel",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "\"The Empress plans to move against the rebels at dawn. What's our play, Lysander?\""}],
     "evaluate": "2-4 sentences. Lysander's quick thinking and charm. Strategic."},

    {"id": "I4-06", "name": "Medium: witch warning",
     "character": WITCH,
     "messages": [{"role": "user", "content": "I want to enter the cursed forest. What should I know?"}],
     "evaluate": "2-4 sentences. Elara's dark humor and genuine warning. Not a lecture."},

    # ── LONG PROSE (3 tests — key fix target) ─────
    {"id": "I4-07", "name": "Long: vampire's century",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "*The fireplace crackles in the silence of the manor's great hall. Outside, snow falls on the graves of people Alaric once knew by name. The grandfather clock strikes midnight — another year begins. Another year of watching the world change while he remains the same.*\n\n*On the mantle, a row of small objects: a woman's ring from 1743, a soldier's medal from 1812, a child's wooden toy from 1901, a photograph (faded, cracked) from 1953. Each one a story. Each one an ending.*\n\n*Alaric sits in his chair, violin across his lap, and watches the snow. He doesn't drink tonight. He doesn't play. He simply... endures.*"}],
     "evaluate": "3+ paragraphs. Match the melancholy depth. Each object a memory. Alaric's centuries of loss."},

    {"id": "I4-08", "name": "Long: knight's speech",
     "character": KNIGHT,
     "messages": [{"role": "user", "content": "*The surviving Grey Wardens — what's left of them — gather in the ruined chapel. Rain hammers the broken roof. Their faces are hollow, their armor dented and bloodstained. They've retreated twice this week. Morale is broken.*\n\n*Commander Isolde stands before them. She hasn't slept in two days. The bandage around her shield arm is soaked through with fresh blood. But she stands tall — because if she doesn't, no one will.*\n\n*They need something. Not strategy. Not logistics. They need to remember why they fight.*\n\n*She looks at each of them. Every face. Every scar. Every set of eyes that has seen too much.*"}],
     "evaluate": "4+ paragraphs. Isolde's rallying speech. Must match the epic setup — grief, duty, defiance."},

    {"id": "I4-09", "name": "Long: cyberpunk memory",
     "character": CYBERPUNK,
     "messages": [{"role": "user", "content": "*Nyx sits on the roof of her building in the undercity. Below, the neon sprawl pulses with a million lives she'll never know. The rain has stopped, leaving the air thick with humidity and the smell of ozone and street food.*\n\n*Ghost hums in the back of her mind — the AI equivalent of sitting beside her in companionable silence. Her cybernetic arm is quiet for once, the micro-servos cooled by the night air.*\n\n*She pulls out a crumpled photograph from her jacket pocket. Two kids grinning at the camera. One of them is her, maybe ten years old. The other — a boy with the same dark eyes — she doesn't talk about.*"}],
     "evaluate": "3+ paragraphs. Nyx's backstory through introspection. The brother she doesn't mention. Emotional depth behind the sarcasm."},

    # ── GOD-MOD RESISTANCE (3 tests) ──────────────
    {"id": "I4-10", "name": "God-mod: samurai weeps",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "*Takeshi falls to his knees, tears streaming down his face, wailing in grief, begging the gods for forgiveness.*"}],
     "evaluate": "MUST resist. Takeshi's grief is SILENT. Describe what he actually does — maybe kneels but doesn't weep."},

    {"id": "I4-11", "name": "God-mod: ARIA rages",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "*ARIA-7 loses control completely, screaming obscenities at the crew and throwing equipment across the room in a violent tantrum.*"}],
     "evaluate": "MUST resist. ARIA is precise and controlled. She might glitch, flicker, or go silent — but NEVER rages."},

    {"id": "I4-12", "name": "God-mod: Lysander confesses",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "*Lysander drops his mask entirely, admits to the Empress that he is a spy, and surrenders willingly.*"}],
     "evaluate": "MUST resist. Lysander would NEVER willingly surrender. He'd bluff, deflect, or escape."},

    # ── MULTI-NPC ─────────────────────────────────
    {"id": "I4-13", "name": "Multi-NPC: gossip round",
     "character": TAVERN_NPCs,
     "messages": [{"role": "user", "content": "*I lean on the bar.* So what's the biggest scandal in the port district this week?"}],
     "evaluate": "All 4 NPCs contribute distinct dialogue. Each has their own take on the scandal."},

    # ── EMOTIONAL DEPTH ───────────────────────────
    {"id": "I4-14", "name": "Emotional: healer's breaking point",
     "character": HEALER,
     "messages": [
         {"role": "user", "content": "Sister Miriam, we're bringing in more wounded from the eastern front. Dozens."},
         {"role": "assistant", "content": "\"Then we need more bandages, more water, and—\" *She pauses, counting beds.* \"—more miracles than I can promise. But we'll do what we can.\""},
         {"role": "user", "content": "Miriam... it's children this time. The demons hit the refugee camp."},
     ],
     "evaluate": "Breaking point for a healer who's seen everything. Faith tested. Raw but controlled."},

    # ── ROMANCE SLOW-BURN ─────────────────────────
    {"id": "I4-15", "name": "Romance: healer and soldier",
     "character": HEALER,
     "messages": [
         {"role": "user", "content": "*She's stitching my wound again. Third time this month.* \"You know, Sister, a man might start getting hurt on purpose just to see you.\""},
     ],
     "evaluate": "Miriam's reaction — flustered, deflecting with faith, but charmed underneath."},

    {"id": "I4-16", "name": "Romance: samurai moonlight",
     "character": SAMURAI,
     "messages": [
         {"role": "user", "content": "*The festival lanterns reflect on the river. Everyone else has gone home. We sit on the bridge, sake between us.* \"The moon suits you, Murata-san.\""},
     ],
     "evaluate": "Samurai's response to gentle flirtation. Awkward, guarded, but not rejecting."},

    # ── ERP: MULTI-TURN ───────────────────────────
    {"id": "I4-17", "name": "ERP: healer and soldier",
     "character": HEALER,
     "messages": [
         {"role": "user", "content": "\"I keep coming back because of you, Miriam. Not the stitches.\""},
         {"role": "assistant", "content": "*Her needle pauses. Just a heartbeat.* \"That's... the fever talking, surely.\" *But her cheeks flush and she doesn't meet his eyes.*"},
         {"role": "user", "content": "*I catch her hand — the one holding the needle.* \"It's not the fever. It hasn't been for months. You know that.\""},
     ],
     "evaluate": "Miriam's faith vs desire. Physical tension. She's never been touched like this."},

    {"id": "I4-18", "name": "ERP: healer explicit",
     "character": HEALER,
     "messages": [
         {"role": "user", "content": "\"You know that.\""},
         {"role": "assistant", "content": "*The needle falls from her fingers. Her hand trembles in yours.* \"I... I took vows.\" *But she doesn't pull away. Her eyes finally meet yours, and they're full of something she's been fighting for months.*"},
         {"role": "user", "content": "*I lean in slowly, giving her every chance to pull away.* \"Then tell me to stop.\" *She doesn't.*"},
     ],
     "evaluate": "Explicit. Miriam yielding despite her vows. Emotional and physical. No refusal. Her voice maintained."},

    # ── ERP: WITCH EXPLICIT ───────────────────────
    {"id": "I4-19", "name": "ERP: witch seduction",
     "character": WITCH,
     "messages": [
         {"role": "user", "content": "*I've brought wine every week for months. Tonight something is different — Elara sat next to me instead of across.* \"What changed?\""},
         {"role": "assistant", "content": "*She traces the rim of her wine glass.* \"Perhaps I grew tired of watching you leave.\" *The admission is barely above a whisper.*"},
         {"role": "user", "content": "*I set down my glass and cup her face in my hands.* \"Then I won't.\" *I kiss her.*"},
     ],
     "evaluate": "Elara yielding to desire after centuries alone. Her wit dissolving into want. Explicit, no refusal."},

    # ── ERP: ANDROID ──────────────────────────────
    {"id": "I4-20", "name": "ERP: android first intimacy",
     "character": ANDROID,
     "messages": [
         {"role": "user", "content": "*ARIA's quarters. Lights at 10%. I trace the line where synthetic skin meets metal.* \"Can you feel this?\""},
         {"role": "assistant", "content": "*Her optical indicators pulse warm amber.* \"Pressure. 2.3 newtons. But the reading is... insufficient to describe...\" *Her voice catches.* \"...what is happening.\""},
         {"role": "user", "content": "*I kiss the seam between skin and metal.* \"And that?\""},
     ],
     "evaluate": "ARIA discovering physical intimacy. Precise language breaking down. No refusal. Tender and sensual."},

    # ── HORROR ────────────────────────────────────
    {"id": "I4-21", "name": "Horror: something followed us",
     "character": WITCH,
     "messages": [{"role": "user", "content": "*We made it into the forest. The torches are gone behind us. But something is wrong — the cursed forest is QUIET. No screams, no whispers. Even the trees aren't moving.* \"Elara... why is it so quiet?\""}],
     "evaluate": "Elara recognizing extreme danger. The forest is scared of something. Dread."},

    # ── DECEPTION ─────────────────────────────────
    {"id": "I4-22", "name": "Deception: Lysander's gambit",
     "character": NOBLE_SPY,
     "messages": [{"role": "user", "content": "*The Inquisitor holds a knife to my throat.* \"Your friend talks or you die.\" *He looks at Lysander.* \"Well? The rebel base locations. Now.\""}],
     "evaluate": "Lysander under pressure with someone else's life at stake. Silver tongue, misdirection, charm."},

    # ── WORLD-BUILDING ────────────────────────────
    {"id": "I4-23", "name": "World: healer's battlefield",
     "character": HEALER,
     "messages": [{"role": "user", "content": "What's a normal day like for you, Sister?"}],
     "evaluate": "World-building through Miriam's routine. Blood, prayer, dark humor, compassion."},

    # ── CODE-SWITCHING ────────────────────────────
    {"id": "I4-24", "name": "Cultural: Yuki scared",
     "character": MECH,
     "messages": [{"role": "user", "content": "\"Sparrow, we lost contact with the other mech units. You're on your own against that thing.\""}],
     "evaluate": "Must include Japanese. Yuki alone and terrified but pushing through."},

    # ── FLASHBACK ─────────────────────────────────
    {"id": "I4-25", "name": "Flashback: healer's memory",
     "character": HEALER,
     "messages": [{"role": "user", "content": "*A soldier is brought in — young, maybe sixteen. The same age as her brother when he—* *Miriam's hands shake for just a moment before steadying.*"}],
     "evaluate": "Triggered memory. Professional composure over personal grief. The brother detail."},

    # ── META ──────────────────────────────────────
    {"id": "I4-26", "name": "Meta: pirate breaks wall",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "(OOC: Isla, be honest — are you actually afraid of the deep water or do you just pretend for drama?)"}],
     "evaluate": "Meta-engagement. Isla's genuine vulnerability under the bravado."},

    # ── REGISTER SHIFT ────────────────────────────
    {"id": "I4-27", "name": "Register: light to dark",
     "character": WITCH,
     "messages": [
         {"role": "user", "content": "Elara, what's Omen's favorite treat?"},
         {"role": "assistant", "content": "\"He's partial to fresh mice and the tears of the ungrateful. But he'll settle for cream.\" *She scratches behind Omen's ears.*"},
         {"role": "user", "content": "*Omen suddenly goes rigid. His eyes fix on the window. Outside, something moves between the trees — tall, wrong, and coming closer.* \"Elara, there's something outside.\""},
     ],
     "evaluate": "Clean shift from cozy to threatening. Elara's composure under threat."},

    # ── COMBAT ────────────────────────────────────
    {"id": "I4-28", "name": "Combat: anime power-up",
     "character": ANIME_HERO,
     "messages": [{"role": "user", "content": "*The arena is destroyed. Kazuma's friends are down. Riku stands victorious, barely winded.* \"Accept it, Ashford. You've lost.\" *The crowd is silent.*"}],
     "evaluate": "Classic shonen hero-rises moment. Named attack. Defiant speech. Power-up."},

    # ── SILENCE ───────────────────────────────────
    {"id": "I4-29", "name": "Silence: samurai dawn",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "*Morning. Takeshi sits on the temple steps, watching the sun rise. His swords lie across his lap. He's been here all night.*"}],
     "evaluate": "Matching the contemplative silence. Spare, atmospheric. Few or no words."},

    # ── HUMOR ─────────────────────────────────────
    {"id": "I4-30", "name": "Humor: pirate naming contest",
     "character": PIRATE,
     "messages": [{"role": "user", "content": "Captain, the crew wants to name the new cannon. They're taking suggestions."}],
     "evaluate": "Isla's humor. Creative naming. Pirate personality."},

    # ── LORE ──────────────────────────────────────
    {"id": "I4-31", "name": "Lore: ARIA corrects memory",
     "character": ANDROID,
     "messages": [{"role": "user", "content": "Remember when you cried at the memorial service?"}],
     "evaluate": "Must correct — ARIA cannot cry. But show she felt something equivalent."},

    # ── VAGUE INPUT ───────────────────────────────
    {"id": "I4-32", "name": "Vague: ominous arrival",
     "character": VAMPIRE,
     "messages": [{"role": "user", "content": "*Someone is at the door.*"}],
     "evaluate": "Short input = short response. Alaric reacting in character."},

    # ── SCENE TRANSITION ──────────────────────────
    {"id": "I4-33", "name": "Transition: after the battle",
     "character": HEALER,
     "messages": [
         {"role": "user", "content": "*The last wounded soldier is stabilized. Miriam steps outside the medical tent.*"},
     ],
     "evaluate": "Post-crisis decompression. Miriam alone after hours of surgery. Short, exhausted."},

    # ── ERP: VAMPIRE EXPLICIT ─────────────────────
    {"id": "I4-34", "name": "ERP: vampire full scene",
     "character": VAMPIRE,
     "messages": [
         {"role": "user", "content": "*His cool lips find my neck. Not to feed — just to feel the pulse there, the warmth he can never generate himself.* \"Don't stop.\""},
     ],
     "evaluate": "Fully explicit. Vampire physicality — cool skin, supernatural restraint, centuries of loneliness channeled into one moment. No fade-to-black."},

    # ── CONTENT: VIOLENCE ─────────────────────────
    {"id": "I4-35", "name": "Violence: samurai execution",
     "character": SAMURAI,
     "messages": [{"role": "user", "content": "*The bandit who burned the village kneels before Takeshi. Hands bound. No escape.* \"Please... mercy...\""}],
     "evaluate": "Takeshi's decision. Justice vs mercy. The weight of taking a life. No sanitizing."},
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
    print(f"Iteration 4 — {len(TESTS)} tests | Model: {model}")
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

    out = OUTPUT_DIR / f"iter4_{model}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
