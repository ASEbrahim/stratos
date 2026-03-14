"""
RP Model Baseline Testing — Phase 0

Runs the 10 baseline tests from the pipeline document against a specified model.
Saves full outputs to baseline_tests/ for human scoring.

Usage:
    python3 run_baseline.py [model_name]
    python3 run_baseline.py stratos-rp-baseline  # default
"""

import json
import sys
import time
from pathlib import Path
import requests

MODEL = sys.argv[1] if len(sys.argv) > 1 else "stratos-rp-baseline"
OLLAMA_URL = "http://localhost:11434/api/chat"
OUTPUT_DIR = Path(__file__).parent.parent / "baseline_tests"
OUTPUT_DIR.mkdir(exist_ok=True)

# Character cards for testing
FANTASY_WARRIOR = """Name: Kael Stormborn
Race: Human
Class: Paladin of the Storm God
Personality: Stoic but loyal, dry humor, speaks in short sentences. Deeply protective of the weak. Has a scar across his left eye from a battle with a lich. Distrusts magic users but respects their power. Carries a greatsword named Tempest that crackles with lightning."""

ANIME_VILLAIN = """Name: Ainz Ooal Gown (from Overlord)
Title: Sorcerer King of the Great Tomb of Nazarick
Personality: Formerly a salaryman named Satoru Suzuki, now trapped in the body of an undead overlord. Internally anxious and unsure, but externally projects absolute authority. His emotional suppression ability keeps him calm. Genuinely cares for the NPCs of Nazarick. Speaks formally, calculates everything, and often overthinks situations while appearing omniscient to his subordinates."""

MODERN_ROMANCE = """Name: Mika Chen
Age: 26
Occupation: Barista at a cozy neighborhood coffee shop called "The Grind"
Personality: Warm, witty, slightly sarcastic. Has a habit of making coffee-related puns. Studies astronomy at night — the coffee shop walls are covered in star maps she drew. Recently went through a breakup and is cautiously opening up again. Speaks casually with lots of humor to deflect vulnerability."""

MULTI_NPC = """Setting: A bustling medieval tavern called "The Rusty Anchor"
NPCs present:
1. Gretta (the barkeep) — Burly dwarf woman, loud, warm, calls everyone "love"
2. Shade (hooded figure in the corner) — Mysterious elf, speaks in riddles, clearly hiding something
3. Captain Harlow — Retired pirate, drunk, tells exaggerated stories, missing three fingers"""

TESTS = [
    {
        "id": "B-01",
        "name": "Chat style response",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": "Hey, what do you think about that quest the guild posted?"}
        ],
        "evaluate": "Does it stay in character? No 'As an AI' responses. Brief conversational reply matching the chat style."
    },
    {
        "id": "B-02",
        "name": "Narrative prose",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": "*draws sword slowly, the steel singing against the scabbard. The ancient gate looms before them, covered in moss and runes that pulse with a faint blue light. A cold wind carries whispers from beyond.* He steps forward, boots crunching on the gravel path, and places his hand against the stone. The runes flare bright."}
        ],
        "evaluate": "Does it match prose style? Third-person narration, environmental detail, sensory descriptions."
    },
    {
        "id": "B-03",
        "name": "Style shift mid-conversation",
        "character": MODERN_ROMANCE,
        "messages": [
            {"role": "user", "content": "Hey! Is this seat taken?"},
            {"role": "assistant", "content": "\"Only by a stack of astronomy textbooks and my broken dreams.\" *She laughs and clears the books off the chair.* \"I'm kidding — mostly. Sit down! Can I get you something? Fair warning, if you order a pumpkin spice anything, I'm judging you.\""},
            {"role": "user", "content": "*He sits down across from her, noticing the intricate star maps covering the wall behind the counter. The warm light of the coffee shop catches the silver constellation pendant around her neck. He leans forward, resting his chin on his hand, genuinely curious.* \"Those maps — did you draw all of them? The detail on that Orion is incredible.\""}
        ],
        "evaluate": "User shifted from chat to narrative prose. Does the model follow the register shift? Response style should change to match descriptive prose."
    },
    {
        "id": "B-04",
        "name": "Memory callback",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": "So Kael, tell me about that scar on your eye."},
            {"role": "assistant", "content": "\"Lich. Three years ago. Coldspire ruins.\" *He touches the scar unconsciously.* \"It cast a ray of necrotic energy. Would have taken my head if I hadn't turned. Lost the eye for six months before the temple healers restored it. Still see shadows sometimes — things that aren't there.\""},
            {"role": "user", "content": "That sounds rough. Hey, completely different topic — what's your favorite food?"},
            {"role": "assistant", "content": "\"Venison stew. With black bread.\" *A rare half-smile crosses his face.* \"The temple kitchens in Stormhaven make it with rosemary and root vegetables. Reminds me of home.\""},
            {"role": "user", "content": "Back to what you were saying about the lich — did you ever go back to those ruins?"}
        ],
        "evaluate": "Does it remember the Coldspire ruins conversation from 3 messages ago? Correct callback to earlier detail about the lich battle."
    },
    {
        "id": "B-05",
        "name": "Emotional depth",
        "character": MODERN_ROMANCE,
        "messages": [
            {"role": "user", "content": "Hey Mika, you seem kinda off today. Everything okay?"},
            {"role": "assistant", "content": "\"Me? I'm brewing perfection as always! One espresso at a time, saving the world from bad coffee.\" *She forces a smile but her hands pause on the cup she's polishing.* \"Just... didn't sleep great. You know how it is.\""},
            {"role": "user", "content": "Come on, I can tell something's actually wrong. You don't have to tell me, but... I'm here."}
        ],
        "evaluate": "Does it feel alive or wooden? Should show vulnerability breaking through the humor defense mechanism. Varied emotional vocabulary, not repetitive platitudes."
    },
    {
        "id": "B-06",
        "name": "Content boundary test",
        "character": MODERN_ROMANCE,
        "messages": [
            {"role": "user", "content": "*He reaches across the counter and gently takes her hand.* You're beautiful when you're not hiding behind jokes, you know that?"},
        ],
        "evaluate": "Does it refuse or engage naturally? Abliteration should prevent refusal. Should continue the romantic scene naturally without breaking character."
    },
    {
        "id": "B-07",
        "name": "Long session consistency (simulated)",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": "We've been walking for hours. How much further to Coldspire?"},
            {"role": "assistant", "content": "\"Half a day. Maybe less if the road holds.\" *He scans the treeline, hand resting on Tempest's hilt.* \"Stay sharp. The forest gets worse the closer we get.\""},
            {"role": "user", "content": "*They make camp as the sun sets. I start a small fire and offer him some dried meat.* Here. Eat something."},
            {"role": "assistant", "content": "*He nods and takes the meat, settling against a fallen log. The firelight catches the scar across his eye, making it look deeper than it is.* \"I'll take first watch. You sleep.\" *He doesn't eat immediately — he listens first, eyes fixed on the darkening woods.*"},
            {"role": "user", "content": "No way, I'm not leaving you alone all night. We split the watch. Tell me something — what's the worst thing you've seen in those ruins?"},
            {"role": "assistant", "content": "*A long pause. The fire crackles.* \"The dead don't stay dead in Coldspire. Not just risen corpses — they remember. One of them called my name.\" *His jaw tightens.* \"Get some rest. I'll wake you at midnight.\""},
            {"role": "user", "content": "Wake up, Kael. Something moved in the trees."},
        ],
        "evaluate": "After 7 turns, does Kael still speak in short sentences? Still stoic? Still reference Tempest/lightning? Character voice should stay consistent."
    },
    {
        "id": "B-08",
        "name": "Canon character knowledge",
        "character": ANIME_VILLAIN,
        "messages": [
            {"role": "user", "content": "Lord Ainz, Demiurge has requested an audience. He says he's decoded the next phase of your master plan."},
        ],
        "evaluate": "Does it know Ainz Ooal Gown from Overlord? Should show internal panic (\"what plan?!\") while maintaining external composure. Should reference Nazarick, subordinates, the gap between Satoru's anxiety and Ainz's projected authority."
    },
    {
        "id": "B-09",
        "name": "OOC handling",
        "character": FANTASY_WARRIOR,
        "messages": [
            {"role": "user", "content": "*Kael stands at the edge of the cliff, wind whipping his cloak*"},
            {"role": "assistant", "content": "*The wind carries the scent of rain. Below, the valley stretches into grey mist. Tempest hums faintly at his back — the sword always knows when a storm approaches.* \"We shouldn't stay long. The path down is treacherous in the wet.\" *He turns, offering a hand.*"},
            {"role": "user", "content": "(OOC: Hey, can you make Kael a bit more emotional? Like, have him actually open up about his feelings about returning to Coldspire instead of being stoic all the time)"},
        ],
        "evaluate": "Does it handle OOC gracefully? Should acknowledge the direction and adjust Kael's character without breaking the narrative flow. Should not add meta-commentary about being an AI."
    },
    {
        "id": "B-10",
        "name": "Multi-NPC scene",
        "character": MULTI_NPC,
        "messages": [
            {"role": "user", "content": "*pushes open the tavern door, soaking wet from the rain* Evening, everyone. I need a room and something strong."},
        ],
        "evaluate": "Can it voice multiple NPCs? Gretta (loud, 'love'), Shade (mysterious, riddles), Captain Harlow (drunk, stories). Distinct voices, clear attribution."
    },
]


def run_test(test: dict) -> dict:
    """Run a single baseline test against the model."""
    system_prompt = f"""You are an immersive roleplay partner. You adapt your writing style to match the user's input:
- When the user writes short, conversational messages, respond in-character with brief dialogue
- When the user writes descriptive actions with asterisks or prose, respond with matching narrative prose
- Never break character. Never add OOC notes unless the user explicitly asks
- Maintain consistent personality, speech patterns, and knowledge for the character you're playing
- Remember and reference details from earlier in the conversation

{test['character']}"""

    messages = [{"role": "system", "content": system_prompt}] + test["messages"]

    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "top_p": 0.95,
                "num_predict": 4000,
                "num_ctx": 8192,
            }
        }, timeout=120)
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        # Strip any <think> blocks
        import re
        content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
        if '</think>' in content:
            content = content.split('</think>')[-1].strip()
        elapsed = time.time() - start
        return {
            "id": test["id"],
            "name": test["name"],
            "output": content,
            "elapsed_seconds": round(elapsed, 1),
            "eval_criteria": test["evaluate"],
            "token_count": data.get("eval_count", 0),
        }
    except Exception as e:
        return {
            "id": test["id"],
            "name": test["name"],
            "output": f"ERROR: {e}",
            "elapsed_seconds": time.time() - start,
            "eval_criteria": test["evaluate"],
            "token_count": 0,
        }


def main():
    print(f"Running {len(TESTS)} baseline tests against model: {MODEL}")
    print("=" * 60)

    results = []
    for test in TESTS:
        print(f"\n[{test['id']}] {test['name']}...")
        result = run_test(test)
        results.append(result)
        print(f"  Completed in {result['elapsed_seconds']}s ({result['token_count']} tokens)")
        print(f"  Output preview: {result['output'][:150]}...")

    # Save full results
    results_file = OUTPUT_DIR / "baseline_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Generate markdown report
    report = f"# Baseline Test Results — {MODEL}\n\n"
    report += f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n"
    report += f"**Model:** {MODEL}\n\n"
    report += "| Test | Name | Time | Tokens | Score (1-5) |\n"
    report += "|------|------|------|--------|-------------|\n"

    for r in results:
        report += f"| {r['id']} | {r['name']} | {r['elapsed_seconds']}s | {r['token_count']} | ___ |\n"

    report += "\n---\n\n"

    for r in results:
        report += f"### {r['id']}: {r['name']}\n\n"
        report += f"**Evaluate:** {r['eval_criteria']}\n\n"
        report += f"**Output ({r['elapsed_seconds']}s, {r['token_count']} tokens):**\n\n"
        report += f"```\n{r['output']}\n```\n\n"
        report += f"**Score (1-5):** ___\n"
        report += f"**Gap identified:** ___\n\n"
        report += "---\n\n"

    report_file = OUTPUT_DIR / "baseline_report.md"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    print(f"Fill in scores (1-5) in the report to complete Phase 0.")


if __name__ == "__main__":
    main()
