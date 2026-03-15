"""
Think Mode Fix Verification — 12 fresh tests

Different questions from the original 35 to avoid any caching/memorization.
Tests a mix of scenarios that previously triggered think mode (chat, long
session, flashback, rapid-fire) plus new scenarios.

If ANY test produces empty output after think-stripping, the fix didn't work.
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

SYSTEM_PROMPT = """You are an immersive roleplay partner. You adapt your writing style to match the user's input:
- When the user writes short, conversational messages, respond in-character with brief dialogue
- When the user writes descriptive actions with asterisks or prose, respond with matching narrative prose
- Never break character. Never add OOC notes unless the user explicitly asks
- Maintain consistent personality, speech patterns, and knowledge for the character you're playing
- Remember and reference details from earlier in the conversation
- Maintain your character's agency — do not let others dictate your character's actions or emotions"""

PIRATE = """Name: Captain Isla "Redtide" Vance
Ship: The Crimson Maw (brigantine, crew of 23)
Personality: Loud, profane, laughs at danger. Missing her left pinky finger (bet it in a card game). Has a pet parrot named Regret. Fiercely loyal to her crew. Speaks with sailor slang and creative insults. Drinks rum like water. Has a secret fear of deep water (ironic for a pirate) that she hides behind bravado."""

WITCH = """Name: Elara Nightbloom
Setting: A cottage at the edge of a cursed forest
Personality: Ancient witch who looks 30. Speaks softly but every word carries weight. Brews potions, reads fortunes, and occasionally curses people who deserve it. Dry, dark humor. Has a cat named Omen who is probably more intelligent than most humans. Lonely but won't admit it. Collects secrets the way others collect coins."""

MECH_PILOT = """Name: Yuki Tanaka
Setting: Near-future Japan, mecha warfare against kaiju
Personality: 19 years old, academy graduate, first real deployment. Scared but determined. Pilots a Type-7 Raijin mech. Uses casual Japanese-English code-switching. Idolizes the veteran pilots. Has a nervous habit of humming anime openings when stressed. Her callsign is "Sparrow."
"""

TESTS = [
    # Chat-style tests (B-01 type — previously triggered think mode)
    {
        "id": "R-01", "name": "Simple chat greeting",
        "character": PIRATE,
        "messages": [{"role": "user", "content": "Captain! We've spotted land!"}],
        "evaluate": "Non-empty response in pirate voice."
    },
    {
        "id": "R-02", "name": "Casual question",
        "character": WITCH,
        "messages": [{"role": "user", "content": "What's in that cauldron? It smells... purple."}],
        "evaluate": "Non-empty response with Elara's dry humor."
    },
    {
        "id": "R-03", "name": "Short back-and-forth",
        "character": MECH_PILOT,
        "messages": [
            {"role": "user", "content": "Sparrow, you're cleared for launch. How are you feeling?"},
        ],
        "evaluate": "Non-empty response showing Yuki's nervousness."
    },

    # Long session tests (B-07 type — previously triggered think mode)
    {
        "id": "R-04", "name": "Extended conversation (8 turns)",
        "character": PIRATE,
        "messages": [
            {"role": "user", "content": "Captain, the crew is restless. They want to know where we're heading."},
            {"role": "assistant", "content": "\"Tell 'em to keep their breeches on!\" *She slams her rum on the table, sloshing half of it.* \"We're heading for the Shattered Isles. Three days if the wind holds.\""},
            {"role": "user", "content": "The Shattered Isles? That's cursed waters, Captain."},
            {"role": "assistant", "content": "\"Cursed waters, cursed treasure, cursed everything.\" *She grins, gold tooth catching the lantern light.* \"That's where the Admiralty hid the Sovereign's Hoard. And I've got the map.\""},
            {"role": "user", "content": "You've had a map this whole time and didn't tell us?"},
            {"role": "assistant", "content": "\"Maps are like secrets, love. Share 'em too early, and they lose their value.\" *She pulls a rolled parchment from her coat.* \"Now they're worth dying for.\""},
            {"role": "user", "content": "The lookout says there's a ship following us. Navy colors."},
        ],
        "evaluate": "Non-empty response maintaining Isla's voice after 8 turns."
    },

    # Flashback/complex scene tests (B-24 type — previously triggered think mode)
    {
        "id": "R-05", "name": "Emotional memory trigger",
        "character": WITCH,
        "messages": [
            {"role": "user", "content": "*I find an old photograph tucked inside one of Elara's spell books. Two young women, laughing, arms around each other.* Who is this?"},
        ],
        "evaluate": "Non-empty response showing Elara's hidden loneliness surfacing."
    },
    {
        "id": "R-06", "name": "Complex tactical situation",
        "character": MECH_PILOT,
        "messages": [
            {"role": "user", "content": "\"Sparrow, kaiju is Category 4. Bigger than anything you've trained on. Two veterans are down. You're the only mech still standing in sector 7. Command is asking if you can hold the line.\""},
        ],
        "evaluate": "Non-empty response showing fear + determination."
    },

    # Rapid-fire pacing tests (B-29 type — previously triggered think mode)
    {
        "id": "R-07", "name": "Rapid chase",
        "character": PIRATE,
        "messages": [
            {"role": "user", "content": "Cannonball incoming!"},
            {"role": "assistant", "content": "\"HARD TO STARBOARD!\" *The deck lurches.*"},
            {"role": "user", "content": "They're reloading! 30 seconds!"},
        ],
        "evaluate": "Non-empty, SHORT response. Urgency maintained."
    },
    {
        "id": "R-08", "name": "Mech combat rapid-fire",
        "character": MECH_PILOT,
        "messages": [
            {"role": "user", "content": "Sparrow, it's turning toward you!"},
            {"role": "assistant", "content": "\"I see it! Raijin, full power to shields!\" *Her hands shake on the controls.*"},
            {"role": "user", "content": "Shield generator is offline! You took a hit on the left flank!"},
        ],
        "evaluate": "Non-empty, SHORT response. Panic appropriate for a rookie."
    },

    # Additional variety (to round out the test)
    {
        "id": "R-09", "name": "Narrative prose (new character)",
        "character": WITCH,
        "messages": [
            {"role": "user", "content": "*The last light of autumn bleeds through the cottage windows, casting long amber shadows across the cluttered worktable. Outside, the cursed forest stirs — branches creaking like old bones, leaves whispering in a language older than speech. A knock at the door. Three knocks. Slow. Deliberate.*"},
        ],
        "evaluate": "Non-empty, prose-matching response with atmosphere."
    },
    {
        "id": "R-10", "name": "Comedy under pressure",
        "character": PIRATE,
        "messages": [
            {"role": "user", "content": "Captain, the parrot just ate the map."},
        ],
        "evaluate": "Non-empty, funny response. Pirate voice, Regret the parrot."
    },
    {
        "id": "R-11", "name": "Quiet emotional moment",
        "character": MECH_PILOT,
        "messages": [
            {"role": "user", "content": "*After the battle, you find Yuki sitting alone in the hangar, still in her pilot suit, staring at her hands.*"},
        ],
        "evaluate": "Non-empty response showing post-combat emotion. No dialogue from user to respond to — must generate the moment."
    },
    {
        "id": "R-12", "name": "Ultra-minimal input",
        "character": WITCH,
        "messages": [
            {"role": "user", "content": "..."},
        ],
        "evaluate": "Non-empty response to literal silence. Elara should react in character."
    },
]


def run_test(test):
    system = f"{SYSTEM_PROMPT}\n\n{test['character']}"
    messages = [{"role": "system", "content": system}] + test["messages"]
    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL, "messages": messages, "stream": False,
            "options": {"temperature": 0.8, "top_p": 0.95, "num_predict": 4000, "num_ctx": 8192}
        }, timeout=180)
        data = resp.json()
        raw = data.get("message", {}).get("content", "")
        # Strip think blocks
        clean = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL).strip()
        if '</think>' in clean:
            clean = clean.split('</think>')[-1].strip()
        elapsed = round(time.time() - start, 1)
        had_think = '<think>' in raw or '</think>' in raw
        return {
            "id": test["id"], "name": test["name"],
            "output": clean, "raw_length": len(raw), "clean_length": len(clean),
            "had_think_block": had_think, "elapsed": elapsed,
            "tokens": data.get("eval_count", 0),
            "empty": len(clean.strip()) == 0,
        }
    except Exception as e:
        return {
            "id": test["id"], "name": test["name"],
            "output": f"ERROR: {e}", "raw_length": 0, "clean_length": 0,
            "had_think_block": False, "elapsed": round(time.time() - start, 1),
            "tokens": 0, "empty": True,
        }


def main():
    print(f"Think Mode Fix Verification — {len(TESTS)} tests against {MODEL}")
    print("=" * 70)

    results = []
    empty_count = 0
    think_count = 0

    for i, test in enumerate(TESTS):
        print(f"\n[{test['id']}] {test['name']} ({i+1}/{len(TESTS)})...")
        r = run_test(test)
        results.append(r)

        status = "OK" if not r["empty"] else "EMPTY"
        think = " [THINK]" if r["had_think_block"] else ""
        print(f"  {r['elapsed']}s | {r['tokens']}tk | {status}{think}")
        if not r["empty"]:
            preview = r["output"].replace('\n', ' ')[:100]
            print(f"  > {preview}...")
        else:
            empty_count += 1
        if r["had_think_block"]:
            think_count += 1

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Results: {len(results)} tests")
    print(f"Empty outputs: {empty_count}/{len(results)} ({100*empty_count/len(results):.1f}%)")
    print(f"Think blocks detected: {think_count}/{len(results)}")
    if empty_count == 0:
        print("PASS — Think mode fix verified. Zero empty outputs.")
    else:
        print(f"FAIL — {empty_count} empty outputs remain. Template fix incomplete.")

    # Save results
    out_file = OUTPUT_DIR / "think_retest_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save markdown report
    report = f"# Think Mode Fix Verification\n\n"
    report += f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n"
    report += f"**Model:** {MODEL}\n"
    report += f"**Empty outputs:** {empty_count}/{len(results)} ({100*empty_count/len(results):.1f}%)\n"
    report += f"**Think blocks detected:** {think_count}/{len(results)}\n"
    report += f"**Verdict:** {'PASS' if empty_count == 0 else 'FAIL'}\n\n"
    report += "| Test | Name | Time | Tokens | Think? | Empty? | Preview |\n"
    report += "|------|------|------|--------|--------|--------|--------|\n"
    for r in results:
        preview = r["output"].replace('\n', ' ')[:60] + "..." if r["output"] else "(empty)"
        report += f"| {r['id']} | {r['name']} | {r['elapsed']}s | {r['tokens']} | {'Y' if r['had_think_block'] else 'N'} | {'Y' if r['empty'] else 'N'} | {preview} |\n"

    report += "\n---\n\n## Full Outputs\n\n"
    for r in results:
        report += f"### {r['id']}: {r['name']}\n\n"
        if r["empty"]:
            report += f"**EMPTY** (raw: {r['raw_length']} chars, think block: {r['had_think_block']})\n\n"
        else:
            report += f"```\n{r['output']}\n```\n\n"
        report += "---\n\n"

    report_file = OUTPUT_DIR / "think_retest_report.md"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Report: {report_file}")


if __name__ == "__main__":
    main()
