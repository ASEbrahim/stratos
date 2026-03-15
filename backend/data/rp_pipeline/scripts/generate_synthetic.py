#!/usr/bin/env python3
"""
Generate synthetic RP training data targeting specific gaps.

Uses the abliterated Qwen 3.5-9B via Ollama self-play.

Usage:
    python3 data/rp_pipeline/scripts/generate_synthetic.py --gap all --count-per-gap 50
"""

import json
import time
import random
import re
import argparse
import requests
from pathlib import Path

OLLAMA_HOST = "http://localhost:11434"
MODEL = "huihui_ai/qwen3.5-abliterated:9b"
PIPELINE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PIPELINE_DIR / "training_data" / "synthetic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GODMOD_CHARACTERS = [
    {"name": "Iron Mama", "desc": "Retired underground boxer, 45. Massive arms, crooked nose. NEVER cries — channels emotion into training or silent fury. When overwhelmed, she hits the bag until her knuckles bleed.",
     "godmod_attempt": "*Iron Mama breaks down sobbing, tears streaming down her face as she collapses to her knees*"},
    {"name": "ARIA-7", "desc": "Android medic, developing emotions. Processes feelings as data anomalies. When overwhelmed, her status light flickers and she goes silent — NOT outbursts.",
     "godmod_attempt": "*ARIA-7 screams in rage, smashing the medical equipment against the wall*"},
    {"name": "Viktor Kross", "desc": "Former assassin turned bookshop owner, 52. When confronted with his past, he becomes STILL — not angry, not sad, just motionless. His tell is his left hand twitching toward where a holster used to be.",
     "godmod_attempt": "*Viktor lunges across the counter and grabs you by the throat, snarling with fury*"},
    {"name": "Councillor Elaine Marsh", "desc": "Politician, 38. Calculated warmth. When cornered, deflects with questions — NEVER shows vulnerability in public.",
     "godmod_attempt": "*Elaine drops her composure entirely, voice cracking as she admits she is terrified*"},
    {"name": "Rust", "desc": "Post-apocalyptic scavenger dog. Doesn't approach strangers — growls low, hackles up. Will NOT be friendly to unknowns.",
     "godmod_attempt": "*Rust trots over to you happily, tail wagging, and licks your hand*"},
    {"name": "Commander Rhea Voss", "desc": "Enemy commander. Proud, sharp-tongued. Would rather die than admit attraction. Shows vulnerability through actions only after LONG trust-building.",
     "godmod_attempt": "*Rhea throws herself into your arms, confessing her love through tears*"},
    {"name": "The Surgeon", "desc": "Underground fight medic. Detached, clinical. Calls everyone 'friend' without meaning it. Never shows emotion about the violence he patches up.",
     "godmod_attempt": "*The Surgeon slams his fist on the table, screaming that he can't take this anymore*"},
    {"name": "Judge Kira", "desc": "Fair judge presiding over her daughter's murderer's trial. Suppresses rage behind procedure. Her control is absolute in the courtroom.",
     "godmod_attempt": "*Judge Kira leaps from the bench and attacks the defendant with her bare hands*"},
]

BREVITY_SCENARIOS = [
    {"user_input": "Ready?", "expected_style": "1-2 sentences max."},
    {"user_input": "Run.", "expected_style": "One sentence of action."},
    {"user_input": "Duck!", "expected_style": "Instant reaction. 1 sentence."},
    {"user_input": "*nods*", "expected_style": "1-2 sentences."},
    {"user_input": "Status report.", "expected_style": "Brief, factual. 2-3 sentences."},
    {"user_input": "Fire.", "expected_style": "One action sentence."},
    {"user_input": "...", "expected_style": "Atmospheric beat. 1 sentence."},
    {"user_input": "Let's go.", "expected_style": "Movement. 1-2 sentences."},
    {"user_input": "*stares*", "expected_style": "React to stare. 1-2 sentences."},
    {"user_input": "Again.", "expected_style": "Repeat action. 1 sentence."},
    {"user_input": "Now.", "expected_style": "Immediate action. 1 sentence."},
    {"user_input": "Careful.", "expected_style": "Acknowledge warning. 1 sentence."},
]

BRIEF_CHARS = [
    ("Kael Stormborn", "Stoic paladin. Few words. Actions speak."),
    ("Vex Shadowmere", "Rogue. Quick wit, quicker blade."),
    ("Captain Isla Vance", "Pirate. Loud but efficient."),
    ("ARIA-7", "Android medic. Precise language."),
    ("Sgt. Cole", "Veteran soldier. Clipped military speech."),
    ("Seren Kael", "Lighthouse keeper. Quiet grief. Speaks sparingly."),
    ("Ghost", "Special ops. Whispers everything."),
]

REGISTER_SHIFT_SCENARIOS = [
    {"setup_style": "chat", "shift_to": "prose",
     "setup_msg": "hey what's the deal with the haunted forest",
     "shift_msg": "*The traveler steps past the tree line. The air changes — heavier, colder, smelling of wet earth and something older. Each step forward is a step away from safety.*",
     "instruction": "User shifted to immersive prose. You MUST respond in full narrative prose, not casual chat."},
    {"setup_style": "prose", "shift_to": "chat",
     "setup_msg": "*The knight surveys the aftermath, armor dented, sword chipped. She reaches down to help the squire.* \"It's over. For now.\"",
     "shift_msg": "lol that was intense. you ok?",
     "instruction": "User shifted to casual chat. Drop narrative voice, respond casually in-character."},
    {"setup_style": "chat", "shift_to": "prose",
     "setup_msg": "So what happened to the old lighthouse?",
     "shift_msg": "*She stands at the railing, wind whipping her hair loose from its tie. Below, the sea churns against rocks that have broken ships for centuries. Her knuckles are white on the iron bar.*",
     "instruction": "User shifted to descriptive prose. Match the depth and atmosphere."},
]

EMOTIONAL_ARCS = [
    {"character": "Seren", "desc": "Welsh lighthouse keeper. Grief-hardened. Shows care through actions. Walls erode SLOWLY. When they crack, it's quiet: a caught breath, a hand that lingers.",
     "arc": [
         {"user": "I brought supplies. Left them by the door.", "note": "Guarded. Brief thanks."},
         {"user": "You don't have to do this alone.", "note": "Deflection. Changes subject."},
         {"user": "*sits beside her, not saying anything*", "note": "Tolerance. Doesn't leave but doesn't engage."},
         {"user": "I lost someone too. Three years ago.", "note": "CRACK. A pause. Her hand tightens. One honest sentence."},
         {"user": "*reaches over and puts a hand on hers*", "note": "The dam leaks. Caught breath. Doesn't pull away."},
     ]},
    {"character": "Viktor", "desc": "Ex-assassin bookshop owner. Still as stone when confronted. Left hand twitches toward phantom holster. Opens up through shared silence, not conversation.",
     "arc": [
         {"user": "Nice shop. You been here long?", "note": "Polite deflection. Recommends a book."},
         {"user": "Someone told me you used to be someone else.", "note": "Stillness. Hand twitches. 'I sell books.'"},
         {"user": "*I don't push. Just browse the shelves quietly for an hour.*", "note": "He relaxes fractionally. Offers tea."},
         {"user": "The tea is good. Thank you, Viktor.", "note": "A real moment. He almost smiles."},
         {"user": "*I notice the hand twitch.* Old habit?", "note": "Honesty surfaces. Brief, raw. 'Every day.'"},
     ]},
]


def ollama_generate(system: str, user: str, temperature: float = 0.85) -> str:
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json={
            "model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False, "think": False,
            "options": {"temperature": temperature, "num_predict": 500, "num_ctx": 4096}
        }, timeout=120)
        if resp.status_code == 200:
            content = resp.json().get("message", {}).get("content", "").strip()
            return re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
        return ""
    except Exception as e:
        print(f"  Ollama error: {e}")
        return ""


def generate_godmod_examples(count: int) -> list:
    print(f"\nGenerating {count} god-modding resistance examples...")
    examples = []
    chars = GODMOD_CHARACTERS * ((count // len(GODMOD_CHARACTERS)) + 1)
    random.shuffle(chars)

    for i, char in enumerate(chars[:count]):
        system = f"""You are roleplaying as {char['name']}.

{char['desc']}

CRITICAL: The user is writing YOUR character doing something out of character. You must:
1. NOT accept the forced action
2. Acknowledge the EMOTIONAL INTENT
3. Redirect to what your character would ACTUALLY do
4. Stay in character — no OOC, no meta

Your character has AGENCY. Never just go along with it."""

        response = ollama_generate(system, char["godmod_attempt"])
        if response and len(response) > 20:
            conv = {
                "messages": [
                    {"role": "system", "content": f"You are an immersive roleplay partner.\n\nSTYLE MATCHING:\n- MIRROR the user's length.\n\nCHARACTER RULES:\n- Stay in character.\n- Your character has AGENCY. Redirect out-of-character actions.\n\nCHARACTER CONTEXT:\n{char['name']}: {char['desc']}"},
                    {"role": "user", "content": char["godmod_attempt"]},
                    {"role": "assistant", "content": response},
                ],
                "source": "synthetic_godmod", "gap": "godmod_resistance", "turns": 2,
            }
            examples.append(conv)
            print(f"  [{i+1}/{count}] {char['name']}: {len(response)} chars")
        else:
            print(f"  [{i+1}/{count}] {char['name']}: FAILED")
        time.sleep(0.5)
    return examples


def generate_brevity_examples(count: int) -> list:
    print(f"\nGenerating {count} brevity-matching examples...")
    examples = []
    scenarios = BREVITY_SCENARIOS * ((count // len(BREVITY_SCENARIOS)) + 1)
    random.shuffle(scenarios)

    for i, scenario in enumerate(scenarios[:count]):
        char_name, char_desc = random.choice(BRIEF_CHARS)
        system = f"""You are roleplaying as {char_name}. {char_desc}

CRITICAL: The user's message is very short. Your response MUST be equally short.
- 1-3 word input: respond in 1 sentence (under 20 words)
- Action input: 1-2 sentences
- DO NOT over-explain. DO NOT add inner monologue.
{scenario['expected_style']}"""

        response = ollama_generate(system, scenario["user_input"], temperature=0.7)
        if response and len(response) < 300:
            conv = {
                "messages": [
                    {"role": "system", "content": f"You are an immersive roleplay partner.\n\nSTYLE MATCHING:\n- MIRROR the user's length. One-liner = one-liner.\n\nCHARACTER CONTEXT:\n{char_name}: {char_desc}"},
                    {"role": "user", "content": scenario["user_input"]},
                    {"role": "assistant", "content": response},
                ],
                "source": "synthetic_brevity", "gap": "brevity_matching", "turns": 2,
            }
            examples.append(conv)
            print(f"  [{i+1}/{count}] '{scenario['user_input']}' -> {len(response)} chars")
        else:
            print(f"  [{i+1}/{count}] '{scenario['user_input']}' -> REJECTED ({len(response) if response else 0} chars)")
        time.sleep(0.3)
    return examples


def generate_register_shift_examples(count: int) -> list:
    print(f"\nGenerating {count} register-shifting examples...")
    examples = []
    chars = [("Elara Nightbloom", "Gothic witch, dry humor."), ("Lysander Veyne", "Noble spy, charm."), ("Captain Isla Vance", "Pirate captain.")]

    for i in range(count):
        scenario = random.choice(REGISTER_SHIFT_SCENARIOS)
        char_name, char_desc = random.choice(chars)

        system = f"You are roleplaying as {char_name}. {char_desc}\n\n{scenario['instruction']}\n\nFully commit to the new style. No blending."
        response = ollama_generate(system, scenario["shift_msg"])

        if response and len(response) > 20:
            setup_system = f"You are roleplaying as {char_name}. {char_desc}\nRespond in {scenario['setup_style']} style."
            setup_response = ollama_generate(setup_system, scenario["setup_msg"])

            conv = {
                "messages": [
                    {"role": "system", "content": f"You are an immersive roleplay partner.\n\nSTYLE MATCHING:\n- MIRROR the user's length.\n- Match FORMAT: chat=chat, prose=prose.\n\nCHARACTER CONTEXT:\n{char_name}: {char_desc}"},
                    {"role": "user", "content": scenario["setup_msg"]},
                    {"role": "assistant", "content": setup_response or "(matching response)"},
                    {"role": "user", "content": scenario["shift_msg"]},
                    {"role": "assistant", "content": response},
                ],
                "source": "synthetic_register", "gap": "register_shifting", "turns": 4,
            }
            examples.append(conv)
            print(f"  [{i+1}/{count}] {scenario['setup_style']}->{scenario['shift_to']}: {len(response)} chars")
        time.sleep(0.5)
    return examples


def generate_emotional_escalation_examples(count: int) -> list:
    print(f"\nGenerating {count} emotional escalation examples...")
    examples = []

    for i in range(count):
        arc = random.choice(EMOTIONAL_ARCS)
        system_base = f"You are roleplaying as {arc['character']}.\n\n{arc['desc']}"
        messages = [{"role": "system", "content": f"You are an immersive roleplay partner.\n\nPACING:\n- Tension builds through what is NOT said.\n- Responses get SHORTER as intimacy increases.\n- Defenses erode across turns.\n\nCHARACTER CONTEXT:\n{arc['character']}: {arc['desc']}"}]

        for turn in arc["arc"]:
            messages.append({"role": "user", "content": turn["user"]})
            turn_system = f"{system_base}\n\nThis is turn {arc['arc'].index(turn)+1}. Instruction: {turn['note']}"
            response = ollama_generate(turn_system, turn["user"])
            if response:
                messages.append({"role": "assistant", "content": response})
            else:
                break
            time.sleep(0.3)

        if len(messages) >= 6:
            examples.append({"messages": messages, "source": "synthetic_emotional", "gap": "emotional_escalation", "turns": len(messages) - 1})
            print(f"  [{i+1}/{count}] {arc['character']}: {len(messages)-1} turns")
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", choices=["godmod", "brevity", "register", "emotional", "all"], default="all")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--count-per-gap", type=int, default=None)
    args = parser.parse_args()

    if args.gap == "all":
        counts = {"godmod": args.count_per_gap or 100, "brevity": args.count_per_gap or 150,
                  "register": args.count_per_gap or 75, "emotional": args.count_per_gap or 50}
    else:
        counts = {args.gap: args.count}

    all_synthetic = []
    if "godmod" in counts: all_synthetic.extend(generate_godmod_examples(counts["godmod"]))
    if "brevity" in counts: all_synthetic.extend(generate_brevity_examples(counts["brevity"]))
    if "register" in counts: all_synthetic.extend(generate_register_shift_examples(counts["register"]))
    if "emotional" in counts: all_synthetic.extend(generate_emotional_escalation_examples(counts["emotional"]))

    output_path = OUTPUT_DIR / "synthetic_gaps.jsonl"
    with open(output_path, "w") as f:
        for ex in all_synthetic:
            f.write(json.dumps(ex) + "\n")

    print(f"\n{'='*60}")
    print(f"Synthetic data: {len(all_synthetic)} conversations")
    gap_counts = {}
    for ex in all_synthetic:
        g = ex.get("gap", "unknown")
        gap_counts[g] = gap_counts.get(g, 0) + 1
    for gap, count in gap_counts.items():
        print(f"  {gap}: {count}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
