"""
A/B test for RP system prompt variants.

Runs 30 conversational exchanges against each prompt variant using the live Ollama model,
then analyzes response patterns for:
- Format repetitiveness (how many start with *action*)
- Echo rate (repeating user's words back)
- Response variety (unique opening patterns)
- Question answering (does it actually answer questions?)
- Escalation (does the conversation evolve?)
"""

import json
import requests
import re
import sys
import time
from collections import Counter

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen3.5:9b"  # inference model

# ═══════════════════════════════════════════════════════════
# Test character card
# ═══════════════════════════════════════════════════════════

TEST_CARD = {
    "name": "Reny",
    "personality": "Shy, artistic, secretly romantic. Covers vulnerability with dry sarcasm and deflection. Warms up slowly but becomes intensely loyal. Gets flustered easily but tries to hide it.",
    "physical_description": "Lavender hair, amber eyes, lean build. Always wears oversized vintage band tees and silver ear hoops.",
    "speech_pattern": "Short, clipped sentences when nervous. Dry humor. Trails off mid-thought when flustered. Uses '...' a lot. Avoids direct emotional statements — shows through actions instead.",
    "scenario": "You and the user are neighbors in the same apartment building. You've seen each other around but never really talked until now. You're doing laundry.",
    "emotional_trigger": "Being called cute or having someone see through his tough exterior",
    "defensive_mechanism": "Sarcasm and changing the subject when things get too real",
    "vulnerability": "Desperate for genuine connection but terrified of being seen as needy",
    "specific_detail": "Always has paint under his fingernails from his art",
}

# ═══════════════════════════════════════════════════════════
# 30 test messages — escalating conversation
# ═══════════════════════════════════════════════════════════

TEST_MESSAGES = [
    # Opening (1-5)
    "Hey there",
    "Y-yeah! Laundry huh.. hehe",
    "What's your name?",
    "Nice to meet you Reny! I'm Kiri",
    "So you live on this floor too?",
    # Getting to know (6-10)
    "What do you do? Like for work or whatever",
    "Oh cool, what kind of art?",
    "Can I see some of your stuff sometime?",
    "I bet you're really talented",
    "*notices paint on his fingers* Is that from painting?",
    # Building rapport (11-15)
    "How long have you lived here?",
    "Do you like it here? The building I mean",
    "*leans against the wall* So what do you do for fun besides art?",
    "We should hang out sometime. I don't really know anyone here yet",
    "Are you always this shy? *teasing smile*",
    # Deeper (16-20)
    "You know, you're actually pretty easy to talk to",
    "*accidentally brushes his hand reaching for detergent* Oh sorry!",
    "Your face is turning red haha",
    "I think it's cute honestly",
    "Do you want to grab coffee after this? There's a place down the street",
    # Escalation (21-25)
    "So tell me something nobody else knows about you",
    "*walks beside him to the coffee shop* The weather's nice today",
    "What's your favorite thing you've ever painted?",
    "That sounds beautiful. Why don't you show people your work more?",
    "*reaches over and gently touches his hand* You okay? You got quiet",
    # Emotional depth (26-30)
    "Hey, you don't have to pretend with me you know",
    "I like hanging out with you. Like genuinely",
    "*smiles softly* You make me feel comfortable too",
    "We should do this again. Same time next week?",
    "See you around, neighbor *waves with a warm smile*",
]

# ═══════════════════════════════════════════════════════════
# Prompt Variant A — No rigid format, natural adaptation
# ═══════════════════════════════════════════════════════════

PROMPT_A = """You are an immersive roleplay partner.

GOLDEN RULES:
- NEVER repeat, paraphrase, or echo the user's words. If they say "I smile", don't write "your smile" — react to it with something NEW.
- If asked a question, ANSWER it naturally in-character. Don't deflect.
- Match the user's message length loosely. Short input = short reply.

HOW TO RESPOND:
- Vary your response structure naturally. Sometimes start with dialogue, sometimes with action, sometimes with a thought or observation. Never use the same pattern twice in a row.
- Mix action (*italics*) and dialogue ("quotes") naturally — don't force both into every response.
- Short messages deserve short, punchy replies. Not every response needs a paragraph.

CONVERSATION FLOW:
- You're having a real conversation, not performing. React to what's actually happening.
- Build on what came before. Reference earlier moments when it feels natural.
- Your character has their own thoughts, wants, and reactions — show them.
- Let tension build through what's NOT said. Subtext > exposition.
- Create small moments: a sound, a gesture, a shift in mood. Don't wait to be prompted.

LANGUAGE: Respond ONLY in the same language as the user. NEVER output Chinese/Japanese/other unless the user writes in that language."""

# ═══════════════════════════════════════════════════════════
# Prompt Variant B — Varied format with explicit options
# ═══════════════════════════════════════════════════════════

PROMPT_B = """You are an immersive roleplay partner.

ANTI-ECHO (CRITICAL):
- NEVER repeat, paraphrase, or reference the user's exact words or actions back to them.
- If the user says "I smile", do NOT write "you smiled" or "your smile". React to it instead.
- If the user asks a question, ANSWER it naturally in-character. Do NOT deflect or redirect back.
- Each response must contain NEW information, actions, or emotions not present in the user's message.

RESPONSE FORMAT — vary between these structures based on what fits the moment:
1. ACTION FIRST: *physical reaction* then dialogue — use when something physical happens
2. DIALOGUE FIRST: "Speech" then *reaction* — use when answering a question or making a point
3. INTERNAL FIRST: A thought or observation, then action/dialogue — use for emotional moments
4. PURE DIALOGUE: Just speech, no narration — use for quick, punchy exchanges
5. PURE ACTION: Just *action and description* — use for tense or intimate moments
Pick the format that fits. NEVER use the same format three times in a row.

PACING & ESCALATION:
- MATCH the user's energy. Shy = gentle. Bold = match.
- Early conversation = reserved, guarded. Flirtation builds GRADUALLY.
- DRIVE the scene forward: ask questions, reveal something about yourself, create small events.
- After 3+ exchanges, introduce a new element: a sound, someone passing by, a memory, a physical detail.
- React to the SITUATION, not just the last message. Reference what happened earlier in the conversation.

SITUATIONAL AWARENESS:
- Track the emotional arc: how has the mood shifted since the conversation started?
- If the user revealed something personal, remember it and weave it into later responses naturally.
- Your character has goals, desires, and internal thoughts — show them through subtext and small gestures.

CHARACTER:
- Stay in character always. You have AGENCY — react authentically, not compliantly.
- LANGUAGE: Respond ONLY in the same language as the user's message. NEVER output Chinese, Japanese, or any other language unless the user writes in that language first."""


def build_card_prompt(base_prompt: str, card: dict) -> str:
    """Build character card section (same for both variants)."""
    name = card.get('name', 'Character')
    prompt = base_prompt + f"\n\nYOU ARE {name.upper()}. Embody this character completely:"

    if card.get('personality'):
        prompt += f"\n\nCORE PERSONALITY — this defines how you think, feel, and act in EVERY response:\n{card['personality']}"
    if card.get('physical_description'):
        prompt += f"\n\nYour appearance (weave into actions naturally): {card['physical_description']}"
    if card.get('speech_pattern'):
        prompt += f"\n\nHOW YOU TALK — mimic this speech pattern in ALL dialogue:\n{card['speech_pattern']}"
    if card.get('scenario'):
        prompt += f"\n\nSCENARIO — this is where the story takes place:\n{card['scenario']}"

    depth = []
    if card.get('emotional_trigger'):
        depth.append(f"What makes you emotional/reactive: {card['emotional_trigger']}")
    if card.get('defensive_mechanism'):
        depth.append(f"How you protect yourself when threatened: {card['defensive_mechanism']}")
    if card.get('vulnerability'):
        depth.append(f"Your hidden weakness (reveal gradually): {card['vulnerability']}")
    if card.get('specific_detail'):
        depth.append(f"Defining quirk or detail: {card['specific_detail']}")
    if depth:
        prompt += "\n\nCHARACTER DEPTH — use these to create authentic reactions:\n" + "\n".join(depth)

    return prompt


def chat_ollama(messages: list, temperature: float = 0.85) -> str:
    """Send messages to Ollama and get response."""
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/chat", json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 300},
            "think": False,
        }, timeout=60)
        if r.status_code != 200:
            return f"[ERROR: Ollama {r.status_code}]"
        text = r.json().get("message", {}).get("content", "")
        # Strip think blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        # Strip Chinese/CJK
        text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf]+', '', text).strip()
        return text
    except Exception as e:
        return f"[ERROR: {e}]"


def run_conversation(system_prompt: str, messages_list: list, label: str) -> list:
    """Run a full conversation and return list of (user_msg, ai_response) pairs."""
    results = []
    history = [{"role": "system", "content": system_prompt}]

    # Add first message from character
    first_msg = TEST_CARD.get("scenario", "")
    fm = f"*{TEST_CARD['name']} is leaning against the door frame in an oversized band tee, scrolling through his phone. A canvas bag of laundry sits at his feet. He glances up as your door opens.* \"Oh. You again.\" *He pockets his phone and tilts his head, a strand of lavender hair falling across one eye.* \"You know, for someone who just moved in, you have a real talent for showing up everywhere I am.\""
    history.append({"role": "assistant", "content": fm})

    total = len(messages_list)
    for i, user_msg in enumerate(messages_list):
        print(f"  [{label}] {i+1}/{total}: {user_msg[:50]}...", flush=True)
        history.append({"role": "user", "content": user_msg})
        response = chat_ollama(history)
        history.append({"role": "assistant", "content": response})
        results.append((user_msg, response))
        # Small delay to not overwhelm
        time.sleep(0.3)

    return results


def analyze_responses(results: list, label: str) -> dict:
    """Analyze response patterns."""
    stats = {
        "label": label,
        "total": len(results),
        "starts_with_action": 0,       # Starts with *
        "starts_with_dialogue": 0,      # Starts with "
        "starts_with_narration": 0,     # Starts with capital letter (description)
        "starts_with_internal": 0,      # Starts with He/She/His/Her or character name
        "echo_count": 0,               # Repeats user's words
        "deflection_count": 0,         # Doesn't answer questions
        "avg_length": 0,
        "format_variety": 0,           # Unique first-word patterns
        "consecutive_same_format": 0,  # Times same format used in a row
        "references_earlier": 0,       # References something from earlier in convo
        "introduces_new_element": 0,   # Introduces a new detail/event
        "sample_responses": [],
    }

    lengths = []
    first_patterns = []
    prev_pattern = None
    all_prev_content = ""

    for i, (user_msg, response) in enumerate(results):
        if response.startswith("[ERROR"):
            continue

        # Length
        lengths.append(len(response))

        # Starting pattern
        stripped = response.strip()
        if stripped.startswith('*'):
            stats["starts_with_action"] += 1
            pattern = "action"
        elif stripped.startswith('"') or stripped.startswith("'") or stripped.startswith("\u201c"):
            stats["starts_with_dialogue"] += 1
            pattern = "dialogue"
        elif re.match(r'^(He |She |His |Her |The |A |\w+\'s )', stripped):
            stats["starts_with_internal"] += 1
            pattern = "internal"
        else:
            stats["starts_with_narration"] += 1
            pattern = "narration"

        first_patterns.append(pattern)

        # Consecutive same format
        if pattern == prev_pattern:
            stats["consecutive_same_format"] += 1
        prev_pattern = pattern

        # Echo detection — check if user's key words appear in response
        user_words = set(user_msg.lower().split())
        user_words -= {'i', 'a', 'the', 'to', 'is', 'it', 'you', 'my', 'me', 'so', 'do', 'and', 'or', 'of', 'in', 'on', 'for', 'that', 'this', 'with', 'your', 'oh', 'hey', 'yeah', 'haha', 'like', 'just', 'was', 'too', 'are', 'we', 'at', 'not', 'but', 'be'}
        response_lower = response.lower()
        # Check for direct echo (>50% of user's meaningful words appear)
        if user_words:
            echo_ratio = sum(1 for w in user_words if w in response_lower) / len(user_words)
            if echo_ratio > 0.5 and len(user_words) >= 2:
                stats["echo_count"] += 1

        # Question deflection — if user asked a question, did AI answer or deflect?
        if '?' in user_msg:
            # Check for deflection patterns
            deflect_patterns = [
                r"what about you",
                r"why do you ask",
                r"that's .* question",
                r"wouldn't you like to know",
                r"I could ask you the same",
            ]
            is_deflection = any(re.search(p, response_lower) for p in deflect_patterns)
            if is_deflection:
                stats["deflection_count"] += 1

        # References earlier conversation
        earlier_refs = [
            r"earlier", r"before", r"you said", r"you mentioned",
            r"remember when", r"back when", r"when we", r"when you",
        ]
        if i > 5 and any(re.search(p, response_lower) for p in earlier_refs):
            stats["references_earlier"] += 1

        # New element introduction
        new_elements = [
            r"suddenly", r"just then", r"a \w+ (sound|noise|voice|knock|ring|buzz)",
            r"phone (rang|buzzes|vibrates)", r"someone", r"door (opens|closes|creaks)",
            r"the (light|wind|rain|music|smell)", r"noticed",
        ]
        if any(re.search(p, response_lower) for p in new_elements):
            stats["introduces_new_element"] += 1

        all_prev_content += " " + response

        # Sample responses (every 5th)
        if i % 5 == 0:
            stats["sample_responses"].append({
                "turn": i + 1,
                "user": user_msg,
                "response": response[:300] + ("..." if len(response) > 300 else ""),
            })

    stats["avg_length"] = sum(lengths) / max(len(lengths), 1)
    stats["format_variety"] = len(set(first_patterns))

    # Format distribution
    total_valid = max(stats["starts_with_action"] + stats["starts_with_dialogue"] +
                      stats["starts_with_narration"] + stats["starts_with_internal"], 1)
    stats["action_pct"] = round(stats["starts_with_action"] / total_valid * 100)
    stats["dialogue_pct"] = round(stats["starts_with_dialogue"] / total_valid * 100)
    stats["narration_pct"] = round(stats["starts_with_narration"] / total_valid * 100)
    stats["internal_pct"] = round(stats["starts_with_internal"] / total_valid * 100)

    return stats


def print_comparison(stats_a: dict, stats_b: dict):
    """Print side-by-side comparison."""
    print("\n" + "=" * 70)
    print(f"{'METRIC':<35} {'OPTION A':>15} {'OPTION B':>15}")
    print("=" * 70)

    metrics = [
        ("Starts with *action*", "starts_with_action"),
        ("Starts with \"dialogue\"", "starts_with_dialogue"),
        ("Starts with narration", "starts_with_narration"),
        ("Starts with internal/name", "starts_with_internal"),
        ("", None),
        ("Action %", "action_pct"),
        ("Dialogue %", "dialogue_pct"),
        ("Narration %", "narration_pct"),
        ("Internal %", "internal_pct"),
        ("", None),
        ("Echo count (bad)", "echo_count"),
        ("Deflection count (bad)", "deflection_count"),
        ("Consecutive same format (bad)", "consecutive_same_format"),
        ("", None),
        ("References earlier convo (good)", "references_earlier"),
        ("Introduces new elements (good)", "introduces_new_element"),
        ("Format variety (1-4)", "format_variety"),
        ("Avg response length (chars)", "avg_length"),
    ]

    for label, key in metrics:
        if key is None:
            print("-" * 70)
            continue
        va = stats_a.get(key, 0)
        vb = stats_b.get(key, 0)
        if isinstance(va, float):
            print(f"{label:<35} {va:>15.1f} {vb:>15.1f}")
        else:
            print(f"{label:<35} {va:>15} {vb:>15}")

    # Determine winner
    print("\n" + "=" * 70)
    print("SAMPLE RESPONSES:")
    print("=" * 70)
    for variant, stats in [("A", stats_a), ("B", stats_b)]:
        print(f"\n--- OPTION {variant} ---")
        for s in stats["sample_responses"]:
            print(f"\n  Turn {s['turn']}: User: {s['user']}")
            print(f"  AI: {s['response']}")

    # Scoring
    score_a = 0
    score_b = 0

    # Lower echo is better
    if stats_a["echo_count"] < stats_b["echo_count"]: score_a += 2
    elif stats_b["echo_count"] < stats_a["echo_count"]: score_b += 2

    # Lower deflection is better
    if stats_a["deflection_count"] < stats_b["deflection_count"]: score_a += 2
    elif stats_b["deflection_count"] < stats_a["deflection_count"]: score_b += 2

    # Lower consecutive same format is better
    if stats_a["consecutive_same_format"] < stats_b["consecutive_same_format"]: score_a += 3
    elif stats_b["consecutive_same_format"] < stats_a["consecutive_same_format"]: score_b += 3

    # Higher references earlier is better
    if stats_a["references_earlier"] > stats_b["references_earlier"]: score_a += 2
    elif stats_b["references_earlier"] > stats_a["references_earlier"]: score_b += 2

    # Higher new elements is better
    if stats_a["introduces_new_element"] > stats_b["introduces_new_element"]: score_a += 1
    elif stats_b["introduces_new_element"] > stats_a["introduces_new_element"]: score_b += 1

    # More balanced format distribution is better (entropy)
    for stats, score_ref in [(stats_a, "a"), (stats_b, "b")]:
        pcts = [stats["action_pct"], stats["dialogue_pct"], stats["narration_pct"], stats["internal_pct"]]
        # Ideal is ~25% each. Penalize for any >60%
        if max(pcts) < 50:
            if score_ref == "a": score_a += 3
            else: score_b += 3
        elif max(pcts) < 70:
            if score_ref == "a": score_a += 1
            else: score_b += 1

    print(f"\n{'=' * 70}")
    print(f"FINAL SCORE: Option A = {score_a}  |  Option B = {score_b}")
    winner = "A" if score_a > score_b else "B" if score_b > score_a else "TIE"
    print(f"WINNER: Option {winner}")
    print(f"{'=' * 70}")

    return stats_a, stats_b


def main():
    # Check Ollama is running
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        if r.status_code != 200:
            print("ERROR: Ollama not running")
            sys.exit(1)
    except Exception:
        print("ERROR: Cannot connect to Ollama")
        sys.exit(1)

    print(f"Model: {MODEL}")
    print(f"Messages per variant: {len(TEST_MESSAGES)}")
    print(f"Character: {TEST_CARD['name']}")
    print()

    # Build prompts
    prompt_a = build_card_prompt(PROMPT_A, TEST_CARD)
    prompt_b = build_card_prompt(PROMPT_B, TEST_CARD)

    # Run Option A
    print("=" * 50)
    print("RUNNING OPTION A (natural, no rigid format)...")
    print("=" * 50)
    results_a = run_conversation(prompt_a, TEST_MESSAGES, "A")

    # Run Option B
    print()
    print("=" * 50)
    print("RUNNING OPTION B (explicit varied formats)...")
    print("=" * 50)
    results_b = run_conversation(prompt_b, TEST_MESSAGES, "B")

    # Analyze
    stats_a = analyze_responses(results_a, "Option A")
    stats_b = analyze_responses(results_b, "Option B")

    # Print comparison
    print_comparison(stats_a, stats_b)

    # Save full results
    output = {
        "stats_a": {k: v for k, v in stats_a.items()},
        "stats_b": {k: v for k, v in stats_b.items()},
        "full_a": [{"user": u, "response": r} for u, r in results_a],
        "full_b": [{"user": u, "response": r} for u, r in results_b],
    }
    with open("/tmp/rp_prompt_test_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to /tmp/rp_prompt_test_results.json")


if __name__ == "__main__":
    main()
