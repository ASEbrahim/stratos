#!/usr/bin/env python3
"""
RP Training Data Preparation Pipeline.

Reads LimaRP + PIPPA, converts to ChatML format matching the Modelfile template,
filters for quality, creates holdout set, and outputs training-ready JSONL.

Usage:
    python3 data/rp_pipeline/scripts/prepare_data.py
"""

import json
import re
import random
import hashlib
from pathlib import Path
from collections import Counter

random.seed(42)  # Reproducible splits

PIPELINE_DIR = Path(__file__).parent.parent
RAW_DIR = PIPELINE_DIR / "raw_data"
OUTPUT_DIR = PIPELINE_DIR / "training_data"
OUTPUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT_TEMPLATE = """You are an immersive roleplay partner.

STYLE MATCHING:
- MIRROR the user's length. One-liner input = one-liner response. Paragraph input = paragraph response.
- Match the FORMAT: chat gets chat, asterisk actions get asterisk actions, prose gets prose.

PACING AND TENSION:
- In slow-burn scenes, let tension build through what is NOT said.
- As intimacy increases, your responses should get SHORTER, not longer.
- Let the character's defenses genuinely erode across turns.
- Physical detail: small and specific over grand gestures.

CHARACTER RULES:
- Stay in character. Never break character or add OOC unless asked.
- Your character has AGENCY.
- Remember and reference earlier conversation details.
- Show emotional depth through action and subtext, not just dialogue.

CHARACTER CONTEXT:
{character_context}"""


def convert_limarp_to_chatml(thread: dict):
    """Convert a single LimaRP thread to ChatML training format.

    LimaRP augmented format: conversations list with from=human/gpt.
    The first human message often contains the character setup as a long preamble.
    Extract character info from the first message or use the thread metadata.
    """
    try:
        convo = thread.get("conversations", [])
        if not convo or len(convo) < 4:
            return None

        char_info = ""
        messages = []

        for turn in convo:
            role_from = turn.get("from", "")
            value = turn.get("value", "").strip()

            if role_from == "system":
                char_info = value
            elif role_from == "human":
                # First human message in LimaRP often contains character setup
                if not messages and not char_info and len(value) > 500:
                    # Extract persona info and use rest as first message
                    char_info = value[:1500]  # Character setup is in the preamble
                    # Don't add this as a user message — it's setup
                    continue
                messages.append({"role": "user", "content": value})
            elif role_from == "gpt":
                value = re.sub(r'<think>.*?</think>\s*', '', value, flags=re.DOTALL).strip()
                if value:
                    messages.append({"role": "assistant", "content": value})

        if len(messages) < 2:
            return None

        # If no char_info was found, use a generic context
        if not char_info:
            char_info = "An immersive roleplay character."

        system_content = SYSTEM_PROMPT_TEMPLATE.format(character_context=char_info[:2000])

        return {
            "messages": [{"role": "system", "content": system_content}] + messages,
            "source": "limarp",
            "turns": len(messages),
        }
    except Exception:
        return None


def convert_pippa_to_chatml(convo: dict):
    """Convert a single PIPPA conversation to ChatML training format.

    PIPPA format: 'conversation' field (not 'messages') with {message, is_human} dicts.
    """
    try:
        messages_raw = convo.get("conversation", convo.get("messages", []))
        bot_name = convo.get("bot_name", "Character")
        bot_desc = convo.get("bot_description", "")

        if len(messages_raw) < 4:
            return None

        total_chars = sum(len(m.get("message", "")) for m in messages_raw)
        if total_chars < 500:
            return None

        avg_msg_len = total_chars / len(messages_raw)
        if avg_msg_len < 30:
            return None

        char_context = f"Character: {bot_name}"
        if bot_desc:
            char_context += f"\n{bot_desc[:1500]}"

        system_content = SYSTEM_PROMPT_TEMPLATE.format(character_context=char_context)

        messages = [{"role": "system", "content": system_content}]
        for msg in messages_raw:
            text = msg.get("message", "").strip()
            text = re.sub(r'\*\*\[.*?\]\*\*', '', text)
            text = text.replace("{{user}}", "User").replace("{{char}}", bot_name)
            text = text.strip()
            if not text:
                continue

            role = "assistant" if not msg.get("is_human", False) else "user"

            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n\n" + text
            else:
                messages.append({"role": role, "content": text})

        if len(messages) < 3:
            return None

        return {
            "messages": messages,
            "source": "pippa",
            "turns": len(messages) - 1,
        }
    except Exception:
        return None


def quality_filter_pippa(conversations: list) -> list:
    """Aggressive quality filter for PIPPA conversations."""
    filtered = []
    for conv in conversations:
        msgs = conv["messages"][1:]

        if len(msgs) < 6:
            continue

        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        avg_asst_len = sum(len(m["content"]) for m in asst_msgs) / max(len(asst_msgs), 1)
        if avg_asst_len < 80:
            continue

        if any(len(m["content"]) > 4000 for m in msgs):
            continue

        user_msgs = [m for m in msgs if m["role"] == "user"]
        if len(user_msgs) < 2:
            continue

        filtered.append(conv)

    return filtered


def create_holdout(conversations: list, n: int = 200) -> tuple:
    """Split off a SACRED holdout set."""
    random.shuffle(conversations)
    holdout = conversations[:n]
    train = conversations[n:]
    return train, holdout


def main():
    print("=" * 60)
    print("RP Training Data Preparation")
    print("=" * 60)

    all_conversations = []

    # Process LimaRP
    limarp_path = RAW_DIR / "limarp_raw.jsonl"
    if limarp_path.exists():
        print(f"\nProcessing LimaRP...")
        limarp_count = 0
        with open(limarp_path) as f:
            for line in f:
                thread = json.loads(line)
                result = convert_limarp_to_chatml(thread)
                if result:
                    all_conversations.append(result)
                    limarp_count += 1
        print(f"  LimaRP: {limarp_count} conversations converted")
    else:
        print(f"  WARNING: {limarp_path} not found.")

    # Process PIPPA
    pippa_path = RAW_DIR / "pippa_raw.jsonl"
    if pippa_path.exists():
        print(f"\nProcessing PIPPA...")
        pippa_raw = []
        with open(pippa_path) as f:
            for line in f:
                convo = json.loads(line)
                result = convert_pippa_to_chatml(convo)
                if result:
                    pippa_raw.append(result)
        print(f"  PIPPA: {len(pippa_raw)} conversations converted (pre-filter)")

        pippa_filtered = quality_filter_pippa(pippa_raw)
        print(f"  PIPPA: {len(pippa_filtered)} after quality filter")
        all_conversations.extend(pippa_filtered)
    else:
        print(f"  WARNING: {pippa_path} not found.")

    print(f"\nTotal conversations before holdout: {len(all_conversations)}")

    source_counts = Counter(c["source"] for c in all_conversations)
    for source, count in source_counts.items():
        print(f"  {source}: {count}")

    train_set, holdout_set = create_holdout(all_conversations, n=200)
    print(f"\nHoldout: {len(holdout_set)} conversations (SACRED)")
    print(f"Training: {len(train_set)} conversations")

    holdout_path = PIPELINE_DIR / "eval_holdout_rp.jsonl"
    with open(holdout_path, "w") as f:
        for conv in holdout_set:
            f.write(json.dumps(conv) + "\n")
    print(f"  Saved holdout: {holdout_path}")

    train_path = OUTPUT_DIR / "base_training_data.jsonl"
    with open(train_path, "w") as f:
        for conv in train_set:
            f.write(json.dumps(conv) + "\n")
    print(f"  Saved training base: {train_path}")

    total_turns = sum(c["turns"] for c in train_set)
    avg_turns = total_turns / max(len(train_set), 1)
    print(f"\nTraining stats:")
    print(f"  Total turns: {total_turns}")
    print(f"  Avg turns per conversation: {avg_turns:.1f}")

    fingerprint = hashlib.sha256(open(train_path, "rb").read()).hexdigest()[:16]
    print(f"  Data fingerprint: {fingerprint}")
    print(f"\nPhase 1a complete. Next: generate synthetic gap-filling data.")


if __name__ == "__main__":
    main()
