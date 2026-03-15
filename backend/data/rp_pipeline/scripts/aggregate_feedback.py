"""
Monthly RP Training Data Aggregation.

Extracts high-quality training data from real user interactions.
Produces: feedback_sft.jsonl, feedback_dpo.jsonl, feedback_report.json

ONLY includes data from users who opted in to training data contribution.

DPO Training Notes:
- Recommended beta=0.1 (Microsoft). Higher=conservative, lower=aggressive.
- On-policy data (from current model) is better than off-policy.
- Each model version should train on data from its immediate predecessor:
  V2 SFT → deploy → collect V2 prefs → V3 DPO → deploy → collect V3 prefs → V4 DPO
  Do NOT accumulate all historical preference data.

Usage:
    python3 aggregate_feedback.py
    python3 aggregate_feedback.py --min-score 0.7 --max-context-turns 20 --export
"""

import json
import sqlite3
import logging
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("aggregate")

DB_PATH = Path(__file__).parent.parent.parent.parent / "strat_os.db"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "training_data" / "feedback"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MAX_CONTEXT_TURNS = 20

SYSTEM_PROMPT_TEMPLATE = """You are an immersive roleplay partner.

STYLE MATCHING:
- MIRROR the user's length. One-liner input = one-liner response.
- Match the FORMAT: chat gets chat, asterisk actions get asterisk actions, prose gets prose.

PACING AND TENSION:
- In slow-burn scenes, let tension build through what is NOT said.
- As intimacy increases, your responses should get SHORTER, not longer.
- Let the character's defenses genuinely erode across turns.

CHARACTER RULES:
- Stay in character. Never break character or add OOC unless asked.
- Your character has AGENCY.
- Remember and reference earlier conversation details.

CHARACTER CONTEXT:
{character_context}"""


def extract_sft_data(db, min_score: float = 0.6) -> list:
    sessions = db.execute("""
        SELECT cs.session_id, cs.quality_score, cs.character_card_id, cs.model_version
        FROM rp_conversation_scores cs
        WHERE cs.quality_score >= ? AND cs.total_turns >= 6
        AND cs.profile_id IN (SELECT id FROM profiles WHERE training_data_opt_in = TRUE)
        ORDER BY cs.quality_score DESC
    """, (min_score,)).fetchall()

    conversations = []
    for session in sessions:
        messages = db.execute("""
            SELECT role, content FROM rp_messages
            WHERE session_id = ? AND branch_id = 'main' AND was_selected = TRUE
            ORDER BY turn_number
        """, (session['session_id'],)).fetchall()

        if len(messages) < 4:
            continue

        card = None
        if session['character_card_id']:
            card = db.execute(
                "SELECT name, physical_description, speech_pattern FROM character_cards WHERE id = ?",
                (session['character_card_id'],)
            ).fetchone()

        char_context = f"{card['name']}: {card['physical_description'] or ''} {card['speech_pattern'] or ''}" if card else ""
        system_content = SYSTEM_PROMPT_TEMPLATE.format(character_context=char_context)

        chatml = [{"role": "system", "content": system_content}]
        for msg in messages:
            if msg['role'] in ('user', 'assistant'):
                chatml.append({"role": msg['role'], "content": msg['content']})

        conversations.append({
            "messages": chatml,
            "source": "user_feedback",
            "quality_score": session['quality_score'],
            "session_id": session['session_id'],
        })

    return conversations


def extract_dpo_pairs(db, max_context_turns: int = DEFAULT_MAX_CONTEXT_TURNS) -> list:
    pairs = []

    # 1. From edits (original=rejected, edited=chosen)
    edits = db.execute("""
        SELECT e.message_id, e.original_content, e.edited_content, e.session_id,
               e.edit_delta_category, e.edit_reason, m.turn_number
        FROM rp_edits e JOIN rp_messages m ON e.message_id = m.id
        WHERE m.profile_id IN (SELECT id FROM profiles WHERE training_data_opt_in = TRUE)
    """).fetchall()

    for edit in edits:
        context = db.execute("""
            SELECT role, content FROM rp_messages
            WHERE session_id = ? AND branch_id = 'main'
            AND turn_number < ? AND was_selected = TRUE ORDER BY turn_number
        """, (edit['session_id'], edit['turn_number'])).fetchall()

        if len(context) < 2:
            continue

        orig_words = set(edit['original_content'].lower().split())
        edit_words = set(edit['edited_content'].lower().split())
        if len(orig_words) > 0:
            overlap = len(orig_words & edit_words) / max(len(orig_words | edit_words), 1)
            if abs(len(edit['original_content']) - len(edit['edited_content'])) < 20 and overlap > 0.9:
                continue

        context_msgs = [{"role": r['role'], "content": r['content']} for r in context]
        if len(context_msgs) > max_context_turns:
            context_msgs = context_msgs[-max_context_turns:]

        category = edit['edit_reason'] or edit['edit_delta_category'] or "unknown"
        pairs.append({
            "context": context_msgs,
            "chosen": edit['edited_content'],
            "rejected": edit['original_content'],
            "source": "user_edit",
            "category": category,
        })

    # 2. From swipe selections
    swipe_groups = db.execute("""
        SELECT DISTINCT swipe_group_id FROM rp_messages
        WHERE swipe_group_id IS NOT NULL
        AND profile_id IN (SELECT id FROM profiles WHERE training_data_opt_in = TRUE)
    """).fetchall()

    for group in swipe_groups:
        swipes = db.execute("""
            SELECT content, was_selected, turn_number, session_id
            FROM rp_messages WHERE swipe_group_id = ?
            ORDER BY was_selected DESC, id
        """, (group['swipe_group_id'],)).fetchall()

        if len(swipes) < 2:
            continue

        chosen = [s for s in swipes if s['was_selected']]
        rejected = [s for s in swipes if not s['was_selected']]
        if not chosen or not rejected:
            continue

        context = db.execute("""
            SELECT role, content FROM rp_messages
            WHERE session_id = ? AND branch_id = 'main'
            AND turn_number < ? AND was_selected = TRUE ORDER BY turn_number
        """, (swipes[0]['session_id'], swipes[0]['turn_number'])).fetchall()

        context_msgs = [{"role": r['role'], "content": r['content']} for r in context]
        if len(context_msgs) > max_context_turns:
            context_msgs = context_msgs[-max_context_turns:]

        for r in rejected:
            pairs.append({
                "context": context_msgs,
                "chosen": chosen[0]['content'],
                "rejected": r['content'],
                "source": "swipe_selection",
                "category": "preference",
            })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-score", type=float, default=0.6)
    parser.add_argument("--max-context-turns", type=int, default=DEFAULT_MAX_CONTEXT_TURNS)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row

    sft_data = extract_sft_data(db, args.min_score)
    dpo_pairs = extract_dpo_pairs(db, args.max_context_turns)

    logger.info(f"SFT conversations: {len(sft_data)}")
    logger.info(f"DPO preference pairs: {len(dpo_pairs)}")

    categories = Counter(p['category'] for p in dpo_pairs)
    logger.info(f"DPO categories: {dict(categories)}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "sft_count": len(sft_data),
        "dpo_count": len(dpo_pairs),
        "dpo_categories": dict(categories),
        "min_score_threshold": args.min_score,
        "max_context_turns": args.max_context_turns,
        "avg_sft_quality": sum(c['quality_score'] for c in sft_data) / max(len(sft_data), 1),
        "dpo_training_notes": {
            "recommended_beta": 0.1,
            "note": "Train DPO on top of latest SFT checkpoint. Use on-policy data from immediate predecessor only.",
        },
    }

    print(json.dumps(report, indent=2))

    if args.export:
        ts = datetime.now().strftime('%Y%m%d')
        sft_path = OUTPUT_DIR / f"feedback_sft_{ts}.jsonl"
        with open(sft_path, "w") as f:
            for conv in sft_data:
                f.write(json.dumps(conv) + "\n")
        logger.info(f"SFT exported: {sft_path}")

        dpo_path = OUTPUT_DIR / f"feedback_dpo_{ts}.jsonl"
        with open(dpo_path, "w") as f:
            for pair in dpo_pairs:
                f.write(json.dumps(pair) + "\n")
        logger.info(f"DPO exported: {dpo_path}")

        report_path = OUTPUT_DIR / f"feedback_report_{ts}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report exported: {report_path}")

    db.close()


if __name__ == "__main__":
    main()
