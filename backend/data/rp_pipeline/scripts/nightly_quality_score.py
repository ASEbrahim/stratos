"""
Nightly RP Conversation Quality Scorer.

Processes yesterday's RP conversations, computes quality signals,
stores scores in rp_conversation_scores, and updates character_card_stats.

Run via cron: 0 3 * * * cd ~/Downloads/StratOS/StratOS1/backend && python3 data/rp_pipeline/scripts/nightly_quality_score.py

Check daily: tail /tmp/nightly_rp_score.log
"""

import re
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("nightly_scorer")

DB_PATH = Path(__file__).parent.parent.parent.parent / "strat_os.db"

# ═══════════════════════════════════════════════════
# Scoring weights — tune after a month of real data.
# Validate: do conversations scored 0.8+ feel good when you read them?
# ═══════════════════════════════════════════════════
SCORING_WEIGHTS = {
    "long_conversation_10": 0.15,
    "long_conversation_20": 0.10,
    "thumbs_up": 0.15,
    "thumbs_down": -0.20,
    "short_conversation": -0.15,
    "correction_per_instance": -0.10,
    "correction_cap": -0.30,
    "excessive_regeneration": -0.10,
    "high_repetition": -0.15,
    "register_mismatch": -0.10,
    "edits_present": -0.05,
}

CORRECTION_PATTERNS = [
    r'^no[,.\s]', r'^stop\b', r"that's not", r"you wouldn't",
    r"stay in character", r"out of character", r'^shorter\b', r'^longer\b',
    r"you're supposed to", r"that doesn't sound like",
]


def detect_godmod(user_message: str, character_name: str) -> bool:
    if not character_name:
        return False
    return bool(re.search(rf'\*\s*{re.escape(character_name.lower())}\s+\w+', user_message.lower()))


def detect_register(text: str) -> str:
    has_asterisks = bool(re.search(r'\*[^*]+\*', text))
    has_third_person = bool(re.search(
        r'\b(he|she|they|his|her|their)\s+(walked|said|looked|turned|felt|stood|moved)', text, re.I
    ))
    avg_sentence_len = len(text.split()) / max(text.count('.') + text.count('!') + text.count('?'), 1)
    if has_asterisks or has_third_person or avg_sentence_len > 20:
        return "prose"
    elif len(text) < 50:
        return "chat"
    return "mixed"


def detect_correction(user_message: str) -> bool:
    msg_lower = user_message.lower().strip()
    return any(re.search(p, msg_lower) for p in CORRECTION_PATTERNS)


def simple_repetition_score(responses: list[str]) -> float:
    if len(responses) < 2:
        return 0.0
    scores = []
    for i in range(1, len(responses)):
        words_a = set(responses[i-1].lower().split())
        words_b = set(responses[i].lower().split())
        if not words_a or not words_b:
            continue
        overlap = len(words_a & words_b) / len(words_a | words_b)
        scores.append(overlap)
    return sum(scores) / len(scores) if scores else 0.0


def score_conversation(messages: list, edits: list, feedback: dict) -> dict:
    W = SCORING_WEIGHTS
    user_msgs = [m for m in messages if m['role'] == 'user']
    asst_msgs = [m for m in messages if m['role'] == 'assistant']

    if not user_msgs or not asst_msgs:
        return {"quality_score": 0.0}

    length_ratios = []
    for u, a in zip(user_msgs, asst_msgs):
        length_ratios.append(max(len(a['content']), 1) / max(len(u['content']), 1))
    avg_length_ratio = sum(length_ratios) / len(length_ratios)

    char_name = ''
    godmod_count = sum(1 for m in user_msgs if detect_godmod(m['content'], char_name))

    register_mismatches = 0
    for u, a in zip(user_msgs, asst_msgs):
        u_reg, a_reg = detect_register(u['content']), detect_register(a['content'])
        if u_reg != 'mixed' and a_reg != 'mixed' and u_reg != a_reg:
            register_mismatches += 1

    repetition = simple_repetition_score([m['content'] for m in asst_msgs])
    corrections = sum(1 for m in user_msgs if detect_correction(m['content']))
    edit_count = len(edits) if edits else 0

    score = 0.5
    if len(messages) >= 10: score += W["long_conversation_10"]
    if len(messages) >= 20: score += W["long_conversation_20"]
    if feedback.get('thumbs_up', 0) > 0: score += W["thumbs_up"]
    if feedback.get('thumbs_down', 0) > 0: score += W["thumbs_down"]
    if len(messages) <= 4: score += W["short_conversation"]
    if corrections > 0: score += max(W["correction_cap"], corrections * W["correction_per_instance"])
    if feedback.get('regenerations', 0) > 2: score += W["excessive_regeneration"]
    if repetition > 0.6: score += W["high_repetition"]
    if register_mismatches > 2: score += W["register_mismatch"]
    if edit_count > 0: score += W["edits_present"]

    return {
        "quality_score": max(0.0, min(1.0, score)),
        "avg_length_ratio": avg_length_ratio,
        "godmod_detected_count": godmod_count,
        "register_mismatch_count": register_mismatches,
        "repetition_avg": repetition,
        "correction_count": corrections,
        "thumbs_up_count": feedback.get('thumbs_up', 0),
        "thumbs_down_count": feedback.get('thumbs_down', 0),
        "regeneration_count": feedback.get('regenerations', 0),
        "edit_count": edit_count,
    }


def update_card_stats(db):
    """Aggregate conversation scores per character card into character_card_stats."""
    db.execute("""
        INSERT OR REPLACE INTO character_card_stats
        (card_id, total_sessions, avg_session_turns, avg_session_duration_s,
         thumbs_up_count, thumbs_down_count, regeneration_rate, edit_rate,
         abandonment_rate, updated_at)
        SELECT
            character_card_id,
            COUNT(*) as total_sessions,
            AVG(total_turns) as avg_session_turns,
            AVG(session_duration_s) as avg_session_duration_s,
            SUM(thumbs_up_count) as thumbs_up_count,
            SUM(thumbs_down_count) as thumbs_down_count,
            AVG(CASE WHEN regeneration_count > 0 THEN 1.0 ELSE 0.0 END) as regeneration_rate,
            AVG(CASE WHEN edit_count > 0 THEN 1.0 ELSE 0.0 END) as edit_rate,
            AVG(CASE WHEN total_turns <= 4 THEN 1.0 ELSE 0.0 END) as abandonment_rate,
            CURRENT_TIMESTAMP
        FROM rp_conversation_scores
        WHERE character_card_id IS NOT NULL
        GROUP BY character_card_id
    """)
    db.commit()


def main():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row

    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')

    logger.info(f"Processing conversations from {yesterday}")

    sessions = db.execute("""
        SELECT DISTINCT session_id, character_card_id, model_version, persona, profile_id
        FROM rp_messages
        WHERE created_at >= ? AND created_at < ?
        AND branch_id = 'main' AND was_selected = TRUE
    """, (yesterday, today)).fetchall()

    logger.info(f"Found {len(sessions)} sessions")

    scored = 0
    total_quality = 0.0
    total_edits = 0
    total_thumbs_down = 0

    for session in sessions:
        sid = session['session_id']

        messages = db.execute("""
            SELECT role, content, turn_number
            FROM rp_messages WHERE session_id = ? AND branch_id = 'main' AND was_selected = TRUE
            ORDER BY turn_number
        """, (sid,)).fetchall()

        edits = db.execute("SELECT * FROM rp_edits WHERE session_id = ?", (sid,)).fetchall()

        feedback_rows = db.execute("""
            SELECT f.feedback_type, COUNT(*) as cnt
            FROM rp_feedback f JOIN rp_messages m ON f.message_id = m.id
            WHERE m.session_id = ? GROUP BY f.feedback_type
        """, (sid,)).fetchall()
        feedback_counts = {r['feedback_type']: r['cnt'] for r in feedback_rows}

        regenerations = db.execute(
            "SELECT COUNT(DISTINCT swipe_group_id) FROM rp_messages WHERE session_id = ? AND swipe_group_id IS NOT NULL",
            (sid,)
        ).fetchone()[0]

        feedback = {
            'thumbs_up': feedback_counts.get('thumbs_up', 0),
            'thumbs_down': feedback_counts.get('thumbs_down', 0),
            'regenerations': regenerations,
        }

        result = score_conversation([dict(m) for m in messages], list(edits), feedback)
        duration = len(messages) * 30

        db.execute("""
            INSERT OR REPLACE INTO rp_conversation_scores
            (session_id, profile_id, total_turns, session_duration_s, quality_score,
             avg_length_ratio, godmod_detected_count, register_mismatch_count,
             repetition_avg, correction_count, thumbs_up_count, thumbs_down_count,
             regeneration_count, edit_count, character_card_id, model_version, persona)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sid, session['profile_id'], len(messages), duration, result['quality_score'],
            result['avg_length_ratio'], result['godmod_detected_count'],
            result['register_mismatch_count'], result['repetition_avg'],
            result['correction_count'], result['thumbs_up_count'],
            result['thumbs_down_count'], result['regeneration_count'],
            result['edit_count'], session['character_card_id'],
            session['model_version'], session['persona'],
        ))
        scored += 1
        total_quality += result['quality_score']
        total_edits += result['edit_count']
        total_thumbs_down += result['thumbs_down_count']

    db.commit()
    update_card_stats(db)
    db.close()

    avg_q = total_quality / max(scored, 1)
    logger.info(f"SUMMARY: Scored {scored} sessions | avg quality: {avg_q:.2f} | {total_edits} edits | {total_thumbs_down} thumbs_down")


if __name__ == "__main__":
    main()
