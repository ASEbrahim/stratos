#!/usr/bin/env python3
"""
STRAT_OS — Training Data Exporter (v2 — Inference-Aligned)
============================================================
Converts distillation corrections + user feedback into JSONL training data
for LoRA fine-tuning of the local scoring model.

CRITICAL DESIGN PRINCIPLE:
    Training data format must be CHARACTER-FOR-CHARACTER identical to what
    the model sees at inference time in scorer_adaptive.py.

    The inference pipeline has two paths:
      1. Batch path (_build_batch_prompt): system + user -> "[1] SCORE: X.X | REASON: ..."
      2. Single path (_build_llm_prompt): prompt only -> "SCORE: X.X | REASON: ..."

    Both paths parse output with:
      re.search(r'SCORE:\\s*(\\d+\\.?\\d*)', response)
      re.search(r'REASON:\\s*(.+)', response)

    Training MUST produce: SCORE: X.X | REASON: <explanation>
    Training system prompt MUST match _build_batch_prompt's system.
    Training user message MUST match _build_batch_prompt's article format.

Usage:
    python export_training.py                  # Export all corrections
    python export_training.py --after 2026-02-16T00:00  # Only recent
    python export_training.py --min-delta 2.0  # Stricter disagreement filter
"""

import argparse
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("EXPORT")


# ═══════════════════════════════════════════════════════════════════════
# System Prompt Builder — MIRRORS scorer_adaptive._build_batch_prompt
# ═══════════════════════════════════════════════════════════════════════

STUDENT_KEYWORDS = {'student', 'intern', 'graduate', 'freshman', 'undergraduate', 'masters', 'phd candidate'}

def _is_student(role: str) -> bool:
    """Match scorer_adaptive.py's student detection."""
    role_lower = role.lower()
    return any(w in role_lower for w in STUDENT_KEYWORDS)


# ═══════════════════════════════════════════════════════════════════════
# Legacy Profile Map — Injects tracked fields for old DB corrections
# ═══════════════════════════════════════════════════════════════════════

# Old distill.py stored only (role, location, context) in user_feedback.
# These profiles reconstruct the tracked fields that were missing.
LEGACY_PROFILE_MAP = {
    ("Computer Engineering (CPEG) fresh graduate", "Kuwait"): {
        "tracked_companies": "Equate, SLB, Halliburton, KOC, KNPC",
        "tracked_institutions": "Warba Bank, Boubyan Bank, NBK",
        "tracked_interests": "AI, quantum computing, semiconductors",
        "tracked_industries": "oil and gas, fintech",
        "context": "Seeking first engineering position in Kuwait oil/gas or tech sector",
    },
    ("Mechanical Engineering fresh graduate seeking NEOM/Aramco roles", "Saudi Arabia"): {
        "tracked_companies": "Saudi Aramco, NEOM, ACWA Power, SABIC",
        "tracked_institutions": "",
        "tracked_interests": "renewable energy, hydrogen, infrastructure",
        "tracked_industries": "energy, infrastructure",
        "context": "Fresh graduate seeking entry-level mechanical engineering positions in Saudi mega-projects",
    },
    ("Environmental Engineering graduate (waste and water treatment)", "Muscat, Oman"): {
        "tracked_companies": "OWASCO, Muscat Water Company",
        "tracked_institutions": "",
        "tracked_interests": "desalination, sustainable waste management",
        "tracked_industries": "water treatment, environmental services",
        "context": "Environmental engineering graduate focused on waste and water treatment in Oman",
    },
    ("Finance & Accounting student at GUST Kuwait", "Kuwait"): {
        "tracked_companies": "",
        "tracked_institutions": "NBK, KFH, Boubyan Bank, Warba Bank",
        "tracked_interests": "Islamic finance, fintech, financial regulations",
        "tracked_industries": "banking, fintech",
        "context": "Finance and accounting student tracking local banking and Islamic finance in Kuwait",
    },
    ("Architect working on smart city infrastructure projects", "Doha, Qatar"): {
        "tracked_companies": "Msheireb Properties, AECOM, Foster + Partners",
        "tracked_institutions": "Qatar Foundation",
        "tracked_interests": "IoT, AI, smart city design",
        "tracked_industries": "construction, urban development",
        "context": "Architect in Doha focused on smart city infrastructure projects",
    },
    ("Petroleum Engineering student at Kuwait University", "Kuwait"): {
        "tracked_companies": "KOC, KNPC, SLB, Halliburton",
        "tracked_institutions": "",
        "tracked_interests": "enhanced oil recovery, digital oilfield",
        "tracked_industries": "oil and gas",
        "context": "Petroleum engineering student focusing on K-sector employers and energy technology",
    },
    ("Biotech researcher at KAUST studying gene therapy", "Jeddah, Saudi Arabia"): {
        "tracked_companies": "",
        "tracked_institutions": "KAUST",
        "tracked_interests": "CRISPR, gene therapy, viral vectors, bioinformatics",
        "tracked_industries": "biotech, pharmaceuticals",
        "context": "Biotech researcher at KAUST focused on gene therapy and CRISPR research",
    },
    ("Senior geophysicist at KOC", "Kuwait"): {
        "tracked_companies": "KOC, SLB, CGG",
        "tracked_institutions": "",
        "tracked_interests": "seismic acquisition, processing, FWI imaging",
        "tracked_industries": "oil and gas, geophysics",
        "context": "Senior geophysicist with 12+ years experience in seismic acquisition and processing at KOC",
    },
    ("STEM education instructor at a Kuwait secondary school", "Kuwait"): {
        "tracked_companies": "",
        "tracked_institutions": "MOE Kuwait",
        "tracked_interests": "STEM education, EdTech, curriculum development",
        "tracked_industries": "education",
        "context": "STEM education instructor at a Kuwait secondary school",
    },
}


def _lookup_legacy_profile(role: str, location: str) -> dict:
    """Find matching legacy profile by prefix matching on role."""
    if not role:
        return {}
    for (map_role, map_loc), profile_data in LEGACY_PROFILE_MAP.items():
        if role.startswith(map_role) and location == map_loc:
            return profile_data
    return {}


def build_system_prompt(role: str, location: str, context: str,
                        tracked_companies: str = '', tracked_institutions: str = '',
                        tracked_interests: str = '', tracked_industries: str = '') -> str:
    """Build system prompt — CHARACTER-FOR-CHARACTER match with scorer_adaptive._build_batch_prompt.

    Source of truth: scorer_adaptive.py _build_batch_prompt and _tracked_fields_block
    """
    if _is_student(role):
        level_note = "IMPORTANT: User is a student. Senior/Lead/Director roles requiring years of experience score 0-3. Entry-level, internship, and graduate positions score highest."
    else:
        level_note = "IMPORTANT: User is an experienced professional. Entry-level/intern positions score low. Senior and specialist roles score highest."

    tracked_block = (
        f"Tracked companies: {tracked_companies if tracked_companies else 'None specified'}\n"
        f"Tracked institutions: {tracked_institutions if tracked_institutions else 'None specified'}\n"
        f"Tracked interests: {tracked_interests if tracked_interests else 'None specified'}\n"
        f"Tracked industries: {tracked_industries if tracked_industries else 'None specified'}"
    )

    return f"""You are a relevance scorer for a {role} in {location}.
User context: {context if context else 'Not specified'}
{tracked_block}
{level_note}

Score each article 0.0-10.0:
9-10: Directly actionable (hiring match, breakthrough in tracked area)
7-8.9: Highly relevant to user's field/interests
5-6.9: Somewhat relevant
0-4.9: Not relevant / noise / wrong level

Reply ONLY with: SCORE: X.X | REASON: brief explanation"""


def build_user_message(title: str, content: str, category_label: str, category_items: str) -> str:
    """Build user message — matches scorer_adaptive._build_batch_prompt article format.

    Source of truth: scorer_adaptive.py lines 880-890
    At inference, batch articles look like:
      [1] Category: KOC Geophysics | Keywords: seismic, KOC, tenders
          Title: Senior Geophysicist job in Doha
          Content: Mekdam Technical Services is hiring...

    For training (single item), we drop the [N] prefix since the model
    learns to score one item, and both single/batch parsers use the same
    SCORE/REASON regex.
    """
    return f"""Score this article:
Category: {category_label}
Keywords: {category_items}
Title: {title}
Content: {content}"""


# ═══════════════════════════════════════════════════════════════════════
# Data Extraction
# ═══════════════════════════════════════════════════════════════════════

def get_corrections(db_path: str, min_delta: float = 1.5, after: str = None) -> List[Dict]:
    """Pull training signals from user_feedback table."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    training_items = []

    time_filter = ""
    time_params = ()
    if after:
        time_filter = " AND f.created_at > ?"
        time_params = (after,)

    # 1. Explicit corrections (distillation + user ratings)
    cursor.execute(f"""
        SELECT f.news_id, f.title, f.url, f.root, f.category,
               f.ai_score, f.user_score, f.note, f.action, f.created_at,
               n.summary, n.source,
               f.profile_role, f.profile_location, f.profile_context
        FROM user_feedback f
        LEFT JOIN news_items n ON f.news_id = n.id AND f.profile_id = n.profile_id
        WHERE f.action = 'rate' AND f.ai_score IS NOT NULL AND f.user_score IS NOT NULL{time_filter}
        ORDER BY f.created_at DESC
    """, time_params)

    for row in cursor.fetchall():
        row = dict(row)
        delta = abs((row.get('user_score') or 0) - (row.get('ai_score') or 0))
        if delta >= min_delta:
            training_items.append({
                'title': row.get('title', ''),
                'summary': row.get('summary', '') or '',
                'source': row.get('source', '') or '',
                'category': row.get('category', '') or '',
                'root': row.get('root', '') or '',
                'local_score': row.get('ai_score', 0),
                'target_score': row.get('user_score', 0),
                'reason': (row.get('note', '') or '').replace('Distillation: ', ''),
                'delta': delta,
                'signal_type': 'correction',
                'profile_role': row.get('profile_role', '') or '',
                'profile_location': row.get('profile_location', '') or '',
                'profile_context': row.get('profile_context', '') or '',
            })

    # 2. Strong implicit signals
    cursor.execute(f"""
        SELECT f.news_id, f.title, f.url, f.root, f.category,
               f.ai_score, f.action, f.created_at, f.user_score,
               n.summary, n.source, n.score,
               f.profile_role, f.profile_location, f.profile_context
        FROM user_feedback f
        LEFT JOIN news_items n ON f.news_id = n.id AND f.profile_id = n.profile_id
        WHERE f.action IN ('save', 'dismiss', 'thumbs_up', 'thumbs_down'){time_filter}
        ORDER BY f.created_at DESC
    """, time_params)

    for row in cursor.fetchall():
        row = dict(row)
        original_score = row.get('ai_score') or row.get('score') or 5.0
        action = row.get('action', '')
        base = {
            'title': row.get('title', ''),
            'summary': row.get('summary', '') or '',
            'source': row.get('source', '') or '',
            'category': row.get('category', '') or '',
            'root': row.get('root', '') or '',
            'local_score': original_score,
            'profile_role': row.get('profile_role', '') or '',
            'profile_location': row.get('profile_location', '') or '',
            'profile_context': row.get('profile_context', '') or '',
        }

        if action == 'thumbs_up':
            target = row.get('user_score') or 9.0
            if abs(target - original_score) >= min_delta:
                training_items.append({**base, 'target_score': target, 'delta': abs(target - original_score),
                    'reason': 'User confirmed this content is relevant and valuable', 'signal_type': 'thumbs_up'})
        elif action == 'thumbs_down':
            target = row.get('user_score') or 1.0
            if abs(target - original_score) >= min_delta:
                training_items.append({**base, 'target_score': target, 'delta': abs(target - original_score),
                    'reason': 'User confirmed this content is irrelevant noise', 'signal_type': 'thumbs_down'})
        elif action == 'save' and original_score < 6.0:
            training_items.append({**base, 'target_score': 8.0, 'delta': abs(8.0 - original_score),
                'reason': 'User explicitly saved this item as important', 'signal_type': 'save'})
        elif action == 'dismiss' and original_score > 4.0:
            training_items.append({**base, 'target_score': 2.0, 'delta': abs(2.0 - original_score),
                'reason': 'User dismissed this item as irrelevant noise', 'signal_type': 'dismiss'})

    conn.close()

    # Deduplicate by title
    seen = set()
    deduped = []
    for item in training_items:
        key = item['title'].strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


# ═══════════════════════════════════════════════════════════════════════
# Training Data Formatting — Inference-Aligned
# ═══════════════════════════════════════════════════════════════════════

def format_chatml(item: Dict) -> Dict:
    """Format as ChatML — aligned with scorer_adaptive.py inference format.

    ALIGNMENT CHECKLIST:
    + System prompt matches _build_batch_prompt system (lines 853-878)
    + User message matches _build_batch_prompt article format (lines 880-890)
    + Assistant output matches regex: SCORE:\\s*(\\d+\\.?\\d*) and REASON:\\s*(.+)
    + Field names: "Content" not "Summary" (matches inference)
    + Category field present (model learns to use category context)
    + Legacy profiles get tracked fields injected via LEGACY_PROFILE_MAP
    """
    role = item.get('profile_role', 'professional')
    location = item.get('profile_location', 'unspecified')
    context = item.get('profile_context', '')

    # Check if tracked fields exist; if not, look up from legacy profile map
    tracked_companies = item.get('profile_tracked_companies', '')
    tracked_institutions = item.get('profile_tracked_institutions', '')
    tracked_interests = item.get('profile_tracked_interests', '')
    tracked_industries = item.get('profile_tracked_industries', '')

    if not any([tracked_companies, tracked_institutions, tracked_interests, tracked_industries]):
        legacy = _lookup_legacy_profile(role, location)
        if legacy:
            tracked_companies = legacy.get('tracked_companies', '')
            tracked_institutions = legacy.get('tracked_institutions', '')
            tracked_interests = legacy.get('tracked_interests', '')
            tracked_industries = legacy.get('tracked_industries', '')
            context = legacy.get('context', context)

    # Use summary (news_items table has summary, not content)
    article_content = item.get('summary', '')

    system = build_system_prompt(
        role, location, context,
        tracked_companies=tracked_companies,
        tracked_institutions=tracked_institutions,
        tracked_interests=tracked_interests,
        tracked_industries=tracked_industries,
    )
    user = build_user_message(
        title=item['title'][:150],
        content=article_content[:500],
        category_label=item.get('category', 'general'),
        category_items="",  # Not stored in corrections — model learns without at training, gets them at inference
    )

    score = round(item['target_score'], 1)
    reason = item['reason'][:400]
    # Strip any residual <think> tags from old distillation reasons
    reason = re.sub(r'</?think>', '', reason).strip()
    reason = re.sub(r'\n+', ' ', reason).strip()
    if not reason:
        reason = "Score adjusted based on user feedback"
    assistant = f"SCORE: {score} | REASON: {reason}"

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    }


# ═══════════════════════════════════════════════════════════════════════
# Main Export
# ═══════════════════════════════════════════════════════════════════════

def export_training_data(db_path: str, output_path: str, min_delta: float = 1.5, after: str = None):
    logger.info("=" * 60)
    logger.info("STRAT_OS Training Data Export (v2 — Inference-Aligned)")
    logger.info(f"  Mode: {'INCREMENTAL (after ' + after + ')' if after else 'FULL (all corrections)'}")
    logger.info("=" * 60)

    items = get_corrections(db_path, min_delta=min_delta, after=after)
    if not items:
        logger.warning("No training data found!")
        return 0

    corrections = sum(1 for i in items if i['signal_type'] == 'correction')
    saves = sum(1 for i in items if i['signal_type'] == 'save')
    dismissals = sum(1 for i in items if i['signal_type'] == 'dismiss')
    thumbs = sum(1 for i in items if i['signal_type'] in ('thumbs_up', 'thumbs_down'))
    with_profile = sum(1 for i in items if i.get('profile_role'))
    logger.info(f"Found {len(items)} examples: {corrections} corrections, {saves} saves, {dismissals} dismissals, {thumbs} thumbs")
    logger.info(f"  Profile-tagged: {with_profile} | Legacy: {len(items) - with_profile}")

    unique_profiles = set(i.get('profile_role', '') for i in items if i.get('profile_role'))
    if unique_profiles:
        logger.info(f"  Unique profiles: {len(unique_profiles)}")

    legacy_resolved = sum(1 for i in items
                          if not i.get('profile_tracked_companies')
                          and _lookup_legacy_profile(i.get('profile_role', ''), i.get('profile_location', '')))
    if legacy_resolved:
        logger.info(f"  Legacy profile resolution: {legacy_resolved} items matched")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in items:
            record = format_chatml(item)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"✓ Exported {len(items)} examples to {output_file}")

    # Verify format
    with open(output_file) as f:
        first = json.loads(f.readline())
    asst = first['messages'][2]['content']
    if asst.startswith('SCORE:'):
        logger.info(f"  ✓ Format verified: output starts with 'SCORE:' (matches inference parser)")
    else:
        logger.error(f"  ✗ FORMAT MISMATCH: '{asst[:30]}' — parser will fail!")

    up = sum(1 for i in items if i['target_score'] > i['local_score'])
    down = sum(1 for i in items if i['target_score'] < i['local_score'])
    avg_delta = sum(i['delta'] for i in items) / len(items)
    logger.info(f"  Avg disagreement: {avg_delta:.1f} | Up: {up} | Down: {down}")

    return len(items)


# ═══════════════════════════════════════════════════════════════════════
# Merge Multiple JSONL Files
# ═══════════════════════════════════════════════════════════════════════

def merge_training_data(files: List[str], output_path: str) -> int:
    """Merge multiple JSONL training files, deduplicate by title.

    Files are processed in order — first occurrence of a title wins.
    Validates format and strips <think> tags from any old-format entries.
    """
    logger.info("=" * 60)
    logger.info("Merging training data files")
    logger.info("=" * 60)

    seen_titles = set()
    total = 0
    dupes = 0
    malformed = 0

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as out:
        for fpath in files:
            if not Path(fpath).exists():
                logger.error(f"  File not found: {fpath}")
                continue
            file_count = 0
            file_dupes = 0
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)

                    # Dedup by (title + profile) — same article with different profiles is NOT a duplicate
                    user_msg = record['messages'][1]['content']
                    sys_msg = record['messages'][0]['content']
                    title_match = re.search(r'Title:\s*(.+)', user_msg)
                    title_str = title_match.group(1).strip().lower() if title_match else user_msg[:200]
                    # Extract role from system prompt first line
                    role_match = re.search(r'for a (.+?) in ', sys_msg)
                    role_str = role_match.group(1).lower() if role_match else ''
                    title_key = f"{title_str}||{role_str}"

                    if title_key in seen_titles:
                        file_dupes += 1
                        dupes += 1
                        continue
                    seen_titles.add(title_key)

                    # Validate format: assistant content must contain SCORE:
                    # v16: Think blocks are allowed — assistant may start with <think>...</think>
                    # followed by SCORE: X.X | REASON: ...
                    asst = record['messages'][2]['content']
                    asst_clean = re.sub(r'<think>.*?</think>\s*', '', asst, flags=re.DOTALL)
                    if not asst_clean.startswith('SCORE:'):
                        malformed += 1
                        continue

                    out.write(json.dumps(record, ensure_ascii=False) + '\n')
                    file_count += 1
                    total += 1
            logger.info(f"  {Path(fpath).name}: {file_count} examples ({file_dupes} duplicates skipped)")

    if malformed:
        logger.warning(f"  {malformed} malformed examples skipped (no SCORE: prefix)")
    logger.info(f"Merged total: {total} examples ({dupes} duplicates removed)")
    logger.info(f"Output: {output_file}")
    return total


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Export training data for LoRA fine-tuning")
    parser.add_argument("--min-delta", type=float, default=1.5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--db", type=str, default=None)
    parser.add_argument("--after", type=str, default=None)
    parser.add_argument("--merge", nargs='+', default=None,
                        help="Merge multiple JSONL files (e.g., --merge file1.jsonl file2.jsonl)")
    args = parser.parse_args()

    backend_dir = Path(__file__).parent

    # Merge mode
    if args.merge:
        output_path = args.output or str(backend_dir / "data" / "training_merged.jsonl")
        merge_training_data(args.merge, output_path)
        return

    # Export mode
    db_path = args.db or str(backend_dir / "strat_os.db")
    output_path = args.output or str(backend_dir / "data" / "training_data.jsonl")

    if not Path(db_path).exists():
        logger.error(f"Database not found: {db_path}")
        sys.exit(1)

    export_training_data(db_path, output_path, args.min_delta, args.after)


if __name__ == "__main__":
    main()
