#!/usr/bin/env python3
"""
V2 Training Pipeline — Stage 4: Data Preparation
==================================================
From scored examples:
  1. Loss weighting (critical > noise)
  2. Contrastive pair extraction (score gap >= 3.0)
  3. Curriculum sorting (easy → hard)
  4. Train/eval split (90/10 stratified)
  5. Format for training (SCORE: X.X | REASON: format)
"""

import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.chdir(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.v2_pipeline.profiles_v2 import ALL_PROFILES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("V2_PREPARE")

OUTPUT_DIR = Path(__file__).parent
SCORES_FILE = OUTPUT_DIR / "scores_v2.json"
ARTICLES_FILE = OUTPUT_DIR / "articles_v2.json"
TRAIN_FILE = OUTPUT_DIR / "training_v2.jsonl"
EVAL_FILE = OUTPUT_DIR / "eval_v2.jsonl"
CONTRASTIVE_FILE = OUTPUT_DIR / "contrastive_pairs_v2.json"
REPORT_FILE = OUTPUT_DIR / "stage4_report.json"

# Profile lookup
PROFILE_MAP = {p['id']: p for p in ALL_PROFILES}

# Score bands (same as train_lora.py)
SCORE_BANDS = {
    "noise": (0.0, 2.0),
    "tangential": (2.5, 4.0),
    "moderate": (4.5, 6.5),
    "high": (7.0, 8.0),
    "critical": (8.5, 10.0),
}

# Loss weights per band
LOSS_WEIGHTS = {
    "noise": 0.5,
    "tangential": 1.0,
    "moderate": 1.5,
    "high": 2.0,
    "critical": 3.0,
}


def score_to_band(score: float) -> str:
    """Map score to band name."""
    if score >= 8.5:
        return "critical"
    elif score >= 7.0:
        return "high"
    elif score >= 4.5:
        return "moderate"
    elif score >= 2.5:
        return "tangential"
    return "noise"


def _is_student(role: str) -> bool:
    """Detect if role is a student."""
    role_lower = role.lower()
    return any(kw in role_lower for kw in ['student', 'freshman', 'sophomore', 'junior', 'senior year',
                                            'undergraduate', 'graduate student', 'phd candidate',
                                            'fresh graduate', 'undeclared'])


def build_training_example(scored: dict, profile: dict, article: dict) -> dict:
    """Build a ChatML training example matching export_training.py format."""
    role = profile['role']
    location = profile['location']
    context = profile.get('context', 'Not specified')
    companies = profile.get('tracked_companies', 'None specified')
    institutions = profile.get('tracked_institutions', 'None specified')
    interests = ', '.join(profile.get('interests', []))
    industries = profile.get('tracked_industries', 'None specified')

    # Student detection
    if _is_student(role):
        level_note = "IMPORTANT: User is a student. Senior/Lead/Director roles requiring years of experience score 0-3. Entry-level, internship, and graduate positions score highest."
    else:
        level_note = "IMPORTANT: User is an experienced professional. Entry-level/intern positions score low. Senior and specialist roles score highest."

    system_prompt = f"""You are a relevance scorer for a {role} in {location}.
User context: {context}
Tracked companies: {companies}
Tracked institutions: {institutions}
Tracked interests: {interests if interests else 'None specified'}
Tracked industries: {industries}
{level_note}

Score each article 0.0-10.0:
9-10: Directly actionable (hiring match, breakthrough in tracked area)
7-8.9: Highly relevant to user's field/interests
5-6.9: Somewhat relevant
0-4.9: Not relevant / noise / wrong level

Reply ONLY with: SCORE: X.X | REASON: brief explanation"""

    title = article.get('title', '')[:200]
    content = article.get('summary', '')[:500]

    user_message = f"""Score this article:
Category: general
Keywords: {interests}
Title: {title}
Content: {content}"""

    score = scored['score']
    reason = scored.get('reason', 'No reason provided')

    assistant_message = f"SCORE: {score:.1f} | REASON: {reason}"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def extract_contrastive_pairs(scored_by_article: dict) -> list:
    """Find all profile pairs where score gap >= 3.0 for the same article."""
    pairs = []
    for article_id, profile_scores in scored_by_article.items():
        # Sort by score
        sorted_scores = sorted(profile_scores, key=lambda x: x['score'])
        for i in range(len(sorted_scores)):
            for j in range(i + 1, len(sorted_scores)):
                gap = sorted_scores[j]['score'] - sorted_scores[i]['score']
                if gap >= 3.0:
                    pairs.append({
                        'article_id': article_id,
                        'low_profile': sorted_scores[i]['profile_id'],
                        'low_score': sorted_scores[i]['score'],
                        'high_profile': sorted_scores[j]['profile_id'],
                        'high_score': sorted_scores[j]['score'],
                        'gap': gap,
                    })
    return pairs


def main():
    print("=" * 80)
    print("V2 TRAINING PIPELINE — STAGE 4: DATA PREPARATION")
    print("=" * 80)

    # Load data
    if not SCORES_FILE.exists():
        print(f"ERROR: {SCORES_FILE} not found. Run stage3_score.py first.")
        sys.exit(1)

    with open(SCORES_FILE) as f:
        scored_examples = json.load(f)

    with open(ARTICLES_FILE) as f:
        articles = json.load(f)

    article_map = {a.get('id', ''): a for a in articles}
    print(f"Loaded {len(scored_examples)} scored examples, {len(articles)} articles")

    # ── 1. Filter sparse reasoning ──
    valid = []
    sparse_rejects = 0
    for s in scored_examples:
        if s.get('think_tokens', 0) < 20:
            sparse_rejects += 1
            continue
        if s['score'] < 0 or s['score'] > 10:
            continue
        valid.append(s)

    print(f"After sparse reasoning filter: {len(valid)} (rejected {sparse_rejects})")

    # ── 2. Loss weighting ──
    for s in valid:
        band = score_to_band(s['score'])
        s['band'] = band
        s['weight'] = LOSS_WEIGHTS[band]

    band_dist = Counter(s['band'] for s in valid)
    print(f"\nScore band distribution:")
    for band in ['noise', 'tangential', 'moderate', 'high', 'critical']:
        count = band_dist.get(band, 0)
        pct = count / max(len(valid), 1) * 100
        print(f"  {band:>12}: {count:>6} ({pct:>5.1f}%) weight={LOSS_WEIGHTS[band]}")

    # ── 3. Contrastive pair extraction ──
    scored_by_article = defaultdict(list)
    for s in valid:
        scored_by_article[s['article_id']].append(s)

    contrastive_pairs = extract_contrastive_pairs(scored_by_article)
    print(f"\nContrastive pairs (score gap >= 3.0): {len(contrastive_pairs)}")

    # Save contrastive pairs
    with open(CONTRASTIVE_FILE, 'w') as f:
        json.dump(contrastive_pairs[:1000], f, indent=2)  # Save sample
    print(f"Saved contrastive pairs sample to {CONTRASTIVE_FILE}")

    if len(contrastive_pairs) < 10000:
        print(f"!! WARNING: Only {len(contrastive_pairs)} contrastive pairs — below 10,000 target")

    # ── 4. Curriculum sorting (easy → hard) ──
    # Easy = clear noise (0-2) or clear critical (8-10)
    # Hard = ambiguous mid-range (3-7)
    def difficulty(score):
        if score <= 2.0 or score >= 8.0:
            return 0  # Easy
        elif score <= 3.0 or score >= 7.0:
            return 1  # Medium
        else:
            return 2  # Hard (ambiguous)

    valid.sort(key=lambda s: (difficulty(s['score']), random.random()))
    print(f"Curriculum sorted: easy → hard")

    # ── 5. Train/eval split (90/10 stratified) ──
    by_band_profile = defaultdict(list)
    for s in valid:
        key = f"{s['band']}_{s['profile_id']}"
        by_band_profile[key].append(s)

    train_examples = []
    eval_examples = []

    for key, examples in by_band_profile.items():
        random.shuffle(examples)
        n_eval = max(1, int(len(examples) * 0.1))
        eval_examples.extend(examples[:n_eval])
        train_examples.extend(examples[n_eval:])

    print(f"\nTrain: {len(train_examples)}, Eval: {len(eval_examples)}")

    # ── 6. Format for training ──
    train_jsonl = []
    eval_jsonl = []

    for s in train_examples:
        profile = PROFILE_MAP.get(s['profile_id'])
        article = article_map.get(s['article_id'])
        if not profile or not article:
            continue
        example = build_training_example(s, profile, article)
        example['sample_weight'] = s['weight']
        train_jsonl.append(example)

    for s in eval_examples:
        profile = PROFILE_MAP.get(s['profile_id'])
        article = article_map.get(s['article_id'])
        if not profile or not article:
            continue
        example = build_training_example(s, profile, article)
        eval_jsonl.append(example)

    # Save
    with open(TRAIN_FILE, 'w') as f:
        for ex in train_jsonl:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    with open(EVAL_FILE, 'w') as f:
        for ex in eval_jsonl:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"\nSaved:")
    print(f"  Training: {TRAIN_FILE} ({len(train_jsonl)} examples, {TRAIN_FILE.stat().st_size / 1024:.0f} KB)")
    print(f"  Eval: {EVAL_FILE} ({len(eval_jsonl)} examples, {EVAL_FILE.stat().st_size / 1024:.0f} KB)")

    # ── 7. Verify format ──
    # Check first training example matches V1 format
    if train_jsonl:
        first = train_jsonl[0]
        msgs = first['messages']
        assert len(msgs) == 3, f"Expected 3 messages, got {len(msgs)}"
        assert msgs[0]['role'] == 'system'
        assert msgs[1]['role'] == 'user'
        assert msgs[2]['role'] == 'assistant'
        assert msgs[2]['content'].startswith('SCORE:'), f"Assistant should start with SCORE:, got: {msgs[2]['content'][:50]}"
        print(f"\nFormat verification: PASSED")
    else:
        print(f"\n!! No training examples generated")

    # ── Report ──
    train_band = Counter(score_to_band(s['score']) for s in train_examples)
    eval_band = Counter(score_to_band(s['score']) for s in eval_examples)

    print(f"\n{'='*80}")
    print("STAGE 4 FINAL REPORT")
    print(f"{'='*80}")
    print(f"Total training examples: {len(train_jsonl)}")
    print(f"Total eval examples: {len(eval_jsonl)}")
    print(f"Total contrastive pairs: {len(contrastive_pairs)}")
    print(f"Sparse reasoning rejects: {sparse_rejects}")
    print(f"Training file size: {TRAIN_FILE.stat().st_size / 1024:.0f} KB")
    print(f"Eval file size: {EVAL_FILE.stat().st_size / 1024:.0f} KB")
    print(f"\nScore distribution (Train vs Eval):")
    for band in ['noise', 'tangential', 'moderate', 'high', 'critical']:
        t = train_band.get(band, 0)
        e = eval_band.get(band, 0)
        t_pct = t / max(len(train_examples), 1) * 100
        e_pct = e / max(len(eval_examples), 1) * 100
        print(f"  {band:>12}: Train {t:>5} ({t_pct:>5.1f}%) | Eval {e:>4} ({e_pct:>5.1f}%)")

    # Unique profiles in train and eval
    train_profiles = set(s['profile_id'] for s in train_examples)
    eval_profiles = set(s['profile_id'] for s in eval_examples)
    print(f"\nProfiles in train: {len(train_profiles)}")
    print(f"Profiles in eval: {len(eval_profiles)}")
    print(f"Profiles in both: {len(train_profiles & eval_profiles)}")

    # Save report
    report = {
        "train_examples": len(train_jsonl),
        "eval_examples": len(eval_jsonl),
        "contrastive_pairs": len(contrastive_pairs),
        "sparse_rejects": sparse_rejects,
        "train_file_kb": TRAIN_FILE.stat().st_size / 1024,
        "eval_file_kb": EVAL_FILE.stat().st_size / 1024,
        "band_distribution": {
            "train": dict(train_band),
            "eval": dict(eval_band),
        },
        "profiles_in_train": len(train_profiles),
        "profiles_in_eval": len(eval_profiles),
        "profiles_in_both": len(train_profiles & eval_profiles),
    }
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
