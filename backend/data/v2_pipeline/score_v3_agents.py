#!/usr/bin/env python3
"""
Score existing articles against V3 profiles using Claude Code agents.
Deploys many small agents, each handling 50 articles × 1 profile.
Each agent receives article data directly (no file reads needed).

Usage:
    python3 data/v2_pipeline/score_v3_agents.py --profile math_teacher_texas
    python3 data/v2_pipeline/score_v3_agents.py --all
    python3 data/v2_pipeline/score_v3_agents.py --merge   # Merge all batch files
"""

import json
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.v2_pipeline.profiles_v2 import V3_NEW_PROFILES

OUTPUT_DIR = Path(__file__).parent
ARTICLES_FILE = OUTPUT_DIR / "articles_v2.json"
SCORES_FILE = OUTPUT_DIR / "scores_v2.json"

BATCH_SIZE = 50  # Articles per agent


def build_agent_prompt(profile: dict, articles: list, batch_idx: int) -> str:
    """Build a prompt for an agent to score a batch of articles."""
    pid = profile['id']
    role = profile['role']
    location = profile['location']
    context = profile.get('context', 'N/A')
    interests = ', '.join(profile.get('interests', []))
    companies = profile.get('tracked_companies', 'N/A')
    industries = profile.get('tracked_industries', 'N/A')
    institutions = profile.get('tracked_institutions', 'N/A')

    # Format articles as compact JSON
    article_data = []
    for a in articles:
        article_data.append({
            'id': a.get('id', ''),
            'title': a.get('title', '')[:200],
            'summary': a.get('summary', '')[:400]
        })

    articles_json = json.dumps(article_data, indent=1, ensure_ascii=False)

    return f"""Score these {len(articles)} articles for relevance to this user profile. You ARE Claude Opus — your judgment is ground truth for training a smaller model.

PROFILE: {pid}
- Role: {role}
- Location: {location}
- Context: {context}
- Interests: {interests}
- Tracked Companies: {companies}
- Tracked Institutions: {institutions}
- Tracked Industries: {industries}

SCORING SCALE:
- 9-10: Directly actionable (hiring match, breakthrough in their tracked area)
- 7-8.9: Highly relevant to their field/interests
- 5-6.9: Somewhat relevant
- 3-4.9: Tangentially relevant
- 0-2.9: Not relevant / noise

ARTICLES TO SCORE:
{articles_json}

INSTRUCTIONS:
1. Read each article's title and summary
2. Think about whether it's relevant to this specific person's role, location, interests, and career
3. Output a JSON array of scores

Write the output as a JSON file at: /home/ahmad/Downloads/StratOS/StratOS1/backend/data/v2_pipeline/v3_scores_{pid}_batch{batch_idx}.json

The file should contain a JSON array like:
[
  {{"article_id": "xxx", "profile_id": "{pid}", "score": 1.0, "reason": "Brief 1-2 sentence explanation", "think_text": "2-4 sentence detailed reasoning about relevance to this profile", "think_tokens": 40}},
  ...
]

CRITICAL RULES:
- DO NOT write a keyword-matching script. Evaluate each article using your intelligence.
- Most articles will be noise (0-2). That's expected and correct.
- think_text must explain WHY this article is/isn't relevant to THIS specific person
- think_tokens = approximate word count of your think_text
- Be fast but genuine — actually consider each article's content against the profile"""


def create_batches(articles: list, batch_size: int) -> list:
    """Split articles into batches."""
    return [articles[i:i+batch_size] for i in range(0, len(articles), batch_size)]


def check_existing_scores(profile_id: str) -> set:
    """Check which articles already have scores for this profile."""
    scored_ids = set()
    for f in OUTPUT_DIR.glob(f"v3_scores_{profile_id}_batch*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
                for s in data:
                    scored_ids.add(s['article_id'])
        except (json.JSONDecodeError, KeyError):
            pass
    return scored_ids


def merge_all_v3_scores():
    """Merge all v3_scores_*.json files into scores_v2.json."""
    with open(SCORES_FILE) as f:
        existing = json.load(f)

    existing_pairs = set((s['article_id'], s['profile_id']) for s in existing)

    new_scores = []
    for f in sorted(OUTPUT_DIR.glob("v3_scores_*_batch*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
                for s in data:
                    key = (s['article_id'], s['profile_id'])
                    if key not in existing_pairs:
                        new_scores.append(s)
                        existing_pairs.add(key)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {f}: {e}")

    if new_scores:
        existing.extend(new_scores)
        # Backup first
        backup = OUTPUT_DIR / "scores_v2_pre_v3_merge.json"
        if not backup.exists():
            import shutil
            shutil.copy(SCORES_FILE, backup)
            print(f"Backed up existing scores to {backup}")

        with open(SCORES_FILE, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"Merged {len(new_scores)} new V3 scores. Total: {len(existing)}")
    else:
        print("No new V3 scores to merge.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", help="Score a single profile")
    parser.add_argument("--all", action="store_true", help="Score all V3 profiles")
    parser.add_argument("--merge", action="store_true", help="Merge V3 scores into scores_v2.json")
    parser.add_argument("--status", action="store_true", help="Show scoring progress")
    args = parser.parse_args()

    if args.merge:
        merge_all_v3_scores()
        return

    if args.status:
        with open(ARTICLES_FILE) as f:
            n_articles = len(json.load(f))
        for p in V3_NEW_PROFILES:
            scored = check_existing_scores(p['id'])
            pct = len(scored) / n_articles * 100
            status = "DONE" if len(scored) >= n_articles else f"{len(scored)}/{n_articles}"
            print(f"  {p['id']:30s}: {status} ({pct:.0f}%)")
        return

    # Load articles
    with open(ARTICLES_FILE) as f:
        articles = json.load(f)

    profiles = V3_NEW_PROFILES
    if args.profile:
        profiles = [p for p in V3_NEW_PROFILES if p['id'] == args.profile]
        if not profiles:
            print(f"Profile '{args.profile}' not found")
            sys.exit(1)

    for profile in profiles:
        pid = profile['id']
        scored_ids = check_existing_scores(pid)
        remaining = [a for a in articles if a.get('id', '') not in scored_ids]

        if not remaining:
            print(f"{pid}: Already fully scored ({len(scored_ids)}/{len(articles)})")
            continue

        batches = create_batches(remaining, BATCH_SIZE)
        print(f"\n{pid}: {len(remaining)} articles remaining, {len(batches)} batches of {BATCH_SIZE}")

        for i, batch in enumerate(batches):
            prompt = build_agent_prompt(profile, batch, i)
            print(f"  Batch {i}: {len(batch)} articles")
            print(f"  Prompt size: {len(prompt)} chars")
            # The actual agent deployment happens from the main Claude session
            # This script just prepares the prompts


if __name__ == "__main__":
    main()
