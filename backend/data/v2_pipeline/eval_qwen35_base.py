#!/usr/bin/env python3
"""
Evaluate Qwen3.5-9B (base, unfine-tuned) on the V2 holdout set.

Compares against Opus ground truth scores to determine if the base model
is already competitive with fine-tuned Qwen3-8B (V2 MAE=1.544).

Also tests Qwen3.5-0.6B as a potential noise pre-filter.

Usage:
  python3 eval_qwen35_base.py                    # Test 9B on full holdout
  python3 eval_qwen35_base.py --model qwen3.5:0.6b --samples 50  # Quick test 0.6B
"""

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

BASE = Path(__file__).parent
HOLDOUT_ARTICLES = BASE / "holdout_articles.json"
HOLDOUT_SCORES = BASE / "holdout_scores_all.json"
HOLDOUT_EVAL = BASE / "eval_holdout_v2.jsonl"

OLLAMA_URL = "http://localhost:11434/api/generate"


def parse_score(text: str):
    """Extract score from model response."""
    if not text:
        return None, None
    # Strip think blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # SCORE: X.X | REASON: ...
    m = re.search(r'SCORE:\s*([\d.]+)(?:\s*\|\s*REASON:\s*(.+))?', text, re.IGNORECASE)
    if m:
        return float(m.group(1)), (m.group(2) or '').strip()
    m = re.search(r'SCORE:\s*([\d.]+)', text)
    if m:
        return float(m.group(1)), ''
    m = re.search(r'\*?\*?Score\*?\*?[:\s]+([\d.]+)', text, re.IGNORECASE)
    if m:
        return float(m.group(1)), ''
    m = re.search(r'^\s*([\d]+\.[\d]+)\s*$', text, re.MULTILINE)
    if m:
        return float(m.group(1)), ''
    return None, None


def score_to_band(score):
    if score >= 8.5: return "critical"
    if score >= 6.5: return "high"
    if score >= 4.5: return "moderate"
    if score >= 2.5: return "tangential"
    return "noise"


def _is_student(role):
    role_lower = role.lower()
    return any(kw in role_lower for kw in [
        'student', 'freshman', 'sophomore', 'junior', 'senior year',
        'undergraduate', 'graduate student', 'phd candidate',
        'fresh graduate', 'undeclared'
    ])


def build_prompt(profile, article):
    """Build scoring prompt matching production format."""
    role = profile['role']
    location = profile['location']
    context = profile.get('context', 'Not specified')
    companies = profile.get('tracked_companies', 'None specified')
    institutions = profile.get('tracked_institutions', 'None specified')
    interests = ', '.join(profile.get('interests', [])) if isinstance(profile.get('interests'), list) else profile.get('interests', 'None specified')
    industries = profile.get('tracked_industries', 'None specified')

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
    category = article.get('category', 'general')

    user_message = f"""Score this article:
Category: {category}
Keywords: {interests}
Title: {title}
Content: {content}"""

    return system_prompt, user_message


def call_ollama(model, system, user, temperature=0.1, num_predict=256):
    """Call Ollama and return response text + timing."""
    # Build prompt using ChatML template
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"

    t0 = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model,
            "prompt": prompt,
            "raw": True,  # We're providing the full template
            "stream": False,
            "options": {
                "num_predict": num_predict,
                "temperature": temperature,
            }
        }, timeout=120)
        data = resp.json()
        elapsed = time.time() - t0
        text = data.get("response", "")
        eval_count = data.get("eval_count", 0)
        return text, elapsed, eval_count
    except Exception as e:
        return f"ERROR: {e}", time.time() - t0, 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5 base models on holdout")
    parser.add_argument("--model", default="qwen3.5:9b", help="Ollama model to test")
    parser.add_argument("--samples", type=int, default=0, help="Limit number of samples (0=all)")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output", default="", help="Output JSON file (default: auto-named)")
    args = parser.parse_args()

    # Load holdout ground truth
    holdout_scores = json.load(open(HOLDOUT_SCORES))
    holdout_articles_raw = json.load(open(HOLDOUT_ARTICLES))
    article_map = {a['id']: a for a in holdout_articles_raw}

    # Load profiles
    from profiles_v2 import ALL_PROFILES
    profile_map = {p['id']: p for p in ALL_PROFILES}

    # Build eval pairs: (article_id, profile_id, opus_score)
    eval_pairs = []
    for s in holdout_scores:
        if s['article_id'] in article_map and s['profile_id'] in profile_map:
            eval_pairs.append((s['article_id'], s['profile_id'], s['score']))

    print(f"Model: {args.model}")
    print(f"Holdout pairs: {len(eval_pairs)}")

    if args.samples > 0:
        import random
        random.seed(42)
        random.shuffle(eval_pairs)
        eval_pairs = eval_pairs[:args.samples]
        print(f"Sampling: {args.samples}")

    # Warm up model
    print("Warming up model...")
    call_ollama(args.model, "You are a test.", "Say OK.", num_predict=8)

    # Run evaluation
    results = []
    errors = []
    total_time = 0
    total_tokens = 0

    for i, (art_id, prof_id, opus_score) in enumerate(eval_pairs):
        article = article_map[art_id]
        profile = profile_map[prof_id]

        system, user = build_prompt(profile, article)
        text, elapsed, tokens = call_ollama(args.model, system, user,
                                             temperature=args.temperature)
        total_time += elapsed
        total_tokens += tokens

        model_score, reason = parse_score(text)

        result = {
            'article_id': art_id,
            'profile_id': prof_id,
            'opus_score': opus_score,
            'model_score': model_score,
            'reason': (reason or '')[:200],
            'elapsed': round(elapsed, 2),
            'tokens': tokens,
        }
        results.append(result)

        if model_score is None:
            errors.append(result)

        # Progress
        if (i + 1) % 25 == 0 or i < 5:
            valid = [r for r in results if r['model_score'] is not None]
            if valid:
                mae = sum(abs(r['opus_score'] - r['model_score']) for r in valid) / len(valid)
                rate = (i + 1) / total_time if total_time > 0 else 0
                eta = (len(eval_pairs) - i - 1) / rate / 60 if rate > 0 else 0
                print(f"  [{i+1}/{len(eval_pairs)}] MAE={mae:.3f} "
                      f"parse_fail={len(errors)} rate={rate:.1f}/s ETA={eta:.0f}min")

    # Compute metrics
    valid = [r for r in results if r['model_score'] is not None]
    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT: {args.model}")
    print(f"{'='*70}")
    print(f"Total pairs: {len(results)}")
    print(f"Valid scores: {len(valid)}")
    print(f"Parse failures: {len(errors)} ({100*len(errors)/len(results):.1f}%)")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Avg time/call: {total_time/len(results):.2f}s")
    print(f"Avg tokens/call: {total_tokens/len(results):.0f}")

    if not valid:
        print("NO VALID SCORES — model cannot follow format")
        return

    # MAE
    abs_errors = [abs(r['opus_score'] - r['model_score']) for r in valid]
    mae = sum(abs_errors) / len(abs_errors)
    median_ae = sorted(abs_errors)[len(abs_errors) // 2]
    biases = [r['model_score'] - r['opus_score'] for r in valid]
    mean_bias = sum(biases) / len(biases)

    print(f"\n--- Accuracy ---")
    print(f"MAE: {mae:.3f}  (V2 fine-tuned baseline: 1.544)")
    print(f"Median AE: {median_ae:.3f}")
    print(f"Mean bias: {mean_bias:+.3f}")

    # Direction accuracy
    correct_dir = sum(1 for r in valid
                      if (r['opus_score'] >= 5.0) == (r['model_score'] >= 5.0))
    dir_acc = correct_dir / len(valid)
    print(f"Direction accuracy (>5 vs <5): {100*dir_acc:.1f}%")

    # Within thresholds
    within_1 = sum(1 for e in abs_errors if e <= 1.0) / len(abs_errors)
    within_2 = sum(1 for e in abs_errors if e <= 2.0) / len(abs_errors)
    print(f"Within 1.0: {100*within_1:.1f}%")
    print(f"Within 2.0: {100*within_2:.1f}%")

    # Per-band analysis
    print(f"\n--- Per-Band MAE ---")
    bands = {'0-2': (0, 2.5), '2-4': (2.5, 4.5), '4-6': (4.5, 6.5), '6-8': (6.5, 8.5), '8-10': (8.5, 10.1)}
    for band_name, (lo, hi) in bands.items():
        band_results = [r for r in valid if lo <= r['opus_score'] < hi]
        if band_results:
            band_mae = sum(abs(r['opus_score'] - r['model_score']) for r in band_results) / len(band_results)
            band_bias = sum(r['model_score'] - r['opus_score'] for r in band_results) / len(band_results)
            print(f"  {band_name}: MAE={band_mae:.3f} bias={band_bias:+.3f} n={len(band_results)}")

    # Score distribution
    print(f"\n--- Model Score Distribution ---")
    model_bands = Counter(score_to_band(r['model_score']) for r in valid)
    opus_bands = Counter(score_to_band(r['opus_score']) for r in valid)
    for band in ['noise', 'tangential', 'moderate', 'high', 'critical']:
        m = model_bands.get(band, 0)
        o = opus_bands.get(band, 0)
        print(f"  {band:>12}: model={m:>4} opus={o:>4}")

    # Save results
    output_file = args.output or f"eval_{args.model.replace(':', '_').replace('.', '')}_holdout.json"
    output_path = BASE / output_file
    report = {
        'model': args.model,
        'temperature': args.temperature,
        'total_pairs': len(results),
        'valid_scores': len(valid),
        'parse_failures': len(errors),
        'mae': round(mae, 3),
        'median_ae': round(median_ae, 3),
        'mean_bias': round(mean_bias, 3),
        'direction_accuracy': round(dir_acc, 3),
        'within_1': round(within_1, 3),
        'within_2': round(within_2, 3),
        'avg_time_per_call': round(total_time / len(results), 2),
        'total_time_seconds': round(total_time),
        'results': results,
    }
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
