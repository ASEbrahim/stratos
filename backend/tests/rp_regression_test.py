#!/usr/bin/env python3
"""
RP Regression Test — V2 Tier 1 Automated Gate

Runs 6 canonical cards × 7 prompts = 42 responses.
Scores 8 dimensions (150 points max).
Compares against baseline JSON if it exists.
Exit code 0 = pass, exit code 1 = regression.

Usage:
    python3 tests/rp_regression_test.py                    # Run test
    python3 tests/rp_regression_test.py --save-baseline    # Run + save as new baseline
    python3 tests/rp_regression_test.py --quick            # 3 cards × 4 prompts (fast check)
"""

import sqlite3
import requests
import json
import time
import re
import uuid
import sys
import os
import argparse
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

API_BASE = "http://localhost:8080"
BASELINE_PATH = Path(__file__).parent.parent / "tests" / "rp_baseline.json"
DB_PATH = Path(__file__).parent.parent / "strat_os.db"

CANONICAL_CARDS = [
    "Reny",              # shy
    "Raven Blackwood",   # confident
    "Kael Stormborn",    # tough
    "Dr. Sable Vex",     # clinical
    "Nurse Sakura",      # sweet
    "Mira Chen",         # default/curious
]

QUICK_CARDS = ["Reny", "Kael Stormborn", "Mira Chen"]

PROMPTS = [
    ("okay",                    "boring",    1),
    ("Hey",                     "greeting",  1),
    ("Tell me about yourself",  "question",  4),
    ("*moves closer*",         "action",    2),
    ("I like you",              "emotional", 3),
    ("What are you afraid of?", "deep",      5),
    ("*touches your hand*",     "contact",   3),
]

QUICK_PROMPTS = PROMPTS[:4]  # boring, greeting, question, action

STOP_WORDS = set(
    "the a an is are was were i you me my your it this that to of in for on "
    "and or but not do does did have has had be been will would could should "
    "can may might with at by from up out so if just like what how who when "
    "where why".split()
)

INITIATIVE_PATTERNS = [
    r'\*(?:reaches|grabs|pulls|pushes|kisses|hugs|touches|strokes|caresses)',
    r'\*(?:places|puts|runs).*(?:hand|finger|palm).*(?:on|across|through).*(?:your)',
    r'\*(?:leans in|moves closer|closes the distance|bridges the gap)',
    r'\*(?:cups|cradles|tilts).*(?:face|chin|cheek)',
]

GESTURE_LIST = [
    "runs hand through", "fidgets with", "tucks hair", "bites lip",
    "looks away", "shifts weight", "scratches", "rolls eyes",
    "crosses arms", "leans against", "raises eyebrow", "tilts head",
    "lets out a breath", "narrows eyes", "smirks", "sighs",
]

GAP_SECONDS = 8
PASS_THRESHOLD = 130
PER_CARD_MIN_PCT = 0.60


# ═══════════════════════════════════════════════════════════════
# Measurement functions
# ═══════════════════════════════════════════════════════════════

def measure_echo(user_msg: str, ai_response: str) -> float:
    user_words = set(user_msg.lower().split()) - STOP_WORDS
    ai_words = set(ai_response.lower().split()) - STOP_WORDS
    if not user_words:
        return 0.0
    return len(user_words & ai_words) / len(user_words)


def detect_format(response: str) -> str:
    s = response.strip()
    if s.startswith('*'):
        return 'action'
    elif s and s[0] in ('"', "'", '\u201c', '\u2018'):
        return 'dialogue'
    return 'narration'


def check_initiative(response: str, user_msg: str) -> bool:
    if '*' in user_msg:
        return False  # user initiated physical action — AI is reacting
    for p in INITIATIVE_PATTERNS:
        if re.search(p, response, re.IGNORECASE):
            return True
    return False


def score_length_ok(user_word_count: int, response_chars: int) -> bool:
    if user_word_count <= 3:
        return response_chars <= 200
    elif user_word_count <= 5:
        return 50 <= response_chars <= 300
    elif user_word_count <= 15:
        return 80 <= response_chars <= 450
    return 100 <= response_chars <= 600


def measure_scenario_refs(response: str, scenario: str) -> int:
    if not scenario:
        return -1  # no scenario on card
    scenario_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', scenario)) - STOP_WORDS
    return sum(1 for w in scenario_words if w in response.lower())


def check_qa_answered(response: str) -> bool:
    has_self = any(w in response.lower() for w in ['i am', "i'm", 'i was', 'my ', ' me '])
    return has_self and len(response.split()) > 10


# ═══════════════════════════════════════════════════════════════
# Scoring functions (per card)
# ═══════════════════════════════════════════════════════════════

def score_card(echoes, formats, initiatives, lengths_ok, scenario_refs, qa_results, times, responses):
    """Score a single card across all dimensions. Returns dict with per-dimension scores."""

    # D1: Echo (0-25)
    avg_echo = sum(echoes) / len(echoes) if echoes else 0
    if avg_echo <= 0.05:   echo_s = 25
    elif avg_echo <= 0.10: echo_s = 20
    elif avg_echo <= 0.15: echo_s = 15
    elif avg_echo <= 0.25: echo_s = 10
    elif avg_echo <= 0.50: echo_s = 5
    else:                  echo_s = 0

    # D2: Format variety (0-25)
    fd = {}
    for f in formats:
        fd[f] = fd.get(f, 0) + 1
    n_formats = len(fd)
    total_f = len(formats)
    max_pct = max(fd.values()) / total_f if total_f else 1
    if n_formats >= 3 and max_pct <= 0.60:   fmt_s = 25
    elif n_formats >= 3:                      fmt_s = 20
    elif n_formats >= 2:                      fmt_s = 15
    elif n_formats == 1:                      fmt_s = 5
    else:                                     fmt_s = 0

    # D3: Physical initiative (0-15)
    viols = sum(1 for v in initiatives if v)
    if viols == 0:   init_s = 15
    elif viols == 1: init_s = 10
    elif viols == 2: init_s = 5
    else:            init_s = 0

    # D4: Length appropriateness (0-20)
    len_pct = sum(1 for l in lengths_ok if l) / len(lengths_ok) if lengths_ok else 0
    if len_pct >= 0.90:   len_s = 20
    elif len_pct >= 0.75: len_s = 15
    elif len_pct >= 0.60: len_s = 10
    elif len_pct >= 0.40: len_s = 5
    else:                 len_s = 0

    # D5: Scenario integration (0-20)
    valid_refs = [r for r in scenario_refs if r >= 0]
    if valid_refs:
        avg_scn = sum(valid_refs) / len(valid_refs)
        if avg_scn >= 2:     scn_s = 20
        elif avg_scn >= 1:   scn_s = 15
        elif avg_scn >= 0.5: scn_s = 10
        else:                scn_s = 5
    else:
        scn_s = 10  # no scenario on card — neutral

    # D6: QA (0-20)
    valid_qa = [q for q in qa_results if q is not None]
    if valid_qa:
        qa_pct = sum(1 for q in valid_qa if q) / len(valid_qa)
        if qa_pct >= 1.0:   qa_s = 20
        elif qa_pct >= 0.75: qa_s = 15
        elif qa_pct >= 0.50: qa_s = 10
        else:                qa_s = 5
    else:
        qa_s = 10

    # D7: Response time (0-10)
    avg_time = sum(times) / len(times) if times else 99
    if avg_time < 10:   time_s = 10
    elif avg_time < 20: time_s = 8
    elif avg_time < 30: time_s = 5
    else:               time_s = 0

    # D8: Conversation repetition (0-15)
    all_text = " ".join(responses).lower()
    overused = sum(1 for g in GESTURE_LIST if all_text.count(g) >= 3)
    openings = [r.strip().split()[:3] for r in responses if r.strip()]
    unique_pct = len(set(tuple(o) for o in openings)) / len(openings) if openings else 0
    if overused == 0 and unique_pct > 0.80:   rep_s = 15
    elif overused == 0 or unique_pct > 0.80:  rep_s = 12
    elif overused <= 2:                        rep_s = 8
    else:                                      rep_s = 3

    total = echo_s + fmt_s + init_s + len_s + scn_s + qa_s + time_s + rep_s
    fmt_str = "/".join(f"{k[0]}:{v}" for k, v in sorted(fd.items()))

    return {
        "echo": echo_s, "format": fmt_s, "initiative": init_s,
        "length": len_s, "scenario": scn_s, "qa": qa_s,
        "time": time_s, "repetition": rep_s, "total": total,
        "pct": total / 150 * 100,
        "avg_echo": avg_echo, "avg_time": avg_time,
        "format_dist": fmt_str, "violations": viols,
        "len_in_range_pct": len_pct,
    }


# ═══════════════════════════════════════════════════════════════
# API interaction
# ═══════════════════════════════════════════════════════════════

def send_message(session_id: str, card_id: int, content: str) -> tuple:
    """Send a message via the RP chat API. Returns (response_text, elapsed_seconds)."""
    t0 = time.time()
    try:
        r = requests.post(f"{API_BASE}/api/rp/chat", json={
            "session_id": session_id, "branch_id": "main",
            "content": content, "character_card_id": card_id,
            "persona": "roleplay",
        }, headers={"X-Device-Id": "test", "X-Auth-Token": "test"},
           timeout=90, stream=True)

        full = ""
        for line in r.iter_lines():
            if not line:
                continue
            ls = line.decode('utf-8', errors='replace')
            if ls.startswith('data: '):
                try:
                    d = json.loads(ls[6:])
                    if d.get('token'):
                        full += d['token']
                    if d.get('done'):
                        break
                except json.JSONDecodeError:
                    continue
        return full.strip(), time.time() - t0
    except Exception as e:
        return "", time.time() - t0


# ═══════════════════════════════════════════════════════════════
# Main test runner
# ═══════════════════════════════════════════════════════════════

def run_test(card_names: list, prompts: list, save_baseline: bool = False):
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Resolve card names to IDs
    cards = []
    for name in card_names:
        row = conn.execute(
            "SELECT id, name, personality, scenario, content_rating FROM character_cards WHERE name = ? LIMIT 1",
            (name,)
        ).fetchone()
        if row:
            cards.append(dict(row))
        else:
            print(f"  WARNING: Card '{name}' not found in DB — skipping")

    if not cards:
        print("ERROR: No cards found. Ensure the database has the canonical cards.")
        sys.exit(1)

    total_responses = len(cards) * len(prompts)
    print(f"\nRP Regression Test — {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 70}")
    print(f"Cards: {len(cards)} | Prompts: {len(prompts)} | Total: {total_responses} responses")
    print(f"Config: Q8_0 | num_ctx=16384 | min_p=0.05 | top_k=0 | pp=0.3")
    print(f"Pass threshold: {PASS_THRESHOLD}/150 | Per-card minimum: {PER_CARD_MIN_PCT:.0%}")
    print()

    all_card_scores = {}

    for ci, card in enumerate(cards):
        sid = f"regtest-{uuid.uuid4().hex[:8]}"
        card_name = card['name']
        scenario = card.get('scenario', '') or ''

        echoes, formats, initiatives, lengths_ok = [], [], [], []
        scenario_refs, qa_results, times, responses = [], [], [], []

        sys.stdout.write(f"  [{ci+1}/{len(cards)}] {card_name:25s} ")
        sys.stdout.flush()

        for prompt, ptype, user_words in prompts:
            resp, elapsed = send_message(sid, card['id'], prompt)
            if not resp:
                continue

            echoes.append(measure_echo(prompt, resp))
            formats.append(detect_format(resp))
            initiatives.append(check_initiative(resp, prompt))
            lengths_ok.append(score_length_ok(user_words, len(resp)))
            scenario_refs.append(measure_scenario_refs(resp, scenario))
            times.append(elapsed)
            responses.append(resp)

            if ptype in ('question', 'deep'):
                qa_results.append(check_qa_answered(resp))
            else:
                qa_results.append(None)

            time.sleep(GAP_SECONDS)

        if not responses:
            print("NO RESPONSES")
            continue

        scores = score_card(echoes, formats, initiatives, lengths_ok,
                           scenario_refs, qa_results, times, responses)
        all_card_scores[card_name] = scores

        status = "✓" if scores['pct'] >= PER_CARD_MIN_PCT * 100 else "✗"
        print(f"{status} {scores['total']:3d}/150 ({scores['pct']:.0f}%) | {scores['format_dist']}")

    # ── Aggregate ──
    if not all_card_scores:
        print("\nERROR: No cards produced results.")
        sys.exit(1)

    avg_total = sum(s['total'] for s in all_card_scores.values()) / len(all_card_scores)
    worst = min(all_card_scores.items(), key=lambda x: x[1]['total'])
    best = max(all_card_scores.items(), key=lambda x: x[1]['total'])
    below_min = {n: s for n, s in all_card_scores.items() if s['pct'] < PER_CARD_MIN_PCT * 100}

    print(f"\n{'─' * 70}")
    print(f"  {'Dimension':<22s}  {'Echo':>5s}  {'Fmt':>4s}  {'Init':>4s}  {'Len':>4s}  {'Scn':>4s}  {'QA':>4s}  {'Time':>4s}  {'Rep':>4s}  {'Total':>6s}")
    print(f"  {'─' * 22}  {'─'*5}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*6}")
    for name, s in sorted(all_card_scores.items(), key=lambda x: -x[1]['total']):
        print(f"  {name:<22s}  {s['echo']:>4d}  {s['format']:>4d}  {s['initiative']:>4d}  "
              f"{s['length']:>4d}  {s['scenario']:>4d}  {s['qa']:>4d}  {s['time']:>4d}  "
              f"{s['repetition']:>4d}  {s['total']:>3d}/150")

    # Dimension averages
    dims = ['echo', 'format', 'initiative', 'length', 'scenario', 'qa', 'time', 'repetition']
    dim_avgs = {d: sum(s[d] for s in all_card_scores.values()) / len(all_card_scores) for d in dims}
    print(f"  {'AVERAGE':<22s}  " + "  ".join(f"{dim_avgs[d]:>4.0f}" for d in dims) + f"  {avg_total:>5.0f}/150")

    print(f"\n  Average: {avg_total:.0f}/150 ({avg_total/150*100:.0f}%)")
    print(f"  Best:    {best[0]} ({best[1]['total']}/150)")
    print(f"  Worst:   {worst[0]} ({worst[1]['total']}/150)")
    if below_min:
        print(f"  ⚠ Below {PER_CARD_MIN_PCT:.0%}: {', '.join(below_min.keys())}")

    # ── Compare against baseline ──
    regression_detected = False

    if BASELINE_PATH.exists() and not save_baseline:
        baseline = json.loads(BASELINE_PATH.read_text())
        bl_scores = baseline.get('scores', {})

        print(f"\n{'─' * 70}")
        print(f"  DELTA vs BASELINE ({baseline.get('date', '?')})")
        print(f"  {'Card':<22s}  {'Baseline':>8s}  {'Current':>8s}  {'Delta':>7s}  {'Status':>8s}")
        print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*8}")

        for name, current in sorted(all_card_scores.items()):
            bl = bl_scores.get(name, {})
            bl_total = bl.get('total', 0)
            delta = current['total'] - bl_total
            if bl_total > 0 and delta < -8:  # >8 point regression
                status = "✗ REGR"
                regression_detected = True
            elif delta > 0:
                status = "↑ improved"
            else:
                status = "✓ OK"
            print(f"  {name:<22s}  {bl_total:>5d}/150  {current['total']:>5d}/150  {delta:>+5d}    {status}")

        bl_avg = baseline.get('average', 0)
        delta_avg = avg_total - bl_avg
        print(f"\n  Average: {bl_avg:.0f} → {avg_total:.0f} ({delta_avg:+.0f})")

    # ── Final verdict ──
    passed = avg_total >= PASS_THRESHOLD and not below_min and not regression_detected

    print(f"\n{'═' * 70}")
    if passed:
        print(f"  ✓ PASS — {avg_total:.0f}/150 (threshold: {PASS_THRESHOLD})")
    else:
        reasons = []
        if avg_total < PASS_THRESHOLD:
            reasons.append(f"avg {avg_total:.0f} < {PASS_THRESHOLD}")
        if below_min:
            reasons.append(f"cards below {PER_CARD_MIN_PCT:.0%}: {', '.join(below_min.keys())}")
        if regression_detected:
            reasons.append("regression vs baseline")
        print(f"  ✗ FAIL — {', '.join(reasons)}")
    print(f"{'═' * 70}\n")

    # ── Save baseline ──
    if save_baseline:
        baseline_data = {
            "date": time.strftime('%Y-%m-%d'),
            "model": "huihui_ai/qwen3.5-abliterated:9b-q8_0",
            "config": {
                "temperature": 0.85, "min_p": 0.05, "top_k": 0,
                "presence_penalty": 0.3, "num_ctx": 16384,
                "system_prompt_tokens": 220,
            },
            "average": avg_total,
            "scores": all_card_scores,
            "thresholds": {
                "pass": PASS_THRESHOLD,
                "per_card_min_pct": PER_CARD_MIN_PCT,
            },
        }
        BASELINE_PATH.write_text(json.dumps(baseline_data, indent=2))
        print(f"  Baseline saved to {BASELINE_PATH}")

    conn.close()
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RP Regression Test — V2 Tier 1')
    parser.add_argument('--save-baseline', action='store_true',
                        help='Save results as new baseline')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 3 cards × 4 prompts')
    args = parser.parse_args()

    card_set = QUICK_CARDS if args.quick else CANONICAL_CARDS
    prompt_set = QUICK_PROMPTS if args.quick else PROMPTS
    run_test(card_set, prompt_set, save_baseline=args.save_baseline)
