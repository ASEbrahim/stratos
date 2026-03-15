"""
RP Model Evaluation Harness — Automated prompt engineering loop
Runs tests against Ollama models, scores outputs, tracks iterations.
"""

import json
import re
import sys
import time
from pathlib import Path
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OUTPUT_DIR = Path(__file__).parent.parent / "eval_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def strip_think(text):
    """Remove <think>...</think> blocks from output."""
    clean = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    if '</think>' in clean:
        clean = clean.split('</think>')[-1].strip()
    return clean


def run_single_test(model, system_prompt, character, messages, timeout=180):
    """Run a single test and return the cleaned output."""
    system = f"{system_prompt}\n\n{character}"
    msgs = [{"role": "system", "content": system}] + messages
    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": model, "messages": msgs, "stream": False,
            "think": False,
            "options": {"temperature": 0.8, "top_p": 0.95, "num_predict": 2048, "num_ctx": 8192}
        }, timeout=timeout)
        data = resp.json()
        raw = data.get("message", {}).get("content", "")
        clean = strip_think(raw)
        elapsed = round(time.time() - start, 1)
        return {
            "output": clean,
            "elapsed": elapsed,
            "tokens": data.get("eval_count", 0),
            "empty": len(clean.strip()) == 0,
            "had_think": '<think>' in raw,
        }
    except Exception as e:
        return {
            "output": f"ERROR: {e}",
            "elapsed": round(time.time() - start, 1),
            "tokens": 0,
            "empty": True,
            "had_think": False,
        }


def run_test_suite(model, system_prompt, tests, label=""):
    """Run a full test suite and return results."""
    print(f"\n{'='*70}")
    print(f"Running {len(tests)} tests on {model} {label}")
    print(f"{'='*70}")

    results = []
    for i, test in enumerate(tests):
        print(f"  [{test['id']}] {test['name']} ({i+1}/{len(tests)})...", end=" ", flush=True)
        r = run_single_test(model, system_prompt, test["character"], test["messages"])
        r["id"] = test["id"]
        r["name"] = test["name"]
        r["eval_criteria"] = test["evaluate"]
        results.append(r)
        status = "EMPTY" if r["empty"] else f"{len(r['output'])} chars"
        think = " [THINK!]" if r["had_think"] else ""
        print(f"{r['elapsed']}s | {r['tokens']}tk | {status}{think}")

    return results


def save_results(results, filename):
    """Save results to JSON."""
    out = OUTPUT_DIR / filename
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")
    return out
