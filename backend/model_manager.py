#!/usr/bin/env python3
"""
StratOS Scorer Model Manager — Safe model switching with rollback support.

Usage:
    python3 model_manager.py status                        # Show current model, available models, history
    python3 model_manager.py switch <model_name>           # Switch active scorer model (with validation)
    python3 model_manager.py rollback                      # Roll back to previous model
    python3 model_manager.py history                       # Show full model change history
    python3 model_manager.py validate [model_name]         # Quick validation (20 samples) against eval set
    python3 model_manager.py register <gguf_path> <name>   # Register a new GGUF as an Ollama model

All changes are logged to data/model_history.json for auditability and rollback.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
HISTORY_PATH = BASE_DIR / "data" / "model_history.json"
EVAL_FILE = BASE_DIR / "data" / "v2_pipeline" / "eval_v2.jsonl"

OLLAMA_HOST = "http://localhost:11434"
SCORE_RE = re.compile(r"SCORE:\s*(\d+\.?\d*)")

MODELFILE_TEMPLATE = """FROM {gguf_path}

TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER num_predict 2048
PARAMETER repeat_penalty 1.1
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("model_manager")


# ---------------------------------------------------------------------------
# YAML helpers — prefer ruamel.yaml (preserves comments), fall back to PyYAML
# ---------------------------------------------------------------------------
_USE_RUAMEL = False
try:
    from ruamel.yaml import YAML as _RuamelYAML

    _ruamel = _RuamelYAML()
    _ruamel.preserve_quotes = True
    _USE_RUAMEL = True
except ImportError:
    pass

if not _USE_RUAMEL:
    import yaml


def _load_yaml(path: Path) -> Any:
    """Load a YAML file, returning the parsed data."""
    with open(path) as f:
        if _USE_RUAMEL:
            return _ruamel.load(f)
        return yaml.safe_load(f)


def _dump_yaml(data: Any, path: Path) -> None:
    """Write data back to a YAML file."""
    with open(path, "w") as f:
        if _USE_RUAMEL:
            _ruamel.dump(data, f)
        else:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def get_current_model() -> str:
    """Read scoring.model from config.yaml."""
    cfg = _load_yaml(CONFIG_PATH)
    return cfg.get("scoring", {}).get("model", "stratos-scorer-v2")


def set_current_model(model_name: str) -> None:
    """Update scoring.model in config.yaml."""
    cfg = _load_yaml(CONFIG_PATH)
    if "scoring" not in cfg:
        cfg["scoring"] = {}
    cfg["scoring"]["model"] = model_name
    _dump_yaml(cfg, CONFIG_PATH)
    logger.info(f"config.yaml updated: scoring.model = {model_name}")


def get_ollama_host() -> str:
    """Read ollama_host from config.yaml, default to localhost."""
    try:
        cfg = _load_yaml(CONFIG_PATH)
        return cfg.get("scoring", {}).get("ollama_host", OLLAMA_HOST)
    except Exception as e:
        logger.warning(f"Failed to read ollama_host from config, using default: {e}")
        return OLLAMA_HOST


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------
def _load_history() -> Dict:
    """Load model history from JSON, creating it if necessary."""
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH) as f:
                data = json.load(f)
            if isinstance(data, dict) and "entries" in data:
                return data
            logger.warning(f"Corrupt model history (unexpected structure), resetting")
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load model history: {e}")
    return {"entries": [], "current_model": get_current_model()}


def _save_history(history: Dict) -> None:
    """Save model history to JSON."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def _add_history_entry(
    action: str,
    from_model: Optional[str],
    to_model: str,
    validation: Optional[Dict] = None,
    extra: Optional[Dict] = None,
) -> None:
    """Append an entry to the history file."""
    history = _load_history()
    entry: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "from_model": from_model,
        "to_model": to_model,
    }
    if validation:
        entry["validation"] = validation
    if extra:
        entry.update(extra)
    history["entries"].append(entry)
    history["current_model"] = to_model
    _save_history(history)


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------
def list_ollama_models(host: str) -> List[str]:
    """Return list of model names registered in Ollama."""
    try:
        resp = requests.get(f"{host}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        logger.error(f"Cannot reach Ollama at {host}: {e}")
        return []


def model_exists_in_ollama(model_name: str, host: str) -> bool:
    """Check if a model (by prefix) exists in Ollama."""
    models = list_ollama_models(host)
    return any(model_name in m for m in models)


# ---------------------------------------------------------------------------
# Validation (mirrors evaluate_scorer.py logic)
# ---------------------------------------------------------------------------
def _load_eval_samples(path: Path, limit: int) -> List[dict]:
    """Load eval samples from JSONL."""
    samples: List[dict] = []
    if not path.exists():
        logger.warning(f"Eval file not found: {path}")
        return samples
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            assistant_msg = d["messages"][-1]["content"]
            m = SCORE_RE.search(assistant_msg)
            if not m:
                continue
            d["ground_truth_score"] = float(m.group(1))
            samples.append(d)
            if len(samples) >= limit:
                break
    return samples


def _score_single(messages: List[dict], model: str, host: str) -> Tuple[float, str]:
    """Score a single sample via Ollama /api/chat. Returns (score, raw_text)."""
    try:
        resp = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": messages[:2],  # system + user only
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 2048},
            },
            timeout=120,
        )
        if resp.status_code != 200:
            return -1.0, f"HTTP {resp.status_code}"
        raw = resp.json().get("message", {}).get("content", "")
        # Strip think blocks (closed and unclosed)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if "<think>" in raw and "</think>" not in raw:
            raw = re.sub(r"<think>.*$", "", raw, flags=re.DOTALL).strip()
        m = SCORE_RE.search(raw)
        if m:
            return float(m.group(1)), raw
        return -1.0, raw
    except Exception as e:
        return -1.0, str(e)


def run_validation(
    model: str, host: str, num_samples: int = 20
) -> Optional[Dict]:
    """Run a quick validation against the eval set.

    Returns dict with mae, direction_acc, samples or None on failure.
    """
    samples = _load_eval_samples(EVAL_FILE, limit=num_samples)
    if not samples:
        logger.error("No eval samples available for validation.")
        return None

    logger.info(f"Validating {model} on {len(samples)} samples...")
    results: List[Dict] = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        score, raw = _score_single(sample["messages"], model, host)
        results.append(
            {
                "predicted_score": score,
                "ground_truth_score": sample["ground_truth_score"],
            }
        )
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(samples) - i - 1) / rate if rate > 0 else 0
            logger.info(f"  [{i+1}/{len(samples)}] {rate:.1f} samples/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    valid = [r for r in results if r["predicted_score"] >= 0]
    if not valid:
        logger.error("No valid predictions obtained.")
        return None

    n = len(valid)
    abs_deltas = [abs(r["predicted_score"] - r["ground_truth_score"]) for r in valid]
    mae = sum(abs_deltas) / n
    direction_correct = sum(
        1
        for r in valid
        if (r["predicted_score"] >= 5.0) == (r["ground_truth_score"] >= 5.0)
    )
    direction_acc = direction_correct / n

    metrics = {
        "mae": round(mae, 3),
        "direction_acc": round(direction_acc, 4),
        "samples": n,
        "parse_failures": len(results) - n,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  Validation Results for {model}:")
    print(f"    Samples:            {n} ({len(results) - n} parse failures)")
    print(f"    MAE:                {metrics['mae']}")
    print(f"    Direction accuracy: {metrics['direction_acc']:.1%}")
    print(f"    Time:               {metrics['elapsed_seconds']}s\n")

    return metrics


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_status(args: argparse.Namespace) -> None:
    """Show current model, available Ollama models, and recent history."""
    host = get_ollama_host()
    current = get_current_model()
    models = list_ollama_models(host)
    history = _load_history()

    print("\n" + "=" * 60)
    print("STRATOS SCORER MODEL STATUS")
    print("=" * 60)
    print(f"  Active model (config.yaml):  {current}")
    print(f"  History model (tracked):     {history.get('current_model', 'N/A')}")
    print(f"  Ollama host:                 {host}")

    print(f"\n  Available Ollama models ({len(models)}):")
    if models:
        for m in sorted(models):
            marker = " <-- active" if current in m else ""
            print(f"    - {m}{marker}")
    else:
        print("    (none found — is Ollama running?)")

    entries = history.get("entries", [])
    if entries:
        print(f"\n  Recent history (last 5):")
        for e in entries[-5:]:
            ts = e.get("timestamp", "?")[:19]
            action = e.get("action", "?")
            to_m = e.get("to_model", "?")
            from_m = e.get("from_model", "?")
            val = e.get("validation", {})
            val_str = ""
            if val:
                val_str = f" | MAE={val.get('mae','?')} dir={val.get('direction_acc','?')}"
            print(f"    [{ts}] {action}: {from_m} -> {to_m}{val_str}")
    else:
        print("\n  No model change history yet.")
    print("=" * 60 + "\n")


def cmd_switch(args: argparse.Namespace) -> None:
    """Switch the active scorer model with validation."""
    host = get_ollama_host()
    new_model = args.model_name
    current = get_current_model()

    if new_model == current:
        print(f"Model '{new_model}' is already the active model. Nothing to do.")
        return

    # Verify new model exists in Ollama
    if not model_exists_in_ollama(new_model, host):
        available = list_ollama_models(host)
        print(f"ERROR: Model '{new_model}' not found in Ollama.")
        if available:
            print(f"  Available: {', '.join(available)}")
        sys.exit(1)

    print(f"\nSwitching scorer model: {current} -> {new_model}")

    # Run quick validation on the new model (10 samples)
    print("\nRunning quick validation (10 samples)...")
    validation = run_validation(new_model, host, num_samples=10)

    # Update config
    set_current_model(new_model)

    # Record in history
    _add_history_entry(
        action="switch",
        from_model=current,
        to_model=new_model,
        validation=validation,
    )

    print("=" * 60)
    print(f"  SWITCH COMPLETE")
    print(f"  Before: {current}")
    print(f"  After:  {new_model}")
    if validation:
        print(f"  MAE:    {validation['mae']}  |  Direction accuracy: {validation['direction_acc']:.1%}")
    print(f"\n  To roll back: python3 model_manager.py rollback")
    print("=" * 60 + "\n")


def cmd_rollback(args: argparse.Namespace) -> None:
    """Roll back to the previous model."""
    history = _load_history()
    entries = history.get("entries", [])

    if not entries:
        print("ERROR: No history entries found. Nothing to roll back to.")
        sys.exit(1)

    # Find the last entry that has a from_model we can revert to
    last_entry = entries[-1]
    previous_model = last_entry.get("from_model")
    current = get_current_model()

    if not previous_model:
        print("ERROR: Last history entry has no from_model. Cannot roll back.")
        sys.exit(1)

    if previous_model == current:
        print(f"Already on the previous model '{previous_model}'. Nothing to do.")
        return

    host = get_ollama_host()

    # Verify previous model still exists in Ollama
    if not model_exists_in_ollama(previous_model, host):
        print(f"WARNING: Previous model '{previous_model}' not found in Ollama.")
        print("Rolling back config anyway (model may need to be re-registered).")

    print(f"\nRolling back: {current} -> {previous_model}")

    # Update config
    set_current_model(previous_model)

    # Record rollback in history
    _add_history_entry(
        action="rollback",
        from_model=current,
        to_model=previous_model,
    )

    print("=" * 60)
    print(f"  ROLLBACK COMPLETE")
    print(f"  Reverted: {current} -> {previous_model}")
    print("=" * 60 + "\n")


def cmd_history(args: argparse.Namespace) -> None:
    """Show full model change history."""
    history = _load_history()
    entries = history.get("entries", [])

    print("\n" + "=" * 60)
    print("STRATOS SCORER MODEL HISTORY")
    print("=" * 60)
    print(f"  Current model: {history.get('current_model', 'N/A')}")
    print(f"  Total entries: {len(entries)}\n")

    if not entries:
        print("  (no entries)")
    else:
        for i, e in enumerate(entries, 1):
            ts = e.get("timestamp", "?")[:19]
            action = e.get("action", "?").upper()
            from_m = e.get("from_model", "N/A")
            to_m = e.get("to_model", "?")
            print(f"  {i:3d}. [{ts}] {action}")
            print(f"       {from_m} -> {to_m}")
            val = e.get("validation")
            if val:
                print(
                    f"       Validation: MAE={val.get('mae','?')} "
                    f"direction_acc={val.get('direction_acc','?')} "
                    f"samples={val.get('samples','?')}"
                )
            gguf = e.get("gguf_path")
            if gguf:
                print(f"       GGUF: {gguf}")
            print()

    print("=" * 60 + "\n")


def cmd_validate(args: argparse.Namespace) -> None:
    """Quick validation of a model against the eval set."""
    host = get_ollama_host()
    model = args.model_name if args.model_name else get_current_model()

    if not model_exists_in_ollama(model, host):
        available = list_ollama_models(host)
        print(f"ERROR: Model '{model}' not found in Ollama.")
        if available:
            print(f"  Available: {', '.join(available)}")
        sys.exit(1)

    print(f"\nValidating model: {model} (20 samples)")
    validation = run_validation(model, host, num_samples=20)

    if validation:
        if validation["mae"] > 0.6:
            print("  Assessment: MAE > 0.6 — model may need retraining.")
        elif validation["direction_acc"] < 0.95:
            print("  Assessment: Direction accuracy < 95% — review recommended.")
        else:
            print("  Assessment: Model performing well.")
    else:
        print("  Validation failed — check Ollama and eval file.")
        sys.exit(1)


def cmd_register(args: argparse.Namespace) -> None:
    """Register a new GGUF file as an Ollama model."""
    gguf_path = Path(args.gguf_path).resolve()
    model_name = args.model_name
    host = get_ollama_host()

    if not gguf_path.exists():
        print(f"ERROR: GGUF file not found: {gguf_path}")
        sys.exit(1)

    if not gguf_path.suffix.lower() == ".gguf":
        print(f"WARNING: File does not have .gguf extension: {gguf_path}")

    print(f"\nRegistering GGUF as Ollama model:")
    print(f"  GGUF:  {gguf_path}")
    print(f"  Name:  {model_name}")

    # Create Modelfile
    modelfile_content = MODELFILE_TEMPLATE.format(gguf_path=str(gguf_path))

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Modelfile", delete=False, dir=str(BASE_DIR)
    ) as tmp:
        tmp.write(modelfile_content)
        modelfile_path = tmp.name

    try:
        # Run ollama create
        print(f"\n  Running: ollama create {model_name} -f {modelfile_path}")
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"ERROR: ollama create failed:\n{result.stderr}")
            sys.exit(1)
        print(f"  {result.stdout.strip()}")
    finally:
        # Clean up temp Modelfile
        try:
            os.unlink(modelfile_path)
        except OSError as e:
            logger.debug(f"Failed to clean up temp Modelfile {modelfile_path}: {e}")

    # Verify model is now available
    if not model_exists_in_ollama(model_name, host):
        print(f"WARNING: Model '{model_name}' not found after registration. "
              "It may still be loading.")

    # Run quick validation
    print("\nRunning quick validation...")
    validation = run_validation(model_name, host, num_samples=10)

    # Record in history
    current = get_current_model()
    _add_history_entry(
        action="register",
        from_model=current,
        to_model=model_name,
        validation=validation,
        extra={"gguf_path": str(gguf_path)},
    )

    print("=" * 60)
    print(f"  REGISTRATION COMPLETE: {model_name}")
    if validation:
        print(f"  MAE: {validation['mae']}  |  Direction accuracy: {validation['direction_acc']:.1%}")
    print(f"\n  To activate: python3 model_manager.py switch {model_name}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="StratOS Scorer Model Manager — safe model switching with rollback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 model_manager.py status
  python3 model_manager.py switch stratos-scorer-v3
  python3 model_manager.py rollback
  python3 model_manager.py history
  python3 model_manager.py validate stratos-scorer-v2
  python3 model_manager.py register /path/to/model.gguf stratos-scorer-v3
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    subparsers.add_parser("status", help="Show current model, available models, history")

    # switch
    sp_switch = subparsers.add_parser("switch", help="Switch the active scorer model")
    sp_switch.add_argument("model_name", help="Name of the Ollama model to switch to")

    # rollback
    subparsers.add_parser("rollback", help="Roll back to the previous model")

    # history
    subparsers.add_parser("history", help="Show full model change history")

    # validate
    sp_validate = subparsers.add_parser("validate", help="Quick validation (20 samples)")
    sp_validate.add_argument("model_name", nargs="?", default=None, help="Model to validate (default: current)")

    # register
    sp_register = subparsers.add_parser("register", help="Register a new GGUF as an Ollama model")
    sp_register.add_argument("gguf_path", help="Path to the GGUF file")
    sp_register.add_argument("model_name", help="Name to register the model as")

    args = parser.parse_args()

    commands = {
        "status": cmd_status,
        "switch": cmd_switch,
        "rollback": cmd_rollback,
        "history": cmd_history,
        "validate": cmd_validate,
        "register": cmd_register,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
