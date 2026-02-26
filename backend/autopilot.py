#!/usr/bin/env python3
"""
STRAT_OS — Autopilot: Fully Autonomous Self-Improvement Loop
══════════════════════════════════════════════════════════════
Each cycle:
  1. Picks a professional profile
  2. Calls /api/suggest-context → AI-suggested tracking context
  3. Calls /api/generate-profile → generates categories + tickers
  4. Saves config → runs scan with DIVERSE data
  5. Sends to Claude Opus for re-scoring (distillation)
  6. Restores original profile
  7. Every N cycles → LoRA fine-tune

Usage:
    python3 autopilot.py                    # Run forever
    python3 autopilot.py --cycles 10        # 10 cycles
    python3 autopilot.py --dry-run          # Preview
    python3 autopilot.py --budget 5.00      # $5 limit
"""

import argparse, copy, json, logging, os, random, re, sqlite3
import subprocess, sys, time, yaml, requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / 'data' / 'autopilot.log', mode='a'),
    ]
)
logger = logging.getLogger("AUTOPILOT")

API_BASE = "http://localhost:8080"

PROFILE_TEMPLATES = [
    # ── Kuwait (core) ──
    {"id": "cpeg_student_kw", "role": "Computer Engineering student at American University of Kuwait", "location": "Kuwait"},
    {"id": "chemeng_student_kw", "role": "Chemical Engineering student at Kuwait University", "location": "Kuwait"},
    {"id": "petrol_eng_kw", "role": "Petroleum Engineering student at Kuwait University", "location": "Kuwait"},
    {"id": "eee_student_kw", "role": "Electrical Engineering student at AUK", "location": "Kuwait"},
    {"id": "civil_eng_kw", "role": "Civil Engineering student at AUK", "location": "Kuwait"},
    {"id": "finance_student_kw", "role": "Finance & Accounting student at GUST Kuwait", "location": "Kuwait"},
    {"id": "medical_student_kw", "role": "Medical student at Kuwait University Faculty of Medicine", "location": "Kuwait"},
    {"id": "cybersecurity_kw", "role": "Cybersecurity analyst at a Kuwaiti bank", "location": "Kuwait"},
    {"id": "geophysicist_koc_kw", "role": "Senior geophysicist at KOC's Geological & Geophysical Solutions (acquisition & processing), also handles legal tenders", "location": "Kuwait"},

    # ── GCC specific countries (high signal — different domains & geographies) ──
    {"id": "meche_grad_sa", "role": "Mechanical Engineering fresh graduate seeking NEOM/Aramco roles", "location": "Saudi Arabia"},
    {"id": "data_scientist_dubai", "role": "Data Scientist at a Dubai fintech startup", "location": "Dubai, UAE"},
    {"id": "enviro_eng_oman", "role": "Environmental Engineering graduate (waste and water treatment)", "location": "Muscat, Oman"},
    {"id": "supply_chain_bahrain", "role": "Supply chain analyst at a logistics company", "location": "Manama, Bahrain"},

    # ── High-contrast profiles (maximize correction signal) ──
    {"id": "quant_finance_kw", "role": "Quantitative finance analyst (algorithmic trading and risk)", "location": "Kuwait"},
    {"id": "stem_teacher_kw", "role": "STEM education instructor at a Kuwait secondary school", "location": "Kuwait"},
    {"id": "biotech_researcher_sa", "role": "Biotech researcher at KAUST studying gene therapy", "location": "Jeddah, Saudi Arabia"},
    {"id": "architect_qatar", "role": "Architect working on smart city infrastructure projects", "location": "Doha, Qatar"},
]


def get_unused_profiles(db_path, all_profiles):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT note FROM user_feedback WHERE note LIKE 'Autopilot:%' AND created_at > datetime('now', '-7 days')")
        recent = {row[0].replace('Autopilot: ', '').split(' ')[0] for row in cursor.fetchall()}
        conn.close()
        unused = [p for p in all_profiles if p['id'] not in recent]
        return unused if unused else all_profiles
    except Exception:
        return all_profiles


# ═══════════════════════════════════════════════════════════════
# Opus Context Generation (fills in questionnaire via Claude API)
# ═══════════════════════════════════════════════════════════════

OPUS_QUESTIONNAIRE_SYSTEM = """You are filling in a professional profile questionnaire for a strategic intelligence dashboard called STRAT_OS.
Given a person's role and location, answer the 6 questions below as if you ARE that person — realistic, specific, grounded in their actual profession.

CRITICAL RULES:
- Be SPECIFIC to their exact profession. A geophysicist mentions seismic tools, not web frameworks. A software engineer mentions GitHub, not drilling rigs.
- For Kuwait roles, know: K-Sector = KOC, KNPC, KIPIC, Equate, PIC; Banks = NBK, Boubyan, KFH, Warba; Service companies = SLB, Halliburton, Baker Hughes, CGG
- For investment assets, use exact Yahoo Finance symbols: NVDA, GC=F (gold), CL=F (oil), BTC-USD, SI=F (silver), etc.
- Keep each answer to 1-2 short sentences. No fluff.

Respond with ONLY the 6 numbered answers, nothing else:
1. [student or professional status]
2. [companies of interest]
3. [investment assets with ticker symbols]
4. [skills/certifications]
5. [main goals]
6. [specific topics to track]"""

OPUS_API_URL = "https://api.anthropic.com/v1/messages"
OPUS_API_VERSION = "2023-06-01"
OPUS_MODEL = "claude-opus-4-5-20251101"


def opus_generate_context(role, location, api_key):
    """Use Claude Opus to fill in questionnaire answers, then combine into context string.
    
    This produces much higher quality context than the local Qwen model's /api/suggest-context,
    and costs <$0.01 per call (tiny prompt, ~200 output tokens).
    """
    import urllib.request
    import urllib.error

    prompt = f"Role: {role}\nLocation: {location}\n\nAnswer the 6 questionnaire questions for this person:"

    payload = json.dumps({
        "model": OPUS_MODEL,
        "max_tokens": 600,
        "system": OPUS_QUESTIONNAIRE_SYSTEM,
        "messages": [{"role": "user", "content": prompt}]
    }).encode('utf-8')

    req = urllib.request.Request(
        OPUS_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": OPUS_API_VERSION,
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            text_parts = [
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            ]
            raw = "\n".join(text_parts).strip()

            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cost = (input_tokens * 5.0 / 1_000_000) + (output_tokens * 25.0 / 1_000_000)
            logger.info(f"  Opus questionnaire: {input_tokens}+{output_tokens} tokens = ${cost:.4f}")

            # Parse numbered answers and combine into context
            context = _parse_questionnaire_to_context(raw, role, location)
            return context

    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8') if e.fp else ''
        logger.warning(f"  Opus questionnaire failed ({e.code}): {body[:200]}")
        return ""
    except Exception as e:
        logger.warning(f"  Opus questionnaire failed: {e}")
        return ""


def _parse_questionnaire_to_context(raw, role, location):
    """Parse numbered answers into a combined context string.
    
    Maps the 6 answers to the fields the settings questionnaire uses:
    1. Student/professional → role context
    2. Companies → company tracking
    3. Assets → investment interests
    4. Skills/certs → certification goals
    5. Goals → objectives
    6. Topics → specific tracking interests
    """
    lines = raw.strip().split('\n')
    answers = {}
    current_num = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Match "1." or "1)" or "1:" patterns
        match = re.match(r'^(\d)[.):\s]+(.+)', line)
        if match:
            current_num = int(match.group(1))
            answers[current_num] = match.group(2).strip()
        elif current_num and current_num in answers:
            answers[current_num] += " " + line

    # Build context string (mirrors what "Add answers to context" button does)
    parts = []
    if 1 in answers:
        parts.append(answers[1])
    if 2 in answers:
        parts.append(f"Companies: {answers[2]}")
    if 3 in answers:
        parts.append(f"Assets: {answers[3]}")
    if 4 in answers:
        parts.append(f"Skills: {answers[4]}")
    if 5 in answers:
        parts.append(f"Goals: {answers[5]}")
    if 6 in answers:
        parts.append(f"Tracking: {answers[6]}")

    context = " | ".join(parts) if parts else f"{role} in {location}"
    return context


# ═══════════════════════════════════════════════════════════════
# HTTP API Calls (to running StratOS server)
# ═══════════════════════════════════════════════════════════════

def api_suggest_context(role, location):
    """POST /api/suggest-context → suggested tracking context. Retries once on failure."""
    for attempt in range(2):
        try:
            timeout = 90 if attempt == 0 else 120  # Longer on retry (model swap)
            r = requests.post(f"{API_BASE}/api/suggest-context",
                              json={"role": role, "location": location}, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                suggestion = data.get("suggestion", "")
                tickers = data.get("tickers", [])
                if suggestion:
                    logger.info(f"  ✓ Suggest: {len(suggestion)} chars, {len(tickers)} tickers")
                    return suggestion
                if attempt == 0:
                    logger.info("  Suggest returned empty, retrying...")
                    time.sleep(5)
                    continue
            else:
                logger.warning(f"  Suggest failed: {r.status_code}")
                if attempt == 0:
                    time.sleep(5)
                    continue
        except Exception as e:
            logger.warning(f"  Suggest failed: {e}")
            if attempt == 0:
                time.sleep(5)
                continue
    return ""


def api_generate_profile(role, location, context):
    """POST /api/generate-profile → categories + tickers. Retries once on failure."""
    for attempt in range(2):
        try:
            timeout = 120 if attempt == 0 else 180
            r = requests.post(f"{API_BASE}/api/generate-profile",
                              json={"role": role, "location": location, "context": context}, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                cats = data.get("categories", [])
                tickers = data.get("tickers", [])
                if cats:
                    logger.info(f"  ✓ Generate: {len(cats)} categories, {len(tickers)} tickers")
                    for c in cats:
                        logger.info(f"    → {c.get('label', '?')} ({len(c.get('items', []))} items)")
                    return data
                if attempt == 0:
                    logger.info("  Generate returned empty, retrying...")
                    time.sleep(5)
                    continue
            else:
                logger.warning(f"  Generate failed: {r.status_code} {r.text[:200]}")
                if attempt == 0:
                    time.sleep(5)
                    continue
        except Exception as e:
            logger.warning(f"  Generate failed: {e}")
            if attempt == 0:
                time.sleep(5)
                continue
    return None


# ═══════════════════════════════════════════════════════════════
# Profile Application (config file)
# ═══════════════════════════════════════════════════════════════

def apply_profile(config_path, profile, generated=None):
    """Apply profile + generated categories. Returns original config for restore."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    original = copy.deepcopy(config)

    config["profile"] = {
        "role": profile["role"],
        "location": profile.get("location", ""),
        "context": profile.get("context", ""),
        "interests": [],
    }

    if generated:
        cats = generated.get("categories", [])
        tickers = generated.get("tickers", [])
        if cats:
            config["dynamic_categories"] = cats
        if tickers:
            ticker_objs = []
            for t in tickers:
                sym = t.strip().upper() if isinstance(t, str) else t.get("symbol", "")
                if sym:
                    ticker_objs.append({"symbol": sym, "name": sym, "category": "custom"})
            if ticker_objs:
                config["market"]["tickers"] = ticker_objs
        gen_ctx = generated.get("context", "")
        if gen_ctx:
            config["profile"]["context"] = gen_ctx

    # Atomic write
    tmp = config_path + ".tmp"
    with open(tmp, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    os.replace(tmp, config_path)
    return original


def restore_profile(config_path, original):
    """Atomic write — prevents corruption if interrupted."""
    tmp = config_path + ".tmp"
    with open(tmp, "w") as f:
        yaml.dump(original, f, default_flow_style=False, sort_keys=False)
    os.replace(tmp, config_path)  # Atomic on POSIX
    logger.info("  Profile restored")


def _clear_memory_for_cycle(memory_path):
    """Clear memory.json to prevent cross-profile contamination during autopilot.
    
    Memory stores high-scoring article examples that get injected as few-shot
    context into the scorer's prompt. If a biotech cycle leaves CRISPR articles
    in memory, they'll bias scoring for the next profile (e.g., a geophysicist).
    
    Memory naturally repopulates with relevant examples during each scan.
    """
    try:
        empty_memory = {"version": "1.0", "last_updated": datetime.now().isoformat(), "examples": []}
        with open(memory_path, 'w') as f:
            json.dump(empty_memory, f, indent=2)
    except Exception:
        pass


def _check_training_diversity(backend_dir, min_profiles=4, after=None):
    """Check if training data has enough profile diversity to avoid catastrophic forgetting.
    
    Incremental training on corrections from only 1-2 profiles causes the model
    to over-fit to those profiles and forget everything it learned about others.
    This is what caused the v12 regression (0.4125 → 0.4913) when 142 examples
    from only biotech + geophysicist were used.
    
    Returns (ok: bool, num_profiles: int, total_examples: int)
    """
    try:
        import sqlite3
        db_path = str(Path(backend_dir) / "strat_os.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT DISTINCT profile_role FROM user_feedback 
            WHERE action = 'rate' AND ai_score IS NOT NULL AND user_score IS NOT NULL
            AND profile_role IS NOT NULL AND profile_role != ''
        """
        params = ()
        if after:
            query += " AND created_at > ?"
            params = (after,)
        
        cursor.execute(query, params)
        profiles = [row[0] for row in cursor.fetchall() if row[0]]
        
        # Count total examples
        count_query = """
            SELECT COUNT(*) FROM user_feedback 
            WHERE action = 'rate' AND ai_score IS NOT NULL AND user_score IS NOT NULL
        """
        if after:
            count_query += " AND created_at > ?"
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]
        
        conn.close()
        return len(profiles) >= min_profiles, len(profiles), total
    except Exception:
        return True, 0, 0  # If check fails, allow training


# ═══════════════════════════════════════════════════════════════
# Scan & Distillation
# ═══════════════════════════════════════════════════════════════

def run_scan(backend_dir):
    """Trigger scan via running server API so dashboard updates live."""
    try:
        # Trigger scan on running server (it reloads config.yaml automatically)
        r = requests.get(f"{API_BASE}/api/refresh", timeout=10)
        if r.status_code != 200:
            logger.warning(f"  Server refresh failed ({r.status_code}), falling back to subprocess")
            return _run_scan_subprocess(backend_dir)
        
        logger.info("  Scan triggered on server (dashboard will update live)...")
        
        # Poll until scan completes
        time.sleep(5)  # Let it start
        for _ in range(120):  # Max 10 min
            try:
                status = requests.get(f"{API_BASE}/api/status", timeout=5).json()
                if not status.get("is_scanning", False):
                    # Extract results from recent scan log
                    scans = status.get("recent_scans", [])
                    if scans:
                        latest = scans[0]
                        items = latest.get("items_scored", 0)
                        logger.info(f"  ✓ Server scan: {items} items scored")
                        return {"items_scored": items}
                    return {"items_scored": 0}
                # Show progress
                progress = status.get("progress", "")
                if progress:
                    logger.info(f"  ... {progress}")
            except Exception:
                pass
            time.sleep(5)
        
        logger.warning("  Scan timed out")
        return {"items_scored": 0}
    except requests.ConnectionError:
        logger.warning("  Server not reachable, using subprocess")
        return _run_scan_subprocess(backend_dir)


def _run_scan_subprocess(backend_dir):
    """Fallback: run scan as subprocess (dashboard won't update)."""
    try:
        result = subprocess.run(
            [sys.executable, str(Path(backend_dir) / "main.py"), "--scan"],
            capture_output=True, text=True, cwd=backend_dir, timeout=600)
        if result.returncode != 0:
            return {"error": result.stderr[:300]}
        items = 0
        for line in result.stdout.splitlines():
            if "Scoring complete:" in line:
                nums = re.findall(r'(\d+)\s+(?:critical|high|medium|noise)', line)
                items = sum(int(n) for n in nums)
        return {"items_scored": items}
    except Exception as e:
        return {"error": str(e)}


def run_distillation(backend_dir, config_path, api_key, hours=24, limit=100):
    try:
        from distill import run_distillation as _distill, get_api_key
        key = api_key or get_api_key()
        if not key:
            return {"error": "no_api_key"}
        return _distill(
            db_path=str(Path(backend_dir) / "strat_os.db"),
            config_path=config_path,
            api_key=key, hours=hours, limit=limit,
            threshold=2.0, dry_run=False) or {}
    except SystemExit:
        return {"error": "api_error"}
    except Exception as e:
        return {"error": str(e)}


def run_training(backend_dir, state=None):
    """Run the full training pipeline: export → train → register.
    
    Supports two modes:
    - Incremental (default): only export NEW corrections, train from previous model
    - Full retrain: export ALL corrections, train from base Qwen3-8B
    
    Full retrain triggers automatically every ~10 training cycles to prevent
    drift accumulation from many incremental updates.
    """
    logger.info("=" * 50)
    logger.info("TRAINING PHASE")
    logger.info("=" * 50)

    # ── Pre-flight: check disk space ──
    try:
        import shutil
        disk = shutil.disk_usage(backend_dir)
        free_gb = disk.free / (1024 ** 3)
        logger.info(f"  Disk: {free_gb:.1f} GB free")
        # Full retrain needs ~40GB (model download + merge + GGUF)
        # Incremental needs ~25GB (current_base already exists + merge + GGUF)
        current_base_exists = (Path(backend_dir) / "data" / "models" / "current_base" / "config.json").exists()
        min_gb = 25 if current_base_exists else 40
        if free_gb < min_gb:
            logger.warning(f"  ✗ Not enough disk space ({free_gb:.1f}GB < {min_gb}GB needed)")
            logger.warning(f"    Free space with: rm -rf ~/.cache/huggingface/hub/models--Qwen*")
            logger.warning(f"    Or clean old models: rm -rf data/models/v*/")
            return False
    except Exception as e:
        logger.warning(f"  Could not check disk space: {e}")

    # Decide: incremental or full retrain
    force_full = state and state.needs_full_retrain()
    is_incremental = not force_full and state and state.last_trained_at_time
    
    # ── Diversity guard: prevent catastrophic forgetting ──
    # If incremental training data covers too few profiles, the model over-fits
    # to those profiles and forgets the rest. This caused v12 regression:
    # 142 examples from 2 profiles → loss 0.4125 → 0.4913 (worse than v2).
    # Force full retrain if incremental data isn't diverse enough.
    if is_incremental and state and state.last_trained_at_time:
        diverse_ok, n_profiles, n_examples = _check_training_diversity(
            backend_dir, min_profiles=4, after=state.last_trained_at_time)
        if not diverse_ok:
            logger.warning(f"  ⚠ Incremental data only covers {n_profiles} profile(s) ({n_examples} examples)")
            logger.warning(f"    Forcing full retrain to prevent catastrophic forgetting")
            force_full = True
            is_incremental = False
    
    # ── Critical: verify current_base/ actually exists for incremental ──
    # If it doesn't (e.g. first run with new system, or cleanup deleted it),
    # we MUST do a full retrain. Otherwise we'd train base Qwen3-8B on only
    # the recent corrections, losing all prior learning.
    current_base_dir = Path(backend_dir) / "data" / "models" / "current_base"
    if is_incremental and not (current_base_dir / "config.json").exists():
        logger.warning("  ⚠ current_base/ not found — forcing full retrain to avoid data loss")
        logger.warning("    (Incremental requires a previous trained model to build on)")
        is_incremental = False
        force_full = True
    
    if force_full:
        logger.info("  ★ Full retrain triggered (periodic drift reset)")
    elif is_incremental:
        logger.info(f"  → Incremental training (new corrections since {state.last_trained_at_time[:16]})")
    else:
        logger.info("  → Full training (first run or no previous timestamp)")

    # Export training data
    export_cmd = [sys.executable, str(Path(backend_dir) / "export_training.py"), "--min-delta", "1.5"]
    if is_incremental and state.last_trained_at_time:
        export_cmd += ["--after", state.last_trained_at_time]
    
    result = subprocess.run(export_cmd, capture_output=True, text=True, cwd=backend_dir)
    if result.returncode != 0:
        logger.error(f"Export failed: {result.stderr[:500]}")
        return False

    training_file = Path(backend_dir) / "data" / "training_data.jsonl"
    if not training_file.exists():
        return False
    with open(training_file) as f:
        n = sum(1 for _ in f)
    
    min_examples = 10 if is_incremental else 30
    if n < min_examples:
        logger.info(f"Only {n} examples — need ≥{min_examples}")
        return False

    # ── Free VRAM: stop Ollama before training ──
    logger.info("Stopping Ollama to free VRAM for training...")
    # Read current models from config to stop them (not hardcoded)
    try:
        with open(str(Path(backend_dir) / "config.yaml"), 'r') as f:
            cfg = yaml.safe_load(f) or {}
        scoring_model = cfg.get("scoring", {}).get("model", "")
        inference_model = cfg.get("scoring", {}).get("inference_model", "")
        for m in [scoring_model, inference_model]:
            if m:
                subprocess.run(["ollama", "stop", m], capture_output=True, timeout=10)
    except Exception:
        pass
    # Also stop common base models that may be loaded
    subprocess.run(["ollama", "stop", "qwen3:8b"], capture_output=True, timeout=10)
    time.sleep(3)  # Let VRAM free up

    logger.info(f"Training with {n} examples ({'incremental' if is_incremental else 'full retrain'})...")
    
    # Build train command
    train_cmd = [sys.executable, str(Path(backend_dir) / "train_lora.py"), "--epochs", "3"]
    if force_full:
        train_cmd.append("--full-retrain")
    # else: --incremental is the default
    
    try:
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=backend_dir)
        
        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"  [TRAIN] {line}")
        
        process.wait(timeout=3600)
        success = process.returncode == 0
    except subprocess.TimeoutExpired:
        process.kill()
        logger.error("Training timed out after 1 hour")
        success = False
    except Exception as e:
        logger.error(f"Training failed: {e}")
        success = False

    # ── Restart Ollama models ──
    logger.info("Restarting Ollama models...")
    subprocess.run(["ollama", "list"], capture_output=True, timeout=10)
    time.sleep(2)

    if success:
        logger.info("✓ Training complete!")
        return "full" if force_full else "incremental"
    else:
        logger.error("Training failed — check logs above")
    return False


# ═══════════════════════════════════════════════════════════════
# Cost Tracking
# ═══════════════════════════════════════════════════════════════

class CostTracker:
    def __init__(self, budget=5.0):
        self.budget = budget
        self.spent = 0.0
        self.log_path = str(Path(__file__).parent / "data" / "cost_tracker.json")
        try:
            with open(self.log_path) as f:
                self.spent = json.load(f).get("total_spent", 0.0)
        except Exception:
            pass

    def _save(self):
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump({"total_spent": round(self.spent, 4), "budget": self.budget,
                        "remaining": round(self.budget - self.spent, 4),
                        "last_updated": datetime.now().isoformat()}, f, indent=2)

    def add(self, cost):
        self.spent += cost
        self._save()

    @property
    def remaining(self):
        return self.budget - self.spent

    def can_afford(self, est=0.20):
        return self.remaining >= est


# ═══════════════════════════════════════════════════════════════
# Persistent State (survives restarts)
# ═══════════════════════════════════════════════════════════════

class AutopilotState:
    """Persists distill_count + total_corrections across restarts."""

    def __init__(self, backend_dir):
        self.path = str(Path(backend_dir) / "data" / "autopilot_state.json")
        self.distill_count = 0
        self.total_corrections = 0
        self.last_profile = ""
        self.last_trained_at = 0  # distill_count when training last ran (0 = never)
        self.corrections_at_last_train = 0  # total_corrections when training last ran
        self.last_trained_at_time = ""  # ISO timestamp when training last ran
        self._force_full_retrain = False  # Set by distill if agreement drops
        self._load()

    def _load(self):
        try:
            with open(self.path) as f:
                d = json.load(f)
                self.distill_count = d.get("distill_count", 0)
                self.total_corrections = d.get("total_corrections", 0)
                self.last_profile = d.get("last_profile", "")
                self.last_trained_at = d.get("last_trained_at", 0)
                self.corrections_at_last_train = d.get("corrections_at_last_train", 0)
                self.last_trained_at_time = d.get("last_trained_at_time", "")
                self._force_full_retrain = d.get("force_full_retrain", False)
        except Exception:
            pass

    def save(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump({
                "distill_count": self.distill_count,
                "total_corrections": self.total_corrections,
                "last_profile": self.last_profile,
                "last_trained_at": self.last_trained_at,
                "corrections_at_last_train": self.corrections_at_last_train,
                "last_trained_at_time": self.last_trained_at_time,
                "force_full_retrain": self._force_full_retrain,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
        os.replace(tmp, self.path)

    def record_cycle(self, profile_id, corrections):
        self.distill_count += 1
        self.total_corrections += corrections
        self.last_profile = profile_id
        self.save()

    def record_training(self):
        self.last_trained_at = self.distill_count
        self.corrections_at_last_train = self.total_corrections
        self.last_trained_at_time = datetime.now().isoformat()
        self._force_full_retrain = False  # Clear flag after successful training
        self.save()

    def should_train(self, train_every):
        """Smart training trigger: only train when enough NEW corrections justify it.
        
        Instead of blindly retraining every N cycles on ALL data, we check how many
        new corrections have been added since the last training run. This avoids
        wasting 36+ minutes retraining on data the model has already learned.
        
        Thresholds:
        - Need at least `train_every` cycles since last training
        - Need at least 50 new corrections since last training (worth the GPU time)
        - OR 200+ new corrections regardless of cycle count (big backlog)
        """
        cycles_since = self.cycles_since_training()
        new_corrections = self.total_corrections - self.corrections_at_last_train
        
        # Big backlog — train regardless of cycle count
        if new_corrections >= 200:
            return True
        
        # Normal trigger: enough cycles AND enough new data
        if cycles_since >= train_every and new_corrections >= 50:
            return True
        
        return False
    
    def needs_full_retrain(self):
        """Determine if we should do a full retrain from base model.
        
        Instead of a fixed schedule, we only trigger a full retrain when
        quality degrades — detected by recent distillation agreement rates
        dropping significantly. This is stored by the distillation pipeline.
        
        Returns False by default. The distill pipeline sets a flag if needed.
        """
        return self._force_full_retrain
    
    def flag_full_retrain(self, reason=""):
        """Called by distillation when agreement rate drops below threshold."""
        self._force_full_retrain = True
        logger.info(f"  ⚠ Full retrain flagged: {reason}")
        self.save()

    def cycles_since_training(self):
        return self.distill_count - self.last_trained_at


# ═══════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════

def run_autopilot(max_cycles=0, train_every=5, scan_interval=300,
                  dry_run=False, skip_training=False, budget=5.0, use_server=True):
    backend_dir = str(Path(__file__).parent)
    config_path = str(Path(backend_dir) / "config.yaml")
    db_path = str(Path(backend_dir) / "strat_os.db")
    memory_path = str(Path(backend_dir) / "data" / "memory.json")
    Path(backend_dir, "data").mkdir(exist_ok=True)

    from distill import get_api_key
    api_key = get_api_key()
    if not api_key and not dry_run:
        logger.error("No API key. Set ANTHROPIC_API_KEY in .env")
        sys.exit(1)

    tracker = CostTracker(budget=budget)
    state = AutopilotState(backend_dir)

    logger.info("═" * 60)
    logger.info("  STRAT_OS AUTOPILOT — Autonomous Self-Improvement")
    logger.info("═" * 60)
    logger.info(f"  Profiles: {len(PROFILE_TEMPLATES)} | Train every: {train_every}")
    logger.info(f"  Budget: ${tracker.remaining:.2f} | Interval: {scan_interval}s")
    logger.info(f"  Server API (suggest/generate): {'Yes' if use_server else 'No'}")
    logger.info(f"  Opus context generation: {'Yes' if api_key else 'No (will use local suggest)'}")
    logger.info(f"  Persistent state: {state.distill_count} cycles, {state.total_corrections} corrections so far")
    logger.info("═" * 60)

    if dry_run:
        for i, p in enumerate(PROFILE_TEMPLATES):
            logger.info(f"  Cycle {i+1}: {p['id']} — {p['role'][:60]}")
        return

    # ── Check if we should train FIRST (accumulated corrections from past runs) ──
    if not skip_training and state.total_corrections >= 50 and state.distill_count > 0:
        new_corrections = state.total_corrections - state.corrections_at_last_train
        cycles_untrained = state.cycles_since_training()
        if cycles_untrained >= train_every and new_corrections >= 50:
            logger.info(f"\n  {new_corrections} new corrections ({state.total_corrections} total), {cycles_untrained} cycles since last training")
            logger.info("  → Triggering catch-up training before starting cycles...")
            result = run_training(backend_dir, state)
            if result:
                state.record_training()

    with open(config_path, 'r') as f:
        original_backup = copy.deepcopy(yaml.safe_load(f) or {})

    cycle = 0

    try:
        while max_cycles == 0 or cycle < max_cycles:
            cycle += 1
            if not tracker.can_afford(0.15):
                logger.warning(f"Budget exhausted (${tracker.remaining:.2f})")
                break

            unused = get_unused_profiles(db_path, PROFILE_TEMPLATES)
            profile = random.choice(unused)

            logger.info(f"\n{'━' * 60}")
            logger.info(f"  CYCLE {cycle} (global #{state.distill_count + 1}) | {profile['id']}")
            logger.info(f"  {profile['role'][:70]}")
            logger.info(f"  Budget: ${tracker.remaining:.2f}")
            logger.info(f"{'━' * 60}")

            # ── 1. Generate Context (Opus questionnaire → local suggest fallback) ──
            logger.info("[1/5] Generate context (Opus questionnaire)...")
            context = ""
            if api_key:
                context = opus_generate_context(
                    profile["role"], profile.get("location", ""), api_key)
                if context:
                    logger.info(f"  ✓ Opus context: {len(context)} chars")
            if not context and use_server:
                logger.info("  → Opus unavailable, falling back to local suggest...")
                context = api_suggest_context(profile["role"], profile.get("location", ""))
            if context:
                profile["context"] = context
            else:
                logger.info("  → No context generated (will use role only)")

            # ── 1b. Clear memory to prevent cross-profile contamination ──
            # Memory accumulates high-scoring examples from previous profiles.
            # A biotech researcher's 9.5-scored CRISPR article would bias the
            # scorer when evaluating articles for a geophysicist or teacher.
            _clear_memory_for_cycle(memory_path)

            # ── 2. Generate Categories ──
            logger.info("[2/5] Generate categories (may take ~60s)...")
            generated = None
            if use_server:
                generated = api_generate_profile(
                    profile["role"], profile.get("location", ""),
                    profile.get("context", ""))
            if not generated:
                logger.info("  → Skipped (will use existing categories)")

            # ── 3. Apply + Scan (via server = dashboard updates live!) ──
            logger.info("[3/5] Apply profile & scan via server...")
            original = apply_profile(config_path, profile, generated)
            time.sleep(2)  # Let config.yaml settle before server reloads it
            t0 = time.time()
            scan_result = run_scan(backend_dir)
            elapsed = time.time() - t0

            if "error" in scan_result:
                logger.error(f"Scan failed: {scan_result['error'][:200]}")
                restore_profile(config_path, original)
                time.sleep(30)
                continue

            logger.info(f"  ✓ {scan_result.get('items_scored', 0)} items in {elapsed:.0f}s")

            # ── 4. Distill ──
            logger.info("[4/5] Distillation (Opus re-scoring)...")
            result = run_distillation(backend_dir, config_path, api_key)

            if "error" in result:
                logger.error(f"Distill failed: {result.get('error')}")
            else:
                corrections = result.get("corrections_saved", 0)
                cost = result.get("estimated_cost", 0.20)
                state.record_cycle(profile["id"], corrections)
                tracker.add(cost)
                
                # Monitor agreement rate — flag full retrain if quality degrades
                items_scored = result.get("items_scored", 0)
                agreed = result.get("agreed", 0)
                if items_scored > 20:
                    agreement_rate = agreed / items_scored
                    logger.info(f"  ✓ {corrections} corrections | ${cost:.2f} | Agreement: {agreement_rate:.0%} | Total: ${tracker.spent:.2f}")
                    if agreement_rate < 0.35:
                        state.flag_full_retrain(f"Agreement dropped to {agreement_rate:.0%} ({agreed}/{items_scored})")
                else:
                    logger.info(f"  ✓ {corrections} corrections | ${cost:.2f} | Total: ${tracker.spent:.2f}")

            # ── 5. Restore ──
            logger.info("[5/5] Restoring original profile...")
            restore_profile(config_path, original)

            # Mark used
            try:
                conn = sqlite3.connect(db_path)
                conn.execute(
                    "INSERT INTO user_feedback (news_id, title, action, note, created_at) VALUES (?,?,?,?,?)",
                    ("autopilot", profile["id"], "autopilot",
                     f"Autopilot: {profile['id']} cycle {cycle}", datetime.now().isoformat()))
                conn.commit()
                conn.close()
            except Exception:
                pass

            # Training trigger (uses persistent count!)
            if not skip_training and state.should_train(train_every):
                new_since = state.total_corrections - state.corrections_at_last_train
                logger.info(f"\n  ★ Training trigger (global cycle #{state.distill_count}, {new_since} new corrections since last train, {state.total_corrections} total)")
                result = run_training(backend_dir, state)
                if result:
                    state.record_training()

            logger.info(f"\n  Status: {cycle} cycles (global #{state.distill_count}) | {state.total_corrections} corrections | ${tracker.spent:.2f}/{tracker.budget:.2f}")
            logger.info(f"  Next in {scan_interval}s...")

            if max_cycles == 0 or cycle < max_cycles:
                time.sleep(scan_interval)

    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    finally:
        restore_profile(config_path, original_backup)
        logger.info(f"\n{'═' * 60}")
        logger.info(f"  COMPLETE: {cycle} cycles (global #{state.distill_count}) | {state.total_corrections} corrections | ${tracker.spent:.2f}")
        logger.info(f"{'═' * 60}")
        if not skip_training and state.total_corrections > 30:
            new_since = state.total_corrections - state.corrections_at_last_train
            if new_since >= 30:  # Only train on exit if meaningful new data
                logger.info(f"  {new_since} new corrections since last train — training before exit...")
                result = run_training(backend_dir, state)
                if result:
                    state.record_training()


def main():
    parser = argparse.ArgumentParser(description="STRAT_OS Autopilot")
    parser.add_argument("--cycles", type=int, default=0)
    parser.add_argument("--train-every", type=int, default=5)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--no-server", action="store_true")
    args = parser.parse_args()
    run_autopilot(args.cycles, args.train_every, args.interval,
                  args.dry_run, args.skip_training, args.budget, not args.no_server)


if __name__ == "__main__":
    main()
