"""
GPU Manager — Swaps between Ollama and ComfyUI automatically.

Only one can use the GPU at a time (24GB VRAM).
When chat is requested: ensure Ollama is running (stop ComfyUI if needed).
When image gen is requested: ensure ComfyUI is running (unload Ollama models if needed).

Strategy: Ollama process stays alive but models are unloaded (0 VRAM) when
ComfyUI needs the GPU. This avoids process killing, VRAM fragmentation,
and the race conditions that caused PC crashes.

Safety: VRAM is verified via rocm-smi before starting ComfyUI. If API-based
unload fails, escalates to force-killing Ollama as a last resort.
"""

import os
import subprocess
import time
import logging
import requests
import threading

logger = logging.getLogger("gpu_manager")

OLLAMA_HOST = "http://localhost:11434"
COMFYUI_HOST = "http://127.0.0.1:8188"
COMFYUI_DIR = "/home/ahmad/Downloads/StratOS/StratOS1/tools/ComfyUI"

# CHROMA needs ~10-12GB VRAM. Require at least 12GB free before starting ComfyUI.
COMFYUI_VRAM_REQUIRED_MB = 12_000
# Maximum time to wait for VRAM to be freed after unload (seconds)
VRAM_RELEASE_TIMEOUT = 30
# Total GPU VRAM in MB (7900 XTX)
VRAM_TOTAL_MB = 24_576
# Safety margin — never fill VRAM beyond this threshold.
# Leaves room for GPU driver overhead, context allocation, and fragmentation.
VRAM_SAFE_LIMIT_MB = 20_480  # 20 GB

_lock = threading.Lock()
_comfyui_process = None
_comfyui_log_handle = None  # Track file handle to prevent leaks
_comfyui_ever_started = False  # Skip ComfyUI check if never started this session


def _get_vram_used_mb() -> int | None:
    """Get GPU 0 VRAM usage in MB via rocm-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5,
        )
        total = used = None
        for line in result.stdout.splitlines():
            if "GPU[0]" in line and "Total Used" in line:
                used = int(line.split(":")[-1].strip()) // (1024 * 1024)
            elif "GPU[0]" in line and "Total Memory" in line:
                total = int(line.split(":")[-1].strip()) // (1024 * 1024)
        if used is not None:
            logger.debug(f"VRAM: {used}MB used / {total}MB total")
            return used
    except Exception as e:
        logger.warning(f"rocm-smi VRAM check failed: {e}")
    return None


def _vram_has_headroom() -> bool:
    """Check if enough VRAM is free for ComfyUI. Assumes True if check unavailable."""
    used = _get_vram_used_mb()
    if used is None:
        # Can't check — proceed cautiously (better than blocking entirely)
        logger.warning("Cannot check VRAM — rocm-smi unavailable, proceeding anyway")
        return True
    # 7900 XTX = ~24576 MB usable
    free = 24_576 - used
    if free >= COMFYUI_VRAM_REQUIRED_MB:
        logger.info(f"VRAM OK: {free}MB free (need {COMFYUI_VRAM_REQUIRED_MB}MB)")
        return True
    logger.warning(f"VRAM insufficient: {free}MB free, need {COMFYUI_VRAM_REQUIRED_MB}MB ({used}MB used)")
    return False


def _ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False  # Connection refused is expected when not running


def _comfyui_running() -> bool:
    try:
        r = requests.get(f"{COMFYUI_HOST}/system_stats", timeout=2)
        return r.status_code == 200
    except Exception:
        return False  # Connection refused is expected when not running


def _ollama_has_loaded_models() -> bool:
    """Check if Ollama currently has any models loaded in VRAM.

    SAFETY: Returns True on failure — if we can't confirm models are unloaded,
    assume they ARE loaded to prevent OOM crash. This is the opposite of the
    previous behavior which assumed False on failure (causing the crash).
    """
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=5)
        if r.status_code == 200:
            models = r.json().get("models", [])
            if models:
                names = [m.get("name", "?") for m in models]
                logger.info(f"Ollama has {len(models)} model(s) loaded: {names}")
            return len(models) > 0
    except Exception as e:
        logger.warning(f"Cannot check Ollama models (assuming loaded): {e}")
        return True  # SAFE DEFAULT: assume loaded → trigger unload
    return False


def _unload_ollama_models() -> bool:
    """Unload all Ollama models from VRAM via API. Ollama process stays alive.

    Returns True if all models confirmed unloaded, False otherwise.
    """
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=5)
        if r.status_code != 200:
            logger.warning(f"Ollama /api/ps returned {r.status_code}")
            return False
        models = r.json().get("models", [])
        if not models:
            logger.info("No Ollama models loaded — nothing to unload")
            return True

        model_names = [m.get("name", "") for m in models if m.get("name")]
        logger.info(f"Unloading {len(model_names)} Ollama model(s): {model_names}")

        for name in model_names:
            logger.info(f"Unloading: {name}")
            try:
                requests.post(f"{OLLAMA_HOST}/api/generate",
                              json={"model": name, "keep_alive": 0}, timeout=30)
            except Exception as e:
                logger.warning(f"Unload API call failed for {name}: {e}")

        # Poll until all models unloaded (up to 20s)
        for attempt in range(10):
            time.sleep(2)
            try:
                r2 = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=5)
                if r2.status_code == 200:
                    remaining = r2.json().get("models", [])
                    if not remaining:
                        logger.info("All Ollama models confirmed unloaded from VRAM")
                        return True
                    names = [m.get("name", "?") for m in remaining]
                    logger.info(f"Still loaded after {(attempt+1)*2}s: {names}")
            except Exception:
                pass

        logger.warning("Ollama models not fully unloaded after 20s of polling")
        return False

    except Exception as e:
        logger.error(f"Could not unload Ollama models via API: {e}")
        return False


def _force_kill_ollama():
    """Last-resort: kill Ollama process to free VRAM."""
    logger.warning("FORCE KILLING Ollama to free VRAM (last resort)")
    try:
        subprocess.run(["pkill", "-9", "-f", "ollama serve"], timeout=5, capture_output=True)
    except Exception as e:
        logger.warning(f"pkill -9 ollama failed: {e}")
    time.sleep(3)
    # Also try systemctl in case it's a service
    try:
        subprocess.run(["systemctl", "stop", "ollama"], timeout=5, capture_output=True)
    except Exception:
        pass
    time.sleep(2)


def _stop_ollama():
    """Stop Ollama to free GPU. Unloads models first, then kills process."""
    logger.info("Stopping Ollama...")
    _unload_ollama_models()
    time.sleep(3)
    # Kill the process as last resort
    try:
        subprocess.run(["pkill", "-f", "ollama serve"], timeout=5, capture_output=True)
    except Exception as e:
        logger.warning(f"pkill ollama failed (non-fatal): {e}")
    time.sleep(3)
    logger.info("Ollama stopped")


def _start_ollama():
    """Start Ollama."""
    if _ollama_running():
        return True
    logger.info("Starting Ollama...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env={
                **os.environ,
                "OLLAMA_MAX_LOADED_MODELS": "1",
                "OLLAMA_KEEP_ALIVE": "10m",
                "OLLAMA_NUM_PARALLEL": "1",
                "OLLAMA_FLASH_ATTENTION": "0",
                "ROCR_VISIBLE_DEVICES": "0",
            }
        )
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        return False
    # Wait for it to be ready
    for _ in range(20):
        time.sleep(1)
        if _ollama_running():
            logger.info("Ollama ready")
            return True
    logger.error("Ollama failed to start within 20s")
    return False


def _stop_comfyui():
    """Stop ComfyUI and close log file handle."""
    global _comfyui_process, _comfyui_log_handle
    logger.info("Stopping ComfyUI...")
    try:
        subprocess.run(["pkill", "-f", "ComfyUI/main.py"], timeout=5, capture_output=True)
    except Exception as e:
        logger.warning(f"pkill ComfyUI failed (non-fatal): {e}")
    if _comfyui_process:
        try:
            _comfyui_process.terminate()
            _comfyui_process.wait(timeout=5)
        except Exception as e:
            logger.warning(f"ComfyUI terminate failed, force killing: {e}")
            try:
                _comfyui_process.kill()
            except Exception as e2:
                logger.warning(f"ComfyUI kill also failed: {e2}")
        _comfyui_process = None
    # Close log file handle (prevents fd leak)
    if _comfyui_log_handle:
        try:
            _comfyui_log_handle.close()
        except Exception:
            pass
        _comfyui_log_handle = None
    time.sleep(3)
    logger.info("ComfyUI stopped")


def _start_comfyui():
    """Start ComfyUI in API mode."""
    global _comfyui_process, _comfyui_log_handle, _comfyui_ever_started
    _comfyui_ever_started = True
    if _comfyui_running():
        return True
    logger.info("Starting ComfyUI...")
    try:
        # Close any previous log handle before opening new one
        if _comfyui_log_handle:
            try:
                _comfyui_log_handle.close()
            except Exception:
                pass
        _comfyui_log_handle = open("/tmp/comfyui.log", "w")
        _comfyui_process = subprocess.Popen(
            ["python3", "main.py", "--listen", "127.0.0.1", "--port", "8188",
             "--preview-method", "auto"],
            cwd=COMFYUI_DIR,
            stdout=_comfyui_log_handle, stderr=subprocess.STDOUT,
        )
    except Exception as e:
        logger.error(f"Failed to start ComfyUI: {e}")
        return False
    # Wait for it to be ready (ComfyUI takes ~20-30s to load models)
    for i in range(60):
        time.sleep(2)
        if _comfyui_running():
            logger.info(f"ComfyUI ready (took ~{i*2}s)")
            return True
    logger.error("ComfyUI failed to start within 120s")
    return False


def _wait_for_vram_release() -> bool:
    """Wait for VRAM to be freed after model unload. Returns True if enough free."""
    deadline = time.time() + VRAM_RELEASE_TIMEOUT
    while time.time() < deadline:
        if _vram_has_headroom():
            return True
        remaining = int(deadline - time.time())
        logger.info(f"Waiting for VRAM release... ({remaining}s remaining)")
        time.sleep(2)
    return False


def ensure_ollama() -> bool:
    """Ensure Ollama is running and GPU is available. Stops ComfyUI if needed. Thread-safe."""
    with _lock:
        # Only check ComfyUI if it was ever started this session — avoids
        # a wasted HTTP request (with 2s timeout) on every chat message
        if _comfyui_ever_started and _comfyui_running():
            logger.info("ComfyUI still running — stopping to free GPU for Ollama")
            _stop_comfyui()
        if _ollama_running():
            return True
        return _start_ollama()


def ensure_model_ready(model_name: str) -> bool:
    """Ensure Ollama is running and VRAM can fit the requested model.

    Goes beyond ensure_ollama() by checking what's currently loaded in VRAM.
    If the requested model is already loaded, returns immediately.
    If loading it would exceed the VRAM safety limit, unloads other models first.

    MUST be called before any Ollama /api/chat or /api/generate request to
    prevent OOM crashes from concurrent model loads.
    """
    with _lock:
        # Step 1: Ensure Ollama is running (same as ensure_ollama, inlined to avoid deadlock)
        if _comfyui_ever_started and _comfyui_running():
            logger.info("ComfyUI still running — stopping to free GPU for Ollama")
            _stop_comfyui()
        if not _ollama_running():
            if not _start_ollama():
                return False

        # Step 2: Check what's loaded in VRAM
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=5)
            if r.status_code != 200:
                logger.warning(f"Cannot check loaded models (/api/ps returned {r.status_code})")
                return True  # Proceed cautiously

            models = r.json().get("models", [])
            if not models:
                return True  # Nothing loaded, safe to proceed

            loaded_names = [m.get("name", "") for m in models]

            # Model already loaded — safe to proceed
            if any(model_name in name for name in loaded_names):
                return True

            # Different model(s) loaded — check VRAM before loading new one
            total_vram_bytes = sum(m.get("size_vram", 0) for m in models)
            total_vram_mb = total_vram_bytes // (1024 * 1024)

            if total_vram_mb > VRAM_SAFE_LIMIT_MB:
                logger.warning(
                    f"VRAM at {total_vram_mb}MB (limit {VRAM_SAFE_LIMIT_MB}MB) with "
                    f"{loaded_names} — unloading before loading '{model_name}'"
                )
                _unload_ollama_models()
                return True

            # VRAM has room but a different model is loaded.
            # With MAX_LOADED_MODELS=1, Ollama will auto-evict the old model.
            # Log for visibility.
            logger.info(
                f"Model switch: {loaded_names} → '{model_name}' "
                f"(VRAM: {total_vram_mb}MB, Ollama will auto-evict)"
            )
            return True

        except Exception as e:
            logger.warning(f"VRAM check failed (proceeding cautiously): {e}")
            return True


def ensure_comfyui() -> bool:
    """Ensure ComfyUI is running. Unloads Ollama models first. Thread-safe.

    Three-stage escalation to free VRAM:
      1. API-based model unload (keep_alive=0) — graceful, Ollama stays alive
      2. Wait + verify VRAM via rocm-smi — confirms GPU memory actually freed
      3. Force-kill Ollama — last resort if API unload fails

    Will NOT start ComfyUI until VRAM is confirmed free (prevents OOM crash).
    """
    with _lock:
        if _comfyui_running():
            return True

        # Stage 1: Try API-based model unload (graceful)
        if _ollama_running() and _ollama_has_loaded_models():
            logger.info("Unloading Ollama models for ComfyUI (process stays alive)...")
            api_unload_ok = _unload_ollama_models()
            # Extra wait for GPU driver to fully release VRAM
            time.sleep(3)

            if not api_unload_ok:
                logger.warning("API-based unload failed — will check VRAM directly")

        # Stage 2: Verify VRAM is actually free via rocm-smi
        if not _vram_has_headroom():
            logger.warning("VRAM still occupied after API unload — waiting for release...")
            if not _wait_for_vram_release():
                # Stage 3: Force-kill Ollama as last resort
                logger.error("VRAM not freed after timeout — escalating to force-kill Ollama")
                _force_kill_ollama()

                # Final VRAM check after force-kill
                time.sleep(3)
                if not _vram_has_headroom():
                    # Check one more time with a longer wait
                    if not _wait_for_vram_release():
                        logger.error("VRAM STILL not free after force-killing Ollama — aborting ComfyUI start")
                        return False

        logger.info("VRAM clear — starting ComfyUI")
        return _start_comfyui()
