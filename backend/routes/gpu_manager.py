"""
GPU Manager — Swaps between Ollama and ComfyUI automatically.

Only one can use the GPU at a time (24GB VRAM).
When chat is requested: ensure Ollama is running (stop ComfyUI if needed).
When image gen is requested: ensure ComfyUI is running (stop Ollama if needed).
"""

import subprocess
import time
import logging
import requests
import threading

logger = logging.getLogger("gpu_manager")

OLLAMA_HOST = "http://localhost:11434"
COMFYUI_HOST = "http://127.0.0.1:8188"
COMFYUI_DIR = "/home/ahmad/Downloads/StratOS/StratOS1/tools/ComfyUI"

_lock = threading.Lock()
_comfyui_process = None


def _ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False  # Connection refused is expected when not running


def _comfyui_running() -> bool:
    try:
        r = requests.get(f"{COMFYUI_HOST}/system_stats", timeout=2)
        return r.status_code == 200
    except Exception:
        return False  # Connection refused is expected when not running


def _unload_ollama_models():
    """Unload all Ollama models from VRAM via API (no sudo needed)."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=3)
        if r.status_code == 200:
            for model in r.json().get("models", []):
                name = model.get("name", "")
                if name:
                    logger.info(f"Unloading Ollama model: {name}")
                    requests.post(f"{OLLAMA_HOST}/api/generate",
                                  json={"model": name, "keep_alive": 0}, timeout=10)
    except Exception as e:
        logger.warning(f"Could not unload Ollama models via API: {e}")


def _stop_ollama():
    """Stop Ollama to free GPU. Uses API to unload models first, then kills process."""
    logger.info("Stopping Ollama...")
    # Unload all models via API (frees VRAM without sudo)
    _unload_ollama_models()
    time.sleep(2)
    # Kill the process (no systemctl — avoids polkit auth popup)
    try:
        subprocess.run(["pkill", "-f", "ollama serve"], timeout=5, capture_output=True)
    except Exception as e:
        logger.warning(f"pkill ollama failed (non-fatal): {e}")
    # Wait for VRAM to free
    time.sleep(2)
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
                **__import__('os').environ,
                "OLLAMA_MAX_LOADED_MODELS": "2",
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
    """Stop ComfyUI."""
    global _comfyui_process
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
    time.sleep(3)
    logger.info("ComfyUI stopped")


def _start_comfyui():
    """Start ComfyUI in API mode."""
    global _comfyui_process
    if _comfyui_running():
        return True
    logger.info("Starting ComfyUI...")
    try:
        _comfyui_log = open("/tmp/comfyui.log", "w")
        _comfyui_process = subprocess.Popen(
            ["python3", "main.py", "--listen", "127.0.0.1", "--port", "8188",
             "--preview-method", "auto"],
            cwd=COMFYUI_DIR,
            stdout=_comfyui_log, stderr=subprocess.STDOUT,
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


def ensure_ollama() -> bool:
    """Ensure Ollama is running. Stops ComfyUI if needed. Thread-safe."""
    with _lock:
        if _ollama_running():
            return True
        if _comfyui_running():
            _stop_comfyui()
        return _start_ollama()


def ensure_comfyui() -> bool:
    """Ensure ComfyUI is running. Stops Ollama if needed. Thread-safe."""
    with _lock:
        if _comfyui_running():
            return True
        if _ollama_running():
            _stop_ollama()
        return _start_comfyui()
