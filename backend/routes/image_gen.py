"""
Image Generation Service for StratOS.

Routes image generation requests to ComfyUI running in API mode.
Supports two models:
- FLUX.1-schnell Q8: All SFW content (4 steps, ~5-10s)
- Pony Diffusion V7: NSFW/explicit content only (30 steps, ~20-30s)

ComfyUI must be running: python main.py --listen 127.0.0.1 --port 8188

Endpoints:
  POST /api/image/generate            — Free-form text-to-image
  POST /api/image/character-portrait  — Generate from character card fields
  GET  /api/image/<image_id>          — Serve generated image
  GET  /api/image/gallery             — User's generation history
  DELETE /api/image/<image_id>        — Delete a generated image
"""

import json
import os
import time
import uuid
import logging
from pathlib import Path

import requests

from routes.helpers import json_response, error_response, read_json_body
from routes.gpu_manager import ensure_comfyui

logger = logging.getLogger("image_gen")

COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "http://127.0.0.1:8188")
WORKFLOW_DIR = Path(__file__).parent.parent / "data" / "comfyui_workflows"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "generated_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PONY_NEGATIVE = (
    "worst quality, low quality, blurry, deformed, bad anatomy, "
    "bad hands, extra fingers, missing fingers, watermark, signature, "
    "text, username, artist name"
)


def select_image_model(style: str, nsfw: bool) -> str:
    """Route to appropriate model. FLUX for SFW, Pony for NSFW only."""
    if nsfw:
        return "pony"
    return "flux"


def _load_workflow(name: str) -> dict:
    path = WORKFLOW_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Workflow not found: {path}")
    with open(path) as f:
        return json.load(f)


def _ensure_comfyui_ready() -> bool:
    """Check if ComfyUI is running."""
    try:
        r = requests.get(f"{COMFYUI_HOST}/system_stats", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _check_ollama_running() -> bool:
    """Check if Ollama is running (VRAM conflict)."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _queue_prompt(workflow: dict) -> str:
    try:
        response = requests.post(
            f"{COMFYUI_HOST}/prompt",
            json={"prompt": workflow},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json().get("prompt_id", "")
        logger.error(f"ComfyUI queue failed: {response.status_code} {response.text}")
        return ""
    except requests.exceptions.ConnectionError:
        logger.error("ComfyUI not running. Start with: bash tools/start_comfyui.sh")
        return ""
    except Exception as e:
        logger.error(f"ComfyUI queue error: {e}")
        return ""


def _poll_result(prompt_id: str, timeout: int = 120) -> str | None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{COMFYUI_HOST}/history/{prompt_id}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if prompt_id in data:
                    outputs = data[prompt_id].get("outputs", {})
                    for node_output in outputs.values():
                        if "images" in node_output:
                            for img in node_output["images"]:
                                return img.get("filename")
            time.sleep(1)
        except Exception:
            time.sleep(2)
    return None


def _download_image(filename: str) -> bytes | None:
    try:
        response = requests.get(f"{COMFYUI_HOST}/view", params={"filename": filename}, timeout=30)
        return response.content if response.status_code == 200 else None
    except Exception as e:
        logger.error(f"Image download error: {e}")
        return None


def character_to_image_prompt(card: dict, style: str = "anime", nsfw: bool = False) -> str:
    """Build optimized image prompt from character card fields."""
    parts = []

    if nsfw:
        parts.append("score_9, score_8_up, score_7_up")
    elif style == "anime":
        parts.append("masterpiece, best quality, highly detailed anime illustration")
    elif style == "realistic":
        parts.append("masterpiece, photorealistic, highly detailed photograph")
    else:
        parts.append("masterpiece, best quality, detailed illustration")

    if card.get("physical_description"):
        parts.append(card["physical_description"][:500])

    if card.get("name"):
        parts.append(f"1person, {card['name']}")

    parts.append("sharp focus, detailed eyes, detailed face")

    if nsfw:
        parts.append("rating_explicit")

    return ", ".join(parts)


def generate_image(prompt: str, negative_prompt: str = "", model: str = "flux",
                   width: int = 768, height: int = 1024, seed: int = -1,
                   steps: int | None = None) -> dict:
    """Generate an image via ComfyUI. Auto-swaps GPU from Ollama if needed."""

    # Ensure ComfyUI is running (swaps from Ollama if needed)
    if not ensure_comfyui():
        return {"success": False, "error": "Failed to start ComfyUI. Check GPU availability."}

    if model == "pony":
        workflow = _load_workflow("pony_t2i")
        steps = steps or 30
        negative_prompt = negative_prompt or PONY_NEGATIVE
        workflow["3"]["inputs"]["text"] = prompt      # CLIPTextEncode positive
        workflow["4"]["inputs"]["text"] = negative_prompt  # CLIPTextEncode negative
        workflow["5"]["inputs"]["width"] = width       # EmptySD3LatentImage
        workflow["5"]["inputs"]["height"] = height
        workflow["6"]["inputs"]["seed"] = seed if seed >= 0 else int(time.time()) % 2**32
        workflow["6"]["inputs"]["steps"] = steps       # KSampler
    else:
        workflow = _load_workflow("flux_t2i")
        steps = steps or 4
        workflow["4"]["inputs"]["text"] = prompt
        workflow["6"]["inputs"]["width"] = width
        workflow["6"]["inputs"]["height"] = height
        workflow["7"]["inputs"]["seed"] = seed if seed >= 0 else int(time.time()) % 2**32
        workflow["7"]["inputs"]["steps"] = steps

    prompt_id = _queue_prompt(workflow)
    if not prompt_id:
        return {"success": False, "error": "Failed to queue. Is ComfyUI running?"}

    filename = _poll_result(prompt_id, timeout=120)
    if not filename:
        return {"success": False, "error": "Generation timed out."}

    image_bytes = _download_image(filename)
    if not image_bytes:
        return {"success": False, "error": "Failed to download image."}

    image_id = uuid.uuid4().hex[:12]
    ext = Path(filename).suffix or ".png"
    local_filename = f"{image_id}{ext}"
    local_path = OUTPUT_DIR / local_filename

    with open(local_path, "wb") as f:
        f.write(image_bytes)

    return {
        "success": True,
        "image_id": image_id,
        "filename": local_filename,
        "path": str(local_path),
        "prompt": prompt,
        "model": model,
        "size": f"{width}x{height}",
    }


# ═══════════════════════════════════════════════════════════
# Route handlers (module pattern matching server.py dispatch)
# ═══════════════════════════════════════════════════════════

def handle_post(handler, strat, auth, path) -> bool:
    if not path.startswith("/api/image/"):
        return False

    profile_id = getattr(handler, '_profile_id', 0)
    db = strat.db

    # ── POST /api/image/generate ──
    if path == "/api/image/generate":
        data = read_json_body(handler)
        prompt = data.get("prompt", "").strip()
        if not prompt:
            error_response(handler, "Prompt required", 400)
            return True

        model = data.get("model", "flux")
        width = min(max(data.get("width", 768), 512), 1536)
        height = min(max(data.get("height", 1024), 512), 1536)
        seed = data.get("seed", -1)
        steps = data.get("steps")
        negative = data.get("negative_prompt", "")

        if model not in ("flux", "pony"):
            model = "flux"

        result = generate_image(prompt, negative, model, width, height, seed, steps)

        if result["success"]:
            db.insert_generated_image(
                result["image_id"], profile_id, prompt, model, width, height,
                result["filename"], result["path"],
                seed=seed, steps=steps
            )
            json_response(handler, result)
        else:
            error_response(handler, result["error"], 503)
        return True

    # ── POST /api/image/character-portrait ──
    if path == "/api/image/character-portrait":
        data = read_json_body(handler)
        name = data.get("character_name", "")
        description = data.get("physical_description", "")
        scenario = data.get("scenario", "")
        style = data.get("style", "anime")
        nsfw = data.get("nsfw", False)
        card_id = data.get("character_card_id")

        if not description:
            error_response(handler, "Physical description required", 400)
            return True

        prompt = character_to_image_prompt(
            {"name": name, "physical_description": description, "scenario": scenario},
            style=style, nsfw=nsfw,
        )
        model = select_image_model(style, nsfw)

        result = generate_image(prompt, model=model, width=768, height=1024)

        if result["success"]:
            db.insert_generated_image(
                result["image_id"], profile_id, prompt, model, 768, 1024,
                result["filename"], result["path"],
                character_card_id=card_id
            )
            json_response(handler, result)
        else:
            error_response(handler, result["error"], 503)
        return True

    return False


def handle_get(handler, strat, auth, path) -> bool:
    if not path.startswith("/api/image/"):
        return False

    profile_id = getattr(handler, '_profile_id', 0)
    db = strat.db

    # ── GET /api/image/gallery ──
    if path == "/api/image/gallery":
        images = db.get_user_images(profile_id)
        json_response(handler, {"images": images})
        return True

    # ── GET /api/image/<image_id> ──
    parts = path.split("/")
    if len(parts) == 4:
        image_id = parts[3]
        matches = list(OUTPUT_DIR.glob(f"{image_id}.*"))
        if not matches:
            handler.send_response(404)
            handler.end_headers()
            return True

        filepath = matches[0]
        content_type = "image/png" if filepath.suffix == ".png" else "image/jpeg"

        with open(filepath, "rb") as f:
            data = f.read()

        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "public, max-age=86400")
        handler.end_headers()
        handler.wfile.write(data)
        return True

    return False


def handle_delete(handler, strat, auth, path) -> bool:
    if not path.startswith("/api/image/"):
        return False

    parts = path.split("/")
    if len(parts) == 4:
        image_id = parts[3]
        matches = list(OUTPUT_DIR.glob(f"{image_id}.*"))
        for f in matches:
            f.unlink()
        strat.db.conn.execute("DELETE FROM generated_images WHERE id = ?", (image_id,))
        strat.db.conn.commit()
        json_response(handler, {"ok": True, "deleted": image_id})
        return True

    return False
