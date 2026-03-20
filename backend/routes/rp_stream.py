"""
RP Streaming — Ollama model selection, streaming, and think-block handling.

Extracted from rp_chat.py for parallelization. Contains:
- _check_model_exists() — verify model in Ollama
- select_rp_model() — A/B split model selection
- _stream_ollama() — SSE streaming from Ollama
- MAX_MESSAGE_LENGTH constant
"""

import hashlib
import json
import logging
import requests as req

from routes.helpers import sse_event, strip_think_blocks

logger = logging.getLogger("rp_stream")

MAX_MESSAGE_LENGTH = 10000  # Max characters per user message

# Cache verified models — no need to check /api/tags on every message
_verified_models: set[str] = set()


def compute_ngram_overlap(text_a: str, text_b: str, n: int = 5) -> float:
    """Compute proportion of n-grams in text_b that also appear in text_a.
    Returns 0.0-1.0 where 1.0 means text_b is entirely contained in text_a.
    Used for dedup detection — if a response repeats >40% of the previous
    response, we inject a warning on the NEXT turn.
    """
    if not text_a or not text_b:
        return 0.0
    words_a = text_a.lower().split()
    words_b = text_b.lower().split()
    if len(words_b) < n:
        return 0.0
    ngrams_a = set(tuple(words_a[i:i+n]) for i in range(len(words_a) - n + 1))
    ngrams_b = [tuple(words_b[i:i+n]) for i in range(len(words_b) - n + 1)]
    if not ngrams_b:
        return 0.0
    matches = sum(1 for ng in ngrams_b if ng in ngrams_a)
    return matches / len(ngrams_b)


def _check_model_exists(ollama_host: str, model: str) -> bool:
    """Check if a model is available in Ollama. Cached after first success."""
    if model in _verified_models:
        return True
    try:
        r = req.get(f"{ollama_host}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            if any(model in m for m in models):
                _verified_models.add(model)
                return True
    except Exception as e:
        logger.warning(f"Could not check Ollama models: {e}")
    return False


def select_rp_model(session_id: str, config: dict) -> str:
    """Select RP model with optional A/B split.

    Deterministic: same session_id always gets same model.
    Falls back to inference_model if RP model not available.
    """
    scoring_cfg = config.get("scoring", {})
    rp_cfg = config.get("rp", {})
    ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
    ab_split = rp_cfg.get("ab_split", 0.0)
    candidate = rp_cfg.get("candidate_model")
    fallback = scoring_cfg.get("inference_model", "qwen3.5:9b")

    rp_model = rp_cfg.get("model", scoring_cfg.get("rp_model", "stratos-rp-q8"))

    if ab_split > 0 and candidate:
        hash_val = int(hashlib.md5(session_id.encode()).hexdigest()[:8], 16)
        if (hash_val % 100) < (ab_split * 100):
            rp_model = candidate

    if not _check_model_exists(ollama_host, rp_model):
        logger.warning(f"RP model '{rp_model}' not found in Ollama, falling back to '{fallback}'")
        return fallback

    return rp_model


def _stream_ollama(handler, ollama_host: str, model: str, messages: list,
                   temperature: float = 0.85, num_predict: int = 350) -> str:
    """Stream Ollama response via SSE. Returns the full accumulated text."""
    try:
        r = req.post(
            f"{ollama_host}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "num_ctx": 32768,
                    "min_p": 0.05,
                    "top_k": 0,
                },
                "think": False,
            },
            timeout=180, stream=True
        )
        if r.status_code != 200:
            sse_event(handler, {"error": f"Ollama returned {r.status_code}"})
            return ""

        full_text = ""
        in_think = False
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_text += token
                    if '<think>' in token:
                        in_think = True
                    if '</think>' in token:
                        in_think = False
                        continue
                    if not in_think:
                        sse_event(handler, {"token": token})
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue

        full_text = strip_think_blocks(full_text)
        return full_text

    except Exception as e:
        logger.error(f"Ollama streaming error: {e}")
        sse_event(handler, {"error": str(e)})
        return ""
