"""
LLM Provider Abstraction — routes calls to Ollama or Gemini Flash
based on config/environment. Drop-in replacement for _call_ollama() sites.

Provider detection:
  - config.llm.provider = 'ollama' | 'gemini' | 'auto'
  - 'auto': uses Gemini if GEMINI_API_KEY is set and Ollama is unreachable
  - Local dev with Ollama running → uses Ollama (zero config change)
  - VPS with GEMINI_API_KEY → uses Gemini Flash automatically

Usage:
    from llm_provider import call_llm, stream_llm

    # Blocking call
    result = call_llm(config, "What is 2+2?", system="You are helpful.", max_tokens=100)

    # Streaming to SSE handler
    text = stream_llm(config, messages, handler, system="...", max_tokens=500)
"""

import json
import logging
import os
import re
import requests

logger = logging.getLogger("LLM")

# ═══════════════════════════════════════════════════
# PROVIDER DETECTION (cached after first call)
# ═══════════════════════════════════════════════════

_provider_cache = None
_gemini_key = os.environ.get('GEMINI_API_KEY', '')
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Default models per use case
_DEFAULT_GEMINI_MODELS = {
    'scoring': 'gemini-2.5-flash',
    'chat': 'gemini-2.5-flash-lite',
    'default': 'gemini-2.5-flash-lite',
}


def get_provider(config):
    """Determine provider: 'gemini' or 'ollama'. Cached after first call."""
    global _provider_cache
    if _provider_cache is not None:
        return _provider_cache

    explicit = (config or {}).get('llm', {}).get('provider', 'auto')
    if explicit in ('gemini', 'ollama'):
        _provider_cache = explicit
        logger.info(f"LLM provider: {explicit} (explicit config)")
        return explicit

    # Auto-detect
    if _gemini_key:
        # Check if Ollama is reachable
        host = (config or {}).get('rp', {}).get('ollama_host',
               (config or {}).get('scoring', {}).get('ollama_host', 'http://localhost:11434'))
        try:
            r = requests.get(f"{host}/api/tags", timeout=2)
            if r.status_code == 200:
                _provider_cache = 'ollama'
                logger.info("LLM provider: ollama (auto — Ollama reachable)")
                return 'ollama'
        except Exception:
            pass
        _provider_cache = 'gemini'
        logger.info("LLM provider: gemini (auto — Ollama unreachable, GEMINI_API_KEY set)")
        return 'gemini'

    _provider_cache = 'ollama'
    logger.info("LLM provider: ollama (auto — no GEMINI_API_KEY)")
    return 'ollama'


def reset_provider_cache():
    """Reset cached provider (for testing or config reload)."""
    global _provider_cache
    _provider_cache = None


def _get_ollama_host(config):
    """Get Ollama host from config."""
    return (config or {}).get('rp', {}).get('ollama_host',
           (config or {}).get('scoring', {}).get('ollama_host', 'http://localhost:11434'))


def _get_gemini_model(config, use_case='default'):
    """Get Gemini model name for a use case."""
    llm_cfg = (config or {}).get('llm', {})
    # Check config overrides first
    if use_case == 'scoring' and llm_cfg.get('scoring_model'):
        return llm_cfg['scoring_model']
    if use_case == 'chat' and llm_cfg.get('chat_model'):
        return llm_cfg['chat_model']
    if llm_cfg.get('model'):
        return llm_cfg['model']
    return _DEFAULT_GEMINI_MODELS.get(use_case, _DEFAULT_GEMINI_MODELS['default'])


# ═══════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════

def call_llm(config, messages, model=None, system=None,
             max_tokens=500, temperature=0.7, timeout=60,
             think=None, use_case='default'):
    """
    Blocking LLM call. Returns response text (str) or None on failure.

    Args:
        config: StratOS config dict
        messages: List of {"role": "user"|"assistant", "content": str}
                  OR a single string (converted to user message)
        model: Model name override (Ollama model name)
        system: System prompt (optional)
        max_tokens: Max output tokens
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        think: Ollama-specific think mode (None=default, False=disable)
        use_case: 'scoring'|'chat'|'default' — selects Gemini model tier
    """
    provider = get_provider(config)

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if provider == 'gemini':
        return _call_gemini(config, messages, system, max_tokens,
                           temperature, timeout, use_case)
    else:
        return _call_ollama(config, messages, model, system,
                           max_tokens, temperature, timeout, think)


def stream_llm(config, messages, handler, model=None, system=None,
               max_tokens=500, temperature=0.7, timeout=120,
               think=None, use_case='default'):
    """
    Streaming LLM call. Sends SSE events to handler.
    Returns accumulated full text (str) or None on failure.

    Each token sent as: data: {"token": "..."}\n\n
    """
    provider = get_provider(config)

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if provider == 'gemini':
        return _stream_gemini(config, messages, handler, system,
                             max_tokens, temperature, timeout, use_case)
    else:
        return _stream_ollama(config, messages, handler, model, system,
                             max_tokens, temperature, timeout, think)


# ═══════════════════════════════════════════════════
# GEMINI FLASH IMPLEMENTATION
# ═══════════════════════════════════════════════════

def _to_gemini_contents(messages):
    """Convert Ollama/OpenAI message format to Gemini contents."""
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            continue  # handled via systemInstruction
        gemini_role = "user" if role == "user" else "model"
        contents.append({
            "role": gemini_role,
            "parts": [{"text": msg["content"]}]
        })
    # Gemini requires at least one message
    if not contents:
        contents = [{"role": "user", "parts": [{"text": ""}]}]
    return contents


def _call_gemini(config, messages, system, max_tokens, temperature, timeout, use_case):
    """Blocking Gemini API call."""
    model = _get_gemini_model(config, use_case)
    url = f"{_GEMINI_BASE}/{model}:generateContent?key={_gemini_key}"

    body = {
        "contents": _to_gemini_contents(messages),
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        }
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    try:
        r = requests.post(url, json=body, timeout=timeout)
        if r.status_code != 200:
            logger.warning(f"Gemini {model} error {r.status_code}: {r.text[:200]}")
            return None
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            logger.warning(f"Gemini {model}: no candidates in response")
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        text = parts[0].get("text", "") if parts else ""
        # Strip any think blocks (safety net)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return text
    except requests.Timeout:
        logger.warning(f"Gemini {model} timeout ({timeout}s)")
        return None
    except Exception as e:
        logger.error(f"Gemini {model} call failed: {e}")
        return None


def _stream_gemini(config, messages, handler, system, max_tokens, temperature, timeout, use_case):
    """Streaming Gemini API call → SSE events to handler."""
    model = _get_gemini_model(config, use_case)
    url = f"{_GEMINI_BASE}/{model}:streamGenerateContent?alt=sse&key={_gemini_key}"

    body = {
        "contents": _to_gemini_contents(messages),
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        }
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    full_text = ""
    try:
        r = requests.post(url, json=body, timeout=timeout, stream=True)
        if r.status_code != 200:
            logger.warning(f"Gemini {model} stream error {r.status_code}: {r.text[:200]}")
            return None

        for line in r.iter_lines():
            if not line:
                continue
            line_str = line.decode('utf-8', errors='replace')
            if not line_str.startswith('data: '):
                continue
            try:
                chunk = json.loads(line_str[6:])
                candidates = chunk.get("candidates", [])
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    continue
                token = parts[0].get("text", "")
                if token:
                    full_text += token
                    handler.wfile.write(f'data: {json.dumps({"token": token})}\n\n'.encode())
                    handler.wfile.flush()
            except (json.JSONDecodeError, KeyError):
                continue

        # Strip think blocks from accumulated text
        full_text = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL).strip()
        return full_text
    except requests.Timeout:
        logger.warning(f"Gemini {model} stream timeout ({timeout}s)")
        return full_text or None
    except Exception as e:
        logger.error(f"Gemini {model} stream failed: {e}")
        return full_text or None


# ═══════════════════════════════════════════════════
# OLLAMA IMPLEMENTATION (for local development)
# ═══════════════════════════════════════════════════

def _call_ollama(config, messages, model, system, max_tokens, temperature, timeout, think):
    """Blocking Ollama /api/chat call."""
    host = _get_ollama_host(config)
    model = model or (config or {}).get('scoring', {}).get('inference_model', 'qwen3.5:9b')

    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)

    body = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    if think is not None:
        body["think"] = think

    try:
        r = requests.post(f"{host}/api/chat", json=body, timeout=timeout)
        if r.status_code != 200:
            logger.warning(f"Ollama {model} error {r.status_code}")
            return None
        text = r.json().get("message", {}).get("content", "")
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return text
    except requests.Timeout:
        logger.warning(f"Ollama {model} timeout ({timeout}s)")
        return None
    except Exception as e:
        logger.error(f"Ollama {model} call failed: {e}")
        return None


def _stream_ollama(config, messages, handler, model, system, max_tokens, temperature, timeout, think):
    """Streaming Ollama /api/chat → SSE events to handler."""
    host = _get_ollama_host(config)
    model = model or (config or {}).get('scoring', {}).get('inference_model', 'qwen3.5:9b')

    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)

    body = {
        "model": model,
        "messages": msgs,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    if think is not None:
        body["think"] = think

    full_text = ""
    try:
        r = requests.post(f"{host}/api/chat", json=body, timeout=timeout, stream=True)
        if r.status_code != 200:
            logger.warning(f"Ollama {model} stream error {r.status_code}")
            return None

        in_think = False
        for line in r.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("message", {}).get("content", "")

            # Skip think blocks
            if '<think>' in token:
                in_think = True
            if in_think:
                if '</think>' in token:
                    in_think = False
                    token = token.split('</think>', 1)[-1]
                else:
                    continue

            if token:
                full_text += token
                handler.wfile.write(f'data: {json.dumps({"token": token})}\n\n'.encode())
                handler.wfile.flush()

            if chunk.get("done"):
                break

        return full_text
    except requests.Timeout:
        logger.warning(f"Ollama {model} stream timeout ({timeout}s)")
        return full_text or None
    except Exception as e:
        logger.error(f"Ollama {model} stream failed: {e}")
        return full_text or None
