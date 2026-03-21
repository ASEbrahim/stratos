"""
Media routes — proxy, file upload/list/delete, persona files.
Extracted from server.py (Sprint 5K Phase 1).
"""

import io
import json
import logging
import os
import struct
from urllib.parse import urlparse, parse_qs

import requests

logger = logging.getLogger("STRAT_OS")


def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 22050,
                channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """Wrap raw PCM audio data in a WAV header."""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_data)
    # RIFF header + fmt chunk + data chunk
    header = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE')
    fmt = struct.pack('<4sIHHIIHH', b'fmt ', 16, 1,  # PCM format
                      channels, sample_rate, byte_rate, block_align, bits_per_sample)
    data_header = struct.pack('<4sI', b'data', data_size)
    return header + fmt + data_header + pcm_data


def _send_json(handler, data, status=200):
    """Send a JSON response with proper headers."""
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def handle_get(handler, strat, auth, path):
    """Handle GET requests for media routes. Returns True if handled."""

    # Media proxy — routes through CF Worker for ISP-blocked content
    if path == "/api/proxy":
        target_url = parse_qs(urlparse(handler.path).query).get("url", [None])[0]
        if not target_url:
            from routes.helpers import error_response
            error_response(handler, "Missing url param", 400)
            return True

        # SSRF protection: validate URL scheme and block private/internal IPs
        from routes.url_validation import validate_url
        is_safe, err_msg = validate_url(target_url)
        if not is_safe:
            logger.warning(f"Proxy SSRF blocked: {err_msg} — url={target_url}")
            from routes.helpers import error_response
            error_response(handler, "Blocked URL", 403)
            return True

        worker_base = strat.config.get("proxy", {}).get("cloudflare_worker", "")
        blocked = strat.config.get("proxy", {}).get("blocked_domains", [])
        domain = urlparse(target_url).hostname or ""
        is_blocked = any(domain == b or domain.endswith("." + b) for b in blocked)

        if is_blocked and worker_base:
            fetch_url = f"{worker_base}/proxy?url={requests.utils.quote(target_url, safe='')}"
        else:
            fetch_url = target_url
        try:
            resp = requests.get(fetch_url, timeout=20, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Referer": urlparse(target_url).scheme + "://" + (urlparse(target_url).hostname or "") + "/",
            }, stream=True)

            # SSRF: re-validate after redirects to prevent redirect-to-internal bypass
            final_url = resp.url
            if final_url != fetch_url:
                from routes.url_validation import validate_url as _redir_validate
                _redir_safe, _redir_err = _redir_validate(final_url)
                if not _redir_safe:
                    resp.close()
                    logger.warning(f"Proxy SSRF blocked after redirect: {_redir_err} — url={final_url}")
                    from routes.helpers import error_response
                    error_response(handler, "Blocked URL (redirect)", 403)
                    return True

            ct = resp.headers.get("Content-Type", "application/octet-stream")
            handler.send_response(resp.status_code)
            handler.send_header("Content-Type", ct)
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.send_header("Cache-Control", "public, max-age=3600")
            handler.end_headers()
            _proxy_bytes = 0
            _PROXY_MAX_BYTES = 50 * 1024 * 1024  # 50MB limit for proxied content
            for chunk in resp.iter_content(8192):
                _proxy_bytes += len(chunk)
                if _proxy_bytes > _PROXY_MAX_BYTES:
                    logger.warning(f"Proxy response too large (>{_PROXY_MAX_BYTES} bytes), truncating: {target_url}")
                    break
                handler.wfile.write(chunk)
        except Exception as e:
            logger.warning(f"Proxy error: {e}")
            from routes.helpers import error_response
            error_response(handler, "Proxy fetch failed", 502)
        return True

    # ── TTS Voice List ─────────────────────────────
    if path == "/api/tts/voices":
        from processors.tts import TTSProcessor
        voices = TTSProcessor.get_available_voices()
        _send_json(handler, voices)
        return True

    # ── TTS Engine Status ──────────────────────────
    if path == "/api/tts/status":
        from processors.tts import TTSProcessor
        status = TTSProcessor.get_engine_status()
        _send_json(handler, status)
        return True

    # ── Persona Files GET endpoints ─────────────────
    if path == "/api/persona-files/read" or path == "/api/persona-files":
        from processors.persona_context import PersonaContextManager
        pcm = PersonaContextManager(strat.config, db=strat.db)
        parsed = urlparse(handler.path)
        params = parse_qs(parsed.query)
        persona = params.get('persona', [''])[0]
        filepath = params.get('path', ['/'])[0]

        if parsed.path == '/api/persona-files/read':
            content = pcm.read_file(handler._profile_id, persona, filepath)
            if content is not None:
                _send_json(handler, {"path": filepath, "content": content})
            else:
                _send_json(handler, {"error": "File not found"}, 404)
            return True
        elif parsed.path == '/api/persona-files':
            entries = pcm.list_files(handler._profile_id, persona, filepath)
            _send_json(handler, {"path": filepath, "entries": entries})
            return True

    return False


def handle_post(handler, strat, auth, path):
    """Handle POST requests for media routes. Returns True if handled."""

    # ── File Upload ──────────────────────────────────────────
    if path == "/api/files/upload":
        try:
            from processors.file_handler import FileHandler
            content_length = int(handler.headers.get('Content-Length', 0))
            if content_length > 10 * 1024 * 1024:
                _send_json(handler, {"error": "File too large (max 10MB)"}, 413)
                return True
            file_data = handler.rfile.read(content_length)
            # Expect multipart or raw upload with X-Filename header
            filename = handler.headers.get('X-Filename', 'upload.txt')

            # Filename sanitization: strip path separators and null bytes
            filename = os.path.basename(filename).replace('\x00', '')
            if not filename or filename.startswith('.'):
                filename = 'upload.txt'
            persona = handler.headers.get('X-Persona', '')
            fh = FileHandler(strat.config, db=strat.db)
            result = fh.save_file(handler._profile_id, filename, file_data, persona=persona)
            if result:
                _send_json(handler, {"ok": True, "file": {
                    "id": result.get("id"),
                    "filename": result["filename"],
                    "type": result["file_type"],
                    "has_text": bool(result.get("content_text")),
                }})
            else:
                _send_json(handler, {"error": "Upload failed"}, 400)
        except Exception as e:
            logger.error(f"File upload error: {e}")
            _send_json(handler, {"error": "Upload failed"}, 500)
        return True

    # ── File List/Search ─────────────────────────────────────
    if path == "/api/files/list":
        try:
            from processors.file_handler import FileHandler
            body = json.loads(handler.rfile.read(int(handler.headers.get('Content-Length', 0))).decode()) if int(handler.headers.get('Content-Length', 0)) > 0 else {}
            fh = FileHandler(strat.config, db=strat.db)
            query = body.get("query", "")
            persona = body.get("persona", "")
            if query:
                files = fh.search_files(handler._profile_id, query, persona=persona)
            else:
                files = fh.list_files(handler._profile_id, persona=persona)
            _send_json(handler, {"files": files})
        except Exception as e:
            logger.error(f"File list error: {e}")
            _send_json(handler, {"error": "Failed to list files"}, 500)
        return True

    # ── TTS — Dual-engine Text-to-Speech ──────────────
    if path == "/api/tts":
        try:
            body = json.loads(handler.rfile.read(int(handler.headers.get('Content-Length', 0))).decode()) if int(handler.headers.get('Content-Length', 0)) > 0 else {}
            text = (body.get('text') or '')[:5000].strip()
            if not text:
                _send_json(handler, {"error": "No text provided"}, 400)
                return True

            from processors.tts import TTSProcessor, PERSONA_DEFAULT_VOICES

            voice = body.get('voice')
            language = body.get('language')
            speed = float(body.get('speed', 1.0))
            persona = body.get('persona')

            # Persona-based default voice
            if not voice and persona and persona in PERSONA_DEFAULT_VOICES:
                voice = PERSONA_DEFAULT_VOICES[persona]

            # User's stored voice preference (sent by frontend)
            if not voice:
                stored_voice = body.get('preferred_voice')
                if stored_voice:
                    voice = stored_voice

            speed = max(0.5, min(2.0, speed))

            result = TTSProcessor.synthesize(text, voice=voice, language=language, speed=speed)

            if 'error' in result:
                _send_json(handler, result, 503)
                return True

            # Edge-TTS returns MP3, Kokoro returns WAV
            content_type = "audio/mpeg" if result['engine'] == 'edge' else "audio/wav"

            handler.send_response(200)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(result['audio'])))
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.send_header("X-TTS-Engine", result['engine'])
            handler.send_header("X-TTS-Voice", result['voice'])
            handler.send_header("X-TTS-Language", result['language'])
            handler.send_header("X-TTS-Processing", str(result['processing_seconds']))
            handler.end_headers()
            handler.wfile.write(result['audio'])
        except Exception as e:
            logger.error(f"TTS endpoint error: {e}")
            _send_json(handler, {"error": "TTS failed"}, 500)
        return True

    # ── TTS Preview ──────────────────────────────────
    if path == "/api/tts/preview":
        try:
            body = json.loads(handler.rfile.read(int(handler.headers.get('Content-Length', 0))).decode()) if int(handler.headers.get('Content-Length', 0)) > 0 else {}
            from processors.tts import TTSProcessor, KOKORO_VOICES, EDGE_TTS_VOICES

            voice = body.get('voice', 'af_heart')

            previews = {
                'en': "Welcome to StratOS. I'm your strategic intelligence assistant.",
                'ar': '\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645 \u0641\u064a \u0633\u062a\u0631\u0627\u062a\u0648\u0633. \u0623\u0646\u0627 \u0645\u0633\u0627\u0639\u062f\u0643\u0645 \u0627\u0644\u0630\u0643\u064a.',
                'ja': 'StratOS\u3078\u3088\u3046\u3053\u305d\u3002\u79c1\u306f\u3042\u306a\u305f\u306e\u6226\u7565\u60c5\u5831\u30a2\u30b7\u30b9\u30bf\u30f3\u30c8\u3067\u3059\u3002',
                'zh': '\u6b22\u8fce\u4f7f\u7528StratOS\u3002\u6211\u662f\u60a8\u7684\u6218\u7565\u60c5\u62a5\u52a9\u624b\u3002',
                'fr': "Bienvenue sur StratOS. Je suis votre assistant d'intelligence strat\u00e9gique.",
                'ko': 'StratOS\uc5d0 \uc624\uc2e0 \uac83\uc744 \ud658\uc601\ud569\ub2c8\ub2e4. \uc800\ub294 \ub2f9\uc2e0\uc758 \uc804\ub7b5 \uc815\ubcf4 \ubcf4\uc870\uc785\ub2c8\ub2e4.',
                'hi': 'StratOS \u092e\u0947\u0902 \u0906\u092a\u0915\u093e \u0938\u094d\u0935\u093e\u0917\u0924 \u0939\u0948\u0964 \u092e\u0948\u0902 \u0906\u092a\u0915\u093e \u0930\u0923\u0928\u0940\u0924\u093f\u0915 \u092c\u0941\u0926\u094d\u0927\u093f\u092e\u0924\u094d\u0924\u093e \u0938\u0939\u093e\u092f\u0915 \u0939\u0942\u0902\u0964',
                'it': "Benvenuto su StratOS. Sono il tuo assistente di intelligence strategica.",
                'pt': "Bem-vindo ao StratOS. Sou seu assistente de intelig\u00eancia estrat\u00e9gica.",
            }

            lang = 'en'
            if voice in KOKORO_VOICES:
                lang = KOKORO_VOICES[voice]['lang']
            elif voice in EDGE_TTS_VOICES or voice.startswith('ar-'):
                lang = 'ar'

            preview_text = previews.get(lang, previews['en'])

            result = TTSProcessor.synthesize(preview_text, voice=voice)

            if 'error' in result:
                _send_json(handler, result, 503)
                return True

            content_type = "audio/mpeg" if result.get('engine') == 'edge' else "audio/wav"
            handler.send_response(200)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(result['audio'])))
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(result['audio'])
        except Exception as e:
            logger.error(f"TTS preview error: {e}")
            _send_json(handler, {"error": "Preview failed"}, 500)
        return True

    # ── TTS Custom Voice Upload ──────────────────────
    if path == "/api/tts/voices/custom":
        try:
            import re as _re
            voice_name = handler.headers.get('X-Voice-Name', 'custom_voice')
            voice_name = _re.sub(r'[^a-zA-Z0-9_]', '_', voice_name).lower()

            content_length = int(handler.headers.get('Content-Length', 0))
            if content_length == 0 or content_length > 10 * 1024 * 1024:
                _send_json(handler, {"error": "Invalid audio size (max 10MB)"}, 400)
                return True

            audio_data = handler.rfile.read(content_length)

            # Validate audio format (WAV/OGG/MP3 magic bytes)
            if not (audio_data[:4] == b'RIFF' or audio_data[:4] == b'OggS' or
                    audio_data[:3] == b'ID3' or audio_data[:2] == b'\xff\xfb'):
                _send_json(handler, {"error": "Invalid audio format (expected WAV, OGG, or MP3)"}, 400)
                return True

            voices_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'tts_voices')
            os.makedirs(voices_dir, exist_ok=True)

            voice_path = os.path.join(voices_dir, f"{voice_name}.wav")

            # Path traversal protection
            if not os.path.realpath(voice_path).startswith(os.path.realpath(voices_dir)):
                _send_json(handler, {"error": "Invalid voice name"}, 400)
                return True

            with open(voice_path, 'wb') as f:
                f.write(audio_data)

            _send_json(handler, {"ok": True, "voice_id": voice_name, "message": f"Custom voice '{voice_name}' saved."})
        except Exception as e:
            logger.error(f"Custom voice upload error: {e}")
            _send_json(handler, {"error": "Upload failed"}, 500)
        return True

    # ── STT — Speech-to-Text via faster-whisper ──────
    if path == "/api/stt":
        from processors.stt import STTProcessor
        available, msg = STTProcessor.is_available()
        if not available:
            _send_json(handler, {"error": f"Speech-to-text unavailable: {msg}"}, 503)
            return True
        content_length = int(handler.headers.get('Content-Length', 0))
        if content_length == 0:
            _send_json(handler, {"error": "No audio data received"}, 400)
            return True
        if content_length > STTProcessor.MAX_AUDIO_BYTES:
            _send_json(handler, {"error": f"Audio too large ({content_length} bytes). Max: {STTProcessor.MAX_AUDIO_BYTES}"}, 413)
            return True
        audio_bytes = handler.rfile.read(content_length)
        language_hint = handler.headers.get('X-Language-Hint', None)
        if language_hint and language_hint not in ('en', 'ar', 'ja', 'ko', 'zh', 'fr', 'de', 'es'):
            language_hint = None
        try:
            result = STTProcessor.transcribe(audio_bytes, language_hint=language_hint)
            _send_json(handler, result)
        except ValueError as e:
            logger.error(f"STT validation error: {e}", exc_info=True)
            _send_json(handler, {"error": "Invalid audio input"}, 400)
        except RuntimeError as e:
            logger.error(f"STT runtime error: {e}", exc_info=True)
            _send_json(handler, {"error": "Internal server error"}, 500)
        except Exception as e:
            logger.error(f"STT error: {e}", exc_info=True)
            _send_json(handler, {"error": "Transcription failed. Please try again."}, 500)
        return True

    # ── Persona Files POST endpoints ────────────────
    if path in ('/api/persona-files/write', '/api/persona-files/mkdir'):
        from processors.persona_context import PersonaContextManager
        pcm = PersonaContextManager(strat.config, db=strat.db)
        body = json.loads(handler.rfile.read(int(handler.headers.get('Content-Length', 0))).decode()) if int(handler.headers.get('Content-Length', 0)) > 0 else {}

        if path == '/api/persona-files/write':
            persona = body.get('persona', '')
            filepath = body.get('path', '')
            content = body.get('content', '')
            ok = pcm.write_file(handler._profile_id, persona, filepath, content)
            _send_json(handler, {"ok": ok})
            return True

        if path == '/api/persona-files/mkdir':
            persona = body.get('persona', '')
            dirpath = body.get('path', '')
            ok = pcm.make_dir(handler._profile_id, persona, dirpath)
            _send_json(handler, {"ok": ok})
            return True

    return False


def handle_delete(handler, strat, auth, path):
    """Handle DELETE requests for media routes. Returns True if handled."""

    # ── File Delete ──
    if path.startswith("/api/files/"):
        try:
            file_id = int(handler.path.split("/")[-1])
            from processors.file_handler import FileHandler
            fh = FileHandler(strat.config, db=strat.db)
            if fh.delete_file(handler._profile_id, file_id):
                _send_json(handler, {"ok": True})
            else:
                _send_json(handler, {"error": "File not found"}, 404)
        except (ValueError, IndexError):
            _send_json(handler, {"error": "Invalid file ID"}, 400)
        except Exception as e:
            logger.error(f"File delete error: {e}")
            _send_json(handler, {"error": "Delete failed"}, 500)
        return True

    # ── Persona Files Delete ──
    if path.startswith("/api/persona-files"):
        from processors.persona_context import PersonaContextManager
        pcm = PersonaContextManager(strat.config, db=strat.db)
        params = parse_qs(urlparse(handler.path).query)
        persona = params.get('persona', [''])[0]
        filepath = params.get('path', [''])[0]
        if persona and filepath:
            ok = pcm.delete_file(handler._profile_id, persona, filepath)
            _send_json(handler, {"ok": ok})
        else:
            _send_json(handler, {"error": "Missing persona or path"}, 400)
        return True

    return False
