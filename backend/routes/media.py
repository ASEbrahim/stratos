"""
Media routes — proxy, file upload/list/delete, persona files.
Extracted from server.py (Sprint 5K Phase 1).
"""

import json
import logging
from urllib.parse import urlparse, parse_qs

import requests

logger = logging.getLogger("STRAT_OS")


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
            ct = resp.headers.get("Content-Type", "application/octet-stream")
            handler.send_response(resp.status_code)
            handler.send_header("Content-Type", ct)
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.send_header("Cache-Control", "public, max-age=3600")
            handler.end_headers()
            for chunk in resp.iter_content(8192):
                handler.wfile.write(chunk)
        except Exception as e:
            logger.warning(f"Proxy error: {e}")
            from routes.helpers import error_response
            error_response(handler, f"Proxy error: {e}", 502)
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
