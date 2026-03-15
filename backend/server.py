"""
HTTP server and route dispatch for StratOS.

Contains CORSHandler (all API routes), ThreadedHTTPServer,
and start_server() which wires everything together.

Extracted from main.py:serve_frontend() (Sprint 4, A1.2).
Route handlers split into routes/ modules (Sprint 5K Phase 1).
"""

import gzip as _gzip_mod
import json
import logging
import mimetypes
import os
import sys
import signal
import time
import threading
import yaml
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

from routes.agent import handle_agent_chat, handle_ask, handle_suggest_context
from routes.auth import handle_auth_routes
from routes.generate import handle_generate_profile
from routes.wizard import handle_wizard_preselect, handle_wizard_tab_suggest, handle_wizard_rv_items
from routes.helpers import json_response, error_response
from routes.config import handle_config_save
from routes import feeds, media, data_endpoints, controls, youtube_endpoints, persona_data, dev_endpoints, rp_chat, image_gen, character_cards
from email_service import EmailService

logger = logging.getLogger("STRAT_OS")

# File extensions eligible for gzip compression on static file serving
_GZIP_TYPES = frozenset({'.js', '.css', '.html', '.json', '.svg', '.txt', '.xml'})

# In-memory gzip cache: {filepath: (mtime, compressed_bytes)}
_gzip_cache = {}
_gzip_cache_lock = threading.Lock()


def _send_json(handler, data, status=200):
    """Send a JSON response with proper headers."""
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def create_handler(strat, auth, frontend_dir, output_dir):
    """Create the HTTP request handler class with bound references.

    Uses closure to bind strat (StratOS instance), auth (AuthManager),
    frontend_dir, and output_dir to the handler class.
    """

    email_service = EmailService(strat.config)

    def _get_profile_id(token):
        """Resolve profile_id from auth session token."""
        if not token:
            return None
        cursor = strat.db.conn.cursor()
        cursor.execute("SELECT profile_id FROM sessions WHERE token = ?", (token,))
        row = cursor.fetchone()
        return row[0] if row else None

    # Cache: device_id → profile_id (avoid DB lookup on every request)
    _device_profile_cache = {}

    def _resolve_device_profile(db, device_id: str) -> int:
        """Resolve or create an anonymous profile from a device ID.

        First request from a new device: creates a user + profile in DB.
        Subsequent requests: returns cached profile_id.
        """
        if device_id in _device_profile_cache:
            return _device_profile_cache[device_id]

        import time as _time
        for _attempt in range(3):
            try:
                cursor = db.conn.cursor()
                # Check if device already has a user
                cursor.execute(
                    "SELECT u.id, p.id FROM users u JOIN profiles p ON p.user_id = u.id "
                    "WHERE u.email = ? LIMIT 1",
                    (f"anon-{device_id}@device.local",)
                )
                row = cursor.fetchone()
                if row:
                    _device_profile_cache[device_id] = row[1]
                    return row[1]

                # Create anonymous user + profile
                import hashlib
                cursor.execute(
                    "INSERT INTO users (email, password_hash, display_name, is_admin) VALUES (?, ?, ?, ?)",
                    (f"anon-{device_id}@device.local", "anon-no-password", f"User-{device_id[:8]}", False)
                )
                user_id = cursor.lastrowid
                cursor.execute(
                    "INSERT INTO profiles (user_id, name, is_default) VALUES (?, ?, ?)",
                    (user_id, "default", True)
                )
                profile_id = cursor.lastrowid
                db.conn.commit()
                logger.info(f"Created anonymous profile {profile_id} for device {device_id[:12]}...")
                _device_profile_cache[device_id] = profile_id
                return profile_id
            except Exception as e:
                if 'locked' in str(e) and _attempt < 2:
                    _time.sleep(0.5)
                    continue
                if 'UNIQUE' in str(e):
                    # Race condition — another thread created it. Retry lookup.
                    db.conn.rollback()
                    cursor.execute(
                        "SELECT p.id FROM users u JOIN profiles p ON p.user_id = u.id "
                        "WHERE u.email = ? LIMIT 1",
                        (f"anon-{device_id}@device.local",)
                    )
                    row = cursor.fetchone()
                    if row:
                        _device_profile_cache[device_id] = row[0]
                        return row[0]
                raise
        return 0

    class CORSHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(frontend_dir), **kwargs)

        def end_headers(self):
            # Disable browser caching for HTML/JS/CSS files so UI updates appear immediately
            if hasattr(self, 'path') and (self.path.endswith('.html') or self.path.endswith('.js') or self.path.endswith('.css') or self.path == '/' or self.path == ''):
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
            # CORS: reflect origin if in allowlist, or wildcard if ["*"] (default)
            cors_origins = strat.config.get("system", {}).get("cors_origins", ["*"])
            if "*" in cors_origins:
                self.send_header("Access-Control-Allow-Origin", "*")
            else:
                req_origin = self.headers.get("Origin", "")
                if req_origin in cors_origins:
                    self.send_header("Access-Control-Allow-Origin", req_origin)
                    self.send_header("Vary", "Origin")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
            super().end_headers()

        def _client_ip(self):
            """Get client IP from X-Forwarded-For header or socket address."""
            forwarded = self.headers.get("X-Forwarded-For", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
            return self.client_address[0] if self.client_address else ""

        def do_GET(self):
            self._profile_id = 0
            self._session_profile = None

            # --- Rate limiting (before any dispatch) ---
            if auth.rate_limited(self.path, self._client_ip()):
                _send_json(self, {"error": "Too many requests. Try again shortly."}, 429)
                return

            # --- Auth check endpoint (always public) ---
            if self.path == "/api/auth-check":
                token = self.headers.get('X-Auth-Token', '')
                device_id = self.headers.get('X-Device-Id', '').strip()
                is_authenticated = auth.validate_session(token)
                # Device-scoped profiles for this device's login screen
                device_profiles = auth.list_profiles(device_id=device_id) if device_id else []
                # Total profile count (so frontend knows if "Login" button should show)
                all_profiles = auth.list_profiles()
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                resp = {
                    "auth_required": True,
                    "authenticated": is_authenticated,
                    "profiles": device_profiles,
                    "all_profiles": all_profiles,
                    "active_profile": auth.get_session_profile(token) if is_authenticated else "",
                    "needs_registration": len(all_profiles) == 0,
                    "total_profiles": len(all_profiles),
                }
                self.wfile.write(json.dumps(resp).encode())
                return

            # --- New email-based auth routes (self-authenticating) ---
            if self.path.startswith("/api/auth/") or self.path == "/api/admin/users" or self.path == "/api/profiles":
                try:
                    if handle_auth_routes(self, "GET", self.path, {}, strat.db, strat, _send_json, email_service):
                        return
                except Exception as e:
                    logger.error(f"Auth route error: {e}")
                    _send_json(self, {"error": "Server busy, please try again."}, 503)
                    return

            # --- Auth enforcement for API endpoints ---
            # /api/proxy exempt: <img> tags can't send X-Auth-Token headers
            _auth_path = self.path.split('?')[0]
            if _auth_path.startswith('/api/') and not _auth_path.startswith('/api/proxy') and _auth_path not in auth.AUTH_EXEMPT:
                token = self.headers.get('X-Auth-Token', '')
                device_id = self.headers.get('X-Device-Id', '')
                # Fallback: accept token as query param for file downloads (e.g. export)
                if not token:
                    _qs = parse_qs(urlparse(self.path).query)
                    token = _qs.get('token', [''])[0]

                if token and auth.validate_session(token):
                    # Logged-in user
                    _session_profile = auth.get_session_profile(token)
                    if _session_profile:
                        strat.ensure_profile(_session_profile)
                    try:
                        cursor = strat.db.conn.cursor()
                        cursor.execute("SELECT profile_id FROM sessions WHERE token = ?", (token,))
                        _pid_row = cursor.fetchone()
                        if _pid_row and _pid_row[0]:
                            self._profile_id = _pid_row[0]
                    except Exception:
                        pass
                    self._session_profile = _session_profile
                elif device_id:
                    # Anonymous user — resolve or create profile from device ID
                    try:
                        self._profile_id = _resolve_device_profile(strat.db, device_id)
                    except Exception as e:
                        logger.error(f"Device profile resolution failed: {e}")
                        self._profile_id = 0
                    self._session_profile = None
                else:
                    self.send_response(401)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": "Authentication required"}')
                    return

            # SSE event stream — replaces polling for real-time updates (keep in server.py: long-lived)
            if self.path.startswith("/api/events"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                # Send current status immediately so client doesn't start blind
                init_data = json.dumps({"type": "status", **strat.scan_status})
                self.wfile.write(f"event: status\ndata: {init_data}\n\n".encode())
                self.wfile.flush()

                # Resolve profile_id from token query param for profile-scoped SSE
                sse_pid = 0
                try:
                    qs = parse_qs(urlparse(self.path).query)
                    sse_token = (qs.get('token') or [''])[0]
                    if sse_token and auth.validate_session(sse_token):
                        sse_pid = _get_profile_id(sse_token) or 0
                except Exception:
                    pass
                strat.sse_register(self.wfile, profile_id=sse_pid)
                try:
                    # Keep alive with heartbeats every 15s
                    while True:
                        time.sleep(15)
                        self.wfile.write(": heartbeat\n\n".encode())
                        self.wfile.flush()
                except Exception:
                    pass
                finally:
                    strat.sse_unregister(self.wfile)
                return

            # --- Route dispatch to modules ---
            clean_path = self.path.split('?')[0]

            if rp_chat.handle_get(self, strat, auth, clean_path): return
            if image_gen.handle_get(self, strat, auth, clean_path): return
            if character_cards.handle_get(self, strat, auth, clean_path): return
            if controls.handle_get(self, strat, auth, clean_path): return
            if feeds.handle_get(self, strat, auth, clean_path): return
            if youtube_endpoints.handle_get(self, strat, auth, clean_path): return
            if persona_data.handle_get(self, strat, auth, clean_path): return
            if media.handle_get(self, strat, auth, clean_path): return
            if data_endpoints.handle_get(self, strat, auth, clean_path, output_dir=output_dir): return
            if dev_endpoints.handle_get(self, strat, auth, clean_path): return

            # --- Static file serving with gzip compression ---
            accepts_gzip = 'gzip' in self.headers.get('Accept-Encoding', '')
            ext = os.path.splitext(self.path.split('?')[0])[1].lower()

            if accepts_gzip and ext in _GZIP_TYPES:
                # Translate URL path to filesystem path
                path = self.translate_path(self.path)
                if os.path.isdir(path):
                    path = os.path.join(path, 'index.html')
                    ext = '.html'
                if os.path.isfile(path):
                    try:
                        mtime = os.path.getmtime(path)
                        with _gzip_cache_lock:
                            cached = _gzip_cache.get(path)
                        if cached and cached[0] == mtime:
                            compressed = cached[1]
                        else:
                            with open(path, 'rb') as f:
                                raw = f.read()
                            compressed = _gzip_mod.compress(raw, compresslevel=6)
                            with _gzip_cache_lock:
                                _gzip_cache[path] = (mtime, compressed)
                        ctype = mimetypes.guess_type(path)[0] or 'application/octet-stream'
                        self.send_response(200)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Content-Encoding", "gzip")
                        self.send_header("Content-Length", str(len(compressed)))
                        self.end_headers()
                        self.wfile.write(compressed)
                        return
                    except Exception:
                        pass  # Fall through to default handler

            return super().do_GET()

        def do_POST(self):
            self._profile_id = 0
            self._session_profile = None

            # --- Rate limiting (before any dispatch) ---
            if auth.rate_limited(self.path, self._client_ip()):
                _send_json(self, {"error": "Too many requests. Try again shortly."}, 429)
                return

            # --- New email-based auth routes (self-authenticating) ---
            if self.path.startswith("/api/auth/") or self.path.startswith("/api/profiles") or self.path.startswith("/api/admin/"):
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length) if content_length else b'{}'
                try:
                    data = json.loads(post_data.decode('utf-8')) if post_data.strip() else {}
                except (json.JSONDecodeError, UnicodeDecodeError):
                    data = {}
                # Username → email resolution for login
                if self.path == "/api/auth/login" and data.get("email") and "@" not in data["email"]:
                    try:
                        cursor = strat.db.conn.cursor()
                        cursor.execute(
                            "SELECT email FROM users WHERE display_name = ? COLLATE NOCASE LIMIT 1",
                            (data["email"],)
                        )
                        row = cursor.fetchone()
                        if row:
                            data["email"] = row[0]
                    except Exception:
                        pass  # Fall through — auth will show "user not found"
                try:
                    if handle_auth_routes(self, "POST", self.path, data, strat.db, strat, _send_json, email_service):
                        return
                    # Auth handler declined — stash parsed body for downstream handlers
                    self._stashed_post_data = data
                except Exception as e:
                    logger.error(f"Auth route error: {e}")
                    _send_json(self, {"error": "Server busy, please try again."}, 503)
                    return

            # --- Login endpoint (always public, legacy PIN-based) ---
            if self.path == "/api/auth":
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    profile_name = data.get('profile', '').strip()
                    pin = data.get('pin', '')

                    if not profile_name:
                        self.send_response(400)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"error": "Profile name required"}')
                        return

                    result = auth.verify_profile_pin(profile_name, pin)
                    if result is None:
                        self.send_response(404)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"error": "Profile not found"}')
                        return
                    if not result:
                        self.send_response(403)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"error": "Invalid PIN"}')
                        return

                    # Load profile config into live system
                    auth.load_profile_config(profile_name, strat)

                    token = auth.create_session(profile_name)
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "token": token,
                        "profile": profile_name
                    }).encode())
                except Exception:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": "Bad request"}')
                return

            # --- Register endpoint (always public) ---
            if self.path == "/api/register":
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    name = data.get('name', '').strip()
                    pin = data.get('pin', '').strip()
                    device_id = data.get('device_id', '').strip()

                    if not name:
                        self.send_response(400)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"error": "Name is required"}')
                        return
                    if len(name) < 2 or len(name) > 30:
                        self.send_response(400)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"error": "Name must be 2-30 characters"}')
                        return
                    if not pin or len(pin) < 4:
                        self.send_response(400)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"error": "PIN must be at least 4 characters"}')
                        return

                    safe = auth.safe_name(name)
                    if not safe:
                        self.send_response(400)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"error": "Invalid name"}')
                        return

                    filepath = auth.profiles_dir() / f"{safe}.yaml"
                    if filepath.exists():
                        self.send_response(409)
                        self.send_header("Content-type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"error": "Profile already exists"}')
                        return

                    # Create new profile with hashed PIN + device binding
                    security_block = {
                        "pin_hash": auth.hash_pin(pin),
                    }
                    if device_id:
                        security_block["devices"] = [device_id]

                    profile_data = {
                        "profile": {
                            "role": "",
                            "location": "Kuwait",
                            "context": "",
                            "interests": [],
                        },
                        "security": security_block,
                        "market": {
                            "tickers": [],
                            "intervals": {},
                            "alert_threshold_percent": 5.0,
                        },
                        "news": {
                            "timelimit": "w",
                            "career": {"root": "kuwait", "keywords": [], "queries": []},
                            "finance": {"root": "kuwait", "keywords": [], "queries": []},
                            "regional": {"root": "regional", "keywords": [], "queries": []},
                            "tech_trends": {"root": "global", "keywords": [], "queries": []},
                            "rss_feeds": [],
                        },
                    }

                    with open(filepath, "w") as f:
                        yaml.dump(profile_data, f, default_flow_style=False, sort_keys=False)

                    logger.info(f"New profile registered: {safe}")

                    # Auto-login after registration
                    auth.load_profile_config(safe, strat)
                    token = auth.create_session(safe)

                    self.send_response(201)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "registered",
                        "token": token,
                        "profile": safe
                    }).encode())
                except Exception as e:
                    logger.error(f"Registration error: {e}")
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": "Registration failed"}')
                return

            # --- Logout endpoint ---
            if self.path == "/api/logout":
                token = self.headers.get('X-Auth-Token', '')
                auth.delete_session(token)
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "logged_out"}')
                return

            # --- Auth enforcement for API endpoints ---
            _auth_path_post = self.path.split('?')[0]
            if _auth_path_post.startswith('/api/') and _auth_path_post not in auth.AUTH_EXEMPT:
                token = self.headers.get('X-Auth-Token', '')
                device_id = self.headers.get('X-Device-Id', '')

                if token and auth.validate_session(token):
                    # Logged-in user — use their session
                    _session_profile = auth.get_session_profile(token)
                    if _session_profile:
                        strat.ensure_profile(_session_profile)
                    try:
                        cursor = strat.db.conn.cursor()
                        cursor.execute("SELECT profile_id FROM sessions WHERE token = ?", (token,))
                        _pid_row = cursor.fetchone()
                        if _pid_row and _pid_row[0]:
                            self._profile_id = _pid_row[0]
                    except Exception:
                        pass
                    self._session_profile = auth.get_session_profile(token)
                elif device_id:
                    # Anonymous user — resolve or create profile from device ID
                    try:
                        self._profile_id = _resolve_device_profile(strat.db, device_id)
                    except Exception as e:
                        logger.error(f"Device profile resolution failed: {e}")
                        self._profile_id = 0
                    self._session_profile = None
                else:
                    # No auth and no device ID — reject
                    self.send_response(401)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": "Authentication required"}')
                    return

            # Save config (existing delegation)
            if self.path == "/api/config":
                handle_config_save(self, strat, auth.auth_helpers_dict())
                return

            # Existing route delegations (keep as-is)
            if self.path == "/api/ask":
                handle_ask(self, strat, output_dir)
                return

            if self.path == "/api/file-assist":
                from routes.agent import handle_file_assist
                handle_file_assist(self, strat)
                return

            if self.path == "/api/suggest-context":
                handle_suggest_context(self, strat)
                return

            if self.path == "/api/generate-profile":
                handle_generate_profile(self, strat)
                return

            if self.path == "/api/wizard-preselect":
                handle_wizard_preselect(self, strat)
                return

            if self.path == "/api/wizard-tab-suggest":
                handle_wizard_tab_suggest(self, strat)
                return

            if self.path == "/api/wizard-rv-items":
                handle_wizard_rv_items(self, strat)
                return

            # ── AI Agent Chat (streaming) ──────────────────────────
            if self.path == "/api/agent-chat":
                _agent_output_file = strat._get_output_path(self._session_profile) if self._session_profile else strat.output_file
                handle_agent_chat(self, strat, _agent_output_file, profile_id=self._profile_id)
                return

            # --- Route dispatch to modules ---
            clean_path = self.path.split('?')[0]

            if rp_chat.handle_post(self, strat, auth, clean_path): return
            if image_gen.handle_post(self, strat, auth, clean_path): return
            if character_cards.handle_post(self, strat, auth, clean_path): return
            if controls.handle_post(self, strat, auth, clean_path): return
            if feeds.handle_post(self, strat, auth, clean_path): return
            if youtube_endpoints.handle_post(self, strat, auth, clean_path): return
            if persona_data.handle_post(self, strat, auth, clean_path): return
            if media.handle_post(self, strat, auth, clean_path): return
            if data_endpoints.handle_post(self, strat, auth, clean_path): return
            if dev_endpoints.handle_post(self, strat, auth, clean_path): return

            self.send_response(404)
            self.end_headers()

        def do_DELETE(self):
            self._profile_id = 0
            self._session_profile = None
            # Auth enforcement for DELETE
            if self.path.startswith('/api/'):
                token = self.headers.get('X-Auth-Token', '')
                device_id = self.headers.get('X-Device-Id', '')
                if token and auth.validate_session(token):
                    _session_profile = auth.get_session_profile(token)
                    if _session_profile:
                        strat.ensure_profile(_session_profile)
                    try:
                        cursor = strat.db.conn.cursor()
                        cursor.execute("SELECT profile_id FROM sessions WHERE token = ?", (token,))
                        _pid_row = cursor.fetchone()
                        if _pid_row and _pid_row[0]:
                            self._profile_id = _pid_row[0]
                    except Exception:
                        pass
                elif device_id:
                    try:
                        self._profile_id = _resolve_device_profile(strat.db, device_id)
                    except Exception:
                        self._profile_id = 0
                else:
                    _send_json(self, {"error": "Authentication required"}, 401)
                    return
                self._session_profile = _session_profile

            # --- Auth route DELETE (profile deletion) ---
            if self.path.startswith("/api/profiles/"):
                if handle_auth_routes(self, "DELETE", self.path, {}, strat.db, strat, _send_json, email_service):
                    return

            # --- Route dispatch to modules ---
            clean_path = self.path.split('?')[0]

            if image_gen.handle_delete(self, strat, auth, clean_path): return
            if character_cards.handle_delete(self, strat, auth, clean_path): return
            if youtube_endpoints.handle_delete(self, strat, auth, clean_path): return
            if persona_data.handle_delete(self, strat, auth, clean_path): return
            if media.handle_delete(self, strat, auth, clean_path): return

            self.send_response(404)
            self.end_headers()

        def do_PUT(self):
            """Route PUT requests through do_POST (conversations use PUT for updates)."""
            self.do_POST()

        def do_OPTIONS(self):
            # Handle CORS preflight (Access-Control-Allow-Origin added by end_headers())
            self.send_response(200)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Auth-Token, X-Device-Id, X-Filename")
            self.end_headers()

        def log_message(self, format, *args):
            # Quieter logging
            if '404' in str(args):
                logger.warning(f"404: {args[0]}")

    return CORSHandler


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start_server(strat, auth, port=8080, open_browser=True):
    """Start the HTTP server with graceful shutdown handling.

    Args:
        strat: StratOS instance
        auth: AuthManager instance
        port: Port to serve on
        open_browser: Whether to open browser automatically
    """
    frontend_dir = Path(__file__).parent.parent / "frontend"
    output_dir = strat.output_file.parent

    HandlerClass = create_handler(strat, auth, frontend_dir, output_dir)
    server = ThreadedHTTPServer(("0.0.0.0", port), HandlerClass)
    logger.info(f"Frontend server started at http://localhost:{port}")

    # Reset stuck transcription statuses — only truly stuck ones (transcribing/extracting/processing)
    # These are NOT reset to 'pending' (which would re-queue for worker), but to 'failed'
    # so the user can manually retranscribe if desired
    try:
        import sqlite3 as _sq
        _db_path = str(strat.db.db_path)
        with _sq.connect(_db_path) as _rc:
            _rc.execute("PRAGMA busy_timeout = 5000")
            _rc.execute(
                "UPDATE youtube_videos SET status = 'failed', error_message = 'Server restarted during processing' "
                "WHERE status IN ('transcribing', 'extracting', 'processing')"
            )
            _stuck_count = _rc.total_changes
            _rc.commit()
        if _stuck_count:
            logger.info(f"Reset {_stuck_count} stuck video(s) to failed")
    except Exception as e:
        logger.debug(f"Stuck status reset: {e}")

    # Start YouTube background worker
    try:
        from processors.youtube_worker import start_youtube_worker
        start_youtube_worker(strat, strat.sse if hasattr(strat, 'sse') else None)
    except Exception as e:
        logger.warning(f"YouTube worker failed to start: {e}")

    if open_browser:
        webbrowser.open(f"http://localhost:{port}")

    # Graceful shutdown handler for SIGINT (Ctrl+C) and SIGTERM
    shutdown_initiated = False

    def _shutdown_handler(signum, frame):
        nonlocal shutdown_initiated
        sig_name = signal.Signals(signum).name

        if shutdown_initiated:
            logger.warning(f"[StratOS] Forced shutdown ({sig_name})")
            sys.exit(1)

        shutdown_initiated = True
        logger.info(f"[StratOS] Shutdown requested ({sig_name}), finishing current operations...")

        # Hard timeout: force exit after 15 seconds if graceful shutdown hangs
        def _force_exit():
            logger.error("[StratOS] Graceful shutdown timed out after 15s, forcing exit")
            os._exit(1)
        watchdog = threading.Timer(15.0, _force_exit)
        watchdog.daemon = True
        watchdog.start()

        # 1. Cancel any in-progress scan
        if hasattr(strat, '_scan_cancelled'):
            strat._scan_cancelled.set()
            logger.info("[StratOS] Scan cancellation signal sent")

        # 2. Wait for scan to finish (up to 10 seconds)
        if strat.scan_status.get("is_scanning"):
            logger.info("[StratOS] Waiting for scan to finish...")
            waited = 0
            while strat.scan_status.get("is_scanning") and waited < 10:
                time.sleep(0.5)
                waited += 0.5
            if strat.scan_status.get("is_scanning"):
                logger.warning("[StratOS] Scan did not finish within 10s, proceeding with shutdown")

        # 3. Checkpoint WAL
        try:
            strat.db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.info("[StratOS] WAL checkpoint complete")
        except Exception as e:
            logger.warning(f"[StratOS] WAL checkpoint failed: {e}")

        # 4. Notify SSE clients
        try:
            strat.sse_broadcast("shutdown", {"reason": sig_name})
        except Exception:
            pass

        # 5. Shutdown HTTP server
        try:
            server.shutdown()
        except Exception:
            pass

        watchdog.cancel()
        logger.info("[StratOS] Shutdown complete")
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, _shutdown_handler)
        signal.signal(signal.SIGTERM, _shutdown_handler)
    except ValueError:
        # signal() only works from main thread — skip if called from a worker thread
        pass

    server.serve_forever()
