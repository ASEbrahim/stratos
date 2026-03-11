"""
HTTP server and route dispatch for StratOS.

Contains CORSHandler (all API routes), ThreadedHTTPServer,
and start_server() which wires everything together.

Extracted from main.py:serve_frontend() (Sprint 4, A1.2).
"""

import gzip as _gzip_mod
import hashlib
import json
import logging
import mimetypes
import os
import sys
import signal
import time
import threading
import yaml
import requests
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urljoin
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

from routes.agent import handle_agent_chat, handle_agent_status, handle_ask, handle_suggest_context
from routes.auth import handle_auth_routes
from routes.generate import handle_generate_profile
from routes.wizard import handle_wizard_preselect, handle_wizard_tab_suggest, handle_wizard_rv_items
from routes.helpers import json_response, error_response
from routes.config import handle_config_save
from email_service import EmailService

logger = logging.getLogger("STRAT_OS")

# File extensions eligible for gzip compression on static file serving
_GZIP_TYPES = frozenset({'.js', '.css', '.html', '.json', '.svg', '.txt', '.xml'})


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

        def do_GET(self):
            self._profile_id = 0
            self._session_profile = None

            # --- Rate limiting (before any dispatch) ---
            if auth.rate_limited(self.path):
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
            if self.path.startswith('/api/') and not self.path.startswith('/api/proxy') and self.path not in auth.AUTH_EXEMPT:
                token = self.headers.get('X-Auth-Token', '')
                if not auth.validate_session(token):
                    self.send_response(401)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": "Authentication required"}')
                    return
                # A2.1: Ensure the correct profile's config is loaded
                _session_profile = auth.get_session_profile(token)
                if _session_profile:
                    strat.ensure_profile(_session_profile)
                # Resolve DB profile_id from session (for data isolation)
                try:
                    cursor = strat.db.conn.cursor()
                    cursor.execute("SELECT profile_id FROM sessions WHERE token = ?", (token,))
                    _pid_row = cursor.fetchone()
                    if _pid_row and _pid_row[0]:
                        self._profile_id = _pid_row[0]
                        strat.active_profile_id = _pid_row[0]
                except Exception:
                    pass
                self._session_profile = _session_profile

            # Lightweight briefing-only endpoint (5KB vs 2MB full /api/data)
            if self.path == "/api/briefing":
                briefings = strat.db.get_recent_briefings(limit=1, profile_id=self._profile_id)
                if briefings:
                    _send_json(self, briefings[0].get("content", {}))
                else:
                    _send_json(self, {"status": "no_briefing"})
                return

            # Serve news_data.json from output directory (per-profile)
            if self.path == "/api/data" or self.path == "/news_data.json":
                # Resolve profile-specific output file
                _prof = self._session_profile
                output_path = strat._get_output_path(_prof) if _prof else strat._output_base
                # If profile-specific file doesn't exist, return empty data (not another profile's)
                if _prof and not output_path.exists():
                    _send_json(self, {"news": [], "market": {}, "briefing": None, "last_updated": None})
                    return
                if output_path.exists():
                    raw = output_path.read_bytes()

                    # ETag: skip sending if client already has this version
                    etag = '"' + hashlib.md5(raw).hexdigest()[:16] + '"'
                    if self.headers.get('If-None-Match') == etag:
                        self.send_response(304)
                        self.end_headers()
                        return

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("ETag", etag)

                    # Gzip if client supports it
                    if 'gzip' in self.headers.get('Accept-Encoding', '') and len(raw) > 1024:
                        compressed = _gzip_mod.compress(raw)
                        self.send_header("Content-Encoding", "gzip")
                        self.send_header("Content-Length", str(len(compressed)))
                        self.end_headers()
                        self.wfile.write(compressed)
                    else:
                        self.send_header("Content-Length", str(len(raw)))
                        self.end_headers()
                        self.wfile.write(raw)
                else:
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(b'{"error": "No data yet. Run a scan first."}')
                return

            # Serve trigger for full refresh (legacy, still works)
            if self.path == "/api/refresh":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"status": "refresh_triggered"}')
                # Resolve profile_id (refresh is AUTH_EXEMPT)
                _refresh_pid = self._profile_id or (_get_profile_id(self.headers.get('X-Auth-Token', '')) or 0)
                threading.Thread(target=strat.run_scan, args=(_refresh_pid,), daemon=True).start()
                return

            # Serve trigger for market-only refresh (fast)
            if self.path == "/api/refresh-market":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"status": "market_refresh_triggered"}')
                # Trigger market refresh in background with error handling (B7)
                _mkt_pid = self._profile_id or (_get_profile_id(self.headers.get('X-Auth-Token', '')) or 0)
                def _safe_market_refresh(pid=_mkt_pid):
                    try:
                        strat.run_market_refresh(profile_id=pid)
                    except Exception as e:
                        logger.error(f"Market refresh thread failed: {e}")
                        strat.scan_status["is_scanning"] = False
                        strat.scan_status["stage"] = "error"
                threading.Thread(target=_safe_market_refresh, daemon=True).start()
                return

            # Single-ticker live update (for fullscreen chart auto-refresh)
            if self.path.startswith("/api/market-tick"):
                qs = parse_qs(urlparse(self.path).query)
                symbol = qs.get("symbol", [""])[0]
                interval = qs.get("interval", ["1m"])[0]
                if not symbol:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error":"symbol required"}')
                    return
                try:
                    tick_data = strat.market_fetcher.fetch_single(symbol, interval)
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(tick_data).encode())
                except Exception as e:
                    logger.error(f"Endpoint error: {e}")
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Internal server error"}).encode())
                return

            # Serve trigger for news-only refresh (slower, uses API)
            if self.path == "/api/refresh-news":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"status": "news_refresh_triggered"}')
                _refresh_pid = self._profile_id or (_get_profile_id(self.headers.get('X-Auth-Token', '')) or 0)
                threading.Thread(target=strat.run_news_refresh, args=(_refresh_pid,), daemon=True).start()
                return

            # Serve scan status
            # Ticker presets — GET list
            if self.path == "/api/ticker-presets":
                user = self._session_profile or 'default'
                presets_dir = os.path.join("profiles", user, "ticker_presets")
                presets = []
                if os.path.isdir(presets_dir):
                    for fname in sorted(os.listdir(presets_dir)):
                        if fname.endswith('.json'):
                            try:
                                with open(os.path.join(presets_dir, fname)) as pf:
                                    preset = json.load(pf)
                                    presets.append({"name": preset.get("name", fname[:-5]), "tickers": preset.get("tickers", [])})
                            except Exception:
                                pass
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"presets": presets}).encode())
                return

            if self.path == "/api/status":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                # Resolve profile_id from token (status is AUTH_EXEMPT so _profile_id may be 0)
                token = self.headers.get('X-Auth-Token', '')
                _status_pid = self._profile_id or (_get_profile_id(token) if token else 0) or 0
                status = {**strat.scan_status, "my_profile_id": _status_pid,
                          "recent_scans": strat.db.get_scan_log(5, profile_id=_status_pid)}
                if token:
                    user = auth.get_session_profile(token)
                    if user:
                        safe = auth.safe_name(user)
                        pf = auth.profiles_dir() / f"{safe}.yaml"
                        if pf.exists():
                            try:
                                pd = yaml.safe_load(pf.read_text()) or {}
                                ai = pd.get("profile", {}).get("avatar_image", "")
                                if ai:
                                    status["avatar_image"] = ai
                                av = pd.get("profile", {}).get("avatar", "")
                                if av:
                                    status["avatar"] = av
                                em = pd.get("profile", {}).get("email", "")
                                if em:
                                    status["email"] = em
                            except Exception:
                                pass
                    # DB fallback for avatar (DB-auth users have no YAML) + ui_state for theme sync
                    if _status_pid:
                        ui_state = strat.db.get_ui_state(_status_pid)
                        if "avatar_image" not in status and ui_state.get("avatar_image"):
                            status["avatar_image"] = ui_state["avatar_image"]
                        if "avatar" not in status and ui_state.get("avatar"):
                            status["avatar"] = ui_state["avatar"]
                        status["ui_state"] = ui_state
                        # DB fallback for email (DB-auth users have no YAML)
                        if "email" not in status:
                            try:
                                cursor = strat.db.conn.cursor()
                                cursor.execute("""
                                    SELECT u.email FROM users u
                                    JOIN profiles p ON p.user_id = u.id
                                    WHERE p.id = ? LIMIT 1
                                """, (_status_pid,))
                                email_row = cursor.fetchone()
                                if email_row and email_row[0]:
                                    status["email"] = email_row[0]
                            except Exception:
                                pass
                self.wfile.write(json.dumps(status).encode())
                return

            if self.path == "/api/ui-state":
                _send_json(self, strat.db.get_ui_state(self._profile_id))
                return

            # Detailed scan progress (for Stop Scan polling)
            if self.path == "/api/scan/status":
                s = strat.scan_status
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "running": s["is_scanning"],
                    "progress": f"{s['scored']}/{s['total']}",
                    "scored": s["scored"],
                    "total": s["total"],
                    "high": s["high"],
                    "medium": s["medium"],
                    "cancelled": s.get("cancelled", False),
                    "stage": s["stage"],
                    "scan_profile_id": s.get("scan_profile_id")
                }).encode())
                return

            # SSE event stream — replaces polling for real-time updates
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

            # Serve search status (Google API quota)
            if self.path == "/api/search-status":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    from fetchers.google_search import get_search_status
                    status = get_search_status(strat.config)
                except ImportError:
                    status = {"provider": "duckduckgo", "limit_reached": False}
                self.wfile.write(json.dumps(status).encode())
                return

            # Scan history log
            if self.path == "/api/scan-log":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                log = strat.db.get_scan_log(50, profile_id=self._profile_id)
                self.wfile.write(json.dumps(log).encode())
                return

            # ── Dashboard Export ──────────────────────────────────
            if self.path.startswith("/api/export"):
                try:
                    query = parse_qs(urlparse(self.path).query)
                    fmt = query.get("format", ["json"])[0]

                    # Load current dashboard data (profile-specific)
                    _export_path = strat._get_output_path(self._session_profile) if self._session_profile else (Path(output_dir) / "news_data.json")
                    dashboard = {}
                    if _export_path.exists():
                        with open(_export_path, "r") as f:
                            dashboard = json.loads(f.read())

                    news_items = dashboard.get("news", [])
                    scan_log = strat.db.get_scan_log(20, profile_id=self._profile_id)

                    if fmt == "csv":
                        # Flat CSV of news signals
                        import csv
                        import io
                        buf = io.StringIO()
                        writer = csv.writer(buf)
                        writer.writerow([
                            "score", "title", "source", "category", "root",
                            "score_reason", "url", "timestamp", "summary"
                        ])
                        for item in sorted(news_items, key=lambda x: x.get("score", 0), reverse=True):
                            writer.writerow([
                                item.get("score", 0),
                                item.get("title", ""),
                                item.get("source", ""),
                                item.get("category", ""),
                                item.get("root", ""),
                                item.get("score_reason", ""),
                                item.get("url", ""),
                                item.get("timestamp", ""),
                                (item.get("summary", "") or "")[:300],
                            ])

                        data = buf.getvalue().encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/csv; charset=utf-8")
                        self.send_header("Content-Disposition", f'attachment; filename="stratos_export_{datetime.now().strftime("%Y%m%d_%H%M")}.csv"')
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(data)

                    else:
                        # Full JSON diagnostic export
                        _pid = self._profile_id
                        cat_stats = strat.db.get_category_stats(days=7, profile_id=_pid)
                        daily_counts = strat.db.get_daily_signal_counts(days=7, profile_id=_pid)
                        top_signals = strat.db.get_top_signals(days=7, min_score=7.0, limit=30, profile_id=_pid)

                        export = {
                            "exported_at": datetime.now().isoformat(),
                            "version": "3.0",
                            "current_feed": {
                                "news_count": len(news_items),
                                "news": news_items,
                                "market": dashboard.get("market", {}),
                                "briefing": dashboard.get("briefing", {}),
                                "last_updated": dashboard.get("last_updated", ""),
                            },
                            "history": {
                                "scan_log": scan_log,
                                "category_stats_7d": cat_stats,
                                "daily_signal_counts_7d": daily_counts,
                                "top_signals_7d": top_signals,
                            },
                            "profile": {
                                "role": strat.config.get("profile", {}).get("role", ""),
                                "location": strat.config.get("profile", {}).get("location", ""),
                            },
                        }

                        data = json.dumps(export, indent=2).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.send_header("Content-Disposition", f'attachment; filename="stratos_export_{datetime.now().strftime("%Y%m%d_%H%M")}.json"')
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(data)

                except Exception as e:
                    logger.error(f"Export failed: {e}")
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Internal server error"}).encode())
                return

            # Feedback stats
            if self.path == "/api/feedback-stats":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    stats = strat.db.get_feedback_stats(profile_id=self._profile_id)
                except Exception as e:
                    logger.error(f"Endpoint error: {e}")
                    stats = {"error": "Internal server error"}
                self.wfile.write(json.dumps(stats).encode())
                return

            # Extra feeds: Finance and Politics (RSS-only, no scoring)
            # Strip query string for route matching
            clean_path = self.path.split('?')[0]

            if clean_path in ("/api/finance-news", "/api/politics-news", "/api/jobs-news"):
                feed_type = "finance" if "finance" in clean_path else "jobs" if "jobs" in clean_path else "politics"
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.end_headers()
                try:
                    from fetchers.extra_feeds import fetch_extra_feeds
                    items = fetch_extra_feeds(feed_type, config=strat.config)
                except Exception as e:
                    logger.error(f"Extra feeds ({feed_type}) error: {e}")
                    items = []
                self.wfile.write(json.dumps({"items": items, "fetched_at": datetime.now().isoformat()}).encode())
                return

            # Media proxy — routes through CF Worker for ISP-blocked content
            if clean_path == "/api/proxy":
                target_url = parse_qs(urlparse(self.path).query).get("url", [None])[0]
                if not target_url:
                    from routes.helpers import error_response
                    error_response(self, "Missing url param", 400)
                    return
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
                    self.send_response(resp.status_code)
                    self.send_header("Content-Type", ct)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Cache-Control", "public, max-age=3600")
                    self.end_headers()
                    for chunk in resp.iter_content(8192):
                        self.wfile.write(chunk)
                except Exception as e:
                    logger.warning(f"Proxy error: {e}")
                    from routes.helpers import error_response
                    error_response(self, f"Proxy error: {e}", 502)
                return

            # Custom user feeds (RSS from user-defined URLs)
            if clean_path == "/api/custom-news":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.end_headers()
                items = []
                try:
                    custom_feeds = strat.config.get("custom_feeds", [])
                    enabled_feeds = [f for f in custom_feeds if f.get("on", True)]
                    if enabled_feeds:
                        import feedparser
                        import re as _re
                        _ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                        _worker_base = strat.config.get("proxy", {}).get("cloudflare_worker", "")
                        _blocked = strat.config.get("proxy", {}).get("blocked_domains", [])

                        for feed_cfg in enabled_feeds:
                            try:
                                feed_url = feed_cfg["url"]
                                feed_domain = urlparse(feed_url).hostname or ""
                                _is_blocked = any(feed_domain == b or feed_domain.endswith("." + b) for b in _blocked)

                                # Route blocked feeds through Cloudflare Worker
                                if _is_blocked and _worker_base:
                                    try:
                                        _proxy_url = f"{_worker_base}/feed?url={requests.utils.quote(feed_url, safe='')}"
                                        _resp = requests.get(_proxy_url, headers={"User-Agent": _ua}, timeout=20)
                                        feed = feedparser.parse(_resp.content)
                                    except Exception as _pe:
                                        logger.warning(f"CF proxy failed for {feed_cfg.get('name','')}: {_pe}")
                                        feed = feedparser.parse("")
                                else:
                                    # Direct fetch with headers
                                    try:
                                        _resp = requests.get(feed_url, headers={
                                            "User-Agent": _ua,
                                            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
                                        }, timeout=15, allow_redirects=True)
                                        feed = feedparser.parse(_resp.content)
                                    except requests.RequestException:
                                        try:
                                            from curl_cffi import requests as cf_req
                                            _resp = cf_req.get(feed_url, impersonate="chrome", timeout=15)
                                            feed = feedparser.parse(_resp.content)
                                        except Exception:
                                            feed = feedparser.parse(feed_url)

                                for entry in feed.entries[:15]:
                                    pub = entry.get("published", entry.get("updated", ""))
                                    link = entry.get("link", "")

                                    # ── Extract thumbnail ──
                                    thumb = ""
                                    # 1. media:thumbnail or media:content
                                    media_thumb = entry.get("media_thumbnail", [])
                                    if media_thumb and isinstance(media_thumb, list):
                                        thumb = media_thumb[0].get("url", "")
                                    if not thumb:
                                        media_content = entry.get("media_content", [])
                                        if media_content and isinstance(media_content, list):
                                            for mc in media_content:
                                                if mc.get("medium") == "image" or "image" in mc.get("type", ""):
                                                    thumb = mc.get("url", "")
                                                    break
                                    # 2. enclosure with image type
                                    if not thumb:
                                        enclosures = entry.get("enclosures", [])
                                        if enclosures:
                                            for enc in enclosures:
                                                if "image" in enc.get("type", ""):
                                                    thumb = enc.get("href", enc.get("url", ""))
                                                    break
                                    # 3. First <img> in summary/content HTML
                                    full_image = ""
                                    if not thumb:
                                        content_html = entry.get("summary", "") or ""
                                        if entry.get("content"):
                                            content_html = entry["content"][0].get("value", content_html)
                                        img_match = _re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content_html)
                                        if img_match:
                                            thumb = img_match.group(1)
                                    # 4. Construct sample + full URLs from known booru patterns
                                    sample_image = ""
                                    if thumb:
                                        # Yande.re: assets.yande.re/data/preview/ab/cd/hash.jpg
                                        #   sample: files.yande.re/sample/hash/yande.re+NNN+sample+tags.jpg (unreliable)
                                        #   jpeg:   files.yande.re/jpeg/ab/cd/hash.jpg (reliable mid-res)
                                        #   image:  files.yande.re/image/ab/cd/hash.png (full, might be png)
                                        if 'yande.re' in thumb and '/preview/' in thumb:
                                            sample_image = thumb.replace('assets.yande.re/data/preview/', 'files.yande.re/jpeg/').replace('/preview/', '/jpeg/')
                                            full_image = thumb.replace('assets.yande.re/data/preview/', 'files.yande.re/image/').replace('/preview/', '/image/')
                                        # Konachan: same structure as yande.re
                                        elif 'konachan.' in thumb and '/preview/' in thumb:
                                            sample_image = thumb.replace('/preview/', '/jpeg/')
                                            full_image = thumb.replace('/preview/', '/image/')
                                        # Danbooru: 360px thumbnail is the max free resolution
                                        # /sample/ and /original/ require different filename formats or paid account
                                        elif 'donmai.us' in thumb:
                                            pass  # Just use thumbnail as-is
                                        # Gelbooru: img*.gelbooru.com/thumbnails/hash/thumbnail_file.jpg
                                        elif 'gelbooru.' in thumb and '/thumbnails/' in thumb:
                                            sample_image = thumb.replace('/thumbnails/', '/samples/').replace('thumbnail_', 'sample_')
                                            full_image = thumb.replace('/thumbnails/', '/images/').replace('thumbnail_', '')
                                        # Safebooru: same as gelbooru
                                        elif 'safebooru.' in thumb and '/thumbnails/' in thumb:
                                            sample_image = thumb.replace('/thumbnails/', '/samples/').replace('thumbnail_', 'sample_')
                                            full_image = thumb.replace('/thumbnails/', '/images/').replace('thumbnail_', '')
                                    # Also scan content HTML for any larger image URLs
                                    if not full_image:
                                        content_html = entry.get("summary", "") or ""
                                        if entry.get("content"):
                                            content_html = entry["content"][0].get("value", content_html)
                                        all_imgs = _re.findall(r'(?:src|href)=["\']([^"\']+\.(?:jpg|jpeg|png|webp))["\']', content_html)
                                        for img_url in all_imgs:
                                            if '/sample/' in img_url or '/jpeg/' in img_url or '/image/' in img_url:
                                                full_image = img_url
                                                break

                                    # ── Detect media type from URL ──
                                    media_type = "article"
                                    embed_id = ""
                                    embed_type = ""

                                    if any(d in link for d in ["youtube.com/watch", "youtu.be/", "youtube.com/shorts"]):
                                        media_type = "video"
                                        yt_match = _re.search(r'(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})', link)
                                        if yt_match:
                                            embed_id = yt_match.group(1)
                                            embed_type = "youtube"
                                            if not thumb:
                                                thumb = f"https://img.youtube.com/vi/{embed_id}/mqdefault.jpg"
                                    elif "twitch.tv/" in link:
                                        media_type = "stream"
                                        tw_match = _re.search(r'twitch\.tv/(?:videos/)?([a-zA-Z0-9_]+)', link)
                                        if tw_match:
                                            embed_id = tw_match.group(1)
                                            embed_type = "twitch"
                                    elif any(d in link for d in ["danbooru.", "yande.re", "gelbooru.", "konachan.", "safebooru."]):
                                        media_type = "image"
                                    elif any(d in link for d in ["mangadex.", "mangaplus.", "webtoons.", "mangakakalot.", "manganato."]):
                                        media_type = "manga"
                                    elif any(d in link for d in ["vimeo.com", "dailymotion.com"]):
                                        media_type = "video"
                                    elif link.lower().endswith((".mp4", ".webm", ".mov")):
                                        media_type = "video"

                                    # ── Proxy blocked thumbnails ──
                                    if thumb:
                                        thumb_domain = urlparse(thumb).hostname or ""
                                        if any(thumb_domain == b or thumb_domain.endswith("." + b) for b in _blocked):
                                            thumb = f"/api/proxy?url={requests.utils.quote(thumb, safe='')}"
                                    if sample_image:
                                        si_domain = urlparse(sample_image).hostname or ""
                                        if any(si_domain == b or si_domain.endswith("." + b) for b in _blocked):
                                            sample_image = f"/api/proxy?url={requests.utils.quote(sample_image, safe='')}"
                                    if full_image:
                                        fi_domain = urlparse(full_image).hostname or ""
                                        if any(fi_domain == b or fi_domain.endswith("." + b) for b in _blocked):
                                            full_image = f"/api/proxy?url={requests.utils.quote(full_image, safe='')}"

                                    # ── Build item ──
                                    item = {
                                        "title": entry.get("title", "No title"),
                                        "url": link,
                                        "summary": _re.sub(r'<[^>]+>', '', entry.get("summary", ""))[:300],
                                        "source": feed_cfg.get("name", "Custom"),
                                        "timestamp": pub,
                                        "media_type": media_type,
                                    }
                                    if thumb:
                                        item["thumbnail"] = thumb
                                    if sample_image:
                                        item["sample_image"] = sample_image
                                    if full_image:
                                        item["full_image"] = full_image
                                    if embed_id:
                                        item["embed_id"] = embed_id
                                        item["embed_type"] = embed_type
                                    items.append(item)
                            except Exception as e:
                                logger.warning(f"Custom feed error ({feed_cfg.get('name','')}): {e}")
                        # Sort by timestamp descending
                        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                        items = items[:80]  # Cap at 80 (more content for media view)
                except Exception as e:
                    logger.error(f"Custom feeds error: {e}")
                self.wfile.write(json.dumps({"items": items, "fetched_at": datetime.now().isoformat()}).encode())
                return

            # Serve feed catalog (for Settings UI)
            if clean_path in ("/api/feed-catalog/finance", "/api/feed-catalog/politics", "/api/feed-catalog/jobs"):
                feed_type = "finance" if "finance" in clean_path else "jobs" if "jobs" in clean_path else "politics"
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                from fetchers.extra_feeds import get_catalog
                catalog = get_catalog(feed_type)
                # Merge with user toggles from config
                toggle_key = f"extra_feeds_{feed_type}"
                user_toggles = strat.config.get(toggle_key, {})
                if user_toggles:
                    # Profile has saved preferences — apply them
                    for item in catalog:
                        if item["id"] in user_toggles:
                            item["on"] = user_toggles[item["id"]]
                else:
                    # No saved preferences (fresh profile) — default all to off
                    for item in catalog:
                        item["on"] = False
                self.wfile.write(json.dumps({"catalog": catalog, "type": feed_type}).encode())
                return

            # Serve config
            if self.path == "/api/config":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                # Use profile-scoped config if available, otherwise fall back to main config
                cfg = strat.config
                editable_config = {
                    "profile": cfg.get("profile", {}),
                    "market": {
                        "tickers": cfg.get("market", {}).get("tickers", [])
                    },
                    "news": {
                        "timelimit": cfg.get("news", {}).get("timelimit", "w"),
                        "career": cfg.get("news", {}).get("career", {}),
                        "finance": cfg.get("news", {}).get("finance", {}),
                        "tech_trends": cfg.get("news", {}).get("tech_trends", {}),
                        "rss_feeds": cfg.get("news", {}).get("rss_feeds", [])
                    },
                    "search": {
                        **cfg.get("search", {}),
                        # Mask sensitive keys — only show last 4 chars
                        "serper_api_key": auth.mask_key(cfg.get("search", {}).get("serper_api_key", "")),
                        "google_api_key": auth.mask_key(cfg.get("search", {}).get("google_api_key", "")),
                    },
                    "dynamic_categories": cfg.get("dynamic_categories", []),
                    "extra_feeds_finance": cfg.get("extra_feeds_finance", {}),
                    "extra_feeds_politics": cfg.get("extra_feeds_politics", {}),
                    "extra_feeds_jobs": cfg.get("extra_feeds_jobs", {}),
                    "custom_feeds_finance": cfg.get("custom_feeds_finance", []),
                    "custom_feeds_politics": cfg.get("custom_feeds_politics", []),
                    "custom_feeds": cfg.get("custom_feeds", []),
                    "custom_tab_name": cfg.get("custom_tab_name", "Custom"),
                    "scoring": {
                        "retain_high_scores": cfg.get("scoring", {}).get("retain_high_scores", True),
                    },
                }
                self.wfile.write(json.dumps(editable_config).encode())
                return

            # List saved profile presets (scoped to logged-in user)
            if self.path == "/api/profiles":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                current_user = self._session_profile

                presets = []
                if current_user:
                    # User-scoped presets directory
                    user_presets_dir = auth.profiles_dir() / auth.safe_name(current_user) / "presets"
                    user_presets_dir.mkdir(parents=True, exist_ok=True)
                    for f in sorted(user_presets_dir.glob("*.yaml")):
                        try:
                            with open(f) as pf:
                                data = yaml.safe_load(pf) or {}
                                presets.append({
                                    "name": f.stem,
                                    "role": data.get("profile", {}).get("role", ""),
                                    "location": data.get("profile", {}).get("location", ""),
                                    "has_pin": False,
                                })
                        except Exception:
                            pass

                self.wfile.write(json.dumps({"presets": presets}).encode())
                return

            if self.path == "/api/top-movers":
                import yfinance as _yf
                try:
                    # Broad universe: mega-caps, popular tech, crypto, commodities, leveraged ETFs
                    UNIVERSE = [
                        # Mega-cap tech
                        'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AMD', 'INTC', 'AVGO',
                        # Popular growth / volatile
                        'COIN', 'MSTR', 'PLTR', 'SMCI', 'ARM', 'SNOW', 'NET', 'SHOP', 'ROKU',
                        # Semis
                        'MU', 'QCOM', 'TSM', 'ASML', 'MRVL',
                        # Energy / commodities futures
                        'GC=F', 'SI=F', 'CL=F', 'HG=F', 'PL=F', 'NG=F',
                        # Crypto
                        'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD', 'AVAX-USD',
                        # Leveraged / volatility ETFs
                        'TQQQ', 'SOXL', 'SQQQ', 'UVXY', 'SPY', 'QQQ',
                        # Other popular
                        'BABA', 'NIO', 'RIVN', 'BA', 'DIS',
                    ]
                    data = _yf.download(UNIVERSE, period='2d', interval='1d', progress=False, threads=True)
                    movers = []
                    # yfinance returns MultiIndex columns for multi-ticker: ('Close','AAPL')
                    try:
                        if hasattr(data.columns, 'levels'):
                            close = data['Close']
                        else:
                            close = data[['Close']]
                            close.columns = [UNIVERSE[0]]
                    except Exception:
                        close = data['Close']

                    for sym in UNIVERSE:
                        try:
                            if sym not in close.columns:
                                continue
                            vals = close[sym].dropna()
                            if len(vals) < 2:
                                continue
                            prev = float(vals.iloc[-2])
                            curr = float(vals.iloc[-1])
                            if prev == 0:
                                continue
                            pct = ((curr - prev) / prev) * 100
                            movers.append({'symbol': sym, 'price': round(curr, 2), 'change_pct': round(pct, 2)})
                        except Exception:
                            continue
                    # Sort by absolute % change, take top 10
                    movers.sort(key=lambda x: abs(x['change_pct']), reverse=True)
                    top10 = movers[:10]
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"movers": top10}).encode())
                except Exception as e:
                    logger.warning(f"Top movers failed: {e}")
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Internal server error", "movers": []}).encode())
                return

            if clean_path == "/api/shadow-scores":
                query = parse_qs(urlparse(self.path).query)
                try:
                    limit = int(query.get("limit", ["200"])[0])
                except (ValueError, IndexError):
                    limit = 200
                try:
                    min_delta = float(query.get("min_delta", ["0"])[0])
                except (ValueError, IndexError):
                    min_delta = 0.0
                scores = strat.db.get_shadow_scores(limit=limit, min_delta=min_delta, profile_id=self._profile_id)
                # Compute summary stats
                if scores:
                    deltas = [abs(s.get('delta', 0)) for s in scores]
                    summary = {
                        "total_comparisons": len(scores),
                        "avg_abs_delta": round(sum(deltas) / len(deltas), 2),
                        "max_abs_delta": round(max(deltas), 2),
                        "agreement_pct": round(100 * sum(1 for d in deltas if d < 1.0) / len(deltas), 1),
                    }
                else:
                    summary = {"total_comparisons": 0, "avg_abs_delta": 0, "max_abs_delta": 0, "agreement_pct": 0}
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"summary": summary, "scores": scores}).encode())
                return

            if self.path == "/api/health":
                import resource
                health = {
                    "uptime_seconds": round(time.time() - strat._server_start_time, 1),
                }
                # Ollama status
                try:
                    r = requests.get(f"{strat.scorer.host}/api/tags", timeout=5)
                    if r.status_code == 200:
                        health["ollama_status"] = "reachable"
                        models = r.json().get("models", [])
                        model_names = [m.get("name", "") for m in models]
                        scorer_model = strat.scorer.model
                        inference_model = getattr(strat.scorer, 'inference_model', '')
                        health["scorer_model"] = {
                            "name": scorer_model,
                            "loaded": any(scorer_model in n or scorer_model.split(":")[0] in n for n in model_names),
                        }
                        health["inference_model"] = {
                            "name": inference_model,
                            "loaded": any(inference_model in n or inference_model.split(":")[0] in n for n in model_names),
                        }
                    else:
                        health["ollama_status"] = "unreachable"
                        health["scorer_model"] = {"name": strat.scorer.model, "loaded": False}
                        health["inference_model"] = {"name": getattr(strat.scorer, 'inference_model', ''), "loaded": False}
                except Exception:
                    health["ollama_status"] = "unreachable"
                    health["scorer_model"] = {"name": strat.scorer.model, "loaded": False}
                    health["inference_model"] = {"name": getattr(strat.scorer, 'inference_model', ''), "loaded": False}
                # DB size
                try:
                    db_path = Path(strat.db.db_path)
                    health["db_size_mb"] = round(db_path.stat().st_size / (1024 * 1024), 2) if db_path.exists() else 0
                except Exception:
                    health["db_size_mb"] = 0
                # Last scan
                try:
                    scans = strat.db.get_scan_log(1)
                    if scans:
                        s = scans[0]
                        health["last_scan"] = {
                            "time": s.get("started_at", ""),
                            "result": "error" if s.get("error") else "ok",
                            "elapsed": s.get("elapsed_secs", 0),
                            "items_scored": s.get("items_scored", 0),
                        }
                    else:
                        health["last_scan"] = None
                except Exception:
                    health["last_scan"] = None
                # SSE clients
                health["active_sse_clients"] = strat.sse.client_count
                # Memory usage
                try:
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    health["memory_usage_mb"] = round(usage.ru_maxrss / 1024, 2)
                except Exception:
                    health["memory_usage_mb"] = 0
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(health).encode())
                return

            if self.path == "/api/agent-status":
                handle_agent_status(self, strat)
                return

            if self.path == "/api/agent-personas":
                from routes.personas import list_personas
                _send_json(self, {"personas": list_personas()})
                return

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
                        with open(path, 'rb') as f:
                            raw = f.read()
                        compressed = _gzip_mod.compress(raw, compresslevel=6)
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
            if auth.rate_limited(self.path):
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
            if self.path.startswith('/api/') and self.path not in auth.AUTH_EXEMPT:
                token = self.headers.get('X-Auth-Token', '')
                if not auth.validate_session(token):
                    self.send_response(401)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": "Authentication required"}')
                    return
                # A2.1: Ensure the correct profile's config is loaded
                _session_profile = auth.get_session_profile(token)
                if _session_profile:
                    strat.ensure_profile(_session_profile)
                # Resolve DB profile_id from session (for data isolation)
                try:
                    cursor = strat.db.conn.cursor()
                    cursor.execute("SELECT profile_id FROM sessions WHERE token = ?", (token,))
                    _pid_row = cursor.fetchone()
                    if _pid_row and _pid_row[0]:
                        self._profile_id = _pid_row[0]
                        strat.active_profile_id = _pid_row[0]
                except Exception:
                    pass
                self._session_profile = _session_profile

            # --- UI state sync (after auth enforcement) ---
            if self.path == "/api/ui-state":
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length) if content_length else b'{}'
                try:
                    body = json.loads(post_data.decode('utf-8')) if post_data.strip() else {}
                except (json.JSONDecodeError, UnicodeDecodeError):
                    body = {}
                if body and self._profile_id:
                    strat.db.save_ui_state(self._profile_id, body)
                _send_json(self, {"ok": True})
                return

            # Save config
            if self.path == "/api/config":
                handle_config_save(self, strat, auth.auth_helpers_dict())
                return

            # Cancel an in-progress scan
            if self.path == "/api/scan/cancel":
                if strat.scan_status.get("is_scanning"):
                    strat._scan_cancelled.set()
                    _send_json(self, {"ok": True, "message": "Scan cancellation requested"})
                else:
                    _send_json(self, {"ok": False, "message": "No scan in progress"})
                return

            # Agent model warmup — pre-load inference model into VRAM
            if self.path == "/api/agent-warmup":
                try:
                    scoring_cfg = strat.config.get("scoring", {})
                    ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
                    model = scoring_cfg.get("inference_model", "qwen3.5:9b")
                    requests.post(f"{ollama_host}/api/generate",
                                  json={"model": model, "prompt": "hi", "stream": False,
                                        "options": {"num_predict": 1}},
                                  timeout=30)
                except Exception:
                    pass
                _send_json(self, {"ok": True})
                return

            # RSS feed auto-discovery from a regular URL
            if self.path == "/api/discover-rss":
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length) if content_length else b'{}'
                try:
                    body = json.loads(post_data.decode('utf-8')) if post_data.strip() else {}
                except (json.JSONDecodeError, UnicodeDecodeError):
                    body = {}
                url = body.get("url", "").strip()
                if not url:
                    _send_json(self, {"error": "URL required"}, 400)
                    return
                try:
                    import re as _re

                    # YouTube shortcut: detect channel URLs and resolve to RSS
                    _yt_match = _re.match(r'https?://(?:www\.)?youtube\.com/(?:@|channel/|c/|user/)([^/?&#]+)', url)
                    if _yt_match:
                        _yt_id_or_handle = _yt_match.group(1)
                        # If it's already a channel ID (starts with UC), use directly
                        if _yt_id_or_handle.startswith('UC') and len(_yt_id_or_handle) == 24:
                            _channel_id = _yt_id_or_handle
                        else:
                            # Use YouTube's internal resolve API (bypasses consent pages)
                            try:
                                _yt_api = requests.post("https://www.youtube.com/youtubei/v1/navigation/resolve_url",
                                    json={"url": url, "context": {"client": {"clientName": "WEB", "clientVersion": "2.20240101"}}},
                                    headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"},
                                    timeout=10)
                                if _yt_api.status_code == 200:
                                    import re as _re2
                                    _uc_matches = _re2.findall(r'UC[a-zA-Z0-9_-]{22}', _yt_api.text)
                                    _channel_id = _uc_matches[0] if _uc_matches else None
                                else:
                                    _channel_id = None
                            except Exception:
                                _channel_id = None
                                _channel_id = None

                        if _channel_id:
                            _yt_feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={_channel_id}"
                            # Verify the feed works
                            try:
                                import feedparser
                                _yt_feed_resp = requests.get(_yt_feed_url, timeout=8)
                                _yt_parsed = feedparser.parse(_yt_feed_resp.content)
                                _yt_title = _yt_parsed.feed.get("title", _yt_id_or_handle)
                                _send_json(self, {"feeds": [{
                                    "url": _yt_feed_url,
                                    "title": _yt_title,
                                    "type": "youtube",
                                    "entries": len(_yt_parsed.entries)
                                }]})
                            except Exception:
                                _send_json(self, {"feeds": [{
                                    "url": _yt_feed_url,
                                    "title": _yt_id_or_handle,
                                    "type": "youtube",
                                    "entries": 0
                                }]})
                            return
                    # Check if domain is blocked
                    from urllib.parse import urlparse as _urlparse, quote_plus as _quote_plus
                    _parsed_url = _urlparse(url)
                    _domain = _parsed_url.netloc.lower().replace('www.', '')
                    _blocked = strat.config.get("proxy", {}).get("blocked_domains", [])
                    if any(_domain == bd or _domain.endswith('.' + bd) for bd in _blocked):
                        _send_json(self, {"error": "Domain is blocked"}, 403)
                        return

                    _disc_headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    }
                    # Route through CF Worker proxy if configured
                    _cf_worker = strat.config.get("proxy", {}).get("cloudflare_worker", "")
                    _fetch_url = f"{_cf_worker}?url={_quote_plus(url)}" if _cf_worker else url
                    try:
                        resp = requests.get(_fetch_url, timeout=10, headers=_disc_headers, allow_redirects=True)
                    except requests.RequestException:
                        try:
                            from curl_cffi import requests as cf_req
                            resp = cf_req.get(_fetch_url, impersonate="chrome", timeout=10)
                        except Exception:
                            raise
                    feeds = []
                    ct = resp.headers.get('content-type', '').lower() if hasattr(resp.headers, 'get') else ''
                    html_text = resp.text[:100000]

                    # Strategy 0: URL already IS a feed (XML/RSS/Atom content-type or content starts with XML)
                    if 'xml' in ct or 'rss' in ct or 'atom' in ct or html_text.lstrip().startswith('<?xml') or html_text.lstrip().startswith('<rss') or html_text.lstrip().startswith('<feed'):
                        try:
                            import feedparser
                            parsed = feedparser.parse(resp.content)
                            if parsed.entries and len(parsed.entries) > 0:
                                feeds.append({"url": url, "title": parsed.feed.get("title", ""), "type": "direct_feed", "entries": len(parsed.entries)})
                        except Exception:
                            pass

                    # Strategy 1: Find all <link> tags and check for RSS/Atom
                    for tag_match in _re.finditer(r'<link\b[^>]*/?>', html_text, _re.IGNORECASE):
                        tag = tag_match.group(0)
                        # Must have type with rss/atom/xml
                        type_m = _re.search(r'type=["\']([^"\']+)["\']', tag, _re.IGNORECASE)
                        href_m = _re.search(r'href=["\']([^"\']+)["\']', tag, _re.IGNORECASE)
                        if not type_m or not href_m:
                            continue
                        ftype = type_m.group(1).lower()
                        if 'rss' not in ftype and 'atom' not in ftype and 'xml' not in ftype:
                            continue
                        href = href_m.group(1)
                        if not href.startswith('http'):
                            href = urljoin(url, href)
                        title_m = _re.search(r'title=["\']([^"\']+)["\']', tag, _re.IGNORECASE)
                        feeds.append({
                            "url": href,
                            "title": title_m.group(1) if title_m else "",
                            "type": ftype
                        })

                    # Strategy 2: Try feedparser directly (some sites serve RSS at their main URL)
                    if not feeds:
                        try:
                            import feedparser
                            parsed = feedparser.parse(resp.content)
                            if parsed.entries and len(parsed.entries) > 0:
                                feeds.append({"url": url, "title": parsed.feed.get("title", ""), "type": "direct"})
                        except Exception:
                            pass

                    # Strategy 3: Probe common RSS paths
                    if not feeds:
                        common_paths = ['/feed', '/rss', '/feed.xml', '/rss.xml', '/atom.xml',
                                        '/index.xml', '/feeds/posts/default', '/rss/index.xml']
                        for path in common_paths:
                            try:
                                test_url = urljoin(url, path)
                                r = requests.get(test_url, timeout=5, headers={
                                    "User-Agent": "Mozilla/5.0"
                                }, allow_redirects=True)
                                ct = r.headers.get('content-type', '').lower()
                                if r.status_code == 200 and ('xml' in ct or 'rss' in ct or 'atom' in ct):
                                    feeds.append({"url": test_url, "title": "", "type": ct.split(';')[0]})
                                    break
                            except Exception:
                                continue
                    _send_json(self, {"feeds": feeds, "source_url": url})
                except Exception as e:
                    logger.warning(f"RSS discovery error for {url}: {e}")
                    _send_json(self, {"feeds": [], "error": str(e)})
                return

            # Sync Serper credits
            if self.path == "/api/serper-credits":
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)

                try:
                    data = json.loads(post_data.decode('utf-8'))
                    credits = int(data.get('credits', 0))

                    # Update tracker
                    from fetchers.serper_search import SerperQueryTracker
                    tracker = SerperQueryTracker()
                    tracker.set_remaining(credits)

                    # Also save to config
                    strat.config.setdefault("search", {})["serper_credits"] = credits
                    with open(strat.config_path, "w") as f:
                        yaml.dump(strat.config, f, default_flow_style=False, sort_keys=False)

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "updated", "credits": credits}).encode())

                except Exception as e:
                    logger.error(f"Endpoint error: {e}")
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Internal server error"}).encode())
                return

            # Dedicated Serper API key save — separate from general config save
            if self.path == "/api/save-serper-key":
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    new_key = data.get('serper_api_key', '').strip()
                    if not new_key or '\u2022' in new_key:
                        raise ValueError("Invalid or masked API key")

                    # Update in-memory config
                    strat.config.setdefault("search", {})["serper_api_key"] = new_key

                    # Reset credit tracker (new key = fresh credits)
                    try:
                        from fetchers.serper_search import SerperQueryTracker
                        SerperQueryTracker().set_remaining(2500)
                    except Exception:
                        pass

                    # Persist to config.yaml
                    with open(strat.config_path, "w") as f:
                        yaml.dump(strat.config, f, default_flow_style=False, sort_keys=False)

                    # Update profile cache so next scan uses new key
                    if strat.active_profile:
                        strat.cache_profile_config(strat.active_profile)

                    logger.info(f"Serper API key updated via dedicated endpoint (last4: ...{new_key[-4:]})")

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "saved"}).encode())
                except Exception as e:
                    logger.error(f"Endpoint error: {e}")
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Internal server error"}).encode())
                return

            # Ticker presets — save/delete
            if self.path == "/api/ticker-presets":
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    action = data.get('action', 'save')
                    name = data.get('name', '').strip()
                    if not name:
                        raise ValueError("Preset name is required")

                    user = self._session_profile or 'default'
                    presets_dir = os.path.join("profiles", user, "ticker_presets")
                    os.makedirs(presets_dir, exist_ok=True)

                    # Sanitize filename
                    safe_name = "".join(c for c in name if c.isalnum() or c in ' _-').strip()
                    if not safe_name:
                        safe_name = "preset"
                    fpath = os.path.join(presets_dir, safe_name + ".json")

                    if action == 'delete':
                        if os.path.exists(fpath):
                            os.remove(fpath)
                        resp = {"status": "deleted", "name": name}
                    elif action == 'load':
                        if os.path.exists(fpath):
                            with open(fpath) as pf:
                                preset = json.load(pf)
                            resp = {"status": "ok", "preset": preset}
                        else:
                            resp = {"error": "Preset not found"}
                    else:  # save
                        tickers = data.get('tickers', [])
                        preset = {"name": name, "tickers": tickers, "created_at": __import__('datetime').datetime.now().isoformat()}
                        with open(fpath, 'w') as pf:
                            json.dump(preset, pf, indent=2)
                        resp = {"status": "saved", "name": name}

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(resp).encode())
                except Exception as e:
                    logger.error(f"Endpoint error: {e}")
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Internal server error"}).encode())
                return

            # User feedback: click, dismiss, rate
            if self.path == "/api/feedback":
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    # Expected: {news_id, title, url, root, category, ai_score, user_score?, action}
                    # action: "click", "dismiss", "rate", "save"
                    action = data.get("action", "click")
                    if action not in ("click", "dismiss", "rate", "save", "thumbs_up", "thumbs_down"):
                        raise ValueError(f"Unknown feedback action: {action}")

                    # Inject active profile so training data preserves context (per-request, not shared config)
                    _fb_profile = {}
                    if self._session_profile:
                        try:
                            _fb_path = auth.profiles_dir() / f"{auth.safe_name(self._session_profile)}.yaml"
                            if _fb_path.exists():
                                _fb_profile = yaml.safe_load(_fb_path.read_text()).get("profile", {})
                        except Exception:
                            pass
                    if not _fb_profile:
                        _fb_profile = strat.config.get("profile", {})
                    data["profile_role"] = _fb_profile.get("role", "")
                    data["profile_location"] = _fb_profile.get("location", "")
                    data["profile_context"] = _fb_profile.get("context", "")[:500]

                    strat.db.save_feedback(data, profile_id=self._profile_id)

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "action": action}).encode())
                except Exception as e:
                    logger.error(f"Endpoint error: {e}")
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Internal server error"}).encode())
                return

            # Feedback stats (GET would be cleaner but keeping consistent with POST pattern)
            if self.path == "/api/feedback-stats":
                try:
                    stats = strat.db.get_feedback_stats(profile_id=self._profile_id)
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(stats).encode())
                except Exception as e:
                    logger.error(f"Endpoint error: {e}")
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Internal server error"}).encode())
                return

            # Ask AI about a news item
            if self.path == "/api/ask":
                handle_ask(self, strat, output_dir)
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

            # ── Update User Profile (name, PIN, avatar) ──────────────
            if self.path == "/api/update-profile":
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    current_user = self._session_profile
                    if not current_user:
                        raise ValueError("Not authenticated")

                    safe = auth.safe_name(current_user)
                    profile_file = auth.profiles_dir() / f"{safe}.yaml"
                    _has_yaml = profile_file.exists()

                    # DB-auth users may not have YAML files — handle updates via DB
                    if not _has_yaml:
                        if self._profile_id:
                            changes = []
                            avatar_state = {}
                            new_avatar = data.get("avatar", "").strip()[:3]
                            avatar_image = data.get("avatar_image", "")
                            if avatar_image and avatar_image.startswith("data:image/"):
                                avatar_state["avatar_image"] = avatar_image
                                changes.append("avatar_image")
                            if new_avatar:
                                avatar_state["avatar"] = new_avatar
                                changes.append("avatar")
                            if avatar_state:
                                strat.db.save_ui_state(self._profile_id, avatar_state)
                            # Update display_name in users table
                            new_name = data.get("display_name", "").strip()
                            resp_name = current_user
                            if new_name and len(new_name) >= 2:
                                try:
                                    cursor = strat.db.conn.cursor()
                                    cursor.execute("""
                                        UPDATE users SET display_name = ?
                                        WHERE id = (SELECT user_id FROM profiles WHERE id = ? LIMIT 1)
                                    """, (new_name, self._profile_id))
                                    strat.db.conn.commit()
                                    changes.append("name")
                                    resp_name = new_name
                                except Exception as e:
                                    logger.warning(f"Failed to update display_name: {e}")
                            _send_json(self, {
                                "status": "updated",
                                "changes": changes,
                                "profile": {"name": resp_name, "avatar": new_avatar or "", "avatar_image": avatar_image or ""}
                            })
                            return
                        raise FileNotFoundError(f"Profile '{current_user}' not found")

                    with open(profile_file) as f:
                        profile_data = yaml.safe_load(f) or {}

                    changes = []

                    # Update display name
                    new_name = data.get("display_name", "").strip()
                    if new_name:
                        profile_data.setdefault("profile", {})["name"] = new_name
                        changes.append("name")

                    # Update avatar initials
                    new_avatar = data.get("avatar", "").strip()[:3]
                    if new_avatar:
                        profile_data.setdefault("profile", {})["avatar"] = new_avatar
                        changes.append("avatar")

                    # Update avatar image (base64 data URL for cross-device sync)
                    avatar_image = data.get("avatar_image", "")
                    if avatar_image and avatar_image.startswith("data:image/"):
                        profile_data.setdefault("profile", {})["avatar_image"] = avatar_image
                        changes.append("avatar_image")

                    # Change PIN
                    new_pin = data.get("new_pin", "").strip()
                    current_pin = data.get("current_pin", "").strip()
                    if new_pin:
                        # Verify current PIN first
                        sec = profile_data.get("security", {})
                        stored_hash = sec.get("pin_hash", "")
                        if stored_hash:
                            if auth.hash_pin(current_pin) != stored_hash:
                                raise ValueError("Current PIN is incorrect")
                        elif sec.get("pin"):
                            if str(sec["pin"]).strip() != current_pin:
                                raise ValueError("Current PIN is incorrect")
                        # Set new PIN
                        profile_data.setdefault("security", {})["pin_hash"] = auth.hash_pin(new_pin)
                        profile_data["security"].pop("pin", None)  # Remove plain text pin if exists
                        changes.append("pin")

                    # Update role
                    new_role = data.get("role", "").strip()
                    if new_role:
                        profile_data.setdefault("profile", {})["role"] = new_role
                        changes.append("role")

                    # Update location
                    new_location = data.get("location", "").strip()
                    if new_location:
                        profile_data.setdefault("profile", {})["location"] = new_location
                        changes.append("location")

                    # Update email
                    if "email" in data:
                        profile_data.setdefault("profile", {})["email"] = data["email"].strip()
                        changes.append("email")

                    # Save
                    with open(profile_file, "w") as f:
                        yaml.dump(profile_data, f, default_flow_style=False, sort_keys=False)

                    # Persist avatar to DB ui_state for cross-device sync
                    if self._profile_id:
                        avatar_state = {}
                        if avatar_image and avatar_image.startswith("data:image/"):
                            avatar_state["avatar_image"] = avatar_image
                        if new_avatar:
                            avatar_state["avatar"] = new_avatar
                        if avatar_state:
                            strat.db.save_ui_state(self._profile_id, avatar_state)

                    # Also update live config if name/role/location changed
                    if any(c in changes for c in ("name", "role", "location", "avatar")):
                        for key in ("name", "role", "location", "avatar"):
                            val = profile_data.get("profile", {}).get(key)
                            if val is not None:
                                strat.config.setdefault("profile", {})[key] = val

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "updated",
                        "changes": changes,
                        "profile": {
                            "name": profile_data.get("profile", {}).get("name", current_user),
                            "avatar": profile_data.get("profile", {}).get("avatar", ""),
                            "avatar_image": profile_data.get("profile", {}).get("avatar_image", ""),
                            "role": profile_data.get("profile", {}).get("role", ""),
                            "location": profile_data.get("profile", {}).get("location", ""),
                        }
                    }).encode())
                except Exception as e:
                    is_client_error = "incorrect" in str(e).lower() or "authenticated" in str(e).lower()
                    self.send_response(400 if is_client_error else 500)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Bad request" if is_client_error else "Internal server error"}).encode())
                return

            if self.path == "/api/profiles":
                # Use stashed body from auth route guard if available (body already consumed)
                data = getattr(self, '_stashed_post_data', None)
                if data is None:
                    content_length = int(self.headers.get('Content-Length', 0))
                    post_data = self.rfile.read(content_length)
                    data = json.loads(post_data.decode('utf-8'))
                try:
                    action = data.get("action", "")
                    name = data.get("name", "").strip()

                    current_user = self._session_profile
                    if not current_user:
                        raise ValueError("Not authenticated")

                    # User-scoped presets directory
                    user_presets_dir = auth.profiles_dir() / auth.safe_name(current_user) / "presets"
                    user_presets_dir.mkdir(parents=True, exist_ok=True)

                    if not name:
                        raise ValueError("Profile name is required")

                    # Sanitize filename
                    safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
                    filepath = user_presets_dir / f"{safe_name}.yaml"

                    if action == "save":
                        # Save current full config as a preset (no security data)
                        import copy
                        preset = copy.deepcopy(strat.config)
                        preset.pop("cache", None)
                        preset.pop("security", None)
                        with open(filepath, "w") as f:
                            yaml.dump(preset, f, default_flow_style=False, sort_keys=False)
                        self.send_response(200)
                        self.send_header("Content-type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "saved", "name": safe_name}).encode())

                    elif action == "load":
                        if not filepath.exists():
                            raise FileNotFoundError(f"Profile '{name}' not found")
                        with open(filepath) as f:
                            preset = yaml.safe_load(f)
                        # Strip security — presets don't carry auth data
                        preset.pop("security", None)
                        # Merge preset into live config (preserve search API key)
                        current_search = strat.config.get("search", {})
                        strat.config.update(preset)
                        # Keep current API key if preset doesn't have one
                        if not preset.get("search", {}).get("serper_api_key"):
                            strat.config.setdefault("search", {})["serper_api_key"] = current_search.get("serper_api_key", "")
                        # Write to disk (live config)
                        with open(strat.config_path, "w") as f:
                            yaml.dump(strat.config, f, default_flow_style=False, sort_keys=False)
                        # Also persist to user's login profile (preserving security block)
                        user_login_file = auth.profiles_dir() / f"{auth.safe_name(current_user)}.yaml"
                        if user_login_file.exists():
                            try:
                                with open(user_login_file) as uf:
                                    user_data = yaml.safe_load(uf) or {}
                                saved_sec = user_data.get("security", {})
                                import copy as _copy
                                updated = _copy.deepcopy(strat.config)
                                updated.pop("security", None)
                                if saved_sec:
                                    updated["security"] = saved_sec
                                with open(user_login_file, "w") as uf:
                                    yaml.dump(updated, uf, default_flow_style=False, sort_keys=False)
                            except Exception as e:
                                logger.warning(f"Failed to persist preset to login profile: {e}")
                        self.send_response(200)
                        self.send_header("Content-type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "loaded", "name": safe_name}).encode())

                    elif action == "delete":
                        # Safety: only delete from user's presets dir
                        if filepath.exists() and str(filepath).startswith(str(user_presets_dir)):
                            filepath.unlink()
                        self.send_response(200)
                        self.send_header("Content-type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        self.wfile.write(json.dumps({"status": "deleted", "name": safe_name}).encode())

                    else:
                        raise ValueError(f"Unknown action: {action}")

                except Exception as e:
                    logger.error(f"Endpoint error: {e}")
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Internal server error"}).encode())
                return

            # ── AI Agent Chat (streaming) ──────────────────────────
            if self.path == "/api/agent-chat":
                _agent_output_file = strat._get_output_path(self._session_profile) if self._session_profile else strat.output_file
                handle_agent_chat(self, strat, _agent_output_file, profile_id=self._profile_id)
                return

            # ── File Upload ──────────────────────────────────────────
            if self.path == "/api/files/upload":
                try:
                    from processors.file_handler import FileHandler
                    content_length = int(self.headers.get('Content-Length', 0))
                    if content_length > 10 * 1024 * 1024:
                        _send_json(self, {"error": "File too large (max 10MB)"}, 413)
                        return
                    file_data = self.rfile.read(content_length)
                    # Expect multipart or raw upload with X-Filename header
                    filename = self.headers.get('X-Filename', 'upload.txt')
                    fh = FileHandler(strat.config, db=strat.db)
                    result = fh.save_file(self._profile_id, filename, file_data)
                    if result:
                        _send_json(self, {"ok": True, "file": {
                            "id": result.get("id"),
                            "filename": result["filename"],
                            "type": result["file_type"],
                            "has_text": bool(result.get("content_text")),
                        }})
                    else:
                        _send_json(self, {"error": "Upload failed"}, 400)
                except Exception as e:
                    logger.error(f"File upload error: {e}")
                    _send_json(self, {"error": "Upload failed"}, 500)
                return

            # ── File List/Search ─────────────────────────────────────
            if self.path == "/api/files/list":
                try:
                    from processors.file_handler import FileHandler
                    body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))).decode()) if int(self.headers.get('Content-Length', 0)) > 0 else {}
                    fh = FileHandler(strat.config, db=strat.db)
                    query = body.get("query", "")
                    if query:
                        files = fh.search_files(self._profile_id, query)
                    else:
                        files = fh.list_files(self._profile_id)
                    _send_json(self, {"files": files})
                except Exception as e:
                    logger.error(f"File list error: {e}")
                    _send_json(self, {"error": "Failed to list files"}, 500)
                return

            self.send_response(404)
            self.end_headers()

        def do_DELETE(self):
            # --- Auth route DELETE (profile deletion) ---
            if self.path.startswith("/api/profiles/"):
                if handle_auth_routes(self, "DELETE", self.path, {}, strat.db, strat, _send_json, email_service):
                    return
            # ── File Delete ──
            if self.path.startswith("/api/files/"):
                try:
                    file_id = int(self.path.split("/")[-1])
                    from processors.file_handler import FileHandler
                    fh = FileHandler(strat.config, db=strat.db)
                    if fh.delete_file(self._profile_id, file_id):
                        _send_json(self, {"ok": True})
                    else:
                        _send_json(self, {"error": "File not found"}, 404)
                except (ValueError, IndexError):
                    _send_json(self, {"error": "Invalid file ID"}, 400)
                except Exception as e:
                    logger.error(f"File delete error: {e}")
                    _send_json(self, {"error": "Delete failed"}, 500)
                return
            self.send_response(404)
            self.end_headers()

        def do_OPTIONS(self):
            # Handle CORS preflight (Access-Control-Allow-Origin added by end_headers())
            self.send_response(200)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
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
