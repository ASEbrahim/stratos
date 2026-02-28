"""
HTTP server and route dispatch for StratOS.

Contains CORSHandler (all API routes), ThreadedHTTPServer,
and start_server() which wires everything together.

Extracted from main.py:serve_frontend() (Sprint 4, A1.2).
"""

import json
import logging
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

    class CORSHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(frontend_dir), **kwargs)

        def end_headers(self):
            # Disable browser caching for HTML/JS/CSS files so UI updates appear immediately
            if hasattr(self, 'path') and (self.path.endswith('.html') or self.path.endswith('.js') or self.path.endswith('.css') or self.path == '/' or self.path == ''):
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            super().end_headers()

        def do_GET(self):
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
                if handle_auth_routes(self, "GET", self.path, {}, strat.db, strat, _send_json, email_service):
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
                        strat.active_profile_id = _pid_row[0]
                except Exception:
                    pass

            # --- Rate limiting ---
            if auth.rate_limited(self.path):
                self.send_response(429)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Rate limit exceeded. Try again shortly."}')
                return

            # Serve news_data.json from output directory (per-profile)
            if self.path == "/api/data" or self.path == "/news_data.json":
                # Resolve profile-specific output file
                _token = self.headers.get('X-Auth-Token', '')
                _prof = auth.get_session_profile(_token) if _token else ''
                output_path = strat._get_output_path(_prof) if _prof else (output_dir / "news_data.json")
                if output_path.exists():
                    raw = output_path.read_bytes()

                    # ETag: skip sending if client already has this version
                    import hashlib as _hl
                    etag = '"' + _hl.md5(raw).hexdigest()[:16] + '"'
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
                        import gzip as _gz
                        compressed = _gz.compress(raw)
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
                # Trigger full scan in background
                threading.Thread(target=strat.run_scan, daemon=True).start()
                return

            # Serve trigger for market-only refresh (fast)
            if self.path == "/api/refresh-market":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"status": "market_refresh_triggered"}')
                # Trigger market refresh in background
                threading.Thread(target=strat.run_market_refresh, daemon=True).start()
                return

            # Serve trigger for news-only refresh (slower, uses API)
            if self.path == "/api/refresh-news":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"status": "news_refresh_triggered"}')
                # Trigger news refresh in background
                threading.Thread(target=strat.run_news_refresh, daemon=True).start()
                return

            # Serve scan status
            # Ticker presets — GET list
            if self.path == "/api/ticker-presets":
                token = self.headers.get('X-Auth-Token', '')
                user = auth.get_session_profile(token) or 'default'
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
                status = {**strat.scan_status, "recent_scans": strat.db.get_scan_log(5)}
                # Include avatar_image if authenticated
                token = self.headers.get('X-Auth-Token', '')
                if token:
                    user = auth.get_session_profile(token)
                    if user:
                        safe = auth.safe_name(user)
                        pf = auth.profiles_dir() / f"{safe}.yaml"
                        if pf.exists():
                            try:
                                import yaml as _yaml
                                pd = _yaml.safe_load(pf.read_text()) or {}
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
                self.wfile.write(json.dumps(status).encode())
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
                    "stage": s["stage"]
                }).encode())
                return

            # SSE event stream — replaces polling for real-time updates
            if self.path == "/api/events":
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

                strat.sse_register(self.wfile)
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
                log = strat.db.get_scan_log(50)
                self.wfile.write(json.dumps(log).encode())
                return

            # ── Dashboard Export ──────────────────────────────────
            if self.path.startswith("/api/export"):
                try:
                    from urllib.parse import urlparse, parse_qs
                    query = parse_qs(urlparse(self.path).query)
                    fmt = query.get("format", ["json"])[0]

                    # Load current dashboard data
                    output_path = Path(output_dir) / "news_data.json"
                    dashboard = {}
                    if output_path.exists():
                        with open(output_path, "r") as f:
                            dashboard = json.loads(f.read())

                    news_items = dashboard.get("news", [])
                    scan_log = strat.db.get_scan_log(20)

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
                        _pid = strat.active_profile_id
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
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
                return

            # Feedback stats
            if self.path == "/api/feedback-stats":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try:
                    stats = strat.db.get_feedback_stats()
                except Exception as e:
                    stats = {"error": str(e)}
                self.wfile.write(json.dumps(stats).encode())
                return

            # Extra feeds: Finance and Politics (RSS-only, no scoring)
            # Strip query string for route matching
            clean_path = self.path.split('?')[0]

            if clean_path in ("/api/finance-news", "/api/politics-news"):
                feed_type = "finance" if "finance" in clean_path else "politics"
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
                        for feed_cfg in enabled_feeds:
                            try:
                                feed = feedparser.parse(feed_cfg["url"])
                                for entry in feed.entries[:15]:
                                    pub = entry.get("published", entry.get("updated", ""))
                                    items.append({
                                        "title": entry.get("title", "No title"),
                                        "url": entry.get("link", ""),
                                        "summary": entry.get("summary", "")[:300],
                                        "source": feed_cfg.get("name", "Custom"),
                                        "timestamp": pub,
                                    })
                            except Exception as e:
                                logger.warning(f"Custom feed error ({feed_cfg.get('name','')}): {e}")
                        # Sort by timestamp descending
                        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                        items = items[:50]  # Cap at 50
                except Exception as e:
                    logger.error(f"Custom feeds error: {e}")
                self.wfile.write(json.dumps({"items": items, "fetched_at": datetime.now().isoformat()}).encode())
                return

            # Serve feed catalog (for Settings UI)
            if clean_path in ("/api/feed-catalog/finance", "/api/feed-catalog/politics"):
                feed_type = "finance" if "finance" in clean_path else "politics"
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
                # Return editable parts of config
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

                # Get current user from session
                token = self.headers.get('X-Auth-Token', '')
                current_user = auth.get_session_profile(token)

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
                    self.wfile.write(json.dumps({"error": str(e), "movers": []}).encode())
                return

            if clean_path == "/api/shadow-scores":
                from urllib.parse import urlparse, parse_qs
                query = parse_qs(urlparse(self.path).query)
                try:
                    limit = int(query.get("limit", ["200"])[0])
                except (ValueError, IndexError):
                    limit = 200
                try:
                    min_delta = float(query.get("min_delta", ["0"])[0])
                except (ValueError, IndexError):
                    min_delta = 0.0
                scores = strat.db.get_shadow_scores(limit=limit, min_delta=min_delta)
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

            return super().do_GET()

        def do_POST(self):
            # --- Scan cancellation (no body needed) ---
            if self.path == "/api/scan/cancel":
                strat._scan_cancelled.set()
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b'{"status": "cancelling"}')
                return

            # --- New email-based auth routes (self-authenticating) ---
            if self.path.startswith("/api/auth/") or self.path.startswith("/api/profiles") or self.path.startswith("/api/admin/"):
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length) if content_length else b'{}'
                try:
                    data = json.loads(post_data.decode('utf-8')) if post_data.strip() else {}
                except (json.JSONDecodeError, UnicodeDecodeError):
                    data = {}
                if handle_auth_routes(self, "POST", self.path, data, strat.db, strat, _send_json, email_service):
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
                        strat.active_profile_id = _pid_row[0]
                except Exception:
                    pass

            # --- Rate limiting ---
            if auth.rate_limited(self.path):
                self.send_response(429)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Rate limit exceeded. Try again shortly."}')
                return

            # Save config
            if self.path == "/api/config":
                handle_config_save(self, strat, auth.auth_helpers_dict())
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
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
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
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
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

                    token = self.headers.get('X-Auth-Token', '')
                    user = auth.get_session_profile(token) or 'default'
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
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
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

                    # Inject active profile so training data preserves context
                    profile = strat.config.get("profile", {})
                    data["profile_role"] = profile.get("role", "")
                    data["profile_location"] = profile.get("location", "")
                    data["profile_context"] = profile.get("context", "")[:500]

                    strat.db.save_feedback(data, profile_id=strat.active_profile_id)

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "action": action}).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
                return

            # Feedback stats (GET would be cleaner but keeping consistent with POST pattern)
            if self.path == "/api/feedback-stats":
                try:
                    stats = strat.db.get_feedback_stats()
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(stats).encode())
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
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
                    token = self.headers.get('X-Auth-Token', '')
                    current_user = auth.get_session_profile(token)
                    if not current_user:
                        raise ValueError("Not authenticated")

                    safe = auth.safe_name(current_user)
                    profile_file = auth.profiles_dir() / f"{safe}.yaml"
                    if not profile_file.exists():
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
                    self.send_response(400 if "incorrect" in str(e).lower() or "authenticated" in str(e).lower() else 500)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
                return

            if self.path == "/api/profiles":
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                try:
                    data = json.loads(post_data.decode('utf-8'))
                    action = data.get("action", "")
                    name = data.get("name", "").strip()

                    # Get current user from session for scoping
                    token = self.headers.get('X-Auth-Token', '')
                    current_user = auth.get_session_profile(token)
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
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
                return

            # ── AI Agent Chat (streaming) ──────────────────────────
            if self.path == "/api/agent-chat":
                handle_agent_chat(self, strat, output_dir)
                return

            self.send_response(404)
            self.end_headers()

        def do_DELETE(self):
            # --- Auth route DELETE (profile deletion) ---
            if self.path.startswith("/api/profiles/"):
                if handle_auth_routes(self, "DELETE", self.path, {}, strat.db, strat, _send_json, email_service):
                    return
            self.send_response(404)
            self.end_headers()

        def do_OPTIONS(self):
            # Handle CORS preflight
            self.send_response(200)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Auth-Token, X-Device-Id")
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
