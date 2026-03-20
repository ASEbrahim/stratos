"""
Data endpoints — news data, briefing, status, config, profiles, export, feedback,
search, health, top movers, scan log, shadow scores, agent status/personas.
Extracted from server.py (Sprint 5K Phase 1).
"""

import gzip as _gzip_mod
import hashlib
import json
import logging
import threading
import time
import yaml
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests

logger = logging.getLogger("STRAT_OS")

# Thread-safe cache for /api/hue (replaces function-attribute dict)
_hue_cache = {}
_hue_cache_lock = threading.Lock()


def _send_json(handler, data, status=200):
    """Send a JSON response with proper headers."""
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _get_profile_id(strat, token):
    """Resolve profile_id from auth session token."""
    if not token:
        return None
    cursor = strat.db.conn.cursor()
    cursor.execute("SELECT profile_id FROM sessions WHERE token = ?", (token,))
    row = cursor.fetchone()
    return row[0] if row else None


def handle_get(handler, strat, auth, path, output_dir=None):
    """Handle GET requests for data endpoints. Returns True if handled."""

    # Lightweight briefing-only endpoint (5KB vs 2MB full /api/data)
    if path == "/api/briefing":
        briefings = strat.db.get_recent_briefings(limit=1, profile_id=handler._profile_id)
        if briefings:
            _send_json(handler, briefings[0].get("content", {}))
        else:
            _send_json(handler, {"status": "no_briefing"})
        return True

    # Intelligence Hue — behavioral profile + computed hue score
    if path == "/api/hue":
        try:
            from behavioral import compute_behavioral_profile, compute_hue
            pid = handler._profile_id
            if not pid:
                _send_json(handler, {"hue": {"overall": -1, "error": "no_profile"}, "behavior_summary": {}})
                return True

            # 5-minute cache keyed by (profile_id, time_bucket)
            cache_key = (pid, int(time.time() / 300))
            with _hue_cache_lock:
                if cache_key in _hue_cache:
                    _send_json(handler, _hue_cache[cache_key])
                    return True
                # Clean cache if it grows too large
                if len(_hue_cache) > 100:
                    _hue_cache.clear()

            bp = compute_behavioral_profile(strat.db, pid, days=30, config=strat.config)
            hue = compute_hue(bp)

            # Trim behavioral profile for the wire
            top_cats = sorted(bp.get("category_engagement", {}).items(),
                              key=lambda x: x[1].get("count", 0), reverse=True)[:5]
            top_sources = sorted(bp.get("source_quality", {}).items(),
                                 key=lambda x: x[1].get("count", 0), reverse=True)[:5]

            result = {
                "hue": hue,
                "behavior_summary": {
                    "top_categories": {k: v for k, v in top_cats},
                    "top_sources": {k: v for k, v in top_sources},
                    "usage_patterns": bp.get("usage_patterns", {}),
                    "trajectory": bp.get("trajectory", "unknown"),
                    "alignment": bp.get("alignment", {}),
                    "article_count": bp.get("article_count", 0),
                    "feedback_count": bp.get("feedback_count", 0),
                },
            }
            with _hue_cache_lock:
                _hue_cache[cache_key] = result
            _send_json(handler, result)
        except Exception as e:
            logger.warning(f"Hue computation failed: {e}")
            _send_json(handler, {"hue": {"overall": -1, "error": "computation_failed"}, "behavior_summary": {}})
        return True

    # Serve news_data.json from output directory (per-profile)
    if path == "/api/data" or path == "/news_data.json":
        # Resolve profile-specific output file
        _prof = handler._session_profile
        output_path = strat._get_output_path(_prof) if _prof else strat._output_base
        # If profile-specific file doesn't exist, return empty data (not another profile's)
        if _prof and not output_path.exists():
            _send_json(handler, {"news": [], "market": {}, "briefing": None, "last_updated": None})
            return True
        if output_path.exists():
            raw = output_path.read_bytes()

            # ETag: skip sending if client already has this version
            etag = '"' + hashlib.md5(raw).hexdigest()[:16] + '"'
            if handler.headers.get('If-None-Match') == etag:
                handler.send_response(304)
                handler.end_headers()
                return True

            handler.send_response(200)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.send_header("Cache-Control", "no-cache")
            handler.send_header("ETag", etag)

            # Gzip if client supports it
            if 'gzip' in handler.headers.get('Accept-Encoding', '') and len(raw) > 1024:
                compressed = _gzip_mod.compress(raw)
                handler.send_header("Content-Encoding", "gzip")
                handler.send_header("Content-Length", str(len(compressed)))
                handler.end_headers()
                handler.wfile.write(compressed)
            else:
                handler.send_header("Content-Length", str(len(raw)))
                handler.end_headers()
                handler.wfile.write(raw)
        else:
            handler.send_response(200)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(b'{"error": "No data yet. Run a scan first."}')
        return True

    if path == "/api/status":
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        # Resolve profile_id from token (status is AUTH_EXEMPT so _profile_id may be 0)
        token = handler.headers.get('X-Auth-Token', '')
        _status_pid = handler._profile_id or (_get_profile_id(strat, token) if token else 0) or 0
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
                    except Exception as e:
                        logger.debug(f"Failed to load avatar from profile YAML: {e}")
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
                        except Exception as e:
                            logger.debug(f"Failed to fetch email for profile {_status_pid}: {e}")
        handler.wfile.write(json.dumps(status).encode())
        return True

    if path == "/api/ui-state":
        _send_json(handler, strat.db.get_ui_state(handler._profile_id))
        return True

    # Saved signals (from user_feedback where action='save')
    if path == "/api/saved-signals":
        try:
            cursor = strat.db.conn.cursor()
            cursor.execute(
                """SELECT title, url, root, category, ai_score, note, created_at
                   FROM user_feedback
                   WHERE action = 'save' AND profile_id = ?
                   ORDER BY created_at DESC
                   LIMIT 200""",
                (handler._profile_id,)
            )
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            signals = []
            for row in rows:
                item = dict(zip(cols, row))
                signals.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "root": item.get("root", "global"),
                    "category": item.get("category", ""),
                    "score": item.get("ai_score", 0),
                    "source": item.get("note", ""),
                    "saved_at": item.get("created_at", ""),
                    "summary": "",
                })
            _send_json(handler, {"signals": signals})
        except Exception as e:
            logger.error(f"Failed to load saved signals: {e}")
            _send_json(handler, {"signals": []})
        return True

    # Serve config
    if path == "/api/config":
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
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
            "ui_preferences": cfg.get("ui_preferences", {}),
        }
        handler.wfile.write(json.dumps(editable_config).encode())
        return True

    # List saved profile presets (scoped to logged-in user)
    if path == "/api/profiles":
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()

        current_user = handler._session_profile

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
                except Exception as e:
                    logger.debug(f"Failed to load preset {f}: {e}")

        handler.wfile.write(json.dumps({"presets": presets}).encode())
        return True

    # ── Dashboard Export ──────────────────────────────────
    if path.startswith("/api/export"):
        try:
            query = parse_qs(urlparse(handler.path).query)
            fmt = query.get("format", ["json"])[0]

            # Load current dashboard data (profile-specific)
            _export_path = strat._get_output_path(handler._session_profile) if handler._session_profile else (Path(output_dir) / "news_data.json") if output_dir else strat._output_base
            dashboard = {}
            if _export_path.exists():
                with open(_export_path, "r") as f:
                    dashboard = json.loads(f.read())

            news_items = dashboard.get("news", [])
            scan_log = strat.db.get_scan_log(20, profile_id=handler._profile_id)

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
                handler.send_response(200)
                handler.send_header("Content-Type", "text/csv; charset=utf-8")
                handler.send_header("Content-Disposition", f'attachment; filename="stratos_export_{datetime.now().strftime("%Y%m%d_%H%M")}.csv"')
                handler.send_header("Access-Control-Allow-Origin", "*")
                handler.end_headers()
                handler.wfile.write(data)

            else:
                # Full JSON diagnostic export
                _pid = handler._profile_id
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
                handler.send_response(200)
                handler.send_header("Content-Type", "application/json; charset=utf-8")
                handler.send_header("Content-Disposition", f'attachment; filename="stratos_export_{datetime.now().strftime("%Y%m%d_%H%M")}.json"')
                handler.send_header("Access-Control-Allow-Origin", "*")
                handler.end_headers()
                handler.wfile.write(data)

        except Exception as e:
            logger.error(f"Export failed: {e}")
            handler.send_response(500)
            handler.send_header("Content-type", "application/json")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "Internal server error"}).encode())
        return True

    # Feedback stats
    if path == "/api/feedback-stats":
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        try:
            stats = strat.db.get_feedback_stats(profile_id=handler._profile_id)
        except Exception as e:
            logger.error(f"Endpoint error: {e}")
            stats = {"error": "Internal server error"}
        handler.wfile.write(json.dumps(stats).encode())
        return True

    # Cross-persona search
    if path.startswith("/api/search-all-contexts"):
        from processors.persona_context import PersonaContextManager
        pcm = PersonaContextManager(strat.config, db=strat.db)
        params = parse_qs(urlparse(handler.path).query)
        q = params.get('q', [''])[0]
        results = pcm.search_all_contexts(handler._profile_id, q)
        _send_json(handler, {"results": results})
        return True

    # Shadow scores
    if path == "/api/shadow-scores":
        query = parse_qs(urlparse(handler.path).query)
        try:
            limit = int(query.get("limit", ["200"])[0])
        except (ValueError, IndexError):
            limit = 200
        try:
            min_delta = float(query.get("min_delta", ["0"])[0])
        except (ValueError, IndexError):
            min_delta = 0.0
        scores = strat.db.get_shadow_scores(limit=limit, min_delta=min_delta, profile_id=handler._profile_id)
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
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(json.dumps({"summary": summary, "scores": scores}).encode())
        return True

    if path == "/api/health":
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
            scans = strat.db.get_scan_log(1, profile_id=None)
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
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(json.dumps(health).encode())
        return True

    if path == "/api/top-movers":
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
            handler.send_response(200)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps({"movers": top10}).encode())
        except Exception as e:
            logger.warning(f"Top movers failed: {e}")
            handler.send_response(500)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "Internal server error", "movers": []}).encode())
        return True

    # Scan history log
    if path == "/api/scan-log":
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        log = strat.db.get_scan_log(50, profile_id=handler._profile_id)
        handler.wfile.write(json.dumps(log).encode())
        return True

    # Serve search status (Google API quota)
    if path == "/api/search-status":
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        try:
            from fetchers.google_search import get_search_status
            status = get_search_status(strat.config)
        except ImportError:
            status = {"provider": "duckduckgo", "limit_reached": False}
        handler.wfile.write(json.dumps(status).encode())
        return True

    # Agent status
    if path == "/api/agent-status":
        from routes.agent import handle_agent_status
        handle_agent_status(handler, strat)
        return True

    # Agent personas
    if path == "/api/agent-personas":
        from routes.personas import list_personas
        _send_json(handler, {"personas": list_personas()})
        return True

    return False


def handle_post(handler, strat, auth, path):
    """Handle POST requests for data endpoints. Returns True if handled."""

    # --- UI state sync ---
    if path == "/api/ui-state":
        content_length = int(handler.headers.get('Content-Length', 0))
        post_data = handler.rfile.read(content_length) if content_length else b'{}'
        try:
            body = json.loads(post_data.decode('utf-8')) if post_data.strip() else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}
        # Resolve profile_id: header auth or sendBeacon fallback (_token in body)
        pid = handler._profile_id
        if not pid and body.get('_token'):
            try:
                cursor = strat.db.conn.cursor()
                cursor.execute("SELECT profile_id FROM sessions WHERE token = ?", (body['_token'],))
                row = cursor.fetchone()
                if row:
                    pid = row[0]
            except Exception as e:
                logger.debug(f"Failed to resolve profile from beacon token: {e}")
            body.pop('_token', None)
        if body and pid:
            strat.db.save_ui_state(pid, body)
        _send_json(handler, {"ok": True})
        return True

    # Unsave a signal
    if path == "/api/unsave-signal":
        content_length = int(handler.headers.get('Content-Length', 0))
        post_data = handler.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode('utf-8'))
            url = data.get("url", "")
            if url:
                cursor = strat.db.conn.cursor()
                cursor.execute(
                    "DELETE FROM user_feedback WHERE action = 'save' AND profile_id = ? AND url = ?",
                    (handler._profile_id, url)
                )
                strat.db.conn.commit()
            _send_json(handler, {"ok": True})
        except Exception as e:
            logger.error(f"Failed to unsave signal: {e}")
            _send_json(handler, {"error": str(e)}, 500)
        return True

    # User feedback: click, dismiss, rate
    if path == "/api/feedback":
        content_length = int(handler.headers.get('Content-Length', 0))
        post_data = handler.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode('utf-8'))
            # Expected: {news_id, title, url, root, category, ai_score, user_score?, action}
            # action: "click", "dismiss", "rate", "save"
            action = data.get("action", "click")
            if action not in ("click", "dismiss", "rate", "save", "thumbs_up", "thumbs_down"):
                raise ValueError(f"Unknown feedback action: {action}")

            # Inject active profile so training data preserves context (per-request, not shared config)
            _fb_profile = {}
            if handler._session_profile:
                try:
                    _fb_path = auth.profiles_dir() / f"{auth.safe_name(handler._session_profile)}.yaml"
                    if _fb_path.exists():
                        _fb_profile = yaml.safe_load(_fb_path.read_text()).get("profile", {})
                except Exception as e:
                    logger.debug(f"Failed to load feedback profile: {e}")
            if not _fb_profile:
                _fb_profile = strat.config.get("profile", {})
            data["profile_role"] = _fb_profile.get("role", "")
            data["profile_location"] = _fb_profile.get("location", "")
            data["profile_context"] = _fb_profile.get("context", "")[:500]

            strat.db.save_feedback(data, profile_id=handler._profile_id)

            handler.send_response(200)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps({"status": "ok", "action": action}).encode())
        except Exception as e:
            logger.error(f"Endpoint error: {e}")
            handler.send_response(500)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "Internal server error"}).encode())
        return True

    # Feedback stats (POST variant)
    if path == "/api/feedback-stats":
        try:
            stats = strat.db.get_feedback_stats(profile_id=handler._profile_id)
            handler.send_response(200)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps(stats).encode())
        except Exception as e:
            logger.error(f"Endpoint error: {e}")
            handler.send_response(500)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "Internal server error"}).encode())
        return True

    # Sync Serper credits
    if path == "/api/serper-credits":
        content_length = int(handler.headers.get('Content-Length', 0))
        post_data = handler.rfile.read(content_length)

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

            handler.send_response(200)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps({"status": "updated", "credits": credits}).encode())

        except Exception as e:
            logger.error(f"Endpoint error: {e}")
            handler.send_response(500)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "Internal server error"}).encode())
        return True

    # Dedicated Serper API key save
    if path == "/api/save-serper-key":
        content_length = int(handler.headers.get('Content-Length', 0))
        post_data = handler.rfile.read(content_length)
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
            except Exception as e:
                logger.debug(f"Failed to reset Serper tracker: {e}")

            # Persist to config.yaml
            with open(strat.config_path, "w") as f:
                yaml.dump(strat.config, f, default_flow_style=False, sort_keys=False)

            # Update profile cache so next scan uses new key
            if strat.active_profile:
                strat.cache_profile_config(strat.active_profile)

            logger.info(f"Serper API key updated via dedicated endpoint (last4: ...{new_key[-4:]})")

            handler.send_response(200)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps({"status": "saved"}).encode())
        except Exception as e:
            logger.error(f"Endpoint error: {e}")
            handler.send_response(400)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "Internal server error"}).encode())
        return True

    # ── Update User Profile (name, PIN, avatar) ──────────────
    if path == "/api/update-profile":
        content_length = int(handler.headers.get('Content-Length', 0))
        post_data = handler.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode('utf-8'))
            current_user = handler._session_profile
            if not current_user:
                raise ValueError("Not authenticated")

            safe = auth.safe_name(current_user)
            profile_file = auth.profiles_dir() / f"{safe}.yaml"
            _has_yaml = profile_file.exists()

            # DB-auth users may not have YAML files — handle updates via DB
            if not _has_yaml:
                if handler._profile_id:
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
                        strat.db.save_ui_state(handler._profile_id, avatar_state)
                    # Update display_name in users table
                    new_name = data.get("display_name", "").strip()
                    resp_name = current_user
                    if new_name and len(new_name) >= 2:
                        try:
                            cursor = strat.db.conn.cursor()
                            cursor.execute("""
                                UPDATE users SET display_name = ?
                                WHERE id = (SELECT user_id FROM profiles WHERE id = ? LIMIT 1)
                            """, (new_name, handler._profile_id))
                            strat.db.conn.commit()
                            changes.append("name")
                            resp_name = new_name
                        except Exception as e:
                            logger.warning(f"Failed to update display_name: {e}")
                    _send_json(handler, {
                        "status": "updated",
                        "changes": changes,
                        "profile": {"name": resp_name, "avatar": new_avatar or "", "avatar_image": avatar_image or ""}
                    })
                    return True
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
            if handler._profile_id:
                avatar_state = {}
                if avatar_image and avatar_image.startswith("data:image/"):
                    avatar_state["avatar_image"] = avatar_image
                if new_avatar:
                    avatar_state["avatar"] = new_avatar
                if avatar_state:
                    strat.db.save_ui_state(handler._profile_id, avatar_state)

            # Also update live config if name/role/location changed
            if any(c in changes for c in ("name", "role", "location", "avatar")):
                for key in ("name", "role", "location", "avatar"):
                    val = profile_data.get("profile", {}).get(key)
                    if val is not None:
                        strat.config.setdefault("profile", {})[key] = val

            handler.send_response(200)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps({
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
            handler.send_response(400 if is_client_error else 500)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "Bad request" if is_client_error else "Internal server error"}).encode())
        return True

    # Profile presets CRUD
    if path == "/api/profiles":
        # Use stashed body from auth route guard if available (body already consumed)
        data = getattr(handler, '_stashed_post_data', None)
        if data is None:
            content_length = int(handler.headers.get('Content-Length', 0))
            post_data = handler.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
        try:
            action = data.get("action", "")
            name = data.get("name", "").strip()

            current_user = handler._session_profile
            if not current_user:
                raise ValueError("Not authenticated")

            # User-scoped presets directory
            user_presets_dir = auth.profiles_dir() / auth.safe_name(current_user) / "presets"
            user_presets_dir.mkdir(parents=True, exist_ok=True)

            if not name:
                raise ValueError("Profile name is required")

            # Sanitize filename
            safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
            if not safe_name:
                safe_name = "preset"
            filepath = user_presets_dir / f"{safe_name}.yaml"

            # Path traversal protection: ensure resolved path stays within presets dir
            import os as _os
            if not _os.path.realpath(filepath).startswith(_os.path.realpath(user_presets_dir)):
                _send_json(handler, {"error": "Invalid preset name"}, 400)
                return True

            if action == "save":
                # Save current full config as a preset (no security data)
                import copy
                preset = copy.deepcopy(strat.config)
                preset.pop("cache", None)
                preset.pop("security", None)
                with open(filepath, "w") as f:
                    yaml.dump(preset, f, default_flow_style=False, sort_keys=False)
                handler.send_response(200)
                handler.send_header("Content-type", "application/json")
                handler.send_header("Access-Control-Allow-Origin", "*")
                handler.end_headers()
                handler.wfile.write(json.dumps({"status": "saved", "name": safe_name}).encode())

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
                            user_data_content = yaml.safe_load(uf) or {}
                        saved_sec = user_data_content.get("security", {})
                        import copy as _copy
                        updated = _copy.deepcopy(strat.config)
                        updated.pop("security", None)
                        if saved_sec:
                            updated["security"] = saved_sec
                        with open(user_login_file, "w") as uf:
                            yaml.dump(updated, uf, default_flow_style=False, sort_keys=False)
                    except Exception as e:
                        logger.warning(f"Failed to persist preset to login profile: {e}")
                handler.send_response(200)
                handler.send_header("Content-type", "application/json")
                handler.send_header("Access-Control-Allow-Origin", "*")
                handler.end_headers()
                handler.wfile.write(json.dumps({"status": "loaded", "name": safe_name}).encode())

            elif action == "delete":
                # Safety: only delete from user's presets dir
                if filepath.exists() and str(filepath).startswith(str(user_presets_dir)):
                    filepath.unlink()
                handler.send_response(200)
                handler.send_header("Content-type", "application/json")
                handler.send_header("Access-Control-Allow-Origin", "*")
                handler.end_headers()
                handler.wfile.write(json.dumps({"status": "deleted", "name": safe_name}).encode())

            else:
                raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Endpoint error: {e}")
            handler.send_response(500)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "Internal server error"}).encode())
        return True

    return False
