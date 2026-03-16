"""
Control routes — refresh triggers, scan status, ticker presets, agent warmup.
Extracted from server.py (Sprint 5K Phase 1).
"""

import json
import logging
import os
import threading
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


def _get_profile_id(strat, token):
    """Resolve profile_id from auth session token."""
    if not token:
        return None
    cursor = strat.db.conn.cursor()
    cursor.execute("SELECT profile_id FROM sessions WHERE token = ?", (token,))
    row = cursor.fetchone()
    return row[0] if row else None


def handle_get(handler, strat, auth, path):
    """Handle GET requests for control routes. Returns True if handled."""

    # Serve trigger for full refresh (legacy, still works)
    if path == "/api/refresh":
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(b'{"status": "refresh_triggered"}')
        # Resolve profile_id (refresh is AUTH_EXEMPT)
        _refresh_pid = handler._profile_id or (_get_profile_id(strat, handler.headers.get('X-Auth-Token', '')) or 0)
        threading.Thread(target=strat.run_scan, args=(_refresh_pid,), daemon=True).start()
        return True

    # Serve trigger for market-only refresh (fast)
    if path == "/api/refresh-market":
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(b'{"status": "market_refresh_triggered"}')
        # Trigger market refresh in background with error handling (B7)
        _mkt_pid = handler._profile_id or (_get_profile_id(strat, handler.headers.get('X-Auth-Token', '')) or 0)
        def _safe_market_refresh(pid=_mkt_pid):
            try:
                strat.run_market_refresh(profile_id=pid)
            except Exception as e:
                logger.error(f"Market refresh thread failed: {e}")
                strat.scan_status["is_scanning"] = False
                strat.scan_status["stage"] = "error"
        threading.Thread(target=_safe_market_refresh, daemon=True).start()
        return True

    # Single-ticker live update (for fullscreen chart auto-refresh)
    if path.startswith("/api/market-tick"):
        qs = parse_qs(urlparse(handler.path).query)
        symbol = qs.get("symbol", [""])[0]
        interval = qs.get("interval", ["1m"])[0]
        if not symbol:
            handler.send_response(400)
            handler.send_header("Content-type", "application/json")
            handler.end_headers()
            handler.wfile.write(b'{"error":"symbol required"}')
            return True
        try:
            tick_data = strat.market_fetcher.fetch_single(symbol, interval)
            handler.send_response(200)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps(tick_data).encode())
        except Exception as e:
            logger.error(f"Endpoint error: {e}")
            handler.send_response(500)
            handler.send_header("Content-type", "application/json")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "Internal server error"}).encode())
        return True

    # Serve trigger for news-only refresh (slower, uses API)
    if path == "/api/refresh-news":
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(b'{"status": "news_refresh_triggered"}')
        _refresh_pid = handler._profile_id or (_get_profile_id(strat, handler.headers.get('X-Auth-Token', '')) or 0)
        threading.Thread(target=strat.run_news_refresh, args=(_refresh_pid,), daemon=True).start()
        return True

    # Ticker presets — GET list
    if path == "/api/ticker-presets":
        user = handler._session_profile or 'default'
        presets_dir = os.path.join("profiles", user, "ticker_presets")
        presets = []
        if os.path.isdir(presets_dir):
            for fname in sorted(os.listdir(presets_dir)):
                if fname.endswith('.json'):
                    try:
                        with open(os.path.join(presets_dir, fname)) as pf:
                            preset = json.load(pf)
                            presets.append({"name": preset.get("name", fname[:-5]), "tickers": preset.get("tickers", [])})
                    except Exception as e:
                        logger.debug(f"Failed to load ticker preset {fname}: {e}")
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(json.dumps({"presets": presets}).encode())
        return True

    # Detailed scan progress (for Stop Scan polling)
    if path == "/api/scan/status":
        s = strat.scan_status
        handler.send_response(200)
        handler.send_header("Content-type", "application/json")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        handler.wfile.write(json.dumps({
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
        return True

    return False


def handle_post(handler, strat, auth, path):
    """Handle POST requests for control routes. Returns True if handled."""

    # Cancel an in-progress scan
    if path == "/api/scan/cancel":
        if strat.scan_status.get("is_scanning"):
            strat._scan_cancelled.set()
            _send_json(handler, {"ok": True, "message": "Scan cancellation requested"})
        else:
            _send_json(handler, {"ok": False, "message": "No scan in progress"})
        return True

    # Agent model warmup — pre-load inference model into VRAM
    if path == "/api/agent-warmup":
        try:
            scoring_cfg = strat.config.get("scoring", {})
            ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
            model = scoring_cfg.get("inference_model", "qwen3.5:9b")
            requests.post(f"{ollama_host}/api/generate",
                          json={"model": model, "prompt": "hi", "stream": False,
                                "options": {"num_predict": 1}},
                          timeout=30)
        except Exception as e:
            logger.debug(f"Agent warmup failed: {e}")
        _send_json(handler, {"ok": True})
        return True

    # Ticker presets — save/delete
    if path == "/api/ticker-presets":
        content_length = int(handler.headers.get('Content-Length', 0))
        post_data = handler.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode('utf-8'))
            action = data.get('action', 'save')
            name = data.get('name', '').strip()
            if not name:
                raise ValueError("Preset name is required")

            user = handler._session_profile or 'default'
            presets_dir = os.path.join("profiles", user, "ticker_presets")
            os.makedirs(presets_dir, exist_ok=True)

            # Sanitize filename
            safe_name = "".join(c for c in name if c.isalnum() or c in ' _-').strip()
            if not safe_name:
                safe_name = "preset"
            fpath = os.path.join(presets_dir, safe_name + ".json")

            # Path traversal protection: ensure resolved path stays within presets_dir
            if not os.path.realpath(fpath).startswith(os.path.realpath(presets_dir)):
                _send_json(handler, {"error": "Invalid preset name"}, 400)
                return True

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

            handler.send_response(200)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps(resp).encode())
        except Exception as e:
            logger.error(f"Endpoint error: {e}")
            handler.send_response(400)
            handler.send_header("Content-type", "application/json")
            handler.send_header("Access-Control-Allow-Origin", "*")
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": "Internal server error"}).encode())
        return True

    return False
