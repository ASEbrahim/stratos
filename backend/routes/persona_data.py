"""
Persona data routes — conversations, persona context, scenarios, preferences,
context compression, workspace export/import.
Extracted from server.py (Sprint 5K Phase 1).
"""

import json
import logging
from datetime import datetime
from urllib.parse import urlparse, parse_qs

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
    """Handle GET requests for persona data routes. Returns True if handled."""

    # ── Persona Context GET endpoints ───────────────
    if path.startswith("/api/persona-context"):
        from processors.persona_context import PersonaContextManager
        pcm = PersonaContextManager(strat.config, db=strat.db)
        parsed = urlparse(handler.path)
        params = parse_qs(parsed.query)
        persona = params.get('persona', [''])[0]

        if parsed.path == '/api/persona-context/list':
            keys = pcm.list_context_keys(handler._profile_id, persona)
            _send_json(handler, {"keys": keys})
            return True
        elif parsed.path == '/api/persona-context/versions':
            versions = pcm.get_versions(handler._profile_id, persona)
            _send_json(handler, {"versions": versions})
            return True
        elif parsed.path == '/api/persona-context':
            key = params.get('key', ['system_context'])[0]
            content = pcm.get_context(handler._profile_id, persona, key)
            _send_json(handler, {"persona": persona, "key": key, "content": content or ''})
            return True

    # ── Conversations GET ───────────────────────────
    if path.startswith("/api/conversations"):
        parsed_url = urlparse(handler.path)
        params = parse_qs(parsed_url.query)
        path_parts = parsed_url.path.rstrip('/').split('/')
        cursor = strat.db.conn.cursor()

        if len(path_parts) == 3:
            # GET /api/conversations?persona=intelligence — list metadata
            persona = params.get('persona', [''])[0]
            cursor.execute(
                "SELECT id, persona, title, is_active, created_at, updated_at, "
                "LENGTH(messages) as msg_size FROM conversations "
                "WHERE profile_id = ? AND archived = 0 "
                + ("AND persona = ? " if persona else "") +
                "ORDER BY updated_at DESC",
                (handler._profile_id, persona) if persona else (handler._profile_id,)
            )
            rows = cursor.fetchall()
            convs = []
            for r in rows:
                d = dict(r)
                try:
                    d['message_count'] = len(json.loads(d.get('messages', '[]') if 'messages' in d.keys() else '[]'))
                except Exception:
                    d['message_count'] = 0
                # Don't send messages in list — get msg_size for count estimation
                msgs_raw = cursor.execute("SELECT messages FROM conversations WHERE id = ?", (d['id'],)).fetchone()
                try:
                    d['message_count'] = len(json.loads(msgs_raw[0])) if msgs_raw else 0
                except Exception:
                    d['message_count'] = 0
                d.pop('msg_size', None)
                convs.append(d)
            _send_json(handler, {"conversations": convs})
            return True

        elif len(path_parts) == 4:
            # GET /api/conversations/:id — full conversation with messages
            try:
                conv_id = int(path_parts[3])
            except ValueError:
                _send_json(handler, {"error": "Invalid conversation ID"}, 400)
                return True
            cursor.execute(
                "SELECT id, profile_id, persona, title, messages, is_active, created_at, updated_at "
                "FROM conversations WHERE id = ? AND profile_id = ?",
                (conv_id, handler._profile_id)
            )
            row = cursor.fetchone()
            if not row:
                _send_json(handler, {"error": "Not found"}, 404)
                return True
            d = dict(row)
            d['messages'] = json.loads(d.get('messages', '[]'))
            d.pop('profile_id', None)
            _send_json(handler, d)
            return True

        _send_json(handler, {"error": "Invalid conversations path"}, 400)
        return True

    # ── Scenario GET ────────────────────────────────
    if path.startswith("/api/scenarios"):
        from processors.scenarios import ScenarioManager
        sm = ScenarioManager(strat.config, db=strat.db)
        params = parse_qs(urlparse(handler.path).query)

        if path.startswith("/api/scenarios/active"):
            active = sm.get_active_scenario(handler._profile_id)
            if active:
                data = sm.get_scenario(handler._profile_id, active)
                _send_json(handler, {"active": active, "data": data})
            else:
                _send_json(handler, {"active": None})
            return True

        name = params.get('name', [''])[0]
        if name:
            data = sm.get_scenario(handler._profile_id, name)
            _send_json(handler, data or {"error": "Scenario not found"}, 200 if data else 404)
        else:
            scenarios = sm.list_scenarios(handler._profile_id)
            _send_json(handler, {"scenarios": scenarios})
        return True

    # ── Context compression GET ─────────────────────
    if path.startswith("/api/persona-state"):
        from processors.context_compression import ContextCompressor
        params = parse_qs(urlparse(handler.path).query)
        persona = params.get('persona', ['intelligence'])[0]
        try:
            cc = ContextCompressor(strat.config, db=strat.db)
            state = cc.get_state(handler._profile_id, persona)
            summaries = cc.get_summaries(handler._profile_id, persona)
            _send_json(handler, {"state": state, "summaries": summaries})
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    # ── Profile workspace GET ───────────────────────
    if path == "/api/profile/workspace-stats":
        from processors.workspace import get_workspace_stats
        try:
            stats = get_workspace_stats(strat, handler._profile_id)
            _send_json(handler, stats)
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    # ── Preference signals GET ──────────────────────
    if path.startswith("/api/preference-signals"):
        params = parse_qs(urlparse(handler.path).query)
        persona = params.get('persona', [''])[0]
        try:
            cursor = strat.db.conn.cursor()
            if persona:
                cursor.execute(
                    "SELECT * FROM user_preference_signals WHERE profile_id = ? AND persona_source = ? ORDER BY created_at DESC",
                    (handler._profile_id, persona)
                )
            else:
                cursor.execute(
                    "SELECT * FROM user_preference_signals WHERE profile_id = ? ORDER BY created_at DESC",
                    (handler._profile_id,)
                )
            _send_json(handler, {"signals": [dict(r) for r in cursor.fetchall()]})
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    return False


def handle_post(handler, strat, auth, path):
    """Handle POST requests for persona data routes. Returns True if handled."""

    # ── Persona Context POST endpoints ──────────────
    if path.startswith("/api/persona-context"):
        from processors.persona_context import PersonaContextManager
        pcm = PersonaContextManager(strat.config, db=strat.db)
        body = json.loads(handler.rfile.read(int(handler.headers.get('Content-Length', 0))).decode()) if int(handler.headers.get('Content-Length', 0)) > 0 else {}

        if path == '/api/persona-context':
            persona = body.get('persona', '')
            key = body.get('key', 'system_context')
            content = body.get('content', '')
            if not persona:
                _send_json(handler, {"error": "No persona specified"}, 400)
                return True
            ok = pcm.save_context(handler._profile_id, persona, key, content)
            _send_json(handler, {"ok": ok})
            return True

        if path == '/api/persona-context/revert':
            persona = body.get('persona', '')
            version = body.get('version', '')
            ok = pcm.revert_to_version(handler._profile_id, persona, version)
            _send_json(handler, {"ok": ok})
            return True

    # ── Conversations POST/PUT ──────────────────────
    if path.startswith("/api/conversations"):
        from urllib.parse import urlparse
        parsed_url = urlparse(handler.path)
        path_parts = parsed_url.path.rstrip('/').split('/')
        content_length = int(handler.headers.get('Content-Length', 0))
        body = json.loads(handler.rfile.read(content_length).decode()) if content_length > 0 else {}
        cursor = strat.db.conn.cursor()

        # POST /api/conversations — create new
        if len(path_parts) == 3 and handler.command == 'POST':
            persona = body.get('persona', 'intelligence')
            title = body.get('title', 'New Chat')
            # Enforce max 10 active per persona — archive oldest
            cursor.execute(
                "SELECT id FROM conversations WHERE profile_id = ? AND persona = ? AND archived = 0 "
                "ORDER BY updated_at ASC",
                (handler._profile_id, persona)
            )
            existing = [r['id'] for r in cursor.fetchall()]
            if len(existing) >= 10:
                cursor.execute("UPDATE conversations SET archived = 1 WHERE id = ?", (existing[0],))
            # Deactivate others
            cursor.execute(
                "UPDATE conversations SET is_active = 0 WHERE profile_id = ? AND persona = ?",
                (handler._profile_id, persona)
            )
            cursor.execute(
                "INSERT INTO conversations (profile_id, persona, title, messages, is_active) "
                "VALUES (?, ?, ?, '[]', 1)",
                (handler._profile_id, persona, title)
            )
            strat.db._commit()
            _send_json(handler, {"ok": True, "id": cursor.lastrowid})
            return True

        # PUT /api/conversations/:id — update
        if len(path_parts) == 4 and handler.command == 'PUT':
            try:
                conv_id = int(path_parts[3])
            except ValueError:
                _send_json(handler, {"error": "Invalid ID"}, 400)
                return True
            # Verify ownership
            cursor.execute("SELECT id FROM conversations WHERE id = ? AND profile_id = ?",
                           (conv_id, handler._profile_id))
            if not cursor.fetchone():
                _send_json(handler, {"error": "Not found"}, 404)
                return True
            if 'messages' in body:
                cursor.execute("UPDATE conversations SET messages = ?, updated_at = ? WHERE id = ?",
                               (json.dumps(body['messages']), datetime.now().isoformat(), conv_id))
            if 'title' in body:
                cursor.execute("UPDATE conversations SET title = ? WHERE id = ?",
                               (body['title'], conv_id))
            if body.get('is_active'):
                # Get persona for this conv
                cursor.execute("SELECT persona FROM conversations WHERE id = ?", (conv_id,))
                row = cursor.fetchone()
                if row:
                    cursor.execute(
                        "UPDATE conversations SET is_active = 0 WHERE profile_id = ? AND persona = ?",
                        (handler._profile_id, row['persona'])
                    )
                cursor.execute("UPDATE conversations SET is_active = 1 WHERE id = ?", (conv_id,))
            strat.db._commit()
            _send_json(handler, {"ok": True})
            return True

        # POST /api/conversations/:id/append — append a message
        if len(path_parts) == 5 and path_parts[4] == 'append':
            try:
                conv_id = int(path_parts[3])
            except ValueError:
                _send_json(handler, {"error": "Invalid ID"}, 400)
                return True
            cursor.execute("SELECT messages FROM conversations WHERE id = ? AND profile_id = ?",
                           (conv_id, handler._profile_id))
            row = cursor.fetchone()
            if not row:
                _send_json(handler, {"error": "Not found"}, 404)
                return True
            messages = json.loads(row['messages'])
            messages.append({
                "id": f"msg_{len(messages)+1}",
                "role": body.get('role', 'user'),
                "content": body.get('content', ''),
                "timestamp": datetime.now().isoformat()
            })
            cursor.execute("UPDATE conversations SET messages = ?, updated_at = ? WHERE id = ?",
                           (json.dumps(messages), datetime.now().isoformat(), conv_id))
            strat.db._commit()
            _send_json(handler, {"ok": True, "message_count": len(messages)})
            return True

        _send_json(handler, {"error": "Invalid conversations path"}, 400)
        return True

    # ── Scenario POST ───────────────────────────────
    if path.startswith("/api/scenarios"):
        from processors.scenarios import ScenarioManager
        sm = ScenarioManager(strat.config, db=strat.db)
        body = json.loads(handler.rfile.read(int(handler.headers.get('Content-Length', 0))).decode()) if int(handler.headers.get('Content-Length', 0)) > 0 else {}

        if path == "/api/scenarios/create":
            result = sm.create_scenario(
                handler._profile_id, body.get('name', ''),
                world_md=body.get('world', ''),
                characters=body.get('characters')
            )
            status = 200 if result.get("ok") else 400
            _send_json(handler, result, status)
            return True

        if path == "/api/scenarios/activate":
            name = body.get('name', '')
            ok = sm.set_active_scenario(handler._profile_id, name)
            _send_json(handler, {"ok": ok}, 200 if ok else 404)
            return True

        if path == "/api/scenarios/save":
            name = body.get('name', '')
            ok = sm.save_scenario(
                handler._profile_id, name,
                state=body.get('state'),
                world=body.get('world'),
                characters=body.get('characters')
            )
            _send_json(handler, {"ok": ok}, 200 if ok else 404)
            return True

        _send_json(handler, {"error": "Unknown scenario action"}, 400)
        return True

    # ── Context compression POST ────────────────────
    if path == "/api/conversation-log":
        from processors.context_compression import ContextCompressor
        body = json.loads(handler.rfile.read(int(handler.headers.get('Content-Length', 0))).decode()) if int(handler.headers.get('Content-Length', 0)) > 0 else {}
        try:
            cc = ContextCompressor(strat.config, db=strat.db)
            persona = body.get('persona', 'intelligence')
            messages = body.get('messages', [])
            cc.log_conversation(handler._profile_id, persona, messages)
            _send_json(handler, {"ok": True})
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    if path == "/api/update-state":
        from processors.context_compression import ContextCompressor
        body = json.loads(handler.rfile.read(int(handler.headers.get('Content-Length', 0))).decode()) if int(handler.headers.get('Content-Length', 0)) > 0 else {}
        try:
            cc = ContextCompressor(strat.config, db=strat.db)
            persona = body.get('persona', 'intelligence')
            messages = body.get('messages', [])
            state = cc.update_state(handler._profile_id, persona, messages)
            _send_json(handler, {"ok": True, "state": state})
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    # ── Profile workspace POST ──────────────────────
    if path == "/api/profile/export":
        from processors.workspace import export_profile
        try:
            content_length = int(handler.headers.get('Content-Length', 0))
            body = json.loads(handler.rfile.read(content_length).decode()) if content_length > 0 else {}
            include_files = body.get('include_files', True)
            include_insights = body.get('include_insights', True)
            zip_buf = export_profile(strat, handler._profile_id, include_files, include_insights)
            zip_data = zip_buf.read()
            handler.send_response(200)
            handler.send_header('Content-Type', 'application/zip')
            handler.send_header('Content-Disposition', f'attachment; filename="stratos_profile_{handler._profile_id}.zip"')
            handler.send_header('Content-Length', str(len(zip_data)))
            handler.end_headers()
            handler.wfile.write(zip_data)
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    if path.startswith("/api/profile/import"):
        from processors.workspace import import_profile
        try:
            content_length = int(handler.headers.get('Content-Length', 0))
            content_type = handler.headers.get('Content-Type', '')
            if 'multipart/form-data' in content_type:
                # Parse multipart: extract ZIP file and strategy
                import cgi
                form = cgi.FieldStorage(fp=handler.rfile, headers=handler.headers,
                                        environ={'REQUEST_METHOD': 'POST',
                                                 'CONTENT_TYPE': content_type})
                zip_data = form['file'].file.read() if 'file' in form else b''
                strategy = form.getvalue('strategy', 'replace')
            else:
                # Raw ZIP body with strategy in query params
                zip_data = handler.rfile.read(content_length)
                params = parse_qs(urlparse(handler.path).query)
                strategy = params.get('strategy', ['replace'])[0]

            if not zip_data:
                _send_json(handler, {"error": "No ZIP data provided"}, 400)
                return True

            result = import_profile(strat, handler._profile_id, zip_data, strategy)
            status = 200 if result.get("ok") else 400
            _send_json(handler, result, status)
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    # ── Preference signals POST ─────────────────────
    if path == "/api/preference-signals":
        body = json.loads(handler.rfile.read(int(handler.headers.get('Content-Length', 0))).decode()) if int(handler.headers.get('Content-Length', 0)) > 0 else {}
        try:
            cursor = strat.db.conn.cursor()
            cursor.execute(
                """INSERT INTO user_preference_signals
                   (profile_id, persona_source, signal_type, signal_key, signal_weight, auto_generated)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(profile_id, persona_source, signal_type, signal_key)
                   DO UPDATE SET signal_weight = excluded.signal_weight""",
                (handler._profile_id, body.get('persona', ''), body.get('type', ''),
                 body.get('key', ''), body.get('weight', 1.0), body.get('auto', 0))
            )
            strat.db._commit()
            _send_json(handler, {"ok": True, "id": cursor.lastrowid})
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    return False


def handle_delete(handler, strat, auth, path):
    """Handle DELETE requests for persona data routes. Returns True if handled."""

    # ── Persona Context Delete ──
    if path.startswith("/api/persona-context"):
        from processors.persona_context import PersonaContextManager
        pcm = PersonaContextManager(strat.config, db=strat.db)
        params = parse_qs(urlparse(handler.path).query)
        persona = params.get('persona', [''])[0]
        key = params.get('key', [''])[0]
        if persona and key:
            pcm.delete_context(handler._profile_id, persona, key)
            _send_json(handler, {"ok": True})
        else:
            _send_json(handler, {"error": "Missing persona or key"}, 400)
        return True

    # ── Conversation Delete ──
    if path.startswith("/api/conversations/"):
        try:
            conv_id = int(handler.path.split("/")[-1])
            cursor = strat.db.conn.cursor()
            cursor.execute(
                "DELETE FROM conversations WHERE id = ? AND profile_id = ?",
                (conv_id, handler._profile_id)
            )
            strat.db._commit()
            _send_json(handler, {"ok": cursor.rowcount > 0})
        except (ValueError, IndexError):
            _send_json(handler, {"error": "Invalid conversation ID"}, 400)
        return True

    # ── Scenario Delete ──
    if path.startswith("/api/scenarios"):
        from processors.scenarios import ScenarioManager
        params = parse_qs(urlparse(handler.path).query)
        name = params.get('name', [''])[0]
        if name:
            sm = ScenarioManager(strat.config, db=strat.db)
            ok = sm.delete_scenario(handler._profile_id, name)
            _send_json(handler, {"ok": ok}, 200 if ok else 404)
        else:
            _send_json(handler, {"error": "Missing scenario name"}, 400)
        return True

    # ── Preference signal Delete ──
    if path.startswith("/api/preference-signals/"):
        try:
            signal_id = int(handler.path.split("/")[-1])
            cursor = strat.db.conn.cursor()
            cursor.execute(
                "DELETE FROM user_preference_signals WHERE id = ? AND profile_id = ?",
                (signal_id, handler._profile_id)
            )
            strat.db._commit()
            _send_json(handler, {"ok": cursor.rowcount > 0})
        except (ValueError, IndexError):
            _send_json(handler, {"error": "Invalid signal ID"}, 400)
        return True

    return False
