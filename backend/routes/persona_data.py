"""
Persona data routes — conversations, persona context, scenarios, preferences,
context compression, workspace export/import.
Extracted from server.py (Sprint 5K Phase 1).
"""

import json
import logging
import re
from datetime import datetime
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("STRAT_OS")

MAX_TITLE_LEN = 200    # Max length for conversation titles, scenario names
MAX_CONTENT_LEN = 50000  # Max length for message content, entity markdown fields
MAX_BODY_SIZE = 2 * 1024 * 1024  # 2 MB max request body for persona data endpoints


def _read_body(handler, max_size=MAX_BODY_SIZE):
    """Read and parse JSON body with size validation. Returns dict or empty dict."""
    content_length = int(handler.headers.get('Content-Length', 0))
    if content_length <= 0:
        return {}
    if content_length > max_size:
        return None  # Caller should return 413
    raw = handler.rfile.read(content_length)
    return json.loads(raw.decode()) if raw else {}


def _sanitize_name(name):
    """Sanitize a name for use in file paths — prevent path traversal."""
    # Strip path separators and null bytes
    name = name.replace('/', '').replace('\\', '').replace('\0', '')
    # Remove .. sequences
    name = name.replace('..', '')
    # Strip leading dots and whitespace
    name = name.lstrip('. ')
    # Only allow alphanumeric, underscore, hyphen, space
    name = re.sub(r'[^\w\s-]', '', name).strip()
    return name[:MAX_TITLE_LEN] if name else ''


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

    # ── Persona Entities GET ─────────────────────────
    if path.startswith("/api/personas/") and "/entities" in path:
        # /api/personas/{persona}/entities?scenario={name}
        # /api/personas/{persona}/entities/{entity_name}?scenario={name}
        parts = path.split("/")  # ['', 'api', 'personas', '{persona}', 'entities', ...]
        if len(parts) >= 5:
            persona = parts[3]
            params = parse_qs(urlparse(handler.path).query)
            scenario = params.get('scenario', [''])[0]
            entity_name = parts[5] if len(parts) > 5 else None

            try:
                cursor = strat.db.conn.cursor()
                if entity_name:
                    cursor.execute(
                        "SELECT * FROM persona_entities WHERE profile_id = ? AND persona = ? AND scenario_name = ? AND name = ?",
                        (handler._profile_id, persona, scenario, entity_name))
                    row = cursor.fetchone()
                    if row:
                        _send_json(handler, dict(row))
                    else:
                        _send_json(handler, {"error": "Entity not found"}, 404)
                else:
                    cursor.execute(
                        "SELECT id, name, display_name, entity_type, personality_md, relationship_md, updated_at "
                        "FROM persona_entities WHERE profile_id = ? AND persona = ? AND scenario_name = ? ORDER BY display_name",
                        (handler._profile_id, persona, scenario))
                    entities = [dict(r) for r in cursor.fetchall()]
                    _send_json(handler, {"entities": entities})
            except Exception as e:
                logger.error(f"Entity GET error: {e}")
                _send_json(handler, {"error": "Failed to retrieve entity data"}, 500)
            return True
        _send_json(handler, {"error": "Invalid entity path"}, 400)
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
        body = _read_body(handler)
        if body is None:
            _send_json(handler, {"error": "Request body too large"}, 413)
            return True

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
        body = _read_body(handler)
        if body is None:
            _send_json(handler, {"error": "Request body too large"}, 413)
            return True
        cursor = strat.db.conn.cursor()

        # POST /api/conversations — create new
        if len(path_parts) == 3 and handler.command == 'POST':
            persona = body.get('persona', 'intelligence')
            title = body.get('title', 'New Chat')[:MAX_TITLE_LEN]
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
                "content": body.get('content', '')[:MAX_CONTENT_LEN],
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
        body = _read_body(handler)
        if body is None:
            _send_json(handler, {"error": "Request body too large"}, 413)
            return True

        if path == "/api/scenarios/create":
            name_raw = body.get('name', '')[:MAX_TITLE_LEN]
            description = body.get('description', body.get('world', ''))[:MAX_CONTENT_LEN]
            genre = body.get('genre', 'fantasy RPG')[:MAX_TITLE_LEN]

            if not name_raw.strip():
                _send_json(handler, {"error": "Scenario name is required"}, 400)
                return True

            # Create DB record (existing flow)
            result = sm.create_scenario(
                handler._profile_id, name_raw,
                world_md=description,
                characters=body.get('characters')
            )
            if not result.get("ok"):
                _send_json(handler, result, 400)
                return True

            safe_name = result.get('name', name_raw)

            # Path traversal prevention — ensure safe_name has no path separators
            if '/' in safe_name or '\\' in safe_name or '..' in safe_name or '\0' in safe_name:
                safe_name = _sanitize_name(safe_name)
                if not safe_name:
                    _send_json(handler, {"error": "Invalid scenario name"}, 400)
                    return True

            # Create file-based folder structure
            try:
                from processors.scenario_templates import create_scenario_skeleton, get_scenario_base_path
                data_dir = strat.config.get("system", {}).get("data_dir", "data")
                base_path = get_scenario_base_path(data_dir, handler._profile_id)
                scenario_path = create_scenario_skeleton(base_path, safe_name)

                # Verify the resolved path is under the expected base
                import os
                resolved = os.path.realpath(scenario_path)
                resolved_base = os.path.realpath(base_path)
                if not resolved.startswith(resolved_base + os.sep) and resolved != resolved_base:
                    logger.warning(f"Path traversal attempt: scenario_path={scenario_path}, base={base_path}")
                    _send_json(handler, {"error": "Invalid scenario path"}, 400)
                    return True
                logger.info(f"Created scenario skeleton: {scenario_path}")

                # Run LLM generation in background thread
                if description.strip():
                    import threading
                    scoring_cfg = strat.config.get("scoring", {})
                    ollama_host = scoring_cfg.get("ollama_host", "http://localhost:11434")
                    model = scoring_cfg.get("inference_model", "qwen3.5:9b")

                    # Track generation status
                    if not hasattr(strat, '_scenario_gen_status'):
                        strat._scenario_gen_status = {}
                    status_key = f"{handler._profile_id}:{safe_name}"
                    strat._scenario_gen_status[status_key] = {"status": "generating", "passes": {}}

                    def _progress_cb(pass_num, pass_name, status):
                        strat._scenario_gen_status[status_key]["passes"][str(pass_num)] = {
                            "name": pass_name, "status": status
                        }
                        if pass_num == 4 and status == "done":
                            strat._scenario_gen_status[status_key]["status"] = "complete"
                            # Update DB world_md with generated setting
                            try:
                                import os
                                setting_path = os.path.join(scenario_path, 'world', 'setting.md')
                                if os.path.exists(setting_path):
                                    with open(setting_path) as f:
                                        setting = f.read()
                                    sm.save_scenario(handler._profile_id, safe_name, world=setting)
                            except Exception as e:
                                logger.warning(f"Failed to save scenario setting from file: {e}")

                    def generate_in_background():
                        try:
                            from processors.scenario_generator import generate_scenario_content
                            generate_scenario_content(
                                ollama_host, scenario_path, safe_name, genre, description,
                                model=model, progress_callback=_progress_cb
                            )
                        except Exception as e:
                            logger.error(f"Scenario generation failed: {e}")
                            strat._scenario_gen_status[status_key]["status"] = "failed"

                    threading.Thread(target=generate_in_background, daemon=True).start()
                    result["status"] = "generating"
                else:
                    result["status"] = "skeleton_only"
            except Exception as e:
                logger.warning(f"Scenario skeleton creation failed: {e}")

            _send_json(handler, result)
            return True

        if path == "/api/scenarios/generate-status":
            profile_id = handler._profile_id
            name = body.get('name', '')
            status_key = f"{profile_id}:{name}"
            gen_status = getattr(strat, '_scenario_gen_status', {}).get(status_key)
            if gen_status:
                _send_json(handler, gen_status)
            else:
                _send_json(handler, {"status": "unknown"})
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

    # ── Persona Entities POST ────────────────────────
    if path.startswith("/api/personas/") and "/entities" in path:
        parts = path.split("/")
        if len(parts) >= 5:
            persona = parts[3]
            body = _read_body(handler)
            if body is None:
                _send_json(handler, {"error": "Request body too large"}, 413)
                return True
            scenario = body.get('scenario', '')
            name = body.get('name', '').strip().lower().replace(' ', '_')
            display_name = body.get('display_name', body.get('name', '')).strip()

            if not name or not display_name:
                _send_json(handler, {"error": "Name is required"}, 400)
                return True

            try:
                cursor = strat.db.conn.cursor()
                # Check if updating existing or creating new
                action = 'update' if len(parts) > 5 else 'create'
                if action == 'update':
                    entity_name = parts[5]
                    updates = []
                    values = []
                    # Allowlist — only these columns can be updated (prevents injection via body keys)
                    _ENTITY_FIELDS = frozenset(['display_name', 'entity_type', 'identity_md', 'personality_md',
                                  'speaking_style_md', 'relationship_md', 'memory_md', 'knowledge_md',
                                  'extra_md', 'auto_save'])
                    for field in _ENTITY_FIELDS:
                        if field in body:
                            updates.append(f"{field} = ?")
                            values.append(body[field])
                    if updates:
                        updates.append("updated_at = ?")
                        values.append(datetime.now().isoformat())
                        values.extend([handler._profile_id, persona, scenario, entity_name])
                        cursor.execute(
                            f"UPDATE persona_entities SET {', '.join(updates)} "
                            "WHERE profile_id = ? AND persona = ? AND scenario_name = ? AND name = ?",
                            values)
                        strat.db._commit()
                    _send_json(handler, {"ok": cursor.rowcount > 0})
                else:
                    cursor.execute(
                        "INSERT INTO persona_entities (profile_id, persona, scenario_name, name, display_name, "
                        "entity_type, identity_md, personality_md, speaking_style_md, relationship_md, "
                        "memory_md, knowledge_md, extra_md) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (handler._profile_id, persona, scenario, name, display_name,
                         body.get('entity_type', 'character'),
                         body.get('identity_md', ''), body.get('personality_md', ''),
                         body.get('speaking_style_md', ''), body.get('relationship_md', ''),
                         body.get('memory_md', ''), body.get('knowledge_md', ''),
                         body.get('extra_md', '')))
                    strat.db._commit()
                    _send_json(handler, {"ok": True, "id": cursor.lastrowid, "name": name})
            except Exception as e:
                if 'UNIQUE constraint' in str(e):
                    _send_json(handler, {"error": f"Entity '{name}' already exists"}, 409)
                else:
                    logger.error(f"Entity POST error: {e}")
                    _send_json(handler, {"error": "Failed to save entity"}, 500)
            return True
        _send_json(handler, {"error": "Invalid entity path"}, 400)
        return True

    # ── Context compression POST ────────────────────
    if path == "/api/conversation-log":
        from processors.context_compression import ContextCompressor
        body = _read_body(handler)
        if body is None:
            _send_json(handler, {"error": "Request body too large"}, 413)
            return True
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
        body = _read_body(handler)
        if body is None:
            _send_json(handler, {"error": "Request body too large"}, 413)
            return True
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
        body = _read_body(handler)
        if body is None:
            _send_json(handler, {"error": "Request body too large"}, 413)
            return True
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

    # ── Entity Delete ──
    if path.startswith("/api/personas/") and "/entities/" in path:
        parts = path.split("/")
        if len(parts) >= 6:
            persona = parts[3]
            entity_name = parts[5]
            params = parse_qs(urlparse(handler.path).query)
            scenario = params.get('scenario', [''])[0]
            try:
                cursor = strat.db.conn.cursor()
                cursor.execute(
                    "DELETE FROM persona_entities WHERE profile_id = ? AND persona = ? AND scenario_name = ? AND name = ?",
                    (handler._profile_id, persona, scenario, entity_name))
                strat.db._commit()
                _send_json(handler, {"ok": cursor.rowcount > 0})
            except Exception as e:
                _send_json(handler, {"error": str(e)}, 500)
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
