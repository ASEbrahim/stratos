"""
Dev endpoints — filesystem context API and prompt builder generation.
Provides live project state for the Sprint Prompt Builder UI.
"""

import glob
import json
import logging
import os
import re
import subprocess
import sqlite3
from pathlib import Path

logger = logging.getLogger("STRAT_OS")

# Project root (two levels up from this file)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)


def _send_json(handler, data, status=200):
    """Send a JSON response with proper headers."""
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _run_git(*args, cwd=None):
    """Run a git command and return stdout, or empty string on error."""
    try:
        result = subprocess.run(
            ['git'] + list(args),
            cwd=cwd or _PROJECT_ROOT,
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return ''


def _read_file_tail(path, lines=80):
    """Read last N lines of a file."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
        return ''.join(all_lines[-lines:])
    except Exception:
        return ''


def _read_file_head(path, lines=50):
    """Read first N lines of a file."""
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            result = []
            for i, line in enumerate(f):
                if i >= lines:
                    break
                result.append(line)
        return ''.join(result)
    except Exception:
        return ''


def _list_files(directory, extension=''):
    """List files in a directory, optionally filtered by extension."""
    try:
        files = sorted(os.listdir(directory))
        if extension:
            files = [f for f in files if f.endswith(extension)]
        return files
    except Exception:
        return []


def _get_db_tables(db_path):
    """Get list of database tables."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception:
        return []


def _get_safety_branches():
    """Get list of safety/pre-* branches."""
    raw = _run_git('branch', '--list', 'pre-*', 'safety/*')
    if not raw:
        return []
    return [b.strip().lstrip('* ') for b in raw.split('\n') if b.strip()]


def _get_sprint_number():
    """Estimate current sprint number from git log."""
    log = _run_git('log', '--oneline', '-100')
    sprint_nums = []
    for line in log.split('\n'):
        m = re.search(r'[Ss]print\s*(\d+)', line)
        if m:
            sprint_nums.append(int(m.group(1)))
    return max(sprint_nums) + 1 if sprint_nums else 1


def _get_pending_items():
    """Extract pending/handoff items from STATE.md."""
    state_path = os.path.join(_PROJECT_ROOT, '_archive', 'strat-docs', 'session-reports', 'STATE.md')
    if not os.path.exists(state_path):
        # Try other locations
        for alt in ['backend/STATE.md', 'STATE.md']:
            alt_path = os.path.join(_PROJECT_ROOT, alt)
            if os.path.exists(alt_path):
                state_path = alt_path
                break
        else:
            return []

    try:
        with open(state_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        # Look for the last "Pending" or "Follow-up" or "TODO" section
        pending = []
        in_pending = False
        for line in content.split('\n'):
            if re.match(r'^#+\s*(Pending|Follow.up|TODO|Next|Handoff)', line, re.IGNORECASE):
                in_pending = True
                continue
            elif in_pending and re.match(r'^#+\s', line):
                break
            elif in_pending and line.strip().startswith('-'):
                pending.append(line.strip().lstrip('- '))
        return pending[-10:]  # Last 10 items
    except Exception:
        return []


def _build_context():
    """Build the full project context dictionary."""
    db_path = os.path.join(_BACKEND_DIR, 'strat_os.db')

    # STATE.md — try multiple locations
    state_path = None
    for candidate in [
        os.path.join(_PROJECT_ROOT, '_archive', 'strat-docs', 'session-reports', 'STATE.md'),
        os.path.join(_BACKEND_DIR, 'STATE.md'),
        os.path.join(_PROJECT_ROOT, 'STATE.md'),
    ]:
        if os.path.exists(candidate):
            state_path = candidate
            break

    claude_md_path = os.path.join(_BACKEND_DIR, 'CLAUDE.md')

    return {
        "git_log": _run_git('log', '--oneline', '-20'),
        "state_md_tail": _read_file_tail(state_path, 80) if state_path else "(STATE.md not found)",
        "claude_md_head": _read_file_head(claude_md_path, 50),
        "sprint_number": _get_sprint_number(),
        "file_structure": {
            "backend_routes": _list_files(os.path.join(_BACKEND_DIR, 'routes'), '.py'),
            "backend_processors": _list_files(os.path.join(_BACKEND_DIR, 'processors'), '.py'),
            "frontend": _list_files(os.path.join(_PROJECT_ROOT, 'frontend'), '.js'),
        },
        "db_tables": _get_db_tables(db_path),
        "test_files": _list_files(os.path.join(_PROJECT_ROOT, 'tests', 'browser'), '.spec.js'),
        "safety_branches": _get_safety_branches(),
        "pending_items": _get_pending_items(),
    }


def handle_get(handler, strat, auth, path):
    """Handle GET requests for dev endpoints. Returns True if handled."""

    if path == "/api/dev/context":
        try:
            context = _build_context()
            _send_json(handler, context)
        except Exception as e:
            logger.error(f"Dev context error: {e}")
            _send_json(handler, {"error": str(e)}, 500)
        return True

    if path == "/api/dev/sprint-log":
        try:
            cursor = strat.db.conn.cursor()
            cursor.execute(
                "SELECT id, sprint_number, sprint_name, owner, status, created_at "
                "FROM sprint_log ORDER BY created_at DESC LIMIT 50"
            )
            rows = cursor.fetchall()
            entries = []
            for row in rows:
                entries.append({
                    "id": row[0], "sprint_number": row[1], "sprint_name": row[2],
                    "owner": row[3], "status": row[4], "created_at": row[5]
                })
            _send_json(handler, {"entries": entries})
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    if path == "/api/dev/sprint-log/prompt":
        # GET with ?id=N to retrieve stored prompt
        from urllib.parse import parse_qs, urlparse
        qs = parse_qs(urlparse(handler.path).query)
        log_id = qs.get('id', [None])[0]
        if not log_id:
            _send_json(handler, {"error": "Missing id parameter"}, 400)
            return True
        try:
            cursor = strat.db.conn.cursor()
            cursor.execute("SELECT generated_prompt FROM sprint_log WHERE id = ?", (int(log_id),))
            row = cursor.fetchone()
            if row:
                _send_json(handler, {"prompt": row[0] or ""})
            else:
                _send_json(handler, {"error": "Not found"}, 404)
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    if path == "/api/dev/templates":
        profile_id = getattr(handler, '_profile_id', 0)
        try:
            cursor = strat.db.conn.cursor()
            cursor.execute(
                "SELECT id, name, template_json, created_at FROM prompt_templates "
                "WHERE profile_id = ? ORDER BY name",
                (profile_id,)
            )
            rows = cursor.fetchall()
            templates = []
            for row in rows:
                templates.append({
                    "id": row[0], "name": row[1],
                    "template": json.loads(row[2]), "created_at": row[3]
                })
            _send_json(handler, {"templates": templates})
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    return False


def handle_post(handler, strat, auth, path):
    """Handle POST requests for dev endpoints. Returns True if handled."""

    if path == "/api/prompt-builder/generate":
        content_length = int(handler.headers.get('Content-Length', 0))
        post_data = handler.rfile.read(content_length) if content_length else b'{}'
        try:
            form_data = json.loads(post_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            _send_json(handler, {"error": "Invalid JSON"}, 400)
            return True

        try:
            context = _build_context()
            prompt = _assemble_prompt(form_data, context)
            _send_json(handler, {"prompt": prompt})
        except Exception as e:
            logger.error(f"Prompt builder error: {e}")
            _send_json(handler, {"error": str(e)}, 500)
        return True

    if path == "/api/dev/sprint-log":
        content_length = int(handler.headers.get('Content-Length', 0))
        post_data = handler.rfile.read(content_length) if content_length else b'{}'
        try:
            data = json.loads(post_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            _send_json(handler, {"error": "Invalid JSON"}, 400)
            return True

        try:
            cursor = strat.db.conn.cursor()
            if data.get('action') == 'update_status':
                cursor.execute(
                    "UPDATE sprint_log SET status = ? WHERE id = ?",
                    (data['status'], data['id'])
                )
                strat.db._commit()
                _send_json(handler, {"ok": True})
            else:
                cursor.execute(
                    "INSERT INTO sprint_log (sprint_number, sprint_name, owner, generated_prompt, status) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (data.get('sprint_number', 0), data.get('sprint_name', ''),
                     data.get('owner', ''), data.get('prompt', ''), 'generated')
                )
                strat.db._commit()
                _send_json(handler, {"ok": True, "id": cursor.lastrowid})
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    if path == "/api/dev/templates":
        content_length = int(handler.headers.get('Content-Length', 0))
        post_data = handler.rfile.read(content_length) if content_length else b'{}'
        try:
            data = json.loads(post_data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            _send_json(handler, {"error": "Invalid JSON"}, 400)
            return True

        profile_id = getattr(handler, '_profile_id', 0)
        try:
            cursor = strat.db.conn.cursor()
            if data.get('action') == 'delete':
                cursor.execute("DELETE FROM prompt_templates WHERE id = ? AND profile_id = ?",
                               (data['id'], profile_id))
            else:
                cursor.execute(
                    "INSERT OR REPLACE INTO prompt_templates (profile_id, name, template_json) "
                    "VALUES (?, ?, ?)",
                    (profile_id, data['name'], json.dumps(data.get('template', {})))
                )
            strat.db._commit()
            _send_json(handler, {"ok": True})
        except Exception as e:
            _send_json(handler, {"error": str(e)}, 500)
        return True

    return False


# =========================================================================
# Prompt Template Engine
# =========================================================================

def _context_block():
    """Static project info block."""
    return f"""## CONTEXT

```
Project root: ~/Downloads/StratOS/StratOS1/
Server: cd backend && python3 main.py --serve --background
Port: 8080
DB: backend/strat_os.db
```"""


def _read_before_starting(relevant_files=None):
    """Standard discovery commands."""
    lines = [
        "## READ BEFORE STARTING",
        "",
        "```bash",
        "cat backend/CLAUDE.md | head -50",
        "git log --oneline -20",
    ]
    if relevant_files:
        for f in relevant_files:
            lines.append(f"cat {f} | head -80")
    lines.extend([
        "ls backend/routes/*.py frontend/*.js",
        "```",
    ])
    return '\n'.join(lines)


def _commit_discipline(branch_name, include_smoke=True):
    """Safety branch and commit rules."""
    lines = [
        "## SAFETY & COMMIT DISCIPLINE",
        "",
        f"```bash",
        f"git branch {branch_name} 2>/dev/null || true",
        f"```",
        "",
        "One commit per logical phase. Descriptive messages. Independently revertable.",
        "",
        "**After EVERY commit:**",
        "```bash",
        "curl -s http://localhost:8080/api/health | head -1",
        'cd backend && python3 -c "import server; print(\'OK\')"',
    ]
    if include_smoke:
        lines.append("npx playwright test tests/browser/smoke.spec.js")
    lines.extend(["```", ""])
    lines.extend([
        "**Rollback:**",
        "```bash",
        f"git reset --hard {branch_name}",
        "```",
    ])
    return '\n'.join(lines)


def _do_not_touch(extra_files=None):
    """Forbidden file list."""
    files = [
        "scorer_adaptive.py", "scorer_base.py", "train_lora.py",
        "export_training.py", "Modelfile.v22", "distill.py",
    ]
    if extra_files:
        files.extend(extra_files)
    return "## DO NOT TOUCH\n\n" + ', '.join(f"`{f}`" for f in files)


def _parallel_session_rules(owner, files_owned, files_forbidden):
    """File ownership for dual-session execution."""
    lines = [
        "## FILE OWNERSHIP (ZERO OVERLAP)",
        "",
        f"| This Session ({owner}) | Other Session |",
        "|---|---|",
    ]
    for f in files_owned:
        lines.append(f"| `{f}` | |")
    if files_forbidden:
        lines.append("")
        lines.append("**Do NOT touch:** " + ', '.join(f"`{f}`" for f in files_forbidden))
    return '\n'.join(lines)


def _handoff_template(sprint_name, branch_name):
    """Handoff block for STATE.md."""
    return f"""## HANDOFF

After all phases complete:

```bash
cat >> backend/STATE.md << 'HANDOFF'

### {sprint_name}
**Commits:** [list all commits]
**New files:** [list]
**Safety branch:** {branch_name}
**Follow-up:** [any next steps]
HANDOFF

git add -A && git commit -m "docs: {sprint_name} handoff"
```"""


def _priority_order(phases):
    """Numbered phase priority list."""
    lines = ["## PRIORITY ORDER", "", "If session runs short, complete in this order:", ""]
    for i, phase in enumerate(phases, 1):
        name = phase.get('name', f'Phase {i}') if isinstance(phase, dict) else str(phase)
        desc = phase.get('description', '') if isinstance(phase, dict) else ''
        line = f"{i}. **{name}**"
        if desc:
            line += f" — {desc}"
        lines.append(line)
    return '\n'.join(lines)


def _rules_section(custom_rules=None):
    """Standard + custom rules."""
    rules = [
        "One commit per logical phase. Descriptive messages.",
        "Run smoke tests after every commit. No regressions.",
        "New files only where possible. Don't restructure existing modules.",
        "Do NOT use the LLM for generation. Template engine is string assembly + file reads.",
        "Do NOT modify existing test files.",
    ]
    if custom_rules:
        rules.extend(custom_rules)
    lines = ["## RULES", ""]
    for i, r in enumerate(rules, 1):
        lines.append(f"{i}. {r}")
    return '\n'.join(lines)


def _auto_context(context):
    """Format live filesystem context as a reference block."""
    lines = [
        "## AUTO-POPULATED CONTEXT (live from filesystem)",
        "",
        "### Recent Commits",
        "```",
        context.get('git_log', '(unavailable)'),
        "```",
        "",
        f"### Sprint Number: {context.get('sprint_number', '?')}",
        "",
        "### File Structure",
    ]

    fs = context.get('file_structure', {})
    for section, files in fs.items():
        if files:
            lines.append(f"**{section}:** {', '.join(files[:20])}")

    tables = context.get('db_tables', [])
    if tables:
        lines.extend(["", f"### DB Tables ({len(tables)})", ', '.join(tables)])

    tests = context.get('test_files', [])
    if tests:
        lines.extend(["", f"### Test Files ({len(tests)})", ', '.join(tests)])

    branches = context.get('safety_branches', [])
    if branches:
        lines.extend(["", f"### Safety Branches", ', '.join(branches)])

    pending = context.get('pending_items', [])
    if pending:
        lines.extend(["", "### Pending Items"])
        for item in pending:
            lines.append(f"- {item}")

    return '\n'.join(lines)


def _assemble_prompt(form_data, context):
    """Assemble a complete sprint prompt from form data + live context."""
    sprint_name = form_data.get('sprint_name', 'Untitled Sprint')
    sprint_num = form_data.get('sprint_number', context.get('sprint_number', '?'))
    owner = form_data.get('owner', 'kirissie')
    feature_spec = form_data.get('feature_spec', '')
    phases = form_data.get('phases', [])
    include = form_data.get('include_sections', {})
    custom_rules = form_data.get('custom_rules', [])
    # Defensive: ensure custom_rules is a list, not a string
    if isinstance(custom_rules, str):
        custom_rules = [r.strip() for r in custom_rules.split('\n') if r.strip()]
    files_owned = form_data.get('files_owned', [])
    files_forbidden = form_data.get('files_forbidden', [])

    branch_name = f"pre-sprint{sprint_num}"

    sections = []

    # Header
    sections.append(f"# StratOS — Sprint {sprint_num}: {sprint_name}\n")

    # Context block
    sections.append(_context_block())

    # Feature spec
    if feature_spec:
        sections.append(f"## FEATURE SPEC\n\n{feature_spec}")

    # Read before starting
    if include.get('read_before_starting', True):
        relevant = files_owned if files_owned else None
        sections.append(_read_before_starting(relevant))

    # Safety branch + commit discipline
    if include.get('safety_branch', True) or include.get('commit_discipline', True):
        sections.append(_commit_discipline(branch_name, include.get('smoke_tests', True)))

    # Do not touch
    if include.get('do_not_touch', True):
        sections.append(_do_not_touch())

    # Parallel session rules
    if include.get('parallel_rules', False) and files_owned:
        sections.append(_parallel_session_rules(owner, files_owned, files_forbidden))

    # Phases
    if phases:
        sections.append("## PHASES\n")
        for i, phase in enumerate(phases, 1):
            name = phase.get('name', f'Phase {i}') if isinstance(phase, dict) else str(phase)
            desc = phase.get('description', '') if isinstance(phase, dict) else ''
            sections.append(f"### Phase {i}: {name}")
            if desc:
                sections.append(f"\n{desc}")
            sections.append("")

    # Priority order
    if include.get('priority_order', True) and phases:
        sections.append(_priority_order(phases))

    # Rules
    sections.append(_rules_section(custom_rules if custom_rules else None))

    # Handoff
    if include.get('handoff', True):
        sections.append(_handoff_template(sprint_name, branch_name))

    # Auto context
    sections.append(_auto_context(context))

    return '\n\n---\n\n'.join(sections)
