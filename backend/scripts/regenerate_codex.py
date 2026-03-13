#!/usr/bin/env python3
"""
Codex Regeneration Script — Incremental rescan + diff output.

Re-scans the codebase for new modules, tables, and endpoints.
Compares against existing codex.json. Adds entries for new items.
Does NOT delete existing entries.

Usage:
    python3 backend/scripts/regenerate_codex.py
    python3 backend/scripts/regenerate_codex.py --dry-run
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from datetime import datetime


def _scan_python_modules(backend_dir: str) -> list:
    """Scan backend Python files and return module info."""
    modules = []
    for subdir in ['', 'routes', 'processors', 'fetchers']:
        dir_path = os.path.join(backend_dir, subdir) if subdir else backend_dir
        if not os.path.isdir(dir_path):
            continue
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith('.py') or fname.startswith('__'):
                continue
            fpath = os.path.join(dir_path, fname)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    first_lines = []
                    for i, line in enumerate(f):
                        if i >= 10:
                            break
                        first_lines.append(line)
                content = ''.join(first_lines)
                # Extract docstring
                doc_match = re.search(r'"""(.+?)"""', content, re.DOTALL)
                docstring = doc_match.group(1).strip().split('\n')[0] if doc_match else ''
                rel_path = os.path.join('backend', subdir, fname) if subdir else os.path.join('backend', fname)
                modules.append({
                    'file': rel_path,
                    'name': fname.replace('.py', ''),
                    'docstring': docstring,
                    'subdir': subdir or 'backend',
                })
            except Exception:
                continue
    return modules


def _scan_js_modules(frontend_dir: str) -> list:
    """Scan frontend JS files and return module info."""
    modules = []
    for fname in sorted(os.listdir(frontend_dir)):
        if not fname.endswith('.js'):
            continue
        fpath = os.path.join(frontend_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                first_lines = []
                for i, line in enumerate(f):
                    if i >= 8:
                        break
                    first_lines.append(line)
            content = ''.join(first_lines)
            # Extract first comment
            comment = ''
            cm = re.search(r'//\s*[=\-]+\s*\n//\s*(.+)', content)
            if cm:
                comment = cm.group(1).strip()
            else:
                cm = re.search(r'\*\s+(.+?)(?:\n|\*)', content)
                if cm:
                    comment = cm.group(1).strip()
            modules.append({
                'file': f'frontend/{fname}',
                'name': fname.replace('.js', ''),
                'docstring': comment,
            })
        except Exception:
            continue
    return modules


def _scan_db_tables(db_path: str) -> list:
    """Get list of database tables."""
    tables = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
    except Exception:
        pass
    return tables


def _get_existing_terms(codex_path: str) -> dict:
    """Load existing codex and return a set of all term names and their categories."""
    try:
        with open(codex_path) as f:
            data = json.load(f)
        terms = {}
        for cat in data['categories']:
            for t in cat['terms']:
                terms[t['term']] = cat['name']
        return terms
    except Exception:
        return {}


def _find_new_items(existing_terms: dict, py_modules: list, js_modules: list, db_tables: list) -> list:
    """Find items that exist in the codebase but not in the codex."""
    new_items = []

    # Check for Python modules not mentioned in any term's files
    existing_files = set()
    for term_name in existing_terms:
        # We'd need the full data to check files, so just check by name
        pass

    # Check by module name — simplistic but effective
    existing_lower = {t.lower() for t in existing_terms}

    for mod in py_modules:
        name = mod['name']
        # Check if any term references this module
        name_variants = [
            name,
            name.replace('_', ' ').title(),
            name.replace('_', ' '),
        ]
        found = any(v.lower() in existing_lower for v in name_variants)
        if not found and mod['subdir'] in ('routes', 'processors', 'fetchers'):
            new_items.append({
                'type': 'python_module',
                'name': name,
                'file': mod['file'],
                'docstring': mod['docstring'],
                'suggested_category': _suggest_category(mod),
            })

    for mod in js_modules:
        name = mod['name']
        name_variants = [
            name,
            name.replace('-', ' ').title(),
            name.replace('-', ' '),
            name + '.js',
        ]
        found = any(v.lower() in existing_lower for v in name_variants)
        if not found:
            new_items.append({
                'type': 'js_module',
                'name': name,
                'file': mod['file'],
                'docstring': mod['docstring'],
                'suggested_category': 'Frontend Modules',
            })

    for table in db_tables:
        if table in ('schema_version', 'sqlite_sequence'):
            continue
        found = table.lower() in existing_lower or table.replace('_', ' ').lower() in existing_lower
        if not found:
            new_items.append({
                'type': 'db_table',
                'name': table,
                'file': 'backend/migrations.py',
                'docstring': f'Database table: {table}',
                'suggested_category': 'Database',
            })

    return new_items


def _suggest_category(mod: dict) -> str:
    """Suggest a codex category for a module."""
    subdir = mod.get('subdir', '')
    name = mod.get('name', '')
    if subdir == 'routes':
        return 'API Endpoints'
    elif subdir == 'processors':
        if 'scor' in name:
            return 'Scoring Pipeline'
        elif 'scenario' in name or 'canon' in name:
            return 'Gaming'
        elif 'youtube' in name or 'lens' in name:
            return 'YouTube Pipeline'
        elif 'tts' in name or 'stt' in name:
            return 'Infrastructure'
        return 'Glossary'
    elif subdir == 'fetchers':
        return 'Data Fetchers'
    return 'Glossary'


def regenerate(codex_path: str, project_root: str, dry_run: bool = False) -> dict:
    """Main regeneration function."""
    backend_dir = os.path.join(project_root, 'backend')
    frontend_dir = os.path.join(project_root, 'frontend')
    db_path = os.path.join(backend_dir, 'strat_os.db')

    print('Scanning codebase...')
    py_modules = _scan_python_modules(backend_dir)
    js_modules = _scan_js_modules(frontend_dir)
    db_tables = _scan_db_tables(db_path)

    print(f'  Python modules: {len(py_modules)}')
    print(f'  JS modules:     {len(js_modules)}')
    print(f'  DB tables:      {len(db_tables)}')

    existing_terms = _get_existing_terms(codex_path)
    print(f'  Existing terms: {len(existing_terms)}')

    new_items = _find_new_items(existing_terms, py_modules, js_modules, db_tables)

    if not new_items:
        print('\nNo new items found. Codex is up to date.')
        return {'new': 0, 'items': []}

    print(f'\nFound {len(new_items)} new item(s):')
    for item in new_items:
        print(f'  [{item["type"]}] {item["name"]} -> {item["suggested_category"]}')
        if item['docstring']:
            print(f'    {item["docstring"][:80]}')

    if dry_run:
        print('\n(dry run — no changes made)')
        return {'new': len(new_items), 'items': new_items}

    # Add new items to codex.json
    with open(codex_path) as f:
        data = json.load(f)

    added = 0
    for item in new_items:
        cat_name = item['suggested_category']
        cat = next((c for c in data['categories'] if c['name'] == cat_name), None)
        if not cat:
            # Create category if it doesn't exist
            cat = {'name': cat_name, 'icon': '', 'terms': []}
            data['categories'].append(cat)

        term = {
            'term': item['name'].replace('_', ' ').replace('-', ' ').title(),
            'definition': item['docstring'] or f'{item["type"].replace("_", " ").title()}: {item["name"]}',
            'files': [item['file']],
            'related': [],
            'added_in': 'Auto-detected',
            'type': item['type'].replace('_', ' '),
        }
        cat['terms'].append(term)
        added += 1

    data['generated'] = datetime.now().strftime('%Y-%m-%d')

    with open(codex_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    total = sum(len(c['terms']) for c in data['categories'])
    print(f'\nAdded {added} terms. Codex now has {total} terms across {len(data["categories"])} categories.')
    return {'new': added, 'items': new_items}


def main():
    parser = argparse.ArgumentParser(description='Regenerate StratOS codex from codebase scan')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be added without modifying codex.json')
    parser.add_argument('--codex', default=None, help='Path to codex.json')
    args = parser.parse_args()

    project_root = str(Path(__file__).resolve().parent.parent.parent)
    codex_path = args.codex or os.path.join(project_root, 'codex.json')

    if not os.path.exists(codex_path):
        print(f'ERROR: codex.json not found at {codex_path}')
        sys.exit(1)

    regenerate(codex_path, project_root, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
