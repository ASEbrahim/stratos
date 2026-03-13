#!/usr/bin/env python3
"""
Obsidian Vault Export — Reads codex.json, generates an Obsidian vault at ~/StratOS-Vault/.

Each note uses Obsidian [[wikilink]] syntax for related terms, includes YAML frontmatter
with category/files/related/sprint/type/_generated fields.

The _generated: true flag tells the script which notes it owns. Never deletes user-added notes.
Re-running only updates notes whose source data changed. Incremental.

Usage:
    python3 backend/scripts/export_obsidian.py
    python3 backend/scripts/export_obsidian.py --vault ~/my-vault
    python3 backend/scripts/export_obsidian.py --codex codex.json
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path


def _safe_filename(term: str) -> str:
    """Convert a term name to a safe filename for Obsidian notes."""
    # Replace characters that are problematic in filenames
    safe = re.sub(r'[<>:"/\\|?*]', '_', term)
    safe = re.sub(r'\s+', ' ', safe).strip()
    # Obsidian doesn't like leading dots or spaces
    safe = safe.lstrip('. ')
    return safe


def _content_hash(text: str) -> str:
    """SHA-256 hash of content for change detection."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _build_note(term_data: dict, category_name: str, all_terms: dict) -> str:
    """Build a complete Obsidian note with YAML frontmatter and wikilinks."""
    term = term_data['term']
    definition = term_data['definition']
    files = term_data.get('files', [])
    related = term_data.get('related', [])
    added_in = term_data.get('added_in', '')
    term_type = term_data.get('type', '')

    # YAML frontmatter
    lines = ['---']
    lines.append(f'term: "{term}"')
    lines.append(f'category: "{category_name}"')
    if term_type:
        lines.append(f'type: "{term_type}"')
    if added_in:
        lines.append(f'added_in: "{added_in}"')
    if files:
        lines.append('files:')
        for f in files:
            lines.append(f'  - "{f}"')
    if related:
        lines.append('related:')
        for r in related:
            lines.append(f'  - "{r}"')
    lines.append('_generated: true')
    lines.append('---')
    lines.append('')

    # Title
    lines.append(f'# {term}')
    lines.append('')

    # Definition with wikilinks for related terms
    def_text = definition
    for rel in related:
        if rel in all_terms:
            # Replace plain text mentions with wikilinks (case-insensitive, word boundary)
            pattern = re.compile(r'\b' + re.escape(rel) + r'\b', re.IGNORECASE)
            # Only link the first occurrence
            def_text = pattern.sub(f'[[{rel}]]', def_text, count=1)
    lines.append(def_text)
    lines.append('')

    # Metadata section
    if term_type:
        lines.append(f'**Type:** {term_type}')
    if added_in:
        lines.append(f'**Added in:** {added_in}')
    if files:
        lines.append(f'**Files:** {", ".join(f"`{f}`" for f in files)}')
    lines.append(f'**Category:** [[{category_name}]]')
    lines.append('')

    # Related terms as wikilinks
    if related:
        lines.append('## Related')
        lines.append('')
        for r in related:
            lines.append(f'- [[{r}]]')
        lines.append('')

    return '\n'.join(lines)


def _build_category_index(category: dict, all_terms: dict) -> str:
    """Build a category index note."""
    name = category['name']
    icon = category.get('icon', '')
    terms = category['terms']

    lines = ['---']
    lines.append(f'category: "{name}"')
    lines.append(f'icon: "{icon}"')
    lines.append(f'term_count: {len(terms)}')
    lines.append('_generated: true')
    lines.append('---')
    lines.append('')
    lines.append(f'# {icon} {name}')
    lines.append('')
    lines.append(f'{len(terms)} terms in this category.')
    lines.append('')
    lines.append('## Terms')
    lines.append('')
    for t in sorted(terms, key=lambda x: x['term']):
        term_type = t.get('type', '')
        type_tag = f' `{term_type}`' if term_type else ''
        lines.append(f'- [[{t["term"]}]]{type_tag}')
    lines.append('')

    return '\n'.join(lines)


def _build_home_note(data: dict) -> str:
    """Build the vault home/index note."""
    total_terms = sum(len(c['terms']) for c in data['categories'])

    lines = ['---']
    lines.append(f'version: "{data["version"]}"')
    lines.append(f'generated: "{data["generated"]}"')
    lines.append(f'categories: {len(data["categories"])}')
    lines.append(f'terms: {total_terms}')
    lines.append('_generated: true')
    lines.append('---')
    lines.append('')
    lines.append('# StratOS Codebase Codex')
    lines.append('')
    lines.append(f'**{len(data["categories"])} categories, {total_terms} terms**')
    lines.append(f'Generated: {data["generated"]}')
    lines.append('')
    lines.append('## Categories')
    lines.append('')
    for cat in data['categories']:
        icon = cat.get('icon', '')
        lines.append(f'- [[{cat["name"]}|{icon} {cat["name"]}]] ({len(cat["terms"])} terms)')
    lines.append('')
    lines.append('## All Terms')
    lines.append('')
    all_terms_sorted = []
    for cat in data['categories']:
        for t in cat['terms']:
            all_terms_sorted.append((t['term'], cat['name']))
    for term, cat in sorted(all_terms_sorted):
        lines.append(f'- [[{term}]] ({cat})')
    lines.append('')

    return '\n'.join(lines)


def export_vault(codex_path: str, vault_path: str) -> dict:
    """Export codex.json to an Obsidian vault. Returns stats."""
    with open(codex_path) as f:
        data = json.load(f)

    vault = Path(vault_path)
    vault.mkdir(parents=True, exist_ok=True)

    # Build lookup of all term names for wikilink resolution
    all_terms = {}
    for cat in data['categories']:
        for t in cat['terms']:
            all_terms[t['term']] = cat['name']

    stats = {'created': 0, 'updated': 0, 'unchanged': 0, 'skipped_user': 0}

    # Track which files the script owns (for incremental mode)
    hash_file = vault / '.codex_hashes.json'
    try:
        prev_hashes = json.loads(hash_file.read_text())
    except Exception:
        prev_hashes = {}
    new_hashes = {}

    def _write_note(rel_path: str, content: str):
        """Write a note if it changed, respecting user-added notes."""
        full_path = vault / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        content_hash = _content_hash(content)
        new_hashes[rel_path] = content_hash

        if full_path.exists():
            # Check if this is a user-added note (no _generated frontmatter)
            existing = full_path.read_text(encoding='utf-8')
            if '_generated: true' not in existing[:500]:
                stats['skipped_user'] += 1
                return

            # Check if content changed
            if prev_hashes.get(rel_path) == content_hash:
                stats['unchanged'] += 1
                return

            full_path.write_text(content, encoding='utf-8')
            stats['updated'] += 1
        else:
            full_path.write_text(content, encoding='utf-8')
            stats['created'] += 1

    # Home note
    home = _build_home_note(data)
    _write_note('StratOS Codex.md', home)

    # Category index notes + term notes (index lives inside the category folder)
    for cat in data['categories']:
        cat_dir = _safe_filename(cat['name'])
        # Category index as _index.md inside the folder
        cat_note = _build_category_index(cat, all_terms)
        _write_note(f'{cat_dir}/_index.md', cat_note)
        # Individual term notes
        for term in cat['terms']:
            note = _build_note(term, cat['name'], all_terms)
            filename = _safe_filename(term['term']) + '.md'
            _write_note(f'{cat_dir}/{filename}', note)

    # Save hashes for next run
    hash_file.write_text(json.dumps(new_hashes, indent=2))

    return stats


def main():
    parser = argparse.ArgumentParser(description='Export StratOS codex to Obsidian vault')
    parser.add_argument('--codex', default=None, help='Path to codex.json')
    parser.add_argument('--vault', default=None, help='Path to vault directory')
    args = parser.parse_args()

    # Default paths
    project_root = Path(__file__).resolve().parent.parent.parent
    codex_path = args.codex or str(project_root / 'codex.json')
    vault_path = args.vault or str(Path.home() / 'StratOS-Vault')

    if not os.path.exists(codex_path):
        print(f'ERROR: codex.json not found at {codex_path}')
        sys.exit(1)

    print(f'Exporting to: {vault_path}')
    stats = export_vault(codex_path, vault_path)
    print(f'  Created:   {stats["created"]}')
    print(f'  Updated:   {stats["updated"]}')
    print(f'  Unchanged: {stats["unchanged"]}')
    print(f'  Skipped (user notes): {stats["skipped_user"]}')
    print(f'  Total notes: {stats["created"] + stats["updated"] + stats["unchanged"]}')


if __name__ == '__main__':
    main()
