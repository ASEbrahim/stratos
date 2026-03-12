"""Auto-update scenario files after each AI response.

Runs in a background thread AFTER the response has been streamed to the user.
Never blocks the user's next message. Uses synchronous requests.post() to Ollama.
"""

import json
import os
import re
import logging
import requests
from datetime import datetime

from routes.helpers import strip_think_blocks

logger = logging.getLogger(__name__)


def post_response_update(ollama_host, model, scenario_path, user_message, ai_response,
                         mode='gm', active_npc=None):
    """Analyze the exchange and update relevant scenario files.

    Called synchronously in a background thread after each gaming response.
    """
    # 1. ALWAYS: Update scenes/current.md
    _update_current_scene(ollama_host, model, scenario_path, user_message, ai_response)

    # 2. ALWAYS: Increment exchange counter
    _increment_exchange_count(scenario_path)

    # 3. CONDITIONAL: Detect what happened and update accordingly
    analysis = _analyze_exchange(ollama_host, model, user_message, ai_response)
    if not analysis:
        return

    # 3a. NPC introduced -> create NPC folder
    new_npcs = analysis.get('new_npcs', [])
    if new_npcs:
        from processors.scenario_templates import create_npc_folder
        for npc in new_npcs:
            if npc.get('id') and npc.get('name'):
                create_npc_folder(scenario_path, npc['id'], {
                    'name': npc['name'],
                    'short': npc.get('short', ''),
                    'location': npc.get('location', ''),
                    'keywords': npc.get('keywords', [npc['id']]),
                    'can_fight': npc.get('can_fight', False),
                    'can_trade': npc.get('can_trade', False),
                    'profile': npc.get('profile', f"# {npc['name']}\n\n(auto-created from game master narration)"),
                })
                logger.info(f"Auto-created NPC: {npc['name']}")

    # 3b. Location changed -> update _index.json
    new_location = analysis.get('new_location')
    if new_location:
        _update_index_field(scenario_path, 'current_location', new_location)
        active_npcs = analysis.get('active_npcs_in_scene', [])
        if active_npcs:
            _update_index_field(scenario_path, 'active_npcs_in_scene', active_npcs)

    # 3c. Stats changed -> update player/stats.md
    stat_changes = analysis.get('stat_changes')
    if stat_changes and any(v for k, v in stat_changes.items() if v):
        _apply_stat_changes(scenario_path, stat_changes)

    # 3d. Inventory changed -> update player/inventory.md
    inventory_changes = analysis.get('inventory_changes')
    if inventory_changes:
        gained = inventory_changes.get('gained', [])
        lost = inventory_changes.get('lost', [])
        currency = inventory_changes.get('currency_delta', 0)
        if gained or lost or currency:
            _apply_inventory_changes(scenario_path, inventory_changes)

    # 3e. New item discovered -> create item file + update catalog
    new_items = analysis.get('new_items', [])
    if new_items:
        from processors.scenario_templates import create_item_file
        for item in new_items:
            if item.get('id'):
                create_item_file(scenario_path, item)

    # 3f. Quest progress -> update player/quests.md
    quest_changes = analysis.get('quest_changes')
    if quest_changes:
        started = quest_changes.get('started', [])
        completed = quest_changes.get('completed', [])
        if started or completed:
            _apply_quest_changes(scenario_path, quest_changes)

    # 3g. NPC memory update (RP mode)
    if mode == 'immersive' and active_npc:
        _update_npc_memory(ollama_host, model, scenario_path, active_npc,
                           user_message, ai_response)


# ═══════════════════════════════════════════════════════════
# EXCHANGE ANALYSIS
# ═══════════════════════════════════════════════════════════

ANALYSIS_PROMPT = """Analyze this game exchange and identify what changed.

Player said: {user_message}
Game master responded: {ai_response}

Return a JSON object identifying ALL changes. Use null/empty for anything that didn't change:

{{
  "new_npcs": [
    {{"id": "filesystem_safe", "name": "Display Name", "short": "one-line role", "location": "location_id", "keywords": ["kw1"], "can_fight": false, "can_trade": false, "profile": "2-3 sentences about this character"}}
  ],
  "new_location": "location_id or null if player didn't move",
  "active_npcs_in_scene": ["npc_ids present after this exchange"],
  "stat_changes": {{
    "hp_delta": 0,
    "mp_delta": 0,
    "xp_gained": 0,
    "level_up": false
  }},
  "inventory_changes": {{
    "gained": ["item names gained"],
    "lost": ["item names lost or used"],
    "currency_delta": 0
  }},
  "new_items": [
    {{"id": "item_id", "name": "Item Name", "type": "weapon|armor|consumable|key_item", "subtype": "", "rarity": "common", "keywords": ["kw"], "description": "2 sentences"}}
  ],
  "quest_changes": {{
    "started": ["quest names started"],
    "progressed": ["quest names with progress"],
    "completed": ["quest names completed"]
  }}
}}

If nothing changed in a category, use null or empty array.
Return ONLY valid JSON."""


def _analyze_exchange(ollama_host, model, user_message, ai_response):
    """Single LLM call to analyze what changed in the exchange."""
    prompt = ANALYSIS_PROMPT.format(
        user_message=user_message[:500],
        ai_response=ai_response[:1000],
    )

    try:
        r = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
                "think": False,
            },
            timeout=30,
        )
        if r.status_code != 200:
            return None
        text = r.json().get('response', '').strip()
        text = strip_think_blocks(text)
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            text = text[brace_start:brace_end + 1]
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Exchange analysis failed — skipping auto-updates: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# FILE UPDATERS
# ═══════════════════════════════════════════════════════════

def _update_current_scene(ollama_host, model, scenario_path, user_msg, ai_resp):
    """Rewrite scenes/current.md based on what just happened."""
    current = _load_file(scenario_path, 'scenes/current.md') or ''

    prompt = f"""Update this scene description based on what just happened.

Previous scene:
{current[:500]}

Player action: {user_msg[:200]}
Result: {ai_resp[:500]}

Write a new scene description (100-150 words) reflecting the current state:
- Where the player is NOW
- Who is present NOW
- What just happened (1 sentence summary)
- Current mood/atmosphere

Clean prose only. No tables, no emoji headers, no options lists.
Return ONLY the scene text, no JSON, no markdown fences."""

    try:
        r = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.3, "num_predict": 256},
                "think": False,
            },
            timeout=30,
        )
        if r.status_code != 200:
            return
        new_scene = strip_think_blocks(r.json().get('response', '').strip())
        if len(new_scene) > 50:
            # Archive previous scene periodically (every 10 exchanges)
            index = _load_json(scenario_path, '_index.json') or {}
            exchanges = index.get('total_exchanges', 0)
            if exchanges % 10 == 0 and current:
                history_dir = os.path.join(scenario_path, 'scenes', 'history')
                os.makedirs(history_dir, exist_ok=True)
                archive_name = f"{str(exchanges).zfill(3)}_{datetime.now().strftime('%H%M')}.md"
                with open(os.path.join(history_dir, archive_name), 'w') as f:
                    f.write(current)

            _write_file_direct(scenario_path, 'scenes/current.md', f"## Current Scene\n\n{new_scene}")
    except Exception as e:
        logger.warning(f"Scene update failed: {e}")


def _update_npc_memory(ollama_host, model, scenario_path, npc_id_or_name, user_msg, ai_resp):
    """Update an NPC's memory.md after an RP interaction."""
    npc_id = npc_id_or_name.strip().lower().replace(' ', '_')
    memory_path = f'characters/npcs/{npc_id}/memory.md'
    current_memory = _load_file(scenario_path, memory_path) or ''

    prompt = f"""Based on this roleplay exchange, write a brief addition to this NPC's memory.

Current memory:
{current_memory[-500:]}

Player said: {user_msg[:200]}
NPC responded: {ai_resp[:300]}

Write 2-3 sentences to append to the interaction log:
- What happened in this exchange
- Any change in relationship or trust
- Anything new the NPC learned about the player

Return ONLY the text to append. No JSON. No headers."""

    try:
        r = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": model, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.3, "num_predict": 128},
                "think": False,
            },
            timeout=15,
        )
        if r.status_code != 200:
            return
        addition = strip_think_blocks(r.json().get('response', '').strip())
        if addition and len(addition) > 20:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            updated = current_memory.rstrip() + f"\n\n### {timestamp}\n{addition}"
            _write_file_direct(scenario_path, memory_path, updated)
    except Exception as e:
        logger.warning(f"NPC memory update failed: {e}")


def _apply_stat_changes(scenario_path, changes):
    """Apply stat deltas to the player's stats.md."""
    stats = _load_file(scenario_path, 'characters/player/stats.md') or ''
    if not stats:
        return

    hp_delta = changes.get('hp_delta', 0)
    mp_delta = changes.get('mp_delta', 0)
    xp_gained = changes.get('xp_gained', 0)

    if hp_delta:
        hp_match = re.search(r'HP:\s*(\d+)/(\d+)', stats)
        if hp_match:
            current_hp = max(0, min(int(hp_match.group(2)), int(hp_match.group(1)) + hp_delta))
            stats = stats[:hp_match.start()] + f"HP: {current_hp}/{hp_match.group(2)}" + stats[hp_match.end():]

    if mp_delta:
        mp_match = re.search(r'MP:\s*(\d+)/(\d+)', stats)
        if mp_match:
            current_mp = max(0, min(int(mp_match.group(2)), int(mp_match.group(1)) + mp_delta))
            stats = stats[:mp_match.start()] + f"MP: {current_mp}/{mp_match.group(2)}" + stats[mp_match.end():]

    if xp_gained and xp_gained > 0:
        xp_match = re.search(r'XP:\s*(\d+)\s*/\s*(\d+)', stats)
        if xp_match:
            current_xp = int(xp_match.group(1)) + xp_gained
            xp_threshold = int(xp_match.group(2))
            if current_xp >= xp_threshold:
                current_xp -= xp_threshold
                level_match = re.search(r'Level:\s*(\d+)', stats)
                if level_match:
                    new_level = int(level_match.group(1)) + 1
                    stats = stats[:level_match.start()] + f"Level: {new_level}" + stats[level_match.end():]
                xp_threshold = int(xp_threshold * 1.5)
            stats = stats[:xp_match.start()] + f"XP: {current_xp} / {xp_threshold}" + stats[xp_match.end():]

    _write_file_direct(scenario_path, 'characters/player/stats.md', stats)


def _apply_inventory_changes(scenario_path, changes):
    """Apply inventory additions/removals."""
    inv = _load_file(scenario_path, 'characters/player/inventory.md') or ''

    for item in changes.get('gained', []):
        if item and item not in inv:
            inv = inv.rstrip() + f"\n- {item}"

    for item in changes.get('lost', []):
        if item:
            lines = inv.split('\n')
            inv = '\n'.join(l for l in lines if item.lower() not in l.lower())

    currency_delta = changes.get('currency_delta', 0)
    if currency_delta:
        col_match = re.search(r'(\d+)\s*(Col|Gold|GP|coins?|gold)', inv, re.IGNORECASE)
        if col_match:
            new_amount = max(0, int(col_match.group(1)) + currency_delta)
            inv = inv[:col_match.start()] + f"{new_amount} {col_match.group(2)}" + inv[col_match.end():]

    _write_file_direct(scenario_path, 'characters/player/inventory.md', inv)


def _apply_quest_changes(scenario_path, changes):
    """Update quest log."""
    quests = _load_file(scenario_path, 'characters/player/quests.md') or ''

    for quest in changes.get('started', []):
        if quest and quest not in quests:
            quests = quests.replace('## Active Quests\n', f'## Active Quests\n- **{quest}** (just started)\n', 1)

    for quest in changes.get('completed', []):
        if quest:
            quests = re.sub(rf'- \*\*{re.escape(quest)}\*\*[^\n]*\n?', '', quests)
            quests = quests.replace('## Completed\n', f'## Completed\n- **{quest}**\n', 1)

    _write_file_direct(scenario_path, 'characters/player/quests.md', quests)


def _increment_exchange_count(scenario_path):
    """Increment the total_exchanges counter in _index.json."""
    index = _load_json(scenario_path, '_index.json') or {}
    index['total_exchanges'] = index.get('total_exchanges', 0) + 1
    _write_json_direct(scenario_path, '_index.json', index)


def _update_index_field(scenario_path, key, value):
    """Update a single field in _index.json."""
    index = _load_json(scenario_path, '_index.json') or {}
    index[key] = value
    _write_json_direct(scenario_path, '_index.json', index)


# ═══════════════════════════════════════════════════════════
# FILE I/O HELPERS
# ═══════════════════════════════════════════════════════════

def _load_file(scenario_path, relative_path):
    try:
        with open(os.path.join(scenario_path, relative_path), 'r') as f:
            return f.read()
    except Exception:
        return None


def _load_json(scenario_path, relative_path):
    try:
        with open(os.path.join(scenario_path, relative_path), 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _write_file_direct(scenario_path, relative_path, content):
    full_path = os.path.join(scenario_path, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w') as f:
        f.write(content)


def _write_json_direct(scenario_path, relative_path, data):
    full_path = os.path.join(scenario_path, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=2)
