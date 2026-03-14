"""LLM-powered scenario content generation.

When a user creates a scenario, this module:
1. Creates the folder skeleton
2. Runs 4 LLM passes to generate all content
3. Parses the LLM output and writes to the correct files
4. The scenario is ready to play immediately

Uses synchronous requests.post() to Ollama (not async).
"""

import json
import os
import re
import logging
import requests
from datetime import datetime

from routes.helpers import strip_think_blocks

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# GENERATION PROMPTS
# ═══════════════════════════════════════════════════════════

PASS_1_WORLD = """You are creating a game world for a {genre} scenario called "{name}".
User's description: {description}

Generate the following as a JSON object with these exact keys:

{{
  "setting": "2-3 paragraphs describing the world, tone, atmosphere, core premise. Max 200 words.",
  "rules": "Game mechanics in prose: death penalty, leveling, combat basics, economy. Max 200 words.",
  "combat": "Combat system details: how attacks work, dice/rolls, damage, defense, healing. Max 150 words.",
  "skills": "Available skill categories and 5-8 starting skills with brief descriptions. Max 150 words.",
  "economy": "Currency name, rough price ranges, how to earn money. Max 100 words.",
  "starting_location_id": "filesystem_safe_id_for_starting_area",
  "starting_location_name": "Human Readable Name",
  "starting_location_description": "The starting area. 100-150 words of prose describing the place, key landmarks, exits to other areas, what the player sees and hears.",
  "additional_locations": [
    {{"id": "area_id", "name": "Area Name", "description": "80-100 words of prose."}}
  ]
}}

CRITICAL RULES:
- All content must be clean narrative prose
- NO markdown tables, NO pipe separators (|), NO emoji section headers
- NO meta-instructions like "Would You Like Me To:" or "Choose your action:"
- Write as if you're briefing a game master, not formatting a wiki page
- Keep each section within its word limit
- Return ONLY valid JSON, no markdown fences, no preamble"""


PASS_2_CHARACTERS = """You are populating the world of "{name}" ({genre}).
World setting: {setting_summary}

Generate characters as JSON:

{{
  "player": {{
    "default_class_options": ["Class1", "Class2", "Class3", "Class4"],
    "stats": "Starting stat block in prose: Level 1, HP, MP, core stats with values, starting XP. Max 80 words.",
    "equipment": "Starting equipment: weapon, armor, accessories. Include stat bonuses. Max 60 words.",
    "starting_inventory": "What the player starts with (items, currency, consumables). Max 60 words."
  }},
  "npcs": [
    {{
      "id": "filesystem_safe_name",
      "name": "Display Name",
      "short": "One-line role description",
      "location": "location_id where they're found",
      "keywords": ["keyword1", "keyword2", "keyword3"],
      "can_fight": false,
      "can_trade": false,
      "profile": "Name, role, appearance, personality in 80-100 words of prose. Include physical description and demeanor.",
      "speaking_style": "How they talk: verbal tics, sentence patterns, tone. 2-3 sentences.",
      "dialogue_samples": "3 example lines this character would say. In quotes, natural speech.",
      "knowledge": "What this NPC knows: world info, rumors, secrets. 2-4 sentences.",
      "stats": "Combat stats if fightable, or 'Non-combatant' if not. Max 40 words."
    }}
  ]
}}

Generate 4-6 NPCs appropriate for the starting area.
CRITICAL: All prose, no tables, no pipes, no emoji headers.
Return ONLY valid JSON."""


PASS_3_ITEMS = """You are creating the item catalog for "{name}" ({genre}).

Generate starting items as JSON:

{{
  "items": [
    {{
      "id": "filesystem_safe_id",
      "name": "Item Name",
      "type": "weapon|armor|consumable|key_item",
      "subtype": "sword|shield|potion|etc",
      "rarity": "common|uncommon|rare|epic|legendary",
      "keywords": ["keyword1", "keyword2"],
      "description": "2-3 sentences: what it looks like, what it does, stat effects."
    }}
  ]
}}

Generate 8-12 items:
- 2-3 weapons (including the starter weapon)
- 2 armor pieces (including starter armor)
- 3-4 consumables (health potions, buff items)
- 1-2 key items or quest items
Return ONLY valid JSON."""


PASS_4_OPENING = """You are writing the opening scene for "{name}" ({genre}).
Starting location: {starting_location}
NPCs present: {npcs_present}

Generate the opening as JSON:

{{
  "current_scene": "Write a vivid opening scene. Where the player is, what they see/hear/smell, who is nearby, the mood. 100-150 words of immersive prose. End with a sense of 'what now?' without listing options — the GM system prompt handles presenting choices.",
  "active_npcs_in_scene": ["npc_id_1", "npc_id_2"],
  "available_quests": [
    {{
      "name": "Quest Name",
      "description": "2 sentences: objective and reward",
      "type": "tutorial|combat|exploration|social"
    }}
  ]
}}

CRITICAL: The scene must be pure immersive prose. NO numbered options. NO "What would you like to do?"
NO stat displays. Just the scene. The game system adds choices and stats separately.
Return ONLY valid JSON."""


# ═══════════════════════════════════════════════════════════
# GENERATION ENGINE
# ═══════════════════════════════════════════════════════════

def generate_scenario_content(ollama_host, scenario_path, name, genre, description,
                              model="qwen3.5:9b", progress_callback=None):
    """Run all 4 generation passes and write files.

    This is called when the user creates a new scenario.
    Runs synchronously — caller should wrap in a thread.

    Args:
        progress_callback: optional callable(pass_num, pass_name, status) for SSE updates
    """
    def _report(pass_num, pass_name, status):
        if progress_callback:
            try:
                progress_callback(pass_num, pass_name, status)
            except Exception:
                pass

    # PASS 1: World
    logger.info(f"Scenario generation pass 1/4: World for {name}")
    _report(1, "World & rules", "generating")
    world_prompt = PASS_1_WORLD.format(name=name, genre=genre, description=description)
    world_data = _llm_json_call(ollama_host, world_prompt, model)

    start_loc_id = ''
    if world_data:
        _write_file(scenario_path, 'world/setting.md', world_data.get('setting', ''))
        _write_file(scenario_path, 'world/rules.md', world_data.get('rules', ''))
        _write_file(scenario_path, 'world/mechanics/combat.md', world_data.get('combat', ''))
        _write_file(scenario_path, 'world/mechanics/skills.md', world_data.get('skills', ''))
        _write_file(scenario_path, 'world/mechanics/economy.md', world_data.get('economy', ''))

        # Write locations
        start_loc_id = world_data.get('starting_location_id', 'starting_area')
        start_loc_name = world_data.get('starting_location_name', 'Starting Area')
        _write_file(scenario_path, f'world/locations/{start_loc_id}.md',
                    f"# {start_loc_name}\n\n{world_data.get('starting_location_description', '')}")

        locations_index = {"locations": [{"id": start_loc_id, "name": start_loc_name, "floor": 1}]}

        for loc in world_data.get('additional_locations', []):
            loc_id = loc.get('id', 'unknown')
            _write_file(scenario_path, f'world/locations/{loc_id}.md',
                        f"# {loc.get('name', loc_id)}\n\n{loc.get('description', '')}")
            locations_index["locations"].append({"id": loc_id, "name": loc.get('name', loc_id)})

        _write_json(scenario_path, 'world/locations/_index.json', locations_index)

        _update_index(scenario_path, {
            "genre": genre,
            "current_location": start_loc_id,
            "created": datetime.now().isoformat(),
        })
        _report(1, "World & rules", "done")
    else:
        _report(1, "World & rules", "failed")

    # PASS 2: Characters
    logger.info(f"Scenario generation pass 2/4: Characters for {name}")
    _report(2, "Characters", "generating")
    setting_summary = world_data.get('setting', '')[:300] if world_data else description
    char_prompt = PASS_2_CHARACTERS.format(name=name, genre=genre, setting_summary=setting_summary)
    char_data = _llm_json_call(ollama_host, char_prompt, model)

    starting_npcs = []
    if char_data:
        player = char_data.get('player', {})
        _write_file(scenario_path, 'characters/player/stats.md', player.get('stats', ''))
        _write_file(scenario_path, 'characters/player/equipment.md', player.get('equipment', ''))
        _write_file(scenario_path, 'characters/player/inventory.md',
                    f"## Equipped\n(see equipment.md)\n\n## Backpack\n{player.get('starting_inventory', '')}")

        class_options = player.get('default_class_options', ['Warrior', 'Mage', 'Rogue', 'Ranger'])
        identity_content = (f"# Player Character\n\nName: (chosen on first interaction)\n"
                            f"Class options: {', '.join(class_options)}\n\n(Identity established through gameplay)")
        _write_file(scenario_path, 'characters/player/identity.md', identity_content)

        # NPC folders
        from processors.scenario_templates import create_npc_folder

        for npc in char_data.get('npcs', []):
            npc_id = npc.get('id', 'unknown')

            profile_content = f"# {npc.get('name', npc_id)}\n\n{npc.get('profile', '')}\n\n## Speaking Style\n{npc.get('speaking_style', '')}"
            dialogue_content = f"# {npc.get('name', npc_id)} — Dialogue\n\n{npc.get('dialogue_samples', '')}"
            knowledge_content = f"# {npc.get('name', npc_id)} — Knowledge\n\n{npc.get('knowledge', '')}"
            stats_content = npc.get('stats', 'Non-combatant')

            create_npc_folder(scenario_path, npc_id, {
                'name': npc.get('name', npc_id),
                'short': npc.get('short', ''),
                'location': npc.get('location', ''),
                'keywords': npc.get('keywords', [npc_id]),
                'can_fight': npc.get('can_fight', False),
                'can_trade': npc.get('can_trade', False),
                'profile': profile_content,
                'stats': stats_content,
                'dialogue': dialogue_content,
                'knowledge': knowledge_content,
            })

            if npc.get('location') == start_loc_id or not npc.get('location'):
                starting_npcs.append(npc_id)

        npc_count = len(char_data.get('npcs', []))
        _report(2, f"Characters ({npc_count} NPCs)", "done")
    else:
        _report(2, "Characters", "failed")

    # PASS 3: Items
    logger.info(f"Scenario generation pass 3/4: Items for {name}")
    _report(3, "Items", "generating")
    items_prompt = PASS_3_ITEMS.format(name=name, genre=genre)
    items_data = _llm_json_call(ollama_host, items_prompt, model)

    if items_data:
        from processors.scenario_templates import create_item_file
        item_count = 0
        for item in items_data.get('items', []):
            if item.get('id'):
                create_item_file(scenario_path, item)
                item_count += 1
        _report(3, f"Items ({item_count} items)", "done")
    else:
        _report(3, "Items", "failed")

    # PASS 4: Opening scene
    logger.info(f"Scenario generation pass 4/4: Opening scene for {name}")
    _report(4, "Opening scene", "generating")
    npcs_present_str = ', '.join(starting_npcs) if starting_npcs else 'various background characters'
    start_loc_desc = world_data.get('starting_location_description', 'the starting area') if world_data else 'the starting area'

    opening_prompt = PASS_4_OPENING.format(
        name=name, genre=genre,
        starting_location=start_loc_desc,
        npcs_present=npcs_present_str
    )
    opening_data = _llm_json_call(ollama_host, opening_prompt, model)

    if opening_data:
        _write_file(scenario_path, 'scenes/current.md',
                    f"## Current Scene\n\n{opening_data.get('current_scene', '')}\n\n"
                    f"## Present NPCs\n{', '.join(opening_data.get('active_npcs_in_scene', starting_npcs))}")

        _update_index(scenario_path, {
            "active_npcs_in_scene": opening_data.get('active_npcs_in_scene', starting_npcs),
        })

        quests = opening_data.get('available_quests', [])
        if quests:
            quests_content = "## Active Quests\n(none yet)\n\n## Available\n"
            for q in quests:
                quests_content += f"- **{q.get('name', 'Unknown')}** — {q.get('description', '')}\n"
            quests_content += "\n## Completed\n(none yet)"
            _write_file(scenario_path, 'characters/player/quests.md', quests_content)
        _report(4, "Opening scene", "done")
    else:
        _report(4, "Opening scene", "failed")

    logger.info(f"Scenario generation complete: {name} — all 4 passes done")
    return True


# ═══════════════════════════════════════════════════════════
# LLM + FILE HELPERS
# ═══════════════════════════════════════════════════════════

def _llm_json_call(ollama_host, prompt, model, max_retries=2):
    """Call the LLM and parse JSON response. Retries on parse failure."""
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.4, "num_predict": 2048},
                    "think": False,
                },
                timeout=120,
            )
            if r.status_code != 200:
                logger.warning(f"Ollama returned {r.status_code}")
                return None

            text = r.json().get('response', '').strip()
            text = strip_think_blocks(text)

            # Strip markdown fences if present
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            # Find JSON object boundaries
            brace_start = text.find('{')
            brace_end = text.rfind('}')
            if brace_start >= 0 and brace_end > brace_start:
                text = text[brace_start:brace_end + 1]

            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                prompt = prompt + "\n\nYour previous response had invalid JSON. Return ONLY a valid JSON object. No text before or after."
            else:
                logger.error(f"JSON parse failed after {max_retries + 1} attempts")
                return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None


def _safe_path(scenario_path, relative_path):
    """Resolve path and ensure it stays within scenario_path (prevent traversal)."""
    full = os.path.realpath(os.path.join(scenario_path, relative_path))
    base = os.path.realpath(scenario_path)
    if not full.startswith(base + os.sep) and full != base:
        raise ValueError(f"Path traversal blocked: {relative_path}")
    return full


def _write_file(scenario_path, relative_path, content):
    """Write content to a file, creating directories as needed."""
    if not content:
        return
    full_path = _safe_path(scenario_path, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w') as f:
        f.write(content)


def _write_json(scenario_path, relative_path, data):
    """Write JSON data to a file."""
    full_path = _safe_path(scenario_path, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=2)


def _update_index(scenario_path, updates):
    """Update _index.json with new values."""
    index_path = os.path.join(scenario_path, '_index.json')
    try:
        with open(index_path, 'r') as f:
            index = json.load(f)
    except Exception:
        index = {}
    index.update(updates)
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
