"""Canon Import — fetch franchise data from Fandom wikis and populate scenario files.

Sprint 8A: "Say the name, get the world."
When the user says "Build the SAO world" or "Import Naruto", this module:
1. Resolves the franchise to a Fandom wiki subdomain
2. Fetches key pages (characters, locations, lore) via MediaWiki API
3. Normalizes raw wikitext into game-ready JSON via LLM
4. Writes everything into the scenario file structure
"""

import json
import os
import re
import time
import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# FRANCHISE ALIAS MAP (top 30 + common abbreviations)
# ═══════════════════════════════════════════════════════════

FRANCHISE_ALIASES = {
    # Anime / Manga
    "sao": {"wiki": "swordartonline", "full_name": "Sword Art Online", "genre": "sci-fi fantasy RPG"},
    "sword art online": {"wiki": "swordartonline", "full_name": "Sword Art Online", "genre": "sci-fi fantasy RPG"},
    "naruto": {"wiki": "naruto", "full_name": "Naruto", "genre": "ninja fantasy RPG"},
    "one piece": {"wiki": "onepiece", "full_name": "One Piece", "genre": "pirate adventure RPG"},
    "attack on titan": {"wiki": "attackontitan", "full_name": "Attack on Titan", "genre": "dark fantasy survival"},
    "aot": {"wiki": "attackontitan", "full_name": "Attack on Titan", "genre": "dark fantasy survival"},
    "jujutsu kaisen": {"wiki": "jujutsu-kaisen", "full_name": "Jujutsu Kaisen", "genre": "supernatural action RPG"},
    "jjk": {"wiki": "jujutsu-kaisen", "full_name": "Jujutsu Kaisen", "genre": "supernatural action RPG"},
    "demon slayer": {"wiki": "kimetsu-no-yaiba", "full_name": "Demon Slayer", "genre": "historical fantasy RPG"},
    "kimetsu no yaiba": {"wiki": "kimetsu-no-yaiba", "full_name": "Demon Slayer", "genre": "historical fantasy RPG"},
    "dragon ball": {"wiki": "dragonball", "full_name": "Dragon Ball", "genre": "martial arts fantasy RPG"},
    "dbz": {"wiki": "dragonball", "full_name": "Dragon Ball", "genre": "martial arts fantasy RPG"},
    "bleach": {"wiki": "bleach", "full_name": "Bleach", "genre": "supernatural action RPG"},
    "my hero academia": {"wiki": "myheroacademia", "full_name": "My Hero Academia", "genre": "superhero academy RPG"},
    "mha": {"wiki": "myheroacademia", "full_name": "My Hero Academia", "genre": "superhero academy RPG"},
    "hunter x hunter": {"wiki": "hunterxhunter", "full_name": "Hunter x Hunter", "genre": "adventure RPG"},
    "hxh": {"wiki": "hunterxhunter", "full_name": "Hunter x Hunter", "genre": "adventure RPG"},
    "fullmetal alchemist": {"wiki": "fma", "full_name": "Fullmetal Alchemist", "genre": "steampunk fantasy RPG"},
    "fma": {"wiki": "fma", "full_name": "Fullmetal Alchemist", "genre": "steampunk fantasy RPG"},
    "death note": {"wiki": "deathnote", "full_name": "Death Note", "genre": "psychological thriller"},
    "chainsaw man": {"wiki": "chainsaw-man", "full_name": "Chainsaw Man", "genre": "dark action RPG"},
    "tokyo ghoul": {"wiki": "tokyoghoul", "full_name": "Tokyo Ghoul", "genre": "dark urban fantasy RPG"},
    "vinland saga": {"wiki": "vinlandsaga", "full_name": "Vinland Saga", "genre": "Viking historical RPG"},
    "spy x family": {"wiki": "spy-x-family", "full_name": "Spy x Family", "genre": "spy comedy adventure"},
    "black clover": {"wiki": "blackclover", "full_name": "Black Clover", "genre": "fantasy magic RPG"},
    "solo leveling": {"wiki": "solo-leveling", "full_name": "Solo Leveling", "genre": "dungeon fantasy RPG"},
    # Games
    "witcher": {"wiki": "witcher", "full_name": "The Witcher", "genre": "dark fantasy RPG"},
    "the witcher": {"wiki": "witcher", "full_name": "The Witcher", "genre": "dark fantasy RPG"},
    "dark souls": {"wiki": "darksouls", "full_name": "Dark Souls", "genre": "dark fantasy souls-like RPG"},
    "elden ring": {"wiki": "eldenring", "full_name": "Elden Ring", "genre": "open-world dark fantasy RPG"},
    "skyrim": {"wiki": "elderscrolls", "full_name": "The Elder Scrolls V: Skyrim", "genre": "fantasy RPG"},
    "elder scrolls": {"wiki": "elderscrolls", "full_name": "The Elder Scrolls", "genre": "fantasy RPG"},
    "zelda": {"wiki": "zelda", "full_name": "The Legend of Zelda", "genre": "action-adventure RPG"},
    "legend of zelda": {"wiki": "zelda", "full_name": "The Legend of Zelda", "genre": "action-adventure RPG"},
    "final fantasy": {"wiki": "finalfantasy", "full_name": "Final Fantasy", "genre": "JRPG"},
    "genshin impact": {"wiki": "genshin-impact", "full_name": "Genshin Impact", "genre": "open-world anime RPG"},
    "genshin": {"wiki": "genshin-impact", "full_name": "Genshin Impact", "genre": "open-world anime RPG"},
    "pokemon": {"wiki": "pokemon", "full_name": "Pokémon", "genre": "creature-collection RPG"},
    "persona 5": {"wiki": "megamitensei", "full_name": "Persona 5", "genre": "JRPG"},
    "hollow knight": {"wiki": "hollowknight", "full_name": "Hollow Knight", "genre": "metroidvania RPG"},
    "god of war": {"wiki": "godofwar", "full_name": "God of War", "genre": "mythological action RPG"},
    # TV / Movies / Books
    "avatar": {"wiki": "avatar", "full_name": "Avatar: The Last Airbender", "genre": "elemental fantasy RPG"},
    "avatar the last airbender": {"wiki": "avatar", "full_name": "Avatar: The Last Airbender", "genre": "elemental fantasy RPG"},
    "atla": {"wiki": "avatar", "full_name": "Avatar: The Last Airbender", "genre": "elemental fantasy RPG"},
    "lord of the rings": {"wiki": "lotr", "full_name": "The Lord of the Rings", "genre": "epic fantasy RPG"},
    "lotr": {"wiki": "lotr", "full_name": "The Lord of the Rings", "genre": "epic fantasy RPG"},
    "harry potter": {"wiki": "harrypotter", "full_name": "Harry Potter", "genre": "wizarding school RPG"},
    "star wars": {"wiki": "starwars", "full_name": "Star Wars", "genre": "sci-fi space opera RPG"},
    "game of thrones": {"wiki": "gameofthrones", "full_name": "Game of Thrones", "genre": "medieval political fantasy RPG"},
    "got": {"wiki": "gameofthrones", "full_name": "Game of Thrones", "genre": "medieval political fantasy RPG"},
}


# ═══════════════════════════════════════════════════════════
# FANDOM WIKI CLIENT
# ═══════════════════════════════════════════════════════════

FANDOM_API_TEMPLATE = "https://{wiki}.fandom.com/api.php"
FETCH_DELAY = 0.5  # seconds between requests
MAX_FETCHES = 15   # hard cap per import


class FandomFetcher:
    """Fetches and parses pages from Fandom MediaWiki APIs."""

    def __init__(self, wiki_subdomain):
        self.wiki = wiki_subdomain
        self.base_url = FANDOM_API_TEMPLATE.format(wiki=wiki_subdomain)
        self.fetch_count = 0
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'StratOS-WorldImport/1.0'

    def _get(self, params, timeout=10):
        """Make a rate-limited GET request."""
        if self.fetch_count >= MAX_FETCHES:
            raise RuntimeError(f"Fetch limit ({MAX_FETCHES}) reached")
        self.fetch_count += 1
        time.sleep(FETCH_DELAY)
        params['format'] = 'json'
        r = self.session.get(self.base_url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def wiki_exists(self):
        """Check if this wiki subdomain is valid."""
        try:
            data = self._get({"action": "query", "titles": "Main_Page"})
            pages = data.get("query", {}).get("pages", {})
            return "-1" not in pages
        except Exception:
            return False

    def search_pages(self, query, limit=10):
        """Search for pages matching a query."""
        try:
            data = self._get({"action": "opensearch", "search": query, "limit": limit})
            if isinstance(data, list) and len(data) >= 2:
                return data[1]  # list of page titles
            return []
        except Exception:
            return []

    def get_category_members(self, category, limit=20):
        """Get pages in a category (e.g., 'Category:Characters')."""
        try:
            data = self._get({
                "action": "query", "list": "categorymembers",
                "cmtitle": category, "cmlimit": limit, "cmtype": "page"
            })
            return [m['title'] for m in data.get("query", {}).get("categorymembers", [])]
        except Exception:
            return []

    def get_page_wikitext(self, title):
        """Fetch the raw wikitext content of a page. Follows redirects."""
        try:
            data = self._get({"action": "parse", "page": title, "prop": "wikitext", "redirects": True})
            return data.get("parse", {}).get("wikitext", {}).get("*", "")
        except Exception as e:
            logger.warning(f"Failed to fetch page '{title}': {e}")
            return ""

    def get_page_extract(self, title, max_chars=1500):
        """Fetch a plain-text extract of a page (intro section). Follows redirects."""
        try:
            data = self._get({
                "action": "query", "titles": title, "prop": "extracts",
                "exchars": max_chars, "explaintext": True, "exintro": True,
                "redirects": True
            })
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    return extract
            return ""
        except Exception:
            return ""


def _parse_wikitext(raw):
    """Convert raw wikitext to clean prose for LLM consumption."""
    if not raw:
        return ""
    text = raw

    # Remove nested templates {{...}} by iterating until none remain
    for _ in range(10):
        cleaned = re.sub(r'\{\{[^{}]*\}\}', '', text)
        if cleaned == text:
            break
        text = cleaned
    # Remove any remaining broken template fragments
    text = re.sub(r'\{\{[^}]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[^{]*\}\}', '', text, flags=re.MULTILINE)

    # Remove file/image links
    text = re.sub(r'\[\[(?:File|Image):[^\]]*\]\]', '', text)
    # Convert [[Link|Display]] to Display, [[Link]] to Link
    text = re.sub(r'\[\[[^\]]*\|([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    # Remove ref tags
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove category links
    text = re.sub(r'\[\[Category:[^\]]*\]\]', '', text)
    # Remove redirect notices
    text = re.sub(r'^:.*?redirects here\..*?\n', '', text, flags=re.MULTILINE)
    # Remove table markup
    text = re.sub(r'\{\|[\s\S]*?\|\}', '', text)
    text = re.sub(r'^\|.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^!.*$', '', text, flags=re.MULTILINE)
    # Clean up wiki markup
    text = re.sub(r"'{2,5}", '', text)  # bold/italic markers
    text = re.sub(r'={2,}([^=]+)={2,}', r'\n\1\n', text)  # section headers
    text = re.sub(r'^\*+\s*', '- ', text, flags=re.MULTILINE)  # bullet lists
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'^\s*\n', '', text, flags=re.MULTILINE)
    return text.strip()


# ═══════════════════════════════════════════════════════════
# FRANCHISE RESOLUTION
# ═══════════════════════════════════════════════════════════

def resolve_franchise(name, serper_search_fn=None):
    """Resolve a franchise name to wiki subdomain and metadata.

    Returns: {"wiki": str, "full_name": str, "genre": str} or None
    """
    key = name.lower().strip()

    # 1. Direct alias match
    if key in FRANCHISE_ALIASES:
        return FRANCHISE_ALIASES[key]

    # 2. Substring match in alias keys
    for alias_key, info in FRANCHISE_ALIASES.items():
        if alias_key in key or key in alias_key:
            return info
        if info['full_name'].lower() in key or key in info['full_name'].lower():
            return info

    # 3. Try slugified name directly as fandom subdomain
    slug = re.sub(r'[^a-z0-9]', '', key)
    fetcher = FandomFetcher(slug)
    try:
        fetcher.fetch_count = 0  # don't count validation against import budget
        if fetcher.wiki_exists():
            return {"wiki": slug, "full_name": name.title(), "genre": "RPG"}
    except Exception:
        pass

    # 4. Web search fallback
    if serper_search_fn:
        try:
            results = serper_search_fn(f'"{name}" site:fandom.com wiki', num_results=3)
            for r in (results or []):
                url = r.get('url', '') or r.get('link', '')
                m = re.search(r'https?://([a-z0-9-]+)\.fandom\.com', url)
                if m:
                    wiki_sub = m.group(1)
                    return {"wiki": wiki_sub, "full_name": name.title(), "genre": "RPG"}
        except Exception as e:
            logger.warning(f"Serper fallback failed: {e}")

    return None


# ═══════════════════════════════════════════════════════════
# LLM NORMALIZATION PROMPTS
# ═══════════════════════════════════════════════════════════

NORMALIZE_CHARACTER = """You are converting a wiki article about a character from {franchise} into game-ready data.

Raw wiki content:
{wiki_text}

Convert to this exact JSON structure:
{{
  "id": "filesystem_safe_name_lowercase",
  "name": "Display Name",
  "short": "One-line role description (max 10 words)",
  "location": "where they're typically found",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "can_fight": true,
  "can_trade": false,
  "profile": "80-100 words: appearance, personality, role in the story. Written as character profile prose.",
  "speaking_style": "How they talk: verbal tics, sentence patterns, tone. 2-3 sentences.",
  "dialogue_samples": "3 example lines this character would say, canon-accurate. In quotes.",
  "knowledge": "What this character knows about the world, other characters, secrets. 2-4 sentences.",
  "stats": "Power level, combat abilities, special techniques. 2-3 sentences."
}}

CRITICAL: Return ONLY valid JSON. No markdown fences. No text before or after."""

NORMALIZE_LOCATION = """You are converting a wiki article about a location from {franchise} into game-ready data.

Raw wiki content:
{wiki_text}

Convert to this exact JSON structure:
{{
  "id": "filesystem_safe_id_lowercase",
  "name": "Location Name",
  "description": "100-150 words: what the place looks like, atmosphere, key landmarks, who can be found here, what activities are available. Vivid prose suitable for a game master to read aloud.",
  "keywords": ["keyword1", "keyword2"]
}}

CRITICAL: Return ONLY valid JSON. No markdown fences."""

NORMALIZE_WORLD = """You are creating a game world overview from wiki lore about {franchise}.

Source material:
{wiki_text}

Generate the following as a JSON object:
{{
  "setting": "2-3 paragraphs describing the world, tone, atmosphere, core premise of {franchise}. Max 250 words. Written as game world lore.",
  "rules": "How the world works: power system, special abilities, technology, magic, key mechanics. Max 200 words. Written as game rules.",
  "combat": "How combat/conflict works in this universe. Key abilities, power scaling, fighting styles. Max 150 words.",
  "skills": "Key abilities/skills characters can have in this world. List 6-10 with brief descriptions. Max 150 words.",
  "factions": [
    {{"id": "faction_id", "name": "Faction Name", "description": "50-80 words about this faction's role, goals, members."}}
  ],
  "power_system": "Detailed explanation of the power/magic system. How powers work, ranks, limitations. Max 200 words."
}}

CRITICAL: Return ONLY valid JSON. No markdown fences."""


# ═══════════════════════════════════════════════════════════
# IMPORT PIPELINE
# ═══════════════════════════════════════════════════════════

def run_canon_import(ollama_host, model, scenario_path, franchise_info, progress_callback=None):
    """Run the full canon import pipeline.

    Args:
        ollama_host: Ollama API base URL
        model: LLM model name for normalization
        scenario_path: Root path of the scenario to populate
        franchise_info: {"wiki": str, "full_name": str, "genre": str}
        progress_callback: optional callable(pass_num, pass_name, status)

    Returns True on success, False on failure.
    """
    wiki_sub = franchise_info['wiki']
    franchise = franchise_info['full_name']
    genre = franchise_info.get('genre', 'RPG')

    def _report(pass_num, pass_name, status):
        if progress_callback:
            try:
                progress_callback(pass_num, pass_name, status)
            except Exception:
                pass

    fetcher = FandomFetcher(wiki_sub)

    # Create _sources directory for caching
    sources_dir = os.path.join(scenario_path, '_sources')
    os.makedirs(sources_dir, exist_ok=True)

    def _cache_source(page_title, content):
        """Cache a raw wiki response."""
        safe_name = re.sub(r'[^a-z0-9_]', '_', page_title.lower())[:60]
        cache_path = os.path.join(sources_dir, f"{safe_name}.json")
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    "source": "fandom",
                    "wiki": wiki_sub,
                    "page": page_title,
                    "fetched_at": datetime.now().isoformat(),
                    "content": content[:10000],
                }, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # ─── PASS 0: Discovery ───
    logger.info(f"Canon import pass 0: Discovery for {franchise} ({wiki_sub}.fandom.com)")
    _report(0, f"Discovering {franchise} wiki...", "generating")

    # Get main page for world lore
    main_text = ""
    try:
        main_text = fetcher.get_page_extract(franchise, max_chars=2000)
        if not main_text:
            main_text = fetcher.get_page_extract("Main Page", max_chars=2000)
        if main_text:
            _cache_source("main_page", main_text)
    except Exception as e:
        logger.warning(f"Main page fetch failed: {e}")

    # Discover characters
    character_titles = []
    for cat in ["Category:Characters", "Category:Main Characters", "Category:Main characters"]:
        titles = fetcher.get_category_members(cat, limit=20)
        if titles:
            character_titles = titles
            break

    if not character_titles:
        # Fallback: search for "characters"
        character_titles = fetcher.search_pages("characters", limit=12)
        # Filter out category/list pages
        character_titles = [t for t in character_titles if not t.lower().startswith(("category:", "list of"))]

    # Discover locations
    location_titles = []
    for cat in ["Category:Locations", "Category:Places", "Category:Areas"]:
        titles = fetcher.get_category_members(cat, limit=10)
        if titles:
            location_titles = titles
            break

    if not location_titles:
        location_titles = fetcher.search_pages("locations", limit=6)
        location_titles = [t for t in location_titles if not t.lower().startswith(("category:", "list of"))]

    # Search for power system / magic system page
    power_titles = fetcher.search_pages("power system OR magic system OR abilities", limit=3)

    _report(0, f"Found {len(character_titles)} characters, {len(location_titles)} locations", "done")
    logger.info(f"Discovery: {len(character_titles)} chars, {len(location_titles)} locs, {len(power_titles)} power pages")

    # ─── PASS 1: World Lore ───
    logger.info(f"Canon import pass 1: World lore for {franchise}")
    _report(1, "World lore & power system", "generating")

    # Gather world source material
    world_source_parts = []
    if main_text:
        world_source_parts.append(f"Main article:\n{main_text}")

    # Fetch power system page if found
    for pt in power_titles[:2]:
        if fetcher.fetch_count >= MAX_FETCHES - 5:
            break  # Reserve fetches for characters
        raw = fetcher.get_page_wikitext(pt)
        if raw:
            clean = _parse_wikitext(raw)[:1500]
            if len(clean) > 100:
                world_source_parts.append(f"Power/abilities page ({pt}):\n{clean}")
                _cache_source(pt, clean)

    world_text = "\n\n".join(world_source_parts) if world_source_parts else f"The world of {franchise}"

    world_data = _llm_json_call(ollama_host, NORMALIZE_WORLD.format(
        franchise=franchise, wiki_text=world_text[:4000]
    ), model)

    if world_data:
        from processors.scenario_generator import _write_file, _write_json, _update_index

        _write_file(scenario_path, 'world/setting.md',
                    f"# {franchise}\n\n{world_data.get('setting', '')}")
        _write_file(scenario_path, 'world/rules.md',
                    f"# World Rules\n\n{world_data.get('rules', '')}")
        _write_file(scenario_path, 'world/mechanics/combat.md',
                    f"# Combat System\n\n{world_data.get('combat', '')}")
        _write_file(scenario_path, 'world/mechanics/skills.md',
                    f"# Skills & Abilities\n\n{world_data.get('skills', '')}")

        # Power system
        power_system = world_data.get('power_system', '')
        if power_system:
            _write_file(scenario_path, 'world/mechanics/power_system.md',
                        f"# Power System\n\n{power_system}")

        # Factions
        factions = world_data.get('factions', [])
        factions_index = {"factions": []}
        for fac in factions:
            fac_id = fac.get('id', 'unknown')
            _write_file(scenario_path, f'world/factions/{fac_id}.md',
                        f"# {fac.get('name', fac_id)}\n\n{fac.get('description', '')}")
            factions_index["factions"].append({
                "id": fac_id, "name": fac.get('name', fac_id)
            })
        if factions:
            _write_json(scenario_path, 'world/factions/_index.json', factions_index)

        _update_index(scenario_path, {
            "genre": genre,
            "source": "canon_import",
            "franchise": franchise,
            "wiki": f"{wiki_sub}.fandom.com",
            "import_date": datetime.now().isoformat(),
            "created": datetime.now().isoformat(),
        })
        _report(1, "World lore & power system", "done")
    else:
        _report(1, "World lore & power system", "failed")

    # ─── PASS 2: Characters ───
    logger.info(f"Canon import pass 2: Characters for {franchise}")
    char_limit = min(12, len(character_titles))
    _report(2, f"Characters (0/{char_limit})", "generating")

    from processors.scenario_templates import create_npc_folder
    from processors.scenario_generator import _write_file as write_f

    starting_npcs = []
    npc_count = 0

    for i, title in enumerate(character_titles[:char_limit]):
        if fetcher.fetch_count >= MAX_FETCHES:
            logger.warning("Fetch limit reached during character import")
            break

        # Use extracts (lighter) first, fallback to wikitext
        raw = fetcher.get_page_extract(title, max_chars=2000)
        if not raw or len(raw) < 100:
            raw = fetcher.get_page_wikitext(title)
            raw = _parse_wikitext(raw)[:2000]

        if not raw or len(raw) < 50:
            continue

        _cache_source(title, raw)
        _report(2, f"Characters ({i+1}/{char_limit}): {title}", "generating")

        char_data = _llm_json_call(ollama_host, NORMALIZE_CHARACTER.format(
            franchise=franchise, wiki_text=raw[:2000]
        ), model)

        if not char_data:
            continue

        npc_id = char_data.get('id', re.sub(r'[^a-z0-9_]', '_', title.lower())[:30])

        profile_content = f"# {char_data.get('name', title)}\n\n{char_data.get('profile', '')}\n\n## Speaking Style\n{char_data.get('speaking_style', '')}"
        dialogue_content = f"# {char_data.get('name', title)} — Dialogue\n\n{char_data.get('dialogue_samples', '')}"
        knowledge_content = f"# {char_data.get('name', title)} — Knowledge\n\n{char_data.get('knowledge', '')}"
        stats_content = char_data.get('stats', '')

        create_npc_folder(scenario_path, npc_id, {
            'name': char_data.get('name', title),
            'short': char_data.get('short', ''),
            'location': char_data.get('location', ''),
            'keywords': char_data.get('keywords', [npc_id]),
            'can_fight': char_data.get('can_fight', True),
            'can_trade': char_data.get('can_trade', False),
            'profile': profile_content,
            'stats': stats_content,
            'dialogue': dialogue_content,
            'knowledge': knowledge_content,
        })

        starting_npcs.append(npc_id)
        npc_count += 1

    _report(2, f"Characters ({npc_count} imported)", "done")

    # ─── PASS 3: Locations ───
    logger.info(f"Canon import pass 3: Locations for {franchise}")
    loc_limit = min(5, len(location_titles))
    _report(3, f"Locations (0/{loc_limit})", "generating")

    locations_index = {"locations": []}
    loc_count = 0
    first_loc_id = ""

    for i, title in enumerate(location_titles[:loc_limit]):
        if fetcher.fetch_count >= MAX_FETCHES:
            break

        raw = fetcher.get_page_extract(title, max_chars=1500)
        if not raw or len(raw) < 50:
            raw = fetcher.get_page_wikitext(title)
            raw = _parse_wikitext(raw)[:1500]

        if not raw or len(raw) < 30:
            continue

        _cache_source(title, raw)
        _report(3, f"Locations ({i+1}/{loc_limit}): {title}", "generating")

        loc_data = _llm_json_call(ollama_host, NORMALIZE_LOCATION.format(
            franchise=franchise, wiki_text=raw[:1500]
        ), model)

        if not loc_data:
            continue

        loc_id = loc_data.get('id', re.sub(r'[^a-z0-9_]', '_', title.lower())[:30])
        if not first_loc_id:
            first_loc_id = loc_id

        _write_file(scenario_path, f'world/locations/{loc_id}.md',
                    f"# {loc_data.get('name', title)}\n\n{loc_data.get('description', '')}")
        locations_index["locations"].append({
            "id": loc_id,
            "name": loc_data.get('name', title),
            "keywords": loc_data.get('keywords', [loc_id]),
        })
        loc_count += 1

    if locations_index["locations"]:
        _write_json(scenario_path, 'world/locations/_index.json', locations_index)

    if first_loc_id:
        _update_index(scenario_path, {"current_location": first_loc_id})

    _report(3, f"Locations ({loc_count} imported)", "done")

    # ─── PASS 4: Opening Scene ───
    logger.info(f"Canon import pass 4: Opening scene for {franchise}")
    _report(4, "Opening scene", "generating")

    setting_text = world_data.get('setting', franchise) if world_data else franchise
    npcs_str = ', '.join(starting_npcs[:5]) if starting_npcs else 'various characters'

    opening_prompt = f"""You are writing the opening scene for a {genre} RPG set in the world of {franchise}.

World: {setting_text[:500]}
NPCs available: {npcs_str}
Starting location: {first_loc_id or 'the main hub area'}

Generate the opening as JSON:
{{
  "current_scene": "Write a vivid 100-150 word opening scene. Where the player is, what they see/hear, the mood. Immerse the player in the {franchise} universe. End with a sense of 'what now?' but do NOT list options.",
  "active_npcs_in_scene": ["npc_id_1", "npc_id_2"],
  "available_quests": [
    {{"name": "Quest Name", "description": "2 sentences about the quest objective and reward", "type": "exploration"}}
  ]
}}

Use canon-appropriate quest hooks and NPCs from {franchise}.
Return ONLY valid JSON."""

    opening_data = _llm_json_call(ollama_host, opening_prompt, model)

    if opening_data:
        scene_npcs = opening_data.get('active_npcs_in_scene', starting_npcs[:3])
        _write_file(scenario_path, 'scenes/current.md',
                    f"## Current Scene\n\n{opening_data.get('current_scene', '')}\n\n"
                    f"## Present NPCs\n{', '.join(scene_npcs)}")

        _update_index(scenario_path, {
            "active_npcs_in_scene": scene_npcs,
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

    logger.info(f"Canon import complete: {franchise} — {npc_count} chars, {loc_count} locs")
    return True


# ═══════════════════════════════════════════════════════════
# LLM HELPER (reuses pattern from scenario_generator.py)
# ═══════════════════════════════════════════════════════════

def _llm_json_call(ollama_host, prompt, model, max_retries=2):
    """Call LLM and parse JSON response with retries."""
    from routes.helpers import strip_think_blocks

    for attempt in range(max_retries + 1):
        try:
            r = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 2048},
                    "think": False,
                },
                timeout=120,
            )
            if r.status_code != 200:
                logger.warning(f"Ollama returned {r.status_code}")
                return None

            text = r.json().get('response', '').strip()
            text = strip_think_blocks(text)

            # Strip markdown fences
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            # Find JSON boundaries
            brace_start = text.find('{')
            brace_end = text.rfind('}')
            if brace_start >= 0 and brace_end > brace_start:
                text = text[brace_start:brace_end + 1]

            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                prompt += "\n\nPrevious response had invalid JSON. Return ONLY a valid JSON object."
            else:
                return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
