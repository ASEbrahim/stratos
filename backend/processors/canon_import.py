"""Canon Import — fetch franchise data from Fandom wikis and populate scenario files.

Sprint 8A: "Say the name, get the world."
When the user says "Build the SAO world" or "Import Naruto", this module:
1. Resolves the franchise to a Fandom wiki subdomain
2. Fetches key pages (characters, locations, lore) via MediaWiki API
3. Normalizes raw wikitext into game-ready JSON via LLM
4. Writes everything into the scenario file structure

Optimized: batch wiki extracts (multi-title API), batch LLM calls (3-4 chars per call),
reduced rate-limit delay, right-sized num_predict per task.
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
FETCH_DELAY = 0.2   # seconds between requests (Fandom rate limit is generous)
MAX_FETCHES = 20     # raised — batch extracts count as 1 fetch each


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
        except Exception as e:
            logger.warning(f"Wiki existence check failed for {self.wiki}: {e}")
            return False

    def search_pages(self, query, limit=10):
        """Search for pages matching a query."""
        try:
            data = self._get({"action": "opensearch", "search": query, "limit": limit})
            if isinstance(data, list) and len(data) >= 2:
                return data[1]  # list of page titles
            return []
        except Exception as e:
            logger.warning(f"Page search failed for '{query}' on {self.wiki}: {e}")
            return []

    def get_category_members(self, category, limit=20):
        """Get pages in a category (e.g., 'Category:Characters')."""
        try:
            data = self._get({
                "action": "query", "list": "categorymembers",
                "cmtitle": category, "cmlimit": limit, "cmtype": "page"
            })
            return [m['title'] for m in data.get("query", {}).get("categorymembers", [])]
        except Exception as e:
            logger.warning(f"Category members fetch failed for '{category}' on {self.wiki}: {e}")
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
        """Fetch wikitext for a single page, parse to clean text, truncate."""
        raw = self.get_page_wikitext(title)
        if raw:
            return _parse_wikitext(raw)[:max_chars]
        return ""

    def get_batch_wikitext(self, titles, max_chars=2000):
        """Fetch raw wikitext for multiple pages in ONE API call via revisions API.

        Uses action=query + prop=revisions (always available on Fandom wikis).
        Returns: dict mapping original title -> cleaned text (parsed wikitext).
        """
        if not titles:
            return {}
        results = {}
        # Process in chunks of 10 (revisions can be large)
        for chunk_start in range(0, len(titles), 10):
            chunk = titles[chunk_start:chunk_start + 10]
            try:
                data = self._get({
                    "action": "query",
                    "titles": "|".join(chunk),
                    "prop": "revisions",
                    "rvprop": "content",
                    "rvslots": "main",
                    "redirects": True,
                })
                # Build redirect map for title resolution
                redirects_map = {}
                for redir in data.get("query", {}).get("redirects", []):
                    redirects_map[redir.get("from", "")] = redir.get("to", "")
                norm_map = {}
                for norm in data.get("query", {}).get("normalized", []):
                    norm_map[norm.get("from", "")] = norm.get("to", "")

                # Extract wikitext from each page's revision
                page_title_to_text = {}
                for page in data.get("query", {}).get("pages", {}).values():
                    title_key = page.get("title", "")
                    revs = page.get("revisions", [])
                    if revs:
                        raw = revs[0].get("slots", {}).get("main", {}).get("*", "")
                        if raw:
                            clean = _parse_wikitext(raw)[:max_chars]
                            if len(clean) >= 50:
                                page_title_to_text[title_key] = clean

                # Map resolved titles back to original requested titles
                for orig_title in chunk:
                    resolved = norm_map.get(orig_title, orig_title)
                    resolved = redirects_map.get(resolved, resolved)
                    if resolved in page_title_to_text:
                        results[orig_title] = page_title_to_text[resolved]
                    elif orig_title in page_title_to_text:
                        results[orig_title] = page_title_to_text[orig_title]
            except Exception as e:
                logger.warning(f"Batch wikitext failed for chunk: {e}")
        return results


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
    if not slug:
        logger.warning(f"Franchise resolution: empty slug for '{name}'")
        return None
    fetcher = FandomFetcher(slug)
    try:
        fetcher.fetch_count = 0  # don't count validation against import budget
        if fetcher.wiki_exists():
            return {"wiki": slug, "full_name": name.title(), "genre": "RPG"}
    except Exception as e:
        logger.warning(f"Franchise slug resolution failed for '{slug}': {e}")

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

NORMALIZE_BATCH_CHARACTERS = """You are converting wiki articles about characters from {franchise} into game-ready data.

{characters_block}

Convert EACH character to a JSON object with this structure, and return a JSON array of all characters:
[
  {{
    "id": "filesystem_safe_name_lowercase",
    "name": "Display Name",
    "short": "One-line role description (max 10 words)",
    "location": "where they're typically found",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "can_fight": true,
    "can_trade": false,
    "profile": "80-100 words: appearance, personality, role in the story.",
    "speaking_style": "How they talk: verbal tics, sentence patterns, tone. 2-3 sentences.",
    "dialogue_samples": "3 example lines this character would say, canon-accurate.",
    "knowledge": "What this character knows about the world. 2-4 sentences.",
    "stats": "Power level, combat abilities, special techniques. 2-3 sentences."
  }}
]

CRITICAL: Return ONLY a valid JSON array. No markdown fences. No text before or after."""

NORMALIZE_BATCH_LOCATIONS = """You are converting wiki articles about locations from {franchise} into game-ready data.

{locations_block}

Convert EACH location to a JSON object, return a JSON array:
[
  {{
    "id": "filesystem_safe_id_lowercase",
    "name": "Location Name",
    "description": "100-150 words: what the place looks like, atmosphere, key landmarks, who can be found here.",
    "keywords": ["keyword1", "keyword2"]
  }}
]

CRITICAL: Return ONLY a valid JSON array. No markdown fences."""

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
# IMPORT PIPELINE (optimized: batch wiki + batch LLM)
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

    # Validate scenario_path exists and is a directory
    if not os.path.isdir(scenario_path):
        logger.error(f"Canon import: scenario_path is not a directory: {scenario_path}")
        return False

    def _report(pass_num, pass_name, status):
        if progress_callback:
            try:
                progress_callback(pass_num, pass_name, status)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")

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
        except Exception as e:
            logger.debug(f"Failed to cache source for '{page_title}': {e}")

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
        character_titles = fetcher.search_pages("characters", limit=12)
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

    # ─── PASS 0.5: Bulk-fetch all extracts in batched API calls ───
    _report(0, "Fetching wiki pages...", "generating")

    char_limit = min(12, len(character_titles))
    loc_limit = min(5, len(location_titles))
    chars_to_fetch = character_titles[:char_limit]
    locs_to_fetch = location_titles[:loc_limit]

    # Batch-fetch character extracts (1 API call per 20 titles instead of 12 separate calls)
    char_extracts = fetcher.get_batch_wikitext(chars_to_fetch, max_chars=2000)
    logger.info(f"Batch-fetched {len(char_extracts)}/{len(chars_to_fetch)} character extracts")

    # Batch-fetch location extracts
    loc_extracts = fetcher.get_batch_wikitext(locs_to_fetch, max_chars=1500)
    logger.info(f"Batch-fetched {len(loc_extracts)}/{len(locs_to_fetch)} location extracts")

    # Cache all fetched content
    for title, text in char_extracts.items():
        _cache_source(title, text)
    for title, text in loc_extracts.items():
        _cache_source(title, text)

    _report(0, f"Fetched {len(char_extracts)} character + {len(loc_extracts)} location pages", "done")

    # ─── PASS 1: World Lore ───
    logger.info(f"Canon import pass 1: World lore for {franchise}")
    _report(1, "World lore & power system", "generating")

    world_source_parts = []
    if main_text:
        world_source_parts.append(f"Main article:\n{main_text}")

    # Fetch power system page if found
    for pt in power_titles[:2]:
        if fetcher.fetch_count >= MAX_FETCHES - 2:
            break
        raw = fetcher.get_page_wikitext(pt)
        if raw:
            clean = _parse_wikitext(raw)[:1500]
            if len(clean) > 100:
                world_source_parts.append(f"Power/abilities page ({pt}):\n{clean}")
                _cache_source(pt, clean)

    world_text = "\n\n".join(world_source_parts) if world_source_parts else f"The world of {franchise}"

    world_data = _llm_json_call(ollama_host, NORMALIZE_WORLD.format(
        franchise=franchise, wiki_text=world_text[:4000]
    ), model, num_predict=2048)

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

        power_system = world_data.get('power_system', '')
        if power_system:
            _write_file(scenario_path, 'world/mechanics/power_system.md',
                        f"# Power System\n\n{power_system}")

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

    # ─── PASS 2: Characters (BATCHED — 3-4 per LLM call) ───
    logger.info(f"Canon import pass 2: Characters for {franchise}")
    _report(2, f"Characters (0/{char_limit})", "generating")

    from processors.scenario_templates import create_npc_folder

    starting_npcs = []
    npc_count = 0

    # Build list of characters that have usable text
    char_texts = []
    for title in chars_to_fetch:
        text = char_extracts.get(title, '')
        if text and len(text) >= 50:
            char_texts.append((title, text[:1500]))  # trim for batch prompt

    # Process in batches of 4
    BATCH_SIZE = 4
    for batch_start in range(0, len(char_texts), BATCH_SIZE):
        batch = char_texts[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(char_texts) + BATCH_SIZE - 1) // BATCH_SIZE

        _report(2, f"Characters batch {batch_num}/{total_batches} ({npc_count} done)", "generating")

        # Build combined prompt with all characters in this batch
        chars_block_parts = []
        for i, (title, text) in enumerate(batch):
            chars_block_parts.append(f"--- CHARACTER {i+1}: {title} ---\n{text}")
        characters_block = "\n\n".join(chars_block_parts)

        # Single LLM call for the whole batch
        batch_result = _llm_json_call(
            ollama_host,
            NORMALIZE_BATCH_CHARACTERS.format(
                franchise=franchise,
                characters_block=characters_block
            ),
            model,
            num_predict=4096,  # more tokens for batch output
            expect_array=True,
        )

        if not batch_result:
            logger.warning(f"Batch {batch_num} LLM call failed, falling back to individual")
            # Fallback: try each character individually
            batch_result = []
            for title, text in batch:
                single_prompt = f"""Convert this {franchise} character wiki article to game-ready JSON.

Raw wiki content for {title}:
{text[:1500]}

Return a single JSON object:
{{"id": "lowercase_id", "name": "Name", "short": "role (max 10 words)", "location": "where found", "keywords": ["k1","k2"], "can_fight": true, "can_trade": false, "profile": "80-100 words", "speaking_style": "2-3 sentences", "dialogue_samples": "3 lines", "knowledge": "2-4 sentences", "stats": "2-3 sentences"}}

Return ONLY valid JSON."""
                char_data = _llm_json_call(ollama_host, single_prompt, model, num_predict=1536)
                if char_data:
                    batch_result.append(char_data)
            if not batch_result:
                continue

        # Handle case where LLM returns a single object instead of array
        if isinstance(batch_result, dict):
            batch_result = [batch_result]

        for char_data in batch_result:
            if not isinstance(char_data, dict):
                continue
            title_guess = char_data.get('name', f'char_{npc_count}')
            npc_id = char_data.get('id', re.sub(r'[^a-z0-9_]', '_', title_guess.lower())[:30])

            profile_content = f"# {char_data.get('name', title_guess)}\n\n{char_data.get('profile', '')}\n\n## Speaking Style\n{char_data.get('speaking_style', '')}"
            dialogue_content = f"# {char_data.get('name', title_guess)} — Dialogue\n\n{char_data.get('dialogue_samples', '')}"
            knowledge_content = f"# {char_data.get('name', title_guess)} — Knowledge\n\n{char_data.get('knowledge', '')}"
            stats_content = char_data.get('stats', '')

            create_npc_folder(scenario_path, npc_id, {
                'name': char_data.get('name', title_guess),
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

    # ─── PASS 3: Locations (BATCHED — all in 1 LLM call) ───
    logger.info(f"Canon import pass 3: Locations for {franchise}")
    _report(3, f"Locations (0/{loc_limit})", "generating")

    from processors.scenario_generator import _write_file, _write_json, _update_index

    locations_index = {"locations": []}
    loc_count = 0
    first_loc_id = ""

    # Build location texts
    loc_texts = []
    for title in locs_to_fetch:
        text = loc_extracts.get(title, '')
        if text and len(text) >= 30:
            loc_texts.append((title, text[:1200]))

    if loc_texts:
        # Build combined prompt for ALL locations in one call
        locs_block_parts = []
        for i, (title, text) in enumerate(loc_texts):
            locs_block_parts.append(f"--- LOCATION {i+1}: {title} ---\n{text}")
        locations_block = "\n\n".join(locs_block_parts)

        _report(3, f"Locations ({len(loc_texts)} in batch)", "generating")

        batch_locs = _llm_json_call(
            ollama_host,
            NORMALIZE_BATCH_LOCATIONS.format(
                franchise=franchise,
                locations_block=locations_block
            ),
            model,
            num_predict=2048,
            expect_array=True,
        )

        if isinstance(batch_locs, dict):
            batch_locs = [batch_locs]

        if not batch_locs:
            # Fallback: process locations individually
            logger.warning("Batch location LLM failed, falling back to individual calls")
            batch_locs = []
            for title, text in loc_texts:
                single_prompt = f"""Convert this {franchise} location wiki article to game-ready JSON.

Location: {title}
{text[:1200]}

Return a single JSON object:
{{"id": "lowercase_id", "name": "Name", "description": "100-150 words vivid prose", "keywords": ["k1","k2"]}}

Return ONLY valid JSON."""
                loc_data = _llm_json_call(ollama_host, single_prompt, model, num_predict=1024)
                if loc_data:
                    batch_locs.append(loc_data)

        if batch_locs:
            for loc_data in batch_locs:
                if not isinstance(loc_data, dict):
                    continue
                loc_id = loc_data.get('id', re.sub(r'[^a-z0-9_]', '_', loc_data.get('name', 'loc').lower())[:30])
                if not first_loc_id:
                    first_loc_id = loc_id

                _write_file(scenario_path, f'world/locations/{loc_id}.md',
                            f"# {loc_data.get('name', loc_id)}\n\n{loc_data.get('description', '')}")
                locations_index["locations"].append({
                    "id": loc_id,
                    "name": loc_data.get('name', loc_id),
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

    opening_data = _llm_json_call(ollama_host, opening_prompt, model, num_predict=1024)

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

def _haiku_json_call(prompt, max_tokens=2048, expect_array=False):
    """Try Haiku API for JSON generation. Returns parsed dict/list or None."""
    from routes.helpers import strip_think_blocks
    try:
        from routes.rp_meta import call_haiku
        raw = call_haiku(prompt, max_tokens=max_tokens,
                         system="You are a game world designer converting wiki data to structured JSON. Return ONLY valid JSON. No markdown, no explanation.")
        if not raw:
            return None
        raw = strip_think_blocks(raw)
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        if expect_array:
            arr_start = raw.find('[')
            arr_end = raw.rfind(']')
            if arr_start >= 0 and arr_end > arr_start:
                return json.loads(raw[arr_start:arr_end + 1])
        brace_start = raw.find('{')
        brace_end = raw.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            result = json.loads(raw[brace_start:brace_end + 1])
            return [result] if expect_array else result
        return json.loads(raw)
    except Exception as e:
        logger.debug(f"Haiku canon import failed (will fall back to Ollama): {e}")
        return None


def _llm_json_call(ollama_host, prompt, model, max_retries=2, num_predict=2048, expect_array=False):
    """Call Haiku first for better quality, fall back to Ollama."""
    from routes.helpers import strip_think_blocks

    # Try Haiku first
    haiku_result = _haiku_json_call(prompt, max_tokens=num_predict, expect_array=expect_array)
    if haiku_result is not None:
        logger.info("Canon import: used Haiku API")
        return haiku_result
    logger.info("Canon import: Haiku unavailable, falling back to Ollama")

    for attempt in range(max_retries + 1):
        try:
            r = requests.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": num_predict},
                    "think": False,
                },
                timeout=240,
            )
            if r.status_code != 200:
                logger.warning(f"Ollama returned {r.status_code}")
                return None

            text = r.json().get('response', '').strip()
            text = strip_think_blocks(text)

            # Strip markdown fences
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            if expect_array:
                # Find array boundaries
                arr_start = text.find('[')
                arr_end = text.rfind(']')
                if arr_start >= 0 and arr_end > arr_start:
                    text = text[arr_start:arr_end + 1]
                    return json.loads(text)
                # Maybe LLM returned a single object — try that
                brace_start = text.find('{')
                brace_end = text.rfind('}')
                if brace_start >= 0 and brace_end > brace_start:
                    text = text[brace_start:brace_end + 1]
                    obj = json.loads(text)
                    return [obj]  # wrap in array
            else:
                # Find JSON object boundaries
                brace_start = text.find('{')
                brace_end = text.rfind('}')
                if brace_start >= 0 and brace_end > brace_start:
                    text = text[brace_start:brace_end + 1]

            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                if expect_array:
                    prompt += "\n\nPrevious response had invalid JSON. Return ONLY a valid JSON array of objects."
                else:
                    prompt += "\n\nPrevious response had invalid JSON. Return ONLY a valid JSON object."
            else:
                return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
