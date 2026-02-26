"""
STRAT_OS - Kuwait & Regional Intelligence Fetcher
Searches for ACTUAL news/announcements about Kuwait careers, bank offers,
and Regional GCC tech/business trends.

=== APPROACH ===
Instead of scraping homepages, we dynamically build search queries from
config.yaml to find press releases, news, and social media signals.

=== COST ANALYSIS ===
FREE:  DuckDuckGo search, Google News RSS
PAID:  Serper.dev (2500 free queries), Google Custom Search (100/day free)
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

# Search client imports
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

try:
    from .google_search import GoogleSearchClient, get_google_client, DailyLimitReached
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False
    GoogleSearchClient = None
    DailyLimitReached = Exception

try:
    from .serper_search import SerperSearchClient, get_serper_client, SerperSearchError
    HAS_SERPER = True
except ImportError:
    HAS_SERPER = False
    SerperSearchClient = None
    SerperSearchError = Exception


def _detect_source(url: str, fallback: str = "Web Search") -> str:
    """Detect the source type from URL."""
    if 'twitter.com' in url or 'x.com' in url:
        return "Twitter/X"
    elif 'linkedin.com' in url:
        return "LinkedIn"
    elif 'instagram.com' in url:
        return "Instagram"
    elif any(s in url for s in ['kuwaittimes', 'arabtimes', 'gulfnews', 'arabianbusiness']):
        return "Kuwait News"
    return fallback


class KuwaitIntelligenceFetcher:
    """
    Fetches actual news and announcements about Kuwait and the GCC region.
    All queries are built dynamically from config - no hardcoded values.
    """

    def __init__(self, config: Dict[str, Any], timelimit: str = "w"):
        self.timelimit = timelimit
        self.config = config
        self.limit_reached = False

        # --- Search provider setup ---
        search_config = config.get('search', {})
        self.search_provider = search_config.get('provider', 'duckduckgo')

        self.google_client: Optional[GoogleSearchClient] = None
        self.serper_client: Optional[SerperSearchClient] = None

        if self.search_provider == 'serper' and HAS_SERPER:
            self.serper_client = get_serper_client(config)
            if self.serper_client:
                logger.info("Kuwait Intel using Serper.dev")
            else:
                logger.warning("Serper configured but no API key - falling back to DuckDuckGo")
                self.search_provider = 'duckduckgo'
        elif self.search_provider == 'google' and HAS_GOOGLE:
            self.google_client = get_google_client(config)
            if self.google_client:
                logger.info("Kuwait Intel using Google Custom Search")
            else:
                self.search_provider = 'duckduckgo'

        if self.search_provider == 'duckduckgo':
            logger.info("Kuwait Intel using DuckDuckGo (free, unlimited)")

        # --- Extract config sections ---
        news_config = config.get('news', {})

        career_cfg = news_config.get('career', {})
        self.career_keywords = career_cfg.get('keywords', [])
        self.career_queries = career_cfg.get('queries', [])

        finance_cfg = news_config.get('finance', {})
        self.bank_keywords = finance_cfg.get('keywords', [])
        self.bank_queries = finance_cfg.get('queries', [])

        regional_cfg = news_config.get('regional', {})
        self.reg_keywords = regional_cfg.get('keywords', [])
        self.reg_queries = regional_cfg.get('queries', [])

        logger.info(
            f"Kuwait Intel: {len(self.career_keywords)} career, "
            f"{len(self.bank_keywords)} bank, "
            f"{len(self.reg_keywords)} regional keywords"
        )
        
        # --- Dynamic categories from Simple mode ---
        self.dynamic_categories = config.get('dynamic_categories', [])
        if self.dynamic_categories:
            logger.info(f"Kuwait Intel: {len(self.dynamic_categories)} dynamic categories loaded")

        # --- User location (drives geographic context in queries) ---
        self.location = config.get('profile', {}).get('location', 'Kuwait').strip()
        if not self.location:
            self.location = 'Kuwait'
        logger.info(f"Kuwait Intel: location = {self.location}")

        # --- DDG region from profile location (same pattern as news.py SearchFetcher) ---
        try:
            from processors.scorer_base import location_to_lang
            _, self.ddg_region, _ = location_to_lang(self.location)
        except ImportError:
            self.ddg_region = 'wt-wt'

    # ─── Query Builders ───────────────────────────────────────────

    def _build_career_searches(self) -> List[str]:
        """Build career search queries. 2 patterns per keyword + custom queries."""
        searches = []
        loc = self.location
        for company in self.career_keywords:
            searches.append(f'"{company}" {loc} hiring 2026')
            searches.append(f'"{company}" {loc} engineer jobs')
        searches.extend(self.career_queries)
        return searches

    def _build_career_evergreen(self) -> List[str]:
        """
        Untimed queries for official K-sector career portals.
        These pages are static/long-lived but contain active recruitment.
        Run WITHOUT time filter to catch always-open hiring campaigns.
        """
        return [
            # Official K-sector career pages (engineering fresh grads go through KOC/KNPC)
            'site:kockw.com careers OR recruitment OR "fresh graduate"',
            'site:knpc.com recruitment OR careers OR "fresh graduate"',
            'site:kipic.com.kw careers OR recruitment',
            'site:kpc.com.kw careers OR recruitment OR "fresh graduate"',
            # Oil sector recruitment announcements
            'Kuwait oil sector "fresh graduate" recruitment announcement 2026',
            'Kuwait oil sector "admin circular" hiring Kuwaiti nationals',
            # Direct KNPC/KOC fresh grad campaigns (often announced in Arabic media)
            '"القطاع النفطي" توظيف خريجين الكويت',
            '"KOC" OR "KNPC" OR "KIPIC" "fresh graduate" program Kuwait',
        ]

    def _build_bank_searches(self) -> List[str]:
        """Build bank offer/deals search queries. 2 patterns per keyword + custom queries."""
        searches = []
        for bank in self.bank_keywords:
            searches.append(f'"{bank}" offer OR promotion 2026')
            searches.append(f'"{bank}" student OR allowance')
        searches.extend(self.bank_queries)
        # Bank hiring queries
        loc = self.location
        for bank in self.bank_keywords:
            searches.append(f'"{bank}" hiring OR career OR vacancy {loc} 2026')
            searches.append(f'"{bank}" IT OR technology OR digital jobs {loc}')
        return searches

    def _build_regional_searches(self) -> List[str]:
        """Build GCC/regional search queries to fill the Regional tab."""
        searches = []
        for kw in self.reg_keywords:
            searches.append(f'"{kw}" technology investment 2026')
            searches.append(f'"{kw}" engineering jobs GCC')
        searches.extend(self.reg_queries)
        return searches

    def _build_social_searches(self) -> List[str]:
        """Build Twitter/social media searches."""
        searches = []
        loc = self.location
        if self.bank_keywords:
            bank_str = '" OR "'.join(self.bank_keywords[:3])
            searches.append(f'site:twitter.com "{bank_str}" offer')
            searches.append(f'site:twitter.com "{bank_str}" promotion')
        if self.career_keywords:
            company_str = '" OR "'.join(self.career_keywords[:3])
            searches.append(f'site:twitter.com "{company_str}" hiring {loc}')
        searches.append(f'site:twitter.com "{loc}" bank "salary transfer"')
        return searches

    def _build_dynamic_searches(self, cat: Dict[str, Any]) -> List[str]:
        """Build search queries from a dynamic category.
        
        KEY DESIGN: Items can be either ENTITIES (company names, abbreviations) or
        TERMS (technical concepts, short phrases). Entities get exact-quoted for 
        precision; terms get used as bare keywords for recall.
        
        Detection heuristic:
        - All-caps or mixed-case short words (KOC, SLB, KNPC, DOW) → entity, quote it
        - Contains known suffixes (Bank, Corp, Inc, Ltd, University) → entity, quote it
        - 1 word ≤ 12 chars → likely entity or technical term, quote it
        - 2+ words that are all lowercase common words → bare keywords, don't quote
        """
        searches = []
        items = cat.get('items', [])
        label = cat.get('label', '').strip()
        scorer_type = cat.get('scorer_type', 'auto')
        root = cat.get('root', 'kuwait')
        
        # Build context from label (skip words that might already be in items)
        STOP_WORDS = {'and', 'or', 'the', 'a', 'an', 'in', 'of', 'for', '&', 'news', 'jobs', 'deals'}
        label_words = [w for w in label.lower().split() if w not in STOP_WORDS and len(w) > 1]
        
        ENTITY_SUFFIXES = ('bank', 'corp', 'inc', 'ltd', 'co', 'university', 'institute', 'company')
        
        def _is_entity(item_str):
            """Detect if an item looks like a named entity vs a descriptive phrase."""
            words = item_str.split()
            # Single word → treat as entity/term
            if len(words) == 1:
                return True
            # ALL CAPS (abbreviation): "KOC", "KNPC", "SLB"
            if item_str.isupper() and len(item_str) <= 10:
                return True
            # Contains a capitalized proper noun pattern: "Warba Bank", "DOW Chemical"
            if any(w[0].isupper() for w in words if w):
                return True
            # Known entity suffixes
            if any(item_str.lower().endswith(s) for s in ENTITY_SUFFIXES):
                return True
            # Short 2-word items (≤20 chars) that aren't common phrases
            if len(words) == 2 and len(item_str) <= 20:
                return True
            return False
        
        def _quote(item_str):
            """Smart-quote: entities get exact match, phrases get bare keywords."""
            if _is_entity(item_str):
                return f'"{item_str}"'
            return item_str
        
        def _item_overlaps_label(item_str):
            """Check if item already contains most of the label's meaning."""
            item_lower = item_str.lower()
            return sum(1 for lw in label_words if lw in item_lower) >= len(label_words) * 0.5
        
        for item in items:
            item_clean = item.strip()
            if not item_clean:
                continue
            
            q_item = _quote(item_clean)
            # Only add label context if item doesn't already contain it
            needs_context = not _item_overlaps_label(item_clean)
            label_ctx = ' '.join(label_words) if needs_context and label_words else ''
            loc = self.location
            
            if scorer_type == 'career':
                if root == 'kuwait':
                    searches.append(f'{q_item} {loc} hiring OR careers OR recruitment 2026')
                    if label_ctx:
                        searches.append(f'{q_item} {label_ctx} {loc} careers 2026')
                else:
                    searches.append(f'{q_item} {label_ctx} hiring OR jobs 2026')
            
            elif scorer_type == 'banks':
                searches.append(f'{q_item} offer OR promotion OR deal 2026')
                searches.append(f'{q_item} student OR allowance OR transfer')
            
            elif scorer_type == 'tech':
                ctx = label_ctx or 'technology'
                searches.append(f'{q_item} {ctx} 2026')
            
            elif scorer_type == 'regional':
                searches.append(f'{q_item} {label_ctx} GCC OR {loc} OR "Middle East" 2026')
            
            else:  # auto
                if root == 'kuwait':
                    searches.append(f'{q_item} {label_ctx} {loc} 2026')
                else:
                    searches.append(f'{q_item} {label_ctx} 2026')
        
        # Cap at 12 queries per category
        searches = searches[:12]
        
        # ── Cross-entity combination queries ──
        # When a category has multiple entities (companies), generate queries that 
        # pair them together to catch partnership/deal/contract news between them.
        entities = [item.strip() for item in items if item.strip() and _is_entity(item.strip())]
        if len(entities) >= 2:
            from itertools import combinations
            pairs = list(combinations(entities[:6], 2))  # Cap at top 6 entities
            # Generate up to 4 cross-entity queries
            for e1, e2 in pairs[:4]:
                q1, q2 = _quote(e1), _quote(e2)
                searches.append(f'{q1} {q2} deal OR partnership OR contract OR agreement 2026')
        
        return searches

    # ─── Search Execution ─────────────────────────────────────────

    def _search(self, query: str, max_results: int = 3,
                category: str = "general", root: str = "kuwait",
                no_time_filter: bool = False) -> List[Dict[str, Any]]:
        """Unified search dispatcher with automatic DDG fallback."""
        if self.limit_reached:
            # Even when limit reached on Serper, try DDG
            return self._search_ddg(query, max_results, category, root)
        if self.search_provider == 'serper' and self.serper_client:
            results = self._search_serper(query, max_results, category, root, no_time_filter)
            # Fallback to DDG if Serper returned nothing (no credits, rate limit, etc.)
            if not results and HAS_DDGS:
                results = self._search_ddg(query, max_results, category, root)
            return results
        elif self.search_provider == 'google' and self.google_client:
            results = self._search_google(query, max_results, category, root)
            if not results and HAS_DDGS:
                results = self._search_ddg(query, max_results, category, root)
            return results
        return self._search_ddg(query, max_results, category, root)

    def _search_serper(self, query: str, max_results: int,
                       category: str, root: str,
                       no_time_filter: bool = False) -> List[Dict[str, Any]]:
        """Search using Serper.dev API (Google results)."""
        if not self.serper_client:
            return []
        results = []
        try:
            time_period = None if no_time_filter else self.timelimit
            search_results = self.serper_client.search(
                query, num_results=max_results, time_period=time_period
            )
            for r in search_results:
                url = r.get('url', '')
                title = r.get('title', '')
                snippet = r.get('snippet', '')
                if not url or not title:
                    continue
                results.append({
                    "title": title,
                    "url": url,
                    "summary": snippet[:250] if snippet else f"Found via search: {query[:50]}",
                    "source": _detect_source(url, "Serper/Google"),
                    "root": root,
                    "category": category,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        except SerperSearchError as e:
            logger.warning(f"Serper search failed for '{query[:40]}...': {e}")
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "not enough credits" in err_msg or "402" in err_msg or "429" in err_msg:
                self.limit_reached = True
                logger.info("Serper limit reached — switching to DuckDuckGo fallback for remaining queries")
        except Exception as e:
            logger.warning(f"Serper search failed for '{query[:40]}...': {e}")
        return results

    def _search_google(self, query: str, max_results: int,
                       category: str, root: str) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API."""
        if not self.google_client:
            return []
        results = []
        date_restrict_map = {'d': 'd1', 'w': 'w1', 'm': 'm1'}
        date_restrict = date_restrict_map.get(self.timelimit, 'w1')
        try:
            search_results = self.google_client.search(
                query, num_results=max_results, date_restrict=date_restrict
            )
            for r in search_results:
                url = r.get('url', '')
                title = r.get('title', '')
                snippet = r.get('snippet', '')
                if not url or not title:
                    continue
                results.append({
                    "title": title,
                    "url": url,
                    "summary": snippet[:250] if snippet else f"Found via Google: {query[:50]}",
                    "source": _detect_source(url, "Google Search"),
                    "root": root,
                    "category": category,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        except DailyLimitReached as e:
            logger.warning(f"Google Search daily limit reached: {e}")
            self.limit_reached = True
        except Exception as e:
            logger.warning(f"Google search failed for '{query[:40]}...': {e}")
        return results

    def _search_ddg(self, query: str, max_results: int,
                    category: str, root: str) -> List[Dict[str, Any]]:
        """Search DuckDuckGo for news/articles."""
        if not HAS_DDGS:
            return []
        results = []
        try:
            with DDGS() as ddgs:
                search_results = list(ddgs.news(
                    query, region=self.ddg_region, max_results=max_results, timelimit=self.timelimit
                ))
                for r in search_results:
                    url = r.get('url', '')
                    title = r.get('title', '')
                    body = r.get('body', '')
                    if not url or not title:
                        continue
                    results.append({
                        "title": title,
                        "url": url,
                        "summary": body[:250] if body else f"Found via search: {query[:50]}",
                        "source": _detect_source(url, "Web Search"),
                        "root": root,
                        "category": category,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
        except Exception as e:
            logger.warning(f"DDG search failed for '{query[:40]}...': {e}")
        return results

    # ─── Main Pipeline ────────────────────────────────────────────

    def fetch_all(self) -> List[Dict[str, Any]]:
        """
        Main pipeline: runs all search tasks and aggregates results.
        Queries within each category run in parallel (8 workers).
        """
        all_items = []
        seen_urls = set()
        lock = threading.Lock()

        # (query_builder, category, root)
        # NOTE: Social media searches (site:twitter.com, site:instagram.com) REMOVED.
        # They return login walls and garbage. Use RSS feeds and direct searches instead.
        
        if self.dynamic_categories:
            # Dynamic mode: build queries from AI-generated categories ONLY
            # Legacy builders would duplicate since their keywords get synced from dynamic cats
            tasks = []
            for cat in self.dynamic_categories:
                # Skip disabled categories
                if cat.get('enabled') is False:
                    continue
                cat_id = cat.get('id', 'unknown')
                cat_root = cat.get('root', 'kuwait')
                def make_builder(c):
                    return lambda: self._build_dynamic_searches(c)
                tasks.append((make_builder(cat), cat_id, cat_root))
            logger.info(f"Dynamic mode: {len(tasks)} enabled category builders (legacy builders skipped)")
        else:
            # Legacy mode: hardcoded career/banks/regional builders
            tasks = [
                (self._build_career_searches,   "career",   "kuwait"),
                (self._build_bank_searches,     "banks",    "kuwait"),
                (self._build_regional_searches, "regional", "regional"),
            ]

        for builder, category, root in tasks:
            queries = builder()
            if not queries:
                continue

            provider_note = " (DDG fallback)" if self.limit_reached else ""
            logger.info(f"Searching {category} intelligence ({len(queries)} queries){provider_note}...")

            def _do_query(query, cat=category, rt=root):
                return self._search(query, max_results=3, category=cat, root=rt)

            category_items = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(_do_query, q): q for q in queries}
                for future in as_completed(futures):
                    try:
                        results = future.result()
                        for item in results:
                            with lock:
                                if item['url'] not in seen_urls:
                                    seen_urls.add(item['url'])
                                    category_items.append(item)
                    except Exception as e:
                        q = futures[future]
                        logger.warning(f"Parallel query failed for '{q[:40]}': {e}")

            all_items.extend(category_items)
            count = len(category_items)
            logger.info(f"  -> {count} {category} items collected")

        # --- Evergreen queries: official K-sector career portals (NO time filter) ---
        # Only run for Kuwait-based users whose config actually tracks oil sector entities.
        # A teacher in Kuwait has no use for KOC/KNPC recruitment pages.
        is_kuwait_user = 'kuwait' in self.location.lower()
        k_sector_entities = {'koc', 'knpc', 'kipic', 'kpc', 'kuwait oil', 'kuwait petroleum'}
        has_k_sector = False
        if is_kuwait_user:
            # Check dynamic categories and legacy career keywords for oil sector entities
            check_keywords = [kw.lower() for kw in self.career_keywords]
            for cat in self.dynamic_categories:
                check_keywords.extend(i.lower() for i in cat.get('items', []))
            has_k_sector = any(ent in ' '.join(check_keywords) for ent in k_sector_entities)
        
        if is_kuwait_user and has_k_sector:
            evergreen_queries = self._build_career_evergreen()
            provider_note = " (DDG fallback)" if self.limit_reached else ""
            logger.info(f"Searching K-sector career portals ({len(evergreen_queries)} untimed queries){provider_note}...")

            def _do_evergreen(query):
                return self._search(query, max_results=3, category="career", root="kuwait",
                                    no_time_filter=True)

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(_do_evergreen, q): q for q in evergreen_queries}
                evergreen_count = 0
                for future in as_completed(futures):
                    try:
                        results = future.result()
                        for item in results:
                            with lock:
                                if item['url'] not in seen_urls:
                                    seen_urls.add(item['url'])
                                    all_items.append(item)
                                    evergreen_count += 1
                    except Exception as e:
                        q = futures[future]
                        logger.warning(f"Evergreen query failed for '{q[:40]}': {e}")
            logger.info(f"  -> {evergreen_count} K-sector career portal items collected")
        else:
            if not is_kuwait_user:
                logger.info("Skipping K-sector evergreen queries (non-Kuwait location)")
            elif not has_k_sector:
                logger.info("Skipping K-sector evergreen queries (no oil sector entities in config)")

        if self.limit_reached:
            logger.info("=== Serper credits exhausted — used DuckDuckGo fallback ===")

        logger.info(f"=== Total Kuwait/Regional intelligence: {len(all_items)} items ===")
        return all_items

    def get_search_status(self) -> Dict[str, Any]:
        """Get current search provider status."""
        if self.serper_client:
            return {
                'provider': 'serper',
                'used': 0,
                'remaining': 'unknown',
                'limit': 2500,
                'percentage': 0,
                'warning': False,
                'limit_reached': self.limit_reached,
                'limit_reached_this_session': self.limit_reached
            }
        if self.google_client:
            status = self.google_client.get_status()
            status['provider'] = 'google'
            status['limit_reached_this_session'] = self.limit_reached
            return status
        return {
            'provider': 'duckduckgo',
            'used': 0,
            'remaining': 'unlimited',
            'limit': 'unlimited',
            'percentage': 0,
            'warning': False,
            'limit_reached': False
        }


def fetch_kuwait_intelligence(config: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main entry point called by the news fetcher.

    Returns:
        Tuple of (items, search_status)
    """
    timelimit = config.get('news', {}).get('timelimit', 'w')
    fetcher = KuwaitIntelligenceFetcher(config, timelimit=timelimit)
    items = fetcher.fetch_all()
    status = fetcher.get_search_status()
    return items, status
