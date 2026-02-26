"""
STRAT_OS - News Intelligence Fetcher
Fetches news from multiple sources: DuckDuckGo search and RSS feeds.
"""

import feedparser
import logging
import hashlib
import re
import time
import json
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from urllib.parse import quote_plus, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

logger = logging.getLogger(__name__)

# Try to import DuckDuckGo search
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False
    logger.warning("duckduckgo_search not installed. Run: pip install duckduckgo_search")

# Try to import Serper Search client
try:
    from .serper_search import SerperSearchClient, get_serper_client
    HAS_SERPER = True
except ImportError:
    HAS_SERPER = False
    SerperSearchClient = None


@dataclass
class NewsItem:
    """Represents a news article."""
    title: str
    url: str
    summary: str = ""
    content: str = ""  # Full article text from deep scraping
    timestamp: str = ""
    source: str = "Unknown"
    root: str = "global"  # global, regional, kuwait, ai
    category: str = "general"
    score: float = 0.0
    score_reason: str = ""
    
    def __post_init__(self):
        """Generate ID after initialization."""
        self._id = None
    
    @property
    def id(self) -> str:
        """Generate unique ID from URL."""
        if self._id is None:
            self._id = hashlib.md5(self.url.encode()).hexdigest()[:12]
        return self._id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching FRS schema."""
        d = {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "summary": self.summary,
            "content": self.content,
            "timestamp": self.timestamp,
            "source": self.source,
            "root": self.root,
            "category": self.category,
            "score": self.score,
            "score_reason": self.score_reason
        }
        # Propagate pre-filter results so scorer can skip re-scoring noise
        if hasattr(self, 'pre_filter_score'):
            d['pre_filter_score'] = self.pre_filter_score
            d['pre_filter_reason'] = self.pre_filter_reason
        return d


class ArticleScraper:
    """
    Scrapes full article text from URLs using ThreadPoolExecutor.
    Implements rate limiting per domain and graceful error handling.
    """

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    # Tags to extract text from (in priority order)
    CONTENT_SELECTORS = ['article', 'main', '.post-content', '.article-body',
                         '.entry-content', '.content', '.story-body']

    # Tags to remove (navigation, ads, etc.)
    REMOVE_SELECTORS = ['nav', 'header', 'footer', 'aside', 'script', 'style',
                        '.advertisement', '.ad', '.sidebar', '.comments', '.related']

    def __init__(self, max_workers: int = 5, timeout: int = 10, max_content_length: int = 1000):
        """
        Initialize scraper.

        Args:
            max_workers: Max parallel scraping threads
            timeout: Request timeout in seconds
            max_content_length: Max chars to extract per article
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_content_length = max_content_length

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract main article text from parsed HTML."""
        # Remove unwanted tags first
        for selector in self.REMOVE_SELECTORS:
            for tag in soup.select(selector):
                tag.decompose()

        # Try to find article content using various selectors
        content = None
        for selector in self.CONTENT_SELECTORS:
            if selector.startswith('.'):
                content = soup.select_one(selector)
            else:
                content = soup.find(selector)
            if content:
                break

        if not content:
            content = soup.body if soup.body else soup

        # Extract and clean text
        text = content.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text[:self.max_content_length]  # Truncate

        return text.strip()

    def scrape_article(self, url: str) -> Optional[str]:
        """
        Scrape a single article URL.

        Args:
            url: Article URL to scrape

        Returns:
            Extracted article text or None on failure
        """
        if not url or not url.startswith('http'):
            return None

        try:
            headers = {
                'User-Agent': random.choice(self.USER_AGENTS),
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'en-US,en;q=0.9',
            }

            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            # Skip non-HTML responses
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                return None

            soup = BeautifulSoup(response.content, 'html.parser')
            return self._extract_text(soup)

        except requests.exceptions.Timeout:
            logger.debug(f"Timeout scraping {url}")
        except requests.exceptions.RequestException as e:
            logger.debug(f"Request error scraping {url}: {e}")
        except Exception as e:
            logger.debug(f"Error scraping {url}: {e}")

        return None

    def scrape_items(self, items: List['NewsItem']) -> List['NewsItem']:
        """
        Scrape content for multiple news items in parallel.
        Falls back to summary if scraping fails.
        Uses daemon threads + hard deadline to never block the pipeline.
        """
        logger.info(f"Deep scraping article content for {len(items)} items...")

        def scrape_single(item: 'NewsItem') -> 'NewsItem':
            content = self.scrape_article(item.url)
            if content and len(content) > 50:
                item.content = content
            else:
                item.content = item.summary
            return item

        # Hard deadline: 90 seconds max for ALL scraping.
        # We use shutdown(wait=False, cancel_futures=True) so the executor
        # doesn't block on stuck threads when the deadline hits.
        global_timeout = 90
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        futures = {executor.submit(scrape_single, item): item for item in items}

        completed = 0
        try:
            for future in as_completed(futures, timeout=global_timeout):
                try:
                    future.result(timeout=1)
                except Exception:
                    pass
                completed += 1
                if completed % 25 == 0:
                    logger.info(f"Scraped {completed}/{len(items)} articles...")
        except TimeoutError:
            logger.warning(f"Scraping global timeout ({global_timeout}s) — {completed}/{len(items)} completed")

        # Kill the executor immediately — don't wait for stuck threads
        executor.shutdown(wait=False, cancel_futures=True)

        # Fallback: any item without content gets its summary
        for item in items:
            if not item.content or len(item.content.strip()) < 50:
                item.content = item.summary

        scraped_count = sum(1 for item in items if item.content and item.content != item.summary)
        logger.info(f"Successfully scraped {scraped_count}/{len(items)} articles with new content")

        return items


# Keywords for smart content-based categorization
# For Kuwait: Must contain "Kuwait" explicitly OR be about specific Kuwait-only entities
# NOTE: Bank names removed - they now come from user config (Settings tab)
KUWAIT_EXPLICIT_KEYWORDS = [
    'kuwait', 'kuwaiti', 'koc ', 'knpc', 'kipic', 'kpc ',
    'kuwait petroleum', 'kuwait oil company', 'kuwait national petroleum',
    # Telecom (always Kuwait-specific)
    'zain kuwait', 'ooredoo kuwait', 'stc kuwait',
    # Education (always Kuwait-specific)
    'paaet', 'kuwait university', 'gust kuwait', 'aum kuwait', 'auk kuwait',
    # Locations (always Kuwait-specific)
    'mina abdullah', 'ahmadi kuwait', 'shuaiba',
    # Companies (always Kuwait-specific)
    'alghanim kuwait', 'alshaya kuwait', 'agility kuwait'
]

# These are global companies that need "Kuwait" context to be tagged as Kuwait
# NOTE: This list is intentionally empty - career keywords from config will be used
# for searches, but categorization will rely on explicit "Kuwait" mentions
KUWAIT_CONTEXT_COMPANIES = []

REGIONAL_KEYWORDS = [
    'gcc', 'gulf cooperation', 'saudi arabia', 'aramco', 'sabic', 
    'uae ', 'emirates', 'dubai', 'abu dhabi', 'adnoc', 
    'qatar', 'qatargas', 'bahrain', 'oman', 'middle east', 'mena region'
]

AI_TECH_KEYWORDS = [
    'artificial intelligence', ' ai ', 'machine learning', 'deep learning', 'neural network',
    'chatgpt', 'openai', 'anthropic', 'google ai', 'gemini ai', 'llm ', 'large language model',
    'quantum computing', 'superconductor', 'room temperature superconductor',
    'semiconductor', 'nvidia', ' amd ',
    '6g ', '5g network', 'autonomous vehicle', 'robotics', 'automation'
]


def categorize_content(title: str, summary: str, default_root: str = "global") -> str:
    """
    Smart categorization based on actual content.
    More strict for Kuwait - requires explicit Kuwait mention or Kuwait-only entities.
    
    When default_root is explicitly 'regional', only override to 'kuwait' if the
    content is PRIMARILY about Kuwait (not just a passing mention).
    
    Args:
        title: Article title
        summary: Article summary/body
        default_root: Default if no match found (also signals caller intent)
        
    Returns:
        Root category: 'kuwait', 'regional', 'ai', or 'global'
    """
    text = f"{title} {summary}".lower()
    
    # If caller explicitly tagged as regional, only override to kuwait
    # if Kuwait is the PRIMARY focus (in title), not just mentioned in body
    if default_root == "regional":
        title_lower = title.lower()
        # Only reclassify to Kuwait if title is clearly Kuwait-focused
        for keyword in KUWAIT_EXPLICIT_KEYWORDS:
            if keyword.lower() in title_lower:
                return "kuwait"
        # Otherwise trust the regional classification
        return "regional"
    
    # Check for explicit Kuwait keywords (high confidence)
    for keyword in KUWAIT_EXPLICIT_KEYWORDS:
        if keyword.lower() in text:
            return "kuwait"
    
    # Check for context-dependent companies (need "kuwait" nearby)
    has_kuwait_word = 'kuwait' in text
    if has_kuwait_word:
        for company in KUWAIT_CONTEXT_COMPANIES:
            if company.lower() in text:
                return "kuwait"
    
    # Check for regional GCC content (but not if it's Kuwait)
    for keyword in REGIONAL_KEYWORDS:
        if keyword.lower() in text:
            return "regional"
    
    # Check for AI/Tech content
    for keyword in AI_TECH_KEYWORDS:
        if keyword.lower() in text:
            return "ai"
    
    return default_root


class RSSFetcher:
    """Fetches news from RSS feeds."""
    
    # Common user agents to rotate
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    ]
    
    def __init__(self, feeds: List[Dict[str, str]]):
        """
        Initialize with feed configuration.
        
        Args:
            feeds: List of dicts with 'url', 'name', 'root', 'category' keys
        """
        self.feeds = feeds
    
    def fetch_all(self, max_per_feed: int = 15) -> List[NewsItem]:
        """
        Fetch from all configured RSS feeds in parallel with global timeout.
        Uses non-blocking shutdown so stuck feeds can't block the pipeline.
        """
        items = []
        feed_timeout = 60  # Max 60 seconds for ALL RSS feeds combined

        def _fetch_one(feed_config):
            try:
                return self._fetch_feed(feed_config, max_per_feed)
            except Exception as e:
                logger.error(f"Failed to fetch RSS {feed_config.get('name', 'unknown')}: {e}")
                return []

        completed_feeds = 0
        executor = ThreadPoolExecutor(max_workers=6)
        futures = {executor.submit(_fetch_one, fc): fc for fc in self.feeds}
        try:
            for future in as_completed(futures, timeout=feed_timeout):
                try:
                    feed_items = future.result(timeout=1)
                    items.extend(feed_items)
                except Exception:
                    pass
                completed_feeds += 1
        except TimeoutError:
            failed = [f.get('name', '?') for fut, f in futures.items() if not fut.done()]
            logger.warning(f"RSS timeout ({feed_timeout}s) — {completed_feeds}/{len(self.feeds)} feeds completed. "
                         f"Hanging: {', '.join(failed)}")

        # Don't wait for stuck feeds
        executor.shutdown(wait=False, cancel_futures=True)

        logger.info(f"Fetched {len(items)} items from {completed_feeds}/{len(self.feeds)} RSS feeds")
        return items
    
    def _fetch_feed(self, feed_config: Dict[str, str], max_items: int) -> List[NewsItem]:
        """Fetch from a single RSS feed."""
        url = feed_config.get("url", "")
        name = feed_config.get("name", "RSS Feed")
        root = feed_config.get("root", "global")
        category = feed_config.get("category", "general")
        
        # Fetch with requests (has timeout), then parse the response
        ua = random.choice(self.USER_AGENTS)
        try:
            response = requests.get(url, headers={'User-Agent': ua}, timeout=15)
            feed = feedparser.parse(response.content)
        except requests.RequestException as e:
            # Don't fall back to feedparser.parse(url) — it has NO timeout and can hang forever
            logger.warning(f"Feed fetch failed for {name}: {e}")
            return []
        
        if feed.bozo and not feed.entries:
            logger.warning(f"Feed error for {name}: {feed.bozo_exception}")
            return []
        
        items = []
        for entry in feed.entries[:max_items]:
            # Extract timestamp
            timestamp = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    timestamp = datetime(*entry.published_parsed[:6]).isoformat()
                except:
                    pass
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                try:
                    timestamp = datetime(*entry.updated_parsed[:6]).isoformat()
                except:
                    pass
            
            # Extract summary
            summary = ""
            if hasattr(entry, "summary"):
                # Strip HTML tags
                summary = re.sub(r'<[^>]+>', '', entry.summary)[:500]
            elif hasattr(entry, "description"):
                summary = re.sub(r'<[^>]+>', '', entry.description)[:500]
            
            item = NewsItem(
                title=entry.get("title", "No title"),
                url=entry.get("link", ""),
                summary=summary,
                timestamp=timestamp or datetime.now().isoformat(),
                source=name,
                root=root,
                category=category
            )
            items.append(item)
        
        return items


class SearchFetcher:
    """Fetches news using Serper (preferred) or DuckDuckGo search."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize search fetcher with config to determine provider."""
        self.config = config or {}
        self.ddgs = DDGS() if HAS_DDGS else None
        self.serper_client = None
        self.search_provider = 'duckduckgo'

        # Derive DDG region from profile location
        profile_location = self.config.get('profile', {}).get('location', '')
        try:
            from processors.scorer_base import location_to_lang
            _, self.ddg_region, lang_label = location_to_lang(profile_location)
        except ImportError:
            self.ddg_region = 'wt-wt'
            lang_label = 'English'

        # Check if Serper is configured
        search_config = self.config.get('search', {})
        provider = search_config.get('provider', 'duckduckgo')

        if provider == 'serper' and HAS_SERPER:
            self.serper_client = get_serper_client(self.config)
            if self.serper_client:
                self.search_provider = 'serper'
                logger.info("SearchFetcher using Serper.dev")

        if self.search_provider == 'duckduckgo' and HAS_DDGS:
            logger.info(f"SearchFetcher using DuckDuckGo (region={self.ddg_region}, lang={lang_label})")

    def search(self, query: str, root: str = "global", category: str = "general",
               max_results: int = 10, timelimit: str = "w") -> List[NewsItem]:
        """
        Search for news using configured provider (Serper or DuckDuckGo).

        Args:
            query: Search query
            root: Default root category (will be overridden by smart categorization)
            category: News category
            max_results: Maximum results to return
            timelimit: Time range - "d" (daily), "w" (weekly), "m" (monthly)

        Returns:
            List of NewsItem objects
        """
        if self.search_provider == 'serper' and self.serper_client:
            results = self._search_serper(query, root, category, max_results, timelimit)
            # Fallback to DuckDuckGo if Serper returned nothing (e.g. no credits)
            if not results and HAS_DDGS and self.ddgs:
                logger.info(f"Serper returned 0 results for '{query}', falling back to DuckDuckGo")
                results = self._search_ddg(query, root, category, max_results, timelimit)
            return results
        else:
            return self._search_ddg(query, root, category, max_results, timelimit)

    def _search_serper(self, query: str, root: str, category: str,
                       max_results: int, timelimit: str) -> List[NewsItem]:
        """Search using Serper.dev API."""
        items = []
        try:
            # Career & bank queries use web search (job pages, company sites, bank offer
            # pages aren't indexed in Google News). Tech & regional use news search for
            # better time filtering on actual articles.
            use_news = category in ("tech", "regional")
            search_type = "news" if use_news else "search"

            results = self.serper_client.search(
                query,
                num_results=max_results,
                search_type=search_type,
                time_period=timelimit
            )

            for r in results:
                title = r.get("title", "")
                summary = r.get("snippet", "")[:500]

                # Smart categorization based on actual content
                actual_root = categorize_content(title, summary, root)

                item = NewsItem(
                    title=title,
                    url=r.get("url", ""),
                    summary=summary,
                    timestamp=r.get("date", datetime.now().isoformat()),
                    source=r.get("source", "Serper/Google"),
                    root=actual_root,
                    category=category
                )
                items.append(item)

            logger.debug(f"Serper search '{query}': {len(items)} results")

        except Exception as e:
            logger.error(f"Serper search failed for '{query}': {e}")

        return items

    def _search_ddg(self, query: str, root: str, category: str,
                    max_results: int, timelimit: str) -> List[NewsItem]:
        """Search using DuckDuckGo."""
        if not HAS_DDGS or not self.ddgs:
            logger.warning("DuckDuckGo search not available")
            return []

        items = []
        try:
            # Use news search for fresher results — region derived from profile location
            results = self.ddgs.news(
                query,
                region=self.ddg_region,
                max_results=max_results,
                timelimit=timelimit
            )

            for r in results:
                title = r.get("title", "")
                summary = r.get("body", "")[:500]

                # Smart categorization based on actual content
                actual_root = categorize_content(title, summary, root)

                item = NewsItem(
                    title=title,
                    url=r.get("url", ""),
                    summary=summary,
                    timestamp=r.get("date", datetime.now().isoformat()),
                    source=r.get("source", "Web Search"),
                    root=actual_root,
                    category=category
                )
                items.append(item)

            logger.debug(f"DDG search '{query}': {len(items)} results")

        except Exception as e:
            logger.error(f"DuckDuckGo search failed for '{query}': {e}")

        return items
    
    def search_multiple(self, queries: List[Dict[str, Any]], delay: float = 1.0, timelimit: str = "w") -> List[NewsItem]:
        """
        Execute multiple searches in parallel with ThreadPoolExecutor.
        
        Args:
            queries: List of dicts with 'query', 'root', 'category' keys
            delay: (ignored — kept for API compat, parallelism replaces rate limiting)
            timelimit: Time range - "d" (daily), "w" (weekly), "m" (monthly)
            
        Returns:
            Combined list of NewsItem objects
        """
        all_items = []
        seen_urls = set()
        
        def _do_search(q):
            return self.search(
                query=q.get("query", ""),
                root=q.get("root", "global"),
                category=q.get("category", "general"),
                max_results=q.get("max_results", 10),
                timelimit=timelimit
            )
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_do_search, q): q for q in queries}
            for future in as_completed(futures):
                try:
                    items = future.result()
                    for item in items:
                        if item.url not in seen_urls:
                            seen_urls.add(item.url)
                            all_items.append(item)
                except Exception as e:
                    q = futures[future]
                    logger.warning(f"Parallel search failed for '{q.get('query', '')[:40]}': {e}")
        
        return all_items


class NewsFetcher:
    """Main news fetcher that combines all sources."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with full configuration.

        Args:
            config: Full config.yaml (needs news section and profile for Kuwait intel)
        """
        self.config = config
        # Support both full config and news-only config for backwards compatibility
        news_config = config.get("news", config) if "news" in config else config
        self.rss_fetcher = RSSFetcher(news_config.get("rss_feeds", []))
        # Pass full config to SearchFetcher so it can use Serper if configured
        self.search_fetcher = SearchFetcher(config) if (HAS_DDGS or HAS_SERPER) else None
        self.timelimit = news_config.get("timelimit", "w")  # d=daily, w=weekly, m=monthly
        self._cache: List[NewsItem] = []
        self._cache_time: Optional[datetime] = None
    
    def fetch_all(self, cache_ttl_seconds: int = 900) -> List[NewsItem]:
        """
        Fetch news from all configured sources.
        
        Args:
            cache_ttl_seconds: Cache time-to-live
            
        Returns:
            List of NewsItem objects
        """
        # Check cache
        if self._cache and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < cache_ttl_seconds:
                logger.debug(f"Using cached news data (age: {age:.0f}s)")
                return self._cache
        
        all_items = []
        seen_urls = set()
        
        # 1. Fetch from RSS feeds (fast, reliable)
        logger.info("Fetching from RSS feeds...")
        rss_items = self.rss_fetcher.fetch_all()
        for item in rss_items:
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                all_items.append(item)
        
        # 2. Search for specific topics (slower, rate-limited)
        if self.search_fetcher:
            logger.info("Searching for specific topics...")
            
            # Build search queries from config
            # NOTE: Career, finance, and regional custom queries are handled by
            # kuwait_scrapers.py (which also sends keyword-based queries).
            # Only tech_trends queries are sent here to avoid duplicate Serper calls.
            search_queries = []

            # Get news config (support both full config and news-only config)
            news_cfg = self.config.get("news", self.config) if "news" in self.config else self.config

            # Tech trends queries (ONLY these — career/finance/regional handled by kuwait_scrapers)
            tech_config = news_cfg.get("tech_trends", {})
            for query in tech_config.get("queries", []):
                search_queries.append({
                    "query": query,
                    "root": tech_config.get("root", "global"),
                    "category": "tech",
                    "max_results": 10
                })
            
            # Execute searches
            search_items = self.search_fetcher.search_multiple(search_queries, delay=1.5, timelimit=self.timelimit)
            for item in search_items:
                if item.url not in seen_urls:
                    seen_urls.add(item.url)
                    all_items.append(item)
        
        # 3. Fetch Kuwait-specific intelligence (careers + bank offers)
        logger.info("Fetching Kuwait-specific intelligence...")
        search_status = None
        try:
            from .kuwait_scrapers import fetch_kuwait_intelligence
            kuwait_items, search_status = fetch_kuwait_intelligence(self.config)  # Returns (items, status)

            for item_dict in kuwait_items:
                if item_dict['url'] not in seen_urls:
                    seen_urls.add(item_dict['url'])
                    # Convert dict to NewsItem
                    news_item = NewsItem(
                        title=item_dict.get('title', ''),
                        url=item_dict.get('url', ''),
                        summary=item_dict.get('summary', ''),
                        timestamp=item_dict.get('timestamp', ''),
                        source=item_dict.get('source', 'Kuwait Intel'),
                        root=item_dict.get('root', 'kuwait'),
                        category=item_dict.get('category', 'general')
                    )
                    all_items.append(news_item)

            logger.info(f"Added {len(kuwait_items)} Kuwait-specific items")

            # Log search status
            if search_status:
                if search_status.get('limit_reached'):
                    logger.warning(f"Search API limit reached! Used {search_status.get('used', '?')}/{search_status.get('limit', '?')} queries today")
                elif search_status.get('warning'):
                    logger.warning(f"Search API warning: {search_status.get('remaining', '?')} queries remaining today")

        except Exception as e:
            logger.warning(f"Kuwait scrapers failed: {e}")

        # 4. Freshness filter: drop items with timestamps older than 2 days
        #    Catches stale results that slip through search engine time filters
        if all_items:
            max_age_hours = 48  # 2 days
            fresh_items = []
            stale_killed = 0
            for item in all_items:
                item_dict = item.to_dict() if hasattr(item, 'to_dict') else item
                ts_str = item_dict.get('timestamp', '')
                if ts_str:
                    try:
                        # Parse various timestamp formats
                        ts_clean = ts_str.replace('Z', '+00:00')
                        if '+' in ts_clean or ts_clean.endswith('00:00'):
                            from datetime import timezone
                            ts = datetime.fromisoformat(ts_clean).replace(tzinfo=None)
                        else:
                            ts = datetime.fromisoformat(ts_clean)
                        age_hours = (datetime.now() - ts).total_seconds() / 3600
                        if age_hours > max_age_hours:
                            stale_killed += 1
                            continue  # Drop stale item
                    except (ValueError, TypeError):
                        pass  # Can't parse timestamp, keep item
                fresh_items.append(item)
            if stale_killed > 0:
                logger.info(f"Freshness filter: dropped {stale_killed}/{len(all_items)} items older than {max_age_hours}h")
            all_items = fresh_items

        # 5. Pre-filter: Quick rule-based scoring on title+snippet to kill obvious noise
        #    before wasting time deep-scraping garbage articles
        scrape_items = all_items  # Default: scrape everything (overridden by pre-filter)
        skip_items = []
        if all_items:
            try:
                from processors.scorer_base import _shared_noise_check, _is_non_latin_title, _is_non_target_language, location_to_lang
                profile_loc = self.config.get('profile', {}).get('location', '')
                allowed_scripts, _, _ = location_to_lang(profile_loc)
                pre_filter_killed = 0
                surviving_items = []
                for item in all_items:
                    item_dict = item.to_dict() if hasattr(item, 'to_dict') else item
                    title = item_dict.get('title', '')
                    text = f"{title} {item_dict.get('summary', '')}".lower()
                    source = item_dict.get('source', '')
                    url = item_dict.get('url', '')
                    root = item_dict.get('root', 'global')

                    noise = _shared_noise_check(title, text, source, url=url, root=root,
                                                allowed_scripts=allowed_scripts)
                    if noise and noise[0] < 4.0:
                        # Mark as pre-filtered so scorer can skip it
                        item.pre_filter_score = noise[0]
                        item.pre_filter_reason = noise[1]
                        pre_filter_killed += 1
                        surviving_items.append(item)  # Still keep for scoring, but skip scraping
                    else:
                        surviving_items.append(item)

                scrape_items = [it for it in surviving_items if not hasattr(it, 'pre_filter_score')]
                skip_items = [it for it in surviving_items if hasattr(it, 'pre_filter_score')]

                logger.info(f"Pre-filter: {pre_filter_killed}/{len(all_items)} items killed as noise, "
                           f"scraping {len(scrape_items)} surviving articles...")
            except Exception as e:
                logger.warning(f"Pre-filter failed, scraping all: {e}")
                scrape_items = all_items
                skip_items = []

        # 6. URL content cache: skip scraping URLs we already have content for
        cached_content = {}
        try:
            output_path = Path(__file__).parent.parent / "output" / "news_data.json"
            if output_path.exists():
                with open(output_path, 'r') as f:
                    prev_data = json.load(f)
                for item in prev_data.get('news', []):
                    content = item.get('content', '')
                    if content and len(content) > 50 and item.get('url'):
                        cached_content[item['url']] = content
                if cached_content:
                    logger.info(f"Loaded {len(cached_content)} cached article contents from previous run")
        except Exception as e:
            logger.debug(f"Could not load content cache: {e}")

        # Apply cache: inject content for already-scraped URLs
        cache_hits = 0
        still_need_scraping = []
        for item in scrape_items:
            if item.url in cached_content:
                item.content = cached_content[item.url]
                cache_hits += 1
            else:
                still_need_scraping.append(item)

        if cache_hits:
            logger.info(f"Content cache: {cache_hits} URLs reused, {len(still_need_scraping)} new to scrape")

        # 6. Deep scrape only NEW article content (parallel)
        if still_need_scraping:
            scraper = ArticleScraper(max_workers=10, timeout=10, max_content_length=1000)
            still_need_scraping = scraper.scrape_items(still_need_scraping)

        # Merge: cached items + freshly scraped items
        scrape_items = [it for it in scrape_items if it.url in cached_content] + still_need_scraping

        # Recombine: scraped items + pre-filtered items (with summary as content)
        for item in skip_items:
            if not item.content:
                item.content = item.summary
        all_items = scrape_items + skip_items

        # Update cache
        self._cache = all_items
        self._cache_time = datetime.now()

        logger.info(f"Total news items fetched: {len(all_items)}")
        return all_items
    
    def extract_entities(self, items: List[NewsItem]) -> Dict[str, int]:
        """
        Extract potential entities (proper nouns) from news items.
        Used for dynamic entity discovery.
        
        Args:
            items: List of news items
            
        Returns:
            Dict of entity name -> mention count
        """
        entity_counts = {}
        
        # Simple pattern: Capitalized words that aren't at sentence start
        # This is basic but works without NLP libraries
        pattern = r'(?<=[.!?]\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)|(?<=\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})'
        
        for item in items:
            text = f"{item.title} {item.summary}"
            
            # Find potential entities
            matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', text)
            
            for match in matches:
                # Filter out common words
                if match.lower() not in {'the', 'a', 'an', 'this', 'that', 'these', 'monday', 
                                          'tuesday', 'wednesday', 'thursday', 'friday', 
                                          'saturday', 'sunday', 'january', 'february', 'march',
                                          'april', 'may', 'june', 'july', 'august', 'september',
                                          'october', 'november', 'december'}:
                    entity_counts[match] = entity_counts.get(match, 0) + 1
        
        return entity_counts


def fetch_news(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch news from config.
    
    Args:
        config: Full configuration dict
        
    Returns:
        List of news item dicts
    """
    news_config = config.get("news", {})
    cache_ttl = config.get("cache", {}).get("news_ttl_seconds", 900)
    
    fetcher = NewsFetcher(news_config)
    items = fetcher.fetch_all(cache_ttl_seconds=cache_ttl)
    
    return [item.to_dict() for item in items]


if __name__ == "__main__":
    # Quick test
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    items = fetch_news(config)
    
    print(f"\n=== Fetched {len(items)} News Items ===")
    for item in items[:10]:
        print(f"\n[{item['source']}] {item['title'][:60]}...")
        print(f"  Root: {item['root']}, Category: {item['category']}")
