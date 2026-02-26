"""
STRAT_OS - Extra RSS Feeds (Finance & Politics)
Lightweight RSS-only fetchers for Finance News and Politics tabs.
No search API calls — purely RSS aggregation with minimal scoring.

Feeds are selected from a master catalog based on user config toggles.
"""

import feedparser
import logging
import hashlib
import random
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
]

# ═══════════════════════════════════════════════════════════════
# MASTER FEED CATALOG
# Every known feed. Users toggle which ones to use.
# "on" = enabled by default for new users.
# ═══════════════════════════════════════════════════════════════

FINANCE_CATALOG = [
    # --- Global Markets ---
    {"id": "cnbc_top",       "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",   "name": "CNBC Top News",      "region": "US",      "category": "markets",    "on": True},
    {"id": "cnbc_finance",   "url": "https://www.cnbc.com/id/10000664/device/rss/rss.html",    "name": "CNBC Finance",       "region": "US",      "category": "markets",    "on": True},
    {"id": "marketwatch",    "url": "https://feeds.marketwatch.com/marketwatch/topstories/",    "name": "MarketWatch",        "region": "US",      "category": "markets",    "on": True},
    {"id": "mw_pulse",       "url": "https://feeds.marketwatch.com/marketwatch/marketpulse/",   "name": "MarketWatch Pulse",  "region": "US",      "category": "markets",    "on": False},
    {"id": "yahoo_fin",      "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US", "name": "Yahoo Finance", "region": "US", "category": "markets", "on": True},
    {"id": "investing",      "url": "https://www.investing.com/rss/news.rss",                   "name": "Investing.com",      "region": "Global",  "category": "markets",    "on": True},
    {"id": "ft",             "url": "https://www.ft.com/?format=rss",                           "name": "Financial Times",    "region": "UK",      "category": "business",   "on": True},
    {"id": "bloomberg",      "url": "https://feeds.bloomberg.com/markets/news.rss",             "name": "Bloomberg Markets",  "region": "Global",  "category": "markets",    "on": True},
    # --- Business & Economy ---
    {"id": "wsj_world",      "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories",  "name": "WSJ / Dow Jones",    "region": "US",      "category": "business",   "on": False},
    {"id": "economist",      "url": "https://www.economist.com/finance-and-economics/rss.xml",  "name": "The Economist",      "region": "Global",  "category": "business",   "on": False},
    {"id": "seekingalpha",   "url": "https://seekingalpha.com/market_currents.xml",              "name": "Seeking Alpha",      "region": "US",      "category": "markets",    "on": False},
    {"id": "biz_insider",    "url": "https://markets.businessinsider.com/rss/news",              "name": "Business Insider",   "region": "US",      "category": "business",   "on": False},
    # --- Crypto ---
    {"id": "coindesk",       "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",          "name": "CoinDesk",           "region": "Global",  "category": "crypto",     "on": False},
    {"id": "cointelegraph",  "url": "https://cointelegraph.com/rss",                            "name": "CoinTelegraph",      "region": "Global",  "category": "crypto",     "on": False},
    # --- GCC / Middle East Finance ---
    {"id": "gn_kuwait_biz",  "url": "https://news.google.com/rss/search?q=Kuwait+business+economy&hl=en&gl=KW&ceid=KW:en", "name": "Kuwait Business (Google)", "region": "Kuwait", "category": "business", "on": True},
    {"id": "gn_gcc_finance", "url": "https://news.google.com/rss/search?q=GCC+finance+economy+Saudi+UAE&hl=en&gl=AE&ceid=AE:en", "name": "GCC Finance (Google)", "region": "GCC", "category": "business", "on": False},
    {"id": "zawya",          "url": "https://www.zawya.com/sitemaps/en/rss",                    "name": "Zawya",              "region": "GCC",     "category": "business",   "on": False},
    {"id": "arab_times_biz", "url": "https://www.arabtimesonline.com/rssFeed/30/",              "name": "Arab Times Business","region": "Kuwait",  "category": "business",   "on": False},
    {"id": "boursa_kuwait",  "url": "https://rss.boursakuwait.com.kw/A/rss/FeedFull.aspx",     "name": "Boursa Kuwait",      "region": "Kuwait",  "category": "markets",    "on": False},
    # --- Energy / Oil ---
    {"id": "oilprice",       "url": "https://oilprice.com/rss/main",                            "name": "OilPrice.com",       "region": "Global",  "category": "energy",     "on": False},
    {"id": "rigzone",        "url": "https://www.rigzone.com/news/rss/rigzone_latest.aspx",     "name": "Rigzone",            "region": "Global",  "category": "energy",     "on": False},
]

POLITICS_CATALOG = [
    # --- International Wire ---
    {"id": "bbc_world",      "url": "https://feeds.bbci.co.uk/news/world/rss.xml",              "name": "BBC World",          "region": "UK",      "category": "world",      "on": True},
    {"id": "bbc_mideast",    "url": "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",  "name": "BBC Middle East",    "region": "UK",      "category": "mideast",    "on": True},
    {"id": "aljazeera",      "url": "https://www.aljazeera.com/xml/rss/all.xml",                "name": "Al Jazeera",         "region": "Qatar",   "category": "world",      "on": True},
    {"id": "npr_world",      "url": "https://feeds.npr.org/1004/rss.xml",                       "name": "NPR World",          "region": "US",      "category": "world",      "on": True},
    # --- US ---
    {"id": "nyt_world",      "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",   "name": "NYT World",          "region": "US",      "category": "world",      "on": True},
    {"id": "nyt_mideast",    "url": "https://rss.nytimes.com/services/xml/rss/nyt/MiddleEast.xml", "name": "NYT Middle East", "region": "US",      "category": "mideast",    "on": True},
    {"id": "wapo_world",     "url": "https://feeds.washingtonpost.com/rss/world",               "name": "WaPo World",         "region": "US",      "category": "world",      "on": True},
    {"id": "wapo_politics",  "url": "https://feeds.washingtonpost.com/rss/politics",            "name": "WaPo Politics",      "region": "US",      "category": "us_politics", "on": False},
    {"id": "cnn_world",      "url": "http://rss.cnn.com/rss/edition_world.rss",                 "name": "CNN World",          "region": "US",      "category": "world",      "on": False},
    {"id": "fox_world",      "url": "https://moxie.foxnews.com/feedburner/world.xml",           "name": "Fox News World",     "region": "US",      "category": "world",      "on": False},
    # --- Europe ---
    {"id": "guardian",       "url": "https://www.theguardian.com/world/rss",                    "name": "The Guardian",       "region": "UK",      "category": "world",      "on": False},
    {"id": "dw",             "url": "https://rss.dw.com/xml/rss-en-top",                        "name": "DW News",            "region": "Germany", "category": "world",      "on": False},
    {"id": "france24",       "url": "https://www.france24.com/en/rss",                          "name": "France 24",          "region": "France",  "category": "world",      "on": False},
    # --- Middle East & GCC ---
    {"id": "gn_kuwait",      "url": "https://news.google.com/rss/search?q=Kuwait&hl=en&gl=KW&ceid=KW:en", "name": "Kuwait News (Google)", "region": "Kuwait", "category": "local", "on": True},
    {"id": "arab_times",     "url": "https://www.arabtimesonline.com/rssFeed/47/",              "name": "Arab Times",         "region": "Kuwait",  "category": "local",      "on": True},
    {"id": "times_kuwait",   "url": "https://timeskuwait.com/feed",                             "name": "Times Kuwait",       "region": "Kuwait",  "category": "local",      "on": False},
    {"id": "gn_mideast",     "url": "https://news.google.com/rss/search?q=Middle+East+news&hl=en&gl=US&ceid=US:en", "name": "Middle East (Google)", "region": "GCC", "category": "mideast", "on": True},
    {"id": "alarabiya",      "url": "https://english.alarabiya.net/Articles/rss.xml",           "name": "Al Arabiya English", "region": "KSA",     "category": "mideast",    "on": False},
    {"id": "middleeasteye",  "url": "https://www.middleeasteye.net/rss",                        "name": "Middle East Eye",    "region": "UK",      "category": "mideast",    "on": False},
    {"id": "newarab",        "url": "https://newarab.com/rss",                                  "name": "The New Arab",       "region": "UK",      "category": "mideast",    "on": False},
    # --- Asia ---
    {"id": "scmp",           "url": "https://www.scmp.com/rss/91/feed",                         "name": "SCMP",               "region": "HK",      "category": "asia",       "on": False},
    {"id": "nikkei_asia",    "url": "https://asia.nikkei.com/rss/feed/nar",                     "name": "Nikkei Asia",        "region": "Japan",   "category": "asia",       "on": False},
]


def get_catalog(feed_type: str = "finance") -> List[Dict[str, Any]]:
    """Return the full catalog for a feed type (for the Settings UI)."""
    catalog = FINANCE_CATALOG if feed_type == "finance" else POLITICS_CATALOG
    # Return clean copies without URLs (frontend only needs id, name, region, category, on)
    return [
        {
            "id": f["id"],
            "name": f["name"],
            "region": f["region"],
            "category": f["category"],
            "on": f["on"],  # Default state
        }
        for f in catalog
    ]


def _get_enabled_feeds(feed_type: str, config: Optional[Dict] = None) -> List[Dict]:
    """Resolve which feeds to use from config toggles + catalog defaults."""
    catalog = FINANCE_CATALOG if feed_type == "finance" else POLITICS_CATALOG
    
    # Check config for user overrides
    feed_key = f"extra_feeds_{feed_type}"
    user_toggles = {}
    if config:
        user_toggles = config.get(feed_key, {})  # {"cnbc_top": true, "reuters_biz": false, ...}
    
    enabled = []
    for feed in catalog:
        fid = feed["id"]
        # User override takes priority, otherwise use catalog default
        if fid in user_toggles:
            if user_toggles[fid]:
                enabled.append(feed)
        elif feed["on"]:
            enabled.append(feed)
    
    return enabled


def _fetch_single_feed(feed_config: Dict[str, str], max_items: int = 5) -> List[Dict[str, Any]]:
    """Fetch a single RSS feed and return normalized items."""
    url = feed_config["url"]
    name = feed_config["name"]
    root = feed_config.get("root", "finance")
    category = feed_config["category"]
    
    ua = random.choice(USER_AGENTS)
    try:
        response = requests.get(url, headers={"User-Agent": ua}, timeout=15, allow_redirects=True)
        if response.status_code >= 400:
            logger.warning(f"  ✗ {name}: HTTP {response.status_code}")
            return []
        feed = feedparser.parse(response.content)
    except requests.RequestException as e:
        logger.warning(f"  ✗ {name}: {type(e).__name__}: {e}")
        return []
    
    if feed.bozo and not feed.entries:
        logger.warning(f"  ✗ {name}: Parse error: {type(feed.bozo_exception).__name__}")
        return []
    
    if not feed.entries:
        logger.warning(f"  ✗ {name}: Feed returned 0 entries")
        return []
    
    items = []
    for entry in feed.entries[:max_items]:
        # Timestamp
        timestamp = ""
        for attr in ("published_parsed", "updated_parsed"):
            parsed = getattr(entry, attr, None)
            if parsed:
                try:
                    timestamp = datetime(*parsed[:6]).isoformat()
                except Exception:
                    pass
                break
        
        # Summary — strip HTML
        summary = ""
        if hasattr(entry, "summary"):
            from bs4 import BeautifulSoup
            summary = BeautifulSoup(entry.summary, "html.parser").get_text()[:300]
        
        link = getattr(entry, "link", "")
        title = getattr(entry, "title", "No Title")
        
        news_id = hashlib.md5(f"{link}{title}".encode()).hexdigest()[:12]
        
        items.append({
            "id": news_id,
            "title": title,
            "url": link,
            "summary": summary,
            "content": "",
            "source": name,
            "root": root,
            "category": category,
            "timestamp": timestamp,
            "score": 6.0,
            "score_reason": f"RSS: {name}",
            "scored_by": "rss_direct",
        })
    
    return items


def fetch_extra_feeds(feed_type: str = "finance", config: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Fetch enabled feeds for a given type.
    
    Args:
        feed_type: "finance" or "politics"
        config: Full app config dict (reads extra_feeds toggles + custom feeds)
    
    Returns:
        List of news item dicts
    """
    feeds = _get_enabled_feeds(feed_type, config)
    
    # Add custom feeds from config
    custom_key = f"custom_feeds_{feed_type}"
    if config:
        custom_feeds = config.get(custom_key, [])
        for cf in custom_feeds:
            if cf.get("url"):
                feeds.append({
                    "id": f"custom_{hashlib.md5(cf['url'].encode()).hexdigest()[:8]}",
                    "url": cf["url"],
                    "name": cf.get("name", "Custom"),
                    "region": "Custom",
                    "category": "custom",
                    "on": True,
                })
    
    # Set root field based on type
    for f in feeds:
        f["root"] = feed_type if feed_type == "finance" else "politics"
    
    all_items = []
    feed_names = [f["name"] for f in feeds]
    logger.info(f"Extra feeds ({feed_type}): fetching {len(feeds)} feeds: {', '.join(feed_names)}")
    
    executor = ThreadPoolExecutor(max_workers=6)
    futures = {executor.submit(_fetch_single_feed, fc): fc for fc in feeds}
    
    completed = 0
    succeeded = 0
    try:
        for future in as_completed(futures, timeout=60):
            fc = futures[future]
            try:
                items = future.result(timeout=1)
                all_items.extend(items)
                completed += 1
                if items:
                    succeeded += 1
                    logger.info(f"  ✓ {fc['name']}: {len(items)} items")
            except Exception as e:
                completed += 1
                logger.warning(f"  ✗ {fc['name']}: {type(e).__name__}: {e}")
    except TimeoutError:
        logger.warning(f"Extra feeds ({feed_type}) timeout — {completed}/{len(feeds)} done")
    
    executor.shutdown(wait=False, cancel_futures=True)
    
    # Deduplicate by URL
    seen = set()
    unique = []
    for item in all_items:
        key = item["url"] or item["title"]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    
    # Sort by timestamp descending (newest first)
    unique.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Cap at 50
    unique = unique[:50]
    
    logger.info(f"Extra feeds ({feed_type}): {len(unique)} items from {succeeded}/{len(feeds)} feeds ({len(feeds) - succeeded} failed)")
    return unique
