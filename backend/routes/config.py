"""
Config routes — POST /api/config (save), GET /api/config (load)
Also handles dynamic category → legacy keyword sync and profile YAML persistence.
"""

import json
import yaml
import logging
from pathlib import Path

from routes.helpers import json_response, error_response, read_json_body

logger = logging.getLogger("STRAT_OS")

# Human-readable names for common tickers
TICKER_NAMES = {
    'NVDA': 'NVIDIA', 'AAPL': 'Apple', 'MSFT': 'Microsoft', 'AMZN': 'Amazon',
    'GOOGL': 'Google', 'META': 'Meta', 'TSLA': 'Tesla', 'AMD': 'AMD',
    'INTC': 'Intel', 'AVGO': 'Broadcom', 'COIN': 'Coinbase', 'MSTR': 'MicroStrategy',
    'PLTR': 'Palantir', 'SMCI': 'Super Micro', 'ARM': 'ARM Holdings',
    'SNOW': 'Snowflake', 'NET': 'Cloudflare', 'SHOP': 'Shopify', 'ROKU': 'Roku',
    'MU': 'Micron', 'QCOM': 'Qualcomm', 'TSM': 'TSMC', 'ASML': 'ASML',
    'MRVL': 'Marvell', 'BABA': 'Alibaba', 'NIO': 'NIO', 'RIVN': 'Rivian',
    'BA': 'Boeing', 'DIS': 'Disney', 'SPY': 'S&P 500 ETF', 'QQQ': 'Nasdaq ETF',
    'TQQQ': '3x Nasdaq', 'SOXL': '3x Semis', 'SQQQ': '3x Short Nasdaq',
    'UVXY': '1.5x VIX', 'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum',
    'SOL-USD': 'Solana', 'XRP-USD': 'XRP', 'DOGE-USD': 'Dogecoin',
    'ADA-USD': 'Cardano', 'AVAX-USD': 'Avalanche',
    'GC=F': 'Gold', 'SI=F': 'Silver', 'CL=F': 'Crude Oil',
    'HG=F': 'Copper', 'PL=F': 'Platinum', 'NG=F': 'Natural Gas',
    'SLB': 'SLB', 'HAL': 'Halliburton', 'BKR': 'Baker Hughes',
    'XOM': 'ExxonMobil', 'CVX': 'Chevron', '^GSPC': 'S&P 500',
    '^IXIC': 'Nasdaq', '^DJI': 'Dow Jones',
}


def _is_masked(value):
    """Check if a value looks like a masked API key."""
    return isinstance(value, str) and '•' in value


def _parse_ticker_objects(raw_tickers: list) -> list:
    """Convert plain strings or dicts into standardized ticker objects."""
    result = []
    for t in raw_tickers:
        if isinstance(t, str):
            sym = t.strip().upper()
            if sym:
                name = TICKER_NAMES.get(sym, sym.replace('-USD', '').replace('=F', ''))
                result.append({"symbol": sym, "name": name})
        elif isinstance(t, dict) and "symbol" in t:
            result.append(t)
    return result



def _update_search_config(config: dict, new_search: dict):
    """Update search provider config, handling masked keys and credit resets."""
    if "search" not in config:
        config["search"] = {}

    if "provider" in new_search:
        config["search"]["provider"] = new_search["provider"]

    if "serper_api_key" in new_search:
        old_key = config["search"].get("serper_api_key", "")
        new_key = new_search["serper_api_key"]
        if not _is_masked(new_key):
            config["search"]["serper_api_key"] = new_key
            if old_key != new_key and new_key:
                try:
                    from fetchers.serper_search import SerperQueryTracker
                    SerperQueryTracker().set_remaining(2500)
                    logger.info("Serper API key changed, reset credits to 2500")
                except Exception as e:
                    logger.warning(f"Failed to reset Serper tracker: {e}")

    if "serper_credits" in new_search and new_search["serper_credits"]:
        config["search"]["serper_credits"] = new_search["serper_credits"]
        try:
            from fetchers.serper_search import SerperQueryTracker
            SerperQueryTracker().set_remaining(new_search["serper_credits"])
        except Exception as e:
            logger.warning(f"Failed to sync Serper credits: {e}")

    if "google_api_key" in new_search:
        if not _is_masked(new_search["google_api_key"]):
            config["search"]["google_api_key"] = new_search["google_api_key"]
    if "google_cx" in new_search:
        config["search"]["google_cx"] = new_search["google_cx"]


def handle_config_save(handler, strat, auth_helpers):
    """POST /api/config — Save configuration changes."""
    try:
        new_config = read_json_body(handler)
        config = strat.config

        # Profile (merge, don't replace)
        if "profile" in new_config:
            if "profile" not in config:
                config["profile"] = {}
            for key in ["role", "location", "context", "interests"]:
                if key in new_config["profile"]:
                    config["profile"][key] = new_config["profile"][key]

        # Market tickers
        if "market" in new_config and "tickers" in new_config["market"]:
            config["market"]["tickers"] = new_config["market"]["tickers"]

        # News config
        if "news" in new_config:
            nc = new_config["news"]
            for key in ["timelimit", "career", "finance", "tech_trends", "regional", "rss_feeds"]:
                if key in nc:
                    config["news"][key] = nc[key]

        # Search config
        if "search" in new_config:
            _update_search_config(config, new_config["search"])

        # Top-level tickers array (from Simple mode)
        if "tickers" in new_config and isinstance(new_config["tickers"], list):
            ticker_objects = _parse_ticker_objects(new_config["tickers"])
            if "market" not in config:
                config["market"] = {}
            config["market"]["tickers"] = ticker_objects
            logger.info(f"  Tickers updated: {len(ticker_objects)} ({', '.join(t['symbol'] for t in ticker_objects[:5])}{'...' if len(ticker_objects) > 5 else ''})")

        # Dynamic categories
        if "dynamic_categories" in new_config:
            config["dynamic_categories"] = new_config["dynamic_categories"]

        # Extra feed toggles
        for key in ["extra_feeds_finance", "extra_feeds_politics",
                     "custom_feeds_finance", "custom_feeds_politics",
                     "custom_feeds", "custom_tab_name"]:
            if key in new_config:
                config[key] = new_config[key]

        # Save to disk
        with open(strat.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Log
        saved_keys = list(new_config.keys())
        logger.info(f"Config saved: keys={saved_keys}")
        for feed_key in ["extra_feeds_finance", "extra_feeds_politics"]:
            if feed_key in new_config:
                enabled = sum(1 for v in new_config[feed_key].values() if v)
                label = feed_key.replace("extra_feeds_", "").capitalize()
                logger.info(f"  {label} sources: {enabled}/{len(new_config[feed_key])} enabled")

        # Reload in memory
        strat.config = config

        # Sync to profile YAML and update profile cache (A2.1)
        token = handler.headers.get('X-Auth-Token', '')
        current_user = auth_helpers['get_session_profile'](token)
        if current_user:
            _sync_to_profile_yaml(config, current_user, auth_helpers)
            strat.cache_profile_config(current_user)

        # Reinitialize MarketFetcher if tickers changed
        if "tickers" in new_config or ("market" in new_config and "tickers" in new_config.get("market", {})):
            from fetchers.market import MarketFetcher
            strat.market_fetcher = MarketFetcher(config.get("market", {}))
            logger.info(f"  MarketFetcher reloaded with {len(strat.market_fetcher.tickers)} tickers")

        json_response(handler, {"status": "saved"}, compress=False)

    except Exception as e:
        error_response(handler, str(e), 500)


def _sync_to_profile_yaml(config: dict, username: str, auth_helpers: dict):
    """Persist config changes to the user's profile YAML."""
    try:
        safe = auth_helpers['safe_name'](username)
        filepath = auth_helpers['profiles_dir']() / f"{safe}.yaml"
        if not filepath.exists():
            return
        with open(filepath) as f:
            profile_data = yaml.safe_load(f) or {}
        for key in ["profile", "market", "news", "dynamic_categories",
                     "extra_feeds_finance", "extra_feeds_politics",
                     "custom_feeds", "custom_tab_name"]:
            if key in config:
                profile_data[key] = config[key]
        with open(filepath, "w") as f:
            yaml.dump(profile_data, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        logger.debug(f"Failed to sync profile YAML for {username}: {e}")
