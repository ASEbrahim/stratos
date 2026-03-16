"""
STRAT_OS - SearXNG Search Integration

SearXNG is a self-hosted metasearch engine that aggregates results
from multiple search engines. Provides unlimited free searches.

Setup: docker run -d --name searxng -p 8888:8080 searxng/searxng
Requires JSON format enabled in SearXNG settings.yml.
"""

import logging
import time
import threading
import requests
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SearXNGSearchClient:
    """SearXNG Search API client."""

    # Minimum seconds between requests to be polite to the instance
    MIN_REQUEST_INTERVAL = 0.1

    def __init__(self, host: str = "http://localhost:8888"):
        self.host = host.rstrip("/")
        # Validate host URL
        parsed = urlparse(self.host)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            logger.warning(f"Invalid SearXNG host URL: {self.host!r}")
            self.host = "http://localhost:8888"
        self._throttle_lock = threading.Lock()
        self._last_request_time = 0.0
        self._session = requests.Session()

    def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: str = "search",
        time_period: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Search using SearXNG JSON API.

        Args:
            query: Search query
            num_results: Number of results (paginate_results not supported, returns first page)
            search_type: "search" for web, "news" for news results
            time_period: Time filter - "d" (day), "w" (week), "m" (month), "y" (year)

        Returns:
            List of search results with title, url, snippet
        """
        params = {
            "q": query,
            "format": "json",
        }

        # SearXNG categories
        if search_type == "news":
            params["categories"] = "news"

        # Time range mapping
        if time_period:
            time_map = {
                "d": "day",
                "w": "week",
                "m": "month",
                "y": "year",
            }
            if time_period in time_map:
                params["time_range"] = time_map[time_period]

        # Throttle requests
        with self._throttle_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self.MIN_REQUEST_INTERVAL:
                time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
            self._last_request_time = time.monotonic()

        try:
            resp = self._session.get(
                f"{self.host}/search",
                params=params,
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning(f"SearXNG returned {resp.status_code} for '{query[:40]}'")
                return []

            data = resp.json()
            raw_results = data.get("results", [])

            # Normalize to same format as Serper
            results = []
            for r in raw_results[:num_results]:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                })

            logger.debug(f"SearXNG: '{query[:30]}' → {len(results)} results")
            return results

        except requests.RequestException as e:
            logger.error(f"SearXNG request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"SearXNG parse error: {e}")
            return []

    def is_available(self) -> bool:
        """Check if SearXNG instance is reachable."""
        try:
            resp = self._session.get(f"{self.host}/", timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"SearXNG availability check failed for {self.host}: {e}")
            return False


def get_searxng_client(config: Dict[str, Any]) -> Optional[SearXNGSearchClient]:
    """Create SearXNG client from config if configured."""
    search_config = config.get("search", {})
    host = search_config.get("searxng_host", "http://localhost:8888")
    client = SearXNGSearchClient(host=host)
    if client.is_available():
        return client
    logger.warning(f"SearXNG not available at {host}")
    return None
