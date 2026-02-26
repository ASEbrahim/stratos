"""
STRAT_OS - Serper.dev Search Integration

Serper.dev provides Google search results via a simple API.
- Free tier: 2,500 queries
- Returns actual Google results (not a separate index)
- Simple setup: just an API key

Get your API key at: https://serper.dev
"""

import logging
import json
import time
import requests
import threading
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Query tracking file
SERPER_TRACKER_FILE = Path(__file__).parent.parent / "data" / "serper_query_tracker.json"

# Serper free tier limit
SERPER_FREE_LIMIT = 2500
WARNING_THRESHOLD_PERCENT = 80  # Warn at 80% usage


class SerperQueryTracker:
    """Tracks Serper API query usage locally."""

    def __init__(self):
        self.tracker_file = SERPER_TRACKER_FILE
        self._lock = threading.Lock()
        self._ensure_data_dir()
        self._load()

    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        """Load tracker from file."""
        if self.tracker_file.exists():
            try:
                with open(self.tracker_file, 'r') as f:
                    data = json.load(f)
                    self.total_count = data.get('total_count', 0)
                    self.daily_counts = data.get('daily_counts', {})
                    self.last_queries = data.get('last_queries', [])
            except Exception as e:
                logger.warning(f"Failed to load Serper query tracker: {e}")
                self._reset()
        else:
            self._reset()

    def _reset(self):
        """Reset tracker."""
        self.total_count = 0
        self.daily_counts = {}
        self.last_queries = []
        self._save()

    def _save(self):
        """Save tracker to file."""
        try:
            with open(self.tracker_file, 'w') as f:
                json.dump({
                    'total_count': self.total_count,
                    'daily_counts': self.daily_counts,
                    'last_queries': self.last_queries[-50:]  # Keep last 50 for debugging
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save Serper query tracker: {e}")

    def increment(self, query: str):
        """Record a query (thread-safe)."""
        with self._lock:
            self._load()  # Reload to get latest

            self.total_count += 1

            # Track daily count
            today = date.today().isoformat()
            self.daily_counts[today] = self.daily_counts.get(today, 0) + 1

            # Keep only last 30 days of daily counts
            recent_dates = sorted(self.daily_counts.keys())[-30:]
            self.daily_counts = {d: self.daily_counts[d] for d in recent_dates}

            # Log the query
            self.last_queries.append({
                'query': query[:100],
                'time': datetime.now().isoformat()
            })

            self._save()

    def get_status(self) -> Dict[str, Any]:
        """Get current usage status (thread-safe)."""
        with self._lock:
            self._load()

        remaining = max(0, SERPER_FREE_LIMIT - self.total_count)
        percentage = round((self.total_count / SERPER_FREE_LIMIT) * 100, 1)

        # Today's usage
        today = date.today().isoformat()
        today_count = self.daily_counts.get(today, 0)

        return {
            'used': self.total_count,
            'remaining': remaining,
            'limit': SERPER_FREE_LIMIT,
            'percentage': percentage,
            'today_count': today_count,
            'warning': percentage >= WARNING_THRESHOLD_PERCENT,
            'limit_reached': remaining <= 0
        }

    def reset_count(self):
        """Reset the query count (for when user gets new credits)."""
        self._reset()
        logger.info("Serper query tracker reset")

    def set_remaining(self, remaining: int):
        """Set remaining credits (syncs from Serper dashboard)."""
        self._load()
        # Calculate used from remaining
        self.total_count = max(0, SERPER_FREE_LIMIT - remaining)
        self._save()
        logger.info(f"Serper tracker synced: {remaining} remaining ({self.total_count} used)")


class SerperSearchError(Exception):
    """Custom exception for Serper Search errors."""
    pass


class SerperSearchClient:
    """Serper.dev Search API client."""

    API_URL = "https://google.serper.dev/search"
    # Minimum seconds between requests to avoid rate limiting
    MIN_REQUEST_INTERVAL = 0.25

    def __init__(self, api_key: str):
        """
        Initialize Serper Search client.

        Args:
            api_key: Serper.dev API key
        """
        self.api_key = api_key
        self.tracker = SerperQueryTracker()
        self._throttle_lock = threading.Lock()
        self._last_request_time = 0.0

    def search(
        self,
        query: str,
        num_results: int = 10,
        search_type: str = "search",
        time_period: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search using Serper.dev API.

        Args:
            query: Search query
            num_results: Number of results (max 100)
            search_type: "search" for web, "news" for news results
            time_period: Time filter - "d" (day), "w" (week), "m" (month), "y" (year)

        Returns:
            List of search results with title, url, snippet

        Raises:
            SerperSearchError: For API errors
        """
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }

        payload = {
            'q': query,
            'num': min(num_results, 100),
        }

        # Add time filter if specified
        if time_period:
            # Serper uses tbs parameter for time
            time_map = {
                'd': 'qdr:d',   # past day
                'w': 'qdr:w',   # past week
                'm': 'qdr:m',   # past month
                'y': 'qdr:y',   # past year
            }
            if time_period in time_map:
                payload['tbs'] = time_map[time_period]

        # Choose endpoint based on search type
        url = self.API_URL
        if search_type == "news":
            url = "https://google.serper.dev/news"

        try:
            # Throttle: ensure minimum interval between requests
            with self._throttle_lock:
                now = time.monotonic()
                elapsed = now - self._last_request_time
                if elapsed < self.MIN_REQUEST_INTERVAL:
                    time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
                self._last_request_time = time.monotonic()

            response = requests.post(url, headers=headers, json=payload, timeout=10)

            # Track the query (even if it fails, it counts against quota)
            self.tracker.increment(query)

            if response.status_code == 401:
                raise SerperSearchError("Invalid API key")

            if response.status_code == 429:
                # Retry once after a pause instead of immediately failing
                time.sleep(2.0)
                with self._throttle_lock:
                    self._last_request_time = time.monotonic()
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                if response.status_code == 429:
                    raise SerperSearchError("Rate limit exceeded")

            if response.status_code != 200:
                error_msg = response.text[:200]
                raise SerperSearchError(f"Serper API error ({response.status_code}): {error_msg}")

            data = response.json()
            results = []

            # Parse organic results (web search)
            for item in data.get('organic', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                })

            # Parse news results if news search
            for item in data.get('news', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'date': item.get('date', ''),
                    'source': item.get('source', ''),
                })

            status = self.tracker.get_status()
            logger.info(f"Serper search '{query[:40]}...' returned {len(results)} results ({status['remaining']} queries left)")
            return results

        except requests.RequestException as e:
            raise SerperSearchError(f"Network error: {e}")

    def search_news(
        self,
        query: str,
        num_results: int = 10,
        time_period: str = None
    ) -> List[Dict[str, Any]]:
        """
        Search news specifically using Serper.dev.

        Args:
            query: Search query
            num_results: Number of results
            time_period: Time filter - "d", "w", "m", "y"

        Returns:
            List of news results
        """
        return self.search(query, num_results, search_type="news", time_period=time_period)

    def get_status(self) -> Dict[str, Any]:
        """Get current API usage status."""
        return self.tracker.get_status()

    def reset_tracker(self):
        """Reset the query tracker (when user resets credits)."""
        self.tracker.reset_count()


def get_serper_client(config: Dict[str, Any]) -> Optional[SerperSearchClient]:
    """
    Get Serper Search client if configured.

    Args:
        config: Full config dict

    Returns:
        SerperSearchClient or None if not configured
    """
    search_config = config.get('search', {})
    api_key = search_config.get('serper_api_key', '')

    if not api_key or api_key == 'YOUR_SERPER_API_KEY':
        return None

    return SerperSearchClient(api_key)


if __name__ == "__main__":
    # Test the client
    import os

    logging.basicConfig(level=logging.INFO)

    api_key = os.environ.get('SERPER_API_KEY', '')
    if api_key:
        client = SerperSearchClient(api_key)
        results = client.search("Kuwait engineering jobs", num_results=5, time_period="w")

        print(f"\n=== Found {len(results)} results ===")
        for r in results:
            print(f"\n{r['title']}")
            print(f"  {r['url']}")
            print(f"  {r['snippet'][:100]}...")
    else:
        print("Set SERPER_API_KEY environment variable to test")
