"""
STRAT_OS - Google Custom Search Integration

Uses Google Custom Search API with:
- 100 free queries/day
- Query counting and limit tracking
- Warning when approaching limit
- Hard stop at limit

Setup required:
1. Create Google Cloud Project: https://console.cloud.google.com
2. Enable Custom Search API
3. Create API Key (no restrictions needed for local use)
4. Create Custom Search Engine: https://programmablesearchengine.google.com
   - Set to "Search the entire web"
   - Get the Search Engine ID (cx)
5. Add to config.yaml:
   search:
     provider: google  # or 'duckduckgo'
     google_api_key: YOUR_API_KEY
     google_cx: YOUR_SEARCH_ENGINE_ID
"""

import logging
import json
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
import requests

logger = logging.getLogger(__name__)

# Query tracking file
QUERY_TRACKER_FILE = Path(__file__).parent.parent / "data" / "google_query_tracker.json"

# Limits
DAILY_LIMIT = 100
WARNING_THRESHOLD = 80  # Warn at 80% usage


class GoogleSearchError(Exception):
    """Custom exception for Google Search errors."""
    pass


class DailyLimitReached(GoogleSearchError):
    """Raised when daily query limit is reached."""
    pass


class QueryTracker:
    """Tracks daily Google API query usage."""

    def __init__(self):
        self.tracker_file = QUERY_TRACKER_FILE
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
                    self.date = data.get('date', '')
                    self.count = data.get('count', 0)
                    self.queries = data.get('queries', [])
            except Exception as e:
                logger.warning(f"Failed to load query tracker: {e}")
                self._reset()
        else:
            self._reset()

        # Reset if new day
        today = date.today().isoformat()
        if self.date != today:
            self._reset()
            self.date = today

    def _reset(self):
        """Reset tracker for new day."""
        self.date = date.today().isoformat()
        self.count = 0
        self.queries = []
        self._save()

    def _save(self):
        """Save tracker to file."""
        try:
            with open(self.tracker_file, 'w') as f:
                json.dump({
                    'date': self.date,
                    'count': self.count,
                    'queries': self.queries[-50:]  # Keep last 50 for debugging
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save query tracker: {e}")

    def can_query(self) -> Tuple[bool, int, str]:
        """
        Check if we can make another query.

        Returns:
            (can_query, remaining, status_message)
        """
        # Reload to check for new day
        self._load()

        remaining = DAILY_LIMIT - self.count

        if remaining <= 0:
            return False, 0, f"Daily limit reached ({DAILY_LIMIT} queries). Resets at midnight PT."

        if self.count >= WARNING_THRESHOLD:
            return True, remaining, f"Warning: {remaining} queries remaining today"

        return True, remaining, "ok"

    def increment(self, query: str):
        """Record a query."""
        self.count += 1
        self.queries.append({
            'query': query[:100],  # Truncate for storage
            'time': datetime.now().isoformat()
        })
        self._save()

    def get_status(self) -> Dict[str, Any]:
        """Get current usage status."""
        self._load()
        remaining = DAILY_LIMIT - self.count
        return {
            'date': self.date,
            'used': self.count,
            'remaining': remaining,
            'limit': DAILY_LIMIT,
            'percentage': round((self.count / DAILY_LIMIT) * 100, 1),
            'warning': self.count >= WARNING_THRESHOLD,
            'limit_reached': remaining <= 0
        }


class GoogleSearchClient:
    """Google Custom Search API client."""

    API_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: str, cx: str):
        """
        Initialize Google Search client.

        Args:
            api_key: Google API key
            cx: Custom Search Engine ID
        """
        self.api_key = api_key
        self.cx = cx
        self.tracker = QueryTracker()

    def search(
        self,
        query: str,
        num_results: int = 5,
        date_restrict: str = "w1"  # w1 = past week, d1 = past day, m1 = past month
    ) -> List[Dict[str, Any]]:
        """
        Search Google and return results.

        Args:
            query: Search query
            num_results: Number of results (max 10 per request)
            date_restrict: Time restriction (d1, w1, m1)

        Returns:
            List of search results with title, url, snippet

        Raises:
            DailyLimitReached: When daily quota is exhausted
            GoogleSearchError: For other API errors
        """
        # Check if we can query
        can_query, remaining, status = self.tracker.can_query()

        if not can_query:
            raise DailyLimitReached(status)

        # Make the request
        params = {
            'key': self.api_key,
            'cx': self.cx,
            'q': query,
            'num': min(num_results, 10),  # Max 10 per request
            'dateRestrict': date_restrict,
        }

        try:
            response = requests.get(self.API_URL, params=params, timeout=10)

            # Record the query (even if it fails, it counts against quota)
            self.tracker.increment(query)

            if response.status_code == 429:
                raise DailyLimitReached("Google API rate limit exceeded")

            if response.status_code != 200:
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                raise GoogleSearchError(f"Google API error: {error_msg}")

            data = response.json()
            results = []

            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                })

            logger.info(f"Google search '{query[:40]}...' returned {len(results)} results ({remaining-1} queries left)")
            return results

        except requests.RequestException as e:
            raise GoogleSearchError(f"Network error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current API usage status."""
        return self.tracker.get_status()


def get_google_client(config: Dict[str, Any]) -> Optional[GoogleSearchClient]:
    """
    Get Google Search client if configured.

    Args:
        config: Full config dict

    Returns:
        GoogleSearchClient or None if not configured
    """
    search_config = config.get('search', {})

    api_key = search_config.get('google_api_key', '')
    cx = search_config.get('google_cx', '')

    if not api_key or not cx:
        return None

    if api_key == 'YOUR_API_KEY' or cx == 'YOUR_SEARCH_ENGINE_ID':
        logger.warning("Google Search not configured - using placeholder values")
        return None

    return GoogleSearchClient(api_key, cx)


# Convenience function for status endpoint
def get_search_status(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get search provider status for API endpoint."""
    search_config = config.get('search', {})
    provider = search_config.get('provider', 'duckduckgo')

    if provider == 'serper':
        serper_key = search_config.get('serper_api_key', '')
        if serper_key and serper_key != 'YOUR_SERPER_API_KEY':
            # Import serper tracker to get usage stats
            try:
                from fetchers.serper_search import SerperQueryTracker
                tracker = SerperQueryTracker()
                status = tracker.get_status()
                status['provider'] = 'serper'
                return status
            except Exception as e:
                logger.warning(f"Failed to get Serper status: {e}")

            # Fallback if import fails
            return {
                'provider': 'serper',
                'limit': 2500,
                'used': 0,
                'remaining': 2500,
                'warning': False,
                'limit_reached': False
            }

    if provider == 'google':
        client = get_google_client(config)
        if client:
            status = client.get_status()
            status['provider'] = 'google'
            return status

    return {
        'provider': 'duckduckgo',
        'limit': 'unlimited',
        'warning': False,
        'limit_reached': False
    }
