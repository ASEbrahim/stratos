"""
STRAT_OS - Main Orchestrator
The central brain that coordinates all modules and generates the final output.
"""

import copy
import hashlib
import json
import logging
import logging.handlers
import os
import yaml
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from database import get_database, Database
from fetchers.market import MarketFetcher
from fetchers.news import NewsFetcher
from fetchers.discovery import EntityDiscovery
from processors.scorer_adaptive import AdaptiveScorer
from processors.briefing import BriefingGenerator
from sse import SSEManager


def _create_scorer(config, db=None):
    """Create the scorer. B3.3: AdaptiveScorer is now the sole scorer."""
    role = config.get('profile', {}).get('role', '?')
    location = config.get('profile', {}).get('location', '?')
    logger.info(f"Using AdaptiveScorer for profile: {role} in {location}")
    return AdaptiveScorer(config, db=db)


# Configure logging — console + rotating file (5 × 10 MB)
_log_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                             datefmt='%H:%M:%S')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

_log_dir = Path(__file__).parent / "logs"
_log_dir.mkdir(exist_ok=True)
_file_handler = logging.handlers.RotatingFileHandler(
    _log_dir / "stratos.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(_log_fmt)
_file_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger("STRAT_OS")


class StratOS:
    """
    Main STRAT_OS orchestrator.

    Coordinates:
    - Market data fetching
    - News fetching
    - AI scoring
    - Entity discovery
    - Briefing generation
    - JSON output
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize STRAT_OS.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._load_env_secrets()
        self._sync_serper_credits()

        # Initialize database
        db_path = self.config.get("system", {}).get("database_file", "strat_os.db")
        self.db = get_database(db_path)

        # Initialize components
        self.market_fetcher = MarketFetcher(self.config.get("market", {}))
        self.news_fetcher = NewsFetcher(self.config)  # Pass full config for Kuwait intel
        self.discovery = EntityDiscovery(self.config.get("discovery", {}), self.db)
        self.scorer = _create_scorer(self.config, db=self.db)  # Pick scorer based on profile
        self.briefing_gen = BriefingGenerator(self.config)  # Pass full config for dynamic prompts

        # Output path (base — per-profile paths derived via _get_output_path)
        self._output_base = Path(self.config.get("system", {}).get("output_file", "output/news_data.json"))
        self._output_base.parent.mkdir(parents=True, exist_ok=True)

        # Background scheduler
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_scheduler = threading.Event()

        # Scan cancellation
        self._scan_cancelled = threading.Event()

        # Scan status tracking
        self.scan_status = {
            "is_scanning": False,
            "stage": "idle",
            "progress": "",
            "scored": 0,
            "total": 0,
            "high": 0,
            "medium": 0,
            "cancelled": False,
            "last_completed": None,
            "data_version": self._get_data_version()
        }
        self._scan_count = 0

        # SSE (Server-Sent Events) — push updates to connected dashboards
        self.sse = SSEManager()

        # Ollama lock — serializes LLM calls so shadow scoring doesn't
        # overlap with primary inference (single-GPU, one model at a time)
        self._ollama_lock = threading.Lock()

        # Scan lock — prevents concurrent scans from overlapping
        self._scan_lock = threading.Lock()

        # Multi-profile session isolation (A2.1)
        self.active_profile = None           # Currently loaded profile name
        self.active_profile_id = 0           # DB profile ID (0 = legacy/unset)
        self._profile_configs = {}           # name -> config snapshot
        self._config_lock = threading.Lock() # Serialize profile switches

        # Deferred briefing thread tracking (BUG-2 fix)
        self._briefing_thread = None
        self._briefing_generation = 0
        self._briefing_done = threading.Event()
        self._briefing_done.set()  # Initially "done" (no pending briefing)

        # Server start time for uptime tracking
        self._server_start_time = time.time()

        logger.info("STRAT_OS initialized")

    # ═══ SSE Push (delegates to SSEManager) ═══════════════════════

    def sse_register(self, wfile):
        """Register a new SSE client connection."""
        self.sse.register(wfile)

    def sse_unregister(self, wfile):
        """Remove a disconnected SSE client."""
        self.sse.unregister(wfile)

    def sse_broadcast(self, event_type: str, data: dict = None):
        """Push an event to all connected SSE clients."""
        self.sse.broadcast(event_type, data)

    # ═══ Profile Isolation (A2.1) ═════════════════════════════════

    def ensure_profile(self, profile_name: str):
        """Ensure the given profile's config is active.

        If a different profile is currently loaded, switch to the
        requested one from the in-memory cache.  This prevents
        a second login from contaminating the first user's config.
        """
        if not profile_name or self.active_profile == profile_name:
            return
        if profile_name not in self._profile_configs:
            return  # Not cached yet — will be loaded on next login
        with self._config_lock:
            if self.active_profile == profile_name:
                return  # Another thread already switched
            self.config = copy.deepcopy(self._profile_configs[profile_name])
            self.active_profile = profile_name
            logger.info(f"Profile switched to: {profile_name}")

    def cache_profile_config(self, profile_name: str):
        """Snapshot current config into the profile cache."""
        safe = profile_name.strip()
        if safe:
            self._profile_configs[safe] = copy.deepcopy(self.config)
            self.active_profile = safe

    # ═══════════════════════════════════════════════════════════════

    def _get_output_path(self, profile_name=None):
        """Get output file path for the given profile (or active profile)."""
        name = profile_name or getattr(self, 'active_profile', None)
        if name:
            return self._output_base.parent / f"{self._output_base.stem}_{name}{self._output_base.suffix}"
        return self._output_base

    @property
    def output_file(self):
        """Output file for the active profile."""
        return self._get_output_path()

    def _get_data_version(self) -> str:
        """Get version hash of current data file."""
        path = self.output_file
        if path.exists():
            stat = path.stat()
            return f"{stat.st_mtime:.0f}"
        return "0"

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_env_secrets(self):
        """Load secrets from .env file and environment variables.

        Environment variables take priority over config.yaml values.
        This allows secrets to live in .env (gitignored) instead of config.yaml (tracked).
        """
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, _, value = line.partition('=')
                            os.environ.setdefault(key.strip(), value.strip())
            except Exception as e:
                logger.warning(f"Failed to read .env: {e}")

        search = self.config.setdefault('search', {})
        if os.environ.get('SERPER_API_KEY'):
            search['serper_api_key'] = os.environ['SERPER_API_KEY']
        if os.environ.get('GOOGLE_API_KEY'):
            search['google_api_key'] = os.environ['GOOGLE_API_KEY']
        if os.environ.get('GOOGLE_CSE_ID'):
            search['google_cx'] = os.environ['GOOGLE_CSE_ID']

    def _sync_serper_credits(self):
        """Sync config.yaml serper_credits with the actual query tracker count."""
        try:
            from fetchers.serper_search import SerperQueryTracker
            tracker = SerperQueryTracker()
            status = tracker.get_status()
            actual_remaining = status.get('remaining', 0)
            config_remaining = self.config.get('search', {}).get('serper_credits', 0)
            if config_remaining != actual_remaining:
                self.config.setdefault('search', {})['serper_credits'] = actual_remaining
                # Persist to config.yaml
                import yaml
                with open(self.config_path, 'r') as f:
                    raw = yaml.safe_load(f)
                raw.setdefault('search', {})['serper_credits'] = actual_remaining
                with open(self.config_path, 'w') as f:
                    yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
                logger.info(f"Serper credits synced: {config_remaining} -> {actual_remaining} (tracker)")
        except Exception as e:
            logger.debug(f"Could not sync Serper credits: {e}")

    def run_scan(self) -> Dict[str, Any]:
        """
        Run a complete intelligence scan.

        This is the main pipeline:
        1. Fetch market data
        2. Fetch news
        3. Score news items
        4. Run entity discovery
        5. Generate briefing
        6. Save to database
        7. Export to JSON

        Returns:
            The complete output data
        """
        if not self._scan_lock.acquire(blocking=False):
            logger.warning("Scan already in progress, skipping")
            return {}
        try:
            return self._run_scan_impl()
        finally:
            self._scan_lock.release()

    def _run_scan_impl(self) -> Dict[str, Any]:
        """Internal scan implementation (called with _scan_lock held)."""
        self._scan_cancelled.clear()
        self._snapshot_previous_articles()  # Snapshot before any writes
        self.scan_status["is_scanning"] = True
        self.scan_status["stage"] = "starting"
        self.scan_status["progress"] = "Reloading configuration..."
        self.scan_status["scored"] = 0
        self.scan_status["total"] = 0
        self.scan_status["high"] = 0
        self.scan_status["medium"] = 0
        self.scan_status["cancelled"] = False
        self.sse_broadcast("scan", {"status": "starting", "progress": "Reloading configuration..."})

        # Reload config — use cached profile config if a profile is active,
        # otherwise fall back to config.yaml (A2.1 profile isolation)
        logger.info("Reloading configuration...")
        if self.active_profile and self.active_profile in self._profile_configs:
            self.config = copy.deepcopy(self._profile_configs[self.active_profile])
        else:
            self.config = self._load_config()
        # Always re-apply .env secrets — config.yaml has ${VAR} placeholders
        # that YAML doesn't expand, and profile cache may have stale keys
        self._load_env_secrets()

        # Reinitialize fetchers with new config
        self.market_fetcher = MarketFetcher(self.config.get("market", {}))
        self.news_fetcher = NewsFetcher(self.config)  # Pass full config for search settings
        self.scorer = _create_scorer(self.config, db=self.db)  # Rebuild scorer with latest profile
        self.briefing_gen = BriefingGenerator(self.config)  # Rebuild briefing with new profile/context

        logger.info("=" * 60)
        logger.info("Starting intelligence scan...")
        start_time = time.time()

        try:
            # 1. Fetch market data
            self.scan_status["stage"] = "market"
            self.scan_status["progress"] = "Fetching market data..."
            self.sse_broadcast("scan", {"status": "market", "progress": "Fetching market data..."})
            logger.info("[1/6] Fetching market data...")
            cache_ttl = self.config.get("cache", {}).get("market_ttl_seconds", 60)
            market_data, market_alerts = self.market_fetcher.fetch_all(cache_ttl_seconds=cache_ttl)

            # Save market snapshots to DB
            for symbol, data in market_data.items():
                if "error" not in data:
                    for interval, interval_data in data.get("data", {}).items():
                        self.db.save_market_snapshot(symbol, data["name"], interval, interval_data)

            # 2. Fetch news
            self.scan_status["stage"] = "news"
            self.scan_status["progress"] = "Fetching news..."
            self.sse_broadcast("scan", {"status": "news", "progress": "Fetching news..."})
            logger.info("[2/6] Fetching news...")
            cache_ttl = self.config.get("cache", {}).get("news_ttl_seconds", 900)
            news_items = self.news_fetcher.fetch_all(cache_ttl_seconds=cache_ttl)
            news_dicts = [item.to_dict() for item in news_items]

            # 2.5 Re-classify items into dynamic categories
            # RSS items arrive with generic category='tech'/'general'. Re-tag them
            # with dynamic category IDs so the scorer routes them correctly and
            # the frontend filters them into the right tabs.
            news_dicts = self._reclassify_dynamic(news_dicts)

            # 2.7 Incremental scan — reuse scores from previous scan if context unchanged
            news_dicts, prescore_reused = self._reuse_snapshot_scores(news_dicts)

            # 3. Score news items (only those that need fresh scoring)
            self.scan_status["stage"] = "scoring"
            total_items = len(news_dicts)
            self.scan_status["total"] = total_items + len(prescore_reused)
            self.scan_status["progress"] = f"Scoring 0/{total_items} items with AI..."
            self.sse_broadcast("scan", {"status": "scoring", "progress": f"Scoring {total_items} items..."})
            logger.info("[3/6] Scoring news items with AI...")

            # Score with progress callback
            def on_score_progress(current, total):
                self.scan_status["scored"] = current
                self.scan_status["total"] = total
                self.scan_status["progress"] = f"Scoring {current}/{total} items with AI..."
                if current % 5 == 0 or current == total:
                    self.sse_broadcast("scan", {"status": "scoring", "progress": f"Scoring {current}/{total} items...", "scored": current, "total": total})

            # === TWO-PASS SCORING ===
            timeout_cfg = self.config.get("scoring", {}).get("timeout", {})
            fallback_score = self.config.get("scoring", {}).get("fallback_score", 3.0)
            fast_buffer = timeout_cfg.get("fast_buffer", 30)
            fast_minimum = timeout_cfg.get("fast_minimum", 45)
            pass1_timeout = self.scorer.scoring_timer.fast_timeout(buffer=fast_buffer, minimum=fast_minimum)

            logger.info(f"[Pass 1] Scoring {total_items} items (timeout={pass1_timeout:.0f}s, "
                        f"avg={self.scorer.scoring_timer.rolling_avg:.1f}s)")

            scored_items, deferred_items = self.scorer.score_items(
                news_dicts,
                progress_callback=on_score_progress,
                cancel_check=lambda: self._scan_cancelled.is_set(),
                timeout_seconds=pass1_timeout
            )

            logger.info(f"[Pass 1] Complete: {len(scored_items)} scored, {len(deferred_items)} deferred")

            # Update score breakdown in scan_status
            _high = sum(1 for i in scored_items if i.get('score', 0) >= 7.0)
            _med = sum(1 for i in scored_items if 5.0 < i.get('score', 0) < 7.0)
            self.scan_status["high"] = _high
            self.scan_status["medium"] = _med

            # Save scored items to DB (even on cancel — partial results are valuable)
            _pid = self.active_profile_id
            for item in scored_items:
                self.db.save_news_item(item, profile_id=_pid)

            # Write Pass 1 results immediately so dashboard can display them
            if deferred_items:
                partial_output = self._build_output(market_data, scored_items, {}, market_alerts=market_alerts)
                self._write_output(partial_output)
                self.scan_status["data_version"] = self._get_data_version()
                self.sse_broadcast("pass1_complete", {
                    "scored": len(scored_items),
                    "deferred": len(deferred_items),
                    "high": _high,
                    "medium": _med,
                    "data_version": self.scan_status["data_version"]
                })

            # Handle cancellation — save partial results and return early
            if self._scan_cancelled.is_set():
                logger.info(f"Scan cancelled after scoring {len(scored_items)}/{total_items} items")
                if not deferred_items:
                    output = self._build_output(market_data, scored_items, {}, market_alerts=market_alerts)
                    self._write_output(output)
                self.scan_status["is_scanning"] = False
                self.scan_status["stage"] = "cancelled"
                self.scan_status["cancelled"] = True
                self.scan_status["progress"] = ""
                self.scan_status["data_version"] = self._get_data_version()
                self.sse_broadcast("scan_cancelled", {
                    "scored": len(scored_items),
                    "total": total_items,
                    "high": _high,
                    "medium": _med,
                    "data_version": self.scan_status["data_version"]
                })
                return self._build_output(market_data, scored_items, {}, market_alerts=market_alerts)

            # === PASS 2: Retry deferred articles with longer timeout ===
            if deferred_items and not self._scan_cancelled.is_set():
                slow_mult = timeout_cfg.get("slow_multiplier", 3)
                slow_buf = timeout_cfg.get("slow_buffer", 60)
                pass2_timeout = self.scorer.scoring_timer.slow_timeout(multiplier=slow_mult, buffer=slow_buf)

                logger.info(f"[Pass 2] Retrying {len(deferred_items)} deferred items (timeout={pass2_timeout:.0f}s)")
                self.scan_status["stage"] = "scoring_pass2"
                self.scan_status["progress"] = f"Pass 2: Retrying {len(deferred_items)} deferred items..."
                self.sse_broadcast("scan", {
                    "status": "scoring_pass2",
                    "progress": f"Pass 2: {len(deferred_items)} deferred items...",
                    "scored": self.scan_status["scored"],
                    "total": total_items
                })

                pass2_scored, still_deferred = self.scorer.score_items(
                    deferred_items,
                    progress_callback=lambda cur, tot: self.sse_broadcast("scan", {
                        "status": "scoring_pass2",
                        "progress": f"Pass 2: {cur}/{tot} deferred...",
                        "scored": self.scan_status["scored"],
                        "total": total_items
                    }),
                    cancel_check=lambda: self._scan_cancelled.is_set(),
                    timeout_seconds=pass2_timeout
                )

                # Permanently timed-out items get fallback score
                for item in still_deferred:
                    item['score'] = fallback_score
                    item['score_reason'] = "Scoring timed out — content may require manual review"
                    item['timeout'] = True
                    pass2_scored.append(item)
                    logger.warning(f"[TIMEOUT x2] \"{item.get('title', '')[:60]}\" — fallback {fallback_score}")

                recovered = len(pass2_scored) - len(still_deferred)
                logger.info(f"[Pass 2] Complete: {recovered} recovered, {len(still_deferred)} timed out permanently")

                # Merge Pass 2 results into scored_items by URL
                pass2_map = {item.get('url', ''): item for item in pass2_scored if item.get('url')}
                for i, item in enumerate(scored_items):
                    url = item.get('url', '')
                    if url and url in pass2_map:
                        scored_items[i] = pass2_map[url]

                # Save Pass 2 items to DB
                for item in pass2_scored:
                    self.db.save_news_item(item, profile_id=_pid)

                # Re-sort and update counts
                scored_items.sort(key=lambda x: x.get('score', 0), reverse=True)
                _high = sum(1 for i in scored_items if i.get('score', 0) >= 7.0)
                _med = sum(1 for i in scored_items if 5.0 < i.get('score', 0) < 7.0)
                self.scan_status["high"] = _high
                self.scan_status["medium"] = _med

            # Merge back pre-scored (reused) items
            if prescore_reused:
                scored_items.extend(prescore_reused)
                scored_items.sort(key=lambda x: x.get('score', 0), reverse=True)
                _high = sum(1 for i in scored_items if i.get('score', 0) >= 7.0)
                _med = sum(1 for i in scored_items if 5.0 < i.get('score', 0) < 7.0)
                self.scan_status["high"] = _high
                self.scan_status["medium"] = _med

            # 4. Run entity discovery
            self.scan_status["stage"] = "discovery"
            self.scan_status["progress"] = "Running entity discovery..."
            self.sse_broadcast("scan", {"status": "discovery", "progress": "Running entity discovery..."})
            logger.info("[4/6] Running entity discovery...")
            discoveries = self.discovery.discover(scored_items)

            # 5. Build output WITHOUT briefing — let the user see results immediately
            self.scan_status["stage"] = "output"
            self.scan_status["progress"] = "Building output..."
            logger.info("[5/6] Building output (briefing deferred)...")
            output = self._build_output(market_data, scored_items, {}, market_alerts=market_alerts)
            self._write_output(output)

            # 6. Spawn briefing in background — non-blocking
            self._spawn_deferred_briefing(market_data, market_alerts, scored_items, discoveries)

            elapsed = time.time() - start_time
            logger.info(f"Scan complete in {elapsed:.1f}s")
            logger.info("=" * 60)

            # Save scan log entry
            critical = sum(1 for i in scored_items if i.get('score', 0) >= 9.0)
            high = sum(1 for i in scored_items if 7.0 <= i.get('score', 0) < 9.0)
            medium = sum(1 for i in scored_items if 5.0 < i.get('score', 0) < 7.0)
            noise = sum(1 for i in scored_items if i.get('score', 0) <= 5.0)
            retained_count = getattr(self, '_last_retained_count', 0)
            scan_id = self.db.save_scan_log({
                'started_at': datetime.fromtimestamp(start_time).isoformat(),
                'elapsed_secs': round(elapsed, 1),
                'items_fetched': len(news_dicts),
                'items_scored': len(scored_items),
                'critical': critical, 'high': high, 'medium': medium, 'noise': noise,
                'rule_scored': self.scorer._stats.get('rule', 0),
                'llm_scored': self.scorer._stats.get('llm', 0),
                'truncated': self.scorer._stats.get('truncated', 0),
                'retained': retained_count,
            }, profile_id=_pid)

            # Update status
            self.scan_status["is_scanning"] = False
            self.scan_status["stage"] = "complete"
            self.scan_status["progress"] = ""
            self.scan_status["last_completed"] = datetime.now().isoformat()
            self.scan_status["data_version"] = self._get_data_version()
            self.sse_broadcast("complete", {"data_version": self.scan_status["data_version"]})

            # Broadcast critical signals for push notifications
            for item in scored_items:
                if item.get('score', 0) >= 9.0:
                    self.sse_broadcast("critical_signal", {
                        "title": item.get('title', ''),
                        "score": item.get('score', 0),
                        "reason": item.get('score_reason', ''),
                        "url": item.get('url', ''),
                        "category": item.get('category', ''),
                    })

            # Shadow scoring — validates AdaptiveScorer vs AIScorer (daemon thread)
            self._run_shadow_scoring(scored_items, scan_id)

            # Periodic DB cleanup — every 10th scan
            self._scan_count += 1
            if self._scan_count % 10 == 0:
                logger.info(f"Running periodic DB cleanup (scan #{self._scan_count})...")
                self.db.cleanup_old_data(days=30, profile_id=_pid)

            # Auto-distillation — every N scans (if API key configured)
            distill_every = self.config.get("distillation", {}).get("auto_every", 0)
            if distill_every > 0 and self._scan_count % distill_every == 0:
                self._run_auto_distillation()

            return output

        except Exception as e:
            logger.error(f"Scan failed: {e}")
            elapsed = time.time() - start_time
            self.db.save_scan_log({
                'started_at': datetime.fromtimestamp(start_time).isoformat(),
                'elapsed_secs': round(elapsed, 1),
                'error': str(e),
                'truncated': getattr(self.scorer, '_stats', {}).get('truncated', 0),
            }, profile_id=self.active_profile_id)
            self.scan_status["is_scanning"] = False
            self.scan_status["stage"] = "error"
            self.scan_status["progress"] = str(e)
            self.sse_broadcast("scan_error", {"message": str(e)})
            raise

    def _run_shadow_scoring(self, scored_items, scan_id):
        """Run shadow scoring in a background thread.

        B3.3: With AIScorer retired, shadow scoring creates a second
        AdaptiveScorer instance for self-consistency checks.
        """
        shadow_enabled = self.config.get('scoring', {}).get('shadow_scoring', False)
        if not shadow_enabled:
            return

        def _shadow_worker():
            try:
                primary_name = 'AdaptiveScorer'
                shadow = AdaptiveScorer(self.config, db=self.db)
                shadow_name = 'AdaptiveScorer-shadow'

                # Sample every 5th item, max 20
                sampled = scored_items[::5][:20]
                if not sampled:
                    return

                # Acquire Ollama lock — wait for any active inference to finish
                # before sending shadow requests (single GPU, one model at a time)
                logger.info(f"Shadow scoring: waiting for Ollama lock ({len(sampled)} items with {shadow_name})...")
                with self._ollama_lock:
                    logger.info(f"Shadow scoring: lock acquired, scoring {len(sampled)} items")
                    for item in sampled:
                        try:
                            primary_score = item.get('score', 0)
                            shadow_score, shadow_reason = shadow.score_item(item)
                            delta = shadow_score - primary_score

                            self.db.save_shadow_score({
                                'scan_id': scan_id,
                                'news_id': item.get('id', ''),
                                'title': item.get('title', ''),
                                'category': item.get('category', ''),
                                'primary_scorer': primary_name,
                                'primary_score': primary_score,
                                'shadow_scorer': shadow_name,
                                'shadow_score': shadow_score,
                                'delta': delta,
                            }, profile_id=self.active_profile_id)
                        except Exception as e:
                            logger.debug(f"Shadow score failed for item: {e}")

                logger.info(f"Shadow scoring complete: {len(sampled)} items compared")

            except Exception as e:
                logger.warning(f"Shadow scoring failed: {e}")

        thread = threading.Thread(target=_shadow_worker, daemon=True)
        thread.start()

    def _run_auto_distillation(self):
        """Run distillation in background thread after a scan completes."""
        import threading

        def _distill_worker():
            try:
                from distill import get_api_key, run_distillation
                from pathlib import Path

                api_key = get_api_key()
                if not api_key:
                    logger.info("Auto-distillation skipped: no API key configured")
                    return

                config_path = str(Path(__file__).parent / "config.yaml")
                db_path = str(Path(__file__).parent / "strat_os.db")

                distill_cfg = self.config.get("distillation", {})
                hours = distill_cfg.get("hours", 168)
                limit = distill_cfg.get("limit", 200)
                threshold = distill_cfg.get("threshold", 2.0)

                logger.info(f"Auto-distillation starting ({limit} items, last {hours}h)...")
                run_distillation(
                    db_path=db_path,
                    config_path=config_path,
                    api_key=api_key,
                    hours=hours,
                    limit=limit,
                    threshold=threshold,
                    dry_run=False
                )
                logger.info("Auto-distillation completed")

            except Exception as e:
                logger.warning(f"Auto-distillation failed: {e}")

        thread = threading.Thread(target=_distill_worker, daemon=True)
        thread.start()

    def run_market_refresh(self) -> Dict[str, Any]:
        """
        Refresh only market data (fast, no API calls).
        Updates the existing output file with fresh market data.
        """
        # If a news scan is in progress, skip — news scan will include market data
        if self.scan_status.get("is_scanning") and self.scan_status.get("stage") not in ("market", "complete", "error"):
            logger.info("Skipping market refresh — news scan in progress")
            return {}

        self.scan_status["is_scanning"] = True
        self.scan_status["stage"] = "market"
        self.scan_status["progress"] = "Refreshing market data..."

        logger.info("=" * 40)
        logger.info("Refreshing market data only...")
        start_time = time.time()

        try:
            # Reload config (profile-aware — A2.1)
            if self.active_profile and self.active_profile in self._profile_configs:
                self.config = copy.deepcopy(self._profile_configs[self.active_profile])
            else:
                self.config = self._load_config()
            self.market_fetcher = MarketFetcher(self.config.get("market", {}))

            # Fetch fresh market data (bypass cache)
            market_data, market_alerts = self.market_fetcher.fetch_all(cache_ttl_seconds=0)

            # Save to DB
            for symbol, data in market_data.items():
                if "error" not in data:
                    for interval, interval_data in data.get("data", {}).items():
                        self.db.save_market_snapshot(symbol, data["name"], interval, interval_data)

            # Load existing output and update market section
            output = {}
            if self.output_file.exists():
                with open(self.output_file, "r") as f:
                    output = json.load(f)

            # Update market data
            market_output = {}
            for symbol, data in market_data.items():
                market_output[symbol] = {
                    "name": data.get("name", symbol),
                    "data": data.get("data", {})
                }
            output["market"] = market_output
            output["alerts"] = market_alerts or []
            output["last_updated"] = datetime.now().strftime("%b %d, %I:%M %p")

            # Update only market timestamp, preserve news timestamp
            if "timestamps" not in output:
                output["timestamps"] = {}
            output["timestamps"]["market"] = datetime.now().isoformat()

            # Write back
            self._write_output(output)

            elapsed = time.time() - start_time
            logger.info(f"Market refresh complete in {elapsed:.1f}s")
            logger.info("=" * 40)

            self.scan_status["is_scanning"] = False
            self.scan_status["stage"] = "complete"
            self.scan_status["progress"] = ""
            self.scan_status["data_version"] = self._get_data_version()
            self.sse_broadcast("complete", {"data_version": self.scan_status["data_version"]})

            return output

        except Exception as e:
            logger.error(f"Market refresh failed: {e}")
            self.scan_status["is_scanning"] = False
            self.scan_status["stage"] = "error"
            self.scan_status["progress"] = str(e)
            self.sse_broadcast("scan_error", {"message": str(e)})
            raise

    def run_news_refresh(self) -> Dict[str, Any]:
        """
        Refresh news, scoring, and briefing (slower, uses API calls).
        Keeps existing market data.
        """
        if not self._scan_lock.acquire(blocking=False):
            logger.warning("Scan already in progress, skipping news refresh")
            return {}
        try:
            return self._run_news_refresh_impl()
        finally:
            self._scan_lock.release()

    def _run_news_refresh_impl(self) -> Dict[str, Any]:
        """Internal news refresh implementation (called with _scan_lock held)."""
        self._scan_cancelled.clear()
        self._snapshot_previous_articles()  # Snapshot before any writes
        self.scan_status["is_scanning"] = True
        self.scan_status["stage"] = "news"
        self.scan_status["progress"] = "Fetching news..."
        self.scan_status["scored"] = 0
        self.scan_status["total"] = 0
        self.scan_status["high"] = 0
        self.scan_status["medium"] = 0
        self.scan_status["cancelled"] = False
        self.sse_broadcast("scan", {"status": "news", "progress": "Fetching news..."})

        logger.info("=" * 60)
        logger.info("Refreshing news and intelligence...")
        start_time = time.time()

        try:
            # Reload config (profile-aware — A2.1)
            if self.active_profile and self.active_profile in self._profile_configs:
                self.config = copy.deepcopy(self._profile_configs[self.active_profile])
            else:
                self.config = self._load_config()
            # Always re-apply .env secrets (config.yaml has ${VAR} placeholders)
            self._load_env_secrets()
            self.news_fetcher = NewsFetcher(self.config)
            self.scorer = _create_scorer(self.config, db=self.db)  # Rebuild scorer with latest profile
            self.briefing_gen = BriefingGenerator(self.config)  # Rebuild briefing with latest profile/context

            # Load existing market data
            market_data = {}
            market_alerts = []
            if self.output_file.exists():
                with open(self.output_file, "r") as f:
                    existing = json.load(f)
                    # Convert back to fetcher format
                    for symbol, data in existing.get("market", {}).items():
                        market_data[symbol] = {
                            "name": data.get("name", symbol),
                            "data": data.get("data", {})
                        }

            # Fetch news
            self.scan_status["progress"] = "Fetching news..."
            self.sse_broadcast("scan", {"status": "news", "progress": "Fetching news..."})
            logger.info("[1/4] Fetching news...")
            news_items = self.news_fetcher.fetch_all(cache_ttl_seconds=0)
            news_dicts = [item.to_dict() for item in news_items]

            # Re-classify into dynamic categories
            news_dicts = self._reclassify_dynamic(news_dicts)

            # Incremental scan — reuse scores from previous scan if context unchanged
            news_dicts, prescore_reused = self._reuse_snapshot_scores(news_dicts)

            # Score news
            self.scan_status["stage"] = "scoring"
            total_items = len(news_dicts)
            self.scan_status["total"] = total_items + len(prescore_reused)
            self.scan_status["progress"] = f"Scoring 0/{total_items} items with AI..."
            self.sse_broadcast("scan", {"status": "scoring", "progress": f"Scoring 0/{total_items} items...", "scored": 0, "total": total_items})
            logger.info("[2/4] Scoring news items with AI...")

            def on_score_progress(current, total):
                self.scan_status["scored"] = current
                self.scan_status["total"] = total
                self.scan_status["progress"] = f"Scoring {current}/{total} items with AI..."
                if current % 5 == 0 or current == total:
                    self.sse_broadcast("scan", {"status": "scoring", "progress": f"Scoring {current}/{total} items...", "scored": current, "total": total})

            # === TWO-PASS SCORING ===
            timeout_cfg = self.config.get("scoring", {}).get("timeout", {})
            fallback_score = self.config.get("scoring", {}).get("fallback_score", 3.0)
            fast_buffer = timeout_cfg.get("fast_buffer", 30)
            fast_minimum = timeout_cfg.get("fast_minimum", 45)
            pass1_timeout = self.scorer.scoring_timer.fast_timeout(buffer=fast_buffer, minimum=fast_minimum)

            logger.info(f"[Pass 1] Scoring {total_items} items (timeout={pass1_timeout:.0f}s, "
                        f"avg={self.scorer.scoring_timer.rolling_avg:.1f}s)")

            scored_items, deferred_items = self.scorer.score_items(
                news_dicts,
                progress_callback=on_score_progress,
                cancel_check=lambda: self._scan_cancelled.is_set(),
                timeout_seconds=pass1_timeout
            )

            logger.info(f"[Pass 1] Complete: {len(scored_items)} scored, {len(deferred_items)} deferred")

            # Update score breakdown
            _high = sum(1 for i in scored_items if i.get('score', 0) >= 7.0)
            _med = sum(1 for i in scored_items if 5.0 < i.get('score', 0) < 7.0)
            self.scan_status["high"] = _high
            self.scan_status["medium"] = _med

            # Save scored items to DB (even on cancel)
            _pid = self.active_profile_id
            for item in scored_items:
                self.db.save_news_item(item, profile_id=_pid)

            # Write Pass 1 results immediately so dashboard can display them
            if deferred_items:
                partial_output = self._build_output(market_data, scored_items, {}, market_alerts=market_alerts)
                self._write_output(partial_output)
                self.scan_status["data_version"] = self._get_data_version()
                self.sse_broadcast("pass1_complete", {
                    "scored": len(scored_items),
                    "deferred": len(deferred_items),
                    "high": _high,
                    "medium": _med,
                    "data_version": self.scan_status["data_version"]
                })

            # Handle cancellation
            if self._scan_cancelled.is_set():
                logger.info(f"News refresh cancelled after scoring {len(scored_items)}/{total_items} items")
                if not deferred_items:
                    output = self._build_output(market_data, scored_items, {}, market_alerts=market_alerts)
                    self._write_output(output)
                self.scan_status["is_scanning"] = False
                self.scan_status["stage"] = "cancelled"
                self.scan_status["cancelled"] = True
                self.scan_status["progress"] = ""
                self.scan_status["data_version"] = self._get_data_version()
                self.sse_broadcast("scan_cancelled", {
                    "scored": len(scored_items),
                    "total": total_items,
                    "high": _high,
                    "medium": _med,
                    "data_version": self.scan_status["data_version"]
                })
                return self._build_output(market_data, scored_items, {}, market_alerts=market_alerts)

            # === PASS 2: Retry deferred articles with longer timeout ===
            if deferred_items and not self._scan_cancelled.is_set():
                slow_mult = timeout_cfg.get("slow_multiplier", 3)
                slow_buf = timeout_cfg.get("slow_buffer", 60)
                pass2_timeout = self.scorer.scoring_timer.slow_timeout(multiplier=slow_mult, buffer=slow_buf)

                logger.info(f"[Pass 2] Retrying {len(deferred_items)} deferred items (timeout={pass2_timeout:.0f}s)")
                self.scan_status["stage"] = "scoring_pass2"
                self.scan_status["progress"] = f"Pass 2: Retrying {len(deferred_items)} deferred items..."
                self.sse_broadcast("scan", {
                    "status": "scoring_pass2",
                    "progress": f"Pass 2: {len(deferred_items)} deferred items...",
                    "scored": self.scan_status["scored"],
                    "total": total_items
                })

                pass2_scored, still_deferred = self.scorer.score_items(
                    deferred_items,
                    progress_callback=lambda cur, tot: self.sse_broadcast("scan", {
                        "status": "scoring_pass2",
                        "progress": f"Pass 2: {cur}/{tot} deferred...",
                        "scored": self.scan_status["scored"],
                        "total": total_items
                    }),
                    cancel_check=lambda: self._scan_cancelled.is_set(),
                    timeout_seconds=pass2_timeout
                )

                # Permanently timed-out items get fallback score
                for item in still_deferred:
                    item['score'] = fallback_score
                    item['score_reason'] = "Scoring timed out — content may require manual review"
                    item['timeout'] = True
                    pass2_scored.append(item)
                    logger.warning(f"[TIMEOUT x2] \"{item.get('title', '')[:60]}\" — fallback {fallback_score}")

                recovered = len(pass2_scored) - len(still_deferred)
                logger.info(f"[Pass 2] Complete: {recovered} recovered, {len(still_deferred)} timed out permanently")

                # Merge Pass 2 results into scored_items by URL
                pass2_map = {item.get('url', ''): item for item in pass2_scored if item.get('url')}
                for i, item in enumerate(scored_items):
                    url = item.get('url', '')
                    if url and url in pass2_map:
                        scored_items[i] = pass2_map[url]

                # Save Pass 2 items to DB
                for item in pass2_scored:
                    self.db.save_news_item(item, profile_id=_pid)

                # Re-sort and update counts
                scored_items.sort(key=lambda x: x.get('score', 0), reverse=True)
                _high = sum(1 for i in scored_items if i.get('score', 0) >= 7.0)
                _med = sum(1 for i in scored_items if 5.0 < i.get('score', 0) < 7.0)
                self.scan_status["high"] = _high
                self.scan_status["medium"] = _med

            # Merge back pre-scored (reused) items
            if prescore_reused:
                scored_items.extend(prescore_reused)
                scored_items.sort(key=lambda x: x.get('score', 0), reverse=True)
                _high = sum(1 for i in scored_items if i.get('score', 0) >= 7.0)
                _med = sum(1 for i in scored_items if 5.0 < i.get('score', 0) < 7.0)
                self.scan_status["high"] = _high
                self.scan_status["medium"] = _med

            # Run entity discovery
            self.scan_status["stage"] = "discovery"
            self.scan_status["progress"] = "Running entity discovery..."
            self.sse_broadcast("scan", {"status": "discovery", "progress": "Running entity discovery..."})
            logger.info("[3/4] Running entity discovery...")
            discoveries = self.discovery.discover(scored_items)

            # Build output WITHOUT briefing — let the user see results immediately
            output = self._build_output(market_data, scored_items, {}, market_alerts=market_alerts)
            self._write_output(output)

            # Spawn briefing in background — non-blocking
            self._spawn_deferred_briefing(market_data, market_alerts, scored_items, discoveries)

            elapsed = time.time() - start_time
            logger.info(f"News refresh complete in {elapsed:.1f}s")
            logger.info("=" * 60)

            # Save scan log entry
            critical = sum(1 for i in scored_items if i.get('score', 0) >= 9.0)
            high = sum(1 for i in scored_items if 7.0 <= i.get('score', 0) < 9.0)
            medium = sum(1 for i in scored_items if 5.0 < i.get('score', 0) < 7.0)
            noise = sum(1 for i in scored_items if i.get('score', 0) <= 5.0)
            retained_count = getattr(self, '_last_retained_count', 0)
            scan_id = self.db.save_scan_log({
                'started_at': datetime.fromtimestamp(start_time).isoformat(),
                'elapsed_secs': round(elapsed, 1),
                'items_fetched': len(news_dicts),
                'items_scored': len(scored_items),
                'critical': critical, 'high': high, 'medium': medium, 'noise': noise,
                'rule_scored': self.scorer._stats.get('rule', 0),
                'llm_scored': self.scorer._stats.get('llm', 0),
                'truncated': self.scorer._stats.get('truncated', 0),
                'retained': retained_count,
            }, profile_id=_pid)

            self.scan_status["is_scanning"] = False
            self.scan_status["stage"] = "complete"
            self.scan_status["progress"] = ""
            self.scan_status["last_completed"] = datetime.now().isoformat()
            self.scan_status["data_version"] = self._get_data_version()
            self.sse_broadcast("complete", {"data_version": self.scan_status["data_version"]})

            # Broadcast critical signals for push notifications
            for item in scored_items:
                if item.get('score', 0) >= 9.0:
                    self.sse_broadcast("critical_signal", {
                        "title": item.get('title', ''),
                        "score": item.get('score', 0),
                        "reason": item.get('score_reason', ''),
                        "url": item.get('url', ''),
                        "category": item.get('category', ''),
                    })

            # Shadow scoring — validates AdaptiveScorer vs AIScorer (daemon thread)
            self._run_shadow_scoring(scored_items, scan_id)

            return output

        except Exception as e:
            logger.error(f"News refresh failed: {e}")
            self.scan_status["is_scanning"] = False
            self.scan_status["stage"] = "error"
            self.scan_status["progress"] = str(e)
            self.sse_broadcast("scan_error", {"message": str(e)})
            raise

    def _reclassify_dynamic(self, items: list) -> list:
        """Re-tag news items with dynamic category IDs.

        RSS/scraper items arrive with generic categories like 'tech' or 'general'.
        This re-classifies them so the scorer routes correctly and the frontend
        filters them into the right dynamic tabs.

        Matching strategy:
        - Short items (1-2 words, ≤12 chars): word-boundary regex for precision
        - Longer items (3+ words): match if ALL significant words appear in text
        - Single long words: substring match (safe for words > 4 chars)
        """
        dynamic_cats = self.config.get('dynamic_categories', [])
        if not dynamic_cats:
            return items

        import re

        STOP_WORDS = {'and', 'or', 'the', 'a', 'an', 'in', 'of', 'for', 'to', 'at', 'on', 'by', 'with'}

        # Pre-compile matchers: [(cat_id, cat_root, [matcher_func])]
        cat_matchers = []
        for cat in dynamic_cats:
            if cat.get('enabled') is False:
                continue
            cat_id = cat.get('id', '')
            cat_root = cat.get('root', 'kuwait')
            matchers = []
            for kw in cat.get('items', []):
                kw_lower = kw.lower().strip()
                if not kw_lower:
                    continue
                words = [w for w in kw_lower.split() if w not in STOP_WORDS]

                if len(words) <= 1 and len(kw_lower) <= 4:
                    # Short abbreviation: word-boundary regex (KOC, SLB, etc.)
                    try:
                        pattern = re.compile(r'\b' + re.escape(kw_lower) + r'\b', re.IGNORECASE)
                        matchers.append(lambda text, p=pattern: bool(p.search(text)))
                    except re.error:
                        pass
                elif len(words) <= 2:
                    # 1-2 significant words: substring match (safe for longer terms)
                    matchers.append(lambda text, k=kw_lower: k in text)
                else:
                    # 3+ words: match if ALL significant words appear
                    matchers.append(lambda text, ws=words: all(w in text for w in ws))

            if matchers:
                cat_matchers.append((cat_id, cat_root, matchers))

        if not cat_matchers:
            return items

        for item in items:
            current_cat = item.get('category', '')
            if any(current_cat == cm[0] for cm in cat_matchers):
                continue

            text = ((item.get('title', '') or '') + ' ' + (item.get('summary', '') or '')).lower()

            best_match = None
            best_score = 0

            for cat_id, cat_root, matchers in cat_matchers:
                match_count = sum(1 for m in matchers if m(text))
                if match_count > best_score:
                    best_score = match_count
                    best_match = (cat_id, cat_root)

            if best_match and best_score >= 1:
                item['category'] = best_match[0]
                if item.get('root', 'global') in ('global', 'general'):
                    item['root'] = best_match[1]

        return items

    def _reuse_snapshot_scores(self, news_dicts: list) -> tuple:
        """Reuse scores from the previous scan's snapshot for already-seen URLs.

        Returns (items_needing_scoring, reused_items).
        Reuse is skipped entirely if the context hash changed since the last scan.
        """
        snapshot = getattr(self, '_retained_snapshot', None)
        if not snapshot:
            return news_dicts, []

        # Check context hash — if profile/role changed, rescore everything
        current_hash = self._get_context_hash(
            self.config.get("profile", {}).get("role", ""),
            self.config.get("profile", {}).get("context", ""),
            self.config.get("profile", {}).get("location", ""),
        )
        prev_hash = getattr(self, '_snapshot_context_hash', "")

        if not prev_hash or prev_hash != current_hash:
            if prev_hash:
                logger.info(f"Incremental scan: context changed ({prev_hash[:6]}→{current_hash[:6]}), rescoring all")
            return news_dicts, []

        # Build URL → scored article lookup from snapshot
        snapshot_map = {}
        for article in snapshot:
            url = article.get("url", "")
            if url and article.get("score") is not None:
                snapshot_map[url] = article

        need_scoring = []
        reused = []
        fetched_urls = {item.get("url", "") for item in news_dicts if item.get("url")}

        for item in news_dicts:
            url = item.get("url", "")
            if url and url in snapshot_map:
                prev = snapshot_map[url]
                # Carry over score + score_reason from previous scan
                item["score"] = prev["score"]
                item["score_reason"] = prev.get("score_reason", "")
                reused.append(item)
            else:
                need_scoring.append(item)

        # Carry forward snapshot articles that weren't re-fetched.
        # They already have scores — no reason to drop them.
        # Exclude retained articles (handled separately by _merge_retained_articles).
        carried = 0
        for url, article in snapshot_map.items():
            if url not in fetched_urls and not article.get("retained"):
                reused.append(article)
                carried += 1

        if reused or carried:
            logger.info(f"Incremental scan: {len(reused) - carried} matched, {carried} carried forward, "
                        f"{len(need_scoring)} new (context={current_hash[:6]})")

        return need_scoring, reused

    def _spawn_deferred_briefing(self, market_data, market_alerts, scored_items, discoveries):
        """Generate briefing in a background thread and patch the output when done."""
        self._briefing_generation += 1
        current_gen = self._briefing_generation
        self._briefing_done.clear()

        def _briefing_worker():
            try:
                if self._scan_cancelled.is_set():
                    return
                logger.info("Deferred briefing: generating...")
                self.sse_broadcast("scan", {"status": "briefing", "progress": "Generating briefing..."})
                briefing = self.briefing_gen.generate_briefing(
                    market_data=market_data,
                    market_alerts=market_alerts,
                    news_items=scored_items,
                    discoveries=discoveries
                )
                if self._scan_cancelled.is_set():
                    return
                # Save to DB
                self.db.save_briefing(briefing, profile_id=self.active_profile_id)
                # Patch the output file with the briefing
                if self.output_file.exists():
                    with open(self.output_file, "r") as f:
                        output = json.load(f)
                    output["briefing"] = briefing
                    output["meta"]["critical_count"] = briefing.get("critical_count", 0)
                    output["meta"]["high_count"] = briefing.get("high_count", 0)
                    self._write_output(output)
                self.sse_broadcast("briefing_ready", {"status": "ready"})
                logger.info("Deferred briefing: complete, output patched")
            except Exception as e:
                logger.warning(f"Deferred briefing failed: {e}")
                self.sse_broadcast("briefing_ready", {"status": "failed"})
            finally:
                # Signal completion only if this is still the current generation
                # (prevents stale thread A from signaling thread B's wait)
                if current_gen == self._briefing_generation:
                    self._briefing_done.set()

        self._briefing_thread = threading.Thread(target=_briefing_worker, daemon=True, name="briefing-deferred")
        self._briefing_thread.start()

    def wait_for_briefing(self, timeout: float = 30) -> bool:
        """Wait for the deferred briefing to complete.
        Returns True if briefing finished, False if timed out."""
        return self._briefing_done.wait(timeout=timeout)

    def _snapshot_previous_articles(self):
        """Snapshot previous feed articles at scan start for retention merge.
        Call this once at the beginning of run_scan()/run_news_refresh() so that
        partial writes during the scan don't corrupt the retention source.
        Also captures meta.context_hash for incremental score reuse."""
        try:
            if self.output_file.exists():
                with open(self.output_file, "r") as f:
                    prev_data = json.load(f)
                    self._retained_snapshot = prev_data.get("news", [])
                    self._snapshot_context_hash = prev_data.get("meta", {}).get("context_hash", "")
            else:
                self._retained_snapshot = []
                self._snapshot_context_hash = ""
        except Exception as e:
            logger.warning(f"Could not snapshot previous feed for retention: {e}")
            self._retained_snapshot = []
            self._snapshot_context_hash = ""

    @staticmethod
    def _get_context_hash(role: str, context: str, location: str) -> str:
        """Hash the full intelligence context into a stable 12-char identifier."""
        parts = []
        for text in [role, context, location]:
            normalized = " ".join(text.lower().split()) if text else ""
            parts.append(normalized)
        combined = "|".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:12]

    def _merge_retained_articles(self, current_articles: list) -> list:
        """Merge high-scoring articles from previous scan into current results."""
        scoring_cfg = self.config.get("scoring", {})
        if not scoring_cfg.get("retain_high_scores", True):
            return current_articles

        threshold = scoring_cfg.get("retention_threshold", 8.0)
        max_age_hours = scoring_cfg.get("retention_max_age_hours", 24)
        max_retained = scoring_cfg.get("retention_max_items", 20)

        # Use snapshot taken at scan start (avoids reading partially-written output file)
        previous_articles = getattr(self, '_retained_snapshot', None)
        if previous_articles is None:
            # Fallback: read from file if no snapshot (shouldn't happen normally)
            try:
                if self.output_file.exists():
                    with open(self.output_file, "r") as f:
                        prev_data = json.load(f)
                        previous_articles = prev_data.get("news", [])
                else:
                    previous_articles = []
            except Exception as e:
                logger.warning(f"Could not load previous feed for retention: {e}")
                return current_articles

        if not previous_articles:
            return current_articles

        current_urls = {a.get("url", "") for a in current_articles if a.get("url")}
        cutoff = datetime.now() - timedelta(hours=max_age_hours)

        # Profile-scoped retention: hash role+context+location as the context key
        profile_cfg = self.config.get("profile", {})
        context_hash = self._get_context_hash(
            profile_cfg.get("role", ""),
            profile_cfg.get("context", ""),
            profile_cfg.get("location", ""),
        )

        retained = []
        for article in previous_articles:
            url = article.get("url", "")
            if not url or url in current_urls:
                continue
            if article.get("score", 0) < threshold:
                continue
            # Check age
            ts = article.get("timestamp", "")
            if ts:
                try:
                    article_time = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
                    if article_time < cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
            # Check if user dismissed this article
            if self.db and self.db.was_dismissed(url, profile_id=self.active_profile_id):
                continue
            # Profile filter: skip articles retained by a different context
            article_profile = article.get("retained_by_profile", "")
            if article_profile and context_hash and article_profile != context_hash:
                continue
            article["retained"] = True
            article["retained_by_profile"] = context_hash
            retained.append(article)

        # Cap retained articles — keep highest scoring
        retained.sort(key=lambda x: x.get("score", 0), reverse=True)
        retained = retained[:max_retained]

        if retained:
            logger.info(f"Retained: {len(retained)} articles from previous scans (>= {threshold})")

        # Store count for scan log
        self._last_retained_count = len(retained)

        return current_articles + retained

    def _build_output(self, market_data: Dict, news_items: list, briefing: Dict, market_alerts: list = None) -> Dict[str, Any]:
        """
        Build the final output matching FRS schema with extensions.

        FRS Schema:
        {
            "market": { ... },
            "news": [ ... ],
            "last_updated": "..."
        }

        Extended with:
        {
            "briefing": { ... }
        }
        """
        # Merge retained high-scoring articles from previous scan
        news_items = self._merge_retained_articles(news_items)
        # Re-sort by score after merging
        news_items.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Pre-compute context hash once for tagging all articles
        profile_cfg = self.config.get("profile", {})
        context_hash = self._get_context_hash(
            profile_cfg.get("role", ""),
            profile_cfg.get("context", ""),
            profile_cfg.get("location", ""),
        )

        # Filter news for output (respect max_items)
        max_items = self.config.get("system", {}).get("max_news_items", 50)
        filtered_news = news_items[:max_items]

        # Build FRS-compliant news array
        news_output = []
        for item in filtered_news:
            news_output.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "summary": item.get("summary", ""),
                "content": item.get("content", ""),
                "timestamp": item.get("timestamp", ""),
                "source": item.get("source", ""),
                "root": item.get("root", "global"),
                "score": item.get("score", 0.0),
                # Extensions
                "id": item.get("id", ""),
                "category": item.get("category", ""),
                "score_reason": item.get("score_reason", ""),
                "retained_by_profile": context_hash,
                **({"retained": True} if item.get("retained") else {})
            })

        # Build market output (FRS compliant)
        market_output = {}
        for symbol, data in market_data.items():
            market_output[symbol] = {
                "name": data.get("name", symbol),
                "data": data.get("data", {})
            }

        return {
            # FRS required fields
            "market": market_output,
            "news": news_output,
            "last_updated": datetime.now().strftime("%b %d, %I:%M %p"),

            # Extended fields
            "alerts": market_alerts or [],
            "briefing": briefing,
            "timestamps": {
                "market": datetime.now().isoformat(),
                "news": datetime.now().isoformat()
            },
            "meta": {
                "version": "3.0",
                "generated_at": datetime.now().isoformat(),
                "news_count": len(news_output),
                "critical_count": briefing.get("critical_count", 0),
                "high_count": briefing.get("high_count", 0),
                "context_hash": context_hash
            }
        }

    def _write_output(self, data: Dict[str, Any]):
        """Write output to JSON file."""
        with open(self.output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Output written to {self.output_file}")

    def start_background_scheduler(self):
        """Start background refresh scheduler."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("Scheduler already running")
            return

        interval_min = self.config.get("schedule", {}).get("background_interval_minutes", 30)

        def scheduler_loop():
            while not self._stop_scheduler.is_set():
                try:
                    self.run_scan()
                except Exception as e:
                    logger.error(f"Scheduled scan failed: {e}")

                # Wait for interval or stop signal
                self._stop_scheduler.wait(timeout=interval_min * 60)

        self._stop_scheduler.clear()
        self._scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info(f"Background scheduler started (interval: {interval_min} min)")

    def stop_background_scheduler(self):
        """Stop the background scheduler."""
        if self._scheduler_thread:
            self._stop_scheduler.set()
            self._scheduler_thread.join(timeout=5)
            logger.info("Background scheduler stopped")

    def serve_frontend(self, port: int = 8080, open_browser: bool = True):
        """Start the HTTP server for the frontend."""
        from auth import AuthManager
        from server import start_server

        auth = AuthManager(self.config_path)

        # Log profile status
        initial_profiles = auth.list_profiles()
        if initial_profiles:
            logger.info(f"Security: {len(initial_profiles)} profile(s) registered")
        else:
            logger.info("Security: No profiles — registration required on first visit")

        start_server(self, auth, port, open_browser)

    def cleanup(self):
        """Cleanup resources."""
        self.stop_background_scheduler()
        self.db.cleanup_old_data(days=30)
        self.db.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="STRAT_OS - Strategic Intelligence Operating System")
    parser.add_argument("--scan", action="store_true", help="Run a single scan")
    parser.add_argument("--serve", action="store_true", help="Start the frontend server")
    parser.add_argument("--background", action="store_true", help="Enable background refresh")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")

    args = parser.parse_args()

    # Initialize STRAT_OS
    strat = StratOS(config_path=args.config)

    try:
        if args.serve:
            # Start server immediately (serve cached data)
            # User must click Refresh to fetch/score
            logger.info("Starting server with cached data...")
            logger.info("Click Refresh in the UI to fetch new data.")

            # Only run initial scan if --scan flag is explicitly provided
            if args.scan:
                def background_scan():
                    time.sleep(2)  # Let server start first
                    strat.run_scan()
                threading.Thread(target=background_scan, daemon=True).start()

            if args.background and strat.config.get("schedule", {}).get("background_enabled", False):
                strat.start_background_scheduler()
            elif args.background:
                logger.info("Background scheduler disabled in config (schedule.background_enabled: false)")

            strat.serve_frontend(port=args.port, open_browser=True)
        else:
            # No server - just run scan
            if args.scan:
                strat.run_scan()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        strat.cleanup()


if __name__ == "__main__":
    main()
