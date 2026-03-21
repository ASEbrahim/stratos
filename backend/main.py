"""
STRAT_OS - Main Orchestrator
The central brain that coordinates all modules and generates the final output.
"""

import copy
import hashlib
import itertools
import json
import logging
import logging.handlers
import os
import tempfile
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
import user_data


def _create_scorer(config, db=None, profile_id=0):
    """Create the scorer. B3.3: AdaptiveScorer is now the sole scorer."""
    role = config.get('profile', {}).get('role', '?')
    location = config.get('profile', {}).get('location', '?')
    logger.info(f"Using AdaptiveScorer for profile: {role} in {location}")
    return AdaptiveScorer(config, db=db, profile_id=profile_id)


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

# ═══ Named Constants (Issue #8: replace magic numbers) ═══════════════
SCHEDULER_JOIN_TIMEOUT_SECS = 5
BRIEFING_DEFAULT_TIMEOUT_SECS = 30
DB_CLEANUP_EVERY_N_SCANS = 10
DB_CLEANUP_MAX_AGE_DAYS = 30
SHADOW_SAMPLE_EVERY_NTH = 5
SHADOW_SAMPLE_MAX = 20
SSE_PROGRESS_EVERY_NTH = 2
SCORE_HIGH_THRESHOLD = 7.0
SCORE_MEDIUM_LOW = 5.0
SCORE_CRITICAL_THRESHOLD = 9.0
DEFAULT_CACHE_MARKET_TTL = 60
DEFAULT_CACHE_NEWS_TTL = 900
DEFAULT_FALLBACK_SCORE = 3.0
DEFAULT_FAST_BUFFER = 30
DEFAULT_FAST_MINIMUM = 45
DEFAULT_SLOW_MULTIPLIER = 3
DEFAULT_SLOW_BUFFER = 60
DEFAULT_MAX_NEWS_ITEMS = 50
DEFAULT_RETENTION_THRESHOLD = 8.0
DEFAULT_RETENTION_MAX_AGE_HOURS = 24
DEFAULT_RETENTION_MAX_ITEMS = 20
DEFAULT_SCHEDULER_INTERVAL_MIN = 30
DEFAULT_DISTILL_HOURS = 168
DEFAULT_DISTILL_LIMIT = 200
DEFAULT_DISTILL_THRESHOLD = 2.0
INITIAL_SCAN_DELAY_SECS = 2
SCAN_CANCEL_WAIT_SECS = 10


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

        # Load TTS persona voice overrides from config
        try:
            from processors.tts import load_persona_voices_from_config
            load_persona_voices_from_config(self.config)
        except Exception as e:
            logger.debug(f"TTS persona voice loading skipped (not critical): {e}")

        # Initialize database (resolve relative paths against backend dir)
        db_path = self.config.get("system", {}).get("database_file", "strat_os.db")
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_path)
        self.db = get_database(db_path)

        # Initialize components
        self.market_fetcher = MarketFetcher(self.config.get("market", {}))
        self.news_fetcher = NewsFetcher(self.config)  # Pass full config for Kuwait intel
        self.discovery = EntityDiscovery(self.config.get("discovery", {}), self.db)
        self.scorer = _create_scorer(self.config, db=self.db, profile_id=0)  # Pick scorer based on profile
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
            "data_version": self._get_data_version(),
            "scan_profile_id": None
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

        # Output file lock — serializes JSON read-modify-write across scan,
        # market refresh, and deferred briefing threads
        self._output_lock = threading.Lock()

        # Deferred briefing thread tracking (BUG-2 fix)
        self._briefing_thread = None
        self._briefing_gen_counter = itertools.count(1)
        self._briefing_generation = 0
        self._briefing_done = threading.Event()
        self._briefing_done.set()  # Initially "done" (no pending briefing)

        # Server start time for uptime tracking
        self._server_start_time = time.time()

        logger.info("STRAT_OS initialized")

    # ═══ SSE Push (delegates to SSEManager) ═══════════════════════

    def sse_register(self, wfile, profile_id=0):
        """Register a new SSE client connection with optional profile scope."""
        self.sse.register(wfile, profile_id=profile_id)

    def sse_unregister(self, wfile):
        """Remove a disconnected SSE client."""
        self.sse.unregister(wfile)

    def sse_broadcast(self, event_type: str, data: dict = None, profile_id=None):
        """Push an event to connected SSE clients (profile_id=None → all)."""
        self.sse.broadcast(event_type, data, profile_id=profile_id)

    # ═══ Profile Isolation (A2.1) ═════════════════════════════════

    def ensure_profile(self, profile_name: str):
        """Ensure the given profile's config is active.

        If a different profile is currently loaded, switch to the
        requested one from the in-memory cache.  If not cached (e.g. after
        server restart), load the config_overlay from DB and cache it.

        Thread-safe: the entire check-and-switch is serialized via _config_lock
        to prevent two threads from racing on different profiles.
        """
        if not profile_name:
            return
        with self._config_lock:
            if self.active_profile == profile_name:
                return  # Already active (or switched by another thread)
            if profile_name not in self._profile_configs:
                # Not cached — try loading from DB (handles server restart)
                self._load_profile_from_db(profile_name)
            if profile_name not in self._profile_configs:
                logger.warning(f"Profile '{profile_name}' not found in cache or DB, skipping switch")
                return  # Still not found — give up
            self.config = copy.deepcopy(self._profile_configs[profile_name])
            self.active_profile = profile_name
            # Update active_profile_id from DB
            try:
                if profile_name.startswith('pid_'):
                    self.active_profile_id = int(profile_name[4:])
                else:
                    row = self.db.conn.execute(
                        "SELECT id FROM profiles WHERE name = ?", (profile_name,)
                    ).fetchone()
                    if row:
                        self.active_profile_id = row[0]
            except Exception as e:
                logger.warning(f"Failed to resolve profile_id for '{profile_name}': {e}")
            logger.info(f"Profile switched to: {profile_name} (id={self.active_profile_id})")

    def _load_profile_from_db(self, profile_name: str):
        """Load a profile's config_overlay from DB into the in-memory cache.
        Supports 'pid_N' format for DB-auth users (avoids 'default' name collision)."""
        try:
            if profile_name.startswith('pid_'):
                pid = int(profile_name[4:])
                row = self.db.conn.execute(
                    "SELECT config_overlay FROM profiles WHERE id = ?",
                    (pid,)
                ).fetchone()
            else:
                row = self.db.conn.execute(
                    "SELECT config_overlay FROM profiles WHERE name = ?",
                    (profile_name,)
                ).fetchone()
            if not row:
                return  # Profile doesn't exist in DB
            # Start from base config, apply overlay (if any)
            base = self._load_config()
            # Inject env var secrets (serper, google keys) — _load_config only reads YAML
            _search = base.setdefault('search', {})
            if os.environ.get('SERPER_API_KEY'):
                _search['serper_api_key'] = os.environ['SERPER_API_KEY']
            if os.environ.get('GOOGLE_API_KEY'):
                _search['google_api_key'] = os.environ['GOOGLE_API_KEY']
            if os.environ.get('GOOGLE_CSE_ID'):
                _search['google_cx'] = os.environ['GOOGLE_CSE_ID']
            if row[0]:
                import json as _json
                overlay = _json.loads(row[0])
                if overlay:  # Non-empty overlay
                    self._apply_overlay(base, overlay)
            else:
                # New profile with no overlay — clear user-specific fields
                # so stale config.yaml data doesn't bleed into new accounts
                base['profile'] = {'role': '', 'location': '', 'context': ''}
                base['dynamic_categories'] = []
                base.get('market', {}).pop('tickers', None)
            self._profile_configs[profile_name] = base
            logger.info(f"Profile '{profile_name}' loaded from DB into cache "
                       f"(role={base.get('profile',{}).get('role','?')}, "
                       f"cats={len(base.get('dynamic_categories',[]))})")
        except Exception as e:
            logger.warning(f"Failed to load profile '{profile_name}' from DB: {e}")

    @staticmethod
    def _apply_overlay(config: dict, overlay: dict):
        """Apply a config_overlay dict onto a base config (in-place)."""
        for key, val in overlay.items():
            existing = config.get(key)
            if isinstance(existing, dict) and isinstance(val, dict):
                existing.update(val)
            else:
                config[key] = val

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
                            val = value.strip().strip('"').strip("'")
                            os.environ.setdefault(key.strip(), val)
            except Exception as e:
                logger.warning(f"Failed to read .env: {e}")

        search = self.config.setdefault('search', {})
        if os.environ.get('SERPER_API_KEY'):
            search['serper_api_key'] = os.environ['SERPER_API_KEY']
        if os.environ.get('GOOGLE_API_KEY'):
            search['google_api_key'] = os.environ['GOOGLE_API_KEY']
        if os.environ.get('GOOGLE_CSE_ID'):
            search['google_cx'] = os.environ['GOOGLE_CSE_ID']

        # SMTP settings from .env override config.yaml
        email_cfg = self.config.setdefault('email', {})
        if os.environ.get('SMTP_HOST'):
            email_cfg['smtp_host'] = os.environ['SMTP_HOST']
        if os.environ.get('SMTP_PORT'):
            email_cfg['smtp_port'] = int(os.environ['SMTP_PORT'])
        if os.environ.get('SMTP_USER'):
            email_cfg['smtp_user'] = os.environ['SMTP_USER']
        if os.environ.get('SMTP_PASSWORD'):
            email_cfg['smtp_password'] = os.environ['SMTP_PASSWORD']

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
                config_dir = os.path.dirname(os.path.abspath(self.config_path))
                data = yaml.dump(raw, default_flow_style=False, sort_keys=False).encode('utf-8')
                fd, tmp = tempfile.mkstemp(dir=config_dir, suffix='.yaml.tmp')
                try:
                    os.write(fd, data)
                    os.close(fd)
                    fd = -1
                    os.replace(tmp, self.config_path)
                except BaseException:
                    if fd >= 0:
                        os.close(fd)
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                    raise
                logger.info(f"Serper credits synced: {config_remaining} -> {actual_remaining} (tracker)")
        except Exception as e:
            logger.debug(f"Could not sync Serper credits: {e}")

    def run_scan(self, profile_id=None) -> Dict[str, Any]:
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

        Args:
            profile_id: Explicit profile_id from the requesting session.
                         If None, falls back to self.active_profile_id.

        Returns:
            The complete output data
        """
        if not self._scan_lock.acquire(blocking=False):
            scanning_pid = self.scan_status.get("scan_profile_id")
            logger.warning(f"Scan already in progress (profile={scanning_pid}), rejecting request from profile={profile_id}")
            return {"error": "scan_in_progress", "scan_profile_id": scanning_pid}
        try:
            return self._run_scan_impl(profile_id)
        finally:
            self._scan_lock.release()

    def _run_scan_impl(self, profile_id=None) -> Dict[str, Any]:
        """Internal scan implementation (called with _scan_lock held)."""
        # Use explicit profile_id if provided (from HTTP handler),
        # otherwise fall back to global (for background scheduler).
        if profile_id is not None:
            _scan_pid = profile_id
        else:
            with self._config_lock:
                _scan_pid = self.active_profile_id
        self._scan_cancelled.clear()
        self._snapshot_previous_articles()  # Snapshot before any writes
        self.scan_status["is_scanning"] = True
        self.scan_status["scan_profile_id"] = _scan_pid
        self.scan_status["stage"] = "starting"
        self.scan_status["progress"] = "Reloading configuration..."
        self.scan_status["scored"] = 0
        self.scan_status["total"] = 0
        self.scan_status["high"] = 0
        self.scan_status["medium"] = 0
        self.scan_status["cancelled"] = False
        self.sse_broadcast("scan", {"status": "starting", "progress": "Reloading configuration..."}, profile_id=_scan_pid)

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
        self.scorer = _create_scorer(self.config, db=self.db, profile_id=_scan_pid)  # Rebuild scorer with latest profile
        self.briefing_gen = BriefingGenerator(self.config)  # Rebuild briefing with new profile/context

        logger.info("=" * 60)
        logger.info(f"Starting intelligence scan (profile_id={_scan_pid})...")
        start_time = time.time()

        try:
            # 1. Fetch market data
            self.scan_status["stage"] = "market"
            self.scan_status["progress"] = "Fetching market data..."
            self.sse_broadcast("scan", {"status": "market", "progress": "Fetching market data..."}, profile_id=_scan_pid)
            logger.info("[1/6] Fetching market data...")
            cache_ttl = self.config.get("cache", {}).get("market_ttl_seconds", DEFAULT_CACHE_MARKET_TTL)
            market_data, market_alerts = self.market_fetcher.fetch_all(cache_ttl_seconds=cache_ttl)

            # Save market snapshots to DB
            for symbol, data in market_data.items():
                if "error" not in data:
                    for interval, interval_data in data.get("data", {}).items():
                        self.db.save_market_snapshot(symbol, data["name"], interval, interval_data)

            # 2. Fetch news
            self.scan_status["stage"] = "news"
            self.scan_status["progress"] = "Fetching news articles..."
            self.sse_broadcast("scan", {"status": "news", "progress": "Fetching news articles..."}, profile_id=_scan_pid)
            logger.info("[2/6] Fetching news...")
            cache_ttl = self.config.get("cache", {}).get("news_ttl_seconds", DEFAULT_CACHE_NEWS_TTL)
            news_items = self.news_fetcher.fetch_all(cache_ttl_seconds=cache_ttl)
            news_dicts = [item.to_dict() for item in news_items]

            # Report fetch complete with count
            self.scan_status["progress"] = f"Fetched {len(news_dicts)} articles, preparing to score..."
            self.sse_broadcast("scan", {"status": "news_done", "progress": f"Fetched {len(news_dicts)} articles", "fetched": len(news_dicts)}, profile_id=_scan_pid)

            # 2.5 Re-classify items into dynamic categories
            # RSS items arrive with generic category='tech'/'general'. Re-tag them
            # with dynamic category IDs so the scorer routes them correctly and
            # the frontend filters them into the right tabs.
            news_dicts = self._reclassify_dynamic(news_dicts)

            # 2.7 Incremental scan — reuse scores from previous scan if context unchanged
            news_dicts, prescore_reused = self._reuse_snapshot_scores(news_dicts, profile_id=_scan_pid)

            # 3. Score news items (only those that need fresh scoring)
            self.scan_status["stage"] = "scoring"
            total_items = len(news_dicts)
            self.scan_status["scored"] = 0
            self.scan_status["total"] = total_items
            self.scan_status["progress"] = f"Scoring 0/{total_items} articles with AI..."
            self.sse_broadcast("scan", {"status": "scoring", "progress": f"Scoring 0/{total_items} articles with AI...", "scored": 0, "total": total_items}, profile_id=_scan_pid)
            logger.info("[3/6] Scoring news items with AI...")

            # Score with progress callback — scorer reports (done, ambiguous_total)
            # for LLM-scored items only (rule-scored items are instant and skipped)
            def on_score_progress(current, total):
                self.scan_status["scored"] = current
                self.scan_status["total"] = total
                self.scan_status["progress"] = f"AI scoring {current}/{total} articles..."
                if current % SSE_PROGRESS_EVERY_NTH == 0 or current == total:
                    self.sse_broadcast("scan", {"status": "scoring", "progress": f"AI scoring {current}/{total} articles...", "scored": current, "total": total}, profile_id=_scan_pid)

            # === TWO-PASS SCORING ===
            timeout_cfg = self.config.get("scoring", {}).get("timeout", {})
            fallback_score = self.config.get("scoring", {}).get("fallback_score", DEFAULT_FALLBACK_SCORE)
            fast_buffer = timeout_cfg.get("fast_buffer", DEFAULT_FAST_BUFFER)
            fast_minimum = timeout_cfg.get("fast_minimum", DEFAULT_FAST_MINIMUM)
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
            _high = sum(1 for i in scored_items if i.get('score', 0) >= SCORE_HIGH_THRESHOLD)
            _med = sum(1 for i in scored_items if SCORE_MEDIUM_LOW < i.get('score', 0) < SCORE_HIGH_THRESHOLD)
            self.scan_status["high"] = _high
            self.scan_status["medium"] = _med

            # Save scored items to DB (even on cancel — partial results are valuable)
            for item in scored_items:
                self.db.save_news_item(item, profile_id=_scan_pid)

            # Write Pass 1 results immediately so dashboard can display them
            if deferred_items:
                partial_output = self._build_output(market_data, scored_items, {}, market_alerts=market_alerts, profile_id=_scan_pid)
                self._write_output(partial_output, profile_id=_scan_pid)
                self.scan_status["data_version"] = self._get_data_version()
                self.sse_broadcast("pass1_complete", {
                    "scored": len(scored_items),
                    "deferred": len(deferred_items),
                    "high": _high,
                    "medium": _med,
                    "data_version": self.scan_status["data_version"]
                }, profile_id=_scan_pid)

            # Handle cancellation — save partial results and return early
            if self._scan_cancelled.is_set():
                logger.info(f"Scan cancelled after scoring {len(scored_items)}/{total_items} items")
                if not deferred_items:
                    output = self._build_output(market_data, scored_items, {}, market_alerts=market_alerts, profile_id=_scan_pid)
                    self._write_output(output, profile_id=_scan_pid)
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
                }, profile_id=_scan_pid)
                return self._build_output(market_data, scored_items, {}, market_alerts=market_alerts, profile_id=_scan_pid)

            # === PASS 2: Retry deferred articles with longer timeout ===
            if deferred_items and not self._scan_cancelled.is_set():
                slow_mult = timeout_cfg.get("slow_multiplier", DEFAULT_SLOW_MULTIPLIER)
                slow_buf = timeout_cfg.get("slow_buffer", DEFAULT_SLOW_BUFFER)
                pass2_timeout = self.scorer.scoring_timer.slow_timeout(multiplier=slow_mult, buffer=slow_buf)

                logger.info(f"[Pass 2] Retrying {len(deferred_items)} deferred items (timeout={pass2_timeout:.0f}s)")
                self.scan_status["stage"] = "scoring_pass2"
                self.scan_status["progress"] = f"Pass 2: Retrying {len(deferred_items)} deferred items..."
                self.sse_broadcast("scan", {
                    "status": "scoring_pass2",
                    "progress": f"Pass 2: {len(deferred_items)} deferred items...",
                    "scored": self.scan_status["scored"],
                    "total": total_items
                }, profile_id=_scan_pid)

                pass2_scored, still_deferred = self.scorer.score_items(
                    deferred_items,
                    progress_callback=lambda cur, tot: self.sse_broadcast("scan", {
                        "status": "scoring_pass2",
                        "progress": f"Pass 2: {cur}/{tot} deferred...",
                        "scored": self.scan_status["scored"],
                        "total": total_items
                    }, profile_id=_scan_pid),
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
                    self.db.save_news_item(item, profile_id=_scan_pid)

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
            self.sse_broadcast("scan", {"status": "discovery", "progress": "Running entity discovery..."}, profile_id=_scan_pid)
            logger.info("[4/6] Running entity discovery...")
            discoveries = self.discovery.discover(scored_items)

            # 5. Build output WITHOUT briefing — let the user see results immediately
            self.scan_status["stage"] = "output"
            self.scan_status["progress"] = "Building output..."
            logger.info("[5/6] Building output (briefing deferred)...")
            output = self._build_output(market_data, scored_items, {}, market_alerts=market_alerts, profile_id=_scan_pid)
            self._write_output(output, profile_id=_scan_pid)

            # 6. Spawn briefing in background — non-blocking
            self._spawn_deferred_briefing(market_data, market_alerts, scored_items, discoveries, profile_id=_scan_pid)

            elapsed = time.time() - start_time
            logger.info(f"Scan complete in {elapsed:.1f}s (profile_id={_scan_pid})")
            logger.info("=" * 60)

            # Save scan log entry
            critical = sum(1 for i in scored_items if i.get('score', 0) >= SCORE_CRITICAL_THRESHOLD)
            high = sum(1 for i in scored_items if SCORE_HIGH_THRESHOLD <= i.get('score', 0) < SCORE_CRITICAL_THRESHOLD)
            medium = sum(1 for i in scored_items if SCORE_MEDIUM_LOW < i.get('score', 0) < SCORE_HIGH_THRESHOLD)
            noise = sum(1 for i in scored_items if i.get('score', 0) <= SCORE_MEDIUM_LOW)
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
            }, profile_id=_scan_pid)

            # Update status
            self.scan_status["is_scanning"] = False
            self.scan_status["scan_profile_id"] = None
            self.scan_status["stage"] = "complete"
            self.scan_status["progress"] = ""
            self.scan_status["last_completed"] = datetime.now().isoformat()
            self.scan_status["data_version"] = self._get_data_version()
            self.sse_broadcast("complete", {"data_version": self.scan_status["data_version"]}, profile_id=_scan_pid)

            # Broadcast critical signals for push notifications (profile-scoped)
            for item in scored_items:
                if item.get('score', 0) >= SCORE_CRITICAL_THRESHOLD:
                    self.sse_broadcast("critical_signal", {
                        "title": item.get('title', ''),
                        "score": item.get('score', 0),
                        "reason": item.get('score_reason', ''),
                        "url": item.get('url', ''),
                        "category": item.get('category', ''),
                    }, profile_id=_scan_pid)

            # Shadow scoring — self-consistency check (daemon thread)
            self._run_shadow_scoring(scored_items, scan_id, profile_id=_scan_pid)

            # Periodic DB cleanup — every 10th scan
            self._scan_count += 1
            if self._scan_count % DB_CLEANUP_EVERY_N_SCANS == 0:
                logger.info(f"Running periodic DB cleanup (scan #{self._scan_count})...")
                self.db.cleanup_old_data(days=DB_CLEANUP_MAX_AGE_DAYS, profile_id=_scan_pid)

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
            }, profile_id=_scan_pid)
            self.scan_status["is_scanning"] = False
            self.scan_status["stage"] = "error"
            self.scan_status["progress"] = str(e)
            self.sse_broadcast("scan_error", {"message": str(e)}, profile_id=_scan_pid)
            raise

    def _run_shadow_scoring(self, scored_items, scan_id, profile_id=0):
        """Run shadow scoring in a background thread.

        Creates a second AdaptiveScorer instance for self-consistency checks.
        """
        shadow_enabled = self.config.get('scoring', {}).get('shadow_scoring', False)
        if not shadow_enabled:
            return

        def _shadow_worker():
            try:
                primary_name = 'AdaptiveScorer'
                shadow = AdaptiveScorer(self.config, db=self.db)
                shadow_name = 'AdaptiveScorer-shadow'

                # Sample every Nth item, capped
                sampled = scored_items[::SHADOW_SAMPLE_EVERY_NTH][:SHADOW_SAMPLE_MAX]
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
                            }, profile_id=profile_id)
                        except Exception as e:
                            logger.debug(f"Shadow score failed for item: {e}")

                logger.info(f"Shadow scoring complete: {len(sampled)} items compared")

            except Exception as e:
                logger.error(f"Shadow scoring thread crashed (profile_id={profile_id}): {e}", exc_info=True)

        thread = threading.Thread(target=_shadow_worker, daemon=True, name="shadow-scoring")
        thread.start()

    def _run_auto_distillation(self):
        """Run distillation in background thread after a scan completes."""
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
                hours = distill_cfg.get("hours", DEFAULT_DISTILL_HOURS)
                limit = distill_cfg.get("limit", DEFAULT_DISTILL_LIMIT)
                threshold = distill_cfg.get("threshold", DEFAULT_DISTILL_THRESHOLD)

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
                logger.error(f"Auto-distillation thread crashed: {e}", exc_info=True)

        thread = threading.Thread(target=_distill_worker, daemon=True, name="auto-distill")
        thread.start()

    def run_market_refresh(self, profile_id=None) -> Dict[str, Any]:
        """
        Refresh only market data (fast, no API calls).
        Updates the existing output file with fresh market data.
        """
        # If a news scan is in progress, skip — news scan will include market data
        if self.scan_status.get("is_scanning") and self.scan_status.get("stage") not in ("market", "complete", "error"):
            logger.info("Skipping market refresh — news scan in progress")
            return {}

        if profile_id is not None:
            _scan_pid = profile_id
        else:
            with self._config_lock:
                _scan_pid = self.active_profile_id

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

            # Load existing output and update market section (under lock to
            # prevent race with deferred briefing writer)
            with self._output_lock:
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

                # Atomic write (inline, lock already held)
                import tempfile
                out_path = self.output_file
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=str(out_path.parent), suffix=".tmp", prefix=".market_"
                )
                try:
                    with os.fdopen(tmp_fd, "w") as f:
                        json.dump(output, f, indent=2)
                    os.replace(tmp_path, str(out_path))
                except Exception:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            logger.info(f"Output written to {self.output_file} (market refresh, profile_id={_scan_pid})")

            elapsed = time.time() - start_time
            logger.info(f"Market refresh complete in {elapsed:.1f}s")
            logger.info("=" * 40)

            self.scan_status["is_scanning"] = False
            self.scan_status["stage"] = "complete"
            self.scan_status["progress"] = ""
            self.scan_status["data_version"] = self._get_data_version()
            self.sse_broadcast("complete", {"data_version": self.scan_status["data_version"]}, profile_id=_scan_pid)

            return output

        except Exception as e:
            logger.error(f"Market refresh failed: {e}")
            self.scan_status["is_scanning"] = False
            self.scan_status["stage"] = "error"
            self.scan_status["progress"] = str(e)
            self.sse_broadcast("scan_error", {"message": str(e)}, profile_id=_scan_pid)
            raise

    def run_news_refresh(self, profile_id=None) -> Dict[str, Any]:
        """
        Refresh news, scoring, and briefing (slower, uses API calls).
        Keeps existing market data.

        Args:
            profile_id: Explicit profile_id from the requesting session.
                         If None, falls back to self.active_profile_id.
        """
        if not self._scan_lock.acquire(blocking=False):
            logger.warning("Scan already in progress, skipping news refresh")
            return {}
        try:
            return self._run_news_refresh_impl(profile_id)
        finally:
            self._scan_lock.release()

    def _run_news_refresh_impl(self, profile_id=None) -> Dict[str, Any]:
        """Internal news refresh implementation (called with _scan_lock held)."""
        # Use explicit profile_id if provided (from HTTP handler),
        # otherwise fall back to global (for background scheduler).
        if profile_id is not None:
            _scan_pid = profile_id
        else:
            with self._config_lock:
                _scan_pid = self.active_profile_id
        self._scan_cancelled.clear()
        self._snapshot_previous_articles()  # Snapshot before any writes
        self.scan_status["is_scanning"] = True
        self.scan_status["scan_profile_id"] = _scan_pid
        self.scan_status["stage"] = "news"
        self.scan_status["progress"] = "Fetching news..."
        self.scan_status["scored"] = 0
        self.scan_status["total"] = 0
        self.scan_status["high"] = 0
        self.scan_status["medium"] = 0
        self.scan_status["cancelled"] = False
        self.sse_broadcast("scan", {"status": "news", "progress": "Fetching news..."}, profile_id=_scan_pid)

        logger.info("=" * 60)
        logger.info(f"Refreshing news and intelligence (profile_id={_scan_pid})...")
        start_time = time.time()

        try:
            # Reload config (profile-aware — A2.1)
            if self.active_profile and self.active_profile in self._profile_configs:
                self.config = copy.deepcopy(self._profile_configs[self.active_profile])
                logger.info(f"Config loaded from cache for '{self.active_profile}' "
                           f"(role={self.config.get('profile',{}).get('role')}, "
                           f"cats={len(self.config.get('dynamic_categories',[]))})")
            else:
                self.config = self._load_config()
                logger.info(f"Config loaded from config.yaml (role={self.config.get('profile',{}).get('role')})")
            # Always re-apply .env secrets (config.yaml has ${VAR} placeholders)
            self._load_env_secrets()
            self.news_fetcher = NewsFetcher(self.config)
            self.scorer = _create_scorer(self.config, db=self.db, profile_id=_scan_pid)  # Rebuild scorer with latest profile
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
            self.sse_broadcast("scan", {"status": "news", "progress": "Fetching news..."}, profile_id=_scan_pid)
            logger.info("[1/4] Fetching news...")
            news_items = self.news_fetcher.fetch_all(cache_ttl_seconds=0)
            news_dicts = [item.to_dict() for item in news_items]

            # Re-classify into dynamic categories
            news_dicts = self._reclassify_dynamic(news_dicts)

            # Incremental scan — reuse scores from previous scan if context unchanged
            news_dicts, prescore_reused = self._reuse_snapshot_scores(news_dicts, profile_id=_scan_pid)

            # Score news
            self.scan_status["stage"] = "scoring"
            total_items = len(news_dicts)
            self.scan_status["scored"] = 0
            self.scan_status["total"] = total_items
            self.scan_status["progress"] = f"Scoring 0/{total_items} articles with AI..."
            self.sse_broadcast("scan", {"status": "scoring", "progress": f"Scoring 0/{total_items} articles with AI...", "scored": 0, "total": total_items}, profile_id=_scan_pid)
            logger.info("[2/4] Scoring news items with AI...")

            def on_score_progress(current, total):
                self.scan_status["scored"] = current
                self.scan_status["total"] = total
                self.scan_status["progress"] = f"AI scoring {current}/{total} articles..."
                if current % SSE_PROGRESS_EVERY_NTH == 0 or current == total:
                    self.sse_broadcast("scan", {"status": "scoring", "progress": f"AI scoring {current}/{total} articles...", "scored": current, "total": total}, profile_id=_scan_pid)

            # === TWO-PASS SCORING ===
            timeout_cfg = self.config.get("scoring", {}).get("timeout", {})
            fallback_score = self.config.get("scoring", {}).get("fallback_score", DEFAULT_FALLBACK_SCORE)
            fast_buffer = timeout_cfg.get("fast_buffer", DEFAULT_FAST_BUFFER)
            fast_minimum = timeout_cfg.get("fast_minimum", DEFAULT_FAST_MINIMUM)
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
            _high = sum(1 for i in scored_items if i.get('score', 0) >= SCORE_HIGH_THRESHOLD)
            _med = sum(1 for i in scored_items if SCORE_MEDIUM_LOW < i.get('score', 0) < SCORE_HIGH_THRESHOLD)
            self.scan_status["high"] = _high
            self.scan_status["medium"] = _med

            # Save scored items to DB (even on cancel)
            for item in scored_items:
                self.db.save_news_item(item, profile_id=_scan_pid)

            # Write Pass 1 results immediately so dashboard can display them
            if deferred_items:
                partial_output = self._build_output(market_data, scored_items, {}, market_alerts=market_alerts, profile_id=_scan_pid)
                self._write_output(partial_output, profile_id=_scan_pid)
                self.scan_status["data_version"] = self._get_data_version()
                self.sse_broadcast("pass1_complete", {
                    "scored": len(scored_items),
                    "deferred": len(deferred_items),
                    "high": _high,
                    "medium": _med,
                    "data_version": self.scan_status["data_version"]
                }, profile_id=_scan_pid)

            # Handle cancellation
            if self._scan_cancelled.is_set():
                logger.info(f"News refresh cancelled after scoring {len(scored_items)}/{total_items} items")
                if not deferred_items:
                    output = self._build_output(market_data, scored_items, {}, market_alerts=market_alerts, profile_id=_scan_pid)
                    self._write_output(output, profile_id=_scan_pid)
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
                }, profile_id=_scan_pid)
                return self._build_output(market_data, scored_items, {}, market_alerts=market_alerts, profile_id=_scan_pid)

            # === PASS 2: Retry deferred articles with longer timeout ===
            if deferred_items and not self._scan_cancelled.is_set():
                slow_mult = timeout_cfg.get("slow_multiplier", DEFAULT_SLOW_MULTIPLIER)
                slow_buf = timeout_cfg.get("slow_buffer", DEFAULT_SLOW_BUFFER)
                pass2_timeout = self.scorer.scoring_timer.slow_timeout(multiplier=slow_mult, buffer=slow_buf)

                logger.info(f"[Pass 2] Retrying {len(deferred_items)} deferred items (timeout={pass2_timeout:.0f}s)")
                self.scan_status["stage"] = "scoring_pass2"
                self.scan_status["progress"] = f"Pass 2: Retrying {len(deferred_items)} deferred items..."
                self.sse_broadcast("scan", {
                    "status": "scoring_pass2",
                    "progress": f"Pass 2: {len(deferred_items)} deferred items...",
                    "scored": self.scan_status["scored"],
                    "total": total_items
                }, profile_id=_scan_pid)

                pass2_scored, still_deferred = self.scorer.score_items(
                    deferred_items,
                    progress_callback=lambda cur, tot: self.sse_broadcast("scan", {
                        "status": "scoring_pass2",
                        "progress": f"Pass 2: {cur}/{tot} deferred...",
                        "scored": self.scan_status["scored"],
                        "total": total_items
                    }, profile_id=_scan_pid),
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
                    self.db.save_news_item(item, profile_id=_scan_pid)

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
            self.sse_broadcast("scan", {"status": "discovery", "progress": "Running entity discovery..."}, profile_id=_scan_pid)
            logger.info("[3/4] Running entity discovery...")
            discoveries = self.discovery.discover(scored_items)

            # Build output WITHOUT briefing — let the user see results immediately
            output = self._build_output(market_data, scored_items, {}, market_alerts=market_alerts, profile_id=_scan_pid)
            self._write_output(output, profile_id=_scan_pid)

            # Spawn briefing in background — non-blocking
            self._spawn_deferred_briefing(market_data, market_alerts, scored_items, discoveries, profile_id=_scan_pid)

            elapsed = time.time() - start_time
            logger.info(f"News refresh complete in {elapsed:.1f}s (profile_id={_scan_pid})")
            logger.info("=" * 60)

            # Save scan log entry
            critical = sum(1 for i in scored_items if i.get('score', 0) >= SCORE_CRITICAL_THRESHOLD)
            high = sum(1 for i in scored_items if SCORE_HIGH_THRESHOLD <= i.get('score', 0) < SCORE_CRITICAL_THRESHOLD)
            medium = sum(1 for i in scored_items if SCORE_MEDIUM_LOW < i.get('score', 0) < SCORE_HIGH_THRESHOLD)
            noise = sum(1 for i in scored_items if i.get('score', 0) <= SCORE_MEDIUM_LOW)
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
            }, profile_id=_scan_pid)

            self.scan_status["is_scanning"] = False
            self.scan_status["scan_profile_id"] = None
            self.scan_status["stage"] = "complete"
            self.scan_status["progress"] = ""
            self.scan_status["last_completed"] = datetime.now().isoformat()
            self.scan_status["data_version"] = self._get_data_version()
            self.sse_broadcast("complete", {"data_version": self.scan_status["data_version"]}, profile_id=_scan_pid)

            # Broadcast critical signals for push notifications (profile-scoped)
            for item in scored_items:
                if item.get('score', 0) >= SCORE_CRITICAL_THRESHOLD:
                    self.sse_broadcast("critical_signal", {
                        "title": item.get('title', ''),
                        "score": item.get('score', 0),
                        "reason": item.get('score_reason', ''),
                        "url": item.get('url', ''),
                        "category": item.get('category', ''),
                    }, profile_id=_scan_pid)

            # Shadow scoring — self-consistency check (daemon thread)
            self._run_shadow_scoring(scored_items, scan_id, profile_id=_scan_pid)

            return output

        except Exception as e:
            logger.error(f"News refresh failed: {e}")
            self.scan_status["is_scanning"] = False
            self.scan_status["stage"] = "error"
            self.scan_status["progress"] = str(e)
            self.sse_broadcast("scan_error", {"message": str(e)}, profile_id=_scan_pid)
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
                    except re.error as e:
                        logger.debug(f"Invalid regex for keyword '{kw_lower}': {e}")
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

    def _reuse_snapshot_scores(self, news_dicts: list, profile_id=0) -> tuple:
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

    def _spawn_deferred_briefing(self, market_data, market_alerts, scored_items, discoveries, profile_id=0):
        """Generate briefing in a background thread and patch the output when done."""
        current_gen = next(self._briefing_gen_counter)
        self._briefing_generation = current_gen
        self._briefing_done.clear()

        def _briefing_worker():
            try:
                if self._scan_cancelled.is_set():
                    return
                logger.info(f"Deferred briefing: generating (profile_id={profile_id})...")
                self.sse_broadcast("scan", {"status": "briefing", "progress": "Generating briefing..."}, profile_id=profile_id)
                # Build behavioral hint for briefing
                _beh_hint = ""
                try:
                    from behavioral import build_briefing_behavioral_hint
                    _beh_hint = build_briefing_behavioral_hint(self.db, profile_id, config=self.config)
                except Exception as _bhe:
                    logger.debug(f"Behavioral briefing hint skipped: {_bhe}")

                briefing = self.briefing_gen.generate_briefing(
                    market_data=market_data,
                    market_alerts=market_alerts,
                    news_items=scored_items,
                    discoveries=discoveries,
                    behavioral_hint=_beh_hint,
                )
                if self._scan_cancelled.is_set():
                    return
                # Save to DB
                self.db.save_briefing(briefing, profile_id=profile_id)
                # Patch the output file with the briefing (lock covers read-modify-write)
                import tempfile
                with self._output_lock:
                    if self.output_file.exists():
                        with open(self.output_file, "r") as f:
                            output = json.load(f)
                        output["briefing"] = briefing
                        output["meta"]["critical_count"] = briefing.get("critical_count", 0)
                        output["meta"]["high_count"] = briefing.get("high_count", 0)
                        out_path = self.output_file
                        tmp_fd, tmp_path = tempfile.mkstemp(
                            dir=str(out_path.parent), suffix=".tmp", prefix=".briefing_"
                        )
                        try:
                            with os.fdopen(tmp_fd, "w") as f:
                                json.dump(output, f, indent=2)
                            os.replace(tmp_path, str(out_path))
                        except Exception:
                            try:
                                os.unlink(tmp_path)
                            except OSError:
                                pass
                            raise
                self.sse_broadcast("briefing_ready", {"status": "ready"}, profile_id=profile_id)
                logger.info(f"Deferred briefing: complete, output patched (profile_id={profile_id})")
            except Exception as e:
                logger.error(f"Deferred briefing failed (profile_id={profile_id}): {e}", exc_info=True)
                self.sse_broadcast("briefing_ready", {"status": "failed"}, profile_id=profile_id)
            finally:
                # Signal completion only if this is still the current generation
                # (prevents stale thread A from signaling thread B's wait)
                if current_gen == self._briefing_generation:
                    self._briefing_done.set()

        self._briefing_thread = threading.Thread(target=_briefing_worker, daemon=True, name="briefing-deferred")
        self._briefing_thread.start()

    def wait_for_briefing(self, timeout: float = BRIEFING_DEFAULT_TIMEOUT_SECS) -> bool:
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

    def _merge_retained_articles(self, current_articles: list, profile_id: int = 0) -> list:
        """Merge high-scoring articles from previous scan into current results."""
        scoring_cfg = self.config.get("scoring", {})
        if not scoring_cfg.get("retain_high_scores", True):
            return current_articles

        threshold = scoring_cfg.get("retention_threshold", DEFAULT_RETENTION_THRESHOLD)
        max_age_hours = scoring_cfg.get("retention_max_age_hours", DEFAULT_RETENTION_MAX_AGE_HOURS)
        max_retained = scoring_cfg.get("retention_max_items", DEFAULT_RETENTION_MAX_ITEMS)

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
                except (ValueError, TypeError) as e:
                    logger.debug(f"Unparseable timestamp '{ts}' in retained article: {e}")
            # Check if user dismissed this article
            if self.db and self.db.was_dismissed(url, profile_id=profile_id):
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

    def _build_output(self, market_data: Dict, news_items: list, briefing: Dict, market_alerts: list = None, profile_id: int = 0) -> Dict[str, Any]:
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
        news_items = self._merge_retained_articles(news_items, profile_id=profile_id)
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
        max_items = self.config.get("system", {}).get("max_news_items", DEFAULT_MAX_NEWS_ITEMS)
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

    def _write_output(self, data: Dict[str, Any], profile_id: int = 0):
        """Write output to JSON file and per-user daily article export.
        Thread-safe: serialized via _output_lock to prevent briefing/scan races.
        Uses atomic write (tmp + rename) to prevent partial JSON on crash."""
        import tempfile
        with self._output_lock:
            out_path = self.output_file
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(out_path.parent), suffix=".tmp", prefix=".news_data_"
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, str(out_path))
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        logger.info(f"Output written to {self.output_file} (profile_id={profile_id})")

        # Per-user daily article JSONL export
        uid = user_data.get_user_id_for_profile(self.db, profile_id)
        if uid > 0:
            today = datetime.now().strftime("%Y-%m-%d")
            for article in data.get("news", []):
                user_data.append_jsonl(uid, f"scans/{today}.jsonl", {
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "score": article.get("score", 0),
                    "score_reason": article.get("score_reason", ""),
                    "category": article.get("category", ""),
                    "source": article.get("source", ""),
                    "timestamp": article.get("timestamp", ""),
                })

    def start_background_scheduler(self):
        """Start background refresh scheduler."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("Scheduler already running")
            return

        interval_min = self.config.get("schedule", {}).get("background_interval_minutes", DEFAULT_SCHEDULER_INTERVAL_MIN)

        def _resolve_scheduler_profile():
            """Resume the most recently active profile from sessions DB."""
            with self._config_lock:
                current_pid = self.active_profile_id
            if current_pid:
                return current_pid
            try:
                cursor = self.db.conn.cursor()
                cursor.execute("""
                    SELECT profile_id FROM sessions
                    WHERE expires_at > ? AND profile_id IS NOT NULL
                    ORDER BY last_active DESC LIMIT 1
                """, (datetime.now().isoformat(),))
                row = cursor.fetchone()
                if row and row[0]:
                    logger.info(f"Scheduler resuming profile_id={row[0]} from last active session")
                    return row[0]
            except Exception as e:
                logger.warning(f"Scheduler profile lookup failed: {e}")
            return None

        def scheduler_loop():
            try:
                while not self._stop_scheduler.is_set():
                    try:
                        pid = _resolve_scheduler_profile()
                        if pid:
                            self.run_scan(profile_id=pid)
                        else:
                            logger.info("Background scan skipped — no active profile")
                    except Exception as e:
                        logger.error(f"Scheduled scan failed: {e}", exc_info=True)

                    # Wait for interval or stop signal
                    self._stop_scheduler.wait(timeout=interval_min * 60)
            except Exception as e:
                logger.critical(f"Scheduler thread crashed: {e}", exc_info=True)

        self._stop_scheduler.clear()
        self._scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info(f"Background scheduler started (interval: {interval_min} min)")

    def stop_background_scheduler(self):
        """Stop the background scheduler."""
        if self._scheduler_thread:
            self._stop_scheduler.set()
            self._scheduler_thread.join(timeout=SCHEDULER_JOIN_TIMEOUT_SECS)
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
        """Cleanup resources: stop all threads, close DB, cancel pending work."""
        logger.info("Cleanup: stopping background scheduler...")
        self.stop_background_scheduler()

        # Cancel any in-progress scan so threads exit promptly
        self._scan_cancelled.set()

        # Wait for deferred briefing thread to finish
        if self._briefing_thread and self._briefing_thread.is_alive():
            logger.info("Cleanup: waiting for deferred briefing thread...")
            self._briefing_thread.join(timeout=SCHEDULER_JOIN_TIMEOUT_SECS)
            if self._briefing_thread.is_alive():
                logger.warning("Cleanup: briefing thread did not exit in time")

        # DB maintenance and close
        try:
            self.db.cleanup_old_data(days=DB_CLEANUP_MAX_AGE_DAYS)
        except Exception as e:
            logger.warning(f"Cleanup: DB cleanup_old_data failed: {e}")
        try:
            self.db.close()
        except Exception as e:
            logger.warning(f"Cleanup: DB close failed: {e}")

        logger.info("Cleanup: complete")


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
                    try:
                        time.sleep(INITIAL_SCAN_DELAY_SECS)  # Let server start first
                        strat.run_scan()
                    except Exception as e:
                        logger.error(f"Initial background scan failed: {e}", exc_info=True)
                threading.Thread(target=background_scan, daemon=True, name="initial-scan").start()

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
