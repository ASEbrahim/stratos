"""
Authentication and session management for StratOS.

Handles profile-based auth with hashed PINs, session tokens,
rate limiting, and API key masking.

Extracted from main.py:serve_frontend() closures (Sprint 4, A1.2).
"""

import copy
import json
import hashlib
import secrets
import time
import threading
import logging
import yaml
from pathlib import Path

logger = logging.getLogger("STRAT_OS")


class AuthManager:
    """Profile-based authentication with hashed PINs and session management."""

    SESSION_TTL = 86400 * 7  # 7 days

    RATE_LIMITS = {
        '/api/refresh': (2, 60),
        '/api/refresh-news': (2, 60),
        '/api/refresh-market': (3, 60),
        '/api/generate-profile': (5, 60),
        '/api/suggest-context': (10, 60),
        '/api/agent-chat': (20, 60),
        '/api/top-movers': (3, 60),
    }

    AUTH_EXEMPT = {
        '/api/auth-check', '/api/auth', '/api/logout', '/api/register',
        '/api/suggest-context', '/api/generate-profile',
        '/api/wizard-preselect', '/api/wizard-tab-suggest', '/api/wizard-rv-items',
        '/api/refresh', '/api/status', '/api/scan/status', '/api/scan/cancel',
        '/api/health', '/api/events',
    }

    def __init__(self, config_path):
        self._config_path = Path(config_path)
        self._sessions_file = self._config_path.parent / ".sessions.json"
        self._active_sessions = {}
        self._rate_buckets = {}
        self._rate_lock = threading.Lock()
        self._last_purge = 0
        self._load_sessions()

    def _save_sessions(self):
        """Persist sessions to disk so they survive server restarts."""
        try:
            self._sessions_file.write_text(json.dumps(self._active_sessions))
        except Exception:
            pass

    def _load_sessions(self):
        """Load sessions from disk on startup."""
        try:
            if self._sessions_file.exists():
                data = json.loads(self._sessions_file.read_text())
                now = time.time()
                self._active_sessions = {k: v for k, v in data.items() if v.get("expiry", 0) > now}
                if self._active_sessions:
                    logger.info(f"Restored {len(self._active_sessions)} active session(s) from disk")
        except Exception:
            self._active_sessions = {}

    @staticmethod
    def hash_pin(pin):
        """Hash a PIN with SHA-256 for storage."""
        return hashlib.sha256(str(pin).strip().encode('utf-8')).hexdigest()

    def profiles_dir(self):
        """Get profiles directory, creating if needed."""
        d = self._config_path.parent / "profiles"
        d.mkdir(exist_ok=True)
        return d

    @staticmethod
    def safe_name(name):
        """Sanitize a profile name for filesystem use."""
        return "".join(c for c in name if c.isalnum() or c in " _-").strip()

    def list_profiles(self, device_id=None):
        """List profiles, optionally filtered to a specific device."""
        profiles = []
        for f in sorted(self.profiles_dir().glob("*.yaml")):
            try:
                with open(f) as pf:
                    data = yaml.safe_load(pf) or {}
                p = data.get("profile", {})
                sec = data.get("security", {})
                devices = sec.get("devices", [])
                if device_id and devices and device_id not in devices:
                    continue
                if device_id and not devices:
                    continue
                profiles.append({
                    "name": f.stem,
                    "role": p.get("role", ""),
                    "location": p.get("location", "Kuwait"),
                    "has_pin": bool(str(sec.get("pin_hash", sec.get("pin", ""))).strip()),
                })
            except Exception:
                pass
        return profiles

    def verify_profile_pin(self, name, pin):
        """Verify PIN against stored hash. Returns True/False/None(not found)."""
        safe = self.safe_name(name)
        filepath = self.profiles_dir() / f"{safe}.yaml"
        if not filepath.exists():
            return None
        try:
            with open(filepath) as f:
                data = yaml.safe_load(f) or {}
            sec = data.get("security", {})
            stored_hash = sec.get("pin_hash", "")
            if not stored_hash:
                stored_plain = str(sec.get("pin", "")).strip()
                if not stored_plain:
                    return False
                return str(pin).strip() == stored_plain
            return self.hash_pin(pin) == stored_hash
        except Exception:
            return None

    # System keys to PRESERVE across profile switches.
    # Everything NOT in this set is cleared on profile switch.
    # This is a blacklist-preserve approach: future profile-specific keys
    # are automatically cleared without needing to update a whitelist.
    SYSTEM_KEYS = frozenset([
        'scoring',      # model names, ollama host, thresholds (profile may overlay retain_*)
        'search',       # API keys, provider settings
        'system',       # database_file, output_file, log_level
        'discovery',    # mention thresholds, baseline hours
        'cache',        # TTLs
        'email',        # SMTP config
        'schedule',     # background scheduler settings
    ])

    def load_profile_config(self, name, strat_instance):
        """Load a profile's config into the live system.

        Uses blacklist-preserve: saves SYSTEM_KEYS, nukes everything else,
        then applies the new profile's config on top. This prevents any
        profile-specific key from leaking between profiles.
        """
        safe = self.safe_name(name)
        filepath = self.profiles_dir() / f"{safe}.yaml"
        if not filepath.exists():
            return False
        try:
            with open(filepath) as f:
                preset = yaml.safe_load(f) or {}

            # 1. Snapshot system keys (deep copy to avoid reference sharing)
            preserved = {}
            for key in self.SYSTEM_KEYS:
                if key in strat_instance.config:
                    preserved[key] = copy.deepcopy(strat_instance.config[key])

            # 2. Nuclear clear — remove EVERYTHING except system keys
            all_keys = list(strat_instance.config.keys())
            for key in all_keys:
                if key not in self.SYSTEM_KEYS:
                    del strat_instance.config[key]

            # 3. Apply new profile's config
            for key, val in preset.items():
                if key == 'security':
                    continue  # Never leak security (pin_hash, devices) into live config
                if key in self.SYSTEM_KEYS:
                    # Deep merge profile overrides into system sections
                    # e.g., profile sets scoring.retain_threshold without nuking scoring.model
                    existing = strat_instance.config.get(key)
                    if isinstance(existing, dict) and isinstance(val, dict):
                        existing.update(val)
                    else:
                        strat_instance.config[key] = val
                else:
                    strat_instance.config[key] = val

            # 4. Restore any system keys the profile YAML didn't touch
            for key in self.SYSTEM_KEYS:
                if key not in strat_instance.config and key in preserved:
                    strat_instance.config[key] = preserved[key]

            # Cache the profile config for session isolation (A2.1)
            strat_instance.cache_profile_config(safe)
            logger.info(f"Profile loaded on login: {safe}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load profile {name}: {e}")
            return False

    def create_session(self, profile_name=""):
        """Create a new auth session, returning the token."""
        token = secrets.token_hex(32)
        self._active_sessions[token] = {
            "expiry": time.time() + self.SESSION_TTL,
            "profile": profile_name,
        }
        self._save_sessions()
        return token

    def validate_session(self, token):
        """Check if a session token is valid and not expired."""
        if not token:
            return False
        session = self._active_sessions.get(token)
        if not session:
            return False
        now = time.time()
        if now > session["expiry"]:
            del self._active_sessions[token]
            self._save_sessions()
            return False
        # Purge all expired sessions every 10 minutes
        if now - self._last_purge > 600:
            expired = [k for k, v in self._active_sessions.items() if now > v.get("expiry", 0)]
            for k in expired:
                del self._active_sessions[k]
            if expired:
                self._save_sessions()
                logger.debug(f"Purged {len(expired)} expired session(s)")
            self._last_purge = now
        return True

    def get_session_profile(self, token):
        """Get the profile name associated with a session token."""
        session = self._active_sessions.get(token)
        return session["profile"] if session else ""

    def delete_session(self, token):
        """Remove a session (logout)."""
        if token in self._active_sessions:
            del self._active_sessions[token]
            self._save_sessions()

    def rate_limited(self, path):
        """Check if a request path is rate-limited."""
        limit = self.RATE_LIMITS.get(path)
        if not limit:
            return False
        max_calls, window = limit
        now = time.time()
        with self._rate_lock:
            bucket = self._rate_buckets.get(path, [])
            bucket = [t for t in bucket if now - t < window]
            if len(bucket) >= max_calls:
                return True
            bucket.append(now)
            self._rate_buckets[path] = bucket
        return False

    @staticmethod
    def mask_key(key):
        """Mask API key for display: show only last 4 chars."""
        if not key or len(key) < 8:
            return key
        return '\u2022' * (len(key) - 4) + key[-4:]

    @staticmethod
    def is_masked(value):
        """Check if a value looks like a masked key (contains mask chars)."""
        return isinstance(value, str) and '\u2022' in value

    def auth_helpers_dict(self):
        """Return dict of auth helpers for routes that need them."""
        return {
            'get_session_profile': self.get_session_profile,
            'safe_name': self.safe_name,
            'profiles_dir': self.profiles_dir,
        }
