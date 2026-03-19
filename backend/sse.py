"""
Server-Sent Events (SSE) manager for StratOS.

Manages client connections and event broadcasting for real-time
dashboard updates during scans.

Extracted from StratOS class (Sprint 4, A1.2).
"""

import json
import logging
import threading

logger = logging.getLogger("STRAT_OS")


class SSEManager:
    """Manages SSE client connections and broadcasts."""

    def __init__(self):
        self._clients = []  # list of (wfile, profile_id)
        self._lock = threading.Lock()

    def register(self, wfile, profile_id=0):
        """Register a new SSE client connection with optional profile scope."""
        with self._lock:
            self._clients.append((wfile, profile_id))
            logger.debug(f"SSE client connected (pid={profile_id}, {len(self._clients)} total)")

    def unregister(self, wfile):
        """Remove a disconnected SSE client."""
        with self._lock:
            self._clients = [(w, p) for w, p in self._clients if w is not wfile]
            logger.debug(f"SSE client disconnected ({len(self._clients)} total)")

    def broadcast(self, event_type: str, data: dict = None, profile_id=None):
        """Push an event to connected SSE clients.

        If profile_id is set, only send to clients with matching profile_id
        or profile_id=0 (wildcard/legacy). If None, send to all (universal events).
        """
        payload = json.dumps({"type": event_type, **(data or {})})
        msg = f"event: {event_type}\ndata: {payload}\n\n"
        dead = []
        with self._lock:
            for wfile, client_pid in self._clients:
                # Skip if this is a profile-scoped broadcast and client doesn't match
                if profile_id is not None and client_pid != 0 and client_pid != profile_id:
                    continue
                try:
                    wfile.write(msg.encode())
                    wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    dead.append(wfile)
                except Exception as e:
                    logger.warning(f"SSE broadcast unexpected error: {e}")
                    dead.append(wfile)
            for d in dead:
                self._clients = [(w, p) for w, p in self._clients if w is not d]

    def heartbeat(self, wfile) -> bool:
        """Send a heartbeat comment to one SSE client under the write lock.

        Returns True on success, False if the client is dead (caller should
        break out of the keep-alive loop so finally: unregister runs).
        """
        with self._lock:
            try:
                wfile.write(b": heartbeat\n\n")
                wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                return False

    @property
    def client_count(self):
        """Number of active SSE connections."""
        with self._lock:
            return len(self._clients)
