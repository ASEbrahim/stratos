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
        self._clients = []
        self._lock = threading.Lock()

    def register(self, wfile):
        """Register a new SSE client connection."""
        with self._lock:
            self._clients.append(wfile)
            logger.debug(f"SSE client connected ({len(self._clients)} total)")

    def unregister(self, wfile):
        """Remove a disconnected SSE client."""
        with self._lock:
            self._clients = [c for c in self._clients if c is not wfile]
            logger.debug(f"SSE client disconnected ({len(self._clients)} total)")

    def broadcast(self, event_type: str, data: dict = None):
        """Push an event to all connected SSE clients."""
        payload = json.dumps({"type": event_type, **(data or {})})
        msg = f"event: {event_type}\ndata: {payload}\n\n"
        dead = []
        with self._lock:
            for wfile in self._clients:
                try:
                    wfile.write(msg.encode())
                    wfile.flush()
                except Exception:
                    dead.append(wfile)
            for d in dead:
                self._clients = [c for c in self._clients if c is not d]

    @property
    def client_count(self):
        """Number of active SSE connections."""
        with self._lock:
            return len(self._clients)
