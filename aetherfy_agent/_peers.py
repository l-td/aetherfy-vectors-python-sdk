"""
The in-process memory of ``agent_url`` and ``agent_urls``.

It holds NAME -> URL and nothing else, for :data:`PEER_CACHE_SECONDS`. That is
the whole of what is safe to remember: a peer's address changes when the peer
is recreated, so anything that outlives the address (the peer's state, its
kind, the lookup's status) would be a copy that could disagree with the
platform. An entry that is not fresh is not served and is dropped on the read
that finds it expired.

ONLY AN ANSWER IS CACHED. A refusal (no such peer, a peer with no URL) is not
remembered, because the very next thing a caller does about it is create or
start the peer, and a cached "not found" would then be served for up to thirty
seconds after the peer exists.

THE CLOCK IS INJECTED. ``clock`` is any callable returning seconds from an
arbitrary origin; the default, :func:`time.monotonic`, is immune to wall-clock
adjustments. Tests hand in an object they move by hand, so no test ever waits
for an expiry.
"""

import threading
import time
from typing import Callable, Dict, Optional, Tuple

#: How long a looked-up address is served without asking the control plane
#: again. ``fresh=True`` skips it. Thirty seconds is the window in which a
#: recreated peer can still be reached at its old address: long enough that a
#: busy caller does one lookup per peer per half minute rather than one per
#: call, short enough that the retired-address retry in ``call_agent`` is the
#: exception and not the rule.
PEER_CACHE_SECONDS = 30


class PeerCache:
    """Name -> URL for each peer looked up by name, and one entry for the list."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._urls: Dict[str, Tuple[str, float]] = {}
        self._list: Optional[Tuple[Dict[str, str], float]] = None

    def _fresh(self, stored_at: float) -> bool:
        return self._clock() - stored_at < PEER_CACHE_SECONDS

    def get_url(self, name: str) -> Optional[str]:
        with self._lock:
            entry = self._urls.get(name)
            if entry is None:
                return None
            url, stored_at = entry
            if self._fresh(stored_at):
                return url
            del self._urls[name]
            return None

    def put_url(self, name: str, url: str) -> None:
        with self._lock:
            self._urls[name] = (url, self._clock())

    def get_list(self) -> Optional[Dict[str, str]]:
        """A COPY: a caller who edits what it was handed must not edit the cache."""
        with self._lock:
            if self._list is None:
                return None
            urls, stored_at = self._list
            if self._fresh(stored_at):
                return dict(urls)
            self._list = None
            return None

    def put_list(self, urls: Dict[str, str]) -> None:
        with self._lock:
            self._list = (dict(urls), self._clock())


#: The one cache of this process. Replaced whole by tests; nothing else writes
#: the name.
cache = PeerCache()
