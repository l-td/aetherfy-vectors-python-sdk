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

#: THE LAST ADDRESS KNOWN FOR A PEER outlives the window above, and is served only when the control plane
#: could not answer (429, 5xx, unreachable): an address that was right thirty seconds ago usually still is,
#: and a call that would have worked must not fail because a lookup was refused. It is dropped the moment
#: the control plane answers that the peer is gone (404 or 409), and it is never served to a caller that
#: asked for a fresh one or to the re-lookup after an address stopped answering.
#:
#: HOW LONG AFTER A FAILED LOOKUP THE CONTROL PLANE IS NOT ASKED AGAIN: its Retry-After, kept between the
#: floor and the cap. The floor is what a failure without a Retry-After waits (a sick control plane must
#: not get a lookup per call); the cap stops one long Retry-After from pinning a stale address for minutes.
PEER_RETRY_FLOOR_SECONDS = 5
PEER_RETRY_CAP_SECONDS = 60


def retry_delay(retry_after: Optional[float]) -> float:
    """Seconds not to ask again after a failed lookup that carried this Retry-After (or none)."""
    if retry_after is None or retry_after != retry_after:  # none, or NaN
        return float(PEER_RETRY_FLOOR_SECONDS)
    return float(min(max(retry_after, 0.0), PEER_RETRY_CAP_SECONDS))


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
        self._known_urls: Dict[str, str] = {}
        self._known_list: Optional[Dict[str, str]] = None
        # key (a name, or None for the list) -> (not before, status, code) of the lookup that failed
        self._blocked: Dict[
            Optional[str], Tuple[float, Optional[int], Optional[str]]
        ] = {}

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
            self._known_urls[name] = url
            self._blocked.pop(name, None)

    def last_known_url(self, name: str) -> Optional[str]:
        with self._lock:
            return self._known_urls.get(name)

    def forget_url(self, name: str) -> None:
        """The control plane said the peer is gone: no stale address may be served for it."""
        with self._lock:
            self._known_urls.pop(name, None)
            self._blocked.pop(name, None)

    def block(
        self,
        key: Optional[str],
        seconds: float,
        status: Optional[int],
        code: Optional[str],
    ) -> None:
        with self._lock:
            self._blocked[key] = (self._clock() + seconds, status, code)

    def blocked(
        self, key: Optional[str]
    ) -> Optional[Tuple[float, Optional[int], Optional[str]]]:
        """(seconds left, status, code) of the failed lookup still being waited out, or None."""
        with self._lock:
            entry = self._blocked.get(key)
            if entry is None:
                return None
            until, status, code = entry
            left = until - self._clock()
            if left <= 0:
                del self._blocked[key]
                return None
            return left, status, code

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
            self._known_list = dict(urls)
            self._blocked.pop(None, None)

    def last_known_list(self) -> Optional[Dict[str, str]]:
        """A COPY, like get_list."""
        with self._lock:
            return dict(self._known_list) if self._known_list is not None else None


#: The one cache of this process. Replaced whole by tests; nothing else writes
#: the name.
cache = PeerCache()
