"""
Models for the Aetherfy Memory SDK.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# Default vector dimension for auto-created scopes. Matches sentence-transformers
# `all-MiniLM-L6-v2`, which is also the planned T2-0 server-side default.
DEFAULT_VECTOR_SIZE = 384

# ---------------------------------------------------------------------------
# The threads collection
#
# Every thread in a workspace lives in ONE collection, with `thread_id` as a
# payload key — Qdrant's documented multitenancy shape. A collection per
# conversation is the shape it warns against: one Aetherfy collection is one
# physical Qdrant collection with its own HNSW graph and a minimum of two
# segments, per region, and it counts against plans.max_collections (Free 3),
# so a collection per conversation capped a Free account at three
# conversations ever.
#
# The name is legal SERVER-side — vectordb's scoping layer accepts
# `[a-zA-Z0-9_-]{1,100}` (utils/collectionScoping.js), no dots — and
# unreachable from the user-facing name regex in client.py, which requires a
# leading letter or digit. So no namespace can ever collide with it.
# ---------------------------------------------------------------------------
THREADS_COLLECTION = "__threads__"

# The tenant key. A NORMAL customer payload key, not a reserved/attested one:
# the attested tier (`__aetherfy_agent_id`, `__aetherfy_deployment_id`) is for
# values the server re-stamps from the credential and can therefore vouch for,
# and a thread id is client-chosen with no server-side source of truth. What
# keeps a caller from forging it is that the memory layer owns every clause
# that mentions it — it is never assembled from a caller-supplied string.
THREAD_ID_KEY = "thread_id"

# The marker key. One marker point per thread, written at create time, is what
# makes an EMPTY thread exist: with rows keyed by payload alone, a thread with
# no messages would be indistinguishable from one that was never created, and
# thread_exists / list_threads / ThreadAlreadyExistsError would all break.
# Markers are never messages — every read path excludes them explicitly.
THREAD_MARKER_KEY = "thread_marker"


@dataclass
class Message:
    """A single message within a conversation Thread.

    `ts` is a Unix timestamp used to order `thread.history()` results.
    The SDK sets it to the wall-clock time at `add` unless the caller
    provides one explicitly (useful for backfilling historical messages).
    """

    role: str
    content: str
    vector: Optional[List[float]] = None
    # A point id is an unsigned integer or a UUID string — an id authored on
    # `add` is carried through as-is (an int stays an int). `from_point`
    # reconstructs the read side (see its note).
    id: Optional[Union[str, int]] = None
    ts: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Flatten into a Qdrant payload dict."""
        payload: Dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "ts": self.ts,
        }
        if self.metadata:
            # User metadata is nested under a key so it can't shadow
            # the reserved role/content/ts fields.
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_point(cls, point: Dict[str, Any]) -> "Message":
        """Reconstruct a Message from a retrieved Qdrant point.

        The id is preserved as stored — an integer point id comes back an
        ``int``, a UUID a ``str``. str()-coercing here would make
        ``add(id=42)`` then ``history()`` return id ``"42"``, breaking the
        caller's ``msg.id == 42`` check (the read-side twin of the write-side
        str() bug).
        """
        payload = point.get("payload") or {}
        return cls(
            id=point.get("id"),
            role=payload.get("role", ""),
            content=payload.get("content", ""),
            ts=payload.get("ts"),
            vector=point.get("vector"),
            metadata=payload.get("metadata") or {},
        )
