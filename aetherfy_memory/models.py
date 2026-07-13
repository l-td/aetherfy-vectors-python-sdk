"""
Models for the Aetherfy Memory SDK.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# Default vector dimension for auto-created scopes. Matches sentence-transformers
# `all-MiniLM-L6-v2`, which is also the planned T2-0 server-side default.
DEFAULT_VECTOR_SIZE = 384


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
