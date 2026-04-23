"""
Models for the Aetherfy Memory SDK.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    id: Optional[str] = None
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
        """Reconstruct a Message from a retrieved Qdrant point."""
        payload = point.get("payload") or {}
        return cls(
            id=str(point.get("id")) if point.get("id") is not None else None,
            role=payload.get("role", ""),
            content=payload.get("content", ""),
            ts=payload.get("ts"),
            vector=point.get("vector"),
            metadata=payload.get("metadata") or {},
        )
