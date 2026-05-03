"""
Thread — a conversation-shaped specialization of Namespace.

A `Thread` is a `Namespace` whose payloads follow a `{role, content, ts, metadata}`
schema and which exposes `history(limit)` for ordered retrieval of messages.

Every add from a Thread writes the three reserved fields on the point payload:
`role` (e.g. "user" / "assistant" / "system"), `content` (the raw text), and
`ts` (Unix timestamp, used to order history).
"""

import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

from aetherfy_vectors.client import AetherfyVectorsClient

from .exceptions import EmbeddingNotSupportedError
from .models import Message
from .namespace import Namespace


class Thread(Namespace):
    """A conversation. Obtain via `memory.thread(id)`."""

    def __init__(self, thread_id: str, collection_name: str, client: AetherfyVectorsClient):
        super().__init__(thread_id, collection_name, client)

    @property
    def id(self) -> str:
        """The thread id (same as `name` for parity with Namespace)."""
        return self._name

    # ---------------------------------------------------------------------
    # Write
    # ---------------------------------------------------------------------

    def add(
        self,
        *,
        role: str,
        content: str,
        vector: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[Union[str, int]] = None,
        ts: Optional[float] = None,
    ) -> str:
        """Append a message to the thread.

        Args:
            role: "user", "assistant", "system", or any agent-defined role.
            content: The message text.
            vector: Embedding for this message. Required today; server-side
                embedding (text-only) lands in DX_ROADMAP T2-0.
            metadata: Optional extra fields; stored nested so it cannot
                shadow `role`/`content`/`ts`.
            id: Optional point ID. UUID4 if omitted.
            ts: Optional Unix timestamp. Wall clock if omitted.

        Returns:
            The point ID used for this message.
        """
        if vector is None:
            raise EmbeddingNotSupportedError()

        if not isinstance(role, str) or not role:
            raise ValueError("role must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError("content must be a string")

        msg = Message(
            role=role,
            content=content,
            vector=vector,
            id=str(id) if id is not None else uuid.uuid4().hex,
            ts=ts if ts is not None else time.time(),
            metadata=metadata or {},
        )

        self._client.upsert(
            self._collection,
            [{"id": msg.id, "vector": vector, "payload": msg.to_payload()}],
        )
        return msg.id

    # ---------------------------------------------------------------------
    # Read — ordered history
    # ---------------------------------------------------------------------

    def history(self, limit: int = 50, *, order: str = "asc") -> List[Message]:
        """Return messages ordered by timestamp.

        Args:
            limit: Maximum messages to return (default 50).
            order: "asc" (oldest first, default — natural reading order) or
                "desc" (newest first, useful for paginating recent messages).

        Returns:
            List of Message objects. Payload-only by default; vectors are
            not re-fetched for history reads.
        """
        if order not in ("asc", "desc"):
            raise ValueError("order must be 'asc' or 'desc'")
        if limit <= 0:
            raise ValueError("limit must be positive")

        # Qdrant's scroll API has no server-side order_by over payload fields
        # without an index; for the MVP we pull up to a bounded cap and sort
        # client-side by `ts`. Longer histories can paginate via `offset` in a
        # future iteration.
        cap = min(max(limit * 20, 100), 5000)

        result = self._client.scroll(
            self._collection, limit=cap, with_payload=True, with_vectors=False
        )
        points = result["points"]
        messages = [Message.from_point(p) for p in points if p.get("payload")]

        # Drop messages without a ts (shouldn't happen for SDK-written points
        # but might for raw AetherfyVectorsClient writes to the same collection).
        messages = [m for m in messages if m.ts is not None]

        reverse = order == "desc"
        messages.sort(key=lambda m: m.ts or 0.0, reverse=reverse)

        return messages[:limit]

    def iter_history(self, *, order: str = "asc") -> Iterator[Message]:
        """Iterate all messages in this thread, sorted by timestamp.

        Unlike `history(limit)` which caps at 5000 for the client-side sort,
        ``iter_history()`` walks the entire thread by paging through the
        underlying scroll iterator and sorting in memory. For threads larger
        than 5000 messages the in-memory sort can be expensive; use
        ``history(limit)`` if you only need the most recent slice.

        Args:
            order: 'asc' (oldest first) or 'desc' (newest first).

        Yields:
            Each Message in the thread, in the requested order.
        """
        if order not in ("asc", "desc"):
            raise ValueError("order must be 'asc' or 'desc'")

        # Reuse Namespace.iter for paging — same scroll_iter under the hood.
        # Skip points without a payload or without a ts (matches history()).
        messages = [
            Message.from_point(p)
            for p in self.iter(with_payload=True, with_vectors=False)
            if p.get("payload")
        ]
        messages = [m for m in messages if m.ts is not None]
        messages.sort(key=lambda m: m.ts or 0.0, reverse=(order == "desc"))
        for m in messages:
            yield m

    def __repr__(self) -> str:
        return f"Thread(id={self._name!r})"
