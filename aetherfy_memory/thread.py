"""
Thread — a conversation-shaped scope.

A `Thread` is a `_Scope` whose payloads follow a `{role, content, ts, metadata}`
schema and which exposes `history(limit)` for ordered retrieval of messages.

Every add from a Thread writes the three reserved fields on the point payload:
`role` (e.g. "user" / "assistant" / "system"), `content` (the raw text), and
`ts` (Unix timestamp, used to order history).

`Thread` is NOT a subclass of `Namespace`: their write APIs differ (a message
requires `role`/`content`; a memory uses `text`), so a Thread is not
add-substitutable for a Namespace. Both share the read/scope surface via
`_Scope`.
"""

import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

from aetherfy_vectors.client import AetherfyVectorsClient

from .exceptions import EmbeddingNotSupportedError
from .models import Message
from .scope import _Scope


class Thread(_Scope):
    """A conversation. Obtain via `memory.thread(id)`."""

    # Thread payload top-level reserved fields — a Thread payload is
    # `{role, content, ts, metadata}`, so role/content/ts are the names that
    # shouldn't appear in a user metadata partial. See `_Scope.merge_metadata`.
    _RESERVED_KEYS: frozenset = frozenset({"role", "content", "ts"})

    def __init__(
        self, thread_id: str, collection_name: str, client: AetherfyVectorsClient
    ):
        super().__init__(thread_id, collection_name, client)

    @property
    def id(self) -> str:
        """The thread id (same as `name`; provided for API parity)."""
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
    ) -> Union[str, int]:
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

        # Explicit id as-authored (an int stays an int, a valid point id);
        # default uuid4 when omitted. No blanket str() — see Namespace.add.
        point_id: Union[str, int] = id if id is not None else str(uuid.uuid4())
        msg = Message(
            role=role,
            content=content,
            vector=vector,
            id=point_id,
            ts=ts if ts is not None else time.time(),
            metadata=metadata or {},
        )

        self._client.upsert(
            self._collection,
            [{"id": point_id, "vector": vector, "payload": msg.to_payload()}],
        )
        return point_id

    def append_many(self, messages: List[Dict[str, Any]]) -> List[Union[str, int]]:
        """Append many messages in a single round trip.

        Each message is a dict with the same shape as ``add`` keyword
        args: ``{"role": "...", "content": "...", "vector": [...],
        "metadata": ..., "id": ..., "ts": ...}``. ``vector``, non-empty
        ``role``, and string ``content`` are required per message.
        Missing IDs get a canonical UUID4 per message; missing ``ts`` gets
        ``time.time()`` per message (each gets its own — NOT one shared
        timestamp, otherwise history ordering for messages appended in
        the same call would be undefined).

        Returns IDs in input order. Empty input returns ``[]`` without
        a round trip. Server handles streaming-chunking; this method
        does not chunk client-side.

        Args:
            messages: List of message dicts.

        Returns:
            List of point IDs in the same order as ``messages``.

        Raises:
            TypeError: if ``messages`` is not a list.
            EmbeddingNotSupportedError / ValueError: with the offending
                index in the message.
        """
        if not isinstance(messages, list):
            raise TypeError("append_many requires a list of message dicts")
        if not messages:
            return []

        points: List[Dict[str, Any]] = []
        for idx, m in enumerate(messages):
            vector = m.get("vector")
            if vector is None:
                raise EmbeddingNotSupportedError(f"append_many[{idx}]")
            role = m.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError(f"append_many[{idx}]: role must be a non-empty string")
            content = m.get("content")
            if not isinstance(content, str):
                raise ValueError(f"append_many[{idx}]: content must be a string")

            msg = Message(
                role=role,
                content=content,
                vector=vector,
                id=m["id"] if m.get("id") is not None else str(uuid.uuid4()),
                ts=m["ts"] if m.get("ts") is not None else time.time(),
                metadata=m.get("metadata") or {},
            )
            points.append({"id": msg.id, "vector": vector, "payload": msg.to_payload()})

        self._client.upsert(self._collection, points)
        return [p["id"] for p in points]

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

        # Reuse _Scope.iter for paging — same scroll_iter under the hood.
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
