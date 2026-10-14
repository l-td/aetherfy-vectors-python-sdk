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

EVERY THREAD IN A WORKSPACE SHARES ONE COLLECTION (`__threads__`), and a
thread is a FILTER over it: `thread_id` is stamped on every point and every
read and write this class issues carries the matching clause. The clause is
assembled here, never from a caller-supplied string, because the proxy
forwards filters verbatim and a misspelled key fails OPEN -- a successful
response with every thread's points in it. A caller's own filter is
COMBINED with the thread clause, never substituted for it.

One MARKER point per thread (written by `create_thread`) is what makes an
empty thread exist. It is not a message and must never read as one, so
`history`, `iter_history`, `search`, `count`, `iter` and the filtered
`delete` all exclude it explicitly.
"""

import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union

from aetherfy_vectors.client import AetherfyVectorsClient
from aetherfy_vectors.exceptions import PointNotFoundError
from aetherfy_vectors.models import Filter
from aetherfy_vectors.utils import serialize_filter

from .exceptions import EmbeddingNotSupportedError
from .models import THREAD_ID_KEY, THREAD_MARKER_KEY, Message
from .scope import _Scope


class Thread(_Scope):
    """A conversation. Obtain via `memory.thread(id)`."""

    # Thread payload top-level reserved fields — a Thread payload is
    # `{role, content, ts, thread_id, metadata}`, so those are the names that
    # shouldn't appear in a user metadata partial. See `_Scope.merge_metadata`.
    # `thread_id` and the marker key are in here for the same reason the
    # other three are: user metadata must not be able to shadow a key the
    # scope itself depends on. They are ordinary customer payload keys, NOT
    # members of vectordb's reserved/attested tier — the server has no source
    # of truth to re-stamp a client-chosen thread id from.
    _RESERVED_KEYS: frozenset = frozenset(
        {"role", "content", "ts", THREAD_ID_KEY, THREAD_MARKER_KEY}
    )

    def __init__(
        self, thread_id: str, collection_name: str, client: AetherfyVectorsClient
    ):
        super().__init__(thread_id, collection_name, client)

    @property
    def id(self) -> str:
        """The thread id (same as `name`; provided for API parity)."""
        return self._name

    # ---------------------------------------------------------------------
    # Scoping — the thread clause
    # ---------------------------------------------------------------------

    def _thread_clause(self) -> Dict[str, Any]:
        """The one condition that scopes an operation to this thread."""
        return {"key": THREAD_ID_KEY, "match": {"value": self._name}}

    @staticmethod
    def _marker_clause() -> Dict[str, Any]:
        """Matches the thread's marker point and nothing else."""
        return {"key": THREAD_MARKER_KEY, "match": {"value": True}}

    def _combine_filter(
        self, filter: Optional[Union[Filter, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Combine a caller filter with the thread clause. Never replaces it.

        The thread clause always lands in `must`, and the marker exclusion
        always lands in `must_not`; a caller's clauses are APPENDED to those
        arrays. Since Aetherfy composes the three clause arrays as a
        conjunction (everything in `must` holds AND at least one `should`
        holds AND nothing in `must_not` holds), no caller clause — `should`
        included — can widen the result past this thread.

        `serialize_filter` does the `Filter` → dict normalisation and
        rejects a clause name outside must / must_not / should, so a caller
        typo at the clause level still fails loudly rather than being
        merged in as an unknown key.
        """
        combined: Dict[str, Any] = {
            "must": [self._thread_clause()],
            "must_not": [self._marker_clause()],
        }
        caller = serialize_filter(filter, "Thread filter")
        if caller:
            combined["must"].extend(caller.get("must") or [])
            combined["must_not"].extend(caller.get("must_not") or [])
            if caller.get("should"):
                combined["should"] = list(caller["should"])
        return combined

    def _own_filter(self) -> Dict[str, Any]:
        """Every row of this thread, marker included. Used by `clear`."""
        return {"must": [self._thread_clause()]}

    def _reads_payload_to_scope(self) -> bool:
        # A thread's point ids are unique within the shared collection, not
        # within the thread, so identifying our own points means reading
        # `thread_id` off the payload.
        return True

    @staticmethod
    def _is_marker(point: Dict[str, Any]) -> bool:
        """True iff `point` is a thread's marker rather than a message.

        The filter already excludes markers server-side. This is the second,
        independent guard, and it earns its place because the FIRST one fails
        open: the proxy forwards a filter verbatim and never validates it, so
        a mistyped clause returns a successful response with unfiltered
        results. A marker reading as a message would put an empty `role` and
        an empty `content` into a caller's conversation.
        """
        return (point.get("payload") or {}).get(THREAD_MARKER_KEY) is True

    def _owns(self, point: Dict[str, Any]) -> bool:
        """True iff `point` is one of THIS thread's messages."""
        payload = point.get("payload") or {}
        return (
            payload.get(THREAD_ID_KEY) == self._name
            and payload.get(THREAD_MARKER_KEY) is not True
        )

    def _retain_owned(
        self, points: List[Dict[str, Any]], *, with_payload: bool
    ) -> List[Dict[str, Any]]:
        kept = [p for p in points if self._owns(p)]
        if with_payload:
            return kept
        # The payload was fetched only to scope the read; the caller asked
        # not to see it.
        return [{k: v for k, v in p.items() if k != "payload"} for p in kept]

    def _point_selector(self, ids: List[Union[str, int]]) -> Dict[str, Any]:
        """Address these ids AND this thread, in one request.

        `has_id` is a first-class Qdrant condition — it is in the pinned
        client's generated schema (@qdrant/js-client-rest 1.15.0, the
        version the fleet runs) alongside FieldCondition in the Condition
        union. That matters because the proxy forwards a filter verbatim
        and an unrecognised key would quietly do nothing: here, silently
        dropping the `has_id` clause would widen a single-point delete to
        the whole thread. It is not an unverified guess.

        Scoping this way rather than checking ids client-side first means
        the ENGINE enforces the boundary, so a later caller who reaches
        past the SDK cannot bypass it, and the round trip that the check
        used to cost is gone.
        """
        return {
            "must": [self._thread_clause(), {"has_id": list(ids)}],
            "must_not": [self._marker_clause()],
        }

    def _owned_ids(self, ids: List[Union[str, int]]) -> List[Union[str, int]]:
        if not ids:
            return []
        points = self._client.retrieve(
            self._collection, ids, with_payload=True, with_vectors=False
        )
        return [p["id"] for p in points if self._owns(p)]

    def _assert_owns(self, ids: List[Union[str, int]]) -> None:
        """Refuse a point id belonging to another thread.

        This one DOES cost a read, and deliberately. The payload endpoints
        accept a filter, so the metadata writers could scope themselves the
        way `delete` now does — but a filter that matches nothing is a
        SUCCESS, and `merge_metadata` / `delete_metadata_keys` are
        documented to raise PointNotFoundError when the point is not there.
        Scoping them by filter would turn a write to a foreign or missing
        id into a silent no-op reported as success. The round trip buys the
        error. `delete` has no such contract to lose: deleting an id that
        is not there was always a no-op that returns True.
        """
        owned = set(self._owned_ids(ids))
        for point_id in ids:
            if point_id not in owned:
                raise PointNotFoundError(str(point_id), self._collection)

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

        # The thread clause is stamped LAST so nothing a caller supplied can
        # displace it — `metadata` is nested a level down and cannot reach
        # this key, but stamping last is what makes that structural rather
        # than incidental.
        payload = {**msg.to_payload(), THREAD_ID_KEY: self._name}
        self._client.upsert(
            self._collection,
            [{"id": point_id, "vector": vector, "payload": payload}],
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
            payload = {**msg.to_payload(), THREAD_ID_KEY: self._name}
            points.append({"id": msg.id, "vector": vector, "payload": payload})

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
        #
        # The cap SURVIVES the move to a shared collection. It was never
        # doing the filter's job: even when a thread had a collection to
        # itself this scroll was already thread-scoped, and the cap was what
        # bounded the client-side sort of an arbitrarily long thread. The
        # filter narrows the same scroll to the same rows it used to see, so
        # removing the cap now would make `history(limit=50)` pull an
        # unbounded thread into memory. It still truncates silently past
        # 5000 messages — that is `iter_history`'s job, which is why that
        # method exists.
        cap = min(max(limit * 20, 100), 5000)

        result = self._client.scroll(
            self._collection,
            limit=cap,
            with_payload=True,
            with_vectors=False,
            scroll_filter=self._combine_filter(None),
        )
        points = [p for p in result["points"] if not self._is_marker(p)]
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
            if p.get("payload") and not self._is_marker(p)
        ]
        messages = [m for m in messages if m.ts is not None]
        messages.sort(key=lambda m: m.ts or 0.0, reverse=(order == "desc"))
        for m in messages:
            yield m

    # ---------------------------------------------------------------------
    # Delete
    # ---------------------------------------------------------------------

    def clear(self) -> bool:
        """Atomically drop this thread, leaving every sibling thread intact.

        Keeps the meaning it has always had — after `clear()` the thread no
        longer exists and `memory.create_thread(id)` re-creates it — but it
        can no longer be a collection drop: the collection now holds every
        OTHER thread in the workspace too. It is a delete-by-filter on this
        thread's rows, marker included (dropping the marker is what makes
        the thread stop existing).
        """
        return self._client.delete(self._collection, self._own_filter())

    def __repr__(self) -> str:
        return f"Thread(id={self._name!r})"
