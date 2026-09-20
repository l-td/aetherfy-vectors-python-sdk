"""
Namespace — a named, scoped memory bucket backed by one Qdrant collection.

A `Namespace` is the generic primitive for any agent shape (non-chat or chat).
It wraps a single underlying `AetherfyVectorsClient` collection and exposes
the scope operations (read / delete / metadata — shared via `_Scope`), the
collection-level schema surface (which only a Namespace owns outright), and
its own generic-memory write API (`add` / `add_many`).

All collection-lifecycle operations (create, list, exists, delete) live on
`MemoryClient` — a `Namespace` instance always points at an existing scope.
"""

from typing import Any, Dict, List, Optional, Union
import uuid

from aetherfy_vectors.schema import AnalysisResult, Schema

from .exceptions import EmbeddingNotSupportedError
from .scope import _Scope


class Namespace(_Scope):
    """A scoped memory bucket. Obtain via `memory.namespace(name)`."""

    # Reserved payload-top-level keys — a Namespace payload is
    # `{text?, metadata?}`, so `text` is the name that can't appear in a
    # user metadata partial. See `_Scope.merge_metadata`.
    _RESERVED_KEYS: frozenset = frozenset({"text"})

    # ---------------------------------------------------------------------
    # Write
    # ---------------------------------------------------------------------

    def add(
        self,
        *,
        vector: Optional[List[float]] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[Union[str, int]] = None,
    ) -> Union[str, int]:
        """Add a memory to this namespace.

        Args:
            vector: The embedding for this memory. Required today; when T2-0
                (server-side embedding) ships, passing only `text` will work.
            text: Original text; stored in the payload under `text`. Optional
                when `vector` is provided, required when it isn't.
            metadata: Arbitrary user metadata; stored under `metadata` in the
                payload so it can't shadow reserved fields.
            id: Optional point ID. If omitted, a UUID4 is generated.

        Returns:
            The point ID used for this memory.

        Raises:
            EmbeddingNotSupportedError: if `vector` is None.
        """
        if vector is None:
            raise EmbeddingNotSupportedError()

        # Pass an explicit id through as-authored — an int stays an int (a
        # valid unsigned-integer point id). str()-coercing it would turn 42
        # into "42", a numeric string the ingress validator rejects. When
        # omitted, default to a canonical uuid4. A non-int/non-uuid explicit
        # id is left for validate_point_id (in client.upsert) to reject loudly.
        point_id = id if id is not None else str(uuid.uuid4())

        payload: Dict[str, Any] = {}
        if text is not None:
            payload["text"] = text
        if metadata:
            payload["metadata"] = metadata

        self._client.upsert(
            self._collection,
            [{"id": point_id, "vector": vector, "payload": payload}],
        )
        return point_id

    def add_many(self, items: List[Dict[str, Any]]) -> List[Union[str, int]]:
        """Add many memories in a single round trip.

        Each item is a dict with the same shape as ``add`` keyword args:
        ``{"vector": [...], "text": ..., "metadata": ..., "id": ...}``.
        ``vector`` is required per item; ``id`` is generated as a
        canonical UUID4 string when omitted (matches what the server
        returns on read). Returns IDs in input order.

        The server handles streaming-chunking of the resulting upsert,
        so this method does NOT itself chunk — pass however many items
        you want.

        Empty input returns ``[]`` without a round trip (degenerate-input
        tolerance for dynamically-built lists).

        Args:
            items: List of memory dicts. Each must have a non-None ``vector``.

        Returns:
            List of point IDs in the same order as ``items``.

        Raises:
            TypeError: if ``items`` is not a list.
            EmbeddingNotSupportedError: if any item lacks a ``vector``;
                the message identifies the offending index.
        """
        if not isinstance(items, list):
            raise TypeError("add_many requires a list of memory dicts")
        if not items:
            return []

        points: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            vector = item.get("vector")
            if vector is None:
                raise EmbeddingNotSupportedError(f"add_many[{idx}]")

            # Explicit id as-authored (an int stays an int); default uuid4 when
            # omitted. See the note in add() — no blanket str() coercion.
            point_id = item["id"] if item.get("id") is not None else str(uuid.uuid4())

            payload: Dict[str, Any] = {}
            if item.get("text") is not None:
                payload["text"] = item["text"]
            if item.get("metadata"):
                payload["metadata"] = item["metadata"]

            points.append({"id": point_id, "vector": vector, "payload": payload})

        self._client.upsert(self._collection, points)
        return [p["id"] for p in points]

    # ---------------------------------------------------------------------
    # Schema
    #
    # A payload schema belongs to a COLLECTION, and a Namespace is the only
    # scope that owns one outright. These used to live on `_Scope`, which
    # gave a Thread the same six methods; now that every thread shares one
    # collection, `thread('a').set_schema(...)` would have imposed a schema
    # on every other thread in the workspace. They are declared here rather
    # than left on the base and documented away.
    # ---------------------------------------------------------------------

    def get_schema(self) -> Optional[Schema]:
        """Return the payload schema for this namespace, or None if unset."""
        return self._client.get_schema(self._collection)

    def set_schema(
        self,
        schema: Schema,
        *,
        enforcement: str = "strict",
        description: Optional[str] = None,
    ) -> str:
        """Set or update the payload schema. `enforcement` is 'strict', 'warn', or 'off'.

        Returns the new schema ETag.
        """
        return self._client.set_schema(
            self._collection, schema, enforcement=enforcement, description=description
        )

    def delete_schema(self) -> bool:
        """Remove the payload schema from this namespace."""
        return self._client.delete_schema(self._collection)

    def analyze_schema(self, sample_size: int = 1000) -> AnalysisResult:
        """Infer a suggested schema from a sample of existing points."""
        return self._client.analyze_schema(self._collection, sample_size=sample_size)

    def refresh_schema(self) -> None:
        """Bust the local schema cache for this namespace."""
        self._client.refresh_schema(self._collection)

    def clear_schema_cache(self) -> None:
        """Clear the client-side schema cache for this namespace."""
        self._client.clear_schema_cache(self._collection)

    def __repr__(self) -> str:
        return f"Namespace(name={self._name!r})"
