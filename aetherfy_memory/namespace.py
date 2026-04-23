"""
Namespace — a named, scoped memory bucket backed by one Qdrant collection.

A `Namespace` is the generic primitive for any agent shape (non-chat or chat).
It wraps a single underlying `AetherfyVectorsClient` collection and exposes
only the operations that make sense at the scope level: add, search, retrieve,
delete, count, schema management, and atomic clear.

All collection-lifecycle operations (create, list, exists, delete) live on
`MemoryClient` — a `Namespace` instance always points at an existing scope.
"""

from typing import Any, Dict, List, Optional, Union
import uuid

from aetherfy_vectors.client import AetherfyVectorsClient
from aetherfy_vectors.models import Filter, SearchResult
from aetherfy_vectors.schema import AnalysisResult, Schema

from .exceptions import EmbeddingNotSupportedError


class Namespace:
    """A scoped memory bucket. Obtain via `memory.namespace(name)`."""

    def __init__(self, name: str, collection_name: str, client: AetherfyVectorsClient):
        """Internal — callers use MemoryClient.namespace(name) to construct."""
        self._name = name
        self._collection = collection_name
        self._client = client

    @property
    def name(self) -> str:
        """The user-facing namespace name (without workspace or thread prefixes)."""
        return self._name

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
    ) -> str:
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

        point_id = str(id) if id is not None else uuid.uuid4().hex

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

    # ---------------------------------------------------------------------
    # Read
    # ---------------------------------------------------------------------

    def search(
        self,
        *,
        vector: List[float],
        limit: int = 10,
        offset: int = 0,
        filter: Optional[Union[Filter, Dict[str, Any]]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Semantic search within this namespace only."""
        return self._client.search(
            self._collection,
            query_vector=vector,
            limit=limit,
            offset=offset,
            query_filter=filter,
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
        )

    def retrieve(
        self,
        ids: List[Union[str, int]],
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch specific points by ID."""
        return self._client.retrieve(
            self._collection, ids, with_payload=with_payload, with_vectors=with_vectors
        )

    def count(self, *, filter: Optional[Dict[str, Any]] = None, exact: bool = True) -> int:
        """Count points in this namespace, optionally filtered."""
        return self._client.count(self._collection, count_filter=filter, exact=exact)

    # ---------------------------------------------------------------------
    # Delete
    # ---------------------------------------------------------------------

    def delete(
        self,
        selector: Union[List[Union[str, int]], Dict[str, Any]],
    ) -> bool:
        """Delete points — by ID list or by filter — without dropping the namespace."""
        return self._client.delete(self._collection, selector)

    def clear(self) -> bool:
        """Atomically drop this namespace (destroys the underlying collection).

        After `clear()`, the namespace no longer exists. Call
        `memory.create_namespace(name)` to re-create it.
        """
        return self._client.delete_collection(self._collection)

    # ---------------------------------------------------------------------
    # Schema
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

    # ---------------------------------------------------------------------
    # Analytics
    # ---------------------------------------------------------------------

    def get_analytics(self, time_range: str = "24h"):
        """Per-namespace analytics (requests, latency, storage).

        time_range: one of '1h', '24h', '7d', '30d'.
        """
        return self._client.get_collection_analytics(
            self._collection, time_range=time_range
        )

    def __repr__(self) -> str:
        return f"Namespace(name={self._name!r})"
