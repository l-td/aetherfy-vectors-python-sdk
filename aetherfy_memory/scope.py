"""
_Scope — the shared base for Namespace and Thread.

Holds every operation that behaves identically for both scope shapes:
read (search / retrieve / count / iter), delete / clear, schema management,
analytics, and the payload-metadata helpers. The two *write* APIs differ by
shape — a Namespace stores a generic memory (`{text?, metadata?}`), a Thread
stores a conversation message (`{role, content, ts, metadata?}`) — so `add`
(and the batch writers) live on the subclasses, not here. That is why
`Thread` is NOT a subclass of `Namespace`: it is not add-substitutable for
one. Both are `_Scope`s that share the substitutable surface.
"""

from typing import Any, Dict, Iterator, List, Optional, Union

from aetherfy_vectors.client import AetherfyVectorsClient
from aetherfy_vectors.exceptions import (
    AetherfyVectorsException,
    CollectionNotFoundError,
    PointNotFoundError,
)
from aetherfy_vectors.models import Filter, SearchResult
from aetherfy_vectors.schema import AnalysisResult, Schema


class _Scope:
    """Internal base for Namespace and Thread. Not instantiated directly."""

    # Reserved payload-top-level keys for this scope shape. Subclasses set
    # their own ({text} for Namespace; {role, content, ts} for Thread). Used
    # as the local guard for merge_metadata / delete_metadata_keys to refuse
    # partials whose keys would mirror a reserved top-level field name.
    _RESERVED_KEYS: frozenset = frozenset()

    def __init__(self, name: str, collection_name: str, client: AetherfyVectorsClient):
        """Internal — callers construct via MemoryClient.namespace / .thread."""
        self._name = name
        self._collection = collection_name
        self._client = client

    @property
    def name(self) -> str:
        """The user-facing scope name (without workspace or thread prefixes)."""
        return self._name

    # ---------------------------------------------------------------------
    # Payload metadata
    # ---------------------------------------------------------------------

    def set_metadata(
        self,
        id: Union[str, int],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Replace the entire metadata sub-key of an existing memory.

        ``set_metadata({tag: 'x'})`` nukes every other key. Use
        ``merge_metadata`` if you want additive updates that preserve
        existing keys.

        Atomically writes ``payload.metadata = metadata``. Reserved fields
        (``text`` for Namespace, plus ``role``/``content``/``ts`` for Thread)
        are untouched. To merge into existing metadata, retrieve + merge +
        ``set_metadata`` explicitly:

            current = ns.retrieve([id])[0]['payload'].get('metadata', {})
            current.update({'reviewed': True})
            ns.set_metadata(id, current)

        The non-atomic compose pattern is intentional — it keeps races
        visible at the call site rather than hidden inside an SDK helper.

        Args:
            id: Point ID of the memory to update.
            metadata: New metadata object. Replaces any existing metadata.

        Returns:
            Server response from the underlying set_payload call.
        """
        return self._client.set_payload(
            self._collection,
            payload={"metadata": metadata},
            points=[id],
        )

    def merge_metadata(
        self,
        id: Union[str, int],
        partial: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Additive merge into existing metadata.

        ``merge_metadata({tag: 'x'})`` adds/updates the listed keys and
        leaves every other key untouched. Use ``set_metadata`` if you
        want to fully replace the metadata sub-key. Concurrent patches
        to different keys all land atomically; concurrent writes to the
        same key resolve via last-writer-wins per the storage operation
        order. Raises ``PointNotFoundError`` if the point doesn't exist.

        Reserved keys (``text`` on Namespace; ``role``, ``content``,
        ``ts`` on Thread) cannot appear in the partial — raises a
        local ``ValueError`` before the request is sent.
        """
        if not isinstance(partial, dict):
            raise TypeError("partial must be a dict")
        bad = [k for k in partial if k in self._RESERVED_KEYS]
        if bad:
            raise ValueError(
                f"Reserved keys cannot appear in metadata partial: {sorted(bad)}"
            )
        try:
            return self._client.set_payload(
                self._collection,
                payload=partial,
                points=[id],
                key="metadata",
            )
        except AetherfyVectorsException as e:
            if e.status_code == 404 and not isinstance(
                e, (PointNotFoundError, CollectionNotFoundError)
            ):
                raise PointNotFoundError(str(id), self._collection) from e
            raise

    def delete_metadata_keys(
        self,
        id: Union[str, int],
        keys: List[str],
    ) -> Dict[str, Any]:
        """Removes the listed keys from metadata.

        Keys not in the list are left untouched. Raises
        ``PointNotFoundError`` if the point doesn't exist.

        Reserved keys (``text`` on Namespace; ``role``, ``content``,
        ``ts`` on Thread) cannot appear in the keys list — raises a
        local ``ValueError`` before the request is sent.
        """
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise TypeError("keys must be a list of strings")
        bad = [k for k in keys if k in self._RESERVED_KEYS]
        if bad:
            raise ValueError(
                f"Reserved keys cannot appear in delete keys list: {sorted(bad)}"
            )
        dotted = [f"metadata.{k}" for k in keys]
        try:
            return self._client.delete_payload(
                self._collection,
                keys=dotted,
                points=[id],
            )
        except AetherfyVectorsException as e:
            if e.status_code == 404 and not isinstance(
                e, (PointNotFoundError, CollectionNotFoundError)
            ):
                raise PointNotFoundError(str(id), self._collection) from e
            raise

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
        """Semantic search within this scope only."""
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

    def count(
        self, *, filter: Optional[Dict[str, Any]] = None, exact: bool = True
    ) -> int:
        """Count points in this scope, optionally filtered."""
        return self._client.count(self._collection, count_filter=filter, exact=exact)

    def iter(
        self,
        *,
        batch_size: int = 256,
        filter: Optional[Union[Filter, Dict[str, Any]]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate all points in this scope.

        Yields each point one at a time, paging transparently through the
        underlying scroll_iter. Returns cleanly when the scope is exhausted.
        Use this for archival, export, or batch-enrichment workflows that
        exceed what `search` and `retrieve` cover.

        Args:
            batch_size: Points per server round-trip (default 256, capped at
                1000 by the server).
            filter: Optional payload filter, same shape as `search`.
            with_payload: Include point payloads (default True).
            with_vectors: Include vectors (default False; large).

        Yields:
            Each point dict from the scope, in unspecified order.
        """
        yield from self._client.scroll_iter(
            self._collection,
            batch_size=batch_size,
            scroll_filter=filter,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

    # ---------------------------------------------------------------------
    # Delete
    # ---------------------------------------------------------------------

    def delete(
        self,
        selector: Union[List[Union[str, int]], Dict[str, Any]],
    ) -> bool:
        """Delete points — by ID list or by filter — without dropping the scope."""
        return self._client.delete(self._collection, selector)

    def clear(self) -> bool:
        """Atomically drop this scope (destroys the underlying collection).

        After `clear()`, the scope no longer exists. Re-create it via
        `memory.create_namespace(name)` / `memory.create_thread(id)`.
        """
        return self._client.delete_collection(self._collection)

    # ---------------------------------------------------------------------
    # Schema
    # ---------------------------------------------------------------------

    def get_schema(self) -> Optional[Schema]:
        """Return the payload schema for this scope, or None if unset."""
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
        """Remove the payload schema from this scope."""
        return self._client.delete_schema(self._collection)

    def analyze_schema(self, sample_size: int = 1000) -> AnalysisResult:
        """Infer a suggested schema from a sample of existing points."""
        return self._client.analyze_schema(self._collection, sample_size=sample_size)

    def refresh_schema(self) -> None:
        """Bust the local schema cache for this scope."""
        self._client.refresh_schema(self._collection)

    def clear_schema_cache(self) -> None:
        """Clear the client-side schema cache for this scope."""
        self._client.clear_schema_cache(self._collection)

    # ---------------------------------------------------------------------
    # Analytics
    # ---------------------------------------------------------------------

    def get_analytics(self, time_range: str = "24h"):
        """Per-scope analytics (requests, latency, storage).

        time_range: one of '1h', '24h', '7d', '30d'.
        """
        return self._client.get_collection_analytics(
            self._collection, time_range=time_range
        )
