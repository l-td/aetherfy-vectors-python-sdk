"""
_Scope — the shared base for Namespace and Thread.

Holds every operation that behaves identically for both scope shapes:
read (search / retrieve / count / iter), delete / clear, and the
payload-metadata helpers. Schema management is NOT here: a schema belongs to a
collection, and a Thread no longer has one to itself (see Namespace).

The two *write* APIs differ by shape — a Namespace stores a generic memory
(`{text?, metadata?}`), a Thread stores a conversation message
(`{role, content, ts, metadata?}`) — so `add` (and the batch writers) live on
the subclasses, not here. That is why `Thread` is NOT a subclass of
`Namespace`: it is not add-substitutable for one. Both are `_Scope`s that
share the substitutable surface.
"""

from typing import Any, Dict, Iterator, List, Optional, Union, overload

from aetherfy_vectors.client import AetherfyVectorsClient
from aetherfy_vectors.exceptions import (
    AetherfyVectorsException,
    CollectionNotFoundError,
    PointNotFoundError,
)
from aetherfy_vectors.models import Filter, SearchResult


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
    # Scoping hooks
    #
    # A Namespace IS its collection, so all four hooks are identities. A
    # Thread shares one collection with every other thread in the workspace
    # and overrides them to carry its own clause. They exist so that every
    # read and write below goes through ONE place that can narrow it — a
    # scope clause bolted onto each call site individually is a scope clause
    # that gets forgotten at the next call site added.
    # ---------------------------------------------------------------------

    # A dict in is a dict out, in BOTH implementations: the identity here
    # hands it straight back and Thread's override always builds a dict.
    # Stated as an overload so `delete`, whose non-list selector is a dict,
    # passes the client a dict rather than something the checker must assume
    # might be a `Filter` or `None` — neither of which that path can produce.
    @overload
    def _combine_filter(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @overload
    def _combine_filter(
        self, filter: Optional[Union[Filter, Dict[str, Any]]]
    ) -> Optional[Union[Filter, Dict[str, Any]]]:
        ...

    def _combine_filter(
        self, filter: Optional[Union[Filter, Dict[str, Any]]]
    ) -> Optional[Union[Filter, Dict[str, Any]]]:
        """Narrow a caller's filter to this scope. Identity for a Namespace."""
        return filter

    def _assert_owns(self, ids: List[Union[str, int]]) -> None:
        """Refuse point ids that do not belong to this scope. No-op here."""
        return None

    def _owned_ids(self, ids: List[Union[str, int]]) -> List[Union[str, int]]:
        """Narrow an id list to the ids this scope owns. Identity here."""
        return ids

    def _point_selector(
        self, ids: List[Union[str, int]]
    ) -> Union[List[Union[str, int]], Dict[str, Any]]:
        """How this scope addresses a list of its own point ids on the wire.

        A Namespace owns its whole collection, so a bare id list is already
        exact. A Thread shares its collection, so it returns a FILTER that
        pins the ids AND the thread — the engine enforces the scope, rather
        than the SDK checking it first and trusting itself afterwards.
        """
        return ids

    def _reads_payload_to_scope(self) -> bool:
        """True when this scope needs payloads to identify its own points."""
        return False

    def _retain_owned(
        self, points: List[Dict[str, Any]], *, with_payload: bool
    ) -> List[Dict[str, Any]]:
        """Drop points belonging to another scope. Identity for a Namespace."""
        return points

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
        # Read-then-check, NOT a scoped filter — see _assert_owns.
        self._assert_owns([id])
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
        self._assert_owns([id])
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
        self._assert_owns([id])
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
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Semantic search within this scope only.

        Args:
            vector: Query vector.
            limit: Maximum number of results.
            offset: Number of results to skip.
            filter: Payload filter conditions.
            with_payload: Include payload in results.
            with_vectors: Include vectors in results.
            score_threshold: Minimum score threshold.
            search_params: Search-time engine parameters, forwarded verbatim
                as the request body's `params` field. The headline use is
                `{"hnsw_ef": 256}`: a larger ef makes the HNSW graph walk
                visit more candidates, buying recall at the cost of latency.
                Recall matters here — retrieving the *right* memory usually
                beats saving a millisecond. Omit it to keep the tuned
                server-side default (hnsw_ef=100). Different params values
                produce different request bodies and therefore different
                server cache entries, so the same query at a different ef is
                a separate entry, never a wrong hit. Not validated or
                translated; see AetherfyVectorsClient.search.

        Returns:
            List of SearchResult objects.

        Raises:
            TypeError: If an unknown keyword argument is passed — this
                signature is keyword-only with no **kwargs sink, so a
                misspelled option fails loudly instead of being dropped.
        """
        return self._client.search(
            self._collection,
            query_vector=vector,
            limit=limit,
            offset=offset,
            query_filter=self._combine_filter(filter),
            with_payload=with_payload,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
            search_params=search_params,
        )

    def retrieve(
        self,
        ids: List[Union[str, int]],
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch specific points by ID.

        Ids that exist in the underlying collection but belong to another
        scope are not returned: a Thread's point ids are unique within the
        shared threads collection, not within the thread.
        """
        points = self._client.retrieve(
            self._collection,
            ids,
            # A Thread has to read payloads to tell its own points from a
            # sibling's. The caller's with_payload choice is still honoured:
            # _retain_owned strips what the caller did not ask for.
            with_payload=with_payload or self._reads_payload_to_scope(),
            with_vectors=with_vectors,
        )
        return self._retain_owned(points, with_payload=with_payload)

    def count(
        self,
        *,
        filter: Optional[Union[Filter, Dict[str, Any]]] = None,
        exact: bool = True,
    ) -> int:
        """Count points in this scope, optionally filtered.

        ``filter`` takes a ``Filter`` or a plain dict, the same as
        ``search`` and ``iter``.
        """
        return self._client.count(
            self._collection, count_filter=self._combine_filter(filter), exact=exact
        )

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
            scroll_filter=self._combine_filter(filter),
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
        """Delete points — by ID list or by filter — without dropping the scope.

        An id list is narrowed to the ids this scope owns before the
        request is sent, so a Thread cannot delete a sibling thread's
        point by naming its id.
        """
        if isinstance(selector, list):
            if not selector:
                # An empty id list is a no-op, and NOT a request. This is a
                # safety property, not a micro-optimisation: a Thread turns
                # an id list into a `has_id` filter, and a request carrying
                # an empty `has_id` is one engine-side semantic away from
                # matching the whole thread. Never send it.
                return True
            return self._client.delete(self._collection, self._point_selector(selector))
        return self._client.delete(self._collection, self._combine_filter(selector))

    def clear(self) -> bool:
        """Atomically drop this scope (destroys the underlying collection).

        After `clear()`, the scope no longer exists. Re-create it via
        `memory.create_namespace(name)` / `memory.create_thread(id)`.
        """
        return self._client.delete_collection(self._collection)
