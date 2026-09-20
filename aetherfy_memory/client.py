"""
MemoryClient — agent-memory SDK layered on aetherfy_vectors.

Provides an opinionated, agent-first API on top of `AetherfyVectorsClient`.
Every add/search operation goes through a named scope (`Namespace` or `Thread`);
there is no root-level `add/search` and no magic default collection. Scopes
must be created explicitly (typo protection).

For operations not exposed here — custom collection configs, raw Qdrant calls,
any current vectors-SDK surface — use `AetherfyVectorsClient` directly:

    from aetherfy_vectors import AetherfyVectorsClient
    from aetherfy_memory import MemoryClient

    memory = MemoryClient()                    # agent-memory, opinionated
    raw = AetherfyVectorsClient(workspace="auto")  # low-level escape hatch

Both share the same auth / workspace / endpoint; MemoryClient is a strict
superset in functionality via delegation.
"""

import re
import uuid
from typing import Any, Dict, List, Optional

from aetherfy_vectors.client import AetherfyVectorsClient
from aetherfy_vectors.models import (
    Collection,
    DistanceMetric,
    UsageStats,
    VectorConfig,
)

from .exceptions import (
    InvalidNameError,
    NamespaceAlreadyExistsError,
    NamespaceNotFoundError,
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
    ThreadVectorSizeMismatchError,
)
from .models import (
    DEFAULT_VECTOR_SIZE,
    THREAD_ID_KEY,
    THREAD_MARKER_KEY,
    THREADS_COLLECTION,
)
from .namespace import Namespace
from .thread import Thread


# User-facing names must start with letter/digit and contain only letters,
# digits, hyphens, underscores, and dots. No leading special chars — the
# `__threads__` collection name is therefore unreachable from this regex,
# so no namespace can collide with it.
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,254}$")


def _validate_user_name(name: str, kind: str) -> None:
    if not isinstance(name, str):
        raise InvalidNameError(f"{kind} must be a string, got {type(name).__name__}")
    if not _NAME_RE.match(name):
        raise InvalidNameError(
            f"Invalid {kind} '{name}'. Must match [a-zA-Z0-9][a-zA-Z0-9._-]* "
            f"(start with letter/digit; letters, digits, dots, hyphens, "
            f"underscores allowed; max 255 chars)."
        )


class MemoryClient:
    """Agent memory client — opinionated wrapper over AetherfyVectorsClient.

    Construction mirrors AetherfyVectorsClient, `workspace="auto"` included
    (it picks up `AETHERFY_WORKSPACE`, which the control plane injects on an
    agent that has a workspace, and resolves to no workspace when unset).
    Override with `workspace=None` for a shared-namespace dev flow, or pass an
    explicit name.
    """

    DEFAULT_ENDPOINT = AetherfyVectorsClient.DEFAULT_ENDPOINT
    DEFAULT_TIMEOUT = AetherfyVectorsClient.DEFAULT_TIMEOUT

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        endpoint: Optional[str] = None,
        api_region: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        workspace: Optional[str] = "auto",
        client: Optional[AetherfyVectorsClient] = None,
        thread_vector_size: int = DEFAULT_VECTOR_SIZE,
        thread_distance: DistanceMetric = DistanceMetric.COSINE,
    ):
        """Initialize a MemoryClient.

        Args:
            api_key: Aetherfy API key. Reads AETHERFY_API_KEY if omitted
                (auto-injected by the control plane at deploy time).
                Ignored if `client` is provided.
            endpoint: API endpoint URL. Resolution order matches
                AetherfyVectorsClient exactly:
                1. Explicit ``endpoint`` argument.
                2. ``AETHERFY_VECTORS_URL`` environment variable — the
                   control plane injects this on every agent machine, so a
                   deployed agent reaches its regional endpoint without any
                   code change.
                3. Default ``https://vectors.aetherfy.com``.
                Leave it unset unless you are targeting a local or custom
                deployment: passing it explicitly SUPPRESSES the injected
                env var, which is how a deployed agent ends up silently
                talking to the default endpoint. Ignored if `client` is
                provided.
            api_region: Which regional API endpoint to CONNECT to
                ('us-east-1', 'eu-central-1', or 'ap-southeast-1') — a
                transport/routing override, NOT where collections live.
                Forwarded verbatim to AetherfyVectorsClient, which owns the
                resolution: it sits BELOW ``endpoint`` and below
                ``AETHERFY_VECTORS_URL``, so the control-plane-injected URL
                always wins and a local-dev ``api_region=`` left in the code
                does not hijack a deployed agent (a warning is logged if both
                are set). Load-bearing only for standalone vectordb usage,
                local development and debugging. Reads
                ``AETHERFY_VECTORS_API_REGION`` when omitted. Ignored if
                `client` is provided. Mirrors the JS SDK, whose
                MemoryClientConfig extends ClientConfig and has always
                accepted `apiRegion`.
            timeout: Request timeout in seconds. Ignored if `client` is provided.
            workspace: Workspace name. Defaults to "auto" (reads
                AETHERFY_WORKSPACE). Pass None to disable workspace scoping
                (collections land in a shared namespace — not recommended
                outside local dev). Ignored if `client` is provided.
            thread_vector_size: Embedding dimension for THREADS. Every thread
                in a workspace lives in one collection and therefore shares
                one dimension, fixed when that collection is first created;
                this is where it comes from. Defaults to 384
                (all-MiniLM-L6-v2). Namespaces are unaffected — each still
                takes its own `vector_size` at `create_namespace`.
            thread_distance: Distance metric for the threads collection,
                fixed the same way. Default cosine.
            client: Bring-your-own AetherfyVectorsClient. When supplied, all
                other parameters (api_key, endpoint, api_region, timeout,
                workspace) are ignored and this client is used as-is. Useful when sharing
                a single vectors client across MemoryClient and other code,
                or when you need a custom session / retry strategy.
        """
        self._thread_vector_size = thread_vector_size
        self._thread_distance = thread_distance

        if client is not None:
            self._client = client
        else:
            # `endpoint` defaults to None, NOT to DEFAULT_ENDPOINT: passing a
            # concrete URL down is indistinguishable from the caller asking
            # for one, and AetherfyVectorsClient treats an explicit endpoint
            # as the highest-precedence source. Defaulting to the constant
            # therefore made AETHERFY_VECTORS_URL unreachable through this
            # constructor — a deployed agent using memory talked to the
            # global default endpoint instead of the injected regional one,
            # silently. Forward None and let the one endpoint resolver in
            # AetherfyVectorsClient.__init__ own the precedence.
            # `api_region` is forwarded, not interpreted. Reimplementing the
            # precedence here is how the endpoint bug happened in the first
            # place: this constructor decided something the resolver already
            # owns. One resolver, one place, both entry points.
            self._client = AetherfyVectorsClient(
                api_key=api_key,
                endpoint=endpoint,
                api_region=api_region,
                timeout=timeout,
                workspace=workspace,
            )

    # ---------------------------------------------------------------------
    # Introspection
    # ---------------------------------------------------------------------

    @property
    def workspace(self) -> Optional[str]:
        """The active workspace, or None if workspace scoping is disabled."""
        return self._client.workspace

    @property
    def vectors(self) -> AetherfyVectorsClient:
        """Direct access to the underlying AetherfyVectorsClient.

        Use this as the low-level escape hatch for any operation not exposed
        on MemoryClient. Collection names are workspace-scoped automatically.
        """
        return self._client

    # ---------------------------------------------------------------------
    # Namespace lifecycle
    # ---------------------------------------------------------------------

    def create_namespace(
        self,
        name: str,
        *,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: DistanceMetric = DistanceMetric.COSINE,
    ) -> Namespace:
        """Create a new namespace.

        Args:
            name: Namespace name. Must match [a-zA-Z0-9][a-zA-Z0-9._-]*.
            vector_size: Embedding dimension. Defaults to 384 (all-MiniLM-L6-v2
                / planned T2-0 default). Override for other models:
                1536 (OpenAI small), 3072 (OpenAI large), 1024 (Cohere v3).
            distance: Distance metric (cosine, dot, or euclid). Default cosine.

        Returns:
            A Namespace handle ready for add/search.

        Raises:
            InvalidNameError: if name doesn't match the allowed pattern.
            ReservedNameError: if name starts with the internal thread prefix.
            NamespaceAlreadyExistsError: if a namespace by that name exists.
        """
        _validate_user_name(name, "namespace name")

        if self._client.collection_exists(name):
            raise NamespaceAlreadyExistsError(name)

        self._client.create_collection(
            name,
            VectorConfig(size=vector_size, distance=distance),
        )
        return Namespace(name, name, self._client)

    def namespace(self, name: str) -> Namespace:
        """Open an existing namespace. Raises if it doesn't exist.

        Use `create_namespace(name)` first to create it.
        """
        _validate_user_name(name, "namespace name")
        if not self._client.collection_exists(name):
            raise NamespaceNotFoundError(name)
        return Namespace(name, name, self._client)

    def namespace_exists(self, name: str) -> bool:
        """True if the namespace exists in this workspace."""
        _validate_user_name(name, "namespace name")
        return self._client.collection_exists(name)

    def get_namespace(self, name: str) -> Collection:
        """Return metadata for a namespace (name, config, points_count, status).

        Distinct from `namespace(name)`, which returns an operation handle.
        Raises NamespaceNotFoundError if it doesn't exist.
        """
        _validate_user_name(name, "namespace name")
        if not self._client.collection_exists(name):
            raise NamespaceNotFoundError(name)
        return self._client.get_collection(name)

    def list_namespaces(self) -> List[str]:
        """All namespace names in this workspace.

        Threads are no longer collections, so there is nothing thread-shaped
        left to filter out of the collection list — except the single
        `__threads__` collection they all share, which is an implementation
        detail and not a namespace.
        """
        return [
            col.name
            for col in self._client.get_collections()
            if col.name != THREADS_COLLECTION
        ]

    def delete_namespace(self, name: str) -> bool:
        """Drop the namespace atomically. Idempotent: returns False if absent."""
        _validate_user_name(name, "namespace name")
        if not self._client.collection_exists(name):
            return False
        return self._client.delete_collection(name)

    # ---------------------------------------------------------------------
    # Thread lifecycle
    # ---------------------------------------------------------------------

    def _threads_marker_filter(self, thread_id: str) -> Dict[str, Any]:
        """Matches exactly the marker point of one thread."""
        return {
            "must": [
                {"key": THREAD_ID_KEY, "match": {"value": thread_id}},
                {"key": THREAD_MARKER_KEY, "match": {"value": True}},
            ]
        }

    def _marker_vector(self, size: int) -> List[float]:
        """A valid unit vector of `size` dimensions for a marker point.

        NOT the zero vector. Under cosine distance Qdrant normalises every
        stored vector by its length, and a zero-length vector has no
        defined normalisation — whether the engine rejects it or stores
        something whose similarity is undefined, neither is a thing to
        build the existence of a thread on. `[1, 0, ...]` has length 1 and
        is well defined under cosine, dot and euclid alike.
        """
        return [1.0] + [0.0] * (size - 1)

    def _ensure_threads_collection(self) -> None:
        """Create the shared threads collection on first use.

        Indexes both keys the thread clause filters on. An unindexed
        payload filter is SCANNED rather than looked up, and this is the
        one collection whose every read carries a tenant filter.
        """
        if self._client.collection_exists(THREADS_COLLECTION):
            existing = self._client.get_collection(THREADS_COLLECTION)
            size = existing.config.size
            if size and size != self._thread_vector_size:
                raise ThreadVectorSizeMismatchError(size, self._thread_vector_size)
            return

        self._client.create_collection(
            THREADS_COLLECTION,
            VectorConfig(
                size=self._thread_vector_size, distance=self._thread_distance
            ),
        )
        self._client.create_field_index(THREADS_COLLECTION, THREAD_ID_KEY, "keyword")
        self._client.create_field_index(
            THREADS_COLLECTION, THREAD_MARKER_KEY, "bool"
        )

    def create_thread(self, thread_id: str) -> Thread:
        """Create a new thread.

        Threads are rows, not collections: every thread in the workspace
        lives in one shared collection with `thread_id` as a payload key,
        so creating one does NOT consume a slot against the account's
        collection limit and a Free account is not capped at three
        conversations.

        That is also why there is no `vector_size` / `distance` here any
        more: one collection has one of each. Both come from the
        MemoryClient (`thread_vector_size` / `thread_distance`) and are
        fixed when the collection is first created.
        `create_namespace` keeps both — a namespace is still one
        collection.

        Creating a thread writes ONE marker point. That is what makes an
        empty thread exist: without it, a thread with no messages would be
        indistinguishable from a thread that was never created.

        Args:
            thread_id: Thread id. Must match [a-zA-Z0-9][a-zA-Z0-9._-]*.

        Returns:
            A Thread handle ready for add/history/search.

        Raises:
            InvalidNameError: if the id doesn't match the allowed pattern.
            ThreadAlreadyExistsError: if a thread by that id exists.
            ThreadVectorSizeMismatchError: if the threads collection
                already exists at a different dimension.
        """
        _validate_user_name(thread_id, "thread id")
        self._ensure_threads_collection()

        if self._thread_marker_exists(thread_id):
            raise ThreadAlreadyExistsError(thread_id)

        self._client.upsert(
            THREADS_COLLECTION,
            [
                {
                    "id": str(uuid.uuid4()),
                    "vector": self._marker_vector(self._thread_vector_size),
                    "payload": {
                        THREAD_ID_KEY: thread_id,
                        THREAD_MARKER_KEY: True,
                    },
                }
            ],
        )
        return Thread(thread_id, THREADS_COLLECTION, self._client)

    def _thread_marker_exists(self, thread_id: str) -> bool:
        """True iff this thread's marker point is present."""
        if not self._client.collection_exists(THREADS_COLLECTION):
            return False
        return (
            self._client.count(
                THREADS_COLLECTION,
                count_filter=self._threads_marker_filter(thread_id),
                exact=True,
            )
            > 0
        )

    def thread(self, thread_id: str) -> Thread:
        """Open an existing thread. Raises if it doesn't exist."""
        _validate_user_name(thread_id, "thread id")
        if not self._thread_marker_exists(thread_id):
            raise ThreadNotFoundError(thread_id)
        return Thread(thread_id, THREADS_COLLECTION, self._client)

    def thread_exists(self, thread_id: str) -> bool:
        """True if the thread exists in this workspace.

        A filtered count over the marker points, so an EMPTY thread still
        reads as existing — the property a payload-keyed model would have
        lost without them.
        """
        _validate_user_name(thread_id, "thread id")
        return self._thread_marker_exists(thread_id)

    def get_thread(self, thread_id: str) -> Collection:
        """Return metadata for a thread (name, config, points_count, status).

        `name` is the thread id and `points_count` is THIS thread's message
        count (the marker is not a message); `config` and `status` describe
        the shared threads collection, which is where a thread's vector
        size and distance actually live now.

        Distinct from `thread(id)`, which returns an operation handle.
        Raises ThreadNotFoundError if it doesn't exist.
        """
        _validate_user_name(thread_id, "thread id")
        if not self._thread_marker_exists(thread_id):
            raise ThreadNotFoundError(thread_id)
        info = self._client.get_collection(THREADS_COLLECTION)
        info.name = thread_id
        info.points_count = Thread(
            thread_id, THREADS_COLLECTION, self._client
        ).count()
        return info

    def list_threads(self) -> List[str]:
        """All thread ids in this workspace.

        A scroll over the MARKER points, so the work is bounded by the
        number of threads rather than the number of messages, and an empty
        thread is listed like any other.
        """
        if not self._client.collection_exists(THREADS_COLLECTION):
            return []
        ids: List[str] = []
        for point in self._client.scroll_iter(
            THREADS_COLLECTION,
            scroll_filter={
                "must": [{"key": THREAD_MARKER_KEY, "match": {"value": True}}]
            },
            with_payload=True,
            with_vectors=False,
        ):
            thread_id = (point.get("payload") or {}).get(THREAD_ID_KEY)
            if isinstance(thread_id, str):
                ids.append(thread_id)
        return ids

    def delete_thread(self, thread_id: str) -> bool:
        """Drop the thread and every message in it. Idempotent.

        A delete-by-filter on this thread's rows, marker included. It
        cannot touch a sibling thread, and it no longer drops a
        collection.
        """
        _validate_user_name(thread_id, "thread id")
        if not self._thread_marker_exists(thread_id):
            return False
        return self._client.delete(
            THREADS_COLLECTION,
            {"must": [{"key": THREAD_ID_KEY, "match": {"value": thread_id}}]},
        )

    # ---------------------------------------------------------------------
    # Usage stats (parity with AetherfyVectorsClient)
    # ---------------------------------------------------------------------

    def get_usage_stats(self) -> UsageStats:
        """Usage stats for this workspace."""
        return self._client.get_usage_stats()

    def clear_schema_cache(self) -> None:
        """Clear the client-side schema cache for every scope in this workspace.

        Per-scope cache clearing lives on `Namespace.clear_schema_cache()` /
        `Thread.clear_schema_cache()`. Use this when bulk-invalidating is
        cheaper than tracking each scope.
        """
        # Passing None to AetherfyVectorsClient.clear_schema_cache clears every
        # entry in its internal cache.
        self._client.clear_schema_cache(None)

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._client.close()

    def __enter__(self) -> "MemoryClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        ws = self.workspace or "<unscoped>"
        return f"MemoryClient(workspace={ws!r})"
