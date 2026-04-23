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
from typing import List, Optional

from aetherfy_vectors.client import AetherfyVectorsClient
from aetherfy_vectors.models import (
    Collection,
    CollectionAnalytics,
    DistanceMetric,
    PerformanceAnalytics,
    UsageStats,
    VectorConfig,
)

from .exceptions import (
    InvalidNameError,
    NamespaceAlreadyExistsError,
    NamespaceNotFoundError,
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
)
from .models import DEFAULT_VECTOR_SIZE
from .namespace import Namespace
from .thread import Thread


# Internal collection-name prefix for threads. Chosen to be an invalid
# user-facing name (starts with `_`), so it can't collide with user
# namespaces under the name validation rule below.
_THREAD_PREFIX = "__thread__"

# User-facing names must start with letter/digit and contain only letters,
# digits, hyphens, underscores, and dots. No leading special chars — the
# reserved `__thread__` prefix is therefore unreachable from this regex.
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

    Construction mirrors AetherfyVectorsClient but defaults `workspace="auto"`
    (which picks up `AETHERFY_WORKSPACE`, injected by the control plane at
    deploy time). Override with `workspace=None` for a shared-namespace dev
    flow, or pass an explicit name.
    """

    DEFAULT_ENDPOINT = AetherfyVectorsClient.DEFAULT_ENDPOINT
    DEFAULT_TIMEOUT = AetherfyVectorsClient.DEFAULT_TIMEOUT

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        workspace: Optional[str] = "auto",
        client: Optional[AetherfyVectorsClient] = None,
    ):
        """Initialize a MemoryClient.

        Args:
            api_key: Aetherfy API key. Reads AETHERFY_API_KEY if omitted
                (auto-injected by the control plane at deploy time).
                Ignored if `client` is provided.
            endpoint: API endpoint URL. Ignored if `client` is provided.
            timeout: Request timeout in seconds. Ignored if `client` is provided.
            workspace: Workspace name. Defaults to "auto" (reads
                AETHERFY_WORKSPACE). Pass None to disable workspace scoping
                (collections land in a shared namespace — not recommended
                outside local dev). Ignored if `client` is provided.
            client: Bring-your-own AetherfyVectorsClient. When supplied, all
                other parameters (api_key, endpoint, timeout, workspace) are
                ignored and this client is used as-is. Useful when sharing
                a single vectors client across MemoryClient and other code,
                or when you need a custom session / retry strategy.
        """
        if client is not None:
            self._client = client
        else:
            self._client = AetherfyVectorsClient(
                api_key=api_key,
                endpoint=endpoint,
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
        """All namespace names in this workspace (excludes threads)."""
        return [
            col.name
            for col in self._client.get_collections()
            if not col.name.startswith(_THREAD_PREFIX)
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

    def create_thread(
        self,
        thread_id: str,
        *,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: DistanceMetric = DistanceMetric.COSINE,
    ) -> Thread:
        """Create a new thread.

        Args:
            thread_id: Thread id. Must match [a-zA-Z0-9][a-zA-Z0-9._-]*.
            vector_size: Embedding dimension (same defaults as namespaces).
            distance: Distance metric. Default cosine.

        Returns:
            A Thread handle ready for add/history/search.
        """
        _validate_user_name(thread_id, "thread id")
        collection = _THREAD_PREFIX + thread_id

        if self._client.collection_exists(collection):
            raise ThreadAlreadyExistsError(thread_id)

        self._client.create_collection(
            collection,
            VectorConfig(size=vector_size, distance=distance),
        )
        return Thread(thread_id, collection, self._client)

    def thread(self, thread_id: str) -> Thread:
        """Open an existing thread. Raises if it doesn't exist."""
        _validate_user_name(thread_id, "thread id")
        collection = _THREAD_PREFIX + thread_id
        if not self._client.collection_exists(collection):
            raise ThreadNotFoundError(thread_id)
        return Thread(thread_id, collection, self._client)

    def thread_exists(self, thread_id: str) -> bool:
        """True if the thread exists in this workspace."""
        _validate_user_name(thread_id, "thread id")
        return self._client.collection_exists(_THREAD_PREFIX + thread_id)

    def get_thread(self, thread_id: str) -> Collection:
        """Return metadata for a thread (name, config, points_count, status).

        The returned Collection's `.name` is remapped to the thread id
        (stripping the internal `__thread__` prefix).

        Distinct from `thread(id)`, which returns an operation handle.
        Raises ThreadNotFoundError if it doesn't exist.
        """
        _validate_user_name(thread_id, "thread id")
        collection_name = _THREAD_PREFIX + thread_id
        if not self._client.collection_exists(collection_name):
            raise ThreadNotFoundError(thread_id)
        info = self._client.get_collection(collection_name)
        info.name = thread_id  # present the user-facing id, not the internal prefix
        return info

    def list_threads(self) -> List[str]:
        """All thread ids in this workspace."""
        return [
            col.name[len(_THREAD_PREFIX):]
            for col in self._client.get_collections()
            if col.name.startswith(_THREAD_PREFIX)
        ]

    def delete_thread(self, thread_id: str) -> bool:
        """Drop the thread atomically. Idempotent."""
        _validate_user_name(thread_id, "thread id")
        collection = _THREAD_PREFIX + thread_id
        if not self._client.collection_exists(collection):
            return False
        return self._client.delete_collection(collection)

    # ---------------------------------------------------------------------
    # Global analytics (parity with AetherfyVectorsClient)
    # ---------------------------------------------------------------------

    def get_performance_analytics(
        self, time_range: str = "24h", region: Optional[str] = None
    ) -> PerformanceAnalytics:
        """Global performance analytics across this workspace."""
        return self._client.get_performance_analytics(time_range=time_range, region=region)

    def get_namespace_analytics(
        self, name: str, time_range: str = "24h"
    ) -> CollectionAnalytics:
        """Per-namespace analytics — mirror of get_collection_analytics."""
        _validate_user_name(name, "namespace name")
        if not self._client.collection_exists(name):
            raise NamespaceNotFoundError(name)
        return self._client.get_collection_analytics(name, time_range=time_range)

    def get_thread_analytics(
        self, thread_id: str, time_range: str = "24h"
    ) -> CollectionAnalytics:
        """Per-thread analytics."""
        _validate_user_name(thread_id, "thread id")
        collection = _THREAD_PREFIX + thread_id
        if not self._client.collection_exists(collection):
            raise ThreadNotFoundError(thread_id)
        return self._client.get_collection_analytics(collection, time_range=time_range)

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
