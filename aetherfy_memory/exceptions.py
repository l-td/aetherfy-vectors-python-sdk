"""
Exceptions for the Aetherfy Memory SDK.

MemoryClient's own error types. Generic vector-db errors continue to bubble
up from the underlying AetherfyVectorsClient via aetherfy_vectors.exceptions.
"""

from typing import Optional

from aetherfy_vectors.exceptions import AetherfyVectorsException


class AetherfyMemoryException(AetherfyVectorsException):
    """Base class for Memory-SDK-specific errors."""


class NamespaceNotFoundError(AetherfyMemoryException):
    """Raised when an operation targets a namespace that does not exist."""

    def __init__(self, name: str):
        super().__init__(
            f"Namespace '{name}' does not exist. Call "
            f"memory.create_namespace('{name}') before adding or searching."
        )
        self.name = name


class ThreadNotFoundError(AetherfyMemoryException):
    """Raised when an operation targets a thread that does not exist."""

    def __init__(self, thread_id: str):
        super().__init__(
            f"Thread '{thread_id}' does not exist. Call "
            f"memory.create_thread('{thread_id}') before adding or searching."
        )
        self.thread_id = thread_id


class NamespaceAlreadyExistsError(AetherfyMemoryException):
    """Raised when create_namespace is called for a name that already exists."""

    def __init__(self, name: str):
        super().__init__(f"Namespace '{name}' already exists.")
        self.name = name


class ThreadAlreadyExistsError(AetherfyMemoryException):
    """Raised when create_thread is called for an id that already exists."""

    def __init__(self, thread_id: str):
        super().__init__(f"Thread '{thread_id}' already exists.")
        self.thread_id = thread_id


class ThreadVectorSizeMismatchError(AetherfyMemoryException):
    """Raised when the threads collection already exists at another dimension.

    Every thread shares ONE collection, so they share one vector size and one
    distance metric — fixed when that collection is first created. Asking for
    a different size later cannot be honoured, and letting it through would
    surface three layers down as a bare dimension ValueError on the first
    ``add``. Raised at create time instead, naming the dimension that is
    actually there.
    """

    def __init__(self, existing: int, requested: int):
        super().__init__(
            f"The threads collection already exists with vector size "
            f"{existing}, but this MemoryClient is configured for "
            f"{requested}. Every thread shares one collection and therefore "
            f"one dimension. Construct MemoryClient(thread_vector_size="
            f"{existing}) to use it, or delete every thread to re-create the "
            f"collection at another size."
        )
        self.existing = existing
        self.requested = requested


class EmbeddingNotSupportedError(AetherfyMemoryException):
    """
    Raised when a caller omits `vector` expecting server-side embedding.

    Server-side embedding lands in a future release (see DX_ROADMAP.md T2-0).
    Until then, callers must compute embeddings client-side and pass `vector=`.
    """

    def __init__(self, context: Optional[str] = None):
        base = (
            "vector is required. Server-side embedding (add(text=...)) is "
            "planned for a future release; for now, compute the embedding "
            "client-side and pass vector=..."
        )
        super().__init__(f"{context}: {base}" if context else base)


class InvalidNameError(AetherfyMemoryException):
    """Raised when a namespace name or thread id does not match the allowed pattern."""
