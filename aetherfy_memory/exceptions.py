"""
Exceptions for the Aetherfy Memory SDK.

MemoryClient's own error types. Generic vector-db errors continue to bubble
up from the underlying AetherfyVectorsClient via aetherfy_vectors.exceptions.
"""

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


class EmbeddingNotSupportedError(AetherfyMemoryException):
    """
    Raised when a caller omits `vector` expecting server-side embedding.

    Server-side embedding lands in a future release (see DX_ROADMAP.md T2-0).
    Until then, callers must compute embeddings client-side and pass `vector=`.
    """

    def __init__(self):
        super().__init__(
            "vector is required. Server-side embedding (add(text=...)) is "
            "planned for a future release; for now, compute the embedding "
            "client-side and pass vector=..."
        )


class InvalidNameError(AetherfyMemoryException):
    """Raised when a namespace name or thread id does not match the allowed pattern."""
