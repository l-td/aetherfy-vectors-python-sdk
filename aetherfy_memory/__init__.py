"""
Aetherfy Memory Python SDK.

Agent memory primitives — conversations, knowledge bases, scraping logs,
customer state, anything an agent needs to remember across turns or runs.

Quickstart:

    from aetherfy_memory import MemoryClient

    memory = MemoryClient()                          # auto workspace

    memory.create_namespace("customer-42")
    customer = memory.namespace("customer-42")
    customer.add(text="Lives in NYC", vector=embed("..."))
    results = customer.search(vector=embed("where does customer live?"))

    memory.create_thread("conv-99")
    chat = memory.thread("conv-99")
    chat.add(role="user", content="hi", vector=embed("hi"))
    for msg in chat.history(limit=20):
        print(msg.role, msg.content)

For raw-Qdrant operations not exposed here, drop to the low-level client:

    memory.vectors.scroll(...)                       # any AetherfyVectorsClient method
    # or directly:
    from aetherfy_vectors import AetherfyVectorsClient
"""

__version__ = "1.0.0"

from .client import MemoryClient
from .exceptions import (
    AetherfyMemoryException,
    EmbeddingNotSupportedError,
    InvalidNameError,
    NamespaceAlreadyExistsError,
    NamespaceNotFoundError,
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
)
from .models import DEFAULT_VECTOR_SIZE, Message
from .namespace import Namespace
from .thread import Thread

__all__ = [
    "MemoryClient",
    "Namespace",
    "Thread",
    "Message",
    "DEFAULT_VECTOR_SIZE",
    "AetherfyMemoryException",
    "EmbeddingNotSupportedError",
    "InvalidNameError",
    "NamespaceAlreadyExistsError",
    "NamespaceNotFoundError",
    "ThreadAlreadyExistsError",
    "ThreadNotFoundError",
]
