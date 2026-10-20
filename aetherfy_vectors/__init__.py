"""
Aetherfy Vectors Python SDK

A client compatible with qdrant-client 1.15.1's core methods (see
aetherfy_vectors.qdrant_compat for the exact contract) that provides global
vector database operations with automatic replication, caching, and sub-50ms latency worldwide.
"""

__version__ = "1.2.2"
__author__ = "Aetherfy"
__email__ = "developers@aetherfy.com"

from .client import AetherfyVectorsClient
from .exceptions import (
    AetherfyVectorsException,
    AuthenticationError,
    RateLimitExceededError,
    ServiceUnavailableError,
    ValidationError,
    CollectionNotFoundError,
    PointNotFoundError,
    RequestTimeoutError,
    NetworkError,
    ConflictError,
    SchemaValidationError,
    SchemaNotFoundError,
    PartialUpsertError,
    CollectionInUseError,
    CollectionInOtherRegionError,
    AgentWorkspaceForbiddenError,
    WorkspaceNotFoundError,
    NotFoundError,
    QuotaExceededError,
    StorageModeNotInPlanError,
)
from .models import (
    SearchResult,
    Point,
    Collection,
    UsageStats,
)
from .storage_modes import StorageMode
from .schema import (
    Schema,
    FieldDefinition,
    AnalysisResult,
)
from .retry_policy import REPEATABLE, ONCE
from .utils import retry_with_backoff

__all__ = [
    "AetherfyVectorsClient",
    "AetherfyVectorsException",
    "AuthenticationError",
    "RateLimitExceededError",
    "ServiceUnavailableError",
    "ValidationError",
    "CollectionNotFoundError",
    "PointNotFoundError",
    "RequestTimeoutError",
    "NetworkError",
    "ConflictError",
    "SchemaValidationError",
    "SchemaNotFoundError",
    "PartialUpsertError",
    "CollectionInUseError",
    "CollectionInOtherRegionError",
    "AgentWorkspaceForbiddenError",
    "WorkspaceNotFoundError",
    "NotFoundError",
    "QuotaExceededError",
    "StorageModeNotInPlanError",
    "SearchResult",
    "Point",
    "Collection",
    "UsageStats",
    "StorageMode",
    "Schema",
    "FieldDefinition",
    "AnalysisResult",
    # The retry helper and the operation classes it takes (retry_policy.py).
    "retry_with_backoff",
    "REPEATABLE",
    "ONCE",
]
