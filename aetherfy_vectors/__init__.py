"""
Aetherfy Vectors Python SDK

A client compatible with qdrant-client 1.15.1's core methods (see
aetherfy_vectors.qdrant_compat for the exact contract) that provides global
vector database operations with automatic replication, caching, and sub-50ms latency worldwide.
"""

__version__ = "1.1.0"
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
    QuotaExceededError,
)
from .models import (
    SearchResult,
    Point,
    Collection,
    UsageStats,
)
from .schema import (
    Schema,
    FieldDefinition,
    AnalysisResult,
)

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
    "QuotaExceededError",
    "SearchResult",
    "Point",
    "Collection",
    "UsageStats",
    "Schema",
    "FieldDefinition",
    "AnalysisResult",
]
