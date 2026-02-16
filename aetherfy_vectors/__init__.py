"""
Aetherfy Vectors Python SDK

A drop-in replacement for qdrant-client that provides global vector database
operations with automatic replication, caching, and sub-50ms latency worldwide.
"""

__version__ = "1.0.0"
__author__ = "Aetherfy"
__email__ = "developers@aetherfy.com"

from .client import AetherfyVectorsClient
from .exceptions import (
    AetherfyVectorsException,
    AuthenticationError,
    RateLimitExceededError,
    ServiceUnavailableError,
    SchemaValidationError,
    SchemaNotFoundError,
    CollectionInUseError,
)
from .models import (
    SearchResult,
    Point,
    Collection,
    PerformanceAnalytics,
    CollectionAnalytics,
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
    "SchemaValidationError",
    "SchemaNotFoundError",
    "CollectionInUseError",
    "SearchResult",
    "Point",
    "Collection",
    "PerformanceAnalytics",
    "CollectionAnalytics",
    "UsageStats",
    "Schema",
    "FieldDefinition",
    "AnalysisResult",
]
