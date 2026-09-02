"""
Data models and type definitions for Aetherfy Vectors SDK.

These models ensure type safety and provide clear interfaces for
all data structures used in the SDK.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum


class DistanceMetric(Enum):
    """Supported distance metrics for vector similarity."""

    COSINE = "Cosine"
    EUCLIDEAN = "Euclidean"
    DOT = "Dot"
    MANHATTAN = "Manhattan"


@dataclass
class Point:
    """Represents a vector point with payload.

    ``id`` is an unsigned integer (<= 2**53 - 1) or a UUID string — the two
    forms the server accepts. See ``utils.validate_point_id``.
    """

    id: Union[str, int]
    vector: List[float]
    payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert point to dictionary format."""
        result: Dict[str, Any] = {"id": self.id, "vector": self.vector}
        if self.payload:
            result["payload"] = self.payload
        return result


@dataclass
class SearchResult:
    """Represents a search result with score and payload."""

    id: Union[str, int]
    score: float
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create SearchResult from dictionary."""
        return cls(
            id=data["id"],
            score=data["score"],
            payload=data.get("payload"),
            vector=data.get("vector"),
        )


@dataclass
class VectorConfig:
    """Configuration for vector storage."""

    size: int
    distance: DistanceMetric

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {"size": self.size, "distance": self.distance.value}


@dataclass
class Collection:
    """Represents a vector collection."""

    name: str
    config: VectorConfig
    description: Optional[str] = None
    points_count: Optional[int] = None
    status: Optional[str] = None
    # §66 per-collection placement regions. Populated on create (the resolved
    # list the server echoes) and on get_collection. None when the server
    # didn't report it (older backend / non-regional response).
    regions: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Collection":
        """Create Collection from dictionary."""
        config_data = data.get("config", {})
        vector_config = data.get("config", {}).get("params", {}).get("vectors", {})

        config = VectorConfig(
            size=vector_config.get("size", 0),
            distance=DistanceMetric(vector_config.get("distance", "Cosine")),
        )

        return cls(
            name=data["name"],
            config=config,
            description=data.get("description"),
            points_count=data.get("points_count"),
            status=data.get("status"),
            regions=data.get("regions"),
        )


@dataclass
class UsageStats:
    """Current usage against the account's plan limits.

    The shape mirrors ``GET /api/v1/analytics/usage`` VERBATIM — same field
    names, same units, no derived or renamed values. It is pinned live by the
    e2e SDK guard (aetherfy-e2e-tests tests/sdk/test_usage_stats_sdk.py), which
    calls the real endpoint and asserts every field below is present with the
    right type. See aetherfy-dashboard docs/TELEMETRY.md for the endpoint's
    contract history.

    The nine invented fields this class used to declare (current_collections,
    max_collections, current_points, max_points, requests_this_month,
    max_requests_per_month, storage_used_mb, max_storage_mb, plan_name) were
    never served by anything: ``from_dict`` raised KeyError on a genuine 200,
    and the unit tests passed only because they mocked the invented payload.
    """

    storage_bytes_used: int
    #: ``None`` on an unlimited tier.
    storage_limit_bytes: Optional[int]
    collections_count: int
    #: The plan limit, ``None`` on an unlimited tier — the same sentinel
    #: ``storage_limit_bytes`` uses, because the endpoint normalises both.
    collections_limit: Optional[int]
    tier: str
    #: The replication footprint: the union of every active collection's
    #: regions.
    active_regions: List[str]
    #: ``0`` when there is no storage limit to be a percentage of.
    usage_percentage: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageStats":
        """Create UsageStats from dictionary."""
        return cls(
            storage_bytes_used=data["storage_bytes_used"],
            storage_limit_bytes=data["storage_limit_bytes"],
            collections_count=data["collections_count"],
            collections_limit=data["collections_limit"],
            tier=data["tier"],
            active_regions=data["active_regions"],
            usage_percentage=data["usage_percentage"],
        )


@dataclass
class Filter:
    """Query filter for search operations."""

    must: Optional[List[Dict[str, Any]]] = None
    must_not: Optional[List[Dict[str, Any]]] = None
    should: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary format."""
        result = {}
        if self.must:
            result["must"] = self.must
        if self.must_not:
            result["must_not"] = self.must_not
        if self.should:
            result["should"] = self.should
        return result
