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
    """Current usage statistics against customer limits."""

    current_collections: int
    max_collections: int
    current_points: int
    max_points: int
    requests_this_month: int
    max_requests_per_month: int
    storage_used_mb: float
    max_storage_mb: float
    plan_name: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UsageStats":
        """Create UsageStats from dictionary."""
        return cls(
            current_collections=data["current_collections"],
            max_collections=data["max_collections"],
            current_points=data["current_points"],
            max_points=data["max_points"],
            requests_this_month=data["requests_this_month"],
            max_requests_per_month=data["max_requests_per_month"],
            storage_used_mb=data["storage_used_mb"],
            max_storage_mb=data["max_storage_mb"],
            plan_name=data["plan_name"],
        )

    @property
    def collections_usage_percent(self) -> float:
        """Calculate collections usage percentage."""
        return (self.current_collections / self.max_collections) * 100

    @property
    def points_usage_percent(self) -> float:
        """Calculate points usage percentage."""
        return (self.current_points / self.max_points) * 100

    @property
    def requests_usage_percent(self) -> float:
        """Calculate requests usage percentage."""
        return (self.requests_this_month / self.max_requests_per_month) * 100

    @property
    def storage_usage_percent(self) -> float:
        """Calculate storage usage percentage."""
        return (self.storage_used_mb / self.max_storage_mb) * 100


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
