"""
Main client implementation for Aetherfy Vectors SDK.

Provides a drop-in replacement for qdrant-client with identical API
that routes requests through the global vector database service.
"""

from typing import List, Dict, Any, Optional, Union
import requests
import json

from .auth import APIKeyManager
from .analytics import AnalyticsClient
from .models import (
    Point,
    SearchResult,
    Collection,
    VectorConfig,
    DistanceMetric,
    Filter,
    PerformanceAnalytics,
    CollectionAnalytics,
    UsageStats,
)
from .exceptions import AetherfyVectorsException, RequestTimeoutError
from .utils import (
    validate_vector,
    validate_collection_name,
    validate_point_id,
    build_api_url,
    parse_error_response,
    format_points_for_upsert,
)


class AetherfyVectorsClient:
    """
    Aetherfy Vectors client - a drop-in replacement for qdrant-client.

    Provides identical API to qdrant-client but routes requests through
    the global vector database service for enhanced performance, automatic
    global replication, and zero DevOps complexity.
    """

    DEFAULT_ENDPOINT = "https://vectors.aetherfy.com"
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs,
    ):
        """Initialize Aetherfy Vectors client.

        Args:
            api_key: Aetherfy API key. If None, will try environment variables.
            endpoint: API endpoint URL (default: https://vectors.aetherfy.com).
            timeout: Request timeout in seconds (default: 30.0).
            **kwargs: Additional parameters for compatibility.

        Raises:
            AuthenticationError: If API key is invalid or missing.
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

        # Initialize authentication
        self.auth_manager = APIKeyManager(api_key)
        self.auth_headers = {
            **self.auth_manager.get_auth_headers(),
            "Content-Type": "application/json",
            "User-Agent": "aetherfy-vectors-python/1.0.0",
        }

        # Initialize schema cache for ETag-based validation
        self._schema_cache: Dict[str, Dict[str, Any]] = {}  # {collection_name: {schema, etag}}

        # Initialize analytics client
        self.analytics = AnalyticsClient(self.endpoint, self.auth_headers, timeout)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        enable_retry: bool = True,
    ) -> Any:
        """Make HTTP request to the API with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE).
            endpoint: API endpoint path.
            data: Request body data.
            params: Query parameters.
            enable_retry: Whether to enable retry logic (default: True).

        Returns:
            Response data.

        Raises:
            AetherfyVectorsException: If request fails.
        """
        from .utils import retry_with_backoff

        def make_single_request():
            url = build_api_url(self.endpoint, endpoint)

            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=self.auth_headers,
                    json=data if data is not None else None,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code in [200, 201]:
                    return response.json() if response.content else None
                else:
                    error_data = response.json() if response.content else {}
                    raise parse_error_response(error_data, response.status_code)

            except requests.Timeout:
                raise RequestTimeoutError(
                    f"Request to {endpoint} timed out after {self.timeout} seconds"
                )
            except requests.RequestException as e:
                raise AetherfyVectorsException(f"Request failed: {str(e)}")

        # Apply retry logic only for write operations (POST, PUT)
        if enable_retry and method in ["POST", "PUT"]:
            return retry_with_backoff(
                make_single_request, max_retries=3, base_delay=1.0
            )
        else:
            return make_single_request()

    # Schema Cache Helpers

    def _get_cached_schema(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get cached schema for a collection if available."""
        return self._schema_cache.get(collection_name)

    def _fetch_and_cache_schema(self, collection_name: str) -> Dict[str, Any]:
        """Fetch collection schema from API and cache it with ETag."""
        response = self._make_request("GET", f"collections/{collection_name}")

        # Extract schema info
        result = response.get("result", {})
        schema_version = response.get("schema_version")
        vector_config = result.get("config", {}).get("params", {}).get("vectors", {})

        schema = {
            "size": vector_config.get("size"),
            "distance": vector_config.get("distance"),
            "etag": schema_version,
            "full_config": result
        }

        # Cache it
        self._schema_cache[collection_name] = schema
        return schema

    def clear_schema_cache(self, collection_name: Optional[str] = None) -> None:
        """Clear schema cache for a collection or all collections."""
        if collection_name:
            self._schema_cache.pop(collection_name, None)
        else:
            self._schema_cache.clear()

    def _normalize_distance_metric(
        self, distance: Union[str, DistanceMetric]
    ) -> DistanceMetric:
        """Normalize distance metric to proper DistanceMetric enum.

        Args:
            distance: Distance metric as string or DistanceMetric enum.

        Returns:
            Normalized DistanceMetric enum value.

        Raises:
            ValueError: If distance metric is invalid.
        """
        if isinstance(distance, DistanceMetric):
            return distance

        # Normalize string to capitalized format matching Qdrant API
        distance_map = {
            "cosine": DistanceMetric.COSINE,
            "euclidean": DistanceMetric.EUCLIDEAN,
            "euclid": DistanceMetric.EUCLIDEAN,
            "dot": DistanceMetric.DOT,
            "manhattan": DistanceMetric.MANHATTAN,
        }

        normalized = distance_map.get(distance.lower())
        if normalized:
            return normalized

        # Try to create DistanceMetric directly (handles already-capitalized strings)
        try:
            return DistanceMetric(distance)
        except ValueError:
            raise ValueError(
                f"Invalid distance metric: {distance}. "
                f"Must be one of: {', '.join(d.value for d in DistanceMetric)}"
            )

    # Collection Management Methods

    def create_collection(
        self,
        collection_name: str,
        vectors_config: Union[VectorConfig, Dict[str, Any]],
        distance: Optional[DistanceMetric] = None,
        **kwargs,
    ) -> bool:
        """Create a new collection.

        Args:
            collection_name: Name of the collection to create.
            vectors_config: Vector configuration or dict with size/distance.
            distance: Distance metric (deprecated, use vectors_config).
            **kwargs: Additional parameters for compatibility.

        Returns:
            True if collection was created successfully.

        Raises:
            ValidationError: If parameters are invalid.
            AetherfyVectorsException: If creation fails.
        """
        validate_collection_name(collection_name)

        # Handle different input formats for compatibility
        if isinstance(vectors_config, dict):
            if "size" in vectors_config and "distance" in vectors_config:
                config = VectorConfig(
                    size=vectors_config["size"],
                    distance=self._normalize_distance_metric(
                        vectors_config["distance"]
                    ),
                )
            else:
                # Handle qdrant-client format
                size = vectors_config.get("size", vectors_config.get("vector_size"))
                if size is None:
                    raise ValueError("Vector size must be specified")
                config = VectorConfig(
                    size=int(size),
                    distance=self._normalize_distance_metric(
                        vectors_config.get("distance", "Cosine")
                    ),
                )
        elif isinstance(vectors_config, VectorConfig):
            config = vectors_config
        else:
            raise ValueError(
                "vectors_config must be VectorConfig instance or dictionary"
            )

        # Override distance if provided separately (for compatibility)
        if distance:
            config.distance = self._normalize_distance_metric(distance)

        data = {"name": collection_name, "vectors": config.to_dict()}

        self._make_request("POST", "collections", data)
        return True

    def delete_collection(self, collection_name: str, **kwargs) -> bool:
        """Delete a collection.

        Args:
            collection_name: Name of the collection to delete.
            **kwargs: Additional parameters for compatibility.

        Returns:
            True if collection was deleted successfully.
        """
        validate_collection_name(collection_name)
        self._make_request("DELETE", f"collections/{collection_name}")
        return True

    def get_collections(self, **kwargs) -> List[Collection]:
        """Get list of all collections.

        Args:
            **kwargs: Additional parameters for compatibility.

        Returns:
            List of Collection objects.
        """
        response = self._make_request("GET", "collections")
        return [Collection.from_dict(col) for col in response.get("collections", [])]

    def collection_exists(self, collection_name: str, **kwargs) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection to check.
            **kwargs: Additional parameters for compatibility.

        Returns:
            True if collection exists, False otherwise.
        """
        try:
            self._make_request("GET", f"collections/{collection_name}")
            return True
        except AetherfyVectorsException:
            return False

    def get_collection(self, collection_name: str, **kwargs) -> Collection:
        """Get collection information.

        Args:
            collection_name: Name of the collection.
            **kwargs: Additional parameters for compatibility.

        Returns:
            Collection object with details.
        """
        validate_collection_name(collection_name)
        response = self._make_request("GET", f"collections/{collection_name}")
        return Collection.from_dict(response)

    # Point Management Methods

    def upsert(
        self, collection_name: str, points: List[Union[Point, Dict[str, Any]]], **kwargs
    ) -> bool:
        """Insert or update points in a collection.

        Args:
            collection_name: Name of the target collection.
            points: List of Point objects or dictionaries.
            **kwargs: Additional parameters for compatibility.

        Returns:
            True if upsert was successful.
        """
        validate_collection_name(collection_name)

        # Get schema (from cache or fetch)
        schema = self._get_cached_schema(collection_name)
        if not schema:
            schema = self._fetch_and_cache_schema(collection_name)

        # Validate vector dimensions
        expected_dim = schema.get("size")
        if expected_dim:
            for point in points:
                vector = point.get("vector") if isinstance(point, dict) else point.vector
                if not vector or not isinstance(vector, (list, tuple)):
                    raise ValueError("Each point must have a vector array")

                if len(vector) != expected_dim:
                    raise ValueError(
                        f"Vector dimension mismatch: expected {expected_dim}, got {len(vector)}"
                    )

        # Convert Point objects to dictionaries if needed
        formatted_points = []
        for point in points:
            if isinstance(point, Point):
                formatted_points.append(point.to_dict())
            elif isinstance(point, dict):
                formatted_points.append(point)
            else:
                raise ValueError("Points must be Point objects or dictionaries")

        # Validate and format points
        formatted_points = format_points_for_upsert(formatted_points)

        # Make request with If-Match header (ETag)
        data = {"points": formatted_points}
        headers = self.auth_headers.copy()
        if schema.get("etag"):
            headers["If-Match"] = schema["etag"]

        try:
            # Use custom headers for this request
            response = requests.put(
                f"{self.endpoint}/collections/{collection_name}/points",
                json=data,
                headers=headers,
                timeout=self.timeout
            )

            # Handle 412 Precondition Failed (schema changed)
            if response.status_code == 412:
                self.clear_schema_cache(collection_name)
                error_data = response.json()
                raise AetherfyVectorsException(
                    f"Schema changed for collection '{collection_name}'. "
                    f"Please retry your request. {error_data.get('error', {}).get('message', '')}"
                )

            # Handle 400 Bad Request (validation error from backend)
            if response.status_code == 400:
                error_data = response.json()
                error_message = error_data.get('error', {}).get('message', 'Validation error')
                raise ValueError(error_message)

            # Handle 500+ Server Errors
            if response.status_code >= 500:
                error_data = response.json() if response.content else {}
                error_message = error_data.get('error', {}).get('message', 'Server error occurred')
                raise AetherfyVectorsException(f"Server error: {error_message}")

            response.raise_for_status()
            return True

        except requests.exceptions.RequestException as e:
            if isinstance(e, (ValueError, AetherfyVectorsException)):
                raise
            raise AetherfyVectorsException(f"Upsert failed: {str(e)}")

    def delete(
        self,
        collection_name: str,
        points_selector: Union[List[Union[str, int]], Dict[str, Any]],
        **kwargs,
    ) -> bool:
        """Delete points from a collection.

        Args:
            collection_name: Name of the collection.
            points_selector: List of point IDs or filter conditions.
            **kwargs: Additional parameters for compatibility.

        Returns:
            True if deletion was successful.
        """
        validate_collection_name(collection_name)

        if isinstance(points_selector, list):
            # Delete by point IDs
            for point_id in points_selector:
                validate_point_id(point_id)
            data: Dict[str, Any] = {"points": points_selector}
        else:
            # Delete by filter
            data = {"filter": points_selector}

        self._make_request("POST", f"collections/{collection_name}/points/delete", data)
        return True

    def retrieve(
        self,
        collection_name: str,
        ids: List[Union[str, int]],
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Retrieve points by IDs.

        Args:
            collection_name: Name of the collection.
            ids: List of point IDs to retrieve.
            with_payload: Include payload in results.
            with_vectors: Include vectors in results.
            **kwargs: Additional parameters for compatibility.

        Returns:
            List of retrieved points.
        """
        validate_collection_name(collection_name)

        for point_id in ids:
            validate_point_id(point_id)

        data = {"ids": ids, "with_payload": with_payload, "with_vector": with_vectors}

        response = self._make_request(
            "POST", f"collections/{collection_name}/points", data
        )
        return response.get("result", [])

    # Search Methods

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        offset: int = 0,
        query_filter: Optional[Union[Filter, Dict[str, Any]]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """Search for similar vectors in a collection.

        Args:
            collection_name: Name of the collection to search in.
            query_vector: Query vector for similarity search.
            limit: Maximum number of results to return.
            offset: Number of results to skip.
            query_filter: Filter conditions for search.
            with_payload: Include payload in results.
            with_vectors: Include vectors in results.
            score_threshold: Minimum score threshold for results.
            **kwargs: Additional parameters for compatibility.

        Returns:
            List of SearchResult objects.
        """
        validate_collection_name(collection_name)
        validate_vector(query_vector)

        data = {
            "vector": query_vector,
            "limit": limit,
            "offset": offset,
            "with_payload": with_payload,
            "with_vector": with_vectors,
        }

        if query_filter:
            if isinstance(query_filter, Filter):
                data["filter"] = query_filter.to_dict()
            else:
                data["filter"] = query_filter

        if score_threshold is not None:
            data["score_threshold"] = score_threshold

        response = self._make_request(
            "POST", f"collections/{collection_name}/points/search", data
        )

        results = []
        for result in response.get("result", []):
            results.append(SearchResult.from_dict(result))

        return results

    def count(
        self,
        collection_name: str,
        count_filter: Optional[Dict[str, Any]] = None,
        exact: bool = True,
        **kwargs,
    ) -> int:
        """Count points in collection.

        Args:
            collection_name: Name of the collection.
            count_filter: Filter conditions for counting.
            exact: Whether to return exact count.
            **kwargs: Additional parameters for compatibility.

        Returns:
            Number of points matching the filter.
        """
        validate_collection_name(collection_name)

        data: Dict[str, Any] = {"exact": exact}
        if count_filter:
            data["filter"] = count_filter

        response = self._make_request(
            "POST", f"collections/{collection_name}/points/count", data
        )
        return response.get("count", 0)

    # Analytics Methods (SDK-specific)

    def get_performance_analytics(
        self, time_range: str = "24h", region: Optional[str] = None
    ) -> PerformanceAnalytics:
        """Retrieve global performance analytics.

        Args:
            time_range: Time range for analytics (1h, 24h, 7d, 30d).
            region: Specific region to filter by (optional).

        Returns:
            Performance analytics data.
        """
        return self.analytics.get_performance_analytics(time_range, region)

    def get_collection_analytics(
        self, collection_name: str, time_range: str = "24h"
    ) -> CollectionAnalytics:
        """Retrieve analytics for a specific collection.

        Args:
            collection_name: Name of the collection.
            time_range: Time range for analytics (1h, 24h, 7d, 30d).

        Returns:
            Collection-specific analytics data.
        """
        return self.analytics.get_collection_analytics(collection_name, time_range)

    def get_usage_stats(self) -> UsageStats:
        """Retrieve current usage statistics against customer limits.

        Returns:
            Current usage statistics.
        """
        return self.analytics.get_usage_stats()

    # Utility Methods

    def close(self) -> None:
        """Close the client connection (for compatibility)."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation of the client."""
        masked_key = self.auth_manager.mask_api_key()
        return (
            f"AetherfyVectorsClient(endpoint='{self.endpoint}', api_key='{masked_key}')"
        )
