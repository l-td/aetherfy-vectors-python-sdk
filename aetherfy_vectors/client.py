"""
Main client implementation for Aetherfy Vectors SDK.

Provides a drop-in replacement for qdrant-client with identical API
that routes requests through the global vector database service.
"""

from typing import List, Dict, Any, Optional, Union
import requests
from requests.adapters import HTTPAdapter

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
from .exceptions import (
    AetherfyVectorsException,
    RequestTimeoutError,
    ValidationError,
    NetworkError,
    SchemaValidationError,
    SchemaNotFoundError,
)
from .schema import (
    Schema,
    FieldDefinition,
    AnalysisResult,
    validate_vectors,
)
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

        # Initialize HTTP session with connection pooling
        # This prevents TCP/TLS handshake overhead on every request
        self.session = self._create_session()

        # Initialize schema cache for ETag-based validation (vector configs)
        self._schema_cache: Dict[
            str, Dict[str, Any]
        ] = {}  # {collection_name: {schema, etag}}

        # Initialize payload schema cache for schema validation
        self._payload_schema_cache: Dict[
            str, Dict[str, Any]
        ] = {}  # {collection_name: {schema: Schema, etag: str, enforcement_mode: str}}

        # Initialize analytics client with shared session
        self.analytics = AnalyticsClient(
            self.endpoint, self.auth_headers, timeout, session=self.session
        )

    def _create_session(self) -> requests.Session:
        """Create a requests Session with connection pooling and retry logic.

        Returns:
            Configured requests Session object with persistent connections.
        """
        session = requests.Session()

        # Configure connection pooling via HTTPAdapter
        # This keeps connections alive and reuses them across requests
        adapter = HTTPAdapter(
            pool_connections=10,  # Number of connection pools to cache
            pool_maxsize=50,  # Max connections to keep in pool
            max_retries=0,  # No automatic retries (we handle this ourselves)
            pool_block=False,  # Don't block when pool is full
        )

        # Mount adapter for both HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers on session
        session.headers.update(self.auth_headers)

        return session

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        enable_retry: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Make HTTP request to the API with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE).
            endpoint: API endpoint path.
            data: Request body data.
            params: Query parameters.
            enable_retry: Whether to enable retry logic (default: True).
            headers: Additional headers to include in request.

        Returns:
            Response data.

        Raises:
            AetherfyVectorsException: If request fails.
        """
        from .utils import retry_with_backoff

        def make_single_request():
            url = build_api_url(self.endpoint, endpoint)

            try:
                # Use session for persistent connections instead of requests.request()
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data if data is not None else None,
                    params=params,
                    headers=headers,  # Pass additional headers if provided
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
            except requests.ConnectionError as e:
                # Network connection errors should be retryable
                raise NetworkError(f"Network connection failed: {str(e)}")
            except requests.RequestException as e:
                # Other request errors - generic exception
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
            "full_config": result,
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
        description: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Create a new collection.

        Args:
            collection_name: Name of the collection to create.
            vectors_config: Vector configuration or dict with size/distance.
            distance: Distance metric (deprecated, use vectors_config).
            description: Optional collection description (max 500 characters).
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

        data = {
            "name": collection_name,
            "vectors": config.to_dict(),
            "description": description,
        }

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
        collection_data = response.get("result", response)
        return Collection.from_dict(collection_data)

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

        Raises:
            SchemaValidationError: If payloads fail schema validation.
        """
        validate_collection_name(collection_name)

        # Get vector config schema (from cache or fetch)
        schema = self._get_cached_schema(collection_name)
        if not schema:
            schema = self._fetch_and_cache_schema(collection_name)

        # Validate vector dimensions
        expected_dim = schema.get("size")
        if expected_dim:
            for point in points:
                vector = (
                    point.get("vector") if isinstance(point, dict) else point.vector
                )
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

        # Get payload schema for validation (if exists)
        payload_schema_data = self._payload_schema_cache.get(collection_name)
        if payload_schema_data is None:  # Not cached yet (different from cached None)
            # Try to fetch schema from server
            try:
                schema_result = self.get_schema(collection_name)
                if schema_result is None:
                    # Cache the fact that no schema exists to avoid repeated fetches
                    self._payload_schema_cache[collection_name] = {
                        "schema": None,
                        "enforcement_mode": "off",
                        "etag": None,
                    }
                payload_schema_data = self._payload_schema_cache.get(collection_name)
            except:
                # Error fetching schema - cache None to avoid retrying
                self._payload_schema_cache[collection_name] = {
                    "schema": None,
                    "enforcement_mode": "off",
                    "etag": None,
                }
                payload_schema_data = None

        # Client-side payload validation
        if payload_schema_data and payload_schema_data["schema"]:
            enforcement_mode = payload_schema_data.get("enforcement_mode", "off")

            # Only validate if enforcement is not 'off'
            if enforcement_mode != "off":
                validation_errors = validate_vectors(
                    formatted_points, payload_schema_data["schema"]
                )
                if validation_errors:
                    # Only raise error in strict mode
                    if enforcement_mode == "strict":
                        # Convert to dict format for exception
                        errors_dict = [e.to_dict() for e in validation_errors]
                        raise SchemaValidationError(errors_dict)
                    # In warn mode, just log the warnings (client-side logging would go here)
                    # For now, we allow the request to proceed

        # Validate and format points
        formatted_points = format_points_for_upsert(formatted_points)

        # Make request with If-Match headers (for both schemas)
        data = {"points": formatted_points}

        # Add If-Match headers if we have ETags
        extra_headers = {}
        if schema.get("etag"):
            extra_headers["If-Match"] = schema["etag"]

        # Add schema ETag for payload validation
        if payload_schema_data and payload_schema_data.get("etag"):
            extra_headers["If-Match"] = payload_schema_data["etag"]

        try:
            # Pass If-Match header directly to the request
            response = self._make_request(
                "PUT",
                f"collections/{collection_name}/points",
                data,
                headers=extra_headers if extra_headers else None,
            )
            return True

        except ValidationError as e:
            # Handle 412 Precondition Failed (schema changed)
            if e.status_code == 412:
                self.clear_schema_cache(collection_name)
                self._payload_schema_cache.pop(collection_name, None)

                # Fetch updated schema and re-validate
                updated_schema = None
                try:
                    self.get_schema(collection_name)
                    updated_schema = self._payload_schema_cache.get(collection_name)
                    if updated_schema and updated_schema["schema"]:
                        enforcement_mode = updated_schema.get("enforcement_mode", "off")
                        if enforcement_mode != "off":
                            validation_errors = validate_vectors(
                                formatted_points, updated_schema["schema"]
                            )
                            if validation_errors and enforcement_mode == "strict":
                                errors_dict = [e.to_dict() for e in validation_errors]
                                raise SchemaValidationError(errors_dict)
                except SchemaValidationError:
                    # Re-raise schema validation errors
                    raise
                except:
                    # Ignore other errors during schema refresh
                    updated_schema = self._payload_schema_cache.get(collection_name)

                # Retry the upsert with updated schema
                try:
                    extra_headers_retry = {}
                    if schema.get("etag"):
                        extra_headers_retry["If-Match"] = schema["etag"]
                    if updated_schema and updated_schema.get("etag"):
                        extra_headers_retry["If-Match"] = updated_schema["etag"]

                    response = self._make_request(
                        "PUT",
                        f"collections/{collection_name}/points",
                        data,
                        headers=extra_headers_retry if extra_headers_retry else None,
                    )
                    return True
                except:
                    # If retry also fails, raise the original 412 error
                    raise ValidationError(
                        f"Schema changed for collection '{collection_name}'. Please retry your request.",
                        status_code=412,
                    )

            # Handle 400 Bad Request (validation error from backend or client-side)
            # Re-raise as ValueError for backward compatibility
            raise ValueError(str(e))

        except AetherfyVectorsException as e:
            # Re-raise other errors
            raise

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

    # Schema Management Methods

    def get_schema(self, collection_name: str) -> Optional[Schema]:
        """Get schema for a collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            Schema object if schema is defined, None otherwise.

        Raises:
            SchemaNotFoundError: If no schema is defined for the collection.
        """
        validate_collection_name(collection_name)

        try:
            response = self._make_request("GET", f"api/v1/schema/{collection_name}")

            schema = Schema.from_dict(response["schema"])
            etag = response["etag"]
            enforcement_mode = response["enforcement_mode"]

            # Cache it
            self._payload_schema_cache[collection_name] = {
                "schema": schema,
                "etag": etag,
                "enforcement_mode": enforcement_mode,
            }

            return schema

        except AetherfyVectorsException as e:
            if e.status_code == 404:
                return None
            raise

    def set_schema(
        self, collection_name: str, schema: Schema, enforcement: str = "off"
    ) -> str:
        """Set schema for a collection.

        Args:
            collection_name: Name of the collection.
            schema: Schema definition.
            enforcement: Enforcement mode - 'off', 'warn', or 'strict' (default: 'off').

        Returns:
            ETag of the new schema.

        Raises:
            ValidationError: If schema or enforcement mode is invalid.
        """
        validate_collection_name(collection_name)

        if enforcement not in ["off", "warn", "strict"]:
            raise ValueError("enforcement must be 'off', 'warn', or 'strict'")

        data = {"schema": schema.to_dict(), "enforcement_mode": enforcement}

        response = self._make_request("PUT", f"api/v1/schema/{collection_name}", data)
        etag = response["etag"]

        # Update cache
        self._payload_schema_cache[collection_name] = {
            "schema": schema,
            "etag": etag,
            "enforcement_mode": enforcement,
        }

        return etag

    def delete_schema(self, collection_name: str) -> bool:
        """Remove schema from a collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            True if schema was deleted successfully.

        Raises:
            SchemaNotFoundError: If no schema is defined for the collection.
        """
        validate_collection_name(collection_name)

        try:
            self._make_request("DELETE", f"api/v1/schema/{collection_name}")

            # Clear from cache
            self._payload_schema_cache.pop(collection_name, None)

            return True

        except AetherfyVectorsException as e:
            if e.status_code == 404:
                raise SchemaNotFoundError(collection_name)
            raise

    def analyze_schema(
        self, collection_name: str, sample_size: int = 1000
    ) -> AnalysisResult:
        """Analyze existing data to understand payload structure.

        Args:
            collection_name: Name of the collection to analyze.
            sample_size: Number of vectors to sample (100-10000, default: 1000).

        Returns:
            Analysis result including field presence, types, and suggested schema.

        Raises:
            ValidationError: If sample_size is out of range.
            CollectionNotFoundError: If collection doesn't exist.
        """
        validate_collection_name(collection_name)

        if sample_size < 100 or sample_size > 10000:
            raise ValueError("sample_size must be between 100 and 10000")

        data = {"sample_size": sample_size}
        response = self._make_request(
            "POST", f"api/v1/schema/{collection_name}/analyze", data
        )

        return AnalysisResult.from_dict(response)

    def refresh_schema(self, collection_name: str) -> None:
        """Force refresh of cached schema.

        Args:
            collection_name: Name of the collection.
        """
        self._payload_schema_cache.pop(collection_name, None)
        self.get_schema(collection_name)

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
        """Close the client connection and cleanup resources."""
        if hasattr(self, "session"):
            self.session.close()

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
