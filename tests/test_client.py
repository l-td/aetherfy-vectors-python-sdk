"""
Tests for the main AetherfyVectorsClient functionality.

Tests all core client operations including collection management,
point operations, and search functionality.
"""

import pytest
from unittest.mock import Mock, patch
import requests

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import VectorConfig, DistanceMetric, Point, SearchResult
from aetherfy_vectors.exceptions import (
    AuthenticationError, ValidationError, CollectionNotFoundError,
    RequestTimeoutError, CollectionInUseError
)


class TestClientInitialization:
    """Test client initialization and configuration."""
    
    def test_client_init_with_api_key(self, api_key, test_endpoint):
        """Test client initialization with API key."""
        client = AetherfyVectorsClient(
            api_key=api_key,
            endpoint=test_endpoint,
            timeout=15.0
        )
        
        assert client.endpoint == test_endpoint
        assert client.timeout == 15.0
        assert client.auth_manager.api_key == api_key
        assert "Authorization" in client.auth_headers
    
    def test_client_init_default_endpoint(self, api_key):
        """Test client initialization with default endpoint."""
        client = AetherfyVectorsClient(api_key=api_key)
        assert client.endpoint == "https://vectors.aetherfy.com"
        assert client.timeout == 30.0
    
    def test_client_init_without_api_key(self):
        """Test client initialization fails without API key."""
        with pytest.raises(AuthenticationError):
            AetherfyVectorsClient()
    
    def test_client_init_with_env_var(self, api_key, monkeypatch):
        """Test client initialization with environment variable."""
        monkeypatch.setenv("AETHERFY_API_KEY", api_key)
        client = AetherfyVectorsClient()
        assert client.auth_manager.api_key == api_key
    
    def test_client_repr(self, client):
        """Test client string representation."""
        repr_str = repr(client)
        assert "AetherfyVectorsClient" in repr_str
        assert "https://test-api.aetherfy.com" in repr_str
        assert "***" in repr_str  # Masked API key
        # Ensure actual API key is not exposed
        assert "afy_test_1234567890abcdef1234" not in repr_str


class TestCollectionManagement:
    """Test collection management operations."""
    
    def test_create_collection_success(self, client, mock_requests, mock_successful_response):
        """Test successful collection creation."""
        mock_requests.request.return_value = mock_successful_response({})
        
        config = VectorConfig(size=128, distance=DistanceMetric.COSINE)
        result = client.create_collection("test_collection", config)
        
        assert result is True
        mock_requests.request.assert_called_once()
        args, kwargs = mock_requests.request.call_args
        assert kwargs["method"] == "POST"
        assert "collections" in kwargs["url"]
        assert kwargs["json"]["name"] == "test_collection"
        assert kwargs["json"]["vectors"]["size"] == 128
    
    def test_create_collection_with_dict_config(self, client, mock_requests, mock_successful_response):
        """Test collection creation with dictionary config."""
        mock_requests.request.return_value = mock_successful_response({})

        config = {"size": 256, "distance": "Euclidean"}
        result = client.create_collection("test_collection", config)

        assert result is True
        args, kwargs = mock_requests.request.call_args
        assert kwargs["json"]["vectors"]["size"] == 256
        assert kwargs["json"]["vectors"]["distance"] == "Euclidean"

    def test_create_collection_with_description(self, client, mock_requests, mock_successful_response):
        """Test collection creation with description."""
        mock_requests.request.return_value = mock_successful_response({})

        config = VectorConfig(size=128, distance=DistanceMetric.COSINE)
        description = "Test collection for product embeddings"
        result = client.create_collection("test_collection", config, description=description)

        assert result is True
        args, kwargs = mock_requests.request.call_args
        assert kwargs["json"]["name"] == "test_collection"
        assert kwargs["json"]["description"] == description

    def test_create_collection_without_description(self, client, mock_requests, mock_successful_response):
        """Test collection creation without description sends null."""
        mock_requests.request.return_value = mock_successful_response({})

        config = VectorConfig(size=128, distance=DistanceMetric.COSINE)
        result = client.create_collection("test_collection", config)

        assert result is True
        args, kwargs = mock_requests.request.call_args
        assert kwargs["json"]["description"] is None
    
    def test_delete_collection_success(self, client, mock_requests, mock_successful_response):
        """Test successful collection deletion."""
        mock_requests.request.return_value = mock_successful_response({})
        
        result = client.delete_collection("test_collection")
        
        assert result is True
        mock_requests.request.assert_called_once()
        args, kwargs = mock_requests.request.call_args
        assert kwargs["method"] == "DELETE"
        assert "collections/test_collection" in kwargs["url"]
    
    def test_get_collections_success(self, client, mock_requests, mock_successful_response):
        """Test successful collections retrieval."""
        collections_data = {
            "collections": [
                {
                    "name": "collection1",
                    "config": {
                        "params": {
                            "vectors": {"size": 128, "distance": "Cosine"}
                        }
                    },
                    "points_count": 100,
                    "status": "green"
                }
            ]
        }
        mock_requests.request.return_value = mock_successful_response(collections_data)
        
        collections = client.get_collections()
        
        assert len(collections) == 1
        assert collections[0].name == "collection1"
        assert collections[0].points_count == 100
    
    def test_collection_exists_true(self, client, mock_requests, mock_successful_response):
        """Test collection_exists returns True for existing collection."""
        mock_requests.request.return_value = mock_successful_response({})
        
        result = client.collection_exists("test_collection")
        
        assert result is True
    
    def test_collection_exists_false(self, client, mock_requests, mock_error_response):
        """Test collection_exists returns False for non-existing collection."""
        mock_requests.request.return_value = mock_error_response(
            message="Collection not found",
            status_code=404
        )

        result = client.collection_exists("nonexistent_collection")

        assert result is False

    def test_collection_exists_reraises_on_auth_error(
        self, client, mock_requests, mock_error_response
    ):
        """A 401 from the existence probe must raise, not silently return False.

        Pre-fix, collection_exists caught any AetherfyVectorsException and
        returned False — so a logged-out / revoked-key state surfaced as
        "collection doesn't exist", masking the real failure. The fix
        narrows the catch to HTTP 404 only and re-raises the rest.
        """
        from aetherfy_vectors.exceptions import AuthenticationError

        mock_requests.request.return_value = mock_error_response(
            message="Unauthorized",
            status_code=401,
        )

        with pytest.raises(AuthenticationError):
            client.collection_exists("any_collection")

    def test_collection_exists_reraises_on_service_unavailable(
        self, client, mock_requests, mock_error_response
    ):
        """A 503 must raise so callers see the real cause, not "doesn't exist"."""
        from aetherfy_vectors.exceptions import ServiceUnavailableError

        mock_requests.request.return_value = mock_error_response(
            message="Service Unavailable",
            status_code=503,
        )

        with pytest.raises(ServiceUnavailableError):
            client.collection_exists("any_collection")

    def test_collection_exists_returns_false_on_non_dict_404_body(
        self, client, mock_requests
    ):
        """Returning False on 404 must work even when the upstream body is
        a bare JSON string (e.g. "Not Found") instead of a dict.

        This is the path that surfaced the parse_error_response crash in
        e2e: the backend returns a non-dict body on 404, parse_error_response
        previously crashed with AttributeError before collection_exists
        ever saw an exception class to inspect.
        """
        # Build a Response stand-in whose .json() returns a bare string.
        response = Mock()
        response.status_code = 404
        response.content = b'"Not Found"'
        response.json.return_value = "Not Found"
        mock_requests.request.return_value = response

        result = client.collection_exists("nonexistent_collection")
        assert result is False

    def test_create_collection_prepopulates_schema_cache(
        self, client, mock_requests, mock_successful_response
    ):
        """create_collection must seed _schema_cache from the request body.

        This closes the create→read consistency window: the backend's
        GET /collections/<name> hits Qdrant, which is eventually
        consistent w.r.t. its own writes, so a read immediately after a
        2xx create can briefly return 4xx. With the cache prepopulated,
        the next upsert/exists call hits local state and skips the GET.
        """
        mock_requests.request.return_value = mock_successful_response({})

        config = VectorConfig(size=384, distance=DistanceMetric.COSINE)
        client.create_collection("fresh_coll", config)

        cached = client._get_cached_schema("fresh_coll")
        assert cached is not None
        assert cached["size"] == 384
        assert cached["distance"] == "Cosine"
        # etag is None until a real GET /collections/<name> assigns one;
        # upsert's `if schema.get("etag"):` guard treats this as
        # "no If-Match header", which is correct.
        assert cached["etag"] is None

    def test_create_collection_cache_unblocks_upsert_without_get(
        self, client, mock_requests, mock_successful_response, sample_points
    ):
        """End-to-end: after create, upsert must skip GET /collections.

        If the cache prepopulation regressed, upsert would fall back to
        _fetch_and_cache_schema, hitting the read-after-write race
        against Qdrant's eventual consistency. This pins the contract:
        the only calls between create and upsert-completion must be the
        POST (create), GET payload schema (returns 404), and PUT (upsert
        points). Any extra GET /collections/<name> would mean the cache
        path regressed.
        """
        from aetherfy_vectors.exceptions import AetherfyVectorsException

        schema_404 = AetherfyVectorsException("Schema not found", status_code=404)
        mock_requests.request.side_effect = [
            mock_successful_response({}),  # POST /collections (create)
            schema_404,                    # GET /schema/<name> -> 404
            mock_successful_response({}),  # PUT /collections/<name>/points
        ]

        config = VectorConfig(size=4, distance=DistanceMetric.COSINE)
        client.create_collection("fresh_coll", config)
        result = client.upsert("fresh_coll", sample_points)

        assert result is True
        # Exactly 3 HTTP calls. A 4th would mean a stray
        # GET /collections/<name> sneaked in, defeating the cache.
        assert mock_requests.request.call_count == 3
        for call in mock_requests.request.call_args_list:
            method = call.kwargs.get("method")
            url = call.kwargs.get("url", "")
            if method == "GET" and url.endswith("/collections/fresh_coll"):
                pytest.fail(
                    f"Upsert hit GET /collections/<name> despite cached schema: {call}"
                )

    def test_delete_collection_clears_schema_cache(
        self, client, mock_requests, mock_successful_response
    ):
        """delete_collection must drop _schema_cache and _payload_schema_cache.

        Otherwise a recreate-with-different-shape would silently use the
        old size/distance/etag, producing dimension-mismatch errors or
        stale ETag conflicts that look like backend bugs.
        """
        mock_requests.request.return_value = mock_successful_response({})

        # Seed both caches as if the collection had been used before.
        config = VectorConfig(size=128, distance=DistanceMetric.COSINE)
        client.create_collection("doomed", config)
        client._payload_schema_cache["doomed"] = {
            "schema": {"fields": []},
            "etag": "v1",
            "enforcement_mode": "warn",
        }
        assert client._get_cached_schema("doomed") is not None
        assert "doomed" in client._payload_schema_cache

        client.delete_collection("doomed")

        assert client._get_cached_schema("doomed") is None
        assert "doomed" not in client._payload_schema_cache

    def test_collection_exists_fast_path_uses_cache(
        self, client, mock_requests, mock_successful_response
    ):
        """A cached entry must short-circuit collection_exists — no GET.

        This is what makes the e2e flow (create → exists) reliable in
        the read-after-write window. Without the fast path, the GET
        races Qdrant's internal replication and can briefly 4xx.
        """
        # Seed the cache directly to isolate the fast path from create.
        client._schema_cache["already_known"] = {
            "size": 128,
            "distance": "Cosine",
            "etag": None,
            "full_config": {},
        }

        result = client.collection_exists("already_known")

        assert result is True
        # Critical: NO HTTP call was made. The cache covers existence.
        mock_requests.request.assert_not_called()

    def test_upsert_404_evicts_cache_self_healing(
        self, client, mock_requests, mock_error_response, sample_points
    ):
        """A 404 from an upsert must drop both caches for that collection.

        Scenario: client A creates a collection (cache seeded); client B
        deletes it; client A tries to upsert. The PUT returns 404. Without
        eviction, A's cache keeps lying ("collection_exists=True") and
        every subsequent operation re-pays the 404. With eviction, the
        next operation goes back to the network and gets the truth.
        """
        # Pre-seed both caches as if a prior create/upsert had populated them.
        client._schema_cache["ghost"] = {
            "size": 4,
            "distance": "Cosine",
            "etag": None,
            "full_config": {},
        }
        client._payload_schema_cache["ghost"] = {
            "schema": None,
            "etag": None,
            "enforcement_mode": "off",
        }

        # The PUT returns 404. retry_with_backoff treats 404 as
        # non-retryable (is_retryable_error → False), so a single 404
        # response is enough to surface the exception immediately.
        mock_requests.request.return_value = mock_error_response(
            message="Collection not found",
            status_code=404,
        )

        # The exception propagates — that's the contract — but the side
        # effect we care about is the cache state after.
        with pytest.raises(Exception):
            client.upsert("ghost", sample_points)

        # Both caches must be dropped for the scoped name.
        assert client._get_cached_schema("ghost") is None
        assert "ghost" not in client._payload_schema_cache

    def test_get_collection_404_evicts_cache(
        self, client, mock_requests, mock_error_response
    ):
        """get_collection 404 must drop both caches.

        Same self-healing contract as upsert, exercised on the read path.
        """
        client._schema_cache["ghost"] = {
            "size": 128,
            "distance": "Cosine",
            "etag": None,
            "full_config": {},
        }
        client._payload_schema_cache["ghost"] = {
            "schema": None,
            "etag": None,
            "enforcement_mode": "off",
        }

        mock_requests.request.return_value = mock_error_response(
            message="Collection not found",
            status_code=404,
        )

        with pytest.raises(Exception):
            client.get_collection("ghost")

        assert client._get_cached_schema("ghost") is None
        assert "ghost" not in client._payload_schema_cache

    def test_non_404_error_does_not_evict_cache(
        self, client, mock_requests, mock_error_response, sample_points
    ):
        """503 / 401 / 500 must NOT evict caches.

        Eviction is only correct when 404 unambiguously means "collection
        is gone." A transient 503 (or auth blip, etc.) leaves the
        collection intact upstream — wiping the cache here would force a
        round trip on the next call for no reason. Pin the contract.
        """
        client._schema_cache["intact"] = {
            "size": 4,
            "distance": "Cosine",
            "etag": None,
            "full_config": {},
        }

        mock_requests.request.return_value = mock_error_response(
            message="Service Unavailable",
            status_code=503,
        )

        with pytest.raises(Exception):
            client.upsert("intact", sample_points)

        # Cache survives the transient error.
        assert client._get_cached_schema("intact") is not None

    def test_get_collection_success(self, client, mock_requests, mock_successful_response):
        """Test successful single collection retrieval."""
        collection_data = {
            "name": "test_collection",
            "config": {
                "params": {
                    "vectors": {"size": 128, "distance": "Cosine"}
                }
            },
            "points_count": 50,
            "status": "green"
        }
        mock_requests.request.return_value = mock_successful_response(collection_data)
        
        collection = client.get_collection("test_collection")
        
        assert collection.name == "test_collection"
        assert collection.points_count == 50
        assert collection.status == "green"


class TestPointOperations:
    """Test point management operations."""
    
    def test_upsert_points_success(self, client, mock_requests, mock_successful_response, sample_points):
        """Test successful point upsert."""
        # Mock GET /collections/{name} - vector config
        collection_response = mock_successful_response({
            'result': {
                'config': {
                    'params': {
                        'vectors': {'size': 4, 'distance': 'Cosine'}
                    }
                }
            },
            'schema_version': 'test123'
        })

        # Mock GET /api/v1/schema/{name} - no payload schema (404)
        from aetherfy_vectors.exceptions import AetherfyVectorsException
        schema_404_error = AetherfyVectorsException("Schema not found", status_code=404)

        # Mock PUT /collections/{name}/points - upsert
        upsert_response = mock_successful_response({})

        mock_requests.request.side_effect = [
            collection_response,  # GET vector config
            schema_404_error,     # GET payload schema (404)
            upsert_response       # PUT upsert
        ]

        result = client.upsert("test_collection", sample_points)

        assert result is True
        assert mock_requests.request.call_count == 3
        # Check the PUT call (third call)
        args, kwargs = mock_requests.request.call_args_list[2]
        assert kwargs["method"] == "PUT"
        assert "collections/test_collection/points" in kwargs["url"]
        assert len(kwargs["json"]["points"]) == 2
    
    def test_upsert_point_objects(self, client, mock_requests, mock_successful_response):
        """Test upsert with Point objects."""
        # Mock GET /collections/{name} - vector config
        collection_response = mock_successful_response({
            'result': {
                'config': {
                    'params': {
                        'vectors': {'size': 3, 'distance': 'Cosine'}
                    }
                }
            },
            'schema_version': 'test123'
        })

        # Mock GET /api/v1/schema/{name} - no payload schema (404)
        from aetherfy_vectors.exceptions import AetherfyVectorsException
        schema_404_error = AetherfyVectorsException("Schema not found", status_code=404)

        # Mock PUT /collections/{name}/points - upsert
        upsert_response = mock_successful_response({})

        mock_requests.request.side_effect = [
            collection_response,  # GET vector config
            schema_404_error,     # GET payload schema (404)
            upsert_response       # PUT upsert
        ]

        points = [
            Point(id="point_1", vector=[0.1, 0.2, 0.3], payload={"test": True}),
            Point(id="point_2", vector=[0.4, 0.5, 0.6])
        ]

        result = client.upsert("test_collection", points)

        assert result is True
        # Check the PUT call (third call)
        args, kwargs = mock_requests.request.call_args_list[2]
        assert len(kwargs["json"]["points"]) == 2
        assert kwargs["json"]["points"][0]["payload"]["test"] is True
    
    def test_delete_points_by_ids(self, client, mock_requests, mock_successful_response):
        """Test point deletion by IDs."""
        mock_requests.request.return_value = mock_successful_response({})
        
        result = client.delete("test_collection", ["point_1", "point_2"])
        
        assert result is True
        args, kwargs = mock_requests.request.call_args
        assert kwargs["method"] == "POST"
        assert "points/delete" in kwargs["url"]
        assert kwargs["json"]["points"] == ["point_1", "point_2"]
    
    def test_delete_points_by_filter(self, client, mock_requests, mock_successful_response):
        """Test point deletion by filter."""
        mock_requests.request.return_value = mock_successful_response({})
        
        filter_condition = {"must": [{"key": "category", "match": {"value": "test"}}]}
        result = client.delete("test_collection", filter_condition)
        
        assert result is True
        args, kwargs = mock_requests.request.call_args
        assert kwargs["json"]["filter"] == filter_condition
    
    def test_retrieve_points_success(self, client, mock_requests, mock_successful_response):
        """Test successful point retrieval. Pins the dedicated retrieve URL.

        retrieve() must POST to /collections/<name>/points/retrieve, not
        /points. The server-side streaming refactor removed the body-shape
        ambiguity at /points (which used to also accept body.ids); retrieve
        has its own URL now and the catch-all proxy will not handle
        body.ids on /points anymore.
        """
        points_data = {
            "result": [
                {
                    "id": "point_1",
                    "vector": [0.1, 0.2, 0.3],
                    "payload": {"category": "test"}
                }
            ]
        }
        mock_requests.request.return_value = mock_successful_response(points_data)

        points = client.retrieve("test_collection", ["point_1"], with_vectors=True)

        assert len(points) == 1
        assert points[0]["id"] == "point_1"
        assert points[0]["vector"] == [0.1, 0.2, 0.3]
        # URL pin — guards against an accidental revert to /points.
        args, kwargs = mock_requests.request.call_args
        assert kwargs["method"] == "POST"
        assert kwargs["url"].endswith("/collections/test_collection/points/retrieve")
        assert kwargs["json"]["ids"] == ["point_1"]
    
    def test_count_points_success(self, client, mock_requests, mock_successful_response):
        """Test successful point counting."""
        count_data = {"count": 42}
        mock_requests.request.return_value = mock_successful_response(count_data)
        
        count = client.count("test_collection")
        
        assert count == 42
        args, kwargs = mock_requests.request.call_args
        assert kwargs["method"] == "POST"
        assert "points/count" in kwargs["url"]


class TestSearchOperations:
    """Test search functionality."""
    
    def test_search_success(self, client, mock_requests, mock_successful_response, sample_search_results):
        """Test successful vector search."""
        search_data = {"result": sample_search_results}
        mock_requests.request.return_value = mock_successful_response(search_data)
        
        query_vector = [0.1, 0.2, 0.3, 0.4]
        results = client.search("test_collection", query_vector, limit=5)
        
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "point_1"
        assert results[0].score == 0.95
        assert results[0].payload["category"] == "test"
        
        args, kwargs = mock_requests.request.call_args
        assert kwargs["method"] == "POST"
        assert "points/search" in kwargs["url"]
        assert kwargs["json"]["vector"] == query_vector
        assert kwargs["json"]["limit"] == 5
    
    def test_search_with_filter(self, client, mock_requests, mock_successful_response, sample_search_results):
        """Test search with filter conditions."""
        search_data = {"result": sample_search_results}
        mock_requests.request.return_value = mock_successful_response(search_data)
        
        query_vector = [0.1, 0.2, 0.3, 0.4]
        query_filter = {"must": [{"key": "category", "match": {"value": "test"}}]}
        
        results = client.search("test_collection", query_vector, query_filter=query_filter)
        
        assert len(results) == 2
        args, kwargs = mock_requests.request.call_args
        assert kwargs["json"]["filter"] == query_filter
    
    def test_search_with_score_threshold(self, client, mock_requests, mock_successful_response, sample_search_results):
        """Test search with score threshold."""
        search_data = {"result": sample_search_results}
        mock_requests.request.return_value = mock_successful_response(search_data)
        
        query_vector = [0.1, 0.2, 0.3, 0.4]
        results = client.search("test_collection", query_vector, score_threshold=0.9)
        
        args, kwargs = mock_requests.request.call_args
        assert kwargs["json"]["score_threshold"] == 0.9


class TestErrorHandling:
    """Test error handling and exception scenarios."""
    
    def test_authentication_error(self, mock_requests, mock_error_response):
        """Test authentication error handling."""
        with pytest.raises(AuthenticationError):
            AetherfyVectorsClient(api_key="invalid_key")
    
    def test_request_error_handling(self, client, mock_requests, mock_error_response):
        """Test general request error handling."""
        mock_requests.request.return_value = mock_error_response(
            message="Collection not found",
            status_code=404,
            error_code="collection_not_found"
        )
        
        with pytest.raises(CollectionNotFoundError):
            client.get_collection("nonexistent")
    
    def test_request_timeout(self, client, mock_requests):
        """Test request timeout handling."""
        mock_requests.request.side_effect = requests.Timeout("Request timed out")
        
        with pytest.raises(RequestTimeoutError):
            client.get_collections()
    
    def test_validation_error_invalid_collection_name(self, client):
        """Test validation error for invalid collection name."""
        with pytest.raises(ValidationError):
            client.create_collection("", VectorConfig(128, DistanceMetric.COSINE))
    
    def test_validation_error_invalid_vector(self, client):
        """Test validation error for invalid vector."""
        with pytest.raises(ValidationError):
            client.search("test_collection", [])  # Empty vector

    def test_delete_collection_in_use_raises_error(self, client, mock_requests):
        """Test that deleting a collection in use raises CollectionInUseError."""
        from unittest.mock import Mock
        mock_response = Mock()
        mock_response.status_code = 409
        mock_response.json.return_value = {
            "error": {
                "code": "COLLECTION_IN_USE",
                "message": "Collection 'test-collection' is in use by agent(s): my-agent",
                "collection_name": "test-collection",
                "agents": ["my-agent", "another-agent"]
            }
        }
        mock_response.content = True
        mock_requests.request.return_value = mock_response

        with pytest.raises(CollectionInUseError) as exc_info:
            client.delete_collection("test-collection")

        assert exc_info.value.collection_name == "test-collection"
        assert "my-agent" in exc_info.value.agents
        assert "another-agent" in exc_info.value.agents


class TestContextManager:
    """Test context manager functionality."""
    
    def test_context_manager(self, api_key, test_endpoint):
        """Test client as context manager."""
        with AetherfyVectorsClient(api_key=api_key, endpoint=test_endpoint) as client:
            assert isinstance(client, AetherfyVectorsClient)
            assert client.auth_manager.api_key == api_key
    
    def test_close_method(self, client):
        """Test close method."""
        client.close()  # Should not raise any exception