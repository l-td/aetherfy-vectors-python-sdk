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
    RequestTimeoutError
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
        assert "X-API-Key" in client.auth_headers
    
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
        mock_requests.request.return_value = mock_successful_response({})
        
        result = client.upsert("test_collection", sample_points)
        
        assert result is True
        mock_requests.request.assert_called_once()
        args, kwargs = mock_requests.request.call_args
        assert kwargs["method"] == "PUT"
        assert "collections/test_collection/points" in kwargs["url"]
        assert len(kwargs["json"]["points"]) == 2
    
    def test_upsert_point_objects(self, client, mock_requests, mock_successful_response):
        """Test upsert with Point objects."""
        mock_requests.request.return_value = mock_successful_response({})
        
        points = [
            Point(id="point_1", vector=[0.1, 0.2, 0.3], payload={"test": True}),
            Point(id="point_2", vector=[0.4, 0.5, 0.6])
        ]
        
        result = client.upsert("test_collection", points)
        
        assert result is True
        args, kwargs = mock_requests.request.call_args
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
        """Test successful point retrieval."""
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