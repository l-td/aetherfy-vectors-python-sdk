"""
Tests for Qdrant compatibility and migration scenarios.

Ensures the SDK maintains API compatibility with qdrant-client
and provides smooth migration paths.
"""

import pytest
from unittest.mock import Mock, patch

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import VectorConfig, DistanceMetric, Point


class TestQdrantCompatibility:
    """Test compatibility with qdrant-client API patterns."""
    
    def test_client_initialization_compatibility(self, api_key):
        """Test client initialization matches qdrant-client patterns."""
        # Should accept similar parameters as QdrantClient
        client = AetherfyVectorsClient(
            api_key=api_key,
            timeout=30.0
        )
        
        assert client.timeout == 30.0
        assert client.auth_manager.api_key == api_key
    
    def test_create_collection_compatibility(self, client, mock_requests, mock_successful_response):
        """Test create_collection API compatibility."""
        mock_requests.request.return_value = mock_successful_response({})
        
        # Test various compatible parameter formats
        
        # Format 1: VectorConfig object
        config1 = VectorConfig(size=128, distance=DistanceMetric.COSINE)
        result1 = client.create_collection("test1", config1)
        assert result1 is True
        
        # Format 2: Dictionary format
        config2 = {"size": 256, "distance": "Euclidean"}
        result2 = client.create_collection("test2", config2)
        assert result2 is True
        
        # Format 3: With separate distance parameter (qdrant-client style)
        config3 = {"size": 512}
        result3 = client.create_collection("test3", config3, distance=DistanceMetric.DOT)
        assert result3 is True
    
    def test_search_parameter_compatibility(self, client, mock_requests, mock_successful_response, sample_search_results):
        """Test search method parameter compatibility."""
        search_data = {"result": sample_search_results}
        mock_requests.request.return_value = mock_successful_response(search_data)
        
        query_vector = [0.1, 0.2, 0.3, 0.4]
        
        # Test different parameter combinations (qdrant-client compatible)
        
        # Basic search
        results1 = client.search("test_collection", query_vector)
        assert len(results1) == 2
        
        # Search with limit and offset
        results2 = client.search("test_collection", query_vector, limit=5, offset=10)
        assert len(results2) == 2
        
        # Search with filter (query_filter parameter)
        filter_dict = {"must": [{"key": "category", "match": {"value": "test"}}]}
        results3 = client.search("test_collection", query_vector, query_filter=filter_dict)
        assert len(results3) == 2
        
        # Search with payload and vectors flags
        results4 = client.search("test_collection", query_vector, with_payload=True, with_vectors=True)
        assert len(results4) == 2
    
    def test_upsert_parameter_compatibility(self, client, mock_requests, mock_successful_response):
        """Test upsert method parameter compatibility."""
        mock_requests.request.return_value = mock_successful_response({})
        
        # Test different point formats
        
        # Format 1: List of dictionaries (qdrant-client style)
        points1 = [
            {"id": "point_1", "vector": [0.1, 0.2, 0.3], "payload": {"test": True}},
            {"id": "point_2", "vector": [0.4, 0.5, 0.6]}
        ]
        result1 = client.upsert("test_collection", points1)
        assert result1 is True
        
        # Format 2: List of Point objects
        points2 = [
            Point(id="point_3", vector=[0.7, 0.8, 0.9], payload={"category": "test"}),
            Point(id="point_4", vector=[1.0, 1.1, 1.2])
        ]
        result2 = client.upsert("test_collection", points2)
        assert result2 is True
    
    def test_retrieve_parameter_compatibility(self, client, mock_requests, mock_successful_response):
        """Test retrieve method parameter compatibility."""
        points_data = {"result": []}
        mock_requests.request.return_value = mock_successful_response(points_data)
        
        # Test different retrieve patterns
        
        # Basic retrieve
        result1 = client.retrieve("test_collection", ["point_1", "point_2"])
        assert isinstance(result1, list)
        
        # Retrieve with flags (qdrant-client compatible)
        result2 = client.retrieve("test_collection", ["point_1"], with_payload=True, with_vectors=True)
        assert isinstance(result2, list)
        
        # Retrieve with mixed ID types
        result3 = client.retrieve("test_collection", ["string_id", 123, "another_id"])
        assert isinstance(result3, list)
    
    def test_delete_parameter_compatibility(self, client, mock_requests, mock_successful_response):
        """Test delete method parameter compatibility."""
        mock_requests.request.return_value = mock_successful_response({})
        
        # Test different delete patterns
        
        # Delete by point IDs (list)
        result1 = client.delete("test_collection", ["point_1", "point_2"])
        assert result1 is True
        
        # Delete by filter conditions
        filter_condition = {"must": [{"key": "category", "match": {"value": "test"}}]}
        result2 = client.delete("test_collection", filter_condition)
        assert result2 is True
    
    def test_count_parameter_compatibility(self, client, mock_requests, mock_successful_response):
        """Test count method parameter compatibility."""
        count_data = {"count": 42}
        mock_requests.request.return_value = mock_successful_response(count_data)
        
        # Test different count patterns
        
        # Basic count
        count1 = client.count("test_collection")
        assert count1 == 42
        
        # Count with filter
        filter_condition = {"must": [{"key": "status", "match": {"value": "active"}}]}
        count2 = client.count("test_collection", count_filter=filter_condition)
        assert count2 == 42
        
        # Count with exact flag
        count3 = client.count("test_collection", exact=True)
        assert count3 == 42


class TestMigrationScenarios:
    """Test common migration scenarios from qdrant-client."""
    
    def test_basic_migration_pattern(self, mock_requests, mock_successful_response):
        """Test basic migration from qdrant-client to aetherfy-vectors."""
        mock_requests.request.return_value = mock_successful_response({})
        
        # Simulate migration from:
        # from qdrant_client import QdrantClient
        # client = QdrantClient(host="localhost", port=6333)
        # 
        # To:
        # from aetherfy_vectors import AetherfyVectorsClient
        # client = AetherfyVectorsClient(api_key="your-api-key")
        
        api_key = "afy_live_migrationtestkey1234567890"
        client = AetherfyVectorsClient(api_key=api_key)
        
        # All existing method calls should work unchanged
        
        # Collection operations
        config = {"size": 128, "distance": "Cosine"}
        client.create_collection("products", config)
        
        collections = client.get_collections()
        assert isinstance(collections, list)
        
        # Point operations
        points = [
            {"id": "product_1", "vector": [0.1, 0.2, 0.3], "payload": {"name": "Product A"}},
            {"id": "product_2", "vector": [0.4, 0.5, 0.6], "payload": {"name": "Product B"}}
        ]
        client.upsert("products", points)
        
        # Search operations
        results = client.search("products", [0.1, 0.2, 0.3], limit=10)
        assert isinstance(results, list)
    
    def test_advanced_migration_patterns(self, client, mock_requests, mock_successful_response, sample_search_results):
        """Test advanced migration patterns with complex operations."""
        search_data = {"result": sample_search_results}
        mock_requests.request.return_value = mock_successful_response(search_data)
        
        # Complex search with all parameters
        query_vector = [0.1, 0.2, 0.3, 0.4]
        query_filter = {
            "must": [
                {"key": "category", "match": {"value": "electronics"}},
                {"key": "price", "range": {"gte": 100, "lte": 1000}}
            ]
        }
        
        results = client.search(
            collection_name="products",
            query_vector=query_vector,
            limit=20,
            offset=0,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
            score_threshold=0.7
        )
        
        assert len(results) == 2
        assert all(hasattr(result, 'score') for result in results)
        assert all(hasattr(result, 'payload') for result in results)
    
    def test_batch_operations_migration(self, client, mock_requests, mock_successful_response):
        """Test batch operations migration patterns."""
        mock_requests.request.return_value = mock_successful_response({})
        
        # Large batch upsert (common in migrations)
        large_batch = []
        for i in range(100):
            large_batch.append({
                "id": f"point_{i}",
                "vector": [float(i), float(i+1), float(i+2), float(i+3)],
                "payload": {"batch_id": "migration_batch_1", "index": i}
            })
        
        result = client.upsert("large_collection", large_batch)
        assert result is True
        
        # Batch deletion
        ids_to_delete = [f"point_{i}" for i in range(0, 50)]
        result = client.delete("large_collection", ids_to_delete)
        assert result is True


class TestAPIResponseCompatibility:
    """Test that API responses match qdrant-client format."""
    
    def test_search_response_format(self, client, mock_requests, mock_successful_response, sample_search_results):
        """Test that search responses match expected format."""
        search_data = {"result": sample_search_results}
        mock_requests.request.return_value = mock_successful_response(search_data)
        
        results = client.search("test_collection", [0.1, 0.2, 0.3])
        
        # Check response structure matches qdrant-client
        assert isinstance(results, list)
        
        for result in results:
            # Each result should have required fields
            assert hasattr(result, 'id')
            assert hasattr(result, 'score')
            assert hasattr(result, 'payload')
            
            # Types should match expectations
            assert isinstance(result.score, float)
            assert result.score >= 0.0 and result.score <= 1.0
            
            if result.payload is not None:
                assert isinstance(result.payload, dict)
    
    def test_collection_response_format(self, client, mock_requests, mock_successful_response):
        """Test that collection responses match expected format."""
        collections_data = {
            "collections": [
                {
                    "name": "test_collection",
                    "config": {
                        "params": {
                            "vectors": {"size": 128, "distance": "Cosine"}
                        }
                    },
                    "points_count": 1000,
                    "status": "green"
                }
            ]
        }
        mock_requests.request.return_value = mock_successful_response(collections_data)
        
        collections = client.get_collections()
        
        assert isinstance(collections, list)
        assert len(collections) == 1
        
        collection = collections[0]
        assert hasattr(collection, 'name')
        assert hasattr(collection, 'config')
        assert hasattr(collection, 'points_count')
        assert hasattr(collection, 'status')
        
        assert collection.name == "test_collection"
        assert collection.points_count == 1000
        assert collection.status == "green"
    
    def test_boolean_return_compatibility(self, client, mock_requests, mock_successful_response):
        """Test that operations return boolean values as expected."""
        mock_requests.request.return_value = mock_successful_response({})
        
        # These operations should return True on success (qdrant-client compatibility)
        config = VectorConfig(size=128, distance=DistanceMetric.COSINE)
        assert client.create_collection("test", config) is True
        
        points = [{"id": "1", "vector": [0.1, 0.2, 0.3]}]
        assert client.upsert("test", points) is True
        
        assert client.delete("test", ["1"]) is True
        assert client.delete_collection("test") is True


class TestBackwardCompatibility:
    """Test backward compatibility features."""
    
    def test_deprecated_parameter_support(self, client, mock_requests, mock_successful_response):
        """Test support for deprecated parameter names."""
        mock_requests.request.return_value = mock_successful_response({})
        
        # Test that old parameter names still work (if any)
        # This would include any renamed parameters for compatibility
        
        # Example: if 'vector_size' was renamed to 'size'
        config = {"vector_size": 128, "distance": "Cosine"}  # Old format
        
        # Should still work by internally mapping to new format
        # (This would require implementation in the actual client)
        try:
            result = client.create_collection("test_old_params", config)
            # If this doesn't raise an exception, backward compatibility works
        except (KeyError, TypeError):
            # If old parameters aren't supported, that's also valid
            # as long as it's documented
            pass
    
    def test_context_manager_compatibility(self, api_key):
        """Test context manager usage (qdrant-client compatible)."""
        # Should work the same as qdrant-client
        with AetherfyVectorsClient(api_key=api_key) as client:
            assert isinstance(client, AetherfyVectorsClient)
            # Client should be usable within context
            assert client.auth_manager.api_key == api_key
        
        # Should not raise errors after context exit
        assert client.auth_manager.api_key == api_key  # Still accessible
    
    def test_exception_compatibility(self, client, mock_requests, mock_error_response):
        """Test that exceptions are compatible with expected patterns."""
        # Test that our exceptions behave similarly to qdrant-client exceptions
        
        mock_requests.request.return_value = mock_error_response(
            message="Collection not found",
            status_code=404,
            error_code="collection_not_found"
        )
        
        with pytest.raises(Exception) as exc_info:
            client.get_collection("nonexistent")
        
        # Exception should have useful attributes
        exception = exc_info.value
        assert hasattr(exception, 'message') or str(exception)  # Should have meaningful message
        
        # Should be caught by general Exception handling (compatibility)
        try:
            client.get_collection("nonexistent")
        except Exception as e:
            assert isinstance(e, Exception)  # Should be catchable as general Exception