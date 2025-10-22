"""
Tests for ETag-based schema validation and caching
"""

import pytest
from unittest.mock import Mock
from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.exceptions import AetherfyVectorsException, ValidationError


class TestETagValidation:
    """Test ETag-based schema validation"""

    @pytest.fixture
    def mock_collection_response(self):
        """Mock collection info response with ETag"""
        return {
            "result": {
                "config": {
                    "params": {
                        "vectors": {
                            "size": 768,
                            "distance": "Cosine"
                        }
                    }
                }
            },
            "schema_version": "abc12345"
        }

    @pytest.fixture
    def mock_successful_upsert_response(self):
        """Mock successful upsert response"""
        def _create_response(status_code=200):
            mock_response = Mock()
            mock_response.status_code = status_code
            mock_response.json.return_value = {"result": {"status": "acknowledged"}}
            mock_response.content = True
            return mock_response
        return _create_response

    def test_schema_cache_on_first_upsert(self, client, mock_requests, mock_collection_response, mock_successful_upsert_response):
        """Test that schema is fetched and cached on first upsert"""
        # First call: GET collection info
        # Second call: PUT upsert
        mock_requests.request.side_effect = [
            Mock(status_code=200, json=lambda: mock_collection_response, content=True),
            mock_successful_upsert_response()
        ]

        # First upsert
        points = [{"id": "1", "vector": [0.1] * 768, "payload": {}}]
        client.upsert("test-collection", points)

        # Should have called request twice (GET schema, PUT upsert)
        assert mock_requests.request.call_count == 2
        assert "test-collection" in client._schema_cache

    def test_schema_cache_reused_on_subsequent_upserts(self, client, mock_requests, mock_collection_response, mock_successful_upsert_response):
        """Test that cached schema is reused on subsequent upserts"""
        # First upsert: GET schema + PUT upsert
        # Second upsert: only PUT (cache hit)
        mock_requests.request.side_effect = [
            Mock(status_code=200, json=lambda: mock_collection_response, content=True),  # GET schema
            mock_successful_upsert_response(),  # First PUT
            mock_successful_upsert_response(),  # Second PUT (no GET)
        ]

        points = [{"id": "1", "vector": [0.1] * 768, "payload": {}}]

        # First upsert - should fetch schema
        client.upsert("test-collection", points)
        call_count_after_first = mock_requests.request.call_count

        # Second upsert - should use cache
        client.upsert("test-collection", points)

        # Should only add one more call (PUT), not two (GET+PUT)
        assert mock_requests.request.call_count == call_count_after_first + 1

    def test_etag_sent_in_upsert_header(self, client, mock_requests, mock_collection_response, mock_successful_upsert_response):
        """Test that ETag is sent in If-Match header"""
        mock_requests.request.side_effect = [
            Mock(status_code=200, json=lambda: mock_collection_response, content=True),
            mock_successful_upsert_response()
        ]

        # Upsert
        points = [{"id": "1", "vector": [0.1] * 768, "payload": {}}]
        client.upsert("test-collection", points)

        # Check If-Match header was sent in the PUT request (second call)
        put_call_kwargs = mock_requests.request.call_args_list[1][1]
        assert "If-Match" in put_call_kwargs["headers"]
        assert put_call_kwargs["headers"]["If-Match"] == "abc12345"

    def test_dimension_mismatch_caught_before_request(self, client, mock_requests, mock_collection_response):
        """Test that dimension mismatch is caught client-side before making request"""
        # Only mock GET schema - PUT should never be called
        mock_requests.request.return_value = Mock(
            status_code=200,
            json=lambda: mock_collection_response,
            content=True
        )

        # Upsert with wrong dimensions
        points = [{"id": "1", "vector": [0.1] * 384, "payload": {}}]  # Wrong size!

        with pytest.raises(ValueError) as exc_info:
            client.upsert("test-collection", points)

        # Should fail with dimension mismatch
        assert "dimension mismatch" in str(exc_info.value).lower()
        assert "expected 768" in str(exc_info.value)
        assert "got 384" in str(exc_info.value)

        # Should only have called GET (not PUT) - failed validation client-side
        assert mock_requests.request.call_count == 1

    def test_schema_changed_412_response(self, client, mock_requests, mock_collection_response):
        """Test handling of 412 response when schema changes"""
        # Mock GET schema, then mock 412 error on PUT
        mock_412_response = Mock()
        mock_412_response.status_code = 412
        mock_412_response.json.return_value = {
            "error": {
                "code": "SCHEMA_VERSION_MISMATCH",
                "message": "Collection schema has changed"
            }
        }
        mock_412_response.content = True

        mock_requests.request.side_effect = [
            Mock(status_code=200, json=lambda: mock_collection_response, content=True),
            mock_412_response
        ]

        # Upsert should fail with schema changed error
        points = [{"id": "1", "vector": [0.1] * 768, "payload": {}}]

        with pytest.raises(ValidationError) as exc_info:
            client.upsert("test-collection", points)

        # Should mention schema changed
        assert "schema changed" in str(exc_info.value).lower()

        # Cache should be cleared
        assert "test-collection" not in client._schema_cache

    def test_clear_schema_cache_single_collection(self, client):
        """Test clearing cache for a single collection"""
        # Populate cache
        client._schema_cache["collection1"] = {"size": 768, "etag": "abc"}
        client._schema_cache["collection2"] = {"size": 384, "etag": "def"}

        # Clear one collection
        client.clear_schema_cache("collection1")

        # collection1 should be cleared
        assert "collection1" not in client._schema_cache
        # collection2 should remain
        assert "collection2" in client._schema_cache

    def test_clear_schema_cache_all_collections(self, client):
        """Test clearing cache for all collections"""
        # Populate cache
        client._schema_cache["collection1"] = {"size": 768, "etag": "abc"}
        client._schema_cache["collection2"] = {"size": 384, "etag": "def"}

        # Clear all
        client.clear_schema_cache()

        # Both should be cleared
        assert len(client._schema_cache) == 0

    def test_backend_validation_error_400(self, client, mock_requests, mock_collection_response):
        """Test handling of 400 validation error from backend"""
        # Mock GET schema, then mock 400 error on PUT
        mock_400_response = Mock()
        mock_400_response.status_code = 400
        mock_400_response.json.return_value = {
            "error": {
                "code": "DIMENSION_MISMATCH",
                "message": "Vector dimension mismatch: expected 768, got 384"
            }
        }
        mock_400_response.content = True

        mock_requests.request.side_effect = [
            Mock(status_code=200, json=lambda: mock_collection_response, content=True),
            mock_400_response
        ]

        # Upsert
        points = [{"id": "1", "vector": [0.1] * 768, "payload": {}}]

        # SDK converts ValidationError to ValueError for backward compatibility
        with pytest.raises(ValueError) as exc_info:
            client.upsert("test-collection", points)

        # Should contain error message from backend
        assert "dimension mismatch" in str(exc_info.value).lower()

    def test_server_error_500(self, client, mock_requests, mock_collection_response):
        """Test handling of 500 server error"""
        # Mock GET schema, then mock 500 error on PUT
        mock_500_response = Mock()
        mock_500_response.status_code = 500
        mock_500_response.json.return_value = {
            "error": {
                "message": "Internal server error"
            }
        }
        mock_500_response.content = b'{"error":{"message":"Internal server error"}}'

        mock_requests.request.side_effect = [
            Mock(status_code=200, json=lambda: mock_collection_response, content=True),
            mock_500_response
        ]

        # Upsert should fail with server error
        points = [{"id": "1", "vector": [0.1] * 768, "payload": {}}]

        with pytest.raises(AetherfyVectorsException) as exc_info:
            client.upsert("test-collection", points)

        # Should mention server error
        assert "server error" in str(exc_info.value).lower()
