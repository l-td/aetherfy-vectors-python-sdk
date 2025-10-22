"""
Tests for ETag-based schema validation and caching
"""

import pytest
from unittest.mock import Mock, patch
from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.exceptions import AetherfyVectorsException


class TestETagValidation:
    """Test ETag-based schema validation"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return AetherfyVectorsClient(api_key="test_key", endpoint="http://localhost:3000")

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

    def test_schema_cache_on_first_upsert(self, client, mock_collection_response):
        """Test that schema is fetched and cached on first upsert"""
        with patch('requests.get') as mock_get, \
             patch('requests.put') as mock_put:

            # Mock collection GET response
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_collection_response

            # Mock upsert response
            mock_put.return_value.status_code = 200
            mock_put.return_value.json.return_value = {"success": True}

            # First upsert
            points = [{"id": "1", "vector": [0.1] * 768, "payload": {}}]
            client.upsert("test-collection", points)

            # Should have called GET to fetch schema
            assert mock_get.called
            assert "test-collection" in client._schema_cache

    def test_schema_cache_reused_on_subsequent_upserts(self, client, mock_collection_response):
        """Test that cached schema is reused on subsequent upserts"""
        with patch('requests.get') as mock_get, \
             patch('requests.put') as mock_put:

            # Mock collection GET response
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_collection_response

            # Mock upsert response
            mock_put.return_value.status_code = 200
            mock_put.return_value.json.return_value = {"success": True}

            # First upsert - should fetch schema
            points = [{"id": "1", "vector": [0.1] * 768, "payload": {}}]
            client.upsert("test-collection", points)

            get_call_count_after_first = mock_get.call_count

            # Second upsert - should use cache
            client.upsert("test-collection", points)

            # GET should not be called again
            assert mock_get.call_count == get_call_count_after_first

    def test_etag_sent_in_upsert_header(self, client, mock_collection_response):
        """Test that ETag is sent in If-Match header"""
        with patch('requests.get') as mock_get, \
             patch('requests.put') as mock_put:

            # Mock collection GET response
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_collection_response

            # Mock upsert response
            mock_put.return_value.status_code = 200
            mock_put.return_value.json.return_value = {"success": True}

            # Upsert
            points = [{"id": "1", "vector": [0.1] * 768, "payload": {}}]
            client.upsert("test-collection", points)

            # Check If-Match header was sent
            _, kwargs = mock_put.call_args
            assert "If-Match" in kwargs["headers"]
            assert kwargs["headers"]["If-Match"] == "abc12345"

    def test_dimension_mismatch_caught_before_request(self, client, mock_collection_response):
        """Test that dimension mismatch is caught client-side before making request"""
        with patch('requests.get') as mock_get, \
             patch('requests.put') as mock_put:

            # Mock collection GET response
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_collection_response

            # Upsert with wrong dimensions
            points = [{"id": "1", "vector": [0.1] * 384, "payload": {}}]  # Wrong size!

            with pytest.raises(ValueError) as exc_info:
                client.upsert("test-collection", points)

            # Should fail with dimension mismatch
            assert "dimension mismatch" in str(exc_info.value).lower()
            assert "expected 768" in str(exc_info.value)
            assert "got 384" in str(exc_info.value)

            # Should NOT have called PUT (failed validation client-side)
            assert not mock_put.called

    def test_schema_changed_412_response(self, client, mock_collection_response):
        """Test handling of 412 response when schema changes"""
        with patch('requests.get') as mock_get, \
             patch('requests.put') as mock_put:

            # Mock collection GET response
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_collection_response

            # Mock 412 response (schema changed)
            mock_put.return_value.status_code = 412
            mock_put.return_value.json.return_value = {
                "error": {
                    "code": "SCHEMA_VERSION_MISMATCH",
                    "message": "Collection schema has changed"
                }
            }

            # Upsert should fail with schema changed error
            points = [{"id": "1", "vector": [0.1] * 768, "payload": {}}]

            with pytest.raises(AetherfyVectorsException) as exc_info:
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

    def test_backend_validation_error_400(self, client, mock_collection_response):
        """Test handling of 400 validation error from backend"""
        with patch('requests.get') as mock_get, \
             patch('requests.put') as mock_put:

            # Mock collection GET response
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_collection_response

            # Mock 400 response (backend validation)
            mock_put.return_value.status_code = 400
            mock_put.return_value.json.return_value = {
                "error": {
                    "code": "DIMENSION_MISMATCH",
                    "message": "Vector dimension mismatch: expected 768, got 384"
                }
            }

            # Upsert
            points = [{"id": "1", "vector": [0.1] * 768, "payload": {}}]

            with pytest.raises(ValueError) as exc_info:
                client.upsert("test-collection", points)

            # Should contain error message from backend
            assert "dimension mismatch" in str(exc_info.value).lower()

    def test_server_error_500(self, client, mock_collection_response):
        """Test handling of 500 server error"""
        with patch('requests.get') as mock_get, \
             patch('requests.put') as mock_put:

            # Mock collection GET response
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_collection_response

            # Mock 500 response
            mock_put.return_value.status_code = 500
            mock_put.return_value.json.return_value = {
                "error": {
                    "message": "Internal server error"
                }
            }
            mock_put.return_value.content = b'{"error":{"message":"Internal server error"}}'

            # Upsert should fail with server error
            points = [{"id": "1", "vector": [0.1] * 768, "payload": {}}]

            with pytest.raises(AetherfyVectorsException) as exc_info:
                client.upsert("test-collection", points)

            # Should mention server error
            assert "server error" in str(exc_info.value).lower()
