"""
Workspace functionality tests for Aetherfy Vectors Python SDK.

Tests workspace auto-detection, collection scoping, and isolation.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import Collection, VectorConfig, DistanceMetric


class TestWorkspaceInitialization:
    """Test workspace initialization and auto-detection."""

    def test_workspace_manual_setting(self):
        """Test setting workspace manually."""
        client = AetherfyVectorsClient(
            api_key="afy_test_1234567890123456",
            workspace="my-workspace"
        )

        assert client.workspace == "my-workspace"

    def test_workspace_auto_detection(self):
        """Test workspace='auto' reads from environment."""
        with patch.dict(os.environ, {'AETHERFY_WORKSPACE': 'env-workspace'}):
            client = AetherfyVectorsClient(
                api_key="afy_test_1234567890123456",
                workspace='auto'
            )

            assert client.workspace == "env-workspace"

    def test_workspace_auto_no_env_var(self):
        """Test workspace='auto' when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            client = AetherfyVectorsClient(
                api_key="afy_test_1234567890123456",
                workspace='auto'
            )

            assert client.workspace is None

    def test_no_workspace(self):
        """Test client without workspace."""
        client = AetherfyVectorsClient(api_key="afy_test_1234567890123456")

        assert client.workspace is None


class TestCollectionScoping:
    """Test collection name scoping with workspaces."""

    def test_scope_collection_with_workspace(self):
        """Test scoping collection name with workspace."""
        client = AetherfyVectorsClient(
            api_key="afy_test_1234567890123456",
            workspace="my-workspace"
        )

        scoped = client._scope_collection("documents")
        assert scoped == "my-workspace/documents"

    def test_scope_collection_without_workspace(self):
        """Test scoping collection name without workspace."""
        client = AetherfyVectorsClient(api_key="afy_test_1234567890123456")

        scoped = client._scope_collection("documents")
        assert scoped == "documents"

    def test_unscope_collection_with_workspace(self):
        """Test unscoping collection name with workspace."""
        client = AetherfyVectorsClient(
            api_key="afy_test_1234567890123456",
            workspace="my-workspace"
        )

        unscoped = client._unscope_collection("my-workspace/documents")
        assert unscoped == "documents"

    def test_unscope_collection_without_workspace(self):
        """Test unscoping collection name without workspace."""
        client = AetherfyVectorsClient(api_key="afy_test_1234567890123456")

        unscoped = client._unscope_collection("documents")
        assert unscoped == "documents"

    def test_unscope_collection_wrong_workspace(self):
        """Test unscoping collection with different workspace prefix."""
        client = AetherfyVectorsClient(
            api_key="afy_test_1234567890123456",
            workspace="my-workspace"
        )

        # Should return as-is if doesn't match workspace
        unscoped = client._unscope_collection("other-workspace/documents")
        assert unscoped == "other-workspace/documents"


class TestVectorOperationsWithWorkspace:
    """Test vector operations use workspace scoping."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP session."""
        client = AetherfyVectorsClient(
            api_key="afy_test_1234567890123456",
            workspace="test-workspace"
        )
        client._make_request = Mock()
        yield client

    def test_search_with_workspace(self, mock_client):
        """Test search operation scopes collection name."""
        mock_client._make_request.return_value = {"result": []}

        mock_client.search(
            collection_name="documents",
            query_vector=[0.1, 0.2, 0.3],
            limit=10
        )

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert call_args[0][0] == "POST"
        assert "test-workspace/documents" in call_args[0][1]
        assert "points/search" in call_args[0][1]

    def test_upsert_with_workspace(self, mock_client):
        """Test upsert operation scopes collection name."""
        # Mock the schema fetch
        mock_client._get_cached_schema = Mock(return_value={"size": 2})
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.upsert(
            collection_name="documents",
            points=[
                {"id": "1", "vector": [0.1, 0.2], "payload": {"text": "test"}}
            ]
        )

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "test-workspace/documents" in call_args[0][1]
        assert "points" in call_args[0][1]

    def test_retrieve_with_workspace(self, mock_client):
        """Test retrieve operation scopes collection name."""
        mock_client._make_request.return_value = {"result": []}

        mock_client.retrieve(
            collection_name="documents",
            ids=["1", "2"]
        )

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "test-workspace/documents" in call_args[0][1]
        assert "points" in call_args[0][1]

    def test_delete_with_workspace(self, mock_client):
        """Test delete operation scopes collection name."""
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.delete(
            collection_name="documents",
            points_selector=["1", "2"]
        )

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "test-workspace/documents" in call_args[0][1]
        assert "points/delete" in call_args[0][1]

    def test_count_with_workspace(self, mock_client):
        """Test count operation scopes collection name."""
        mock_client._make_request.return_value = {"count": 42}

        result = mock_client.count(collection_name="documents")

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "test-workspace/documents" in call_args[0][1]
        assert result == 42


class TestCollectionManagementWithWorkspace:
    """Test collection management operations with workspace."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP session."""
        client = AetherfyVectorsClient(
            api_key="afy_test_1234567890123456",
            workspace="test-workspace"
        )
        client._make_request = Mock()
        yield client

    def test_create_collection_with_workspace(self, mock_client):
        """Test create_collection scopes collection name."""
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.create_collection(
            collection_name="documents",
            vectors_config=VectorConfig(size=384, distance=DistanceMetric.COSINE)
        )

        # Verify the request data contains scoped collection name
        call_args = mock_client._make_request.call_args
        request_data = call_args[0][2]
        assert request_data['name'] == 'test-workspace/documents'

    def test_delete_collection_with_workspace(self, mock_client):
        """Test delete_collection scopes collection name."""
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.delete_collection(collection_name="documents")

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "test-workspace/documents" in call_args[0][1]

    def test_collection_exists_with_workspace(self, mock_client):
        """Test collection_exists scopes collection name."""
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.collection_exists(collection_name="documents")

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "test-workspace/documents" in call_args[0][1]

    def test_get_collection_with_workspace(self, mock_client):
        """Test get_collection scopes and unscopes collection name."""
        mock_client._make_request.return_value = {
            "name": "test-workspace/documents",
            "config": {"params": {"vectors": {"size": 384, "distance": "Cosine"}}}
        }

        result = mock_client.get_collection(collection_name="documents")

        # Request should use scoped name
        call_args = mock_client._make_request.call_args
        assert "test-workspace/documents" in call_args[0][1]

        # Response should have unscoped name
        assert result.name == 'documents'

    def test_get_collections_with_workspace(self, mock_client):
        """Test get_collections filters and unscopes workspace collections."""
        mock_client._make_request.return_value = {
            "collections": [
                {"name": "test-workspace/documents", "config": {"params": {"vectors": {"size": 384, "distance": "Cosine"}}}},
                {"name": "test-workspace/images", "config": {"params": {"vectors": {"size": 512, "distance": "Cosine"}}}},
                {"name": "other-workspace/data", "config": {"params": {"vectors": {"size": 256, "distance": "Cosine"}}}},
                {"name": "no-workspace", "config": {"params": {"vectors": {"size": 128, "distance": "Cosine"}}}}
            ]
        }

        result = mock_client.get_collections()

        # Should only return collections from this workspace, unscoped
        assert len(result) == 2
        names = [c.name for c in result]
        assert 'documents' in names
        assert 'images' in names
        assert 'data' not in names
        assert 'no-workspace' not in names


class TestSchemaOperationsWithWorkspace:
    """Test schema operations with workspace."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP session."""
        client = AetherfyVectorsClient(
            api_key="afy_test_1234567890123456",
            workspace="test-workspace"
        )
        client._make_request = Mock()
        yield client

    def test_get_schema_with_workspace(self, mock_client):
        """Test get_schema scopes collection name."""
        mock_client._make_request.return_value = {
            "schema": {"fields": {}},
            "etag": "schema-v1",
            "enforcement_mode": "off"
        }

        mock_client.get_schema(collection_name="documents")

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "test-workspace/documents" in call_args[0][1]
        assert "schema" in call_args[0][1]

    def test_set_schema_with_workspace(self, mock_client):
        """Test set_schema scopes collection name."""
        from aetherfy_vectors.schema import Schema, FieldDefinition

        mock_client._make_request.return_value = {"etag": "schema-v2"}

        schema = Schema(fields={"title": FieldDefinition(type="keyword", required=True)})

        mock_client.set_schema(
            collection_name="documents",
            schema=schema
        )

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "test-workspace/documents" in call_args[0][1]
        assert "schema" in call_args[0][1]

    def test_delete_schema_with_workspace(self, mock_client):
        """Test delete_schema scopes collection name."""
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.delete_schema(collection_name="documents")

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "test-workspace/documents" in call_args[0][1]
        assert "schema" in call_args[0][1]

    def test_analyze_schema_with_workspace(self, mock_client):
        """Test analyze_schema scopes collection name."""
        mock_client._make_request.return_value = {
            "collection": "test-workspace/documents",
            "sample_size": 1000,
            "total_points": 5000,
            "fields": {},
            "suggested_schema": {"fields": {}},
            "processing_time_ms": 150
        }

        mock_client.analyze_schema(collection_name="documents")

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "test-workspace/documents" in call_args[0][1]
        assert "schema" in call_args[0][1]
        assert "analyze" in call_args[0][1]

    def test_refresh_schema_with_workspace(self, mock_client):
        """Test refresh_schema scopes collection name."""
        mock_client._make_request.return_value = {
            "schema": {"fields": {}},
            "etag": "schema-v3",
            "enforcement_mode": "off"
        }

        mock_client.refresh_schema(collection_name="documents")

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "test-workspace/documents" in call_args[0][1]
        assert "schema" in call_args[0][1]


class TestBackwardCompatibility:
    """Test backward compatibility without workspace."""

    @pytest.fixture
    def mock_client(self):
        """Create a client without workspace."""
        client = AetherfyVectorsClient(api_key="afy_test_1234567890123456")
        client._make_request = Mock()
        yield client

    def test_operations_without_workspace(self, mock_client):
        """Test operations work without workspace (backward compat)."""
        mock_client._make_request.return_value = {"result": []}

        # Should work without workspace scoping
        mock_client.search(
            collection_name="documents",
            query_vector=[0.1, 0.2, 0.3]
        )

        call_args = mock_client._make_request.call_args
        # Collection should NOT be scoped (no workspace prefix)
        assert "collections/documents" in call_args[0][1]
        assert "test-workspace" not in call_args[0][1]

    def test_get_collections_without_workspace(self, mock_client):
        """Test get_collections returns all collections without workspace."""
        mock_client._make_request.return_value = {
            "collections": [
                {"name": "documents", "config": {"params": {"vectors": {"size": 384, "distance": "Cosine"}}}},
                {"name": "images", "config": {"params": {"vectors": {"size": 512, "distance": "Cosine"}}}}
            ]
        }

        result = mock_client.get_collections()

        # Should return all collections as-is
        assert len(result) == 2
        names = [c.name for c in result]
        assert 'documents' in names
        assert 'images' in names
