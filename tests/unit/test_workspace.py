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

    # _unscope_collection was removed post-A/B (vectordb returns bare
    # names; no client-side unscoping needed). The slash-form scoped
    # name is now only used as a local schemaCache key — no read path
    # needs the inverse. Tests for the deleted method removed.


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
        # Post-A/B: nested URL form, not URL-encoded slash.
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]
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
        # Post-A/B: nested URL form, not URL-encoded slash.
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]
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
        # Post-A/B: nested URL form, not URL-encoded slash.
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]
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
        # Post-A/B: nested URL form, not URL-encoded slash.
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]
        assert "points/delete" in call_args[0][1]

    def test_count_with_workspace(self, mock_client):
        """Test count operation scopes collection name.

        Mock matches Qdrant's actual response shape:
        {"result": {"count": N}, "status": "ok"}. A flat {"count": N}
        would have hidden the bug where the SDK was reading the wrong
        path (fixed in client.py:count — see test_count_points_success
        in test_client.py).
        """
        mock_client._make_request.return_value = {
            "result": {"count": 42}, "status": "ok"
        }

        result = mock_client.count(collection_name="documents")

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        # Post-A/B: nested URL form, not URL-encoded slash.
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]
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
        """Post-A/B: workspace lives in URL path, body name is bare.
        vectordb rejects body name containing "/" with 400
        INVALID_COLLECTION_NAME."""
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.create_collection(
            collection_name="documents",
            vectors_config=VectorConfig(size=384, distance=DistanceMetric.COSINE)
        )

        # URL must be the nested list/create endpoint.
        call_args = mock_client._make_request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "workspaces/test-workspace/collections"

        # Body name must be BARE.
        request_data = call_args[0][2]
        assert request_data['name'] == 'documents'
        assert '/' not in request_data['name']

    def test_delete_collection_with_workspace(self, mock_client):
        """Test delete_collection scopes collection name."""
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.delete_collection(collection_name="documents")

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        # Post-A/B: nested URL form, not URL-encoded slash.
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]

    def test_collection_exists_with_workspace(self, mock_client):
        """Test collection_exists scopes collection name."""
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.collection_exists(collection_name="documents")

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        # Post-A/B: nested URL form, not URL-encoded slash.
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]

    def test_get_collection_with_workspace(self, mock_client):
        """Post-A/B: vectordb returns the bare collection name (PG
        stores name without workspace prefix; workspace_id is the join
        key). The SDK no longer needs to unscope client-side."""
        mock_client._make_request.return_value = {
            "name": "documents",
            "config": {"params": {"vectors": {"size": 384, "distance": "Cosine"}}}
        }

        result = mock_client.get_collection(collection_name="documents")

        # Request should use nested URL form.
        call_args = mock_client._make_request.call_args
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]

        # Response name is bare as returned by vectordb.
        assert result.name == 'documents'

    def test_get_collections_with_workspace(self, mock_client):
        """Post-A/B: GET /workspaces/{ws}/collections returns ONLY this
        workspace's collections (server-side filter by workspace_id),
        with bare names (PG schema). The SDK no longer filters or
        unscopes client-side — pinning that contract here.

        The OLD mock returned mixed-workspace names and verified
        client-side prefix-matching; that's gone."""
        mock_client._make_request.return_value = {
            "collections": [
                {"name": "documents", "config": {"params": {"vectors": {"size": 384, "distance": "Cosine"}}}},
                {"name": "images", "config": {"params": {"vectors": {"size": 512, "distance": "Cosine"}}}},
            ]
        }

        result = mock_client.get_collections()

        # Request hits the nested list endpoint.
        call_args = mock_client._make_request.call_args
        assert call_args[0][0] == "GET"
        assert call_args[0][1] == "workspaces/test-workspace/collections"

        # No client-side filtering; names returned verbatim.
        assert len(result) == 2
        names = [c.name for c in result]
        assert 'documents' in names
        assert 'images' in names


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

    # Schema URLs stay in slash-form (vectordb's /api/v1/schema/:collection
    # router is unchanged in PR 1 of the A/B refactor — only /collections
    # got the nested route family). The SDK keeps using
    # f"schema/{quote_collection_name(scoped_name)}" which URL-encodes
    # the slash. Pinning that contract here so a future migration of
    # schema endpoints to nested form is a visible test failure.

    def test_get_schema_with_workspace(self, mock_client):
        """get_schema URL keeps slash-form (schema endpoints not nested)."""
        mock_client._make_request.return_value = {
            "schema": {"fields": {}},
            "etag": "schema-v1",
            "enforcement_mode": "off",
            "description": None
        }

        mock_client.get_schema(collection_name="documents")

        call_args = mock_client._make_request.call_args
        assert "schema/test-workspace%2Fdocuments" in call_args[0][1]

    def test_set_schema_with_workspace(self, mock_client):
        """set_schema URL keeps slash-form."""
        from aetherfy_vectors.schema import Schema, FieldDefinition

        mock_client._make_request.return_value = {"etag": "schema-v2"}

        schema = Schema(fields={"title": FieldDefinition(type="keyword", required=True)})

        mock_client.set_schema(
            collection_name="documents",
            schema=schema
        )

        call_args = mock_client._make_request.call_args
        assert "schema/test-workspace%2Fdocuments" in call_args[0][1]

    def test_delete_schema_with_workspace(self, mock_client):
        """delete_schema URL keeps slash-form."""
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.delete_schema(collection_name="documents")

        call_args = mock_client._make_request.call_args
        assert "schema/test-workspace%2Fdocuments" in call_args[0][1]

    def test_analyze_schema_with_workspace(self, mock_client):
        """analyze_schema URL keeps slash-form."""
        mock_client._make_request.return_value = {
            "collection": "test-workspace/documents",
            "sample_size": 1000,
            "total_points": 5000,
            "fields": {},
            "suggested_schema": {"fields": {}},
            "processing_time_ms": 150
        }

        mock_client.analyze_schema(collection_name="documents")

        call_args = mock_client._make_request.call_args
        assert "schema/test-workspace%2Fdocuments" in call_args[0][1]
        assert "analyze" in call_args[0][1]

    def test_refresh_schema_with_workspace(self, mock_client):
        """Test refresh_schema scopes collection name."""
        mock_client._make_request.return_value = {
            "schema": {"fields": {}},
            "etag": "schema-v3",
            "enforcement_mode": "off",
            "description": None
        }

        mock_client.refresh_schema(collection_name="documents")

        call_args = mock_client._make_request.call_args
        assert "schema/test-workspace%2Fdocuments" in call_args[0][1]


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
