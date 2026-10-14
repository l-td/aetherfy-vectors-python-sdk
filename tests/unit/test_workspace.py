"""
Workspace functionality tests for Aetherfy Vectors Python SDK.

Every collection lives in a workspace: a name, or (workspace="auto") the
agent's workspace id from AETHERFY_WORKSPACE_ID. There is no "no workspace".
"""

import pytest
from unittest.mock import Mock
from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import Collection, VectorConfig, DistanceMetric


WS_ID = "0f8e2b9a-7c41-4d2e-9b3a-5e6f7a8b9c0d"
KEY = "afy_test_1234567890123456"


class TestWorkspaceInitialization:
    """Every collection lives in a workspace: a name, or the agent's id."""

    def test_workspace_by_name(self):
        client = AetherfyVectorsClient(api_key=KEY, workspace="my-workspace")

        assert client.workspace == "my-workspace"
        assert client.workspace_id is None
        assert client._build_collections_list_path() == (
            "workspaces/my-workspace/collections"
        )

    def test_auto_reads_the_workspace_id(self, monkeypatch):
        monkeypatch.setenv("AETHERFY_WORKSPACE_ID", WS_ID)
        client = AetherfyVectorsClient(api_key=KEY, workspace="auto")

        assert client.workspace is None
        assert client.workspace_id == WS_ID
        assert client._build_collections_list_path() == (
            f"workspace-ids/{WS_ID}/collections"
        )

    def test_default_is_auto(self, monkeypatch):
        monkeypatch.setenv("AETHERFY_WORKSPACE_ID", WS_ID)
        client = AetherfyVectorsClient(api_key=KEY)

        assert client.workspace_id == WS_ID
        assert client._build_collection_path("docs", "/points") == (
            f"workspace-ids/{WS_ID}/collections/docs/points"
        )

    def test_the_old_name_variable_is_not_read(self, monkeypatch):
        """AETHERFY_WORKSPACE (the name) is gone; only the id is read."""
        monkeypatch.delenv("AETHERFY_WORKSPACE_ID", raising=False)
        monkeypatch.setenv("AETHERFY_WORKSPACE", "env-workspace")

        with pytest.raises(ValueError, match="AETHERFY_WORKSPACE_ID"):
            AetherfyVectorsClient(api_key=KEY)

    @pytest.mark.parametrize("env", [None, ""])
    def test_auto_outside_an_agent_is_refused(self, monkeypatch, env):
        if env is None:
            monkeypatch.delenv("AETHERFY_WORKSPACE_ID", raising=False)
        else:
            monkeypatch.setenv("AETHERFY_WORKSPACE_ID", env)

        with pytest.raises(ValueError) as exc:
            AetherfyVectorsClient(api_key=KEY, workspace="auto")

        # Both fixes are named.
        assert "workspace name" in str(exc.value)
        assert "inside an Aetherfy agent" in str(exc.value)

    @pytest.mark.parametrize("workspace", [None, ""])
    def test_no_workspace_is_refused(self, monkeypatch, workspace):
        """None and "" are refused even inside an agent: no fallback."""
        monkeypatch.setenv("AETHERFY_WORKSPACE_ID", WS_ID)

        with pytest.raises(ValueError) as exc:
            AetherfyVectorsClient(api_key=KEY, workspace=workspace)

        assert "workspace name" in str(exc.value)
        assert "inside an Aetherfy agent" in str(exc.value)

    def test_workspace_is_read_only(self):
        client = AetherfyVectorsClient(api_key=KEY, workspace="my-workspace")

        with pytest.raises(AttributeError):
            client.workspace = "other"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            client.workspace_id = WS_ID  # type: ignore[misc]


class TestCollectionScoping:
    """The local cache key never lets two workspaces share an entry."""

    def test_scope_by_name(self):
        client = AetherfyVectorsClient(api_key=KEY, workspace="my-workspace")

        assert client._scope_collection("documents") == "my-workspace/documents"

    def test_scope_by_id(self, monkeypatch):
        monkeypatch.setenv("AETHERFY_WORKSPACE_ID", WS_ID)
        client = AetherfyVectorsClient(api_key=KEY)

        assert client._scope_collection("documents") == f"id:{WS_ID}/documents"

    def test_name_and_id_forms_never_share_a_cache_entry(self, monkeypatch):
        """A workspace NAMED like an id still gets a different key."""
        monkeypatch.setenv("AETHERFY_WORKSPACE_ID", WS_ID)
        by_id = AetherfyVectorsClient(api_key=KEY)
        by_name = AetherfyVectorsClient(api_key=KEY, workspace=WS_ID)
        other = AetherfyVectorsClient(api_key=KEY, workspace="other")

        keys = {
            by_id._scope_collection("documents"),
            by_name._scope_collection("documents"),
            other._scope_collection("documents"),
        }
        assert len(keys) == 3

    def test_workspace_in_the_url_is_encoded(self):
        client = AetherfyVectorsClient(api_key=KEY, workspace="a b")

        assert client._build_collection_path("c/d") == (
            "workspaces/a%20b/collections/c%2Fd"
        )


class TestVectorOperationsWithWorkspace:
    """Test vector operations use workspace scoping."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP session."""
        client = AetherfyVectorsClient(api_key=KEY, workspace="test-workspace")
        client._make_request = Mock()
        yield client

    def test_search_with_workspace(self, mock_client):
        """Test search operation scopes collection name."""
        mock_client._make_request.return_value = {"result": []}

        mock_client.search(
            collection_name="documents", query_vector=[0.1, 0.2, 0.3], limit=10
        )

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert call_args[0][0] == "POST"
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]
        assert call_args[0][1].endswith("points/query")

    def test_upsert_with_workspace(self, mock_client):
        """Test upsert operation scopes collection name."""
        # Mock the schema fetch
        mock_client._get_cached_schema = Mock(return_value={"size": 2})
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.upsert(
            collection_name="documents",
            points=[{"id": 1, "vector": [0.1, 0.2], "payload": {"text": "test"}}],
        )

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]
        assert "points" in call_args[0][1]

    def test_retrieve_with_workspace(self, mock_client):
        """Test retrieve operation scopes collection name."""
        mock_client._make_request.return_value = {"result": []}

        mock_client.retrieve(collection_name="documents", ids=[1, 2])

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]
        assert "points" in call_args[0][1]

    def test_delete_with_workspace(self, mock_client):
        """Test delete operation scopes collection name."""
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.delete(collection_name="documents", points_selector=[1, 2])

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
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
            "result": {"count": 42},
            "status": "ok",
        }

        result = mock_client.count(collection_name="documents")

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]
        assert result == 42


class TestCollectionManagementWithWorkspace:
    """Test collection management operations with workspace."""

    @pytest.fixture
    def mock_client(self):
        """Create a client with mocked HTTP session."""
        client = AetherfyVectorsClient(api_key=KEY, workspace="test-workspace")
        client._make_request = Mock()
        yield client

    def test_create_collection_with_workspace(self, mock_client):
        """Post-A/B: workspace lives in URL path, body name is bare.
        vectordb rejects body name containing "/" with 400
        INVALID_COLLECTION_NAME."""
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.create_collection(
            collection_name="documents",
            vectors_config=VectorConfig(size=384, distance=DistanceMetric.COSINE),
        )

        # URL must be the nested list/create endpoint.
        call_args = mock_client._make_request.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "workspaces/test-workspace/collections"

        # Body name must be BARE.
        request_data = call_args[0][2]
        assert request_data["name"] == "documents"
        assert "/" not in request_data["name"]

    def test_delete_collection_with_workspace(self, mock_client):
        """Test delete_collection scopes collection name."""
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.delete_collection(collection_name="documents")

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]

    def test_collection_exists_with_workspace(self, mock_client):
        """Test collection_exists scopes collection name."""
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.collection_exists(collection_name="documents")

        # Verify the request was made with scoped collection in URL
        call_args = mock_client._make_request.call_args
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]

    def test_get_collection_with_workspace(self, mock_client):
        """Post-A/B: vectordb returns the bare collection name (PG
        stores name without workspace prefix; workspace_id is the join
        key). The SDK no longer needs to unscope client-side."""
        mock_client._make_request.return_value = {
            "name": "documents",
            "config": {"params": {"vectors": {"size": 384, "distance": "Cosine"}}},
        }

        result = mock_client.get_collection(collection_name="documents")

        # Request should use nested URL form.
        call_args = mock_client._make_request.call_args
        assert "workspaces/test-workspace/collections/documents" in call_args[0][1]

        # Response name is bare as returned by vectordb.
        assert result.name == "documents"

    def test_get_collections_with_workspace(self, mock_client):
        """Post-A/B: GET /workspaces/{ws}/collections returns ONLY this
        workspace's collections (server-side filter by workspace_id),
        with bare names (PG schema). The SDK no longer filters or
        unscopes client-side — pinning that contract here.

        The OLD mock returned mixed-workspace names and verified
        client-side prefix-matching; that's gone."""
        mock_client._make_request.return_value = {
            "collections": [
                {
                    "name": "documents",
                    "config": {
                        "params": {"vectors": {"size": 384, "distance": "Cosine"}}
                    },
                },
                {
                    "name": "images",
                    "config": {
                        "params": {"vectors": {"size": 512, "distance": "Cosine"}}
                    },
                },
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
        assert "documents" in names
        assert "images" in names


class TestSchemaOperationsWithWorkspace:
    """Schema routes live under the collection, in its workspace."""

    SCHEMA = "workspaces/test-workspace/collections/documents/schema"

    @pytest.fixture
    def mock_client(self):
        client = AetherfyVectorsClient(api_key=KEY, workspace="test-workspace")
        client._make_request = Mock()
        yield client

    def test_get_schema_with_workspace(self, mock_client):
        mock_client._make_request.return_value = {
            "schema": {"fields": {}},
            "etag": "schema-v1",
            "enforcement_mode": "off",
            "description": None,
        }

        mock_client.get_schema(collection_name="documents")

        assert mock_client._make_request.call_args[0][:2] == ("GET", self.SCHEMA)

    def test_set_schema_with_workspace(self, mock_client):
        from aetherfy_vectors.schema import Schema, FieldDefinition

        mock_client._make_request.return_value = {"etag": "schema-v2"}
        schema = Schema(
            fields={"title": FieldDefinition(type="keyword", required=True)}
        )

        mock_client.set_schema(collection_name="documents", schema=schema)

        assert mock_client._make_request.call_args[0][:2] == ("PUT", self.SCHEMA)

    def test_delete_schema_with_workspace(self, mock_client):
        mock_client._make_request.return_value = {"status": "ok"}

        mock_client.delete_schema(collection_name="documents")

        assert mock_client._make_request.call_args[0][:2] == (
            "DELETE",
            self.SCHEMA,
        )

    def test_analyze_schema_with_workspace(self, mock_client):
        mock_client._make_request.return_value = {
            "collection": "documents",
            "sample_size": 1000,
            "total_points": 5000,
            "fields": {},
            "suggested_schema": {"fields": {}},
            "processing_time_ms": 150,
        }

        mock_client.analyze_schema(collection_name="documents")

        assert mock_client._make_request.call_args[0][:2] == (
            "POST",
            self.SCHEMA + "/analyze",
        )

    def test_refresh_schema_with_workspace(self, mock_client):
        mock_client._make_request.return_value = {
            "schema": {"fields": {}},
            "etag": "schema-v3",
            "enforcement_mode": "off",
            "description": None,
        }

        mock_client.refresh_schema(collection_name="documents")

        assert mock_client._make_request.call_args[0][:2] == ("GET", self.SCHEMA)

    def test_schema_by_workspace_id(self, monkeypatch):
        monkeypatch.setenv("AETHERFY_WORKSPACE_ID", WS_ID)
        client = AetherfyVectorsClient(api_key=KEY)
        client._make_request = Mock(return_value={"status": "ok"})

        client.delete_schema(collection_name="documents")

        assert client._make_request.call_args[0][1] == (
            f"workspace-ids/{WS_ID}/collections/documents/schema"
        )

    def test_clear_schema_cache_takes_the_bare_name(self, mock_client):
        key = mock_client._scope_collection("documents")
        mock_client._schema_cache[key] = {"size": 3}

        mock_client.clear_schema_cache("documents")

        assert key not in mock_client._schema_cache
