"""
Workspaced URL contract tests for AetherfyVectorsClient.

Every collection lives in a workspace, so every collection-scoped HTTP call
uses one of two nested forms, with a bare collection name in the URL/body:

- ``workspaces/{name}/collections/{coll}[/<suffix>]`` for a workspace name;
- ``workspace-ids/{id}/collections/{coll}[/<suffix>]`` for workspace="auto",
  the agent's workspace id from ``AETHERFY_WORKSPACE_ID``.

There is no bare ``collections/...`` form: vectordb answers it with
400 WORKSPACE_REQUIRED.

The tests assert on the ``requests`` mock's call args — exact URLs,
exact body shapes — so any regression fails loudly.
"""

import pytest
from unittest.mock import Mock

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import DistanceMetric


WS = "my-workspace"
COLL = "my-coll"
ENDPOINT = "https://test-api.aetherfy.com"
API_KEY = "afy_test_1234567890abcdef1234"


def _ok_response(json_data=None, status_code=200):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.content = True
    return resp


@pytest.fixture
def workspaced_client(mock_requests):
    """Client with `workspace` set — every collection call should hit
    the nested URL family."""
    return AetherfyVectorsClient(
        api_key=API_KEY,
        endpoint=ENDPOINT,
        workspace=WS,
        timeout=10.0,
    )


def _last_request_url(mock_requests):
    """The URL passed to the most recent requests.request() call. The
    session wrapper in conftest's mock_requests routes session.request
    through mock.request, so the call args we want live on
    mock_requests.request.call_args."""
    args, kwargs = mock_requests.request.call_args
    # session.request("METHOD", url, ...) — url is the second positional
    # arg. Be tolerant of kwargs-style passing.
    if len(args) >= 2:
        return args[1]
    return kwargs.get("url")


def _last_request_method(mock_requests):
    args, kwargs = mock_requests.request.call_args
    if len(args) >= 1:
        return args[0]
    return kwargs.get("method")


def _last_request_json(mock_requests):
    _, kwargs = mock_requests.request.call_args
    return kwargs.get("json")


class TestWorkspacedCreateCollection:
    def test_posts_to_nested_url_with_bare_name(self, workspaced_client, mock_requests):
        mock_requests.request.return_value = _ok_response({"success": True}, 201)
        workspaced_client.create_collection(
            COLL, {"size": 128, "distance": "Cosine"}
        )
        url = _last_request_url(mock_requests)
        body = _last_request_json(mock_requests)
        assert url.endswith(f"/workspaces/{WS}/collections")
        # Bare name in body — vectordb rejects "ws/coll" with 400.
        assert body["name"] == COLL
        assert "/" not in body["name"]


class TestWorkspacedListCollections:
    def test_uses_nested_list_url(self, workspaced_client, mock_requests):
        mock_requests.request.return_value = _ok_response(
            {"collections": [{"name": COLL, "config": {"size": 128, "distance": "Cosine"}}]}
        )
        workspaced_client.get_collections()
        url = _last_request_url(mock_requests)
        assert _last_request_method(mock_requests) == "GET"
        assert url.endswith(f"/workspaces/{WS}/collections")


class TestWorkspacedCollectionOps:
    def test_get_collection_uses_nested_url(self, workspaced_client, mock_requests):
        mock_requests.request.return_value = _ok_response(
            {"result": {"name": COLL, "config": {"params": {"vectors": {"size": 128, "distance": "Cosine"}}}}}
        )
        workspaced_client.get_collection(COLL)
        url = _last_request_url(mock_requests)
        assert _last_request_method(mock_requests) == "GET"
        assert url.endswith(f"/workspaces/{WS}/collections/{COLL}")

    def test_delete_collection_uses_nested_url(self, workspaced_client, mock_requests):
        mock_requests.request.return_value = _ok_response({"success": True})
        workspaced_client.delete_collection(COLL)
        url = _last_request_url(mock_requests)
        assert _last_request_method(mock_requests) == "DELETE"
        assert url.endswith(f"/workspaces/{WS}/collections/{COLL}")

    def test_collection_exists_uses_nested_url(self, workspaced_client, mock_requests):
        mock_requests.request.return_value = _ok_response(
            {"result": {"name": COLL, "config": {"params": {"vectors": {"size": 128, "distance": "Cosine"}}}}}
        )
        workspaced_client.collection_exists(COLL)
        url = _last_request_url(mock_requests)
        assert url.endswith(f"/workspaces/{WS}/collections/{COLL}")


class TestWorkspacedPointOps:
    @pytest.mark.parametrize(
        "op,suffix",
        [
            ("search", "/points/query"),
            ("scroll", "/points/scroll"),
            ("count", "/points/count"),
        ],
    )
    def test_point_op_uses_nested_url(self, workspaced_client, mock_requests, op, suffix):
        # count expects `{"result": {"count": N}}` shape; others get
        # `result: []` for an empty match.
        if op == "count":
            mock_requests.request.return_value = _ok_response(
                {"result": {"count": 0}, "status": "ok"}
            )
        else:
            mock_requests.request.return_value = _ok_response(
                {"result": [], "status": "ok", "schema_version": "abc12345", "points_count": 0}
            )
        # search needs a query vector; others take just collection.
        if op == "search":
            workspaced_client.search(COLL, [0.1] * 128, limit=1)
        elif op == "scroll":
            workspaced_client.scroll(COLL, limit=1)
        elif op == "count":
            workspaced_client.count(COLL)
        url = _last_request_url(mock_requests)
        assert url.endswith(f"/workspaces/{WS}/collections/{COLL}{suffix}"), (
            f"{op} expected suffix {suffix}, got url {url}"
        )


class TestWorkspaceIdForm:
    """workspace="auto" addresses the agent's workspace by id, on the wire."""

    WS_ID = "0f8e2b9a-7c41-4d2e-9b3a-5e6f7a8b9c0d"

    @pytest.fixture
    def id_client(self, mock_requests, monkeypatch):
        monkeypatch.setenv("AETHERFY_WORKSPACE_ID", self.WS_ID)
        return AetherfyVectorsClient(api_key=API_KEY, endpoint=ENDPOINT, timeout=10.0)

    def test_create_collection_uses_the_id_form(self, id_client, mock_requests):
        mock_requests.request.return_value = _ok_response({"success": True}, 201)
        id_client.create_collection(COLL, {"size": 128, "distance": "Cosine"})
        url = _last_request_url(mock_requests)
        body = _last_request_json(mock_requests)
        assert url == f"{ENDPOINT}/api/v1/workspace-ids/{self.WS_ID}/collections"
        assert body["name"] == COLL

    def test_point_op_uses_the_id_form(self, id_client, mock_requests):
        mock_requests.request.return_value = _ok_response(
            {"result": {"count": 0}, "status": "ok"}
        )
        id_client.count(COLL)
        url = _last_request_url(mock_requests)
        assert url == (
            f"{ENDPOINT}/api/v1/workspace-ids/{self.WS_ID}"
            f"/collections/{COLL}/points/count"
        )
