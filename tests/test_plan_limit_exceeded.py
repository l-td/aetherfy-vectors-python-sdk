"""The plan limit's answer (403 PLAN_LIMIT_EXCEEDED, the control plane's shape)
reaches the typed QuotaExceededError through the real client request path,
including the memory SDK's create_namespace.

The body is the one vectordb sends (routes/proxy.js, the collection limit):
`limit` names the axis, `current` is the count in use and the plan's number
sits under the axis' own name.
"""

from unittest.mock import Mock

import pytest

from aetherfy_memory import MemoryClient
from aetherfy_vectors import AetherfyVectorsClient, QuotaExceededError

PLAN_LIMIT = {
    "error": {
        "code": "PLAN_LIMIT_EXCEEDED",
        "message": "Collection limit reached. Your free plan allows 3 collections.",
        "limit": "max_collections",
        "current": 3,
        "max_collections": 3,
        "upgrade_url": "/billing/upgrade",
    }
}
COLLECTION_NOT_FOUND = {
    "error": {"code": "COLLECTION_NOT_FOUND", "message": "not found"}
}


def _response(status, payload):
    r = Mock()
    r.status_code = status
    r.json.return_value = payload
    r.content = True
    return r


@pytest.fixture
def wired(mock_requests):
    client = AetherfyVectorsClient(
        api_key="afy_test_1234567890abcdef1234",
        endpoint="https://test-api.aetherfy.com",
        workspace="home",
    )

    def queue(*responses):
        mock_requests.request.side_effect = list(responses)
        return mock_requests.request

    return client, queue


def test_create_collection_raises_the_quota_error(wired):
    client, queue = wired
    calls = queue(_response(403, PLAN_LIMIT))

    with pytest.raises(QuotaExceededError) as exc:
        client.create_collection("docs", {"size": 3, "distance": "Cosine"})

    assert calls.call_count == 1
    assert exc.value.error_code == "PLAN_LIMIT_EXCEEDED"
    assert exc.value.quota_type == "collections"
    assert exc.value.current == 3
    assert exc.value.limit == 3
    assert str(PLAN_LIMIT["error"]["message"]) in str(exc.value)


def test_create_namespace_raises_the_same_quota_error(wired):
    client, queue = wired
    queue(_response(404, COLLECTION_NOT_FOUND), _response(403, PLAN_LIMIT))

    with pytest.raises(QuotaExceededError) as exc:
        MemoryClient(client=client).create_namespace("kb")

    assert exc.value.quota_type == "collections"
    assert exc.value.current == 3
    assert exc.value.limit == 3


def test_another_axis_maps_the_same_way(wired):
    client, queue = wired
    queue(
        _response(
            403,
            {
                "error": {
                    "code": "PLAN_LIMIT_EXCEEDED",
                    "message": "limit",
                    "limit": "max_workspaces",
                    "current": 1,
                    "max_workspaces": 1,
                }
            },
        )
    )

    with pytest.raises(QuotaExceededError) as exc:
        client.create_collection("docs", {"size": 3, "distance": "Cosine"})
    assert exc.value.quota_type == "workspaces"
    assert exc.value.limit == 1


def test_a_body_without_the_numbers_still_maps(wired):
    client, queue = wired
    queue(
        _response(403, {"error": {"code": "PLAN_LIMIT_EXCEEDED", "message": "limit"}})
    )

    with pytest.raises(QuotaExceededError) as exc:
        client.create_collection("docs", {"size": 3, "distance": "Cosine"})
    assert exc.value.current is None
    assert exc.value.limit is None
