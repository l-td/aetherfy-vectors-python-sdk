"""A plan limit (403 PLAN_LIMIT_EXCEEDED) reaches the typed QuotaExceededError
through the real client request path, including the memory SDK's
create_namespace.

The body is ``{code, message, upgrade_url, limit, allowed, current}`` inside
``error``: ``limit`` names the limit, ``allowed`` is the plan's number (or
``false`` for a feature the plan lacks, or ``null`` for no cap) and
``current`` is the usage now (absent for a yes/no feature).
"""

from unittest.mock import Mock

import pytest

from aetherfy_memory import MemoryClient
from aetherfy_vectors import AetherfyVectorsClient, QuotaExceededError
from aetherfy_vectors.exceptions import PermissionDeniedError
from aetherfy_vectors.utils import parse_error_response

UPGRADE_URL = "https://app.aetherfy.com/dashboard/settings/billing"


def plan_limit(limit="max_collections", **fields):
    error = {
        "code": "PLAN_LIMIT_EXCEEDED",
        "message": "Collection limit reached. Your free plan allows 3 collections.",
        "upgrade_url": UPGRADE_URL,
        "limit": limit,
    }
    error.update(fields)
    return {"error": error}


PLAN_LIMIT = plan_limit(allowed=3, current=3)
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

    assert calls.call_count == 1  # a refusal of the account is not retried
    err = exc.value
    assert err.error_code == "PLAN_LIMIT_EXCEEDED"
    assert err.status_code == 403
    assert err.quota_type == "collections"
    assert err.limit == "max_collections"
    assert err.allowed == 3
    assert err.current == 3
    assert err.upgrade_url == UPGRADE_URL
    assert err.details["limit"] == "max_collections"
    assert PLAN_LIMIT["error"]["message"] in str(err)


def test_create_namespace_raises_the_same_quota_error(wired):
    client, queue = wired
    queue(_response(404, COLLECTION_NOT_FOUND), _response(403, PLAN_LIMIT))

    with pytest.raises(QuotaExceededError) as exc:
        MemoryClient(client=client).create_namespace("kb")

    assert exc.value.quota_type == "collections"
    assert exc.value.limit == "max_collections"
    assert exc.value.allowed == 3
    assert exc.value.current == 3


@pytest.mark.parametrize(
    "allowed",
    [3, 0, False, None],
    ids=["a number", "zero", "a feature the plan lacks", "no cap"],
)
def test_every_kind_of_allowed_is_carried_as_sent(allowed):
    err = parse_error_response(plan_limit(allowed=allowed, current=2), 403)
    assert isinstance(err, QuotaExceededError)
    assert err.allowed is allowed or err.allowed == allowed
    assert type(err.allowed) is type(allowed)


def test_a_yes_no_feature_has_no_current():
    body = plan_limit("webhooks", allowed=False)
    assert "current" not in body["error"]
    err = parse_error_response(body, 403)
    assert isinstance(err, QuotaExceededError)
    assert err.quota_type == "webhooks"
    assert err.limit == "webhooks"
    assert err.allowed is False
    assert err.current is None


def test_another_axis_maps_the_same_way():
    err = parse_error_response(plan_limit("max_workspaces", allowed=1, current=1), 403)
    assert isinstance(err, QuotaExceededError)
    assert err.quota_type == "workspaces"
    assert err.limit == "max_workspaces"
    assert err.allowed == 1


def test_a_body_without_the_fields_still_maps():
    err = parse_error_response(
        {"error": {"code": "PLAN_LIMIT_EXCEEDED", "message": "limit"}}, 403
    )
    assert isinstance(err, QuotaExceededError)
    assert err.limit is None
    assert err.allowed is None
    assert err.current is None
    assert err.upgrade_url is None


@pytest.mark.parametrize("bad", ["3", True, [3], {"n": 3}])
def test_an_allowed_of_another_type_is_not_carried(bad):
    err = parse_error_response(plan_limit(allowed=bad, current=bad), 403)
    assert isinstance(err, QuotaExceededError)
    assert err.allowed is None
    assert err.current is None


def test_a_plan_limit_is_not_a_permission_error():
    """The key may do this; the account is at its cap."""
    err = parse_error_response(PLAN_LIMIT, 403)
    assert not isinstance(err, PermissionDeniedError)
