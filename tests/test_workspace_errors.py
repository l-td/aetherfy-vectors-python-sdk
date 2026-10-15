"""The workspace-required error codes, mapped on `code`, never on the message.

Bodies are the contract sheet's (vectordb envelope `{"error": {...}}`), sent
through the one status/code mapping every error goes through
(utils.parse_error_response) and, where a client method decides on a 404,
through that method: an unknown workspace must never read as "not there".
"""

from unittest.mock import Mock, patch

import pytest

from aetherfy_vectors.retry_policy import REPEATABLE, should_retry_class

from aetherfy_memory import MemoryClient, NamespaceNotFoundError
from aetherfy_vectors import AetherfyVectorsClient, WorkspaceNotFoundError
from aetherfy_vectors.exceptions import (
    AetherfyVectorsException,
    AuthenticationError,
    CollectionNotFoundError,
    ConflictError,
    PointNotFoundError,
    SchemaNotFoundError,
    ServiceUnavailableError,
    ValidationError,
)
from aetherfy_vectors.utils import parse_error_response

WS_ID = "0f8e2b9a-7c41-4d2e-9b3a-5e6f7a8b9c0d"


def body(code, **fields):
    return {"error": {"code": code, "message": f"{code} (prose)", **fields}}


WORKSPACE_NOT_FOUND_BY_NAME = body("WORKSPACE_NOT_FOUND", workspace="typo")
WORKSPACE_NOT_FOUND_BY_ID = body("WORKSPACE_NOT_FOUND", workspace_id=WS_ID)
# Qdrant's own 404 (a missing point), as vectordb re-wraps it.
POINT_NOT_THERE = body("NOT_FOUND", message="No point with id 1 found")
COLLECTION_NOT_FOUND = body(
    "COLLECTION_NOT_FOUND",
    message="Collection 'docs' not found",
    collection_name="docs",
)


# ===========================================================================
# The mapping table
# ===========================================================================


@pytest.mark.parametrize(
    "status,payload,cls",
    [
        (400, body("WORKSPACE_REQUIRED"), ValidationError),
        (404, WORKSPACE_NOT_FOUND_BY_NAME, WorkspaceNotFoundError),
        (404, WORKSPACE_NOT_FOUND_BY_ID, WorkspaceNotFoundError),
        (404, COLLECTION_NOT_FOUND, CollectionNotFoundError),
        (404, body("SCHEMA_NOT_DEFINED"), SchemaNotFoundError),
        (409, body("ADDRESSED_AN_ALIAS", alias="live", workspace="w"), ConflictError),
        (
            409,
            body(
                "COLLECTION_NAME_TAKEN",
                collection_name="docs",
                workspace="w",
                taken_by="alias",
            ),
            ConflictError,
        ),
        (
            409,
            body("COLLECTION_NAME_IS_AN_ALIAS", collection_name="docs", workspace="w"),
            ConflictError,
        ),
        (422, body("COLLECTION_SDK_MANAGED", collection_name="__threads__"), ValidationError),
        (
            403,
            body(
                "AUTH_AGENT_KEY_OUT_OF_SCOPE",
                agent_id="a1",
                required_scope=None,
                documentation_url="https://docs.aetherfy.com/platform/api-keys",
            ),
            AuthenticationError,
        ),
        (503, body("WORKSPACE_LOOKUP_UNAVAILABLE"), ServiceUnavailableError),
        (
            503,
            body("COLLECTION_LOOKUP_UNAVAILABLE", collection_name="docs"),
            ServiceUnavailableError,
        ),
        (503, body("PLACEMENT_UNAVAILABLE"), ServiceUnavailableError),
    ],
)
def test_each_code_maps_to_a_typed_error_carrying_it(status, payload, cls):
    err = parse_error_response(payload, status)
    assert type(err) is cls
    assert err.error_code == payload["error"]["code"]
    assert err.status_code == status


@pytest.mark.parametrize(
    "code",
    ["WORKSPACE_LOOKUP_UNAVAILABLE", "COLLECTION_LOOKUP_UNAVAILABLE", "PLACEMENT_UNAVAILABLE"],
)
def test_the_lookup_outages_are_retryable(code):
    assert should_retry_class(REPEATABLE, parse_error_response(body(code), 503))


@pytest.mark.parametrize(
    "status,code",
    [(404, "WORKSPACE_NOT_FOUND"), (400, "WORKSPACE_REQUIRED"), (409, "ADDRESSED_AN_ALIAS")],
)
def test_control_the_refusals_are_not_retryable(status, code):
    assert not should_retry_class(REPEATABLE, parse_error_response(body(code), status))


def test_the_workspace_not_found_names_what_was_sent():
    by_name = parse_error_response(WORKSPACE_NOT_FOUND_BY_NAME, 404)
    assert (by_name.workspace, by_name.workspace_id) == ("typo", None)

    by_id = parse_error_response(WORKSPACE_NOT_FOUND_BY_ID, 404)
    assert (by_id.workspace, by_id.workspace_id) == (None, WS_ID)


def test_control_the_code_not_the_message_decides():
    """A message that says "workspace not found" under another code is not
    a WorkspaceNotFoundError, and the code under any message is."""
    lying = {"error": {"code": "COLLECTION_NOT_FOUND", "message": "Workspace not found"}}
    assert type(parse_error_response(lying, 404)) is CollectionNotFoundError

    terse = {"error": {"code": "WORKSPACE_NOT_FOUND", "message": "x"}}
    assert type(parse_error_response(terse, 404)) is WorkspaceNotFoundError


# ===========================================================================
# Through the client: the retry loop, and the methods that decide on a 404
# ===========================================================================


def _response(status, payload):
    r = Mock()
    r.status_code = status
    r.json.return_value = payload
    r.content = True
    return r


@pytest.fixture
def wired(mock_requests):
    """A client whose every request returns the next queued response."""
    client = AetherfyVectorsClient(
        api_key="afy_test_1234567890abcdef1234",
        endpoint="https://test-api.aetherfy.com",
        workspace="typo",
    )

    def queue(*responses):
        mock_requests.request.side_effect = list(responses)
        return mock_requests.request

    return client, queue


@pytest.mark.parametrize(
    "code",
    ["WORKSPACE_LOOKUP_UNAVAILABLE", "COLLECTION_LOOKUP_UNAVAILABLE", "PLACEMENT_UNAVAILABLE"],
)
@patch("time.sleep")
def test_a_lookup_outage_is_retried_then_succeeds(_sleep, wired, code):
    client, queue = wired
    calls = queue(
        _response(503, body(code)),
        _response(200, {"result": {"count": 7}, "status": "ok"}),
    )

    assert client.count("docs") == 7
    assert calls.call_count == 2


@patch("time.sleep")
def test_a_lasting_outage_raises_the_typed_error_with_its_code(_sleep, wired):
    client, queue = wired
    calls = queue(*[_response(503, body("WORKSPACE_LOOKUP_UNAVAILABLE"))] * 10)

    with pytest.raises(ServiceUnavailableError) as exc:
        client.count("docs")
    assert exc.value.error_code == "WORKSPACE_LOOKUP_UNAVAILABLE"
    assert calls.call_count > 1


@patch("time.sleep")
def test_control_an_unknown_workspace_is_tried_once(_sleep, wired):
    client, queue = wired
    calls = queue(*[_response(404, WORKSPACE_NOT_FOUND_BY_NAME)] * 10)

    with pytest.raises(WorkspaceNotFoundError):
        client.count("docs")
    assert calls.call_count == 1


def test_collection_exists_raises_on_an_unknown_workspace(wired):
    client, queue = wired
    queue(_response(404, WORKSPACE_NOT_FOUND_BY_NAME))

    with pytest.raises(WorkspaceNotFoundError):
        client.collection_exists("docs")


def test_control_collection_exists_is_false_on_collection_not_found(wired):
    client, queue = wired
    queue(_response(404, COLLECTION_NOT_FOUND))

    assert client.collection_exists("docs") is False


def test_delete_field_index_raises_on_an_unknown_workspace(wired):
    client, queue = wired
    queue(_response(404, WORKSPACE_NOT_FOUND_BY_NAME))

    with pytest.raises(WorkspaceNotFoundError):
        client.delete_field_index("docs", "tag")


def test_control_delete_field_index_is_false_on_collection_not_found(wired):
    client, queue = wired
    queue(_response(404, COLLECTION_NOT_FOUND))

    assert client.delete_field_index("docs", "tag") is False


def test_get_schema_raises_on_an_unknown_workspace(wired):
    client, queue = wired
    queue(_response(404, WORKSPACE_NOT_FOUND_BY_NAME))

    with pytest.raises(WorkspaceNotFoundError):
        client.get_schema("docs")


@pytest.mark.parametrize("payload", [COLLECTION_NOT_FOUND, body("SCHEMA_NOT_DEFINED")])
def test_control_get_schema_is_none_on_no_schema(wired, payload):
    client, queue = wired
    queue(_response(404, payload))

    assert client.get_schema("docs") is None


def test_delete_schema_raises_on_an_unknown_workspace(wired):
    client, queue = wired
    queue(_response(404, WORKSPACE_NOT_FOUND_BY_NAME))

    with pytest.raises(WorkspaceNotFoundError):
        client.delete_schema("docs")


def test_control_delete_schema_is_schema_not_found_on_no_schema(wired):
    client, queue = wired
    queue(_response(404, body("SCHEMA_NOT_DEFINED")))

    with pytest.raises(SchemaNotFoundError):
        client.delete_schema("docs")


@pytest.mark.parametrize("op", ["merge_metadata", "delete_metadata_keys"])
def test_metadata_ops_do_not_call_an_unknown_workspace_a_missing_point(wired, op):
    client, queue = wired
    queue(_response(404, WORKSPACE_NOT_FOUND_BY_NAME))
    arg = {"a": 1} if op == "merge_metadata" else ["a"]

    with pytest.raises(WorkspaceNotFoundError):
        getattr(client, op)("docs", 1, arg)


@pytest.mark.parametrize("op", ["merge_metadata", "delete_metadata_keys"])
def test_control_metadata_ops_still_name_a_missing_point(wired, op):
    client, queue = wired
    queue(_response(404, POINT_NOT_THERE))
    arg = {"a": 1} if op == "merge_metadata" else ["a"]

    with pytest.raises(PointNotFoundError):
        getattr(client, op)("docs", 1, arg)


def test_an_unknown_workspace_does_not_evict_the_schema_cache(wired):
    client, queue = wired
    key = client._scope_collection("docs")
    client._schema_cache[key] = {"size": 3, "distance": "Cosine"}
    queue(_response(404, WORKSPACE_NOT_FOUND_BY_NAME))

    with pytest.raises(WorkspaceNotFoundError):
        client.get_collection("docs")
    assert key in client._schema_cache


def test_control_collection_not_found_evicts_the_schema_cache(wired):
    client, queue = wired
    key = client._scope_collection("docs")
    client._schema_cache[key] = {"size": 3, "distance": "Cosine"}
    queue(_response(404, COLLECTION_NOT_FOUND))

    with pytest.raises(CollectionNotFoundError):
        client.get_collection("docs")
    assert key not in client._schema_cache


def test_usage_stats_with_an_agent_key_is_the_typed_scope_refusal(wired):
    """GET /analytics/usage is account-only: an agent key gets 403
    AUTH_AGENT_KEY_OUT_OF_SCOPE (vectordb utils/agentRoutePolicy.js)."""
    client, _ = wired
    client.session.request = Mock(
        return_value=_response(
            403, body("AUTH_AGENT_KEY_OUT_OF_SCOPE", agent_id="a1", required_scope=None)
        )
    )

    with pytest.raises(AuthenticationError) as exc:
        client.get_usage_stats()
    assert exc.value.error_code == "AUTH_AGENT_KEY_OUT_OF_SCOPE"


# ===========================================================================
# The memory SDK: NamespaceNotFoundError only for a missing collection
# ===========================================================================


def test_namespace_raises_the_unknown_workspace_not_namespace_not_found(wired):
    client, queue = wired
    queue(_response(404, WORKSPACE_NOT_FOUND_BY_NAME))

    with pytest.raises(WorkspaceNotFoundError):
        MemoryClient(client=client).namespace("kb")


def test_control_namespace_not_found_on_collection_not_found(wired):
    client, queue = wired
    queue(_response(404, COLLECTION_NOT_FOUND))

    with pytest.raises(NamespaceNotFoundError):
        MemoryClient(client=client).namespace("kb")


def test_namespace_exists_raises_on_an_unknown_workspace(wired):
    client, queue = wired
    queue(_response(404, WORKSPACE_NOT_FOUND_BY_NAME))

    with pytest.raises(AetherfyVectorsException):
        MemoryClient(client=client).namespace_exists("kb")


@pytest.mark.parametrize("op", ["merge_metadata", "delete_metadata_keys"])
def test_a_namespace_does_not_call_an_unknown_workspace_a_missing_point(wired, op):
    from aetherfy_memory.namespace import Namespace

    client, queue = wired
    queue(_response(404, WORKSPACE_NOT_FOUND_BY_NAME))
    arg = {"a": 1} if op == "merge_metadata" else ["a"]

    with pytest.raises(WorkspaceNotFoundError):
        getattr(Namespace("kb", "kb", client), op)(1, arg)


@pytest.mark.parametrize("op", ["merge_metadata", "delete_metadata_keys"])
def test_control_a_namespace_still_names_a_missing_point(wired, op):
    from aetherfy_memory.namespace import Namespace

    client, queue = wired
    queue(_response(404, POINT_NOT_THERE))
    arg = {"a": 1} if op == "merge_metadata" else ["a"]

    with pytest.raises(PointNotFoundError):
        getattr(Namespace("kb", "kb", client), op)(1, arg)


# ===========================================================================
# NotFoundError: the parent of every not-found error, and the 404 fallback
# ===========================================================================


@pytest.mark.parametrize(
    "payload",
    [
        COLLECTION_NOT_FOUND,
        WORKSPACE_NOT_FOUND_BY_NAME,
        body("SCHEMA_NOT_DEFINED"),
        body("POINT_NOT_FOUND", point_id="1", collection_name="docs"),
    ],
)
def test_every_not_found_error_is_a_not_found_error(payload):
    from aetherfy_vectors import NotFoundError

    assert isinstance(parse_error_response(payload, 404), NotFoundError)


def test_an_unknown_404_code_is_a_typed_not_found_carrying_it():
    from aetherfy_vectors import NotFoundError

    err = parse_error_response(body("ALIAS_NOT_FOUND", alias_name="live"), 404)
    assert type(err) is NotFoundError
    assert err.error_code == "ALIAS_NOT_FOUND"
    assert err.status_code == 404


# ===========================================================================
# A missing point on the payload routes: NOT_FOUND, by code
# ===========================================================================


@pytest.mark.parametrize(
    "op,args",
    [
        ("set_payload", lambda: ("docs", {"a": 1}, [1])),
        ("overwrite_payload", lambda: ("docs", {"a": 1}, [1])),
        ("delete_payload", lambda: ("docs", ["a"], [1])),
    ],
)
def test_a_payload_route_s_not_found_is_a_missing_point(wired, op, args):
    client, queue = wired
    queue(_response(404, POINT_NOT_THERE))

    with pytest.raises(PointNotFoundError) as exc:
        getattr(client, op)(*args())
    assert exc.value.error_code == "NOT_FOUND"
    assert exc.value.point_id == "1"


def test_control_a_code_less_404_on_a_payload_route_is_not_a_missing_point(wired):
    """vectordb never answers a 404 without a code: one is not read as a
    missing point, it is the NotFoundError fallback."""
    from aetherfy_vectors import NotFoundError

    client, queue = wired
    queue(_response(404, {"status": {"error": "No point with id 1 found"}}))

    with pytest.raises(NotFoundError) as exc:
        client.set_payload("docs", {"a": 1}, [1])
    assert not isinstance(exc.value, PointNotFoundError)
