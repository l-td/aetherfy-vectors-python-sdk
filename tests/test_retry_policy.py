"""The retry rule (aetherfy_vectors/retry_policy.py), driven through the client.

A refusal before anything ran is retried for every operation; an ambiguous
failure only for an operation whose repeat gives the same result; a 4xx never.
The session's `request` is the seam: each test queues what the wire returns
and counts the attempts.
"""

import re
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from urllib3.exceptions import NewConnectionError

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.exceptions import (
    NetworkError,
    RequestTimeoutError,
    ServiceUnavailableError,
    ValidationError,
)
from aetherfy_vectors.models import DistanceMetric, VectorConfig
from aetherfy_vectors.retry_policy import OPERATIONS, ONCE, REPEATABLE

LOOKUP = {"error": {"code": "WORKSPACE_LOOKUP_UNAVAILABLE", "message": "x"}}
OTHER_503 = {"error": {"code": "SOMETHING_ELSE", "message": "x"}}


def _response(status, payload):
    r = Mock()
    r.status_code = status
    r.json.return_value = payload
    r.content = True
    return r


def _ok(payload=None):
    return _response(200, payload if payload is not None else {"result": True})


def _never_connected():
    """What requests raises for DNS failure or a refused connection."""
    return requests.ConnectionError(
        NewConnectionError(None, "Failed to establish a new connection")
    )


@pytest.fixture(autouse=True)
def _no_backoff():
    with patch("time.sleep"):
        yield


@pytest.fixture
def client():
    c = AetherfyVectorsClient(
        api_key="afy_test_1234567890abcdef1234",
        endpoint="https://test-api.aetherfy.com",
        workspace="w",
    )
    c.session.request = Mock()
    return c


def queue(client, *outcomes):
    client.session.request.reset_mock()
    client.session.request.side_effect = list(outcomes)
    return client.session.request


COLLECTION = {
    "result": {"name": "docs", "config": {"params": {"vectors": {"size": 3, "distance": "Cosine"}}}}
}

# One call per operation class the ruling names, each with a success body.
CALLS = {
    "get_collection": (lambda c: c.get_collection("docs"), COLLECTION),
    "delete_collection": (lambda c: c.delete_collection("docs"), {"result": True}),
    "create_collection": (
        lambda c: c.create_collection(
            "docs", VectorConfig(size=3, distance=DistanceMetric.COSINE)
        ),
        {"result": True},
    ),
    "count": (lambda c: c.count("docs"), {"result": {"count": 1}}),
    "scroll": (lambda c: c.scroll("docs", limit=1), {"result": {"points": []}}),
    "retrieve": (lambda c: c.retrieve("docs", [1]), {"result": []}),
}


# ===========================================================================
# 1. Refused before anything happened: every operation
# ===========================================================================


def test_a_get_retries_a_lookup_unavailable_503(client):
    call, body = CALLS["get_collection"]
    sent = queue(client, _response(503, LOOKUP), _ok(body))

    call(client)
    assert sent.call_count == 2
    assert sent.call_args.kwargs["method"] == "GET"


def test_a_delete_retries_a_lookup_unavailable_503(client):
    call, body = CALLS["delete_collection"]
    sent = queue(client, _response(503, LOOKUP), _ok(body))

    call(client)
    assert sent.call_count == 2
    assert sent.call_args.kwargs["method"] == "DELETE"


def test_a_delete_retries_a_connection_that_was_never_made(client):
    call, body = CALLS["delete_collection"]
    sent = queue(client, _never_connected(), _ok(body))

    call(client)
    assert sent.call_count == 2


def test_a_delete_retries_a_connect_timeout(client):
    call, body = CALLS["delete_collection"]
    sent = queue(client, requests.ConnectTimeout("connect"), _ok(body))

    call(client)
    assert sent.call_count == 2


# ===========================================================================
# 2. Ambiguous: only operations whose repeat gives the same result
# ===========================================================================


@pytest.mark.parametrize("op", ["delete_collection", "create_collection"])
def test_a_once_operation_does_not_retry_a_read_timeout(client, op):
    call, _ = CALLS[op]
    sent = queue(client, requests.ReadTimeout("read"), _ok())

    with pytest.raises(RequestTimeoutError):
        call(client)
    assert sent.call_count == 1


@pytest.mark.parametrize("op", ["delete_collection", "create_collection"])
def test_a_once_operation_does_not_retry_a_reset_or_another_503(client, op):
    call, _ = CALLS[op]
    sent = queue(client, requests.ConnectionError("reset by peer"), _ok())
    with pytest.raises(NetworkError):
        call(client)
    assert sent.call_count == 1

    sent = queue(client, _response(503, OTHER_503), _ok())
    with pytest.raises(ServiceUnavailableError):
        call(client)
    assert sent.call_count == 1


@pytest.mark.parametrize("op", ["get_collection", "count", "scroll", "retrieve"])
def test_a_read_retries_a_read_timeout_and_another_503(client, op):
    call, body = CALLS[op]
    sent = queue(client, requests.ReadTimeout("read"), _ok(body))
    call(client)
    assert sent.call_count == 2

    sent = queue(client, _response(503, OTHER_503), _ok(body))
    call(client)
    assert sent.call_count == 2


def test_an_ambiguous_failure_is_retried_three_times_then_raised(client):
    call, _ = CALLS["count"]
    sent = queue(client, *[requests.ReadTimeout("read")] * 10)

    with pytest.raises(RequestTimeoutError):
        call(client)
    assert sent.call_count == 4  # the first attempt and 3 retries


# ===========================================================================
# 3. A 4xx is never retried
# ===========================================================================


@pytest.mark.parametrize("op", ["get_collection", "count", "delete_collection"])
def test_a_4xx_is_never_retried(client, op):
    call, _ = CALLS[op]
    sent = queue(
        client, _response(400, {"error": {"code": "VALIDATION_ERROR", "message": "x"}})
    )

    with pytest.raises(ValidationError):
        call(client)
    assert sent.call_count == 1


# ===========================================================================
# The table is the call sites
# ===========================================================================


def test_every_request_declares_an_operation_of_the_table_and_vice_versa():
    source = (Path(__file__).parent.parent / "aetherfy_vectors" / "client.py").read_text(
        encoding="utf-8"
    )
    declared = set(re.findall(r'operation="([a-z_]+)"', source))
    assert declared == set(OPERATIONS)
    calls = re.findall(
        r"self\._(?:make|payload)_request\((.*?)operation=([^,)\s]+)",
        source,
        re.S,
    )
    # Every call names its operation as a literal, except the one
    # pass-through inside _payload_request.
    passed_through = [op for _, op in calls if not op.startswith('"')]
    assert passed_through == ["operation"]
    assert len(calls) - 1 == source.count('operation="')


def test_the_ruling_s_once_operations_are_once():
    for op in [
        "create_collection",
        "delete_collection",
        "delete",
        "delete_schema",
        "delete_field_index",
    ]:
        assert OPERATIONS[op] == ONCE, op
    for op in ["search", "count", "scroll", "retrieve", "upsert", "set_payload",
               "overwrite_payload", "create_field_index", "get_collection",
               "update_collection"]:
        assert OPERATIONS[op] == REPEATABLE, op


def test_set_schema_is_repeatable_only_while_it_sends_no_if_match(client):
    """A repeat of an If-Match write after a hidden success gets a false 412:
    if set_schema ever sends If-Match, it must be classed ONCE."""
    from aetherfy_vectors.schema import Schema

    sent = queue(client, _ok({"etag": "v1"}))
    client.set_schema("docs", Schema(fields={}))

    headers = sent.call_args.kwargs.get("headers") or {}
    if any(h.lower() == "if-match" for h in headers):
        assert OPERATIONS["set_schema"] == ONCE
    else:
        assert OPERATIONS["set_schema"] == REPEATABLE


def _http_method_of_each_operation():
    """{operation: HTTP method}, read from client.py's request call sites."""
    source = (Path(__file__).parent.parent / "aetherfy_vectors" / "client.py").read_text(
        encoding="utf-8"
    )
    methods = {}
    for call in re.finditer(
        r"self\._(?:make|payload)_request\((.*?)operation=\"([a-z_]+)\"", source, re.S
    ):
        verb = re.search(r'"(GET|POST|PUT|PATCH|DELETE)"', call.group(1))
        methods.setdefault(call.group(2), set()).add(verb.group(1) if verb else "?")
    return methods


def test_a_delete_or_a_create_is_once_unless_listed_with_a_reason():
    """The table guard catches a missing entry; this catches a wrong one."""
    from aetherfy_vectors.retry_policy import REPEATABLE_DESPITE_ITS_NAME

    methods = _http_method_of_each_operation()
    assert set(methods) == set(OPERATIONS)
    for op, verbs in methods.items():
        risky = "DELETE" in verbs or op.startswith(("delete", "create"))
        if risky and op not in REPEATABLE_DESPITE_ITS_NAME:
            assert OPERATIONS[op] == ONCE, f"{op} sends {verbs}: class it ONCE"
    for op, reason in REPEATABLE_DESPITE_ITS_NAME.items():
        assert OPERATIONS[op] == REPEATABLE and reason.strip(), op


# ===========================================================================
# A connection never made, on a real network (no server, no mock transport)
# ===========================================================================


def _real_client(endpoint, timeout):
    c = AetherfyVectorsClient(
        api_key="afy_test_1234567890abcdef1234", endpoint=endpoint, workspace="w",
        timeout=timeout,
    )
    real = c.session.request
    c.session.request = Mock(side_effect=real)  # counts, sends for real
    return c


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        # TEST-NET-1 (RFC 5737): never routed, so the connect times out.
        ("http://192.0.2.1", RequestTimeoutError),
        # .invalid (RFC 2606) never resolves: a DNS failure.
        ("http://no-such-host.invalid", NetworkError),
    ],
)
def test_a_real_connection_never_made_is_retried_even_for_a_once_operation(
    endpoint, expected
):
    client = _real_client(endpoint, timeout=0.3)

    with pytest.raises(expected) as exc:
        client.delete_collection("docs")  # ONCE: only a never-sent retries
    assert exc.value.request_sent is False
    assert client.session.request.call_count == 4


def test_a_real_refused_connection_is_never_connected():
    import socket

    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    try:
        requests.get(f"http://127.0.0.1:{port}/", timeout=5)
    except requests.ConnectTimeout:
        pytest.skip("this host reports a refused port as a connect timeout")
    except requests.ConnectionError as e:
        from aetherfy_vectors.client import _never_connected

        assert _never_connected(e)
    else:
        pytest.fail("a closed port answered")


def test_retry_with_backoff_around_a_client_call_retries_a_never_made_connection():
    """The typed error a client call raises carries request_sent=False, so
    the public helper retries a ONCE operation on it, as the client does."""
    from aetherfy_vectors.utils import retry_with_backoff

    client = _real_client("http://192.0.2.1", timeout=0.3)

    with pytest.raises(RequestTimeoutError) as exc:
        retry_with_backoff(
            lambda: client.delete_collection("docs"), ONCE, max_retries=1, base_delay=0.01
        )
    assert exc.value.request_sent is False
    assert client.session.request.call_count == 8  # the client's 4, twice
