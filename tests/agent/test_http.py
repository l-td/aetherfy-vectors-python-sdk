"""
The transport, exercised directly.

Every other test in this directory replaces `_http.request_json`, which proves
what the helper SENDS but nothing about what the transport DOES. These tests
cover the other half: that an HTTP status is returned rather than raised, that
a request that never left is retried exactly once, that one that may have been
delivered is retried only when the call is repeatable, and that a status is
never retried at all.

`urlopen` is the only thing stubbed. Nothing here opens a socket.
"""

import io
import json
import socket
import urllib.error

import pytest

from aetherfy_agent import _http
from aetherfy_agent.exceptions import AgentTransportError


class FakeResponse(io.BytesIO):
    def __init__(self, status, body):
        super().__init__(json.dumps(body).encode("utf-8") if body is not None else b"")
        self._status = status

    def getcode(self):
        return self._status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def call(monkeypatch, urlopen, repeatable=True):
    monkeypatch.setattr(_http.urllib.request, "urlopen", urlopen)
    return _http.request_json(
        "POST",
        "https://agents.aetherfy.com/api/v1/agents/a/spawn",
        api_key="afy_test_key",
        ua="aetherfy-agent-python/test",
        body={"child_agent_id": "worker", "payload": {}},
        repeatable=repeatable,
    )


#: A request that NEVER LEFT: the name did not resolve, the connection was refused.
NEVER_DELIVERED = [
    urllib.error.URLError(socket.gaierror(-2, "Name or service not known")),
    urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")),
    ConnectionRefusedError(111, "Connection refused"),
]
#: A request that may have been DELIVERED: a timeout or a reset after it was sent.
MAYBE_DELIVERED = [
    urllib.error.URLError(socket.timeout("timed out")),
    socket.timeout("timed out"),
    ConnectionResetError(104, "Connection reset by peer"),
    urllib.error.URLError("connection reset"),
]


def test_the_headers_and_body_go_out_as_json(monkeypatch):
    seen = {}

    def urlopen(request, timeout=None):
        seen["headers"] = dict(request.headers)
        seen["data"] = request.data
        seen["method"] = request.get_method()
        return FakeResponse(202, {"status": "queued"})

    status, body, _ = call(monkeypatch, urlopen)

    assert status == 202
    assert body == {"status": "queued"}
    assert seen["method"] == "POST"
    # urllib title-cases header names it stores.
    assert seen["headers"]["Authorization"] == "Bearer afy_test_key"
    assert seen["headers"]["User-agent"] == "aetherfy-agent-python/test"
    assert seen["headers"]["Content-type"] == "application/json"
    assert json.loads(seen["data"]) == {"child_agent_id": "worker", "payload": {}}


def test_an_http_error_status_is_returned_not_raised(monkeypatch):
    envelope = {"detail": {"code": "AGENT_WORKER_PAUSED", "message": "paused"}}

    def urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            409,
            "Conflict",
            {},
            io.BytesIO(json.dumps(envelope).encode("utf-8")),
        )

    status, body, _ = call(monkeypatch, urlopen)

    assert status == 409
    assert body == envelope


def test_an_error_status_is_not_retried(monkeypatch):
    attempts = []

    def urlopen(request, timeout=None):
        attempts.append(1)
        raise urllib.error.HTTPError(
            request.full_url, 409, "Conflict", {}, io.BytesIO(b"{}")
        )

    call(monkeypatch, urlopen)

    assert len(attempts) == 1


@pytest.mark.parametrize("failure", NEVER_DELIVERED, ids=repr)
@pytest.mark.parametrize("repeatable", [True, False])
def test_a_request_that_never_left_is_retried_once_whatever_the_call(monkeypatch, failure, repeatable):
    attempts = []

    def urlopen(request, timeout=None):
        attempts.append(1)
        raise failure

    with pytest.raises(AgentTransportError) as excinfo:
        call(monkeypatch, urlopen, repeatable=repeatable)

    assert len(attempts) == 2, "one retry, not zero and not a loop"
    assert "did not reach the Aetherfy control plane after 2 attempt(s)" in str(excinfo.value)


@pytest.mark.parametrize("failure", MAYBE_DELIVERED, ids=repr)
def test_a_request_that_may_have_been_delivered_is_retried_once_when_the_call_is_repeatable(monkeypatch, failure):
    attempts = []

    def urlopen(request, timeout=None):
        attempts.append(1)
        raise failure

    with pytest.raises(AgentTransportError):
        call(monkeypatch, urlopen, repeatable=True)

    assert len(attempts) == 2, "one retry, not zero and not a loop"


@pytest.mark.parametrize("failure", MAYBE_DELIVERED, ids=repr)
def test_a_request_that_may_have_been_delivered_is_never_repeated_when_the_call_is_not_repeatable(monkeypatch, failure):
    attempts = []

    def urlopen(request, timeout=None):
        attempts.append(1)
        raise failure

    with pytest.raises(AgentTransportError) as excinfo:
        call(monkeypatch, urlopen, repeatable=False)

    assert len(attempts) == 1, "a repeat could do the work a second time"
    assert "may have been applied" in str(excinfo.value)


def test_a_retried_request_can_succeed(monkeypatch):
    attempts = []

    def urlopen(request, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.URLError(socket.gaierror(-3, "Temporary failure in name resolution"))
        return FakeResponse(202, {"status": "queued"})

    status, body, _ = call(monkeypatch, urlopen, repeatable=False)

    assert len(attempts) == 2
    assert (status, body) == (202, {"status": "queued"})


def test_the_retry_sends_the_same_request_its_body_and_its_id_included(monkeypatch):
    # The idempotency of a spawn rests on this: the retry carries the SAME
    # request_id as the first attempt, so the control plane recognises it.
    sent = []

    def urlopen(request, timeout=None):
        sent.append(request.data)
        if len(sent) == 1:
            raise socket.timeout("timed out")
        return FakeResponse(202, {"status": "queued"})

    monkeypatch.setattr(_http.urllib.request, "urlopen", urlopen)
    _http.request_json(
        "POST",
        "https://agents.aetherfy.com/api/v1/agents/a/spawn",
        api_key="afy_test_key",
        ua="aetherfy-agent-python/test",
        body={"child_agent_id": "worker", "payload": {}, "request_id": "r-1"},
        repeatable=True,
    )
    assert len(sent) == 2 and sent[0] == sent[1]


def test_repeatable_has_no_default():
    # Every caller decides it in the open; a default would let a new call
    # inherit a retry it never thought about.
    import inspect

    assert inspect.signature(_http.request_json).parameters["repeatable"].default is inspect.Parameter.empty


def test_a_body_that_fails_mid_read_is_not_retried(monkeypatch):
    """The server ANSWERED. On a spawn that means the control plane already
    recorded the run, so a second attempt would create a second one — the body
    is lost, the status is not."""
    attempts = []

    class Truncated(io.BytesIO):
        def read(self, *args):
            raise OSError("connection reset while reading the body")

        def getcode(self):
            return 202

    def urlopen(request, timeout=None):
        attempts.append(1)
        return Truncated()

    status, body, _ = call(monkeypatch, urlopen)

    assert len(attempts) == 1, "a body failure must not re-send the request"
    assert status == 202
    assert body is None


def test_a_non_json_body_decodes_to_none(monkeypatch):
    def urlopen(request, timeout=None):
        response = io.BytesIO(b"<html>an edge error page</html>")
        response.getcode = lambda: 403
        response.__enter__ = lambda self=response: self
        response.__exit__ = lambda *exc: False
        return response

    status, body, _ = call(monkeypatch, urlopen)

    assert status == 403
    assert body is None


# -- Retry-After --------------------------------------------------------------


def _refusal(status, headers):
    def urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, status, "x", headers, io.BytesIO(b"{}")
        )

    return urlopen


@pytest.mark.parametrize("status", [429, 503])
def test_the_retry_after_header_is_returned_in_seconds(monkeypatch, status):
    _, _, retry_after = call(monkeypatch, _refusal(status, {"Retry-After": "7"}))
    assert retry_after == 7.0


@pytest.mark.parametrize("value", ["", "soon", "-3", "nan"])
def test_an_unreadable_retry_after_is_none(monkeypatch, value):
    _, _, retry_after = call(monkeypatch, _refusal(429, {"Retry-After": value}))
    assert retry_after is None


def test_a_refusal_without_the_header_has_none(monkeypatch):
    assert call(monkeypatch, _refusal(503, {}))[2] is None


def test_a_status_with_retry_after_is_still_never_retried(monkeypatch):
    """The helper reports the time; waiting it is the caller's decision."""
    attempts = []

    def urlopen(request, timeout=None):
        attempts.append(1)
        return _refusal(429, {"Retry-After": "1"})(request, timeout)

    call(monkeypatch, urlopen)
    assert len(attempts) == 1
