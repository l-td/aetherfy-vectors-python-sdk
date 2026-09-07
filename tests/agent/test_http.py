"""
The transport, exercised directly.

Every other test in this directory replaces `_http.request_json`, which proves
what the helper SENDS but nothing about what the transport DOES. These tests
cover the other half: that an HTTP status is returned rather than raised, that
a connection failure is retried exactly once, and that a status is never
retried at all.

`urlopen` is the only thing stubbed. Nothing here opens a socket.
"""

import io
import json
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


def call(monkeypatch, urlopen):
    monkeypatch.setattr(_http.urllib.request, "urlopen", urlopen)
    return _http.request_json(
        "POST",
        "https://agents.aetherfy.com/api/v1/agents/a/spawn",
        api_key="afy_test_key",
        ua="aetherfy-agent-python/test",
        body={"child_agent_id": "worker", "payload": {}},
    )


def test_the_headers_and_body_go_out_as_json(monkeypatch):
    seen = {}

    def urlopen(request, timeout=None):
        seen["headers"] = dict(request.headers)
        seen["data"] = request.data
        seen["method"] = request.get_method()
        return FakeResponse(202, {"status": "queued"})

    status, body = call(monkeypatch, urlopen)

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

    status, body = call(monkeypatch, urlopen)

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


def test_a_connection_failure_is_retried_once_then_raises(monkeypatch):
    attempts = []

    def urlopen(request, timeout=None):
        attempts.append(1)
        raise urllib.error.URLError("connection reset")

    with pytest.raises(AgentTransportError) as excinfo:
        call(monkeypatch, urlopen)

    assert len(attempts) == 2, "one retry, not zero and not a loop"
    assert "connection reset" in str(excinfo.value)


def test_a_retried_connection_failure_can_succeed(monkeypatch):
    attempts = []

    def urlopen(request, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.URLError("dns blip")
        return FakeResponse(202, {"status": "queued"})

    status, body = call(monkeypatch, urlopen)

    assert len(attempts) == 2
    assert (status, body) == (202, {"status": "queued"})


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

    status, body = call(monkeypatch, urlopen)

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

    status, body = call(monkeypatch, urlopen)

    assert status == 403
    assert body is None
