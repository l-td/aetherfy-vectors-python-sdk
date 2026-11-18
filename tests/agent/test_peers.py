"""
`agent_url()`, `agent_urls()` and `call_agent()`: finding a peer at call time.

The server side is built apart from this SDK, so the contract is mocked
exactly as it was specified:

  GET <AETHERFY_PEERS_URL, {id} = this agent>/<name>
      200 {"name": ..., "url": ...}
      404 AGENT_NOT_FOUND, 409 PEER_HAS_NO_URL, 403 AUTH_AGENT_KEY_OUT_OF_SCOPE
  GET <AETHERFY_PEERS_URL, {id} = this agent>
      200 {"peers": [{"name": ..., "url": ...}]}

Two stand-ins, one per network:

  * the control plane's lookup replaces `_http.request_json`, as the spawn
    tests do, so every test counts the lookups the code made;
  * the peers replace the opener `request_peer` sends through, so `request_peer` itself runs and what it puts
    on the wire (above all: no Aetherfy key) is read off the request.

TIME NEVER PASSES BY WAITING. The cache takes an injected clock; a test moves
it by hand.
"""

import io
import json
import socket
import threading
import urllib.error
from urllib.parse import unquote
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import aetherfy_agent
from aetherfy_agent import (
    PEER_CACHE_SECONDS,
    PEER_RETRY_CAP_SECONDS,
    PEER_RETRY_FLOOR_SECONDS,
    agent_url,
    agent_urls,
    call_agent,
)
from aetherfy_agent import _http, _peers
from aetherfy_agent.exceptions import (
    AgentTransportError,
    NotRunningOnAgent,
    PeerHasNoUrl,
    PeerLookupError,
    PeerNotFound,
)
from aetherfy_agent.models import PeerResponse

PEERS_TEMPLATE = "https://agents.aetherfy.com/api/v1/agents/{id}/peers"
PEERS = "https://agents.aetherfy.com/api/v1/agents/parent-agent-id/peers"
KEY = "afy_test_key_do_not_leak"

OLD = "https://analyst-k3m7x2.aetherfy.dev"
NEW = "https://analyst-q9w4h1.aetherfy.dev"


class Clock:
    """The injected clock: a number a test moves, never a wait."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture(autouse=True)
def on_a_machine(monkeypatch):
    # Exactly what the platform injects, including the literal `{id}`.
    monkeypatch.setenv("AETHERFY_PEERS_URL", PEERS_TEMPLATE)
    monkeypatch.setenv("AETHERFY_AGENT_ID", "parent-agent-id")
    monkeypatch.setenv("AETHERFY_API_KEY", KEY)


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    # A fresh cache per test, holding the test's clock.
    monkeypatch.setattr(_peers, "cache", _peers.PeerCache(clock=fake))
    return fake


class Lookup:
    """The control plane's peers route. Counts what reaches it."""

    def __init__(self):
        self.calls = []
        # name -> (status, body); None as the name is the list route.
        self.replies = {}
        self.retry_after = None
        # set to an exception and every lookup raises it, after being counted
        self.fail_with = None

    def answer(self, name, url):
        self.replies[name] = (200, {"name": name, "url": url})

    def __call__(self, method, url, *, api_key, ua, body=None, **kwargs):
        self.calls.append(
            {"method": method, "url": url, "api_key": api_key, "ua": ua, "body": body}
        )
        if url == PEERS:
            key = None
        else:
            assert url.startswith(PEERS + "/"), url
            key = unquote(url[len(PEERS) + 1 :])
        if self.fail_with is not None:
            raise self.fail_with
        status, reply = self.replies[key]
        return status, reply, self.retry_after

    @property
    def count(self):
        return len(self.calls)


@pytest.fixture
def lookup(monkeypatch, clock):
    fake = Lookup()
    fake.answer("analyst", OLD)
    monkeypatch.setattr(_http, "request_json", fake)
    return fake


def envelope(code, message="refused"):
    return {"detail": {"code": code, "message": message}}


# ----------------------------------------------------------------- agent_url


def test_the_lookup_request_matches_the_route(lookup):
    assert agent_url("analyst") == OLD

    call = lookup.calls[0]
    assert call["method"] == "GET"
    # THE VARIABLE IS A TEMPLATE, resolved with this machine's own agent id.
    assert call["url"] == PEERS + "/analyst"
    assert "{id}" not in call["url"]
    assert call["api_key"] == KEY
    assert call["ua"].startswith("aetherfy-agent-python/")
    assert call["body"] is None


def test_the_name_is_quoted_into_the_path(lookup):
    lookup.replies["a/b c"] = (200, {"name": "a/b c", "url": OLD})

    agent_url("a/b c")

    assert lookup.calls[0]["url"] == PEERS + "/a%2Fb%20c"


def test_an_already_resolved_url_is_left_alone(lookup, monkeypatch):
    monkeypatch.setenv(
        "AETHERFY_PEERS_URL", "https://agents.aetherfy.com/api/v1/agents/abc/peers/"
    )
    monkeypatch.delenv("AETHERFY_AGENT_ID")
    seen = []

    def spy(method, url, **kwargs):
        seen.append(url)
        return 200, {"name": "analyst", "url": OLD}, None

    monkeypatch.setattr(_http, "request_json", spy)

    agent_url("analyst")

    assert seen == ["https://agents.aetherfy.com/api/v1/agents/abc/peers/analyst"]


def test_a_fresh_answer_is_served_without_a_second_lookup(lookup):
    assert agent_url("analyst") == OLD
    assert agent_url("analyst") == OLD
    assert agent_url("analyst") == OLD

    assert lookup.count == 1


def test_the_cache_window_is_thirty_seconds():
    assert PEER_CACHE_SECONDS == 30
    assert aetherfy_agent.PEER_CACHE_SECONDS is _peers.PEER_CACHE_SECONDS


def test_the_cache_is_keyed_by_name(lookup):
    lookup.answer("indexer", "https://indexer-a1b2c3.aetherfy.dev")

    assert agent_url("analyst") == OLD
    assert agent_url("indexer") == "https://indexer-a1b2c3.aetherfy.dev"
    assert agent_url("analyst") == OLD

    assert [c["url"] for c in lookup.calls] == [
        PEERS + "/analyst",
        PEERS + "/indexer",
    ]


def test_the_entry_is_served_until_the_window_closes_and_not_after(lookup, clock):
    agent_url("analyst")
    clock.advance(PEER_CACHE_SECONDS - 0.001)
    agent_url("analyst")
    assert lookup.count == 1

    clock.advance(0.001)
    agent_url("analyst")
    assert lookup.count == 2


def test_the_window_runs_from_the_lookup_not_from_the_last_read(lookup, clock):
    """A busy caller must not keep an address alive forever by reading it."""
    agent_url("analyst")  # t = 0
    clock.advance(14)
    agent_url("analyst")  # t = 14: a read, which must not renew the entry
    clock.advance(14)
    agent_url("analyst")  # t = 28
    assert lookup.count == 1

    clock.advance(2)
    agent_url("analyst")  # t = 30, thirty seconds after the LOOKUP
    assert lookup.count == 2


def test_fresh_skips_the_cache_and_replaces_the_entry(lookup):
    assert agent_url("analyst") == OLD
    lookup.answer("analyst", NEW)

    assert agent_url("analyst") == OLD  # still inside the window
    assert agent_url("analyst", fresh=True) == NEW
    assert lookup.count == 2

    # The refresh REPLACED the entry: the next plain call is served it.
    assert agent_url("analyst") == NEW
    assert lookup.count == 2


def test_a_recreated_peer_is_found_after_the_window(lookup, clock):
    assert agent_url("analyst") == OLD
    lookup.answer("analyst", NEW)  # the peer was deleted and created again

    clock.advance(PEER_CACHE_SECONDS)

    assert agent_url("analyst") == NEW


def test_a_recreated_peer_is_found_at_once_with_fresh(lookup):
    assert agent_url("analyst") == OLD
    lookup.answer("analyst", NEW)

    assert agent_url("analyst", fresh=True) == NEW


def test_a_refusal_is_not_cached(lookup):
    lookup.replies["analyst"] = (404, envelope("AGENT_NOT_FOUND"))
    with pytest.raises(PeerNotFound):
        agent_url("analyst")

    lookup.answer("analyst", OLD)  # the peer is created a moment later

    assert agent_url("analyst") == OLD
    assert lookup.count == 2


def test_a_failed_refresh_leaves_the_entry_it_could_not_replace(lookup):
    agent_url("analyst")
    lookup.replies["analyst"] = (404, envelope("AGENT_NOT_FOUND"))

    with pytest.raises(PeerNotFound):
        agent_url("analyst", fresh=True)

    lookup.replies["analyst"] = (500, None)
    assert agent_url("analyst") == OLD  # nothing was asked
    assert lookup.count == 2


@pytest.mark.parametrize(
    "status, code, expected",
    [
        (404, "AGENT_NOT_FOUND", PeerNotFound),
        (409, "PEER_HAS_NO_URL", PeerHasNoUrl),
        (403, "AUTH_AGENT_KEY_OUT_OF_SCOPE", PeerLookupError),
    ],
)
def test_every_code_maps_to_its_exception(lookup, status, code, expected):
    lookup.replies["analyst"] = (status, envelope(code, "prose that may change"))

    with pytest.raises(PeerLookupError) as excinfo:
        agent_url("analyst")

    error = excinfo.value
    assert type(error) is expected
    assert error.error_code == code
    assert error.status_code == status
    assert "prose that may change" in str(error)
    assert error.details["code"] == code


@pytest.mark.parametrize(
    "status, code",
    [
        # THE PAIRING SELECTS THE TYPE: the status alone, or the code on
        # another status, must not wear a type whose code the platform never
        # sent with it.
        (404, "SOMETHING_ELSE"),
        (404, None),
        (409, "SOMETHING_ELSE"),
        (409, "AGENT_NOT_FOUND"),
        (403, "AGENT_NOT_FOUND"),
        (500, "PEER_HAS_NO_URL"),
    ],
)
def test_the_pairing_selects_the_type(lookup, status, code):
    lookup.replies["analyst"] = (status, envelope(code) if code else {"detail": "x"})

    with pytest.raises(PeerLookupError) as excinfo:
        agent_url("analyst")

    assert type(excinfo.value) is PeerLookupError
    assert excinfo.value.error_code == code
    assert excinfo.value.status_code == status


def test_a_refusal_carries_retry_after(lookup):
    lookup.replies["analyst"] = (503, envelope("SERVICE_UNAVAILABLE"))
    lookup.retry_after = 7.0

    with pytest.raises(PeerLookupError) as excinfo:
        agent_url("analyst")

    assert excinfo.value.retry_after == 7.0


def test_the_peer_errors_are_agent_errors():
    for exc in (PeerNotFound, PeerHasNoUrl, PeerLookupError):
        assert issubclass(exc, aetherfy_agent.AgentError)
    assert issubclass(PeerNotFound, PeerLookupError)
    assert issubclass(PeerHasNoUrl, PeerLookupError)
    assert not issubclass(PeerNotFound, PeerHasNoUrl)


@pytest.mark.parametrize(
    "reply",
    [
        (200, {"name": "analyst"}),
        (200, {"name": "analyst", "url": ""}),
        (200, {"name": "analyst", "url": 7}),
        (200, ["not", "an", "object"]),
        (200, None),
    ],
)
def test_a_200_with_no_address_is_an_error_not_a_none(lookup, reply):
    lookup.replies["analyst"] = reply

    with pytest.raises(PeerLookupError) as excinfo:
        agent_url("analyst")

    assert excinfo.value.status_code == 200
    assert lookup.count == 1
    # And nothing was remembered from it.
    lookup.answer("analyst", OLD)
    assert agent_url("analyst") == OLD


@pytest.mark.parametrize(
    "variable", ["AETHERFY_PEERS_URL", "AETHERFY_API_KEY", "AETHERFY_AGENT_ID"]
)
def test_a_missing_variable_is_the_same_helpful_error_spawn_gives(
    lookup, monkeypatch, variable
):
    monkeypatch.delenv(variable)

    with pytest.raises(NotRunningOnAgent) as excinfo:
        agent_url("analyst")

    assert excinfo.value.variable == variable
    assert variable in str(excinfo.value)
    assert "Aetherfy agent" in str(excinfo.value)
    assert lookup.count == 0


def test_a_fresh_answer_needs_no_environment_to_be_served(lookup, monkeypatch):
    """Nothing is read to serve it; only a lookup needs the variables."""
    agent_url("analyst")
    monkeypatch.delenv("AETHERFY_PEERS_URL")

    assert agent_url("analyst") == OLD
    with pytest.raises(NotRunningOnAgent):
        agent_url("analyst", fresh=True)


def test_an_empty_name_is_refused_before_any_request(lookup):
    with pytest.raises(ValueError):
        agent_url("")
    assert lookup.count == 0


# ---------------------------------------------------------------- agent_urls

LISTED = {
    "peers": [
        {"name": "analyst", "url": OLD},
        {"name": "indexer", "url": "https://indexer-a1b2c3.aetherfy.dev"},
    ]
}


@pytest.fixture
def listed(lookup):
    lookup.replies[None] = (200, LISTED)
    return lookup


def test_the_list_request_matches_the_route(listed):
    assert agent_urls() == {
        "analyst": OLD,
        "indexer": "https://indexer-a1b2c3.aetherfy.dev",
    }

    call = listed.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == PEERS
    assert call["api_key"] == KEY


def test_an_empty_workspace_is_an_empty_dict(listed):
    listed.replies[None] = (200, {"peers": []})

    assert agent_urls() == {}
    assert agent_urls() == {}
    assert listed.count == 1  # an empty answer is an answer, and is remembered


def test_the_list_is_one_cache_entry(listed, clock):
    agent_urls()
    agent_urls()
    assert listed.count == 1

    clock.advance(PEER_CACHE_SECONDS - 0.001)
    agent_urls()
    assert listed.count == 1

    clock.advance(0.001)
    agent_urls()
    assert listed.count == 2


def test_fresh_refreshes_the_list(listed):
    agent_urls()
    listed.replies[None] = (200, {"peers": [{"name": "analyst", "url": NEW}]})

    assert agent_urls() == {
        "analyst": OLD,
        "indexer": "https://indexer-a1b2c3.aetherfy.dev",
    }
    assert agent_urls(fresh=True) == {"analyst": NEW}
    assert agent_urls() == {"analyst": NEW}
    assert listed.count == 2


def test_a_recreated_peer_is_in_the_list_after_the_window(listed, clock):
    agent_urls()
    listed.replies[None] = (200, {"peers": [{"name": "analyst", "url": NEW}]})
    clock.advance(PEER_CACHE_SECONDS)

    assert agent_urls() == {"analyst": NEW}


def test_editing_the_returned_dict_does_not_edit_the_cache(listed):
    first = agent_urls()
    first["intruder"] = "https://evil.example"
    first["analyst"] = "https://evil.example"

    again = agent_urls()
    assert "intruder" not in again
    assert again["analyst"] == OLD
    again["x"] = "y"
    assert "x" not in agent_urls()


def test_the_list_and_a_name_are_separate_memories(listed):
    agent_urls()
    assert listed.count == 1

    assert agent_url("analyst") == OLD  # not answered from the list
    assert listed.count == 2

    agent_url("analyst")
    agent_urls()
    assert listed.count == 2


@pytest.mark.parametrize(
    "status, code, expected",
    [
        (403, "AUTH_AGENT_KEY_OUT_OF_SCOPE", PeerLookupError),
        (404, "AGENT_NOT_FOUND", PeerNotFound),
        (409, "PEER_HAS_NO_URL", PeerHasNoUrl),
    ],
)
def test_the_list_maps_codes_as_the_name_does(listed, status, code, expected):
    listed.replies[None] = (status, envelope(code))

    with pytest.raises(PeerLookupError) as excinfo:
        agent_urls()

    assert type(excinfo.value) is expected
    assert excinfo.value.error_code == code


@pytest.mark.parametrize(
    "reply",
    [
        (200, {}),
        (200, {"peers": "analyst"}),
        (200, {"peers": [{"name": "analyst"}]}),
        (200, {"peers": [{"url": OLD}]}),
        (200, {"peers": ["analyst"]}),
        (200, {"peers": [{"name": "", "url": OLD}]}),
        (200, ["x"]),
    ],
)
def test_a_malformed_list_is_an_error_and_is_not_remembered(listed, reply):
    listed.replies[None] = reply
    with pytest.raises(PeerLookupError):
        agent_urls()

    listed.replies[None] = (200, LISTED)
    assert len(agent_urls()) == 2
    assert listed.count == 2


def test_the_list_needs_the_same_environment(listed, monkeypatch):
    monkeypatch.delenv("AETHERFY_PEERS_URL")

    with pytest.raises(NotRunningOnAgent) as excinfo:
        agent_urls()

    assert excinfo.value.variable == "AETHERFY_PEERS_URL"


# ---------------------------------------------------------------- call_agent


class FakeResponse(io.BytesIO):
    def __init__(self, status, body, headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        super().__init__(body.encode("utf-8") if isinstance(body, str) else body)
        self._status = status
        self.headers = headers or {}

    def getcode(self):
        return self._status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class BrokenBody(io.BytesIO):
    """A response whose body dies mid-stream, after the status arrived."""

    headers = {}

    def getcode(self):
        return 200

    def read(self, *args):
        raise ConnectionResetError("reset mid-body")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Peers:
    """The peers' network. One behaviour per base URL; every request recorded.

    A behaviour is an Exception to raise, or `(status, body)` to answer, or a
    list of those used in turn.
    """

    def __init__(self):
        self.requests = []
        self.behaviour = {}

    def __call__(self, request, timeout=None):
        self.requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "data": request.data,
                "headers": {k.lower(): v for k, v in request.header_items()},
                "timeout": timeout,
            }
        )
        base = next(b for b in self.behaviour if request.full_url.startswith(b))
        behaviour = self.behaviour[base]
        if isinstance(behaviour, list):
            behaviour = behaviour.pop(0)
        if isinstance(behaviour, BaseException):
            raise behaviour
        if isinstance(behaviour, (FakeResponse, BrokenBody)):
            return behaviour
        status, body = behaviour[:2]
        headers = behaviour[2] if len(behaviour) > 2 else {}
        if status >= 400:
            raise urllib.error.HTTPError(
                request.full_url,
                status,
                "error",
                headers,
                FakeResponse(status, body),
            )
        return FakeResponse(status, body, headers)

    def open(self, request, timeout=None):
        """The opener's method: `request_peer` sends through `_PEER_OPENER`."""
        return self(request, timeout)

    def urls(self):
        return [r["url"] for r in self.requests]


@pytest.fixture
def peers(monkeypatch, lookup):
    fake = Peers()
    monkeypatch.setattr(_http, "_PEER_OPENER", fake)
    return fake


REFUSED_NOT_FOUND = (404, {"error": {"code": "AGENT_NOT_FOUND", "message": "gone"}})
REFUSED_NOT_REACHABLE = (404, {"error": {"code": "AGENT_NOT_REACHABLE"}})
UNREACHABLE = urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed"))


def test_call_agent_sends_the_request_to_the_url_plus_the_path(peers):
    peers.behaviour[OLD] = (200, {"ok": True})

    response = call_agent(
        "analyst",
        "/summarise?depth=2",
        method="POST",
        json={"id": 7},
        headers={"X-Trace": "t-1"},
        timeout=5,
    )

    request = peers.requests[0]
    assert request["url"] == OLD + "/summarise?depth=2"
    assert request["method"] == "POST"
    assert json.loads(request["data"]) == {"id": 7}
    assert request["headers"]["content-type"] == "application/json"
    assert request["headers"]["x-trace"] == "t-1"
    assert request["headers"]["user-agent"].startswith("aetherfy-agent-python/")
    assert request["timeout"] == 5
    assert response.status == 200


def test_a_get_sends_no_body_and_a_trailing_slash_is_not_doubled(peers, lookup):
    lookup.answer("analyst", OLD + "/")
    peers.behaviour[OLD] = (200, "hello")

    call_agent("analyst", "/health")

    request = peers.requests[0]
    assert request["url"] == OLD + "/health"
    assert request["method"] == "GET"
    assert request["data"] is None
    assert "content-type" not in request["headers"]


def test_a_caller_content_type_is_kept(peers):
    peers.behaviour[OLD] = (200, "ok")

    call_agent(
        "analyst",
        "/x",
        method="PUT",
        json=[1, 2],
        headers={"content-type": "application/vnd.api+json"},
    )

    headers = peers.requests[0]["headers"]
    assert headers["content-type"] == "application/vnd.api+json"
    assert len([k for k in peers.requests[0]["headers"] if k == "content-type"]) == 1


def test_json_false_and_zero_are_bodies(peers):
    """`json=None` is the absence; every other JSON value is a body."""
    peers.behaviour[OLD] = (200, "ok")

    call_agent("analyst", "/x", method="POST", json=False)
    call_agent("analyst", "/x", method="POST", json=0)

    assert [r["data"] for r in peers.requests] == [b"false", b"0"]


def test_the_aetherfy_key_is_never_sent_to_the_peer(peers):
    peers.behaviour[OLD] = (200, "ok")

    call_agent("analyst", "/x", method="POST", json={"id": 7}, headers={"X-A": "b"})

    request = peers.requests[0]
    assert "authorization" not in request["headers"]
    assert KEY not in json.dumps(request["headers"])
    assert KEY not in request["url"]
    assert KEY.encode() not in request["data"]
    # The lookup, and only the lookup, carried it.


def test_the_key_goes_to_every_lookup_and_to_no_peer_attempt(peers, lookup):
    peers.behaviour[OLD] = UNREACHABLE
    peers.behaviour[NEW] = (200, "ok")
    agent_url("analyst")
    lookup.answer("analyst", NEW)

    call_agent("analyst", "/x")  # OLD is down; the re-resolve finds NEW

    assert len(peers.requests) == 2
    assert lookup.count == 2
    assert all(c["api_key"] == KEY for c in lookup.calls)
    for request in peers.requests:
        assert "authorization" not in request["headers"]
        assert KEY not in json.dumps(request["headers"])


def test_the_answer_is_a_small_response_object(peers):
    peers.behaviour[OLD] = (
        201,
        {"summary": "done"},
        {"Content-Type": "application/json", "X-Request-Id": "r-9"},
    )

    response = call_agent("analyst", "/x")

    assert isinstance(response, PeerResponse)
    assert response.status == 201
    assert response.headers == {
        "content-type": "application/json",
        "x-request-id": "r-9",
    }
    assert response.text == '{"summary": "done"}'
    assert response.json() == {"summary": "done"}


def test_json_raises_value_error_on_a_body_that_is_not_json(peers):
    peers.behaviour[OLD] = (200, "plain text")

    response = call_agent("analyst", "/x")

    assert response.text == "plain text"
    with pytest.raises(ValueError):
        response.json()


def test_a_body_that_is_not_utf8_is_replaced_not_raised(peers):
    peers.behaviour[OLD] = (200, b"ok \xff\xfe")

    assert call_agent("analyst", "/x").text == "ok ��"


def test_call_agent_uses_the_cache_for_the_lookup(peers, lookup):
    peers.behaviour[OLD] = (200, "ok")

    call_agent("analyst", "/x")
    call_agent("analyst", "/y")
    call_agent("analyst", "/z")

    assert lookup.count == 1
    assert len(peers.requests) == 3


@pytest.mark.parametrize(
    "failure",
    [
        UNREACHABLE,
        urllib.error.URLError(ConnectionRefusedError(111, "refused")),
        ConnectionRefusedError(111, "refused"),
    ],
    ids=["dns", "refused-urlerror", "refused"],
)
def test_it_re_resolves_once_when_no_peer_was_reached(peers, lookup, failure):
    peers.behaviour[OLD] = failure
    peers.behaviour[NEW] = (200, {"ok": True})
    agent_url("analyst")  # the cache now holds OLD
    lookup.answer("analyst", NEW)  # the peer was recreated meanwhile
    lookups_before = lookup.count

    response = call_agent("analyst", "/x")

    assert response.status == 200
    assert peers.urls() == [OLD + "/x", NEW + "/x"]
    # One extra lookup, and it was a FRESH one: the cache still held OLD.
    assert lookup.count == lookups_before + 1
    assert lookup.calls[-1]["url"] == PEERS + "/analyst"
    # The re-resolve replaced the entry: the next call needs no lookup.
    call_agent("analyst", "/y")
    assert lookup.count == lookups_before + 1
    assert peers.urls()[-1] == NEW + "/y"


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionResetError(104, "reset"),
        TimeoutError("timed out"),
        socket.timeout("timed out"),
        urllib.error.URLError(socket.timeout("timed out")),
    ],
    ids=["reset", "timeout", "socket-timeout", "urlerror-timeout"],
)
def test_a_reset_or_a_timeout_is_not_replayed_because_the_peer_may_have_acted(
    peers, lookup, failure
):
    peers.behaviour[OLD] = failure
    peers.behaviour[NEW] = (200, {"ok": True})
    agent_url("analyst")
    lookup.answer("analyst", NEW)
    lookups_before = lookup.count

    with pytest.raises(AgentTransportError, match="may or may not have run"):
        call_agent("analyst", "/charge", method="POST", json={"cents": 5})

    # Exactly ONE request left the machine, and no second lookup was made.
    assert peers.urls() == [OLD + "/charge"]
    assert lookup.count == lookups_before


@pytest.mark.parametrize(
    "refusal",
    [REFUSED_NOT_FOUND, REFUSED_NOT_REACHABLE],
    ids=["not-found", "not-reachable"],
)
def test_it_re_resolves_once_when_the_edge_refuses_a_retired_address(
    peers, lookup, refusal
):
    peers.behaviour[OLD] = refusal
    peers.behaviour[NEW] = (200, {"ok": True})
    agent_url("analyst")
    lookup.answer("analyst", NEW)

    response = call_agent("analyst", "/x", method="POST", json={"a": 1})

    assert response.status == 200
    assert peers.urls() == [OLD + "/x", NEW + "/x"]
    # The retry carries the same method and body.
    assert [r["method"] for r in peers.requests] == ["POST", "POST"]
    assert [r["data"] for r in peers.requests] == [b'{"a": 1}'] * 2
    assert lookup.count == 2


@pytest.mark.parametrize(
    "answer",
    [
        (500, "boom"),
        (502, {"detail": "bad gateway"}),
        (503, {"error": {"code": "AGENT_NOT_REACHABLE"}}),  # 503 is not the refusal
        (404, "not json at all"),
        (404, {"error": {"code": "ROUTE_NOT_FOUND"}}),  # the peer's own 404
        (404, {"detail": {"code": "AGENT_NOT_FOUND"}}),  # not the edge's envelope
        (404, {"error": "AGENT_NOT_FOUND"}),
        (404, ["AGENT_NOT_FOUND"]),
        (404, ""),
        (403, {"error": {"code": "AGENT_NOT_FOUND"}}),
        (200, {"error": {"code": "AGENT_NOT_FOUND"}}),
        (400, {"error": {"code": "BAD"}}),
    ],
)
def test_any_other_status_is_returned_as_is_and_never_retried(peers, lookup, answer):
    peers.behaviour[OLD] = answer

    response = call_agent("analyst", "/x")

    assert response.status == answer[0]
    assert len(peers.requests) == 1
    assert lookup.count == 1


def test_a_second_connect_failure_raises_and_stops(peers, lookup):
    peers.behaviour[OLD] = UNREACHABLE
    peers.behaviour[NEW] = urllib.error.URLError(ConnectionRefusedError(111, "x"))
    agent_url("analyst")
    lookup.answer("analyst", NEW)

    with pytest.raises(AgentTransportError):
        call_agent("analyst", "/x")

    assert peers.urls() == [OLD + "/x", NEW + "/x"]  # not a third attempt
    assert lookup.count == 2


def test_a_second_refusal_raises_and_stops(peers, lookup):
    peers.behaviour[OLD] = REFUSED_NOT_REACHABLE
    peers.behaviour[NEW] = REFUSED_NOT_FOUND
    agent_url("analyst")
    lookup.answer("analyst", NEW)

    with pytest.raises(AgentTransportError) as excinfo:
        call_agent("analyst", "/x")

    assert "AGENT_NOT_FOUND" in str(excinfo.value)
    assert peers.urls() == [OLD + "/x", NEW + "/x"]


def test_a_failure_then_a_peer_error_status_is_returned(peers, lookup):
    """The second attempt's answer is any answer: only the FIRST failure kinds
    are retried, and the second attempt is the last whatever it says."""
    peers.behaviour[OLD] = UNREACHABLE
    peers.behaviour[NEW] = (500, "boom")
    agent_url("analyst")
    lookup.answer("analyst", NEW)

    assert call_agent("analyst", "/x").status == 500
    assert len(peers.requests) == 2


def test_the_same_address_after_a_re_resolve_is_still_tried_once_more(peers, lookup):
    """A blip, not a recreation: the fresh lookup answers the same URL."""
    peers.behaviour[OLD] = [UNREACHABLE, (200, "ok")]

    assert call_agent("analyst", "/x").status == 200
    assert peers.urls() == [OLD + "/x", OLD + "/x"]
    assert lookup.count == 2


def test_the_re_resolve_failing_to_find_the_peer_surfaces_that(peers, lookup):
    peers.behaviour[OLD] = REFUSED_NOT_FOUND
    agent_url("analyst")
    lookup.replies["analyst"] = (404, envelope("AGENT_NOT_FOUND"))

    with pytest.raises(PeerNotFound):
        call_agent("analyst", "/x")

    assert len(peers.requests) == 1


def test_a_peer_with_no_url_raises_before_any_peer_request(peers, lookup):
    lookup.replies["analyst"] = (409, envelope("PEER_HAS_NO_URL"))

    with pytest.raises(PeerHasNoUrl):
        call_agent("analyst", "/x")

    assert peers.requests == []


def test_a_body_that_dies_after_the_status_is_not_retried(peers, lookup):
    """The peer answered. A retry could repeat what it already did."""
    peers.behaviour[OLD] = BrokenBody()

    with pytest.raises(AgentTransportError) as excinfo:
        call_agent("analyst", "/x", method="POST", json={"charge": 1})

    assert type(excinfo.value) is AgentTransportError
    assert "body could not be read" in str(excinfo.value)
    assert len(peers.requests) == 1
    assert lookup.count == 1


@pytest.mark.parametrize("path", ["summarise", "", "http://evil.example/x"])
def test_a_path_must_start_with_a_slash(peers, lookup, path):
    with pytest.raises(ValueError):
        call_agent("analyst", path)

    assert peers.requests == [] and lookup.count == 0


@pytest.mark.parametrize("timeout", [0, -1, float("nan")])
def test_a_timeout_must_be_positive(peers, lookup, timeout):
    with pytest.raises(ValueError):
        call_agent("analyst", "/x", timeout=timeout)

    assert peers.requests == [] and lookup.count == 0


def test_call_agent_needs_the_lookup_environment(peers, monkeypatch):
    monkeypatch.delenv("AETHERFY_PEERS_URL")

    with pytest.raises(NotRunningOnAgent) as excinfo:
        call_agent("analyst", "/x")

    assert excinfo.value.variable == "AETHERFY_PEERS_URL"
    assert peers.requests == []


# ------------------------------------------- over a real socket (loopback only)


class _Handler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        type(self).seen.append((self.path, dict(self.headers), body))
        payload = json.dumps({"echo": json.loads(body)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_over_a_real_socket_a_dead_address_is_re_resolved_to_the_live_one(
    monkeypatch, lookup
):
    """The one test that opens sockets, on 127.0.0.1 only: a real refused
    connection, a real server, nothing else on the network."""
    _Handler.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # A port nothing listens on: bind it, read the number, close it.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead = "http://127.0.0.1:{0}".format(probe.getsockname()[1])
    probe.close()
    live = "http://127.0.0.1:{0}".format(server.server_address[1])
    try:
        lookup.answer("analyst", dead)
        agent_url("analyst")
        lookup.answer("analyst", live)  # recreated, now at the live address

        response = call_agent(
            "analyst", "/echo", method="POST", json={"n": 1}, timeout=10
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert response.status == 200
    assert response.json() == {"echo": {"n": 1}}
    ((path, headers, body),) = _Handler.seen
    assert path == "/echo"
    assert KEY not in json.dumps(headers)
    assert "Authorization" not in headers
    assert lookup.count == 2


@pytest.mark.parametrize("method", ["GET", "POST", "PUT"])
def test_over_a_real_socket_a_redirect_is_returned_not_followed(lookup, method):
    """A 3xx is the peer's answer. Following it would replay a POST as a GET
    (301/302), refuse a 307/308, and send the caller's headers wherever
    `Location` points."""

    class Redirecting(_Handler):
        seen = []

        def _redirect(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            type(self).seen.append((self.command, self.path))
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_POST = do_PUT = _redirect

    server = HTTPServer(("127.0.0.1", 0), Redirecting)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    live = "http://127.0.0.1:{0}".format(server.server_address[1])
    try:
        lookup.answer("analyst", live)

        response = call_agent(
            "analyst",
            "/moved",
            method=method,
            json=None if method == "GET" else {"n": 1},
            headers={"X-Mine": "1"},
            timeout=10,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert response.status == 302
    assert response.headers["location"] == "/elsewhere"
    # ONE request reached the peer: the redirect was not followed.
    assert Redirecting.seen == [(method, "/moved")]
    assert lookup.count == 1


# ------------------------------------------- a control plane that cannot answer
#
# A looked-up address outlives its window as the LAST KNOWN ONE and is served when the control plane
# cannot answer (429, 5xx, no connection), and the control plane is not asked again before its
# Retry-After. Every test counts lookups (events) and moves the injected clock; none waits.

WINDOW_PAST = PEER_CACHE_SECONDS + 1


def _known_then_expired(lookup, clock):
    assert agent_url("analyst") == OLD
    clock.advance(WINDOW_PAST)


@pytest.mark.parametrize(
    "status, retry_after",
    [(429, 7.0), (500, None), (503, 7.0), (502, None)],
    ids=["429", "500", "503", "502"],
)
def test_an_unanswered_lookup_serves_the_last_known_address(lookup, clock, status, retry_after):
    _known_then_expired(lookup, clock)
    lookup.replies["analyst"] = (status, envelope("SERVICE_UNAVAILABLE"))
    lookup.retry_after = retry_after

    assert agent_url("analyst") == OLD
    assert lookup.count == 2  # asked once, was refused, served the known address


def test_an_unreachable_control_plane_serves_the_last_known_address_too(lookup, clock):
    _known_then_expired(lookup, clock)
    lookup.fail_with = AgentTransportError("no connection")

    assert agent_url("analyst") == OLD


def test_the_control_plane_is_not_asked_again_before_its_retry_after(lookup, clock):
    _known_then_expired(lookup, clock)
    lookup.replies["analyst"] = (429, envelope("RATE_LIMIT_EXCEEDED"))
    lookup.retry_after = 10.0
    agent_url("analyst")
    asked = lookup.count

    clock.advance(9.9)
    assert agent_url("analyst") == OLD
    assert lookup.count == asked  # waiting: no request

    lookup.answer("analyst", NEW)
    clock.advance(0.1)
    assert agent_url("analyst") == NEW  # the wait is over: asked, and answered
    assert lookup.count == asked + 1


@pytest.mark.parametrize(
    "retry_after, waits",
    [
        (None, PEER_RETRY_FLOOR_SECONDS),
        (3600.0, PEER_RETRY_CAP_SECONDS),
        (20.0, 20.0),
    ],
    ids=["none-waits-the-floor", "long-is-capped", "in-range-is-honoured"],
)
def test_the_wait_is_the_retry_after_between_the_floor_and_the_cap(
    lookup, clock, retry_after, waits
):
    _known_then_expired(lookup, clock)
    lookup.replies["analyst"] = (503, envelope("SERVICE_UNAVAILABLE"))
    lookup.retry_after = retry_after
    agent_url("analyst")
    asked = lookup.count

    clock.advance(waits - 0.01)
    agent_url("analyst")
    assert lookup.count == asked
    clock.advance(0.02)
    agent_url("analyst")
    assert lookup.count == asked + 1


@pytest.mark.parametrize(
    "retry_after, waits",
    [
        (None, 5.0),
        (0.0, 5.0),
        (1.0, 5.0),
        (4.9, 5.0),
        (-3.0, 5.0),
        (float("nan"), 5.0),
        (5.0, 5.0),
        (20.0, 20.0),
        (60.0, 60.0),
        (120.0, 60.0),
    ],
    ids=["none", "zero", "one", "4.9", "negative", "nan", "5", "20", "60", "120"],
)
def test_retry_delay_clamps_both_ends(retry_after, waits):
    """The floor applies to a GIVEN Retry-After too: the control plane's own limiter sends small values, and a
    wait of 0 would give a sick control plane a lookup on every call."""
    assert _peers.retry_delay(retry_after) == waits


@pytest.mark.parametrize("retry_after", [0.0, 1.0, 4.9, -3.0], ids=["zero", "one", "4.9", "negative"])
def test_a_small_retry_after_still_waits_the_floor_before_asking_again(lookup, clock, retry_after):
    _known_then_expired(lookup, clock)
    lookup.replies["analyst"] = (429, envelope("RATE_LIMIT_EXCEEDED"))
    lookup.retry_after = retry_after
    agent_url("analyst")
    asked = lookup.count

    clock.advance(PEER_RETRY_FLOOR_SECONDS - 0.01)
    assert agent_url("analyst") == OLD
    assert lookup.count == asked  # still waiting: no request

    clock.advance(0.02)
    agent_url("analyst")
    assert lookup.count == asked + 1


@pytest.mark.parametrize(
    "status, code",
    [(403, "AUTH_AGENT_KEY_OUT_OF_SCOPE"), (404, "AGENT_NOT_FOUND"), (409, "PEER_HAS_NO_URL")],
)
def test_an_answer_is_never_served_around(lookup, clock, status, code):
    """The control plane SAID no: the known address is not a better answer than that."""
    _known_then_expired(lookup, clock)
    lookup.replies["analyst"] = (status, envelope(code))

    with pytest.raises(PeerLookupError) as excinfo:
        agent_url("analyst")

    assert excinfo.value.status_code == status


def test_a_peer_the_control_plane_says_is_gone_is_not_served_after_it(lookup, clock):
    _known_then_expired(lookup, clock)
    lookup.replies["analyst"] = (404, envelope("AGENT_NOT_FOUND"))
    with pytest.raises(PeerNotFound):
        agent_url("analyst")

    lookup.replies["analyst"] = (503, envelope("SERVICE_UNAVAILABLE"))
    with pytest.raises(PeerLookupError) as excinfo:
        agent_url("analyst")

    assert excinfo.value.status_code == 503  # nothing known to fall back to


def test_with_no_known_address_the_failure_is_raised_and_then_waited_out(lookup, clock):
    lookup.replies["analyst"] = (503, envelope("SERVICE_UNAVAILABLE"))
    lookup.retry_after = 7.0

    with pytest.raises(PeerLookupError) as first:
        agent_url("analyst")
    assert first.value.status_code == 503 and first.value.retry_after == 7.0

    clock.advance(2)
    with pytest.raises(PeerLookupError) as second:
        agent_url("analyst")

    assert lookup.count == 1  # the second did not ask
    assert second.value.status_code == 503
    assert second.value.retry_after == pytest.approx(5.0)
    assert "no earlier address is known" in str(second.value)


def test_fresh_is_never_answered_from_memory(lookup, clock):
    _known_then_expired(lookup, clock)
    lookup.replies["analyst"] = (503, envelope("SERVICE_UNAVAILABLE"))

    with pytest.raises(PeerLookupError):
        agent_url("analyst", fresh=True)
    with pytest.raises(PeerLookupError):  # and waiting is not an answer either
        agent_url("analyst", fresh=True)

    assert lookup.count == 2  # the first fresh asked; the second waited


def test_one_peers_outage_does_not_wait_out_another(lookup, clock):
    lookup.answer("indexer", "https://indexer-a1b2c3.aetherfy.dev")
    _known_then_expired(lookup, clock)
    lookup.replies["analyst"] = (503, envelope("SERVICE_UNAVAILABLE"))
    agent_url("analyst")

    assert agent_url("indexer") == "https://indexer-a1b2c3.aetherfy.dev"


def test_the_list_falls_back_to_the_last_list_known(lookup, clock):
    lookup.replies[None] = (200, {"peers": [{"name": "analyst", "url": OLD}]})
    assert agent_urls() == {"analyst": OLD}
    clock.advance(WINDOW_PAST)
    lookup.replies[None] = (429, envelope("RATE_LIMIT_EXCEEDED"))
    lookup.retry_after = 9.0

    served = agent_urls()
    served["analyst"] = "edited"  # a copy

    assert agent_urls() == {"analyst": OLD}
    assert lookup.count == 2  # the second read waited
    with pytest.raises(PeerLookupError):
        agent_urls(fresh=True)


def test_the_list_with_nothing_known_raises(lookup, clock):
    lookup.replies[None] = (503, envelope("SERVICE_UNAVAILABLE"))
    with pytest.raises(PeerLookupError) as excinfo:
        agent_urls()
    assert excinfo.value.status_code == 503


def test_call_agent_reaches_the_peer_at_the_known_address_through_an_outage(peers, lookup, clock):
    peers.behaviour[OLD] = (200, {"ok": True})
    agent_url("analyst")
    clock.advance(WINDOW_PAST)
    lookup.replies["analyst"] = (429, envelope("RATE_LIMIT_EXCEEDED"))
    lookup.retry_after = 5.0

    response = call_agent("analyst", "/x")

    assert response.status == 200
    assert peers.urls() == [OLD + "/x"]


def test_call_agent_does_not_retry_the_address_that_just_stopped_answering(peers, lookup, clock):
    """The re-lookup is for an address that is GONE: serving that same address as the stale one
    would be a second try of what just failed."""
    peers.behaviour[OLD] = UNREACHABLE
    agent_url("analyst")
    lookup.replies["analyst"] = (503, envelope("SERVICE_UNAVAILABLE"))

    with pytest.raises(PeerLookupError) as excinfo:
        call_agent("analyst", "/x")

    assert excinfo.value.status_code == 503
    assert peers.urls() == [OLD + "/x"]  # one request, to the address that was known


def test_a_failed_list_does_not_hold_back_a_lookup_by_name(lookup, clock):
    lookup.replies[None] = (503, envelope("SERVICE_UNAVAILABLE"))
    with pytest.raises(PeerLookupError):
        agent_urls()

    assert agent_url("analyst") == OLD  # asked, and answered: the two waits are separate


# ---------------------------------------- a lookup the control plane answered 503 RESOURCE_BUSY
#
# With a last known address the peer policy stands: it is served at once, and Retry-After is kept before asking again.
# With NOTHING known (the first lookup in this process, or a fresh one), a 503 RESOURCE_BUSY (nothing ran) is waited
# out once and asked again before the lookup fails; any other 429 or 5xx fails as it came. These run the REAL
# request_json: only `urlopen` is stubbed, and every wait is recorded (`_http._sleep`), never taken.


class Wire:
    """The control plane's peers route behind the real request_json: each lookup takes the next queued reply."""

    def __init__(self):
        self.replies = {}   # key (a name, or None for the list) -> [(status, body, retry_after), ...]
        self.calls = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        key = None if url == PEERS else unquote(url[len(PEERS) + 1:])
        self.calls.append(key)
        queued = self.replies.get(key) or []
        assert queued, "lookup {0}: a request the test did not expect".format(len(self.calls))
        status, body, retry_after = queued.pop(0)
        payload = io.BytesIO(json.dumps(body).encode("utf-8"))
        if status != 200:
            headers = {"Retry-After": retry_after} if retry_after else {}
            raise urllib.error.HTTPError(url, status, "refused", headers, payload)

        class Response(io.BytesIO):
            headers = {}

            def getcode(self):
                return 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Response(payload.getvalue())


BUSY = (503, envelope("RESOURCE_BUSY"), "5")


@pytest.fixture
def wire(monkeypatch, clock):
    fake = Wire()
    monkeypatch.setattr(_http.urllib.request, "urlopen", fake)
    return fake


@pytest.fixture
def waits(monkeypatch):
    waited = []
    monkeypatch.setattr(_http, "_sleep", waited.append)
    return waited


def outcome(call):
    """What a call returned, or the exception it raised: a test then fails on its assertion, naming both."""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001
        return exc


def found(name, url):
    return (200, {"name": name, "url": url}, None)


def test_with_no_known_address_resource_busy_is_waited_out_once_and_asked_again(wire, waits):
    wire.replies["analyst"] = [BUSY, found("analyst", OLD)]
    assert outcome(lambda: agent_url("analyst")) == OLD
    assert (len(wire.calls), waits) == (2, [5.0])


def test_with_no_known_address_any_other_503_fails_as_it_came(wire, waits):
    wire.replies["analyst"] = [(503, envelope("SERVICE_UNAVAILABLE"), "5")]
    with pytest.raises(PeerLookupError):
        agent_url("analyst")
    assert (len(wire.calls), waits) == (1, [])


def test_with_a_known_address_resource_busy_serves_it_at_once_and_waits_for_nothing(wire, waits, clock):
    wire.replies["analyst"] = [found("analyst", OLD)]
    assert agent_url("analyst") == OLD
    clock.advance(PEER_CACHE_SECONDS + 1)
    wire.replies["analyst"] = [BUSY]
    assert agent_url("analyst") == OLD
    assert (len(wire.calls), waits) == (2, [])


def test_a_fresh_lookup_has_nothing_known_to_serve_so_resource_busy_is_waited_out_once(wire, waits):
    wire.replies["analyst"] = [found("analyst", OLD), BUSY, found("analyst", NEW)]
    assert agent_url("analyst") == OLD
    assert outcome(lambda: agent_url("analyst", fresh=True)) == NEW
    assert (len(wire.calls), waits) == (3, [5.0])


def test_the_list_with_no_known_list_waits_out_resource_busy_once(wire, waits):
    wire.replies[None] = [BUSY, (200, {"peers": [{"name": "analyst", "url": OLD}]}, None)]
    assert agent_urls() == {"analyst": OLD}
    assert (len(wire.calls), waits) == (2, [5.0])


def test_the_list_with_a_known_list_serves_it_at_once(wire, waits, clock):
    wire.replies[None] = [(200, {"peers": [{"name": "analyst", "url": OLD}]}, None)]
    agent_urls()
    clock.advance(PEER_CACHE_SECONDS + 1)
    wire.replies[None] = [BUSY]
    assert agent_urls() == {"analyst": OLD}
    assert (len(wire.calls), waits) == (2, [])
