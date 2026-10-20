"""
`connection()` and `list_connections()`: the requests the control plane's routes
accept, the refusals a caller branches on, and the per-id cache.

Pinned against aetherfy-control-plane `api/routes/connections.py`
(`POST /connections/{connection_id}/token`, body `{min_valid_seconds}`, answer
`ConnectionTokenResponse`; `GET /connections/usable`) and `shared/connections/broker.py` (the refusal
codes and statuses). The transport is faked the way every test in this
directory fakes it — `_http.request_json` replaced — because the helper speaks
urllib, which a requests-level mock would not see.
"""

from datetime import datetime, timedelta, timezone

import pytest

import aetherfy_agent
from aetherfy_agent import connection, list_connections
from aetherfy_agent.exceptions import (
    ConnectionAccessDenied,
    ConnectionGrantNotEnabled,
    ConnectionNeedsReauth,
    ConnectionNotFound,
    ConnectionTokenError,
    ConnectionUnavailable,
    NotRunningOnAgent,
)


GOOGLE = "6f1c2b7e-0a2d-4f8e-9c31-2b0d5a7e4411"
NOTION = "0b6f2d3a-91c4-4e0e-8a5d-7c1f3e9b2a64"


def answer(expires_in=3600, token="ya29.token", **extra):
    body = {
        "id": GOOGLE,
        "access_token": token,
        "token_type": "Bearer",
        "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=expires_in))
        .isoformat()
        .replace("+00:00", "Z")
        if expires_in is not None
        else None,
        "provider": "google",
        "name": "google",
        "account_label": "you@example.com",
        "scopes": ["openid", "https://www.googleapis.com/auth/drive.file"],
    }
    body.update(extra)
    return body


@pytest.fixture(autouse=True)
def on_a_machine(monkeypatch):
    monkeypatch.setenv("AETHERFY_API_URL", "https://agents.aetherfy.com/api/v1")
    monkeypatch.setenv("AETHERFY_API_KEY", "afy_test_agentkey")
    # Every test starts with an empty cache.
    monkeypatch.setattr(aetherfy_agent.connections, "_cache", {})


@pytest.fixture
def transport(monkeypatch):
    class Transport:
        def __init__(self):
            self.calls = []
            self.replies = [(200, answer())]

        def __call__(self, method, url, *, api_key, ua, body=None, **kwargs):
            self.calls.append(
                {
                    "method": method,
                    "url": url,
                    "api_key": api_key,
                    "ua": ua,
                    "body": body,
                }
            )
            return self.replies[min(len(self.calls), len(self.replies)) - 1]

    fake = Transport()
    monkeypatch.setattr(aetherfy_agent._http, "request_json", fake)
    return fake


def test_the_request_matches_the_route(transport):
    token = connection(GOOGLE, min_valid_seconds=120)

    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://agents.aetherfy.com/api/v1/connections/{0}/token".format(GOOGLE)
    assert call["api_key"] == "afy_test_agentkey"
    assert call["body"] == {"min_valid_seconds": 120}
    assert call["ua"].startswith("aetherfy-agent-python/")
    assert token.access_token == "ya29.token" and token.token_type == "Bearer"
    assert token.scopes == ("openid", "https://www.googleapis.com/auth/drive.file")
    assert token.expires_at.tzinfo is not None


def test_an_id_that_is_not_a_uuid_is_refused_before_any_request(transport):
    # "google" was a name; connections are addressed by id now.
    for bad in ("google", "a/b", "", None, 7):
        with pytest.raises(ValueError):
            connection(bad)
    assert transport.calls == []


def test_the_token_is_not_in_its_repr(transport):
    assert "ya29.token" not in repr(connection(GOOGLE))


def test_a_fresh_token_is_reused_from_the_cache(transport):
    first = connection(GOOGLE)
    second = connection(GOOGLE, min_valid_seconds=600)
    assert second is first
    assert len(transport.calls) == 1


def test_a_token_inside_the_margin_is_fetched_again(transport):
    transport.replies = [
        (200, answer(expires_in=200, token="old")),
        (200, answer(token="new")),
    ]
    assert connection(GOOGLE, min_valid_seconds=60).access_token == "old"
    # 200 s left is not enough for a caller asking for 300.
    assert connection(GOOGLE, min_valid_seconds=300).access_token == "new"
    assert len(transport.calls) == 2


def test_the_one_minute_floor_applies_to_the_cache_too(transport):
    transport.replies = [
        (200, answer(expires_in=50, token="old")),
        (200, answer(token="new")),
    ]
    connection(GOOGLE, min_valid_seconds=0)
    assert connection(GOOGLE, min_valid_seconds=0).access_token == "new"


def test_a_never_expiring_token_is_rechecked_after_a_while(transport, monkeypatch):
    # Cached, but not for good: a disconnect on the dashboard revokes it, and a
    # long-running agent must hear about that.
    transport.replies = [
        (200, answer(expires_in=None, token="first", provider="notion")),
        (200, answer(expires_in=None, token="second", provider="notion")),
    ]
    clock = [1000.0]
    monkeypatch.setattr(aetherfy_agent.connections, "monotonic", lambda: clock[0])
    first = connection(NOTION, min_valid_seconds=3000)
    assert first.expires_at is None
    clock[0] += aetherfy_agent.connections._NO_EXPIRY_RECHECK_SECONDS - 1
    assert connection(NOTION, min_valid_seconds=3000) is first
    assert len(transport.calls) == 1
    clock[0] += 1
    assert connection(NOTION, min_valid_seconds=3000).access_token == "second"
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    "status,code,exc,retryable",
    [
        (404, "CONNECTION_NOT_FOUND", ConnectionNotFound, False),
        (403, "CONNECTION_GRANT_NOT_ENABLED", ConnectionGrantNotEnabled, False),
        (409, "CONNECTION_NEEDS_REAUTH", ConnectionNeedsReauth, False),
        (502, "CONNECTION_PROVIDER_UNAVAILABLE", ConnectionUnavailable, True),
        (403, "CONNECTION_REQUIRES_AGENT_KEY", ConnectionAccessDenied, False),
    ],
)
def test_each_refusal_gets_its_type(transport, status, code, exc, retryable):
    transport.replies = [
        (status, {"detail": {"code": code, "message": "said the server"}})
    ]
    with pytest.raises(exc) as raised:
        connection(GOOGLE)
    assert raised.value.error_code == code
    assert raised.value.status_code == status
    assert raised.value.retryable is retryable
    assert str(raised.value).startswith("said the server")


def test_a_key_without_the_scope_is_the_generic_error_with_the_403(transport):
    # The key's route policy refuses with 403 AUTH_AGENT_KEY_OUT_OF_SCOPE. That is
    # NOT "no such connection": it must arrive as the generic
    # ConnectionTokenError, never as ConnectionNotFound.
    transport.replies = [
        (
            403,
            {
                "detail": {
                    "code": "AUTH_AGENT_KEY_OUT_OF_SCOPE",
                    "message": "this key cannot do that",
                    "required_scope": "connections:token",
                }
            },
        )
    ]
    with pytest.raises(ConnectionTokenError) as raised:
        connection(GOOGLE)
    assert type(raised.value) is ConnectionTokenError
    assert raised.value.status_code == 403
    assert raised.value.error_code == "AUTH_AGENT_KEY_OUT_OF_SCOPE"
    assert not isinstance(raised.value, ConnectionNotFound)


def test_a_grant_whose_switch_is_off_is_its_own_type(transport):
    transport.replies = [
        (
            403,
            {
                "detail": {
                    "code": "CONNECTION_GRANT_NOT_ENABLED",
                    "message": "the grant does not cover connections",
                    "connection_id": GOOGLE,
                }
            },
        )
    ]
    with pytest.raises(ConnectionGrantNotEnabled) as raised:
        connection(GOOGLE)
    # 403 CONNECTION_REQUIRES_AGENT_KEY is a different thing: the code picks the type.
    assert not isinstance(raised.value, ConnectionAccessDenied)
    assert raised.value.details["connection_id"] == GOOGLE


def test_an_unrecognised_pairing_is_reported_as_it_came(transport):
    # A 404 with another code (the key's agent was deleted) is not "no such
    # connection", and must not wear that type.
    transport.replies = [
        (404, {"detail": {"code": "AGENT_NOT_FOUND", "message": "gone"}})
    ]
    with pytest.raises(ConnectionTokenError) as raised:
        connection(GOOGLE)
    assert type(raised.value) is ConnectionTokenError
    assert raised.value.error_code == "AGENT_NOT_FOUND"


def test_needs_reauth_drops_the_cached_token(transport):
    transport.replies = [
        (200, answer(expires_in=200, token="old")),
        (409, {"detail": {"code": "CONNECTION_NEEDS_REAUTH", "message": "reconnect"}}),
    ]
    connection(GOOGLE, min_valid_seconds=60)
    with pytest.raises(ConnectionNeedsReauth):
        connection(GOOGLE, min_valid_seconds=300)
    assert GOOGLE not in aetherfy_agent.connections._cache


@pytest.mark.parametrize("value", [-1, 3001, 1.5, True])
def test_min_valid_seconds_is_checked_before_any_request(transport, value):
    with pytest.raises(ValueError):
        connection(GOOGLE, min_valid_seconds=value)
    assert transport.calls == []


def test_off_a_machine_it_says_so(transport, monkeypatch):
    monkeypatch.delenv("AETHERFY_API_KEY")
    with pytest.raises(NotRunningOnAgent):
        connection(GOOGLE)
    assert transport.calls == []


def test_list_connections_asks_the_usable_route_and_builds_infos(transport):
    transport.replies = [
        (
            200,
            [
                {
                    "id": GOOGLE,
                    "name": "google",
                    "provider": "google",
                    "scope": "workspace",
                    "account_label": "you@example.com",
                    "status": "connected",
                    "scopes": ["openid"],
                    "used_by": None,
                }
            ],
        )
    ]
    found = list_connections()
    call = transport.calls[0]
    assert (call["method"], call["url"]) == (
        "GET",
        "https://agents.aetherfy.com/api/v1/connections/usable",
    )
    assert call["api_key"] == "afy_test_agentkey" and call["body"] is None
    assert [(c.id, c.name, c.provider, c.scope, c.status) for c in found] == [
        (GOOGLE, "google", "google", "workspace", "connected")
    ]
    assert found[0].scopes == ("openid",)


def test_list_connections_refusals_are_typed(transport):
    transport.replies = [
        (403, {"detail": {"code": "CONNECTION_REQUIRES_AGENT_KEY", "message": "no"}})
    ]
    with pytest.raises(ConnectionAccessDenied):
        list_connections()
