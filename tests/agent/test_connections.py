"""
`connection()`: the request the control plane's token route accepts, the four
refusals a caller branches on, and the per-name cache.

Pinned against aetherfy-control-plane `api/routes/connections.py`
(`POST /connections/{name}/token`, body `{min_valid_seconds}`, answer
`ConnectionTokenResponse`) and `shared/connections/broker.py` (the refusal
codes and statuses). The transport is faked the way every test in this
directory fakes it — `_http.request_json` replaced — because the helper speaks
urllib, which a requests-level mock would not see.
"""

from datetime import datetime, timedelta, timezone

import pytest

import aetherfy_agent
from aetherfy_agent import connection
from aetherfy_agent.exceptions import (
    ConnectionAccessDenied,
    ConnectionNeedsReauth,
    ConnectionNotFound,
    ConnectionTokenError,
    ConnectionUnavailable,
    NotRunningOnAgent,
)


def answer(expires_in=3600, token="ya29.token", **extra):
    body = {
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
    token = connection("google", min_valid_seconds=120)

    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://agents.aetherfy.com/api/v1/connections/google/token"
    assert call["api_key"] == "afy_test_agentkey"
    assert call["body"] == {"min_valid_seconds": 120}
    assert call["ua"].startswith("aetherfy-agent-python/")
    assert token.access_token == "ya29.token" and token.token_type == "Bearer"
    assert token.scopes == ("openid", "https://www.googleapis.com/auth/drive.file")
    assert token.expires_at.tzinfo is not None


def test_the_name_is_escaped_into_one_path_segment(transport):
    connection("a/b")
    assert transport.calls[0]["url"].endswith("/connections/a%2Fb/token")


def test_the_token_is_not_in_its_repr(transport):
    assert "ya29.token" not in repr(connection("google"))


def test_a_fresh_token_is_reused_from_the_cache(transport):
    first = connection("google")
    second = connection("google", min_valid_seconds=600)
    assert second is first
    assert len(transport.calls) == 1


def test_a_token_inside_the_margin_is_fetched_again(transport):
    transport.replies = [
        (200, answer(expires_in=200, token="old")),
        (200, answer(token="new")),
    ]
    assert connection("google", min_valid_seconds=60).access_token == "old"
    # 200 s left is not enough for a caller asking for 300.
    assert connection("google", min_valid_seconds=300).access_token == "new"
    assert len(transport.calls) == 2


def test_the_one_minute_floor_applies_to_the_cache_too(transport):
    transport.replies = [
        (200, answer(expires_in=50, token="old")),
        (200, answer(token="new")),
    ]
    connection("google", min_valid_seconds=0)
    assert connection("google", min_valid_seconds=0).access_token == "new"


def test_a_never_expiring_token_is_rechecked_after_a_while(transport, monkeypatch):
    # Cached, but not for good: a disconnect on the dashboard revokes it, and a
    # long-running agent must hear about that.
    transport.replies = [
        (200, answer(expires_in=None, token="first", provider="notion")),
        (200, answer(expires_in=None, token="second", provider="notion")),
    ]
    clock = [1000.0]
    monkeypatch.setattr(aetherfy_agent.connections, "monotonic", lambda: clock[0])
    first = connection("notion", min_valid_seconds=3000)
    assert first.expires_at is None
    clock[0] += aetherfy_agent.connections._NO_EXPIRY_RECHECK_SECONDS - 1
    assert connection("notion", min_valid_seconds=3000) is first
    assert len(transport.calls) == 1
    clock[0] += 1
    assert connection("notion", min_valid_seconds=3000).access_token == "second"
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    "status,code,exc,retryable",
    [
        (404, "CONNECTION_NOT_FOUND", ConnectionNotFound, False),
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
        connection("google")
    assert raised.value.error_code == code
    assert raised.value.status_code == status
    assert raised.value.retryable is retryable
    assert str(raised.value).startswith("said the server")


def test_an_unrecognised_pairing_is_reported_as_it_came(transport):
    # A 404 with another code (the key's agent was deleted) is not "no such
    # connection", and must not wear that type.
    transport.replies = [
        (404, {"detail": {"code": "AGENT_NOT_FOUND", "message": "gone"}})
    ]
    with pytest.raises(ConnectionTokenError) as raised:
        connection("google")
    assert type(raised.value) is ConnectionTokenError
    assert raised.value.error_code == "AGENT_NOT_FOUND"


def test_needs_reauth_drops_the_cached_token(transport):
    transport.replies = [
        (200, answer(expires_in=200, token="old")),
        (409, {"detail": {"code": "CONNECTION_NEEDS_REAUTH", "message": "reconnect"}}),
    ]
    connection("google", min_valid_seconds=60)
    with pytest.raises(ConnectionNeedsReauth):
        connection("google", min_valid_seconds=300)
    assert "google" not in aetherfy_agent.connections._cache


@pytest.mark.parametrize("value", [-1, 3001, 1.5, True])
def test_min_valid_seconds_is_checked_before_any_request(transport, value):
    with pytest.raises(ValueError):
        connection("google", min_valid_seconds=value)
    assert transport.calls == []


def test_off_a_machine_it_says_so(transport, monkeypatch):
    monkeypatch.delenv("AETHERFY_API_KEY")
    with pytest.raises(NotRunningOnAgent):
        connection("google")
    assert transport.calls == []
