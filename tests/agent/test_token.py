"""
`token()`: the exchange the control plane's route accepts, the cache in front of
it, and the refusals a caller branches on.

Pinned against aetherfy-control-plane `api/routes/agent_tokens.py`:
`POST {api_prefix}/agent-tokens` with `{audience, scopes?}`, answered 201 with
`{token, token_type, expires_at, audience, scopes}`, and the `{"detail": {...}}`
envelope for AGENT_TOKEN_AUDIENCE_UNKNOWN / _SCOPE_NOT_GRANTED /
AGENT_TOKENS_UNCONFIGURED / AGENT_TOKEN_REQUIRES_AGENT_KEY.
"""

from datetime import datetime, timedelta, timezone

import pytest

import aetherfy_agent
from aetherfy_agent import AgentToken, TokenError, token
from aetherfy_agent.exceptions import NotRunningOnAgent

AUDIENCE = "aetherfy-control-plane"


def _expiring_in(seconds):
    instant = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return instant.strftime("%Y-%m-%dT%H:%M:%SZ")


def _minted(expires_at, value="afyat_test_header.claims.sig"):
    return (
        201,
        {
            "token": value,
            "token_type": "Bearer",
            "expires_at": expires_at,
            "audience": AUDIENCE,
            "scopes": ["runs:read"],
        },
    )


@pytest.fixture(autouse=True)
def on_a_machine(monkeypatch):
    monkeypatch.setenv("AETHERFY_API_URL", "https://agents.aetherfy.com/api/v1/")
    monkeypatch.setenv("AETHERFY_API_KEY", "afy_test_key")
    aetherfy_agent._token_cache.clear()
    yield
    aetherfy_agent._token_cache.clear()


@pytest.fixture
def transport(monkeypatch):
    class Transport:
        def __init__(self):
            self.calls = []
            self.replies = []

        def __call__(self, method, url, *, api_key, ua, body=None, **kwargs):
            self.calls.append(
                {"method": method, "url": url, "api_key": api_key, "body": body}
            )
            return self.replies.pop(0)

    fake = Transport()
    monkeypatch.setattr(aetherfy_agent._http, "request_json", fake)
    return fake


def test_the_request_matches_the_route(transport):
    transport.replies.append(_minted(_expiring_in(600)))

    minted = token(AUDIENCE, scopes=["runs:spawn", "runs:read"])

    assert minted == AgentToken(
        token="afyat_test_header.claims.sig", expires_at=minted.expires_at
    )
    (call,) = transport.calls
    assert call["method"] == "POST"
    assert call["url"] == "https://agents.aetherfy.com/api/v1/agent-tokens"
    assert call["api_key"] == "afy_test_key"
    assert call["body"] == {"audience": AUDIENCE, "scopes": ["runs:read", "runs:spawn"]}


def test_omitted_scopes_are_not_sent(transport):
    """The route's default is every scope the key holds for the audience."""
    transport.replies.append(_minted(_expiring_in(600)))
    token(AUDIENCE)
    assert transport.calls[0]["body"] == {"audience": AUDIENCE}


def test_a_live_token_is_served_from_the_cache(transport):
    transport.replies.append(_minted(_expiring_in(600)))
    first = token(AUDIENCE, scopes=["runs:read"])
    second = token(AUDIENCE, scopes=["runs:read"])
    assert first is second
    assert len(transport.calls) == 1


def test_a_token_within_a_minute_of_expiry_is_replaced(transport):
    transport.replies.append(_minted(_expiring_in(59), value="afyat_test_old"))
    transport.replies.append(_minted(_expiring_in(600), value="afyat_test_new"))
    assert token(AUDIENCE).token == "afyat_test_old"
    assert token(AUDIENCE).token == "afyat_test_new"
    assert len(transport.calls) == 2


def test_the_cache_is_per_key_audience_and_scopes(transport, monkeypatch):
    """A task machine gets a new key for every run; the previous run's token
    dies with that run and must never be handed to the next one."""
    for value in ("a", "b", "c"):
        transport.replies.append(_minted(_expiring_in(600), value=value))
    assert token(AUDIENCE, scopes=["runs:read"]).token == "a"
    assert token(AUDIENCE, scopes=["runs:spawn"]).token == "b"
    monkeypatch.setenv("AETHERFY_API_KEY", "afy_test_next_run")
    assert token(AUDIENCE, scopes=["runs:read"]).token == "c"


def test_a_single_string_of_scopes_is_refused_before_any_request(transport):
    with pytest.raises(TypeError):
        token(AUDIENCE, scopes="runs:read")
    assert transport.calls == []


def test_an_unknown_option_is_refused():
    with pytest.raises(TypeError):
        token(AUDIENCE, ttl=60)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "status, code",
    [
        (400, "AGENT_TOKEN_AUDIENCE_UNKNOWN"),
        (403, "AGENT_TOKEN_SCOPE_NOT_GRANTED"),
        (403, "AGENT_TOKEN_REQUIRES_AGENT_KEY"),
        (503, "AGENT_TOKENS_UNCONFIGURED"),
    ],
)
def test_a_refusal_carries_the_platform_code(transport, status, code):
    transport.replies.append(
        (status, {"detail": {"code": code, "message": "no", "not_granted": ["x"]}})
    )
    with pytest.raises(TokenError) as exc:
        token(AUDIENCE)
    assert exc.value.status_code == status
    assert exc.value.error_code == code
    assert exc.value.details["not_granted"] == ["x"]


def test_a_refusal_is_not_cached(transport):
    transport.replies.append((503, {"detail": {"code": "AGENT_TOKENS_UNCONFIGURED"}}))
    transport.replies.append(_minted(_expiring_in(600)))
    with pytest.raises(TokenError):
        token(AUDIENCE)
    assert token(AUDIENCE).token.startswith("afyat_test_")


def test_an_unreadable_expiry_is_an_error_not_a_forever_token(transport):
    transport.replies.append(_minted("soon"))
    with pytest.raises(TokenError):
        token(AUDIENCE)
    assert aetherfy_agent._token_cache == {}


def test_off_a_machine_it_says_which_variable_is_missing(monkeypatch):
    monkeypatch.delenv("AETHERFY_API_KEY")
    with pytest.raises(NotRunningOnAgent):
        token(AUDIENCE)
