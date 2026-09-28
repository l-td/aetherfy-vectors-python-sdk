"""
Connections: a fresh OAuth access token for Google, Slack or Notion, by name.

The platform runs the OAuth sign-in on the dashboard and keeps the grant; this
agent never sees a client secret or a refresh token. It asks the control plane
for an access token when it needs one:

    POST {AETHERFY_API_URL}/connections/{name}/token   {"min_valid_seconds": N}
    Authorization: Bearer {AETHERFY_API_KEY}

and gets one valid for at least N seconds (never less than a minute), refreshed
first when it had to be. https://docs.aetherfy.com/agents/connections is the
same contract, hand-rolled.

ONLY THIS MACHINE'S OWN KEY WORKS. The route answers the deployment-bound
``AETHERFY_API_KEY`` the platform injects into each agent machine, and refuses
an account key (:class:`~.exceptions.ConnectionAccessDenied`). So this call
works on an agent and nowhere else.

A PER-NAME IN-PROCESS CACHE. A token is reused until it would have less than
``max(min_valid_seconds, 60)`` seconds left — the same margin the control plane
itself applies — so a loop calling ``connection("google")`` per item costs one
request per token lifetime, not one per item. The cache is per process: a
fan-out's worker processes each keep their own. A token with no expiry
(Notion's) is asked for again after ``_NO_EXPIRY_RECHECK_SECONDS``, so a
disconnect or a reconnect on the dashboard reaches a long-running agent.
"""

import threading
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Dict, Optional, Tuple
from urllib.parse import quote

from . import _http
from .exceptions import (
    CONNECTION_NEEDS_REAUTH,
    CONNECTION_NOT_FOUND,
    CONNECTION_PROVIDER_UNAVAILABLE,
    CONNECTION_REQUIRES_AGENT_KEY,
    ConnectionAccessDenied,
    ConnectionNeedsReauth,
    ConnectionNotFound,
    ConnectionTokenError,
    ConnectionUnavailable,
)
from .models import ConnectionToken

#: The bound the control plane enforces on ``min_valid_seconds``. Checked here
#: too, so a bad argument costs no round trip.
MIN_VALID_SECONDS_MAX = 3000
MIN_VALID_SECONDS_DEFAULT = 300

#: The control plane never hands out a token with less than this left, and the
#: cache holds itself to the same floor.
_REFRESH_FLOOR_SECONDS = 60

#: How long a token with no expiry is reused before the control plane is asked
#: again. Without a bound it would outlive its own revocation.
_NO_EXPIRY_RECHECK_SECONDS = 300

#: name -> (token, monotonic() when it was fetched)
_cache: Dict[str, Tuple[ConnectionToken, float]] = {}
_cache_lock = threading.Lock()


def _fresh_enough(
    token: ConnectionToken, fetched_at: float, min_valid_seconds: int
) -> bool:
    if token.expires_at is None:
        return monotonic() - fetched_at < _NO_EXPIRY_RECHECK_SECONDS
    margin = timedelta(seconds=max(min_valid_seconds, _REFRESH_FLOOR_SECONDS))
    return token.expires_at - datetime.now(timezone.utc) > margin


def _parse_expiry(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    # fromisoformat before 3.11 does not read a trailing "Z".
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def connection(
    name: str, *, min_valid_seconds: int = MIN_VALID_SECONDS_DEFAULT
) -> ConnectionToken:
    """
    A fresh access token for the connection ``name``.

    The agent sees its own connections and its workspace's; on a name clash its
    own wins. The token stays valid for at least ``min_valid_seconds`` (0 to
    3000, default 300), and never less than a minute.

    :raises ValueError: ``min_valid_seconds`` is out of range.
    :raises NotRunningOnAgent: ``AETHERFY_API_URL`` or ``AETHERFY_API_KEY`` is unset.
    :raises ConnectionNotFound: no connection by that name here.
    :raises ConnectionNeedsReauth: the provider refused the grant; reconnect it.
    :raises ConnectionUnavailable: the provider did not answer; retry shortly.
    :raises ConnectionAccessDenied: the key is not this agent's own.
    :raises ConnectionTokenError: any other refusal — read ``error_code``.
    :raises AgentTransportError: the request never reached the control plane.
    """
    if isinstance(min_valid_seconds, bool) or not isinstance(min_valid_seconds, int):
        raise ValueError("min_valid_seconds must be an integer")
    if not 0 <= min_valid_seconds <= MIN_VALID_SECONDS_MAX:
        raise ValueError(
            "min_valid_seconds must be between 0 and {0}, got {1}".format(
                MIN_VALID_SECONDS_MAX, min_valid_seconds
            )
        )

    with _cache_lock:
        cached = _cache.get(name)
    if cached is not None and _fresh_enough(*cached, min_valid_seconds):
        return cached[0]

    # Imported here, not at module top: they live in the package's __init__,
    # which imports this module to re-export connection().
    from . import _require, _user_agent

    api_url = _require(
        "AETHERFY_API_URL", "the Aetherfy API a connection token comes from"
    )
    api_key = _require("AETHERFY_API_KEY", "the key a connection token is issued to")
    status, body = _http.request_json(
        "POST",
        "{0}/connections/{1}/token".format(api_url.rstrip("/"), quote(name, safe="")),
        api_key=api_key,
        ua=_user_agent(),
        body={"min_valid_seconds": min_valid_seconds},
    )

    if status == 200 and isinstance(body, dict) and body.get("access_token"):
        token = ConnectionToken(
            access_token=str(body["access_token"]),
            token_type=str(body.get("token_type") or "Bearer"),
            expires_at=_parse_expiry(body.get("expires_at")),
            provider=str(body.get("provider")),
            name=str(body.get("name") or name),
            account_label=body.get("account_label"),
            scopes=tuple(body.get("scopes") or ()),
        )
        with _cache_lock:
            _cache[name] = (token, monotonic())
        return token

    detail = body.get("detail") if isinstance(body, dict) else None
    detail = detail if isinstance(detail, dict) else {}
    message = detail.get(
        "message"
    ) or "The token for connection '{0}' was refused with status {1}.".format(
        name, status
    )
    code = detail.get("code")

    # The status AND the code select a type; an unrecognised pairing arrives as
    # ConnectionTokenError reporting exactly what came back.
    if status == 404 and code == CONNECTION_NOT_FOUND:
        raise ConnectionNotFound(message, details=detail)
    if status == 409 and code == CONNECTION_NEEDS_REAUTH:
        with _cache_lock:
            _cache.pop(name, None)
        raise ConnectionNeedsReauth(message, details=detail)
    if status == 502 and code == CONNECTION_PROVIDER_UNAVAILABLE:
        raise ConnectionUnavailable(message, details=detail)
    if status == 403 and code == CONNECTION_REQUIRES_AGENT_KEY:
        raise ConnectionAccessDenied(message, details=detail)
    raise ConnectionTokenError(
        message, status_code=status, error_code=code, details=detail
    )
