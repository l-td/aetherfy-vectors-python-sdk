"""
Connections: a fresh OAuth access token for a connected provider (Google, Slack,
Notion, Microsoft 365, HubSpot, Atlassian), by the connection's id.

The platform runs the OAuth sign-in on the dashboard and keeps the grant; this
agent never sees a client secret or a refresh token. It asks the control plane
for an access token when it needs one:

    GET  {AETHERFY_API_URL}/connections/usable          the ids this agent may use
    POST {AETHERFY_API_URL}/connections/{id}/token      {"min_valid_seconds": N}
    Authorization: Bearer {AETHERFY_API_KEY}

and gets one valid for at least N seconds (never less than a minute), refreshed
first when it had to be. https://docs.aetherfy.com/agents/connections is the
same contract, hand-rolled.

ONLY THIS MACHINE'S OWN KEY WORKS. The route answers the deployment-bound
``AETHERFY_API_KEY`` the platform injects into each agent machine, and refuses
an account key (:class:`~.exceptions.ConnectionAccessDenied`). So this call
works on an agent and nowhere else.

A PER-ID IN-PROCESS CACHE. A token is reused until it would have less than
``max(min_valid_seconds, 60)`` seconds left — the same margin the control plane
itself applies — so a loop calling ``connection(google.id)`` per item costs one
request per token lifetime, not one per item. The cache is per process: a
fan-out's worker processes each keep their own. A token with no expiry
(Notion's) is asked for again after ``_NO_EXPIRY_RECHECK_SECONDS``, so a
disconnect or a reconnect on the dashboard reaches a long-running agent.
"""

import threading
import uuid
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Any, Dict, List, Optional, Tuple

from . import _http
from .exceptions import (
    CONNECTION_GRANT_NOT_ENABLED,
    CONNECTION_NEEDS_REAUTH,
    CONNECTION_NOT_FOUND,
    CONNECTION_PROVIDER_UNAVAILABLE,
    CONNECTION_REQUIRES_AGENT_KEY,
    ConnectionAccessDenied,
    ConnectionGrantNotEnabled,
    ConnectionNeedsReauth,
    ConnectionNotFound,
    ConnectionTokenError,
    ConnectionUnavailable,
)
from .models import ConnectionInfo, ConnectionToken

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

#: connection id -> (token, monotonic() when it was fetched)
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


def list_connections() -> List[ConnectionInfo]:
    """
    The connections this agent's key may ask a token for: its own, its
    workspace's, and those of a granted workspace whose grant has the connections
    switch on. Each has an ``id`` to give :func:`connection`.

    :raises NotRunningOnAgent: ``AETHERFY_API_URL`` or ``AETHERFY_API_KEY`` is unset.
    :raises ConnectionAccessDenied: the key is not this agent's own.
    :raises ConnectionTokenError: any other refusal — read ``error_code``.
    :raises AgentTransportError: the request never reached the control plane.
    """
    from . import _require, _user_agent

    api_url = _require("AETHERFY_API_URL", "the Aetherfy API the connections come from")
    api_key = _require("AETHERFY_API_KEY", "the key the connections are listed for")
    status, body = _http.request_json(
        "GET",
        "{0}/connections/usable".format(api_url.rstrip("/")),
        api_key=api_key,
        ua=_user_agent(),
    )
    if status == 200 and isinstance(body, list):
        return [
            ConnectionInfo(
                id=str(item["id"]),
                name=str(item.get("name")),
                provider=str(item.get("provider")),
                scope=str(item.get("scope")),
                account_label=item.get("account_label"),
                status=str(item.get("status")),
                scopes=tuple(item.get("scopes") or ()),
            )
            for item in body
            if isinstance(item, dict) and item.get("id")
        ]
    raise _refusal(status, body, "The connection list was refused with status {0}.".format(status))


def _refusal(status: int, body: Any, fallback: str) -> ConnectionTokenError:
    detail = body.get("detail") if isinstance(body, dict) else None
    detail = detail if isinstance(detail, dict) else {}
    message = detail.get("message") or fallback
    code = detail.get("code")
    # The status AND the code select a type; an unrecognised pairing arrives as
    # ConnectionTokenError reporting exactly what came back.
    if status == 404 and code == CONNECTION_NOT_FOUND:
        return ConnectionNotFound(message, details=detail)
    if status == 403 and code == CONNECTION_GRANT_NOT_ENABLED:
        return ConnectionGrantNotEnabled(message, details=detail)
    if status == 409 and code == CONNECTION_NEEDS_REAUTH:
        return ConnectionNeedsReauth(message, details=detail)
    if status == 502 and code == CONNECTION_PROVIDER_UNAVAILABLE:
        return ConnectionUnavailable(message, details=detail)
    if status == 403 and code == CONNECTION_REQUIRES_AGENT_KEY:
        return ConnectionAccessDenied(message, details=detail)
    return ConnectionTokenError(
        message, status_code=status, error_code=code, details=detail
    )


def connection(
    connection_id: str, *, min_valid_seconds: int = MIN_VALID_SECONDS_DEFAULT
) -> ConnectionToken:
    """
    A fresh access token for the connection ``connection_id`` (an id from
    :func:`list_connections`).

    The agent may use its own connections, its workspace's, and those of a
    workspace it is granted whose grant has the connections switch on. The token
    stays valid for at least ``min_valid_seconds`` (0 to 3000, default 300), and
    never less than a minute.

    :raises ValueError: ``connection_id`` is not a UUID, or ``min_valid_seconds``
        is out of range.
    :raises NotRunningOnAgent: ``AETHERFY_API_URL`` or ``AETHERFY_API_KEY`` is unset.
    :raises ConnectionNotFound: ``404 CONNECTION_NOT_FOUND``: no such connection, or
        not one this agent may use (the control plane does not say which).
    :raises ConnectionGrantNotEnabled: ``403 CONNECTION_GRANT_NOT_ENABLED``: the
        connection is in a granted workspace whose grant does not cover connections.
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
    try:
        connection_id = str(uuid.UUID(str(connection_id)))
    except ValueError:
        raise ValueError(
            "connection_id must be a connection id (see aetherfy_agent.list_connections()), got {0!r}".format(
                connection_id
            )
        )

    with _cache_lock:
        cached = _cache.get(connection_id)
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
        "{0}/connections/{1}/token".format(api_url.rstrip("/"), connection_id),
        api_key=api_key,
        ua=_user_agent(),
        body={"min_valid_seconds": min_valid_seconds},
    )

    if status == 200 and isinstance(body, dict) and body.get("access_token"):
        token = ConnectionToken(
            id=str(body.get("id") or connection_id),
            access_token=str(body["access_token"]),
            token_type=str(body.get("token_type") or "Bearer"),
            expires_at=_parse_expiry(body.get("expires_at")),
            provider=str(body.get("provider")),
            name=str(body.get("name") or ""),
            account_label=body.get("account_label"),
            scopes=tuple(body.get("scopes") or ()),
        )
        with _cache_lock:
            _cache[connection_id] = (token, monotonic())
        return token

    error = _refusal(
        status,
        body,
        "The token for connection '{0}' was refused with status {1}.".format(
            connection_id, status
        ),
    )
    if isinstance(error, (ConnectionNeedsReauth, ConnectionNotFound)):
        with _cache_lock:
            _cache.pop(connection_id, None)
    raise error
