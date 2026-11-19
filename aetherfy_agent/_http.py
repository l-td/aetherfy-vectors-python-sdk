"""
The helper's HTTP layer: ``urllib`` from the standard library, nothing else.

WHY NOT ``requests``, which this distribution already depends on. Nothing
here needs a session, a connection pool, a retry policy or an adapter — two
requests, both one-shot — so the import would buy nothing it does not
already cost. This is a preference, not an isolation claim: importing this
module pulls in ``aetherfy_vectors`` for the exception base, and that does
import ``requests``.

EVERY REQUEST SETS AN EXPLICIT User-Agent. Python's urllib otherwise sends
``Python-urllib/3.x``, a signature the edge's bot protection blocks outright —
producing a 403 that reads exactly like an auth failure. The docs make the same
point in the hand-rolled examples; the helper is where it stops being the
customer's problem.
"""

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, cast

from aetherfy_vectors.utils import parse_retry_after

from .exceptions import AgentTransportError
from .models import PeerResponse

#: Sent on every request this module makes. Version-stamped so a platform-side
#: log can tell which helper release a run was built against.
USER_AGENT = "aetherfy-agent-python/{version}"

DEFAULT_TIMEOUT = 30.0


def user_agent(version: str) -> str:
    return USER_AGENT.format(version=version)


def _read_json(stream: Any) -> Any:
    """
    Never raises. A body that cannot be read or parsed is reported as ``None``
    and the STATUS carries the meaning — an edge error page is not JSON, and a
    truncated stream is not a reason to lose a status the server did send.
    """
    try:
        raw = stream.read()
    except Exception:
        return None
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def _retry_after_of(response: Any) -> Optional[float]:
    """The ``Retry-After`` header of a response, in seconds, or None."""
    headers = getattr(response, "headers", None)
    return parse_retry_after(headers.get("Retry-After") if headers else None)


#: The control plane's 503s worth one more try after their Retry-After. RESOURCE_BUSY: the request was refused before
#: anything ran (its first unit found no database slot), so ANY request may be sent again, a create or a delete
#: included. REQUEST_TIMED_OUT and SERVICE_UNAVAILABLE can come PARTWAY through a request (a later unit timed out, or
#: the archive storage an upload or a rebuild needs was unreachable), so only a call whose repeat gives the same
#: result (`repeatable`) is sent again.
RESOURCE_BUSY = "RESOURCE_BUSY"
REQUEST_TIMED_OUT = "REQUEST_TIMED_OUT"
SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
#: The 503s that may have partly run: repeated only for a repeatable call.
_PARTWAY = frozenset({REQUEST_TIMED_OUT, SERVICE_UNAVAILABLE})
#: The longest Retry-After the helper waits out itself: the request's own timeout. A longer one is returned to the
#: caller with the 503, as any other status is.
MAX_RETRY_AFTER_WAITED = DEFAULT_TIMEOUT

#: How a Retry-After is waited out. A name of its own so a test records the wait instead of taking it.
_sleep = time.sleep


#: Which of the control plane's 503s request_json waits out and repeats: BY THE RULE (the default: RESOURCE_BUSY for
#: any call, REQUEST_TIMED_OUT and SERVICE_UNAVAILABLE for a repeatable one), RESOURCE_BUSY alone, or NONE (returned
#: as they came). The peer lookup chooses: with a last known address it has an answer to serve at once instead.
REFUSALS_BY_THE_RULE = "by-the-rule"
REFUSALS_RESOURCE_BUSY_ONLY = "resource-busy-only"
REFUSALS_NONE = "none"


def _worth_one_more_try(
    status: int,
    body: Any,
    retry_after: Optional[float],
    *,
    repeatable: bool,
    refusals: str,
) -> bool:
    """A 503 the helper repeats after its Retry-After, only with a Retry-After it may wait out: RESOURCE_BUSY for
    any call (nothing ran), REQUEST_TIMED_OUT and SERVICE_UNAVAILABLE for a repeatable one (they may have partly
    run), as `refusals` allows."""
    detail = body.get("detail") if isinstance(body, dict) else None
    code = detail.get("code") if isinstance(detail, dict) else None
    if refusals == REFUSALS_NONE:
        retryable = False
    elif code == RESOURCE_BUSY:
        retryable = True
    else:
        retryable = refusals == REFUSALS_BY_THE_RULE and repeatable and code in _PARTWAY
    return (
        status == 503
        and retryable
        and retry_after is not None
        and 0 <= retry_after <= MAX_RETRY_AFTER_WAITED
    )


def never_delivered(exc: BaseException) -> bool:
    """Did this transport failure happen BEFORE the request could reach anyone?

    Only two failures prove it: the name did not resolve, and the connection
    was refused. Anything else (a timeout, a reset, a broken pipe) can come
    after the far side received the request and acted on it. ONE answer for
    every request this module sends: :func:`request_json` to the control plane
    and :func:`request_peer` to a peer read it, and so does the vectors client's
    own retry policy (``aetherfy_vectors/retry_policy.py``, "refused before
    anything happened").
    """
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    return isinstance(reason, (socket.gaierror, ConnectionRefusedError))


def request_json(
    method: str,
    url: str,
    *,
    api_key: str,
    ua: str,
    repeatable: bool,
    body: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    refusals: str = REFUSALS_BY_THE_RULE,
) -> Tuple[int, Any, Optional[float]]:
    """
    Send one JSON request and return ``(status_code, parsed_body, retry_after)``.

    ``retry_after`` is the seconds the response's ``Retry-After`` header asked
    to wait (a number or an HTTP date), or None when it sent none or an
    unreadable one. Nothing here waits it: see below.

    An HTTP error status is a RETURN, not a raise: the caller maps 413 and 429
    onto its own types and needs the body to do it. Only a request that never
    got an answer raises, as :class:`AgentTransportError`.

    AT MOST ONE RETRY, and only where a repeat cannot do the work twice:

    - a request that NEVER LEFT (:func:`never_delivered`: the name did not
      resolve, the connection was refused) is retried, whatever the call;
    - a 503 with a ``Retry-After`` is retried after that ``Retry-After``:
      ``RESOURCE_BUSY`` whatever the call (it was refused before anything
      ran), ``REQUEST_TIMED_OUT`` and ``SERVICE_UNAVAILABLE`` only when the
      call is ``repeatable`` (either can come partway through a request). It
      waits at most :data:`MAX_RETRY_AFTER_WAITED`; a longer one is returned
      as it came;
    - a request that may have been DELIVERED (a timeout or a reset after it was
      sent) is retried only when the caller says the call is ``repeatable``:
      its repeat gives the same result (a read, a cancel, a spawn or a message
      that carries its own id, which the control plane stores once). A call
      that is not repeatable raises at once, saying the request may have been
      applied, because a blind repeat could do the work a second time.

    ``repeatable`` has NO default: every caller decides it, in the open.

    No other HTTP status is retried here: a 4xx repeated is a 4xx. The caller
    decides whether a 429 is worth waiting on, and ``retry_after`` is what it
    decides with: a retry before that time is refused again.
    """
    data = None
    headers = {
        "Authorization": "Bearer {0}".format(api_key),
        "User-Agent": ua,
        "Accept": "application/json",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)

    last_error: Optional[BaseException] = None
    attempts = 0
    while attempts < 2:
        attempts += 1
        # ONLY the call itself is inside the retry. READING THE BODY IS
        # DELIBERATELY OUTSIDE IT: a body that fails mid-stream arrives AFTER
        # the server answered, and the work is done. `_read_json` swallows that
        # failure into a None body and the status stands.
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:  # a status, not a transport failure
            status, body, retry_after = exc.code, _read_json(exc), _retry_after_of(exc)
            if attempts < 2 and _worth_one_more_try(
                status, body, retry_after, repeatable=repeatable, refusals=refusals
            ):
                # The rule refused a missing Retry-After: here it is a number.
                _sleep(cast(float, retry_after))
                continue
            return status, body, retry_after
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            if never_delivered(exc) or repeatable:
                continue
            raise AgentTransportError(
                "{0} {1} got no answer and may have been applied: {2}".format(
                    method, url, exc
                )
            )
        with response:
            return (
                response.getcode(),
                _read_json(response),
                _retry_after_of(response),
            )

    raise AgentTransportError(
        "{0} {1} did not reach the Aetherfy control plane after "
        "{2} attempt(s): {3}".format(method, url, attempts, last_error)
    )


class _DoNotFollow(urllib.request.HTTPRedirectHandler):
    """
    Answer a redirect with the redirect.

    ``urllib`` otherwise follows it: a 301 or 302 on a POST silently becomes a
    GET, a 307 or 308 on a POST is refused with an error of its own, and the
    caller's headers are replayed to whatever host ``Location`` names. A peer's
    3xx is the peer's answer, like any other status, and what to do about it is
    the caller's to say. ``None`` is how a handler declines to follow; the 3xx
    then surfaces as an ``HTTPError``, which :func:`request_peer` returns.
    """

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


#: What peer requests go through. ``urlopen`` cannot be told not to follow a
#: redirect; an opener built without that handler can.
_PEER_OPENER = urllib.request.build_opener(_DoNotFollow)


@dataclass(frozen=True)
class PeerAttempt:
    """
    One request to a peer: either its ``response``, or the ``failure`` that
    kept any response from coming.

    ``failure`` is set only when the request PROVABLY NEVER REACHED A PEER: the
    name did not resolve, or the connection was refused. That is the one outcome
    ``call_agent`` answers by looking the peer up again, because repeating the
    request cannot run it twice. A reset or a timeout is NOT that: either can
    come after the peer acted, so :func:`request_peer` raises
    :class:`AgentTransportError` for it, as it does for a body that fails
    mid-stream. A VALUE, NOT AN EXCEPTION, for the retryable case, so the
    helper's exception hierarchy stays what a caller can catch.
    """

    response: Optional[PeerResponse] = None
    failure: Optional[str] = None


def encode_json(value: Any) -> bytes:
    """A request body's JSON, as bytes. Here because ``call_agent`` has a
    parameter of its own called ``json`` that hides the module."""
    return json.dumps(value).encode("utf-8")


def request_peer(
    method: str,
    url: str,
    *,
    ua: str,
    body: Optional[bytes] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> PeerAttempt:
    """
    Send one request to a PEER and return whatever it answered.

    THE AETHERFY KEY IS NOT A PARAMETER, and that is the point: a peer is
    another service, and a bearer token for the whole account sent to it would
    hand the account to whatever that service logs. The only headers are the
    explicit ``User-Agent`` this module always sets and the caller's own.

    No retry here. Where a retry makes sense is the caller's to say: it needs
    the peer looked up again first, and repeating the same address would only
    repeat the failure.

    An HTTP status of any kind is a RETURN, a 3xx included: it is not followed.
    Only a request that never reached a peer (the name did not resolve, the
    connection was refused) comes back with a ``failure`` and no ``response``;
    a reset, a timeout, or a body that cannot be read after the status arrived
    raises a plain :class:`AgentTransportError`, and is not retried, because
    the peer may have acted on the request.
    """
    request_headers = {"User-Agent": ua}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url, data=body, headers=request_headers, method=method
    )

    try:
        response = _PEER_OPENER.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:  # a status, not a transport failure
        return _peer_response(exc.code, exc.headers, exc, method, url)
    except (urllib.error.URLError, OSError) as exc:
        if never_delivered(exc):
            return PeerAttempt(failure="{0} {1}: {2}".format(method, url, exc))
        raise AgentTransportError(
            "{0} {1} got no answer and may or may not have run: {2}".format(
                method, url, exc
            )
        )
    with response:
        return _peer_response(
            response.getcode(), response.headers, response, method, url
        )


def _peer_response(
    status: int, headers: Any, stream: Any, method: str, url: str
) -> PeerAttempt:
    try:
        raw = stream.read()
    except Exception as exc:
        raise AgentTransportError(
            "{0} {1} answered {2} but its body could not be read: {3}".format(
                method, url, status, exc
            )
        )
    return PeerAttempt(
        response=PeerResponse(
            status=status,
            headers={name.lower(): value for name, value in (headers or {}).items()},
            text=raw.decode("utf-8", errors="replace"),
        )
    )
