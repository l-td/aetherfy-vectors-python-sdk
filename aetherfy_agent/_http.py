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
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from .exceptions import AgentTransportError

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


def request_json(
    method: str,
    url: str,
    *,
    api_key: str,
    ua: str,
    body: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    retry_connection_errors: bool = True,
) -> Tuple[int, Any]:
    """
    Send one JSON request and return ``(status_code, parsed_body)``.

    An HTTP error status is a RETURN, not a raise: the caller maps 413 and 429
    onto its own types and needs the body to do it. Only a request that never
    got an answer raises, as :class:`AgentTransportError`.

    ONE retry, and only for a connection-level failure — a DNS blip or a reset
    socket on the way out, where nothing was recorded and repeating is safe. An
    HTTP status is never retried here: a 4xx repeated is a 4xx, and retrying a
    spawn that the control plane may already have recorded would create a
    second run. The caller decides whether a 429 is worth waiting on.
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

    attempts = 2 if retry_connection_errors else 1
    last_error = None
    for attempt in range(attempts):
        # ONLY the call itself is inside the retry. READING THE BODY IS
        # DELIBERATELY OUTSIDE IT: a body that fails mid-stream arrives AFTER
        # the server answered, and on a spawn the control plane has already
        # recorded the run — retrying there would create a second one.
        # `_read_json` swallows that failure into a None body and the status
        # stands.
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:  # a status, not a transport failure
            return exc.code, _read_json(exc)
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            continue
        with response:
            return response.getcode(), _read_json(response)

    raise AgentTransportError(
        "{0} {1} did not reach the Aetherfy control plane after "
        "{2} attempt(s): {3}".format(method, url, attempts, last_error)
    )
