"""
Helper functions and utilities for Aetherfy Vectors SDK.

Provides common functionality for request handling, response processing,
and data validation across the SDK.
"""

import json
import random
import re
import time
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional, Union, Callable

from .exceptions import ValidationError, AetherfyVectorsException


def server_fields(error: Optional[BaseException]) -> Dict[str, Any]:
    """What an exception built from a server's answer must carry on.

    When code catches a server error and raises another exception in its
    place (a more specific class, a clearer message), the new one is built
    from the same answer, so it carries the same ``error_code``,
    ``status_code``, ``request_id`` and ``details``: ``raise
    SchemaNotFoundError(name, **server_fields(e))``. A branch on ``code``
    must see the server's code whichever class it arrives as. Anything that
    is not an SDK error (or None) carries nothing.
    """
    if not isinstance(error, AetherfyVectorsException):
        return {}
    return {
        "request_id": error.request_id,
        "status_code": error.status_code,
        "details": error.details,
        "error_code": error.error_code,
    }


def validate_vector(
    vector: List[float], expected_dimension: Optional[int] = None
) -> None:
    """Validate a vector's format and dimensions.

    Args:
        vector: The vector to validate.
        expected_dimension: Expected vector dimension (optional).

    Raises:
        ValidationError: If vector is invalid.
    """
    if not isinstance(vector, list):
        raise ValidationError("Vector must be a list of floats")

    if not vector:
        raise ValidationError("Vector cannot be empty")

    if not all(isinstance(x, (int, float)) for x in vector):
        raise ValidationError("Vector must contain only numeric values")

    if expected_dimension is not None and len(vector) != expected_dimension:
        raise ValidationError(
            f"Vector dimension mismatch: expected {expected_dimension}, got {len(vector)}"
        )


def validate_collection_name(collection_name: str) -> None:
    """Validate collection name format.

    Args:
        collection_name: The collection name to validate.

    Raises:
        ValidationError: If collection name is invalid.
    """
    if not isinstance(collection_name, str):
        raise ValidationError("Collection name must be a string")

    if not collection_name.strip():
        raise ValidationError("Collection name cannot be empty")

    if len(collection_name) > 255:
        raise ValidationError("Collection name must be 255 characters or less")

    # Check for invalid characters (basic validation)
    if any(
        char in collection_name
        for char in ["/", "\\", "?", "%", "*", ":", "|", '"', "<", ">"]
    ):
        raise ValidationError("Collection name contains invalid characters")


# Point-id validation — mirrors the server's ingress rule (vectordb
# utils/pointIds.js, 400 INVALID_POINT_ID). The UUID core after any
# urn:uuid: / brace wrapper is stripped: canonical hyphenated or simple
# (bare 32 hex), case-insensitive. Explicit regexes rather than
# uuid.UUID(), which accepts MORE than Qdrant's four forms (its parser
# strips braces/urn:/hyphens loosely, so e.g. misplaced hyphens pass).
_UUID_CANONICAL_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_UUID_SIMPLE_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)
_URN_UUID_PREFIX = "urn:uuid:"

# The server's integer bound (JS Number.MAX_SAFE_INTEGER = 2^53 − 1),
# NOT a Python limitation — see validate_point_id's docstring.
_MAX_POINT_ID_INT = 2**53 - 1


def _is_uuid_point_id(value: str) -> bool:
    """True iff ``value`` is any of Qdrant's four accepted UUID string forms:
    canonical (8-4-4-4-12), simple (32 hex), braced ({…}), URN (urn:uuid:…).
    """
    core = value
    # The urn:uuid: prefix and { } braces are mutually exclusive wrappers in
    # the forms Qdrant accepts — strip whichever is present, then hex-check
    # the core.
    if core[: len(_URN_UUID_PREFIX)].lower() == _URN_UUID_PREFIX:
        core = core[len(_URN_UUID_PREFIX) :]
    elif len(core) >= 2 and core[0] == "{" and core[-1] == "}":
        core = core[1:-1]
    return bool(_UUID_CANONICAL_RE.match(core) or _UUID_SIMPLE_RE.match(core))


def validate_point_id(point_id: Union[str, int]) -> None:
    """Validate a point ID against the server's ingress rule.

    Valid point ids are exactly two shapes:

    - An unsigned integer no larger than 2**53 − 1. The bound is the
      SERVER's, not a Python limitation: Python ints are
      arbitrary-precision, but the server's Node parse layer decodes JSON
      numbers as IEEE-754 doubles, which represent integers exactly only
      up to 2^53 − 1 — an id past that would be silently mangled before
      storage, so the server rejects it (an id it accepts is an id it can
      round-trip intact). Enforcing the same bound here mirrors the wire
      truth. Callers needing bigger stable ids use a UUID.
    - A UUID string in any of the four forms Qdrant accepts: canonical
      (8-4-4-4-12), simple (32 hex, no hyphens), braced ({…}), or URN
      (urn:uuid:…), case-insensitive.

    Rejecting anything else does not change which ids WORK — the server
    already responds 400 INVALID_POINT_ID to them; this raises the same
    rejection client-side, before the request is sent.

    Args:
        point_id: The point ID to validate.

    Raises:
        ValidationError: If the point ID is invalid — same wording as the
            server's 400 INVALID_POINT_ID response, so client-side and
            server-side rejections read the same.
    """
    # bool is an int subclass in Python, but JSON true/false is not a
    # valid point id — reject it before the int branch would accept it.
    if isinstance(point_id, int) and not isinstance(point_id, bool):
        if 0 <= point_id <= _MAX_POINT_ID_INT:
            return
    elif isinstance(point_id, str):
        if _is_uuid_point_id(point_id):
            return
    raise ValidationError(
        f"Point ID '{point_id}' is invalid — use an unsigned integer "
        "or a UUID string."
    )


# Filter clauses, in the order the request body carries them.
#
# The API judges every key of a filter: an unknown one is a 400
# INVALID_FILTER naming its path and the nearest valid key, which becomes
# `InvalidFilterError`. The SDK therefore keeps no clause vocabulary of its
# own to check against (a local list can only lag the API: it refused
# `min_should` for as long as it existed), and a plain dict is forwarded as
# written. The names below fix only the ORDER of the keys the SDK knows.
#
# Insertion order IS the emitted order. Server cache keys are derived from
# the request body BYTES, so a fixed order means two callers who wrote the
# same clauses in different orders share one cache entry instead of two.
_FILTER_CLAUSES = ("must", "must_not", "should", "min_should")


def serialize_filter(filter_obj: Any, method_name: str) -> Optional[Dict[str, Any]]:
    """Normalize a filter (``Filter`` or plain dict) into the request body form.

    Args:
        filter_obj: A ``Filter`` instance, a plain dict of clauses, or None.
        method_name: Calling method, used in the error message.

    Returns:
        The body's ``filter`` value, or None when nothing was supplied (so
        callers can skip the key entirely and leave a filterless body's
        bytes unchanged).

    Raises:
        ValidationError: If ``filter_obj`` is neither a ``Filter`` nor a dict.
            The keys of a dict are not judged here: the API answers an
            unknown one with ``InvalidFilterError``.
    """
    # Import here: models imports nothing from utils today, and a
    # module-level import would create a cycle if that ever changes.
    from .models import Filter

    if filter_obj is None:
        return None

    if isinstance(filter_obj, Filter):
        return filter_obj.to_dict()

    if not isinstance(filter_obj, dict):
        raise ValidationError(
            f"{method_name}: filter must be a Filter or a dict with "
            f"must / must_not / should / min_should clauses, got "
            f"{type(filter_obj).__name__}."
        )

    known = {k: filter_obj[k] for k in _FILTER_CLAUSES if filter_obj.get(k) is not None}
    others = {
        k: v
        for k, v in filter_obj.items()
        if k not in _FILTER_CLAUSES and v is not None
    }
    return {**known, **others}


def build_api_url(base_url: str, endpoint: str) -> str:
    """Build a fully-qualified API URL from a base host and an endpoint path.

    Args:
        base_url: Base URL with optional trailing slash.
        endpoint: Endpoint path, with or without a leading slash.

    Returns:
        The joined URL.
    """
    base = base_url.rstrip("/")
    path = endpoint.lstrip("/")
    return f"{base}/api/v1/{path}"


def _is_count(value: Any) -> bool:
    """A number the server counted with: an int, and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_allowed(value: Any) -> bool:
    """A plan limit's ``allowed``: a number, ``False``, or None for no cap."""
    return _is_count(value) or value is False or value is None


def _text(value: Any) -> Optional[str]:
    """A string field of an error body, or None when absent or not a string."""
    return value if isinstance(value, str) else None


def parse_retry_after(value: Any, now: Optional[float] = None) -> Optional[float]:
    """Seconds a ``Retry-After`` header asks to wait, or None when unreadable.

    The header is either a number of seconds or an HTTP date. Anything else
    (absent, empty, negative, not a string) is None, and the caller falls back
    to its own backoff. ``now`` (epoch seconds) is for tests.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            return None
        seconds = when.timestamp() - (time.time() if now is None else now)
        return max(seconds, 0.0)
    if seconds != seconds or seconds < 0 or seconds == float("inf"):
        return None
    return seconds


def parse_error_response(
    response_data: Any, status_code: int, headers: Any = None
) -> AetherfyVectorsException:
    """Parse error response from API and return appropriate exception.

    Args:
        response_data: The error response body. Normally a dict in one of
            the documented backend shapes (``{"error": {...}}`` or flat
            ``{"message": "..."}``), but can be a bare string, list, None,
            or any other JSON-decodable value when an upstream returns an
            unstructured body (a static 502 page from a CDN, a bare
            ``"Not Found"`` string from a misconfigured route, etc.).
            The function never raises on shape — it coerces non-dict
            input into a synthetic ``{"message": ...}`` so the rest of
            the mapping logic is uniform.
        status_code: HTTP status code.
        headers: The response headers, when there are any: a 5xx refusal
            carries the ``Retry-After`` it names.

    Returns:
        Appropriate exception instance.
    """
    from .exceptions import (
        AuthenticationError,
        RateLimitExceededError,
        ServiceUnavailableError,
        CollectionNotFoundError,
        PointNotFoundError,
        ValidationError,
        RequestTimeoutError,
    )

    # Defensive coercion: a non-dict body would crash response_data.get(...)
    # below. Coerce to a synthetic {"message": ...} so every downstream
    # branch handles a uniform shape. This is the parity the JS SDK gets
    # for free via optional chaining (responseData?.error / ?.message).
    if not isinstance(response_data, dict):
        if response_data is None:
            response_data = {"message": f"HTTP {status_code}"}
        else:
            response_data = {"message": str(response_data)}

    # Canonical vectordb error envelope: {"error": {"code": "...", "message": "...", **extras}}.
    # The Python SDK only talks to vectordb (Node/Express); it does not
    # parse FastAPI's `detail` key. Per docs/REVIEW_FAQ.md section 56,
    # each consumer reads only from the surface(s) it talks to so its
    # parser stays simple and asserts the contract.
    if "error" in response_data and isinstance(response_data["error"], dict):
        error_obj = response_data["error"]
        message = error_obj.get("message", "Unknown error")
        error_code = error_obj.get("code")
        request_id = response_data.get("request_id")
        details = error_obj
    elif "error" in response_data and isinstance(response_data["error"], str):
        # Some backend routes return {"error": "<message>"} (string-shaped).
        # Treat the string as the message rather than letting it become
        # the details bag below.
        message = response_data["error"]
        error_code = response_data.get("error_code")
        request_id = response_data.get("request_id")
        details = {}
    else:
        # Handle flat format: {"message": "...", "error_code": "..."}
        message = response_data.get("message", "Unknown error")
        request_id = response_data.get("request_id")
        error_code = response_data.get("error_code")
        details = response_data.get("details", {})

    # Map status codes to exceptions. Every typed exception gets the
    # backend's stable `error_code` so SDK-level code (e.g.
    # client._extract_error_code) can read it back without poking at
    # the body-shape-specific details bag.
    if status_code == 401:
        return AuthenticationError(
            message,
            request_id=request_id,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )
    elif status_code == 429:
        if error_code == "STORAGE_LIMIT_EXCEEDED":
            from .exceptions import QuotaExceededError

            # The numbers are nested: error.details.{current_bytes, limit_bytes}.
            nested = details.get("details") if isinstance(details, dict) else None
            nested = nested if isinstance(nested, dict) else {}
            current = nested.get("current_bytes")
            allowed = nested.get("limit_bytes")
            return QuotaExceededError(
                message,
                "storage",
                current=current if _is_count(current) else None,
                allowed=allowed if _is_count(allowed) else None,
                request_id=request_id,
                status_code=status_code,
                details=details,
                error_code=error_code,
            )
        retry_after = details.get("retry_after") if isinstance(details, dict) else None
        return RateLimitExceededError(
            message,
            request_id=request_id,
            status_code=status_code,
            details=details,
            retry_after=retry_after,
            error_code=error_code,
        )
    elif status_code == 403:
        # By code first, then by status: a 403 is "the key is valid and this
        # is not allowed", never "the key is wrong" (that is the 401 above).
        d = details if isinstance(details, dict) else {}
        carried = dict(
            request_id=request_id,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )
        if error_code == "PLAN_LIMIT_EXCEEDED":
            from .exceptions import QuotaExceededError

            axis = d.get("limit")
            current = d.get("current")
            return QuotaExceededError(
                message,
                axis.removeprefix("max_") if isinstance(axis, str) else "plan",
                current=current if _is_count(current) else None,
                limit=axis if isinstance(axis, str) else None,
                allowed=d.get("allowed") if _is_allowed(d.get("allowed")) else None,
                upgrade_url=_text(d.get("upgrade_url")),
                **carried,
            )
        if error_code == "AGENT_KEY_WORKSPACE_FORBIDDEN":
            from .exceptions import AgentWorkspaceForbiddenError

            return AgentWorkspaceForbiddenError(
                _text(d.get("workspace")), message=None, **carried
            )
        if error_code == "AGENT_ACCESS_LEVEL_INSUFFICIENT":
            from .exceptions import AccessLevelInsufficientError

            return AccessLevelInsufficientError(
                message,
                workspace=_text(d.get("workspace")),
                level=_text(d.get("level")),
                required_level=_text(d.get("required_level")),
                **carried,
            )
        if error_code == "COLLECTION_PROTECTED":
            from .exceptions import CollectionProtectedError

            return CollectionProtectedError(
                message, collection_name=_text(d.get("collection_name")), **carried
            )
        if error_code == "AUTH_AGENT_KEY_OUT_OF_SCOPE":
            from .exceptions import AgentKeyOutOfScopeError

            return AgentKeyOutOfScopeError(
                message,
                agent_id=_text(d.get("agent_id")),
                required_scope=_text(d.get("required_scope")),
                **carried,
            )
        from .exceptions import PermissionDeniedError

        return PermissionDeniedError(message, **carried)
    elif status_code in [502, 503, 504]:
        return ServiceUnavailableError(
            message,
            retry_after=parse_retry_after(
                headers.get("Retry-After") if hasattr(headers, "get") else None
            ),
            request_id=request_id,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )
    elif status_code == 404:
        if error_code == "COLLECTION_NOT_FOUND":
            collection_name = details.get("collection_name", "unknown")
            return CollectionNotFoundError(
                collection_name,
                request_id=request_id,
                status_code=status_code,
                details=details,
                error_code=error_code,
            )
        elif error_code == "POINT_NOT_FOUND":
            point_id = details.get("point_id", "unknown")
            collection_name = details.get("collection_name", "unknown")
            return PointNotFoundError(
                point_id,
                collection_name,
                request_id=request_id,
                status_code=status_code,
                details=details,
                error_code=error_code,
            )
        elif error_code == "SCHEMA_NOT_DEFINED":
            from .exceptions import SchemaNotFoundError

            d = details if isinstance(details, dict) else {}
            return SchemaNotFoundError(
                d.get("collection_name", "unknown"),
                request_id=request_id,
                status_code=status_code,
                details=details,
                error_code=error_code,
            )
        elif error_code == "WORKSPACE_NOT_FOUND":
            from .exceptions import WorkspaceNotFoundError

            d = details if isinstance(details, dict) else {}
            return WorkspaceNotFoundError(
                d.get("workspace"),
                d.get("workspace_id"),
                message=message,
                request_id=request_id,
                status_code=status_code,
                details=details,
                error_code=error_code,
            )
        # A 404 whose code this SDK does not know: still a typed not-found.
        from .exceptions import NotFoundError

        return NotFoundError(
            message,
            request_id=request_id,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )
    elif status_code == 400:
        d = details if isinstance(details, dict) else {}
        if error_code == "INVALID_FILTER":
            from .exceptions import InvalidFilterError

            return InvalidFilterError(
                message,
                field=_text(d.get("field")),
                key=_text(d.get("key")),
                suggestion=_text(d.get("suggestion")),
                request_id=request_id,
                status_code=status_code,
                details=details,
                error_code=error_code,
            )
        if error_code == "INVALID_REQUEST_FIELD":
            from .exceptions import InvalidRequestFieldError

            return InvalidRequestFieldError(
                message,
                field=_text(d.get("field")),
                key=_text(d.get("key")),
                suggestion=_text(d.get("suggestion")),
                request_id=request_id,
                status_code=status_code,
                details=details,
                error_code=error_code,
            )
        return ValidationError(
            message,
            request_id=request_id,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )
    elif status_code == 409:
        from .exceptions import (
            CollectionInUseError,
            CollectionInOtherRegionError,
            ConflictError,
        )

        if error_code == "COLLECTION_IN_USE":
            collection_name = (
                details.get("collection_name", "unknown")
                if isinstance(details, dict)
                else "unknown"
            )
            agents = details.get("agents", []) if isinstance(details, dict) else []
            return CollectionInUseError(
                collection_name,
                agents,
                request_id=request_id,
                status_code=status_code,
                details=details,
                error_code=error_code,
            )
        if error_code == "COLLECTION_EXISTS_IN_OTHER_REGION":
            d = details if isinstance(details, dict) else {}
            return CollectionInOtherRegionError(
                d.get("collection_name", "unknown"),
                d.get("existing_regions", []) or [],
                d.get("requesting_region", ""),
                message=message,
                request_id=request_id,
                status_code=status_code,
                details=details,
                error_code=error_code,
            )
        return ConflictError(
            message,
            request_id=request_id,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )
    elif status_code == 422:
        # Unprocessable: VALIDATION_ERROR, COLLECTION_SDK_MANAGED, ...
        return ValidationError(
            message,
            request_id=request_id,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )
    elif status_code == 412:
        # Schema version mismatch - return ValidationError to trigger cache clear
        return ValidationError(
            message,
            request_id=request_id,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )
    elif status_code == 408:
        return RequestTimeoutError(
            message,
            request_id=request_id,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )

    # Default to base exception. Pass error_code so SDK-level code can
    # branch on the backend's stable identifier without poking at the
    # body-shape-specific details bag.
    return AetherfyVectorsException(
        message,
        request_id=request_id,
        status_code=status_code,
        details=details,
        error_code=error_code,
    )


def format_points_for_upsert(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format points data for upsert operation.

    Args:
        points: List of point dictionaries.

    Returns:
        Formatted points data.

    Raises:
        ValidationError: If points data is invalid.
    """
    if not isinstance(points, list):
        raise ValidationError("Points must be a list")

    if not points:
        raise ValidationError("Points list cannot be empty")

    formatted_points = []
    for i, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValidationError(f"Point at index {i} must be a dictionary")

        if "id" not in point:
            raise ValidationError(f"Point at index {i} must have an 'id' field")

        if "vector" not in point:
            raise ValidationError(f"Point at index {i} must have a 'vector' field")

        validate_point_id(point["id"])
        validate_vector(point["vector"])

        formatted_point = {"id": point["id"], "vector": point["vector"]}

        if "payload" in point and point["payload"] is not None:
            formatted_point["payload"] = point["payload"]

        formatted_points.append(formatted_point)

    return formatted_points


def retry_with_backoff(
    func: Callable,
    retry_class: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    deadline: Optional[float] = None,
):
    """Retry ``func`` with exponential backoff, under the retry rule.

    Which failures are retried is decided by retry_policy for the
    operation's class, never by the error's type alone: a refusal before
    anything ran is retried for any class, an ambiguous failure only for
    ``retry_policy.REPEATABLE``, a 4xx never.

    Args:
        func: Function to retry.
        retry_class: ``retry_policy.REPEATABLE`` or ``retry_policy.ONCE``:
            whether repeating the operation gives the same result.
        max_retries: Maximum number of retries.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay cap in seconds. It also caps a wait the
            server asked for with ``Retry-After`` on a 503.
        deadline: Optional ``time.monotonic()`` instant after which no retry
            may start. A backoff sleep that would end at or past it is not
            taken: the last error is raised instead. ``func`` owns bounding
            its own attempt by the same deadline.

    Returns:
        Function result.

    Raises:
        Last exception if all retries fail.
    """
    from .exceptions import ServiceUnavailableError
    from .retry_policy import should_retry_class

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e

            should_retry = should_retry_class(retry_class, e)

            if should_retry and attempt < max_retries:
                asked = getattr(e, "retry_after", None)
                if isinstance(e, ServiceUnavailableError) and asked is not None:
                    # The server said how long to wait: wait that, no jitter,
                    # but never past the cap. The deadline check below still
                    # applies, so a long ask cannot outlive the call.
                    delay = min(asked, max_delay)
                else:
                    delay = min(base_delay * (2**attempt), max_delay)
                    # Add jitter (50-100% of delay)
                    delay = delay * (0.5 + 0.5 * random.random())
                if deadline is not None and time.monotonic() + delay >= deadline:
                    break
                time.sleep(delay)
            else:
                break

    if last_exception is not None:
        raise last_exception
    else:
        raise RuntimeError("Maximum retries exceeded")


def sanitize_for_logging(data: Any) -> Any:
    """Sanitize data for safe logging (remove sensitive information).

    Args:
        data: Data to sanitize.

    Returns:
        Sanitized data.
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if any(
                sensitive in key.lower()
                for sensitive in ["key", "token", "password", "secret"]
            ):
                sanitized[key] = "***"
            else:
                sanitized[key] = sanitize_for_logging(value)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_for_logging(item) for item in data]
    elif isinstance(data, str) and len(data) > 100:
        # Truncate very long strings
        return data[:100] + "..."
    else:
        return data
