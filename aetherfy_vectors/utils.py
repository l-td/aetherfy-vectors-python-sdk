"""
Helper functions and utilities for Aetherfy Vectors SDK.

Provides common functionality for request handling, response processing,
and data validation across the SDK.
"""

import json
import random
import time
from typing import Any, Dict, List, Optional, Union, Callable
from urllib.parse import quote, urlparse

from .exceptions import ValidationError, AetherfyVectorsException


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


def validate_point_id(point_id: Union[str, int]) -> None:
    """Validate point ID format.

    Args:
        point_id: The point ID to validate.

    Raises:
        ValidationError: If point ID is invalid.
    """
    if not isinstance(point_id, (str, int)):
        raise ValidationError("Point ID must be a string or integer")

    if isinstance(point_id, str) and not point_id.strip():
        raise ValidationError("Point ID cannot be empty string")


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


def quote_collection_name(name: str) -> str:
    """URL-quote a collection name for safe use as a path segment.

    Workspace-scoped collection names contain ``/`` (the separator
    between workspace and collection, e.g. ``"my-bot/customer-42"``).
    Embedded literally in a URL path the slash splits the segment and
    the request reaches the wrong route. ``quote(name, safe='')``
    percent-encodes every reserved character (including ``/``) so the
    segment lands intact at the server.
    """
    return quote(name, safe="")


def parse_error_response(
    response_data: Any, status_code: int
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

            current = details.get("current") if isinstance(details, dict) else None
            limit = details.get("limit") if isinstance(details, dict) else None
            return QuotaExceededError(
                message,
                "storage",
                current=current,
                limit=limit,
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
    elif status_code in [502, 503, 504]:
        return ServiceUnavailableError(
            message,
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
    elif status_code == 400:
        if error_code == "COLLECTION_LIMIT_EXCEEDED":
            from .exceptions import QuotaExceededError

            current = details.get("current") if isinstance(details, dict) else None
            limit = details.get("limit") if isinstance(details, dict) else None
            return QuotaExceededError(
                message,
                "collections",
                current=current,
                limit=limit,
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
        from .exceptions import CollectionInUseError, CollectionInOtherRegionError

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
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_condition: Optional[Callable[[Exception], bool]] = None,
):
    """Retry function with exponential backoff.

    Args:
        func: Function to retry.
        max_retries: Maximum number of retries.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay cap in seconds.
        retry_condition: Optional function to determine if error is retryable.

    Returns:
        Function result.

    Raises:
        Last exception if all retries fail.
    """
    from .exceptions import is_retryable_error

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e

            # Check if error is retryable
            should_retry = (
                retry_condition(e) if retry_condition else is_retryable_error(e)
            )

            if should_retry and attempt < max_retries:
                delay = min(base_delay * (2**attempt), max_delay)
                # Add jitter (50-100% of delay)
                delay = delay * (0.5 + 0.5 * random.random())
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
