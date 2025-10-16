"""
Helper functions and utilities for Aetherfy Vectors SDK.

Provides common functionality for request handling, response processing,
and data validation across the SDK.
"""

import json
import random
import time
from typing import Any, Dict, List, Optional, Union, Callable
from urllib.parse import urljoin, urlparse

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
    """Build a complete API URL from base URL and endpoint.

    Args:
        base_url: The base API URL.
        endpoint: The API endpoint path.

    Returns:
        Complete URL.
    """
    # Ensure base_url ends with /
    if not base_url.endswith("/"):
        base_url += "/"

    # Remove leading / from endpoint if present
    if endpoint.startswith("/"):
        endpoint = endpoint[1:]

    return urljoin(base_url, endpoint)


def parse_error_response(
    response_data: Dict[str, Any], status_code: int
) -> AetherfyVectorsException:
    """Parse error response from API and return appropriate exception.

    Args:
        response_data: The error response data.
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

    message = response_data.get("message", "Unknown error")
    request_id = response_data.get("request_id")
    error_code = response_data.get("error_code")
    details = response_data.get("details", {})

    # Map status codes to exceptions
    if status_code == 401:
        return AuthenticationError(
            message, request_id=request_id, status_code=status_code, details=details
        )
    elif status_code == 429:
        retry_after = details.get("retry_after")
        return RateLimitExceededError(
            message,
            request_id=request_id,
            status_code=status_code,
            details=details,
            retry_after=retry_after,
        )
    elif status_code in [502, 503, 504]:
        return ServiceUnavailableError(
            message, request_id=request_id, status_code=status_code, details=details
        )
    elif status_code == 404:
        if error_code == "collection_not_found":
            collection_name = details.get("collection_name", "unknown")
            return CollectionNotFoundError(
                collection_name,
                request_id=request_id,
                status_code=status_code,
                details=details,
            )
        elif error_code == "point_not_found":
            point_id = details.get("point_id", "unknown")
            collection_name = details.get("collection_name", "unknown")
            return PointNotFoundError(
                point_id,
                collection_name,
                request_id=request_id,
                status_code=status_code,
                details=details,
            )
    elif status_code == 400:
        return ValidationError(
            message, request_id=request_id, status_code=status_code, details=details
        )
    elif status_code == 408:
        return RequestTimeoutError(
            message, request_id=request_id, status_code=status_code, details=details
        )

    # Default to base exception
    return AetherfyVectorsException(
        message, request_id=request_id, status_code=status_code, details=details
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
