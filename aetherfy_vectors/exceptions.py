"""
Custom exceptions for Aetherfy Vectors SDK.

These exceptions provide specific error handling for various scenarios
that can occur when interacting with the global vector database service.
"""

from typing import Optional, Dict, Any


class AetherfyVectorsException(Exception):
    """Base exception for all Aetherfy Vectors errors."""

    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.request_id = request_id
        self.status_code = status_code
        self.details = details or {}

    def __str__(self) -> str:
        base_msg = self.message
        if self.request_id:
            base_msg += f" (Request ID: {self.request_id})"
        return base_msg


class AuthenticationError(AetherfyVectorsException):
    """Invalid or missing API key."""

    def __init__(self, message: str = "Invalid or missing API key", **kwargs):
        super().__init__(message, **kwargs)


class RateLimitExceededError(AetherfyVectorsException):
    """Customer has exceeded their usage limits."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.retry_after:
            base_msg += f" (Retry after {self.retry_after} seconds)"
        return base_msg


class ServiceUnavailableError(AetherfyVectorsException):
    """Backend service is temporarily unavailable."""

    def __init__(self, message: str = "Service temporarily unavailable", **kwargs):
        super().__init__(message, **kwargs)


class ValidationError(AetherfyVectorsException):
    """Invalid request parameters or data."""

    def __init__(self, message: str = "Invalid request parameters", **kwargs):
        super().__init__(message, **kwargs)


class CollectionNotFoundError(AetherfyVectorsException):
    """Specified collection does not exist."""

    def __init__(self, collection_name: str, **kwargs):
        message = f"Collection '{collection_name}' not found"
        super().__init__(message, **kwargs)
        self.collection_name = collection_name


class PointNotFoundError(AetherfyVectorsException):
    """Specified point does not exist."""

    def __init__(self, point_id: str, collection_name: str, **kwargs):
        message = f"Point '{point_id}' not found in collection '{collection_name}'"
        super().__init__(message, **kwargs)
        self.point_id = point_id
        self.collection_name = collection_name


class RequestTimeoutError(AetherfyVectorsException):
    """Request timed out."""

    def __init__(self, message: str = "Request timed out", **kwargs):
        super().__init__(message, **kwargs)


class NetworkError(AetherfyVectorsException):
    """Network connection error."""

    def __init__(self, message: str = "Network connection error", **kwargs):
        super().__init__(message, **kwargs)


class SchemaValidationError(AetherfyVectorsException):
    """Payload failed schema validation."""

    def __init__(self, errors: list, **kwargs):
        """Initialize SchemaValidationError.

        Args:
            errors: List of VectorValidationError objects.
            **kwargs: Additional exception parameters.
        """
        self.errors = errors

        # Create human-readable message
        messages = []
        for vector_error in errors:
            for error in vector_error.get("errors", []):
                messages.append(f"Vector {vector_error['index']}: {error['message']}")

        message = f"Schema validation failed:\n" + "\n".join(messages)
        super().__init__(message, **kwargs)


class CollectionInUseError(AetherfyVectorsException):
    """Collection cannot be deleted because it is referenced by one or more agents."""

    def __init__(self, collection_name: str, agents: list, **kwargs):
        agent_list = ", ".join(agents) if agents else "unknown"
        message = f"Collection '{collection_name}' is in use by agent(s): {agent_list}"
        super().__init__(message, **kwargs)
        self.collection_name = collection_name
        self.agents = agents


class SchemaNotFoundError(AetherfyVectorsException):
    """No schema defined for collection."""

    def __init__(self, collection_name: str, **kwargs):
        message = f"No schema defined for collection '{collection_name}'"
        super().__init__(message, **kwargs)
        self.collection_name = collection_name


def is_retryable_error(error: Exception) -> bool:
    """Check if an error should be retried.

    Args:
        error: The exception to check.

    Returns:
        True if error is retryable, False otherwise.
    """
    return (
        isinstance(error, ServiceUnavailableError)
        or isinstance(error, RequestTimeoutError)
        or isinstance(error, NetworkError)
        or (isinstance(error, RateLimitExceededError) and error.retry_after is not None)
    )
