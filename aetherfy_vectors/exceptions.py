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
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.request_id = request_id
        self.status_code = status_code
        self.details = details or {}
        # error_code carries the backend's stable error identifier
        # (e.g. COLLECTION_NOT_FOUND, SCHEMA_NOT_DEFINED). The status
        # code is for the HTTP layer; this is for SDK-level branching
        # — most importantly disambiguating which kind of 404 happened.
        self.error_code = error_code

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


class CollectionInOtherRegionError(AetherfyVectorsException):
    """A collection name created in one region already belongs to another.

    Raised on create_collection when the same name is already owned by a
    different region. Carries typed fields so callers can offer a
    different name or pin to the existing region without parsing the
    error message string.
    """

    def __init__(
        self,
        collection_name: str,
        existing_regions: list,
        requesting_region: str,
        message: Optional[str] = None,
        **kwargs,
    ):
        if message is None:
            message = (
                f"Collection '{collection_name}' already exists in region "
                f"{', '.join(existing_regions)}. Collection names are unique "
                f"per account; pick a different name or use the existing one."
            )
        super().__init__(message, **kwargs)
        self.collection_name = collection_name
        self.existing_regions = list(existing_regions)
        self.requesting_region = requesting_region


class QuotaExceededError(AetherfyVectorsException):
    """Tier quota exceeded (e.g. collection count limit)."""

    def __init__(
        self,
        message: str,
        quota_type: str,
        current: Optional[int] = None,
        limit: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.quota_type = quota_type
        self.current = current
        self.limit = limit


class SchemaNotFoundError(AetherfyVectorsException):
    """No schema defined for collection."""

    def __init__(self, collection_name: str, **kwargs):
        message = f"No schema defined for collection '{collection_name}'"
        super().__init__(message, **kwargs)
        self.collection_name = collection_name


class PartialUpsertError(AetherfyVectorsException):
    """A multi-chunk upsert succeeded for some chunks and failed for others.

    Carries which point IDs were saved and which weren't so callers can
    retry the failed ones surgically without double-upserting the saved
    ones (Qdrant upsert is idempotent by point ID, so a retry of an
    already-saved point is also safe — this just avoids the wasted
    round-trip).

    Raised only when the SDK had to split the upsert into multiple HTTP
    requests due to byte-size limits and at least one of those requests
    failed after the SDK's retry budget was exhausted. Single-request
    upserts that fail raise the more specific exception directly
    (ValidationError, NetworkError, ServiceUnavailableError, etc.) —
    same behaviour as before chunking.

    Attributes:
        saved: Count of points successfully upserted (across all
            successful chunks).
        total: Total points the caller passed to upsert().
        failed: List of dicts, one per failed chunk:
            {"point_ids": [...], "error": AetherfyVectorsException}.
    """

    def __init__(
        self,
        saved: int,
        total: int,
        failed: list,
        **kwargs,
    ):
        failed_count = sum(len(f["point_ids"]) for f in failed)
        message = (
            f"Partial upsert: {saved} of {total} points saved; "
            f"{failed_count} failed across {len(failed)} chunk(s). "
            f"See .failed for per-chunk point IDs and errors."
        )
        super().__init__(message, **kwargs)
        self.saved = saved
        self.total = total
        self.failed = failed


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
