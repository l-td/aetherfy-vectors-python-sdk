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


class NotFoundError(AetherfyVectorsException):
    """Something the request named does not exist (HTTP 404).

    The parent of every not-found error (collection, point, schema,
    workspace), and what a 404 whose ``error_code`` this SDK does not know
    arrives as, with that code. Catch it to handle any of them.
    """


class CollectionNotFoundError(NotFoundError):
    """Specified collection does not exist."""

    def __init__(self, collection_name: str, **kwargs):
        message = f"Collection '{collection_name}' not found"
        super().__init__(message, **kwargs)
        self.collection_name = collection_name


class PointNotFoundError(NotFoundError):
    """Specified point does not exist."""

    def __init__(self, point_id: str, collection_name: str, **kwargs):
        message = f"Point '{point_id}' not found in collection '{collection_name}'"
        super().__init__(message, **kwargs)
        self.point_id = point_id
        self.collection_name = collection_name


class RequestTimeoutError(AetherfyVectorsException):
    """Request timed out.

    ``request_sent`` is False for a connect timeout (the request never left),
    None when it may have been sent (a read timeout). See retry_policy.
    """

    def __init__(
        self,
        message: str = "Request timed out",
        request_sent: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.request_sent = request_sent


class NetworkError(AetherfyVectorsException):
    """Network connection error.

    ``request_sent`` is False when the connection was never made (DNS,
    connection refused), None when it may have been sent (a reset after
    sending). See retry_policy.
    """

    def __init__(
        self,
        message: str = "Network connection error",
        request_sent: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.request_sent = request_sent


class ConflictError(AetherfyVectorsException):
    """A 409 the SDK has no more specific class for.

    The two 409s that carry typed fields keep their own classes —
    ``CollectionInUseError`` (``COLLECTION_IN_USE``) and
    ``CollectionInOtherRegionError`` (``COLLECTION_EXISTS_IN_OTHER_REGION``)
    — and are NOT subclasses of this one, matching the JavaScript SDK. Every
    other 409 lands here, with the backend's code in ``error_code``. It is an
    ``AetherfyVectorsException``, so a handler catching the base still
    catches it; before this class existed a generic 409 was the base itself.
    """

    def __init__(self, message: str = "Resource conflict", **kwargs):
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
                f"per workspace; pick a different name or use the existing one."
            )
        super().__init__(message, **kwargs)
        self.collection_name = collection_name
        self.existing_regions = list(existing_regions)
        self.requesting_region = requesting_region


class AgentWorkspaceForbiddenError(AetherfyVectorsException):
    """An agent's API key addressed a workspace it has not been granted.

    The key Aetherfy injects into a deployed agent reaches its own workspace
    plus the workspaces the account owner granted it; any other is refused
    (403 ``AGENT_KEY_WORKSPACE_FORBIDDEN``) whether or not it exists.
    ``workspace`` names it, or is None when it was addressed by id. Not
    retryable: the owner grants access on the agent's page in the dashboard,
    or with ``afy access <agent> --add <workspace>``, and it takes effect
    within 60 seconds.
    """

    def __init__(
        self,
        workspace: Optional[str],
        message: Optional[str] = None,
        **kwargs,
    ):
        where = f"workspace '{workspace}'" if workspace else "this workspace"
        if message is None:
            message = f"This agent's API key cannot use {where}."
        add = workspace if workspace else "<workspace>"
        message += (
            f" Ask the account owner to grant this agent access to {where}: "
            f"the agent's page in the dashboard, or `afy access <agent> --add {add}`."
        )
        super().__init__(message, **kwargs)
        self.workspace = workspace


class WorkspaceNotFoundError(NotFoundError):
    """No such workspace in this account (404 ``WORKSPACE_NOT_FOUND``).

    ``workspace`` is the name as sent, or None when it was addressed by id;
    ``workspace_id`` is then the id as sent. Not retryable: create the
    workspace, or fix the name. A workspace of another account answers the
    same way, by design.
    """

    def __init__(
        self,
        workspace: Optional[str],
        workspace_id: Optional[str] = None,
        message: Optional[str] = None,
        **kwargs,
    ):
        if message is None:
            where = (
                f"Workspace '{workspace}'"
                if workspace
                else f"Workspace id '{workspace_id}'"
            )
            message = f"{where} not found in this account."
        super().__init__(message, **kwargs)
        self.workspace = workspace
        self.workspace_id = workspace_id


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


class SchemaNotFoundError(NotFoundError):
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
        # A failed chunk's outcome is unknown, never "not written": a chunk
        # that timed out or lost its connection may have been applied, and
        # the server writes a large chunk in parts, so a refusal can come
        # after part of it was written.
        message = (
            f"Partial upsert: {saved} of {total} points confirmed saved; the outcome "
            f"of the other {failed_count} point(s), in {len(failed)} chunk(s) that "
            f"failed, is unknown: a chunk that timed out, lost its connection or was "
            f"refused may have been written in whole or in part. Retrying the upsert "
            f"is safe: it replaces points by id. See .failed for per-chunk point IDs "
            f"and errors."
        )
        super().__init__(message, **kwargs)
        self.saved = saved
        self.total = total
        self.failed = failed


def is_retryable_error(error: Exception) -> bool:
    """True when the error's KIND may be retried: a 5xx, a timeout, a network
    failure, or a 429 with a Retry-After.

    It does not know the operation, so it decides no retry: the client's
    retries (and utils.retry_with_backoff) follow retry_policy, which weighs
    the operation's class. Retrying an ambiguous failure is safe only for an
    operation retry_policy classes REPEATABLE.

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
