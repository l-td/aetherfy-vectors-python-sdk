"""When a request is retried: one rule, the same in the JS SDK (src/retry-policy.ts).

Three kinds of failure:

1. REFUSED BEFORE ANYTHING HAPPENED. Retried for EVERY operation.
   - 503 ``WORKSPACE_LOOKUP_UNAVAILABLE``, ``COLLECTION_LOOKUP_UNAVAILABLE``,
     ``PLACEMENT_UNAVAILABLE``: the server refused at its edge; nothing ran.
   - A connection failure before the request was sent: DNS, connection
     refused, connect timeout.
   - 429 with a Retry-After: refused, nothing ran (as before this rule).
2. AMBIGUOUS: a read timeout, a connection reset after sending, any other
   5xx. The request may have run. Retried only for an operation whose repeat
   gives the same result (REPEATABLE below). A ONCE operation raises after one
   attempt: its repeat after a hidden success reports a false error (a delete
   finds nothing, a create finds the collection).
3. Everything else (4xx) is never retried.

Attempts and backoff: 3 retries after the first attempt, waiting about 1, 2,
then 4 s, each jittered to 50-100% (``utils.retry_with_backoff``), within a
call's ``timeout=`` deadline.

The operations, by class:

=========================  ==========  ==============================================
Operation                  Class       Why
=========================  ==========  ==============================================
get_collections            REPEATABLE  read
get_collection             REPEATABLE  read
collection_exists          REPEATABLE  read
get_schema                 REPEATABLE  read (also the vector-config read before upsert)
analyze_schema             REPEATABLE  read
search                     REPEATABLE  read (POST)
retrieve                   REPEATABLE  read (POST)
scroll                     REPEATABLE  read (POST)
count                      REPEATABLE  read (POST)
get_usage_stats            REPEATABLE  read
upsert                     REPEATABLE  write keyed by point id
set_payload                REPEATABLE  write keyed by point id
overwrite_payload          REPEATABLE  write keyed by point id
delete_payload             REPEATABLE  removing the same keys again changes nothing
create_field_index         REPEATABLE  creating the same index again changes nothing
set_schema                 REPEATABLE  BECAUSE it is sent without If-Match (below)
create_collection          ONCE        a repeat finds the collection it just made
delete_collection          ONCE        a repeat finds nothing to delete
delete                     ONCE        deleting points; a repeat reports a false error
delete_schema              ONCE        a repeat finds no schema
delete_field_index         ONCE        a repeat finds no index
=========================  ==========  ==============================================
"""

from typing import Dict

from .exceptions import (
    NetworkError,
    RateLimitExceededError,
    RequestTimeoutError,
    ServiceUnavailableError,
)

REPEATABLE = "repeatable"
ONCE = "once"

OPERATIONS: Dict[str, str] = {
    "get_collections": REPEATABLE,
    "get_collection": REPEATABLE,
    "collection_exists": REPEATABLE,
    "get_schema": REPEATABLE,
    "analyze_schema": REPEATABLE,
    "search": REPEATABLE,
    "retrieve": REPEATABLE,
    "scroll": REPEATABLE,
    "count": REPEATABLE,
    "get_usage_stats": REPEATABLE,
    "upsert": REPEATABLE,
    "set_payload": REPEATABLE,
    "overwrite_payload": REPEATABLE,
    "delete_payload": REPEATABLE,
    "create_field_index": REPEATABLE,
    # Repeatable BECAUSE set_schema is sent without If-Match: a repeat writes
    # the same schema again. If it ever sends If-Match, a repeat after a
    # hidden success gets a false 412, and it must move to ONCE
    # (tests/test_retry_policy.py holds the two together).
    "set_schema": REPEATABLE,
    "create_collection": ONCE,
    "delete_collection": ONCE,
    "delete": ONCE,
    "delete_schema": ONCE,
    "delete_field_index": ONCE,
}

# The "wrong class" guard (tests/test_retry_policy.py): an operation that
# sends DELETE, or whose name starts with delete/create, is ONCE unless it is
# listed here, with the reason its repeat gives the same result.
REPEATABLE_DESPITE_ITS_NAME: Dict[str, str] = {
    "delete_payload": "removes keys from points; removing them again changes nothing",
    "create_field_index": "re-creating an existing index with the same schema returns at once",
}

# 503s the server answers at its edge, before running anything.
REFUSED_BEFORE_ANYTHING_CODES = frozenset(
    {
        "WORKSPACE_LOOKUP_UNAVAILABLE",
        "COLLECTION_LOOKUP_UNAVAILABLE",
        "PLACEMENT_UNAVAILABLE",
    }
)


def refused_before_anything(error: Exception) -> bool:
    """Kind 1: nothing ran, so any operation may be repeated."""
    if isinstance(error, ServiceUnavailableError):
        return error.error_code in REFUSED_BEFORE_ANYTHING_CODES
    if isinstance(error, (NetworkError, RequestTimeoutError)):
        return getattr(error, "request_sent", None) is False
    if isinstance(error, RateLimitExceededError):
        return error.retry_after is not None
    return False


def ambiguous(error: Exception) -> bool:
    """Kind 2: the request may have run (read timeout, reset, other 5xx)."""
    return isinstance(
        error, (ServiceUnavailableError, NetworkError, RequestTimeoutError)
    ) and not refused_before_anything(error)


def should_retry_class(retry_class: str, error: Exception) -> bool:
    """The rule, for an operation of ``retry_class`` (REPEATABLE or ONCE)."""
    if retry_class not in (REPEATABLE, ONCE):
        raise ValueError(f"retry_class must be {REPEATABLE!r} or {ONCE!r}")
    if refused_before_anything(error):
        return True
    return retry_class == REPEATABLE and ambiguous(error)
