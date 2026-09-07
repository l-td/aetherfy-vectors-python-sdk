"""
Exceptions for the Aetherfy Agent helper.

Mirrors the Memory SDK's shape: one base for everything this module raises,
subclasses for the failures a caller is expected to branch on. The base
derives from ``AetherfyVectorsException`` so a single ``except`` around agent
code catches vector-db errors and agent-runtime errors alike.

Only three failures are worth telling apart when spawning, and they are the
three the control plane distinguishes: the payload was too big (413), too many
runs are already in flight (429), and everything else (any other status).
"""

from typing import Any, Dict, Optional

from aetherfy_vectors.exceptions import AetherfyVectorsException


class AgentError(AetherfyVectorsException):
    """Base class for every error raised by :mod:`aetherfy_agent`."""


class NotRunningOnAgent(AgentError):
    """
    Raised when a variable the platform sets on every agent machine is missing.

    This helper reads the run's environment; off a machine there is no run and
    nothing to read. Seeing this locally means the code is running somewhere
    Aetherfy did not start it.
    """

    def __init__(self, variable: str, purpose: str):
        super().__init__(
            f"{variable} is not set, so {purpose} cannot be read. This helper "
            f"is for code running on an Aetherfy agent machine; the platform "
            f"sets {variable} before your entrypoint starts."
        )
        self.variable = variable


class PayloadUnavailable(AgentError):
    """
    Raised when neither the payload file nor the HTTP fallback yielded a payload.

    Both routes to the same bytes failed — see the message for which one was
    tried and why it did not answer.
    """

    def __init__(self, message: str):
        super().__init__(message)


class SpawnError(AgentError):
    """
    Raised when the control plane refuses a spawn.

    ``code`` is the platform's stable error code (``detail.code`` in the
    control-plane envelope) and is the thing to branch on; ``message`` is
    prose that may be reworded at any time.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )


class PayloadTooLarge(SpawnError):
    """
    Raised on ``413 RUN_PAYLOAD_TOO_LARGE`` — the spawn payload crossed the
    inline cap.

    The payload carries parameters and references, not data. Write the data to
    a collection and pass its id.
    """

    def __init__(
        self,
        message: str,
        *,
        payload_bytes: Optional[int] = None,
        max_bytes: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            status_code=413,
            error_code="RUN_PAYLOAD_TOO_LARGE",
            details=details,
        )
        self.payload_bytes = payload_bytes
        self.max_bytes = max_bytes


class TooManyRunsInFlight(SpawnError):
    """
    Raised on ``429 AGENT_SPAWN_CONCURRENCY_LIMIT_EXCEEDED`` — the account's
    runs-in-flight cap is full.

    Retryable, unlike the other two: wait for runs to finish and spawn again.

    THE CAP IS THE ACCOUNT'S, NOT THE AGENT'S, and it is set by the plan.
    ``limit`` names WHICH plan limit was hit — a stable string the platform
    documents as switchable-on — and ``max_in_flight_runs`` is its value.
    ``max_in_flight_runs`` is ``None`` when the plan declares no cap, so a
    caller building a message from it must not assume a number.

    ``limit`` was ``"max_in_flight_runs"`` at the time of writing and is the
    only value the platform sends today; it is read from the envelope rather
    than assumed, so a second named limit reaching this status arrives intact
    instead of being reported as the first one.
    """

    def __init__(
        self,
        message: str,
        *,
        in_flight_count: Optional[int] = None,
        limit: Optional[str] = None,
        max_in_flight_runs: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            status_code=429,
            error_code="AGENT_SPAWN_CONCURRENCY_LIMIT_EXCEEDED",
            details=details,
        )
        self.in_flight_count = in_flight_count
        self.limit = limit
        self.max_in_flight_runs = max_in_flight_runs


class AgentTransportError(AgentError):
    """Raised when a request never reached the control plane at all."""
