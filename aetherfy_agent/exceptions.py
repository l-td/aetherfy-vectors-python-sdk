"""
Exceptions for the Aetherfy Agent helper.

Mirrors the Memory SDK's shape: one base for everything this module raises,
subclasses for the failures a caller is expected to branch on. The base
derives from ``AetherfyVectorsException`` so a single ``except`` around agent
code catches vector-db errors and agent-runtime errors alike.

Only three failures are worth telling apart when spawning, and they are the
three the control plane distinguishes: the payload was too big
(413 RUN_PAYLOAD_TOO_LARGE), too many runs are already in flight
(429 AGENT_RUN_CONCURRENCY_LIMIT_EXCEEDED), and everything else. "Everything
else" is any other status AND any other code on those two statuses: the pairing
is what selects a type, so a 413 the platform grows for some new reason arrives
as a plain SpawnError reporting its own code rather than wearing this one's.

READING A RUN BACK has its own small family below, under ``RunReadError``, and
it follows exactly the same rule. It is a SEPARATE family from ``SpawnError``
rather than a widening of it, because the two calls fail at different things:
a spawn is refused for what you asked to start, a read for what you asked to
see. ``ResultTooLarge`` sits outside both — it is raised before anything leaves
the machine.
"""

from typing import Any, Dict, Optional

from aetherfy_vectors.exceptions import AetherfyVectorsException

#: The two platform error codes this module gives a type of its own.
#:
#: ONE definition each, because they are used TWICE: to decide which type a
#: refusal becomes, and to stamp that type's ``error_code``. Two literals would
#: let the dispatch and the stamp disagree, which is the one way an error could
#: report a code the platform never sent.
RUN_PAYLOAD_TOO_LARGE = "RUN_PAYLOAD_TOO_LARGE"
AGENT_RUN_CONCURRENCY_LIMIT_EXCEEDED = "AGENT_RUN_CONCURRENCY_LIMIT_EXCEEDED"


class AgentError(AetherfyVectorsException):
    """Base class for every error raised by :mod:`aetherfy_agent`."""


class NotRunningOnAgent(AgentError):
    """
    Raised when a variable the platform sets on every agent machine is missing.

    This helper reads the run's environment; off a machine there is no run and
    nothing to read. Seeing this locally means the code is running somewhere
    Aetherfy did not start it.

    ``remedy`` REPLACES THE SECOND SENTENCE, and exists because the default one
    is not true of every variable. "The platform sets it before your entrypoint
    starts" holds for the variables Aetherfy injects unconditionally; it is a
    lie for ``AETHERFY_SPAWN_RESULT_PATH``, which a task machine is offered only
    when it also carries a result cap, and which a ``service`` machine never
    gets at all. A caller sent looking for a bug in their own code by a message
    that confidently describes the wrong world is worse off than one told
    nothing.
    """

    def __init__(self, variable: str, purpose: str, remedy: Optional[str] = None):
        super().__init__(
            f"{variable} is not set, so {purpose} cannot be read. "
            + (
                remedy
                or f"This helper is for code running on an Aetherfy agent "
                f"machine; the platform sets {variable} before your "
                f"entrypoint starts."
            )
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
            error_code=RUN_PAYLOAD_TOO_LARGE,
            details=details,
        )
        self.payload_bytes = payload_bytes
        self.max_bytes = max_bytes


class TooManyRunsInFlight(SpawnError):
    """
    Raised on ``429 AGENT_RUN_CONCURRENCY_LIMIT_EXCEEDED`` — the account's
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
            error_code=AGENT_RUN_CONCURRENCY_LIMIT_EXCEEDED,
            details=details,
        )
        self.in_flight_count = in_flight_count
        self.limit = limit
        self.max_in_flight_runs = max_in_flight_runs


class AgentTransportError(AgentError):
    """Raised when a request never reached the control plane at all."""


#: The three control-plane error codes the run-reading calls give a type of
#: their own. ONE definition each, for the same reason as the two above: the
#: code both SELECTS the type and is STAMPED on it, and two literals could
#: disagree.
#:
#: The first two are the deployment read's existing contract, and the /wait
#: route answers with them identically by construction — one loader serves both
#: routes upstream, so a caller need not know which one it called.
DEPLOYMENT_NOT_FOUND = "DEPLOYMENT_NOT_FOUND"
DEPLOYMENT_ACCESS_DENIED = "DEPLOYMENT_ACCESS_DENIED"
DEPLOYMENT_WAIT_TIMEOUT_INVALID = "DEPLOYMENT_WAIT_TIMEOUT_INVALID"


class ResultTooLarge(AgentError):
    """
    Raised by :func:`~aetherfy_agent.write_result` when the encoded result
    crosses this machine's inline cap.

    THE PLATFORM WOULD NOT HAVE FAILED THE RUN. A result over the cap is
    dropped and the run records ``result_error = "too_large"`` beside an empty
    result — the exit code is still the run's outcome. This helper refuses at
    the write instead, because a value discarded silently is a value the caller
    never learns to shrink: whoever spawned the run finds out, and the code
    that could have written the data to a collection and returned its id does
    not.

    Mirrors :class:`PayloadTooLarge`, which is the same cap in the other
    direction — one number bounds both. It is NOT a subclass of it, and not of
    :class:`SpawnError` either: nothing here crossed the network, so there is
    no status and no platform code to carry.

    ``max_bytes`` is read from ``AETHERFY_RUN_INLINE_MAX_BYTES``, which the
    platform injects on every task machine.
    """

    def __init__(
        self,
        message: str,
        *,
        result_bytes: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ):
        super().__init__(message)
        self.result_bytes = result_bytes
        self.max_bytes = max_bytes


class RunReadError(AgentError):
    """
    Raised when the control plane refuses to hand over a run.

    ``error_code`` is the platform's stable code (``detail.code`` in the
    control-plane envelope) and is the thing to branch on; ``message`` is prose
    that may be reworded at any time.

    Same discipline as :class:`SpawnError`: the STATUS AND THE CODE together
    select a subclass, and an unrecognised pairing arrives as this class
    reporting exactly what came back rather than wearing a type whose code the
    platform never sent.
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


class RunNotFound(RunReadError):
    """
    Raised on ``404 DEPLOYMENT_NOT_FOUND`` — no run has that id.

    A spawn returns the child run's id in ``Spawn.spawn_id``; anything else is
    a guess. Note that a run row is not immortal: an archived agent takes its
    runs with it.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            status_code=404,
            error_code=DEPLOYMENT_NOT_FOUND,
            details=details,
        )


class RunAccessDenied(RunReadError):
    """
    Raised on ``403 DEPLOYMENT_ACCESS_DENIED`` — the run belongs to another
    account.

    Distinct from :class:`RunNotFound` because the platform distinguishes them,
    and the two are different problems: an id that does not exist is a bug in
    what you passed, an id you may not read is a bug in whose key you used.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            status_code=403,
            error_code=DEPLOYMENT_ACCESS_DENIED,
            details=details,
        )


class WaitTimeoutInvalid(RunReadError):
    """
    Raised on ``422 DEPLOYMENT_WAIT_TIMEOUT_INVALID`` — the server rejected the
    ``timeout_seconds`` it was sent.

    :func:`~aetherfy_agent.wait` checks the same bound before it sends
    anything, and raises ``ValueError`` when the CALLER is out of range — that
    is a bad argument, not a refusal, and it costs no round trip. This type is
    for the case that check did not catch: the server's bound moved. Kept as a
    named type so that day arrives as something to read rather than as a bare
    422 the helper had no shape for.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            status_code=422,
            error_code=DEPLOYMENT_WAIT_TIMEOUT_INVALID,
            details=details,
        )
