"""
Exceptions for the Aetherfy Agent helper.

Mirrors the Memory SDK's shape: one base for everything this module raises,
subclasses for the failures a caller is expected to branch on. The base
derives from ``AetherfyVectorsException`` so a single ``except`` around agent
code catches vector-db errors and agent-runtime errors alike.

Only three failures are worth telling apart when spawning, and they are the
three the control plane distinguishes: the payload was too big
(413 RUN_PAYLOAD_TOO_LARGE), the account's line of waiting runs is full
(429 AGENT_RUN_LINE_FULL), and everything else. "Everything else" is any other
status AND any other code on those two statuses: the pairing is what selects a
type, so a 413 the platform grows for some new reason arrives as a plain
SpawnError reporting its own code rather than wearing this one's.

READING A RUN BACK has its own small family below, under ``RunReadError``, and
it follows exactly the same rule. It is a SEPARATE family from ``SpawnError``
rather than a widening of it, because the two calls fail at different things:
a spawn is refused for what you asked to start, a read for what you asked to
see. ``ResultTooLarge`` sits outside both — it is raised before anything leaves
the machine.

CANCELLING A RUN has a third family, under ``RunCancelError``, with the same
rule. Two of its refusals are the read family's own types, because they are the
same two problems: ``RunNotFound`` (no run has that id) and ``RunAccessDenied``
(the run is not the caller's to touch). Code that catches only
``RunCancelError`` therefore misses those two; catch ``AgentError`` to see every
failure of the module. ``NotAServiceRun`` is raised before anything leaves the
machine, by ``service_run``.

LOOKING A PEER UP is a third family, under ``PeerLookupError``, with the same
rule again: the status and the code together select a subclass
(``PeerNotFound``, ``PeerHasNoUrl``) and anything else arrives as the base
reporting what came back. Reaching the peer itself, once it is found, raises
nothing of the platform's: the peer is another service, and what it answers is
returned as it came (see ``call_agent``).
"""

from typing import Any, Dict, Optional, Sequence

from aetherfy_vectors.exceptions import AetherfyVectorsException

#: The two platform error codes this module gives a type of its own.
#:
#: ONE definition each, because they are used TWICE: to decide which type a
#: refusal becomes, and to stamp that type's ``error_code``. Two literals would
#: let the dispatch and the stamp disagree, which is the one way an error could
#: report a code the platform never sent.
RUN_PAYLOAD_TOO_LARGE = "RUN_PAYLOAD_TOO_LARGE"
AGENT_RUN_LINE_FULL = "AGENT_RUN_LINE_FULL"

#: The two headers the platform sets on the call to a service's
#: ``POST /aetherfy/run``: the run's id and the run's own key. ONE definition
#: each: ``service_run`` reads them and ``NotAServiceRun`` names them.
RUN_ID_HEADER = "X-Aetherfy-Run"
RUN_KEY_HEADER = "X-Aetherfy-Run-Key"


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
        retry_after: Optional[float] = None,
    ):
        super().__init__(
            message,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )
        self.retry_after = retry_after


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


class RunLineFull(SpawnError):
    """
    Raised on ``429 AGENT_RUN_LINE_FULL`` — the account's line of waiting runs is
    full.

    A run started while the account is at its in-flight limit does not fail: it
    waits in the account's line, and the spawn is accepted with status
    ``waiting``. This is raised only when the line itself already holds the most
    runs the plan allows, so there is nowhere left to wait.

    THE LIMITS ARE THE ACCOUNT'S, NOT THE AGENT'S, and the plan sets them.
    ``waiting_count`` runs are in the line now, of at most ``max_waiting_runs``;
    ``in_flight_count`` runs are executing, of at most ``max_in_flight_runs``.
    ``limit`` names WHICH plan limit was hit (``"max_waiting_runs"`` today) and
    is read from the envelope rather than assumed, so a second named limit
    reaching this status arrives intact. Any of the numbers is ``None`` when the
    envelope does not carry it, so a caller building a message from them must
    not assume a number.

    NOTHING IS RETRIED FOR YOU. This refusal comes with no ``Retry-After``;
    ``retry_after`` is the seconds that header asks to wait, and is ``None``
    whenever it is absent. Spawn again once runs have left the line, and decide
    for yourself how long to wait.
    """

    def __init__(
        self,
        message: str,
        *,
        waiting_count: Optional[int] = None,
        max_waiting_runs: Optional[int] = None,
        in_flight_count: Optional[int] = None,
        max_in_flight_runs: Optional[int] = None,
        limit: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(
            message,
            status_code=429,
            error_code=AGENT_RUN_LINE_FULL,
            details=details,
            retry_after=retry_after,
        )
        self.waiting_count = waiting_count
        self.max_waiting_runs = max_waiting_runs
        self.in_flight_count = in_flight_count
        self.max_in_flight_runs = max_in_flight_runs
        self.limit = limit


class AgentTransportError(AgentError):
    """Raised when a request never reached the control plane at all."""


#: The control-plane error codes the run-reading and run-cancelling calls give a
#: type of their own. ONE definition each, for the same reason as the two above:
#: the code both SELECTS the type and is STAMPED on it, and two literals could
#: disagree.
#:
#: The first two are the deployment read's existing contract, and the /wait
#: route answers with them identically by construction — one loader serves both
#: routes upstream, so a caller need not know which one it called. The cancel
#: route answers with the same two.
DEPLOYMENT_NOT_FOUND = "DEPLOYMENT_NOT_FOUND"
DEPLOYMENT_ACCESS_DENIED = "DEPLOYMENT_ACCESS_DENIED"
DEPLOYMENT_WAIT_TIMEOUT_INVALID = "DEPLOYMENT_WAIT_TIMEOUT_INVALID"
#: A cancel of an id that is a release, not a run.
DEPLOYMENT_NOT_A_RUN = "DEPLOYMENT_NOT_A_RUN"
#: An agent's key used on a run outside what it may touch. A cancel of a run
#: that is not below the key's own run is refused with it.
AUTH_AGENT_KEY_OUT_OF_SCOPE = "AUTH_AGENT_KEY_OUT_OF_SCOPE"


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
        retry_after: Optional[float] = None,
    ):
        super().__init__(
            message,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )
        self.retry_after = retry_after


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
    Raised when the control plane will not let this key touch the run.

    Two codes arrive as this type, and ``error_code`` is the one that came, the
    thing to tell them apart by:

    - ``403 DEPLOYMENT_ACCESS_DENIED``, the run belongs to another account, from
      a read or a cancel;
    - ``403 AUTH_AGENT_KEY_OUT_OF_SCOPE``, from :func:`~aetherfy_agent.cancel`
      ONLY: an agent's key may cancel just the runs below its own run.

    A READ is different. An agent's own key reading a run outside its run tree
    is ``403 AUTH_AGENT_KEY_OUT_OF_SCOPE`` too, but it arrives as a plain
    :class:`RunReadError` carrying that ``error_code``: what the key may read,
    not whose run it is.

    Distinct from :class:`RunNotFound` because the platform distinguishes them,
    and the two are different problems: an id that does not exist is a bug in
    what you passed, an id you may not touch is a bug in whose key you used.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
        error_code: str = DEPLOYMENT_ACCESS_DENIED,
    ):
        super().__init__(
            message,
            status_code=403,
            error_code=error_code,
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


class RunCancelError(AgentError):
    """
    Raised when the control plane refuses to cancel a run, for a reason with no
    type of its own.

    ``error_code`` is the platform's stable code (``detail.code`` in the
    control-plane envelope) and is the thing to branch on; ``message`` is prose
    that may be reworded at any time.

    Same discipline as :class:`SpawnError`: the STATUS AND THE CODE together
    select a subclass, and an unrecognised pairing arrives as this class
    reporting exactly what came back. A run that had already ended is not a
    refusal at all: the cancel answers 200 with outcome ``already_finished``.

    The two refusals every run call shares are NOT subclasses of this: an id no
    run has is :class:`RunNotFound`, and a run this key may not cancel is
    :class:`RunAccessDenied`. Catch :class:`AgentError` to see all of them.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(
            message,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )
        self.retry_after = retry_after


class NotARun(RunCancelError):
    """
    Raised on ``409 DEPLOYMENT_NOT_A_RUN`` — the id names a release, not a run.

    A release is cancelled with the CLI's ``afy cancel <agent>``; this call
    cancels runs. ``Spawn.spawn_id`` is always a run's id.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            status_code=409,
            error_code=DEPLOYMENT_NOT_A_RUN,
            details=details,
        )


class NotAServiceRun(AgentError):
    """
    Raised by :func:`~aetherfy_agent.service_run` when the headers do not carry
    a run's id and key.

    ``service_run`` is for the handler of ``POST /aetherfy/run`` ONLY. The
    platform sets ``X-Aetherfy-Run`` and ``X-Aetherfy-Run-Key`` on that one call
    and removes both from every public request, so headers without them belong
    to an ordinary request, or to a service deployed before a service run had a
    key of its own, which has to be deployed again. ``missing`` names the headers
    that were not there.
    """

    def __init__(self, missing: Sequence[str]):
        super().__init__(
            "service_run() is for the handler of POST /aetherfy/run only: the "
            "platform sets {0} and {1} on that call, and these headers lack "
            "{2}. A public request carries neither; a service deployed before a "
            "service run had a key of its own sends no key and has to be "
            "deployed again.".format(
                RUN_ID_HEADER, RUN_KEY_HEADER, " and ".join(missing)
            )
        )
        self.missing = tuple(missing)


#: The two control-plane error codes a peer lookup gives a type of its own. ONE
#: definition each, for the same reason as the codes above: the code both
#: SELECTS the type and is STAMPED on it.
AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
PEER_HAS_NO_URL = "PEER_HAS_NO_URL"

#: The code the edge in front of ``*.aetherfy.dev`` answers a RETIRED address
#: with, beside ``AGENT_NOT_FOUND`` (which it uses for an address that never
#: existed or was deleted). Not a lookup error: it arrives in the body of a
#: peer's response, never from the control plane, and ``call_agent`` reads it
#: to decide whether to look the peer up again.
AGENT_NOT_REACHABLE = "AGENT_NOT_REACHABLE"


class PeerLookupError(AgentError):
    """
    Raised when the control plane does not hand over a peer's address.

    ``error_code`` is the platform's stable code (``detail.code`` in the
    control-plane envelope) and is the thing to branch on; ``message`` is prose
    that may be reworded at any time.

    Same discipline as :class:`SpawnError`: the STATUS AND THE CODE together
    select a subclass, and an unrecognised pairing arrives as this class
    reporting exactly what came back — among them ``403
    AUTH_AGENT_KEY_OUT_OF_SCOPE``, when this machine's own key may not look
    peers up.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(
            message,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )
        self.retry_after = retry_after


class PeerNotFound(PeerLookupError):
    """
    Raised on ``404 AGENT_NOT_FOUND`` — this agent's own workspace holds no
    agent of that name.

    Peers are the agents of the caller's own workspace, found by the name the
    agent has there; another workspace's agent is not a peer, and neither is a
    name that was spelled differently when the agent was created.
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
            error_code=AGENT_NOT_FOUND,
            details=details,
        )


class PeerHasNoUrl(PeerLookupError):
    """
    Raised on ``409 PEER_HAS_NO_URL`` — the peer exists, but nothing listens
    for requests on it.

    A peer that is not reachable by URL cannot be called, whatever else it can
    do: it is not a service. Start it with :func:`~aetherfy_agent.spawn`
    instead, or call a peer that serves HTTP.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            status_code=409,
            error_code=PEER_HAS_NO_URL,
            details=details,
        )


# --- MESSAGES BETWEEN RUNNING AGENTS -----------------------------------------
#
# The codes a message call refuses with (control plane, messaging between running
# agents). The STATUS AND THE CODE together select a subclass; an unrecognised
# pairing arrives as :class:`MessageError` reporting exactly what came back.

#: A message larger than the inline cap (the run payload's).
RUN_MESSAGE_TOO_LARGE = "RUN_MESSAGE_TOO_LARGE"
#: More messages from one sending run in a minute than the rate allows.
RUN_MESSAGE_RATE_LIMITED = "RUN_MESSAGE_RATE_LIMITED"
#: The recipient's mailbox holds as many unread messages as it may.
RUN_MAILBOX_FULL = "RUN_MAILBOX_FULL"
#: The account's live runs' messages would pass its stored-logs budget.
RUN_MESSAGE_STORAGE_FULL = "RUN_MESSAGE_STORAGE_FULL"
#: The recipient has ended: its mailbox takes no more messages.
RUN_MESSAGE_RECIPIENT_ENDED = "RUN_MESSAGE_RECIPIENT_ENDED"
#: receive(from_run=...) on a run that has ended and left nothing unread.
RUN_MESSAGE_SENDER_ENDED = "RUN_MESSAGE_SENDER_ENDED"


class MessageError(AgentError):
    """
    Raised when the control plane refuses a message call (send, receive, the
    history) for a reason with no type of its own.

    ``error_code`` is the platform's stable code and the thing to branch on;
    ``message`` is prose that may be reworded at any time. An id no run has is
    :class:`RunNotFound`, and a run this key may not reach is
    :class:`RunAccessDenied`, as for every run call.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(
            message,
            status_code=status_code,
            details=details,
            error_code=error_code,
        )
        self.retry_after = retry_after


class MessageTooLarge(MessageError):
    """
    Raised on ``413 RUN_MESSAGE_TOO_LARGE``: the message is larger than the
    inline cap, the same one a run's payload has. ``message_bytes`` is its size
    and ``max_bytes`` the cap. A message is for instructions and answers: put
    large data in a collection and send its id.
    """

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message, status_code=413, error_code=RUN_MESSAGE_TOO_LARGE, details=details
        )
        details = details or {}
        self.message_bytes = details.get("message_bytes")
        self.max_bytes = details.get("max_bytes")


class MessageRateLimited(MessageError):
    """
    Raised on ``429 RUN_MESSAGE_RATE_LIMITED``: this run sent as many messages
    in the last minute as it may. ``retry_after`` is the seconds until the next
    one is taken.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(
            message,
            status_code=429,
            error_code=RUN_MESSAGE_RATE_LIMITED,
            details=details,
            retry_after=retry_after,
        )


class MailboxFull(MessageError):
    """
    Raised on ``429 RUN_MAILBOX_FULL``: the recipient has as many unread
    messages as its mailbox holds. It clears when the recipient reads, not with
    time, so there is no ``retry_after``. ``unread`` and ``max_unread`` say how
    many.
    """

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message, status_code=429, error_code=RUN_MAILBOX_FULL, details=details
        )
        details = details or {}
        self.unread = details.get("unread")
        self.max_unread = details.get("max_unread")


class MessageStorageFull(MessageError):
    """
    Raised on ``429 RUN_MESSAGE_STORAGE_FULL``: the messages of the account's
    live runs would pass its plan's stored-logs budget. It clears as runs read
    and end, so there is no ``retry_after``.
    """

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            status_code=429,
            error_code=RUN_MESSAGE_STORAGE_FULL,
            details=details,
        )


class RecipientEnded(MessageError):
    """
    Raised on ``409 RUN_MESSAGE_RECIPIENT_ENDED``: the run the message is for
    has ended, and an ended run's mailbox takes no more messages. Start a new
    run instead. ``recipient_state`` is how it ended.
    """

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            status_code=409,
            error_code=RUN_MESSAGE_RECIPIENT_ENDED,
            details=details,
        )
        self.recipient_state = (details or {}).get("recipient_state")


class SenderEnded(MessageError):
    """
    Raised by ``receive(from_run=...)`` on ``409 RUN_MESSAGE_SENDER_ENDED``:
    that run has ended and nothing it sent is left unread, so no message from it
    will come. ``sender_state`` is how it ended.
    """

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            status_code=409,
            error_code=RUN_MESSAGE_SENDER_ENDED,
            details=details,
        )
        self.sender_state = (details or {}).get("sender_state")
