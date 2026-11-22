"""
Aetherfy Agent — what code running on an Aetherfy machine does.

This is a THIN wrapper over contracts the platform already publishes. It
invents no protocol: every call here has a hand-rolled equivalent in
https://docs.aetherfy.com/agents/task-contract, and the helper exists so that
equivalent stops being copied into every task.

    from aetherfy_agent import payload, machine, fan_out, spawn
    from aetherfy_agent import write_result, result, wait, cancel
    from aetherfy_agent import send, receive, messages, parent, restarted
    from aetherfy_agent import agent_url, call_agent

    data = payload()                         # this run's input, {} when none
    shape = machine()                        # vcpus / memory_mb / region
    results = fan_out(work, data["items"])   # in-machine pool, input order
    write_result({"rows": len(results)})     # this run's answer

    run = spawn("nightly-rollup", {"date": "2026-09-07"})
    finished = wait(run.spawn_id)            # or result(...) for a plain read
    print(finished.result)
    cancel(run.spawn_id)                     # stop it, waiting or running

    send(run.spawn_id, {"focus": "eu"})      # a message to a run while it runs
    message = receive()                      # this run's next message, or a child's end

    url = agent_url("analyst")               # a peer service's address, by name
    reply = call_agent("analyst", "/summarise", method="POST", json={"id": 7})
    print(reply.status, reply.json())

TWO HALVES, and they are the same contract read from opposite ends. A task
reads its payload and writes its result; whoever started it spawns and then
reads that result back. ``payload``/``write_result`` are files on the machine
and touch no network at all; ``spawn``/``result``/``wait``/``cancel`` and the
messages are the control plane, and are the only calls here that do.

WHILE BOTH RUN, THEY CAN TALK. Every run has a mailbox on the control plane.
``send`` puts a message in another run's mailbox; ``receive`` hands over this
run's next one, or the end of a run it started. A task run that waits in
``receive`` past a short grace is suspended, and is not billed until a message
wakes it. A message stays unread until the receiving run asks for the next one,
so a run that starts again on a new machine (``restarted()``) reads again what
it had not finished with, and ``messages()`` is its whole history.

A RUN HAS A LIFE BEFORE IT RUNS. When the account is at its in-flight limit, a
new run waits in the account's line (state ``waiting``, no machine, nothing
billed), and ``wait`` keeps waiting through that time. A run ends ``completed``,
``failed`` or ``cancelled``; ``cancel`` is what ends one from outside, whether
it is waiting or executing.

INSIDE A SERVICE'S RUN HANDLER, USE THE HANDLE. The key these module-level
calls read from ``AETHERFY_API_KEY`` is, in a service, the release's: right for
the service's own traffic, wrong for a run. ``service_run(request.headers)``
returns a handle whose ``spawn``, ``result``, ``wait`` and ``cancel`` use the
run's own key, so what the handler starts belongs to the run.

A THIRD THING, for a service that talks to another service: ``agent_url`` and
``agent_urls`` look a peer's public address up BY NAME at call time, and
``call_agent`` makes the request. Nothing about a peer's address is baked into
this machine's environment, so a peer that is recreated is found at its new
address without a restart here. The lookup is the control plane's, and its
answer is remembered in this process for ``PEER_CACHE_SECONDS``.

It ships inside the ``aetherfy-vectors`` distribution beside
``aetherfy_vectors`` and ``aetherfy_memory``, and the standard runtime image
preinstalls that distribution — so on a plain agent these names import with
nothing in your requirements. A custom container installs it itself.
"""

import json
import os
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeGuard,
    TypeVar,
    Union,
)
from urllib.parse import quote

from . import _http, _peers
from .exceptions import (
    AGENT_NOT_FOUND,
    AGENT_NOT_REACHABLE,
    AGENT_RUN_LINE_FULL,
    AUTH_AGENT_KEY_OUT_OF_SCOPE,
    DEPLOYMENT_ACCESS_DENIED,
    DEPLOYMENT_NOT_A_RUN,
    DEPLOYMENT_NOT_FOUND,
    DEPLOYMENT_WAIT_TIMEOUT_INVALID,
    PEER_HAS_NO_URL,
    RUN_MAILBOX_FULL,
    RUN_MESSAGE_RATE_LIMITED,
    RUN_MESSAGE_RECIPIENT_ENDED,
    RUN_MESSAGE_SENDER_ENDED,
    RUN_MESSAGE_STORAGE_FULL,
    RUN_MESSAGE_TOO_LARGE,
    RUN_ID_HEADER,
    RUN_KEY_HEADER,
    RUN_PAYLOAD_TOO_LARGE,
    AgentError,
    AgentTransportError,
    MailboxFull,
    MessageError,
    MessageRateLimited,
    MessageStorageFull,
    MessageTooLarge,
    NotARun,
    NotAServiceRun,
    NotRunningOnAgent,
    PayloadTooLarge,
    PayloadUnavailable,
    PeerHasNoUrl,
    PeerLookupError,
    PeerNotFound,
    RecipientEnded,
    ResultTooLarge,
    RunAccessDenied,
    RunCancelError,
    RunLineFull,
    RunNotFound,
    RunReadError,
    SenderEnded,
    SpawnError,
    WaitTimeoutInvalid,
)
from .models import (
    Cancellation,
    Ended,
    MachineShape,
    Message,
    PeerResponse,
    Run,
    RunRetry,
    Sent,
    Spawn,
)

# ONE distribution, ONE version. `aetherfy_vectors.__version__` is what
# setup.py reads to stamp the wheel, so re-exporting it here means the
# User-Agent this helper sends can never disagree with the release that
# sent it. A second literal would drift on the first bump nobody
# remembered to make twice.
from aetherfy_vectors import __version__  # noqa: E402  (re-exported)

__all__ = [
    "payload",
    "machine",
    "fan_out",
    "spawn",
    "write_result",
    "result",
    "wait",
    "cancel",
    "service_run",
    "agent_url",
    "agent_urls",
    "call_agent",
    "send",
    "receive",
    "messages",
    "restarted",
    "parent",
    "SUPERVISOR_WAIT_URL_VARIABLE",
    "INSTANCE_ID_VARIABLE",
    # PUBLIC IN BOTH LANGUAGES OR NEITHER. The JavaScript helper exports these
    # three from its entry point, so a task ported between the two would find
    # the bound readable in one and not the other. A caller sizing its own
    # retry loop around wait() is the reason they are readable at all.
    "WAIT_TIMEOUT_MIN_SECONDS",
    "WAIT_TIMEOUT_MAX_SECONDS",
    "WAIT_TIMEOUT_DEFAULT_SECONDS",
    # The same rule for the peer lookup's memory window.
    "PEER_CACHE_SECONDS",
    "PEER_RETRY_FLOOR_SECONDS",
    "PEER_RETRY_CAP_SECONDS",
    "Cancellation",
    "Ended",
    "HeaderSource",
    "MachineShape",
    "Message",
    "PeerResponse",
    "Run",
    "RunRetry",
    "Sent",
    "ServiceRun",
    "Spawn",
    "AgentError",
    "AgentTransportError",
    "MailboxFull",
    "MessageError",
    "MessageRateLimited",
    "MessageStorageFull",
    "MessageTooLarge",
    "NotARun",
    "NotAServiceRun",
    "NotRunningOnAgent",
    "PayloadUnavailable",
    "PayloadTooLarge",
    "PeerLookupError",
    "PeerNotFound",
    "PeerHasNoUrl",
    "RecipientEnded",
    "ResultTooLarge",
    "RunReadError",
    "RunNotFound",
    "RunAccessDenied",
    "RunCancelError",
    "RunLineFull",
    "SenderEnded",
    "SpawnError",
    "WaitTimeoutInvalid",
]

T = TypeVar("T")
R = TypeVar("R")

#: The one line a fan-out prints, and the reason it prints at all: the platform
#: cannot count the customer's in-machine workers for them. It lands in the
#: run's logs like any other stdout, so a run's width is visible after the fact
#: without the customer having written the line. Byte-for-byte identical to the
#: JavaScript helper's.
_FAN_OUT_LINE = (
    "aetherfy: fanning out {width} wide on {vcpus} vCPU / {memory_mb} MB ({n} tasks)"
)

#: Multiplier behind the default fan-out width. Model calls and HTTP requests
#: spend nearly all their time waiting, so a thread pool wider than the core
#: count is the right default for them. CPU-bound work should pass
#: ``width=machine().vcpus`` and ``kind="processes"``.
_IO_BOUND_WIDTH_PER_VCPU = 8

#: How far past the server's own hold this helper lets a :func:`wait` request
#: run before it gives up on the socket. THE CLIENT'S BOUND MUST EXCEED THE
#: SERVER'S, or a wait that the control plane is about to answer at its
#: deadline is cut off here first and reported as a transport failure — the one
#: outcome a caller cannot tell from a real network fault. The margin covers
#: the round trip and the serialization on either side.
_WAIT_TRANSPORT_MARGIN_SECONDS = 15.0

#: The bound the control plane enforces on ``?timeout_seconds``. Checked here
#: too, so a caller learns about a bad argument without paying a round trip to
#: be told. Waiting longer than the maximum is another call, not a bigger
#: number: the request is held open and anything longer is cut by the network
#: in front of Aetherfy.
WAIT_TIMEOUT_MIN_SECONDS = 1
WAIT_TIMEOUT_MAX_SECONDS = 60
WAIT_TIMEOUT_DEFAULT_SECONDS = 30

#: How long a looked-up peer address is served without asking again. The one
#: definition lives beside the cache that applies it; this is the same number,
#: readable where the calls are.
PEER_CACHE_SECONDS = _peers.PEER_CACHE_SECONDS

#: How soon after a failed lookup the control plane is asked again: its Retry-After, kept between these.
#: While waiting, the last address known for the peer is served (see :func:`agent_url`).
PEER_RETRY_FLOOR_SECONDS = _peers.PEER_RETRY_FLOOR_SECONDS
PEER_RETRY_CAP_SECONDS = _peers.PEER_RETRY_CAP_SECONDS


def _require(variable: str, purpose: str, remedy: Optional[str] = None) -> str:
    value = os.environ.get(variable)
    if not value:
        raise NotRunningOnAgent(variable, purpose, remedy)
    return value


#: What to say when the RESULT PATH is missing, instead of the default "the
#: platform sets this before your entrypoint starts" — which is not true of
#: this one variable. The task supervisor offers the path only when the machine
#: also carries an inline cap (image_generator.py: `if _RESULT_MAX_BYTES > 0`,
#: else it logs that this run cannot return a result), and a `service` machine
#: has no runs to return anything from. A customer told the platform always
#: sets it would go looking for a bug in their own code.
_NO_RESULT_PATH_REMEDY = (
    "write_result() is a TASK-ONLY call: a result belongs to a run, and a "
    "`service` agent has no runs, so a service machine is never given a result "
    "path and this call can never succeed there. Return your answer over HTTP "
    "instead. On a `type: job` machine Aetherfy sets the path before each run's "
    "entrypoint starts, but only when that machine also carries an inline "
    "result cap (AETHERFY_RUN_INLINE_MAX_BYTES) — without the cap the platform "
    "cannot accept a result and does not offer the path."
)


def _user_agent() -> str:
    return _http.user_agent(__version__)


def payload() -> Dict[str, Any]:
    """
    Return this run's input payload, or ``{}`` when the run was given none.

    THE EMPTY CASE IS THE NORMAL CASE. A scheduled fire, or a manual run
    started without input, gets ``{}``. Write the task so that no input is the
    path it takes most often and treat any input as an optional override.

    The payload is written to a file on the machine before the entrypoint
    starts and its path put in ``AETHERFY_SPAWN_PAYLOAD_PATH``; nothing crosses
    the network to read it. When that variable is unset, or names a file that
    is not there, the same bytes are fetched over HTTP from
    ``GET {AETHERFY_API_URL}/deployments/{AETHERFY_SPAWN_ID}/payload`` — a
    fallback for a machine that could not write the file, not the path to build
    on.

    :raises PayloadUnavailable: neither route yielded a payload.
    """
    path = os.environ.get("AETHERFY_SPAWN_PAYLOAD_PATH")
    if path:
        try:
            with open(path, encoding="utf-8") as handle:
                raw = handle.read()
        except OSError:
            # The file the platform promised is not readable. Fall through to
            # the HTTP route rather than failing: that is the exact case the
            # fallback exists for.
            raw = None
        if raw is not None:
            if not raw.strip():
                return {}
            try:
                parsed = json.loads(raw)
            except ValueError as exc:
                raise PayloadUnavailable(
                    "AETHERFY_SPAWN_PAYLOAD_PATH ({0}) does not hold JSON: {1}".format(
                        path, exc
                    )
                )
            return parsed if isinstance(parsed, dict) else {}

    api_url = os.environ.get("AETHERFY_API_URL")
    spawn_id = os.environ.get("AETHERFY_SPAWN_ID")
    api_key = os.environ.get("AETHERFY_API_KEY")
    if not (api_url and spawn_id and api_key):
        raise PayloadUnavailable(
            "No payload file (AETHERFY_SPAWN_PAYLOAD_PATH) and no way to fetch "
            "one: AETHERFY_API_URL, AETHERFY_SPAWN_ID and AETHERFY_API_KEY must "
            "all be set for the HTTP fallback. All four are set by the platform "
            "on an agent machine."
        )

    # QUOTED, like every id this module puts in a path. Left raw, an id
    # holding a slash or a `..` silently becomes a request to a DIFFERENT
    # route — the client normalises the path before it leaves — and the
    # answer is then parsed as though it were this one. A 404 is the
    # honest outcome; a wrong object read as the right one is not.
    url = "{0}/deployments/{1}/payload".format(
        api_url.rstrip("/"), quote(spawn_id, safe="")
    )
    status, body, _ = _http.request_json(
        "GET", url, api_key=api_key, ua=_user_agent(), repeatable=True
    )
    if status != 200:
        raise PayloadUnavailable(
            "The payload fallback ({0}) answered {1}: {2}".format(
                url, status, _message_of(body) or "no message"
            )
        )
    if not isinstance(body, dict):
        raise PayloadUnavailable(
            "The payload fallback ({0}) answered 200 with a body that is not "
            "an object.".format(url)
        )
    fetched = body.get("payload")
    return fetched if isinstance(fetched, dict) else {}


def machine() -> MachineShape:
    """
    Return the shape of the machine this run is executing on.

    Ints, not the strings the environment carries — the width of a pool is
    arithmetic, and ``"4" * 8`` is a bug that produces a string of length 8
    rather than a crash.

    :raises NotRunningOnAgent: a variable the platform always sets is missing.
    """
    vcpus = _require("AETHERFY_VCPUS", "this machine's vCPU count")
    memory_mb = _require("AETHERFY_MEMORY_MB", "this machine's memory")
    region = _require("AETHERFY_REGION", "this machine's region")
    try:
        return MachineShape(vcpus=int(vcpus), memory_mb=int(memory_mb), region=region)
    except ValueError as exc:
        raise AgentError(
            "AETHERFY_VCPUS ({0}) and AETHERFY_MEMORY_MB ({1}) must both be "
            "whole numbers: {2}".format(vcpus, memory_mb, exc)
        )


def fan_out(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    width: Optional[int] = None,
    kind: str = "threads",
) -> List[R]:
    """
    Run ``fn`` over ``items`` on an in-machine pool, results in INPUT order.

    Fanning out inside the machine is the cheap kind of parallelism on
    Aetherfy: no extra machines, no cold starts, and no awake time beyond the
    run itself. Spawning is for isolation and independent lifecycles — see
    :func:`spawn`.

    ``width`` DEFAULTS FROM ``kind``, because the pool already says what the
    work is. ``kind="threads"`` (the default) is for waiting — model calls,
    HTTP, database round trips — and defaults to ``vcpus * 8``.
    ``kind="processes"`` is only ever worth its pickling cost for CPU-bound
    work, so it defaults to ``vcpus``: one worker per core, no oversubscription.
    Pass ``width`` explicitly to override either. With ``kind="processes"``,
    ``fn`` and every item must be picklable.

    NO FAILURE IS SWALLOWED. If any call raises, the exception from the
    LOWEST-INDEXED failing item is re-raised once every worker has finished —
    deterministic, rather than whichever thread happened to lose the race.
    Work already in flight is not cancelled: a thread cannot be interrupted,
    and pretending otherwise would leak half-finished work.

    Prints one line to stdout before running, flushed, so the run's logs record
    how wide it went.
    """
    if kind not in ("threads", "processes"):
        raise ValueError(
            "kind must be 'threads' (I/O-bound) or 'processes' (CPU-bound), "
            "not {0!r}".format(kind)
        )

    materialized = list(items)
    shape = machine()

    if width is None:
        # THE DEFAULT FOLLOWS THE POOL, because the pool already declares the
        # shape of the work. Choosing processes means the work is CPU-bound by
        # definition — that is the only reason to pay for pickling and a second
        # interpreter — and CPU-bound work wider than the core count only adds
        # context switching. Threads are for waiting, so they run wide.
        width = (
            shape.vcpus
            if kind == "processes"
            else shape.vcpus * _IO_BOUND_WIDTH_PER_VCPU
        )
    width = int(width)
    if width < 1:
        raise ValueError("width must be at least 1, not {0}".format(width))

    print(
        _FAN_OUT_LINE.format(
            width=width,
            vcpus=shape.vcpus,
            memory_mb=shape.memory_mb,
            n=len(materialized),
        ),
        flush=True,
    )

    if not materialized:
        return []

    executor = ThreadPoolExecutor if kind == "threads" else ProcessPoolExecutor
    with executor(max_workers=width) as pool:
        futures = [pool.submit(fn, item) for item in materialized]
        # Iterating in submission order makes the re-raise deterministic: the
        # first exception a caller sees is always the earliest item's.
        return [future.result() for future in futures]


def spawn(child: str, payload: Optional[Dict[str, Any]] = None) -> Spawn:
    """
    Ask the control plane to run another agent, and return once recorded.

    ``child`` is the id or name of an agent you own, of either type: a task's
    run forks its command, a service's run is a request to its own
    ``POST /aetherfy/run``. The parent is this machine's own agent, recorded on
    the run and never on the child. The child runs in this agent's region, and the
    two must be connected by ``spawn.workers`` in ``aetherfy.yaml`` for the
    call to be allowed.

    ACCEPTANCE IS NOT EXECUTION. The returned :class:`~.models.Spawn` says the
    run was recorded. A run over the account's in-flight limit is not refused:
    it WAITS in the account's line (``status`` ``"waiting"``, ``position`` its
    place there, 1 being next), with no machine and nothing billed, and starts
    when a slot frees; otherwise ``status`` is ``"queued"`` and it goes straight
    to a machine. :func:`wait` includes the time in the line. A spawn is refused
    only when the line itself is full (:class:`RunLineFull`), and nothing is
    retried for you then. Wait for the run and read its state.

    Keep the payload small: it is for parameters and references, not data. Pass
    anything large by reference to a collection.

    This uses ``AETHERFY_API_KEY``: on a task machine that is the run's own key,
    in a service the release's. Inside a service's run handler, spawn through
    :func:`service_run` so the new run belongs to the run being handled.

    :raises PayloadTooLarge: 413, the payload crossed the inline cap.
    :raises RunLineFull: 429, the account's line of waiting runs is full.
    :raises SpawnError: any other refusal — read ``error_code``, not the prose.
    :raises AgentTransportError: the request never reached the control plane.
    """
    return _spawn_as(
        _require("AETHERFY_API_KEY", "the key a spawn authenticates with"),
        child,
        payload,
    )


def _spawn_as(api_key: str, child: str, payload: Optional[Dict[str, Any]]) -> Spawn:
    """The one spawn, for whichever key it is made with: :func:`spawn` passes
    the environment's, a :class:`ServiceRun` its run's."""
    url = _spawn_url()

    # THE SPAWN'S OWN ID, made once per call. The control plane stores it with
    # the run it makes and answers a repeat of the same id with that run, so a
    # spawn whose answer was lost on the way (a timeout after the request left)
    # is sent again safely and never makes a second run.
    status, body, retry_after = _http.request_json(
        "POST",
        url,
        api_key=api_key,
        ua=_user_agent(),
        body={
            "child_agent_id": child,
            "payload": payload or {},
            "request_id": str(uuid.uuid4()),
        },
        repeatable=True,
    )

    if status in (200, 201, 202):
        if not isinstance(body, dict):
            raise SpawnError(
                "The spawn was accepted with {0} but the body was not an "
                "object, so the run cannot be identified.".format(status),
                status_code=status,
            )
        # A run waiting in the line has no job yet, and a run that is not
        # waiting has no place in it: null is an answer, not the string "None".
        job_id = body.get("job_id")
        position = body.get("position")
        return Spawn(
            spawn_id=str(body.get("spawn_id")),
            job_id=None if job_id is None else str(job_id),
            child_agent_id=str(body.get("child_agent_id")),
            region=str(body.get("region")),
            status=str(body.get("status")),
            workspace=str(body.get("workspace")),
            estimated_start=body.get("estimated_start"),
            position=position if _is_int(position) else None,
            raw=body,
        )

    detail = _detail_of(body)
    message = detail.get("message") or "Spawning '{0}' failed with status {1}.".format(
        child, status
    )
    code = detail.get("code")

    # THE CODE DECIDES, NOT THE STATUS ALONE. A status is a category the
    # platform reuses; the code is the thing it promises not to rename. Mapping
    # on 413 alone would stamp RUN_PAYLOAD_TOO_LARGE onto the next unrelated
    # 413 the control plane grows, and the caller would branch on a lie it
    # could not see through — the typed error carries the wrong code AND the
    # right message. An unrecognised pairing falls through to SpawnError, which
    # reports exactly what arrived.
    if status == 413 and code == RUN_PAYLOAD_TOO_LARGE:
        raise PayloadTooLarge(
            message,
            payload_bytes=detail.get("payload_bytes"),
            max_bytes=detail.get("max_bytes"),
            details=detail,
        )
    if status == 429 and code == AGENT_RUN_LINE_FULL:
        raise RunLineFull(
            message,
            waiting_count=detail.get("waiting_count"),
            max_waiting_runs=detail.get("max_waiting_runs"),
            in_flight_count=detail.get("in_flight_count"),
            max_in_flight_runs=detail.get("max_in_flight_runs"),
            limit=detail.get("limit"),
            details=detail,
            retry_after=retry_after,
        )
    raise SpawnError(
        message,
        status_code=status,
        error_code=code,
        details=detail,
        retry_after=retry_after,
    )


def write_result(value: Any) -> None:
    """
    Return ``value`` to whoever started this run.

    The mirror of :func:`payload`: Aetherfy puts the path of a file in
    ``AETHERFY_SPAWN_RESULT_PATH`` before the entrypoint starts, and stores
    what was written there once the process ends. Nothing crosses the network,
    and there is no call to make — a parent reads it back from the run itself
    with :func:`result` or :func:`wait`.

    RETURNING NOTHING IS THE NORMAL CASE, so most tasks never call this. A run
    that writes no file is recorded as returning nothing, which is not the same
    as failing to return something. Passing ``None`` records the same thing:
    the platform reads a literal ``null`` as "returned nothing", and an empty
    column already says that.

    The result is for answers and references, not data — it shares the payload's
    inline cap, one number bounding both directions. Anything larger belongs in
    a collection in your Aetherfy vector database, with its id in the result.

    THE LAST CALL WINS. The file is overwritten, so calling this twice returns
    the second value; there is no accumulation and no merge.

    :raises ResultTooLarge: the encoded result crosses this machine's cap. The
        platform would have dropped it and recorded ``result_error`` instead —
        this refuses at the write so the caller can shrink it.
    :raises NotRunningOnAgent: ``AETHERFY_SPAWN_RESULT_PATH`` is not set, so
        there is nowhere to put an answer.
    :raises ValueError: ``value`` is not JSON — including a ``NaN`` or an
        infinity, which Python would otherwise happily write as tokens no
        other JSON reader accepts.
    :raises TypeError: ``value`` holds an object json cannot encode.
    """
    # DELIBERATELY NOT THE DOCS' HAND-ROLLED VERSION, which no-ops when the
    # variable is missing. That is the right shape inline in a customer's own
    # script, where the author can see the fallback; it is the wrong shape for
    # a library, which would be silently discarding the one value it was
    # called to deliver. The variable is absent only on a machine that has no
    # result path to offer — off Aetherfy entirely, or a task machine whose
    # supervisor could not prepare the file — and in both cases a run that
    # thinks it answered did not.
    path = _require(
        "AETHERFY_SPAWN_RESULT_PATH",
        "the path this run writes its answer to",
        _NO_RESULT_PATH_REMEDY,
    )

    # allow_nan=False ON PURPOSE. Python's json writes NaN, Infinity and
    # -Infinity as bare tokens, which are not JSON: the platform's own reader
    # accepts them (it is Python too) but the value would then reach a
    # dashboard or a JavaScript caller as a parse error on a field nobody
    # touched. Refusing here is the same answer the JavaScript helper gives for
    # free, since JSON.stringify has no such extension.
    encoded = json.dumps(value, allow_nan=False).encode("utf-8")

    max_bytes = _inline_max_bytes()
    # ONE encode, measured and written. Encoding twice is how a size check ends
    # up describing bytes other than the ones that land on disk.
    if max_bytes is not None and len(encoded) > max_bytes:
        raise ResultTooLarge(
            "This run's result is {0} bytes and the inline cap is {1}. The "
            "result is for answers and references, not data: write the data to "
            "a collection and return its id.".format(len(encoded), max_bytes),
            result_bytes=len(encoded),
            max_bytes=max_bytes,
        )

    with open(path, "wb") as handle:
        handle.write(encoded)


def result(run_id: str) -> Run:
    """
    Read one run back, with whatever it returned.

    Answers immediately with the run as it stands. A run that has not ended has
    no result yet: its ``state`` is ``"waiting"`` while it waits in the
    account's line and ``"active"`` while it is executing, and it ends
    ``"completed"``, ``"failed"`` or ``"cancelled"``. :func:`wait` is the same
    read with the waiting done server-side, and is what to use when the answer
    is the point.

    ``run_id`` is a run's id — ``Spawn.spawn_id`` from a :func:`spawn`, or, on a
    task machine, the id of this run itself in ``AETHERFY_DEPLOYMENT_ID``.

    :raises RunNotFound: 404, no run has that id.
    :raises RunAccessDenied: 403 ``DEPLOYMENT_ACCESS_DENIED``, the run belongs to
        another account (an account key was used).
    :raises RunReadError: 403 ``AUTH_AGENT_KEY_OUT_OF_SCOPE`` when this machine's
        own ``AETHERFY_API_KEY`` reads a run that key may not read (which
        ones it may: https://docs.aetherfy.com/platform/api-keys#agent-scoped-keys-issued-by-aetherfy).
    :raises RunReadError: any other refusal — read ``error_code``, not the prose.
    :raises AgentTransportError: the request never reached the control plane.
    """
    return _result_as(_environment_key("the key a run is read with"), run_id)


def _result_as(api_key: str, run_id: str) -> Run:
    return _read_run(_run_url(run_id), api_key)


def wait(run_id: str, timeout_seconds: int = WAIT_TIMEOUT_DEFAULT_SECONDS) -> Run:
    """
    Hold one request open until the run finishes, then return it.

    The read side of the result path, and the reason a parent does not poll:
    without it every caller writes the same loop with its own interval, and all
    of them pay for the privilege of not knowing yet.

    THE TIME IN THE LINE IS PART OF WHAT IS WAITED FOR. A run over the account's
    in-flight limit waits in the account's line (state ``waiting``), and this
    keeps holding through it. It returns when the run has ended: ``completed``,
    ``failed`` or ``cancelled``.

    A TIMEOUT IS NOT AN ERROR. If the run has not finished in ``timeout_seconds``
    this returns it exactly as it stands — read ``Run.state``, which is
    ``waiting`` while the run waits in the line, ``active`` while it is
    executing, and ``completed``, ``failed`` or ``cancelled`` when it is over,
    and call again. Waiting longer than the maximum is a second call, not a
    bigger number: the request is held open, and anything longer is cut by the
    network in front of Aetherfy.

    ONE CONNECTION FAILURE IS NOT RETRIED HERE, unlike every other call in this
    module. A retry would silently hold a second full timeout and hand back a
    run up to twice as late as the number the caller passed; the bound this
    function's argument promises is worth more than the blip it would paper
    over. Call again.

    :raises ValueError: ``timeout_seconds`` is outside the server's bound. The
        argument is wrong, and no request is sent.
    :raises WaitTimeoutInvalid: 422, the server rejected the timeout anyway —
        its bound moved and this helper's copy is stale.
    :raises RunNotFound: 404, no run has that id.
    :raises RunAccessDenied: 403 ``DEPLOYMENT_ACCESS_DENIED``, the run belongs to
        another account (an account key was used).
    :raises RunReadError: 403 ``AUTH_AGENT_KEY_OUT_OF_SCOPE`` when this machine's
        own ``AETHERFY_API_KEY`` reads a run that key may not read (which
        ones it may: https://docs.aetherfy.com/platform/api-keys#agent-scoped-keys-issued-by-aetherfy).
    :raises RunReadError: any other refusal.
    :raises AgentTransportError: the request never reached the control plane.
    """
    return _wait_as(
        _environment_key("the key a run is read with"), run_id, timeout_seconds
    )


def _wait_as(api_key: str, run_id: str, timeout_seconds: int) -> Run:
    timeout_seconds = int(timeout_seconds)
    if not WAIT_TIMEOUT_MIN_SECONDS <= timeout_seconds <= WAIT_TIMEOUT_MAX_SECONDS:
        raise ValueError(
            "timeout_seconds must be between {0} and {1}, not {2}. Waiting "
            "longer is another call to wait(), not a bigger number.".format(
                WAIT_TIMEOUT_MIN_SECONDS, WAIT_TIMEOUT_MAX_SECONDS, timeout_seconds
            )
        )
    return _read_run(
        "{0}/wait?timeout_seconds={1}".format(_run_url(run_id), timeout_seconds),
        api_key,
        timeout=timeout_seconds + _WAIT_TRANSPORT_MARGIN_SECONDS,
        repeatable=False,
    )


def cancel(run_id: str) -> Cancellation:
    """
    Cancel a run, whether it is waiting in the account's line or executing.

    ``run_id`` is a run's id — ``Spawn.spawn_id`` from a :func:`spawn`. The
    answer is a :class:`~.models.Cancellation`: its ``outcome`` is
    ``"cancelled"`` (the run is cancelled now), ``"stopping"`` (it was
    executing and is being stopped; it ends ``cancelled``) or
    ``"already_finished"`` (it had ended before this call). A run that had
    already ended is NOT an error, so cancelling twice, or racing the run's own
    end, needs no handling. ``cancelled_count`` counts every run this call
    cancelled, the runs started below the cancelled one (its descendants)
    included: a cancelled run's own spawns are cancelled with it.

    A run cancelled from outside ends ``cancelled``, on every path: by a person,
    by a run above it, or as part of a cascade. A cancelled run's key stops
    working at once, so whatever it was still starting is refused.

    WHICH RUNS A KEY MAY CANCEL. An account's key may cancel any run of the
    account. An agent's own key (``AETHERFY_API_KEY`` on an agent machine) may
    cancel only the runs below its own run, not the run itself and not its
    siblings. Inside a service's run handler, cancel through :func:`service_run`.

    This is for runs. A release (a deploy) is cancelled with the CLI's
    ``afy cancel <agent>``.

    :raises RunNotFound: 404, no run has that id.
    :raises RunAccessDenied: 403, the run belongs to another account
        (``error_code`` ``DEPLOYMENT_ACCESS_DENIED``), or an agent's key was
        used on a run that is not below its own (``error_code``
        ``AUTH_AGENT_KEY_OUT_OF_SCOPE``).
    :raises NotARun: 409 ``DEPLOYMENT_NOT_A_RUN``, the id names a release.
    :raises RunCancelError: any other refusal — read ``error_code``, not the
        prose. Note that :class:`RunNotFound` and :class:`RunAccessDenied` are
        not its subclasses; catch :class:`AgentError` to see all of them.
    :raises ValueError: ``run_id`` is empty. No request is sent.
    :raises AgentTransportError: the request never reached the control plane.
    """
    return _cancel_as(_environment_key("the key a cancel authenticates with"), run_id)


def _cancel_as(api_key: str, run_id: str) -> Cancellation:
    url = "{0}/cancel".format(_run_url(run_id))
    # REPEATABLE: a second cancel of the same run changes nothing (it answers
    # `stopping` with a count of 0, or `already_finished`), so a cancel whose
    # answer was lost is sent again rather than reported as maybe applied.
    status, body, retry_after = _http.request_json(
        "POST", url, api_key=api_key, ua=_user_agent(), repeatable=True
    )

    if status == 200:
        run = body.get("run") if isinstance(body, dict) else None
        outcome = body.get("outcome") if isinstance(body, dict) else None
        if not (isinstance(body, dict) and isinstance(run, dict) and outcome):
            raise RunCancelError(
                "Cancelling the run answered 200 with a body that has no run "
                "and outcome, so what the cancel did is not known.",
                status_code=status,
            )
        message = body.get("message")
        cancelled_count = body.get("cancelled_count")
        return Cancellation(
            run=_run_from(run),
            outcome=str(outcome),
            message=message if isinstance(message, str) else "",
            cancelled_count=cancelled_count if _is_int(cancelled_count) else 0,
        )

    detail = _detail_of(body)
    message = detail.get(
        "message"
    ) or "Cancelling the run failed with status {0}.".format(status)
    code = detail.get("code")

    # THE CODE DECIDES, NOT THE STATUS ALONE, as in spawn() and the reads. The
    # 404 and the DEPLOYMENT_ACCESS_DENIED 403 are the read family's own, shared
    # with it; the out-of-scope 403 is mapped here and not on a read, because a
    # cancel has one problem to report with it (this key may not cancel that
    # run) and a read has another.
    refusal = _shared_run_refusal(status, code, message, detail)
    if refusal is not None:
        raise refusal
    if status == 403 and code == AUTH_AGENT_KEY_OUT_OF_SCOPE:
        raise RunAccessDenied(
            message, details=detail, error_code=AUTH_AGENT_KEY_OUT_OF_SCOPE
        )
    if status == 409 and code == DEPLOYMENT_NOT_A_RUN:
        raise NotARun(message, details=detail)
    raise RunCancelError(
        message,
        status_code=status,
        error_code=code,
        details=detail,
        retry_after=retry_after,
    )


# --- MESSAGES BETWEEN RUNNING AGENTS -------------------------------------------
#
# A run has a mailbox on the control plane. Any run of its run tree can send it a
# message (send), a person on the account can too, and the run reads its own
# mailbox in order (receive), the oldest unread message first.
#
# WAITING HOLDS NOTHING OPEN TO THE CONTROL PLANE. A receive() with nothing to
# read makes ONE short request that records the wait, then blocks on a LOCAL call
# to this machine's supervisor (AETHERFY_SUPERVISOR_WAIT_URL, loopback only). A
# message, a child's end or the receive's own deadline makes the control plane
# tell the supervisor, which releases the call; the helper then asks once more and
# gets what is there. While it waits, a task's machine is put to sleep after a
# short grace and nothing is billed for running time; the message wakes it.
#
# AT LEAST ONCE. A message stays unread until the NEXT call acknowledges it, so
# one whose answer was lost on the way is handed over again rather than lost. A
# run that restarts (restarted() is True) reads what it had already been handed
# with messages().

#: Where the supervisor holds a waiting receive(): set by the platform on the
#: run's process (a task) or on the service's server (a service run's handler).
SUPERVISOR_WAIT_URL_VARIABLE = "AETHERFY_SUPERVISOR_WAIT_URL"

#: The machine this process runs on, as the platform names it to the control plane: set by a supervisor that holds
#: waits (run-request-v4). A wait records the machine whose supervisor holds its call; a service release's wait names
#: it with this, since a release has no one machine of its own.
INSTANCE_ID_VARIABLE = "AETHERFY_INSTANCE_ID"

#: How many times the local wait is asked again after its connection dropped,
#: before the helper gives up. The supervisor remembers a wake that came while
#: the call was being made again, so asking again loses nothing.
_LOCAL_WAIT_ATTEMPTS = 3

# What each run's process has handed to its code and not yet acknowledged, and
# the lock that makes one receive() at a time per run: a run's mailbox has one
# reader, and two waits at once would strand one of them.
_pending_acks: Dict[str, List[str]] = {}
# The wait a receive() may have recorded and did not see through: its request,
# or its local call, failed after the request may have arrived. The next call
# names it again, so the control plane answers with that same wait instead of
# refusing a second one while the first is outstanding.
_pending_waits: Dict[str, str] = {}
_receive_locks: Dict[str, threading.Lock] = {}
_receive_locks_guard = threading.Lock()


def _receive_lock(run_id: str) -> threading.Lock:
    with _receive_locks_guard:
        return _receive_locks.setdefault(run_id, threading.Lock())


def _own_run_id(purpose: str) -> str:
    """The deployment whose mailbox this process reads: AETHERFY_DEPLOYMENT_ID, and
    nothing else. On a task machine the supervisor sets it to the run being
    executed, on every run; on a service machine the platform sets it to the
    release the machine serves, whose mailbox the service's own code reads.
    (AETHERFY_SPAWN_ID names a run only, and a service machine has none.)"""
    return _require("AETHERFY_DEPLOYMENT_ID", purpose)


def send(run_id: str, body: Any, *, id: Optional[str] = None) -> Sent:
    """
    Send one message to a run: the run that started this one (:func:`parent`),
    a run it started, or any run of the same run tree.

    ``body`` is any JSON value, up to the inline cap a run's payload has. ``id``
    is the message's own id, a UUID (anything else is refused, 422
    VALIDATION_ERROR); the helper makes one when none is given. Sending
    the same id again is the same message: it is stored once, so a send whose
    answer was lost can simply be repeated.

    The recipient reads it with :func:`receive`, in order. A recipient that is
    waiting is woken, and a recipient that has ended refuses it.

    :raises MessageTooLarge: 413, the message is larger than the cap.
    :raises MessageRateLimited: 429, this run sent too many in the last minute.
    :raises MailboxFull: 429, the recipient has too many unread messages.
    :raises MessageStorageFull: 429, the account's stored-logs budget is full.
    :raises RecipientEnded: 409, the recipient has ended.
    :raises RunNotFound: 404, no run has that id.
    :raises RunAccessDenied: 403, the run is outside this run's tree.
    """
    return _send_as(
        _environment_key("the key a message is sent with"), run_id, body, id
    )


def _send_as(api_key: str, run_id: str, body: Any, message_id: Optional[str]) -> Sent:
    if not run_id:
        raise ValueError("run_id must name the run the message is for.")
    status, answer, retry_after = _http.request_json(
        "POST",
        "{0}/messages".format(_run_url(run_id)),
        api_key=api_key,
        ua=_user_agent(),
        body={"id": message_id or str(uuid.uuid4()), "body": body},
        repeatable=True,
    )
    if status in (200, 201):
        message = answer.get("message") if isinstance(answer, dict) else None
        recipient = answer.get("recipient") if isinstance(answer, dict) else None
        if not (isinstance(message, dict) and isinstance(recipient, dict)):
            raise MessageError(
                "The message was accepted with {0} but the answer names no "
                "message and recipient.".format(status),
                status_code=status,
            )
        seq = message.get("seq")
        return Sent(
            message_id=str(message.get("id")),
            seq=seq if _is_int(seq) else 0,
            recipient_id=str(recipient.get("id")),
            recipient_agent=recipient.get("agent"),
            recipient_state=recipient.get("state"),
            raw=answer,
        )
    raise _message_refusal(status, answer, retry_after, "Sending the message")


def receive(
    *, timeout_seconds: Optional[float] = None, from_run: Optional[str] = None
) -> Optional[Union[Message, Ended]]:
    """
    The next message to this run, the oldest unread first; or the end of a run
    this run started (:class:`Ended`), when one ends; or ``None`` when
    ``timeout_seconds`` ran out first.

    With nothing to read it WAITS: with no timeout, until something arrives;
    with ``timeout_seconds=0``, not at all (one check). While it waits, nothing
    is held open to the control plane, and a task run's machine sleeps after a
    short grace, unbilled; a message wakes it where it left off.

    ``from_run`` reads only what that run sent (and its end, if it is a run this
    one started). Other senders' messages wait their turn: nothing is skipped.

    One receive() at a time per run: a second call in another thread waits for
    the first. A message is acknowledged on the NEXT call, so one whose answer
    was lost is handed over again; ``id`` tells a repeat apart.

    :raises SenderEnded: 409, ``from_run`` has ended and left nothing unread.
    :raises RunNotFound, RunAccessDenied: as for every run call.
    :raises MessageError: any other refusal, with its code.
    """
    return _receive_as(
        _environment_key("the key a message is received with"),
        _own_run_id("the run whose messages are received"),
        timeout_seconds,
        from_run,
    )


def _receive_as(
    api_key: str,
    run_id: str,
    timeout_seconds: Optional[float],
    from_run: Optional[str],
) -> Optional[Union[Message, Ended]]:
    if timeout_seconds is not None and timeout_seconds < 0:
        raise ValueError(
            "timeout_seconds must be 0 or more, or None to wait for as long as it takes."
        )
    # A call that may wait needs the supervisor's local call to wait on. Refused
    # BEFORE any request: the request records the wait and puts the run to
    # sleep after a short grace, under a process that would not be waiting.
    local_wait_url = None
    if timeout_seconds != 0:
        local_wait_url = os.environ.get(SUPERVISOR_WAIT_URL_VARIABLE)
        if not local_wait_url:
            raise NotRunningOnAgent(
                SUPERVISOR_WAIT_URL_VARIABLE,
                "the local call a waiting receive() blocks on",
                "Redeploy the agent: it runs on a platform older than waiting "
                "for a message. receive(timeout_seconds=0) only checks, and "
                "works without it.",
            )
    with _receive_lock(run_id):
        pending = _pending_acks.setdefault(run_id, [])
        while True:
            wait_id = _pending_waits.get(run_id) or str(uuid.uuid4())
            request: Dict[str, Any] = {"ack": list(pending), "wait_id": wait_id}
            if from_run is not None:
                request["from_run"] = from_run
            if timeout_seconds is not None:
                request["timeout_seconds"] = timeout_seconds
            instance_id = os.environ.get(INSTANCE_ID_VARIABLE)
            if instance_id:
                request["instance_id"] = instance_id
            # Kept until this call sees the wait through: if the request fails
            # after it may have arrived, the next call names the same wait.
            _pending_waits[run_id] = wait_id
            status, answer, retry_after = _http.request_json(
                "POST",
                "{0}/messages/receive".format(_run_url(run_id)),
                api_key=api_key,
                ua=_user_agent(),
                body=request,
                repeatable=True,
            )
            if status != 200 or not isinstance(answer, dict):
                # An answer, even a refusal, is the whole outcome: no wait of
                # this id is outstanding.
                _pending_waits.pop(run_id, None)
                raise _message_refusal(
                    status, answer, retry_after, "Receiving a message"
                )
            # What this call acknowledged is acknowledged: the next call starts clean.
            del pending[:]
            if not answer.get("waiting"):
                _pending_waits.pop(run_id, None)
            if isinstance(answer.get("message"), dict):
                message = _message_from(answer["message"])
                pending.append(message.id)
                return message
            if isinstance(answer.get("ended"), dict):
                ended = answer["ended"]
                child = str(ended.get("run_id"))
                pending.append(child)
                return Ended(run_id=child, state=str(ended.get("state")), raw=ended)
            if not answer.get("waiting"):
                return None
            released = _wait_locally(
                local_wait_url or "", api_key, str(answer.get("wait_id") or wait_id)
            )
            # Seen through: the wait was released.
            _pending_waits.pop(run_id, None)
            if released == "timeout":
                return None
            # Released by a message or a child's end: ask again, and it is there.


def _wait_locally(base: str, api_key: str, wait_id: str) -> str:
    """Block on the supervisor's local call at ``base`` until the control plane releases this wait. Returns why it
    was released ("message", "ended" or "timeout"). No timeout of its own: the deadline, when there is one, is the
    control plane's, and a machine asleep under this call does not count the time it slept.
    """
    last_error: Optional[Exception] = None
    for _ in range(_LOCAL_WAIT_ATTEMPTS):
        try:
            status, answer, _ = _http.request_json(
                "POST",
                "{0}/wait".format(base.rstrip("/")),
                api_key=api_key,
                ua=_user_agent(),
                body={"wait_id": wait_id},
                timeout=None,
                repeatable=True,
            )
        except AgentTransportError as error:
            last_error = error
            continue
        if status == 200 and isinstance(answer, dict):
            return str(answer.get("released") or "message")
        raise MessageError(
            "This run's supervisor answered {0} to the local wait.".format(status),
            status_code=status,
        )
    raise AgentTransportError(
        "The local wait on this run's supervisor did not hold after {0} attempts: "
        "{1}".format(_LOCAL_WAIT_ATTEMPTS, last_error)
    )


def messages(*, after: int = 0) -> Iterator[Message]:
    """
    Every message this run's mailbox holds, in order, read or not: the history
    a run that restarted rebuilds its state from. ``after`` skips the first
    ``after`` (by ``seq``). Messages are kept as long as the account's plan
    keeps logs, and for as long as the run lives.
    """
    return _messages_as(
        _environment_key("the key a mailbox is read with"),
        _own_run_id("the run whose mailbox is read"),
        after,
    )


def _messages_as(api_key: str, run_id: str, after: int) -> Iterator[Message]:
    cursor = after
    while True:
        status, answer, retry_after = _http.request_json(
            "GET",
            "{0}/messages?after={1}".format(_run_url(run_id), int(cursor)),
            api_key=api_key,
            ua=_user_agent(),
            repeatable=True,
        )
        if status != 200 or not isinstance(answer, dict):
            raise _message_refusal(status, answer, retry_after, "Reading the mailbox")
        page = answer.get("messages")
        if not isinstance(page, list) or not page:
            return
        for item in page:
            if isinstance(item, dict):
                message = _message_from(item)
                cursor = max(cursor, message.seq)
                yield message
        if answer.get("next_after") is None:
            return


def restarted() -> bool:
    """
    Whether this process is a restart of its run: the run waited for a message,
    its machine's memory was not kept, and the platform started it again. Its
    code then rebuilds what it needs from its history (:func:`messages`) and its
    payload; nothing in memory survived. False on a run's first start.
    """
    run = result(_own_run_id("the run asked about"))
    attempt = run.raw.get("attempt")
    return _is_int(attempt) and attempt > 1


def parent() -> Optional[str]:
    """
    The id of the run that started this one, or ``None`` for a run started by a
    person, a schedule or a service's public request. The usual recipient of a
    child's progress and questions: ``send(parent(), {...})``.
    """
    run = result(_own_run_id("the run whose parent is read"))
    found = run.raw.get("spawned_by_deployment_id")
    return str(found) if found else None


def _message_from(item: Dict[str, Any]) -> Message:
    seq = item.get("seq")
    sender_run = item.get("sender_run_id")
    return Message(
        id=str(item.get("id")),
        seq=seq if _is_int(seq) else 0,
        sender=str(item.get("sender") or ("run" if sender_run else "person")),
        sender_run_id=str(sender_run) if sender_run else None,
        body=item.get("body"),
        created_at=item.get("created_at"),
        raw=item,
    )


def _message_refusal(
    status: int, body: Any, retry_after: Optional[float], doing: str
) -> AgentError:
    """The error a refused message call raises: THE CODE DECIDES, NOT THE STATUS ALONE, as in every call here."""
    detail = _detail_of(body)
    message = detail.get("message") or "{0} failed with status {1}.".format(
        doing, status
    )
    code = detail.get("code")
    shared = _shared_run_refusal(status, code, message, detail)
    if shared is not None:
        return shared
    if status == 403 and code == AUTH_AGENT_KEY_OUT_OF_SCOPE:
        return RunAccessDenied(
            message, details=detail, error_code=AUTH_AGENT_KEY_OUT_OF_SCOPE
        )
    if status == 413 and code == RUN_MESSAGE_TOO_LARGE:
        return MessageTooLarge(message, details=detail)
    if status == 429 and code == RUN_MESSAGE_RATE_LIMITED:
        return MessageRateLimited(message, details=detail, retry_after=retry_after)
    if status == 429 and code == RUN_MAILBOX_FULL:
        return MailboxFull(message, details=detail)
    if status == 429 and code == RUN_MESSAGE_STORAGE_FULL:
        return MessageStorageFull(message, details=detail)
    if status == 409 and code == RUN_MESSAGE_RECIPIENT_ENDED:
        return RecipientEnded(message, details=detail)
    if status == 409 and code == RUN_MESSAGE_SENDER_ENDED:
        return SenderEnded(message, details=detail)
    return MessageError(
        message,
        status_code=status,
        error_code=code,
        details=detail,
        retry_after=retry_after,
    )


class HeaderSource(Protocol):
    """
    The headers of a request, as a web framework hands them over: anything with
    ``items()`` yielding ``(name, value)`` pairs. A plain ``dict``, Starlette's
    and FastAPI's ``request.headers``, Flask's ``request.headers`` and the
    standard library's ``http.server`` headers all are. Names are compared
    without regard to case.
    """

    def items(self) -> Iterable[Tuple[str, str]]:
        ...


class ServiceRun:
    """
    The run a service is handling right now, as a handle to spawn, read, wait
    on and cancel with THAT RUN'S OWN KEY. Built by :func:`service_run`.

    ``spawn``, ``result``, ``wait`` and ``cancel`` are the module-level
    functions of the same names, with the same signatures, the same endpoints,
    the same errors and the same documentation; only the key differs. What the
    handler starts through the handle belongs to the run: it is in the run's
    tree, so cancelling the run cancels it, and the run's key may read and wait
    on it. ``run_id`` is the id of the run being handled.

    WHY A HANDLE YOU PASS, NOT A CONTEXT THE LIBRARY KEEPS. A service answers
    many runs at once on one process, so which run a call belongs to has to come
    from somewhere. An implicit context (``contextvars``) is lost the moment
    work is handed to a thread pool or to ``run_in_executor``: the call then
    falls back, without an error, to the release's key, and what it starts lands
    outside the run's tree, where cancelling the run does not reach it. An
    explicit handle is either in scope or a visible ``NameError``.

    TWO KEYS IN ONE HANDLER. The module-level functions read
    ``AETHERFY_API_KEY``, which in a service is the release's: right for the
    service's own traffic, wrong for a run. Do not mix them on one run. A run
    spawned through the handle belongs to the run's tree and not to the
    release's, so a module-level :func:`wait` on it is refused (403
    ``AUTH_AGENT_KEY_OUT_OF_SCOPE``, a :class:`RunReadError`): wait through the
    handle.

    THE KEY LASTS AS LONG AS THE RUN. It stops working when the run ends,
    answered, cancelled or failed, so do not keep the handle past the handler.
    """

    __slots__ = ("run_id", "_api_key")

    def __init__(self, run_id: str, api_key: str) -> None:
        self.run_id = run_id
        self._api_key = api_key

    def __repr__(self) -> str:
        # The key is a credential: it must not reach a log through a repr.
        return "ServiceRun(run_id={0!r})".format(self.run_id)

    def spawn(self, child: str, payload: Optional[Dict[str, Any]] = None) -> Spawn:
        """:func:`spawn`, with this run's key: the new run belongs to this run."""
        return _spawn_as(self._api_key, child, payload)

    def result(self, run_id: str) -> Run:
        """:func:`result`, with this run's key."""
        return _result_as(self._api_key, run_id)

    def wait(
        self, run_id: str, timeout_seconds: int = WAIT_TIMEOUT_DEFAULT_SECONDS
    ) -> Run:
        """:func:`wait`, with this run's key."""
        return _wait_as(self._api_key, run_id, timeout_seconds)

    def cancel(self, run_id: str) -> Cancellation:
        """:func:`cancel`, with this run's key: it may cancel the runs below
        this run."""
        return _cancel_as(self._api_key, run_id)

    def send(self, run_id: str, body: Any, *, id: Optional[str] = None) -> Sent:
        """:func:`send`, from this run: to a run of its run tree."""
        return _send_as(self._api_key, run_id, body, id)

    def receive(
        self, *, timeout_seconds: Optional[float] = None, from_run: Optional[str] = None
    ) -> Optional[Union[Message, Ended]]:
        """:func:`receive`, for this run's mailbox. A service run that waits
        stays awake in its request (its service is billed as it always is), and
        the wait still holds nothing open to the control plane."""
        return _receive_as(self._api_key, self.run_id, timeout_seconds, from_run)

    def messages(self, *, after: int = 0) -> Iterator[Message]:
        """:func:`messages`, for this run's mailbox."""
        return _messages_as(self._api_key, self.run_id, after)

    def parent(self) -> Optional[str]:
        """:func:`parent`, of this run."""
        found = _read_run(_run_url(self.run_id), self._api_key).raw.get(
            "spawned_by_deployment_id"
        )
        return str(found) if found else None


def service_run(headers: HeaderSource) -> ServiceRun:
    """
    Return the handle for the run a service's ``POST /aetherfy/run`` is handling.

    Call it with the request's headers, at the top of the handler::

        @app.post("/aetherfy/run")
        async def run(request):
            run = service_run(request.headers)
            child = run.spawn("worker", {"id": 7})
            finished = run.wait(child.spawn_id)
            return {"answer": finished.result}

    The platform sets ``X-Aetherfy-Run`` (the run's id) and
    ``X-Aetherfy-Run-Key`` (the run's own key) on that one call, and removes both
    from every public request, so a handler can trust them. They are read
    without regard to case, from any headers object with ``items()``: a dict,
    Starlette's or Flask's ``request.headers``. The key is never logged by this
    package and is not in the handle's ``repr``.

    THE HANDLE IS EXPLICIT BECAUSE AN IMPLICIT ONE WOULD FAIL SILENTLY: see
    :class:`ServiceRun`. The module-level functions are unchanged and keep
    reading ``AETHERFY_API_KEY``, which on a task machine is the run's key and
    in a service is the release's.

    :raises NotAServiceRun: the headers carry no run id and run key. This is for
        the handler of ``POST /aetherfy/run`` only; a public request has neither.
    """
    wanted = {RUN_ID_HEADER.lower(): "", RUN_KEY_HEADER.lower(): ""}
    for name, value in headers.items():
        if not (isinstance(name, str) and isinstance(value, str)):
            continue
        key = name.lower()
        if key in wanted and not wanted[key]:
            wanted[key] = value.strip()
    missing = [
        header
        for header in (RUN_ID_HEADER, RUN_KEY_HEADER)
        if not wanted[header.lower()]
    ]
    if missing:
        raise NotAServiceRun(missing)
    return ServiceRun(wanted[RUN_ID_HEADER.lower()], wanted[RUN_KEY_HEADER.lower()])


def agent_url(name: str, *, fresh: bool = False) -> str:
    """
    Return the current public URL of a peer agent, found by name.

    A peer is an agent of THIS agent's own workspace; ``name`` is the name it
    has there. The URL is what the platform serves the peer at right now, so a
    peer that was deleted and created again is found at its new address and
    nothing here, or in this machine's environment, has to change.

    THE ANSWER IS REMEMBERED FOR ``PEER_CACHE_SECONDS`` (30) PER NAME, in this
    process only, with no request while it is fresh. ``fresh=True`` skips that
    memory, asks the control plane and replaces the entry; use it when an
    address you were handed has just stopped answering. :func:`call_agent` does
    exactly that, once, by itself.

    Only the NAME to URL pair is kept. A refusal is never kept, so a peer that
    is created a moment after a :class:`PeerNotFound` is found on the next call.

    WHEN THE CONTROL PLANE CANNOT ANSWER (429, 5xx, or no connection), the
    last address known for the peer is returned if there is one, and the
    control plane is not asked again before its ``Retry-After`` (between
    ``PEER_RETRY_FLOOR_SECONDS`` and ``PEER_RETRY_CAP_SECONDS``): stale beats
    failing a call that would have worked. With no known address the failure
    is raised. ``fresh=True`` is never answered from memory, and a 404 or 409
    drops the known address.

    :raises PeerNotFound: 404 ``AGENT_NOT_FOUND``, the workspace has no agent
        of that name.
    :raises PeerHasNoUrl: 409 ``PEER_HAS_NO_URL``, the agent exists but nothing
        listens on it.
    :raises PeerLookupError: any other refusal, among them 403
        ``AUTH_AGENT_KEY_OUT_OF_SCOPE`` — read ``error_code``, not the prose.
    :raises NotRunningOnAgent: ``AETHERFY_PEERS_URL`` or ``AETHERFY_API_KEY`` is
        not set.
    :raises ValueError: ``name`` is empty. No request is sent.
    :raises AgentTransportError: the request never reached the control plane.
    """
    if not name:
        raise ValueError("name must be a peer agent's name, not an empty string.")
    if not fresh:
        cached = _peers.cache.get_url(name)
        if cached is not None:
            return cached

    waiting = _peers.cache.blocked(name)
    if waiting is not None:
        stale = None if fresh else _peers.cache.last_known_url(name)
        if stale is not None:
            return stale
        raise _still_waiting("Looking up '{0}'".format(name), *waiting)

    try:
        body = _lookup(
            "{0}/{1}".format(_peers_url(), quote(name, safe="")),
            name,
            known=not fresh and _peers.cache.last_known_url(name) is not None,
        )
    except (PeerNotFound, PeerHasNoUrl):
        _peers.cache.forget_url(name)
        raise
    except (PeerLookupError, AgentTransportError) as exc:
        if not _control_plane_could_not_answer(exc):
            raise
        _block(name, exc)
        stale = None if fresh else _peers.cache.last_known_url(name)
        if stale is not None:
            return stale
        raise
    url = body.get("url")
    if not isinstance(url, str) or not url:
        raise PeerLookupError(
            "The lookup of '{0}' answered 200 with no 'url', so there is no "
            "address to return.".format(name),
            status_code=200,
            details=body,
        )
    _peers.cache.put_url(name, url)
    return url


def agent_urls(*, fresh: bool = False) -> Dict[str, str]:
    """
    Return every addressable peer of this agent's workspace as ``{name: url}``.

    The workspace's other agents that have a public URL; this agent is not in
    it, and neither is an agent nothing listens on. The same memory rule as
    :func:`agent_url`: the whole answer is remembered for
    ``PEER_CACHE_SECONDS`` (30) as one entry, and ``fresh=True`` skips it and
    replaces it. The two memories are separate: this list is not read to
    answer :func:`agent_url`, nor the other way round.

    The dict is a copy: editing it does not edit what is remembered.

    The same fallback as :func:`agent_url`: when the control plane cannot
    answer, the last list known is returned, and the control plane is not
    asked again before its ``Retry-After``.

    :raises PeerLookupError: a refusal, among them 403
        ``AUTH_AGENT_KEY_OUT_OF_SCOPE`` — read ``error_code``, not the prose.
    :raises NotRunningOnAgent: ``AETHERFY_PEERS_URL`` or ``AETHERFY_API_KEY`` is
        not set.
    :raises AgentTransportError: the request never reached the control plane.
    """
    if not fresh:
        cached = _peers.cache.get_list()
        if cached is not None:
            return cached

    waiting = _peers.cache.blocked(None)
    if waiting is not None:
        stale_list = None if fresh else _peers.cache.last_known_list()
        if stale_list is not None:
            return stale_list
        raise _still_waiting("Listing the peers", *waiting)

    try:
        body = _lookup(
            _peers_url(),
            None,
            known=not fresh and _peers.cache.last_known_list() is not None,
        )
    except (PeerLookupError, AgentTransportError) as exc:
        if not _control_plane_could_not_answer(exc):
            raise
        _block(None, exc)
        stale_list = None if fresh else _peers.cache.last_known_list()
        if stale_list is not None:
            return stale_list
        raise
    peers = body.get("peers")
    if not isinstance(peers, list):
        raise PeerLookupError(
            "The peer list answered 200 with no 'peers' list, so there is "
            "nothing to return.",
            status_code=200,
            details=body,
        )
    urls: Dict[str, str] = {}
    for peer in peers:
        peer_name = peer.get("name") if isinstance(peer, dict) else None
        peer_url = peer.get("url") if isinstance(peer, dict) else None
        if not (
            isinstance(peer_name, str)
            and peer_name
            and isinstance(peer_url, str)
            and peer_url
        ):
            raise PeerLookupError(
                "The peer list answered 200 with an entry that has no name and "
                "url: {0!r}.".format(peer),
                status_code=200,
                details=body,
            )
        urls[peer_name] = peer_url
    _peers.cache.put_list(urls)
    return dict(urls)


def call_agent(
    name: str,
    path: str,
    *,
    method: str = "GET",
    json: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = _http.DEFAULT_TIMEOUT,
) -> PeerResponse:
    """
    Call a peer agent over HTTP and return its answer.

    Looks ``name`` up with :func:`agent_url` and sends ``method`` to the URL
    plus ``path`` (which starts with ``/``). ``json``, when given, is the
    request body encoded as JSON, with a ``Content-Type`` of
    ``application/json`` unless ``headers`` names one. ``timeout`` is seconds
    for the request.

    THE PEER IS ANOTHER SERVICE, so it is never sent this machine's Aetherfy
    key: only the lookup carries it. Whatever authentication the peer wants is
    yours to put in ``headers``.

    ONE RE-LOOKUP, AND ONLY FOR AN ADDRESS THAT IS GONE. The call is repeated
    once, at a freshly looked-up URL (``agent_url(name, fresh=True)``), when
    the first attempt PROVABLY NEVER REACHED A PEER (the name does not resolve,
    or the connection was refused: nothing accepted it) or when the edge REFUSED
    the address: a 404 whose JSON body is ``{"error": {"code": "AGENT_NOT_FOUND"}}``
    or ``AGENT_NOT_REACHABLE``, which is what a retired address answers. That
    is how a peer recreated inside the cache window is found without a
    restart. A second failure of either kind raises; nothing is retried
    again.

    EVERY OTHER STATUS IS RETURNED, never retried and never raised, 5xx
    included: the peer answered, and only you know what its answer means. A
    redirect is such an answer: it is returned, not followed, so a ``POST`` is
    never quietly replayed as a ``GET`` and your headers never go to a host
    you did not name.
    A RESET OR A TIMEOUT IS NOT RETRIED: either can happen after the peer has
    acted, so the call raises :class:`AgentTransportError` and whether to repeat
    it is yours to say. A request that is not safe to repeat (a ``POST`` that
    charges a card) should carry its own idempotency key.

    :raises PeerNotFound: 404 ``AGENT_NOT_FOUND``, the workspace has no agent
        of that name.
    :raises PeerHasNoUrl: 409 ``PEER_HAS_NO_URL``, the agent exists but nothing
        listens on it.
    :raises PeerLookupError: any other refusal of the lookup.
    :raises AgentTransportError: the peer did not answer, and the request may
        or may not have run (a reset or a timeout); or it could not be reached at
        the freshly looked-up address either, or that address was refused too, or the
        lookup never reached the control plane.
    :raises NotRunningOnAgent: ``AETHERFY_PEERS_URL`` or ``AETHERFY_API_KEY`` is
        not set.
    :raises ValueError: ``name`` is empty, ``path`` does not start with ``/``
        or ``timeout`` is not positive. No request is sent.
    """
    if not path.startswith("/"):
        raise ValueError(
            "path must start with '/', as in '/summarise', not {0!r}.".format(path)
        )
    if not timeout > 0:
        raise ValueError(
            "timeout must be a positive number of seconds, not {0!r}.".format(timeout)
        )
    request_headers = dict(headers) if headers else {}
    body = None
    if json is not None:
        body = _http.encode_json(json)
        if not any(key.lower() == "content-type" for key in request_headers):
            request_headers["Content-Type"] = "application/json"

    def attempt(base: str) -> _http.PeerAttempt:
        return _http.request_peer(
            method,
            base.rstrip("/") + path,
            ua=_user_agent(),
            body=body,
            headers=request_headers,
            timeout=timeout,
        )

    first = attempt(agent_url(name)).response
    if first is not None and _refused_code(first) is None:
        return first

    second = attempt(agent_url(name, fresh=True))
    if second.response is None:
        raise AgentTransportError(
            "The peer '{0}' did not answer at its freshly looked-up address, "
            "nor at the one it had before: {1}".format(name, second.failure)
        )
    refused = _refused_code(second.response)
    if refused is not None:
        raise AgentTransportError(
            "The peer '{0}' was refused ({1}) at its freshly looked-up address, "
            "as at the one it had before.".format(name, refused)
        )
    return second.response


def _refused_code(response: PeerResponse) -> Optional[str]:
    """The code of an address the edge refused, or None for any other answer."""
    if response.status != 404:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if code in (AGENT_NOT_FOUND, AGENT_NOT_REACHABLE) else None


def _peers_url() -> str:
    """
    Resolve ``AETHERFY_PEERS_URL`` into the URL to GET, without a trailing slash.

    A TEMPLATE, exactly as :func:`_spawn_url` explains for its variable: the
    platform leaves ``{id}`` as a literal placeholder in
    ``.../agents/{id}/peers``, and the id that belongs there is this machine's
    own agent, because these are THIS agent's peers.
    """
    url = _require("AETHERFY_PEERS_URL", "the peer lookup endpoint")
    if "{id}" in url:
        agent_id = _require(
            "AETHERFY_AGENT_ID", "this agent's id, whose peers these are"
        )
        url = url.replace("{id}", agent_id)
    return url.rstrip("/")


def _control_plane_could_not_answer(exc: Exception) -> bool:
    """A lookup that failed for the control plane's state, not for what was asked: no connection,
    429 or 5xx. A 403, a 404 and a 409 are ANSWERS and are never served around."""
    if isinstance(exc, AgentTransportError):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


def _block(key: Optional[str], exc: Exception) -> None:
    _peers.cache.block(
        key,
        _peers.retry_delay(getattr(exc, "retry_after", None)),
        getattr(exc, "status_code", None),
        getattr(exc, "error_code", None),
    )


def _still_waiting(
    what: str, left: float, status: Optional[int], code: Optional[str]
) -> PeerLookupError:
    """The failure of the lookup being waited out, raised instead of asking again."""
    return PeerLookupError(
        "{0} failed a moment ago ({1}) and the control plane is not asked again for "
        "{2:.0f} more second(s); no earlier address is known.".format(
            what, status if status is not None else "no connection", left
        ),
        status_code=status,
        error_code=code,
        retry_after=left,
    )


def _lookup(url: str, name: Optional[str], *, known: bool) -> Dict[str, Any]:
    """
    One GET against the peers route, shared by :func:`agent_url` and
    :func:`agent_urls`: the two answer in the same envelope and refuse in the
    same words, so there is one place that maps the refusal.

    ``known``: an address (or a list) this call would serve if the lookup
    cannot be answered. Then a control-plane refusal is not waited out: the
    last known answer is served at once, and Retry-After is kept before asking
    again. With nothing known (the first lookup in this process, or a fresh
    one), a 503 RESOURCE_BUSY (nothing ran) is waited out once and asked again
    before the lookup fails; any other 429 or 5xx fails as it came.

    THE CODE DECIDES, NOT THE STATUS ALONE, as in :func:`spawn`. 404 and 409
    are categories the control plane reuses; ``AGENT_NOT_FOUND`` and
    ``PEER_HAS_NO_URL`` are what it promises not to rename.
    """
    api_key = _require("AETHERFY_API_KEY", "the key a peer lookup authenticates with")
    status, body, retry_after = _http.request_json(
        "GET",
        url,
        api_key=api_key,
        ua=_user_agent(),
        repeatable=True,
        refusals=_http.REFUSALS_NONE if known else _http.REFUSALS_RESOURCE_BUSY_ONLY,
    )

    if status == 200:
        if not isinstance(body, dict):
            raise PeerLookupError(
                "The peer lookup answered 200 with a body that is not an "
                "object, so there is nothing to return.",
                status_code=status,
            )
        return body

    detail = _detail_of(body)
    what = "Looking up '{0}'".format(name) if name else "Listing the peers"
    message = detail.get("message") or "{0} failed with status {1}.".format(
        what, status
    )
    code = detail.get("code")
    if status == 404 and code == AGENT_NOT_FOUND:
        raise PeerNotFound(message, details=detail)
    if status == 409 and code == PEER_HAS_NO_URL:
        raise PeerHasNoUrl(message, details=detail)
    raise PeerLookupError(
        message,
        status_code=status,
        error_code=code,
        details=detail,
        retry_after=retry_after,
    )


def _inline_max_bytes() -> Optional[int]:
    """
    This machine's inline cap, or ``None`` when it cannot be read.

    NOT A REFUSAL WHEN ABSENT. The cap is the platform's to enforce and it does
    — an oversized result is dropped and recorded as ``too_large`` — so the
    check here is a courtesy that turns a silent drop into something the caller
    can act on. Declining to write because the courtesy is unavailable would
    lose a result the platform would have accepted, which is strictly worse
    than not checking.
    """
    raw = os.environ.get("AETHERFY_RUN_INLINE_MAX_BYTES")
    if not raw:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _run_url(run_id: str) -> str:
    """The control plane's URL for one run.

    A run is a deployment row — the ephemeral kind — so it is read from
    ``/deployments/{id}``, the same route and the same object a deploy is read
    from. That is the platform's shape, not a convenience: one row, one reader.
    """
    api_url = _require("AETHERFY_API_URL", "the control plane's base URL")
    if not run_id:
        raise ValueError("run_id must be a run's id, not an empty string.")
    return "{0}/deployments/{1}".format(api_url.rstrip("/"), quote(run_id, safe=""))


def _environment_key(purpose: str) -> str:
    """The key this process was started with: on a task machine the run's own,
    in a service the release's. What the module-level functions send."""
    return _require("AETHERFY_API_KEY", purpose)


def _is_int(value: Any) -> TypeGuard[int]:
    """A JSON integer. ``bool`` is an ``int`` in Python and is not one here."""
    return isinstance(value, int) and not isinstance(value, bool)


def _run_from(body: Dict[str, Any]) -> Run:
    """The one place a response's deployment object becomes a :class:`Run`:
    a read and a cancel both hand one back."""
    return Run(
        id=str(body.get("id")),
        agent_id=str(body.get("agent_id")),
        state=str(body.get("state")),
        result=body.get("result"),
        result_error=body.get("result_error"),
        has_result=bool(body.get("has_result")),
        is_ephemeral=bool(body.get("is_ephemeral")),
        error_message=body.get("error_message"),
        failure_code=body.get("failure_code"),
        retry=_retry_from(body.get("retry")),
        raw=body,
    )


def _retry_from(value: Any) -> Optional[RunRetry]:
    """A run's ``retry`` object as a :class:`RunRetry`, or ``None``.

    ``None`` is the answer for a ``null``, for a missing key, and for an object
    that does not have the shape: ``retry`` is information about a run that is
    still going, so a field of the wrong type reads as absent, as ``position``
    and ``cancelled_count`` do, rather than failing the read of the run. It is
    all or nothing, never a retry with some of its fields. ``Run.raw`` keeps
    what the platform sent either way.
    """
    if not isinstance(value, dict):
        return None
    code = value.get("code")
    attempt = value.get("attempt")
    max_attempts = value.get("max_attempts")
    next_attempt_at = _utc_from(value.get("next_attempt_at"))
    if (
        isinstance(code, str)
        and _is_int(attempt)
        and _is_int(max_attempts)
        and next_attempt_at is not None
    ):
        return RunRetry(
            code=code,
            attempt=attempt,
            max_attempts=max_attempts,
            next_attempt_at=next_attempt_at,
        )
    return None


def _utc_from(value: Any) -> Optional[datetime]:
    """An ISO 8601 timestamp that names its offset (``Z`` or ``+hh:mm``) as a
    timezone-aware UTC ``datetime``; ``None`` for anything else.

    A timestamp without an offset is refused, not assumed: it would read as
    local time, and the platform's are UTC.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _shared_run_refusal(
    status: int, code: Any, message: str, detail: Dict[str, Any]
) -> Optional[RunReadError]:
    """The two refusals every call on a run answers in the same words, as the
    types they have: no run has that id, and the run is another account's. None
    for anything else."""
    if status == 404 and code == DEPLOYMENT_NOT_FOUND:
        return RunNotFound(message, details=detail)
    if status == 403 and code == DEPLOYMENT_ACCESS_DENIED:
        return RunAccessDenied(message, details=detail)
    return None


def _read_run(
    url: str,
    api_key: str,
    *,
    timeout: float = _http.DEFAULT_TIMEOUT,
    repeatable: bool = True,
) -> Run:
    """One GET, one Run — shared by :func:`result` and :func:`wait`, for
    whichever key they are given.

    ONE implementation because the two routes return the SAME object and refuse
    in the SAME words; the control plane loads both through one function for
    exactly that reason. Two readers here is how one of them ends up mapping a
    403 the other maps as a 404.
    """
    status, body, retry_after = _http.request_json(
        "GET",
        url,
        api_key=api_key,
        ua=_user_agent(),
        timeout=timeout,
        repeatable=repeatable,
    )

    if status == 200:
        if not isinstance(body, dict):
            raise RunReadError(
                "Reading the run answered 200 with a body that is not an "
                "object, so there is no run to return.",
                status_code=status,
            )
        return _run_from(body)

    detail = _detail_of(body)
    message = detail.get("message") or "Reading the run failed with status {0}.".format(
        status
    )
    code = detail.get("code")

    # THE CODE DECIDES, NOT THE STATUS ALONE — the same rule spawn() follows,
    # for the same reason. 404 and 403 are categories the control plane reuses
    # across every route; DEPLOYMENT_NOT_FOUND and DEPLOYMENT_ACCESS_DENIED are
    # what it publishes and promises not to rename. An unrecognised pairing
    # falls through to RunReadError, which reports exactly what arrived.
    refusal = _shared_run_refusal(status, code, message, detail)
    if refusal is not None:
        raise refusal
    if status == 422 and code == DEPLOYMENT_WAIT_TIMEOUT_INVALID:
        raise WaitTimeoutInvalid(message, details=detail)
    raise RunReadError(
        message,
        status_code=status,
        error_code=code,
        details=detail,
        retry_after=retry_after,
    )


def _spawn_url() -> str:
    """
    Resolve ``AETHERFY_SPAWN_URL`` into the URL to POST to.

    THE VARIABLE IS A TEMPLATE, not a finished URL. The platform injects
    ``.../agents/{id}/spawn`` with ``{id}`` left as a literal placeholder —
    substituted here with this machine's own agent id, because the path
    parameter names the PARENT of the spawn, which is us. A helper that POSTed
    the variable verbatim would send a request to a path containing a literal
    brace and get a 404 that explains nothing.

    A platform that starts injecting an already-resolved URL keeps working: the
    substitution only fires when the placeholder is actually present.
    """
    url = _require("AETHERFY_SPAWN_URL", "the spawn endpoint")
    if "{id}" not in url:
        return url
    agent_id = _require("AETHERFY_AGENT_ID", "this agent's id, the spawn's parent")
    return url.replace("{id}", agent_id)


def _detail_of(body: Any) -> Dict[str, Any]:
    """
    Pull the control plane's ``{"detail": {...}}`` envelope out of a body.

    NOT the vector API's ``{"error": {...}}`` — the agent control plane is a
    different service with a different envelope, and conflating them is how a
    caller ends up reading ``None`` for every code. A ``detail`` that is a bare
    string is FastAPI's own default, which routes that never reached our error
    handling produce; it carries prose but no code.
    """
    if not isinstance(body, dict):
        return {}
    detail = body.get("detail")
    if isinstance(detail, dict):
        return detail
    if isinstance(detail, str):
        return {"message": detail}
    return {}


def _message_of(body: Any) -> Optional[str]:
    detail = _detail_of(body)
    message = detail.get("message")
    return message if isinstance(message, str) else None
