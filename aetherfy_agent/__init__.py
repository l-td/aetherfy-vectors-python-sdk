"""
Aetherfy Agent — what code running on an Aetherfy machine does.

This is a THIN wrapper over contracts the platform already publishes. It
invents no protocol: every call here has a hand-rolled equivalent in
https://docs.aetherfy.com/agents/task-contract, and the helper exists so that
equivalent stops being copied into every task.

    from aetherfy_agent import payload, machine, fan_out, spawn
    from aetherfy_agent import write_result, result, wait

    data = payload()                         # this run's input, {} when none
    shape = machine()                        # vcpus / memory_mb / region
    results = fan_out(work, data["items"])   # in-machine pool, input order
    write_result({"rows": len(results)})     # this run's answer

    run = spawn("nightly-rollup", {"date": "2026-09-07"})
    finished = wait(run.spawn_id)            # or result(...) for a plain read
    print(finished.result)

TWO HALVES, and they are the same contract read from opposite ends. A task
reads its payload and writes its result; whoever started it spawns and then
reads that result back. ``payload``/``write_result`` are files on the machine
and touch no network at all; ``spawn``/``result``/``wait`` are the control
plane, and are the only calls here that do.

It ships inside the ``aetherfy-vectors`` distribution beside
``aetherfy_vectors`` and ``aetherfy_memory``, and the standard runtime image
preinstalls that distribution — so on a plain agent these names import with
nothing in your requirements. A custom container installs it itself.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar
from urllib.parse import quote

from . import _http
from .exceptions import (
    AGENT_SPAWN_CONCURRENCY_LIMIT_EXCEEDED,
    DEPLOYMENT_ACCESS_DENIED,
    DEPLOYMENT_NOT_FOUND,
    DEPLOYMENT_WAIT_TIMEOUT_INVALID,
    RUN_PAYLOAD_TOO_LARGE,
    AgentError,
    AgentTransportError,
    NotRunningOnAgent,
    PayloadTooLarge,
    PayloadUnavailable,
    ResultTooLarge,
    RunAccessDenied,
    RunNotFound,
    RunReadError,
    SpawnError,
    TooManyRunsInFlight,
    WaitTimeoutInvalid,
)
from .models import MachineShape, Run, Spawn

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
    # PUBLIC IN BOTH LANGUAGES OR NEITHER. The JavaScript helper exports these
    # three from its entry point, so a task ported between the two would find
    # the bound readable in one and not the other. A caller sizing its own
    # retry loop around wait() is the reason they are readable at all.
    "WAIT_TIMEOUT_MIN_SECONDS",
    "WAIT_TIMEOUT_MAX_SECONDS",
    "WAIT_TIMEOUT_DEFAULT_SECONDS",
    "MachineShape",
    "Run",
    "Spawn",
    "AgentError",
    "AgentTransportError",
    "NotRunningOnAgent",
    "PayloadUnavailable",
    "PayloadTooLarge",
    "ResultTooLarge",
    "RunReadError",
    "RunNotFound",
    "RunAccessDenied",
    "SpawnError",
    "TooManyRunsInFlight",
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
    "Aetherfy offers it to a `type: job` machine before each run's entrypoint "
    "starts, and only when that machine also carries an inline result cap "
    "(AETHERFY_RUN_INLINE_MAX_BYTES) — without the cap the platform cannot "
    "accept a result and does not offer the path. A `service` agent never gets "
    "one: a result belongs to a run."
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
    status, body = _http.request_json("GET", url, api_key=api_key, ua=_user_agent())
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
    Ask the control plane to run another task agent, and return once recorded.

    ``child`` is the id or name of a ``type: job`` agent you own; the parent is
    this machine's own agent. The child runs in this agent's region, and the
    two must be connected by ``spawn.workers`` in ``aetherfy.yaml`` for the
    call to be allowed.

    ACCEPTANCE IS NOT EXECUTION. The returned :class:`~.models.Spawn` says the
    run was recorded and its deploy queued. Aetherfy never queues a run behind
    another, so a spawn aimed at an agent already running fails as busy rather
    than waiting — check the run's status.

    Keep the payload small: it is for parameters and references, not data. Pass
    anything large by reference to a collection.

    :raises PayloadTooLarge: 413, the payload crossed the inline cap.
    :raises TooManyRunsInFlight: 429, the concurrent-run cap is full. The one
        failure here worth retrying.
    :raises SpawnError: any other refusal — read ``error_code``, not the prose.
    :raises AgentTransportError: the request never reached the control plane.
    """
    api_key = _require("AETHERFY_API_KEY", "the key a spawn authenticates with")
    url = _spawn_url()

    status, body = _http.request_json(
        "POST",
        url,
        api_key=api_key,
        ua=_user_agent(),
        body={"child_agent_id": child, "payload": payload or {}},
    )

    if status in (200, 201, 202):
        if not isinstance(body, dict):
            raise SpawnError(
                "The spawn was accepted with {0} but the body was not an "
                "object, so the run cannot be identified.".format(status),
                status_code=status,
            )
        return Spawn(
            spawn_id=str(body.get("spawn_id")),
            job_id=str(body.get("job_id")),
            child_agent_id=str(body.get("child_agent_id")),
            region=str(body.get("region")),
            status=str(body.get("status")),
            workspace=body.get("workspace"),
            estimated_start=body.get("estimated_start"),
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
    if status == 429 and code == AGENT_SPAWN_CONCURRENCY_LIMIT_EXCEEDED:
        raise TooManyRunsInFlight(
            message,
            in_flight_count=detail.get("in_flight_count"),
            limit=detail.get("limit"),
            max_in_flight_runs=detail.get("max_in_flight_runs"),
            details=detail,
        )
    raise SpawnError(message, status_code=status, error_code=code, details=detail)


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

    Answers immediately with the run as it stands. A run that is still going
    has ``state == "active"`` and no result yet; :func:`wait` is the same read
    with the waiting done server-side, and is what to use when the answer is
    the point.

    ``run_id`` is a run's id — ``Spawn.spawn_id`` from a :func:`spawn`, or the
    id of this run itself in ``AETHERFY_SPAWN_ID``.

    :raises RunNotFound: 404, no run has that id.
    :raises RunAccessDenied: 403, the run belongs to another account.
    :raises RunReadError: any other refusal — read ``error_code``, not the prose.
    :raises AgentTransportError: the request never reached the control plane.
    """
    return _read_run(_run_url(run_id))


def wait(run_id: str, timeout_seconds: int = WAIT_TIMEOUT_DEFAULT_SECONDS) -> Run:
    """
    Hold one request open until the run finishes, then return it.

    The read side of the result path, and the reason a parent does not poll:
    without it every caller writes the same loop with its own interval, and all
    of them pay for the privilege of not knowing yet.

    A TIMEOUT IS NOT AN ERROR. If the run has not finished in ``timeout_seconds``
    this returns it exactly as it stands — read ``Run.state``, which is
    ``active`` while a run is executing and ``completed`` or ``failed`` when it
    is over, and call again. Waiting longer than the maximum is a second call,
    not a bigger number: the request is held open, and anything longer is cut
    by the network in front of Aetherfy.

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
    :raises RunAccessDenied: 403, the run belongs to another account.
    :raises RunReadError: any other refusal.
    :raises AgentTransportError: the request never reached the control plane.
    """
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
        timeout=timeout_seconds + _WAIT_TRANSPORT_MARGIN_SECONDS,
        retry_connection_errors=False,
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


def _read_run(
    url: str,
    *,
    timeout: float = _http.DEFAULT_TIMEOUT,
    retry_connection_errors: bool = True,
) -> Run:
    """One GET, one Run — shared by :func:`result` and :func:`wait`.

    ONE implementation because the two routes return the SAME object and refuse
    in the SAME words; the control plane loads both through one function for
    exactly that reason. Two readers here is how one of them ends up mapping a
    403 the other maps as a 404.
    """
    api_key = _require("AETHERFY_API_KEY", "the key a run is read with")
    status, body = _http.request_json(
        "GET",
        url,
        api_key=api_key,
        ua=_user_agent(),
        timeout=timeout,
        retry_connection_errors=retry_connection_errors,
    )

    if status == 200:
        if not isinstance(body, dict):
            raise RunReadError(
                "Reading the run answered 200 with a body that is not an "
                "object, so there is no run to return.",
                status_code=status,
            )
        return Run(
            id=str(body.get("id")),
            agent_id=str(body.get("agent_id")),
            state=str(body.get("state")),
            result=body.get("result"),
            result_error=body.get("result_error"),
            has_result=bool(body.get("has_result")),
            is_ephemeral=bool(body.get("is_ephemeral")),
            error_message=body.get("error_message"),
            raw=body,
        )

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
    if status == 404 and code == DEPLOYMENT_NOT_FOUND:
        raise RunNotFound(message, details=detail)
    if status == 403 and code == DEPLOYMENT_ACCESS_DENIED:
        raise RunAccessDenied(message, details=detail)
    if status == 422 and code == DEPLOYMENT_WAIT_TIMEOUT_INVALID:
        raise WaitTimeoutInvalid(message, details=detail)
    raise RunReadError(message, status_code=status, error_code=code, details=detail)


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
