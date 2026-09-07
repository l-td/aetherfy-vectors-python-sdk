"""
Aetherfy Agent — the four things code running on an Aetherfy machine does.

This is a THIN wrapper over contracts the platform already publishes. It
invents no protocol: every call here has a hand-rolled equivalent in
https://docs.aetherfy.com/agents/task-contract, and the helper exists so that
equivalent stops being copied into every task.

    from aetherfy_agent import payload, machine, fan_out, spawn

    data = payload()                         # this run's input, {} when none
    shape = machine()                        # vcpus / memory_mb / region
    results = fan_out(work, data["items"])   # in-machine pool, input order
    spawn("nightly-rollup", {"date": "2026-09-07"})

It ships inside the ``aetherfy-vectors`` distribution beside
``aetherfy_vectors`` and ``aetherfy_memory``, and the standard runtime image
preinstalls that distribution — so on a plain agent these four names import
with nothing in your requirements. A custom container installs it itself.

Nothing here reaches the network except :func:`spawn` and the fallback branch
of :func:`payload`.

There is deliberately no ``result()`` and no ``wait()``. A run reports its
outcome through its exit code, and the platform's result path is not built
yet; adding a method that pretended otherwise would be inventing protocol.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar

from . import _http
from .exceptions import (
    AgentError,
    AgentTransportError,
    NotRunningOnAgent,
    PayloadTooLarge,
    PayloadUnavailable,
    SpawnError,
    TooManyRunsInFlight,
)
from .models import MachineShape, Spawn

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
    "MachineShape",
    "Spawn",
    "AgentError",
    "AgentTransportError",
    "NotRunningOnAgent",
    "PayloadUnavailable",
    "PayloadTooLarge",
    "SpawnError",
    "TooManyRunsInFlight",
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


def _require(variable: str, purpose: str) -> str:
    value = os.environ.get(variable)
    if not value:
        raise NotRunningOnAgent(variable, purpose)
    return value


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

    url = "{0}/deployments/{1}/payload".format(api_url.rstrip("/"), spawn_id)
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

    if status == 413:
        raise PayloadTooLarge(
            message,
            payload_bytes=detail.get("payload_bytes"),
            max_bytes=detail.get("max_bytes"),
            details=detail,
        )
    if status == 429:
        raise TooManyRunsInFlight(
            message,
            in_flight_count=detail.get("in_flight_count"),
            limit=detail.get("limit"),
            max_in_flight_runs=detail.get("max_in_flight_runs"),
            details=detail,
        )
    raise SpawnError(message, status_code=status, error_code=code, details=detail)


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
