"""
Value types returned by the Aetherfy Agent helper.

All of them are plain frozen dataclasses: they describe what the platform already
told this process, so there is nothing to configure and nothing to mutate.
"""

import json as _json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MachineShape:
    """
    The machine this run is executing on.

    ``vcpus`` is the shared vCPU count the platform derives from the agent's
    memory step — size an in-machine pool from it. ``memory_mb`` is the memory
    the agent was deployed with, and the limit that ends the whole run if the
    process crosses it.
    """

    vcpus: int
    memory_mb: int
    region: str


@dataclass(frozen=True)
class Spawn:
    """
    An accepted spawn: the control plane recorded the run.

    Acceptance is not execution. ``status`` is the run's INITIAL status, and
    there are two:

    - ``"queued"``: the run goes straight to a machine.
    - ``"waiting"``: the account is at its in-flight limit, so the run waits in
      the account's line, with no machine and nothing billed, until a slot
      frees. ``position`` is its place there, ``1`` being next. A run that is
      waiting has no ``job_id`` yet.

    ``wait`` includes the time in the line, so a caller that waits on a run
    needs no other handling for ``waiting``. A spawn is refused with
    :class:`~aetherfy_agent.exceptions.RunLineFull` only when the line itself
    is full.

    ``position`` is ``None`` unless the run is waiting. ``raw`` is the whole
    response body, unmodified, like :attr:`Run.raw`.
    """

    spawn_id: str
    job_id: Optional[str]
    child_agent_id: str
    region: str
    status: str
    # The spawned agent's own workspace, by name. Every agent lives in one.
    workspace: str
    estimated_start: Optional[str] = None
    position: Optional[int] = None
    #: Everything the control plane sent, unmodified. Left out of the hash, so a
    #: Spawn stays usable as a dict key or in a set, as it was before it had a
    #: ``raw`` (a dict cannot be hashed).
    raw: Dict[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class Run:
    """
    One run, read back from the control plane.

    THE RUN ROW IS THE RECORD. A run's answer is not delivered anywhere — it is
    stored on the run and read from it, so a parent hears from a child in
    another region with no side channel and no shared storage between them.

    Only the fields a caller of :func:`~aetherfy_agent.result` or
    :func:`~aetherfy_agent.wait` reads are named here. The response carries a
    deployment object with a good deal more on it (regions, versions, rollback
    and cancellation flags), all of which is deploy-shaped rather than
    run-shaped; it is kept verbatim in ``raw`` rather than re-declared field by
    field in a place that would rot the first time the platform adds one.

    ``state`` IS THE ONE TO READ AFTER A WAIT. A run is ``waiting`` while it
    waits in the account's line (no machine, nothing billed, and
    ``raw["position"]`` is its place there, 1 being next), ``active`` while it
    is executing, and it ends ``completed``, ``failed`` or ``cancelled``. A wait
    that times out returns the run exactly as it stands, which is not an error:
    ``waiting`` and ``active`` both mean call again. There is deliberately no
    ``is_finished`` here, and no enum of the states: the set belongs to the
    platform, and a second copy of it in this package would be a copy that could
    disagree.

    ``result`` is ``None`` both for a run that returned nothing and for one
    whose result was refused — ``result_error`` is what tells those apart, and
    ``has_result`` is the platform's own answer to "did this run answer at
    all". A refused result is not a result, so ``has_result`` is ``False``
    whenever ``result_error`` is set.

    ``failure_code`` IS WHY A FAILED RUN FAILED, as the platform's code, beside
    ``error_message``, its sentence: the value to switch on (the code is the
    contract, the sentence is prose). ``None`` for a run that did not fail, and
    for one that failed before the platform recorded codes. A ``cancelled`` run
    carries none: why it was stopped is ``raw["cancellation_reason"]``.
    """

    id: str
    agent_id: str
    state: str
    result: Any = None
    result_error: Optional[str] = None
    has_result: bool = False
    is_ephemeral: bool = False
    error_message: Optional[str] = None
    failure_code: Optional[str] = None
    #: Everything the control plane sent, unmodified.
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Cancellation:
    """
    What cancelling a run answered.

    ``outcome`` says what the call did, and is one of:

    - ``"cancelled"``: the run is cancelled now (it was waiting, or had not
      started).
    - ``"stopping"``: the run was executing and is being stopped; it ends
      ``cancelled``.
    - ``"already_finished"``: the run had ended before the call. Not an error:
      nothing was cancelled.

    ``run`` is the run as the call left it. ``cancelled_count`` is how many runs
    this call cancelled, the runs started below it (its descendants) included.
    ``message`` is prose for a person and may be reworded at any time; branch on
    ``outcome``.
    """

    run: Run
    outcome: str
    message: str
    cancelled_count: int


@dataclass(frozen=True)
class PeerResponse:
    """
    What a peer answered to :func:`~aetherfy_agent.call_agent`.

    Small on purpose: the peer is another service and its answer is its own, so
    this carries the three things every answer has and no more. ``status`` is
    the HTTP status exactly as the peer sent it — an error status is a return,
    not a raise, because only the caller knows what a 404 from that peer means.
    ``headers`` has lower-case names. ``text`` is the body decoded as UTF-8,
    with undecodable bytes replaced.
    """

    status: int
    headers: Dict[str, str]
    text: str

    def json(self) -> Any:
        """
        The body parsed as JSON.

        :raises ValueError: the body is not JSON (an empty body included).
        """
        return _json.loads(self.text)
