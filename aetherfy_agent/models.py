"""
Value types returned by the Aetherfy Agent helper.

All three are plain frozen dataclasses: they describe what the platform already
told this process, so there is nothing to configure and nothing to mutate.
"""

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
    An accepted spawn: the control plane recorded the run and queued its deploy.

    Acceptance is not execution. ``status`` is the run's INITIAL status.
    Aetherfy never queues a run behind another: a spawned run that finds every
    machine of the child busy gets a machine of its own and runs at once.
    """

    spawn_id: str
    job_id: str
    child_agent_id: str
    region: str
    status: str
    workspace: Optional[str] = None
    estimated_start: Optional[str] = None


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

    ``state`` IS THE ONE TO READ AFTER A WAIT. A wait that times out returns
    the run exactly as it stands, which is not an error — ``active`` on a run
    means it is executing right now, and the terminal states are ``completed``
    and ``failed``. There is deliberately no ``is_finished`` here: the set of
    in-flight states belongs to the platform, and a second copy of it in this
    package would be a copy that could disagree.

    ``result`` is ``None`` both for a run that returned nothing and for one
    whose result was refused — ``result_error`` is what tells those apart, and
    ``has_result`` is the platform's own answer to "did this run answer at
    all". A refused result is not a result, so ``has_result`` is ``False``
    whenever ``result_error`` is set.
    """

    id: str
    agent_id: str
    state: str
    result: Any = None
    result_error: Optional[str] = None
    has_result: bool = False
    is_ephemeral: bool = False
    error_message: Optional[str] = None
    #: Everything the control plane sent, unmodified.
    raw: Dict[str, Any] = field(default_factory=dict)
