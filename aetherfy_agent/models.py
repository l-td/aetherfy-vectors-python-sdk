"""
Value types returned by the Aetherfy Agent helper.

Both are plain dataclasses: they describe what the platform already told this
process, so there is nothing to configure and nothing to mutate.
"""

from dataclasses import dataclass
from typing import Optional


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

    Acceptance is not execution. ``status`` is the run's INITIAL status, and a
    spawned run that lands on an agent already busy fails as busy rather than
    queueing — Aetherfy never queues a run behind another.
    """

    spawn_id: str
    job_id: str
    child_agent_id: str
    region: str
    status: str
    workspace: Optional[str] = None
    estimated_start: Optional[str] = None
