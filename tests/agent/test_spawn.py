"""
`spawn()`: the request the control plane's route actually accepts, and the
two refusals a caller has to tell apart.

Pinned against aetherfy-control-plane `api/routes/agents.py`:
`SpawnRequest` (child_agent_id + payload), the route
`POST /agents/{agent_id_or_name}/spawn` returning 202 with `SpawnResponse`,
and the `{"detail": {...}}` error envelope built by `shared/api_errors.py`.
"""

import pytest

import aetherfy_agent
from aetherfy_agent import spawn
from aetherfy_agent.exceptions import (
    NotRunningOnAgent,
    PayloadTooLarge,
    SpawnError,
    TooManyRunsInFlight,
)

ACCEPTED = {
    "spawn_id": "11111111-1111-1111-1111-111111111111",
    "job_id": "22222222-2222-2222-2222-222222222222",
    "child_agent_id": "33333333-3333-3333-3333-333333333333",
    "workspace": "research",
    "region": "us-east-1",
    "status": "queued",
    "estimated_start": "~1s",
}


@pytest.fixture(autouse=True)
def on_a_machine(monkeypatch):
    # Exactly what orchestrator/fly_manager.py injects, including the literal
    # `{id}` placeholder in the spawn URL.
    monkeypatch.setenv(
        "AETHERFY_SPAWN_URL",
        "https://agents.aetherfy.com/api/v1/agents/{id}/spawn",
    )
    monkeypatch.setenv("AETHERFY_AGENT_ID", "parent-agent-id")
    monkeypatch.setenv("AETHERFY_API_KEY", "afy_test_key")


@pytest.fixture
def transport(monkeypatch):
    """Records the request and replies with whatever the test queued."""

    class Transport:
        def __init__(self):
            self.calls = []
            self.reply = (202, ACCEPTED)

        def __call__(self, method, url, *, api_key, ua, body=None, **kwargs):
            self.calls.append(
                {
                    "method": method,
                    "url": url,
                    "api_key": api_key,
                    "ua": ua,
                    "body": body,
                    "kwargs": kwargs,
                }
            )
            return self.reply

    fake = Transport()
    monkeypatch.setattr(aetherfy_agent._http, "request_json", fake)
    return fake


def test_the_request_matches_the_route(transport):
    spawn("nightly-rollup", {"date": "2026-09-07"})

    call = transport.calls[0]
    assert call["method"] == "POST"
    # THE VARIABLE IS A TEMPLATE. Posting it verbatim would send a request to a
    # path holding a literal brace.
    assert call["url"] == (
        "https://agents.aetherfy.com/api/v1/agents/parent-agent-id/spawn"
    )
    assert "{id}" not in call["url"]
    assert call["body"] == {
        "child_agent_id": "nightly-rollup",
        "payload": {"date": "2026-09-07"},
    }
    assert call["api_key"] == "afy_test_key"
    assert call["ua"].startswith("aetherfy-agent-python/")


def test_no_payload_sends_an_empty_object(transport):
    spawn("nightly-rollup")

    assert transport.calls[0]["body"]["payload"] == {}


def test_an_already_resolved_url_is_left_alone(transport, monkeypatch):
    """A platform that stops templating the variable keeps working."""
    monkeypatch.setenv(
        "AETHERFY_SPAWN_URL",
        "https://agents.aetherfy.com/api/v1/agents/abc/spawn",
    )
    monkeypatch.delenv("AETHERFY_AGENT_ID")

    spawn("nightly-rollup")

    assert transport.calls[0]["url"].endswith("/agents/abc/spawn")


def test_the_accepted_response_becomes_a_spawn(transport):
    result = spawn("nightly-rollup")

    assert result.spawn_id == ACCEPTED["spawn_id"]
    assert result.job_id == ACCEPTED["job_id"]
    assert result.child_agent_id == ACCEPTED["child_agent_id"]
    assert result.region == "us-east-1"
    assert result.status == "queued"
    assert result.workspace == "research"
    assert result.estimated_start == "~1s"


def test_a_workspaceless_spawn_carries_none(transport):
    transport.reply = (202, dict(ACCEPTED, workspace=None))

    assert spawn("nightly-rollup").workspace is None


def test_413_is_payload_too_large(transport):
    transport.reply = (
        413,
        {
            "detail": {
                "code": "RUN_PAYLOAD_TOO_LARGE",
                "message": "The run payload is 300000 bytes; the inline cap is 262144 bytes.",
                "payload_bytes": 300000,
                "max_bytes": 262144,
            }
        },
    )

    with pytest.raises(PayloadTooLarge) as excinfo:
        spawn("nightly-rollup", {"blob": "..."})
    error = excinfo.value
    assert error.payload_bytes == 300000
    assert error.max_bytes == 262144
    assert error.error_code == "RUN_PAYLOAD_TOO_LARGE"
    assert error.status_code == 413


def test_429_is_too_many_runs_in_flight(transport):
    # The envelope shared/job_runs.py builds at the 429 raise: `limit` names
    # WHICH plan limit was hit and `max_in_flight_runs` is its value. The cap
    # is the ACCOUNT's, set by the plan — not a per-agent spawn ceiling.
    transport.reply = (
        429,
        {
            "detail": {
                "code": "AGENT_SPAWN_CONCURRENCY_LIMIT_EXCEEDED",
                "message": (
                    "Too many runs in flight on this account (25/25); wait for "
                    "some to finish. The limit is set by your plan."
                ),
                "limit": "max_in_flight_runs",
                "in_flight_count": 25,
                "max_in_flight_runs": 25,
            }
        },
    )

    with pytest.raises(TooManyRunsInFlight) as excinfo:
        spawn("nightly-rollup")
    error = excinfo.value
    assert error.in_flight_count == 25
    assert error.limit == "max_in_flight_runs"
    assert error.max_in_flight_runs == 25
    assert error.error_code == "AGENT_SPAWN_CONCURRENCY_LIMIT_EXCEEDED"


def test_429_survives_an_uncapped_plan(transport):
    """`max_in_flight_runs` is None when the plan declares no cap. A caller
    building a message from it must not be handed a crash instead."""
    transport.reply = (
        429,
        {
            "detail": {
                "code": "AGENT_SPAWN_CONCURRENCY_LIMIT_EXCEEDED",
                "message": "Too many runs in flight on this account.",
                "limit": "max_in_flight_runs",
                "in_flight_count": 400,
                "max_in_flight_runs": None,
            }
        },
    )

    with pytest.raises(TooManyRunsInFlight) as excinfo:
        spawn("nightly-rollup")
    assert excinfo.value.max_in_flight_runs is None
    assert excinfo.value.in_flight_count == 400


def test_429_reads_the_limit_name_rather_than_assuming_it(transport):
    """`limit` is read off the envelope, not hardcoded. A second named limit
    reaching this status must arrive intact, not be reported as the first."""
    transport.reply = (
        429,
        {
            "detail": {
                "code": "AGENT_SPAWN_CONCURRENCY_LIMIT_EXCEEDED",
                "message": "Some other cap.",
                "limit": "max_agents",
                "in_flight_count": 3,
                "max_in_flight_runs": None,
            }
        },
    )

    with pytest.raises(TooManyRunsInFlight) as excinfo:
        spawn("nightly-rollup")
    assert excinfo.value.limit == "max_agents"


def test_a_413_carrying_another_code_is_a_plain_spawn_error(transport):
    """THE PAIRING SELECTS THE TYPE. A status is a category the platform reuses;
    the code is what it promises not to rename. A 413 grown for some new reason
    must arrive reporting ITS code, not wearing RUN_PAYLOAD_TOO_LARGE's."""
    transport.reply = (
        413,
        {
            "detail": {
                "code": "AGENT_IMAGE_TOO_LARGE",
                "message": "The built image is larger than the runtime allows.",
            }
        },
    )

    with pytest.raises(SpawnError) as excinfo:
        spawn("nightly-rollup")
    error = excinfo.value
    assert type(error) is SpawnError
    assert not isinstance(error, PayloadTooLarge)
    assert error.error_code == "AGENT_IMAGE_TOO_LARGE"
    assert error.status_code == 413
    assert "larger than the runtime allows" in str(error)


def test_a_429_carrying_another_code_is_a_plain_spawn_error(transport):
    transport.reply = (
        429,
        {
            "detail": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests.",
            }
        },
    )

    with pytest.raises(SpawnError) as excinfo:
        spawn("nightly-rollup")
    error = excinfo.value
    assert type(error) is SpawnError
    assert not isinstance(error, TooManyRunsInFlight)
    assert error.error_code == "RATE_LIMIT_EXCEEDED"
    assert error.status_code == 429


@pytest.mark.parametrize("status", [413, 429])
def test_a_codeless_body_on_either_status_is_a_plain_spawn_error(transport, status):
    """No code is not the expected code. Reporting one of the typed errors here
    would attach a code the platform never sent."""
    transport.reply = (status, {"detail": {"message": "no code here"}})

    with pytest.raises(SpawnError) as excinfo:
        spawn("nightly-rollup")
    error = excinfo.value
    assert type(error) is SpawnError
    assert error.error_code is None
    assert error.status_code == status


def test_429_with_no_extras_still_maps(transport):
    transport.reply = (
        429,
        {
            "detail": {
                "code": "AGENT_SPAWN_CONCURRENCY_LIMIT_EXCEEDED",
                "message": "busy",
            }
        },
    )

    with pytest.raises(TooManyRunsInFlight) as excinfo:
        spawn("nightly-rollup")
    error = excinfo.value
    assert error.limit is None
    assert error.max_in_flight_runs is None
    assert error.in_flight_count is None


def test_the_two_refusals_are_distinguishable(transport):
    """They must not collapse into one another: PayloadTooLarge is permanent
    for this payload, TooManyRunsInFlight is the one worth retrying."""
    assert not issubclass(PayloadTooLarge, TooManyRunsInFlight)
    assert not issubclass(TooManyRunsInFlight, PayloadTooLarge)
    assert issubclass(PayloadTooLarge, SpawnError)
    assert issubclass(TooManyRunsInFlight, SpawnError)


@pytest.mark.parametrize(
    "status,code",
    [
        (400, "AGENT_CHILD_NOT_JOB_TYPE"),
        (403, "AGENT_NOT_SPAWN_ENABLED"),
        (403, "AGENT_WORKER_NOT_ALLOWED"),
        (409, "AGENT_PARENT_NOT_SPAWNABLE"),
        (409, "AGENT_WORKER_PAUSED"),
        (503, "AGENT_SPAWN_RATE_LIMITED"),
        (500, "INTERNAL_ERROR"),
    ],
)
def test_every_other_status_carries_the_platform_code(transport, status, code):
    transport.reply = (status, {"detail": {"code": code, "message": "refused"}})

    with pytest.raises(SpawnError) as excinfo:
        spawn("nightly-rollup")
    error = excinfo.value
    assert type(error) is SpawnError
    assert error.error_code == code
    assert error.status_code == status
    assert "refused" in str(error)


def test_a_bare_string_detail_is_survivable(transport):
    """FastAPI's own default for a route that never reached our error
    handling — prose, no code."""
    transport.reply = (404, {"detail": "Not Found"})

    with pytest.raises(SpawnError) as excinfo:
        spawn("nightly-rollup")
    assert excinfo.value.error_code is None
    assert "Not Found" in str(excinfo.value)


def test_a_body_free_error_still_raises(transport):
    transport.reply = (502, None)

    with pytest.raises(SpawnError) as excinfo:
        spawn("nightly-rollup")
    assert excinfo.value.status_code == 502


def test_an_error_status_is_never_retried(transport):
    """A repeated 4xx is still a 4xx, and a repeated spawn the control plane
    may already have recorded is a second run."""
    transport.reply = (
        409,
        {"detail": {"code": "AGENT_WORKER_PAUSED", "message": "paused"}},
    )

    with pytest.raises(SpawnError):
        spawn("nightly-rollup")
    assert len(transport.calls) == 1


def test_a_missing_spawn_url_names_the_variable(transport, monkeypatch):
    monkeypatch.delenv("AETHERFY_SPAWN_URL")

    with pytest.raises(NotRunningOnAgent) as excinfo:
        spawn("nightly-rollup")
    assert excinfo.value.variable == "AETHERFY_SPAWN_URL"


def test_a_missing_api_key_names_the_variable(transport, monkeypatch):
    monkeypatch.delenv("AETHERFY_API_KEY")

    with pytest.raises(NotRunningOnAgent) as excinfo:
        spawn("nightly-rollup")
    assert excinfo.value.variable == "AETHERFY_API_KEY"
