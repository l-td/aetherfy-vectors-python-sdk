"""
`cancel()`: the request the control plane's cancel route accepts, the three
things it answers 200 with, and every refusal it can send.

Pinned against the run queue's contract (`POST /deployments/{run_id}/cancel`):
200 `RunCancelResponse {run, outcome, message, cancelled_count}`, and the
refusals 404 `DEPLOYMENT_NOT_FOUND`, 403 `DEPLOYMENT_ACCESS_DENIED`, 403
`AUTH_AGENT_KEY_OUT_OF_SCOPE` and 409 `DEPLOYMENT_NOT_A_RUN`. A run that had
already ended is an answer, not a refusal.
"""

import pytest

from aetherfy_agent import cancel, result
from aetherfy_agent.exceptions import (
    AgentError,
    NotARun,
    NotRunningOnAgent,
    RunAccessDenied,
    RunCancelError,
    RunNotFound,
    RunReadError,
)
from aetherfy_agent.models import Cancellation, Run

RUN_ID = "55555555-5555-5555-5555-555555555555"

# The deployment object a cancel answers with, as DeploymentResponse serializes
# a run: kept whole in `raw`.
CANCELLED_RUN = {
    "id": RUN_ID,
    "agent_id": "6f1c2b7e-0a2d-4f8e-9c31-2b0d5a7e4411",
    "state": "cancelled",
    "is_ephemeral": True,
    "result": None,
    "result_error": None,
    "has_result": False,
    "error_message": None,
    "cancellation_reason": "user_cancel",
    "version": 7,
}


def answer(outcome, count=1, state="cancelled", message="done"):
    return {
        "run": dict(CANCELLED_RUN, state=state),
        "outcome": outcome,
        "message": message,
        "cancelled_count": count,
    }


def refusal(code, message="nope", **extras):
    return {"detail": dict({"code": code, "message": message}, **extras)}


@pytest.fixture(autouse=True)
def on_a_machine(monkeypatch):
    monkeypatch.setenv("AETHERFY_API_URL", "https://agents.aetherfy.com/api/v1")
    monkeypatch.setenv("AETHERFY_API_KEY", "afy_test_key")


# --- THE REQUEST -------------------------------------------------------------


def test_cancel_posts_to_the_runs_cancel_route(transport):
    transport.reply = (200, answer("cancelled"))

    cancel(RUN_ID)

    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == (
        "https://agents.aetherfy.com/api/v1/deployments/" + RUN_ID + "/cancel"
    )
    # Nothing to say but which run: the id is in the path.
    assert call["body"] is None
    assert call["api_key"] == "afy_test_key"
    assert call["ua"].startswith("aetherfy-agent-python/")


def test_a_trailing_slash_on_the_base_url_does_not_double(transport, monkeypatch):
    monkeypatch.setenv("AETHERFY_API_URL", "https://agents.aetherfy.com/api/v1/")
    transport.reply = (200, answer("cancelled"))

    cancel(RUN_ID)

    assert "/v1/deployments/" in transport.calls[0]["url"]


def test_a_run_id_cannot_walk_out_of_its_route(transport):
    # UNESCAPED, `../agents/x` normalises to a DIFFERENT route before the
    # request leaves, and a POST there is a different action. Quoted, the
    # platform answers 404.
    transport.reply = (200, answer("cancelled"))

    cancel("../agents/other")

    assert transport.calls[0]["url"] == (
        "https://agents.aetherfy.com/api/v1/deployments/..%2Fagents%2Fother/cancel"
    )


def test_an_empty_run_id_is_refused_before_a_request(transport):
    with pytest.raises(ValueError):
        cancel("")

    assert transport.calls == []


@pytest.mark.parametrize("variable", ["AETHERFY_API_URL", "AETHERFY_API_KEY"])
def test_a_missing_variable_is_named(transport, monkeypatch, variable):
    monkeypatch.delenv(variable, raising=False)

    with pytest.raises(NotRunningOnAgent) as exc:
        cancel(RUN_ID)

    assert exc.value.variable == variable
    assert transport.calls == []


# --- THE THREE ANSWERS -------------------------------------------------------


@pytest.mark.parametrize(
    "outcome,count,state",
    [
        # Waiting or not yet started: it is cancelled now.
        ("cancelled", 1, "cancelled"),
        # Executing: being stopped, and it ends cancelled.
        ("stopping", 1, "active"),
        # Ended before the call: nothing was cancelled.
        ("already_finished", 0, "completed"),
    ],
)
def test_each_outcome_arrives_as_a_cancellation(transport, outcome, count, state):
    transport.reply = (
        200,
        answer(outcome, count=count, state=state, message="what happened"),
    )

    cancellation = cancel(RUN_ID)

    assert isinstance(cancellation, Cancellation)
    assert cancellation.outcome == outcome
    assert cancellation.cancelled_count == count
    assert cancellation.message == "what happened"
    assert isinstance(cancellation.run, Run)
    assert cancellation.run.id == RUN_ID
    assert cancellation.run.state == state


def test_a_run_that_had_already_ended_is_not_an_error(transport):
    """Cancelling twice, or racing the run's own end, is an answer. The caller
    reads `outcome`; nothing is raised."""
    transport.reply = (200, answer("already_finished", count=0, state="completed"))

    assert cancel(RUN_ID).outcome == "already_finished"


def test_the_count_includes_the_runs_the_cascade_cancelled(transport):
    transport.reply = (200, answer("cancelled", count=4))

    assert cancel(RUN_ID).cancelled_count == 4


def test_the_run_keeps_the_whole_object(transport):
    transport.reply = (200, answer("cancelled"))

    run = cancel(RUN_ID).run

    assert run.raw["cancellation_reason"] == "user_cancel"
    assert run.raw["version"] == 7
    assert run.is_ephemeral is True


def test_the_run_a_cancel_answers_with_is_read_like_any_other_run(transport):
    # One reader for every run the SDK returns: a cancel's carries the retry it
    # was sent with, and none when it was sent without.
    retrying = {
        "code": "PLATFORM_TEMPORARILY_UNAVAILABLE",
        "attempt": 3,
        "max_attempts": 4,
        "next_attempt_at": "2026-11-18T03:12:45Z",
    }
    body = answer("cancelled")
    body["run"]["retry"] = retrying
    transport.reply = (200, body)

    retry = cancel(RUN_ID).run.retry

    assert (retry.code, retry.attempt, retry.max_attempts) == (
        "PLATFORM_TEMPORARILY_UNAVAILABLE",
        3,
        4,
    )
    assert retry.next_attempt_at.isoformat() == "2026-11-18T03:12:45+00:00"

    transport.reply = (200, answer("cancelled"))
    assert cancel(RUN_ID).run.retry is None


def test_a_200_that_is_not_a_cancellation_is_not_taken_for_one(transport):
    for body in ("OK", {}, {"outcome": "cancelled"}, {"run": CANCELLED_RUN}):
        transport.reply = (200, body)
        with pytest.raises(RunCancelError) as exc:
            cancel(RUN_ID)
        assert exc.value.status_code == 200


# --- THE REFUSALS ------------------------------------------------------------


def test_an_unknown_id_is_run_not_found(transport):
    transport.reply = (404, refusal("DEPLOYMENT_NOT_FOUND", "Deployment x not found"))

    with pytest.raises(RunNotFound) as exc:
        cancel(RUN_ID)

    assert exc.value.status_code == 404
    assert exc.value.error_code == "DEPLOYMENT_NOT_FOUND"
    assert "not found" in str(exc.value)


def test_someone_elses_run_is_run_access_denied(transport):
    transport.reply = (403, refusal("DEPLOYMENT_ACCESS_DENIED", "Access denied"))

    with pytest.raises(RunAccessDenied) as exc:
        cancel(RUN_ID)

    assert exc.value.status_code == 403
    assert exc.value.error_code == "DEPLOYMENT_ACCESS_DENIED"


def test_a_run_outside_the_keys_reach_is_run_access_denied_with_its_own_code(
    transport,
):
    """An agent's key may cancel only the runs below its own run. The refusal is
    a RunAccessDenied, and it carries the code the platform SENT: it must not
    report DEPLOYMENT_ACCESS_DENIED, a different problem (whose run it is)."""
    transport.reply = (
        403,
        refusal(
            "AUTH_AGENT_KEY_OUT_OF_SCOPE",
            "This agent's key may cancel only the runs below its own run.",
        ),
    )

    with pytest.raises(RunAccessDenied) as exc:
        cancel(RUN_ID)

    assert exc.value.status_code == 403
    assert exc.value.error_code == "AUTH_AGENT_KEY_OUT_OF_SCOPE"
    assert "below its own run" in str(exc.value)
    assert exc.value.details["code"] == "AUTH_AGENT_KEY_OUT_OF_SCOPE"


def test_the_same_403_on_a_read_is_still_a_plain_read_error(transport):
    """The cancel maps the out-of-scope 403 onto RunAccessDenied; a READ does
    not, and must not start to: what the key may read is not whose run it is."""
    transport.reply = (403, refusal("AUTH_AGENT_KEY_OUT_OF_SCOPE", "no"))

    with pytest.raises(RunReadError) as exc:
        result(RUN_ID)

    assert not isinstance(exc.value, RunAccessDenied)
    assert exc.value.error_code == "AUTH_AGENT_KEY_OUT_OF_SCOPE"


def test_a_release_is_not_a_run(transport):
    transport.reply = (
        409,
        refusal(
            "DEPLOYMENT_NOT_A_RUN",
            "That is a release. Cancel it with `afy cancel <agent>`.",
        ),
    )

    with pytest.raises(NotARun) as exc:
        cancel(RUN_ID)

    assert exc.value.status_code == 409
    assert exc.value.error_code == "DEPLOYMENT_NOT_A_RUN"
    assert isinstance(exc.value, RunCancelError)
    assert "afy cancel" in str(exc.value)


@pytest.mark.parametrize(
    "status,code",
    [
        (404, "AGENT_NOT_FOUND"),
        (404, None),
        (403, "SUBSCRIPTION_SUSPENDED"),
        (403, None),
        (409, "DEPLOYMENT_TERMINAL_CANNOT_CANCEL"),
        (409, None),
    ],
)
def test_the_code_decides_not_the_status_alone(transport, status, code):
    # A status is a category the control plane reuses across every route; the
    # code is the thing it promises not to rename. A 409 grown for some other
    # reason must not arrive wearing DEPLOYMENT_NOT_A_RUN.
    transport.reply = (status, refusal(code) if code else {"detail": {}})

    with pytest.raises(RunCancelError) as exc:
        cancel(RUN_ID)

    assert type(exc.value) is RunCancelError
    assert exc.value.status_code == status
    assert exc.value.error_code == code


@pytest.mark.parametrize("status", [401, 429, 500, 502, 503])
def test_every_other_status_reports_what_arrived(transport, status):
    transport.reply = (status, refusal("SOMETHING_ELSE", "upstream said no"))

    with pytest.raises(RunCancelError) as exc:
        cancel(RUN_ID)

    assert type(exc.value) is RunCancelError
    assert exc.value.status_code == status
    assert exc.value.error_code == "SOMETHING_ELSE"
    assert "upstream said no" in str(exc.value)


def test_a_refusal_carries_the_time_the_control_plane_asked_for(transport):
    transport.retry_after = 9.0
    transport.reply = (503, {"detail": {"message": "busy"}})

    with pytest.raises(RunCancelError) as exc:
        cancel(RUN_ID)

    assert exc.value.retry_after == 9.0
    assert len(transport.calls) == 1, "the helper never retries a cancel itself"


def test_a_bare_string_detail_still_cancels_with_prose(transport):
    transport.reply = (500, {"detail": "Internal Server Error"})

    with pytest.raises(RunCancelError) as exc:
        cancel(RUN_ID)

    assert exc.value.error_code is None
    assert "Internal Server Error" in str(exc.value)


def test_a_body_free_refusal_still_names_the_status(transport):
    transport.reply = (502, None)

    with pytest.raises(RunCancelError) as exc:
        cancel(RUN_ID)

    assert exc.value.status_code == 502
    assert "502" in str(exc.value)


def test_the_two_shared_refusals_are_not_cancel_errors():
    """A caller that catches only RunCancelError misses the 404 and the 403, and
    the docs say so. AgentError sees all of them."""
    assert not issubclass(RunNotFound, RunCancelError)
    assert not issubclass(RunAccessDenied, RunCancelError)
    assert issubclass(NotARun, RunCancelError)
    for error in (RunNotFound, RunAccessDenied, RunCancelError, NotARun):
        assert issubclass(error, AgentError)
