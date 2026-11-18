"""
`result()` and `wait()`: the run a caller reads back, and the refusals the two
routes answer identically.

Pinned against aetherfy-control-plane `api/routes/deployments.py`:
`GET /deployments/{id}` and `GET /deployments/{id}/wait`, both returning
`DeploymentResponse`, both loading through `_load_customer_deployment` — which
is why a 404 and a 403 must arrive here identically whichever call was made.
The wait bound (1..60, default 30) is `WAIT_TIMEOUT_MIN_SECONDS` /
`WAIT_TIMEOUT_MAX_SECONDS` / `WAIT_TIMEOUT_DEFAULT_SECONDS` there.
"""

from datetime import datetime, timedelta, timezone

import pytest

import aetherfy_agent
from aetherfy_agent import result, wait
from aetherfy_agent._http import DEFAULT_TIMEOUT
from aetherfy_agent.exceptions import (
    NotRunningOnAgent,
    RunAccessDenied,
    RunNotFound,
    RunReadError,
    WaitTimeoutInvalid,
)
from aetherfy_agent.models import RunRetry

RUN_ID = "44444444-4444-4444-4444-444444444444"

# A finished run, exactly as DeploymentResponse serializes one — including the
# deploy-shaped fields a run carries but nobody reads off it, because `raw`
# promises the whole object.
FINISHED = {
    "id": RUN_ID,
    "agent_id": "6f1c2b7e-0a2d-4f8e-9c31-2b0d5a7e4411",
    "version": 7,
    "state": "completed",
    "is_ephemeral": True,
    "result": {"rows": 128},
    "result_error": None,
    "has_result": True,
    "error_message": None,
    "failure_code": None,
    "retry": None,
    "regions": ["iad"],
    "pending_regions": [],
    "created_at": "2026-09-08T09:04:11Z",
}


@pytest.fixture(autouse=True)
def on_a_machine(monkeypatch):
    monkeypatch.setenv("AETHERFY_API_URL", "https://agents.aetherfy.com/api/v1")
    monkeypatch.setenv("AETHERFY_API_KEY", "afy_test_key")


@pytest.fixture
def transport(monkeypatch):
    """Records the request and replies with whatever the test queued."""

    class Transport:
        def __init__(self):
            self.calls = []
            self.retry_after = None
            self.reply = (200, FINISHED)

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
            status, body = self.reply
            return status, body, self.retry_after

    fake = Transport()
    monkeypatch.setattr(aetherfy_agent._http, "request_json", fake)
    return fake


def refusal(code, message="nope", **extras):
    return {"detail": dict({"code": code, "message": message}, **extras)}


# --- THE REQUEST -------------------------------------------------------------


def test_result_reads_the_deployment_route(transport):
    result(RUN_ID)

    call = transport.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == ("https://agents.aetherfy.com/api/v1/deployments/" + RUN_ID)
    assert call["body"] is None
    assert call["api_key"] == "afy_test_key"
    assert call["ua"].startswith("aetherfy-agent-python/")


def test_a_trailing_slash_on_the_base_url_does_not_double(transport, monkeypatch):
    monkeypatch.setenv("AETHERFY_API_URL", "https://agents.aetherfy.com/api/v1/")
    result(RUN_ID)
    assert "/v1/deployments/" in transport.calls[0]["url"]


def test_wait_sends_the_default_timeout_the_server_documents(transport):
    wait(RUN_ID)
    assert transport.calls[0]["url"].endswith(
        "/deployments/" + RUN_ID + "/wait?timeout_seconds=30"
    )


def test_wait_sends_the_timeout_it_was_given(transport):
    wait(RUN_ID, timeout_seconds=45)
    assert transport.calls[0]["url"].endswith("/wait?timeout_seconds=45")


def test_the_socket_outlives_the_hold_the_server_promised(transport):
    # THE CLIENT'S BOUND MUST EXCEED THE SERVER'S. A wait the control plane is
    # about to answer at its own deadline must not be cut off here first and
    # reported as a transport failure — the one outcome a caller cannot tell
    # from a real network fault.
    wait(RUN_ID, timeout_seconds=60)
    assert transport.calls[0]["kwargs"]["timeout"] > 60


def test_wait_does_not_retry_a_dropped_connection(transport):
    # A retry would hold a SECOND full timeout and hand back a run up to twice
    # as late as the number the caller passed. Every other call in this module
    # retries once; this one buys the bound instead.
    wait(RUN_ID, timeout_seconds=30)
    assert transport.calls[0]["kwargs"]["retry_connection_errors"] is False


def test_result_keeps_the_one_retry_and_the_ordinary_timeout(transport):
    result(RUN_ID)
    kwargs = transport.calls[0]["kwargs"]
    # Read explicitly, not with a default: a `.get(key, want)` here would pass
    # just as happily on a call that never sent the argument at all.
    assert kwargs["retry_connection_errors"] is True
    assert kwargs["timeout"] == DEFAULT_TIMEOUT


# --- THE RUN -----------------------------------------------------------------


def test_a_finished_run_carries_its_answer(transport):
    run = result(RUN_ID)

    assert run.id == RUN_ID
    assert run.agent_id == "6f1c2b7e-0a2d-4f8e-9c31-2b0d5a7e4411"
    assert run.state == "completed"
    assert run.result == {"rows": 128}
    assert run.result_error is None
    assert run.has_result is True
    assert run.is_ephemeral is True
    assert run.error_message is None
    assert run.failure_code is None
    assert run.retry is None


def test_raw_keeps_the_whole_object_including_what_is_not_named(transport):
    run = result(RUN_ID)
    assert run.raw == FINISHED
    assert run.raw["version"] == 7
    assert run.raw["regions"] == ["iad"]


def test_a_refused_result_is_not_a_result(transport):
    # image_generator.py's _collect_result classifies an oversized or unparsable
    # file, and Deployment.has_result is False whenever result_error is set:
    # "a refused result is not a result".
    transport.reply = (
        200,
        dict(FINISHED, result=None, result_error="too_large", has_result=False),
    )
    run = result(RUN_ID)

    assert run.result is None
    assert run.result_error == "too_large"
    assert run.has_result is False


@pytest.mark.parametrize("call", [result, wait])
def test_a_failed_run_is_returned_with_its_failure_code_beside_its_sentence(transport, call):
    # The code is the value to switch on, the sentence the prose; a failed run is
    # an answer, so it is returned, never raised.
    transport.reply = (
        200,
        dict(FINISHED, state="failed", result=None, has_result=False,
             error_message="run exited with code 3", failure_code="RUN_EXITED_NONZERO"),
    )
    run = call(RUN_ID)

    assert (run.state, run.failure_code, run.error_message) == (
        "failed", "RUN_EXITED_NONZERO", "run exited with code 3")


def test_control_a_run_without_the_field_has_no_failure_code(transport):
    # A run that failed before the platform recorded codes is sent without one.
    body = dict(FINISHED, state="failed", error_message="it failed")
    del body["failure_code"]
    transport.reply = (200, body)

    assert result(RUN_ID).failure_code is None


# What the control plane sends while it retries an attempt that met a temporary
# unavailability. The run's `state` is not this field's business: a retry is read
# off the run whatever it is.
RETRYING = {
    "code": "PLATFORM_TEMPORARILY_UNAVAILABLE",
    "attempt": 2,
    "max_attempts": 4,
    "next_attempt_at": "2026-11-18T03:12:45Z",
}
NEXT_ATTEMPT = datetime(2026, 11, 18, 3, 12, 45, tzinfo=timezone.utc)


@pytest.mark.parametrize("call", [result, wait])
def test_a_run_being_retried_carries_its_retry_from_both_reads(transport, call):
    transport.reply = (200, dict(FINISHED, retry=RETRYING))

    run = call(RUN_ID)

    assert run.retry == RunRetry(
        code="PLATFORM_TEMPORARILY_UNAVAILABLE",
        attempt=2,
        max_attempts=4,
        next_attempt_at=NEXT_ATTEMPT,
    )
    assert run.raw["retry"] == RETRYING


@pytest.mark.parametrize(
    "sent,instant",
    [
        ("2026-11-18T03:12:45Z", NEXT_ATTEMPT),
        ("2026-11-18T03:12:45.250000Z", NEXT_ATTEMPT.replace(microsecond=250000)),
        ("2026-11-18T03:12Z", NEXT_ATTEMPT.replace(second=0)),
        ("2026-11-18T03:12:45+00:00", NEXT_ATTEMPT),
        # The same instant, said in another offset, is read in UTC.
        ("2026-11-18T05:12:45+02:00", NEXT_ATTEMPT),
        ("2026-11-17T22:12:45-05:00", NEXT_ATTEMPT),
    ],
)
def test_next_attempt_at_is_an_aware_utc_datetime(transport, sent, instant):
    transport.reply = (200, dict(FINISHED, retry=dict(RETRYING, next_attempt_at=sent)))

    when = result(RUN_ID).retry.next_attempt_at

    assert when == instant
    # `==` compares instants, so an unconverted +02:00 would pass it: the offset
    # is what says the value is in UTC.
    assert when.utcoffset() == timedelta(0)
    assert when.tzinfo is timezone.utc


def test_a_run_not_being_retried_has_no_retry(transport):
    transport.reply = (200, dict(FINISHED, retry=None))

    assert result(RUN_ID).retry is None


def test_control_a_run_without_the_field_has_no_retry(transport):
    # A run from before the platform sent it, or one that is not a run being read
    # back with it, has no key at all: that reads as null too.
    body = dict(FINISHED)
    del body["retry"]
    transport.reply = (200, body)

    assert result(RUN_ID).retry is None


@pytest.mark.parametrize(
    "sent",
    [
        pytest.param("soon", id="a string"),
        pytest.param([2, 4], id="a list"),
        pytest.param({}, id="an empty object"),
        *[
            pytest.param(
                {key: value for key, value in RETRYING.items() if key != missing},
                id="without " + missing,
            )
            for missing in RETRYING
        ],
        pytest.param(dict(RETRYING, code=7), id="code is not a string"),
        pytest.param(dict(RETRYING, attempt="2"), id="attempt is a string"),
        pytest.param(dict(RETRYING, attempt=True), id="attempt is a boolean"),
        pytest.param(dict(RETRYING, attempt=2.5), id="attempt is a fraction"),
        pytest.param(dict(RETRYING, max_attempts=None), id="max_attempts is null"),
        pytest.param(dict(RETRYING, next_attempt_at=1794971565), id="a number of seconds"),
        pytest.param(dict(RETRYING, next_attempt_at="tomorrow"), id="not a timestamp"),
        pytest.param(dict(RETRYING, next_attempt_at=""), id="an empty timestamp"),
        pytest.param(
            dict(RETRYING, next_attempt_at="2026-11-18T03:12:45"),
            id="a timestamp with no offset",
        ),
        pytest.param(
            dict(RETRYING, next_attempt_at="2026-11-18"), id="a date with no time"
        ),
        pytest.param(
            dict(RETRYING, next_attempt_at="2026-13-01T00:00:00Z"),
            id="a month that does not exist",
        ),
        pytest.param(
            dict(RETRYING, next_attempt_at="0001-01-01T00:00:00+05:00"),
            id="a timestamp that has no UTC form",
        ),
    ],
)
def test_a_retry_that_is_not_shaped_so_reads_as_none_and_the_run_is_still_returned(
    transport, sent
):
    # An optional field of the wrong type reads as absent, as `position` does on
    # a spawn: the run itself is still an answer, and raw keeps what was sent.
    transport.reply = (200, dict(FINISHED, retry=sent))

    run = result(RUN_ID)

    assert run.retry is None
    assert run.state == "completed"
    assert run.raw["retry"] == sent


def test_a_run_that_returned_nothing_is_not_a_run_that_failed(transport):
    transport.reply = (
        200,
        dict(FINISHED, result=None, result_error=None, has_result=False),
    )
    run = result(RUN_ID)

    assert run.state == "completed"
    assert run.result is None
    assert run.result_error is None


def test_a_wait_that_times_out_returns_the_run_it_has(transport):
    # A TIMEOUT IS NOT AN ERROR: the route answers 200 with the run exactly as
    # it stands, and `active` on a run means it is executing right now.
    transport.reply = (
        200,
        dict(FINISHED, state="active", result=None, has_result=False),
    )
    run = wait(RUN_ID, timeout_seconds=1)

    assert run.state == "active"
    assert run.result is None


@pytest.mark.parametrize(
    "state", ["waiting", "active", "completed", "failed", "cancelled"]
)
@pytest.mark.parametrize("call", [result, wait])
def test_a_run_in_any_state_is_returned_as_it_stands(transport, call, state):
    # The platform owns the vocabulary: the helper adds no enum and rejects no
    # state, so `waiting` (in the account's line) and `cancelled` (stopped from
    # outside) arrive like the rest.
    transport.reply = (
        200,
        dict(FINISHED, state=state, result=None, has_result=False),
    )

    assert call(RUN_ID).state == state


def test_the_place_in_the_line_is_in_raw_while_a_run_waits(transport):
    transport.reply = (
        200,
        dict(FINISHED, state="waiting", result=None, has_result=False, position=4),
    )

    run = result(RUN_ID)

    assert run.state == "waiting"
    assert run.raw["position"] == 4


def test_a_body_that_is_not_an_object_is_not_a_run(transport):
    transport.reply = (200, "OK")
    with pytest.raises(RunReadError):
        result(RUN_ID)


# --- THE REFUSALS ------------------------------------------------------------


@pytest.mark.parametrize("call", [result, wait])
def test_an_unknown_id_is_the_same_refusal_from_both_reads(transport, call):
    transport.reply = (404, refusal("DEPLOYMENT_NOT_FOUND", "Deployment x not found"))

    with pytest.raises(RunNotFound) as exc:
        call(RUN_ID)

    assert exc.value.status_code == 404
    assert exc.value.error_code == "DEPLOYMENT_NOT_FOUND"
    assert "not found" in str(exc.value)


@pytest.mark.parametrize("call", [result, wait])
def test_someone_elses_run_is_the_same_refusal_from_both_reads(transport, call):
    # The control plane loads both routes through _load_customer_deployment for
    # exactly this reason: a caller must not have to know which one it called
    # to handle the error.
    transport.reply = (403, refusal("DEPLOYMENT_ACCESS_DENIED", "Access denied"))

    with pytest.raises(RunAccessDenied) as exc:
        call(RUN_ID)

    assert exc.value.status_code == 403
    assert exc.value.error_code == "DEPLOYMENT_ACCESS_DENIED"


@pytest.mark.parametrize("call", [result, wait])
def test_a_run_outside_the_agent_keys_scope_is_a_read_error_not_access_denied(
    transport, call
):
    """The agent's own key reads only its own run and the runs it spawned
    (aetherfy-control-plane api/middleware/agent_scope.py). Any other run is
    403 AUTH_AGENT_KEY_OUT_OF_SCOPE -- a different code from another account's
    run, so it must not be dressed up as RunAccessDenied."""
    transport.reply = (
        403,
        refusal(
            "AUTH_AGENT_KEY_OUT_OF_SCOPE", "This agent's API key cannot call it"
        ),
    )

    with pytest.raises(RunReadError) as exc:
        call(RUN_ID)

    assert not isinstance(exc.value, RunAccessDenied)
    assert exc.value.status_code == 403
    assert exc.value.error_code == "AUTH_AGENT_KEY_OUT_OF_SCOPE"


def test_the_two_refusals_are_provably_distinct(transport):
    transport.reply = (404, refusal("DEPLOYMENT_NOT_FOUND"))
    with pytest.raises(RunNotFound):
        result(RUN_ID)

    transport.reply = (403, refusal("DEPLOYMENT_ACCESS_DENIED"))
    with pytest.raises(RunReadError) as exc:
        result(RUN_ID)
    assert not isinstance(exc.value, RunNotFound)


@pytest.mark.parametrize(
    "status,code",
    [
        (404, "AGENT_NOT_FOUND"),
        (404, None),
        (403, "SUBSCRIPTION_SUSPENDED"),
        (422, "VALIDATION_ERROR"),
    ],
)
def test_the_code_decides_not_the_status_alone(transport, status, code):
    # A status is a category the control plane reuses across every route; the
    # code is the thing it promises not to rename. A 404 the platform grows for
    # some other reason must not arrive wearing DEPLOYMENT_NOT_FOUND — the
    # caller would branch on a code nothing sent.
    transport.reply = (status, refusal(code) if code else {"detail": {}})

    with pytest.raises(RunReadError) as exc:
        result(RUN_ID)

    assert type(exc.value) is RunReadError
    assert exc.value.status_code == status
    assert exc.value.error_code == code


@pytest.mark.parametrize("status", [401, 429, 500, 502, 503])
def test_every_other_status_reports_what_arrived(transport, status):
    transport.reply = (status, refusal("SOMETHING_ELSE", "upstream said no"))

    with pytest.raises(RunReadError) as exc:
        result(RUN_ID)

    assert exc.value.status_code == status
    assert exc.value.error_code == "SOMETHING_ELSE"
    assert "upstream said no" in str(exc.value)


def test_a_bare_string_detail_still_reads(transport):
    # FastAPI's own default for a route that never reached our error handling:
    # prose, no code.
    transport.reply = (500, {"detail": "Internal Server Error"})

    with pytest.raises(RunReadError) as exc:
        result(RUN_ID)

    assert exc.value.error_code is None
    assert "Internal Server Error" in str(exc.value)


def test_a_body_free_refusal_still_names_the_status(transport):
    transport.reply = (502, None)

    with pytest.raises(RunReadError) as exc:
        result(RUN_ID)

    assert exc.value.status_code == 502
    assert "502" in str(exc.value)


# --- THE WAIT BOUND ----------------------------------------------------------


@pytest.mark.parametrize("timeout", [0, -1, 61, 3600])
def test_a_timeout_outside_the_bound_never_leaves_the_process(transport, timeout):
    with pytest.raises(ValueError) as exc:
        wait(RUN_ID, timeout_seconds=timeout)

    assert "between 1 and 60" in str(exc.value)
    # POSITIVE CONTROL on the claim: no request was sent, so the check really
    # is client-side and not the server's 422 arriving in a different coat.
    assert transport.calls == []


@pytest.mark.parametrize("timeout", [1, 60])
def test_the_bound_is_inclusive_at_both_ends(transport, timeout):
    wait(RUN_ID, timeout_seconds=timeout)
    assert transport.calls[0]["url"].endswith("?timeout_seconds=" + str(timeout))


def test_the_servers_own_422_arrives_as_a_type(transport):
    # The client-side check has a copy of the server's bound, and a copy can go
    # stale. When it does, the refusal is something to read rather than a bare
    # 422 the helper had no shape for.
    transport.reply = (
        422,
        refusal(
            "DEPLOYMENT_WAIT_TIMEOUT_INVALID",
            "timeout_seconds must be between 1 and 30; got 45.",
            field="timeout_seconds",
            min_seconds=1,
            max_seconds=30,
        ),
    )

    with pytest.raises(WaitTimeoutInvalid) as exc:
        wait(RUN_ID, timeout_seconds=45)

    assert exc.value.status_code == 422
    assert exc.value.error_code == "DEPLOYMENT_WAIT_TIMEOUT_INVALID"
    assert exc.value.details["max_seconds"] == 30


# --- OFF A MACHINE -----------------------------------------------------------


@pytest.mark.parametrize("variable", ["AETHERFY_API_URL", "AETHERFY_API_KEY"])
@pytest.mark.parametrize("call", [result, wait])
def test_a_missing_variable_is_named(transport, monkeypatch, variable, call):
    monkeypatch.delenv(variable, raising=False)

    with pytest.raises(NotRunningOnAgent) as exc:
        call(RUN_ID)

    assert exc.value.variable == variable
    assert transport.calls == []


@pytest.mark.parametrize("call", [result, wait])
def test_an_empty_run_id_is_refused_before_a_request(transport, call):
    # Without this the URL ends in a bare `/deployments/`, which is the LIST
    # route — a 200 carrying an array, and `str(body.get("id"))` on a list is
    # the string "None". A wrong answer, not an error.
    with pytest.raises(ValueError):
        call("")

    assert transport.calls == []


# --- THE PUBLIC SURFACE ------------------------------------------------------


def test_the_wait_bound_is_readable_in_both_languages():
    # ONE API IN TWO LANGUAGES. The JavaScript helper exports these three from
    # its entry point; a task ported between the two must not find the bound
    # readable in one and missing from the other's declared surface.
    import aetherfy_agent

    for name in (
        "WAIT_TIMEOUT_MIN_SECONDS",
        "WAIT_TIMEOUT_MAX_SECONDS",
        "WAIT_TIMEOUT_DEFAULT_SECONDS",
    ):
        assert name in aetherfy_agent.__all__, name
        assert isinstance(getattr(aetherfy_agent, name), int)


def test_the_default_the_signature_uses_is_the_one_it_publishes():
    # Two numbers that could disagree: the constant a caller reads and the
    # default wait() actually applies.
    import inspect

    import aetherfy_agent

    signature = inspect.signature(aetherfy_agent.wait)
    assert (
        signature.parameters["timeout_seconds"].default
        == aetherfy_agent.WAIT_TIMEOUT_DEFAULT_SECONDS
    )


def test_a_run_id_cannot_walk_out_of_its_route(transport):
    # UNESCAPED, `../agents/x` normalises to a DIFFERENT route before the
    # request leaves, and its answer is parsed as though it were a run: `id`
    # becomes the agent's, `state` becomes the string "None". A wrong object
    # read as the right one, silently. Quoted, the platform answers 404.
    result("../agents/other")

    url = transport.calls[0]["url"]
    assert url == ("https://agents.aetherfy.com/api/v1/deployments/..%2Fagents%2Fother")
    assert "/agents/" not in url


def test_the_wait_route_quotes_the_id_too(transport):
    wait("../agents/other", timeout_seconds=5)
    assert transport.calls[0]["url"] == (
        "https://agents.aetherfy.com/api/v1/deployments/..%2Fagents%2Fother"
        "/wait?timeout_seconds=5"
    )


# -- Retry-After --------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 503])
def test_a_read_refusal_carries_the_time_the_control_plane_asked_for(transport, status):
    transport.retry_after = 9.0
    transport.reply = (status, {"detail": {"message": "slow down"}})
    with pytest.raises(RunReadError) as excinfo:
        result("dep-1")
    assert excinfo.value.status_code == status
    assert excinfo.value.retry_after == 9.0
    assert len(transport.calls) == 1, "the helper never retries a read itself"


def test_a_read_refusal_without_the_header_has_no_time(transport):
    transport.reply = (503, {"detail": {"message": "busy"}})
    with pytest.raises(RunReadError) as excinfo:
        result("dep-1")
    assert excinfo.value.retry_after is None
