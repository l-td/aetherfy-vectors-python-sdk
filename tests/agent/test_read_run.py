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
            return self.reply

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
