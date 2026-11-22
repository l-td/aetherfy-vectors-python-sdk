"""
`service_run()`: the handle a service's run handler uses to spawn, read, wait
on and cancel with THE RUN'S OWN KEY.

Pinned against the run queue's contract: the platform sets `X-Aetherfy-Run` (the
run's id) and `X-Aetherfy-Run-Key` (the run's key) on the loopback call to
`POST /aetherfy/run`, and removes both from public requests. The module-level
functions are unchanged and keep reading `AETHERFY_API_KEY`.

The framework tests use the real `starlette` and `werkzeug` header classes
(`requirements-ci.txt` installs them for exactly this): a fake that mimicked
them would prove nothing about them.
"""

import http.client
import inspect
from concurrent.futures import ThreadPoolExecutor

import pytest
from requests.structures import CaseInsensitiveDict
from starlette.datastructures import Headers as StarletteHeaders
from werkzeug.datastructures import EnvironHeaders, Headers as WerkzeugHeaders

import aetherfy_agent
from aetherfy_agent import cancel, result, service_run, spawn, wait
from aetherfy_agent.exceptions import (
    NotARun,
    NotAServiceRun,
    RunLineFull,
    RunNotFound,
)
from aetherfy_agent.models import Cancellation, Run, Spawn

RUN_ID = "77777777-7777-7777-7777-777777777777"
CHILD_ID = "88888888-8888-8888-8888-888888888888"
RUN_KEY = "afy_run_key_for_this_run"
RELEASE_KEY = "afy_release_key_of_the_service"

ACCEPTED = {
    "spawn_id": CHILD_ID,
    "job_id": None,
    "child_agent_id": "33333333-3333-3333-3333-333333333333",
    "workspace": "research",
    "region": "us-east-1",
    "status": "waiting",
    "position": 2,
    "estimated_start": "~1s",
}
RUN_BODY = {
    "id": CHILD_ID,
    "agent_id": "6f1c2b7e-0a2d-4f8e-9c31-2b0d5a7e4411",
    "state": "completed",
    "is_ephemeral": True,
    "result": {"rows": 1},
    "has_result": True,
}
CANCELLED = {
    "run": dict(RUN_BODY, state="cancelled"),
    "outcome": "cancelled",
    "message": "done",
    "cancelled_count": 1,
}


@pytest.fixture(autouse=True)
def in_a_service(monkeypatch):
    """A service machine: the release's key in the environment, and the URLs the
    platform injects."""
    monkeypatch.setenv("AETHERFY_API_URL", "https://agents.aetherfy.com/api/v1")
    monkeypatch.setenv(
        "AETHERFY_SPAWN_URL",
        "https://agents.aetherfy.com/api/v1/agents/{id}/spawn",
    )
    monkeypatch.setenv("AETHERFY_AGENT_ID", "the-service-agent")
    monkeypatch.setenv("AETHERFY_API_KEY", RELEASE_KEY)


def run_headers():
    return {"X-Aetherfy-Run": RUN_ID, "X-Aetherfy-Run-Key": RUN_KEY}


@pytest.fixture
def routed(transport):
    """The transport, answering each route with a plausible success, so one
    handle can be driven through all four calls without queueing replies."""

    def by_url(method, url):
        if url.endswith("/spawn"):
            return 202, ACCEPTED
        if url.endswith("/cancel"):
            return 200, CANCELLED
        return 200, RUN_BODY

    transport.route = by_url
    return transport


# --- READING THE HEADERS -----------------------------------------------------


def _starlette():
    return StarletteHeaders(headers=run_headers())


def _starlette_lowercase():
    return StarletteHeaders(
        headers={"x-aetherfy-run": RUN_ID, "x-aetherfy-run-key": RUN_KEY}
    )


def _werkzeug_headers():
    return WerkzeugHeaders(list(run_headers().items()))


def _werkzeug_environ():
    # What Flask's `request.headers` is: the WSGI environ, headers as HTTP_*.
    return EnvironHeaders(
        {
            "HTTP_X_AETHERFY_RUN": RUN_ID,
            "HTTP_X_AETHERFY_RUN_KEY": RUN_KEY,
        }
    )


def _case_insensitive_dict():
    return CaseInsensitiveDict(run_headers())


def _http_message():
    # What `http.server` hands a handler as `self.headers`.
    message = http.client.HTTPMessage()
    message["X-Aetherfy-Run"] = RUN_ID
    message["X-Aetherfy-Run-Key"] = RUN_KEY
    return message


MAPPINGS = {
    "dict, as sent": run_headers,
    "dict, lower case": lambda: {
        "x-aetherfy-run": RUN_ID,
        "x-aetherfy-run-key": RUN_KEY,
    },
    "dict, upper case": lambda: {
        "X-AETHERFY-RUN": RUN_ID,
        "X-AETHERFY-RUN-KEY": RUN_KEY,
    },
    "dict, mixed case": lambda: {
        "x-AetherFY-run": RUN_ID,
        "X-aetherfy-RUN-key": RUN_KEY,
    },
    "starlette Headers": _starlette,
    "starlette Headers, lower case": _starlette_lowercase,
    "werkzeug Headers (flask)": _werkzeug_headers,
    "werkzeug EnvironHeaders (flask)": _werkzeug_environ,
    "requests CaseInsensitiveDict": _case_insensitive_dict,
    "http.client HTTPMessage": _http_message,
}


@pytest.mark.parametrize("make", MAPPINGS.values(), ids=MAPPINGS.keys())
def test_the_run_and_its_key_are_read_from_any_header_mapping(make):
    handle = service_run(make())

    assert handle.run_id == RUN_ID
    # The key is private; what proves it was read is what the handle SENDS, in
    # the tests below. Here, that it is held and is not the release's.
    assert handle._api_key == RUN_KEY


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Content-Type": "application/json", "Authorization": "Bearer x"},
        {"X-Aetherfy-Run": RUN_ID},
        {"X-Aetherfy-Run-Key": RUN_KEY},
        {"X-Aetherfy-Run": "", "X-Aetherfy-Run-Key": RUN_KEY},
        {"X-Aetherfy-Run": RUN_ID, "X-Aetherfy-Run-Key": ""},
        {"X-Aetherfy-Run": "   ", "X-Aetherfy-Run-Key": "  "},
        # Look-alikes are not the headers.
        {"X-Aetherfy-Run-Id": RUN_ID, "X-Aetherfy-Run-Keys": RUN_KEY},
        {"Aetherfy-Run": RUN_ID, "Aetherfy-Run-Key": RUN_KEY},
    ],
)
def test_headers_without_the_run_and_its_key_are_not_a_service_run(headers):
    with pytest.raises(NotAServiceRun):
        service_run(headers)


@pytest.mark.parametrize(
    "headers,missing",
    [
        ({}, ("X-Aetherfy-Run", "X-Aetherfy-Run-Key")),
        ({"X-Aetherfy-Run": RUN_ID}, ("X-Aetherfy-Run-Key",)),
        ({"x-aetherfy-run-key": RUN_KEY}, ("X-Aetherfy-Run",)),
    ],
)
def test_the_error_says_which_headers_were_missing(headers, missing):
    with pytest.raises(NotAServiceRun) as exc:
        service_run(headers)

    assert exc.value.missing == missing
    # What to do about it: the one place this is for, and why a public request
    # fails it.
    assert "POST /aetherfy/run" in str(exc.value)
    assert "public request" in str(exc.value)


def test_a_public_request_is_not_a_service_run():
    """Headers a browser or a curl sends. The platform removes the two headers
    from public traffic, so a handler that gets here without them is not inside
    a run."""
    with pytest.raises(NotAServiceRun):
        service_run(
            StarletteHeaders(headers={"user-agent": "curl/8", "accept": "*/*"})
        )


def test_the_handle_never_shows_its_key():
    handle = service_run(run_headers())

    assert RUN_KEY not in repr(handle)
    assert RUN_ID in repr(handle)
    assert RUN_KEY not in str(handle)


# --- THE KEY EACH CALL SENDS -------------------------------------------------


def test_every_call_through_the_handle_sends_the_runs_key(routed):
    handle = service_run(run_headers())

    handle.spawn("worker", {"n": 1})
    handle.result(CHILD_ID)
    handle.wait(CHILD_ID, timeout_seconds=5)
    handle.cancel(CHILD_ID)

    assert [call["method"] for call in routed.calls] == ["POST", "GET", "GET", "POST"]
    assert [call["api_key"] for call in routed.calls] == [RUN_KEY] * 4


def test_the_module_level_functions_send_the_environments_key(routed):
    """UNCHANGED: in a service that is the release's key, right for the
    service's own traffic."""
    spawn("worker", {"n": 1})
    result(CHILD_ID)
    wait(CHILD_ID, timeout_seconds=5)
    cancel(CHILD_ID)

    assert [call["api_key"] for call in routed.calls] == [RELEASE_KEY] * 4


def test_the_two_keys_never_cross_in_one_process(routed):
    handle = service_run(run_headers())

    handle.result(CHILD_ID)
    result(CHILD_ID)
    handle.result(CHILD_ID)

    assert [call["api_key"] for call in routed.calls] == [
        RUN_KEY,
        RELEASE_KEY,
        RUN_KEY,
    ]


def test_the_handle_needs_no_key_in_the_environment(routed, monkeypatch):
    """Its key is the run's, from the headers. The environment's key is neither
    read nor required."""
    monkeypatch.delenv("AETHERFY_API_KEY")
    handle = service_run(run_headers())

    assert handle.result(CHILD_ID).state == "completed"
    assert routed.calls[0]["api_key"] == RUN_KEY
    with pytest.raises(aetherfy_agent.NotRunningOnAgent):
        result(CHILD_ID)


def test_a_worker_thread_uses_the_handle_it_was_given(routed):
    """WHY THE HANDLE IS EXPLICIT. The handle is passed to the work, so work
    handed to a thread pool still uses the run's key; an ambient context would
    not have crossed into the thread."""
    handle = service_run(run_headers())

    with ThreadPoolExecutor(max_workers=2) as pool:
        pool.submit(handle.spawn, "worker", {"n": 1}).result()
        pool.submit(handle.cancel, CHILD_ID).result()

    assert [call["api_key"] for call in routed.calls] == [RUN_KEY, RUN_KEY]


def test_the_handle_makes_the_same_requests_as_the_module_functions(routed):
    """SAME ENDPOINTS, BODIES AND TRANSPORT OPTIONS: only the key differs."""
    handle = service_run(run_headers())

    handle.spawn("worker", {"n": 1})
    handle.result(CHILD_ID)
    handle.wait(CHILD_ID, timeout_seconds=45)
    handle.cancel(CHILD_ID)
    spawn("worker", {"n": 1})
    result(CHILD_ID)
    wait(CHILD_ID, timeout_seconds=45)
    cancel(CHILD_ID)

    through_handle, module_level = routed.calls[:4], routed.calls[4:]
    # A spawn's request_id is made once per CALL (its idempotency), so the two
    # spawns differ by it alone: both carry one, and nothing else differs.
    for call in (through_handle[0], module_level[0]):
        assert call["body"].pop("request_id")
    for a, b in zip(through_handle, module_level):
        assert {k: v for k, v in a.items() if k != "api_key"} == {
            k: v for k, v in b.items() if k != "api_key"
        }


@pytest.mark.parametrize("name", ["spawn", "result", "wait", "cancel"])
def test_the_handles_signatures_are_the_module_functions(name):
    module_level = inspect.signature(getattr(aetherfy_agent, name))
    on_handle = inspect.signature(getattr(aetherfy_agent.ServiceRun, name))

    assert list(on_handle.parameters)[0] == "self"
    handle_params = list(on_handle.parameters.values())[1:]
    assert [
        (p.name, p.kind, p.default, p.annotation) for p in handle_params
    ] == [
        (p.name, p.kind, p.default, p.annotation)
        for p in module_level.parameters.values()
    ]
    assert on_handle.return_annotation == module_level.return_annotation


# --- THE SAME ANSWERS AND THE SAME ERRORS ------------------------------------


def test_the_handle_returns_the_same_types(routed):
    handle = service_run(run_headers())

    spawned = handle.spawn("worker")
    assert isinstance(spawned, Spawn)
    assert spawned.status == "waiting"
    assert spawned.position == 2
    assert spawned.job_id is None
    assert isinstance(handle.result(CHILD_ID), Run)
    assert isinstance(handle.wait(CHILD_ID), Run)
    assert isinstance(handle.cancel(CHILD_ID), Cancellation)


def test_the_handle_raises_the_same_errors(transport):
    handle = service_run(run_headers())

    transport.reply = (
        429,
        {"detail": {"code": "AGENT_RUN_LINE_FULL", "message": "full"}},
    )
    with pytest.raises(RunLineFull):
        handle.spawn("worker")

    transport.reply = (
        404,
        {"detail": {"code": "DEPLOYMENT_NOT_FOUND", "message": "no"}},
    )
    with pytest.raises(RunNotFound):
        handle.result(CHILD_ID)

    transport.reply = (
        409,
        {"detail": {"code": "DEPLOYMENT_NOT_A_RUN", "message": "release"}},
    )
    with pytest.raises(NotARun):
        handle.cancel(CHILD_ID)


@pytest.mark.parametrize("timeout", [0, 61])
def test_the_handle_checks_the_wait_bound_before_sending(transport, timeout):
    handle = service_run(run_headers())

    with pytest.raises(ValueError):
        handle.wait(CHILD_ID, timeout_seconds=timeout)

    assert transport.calls == []


def test_a_module_level_wait_on_the_runs_child_is_the_releases_to_refuse(transport):
    """What the docs warn about: a run spawned through the handle belongs to the
    run's tree, so a module-level wait (the release's key) is refused by the
    control plane. The helper reports the refusal as it came."""
    from aetherfy_agent.exceptions import RunReadError

    transport.reply = (
        403,
        {
            "detail": {
                "code": "AUTH_AGENT_KEY_OUT_OF_SCOPE",
                "message": "Wait through the run's handle: service_run(headers).",
            }
        },
    )

    with pytest.raises(RunReadError) as exc:
        wait(CHILD_ID, timeout_seconds=5)

    assert exc.value.error_code == "AUTH_AGENT_KEY_OUT_OF_SCOPE"
    assert transport.calls[0]["api_key"] == RELEASE_KEY
