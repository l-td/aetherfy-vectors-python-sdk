"""
`payload()`: the file first, the HTTP fallback second, `{}` for no input.

No network is touched. The fallback tests replace the module's own transport,
so what they pin is this module's request — not urllib's behaviour.
"""

import json

import pytest

import aetherfy_agent
from aetherfy_agent import payload
from aetherfy_agent.exceptions import PayloadUnavailable

AGENT_VARS = [
    "AETHERFY_SPAWN_PAYLOAD_PATH",
    "AETHERFY_API_URL",
    "AETHERFY_SPAWN_ID",
    "AETHERFY_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in AGENT_VARS:
        monkeypatch.delenv(name, raising=False)


def test_reads_the_payload_file(tmp_path, monkeypatch):
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"date": "2026-09-07", "items": [1, 2]}), "utf-8")
    monkeypatch.setenv("AETHERFY_SPAWN_PAYLOAD_PATH", str(path))

    assert payload() == {"date": "2026-09-07", "items": [1, 2]}


def test_empty_file_is_an_empty_payload(tmp_path, monkeypatch):
    """The normal case: a scheduled fire gets a file holding nothing."""
    path = tmp_path / "payload.json"
    path.write_text("", "utf-8")
    monkeypatch.setenv("AETHERFY_SPAWN_PAYLOAD_PATH", str(path))

    assert payload() == {}


def test_file_holding_an_empty_object_is_an_empty_payload(tmp_path, monkeypatch):
    path = tmp_path / "payload.json"
    path.write_text("{}", "utf-8")
    monkeypatch.setenv("AETHERFY_SPAWN_PAYLOAD_PATH", str(path))

    assert payload() == {}


def test_file_holding_null_is_an_empty_payload(tmp_path, monkeypatch):
    """`json.load` on a literal null yields None, which is not a payload."""
    path = tmp_path / "payload.json"
    path.write_text("null", "utf-8")
    monkeypatch.setenv("AETHERFY_SPAWN_PAYLOAD_PATH", str(path))

    assert payload() == {}


def test_unparseable_file_names_the_path(tmp_path, monkeypatch):
    path = tmp_path / "payload.json"
    path.write_text("{not json", "utf-8")
    monkeypatch.setenv("AETHERFY_SPAWN_PAYLOAD_PATH", str(path))

    with pytest.raises(PayloadUnavailable) as excinfo:
        payload()
    assert str(path) in str(excinfo.value)


def test_falls_back_to_http_when_the_variable_is_unset(monkeypatch):
    monkeypatch.setenv("AETHERFY_API_URL", "https://agents.aetherfy.com/api/v1")
    monkeypatch.setenv("AETHERFY_SPAWN_ID", "dep-42")
    monkeypatch.setenv("AETHERFY_API_KEY", "afy_test_key")

    calls = []

    def fake_request(method, url, *, api_key, ua, body=None, **kwargs):
        calls.append((method, url, api_key, ua, body))
        return 200, {"payload": {"date": "2026-09-07"}}

    monkeypatch.setattr(aetherfy_agent._http, "request_json", fake_request)

    assert payload() == {"date": "2026-09-07"}
    method, url, api_key, ua, body = calls[0]
    assert method == "GET"
    assert url == "https://agents.aetherfy.com/api/v1/deployments/dep-42/payload"
    assert api_key == "afy_test_key"
    assert body is None
    # An explicit User-Agent, always: urllib's default gets a 403 at the edge
    # that reads exactly like an auth failure.
    assert ua.startswith("aetherfy-agent-python/")


def test_falls_back_to_http_when_the_file_is_missing(tmp_path, monkeypatch):
    """The variable is set but the machine never wrote the file — the exact
    case the fallback exists for."""
    monkeypatch.setenv("AETHERFY_SPAWN_PAYLOAD_PATH", str(tmp_path / "nope.json"))
    monkeypatch.setenv("AETHERFY_API_URL", "https://agents.aetherfy.com/api/v1")
    monkeypatch.setenv("AETHERFY_SPAWN_ID", "dep-42")
    monkeypatch.setenv("AETHERFY_API_KEY", "afy_test_key")
    monkeypatch.setattr(
        aetherfy_agent._http,
        "request_json",
        lambda *a, **k: (200, {"payload": {"from": "fallback"}}),
    )

    assert payload() == {"from": "fallback"}


def test_fallback_empty_payload_is_an_empty_dict(monkeypatch):
    monkeypatch.setenv("AETHERFY_API_URL", "https://agents.aetherfy.com/api/v1")
    monkeypatch.setenv("AETHERFY_SPAWN_ID", "dep-42")
    monkeypatch.setenv("AETHERFY_API_KEY", "afy_test_key")
    monkeypatch.setattr(
        aetherfy_agent._http, "request_json", lambda *a, **k: (200, {"payload": {}})
    )

    assert payload() == {}


def test_fallback_error_status_raises(monkeypatch):
    monkeypatch.setenv("AETHERFY_API_URL", "https://agents.aetherfy.com/api/v1")
    monkeypatch.setenv("AETHERFY_SPAWN_ID", "dep-42")
    monkeypatch.setenv("AETHERFY_API_KEY", "afy_test_key")
    monkeypatch.setattr(
        aetherfy_agent._http,
        "request_json",
        lambda *a, **k: (
            404,
            {"detail": {"code": "DEPLOYMENT_NOT_FOUND", "message": "no such run"}},
        ),
    )

    with pytest.raises(PayloadUnavailable) as excinfo:
        payload()
    assert "404" in str(excinfo.value)
    assert "no such run" in str(excinfo.value)


def test_no_file_and_no_credentials_raises(monkeypatch):
    with pytest.raises(PayloadUnavailable) as excinfo:
        payload()
    assert "AETHERFY_SPAWN_PAYLOAD_PATH" in str(excinfo.value)


def test_nothing_reaches_the_network_on_the_file_path(tmp_path, monkeypatch):
    """A positive control for the mocking above: with a readable file, the
    transport is never called at all — so a test that mocks it and passes is
    genuinely exercising the fallback branch, not this one."""
    path = tmp_path / "payload.json"
    path.write_text('{"a": 1}', "utf-8")
    monkeypatch.setenv("AETHERFY_SPAWN_PAYLOAD_PATH", str(path))

    def explode(*args, **kwargs):
        raise AssertionError("the file path must not make a request")

    monkeypatch.setattr(aetherfy_agent._http, "request_json", explode)
    assert payload() == {"a": 1}
