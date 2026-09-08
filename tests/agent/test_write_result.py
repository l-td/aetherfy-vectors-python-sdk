"""
`write_result()`: the bytes that land on disk, and the one refusal that
happens before they do.

Pinned against aetherfy-control-plane `orchestrator/image_generator.py`, whose
task supervisor prepares the file, injects its path as
`AETHERFY_SPAWN_RESULT_PATH`, and afterwards reads it back with
`len(raw) > _RESULT_MAX_BYTES` over the RAW BYTES — which is why the check here
measures the encoded bytes and not the string. The cap itself is injected as
`AETHERFY_RUN_INLINE_MAX_BYTES` (`orchestrator/fly_manager.py`).
"""

import json

import pytest

from aetherfy_agent import write_result
from aetherfy_agent.exceptions import NotRunningOnAgent, ResultTooLarge

CAP = 64


@pytest.fixture
def result_path(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    monkeypatch.setenv("AETHERFY_SPAWN_RESULT_PATH", str(path))
    monkeypatch.setenv("AETHERFY_RUN_INLINE_MAX_BYTES", str(CAP))
    return path


def test_the_file_holds_exactly_the_json_the_platform_will_read(result_path):
    write_result({"rows": 128, "date": "2026-09-08"})

    raw = result_path.read_bytes()
    assert json.loads(raw.decode("utf-8")) == {"rows": 128, "date": "2026-09-08"}


def test_a_result_is_not_only_an_object(result_path):
    # The control plane's column is `Any`, so a list or a scalar is a result.
    write_result([1, 2, 3])
    assert json.loads(result_path.read_text(encoding="utf-8")) == [1, 2, 3]


def test_none_writes_the_literal_null_the_platform_reads_as_nothing(result_path):
    # image_generator.py's _collect_result: "A child that writes the literal
    # `null` reads here as 'returned nothing'." Documented, not invented.
    write_result(None)
    assert result_path.read_bytes() == b"null"


def test_the_last_call_wins(result_path):
    write_result({"attempt": 1})
    write_result({"attempt": 2})
    assert json.loads(result_path.read_text(encoding="utf-8")) == {"attempt": 2}


def test_no_result_path_names_the_variable_and_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv("AETHERFY_SPAWN_RESULT_PATH", raising=False)
    monkeypatch.setenv("AETHERFY_RUN_INLINE_MAX_BYTES", str(CAP))

    with pytest.raises(NotRunningOnAgent) as exc:
        write_result({"rows": 1})

    assert exc.value.variable == "AETHERFY_SPAWN_RESULT_PATH"
    assert list(tmp_path.iterdir()) == []


def test_the_message_does_not_claim_the_platform_always_sets_the_path(monkeypatch):
    # THE DEFAULT SENTENCE IS FALSE FOR THIS ONE VARIABLE, and this is the only
    # variable in the module for which it is. image_generator.py offers the
    # result path only inside `if _RESULT_MAX_BYTES > 0`; a task machine with no
    # cap, and every service machine, reaches this error on a platform that
    # deliberately did not set it. Telling that customer "the platform sets it
    # before your entrypoint starts" sends them to debug their own code.
    monkeypatch.delenv("AETHERFY_SPAWN_RESULT_PATH", raising=False)

    with pytest.raises(NotRunningOnAgent) as exc:
        write_result({"rows": 1})

    message = str(exc.value)
    assert "the platform sets" not in message
    assert "AETHERFY_RUN_INLINE_MAX_BYTES" in message
    # ...and it says WRITE, not read, because that is what this call does.
    assert "writes its answer to" in message


def test_the_message_says_this_is_a_task_only_call(monkeypatch):
    # THE SERVICE CASE IS NOT A MISCONFIGURATION, it is the wrong call. A
    # service machine has no runs, so it is never given a result path and this
    # can never succeed there — no cap, no redeploy and no support ticket will
    # change that. Saying only "the path is missing" would leave a service
    # author hunting for the setting that turns it on.
    monkeypatch.delenv("AETHERFY_SPAWN_RESULT_PATH", raising=False)

    with pytest.raises(NotRunningOnAgent) as exc:
        write_result({"rows": 1})

    message = str(exc.value)
    assert "TASK-ONLY" in message
    assert "service" in message
    assert "never" in message
    # And it names the way out, rather than only the wall.
    assert "HTTP" in message


def test_every_other_variable_keeps_the_default_sentence(monkeypatch):
    # A negative control on the override: widening it to every call site would
    # make the message above unremarkable and would drop a true sentence from
    # the variables Aetherfy really does always inject.
    from aetherfy_agent import machine

    monkeypatch.delenv("AETHERFY_VCPUS", raising=False)
    with pytest.raises(NotRunningOnAgent) as exc:
        machine()

    assert "the platform sets AETHERFY_VCPUS before your entrypoint starts" in str(
        exc.value
    )


# --- THE CAP -----------------------------------------------------------------


def test_over_the_cap_refuses_and_leaves_no_file(result_path):
    oversized = {"blob": "x" * CAP}

    with pytest.raises(ResultTooLarge) as exc:
        write_result(oversized)

    assert exc.value.max_bytes == CAP
    assert exc.value.result_bytes == len(json.dumps(oversized).encode("utf-8"))
    assert exc.value.result_bytes > CAP
    # THE REFUSAL IS BEFORE THE WRITE. A half-written oversized file would be
    # read by the supervisor as this run's answer and reported as too_large —
    # the same outcome the refusal exists to replace.
    assert not result_path.exists()


def test_exactly_the_cap_is_written(result_path):
    # The supervisor's own comparison is `len(raw) > cap`, so the boundary
    # value is accepted. A `>=` here would refuse a result the platform stores.
    value = "x" * (CAP - len('""'))
    encoded = json.dumps(value).encode("utf-8")
    assert len(encoded) == CAP

    write_result(value)
    assert result_path.read_bytes() == encoded


def test_the_cap_is_read_from_the_environment_not_assumed(result_path, monkeypatch):
    # A hardcoded cap would pass every test above and be wrong on every machine
    # whose memory step buys a different one.
    monkeypatch.setenv("AETHERFY_RUN_INLINE_MAX_BYTES", "8")

    with pytest.raises(ResultTooLarge) as exc:
        write_result({"rows": 128})

    assert exc.value.max_bytes == 8


@pytest.mark.parametrize("cap", ["", "0", "not-a-number", "-1"])
def test_an_unusable_cap_writes_anyway(result_path, monkeypatch, cap):
    # THE PLATFORM ENFORCES THE CAP; this check is a courtesy. Refusing because
    # the courtesy is unavailable would lose a result the platform would have
    # accepted, which is strictly worse than not checking.
    monkeypatch.setenv("AETHERFY_RUN_INLINE_MAX_BYTES", cap)
    big = {"blob": "x" * (CAP * 4)}

    write_result(big)
    assert json.loads(result_path.read_text(encoding="utf-8")) == big


def test_the_cap_is_measured_in_bytes_not_characters(result_path, monkeypatch):
    # The supervisor reads the file in binary and compares byte lengths. What
    # lands on disk is what must be measured — one encode, measured and
    # written, so the number in the error is the number of bytes in the file.
    monkeypatch.setenv("AETHERFY_RUN_INLINE_MAX_BYTES", "10000")
    value = {"note": "héllo — ünïcode"}

    write_result(value)

    written = result_path.read_bytes()
    assert len(written) == len(json.dumps(value).encode("utf-8"))
    assert json.loads(written.decode("utf-8")) == value


# --- WHAT IS NOT JSON --------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_refused_and_nothing_is_written(result_path, value):
    # Python's json would write bare `NaN` / `Infinity` tokens, which no other
    # JSON reader accepts — the value would reach a dashboard or a JavaScript
    # caller as a parse error on a field nobody touched.
    with pytest.raises(ValueError):
        write_result({"score": value})

    assert not result_path.exists()


def test_an_unencodable_object_is_refused_and_nothing_is_written(result_path):
    with pytest.raises(TypeError):
        write_result({"when": object()})

    assert not result_path.exists()
