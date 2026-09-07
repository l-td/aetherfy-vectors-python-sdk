"""`machine()`: the shape the platform set, as ints."""

import pytest

from aetherfy_agent import machine
from aetherfy_agent.exceptions import AgentError, NotRunningOnAgent

SHAPE_VARS = ["AETHERFY_VCPUS", "AETHERFY_MEMORY_MB", "AETHERFY_REGION"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in SHAPE_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def on_a_machine(monkeypatch):
    monkeypatch.setenv("AETHERFY_VCPUS", "4")
    monkeypatch.setenv("AETHERFY_MEMORY_MB", "8192")
    monkeypatch.setenv("AETHERFY_REGION", "us-east-1")


def test_returns_ints_not_strings(on_a_machine):
    shape = machine()

    assert shape.vcpus == 4
    assert shape.memory_mb == 8192
    assert shape.region == "us-east-1"
    # The whole point of the conversion: `"4" * 8` is a string of length 8,
    # not a pool width, and it would not raise anywhere.
    assert isinstance(shape.vcpus, int)
    assert isinstance(shape.memory_mb, int)
    assert shape.vcpus * 8 == 32


@pytest.mark.parametrize("missing", SHAPE_VARS)
def test_each_missing_variable_is_named(on_a_machine, monkeypatch, missing):
    monkeypatch.delenv(missing)

    with pytest.raises(NotRunningOnAgent) as excinfo:
        machine()
    assert excinfo.value.variable == missing
    assert missing in str(excinfo.value)


def test_a_non_numeric_shape_is_an_agent_error(on_a_machine, monkeypatch):
    monkeypatch.setenv("AETHERFY_VCPUS", "four")

    with pytest.raises(AgentError):
        machine()
