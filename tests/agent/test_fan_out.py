"""
`fan_out()`: input order, no swallowed failure, and the one log line.

THE LOG LINE IS ASSERTED BYTE FOR BYTE, here and in the JavaScript SDK's
matching test. It is the convention the platform cannot produce for the
customer — nothing outside the machine can count in-process workers — so its
text is a contract between the two helpers, not a debug print. A drift between
the two languages would make a run's width unreadable in exactly half the fleet
and nothing else would notice.
"""

import time

import pytest

from aetherfy_agent import fan_out


@pytest.fixture(autouse=True)
def on_a_machine(monkeypatch):
    monkeypatch.setenv("AETHERFY_VCPUS", "4")
    monkeypatch.setenv("AETHERFY_MEMORY_MB", "8192")
    monkeypatch.setenv("AETHERFY_REGION", "us-east-1")


def test_results_come_back_in_input_order(capsys):
    """Every worker sleeps for the inverse of its value, so completion order is
    the exact reverse of input order — an implementation that returned results
    as they finished would return them backwards."""

    def slow(n):
        time.sleep((5 - n) * 0.02)
        return n * 10

    assert fan_out(slow, [1, 2, 3, 4]) == [10, 20, 30, 40]


def test_the_log_line_is_exact(capsys):
    fan_out(lambda n: n, [1, 2, 3])

    assert (
        capsys.readouterr().out
        == "aetherfy: fanning out 32 wide on 4 vCPU / 8192 MB (3 tasks)\n"
    )


def test_the_default_width_is_eight_per_vcpu(capsys, monkeypatch):
    monkeypatch.setenv("AETHERFY_VCPUS", "1")
    monkeypatch.setenv("AETHERFY_MEMORY_MB", "1024")

    fan_out(lambda n: n, [1])

    assert (
        capsys.readouterr().out
        == "aetherfy: fanning out 8 wide on 1 vCPU / 1024 MB (1 tasks)\n"
    )


def test_an_explicit_width_is_the_width_reported(capsys):
    fan_out(lambda n: n, [1, 2], width=2)

    assert (
        capsys.readouterr().out
        == "aetherfy: fanning out 2 wide on 4 vCPU / 8192 MB (2 tasks)\n"
    )


def test_the_line_prints_once_and_only_once(capsys):
    fan_out(lambda n: n, list(range(20)))

    assert capsys.readouterr().out.count("aetherfy: fanning out") == 1


def test_an_empty_item_list_still_reports(capsys):
    assert fan_out(lambda n: n, []) == []
    assert (
        capsys.readouterr().out
        == "aetherfy: fanning out 32 wide on 4 vCPU / 8192 MB (0 tasks)\n"
    )


def test_the_first_failure_is_re_raised_not_swallowed(capsys):
    def sometimes(n):
        if n in (1, 3):
            raise ValueError("item {0} failed".format(n))
        return n

    with pytest.raises(ValueError) as excinfo:
        fan_out(sometimes, [0, 1, 2, 3])
    # The LOWEST-INDEXED failure, deterministically — not whichever thread lost
    # the race. Item 3 also failed and must not be the one reported.
    assert "item 1 failed" in str(excinfo.value)


def test_a_failure_does_not_hide_the_log_line(capsys):
    def always_fails(n):
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        fan_out(always_fails, [1])
    assert "aetherfy: fanning out" in capsys.readouterr().out


def test_an_iterable_is_materialized_once(capsys):
    """A generator must not be consumed by the count in the log line and then
    found empty by the pool."""
    assert fan_out(lambda n: n * 2, (n for n in [1, 2, 3])) == [2, 4, 6]
    assert "(3 tasks)" in capsys.readouterr().out


def test_a_bad_kind_is_refused(capsys):
    with pytest.raises(ValueError):
        fan_out(lambda n: n, [1], kind="greenlets")


def test_a_width_below_one_is_refused(capsys):
    with pytest.raises(ValueError):
        fan_out(lambda n: n, [1], width=0)


def _double(n):
    """Module level so a process pool can pickle it."""
    return n * 2


def test_processes_run_the_same_contract(capsys):
    assert fan_out(_double, [1, 2, 3], width=2, kind="processes") == [2, 4, 6]
    assert (
        capsys.readouterr().out
        == "aetherfy: fanning out 2 wide on 4 vCPU / 8192 MB (3 tasks)\n"
    )


def test_processes_default_to_one_worker_per_core(capsys):
    """The default follows the POOL. Choosing processes means the work is
    CPU-bound by definition, and CPU-bound work wider than the core count only
    adds context switching — so the thread default of vcpus * 8 must NOT apply
    here."""
    assert fan_out(_double, [1, 2, 3], kind="processes") == [2, 4, 6]

    assert (
        capsys.readouterr().out
        == "aetherfy: fanning out 4 wide on 4 vCPU / 8192 MB (3 tasks)\n"
    )


def test_threads_still_default_to_eight_per_core(capsys):
    """The other half of the same rule, asserted beside it: narrowing the
    process default must not narrow the thread default with it."""
    fan_out(lambda n: n, [1], kind="threads")

    assert (
        capsys.readouterr().out
        == "aetherfy: fanning out 32 wide on 4 vCPU / 8192 MB (1 tasks)\n"
    )
