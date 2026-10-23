"""tests/fixtures/request-bodies.json is what the client sends, today.

The API's own tests read that file and send every body through its request
validation, so the file is only worth anything while it is true. This test
regenerates it from the client's HTTP layer and fails when it differs: a request
the client builds differently needs the file regenerated and committed
(``python scripts/record_request_bodies.py``), and that is the moment the API's
validation sees the new body.
"""

import json

import pytest

from tests import request_bodies


def describe_difference(recorded, committed):
    """What differs, by row name, in words; empty when they match."""
    by_name = lambda rows: {(r["name"], r["method"], r["route"]): r["body"] for r in rows}
    now, then = by_name(recorded), by_name(committed)
    lines = []
    for key in sorted(now.keys() - then.keys()):
        lines.append(f"new request not in the file: {key[0]} ({key[1]} {key[2]})")
    for key in sorted(then.keys() - now.keys()):
        lines.append(f"the file has a request the client no longer sends: {key[0]}")
    for key in sorted(now.keys() & then.keys()):
        if now[key] != then[key]:
            lines.append(f"the body of {key[0]} changed: {then[key]} -> {now[key]}")
    return lines


def test_the_committed_file_is_what_the_client_sends():
    committed = json.loads(request_bodies.FIXTURE.read_text(encoding="utf-8"))
    recorded = request_bodies.record()
    difference = describe_difference(recorded, committed)
    assert not difference, (
        "tests/fixtures/request-bodies.json is stale; run "
        "`python scripts/record_request_bodies.py` and commit it:\n  "
        + "\n  ".join(difference)
    )
    assert request_bodies.dump(recorded) == request_bodies.FIXTURE.read_text(
        encoding="utf-8"
    ), "the file differs from the regenerated one only in formatting: regenerate it"


def test_the_file_covers_every_endpoint_the_client_writes():
    committed = json.loads(request_bodies.FIXTURE.read_text(encoding="utf-8"))
    seen = {(r["method"], r["route"].replace("/collections/docs", "")) for r in committed}
    assert seen >= {
        ("POST", "/points/query"),
        ("POST", "/points/scroll"),
        ("POST", "/points/count"),
        ("POST", "/points/delete"),
        ("POST", "/points/payload"),
        ("POST", "/points/payload/delete"),
        ("POST", "/points/retrieve"),
        ("PUT", "/points"),
        ("PUT", "/points/payload"),
        ("POST", "/collections"),
        ("PUT", "/index"),
        ("PUT", "/schema"),
        ("POST", "/schema/analyze"),
    }
    assert any(
        r["body"].get("filter", {}).get("min_should") for r in committed
    ), "a min_should filter is part of what the client sends"


def test_every_deliberately_invalid_row_is_in_the_file():
    committed = json.loads(request_bodies.FIXTURE.read_text(encoding="utf-8"))
    names = {r["name"] for r in committed}
    assert set(request_bodies.DELIBERATELY_INVALID) <= names


# -- CONTROLS: the comparison is not vacuous ------------------------------------


@pytest.fixture
def committed():
    return json.loads(request_bodies.FIXTURE.read_text(encoding="utf-8"))


def test_control_a_changed_body_is_reported(committed):
    mutated = json.loads(json.dumps(committed))
    mutated[0]["body"]["added_member"] = 1
    difference = describe_difference(committed, mutated)
    assert len(difference) == 1 and "changed" in difference[0]


def test_control_a_new_request_is_reported(committed):
    assert describe_difference(committed, committed[1:])[0].startswith("new request")


def test_control_a_vanished_request_is_reported(committed):
    assert "no longer sends" in describe_difference(committed[1:], committed)[0]
