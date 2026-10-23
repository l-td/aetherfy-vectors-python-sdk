"""The members of the request bodies the API does not validate itself.

The API checks every body of the filtered and point-writing endpoints (query,
scroll, count, delete, payload, upsert) key by key; those bodies are held to it
by its own tests, which read ``tests/fixtures/request-bodies.json`` (what the
client sends, kept true by ``test_request_bodies_fixture.py``) and send each
through that validation. There is no copy of that list here: one rule, one gate.

What is left is the endpoints that validation does NOT judge, where the API's
handler simply reads the members it documents: collection create, field index,
schema and analyze, and retrieve. For those this file keeps the documented
members, and holds every body in the fixture to them.
"""

import json

import pytest

from tests import request_bodies

# "METHOD path-suffix" -> the members the endpoint documents.
UNJUDGED = {
    "POST collections": {"name", "vectors", "description", "regions"},
    "PUT index": {"field_name", "field_schema"},
    "PUT schema": {"schema", "enforcement_mode", "description"},
    "POST schema/analyze": {"sample_size"},
    "POST points/retrieve": {"ids", "with_payload", "with_vector"},
}


def _endpoint(method, route):
    for key in sorted(UNJUDGED, key=len, reverse=True):
        verb, suffix = key.split(" ", 1)
        if verb == method and route.endswith(suffix):
            return key
    return None


def refused(method, route, body):
    """The members of ``body`` the endpoint does not document."""
    return set(body) - UNJUDGED[_endpoint(method, route)]


@pytest.fixture
def rows():
    return json.loads(request_bodies.FIXTURE.read_text(encoding="utf-8"))


def test_every_unjudged_body_in_the_fixture_carries_only_documented_members(rows):
    judged = [r for r in rows if _endpoint(r["method"], r["route"])]
    built = {_endpoint(r["method"], r["route"]) for r in judged}
    assert built == set(UNJUDGED), f"a shape is missing: {set(UNJUDGED) - built}"
    for r in judged:
        extra = refused(r["method"], r["route"], r["body"])
        assert not extra, f"{r['name']}: {r['method']} {r['route']} carries {extra}"


def test_the_judged_endpoints_have_no_copy_here():
    """One rule, one gate: query, scroll, count, delete, payload and upsert are
    not listed in this file."""
    assert _endpoint("POST", "/collections/docs/points/query") is None
    assert _endpoint("POST", "/collections/docs/points/scroll") is None
    assert _endpoint("POST", "/collections/docs/points/count") is None
    assert _endpoint("POST", "/collections/docs/points/delete") is None
    assert _endpoint("POST", "/collections/docs/points/payload") is None
    assert _endpoint("PUT", "/collections/docs/points") is None


@pytest.mark.parametrize(
    "method,route,body,expected",
    [
        ("POST", "/collections", {"name": "x", "vectors": {}, "bogus": 1}, {"bogus"}),
        ("PUT", "/collections/docs/index", {"field_name": "a", "field_type": "keyword"}, {"field_type"}),
        ("PUT", "/collections/docs/schema", {"schema": {}, "mode": "strict"}, {"mode"}),
        ("POST", "/collections/docs/schema/analyze", {"sample_size": 1, "limit": 2}, {"limit"}),
        ("POST", "/collections/docs/points/retrieve", {"ids": [1], "with_vectors": True}, {"with_vectors"}),
    ],
)
def test_control_a_stray_member_is_named(method, route, body, expected):
    """CONTROL: the comparison is not vacuous."""
    assert refused(method, route, body) == expected
