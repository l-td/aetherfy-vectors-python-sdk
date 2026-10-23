"""No request the SDK builds carries a member the API refuses.

The API checks the top-level members of a request body, the keys of every
filter in it and the keys of ``params`` (a misspelt one is a 400
INVALID_REQUEST_FIELD / INVALID_FILTER). Every request shape the client builds
is recorded here with its options all set, and each member is held to the
members the endpoint takes.
"""

import pytest

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import Filter
from aetherfy_vectors.schema import Schema

# The members each endpoint takes ("METHOD path-suffix" -> members).
ALLOWED = {
    "POST points/query": {
        "shard_key", "prefetch", "query", "using", "filter", "params",
        "score_threshold", "limit", "offset", "with_vector", "with_payload",
        "lookup_from", "with_vectors",
    },
    "POST points/scroll": {
        "shard_key", "offset", "limit", "filter", "with_payload", "with_vector",
        "order_by", "with_vectors",
    },
    "POST points/count": {"shard_key", "filter", "exact"},
    "POST points/delete": {"points", "shard_key", "filter"},
    "POST points/payload": {"payload", "points", "filter", "shard_key", "key"},
    # The upsert is a streaming write: it takes the points and nothing else
    # (a conditional upsert's update_filter / update_mode is refused).
    "PUT points": {"points"},
    "PUT points/payload": {"payload", "points", "filter", "shard_key", "key"},
    "POST collections": {"name", "vectors", "description", "regions"},
    "PUT index": {"field_name", "field_schema"},
    "PUT schema": {"schema", "enforcement_mode", "description"},
    "POST schema/analyze": {"sample_size"},
    "POST points/payload/delete": {"keys", "points", "filter", "shard_key"},
    "POST points/retrieve": {"ids", "with_payload", "with_vector"},
}
PARAMS = {"hnsw_ef", "exact", "quantization", "indexed_only", "acorn", "idf"}
FILTER_KEYS = {"must", "must_not", "should", "min_should"}

FILTER = Filter(
    must=[{"key": "a", "match": {"value": 1}}],
    must_not=[{"key": "b", "match": {"value": 2}}],
    should=[{"key": "c", "match": {"value": 3}}],
    min_should={"conditions": [{"key": "d", "match": {"value": 4}}], "min_count": 1},
)


@pytest.fixture
def recorded(monkeypatch):
    client = AetherfyVectorsClient(
        api_key="afy_test_1234567890abcdef1234",
        endpoint="https://test-api.aetherfy.com",
        workspace="home",
    )
    calls = []

    def fake(method, path, data=None, *args, **kwargs):
        calls.append((method, path, data))
        return {"result": {"points": [], "count": 0, "next_page_offset": None}}

    monkeypatch.setattr(client, "_make_request", fake)
    return client, calls


def _drive(client):
    # Every call is made with every option set; a method whose result shape the
    # fake does not satisfy still records its request first.
    attempts = [
        lambda: client.search(
            "docs", [0.1, 0.2], limit=5, offset=1, query_filter=FILTER,
            with_payload=False, with_vectors=True, score_threshold=0.5,
            search_params={"hnsw_ef": 64, "exact": False, "indexed_only": True},
        ),
        lambda: client.scroll(
            "docs", limit=5, offset=3, scroll_filter=FILTER, with_payload=False,
            with_vectors=True, order_by={"key": "ts", "direction": "desc"},
        ),
        lambda: client.count("docs", count_filter=FILTER, exact=False),
        lambda: client.delete("docs", [1, 2]),
        lambda: client.delete("docs", FILTER.to_dict()),
        lambda: client.set_payload("docs", {"a": 1}, [1], key="nested"),
        lambda: client.overwrite_payload("docs", {"a": 1}, [1]),
        lambda: client.delete_payload("docs", ["a"], [1]),
        lambda: client.retrieve("docs", [1], with_payload=False, with_vectors=True),
        lambda: client.merge_metadata("docs", 1, {"a": 1}),
        lambda: client.delete_metadata_keys("docs", 1, ["a"]),
        lambda: client.upsert("docs", [{"id": 1, "vector": [0.1, 0.2]}]),
        lambda: client.create_collection(
            "docs",
            {"size": 2, "distance": "Cosine"},
            description="d",
            regions=["us-east-1"],
        ),
        lambda: client.create_field_index("docs", "tag", "keyword"),
        lambda: client.set_schema(
            "docs", Schema(fields={}), enforcement="strict", description="d"
        ),
        lambda: client.analyze_schema("docs", sample_size=500),
    ]
    for attempt in attempts:
        try:
            attempt()
        except Exception:
            pass


def _endpoint(method, path):
    for key in sorted(ALLOWED, key=len, reverse=True):
        verb, suffix = key.split(" ", 1)
        if verb == method and path.endswith(suffix):
            return key
    return None


def refused(method, path, body):
    """The members of ``body`` the endpoint at ``path`` does not take."""
    extra = set(body) - ALLOWED[_endpoint(method, path)]
    extra |= {f"filter.{k}" for k in set(body.get("filter") or {}) - FILTER_KEYS}
    extra |= {f"params.{k}" for k in set(body.get("params") or {}) - PARAMS}
    return extra


def test_every_member_of_every_request_is_one_the_endpoint_takes(recorded):
    client, calls = recorded
    _drive(client)
    judged = [
        (m, p, d) for m, p, d in calls if _endpoint(m, p) and isinstance(d, dict)
    ]
    built = {_endpoint(m, p) for m, p, _ in judged}
    assert built >= set(ALLOWED), f"a shape was not built: {set(ALLOWED) - built}"
    for method, path, body in judged:
        assert not refused(method, path, body), (
            f"{method} {path} carries {refused(method, path, body)}"
        )


def test_a_filter_with_min_should_is_sent_whole(recorded):
    client, calls = recorded
    client.count("docs", count_filter=FILTER)
    body = calls[-1][2]
    assert body["filter"]["min_should"] == FILTER.min_should
    assert set(body["filter"]) == FILTER_KEYS


@pytest.mark.parametrize(
    "body,expected",
    [
        ({"filter": {"must": []}, "with_vectors": True}, {"with_vectors"}),
        ({"filter": {"mustNot": []}}, {"filter.mustNot"}),
        ({"exact": True, "params": {"hnsw_ef": 1, "bogus": 2}}, {"params", "params.bogus"}),
    ],
)
def test_control_the_check_names_a_member_the_endpoint_does_not_take(body, expected):
    """CONTROL: the comparison is not vacuous (count takes filter and exact)."""
    assert refused("POST", "/collections/docs/points/count", body) >= expected


@pytest.mark.parametrize(
    "method,path,body,expected",
    [
        ("PUT", "/collections/docs/points", {"points": [], "update_filter": {}}, {"update_filter"}),
        ("PUT", "/collections/docs/points/payload", {"payload": {}, "points": [], "bogus": 1}, {"bogus"}),
        ("POST", "/collections", {"name": "x", "vectors": {}, "bogus": 1}, {"bogus"}),
        ("PUT", "/collections/docs/index", {"field_name": "a", "field_type": "keyword"}, {"field_type"}),
        ("PUT", "/collections/docs/schema", {"schema": {}, "mode": "strict"}, {"mode"}),
        ("POST", "/collections/docs/schema/analyze", {"sample_size": 1, "limit": 2}, {"limit"}),
    ],
)
def test_control_the_remaining_endpoints_are_judged_too(method, path, body, expected):
    """CONTROL: upsert, payload overwrite, create, index, schema and analyze are
    held to their members as well, not only the filter-taking reads."""
    assert refused(method, path, body) == expected
