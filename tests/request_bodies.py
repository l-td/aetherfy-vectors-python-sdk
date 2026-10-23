"""Every request body the client builds, recorded from its HTTP layer.

``record()`` drives each public method that sends a body, with its options all
set, against a client whose HTTP layer is replaced (nothing leaves the process),
and returns what would have been sent: ``[{name, method, route, body}]``, sorted.
The routes are the ones the API sees (``/collections/{name}/...``), the
workspace prefix removed.

``tests/fixtures/request-bodies.json`` is that list, committed. A test holds the
client to the file (it fails when the file is stale), and the API's own tests
read the file and send each body through its request validation, so a body the
API would refuse is found where the API is, not by a hand-kept list here.

A row whose ``name`` is in ``DELIBERATELY_INVALID`` is a body the API is MEANT to
refuse: the tests that check the typed errors send it on purpose.

Regenerate with ``python scripts/record_request_bodies.py``.
"""

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import Filter
from aetherfy_vectors.schema import Schema

FIXTURE = Path(__file__).parent / "fixtures" / "request-bodies.json"

# Bodies the API is meant to refuse, by row name: a misspelt filter key and an
# unknown search parameter, sent so the typed errors can be tested.
DELIBERATELY_INVALID = {
    "search/misspelt-filter-key": "INVALID_FILTER",
    "search/unknown-search-param": "INVALID_REQUEST_FIELD",
}

CONDITION = [{"key": "category", "match": {"value": "books"}}]
FILTER = Filter(
    must=CONDITION,
    must_not=[{"key": "in_stock", "match": {"value": False}}],
    should=[{"key": "tag", "match": {"value": "sale"}}],
    min_should={"conditions": CONDITION, "min_count": 1},
)
# The shape the memory layer sends for a thread: a scope clause, a point-id
# selector and an exclusion of the marker point.
THREAD_FILTER = {
    "must": [
        {"key": "thread_id", "match": {"value": "t1"}},
        {"has_id": [1, 2]},
    ],
    "must_not": [{"key": "thread_marker", "match": {"value": True}}],
}
VECTOR = [0.1, 0.2]


def _calls(client: AetherfyVectorsClient) -> List[Tuple[str, Callable[[], Any]]]:
    return [
        (
            "search/all-options",
            lambda: client.search(
                "docs",
                VECTOR,
                limit=5,
                offset=1,
                query_filter=FILTER,
                with_payload=False,
                with_vectors=True,
                score_threshold=0.5,
                search_params={"hnsw_ef": 64, "exact": False, "indexed_only": True},
            ),
        ),
        ("search/defaults", lambda: client.search("docs", VECTOR)),
        (
            "search/thread-filter",
            lambda: client.search("docs", VECTOR, query_filter=THREAD_FILTER),
        ),
        (
            "search/misspelt-filter-key",
            lambda: client.search("docs", VECTOR, query_filter={"mustNot": []}),
        ),
        (
            "search/unknown-search-param",
            lambda: client.search("docs", VECTOR, search_params={"bogus": 1}),
        ),
        (
            "scroll/all-options",
            lambda: client.scroll(
                "docs",
                limit=5,
                offset=3,
                scroll_filter=FILTER,
                with_payload=False,
                with_vectors=True,
                order_by={"key": "ts", "direction": "desc"},
            ),
        ),
        ("scroll/thread-filter", lambda: client.scroll("docs", scroll_filter=THREAD_FILTER)),
        ("count/all-options", lambda: client.count("docs", count_filter=FILTER, exact=False)),
        ("count/thread-filter", lambda: client.count("docs", count_filter=THREAD_FILTER)),
        ("delete/ids", lambda: client.delete("docs", [1, 2])),
        ("delete/filter", lambda: client.delete("docs", FILTER.to_dict())),
        ("delete/thread-filter", lambda: client.delete("docs", THREAD_FILTER)),
        ("set-payload/key", lambda: client.set_payload("docs", {"a": 1}, [1], key="nested")),
        ("set-payload", lambda: client.set_payload("docs", {"a": 1}, [1])),
        ("overwrite-payload", lambda: client.overwrite_payload("docs", {"a": 1}, [1])),
        ("delete-payload", lambda: client.delete_payload("docs", ["a"], [1])),
        ("merge-metadata", lambda: client.merge_metadata("docs", 1, {"a": 1})),
        ("delete-metadata-keys", lambda: client.delete_metadata_keys("docs", 1, ["a"])),
        (
            "retrieve",
            lambda: client.retrieve("docs", [1], with_payload=False, with_vectors=True),
        ),
        ("upsert", lambda: client.upsert("docs", [{"id": 1, "vector": VECTOR}])),
        (
            "upsert/payload",
            lambda: client.upsert(
                "docs", [{"id": 2, "vector": VECTOR, "payload": {"a": "b"}}]
            ),
        ),
        (
            "create-collection",
            lambda: client.create_collection(
                "docs",
                {"size": 2, "distance": "Cosine"},
                description="d",
                regions=["us-east-1"],
            ),
        ),
        (
            "create-collection/defaults",
            lambda: client.create_collection("docs", {"size": 2, "distance": "Cosine"}),
        ),
        ("create-field-index", lambda: client.create_field_index("docs", "tag", "keyword")),
        (
            "set-schema",
            lambda: client.set_schema(
                "docs", Schema(fields={}), enforcement="strict", description="d"
            ),
        ),
        ("analyze-schema", lambda: client.analyze_schema("docs", sample_size=500)),
    ]


def _route(path: str) -> str:
    """The path as the API sees it: the workspace prefix removed."""
    path = "/" + path.lstrip("/")
    path = re.sub(r"^/api/v1", "", path)
    return re.sub(r"^/workspace(s|-ids)/[^/]+", "", path)


def record() -> List[Dict[str, Any]]:
    """Drive every call and return the bodies sent, sorted by name."""
    client = AetherfyVectorsClient(
        api_key="afy_test_1234567890abcdef1234",
        endpoint="https://test-api.aetherfy.com",
        workspace="home",
    )
    sent: List[Tuple[str, str, Any]] = []

    def fake(method, path, data=None, *args, **kwargs):
        sent.append((method, path, data))
        return {"result": {"points": [], "count": 0, "next_page_offset": None}}

    client._make_request = fake  # type: ignore[method-assign]
    rows: List[Dict[str, Any]] = []
    for name, call in _calls(client):
        before = len(sent)
        try:
            call()
        except Exception:
            pass  # a method whose answer the stub does not satisfy still sent its request
        for method, path, data in sent[before:]:
            if isinstance(data, dict):
                rows.append(
                    {"name": name, "method": method, "route": _route(path), "body": data}
                )
    rows.sort(key=lambda r: (r["name"], r["method"], r["route"]))
    return rows


def dump(rows: List[Dict[str, Any]]) -> str:
    return json.dumps(rows, indent=2, sort_keys=True) + "\n"
