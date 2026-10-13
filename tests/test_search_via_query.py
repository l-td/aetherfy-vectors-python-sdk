"""search() goes through POST /collections/{name}/points/query.

The API refuses Qdrant's retired /points/search (410 ROUTE_RETIRED), so
search() sends the query route. Its public call and arguments are unchanged;
this pins the translation of EVERY argument into the query body, and the
reading of the query response (result.points, not result).
"""

from aetherfy_vectors.models import SearchResult

QUERY_VECTOR = [0.1, 0.2, 0.3, 0.4]


def _sent(mock_requests):
    _, kwargs = mock_requests.request.call_args
    return kwargs


def test_every_argument_is_translated_into_the_query_body(
    client, mock_requests, mock_successful_response
):
    mock_requests.request.return_value = mock_successful_response({"result": {"points": []}})

    client.search(
        "test_collection",
        QUERY_VECTOR,
        limit=7,
        offset=3,
        query_filter={
            "must": [{"key": "city", "match": {"value": "Rome"}}],
            "must_not": [{"key": "n", "range": {"gt": 5}}],
        },
        with_payload=False,
        with_vectors=True,
        score_threshold=0.42,
        search_params={"hnsw_ef": 256, "exact": False},
    )

    sent = _sent(mock_requests)
    assert sent["method"] == "POST"
    assert sent["url"].endswith("/collections/test_collection/points/query")
    assert sent["json"] == {
        "query": QUERY_VECTOR,
        "limit": 7,
        "offset": 3,
        "with_payload": False,
        "with_vector": True,
        "filter": {
            "must": [{"key": "city", "match": {"value": "Rome"}}],
            "must_not": [{"key": "n", "range": {"gt": 5}}],
        },
        "score_threshold": 0.42,
        "params": {"hnsw_ef": 256, "exact": False},
    }


def test_matches_are_read_from_result_points(client, mock_requests, mock_successful_response):
    points = [
        {"id": 1, "version": 0, "score": 0.99, "payload": {"t": "a"}},
        {"id": "b3f7", "version": 2, "score": 0.5, "payload": {"t": "b"}, "vector": [1.0, 0.0, 0.0, 0.0]},
    ]
    mock_requests.request.return_value = mock_successful_response(
        {"result": {"points": points}, "status": "ok", "time": 0.001}
    )

    results = client.search("test_collection", QUERY_VECTOR)

    assert [type(r) for r in results] == [SearchResult, SearchResult]
    assert [(r.id, r.score, r.payload) for r in results] == [
        (1, 0.99, {"t": "a"}),
        ("b3f7", 0.5, {"t": "b"}),
    ]
    assert results[1].vector == [1.0, 0.0, 0.0, 0.0]


def test_the_retired_route_is_never_called(client, mock_requests, mock_successful_response):
    mock_requests.request.return_value = mock_successful_response({"result": {"points": []}})
    client.search("test_collection", QUERY_VECTOR)
    for _, kwargs in mock_requests.request.call_args_list:
        assert "/points/search" not in kwargs["url"]
