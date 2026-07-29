"""Wire-level contract for search(search_params=...).

The backend forwards the search body verbatim to Qdrant (raw-body passthrough
on POST /points/search), so `params` is a pure serialization contract: what the
SDK puts in the body is what the engine gets. These tests pin the body, not the
behavior — no live search is needed to prove the contract.

Two invariants matter beyond "it's in there":

  1. The default body must not change shape. Server cache keys are derived from
     the request body BYTES; a new key (even one carrying null/{}) would
     invalidate every cached search entry in the fleet on deploy.
  2. Unknown kwargs must raise. search() used to end in `**kwargs`, which
     swallowed anything a caller invented — that silent drop is why the missing
     params passthrough went unnoticed for so long.
"""

import json

import pytest


QUERY_VECTOR = [0.1, 0.2, 0.3, 0.4]

# The body a default search() has always produced. Kept as a literal (not
# built from the client) so a change in the client is a diff here, not a
# self-fulfilling assertion. Key order included: the cache key is byte-derived.
BASELINE_BODY = {
    "vector": QUERY_VECTOR,
    "limit": 10,
    "offset": 0,
    "with_payload": True,
    "with_vector": False,
}


@pytest.fixture
def searching_client(client, mock_requests, mock_successful_response):
    """Client whose next search returns one well-formed result."""
    mock_requests.request.return_value = mock_successful_response(
        {"result": [{"id": 1, "score": 0.9, "payload": {}, "vector": QUERY_VECTOR}]}
    )
    return client


def sent_body(mock_requests):
    """The JSON body handed to requests for the last call."""
    _, kwargs = mock_requests.request.call_args
    return kwargs["json"]


class TestSearchParamsSerialization:
    """search_params lands in the body as `params`, untranslated."""

    def test_search_params_sent_verbatim_as_params(
        self, searching_client, mock_requests
    ):
        searching_client.search(
            "test_collection", QUERY_VECTOR, search_params={"hnsw_ef": 256}
        )

        body = sent_body(mock_requests)
        assert body["params"] == {"hnsw_ef": 256}

    def test_params_are_not_enumerated_or_translated(
        self, searching_client, mock_requests
    ):
        """The SDK must not know the schema. An arbitrary key the SDK has
        never heard of — including one Qdrant may add tomorrow — passes
        through byte-identically, snake_case and nesting intact. If this test
        ever needs updating for a new param name, the pass-through has grown
        an allowlist and become a compatibility treadmill."""
        opaque = {
            "hnsw_ef": 512,
            "exact": False,
            "quantization": {"rescore": True, "oversampling": 2.0},
            "some_param_invented_after_this_sdk_shipped": ["a", 1, None],
        }

        searching_client.search("test_collection", QUERY_VECTOR, search_params=opaque)

        assert sent_body(mock_requests)["params"] == opaque

    def test_params_coexist_with_filter_and_threshold(
        self, searching_client, mock_requests
    ):
        query_filter = {"must": [{"key": "category", "match": {"value": "test"}}]}

        searching_client.search(
            "test_collection",
            QUERY_VECTOR,
            limit=5,
            query_filter=query_filter,
            score_threshold=0.7,
            search_params={"hnsw_ef": 128},
        )

        body = sent_body(mock_requests)
        assert body["filter"] == query_filter
        assert body["score_threshold"] == 0.7
        assert body["params"] == {"hnsw_ef": 128}
        assert body["limit"] == 5


class TestDefaultBodyUnchanged:
    """Omitting search_params must produce today's body, byte for byte."""

    def test_default_body_is_byte_identical_to_baseline(
        self, searching_client, mock_requests
    ):
        searching_client.search("test_collection", QUERY_VECTOR)

        body = sent_body(mock_requests)
        assert body == BASELINE_BODY
        # Byte-level, not just value-level: the server cache key is derived
        # from the serialized body, so key ORDER is part of the contract.
        assert json.dumps(body) == json.dumps(BASELINE_BODY)

    def test_no_params_key_when_omitted(self, searching_client, mock_requests):
        """Not `"params": null`, not `"params": {}` — absent. A null would
        change the body bytes (and reach Qdrant as an explicit null)."""
        searching_client.search("test_collection", QUERY_VECTOR)

        assert "params" not in sent_body(mock_requests)

    def test_explicit_none_is_the_same_as_omitting(
        self, searching_client, mock_requests
    ):
        """Callers threading an Optional through (params or None) get the
        default path, not a null on the wire."""
        searching_client.search("test_collection", QUERY_VECTOR, search_params=None)

        assert json.dumps(sent_body(mock_requests)) == json.dumps(BASELINE_BODY)

    def test_empty_dict_is_sent_as_given(self, searching_client, mock_requests):
        """`{}` is a value, not an absence: the SDK does not second-guess it.
        It is a different body than the default, hence a different cache
        entry — which is correct, and the reason the default path checks for
        None rather than falsiness."""
        searching_client.search("test_collection", QUERY_VECTOR, search_params={})

        body = sent_body(mock_requests)
        assert body["params"] == {}
        assert json.dumps(body) != json.dumps(BASELINE_BODY)


class TestCacheKeyDivergence:
    """Different params ⇒ different body ⇒ different server cache entry.

    Documented behavior, not a bug: the same query at ef=64 and ef=256 must
    never serve each other's results. The SDK's only obligation is to make the
    bytes differ; these assertions pin that they do.
    """

    def test_different_ef_produces_different_body_bytes(
        self, searching_client, mock_requests
    ):
        searching_client.search(
            "test_collection", QUERY_VECTOR, search_params={"hnsw_ef": 64}
        )
        low = json.dumps(sent_body(mock_requests))

        searching_client.search(
            "test_collection", QUERY_VECTOR, search_params={"hnsw_ef": 256}
        )
        high = json.dumps(sent_body(mock_requests))

        assert low != high

    def test_same_ef_produces_identical_body_bytes(
        self, searching_client, mock_requests
    ):
        """The flip side: identical params must hit the same cache entry, so
        repeating the call must not perturb the bytes (no timestamps, no
        iteration-order wobble)."""
        searching_client.search(
            "test_collection", QUERY_VECTOR, search_params={"hnsw_ef": 256}
        )
        first = json.dumps(sent_body(mock_requests))

        searching_client.search(
            "test_collection", QUERY_VECTOR, search_params={"hnsw_ef": 256}
        )
        second = json.dumps(sent_body(mock_requests))

        assert first == second


class TestUnknownKwargsRaise:
    """The silent-drop footgun that hid this gap is closed.

    search() no longer has a **kwargs sink, so Python raises TypeError itself
    — the same contract scroll_iter already has.
    """

    def test_unknown_kwarg_raises_type_error(self, searching_client):
        with pytest.raises(TypeError):
            searching_client.search(
                "test_collection", QUERY_VECTOR, definitely_not_a_real_option=1
            )

    def test_near_miss_param_name_raises_instead_of_being_dropped(
        self, searching_client
    ):
        """The exact shape of the original bug: a caller reaching for
        engine-level tuning by inventing a kwarg. Before, this searched at the
        default ef and returned plausible results with no signal at all."""
        with pytest.raises(TypeError):
            searching_client.search("test_collection", QUERY_VECTOR, hnsw_ef=256)

        with pytest.raises(TypeError):
            searching_client.search(
                "test_collection", QUERY_VECTOR, params={"hnsw_ef": 256}
            )

    def test_no_request_is_made_when_kwargs_are_rejected(
        self, searching_client, mock_requests
    ):
        """Fails before the network, so a typo can't burn a request or,
        worse, populate a cache entry under a body the caller didn't mean."""
        mock_requests.request.reset_mock()

        with pytest.raises(TypeError):
            searching_client.search("test_collection", QUERY_VECTOR, hnsw_ef=256)

        assert mock_requests.request.call_count == 0

    def test_known_kwargs_still_accepted_by_keyword(self, searching_client):
        """Guard against over-correcting into positional-only."""
        searching_client.search(
            collection_name="test_collection",
            query_vector=QUERY_VECTOR,
            limit=3,
            offset=1,
            with_payload=False,
            with_vectors=True,
            score_threshold=0.5,
            search_params={"hnsw_ef": 200},
        )
