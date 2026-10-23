"""
Wire contract for the filter clause vocabulary.

Python's vocabulary IS the wire vocabulary — `Filter.to_dict()` emits
`must` / `must_not` / `should` / `min_should` as they go on the wire. A
filter-taking method also accepts a plain dict, and a clause name the engine
does not know used to be dropped without a word (`{"mustNot": [...]}`, the
JavaScript spelling, excluded nothing).

The API now judges every key of a filter and answers an unknown one with a
400 INVALID_FILTER naming its path and the nearest valid key, so the SDK keeps
no clause list of its own (one that can only lag the API: it refused
`min_should` for as long as it existed) and forwards a dict as written. The
mapping of that answer is tested in test_invalid_filter.py.

Four call sites take a filter — search, scroll, delete-by-filter, count —
and all four are covered here. `count` also gained `Filter` support; it
previously accepted a dict only, so a `Filter` reached the wire as an
un-serialized dataclass.
"""

import pytest

from aetherfy_vectors.exceptions import ValidationError
from aetherfy_vectors.models import Filter
from aetherfy_vectors.utils import serialize_filter

MUST = [{"key": "category", "match": {"value": "books"}}]
MUST_NOT = [{"key": "in_stock", "match": {"value": False}}]
SHOULD = [{"key": "tag", "match": {"value": "sale"}}]


class TestSerializeFilter:
    def test_filter_dataclass_round_trips(self):
        out = serialize_filter(
            Filter(must=MUST, must_not=MUST_NOT, should=SHOULD), "search"
        )
        assert out == {"must": MUST, "must_not": MUST_NOT, "should": SHOULD}

    def test_dict_round_trips(self):
        out = serialize_filter(
            {"must": MUST, "must_not": MUST_NOT, "should": SHOULD}, "search"
        )
        assert out == {"must": MUST, "must_not": MUST_NOT, "should": SHOULD}

    def test_key_order_is_fixed_regardless_of_insertion_order(self):
        # Byte-level, not dict-equality: the server cache key is derived from
        # the request body bytes, so two callers writing the same clauses in
        # different orders must produce identical bytes.
        import json

        a = json.dumps(
            serialize_filter(
                {"should": SHOULD, "must_not": MUST_NOT, "must": MUST}, "search"
            )
        )
        b = json.dumps(
            serialize_filter(
                {"must": MUST, "should": SHOULD, "must_not": MUST_NOT}, "search"
            )
        )
        assert a == b
        assert a == json.dumps({"must": MUST, "must_not": MUST_NOT, "should": SHOULD})

    def test_none_passes_through(self):
        assert serialize_filter(None, "search") is None

    def test_unset_clauses_are_omitted(self):
        assert serialize_filter({"must_not": MUST_NOT}, "search") == {
            "must_not": MUST_NOT
        }
        assert serialize_filter({"must": MUST, "should": None}, "search") == {
            "must": MUST
        }

    def test_min_should_is_a_clause_of_both_forms(self):
        min_should = {"conditions": SHOULD, "min_count": 1}
        assert serialize_filter(Filter(min_should=min_should), "search") == {
            "min_should": min_should
        }
        assert serialize_filter({"must": MUST, "min_should": min_should}, "search") == {
            "must": MUST,
            "min_should": min_should,
        }

    def test_a_clause_the_sdk_does_not_know_is_forwarded_as_written(self):
        """The API judges clause names (400 INVALID_FILTER with the path and
        the nearest valid key), so the SDK neither keeps a list that can lag
        it nor drops what it does not recognise. Known clauses keep their
        fixed order; the rest follow in the caller's order."""
        out = serialize_filter({"mustNot": MUST_NOT, "must": MUST, "nope": []}, "count")
        assert list(out) == ["must", "mustNot", "nope"]
        assert out["mustNot"] == MUST_NOT

    def test_non_dict_non_filter_raises(self):
        with pytest.raises(ValidationError) as exc:
            serialize_filter([MUST], "search")
        assert "must be a Filter or a dict" in str(exc.value)


class TestCallSites:
    """Every filter-taking method routes through the same normalizer."""

    @staticmethod
    def _client(monkeypatch):
        from aetherfy_vectors import AetherfyVectorsClient

        monkeypatch.delenv("AETHERFY_VECTORS_URL", raising=False)
        return AetherfyVectorsClient(api_key="afy_test_1234567890abcdef1234")

    @staticmethod
    def _capture(client, monkeypatch, result):
        """Replace _make_request and record the body it was handed."""
        sent = {}

        def fake(method, path, data=None, **kwargs):
            sent["method"] = method
            sent["path"] = path
            sent["data"] = data
            return result

        monkeypatch.setattr(client, "_make_request", fake)
        return sent

    def test_search_sends_must_not(self, monkeypatch):
        client = self._client(monkeypatch)
        sent = self._capture(client, monkeypatch, {"result": []})
        client.search("c1", [0.1, 0.2], query_filter={"must_not": MUST_NOT})
        assert sent["data"]["filter"] == {"must_not": MUST_NOT}

    def test_scroll_sends_must_not(self, monkeypatch):
        client = self._client(monkeypatch)
        sent = self._capture(
            client, monkeypatch, {"result": {"points": [], "next_page_offset": None}}
        )
        client.scroll("c1", scroll_filter=Filter(must_not=MUST_NOT))
        assert sent["data"]["filter"] == {"must_not": MUST_NOT}

    def test_count_accepts_a_filter_object(self, monkeypatch):
        client = self._client(monkeypatch)
        sent = self._capture(client, monkeypatch, {"result": {"count": 0}})
        client.count("c1", count_filter=Filter(must_not=MUST_NOT))
        assert sent["data"]["filter"] == {"must_not": MUST_NOT}

    def test_delete_by_filter_sends_must_not(self, monkeypatch):
        client = self._client(monkeypatch)
        sent = self._capture(client, monkeypatch, {"result": True})
        client.delete("c1", {"must_not": MUST_NOT})
        assert sent["data"]["filter"] == {"must_not": MUST_NOT}

    @pytest.mark.parametrize(
        "call",
        [
            lambda c: c.search("c1", [0.1], query_filter={"mustNot": MUST_NOT}),
            lambda c: c.scroll("c1", scroll_filter={"mustNot": MUST_NOT}),
            lambda c: c.count("c1", count_filter={"mustNot": MUST_NOT}),
            lambda c: c.delete("c1", {"mustNot": MUST_NOT}),
        ],
        ids=["search", "scroll", "count", "delete"],
    )
    def test_an_unknown_clause_reaches_the_api_to_be_judged(self, monkeypatch, call):
        client = self._client(monkeypatch)
        sent = self._capture(client, monkeypatch, {"result": []})
        try:
            call(client)
        except Exception:
            pass  # the fake answers a shape some methods do not parse
        assert sent["data"]["filter"] == {"mustNot": MUST_NOT}
