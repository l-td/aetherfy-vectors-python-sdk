"""
Wire contract for the filter clause vocabulary.

Python's vocabulary IS the wire vocabulary — `Filter.to_dict()` has always
emitted `must` / `must_not` / `should` correctly, which is why the JS SDK's
`mustNot` defect had no direct Python analogue. But only half the contract
was covered: every filter-taking method also accepts a plain dict, and that
dict went to the engine VERBATIM. A caller who wrote `{"mustNot": [...]}` —
the JavaScript spelling, and the natural guess for anyone arriving from
those docs — had the entire exclusion clause dropped. No error, no warning;
the points they meant to exclude came back in the results, which is easy to
miss in a test that only asserts on the top hit.

So the two SDKs now enforce the same discipline from opposite directions:
each accepts its own idiom and rejects the other's LOUDLY, rather than
forwarding a clause the engine will ignore.

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

    def test_camel_case_clause_raises_and_names_the_right_spelling(self):
        """The regression case: this used to reach the wire and be ignored."""
        with pytest.raises(ValidationError) as exc:
            serialize_filter({"mustNot": MUST_NOT}, "search")
        assert "mustNot" in str(exc.value)
        assert "must_not" in str(exc.value)

    def test_typo_raises(self):
        with pytest.raises(ValidationError) as exc:
            serialize_filter({"mustnot": MUST_NOT}, "search")
        assert "mustnot" in str(exc.value)

    def test_error_names_the_calling_method(self):
        with pytest.raises(ValidationError) as exc:
            serialize_filter({"nope": []}, "count")
        assert str(exc.value).startswith("count: unknown filter clause")

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
    def test_camel_case_never_reaches_the_wire(self, monkeypatch, call):
        client = self._client(monkeypatch)
        sent = self._capture(client, monkeypatch, {"result": []})
        with pytest.raises(ValidationError):
            call(client)
        assert sent == {}, "the request must not be sent at all"
