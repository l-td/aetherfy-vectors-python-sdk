"""Unit tests for client.scroll_iter() — auto-paginating scroll generator.

Pins the contract that:
  - scroll_iter delegates to scroll() (no duplicate HTTP path).
  - The kwarg allowlist is enforced; `limit` and `offset` are owned by the
    iterator and not caller-controllable.
  - batch_size is range-validated (1..1000, the server cap from vectordb WS1).
  - The generator stops cleanly when next_page_offset becomes None.
"""

from unittest.mock import MagicMock

import pytest

from aetherfy_vectors.client import AetherfyVectorsClient


def _make_client():
    """Construct a client without firing real HTTP at __init__."""
    client = AetherfyVectorsClient.__new__(AetherfyVectorsClient)
    return client


def test_scroll_iter_yields_all_pages_in_order():
    """Three-page scroll: cursor1 → cursor2 → None."""
    client = _make_client()
    page1 = {"points": [{"id": "a"}, {"id": "b"}], "next_page_offset": "c1"}
    page2 = {"points": [{"id": "c"}, {"id": "d"}], "next_page_offset": "c2"}
    page3 = {"points": [{"id": "e"}], "next_page_offset": None}
    scroll_mock = MagicMock(side_effect=[page1, page2, page3])
    client.scroll = scroll_mock

    out = list(client.scroll_iter("col", batch_size=2))

    assert [p["id"] for p in out] == ["a", "b", "c", "d", "e"]
    assert scroll_mock.call_count == 3
    # Each call uses the previous page's next_page_offset as `offset`.
    first, second, third = scroll_mock.call_args_list
    assert first.kwargs["offset"] is None
    assert second.kwargs["offset"] == "c1"
    assert third.kwargs["offset"] == "c2"


def test_scroll_iter_empty_collection_yields_nothing():
    client = _make_client()
    client.scroll = MagicMock(
        return_value={"points": [], "next_page_offset": None}
    )

    out = list(client.scroll_iter("col"))

    assert out == []
    assert client.scroll.call_count == 1


def test_scroll_iter_forwards_filter_and_payload_flags():
    client = _make_client()
    client.scroll = MagicMock(
        return_value={"points": [], "next_page_offset": None}
    )
    flt = {"must": [{"key": "category", "match": {"value": "docs"}}]}

    list(
        client.scroll_iter(
            "col",
            batch_size=128,
            scroll_filter=flt,
            with_payload=False,
            with_vectors=True,
        )
    )

    kwargs = client.scroll.call_args.kwargs
    assert kwargs["limit"] == 128
    assert kwargs["scroll_filter"] is flt
    assert kwargs["with_payload"] is False
    assert kwargs["with_vectors"] is True


def test_scroll_iter_rejects_batch_size_zero():
    client = _make_client()
    client.scroll = MagicMock()
    with pytest.raises(ValueError, match="batch_size"):
        # Generator validation fires on first iteration.
        list(client.scroll_iter("col", batch_size=0))
    assert client.scroll.call_count == 0


def test_scroll_iter_rejects_batch_size_above_server_cap():
    client = _make_client()
    client.scroll = MagicMock()
    with pytest.raises(ValueError, match="batch_size"):
        list(client.scroll_iter("col", batch_size=1001))


def test_scroll_iter_rejects_caller_supplied_limit_kwarg():
    """limit is owned by the iterator — TypeError if caller tries to pass it."""
    client = _make_client()
    client.scroll = MagicMock()
    with pytest.raises(TypeError):
        # Passing as kwargs not in the explicit allowlist raises TypeError
        # automatically because scroll_iter doesn't accept **kwargs.
        list(client.scroll_iter("col", limit=100))  # type: ignore[call-arg]


def test_scroll_iter_rejects_caller_supplied_offset_kwarg():
    """offset is owned by the iterator — starting mid-stream is a footgun."""
    client = _make_client()
    client.scroll = MagicMock()
    with pytest.raises(TypeError):
        list(client.scroll_iter("col", offset="x"))  # type: ignore[call-arg]
