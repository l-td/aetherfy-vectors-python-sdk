"""Unit tests for Namespace.set_metadata() and Namespace.iter().

These are thin delegating helpers; the tests just pin that they call the
right vectors-client method with the right arguments.
"""

from unittest.mock import MagicMock

from aetherfy_memory.namespace import Namespace


def _make_ns():
    client = MagicMock()
    return Namespace("my-ns", "user_X_my-ns", client), client


# ---------- set_metadata ---------------------------------------------------


def test_set_metadata_calls_set_payload_with_metadata_wrapper():
    ns, client = _make_ns()
    client.set_payload.return_value = {"status": "ok"}

    out = ns.set_metadata("p1", {"foo": 1, "bar": True})

    client.set_payload.assert_called_once_with(
        "user_X_my-ns",
        payload={"metadata": {"foo": 1, "bar": True}},
        points=["p1"],
    )
    assert out == {"status": "ok"}


def test_set_metadata_passes_integer_id_through():
    ns, client = _make_ns()
    ns.set_metadata(42, {"x": 1})
    assert client.set_payload.call_args.kwargs["points"] == [42]


# ---------- iter -----------------------------------------------------------


def test_iter_delegates_to_scroll_iter_and_yields_each_point():
    ns, client = _make_ns()
    client.scroll_iter.return_value = iter(
        [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    )

    out = list(ns.iter())

    assert [p["id"] for p in out] == ["a", "b", "c"]
    client.scroll_iter.assert_called_once()
    kwargs = client.scroll_iter.call_args.kwargs
    assert kwargs["batch_size"] == 256
    assert kwargs["scroll_filter"] is None
    assert kwargs["with_payload"] is True
    assert kwargs["with_vectors"] is False


def test_iter_forwards_batch_size_filter_payload_flags():
    ns, client = _make_ns()
    client.scroll_iter.return_value = iter([])
    flt = {"must": [{"key": "x", "match": {"value": "y"}}]}

    list(
        ns.iter(
            batch_size=100,
            filter=flt,
            with_payload=False,
            with_vectors=True,
        )
    )

    kwargs = client.scroll_iter.call_args.kwargs
    assert kwargs["batch_size"] == 100
    assert kwargs["scroll_filter"] is flt
    assert kwargs["with_payload"] is False
    assert kwargs["with_vectors"] is True
