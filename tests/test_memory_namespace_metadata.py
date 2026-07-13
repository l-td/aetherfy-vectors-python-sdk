"""Unit tests for Namespace.set_metadata(), merge_metadata(),
delete_metadata_keys(), and iter().

These are thin delegating helpers; the tests just pin that they call the
right vectors-client method with the right arguments — plus the local
reserved-field guard for the merge/delete-keys helpers.
"""

import pytest
from unittest.mock import MagicMock

from aetherfy_memory.namespace import Namespace
from aetherfy_memory.thread import Thread
from aetherfy_vectors.exceptions import (
    AetherfyVectorsException,
    PointNotFoundError,
)

# Point-id fixtures — valid ids (UUID strings). Arbitrary labels like "p1"
# would be rejected by validate_point_id on a real upsert/payload call; the
# MagicMock here would otherwise hide that contract violation.
P1 = "00000000-0000-4000-8000-000000000001"
MISSING = "00000000-0000-4000-8000-0000000000ff"


def _make_ns():
    client = MagicMock()
    return Namespace("my-ns", "user_X_my-ns", client), client


def _make_thread():
    client = MagicMock()
    return Thread("conv-1", "user_X___thread__conv-1", client), client


# ---------- set_metadata ---------------------------------------------------


def test_set_metadata_calls_set_payload_with_metadata_wrapper():
    ns, client = _make_ns()
    client.set_payload.return_value = {"status": "ok"}

    out = ns.set_metadata(P1, {"foo": 1, "bar": True})

    client.set_payload.assert_called_once_with(
        "user_X_my-ns",
        payload={"metadata": {"foo": 1, "bar": True}},
        points=[P1],
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
        [{"id": "00000000-0000-4000-8000-00000000000a"}, {"id": "00000000-0000-4000-8000-00000000000b"}, {"id": "00000000-0000-4000-8000-00000000000c"}]
    )

    out = list(ns.iter())

    assert [p["id"] for p in out] == [
        "00000000-0000-4000-8000-00000000000a",
        "00000000-0000-4000-8000-00000000000b",
        "00000000-0000-4000-8000-00000000000c",
    ]
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


# ---------- merge_metadata ------------------------------------------------


def test_merge_metadata_calls_set_payload_with_metadata_key():
    ns, client = _make_ns()
    client.set_payload.return_value = {"status": "ok"}

    out = ns.merge_metadata(P1, {"reviewed": True})

    client.set_payload.assert_called_once_with(
        "user_X_my-ns",
        payload={"reviewed": True},
        points=[P1],
        key="metadata",
    )
    assert out == {"status": "ok"}


def test_merge_metadata_passes_integer_id_through():
    ns, client = _make_ns()
    ns.merge_metadata(42, {"x": 1})
    assert client.set_payload.call_args.kwargs["points"] == [42]


def test_merge_metadata_rejects_non_dict_partial():
    ns, _ = _make_ns()
    with pytest.raises(TypeError):
        ns.merge_metadata(P1, "not-a-dict")  # type: ignore[arg-type]


def test_namespace_merge_metadata_rejects_reserved_text():
    ns, client = _make_ns()
    with pytest.raises(ValueError, match="Reserved keys"):
        ns.merge_metadata(P1, {"text": "oops", "ok": 1})
    client.set_payload.assert_not_called()


def test_namespace_merge_metadata_allows_role_content_ts():
    # role/content/ts are reserved on Thread but free on plain Namespace
    ns, client = _make_ns()
    ns.merge_metadata(P1, {"role": "x", "content": "y", "ts": 1})
    client.set_payload.assert_called_once()


def test_thread_merge_metadata_rejects_role_content_ts():
    th, client = _make_thread()
    for bad in ("role", "content", "ts"):
        client.reset_mock()
        with pytest.raises(ValueError, match="Reserved keys"):
            th.merge_metadata(P1, {bad: "x"})
        client.set_payload.assert_not_called()


def test_thread_merge_metadata_allows_text():
    # text is reserved on Namespace but free on Thread (Thread payload has
    # role/content/ts, no text at the top level)
    th, client = _make_thread()
    th.merge_metadata(P1, {"text": "free-on-thread"})
    client.set_payload.assert_called_once()


def test_merge_metadata_translates_404_to_point_not_found():
    ns, client = _make_ns()
    client.set_payload.side_effect = AetherfyVectorsException(
        "Not found", status_code=404
    )
    with pytest.raises(PointNotFoundError) as excinfo:
        ns.merge_metadata(MISSING, {"k": 1})
    assert excinfo.value.point_id == MISSING
    assert excinfo.value.collection_name == "user_X_my-ns"


# ---------- delete_metadata_keys -----------------------------------------


def test_delete_metadata_keys_calls_delete_payload_with_dotted_keys():
    ns, client = _make_ns()
    client.delete_payload.return_value = {"status": "ok"}

    out = ns.delete_metadata_keys(P1, ["k1", "k2"])

    client.delete_payload.assert_called_once_with(
        "user_X_my-ns",
        keys=["metadata.k1", "metadata.k2"],
        points=[P1],
    )
    assert out == {"status": "ok"}


def test_delete_metadata_keys_rejects_non_list():
    ns, _ = _make_ns()
    with pytest.raises(TypeError):
        ns.delete_metadata_keys(P1, "k1")  # type: ignore[arg-type]


def test_delete_metadata_keys_rejects_non_string_items():
    ns, _ = _make_ns()
    with pytest.raises(TypeError):
        ns.delete_metadata_keys(P1, ["k1", 2])  # type: ignore[list-item]


def test_namespace_delete_metadata_keys_rejects_reserved_text():
    ns, client = _make_ns()
    with pytest.raises(ValueError, match="Reserved keys"):
        ns.delete_metadata_keys(P1, ["text"])
    client.delete_payload.assert_not_called()


def test_thread_delete_metadata_keys_rejects_role_content_ts():
    th, client = _make_thread()
    for bad in ("role", "content", "ts"):
        client.reset_mock()
        with pytest.raises(ValueError, match="Reserved keys"):
            th.delete_metadata_keys(P1, [bad])
        client.delete_payload.assert_not_called()


def test_delete_metadata_keys_translates_404_to_point_not_found():
    ns, client = _make_ns()
    client.delete_payload.side_effect = AetherfyVectorsException(
        "Not found", status_code=404
    )
    with pytest.raises(PointNotFoundError) as excinfo:
        ns.delete_metadata_keys(MISSING, ["k1"])
    assert excinfo.value.point_id == MISSING
    assert excinfo.value.collection_name == "user_X_my-ns"
