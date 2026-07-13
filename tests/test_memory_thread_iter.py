"""Unit tests for Thread.iter_history() — full-thread message iteration.

Pins the contract that iter_history walks scroll_iter (not scroll), sorts
by ts in memory, and skips messages without payload or ts.
"""

from unittest.mock import MagicMock

import pytest

from aetherfy_memory.thread import Thread


def _make_thread():
    client = MagicMock()
    return Thread("t1", "user_X_threads/t1", client), client


def test_iter_history_asc_yields_messages_oldest_first():
    th, client = _make_thread()
    # Mixed-ts points, intentionally out of order on the wire.
    client.scroll_iter.return_value = iter([
        {"id": "00000000-0000-4000-8000-000000000003", "payload": {"role": "user", "content": "c", "ts": 30.0}},
        {"id": "00000000-0000-4000-8000-000000000001", "payload": {"role": "user", "content": "a", "ts": 10.0}},
        {"id": "00000000-0000-4000-8000-000000000002", "payload": {"role": "user", "content": "b", "ts": 20.0}},
    ])

    out = list(th.iter_history())

    assert [m.id for m in out] == [
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
        "00000000-0000-4000-8000-000000000003",
    ]
    assert [m.ts for m in out] == [10.0, 20.0, 30.0]


def test_iter_history_desc_yields_newest_first():
    th, client = _make_thread()
    client.scroll_iter.return_value = iter([
        {"id": "00000000-0000-4000-8000-000000000001", "payload": {"role": "user", "content": "a", "ts": 10.0}},
        {"id": "00000000-0000-4000-8000-000000000003", "payload": {"role": "user", "content": "c", "ts": 30.0}},
        {"id": "00000000-0000-4000-8000-000000000002", "payload": {"role": "user", "content": "b", "ts": 20.0}},
    ])

    out = list(th.iter_history(order="desc"))

    assert [m.id for m in out] == [
        "00000000-0000-4000-8000-000000000003",
        "00000000-0000-4000-8000-000000000002",
        "00000000-0000-4000-8000-000000000001",
    ]


def test_iter_history_invalid_order_raises():
    th, _ = _make_thread()
    with pytest.raises(ValueError, match="order"):
        # Generator-validator fires on first iteration.
        list(th.iter_history(order="random"))


def test_iter_history_skips_points_without_payload():
    th, client = _make_thread()
    client.scroll_iter.return_value = iter([
        {"id": "00000000-0000-4000-8000-000000000001", "payload": {"role": "user", "content": "a", "ts": 1.0}},
        {"id": "00000000-0000-4000-8000-0000000000fa"},  # no payload key — skipped
        {"id": "00000000-0000-4000-8000-000000000002", "payload": {"role": "user", "content": "b", "ts": 2.0}},
    ])

    out = list(th.iter_history())
    assert [m.id for m in out] == [
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
    ]


def test_iter_history_skips_messages_without_ts():
    th, client = _make_thread()
    client.scroll_iter.return_value = iter([
        {"id": "00000000-0000-4000-8000-000000000001", "payload": {"role": "user", "content": "a", "ts": 1.0}},
        # Message.from_point with payload missing ts → m.ts is None → skipped.
        {"id": "00000000-0000-4000-8000-0000000000fb", "payload": {"role": "user", "content": "x"}},
        {"id": "00000000-0000-4000-8000-000000000002", "payload": {"role": "user", "content": "b", "ts": 2.0}},
    ])

    out = list(th.iter_history())
    assert [m.id for m in out] == [
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
    ]


def test_iter_history_walks_scroll_iter_not_single_shot_scroll():
    """Pins delegation: iter_history calls scroll_iter (paged) under the hood,
    not scroll (single-shot). If a future refactor accidentally swaps these,
    the iterator semantics would silently degrade for threads >5000 msgs."""
    th, client = _make_thread()
    client.scroll_iter.return_value = iter([])

    list(th.iter_history())

    assert client.scroll_iter.call_count == 1
    assert client.scroll.call_count == 0
