"""Threads are payload rows in one collection, not a collection each.

These tests run the memory layer against `tests/fake_vectors_store`, a double
that actually stores points and actually evaluates filters. That matters here:
every claim in this file — a thread is isolated from its siblings, an empty
thread exists, a caller's filter cannot widen the scope, `clear()` does not
take the neighbours with it — is a claim about what the FILTER does, and a
recorded-call assertion would only prove which filter was sent.

The headline case is `test_more_threads_than_the_plan_allows_collections`: the
defect this model change exists to remove.
"""

import pytest

from aetherfy_memory import MemoryClient
from aetherfy_memory.exceptions import (
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
    ThreadVectorSizeMismatchError,
)
from aetherfy_memory.models import (
    DEFAULT_VECTOR_SIZE,
    THREAD_ID_KEY,
    THREAD_MARKER_KEY,
    THREADS_COLLECTION,
)
from aetherfy_vectors.exceptions import PointNotFoundError

from .fake_vectors_store import FakeVectorsClient, unit_vector

DIM = 4


@pytest.fixture
def store():
    return FakeVectorsClient()


@pytest.fixture
def memory(store):
    return MemoryClient(client=store, thread_vector_size=DIM)


def _v(axis=0):
    return unit_vector(DIM, axis)


def _msgs(thread, n, prefix="m", axis=0):
    return [
        thread.add(role="user", content=f"{prefix}{i}", vector=_v(axis), ts=float(i))
        for i in range(n)
    ]


# ===========================================================================
# The cross-repo pin
# ===========================================================================


def test_the_threads_collection_name_is_the_pinned_literal():
    """The e2e suite hard-codes "__threads__" on purpose: a cross-repo
    literal should be a literal there, so a rename is caught rather than
    followed. This is the other half of that pin. Without it a rename goes
    green here and reds in a repo that cannot explain why — so the gate
    lives where the rename would happen.
    """
    assert THREADS_COLLECTION == "__threads__"
    assert THREAD_ID_KEY == "thread_id"
    assert THREAD_MARKER_KEY == "thread_marker"


# ===========================================================================
# The defect this change removes
# ===========================================================================


def test_more_threads_than_the_plan_allows_collections(memory, store):
    """A Free plan allows three collections. It must not cap conversations.

    Under the old model every thread was its own collection, so the fourth
    `create_thread` on a Free account returned COLLECTION_LIMIT_EXCEEDED (and
    fired the "you hit your plan limit" email). Ten threads here, and the
    collection count does not move.
    """
    for i in range(10):
        memory.create_thread(f"conv-{i}")

    assert len(store.get_collections()) == 1
    assert [c.name for c in store.get_collections()] == [THREADS_COLLECTION]
    assert sorted(memory.list_threads()) == sorted(f"conv-{i}" for i in range(10))
    # And none of them is a namespace.
    assert memory.list_namespaces() == []


# ===========================================================================
# An empty thread exists
# ===========================================================================


def test_an_empty_thread_exists(memory):
    memory.create_thread("empty")
    assert memory.thread_exists("empty") is True
    assert memory.list_threads() == ["empty"]
    # ...and holds no messages.
    assert memory.thread("empty").count() == 0
    assert memory.thread("empty").history() == []


def test_creating_an_empty_thread_twice_still_raises(memory):
    memory.create_thread("empty")
    with pytest.raises(ThreadAlreadyExistsError):
        memory.create_thread("empty")


def test_a_thread_that_was_never_created_does_not_exist(memory):
    assert memory.thread_exists("never") is False
    with pytest.raises(ThreadNotFoundError):
        memory.thread("never")
    assert memory.delete_thread("never") is False


def test_a_second_marker_does_not_list_the_thread_twice(memory, store):
    """Creating a thread is a check-then-write, so it can lose a race.

    Two callers that both pass the exists-check before either marker lands
    both write one. Every other read tolerates that — `thread_exists` counts,
    `count` and `history` exclude markers, `delete_thread` removes every row
    with the id — but `list_threads` reads the id off each marker, so without
    de-duplication it reported the thread twice. Replays the losing caller's
    write directly, since the race itself is not reproducible in-process.
    """
    import uuid

    memory.create_thread("a")
    store.upsert(
        THREADS_COLLECTION,
        [
            {
                "id": str(uuid.uuid4()),
                "vector": _v(),
                "payload": {THREAD_ID_KEY: "a", THREAD_MARKER_KEY: True},
            }
        ],
    )

    assert memory.list_threads() == ["a"]
    # ...and nothing else was disturbed.
    assert memory.thread_exists("a") is True
    assert memory.thread("a").count() == 0
    assert memory.delete_thread("a") is True
    assert memory.list_threads() == []


def test_list_threads_keeps_first_seen_order(memory):
    """Order is first-seen, not hash order: a set would have made this
    assertion depend on Python's string hashing."""
    for name in ("zeta", "alpha", "mid"):
        memory.create_thread(name)
    assert memory.list_threads() == ["zeta", "alpha", "mid"]


def test_marker_vector_is_a_unit_vector_not_a_zero_vector(memory, store):
    memory.create_thread("conv-1")
    (marker,) = list(store.collections[THREADS_COLLECTION]["points"].values())
    assert marker["payload"][THREAD_MARKER_KEY] is True
    assert sum(v * v for v in marker["vector"]) == pytest.approx(1.0)


# ===========================================================================
# Isolation between threads
# ===========================================================================


def test_history_returns_only_this_threads_messages(memory):
    a = memory.create_thread("a")
    b = memory.create_thread("b")
    _msgs(a, 3, "a")
    _msgs(b, 2, "b")

    assert [m.content for m in a.history()] == ["a0", "a1", "a2"]
    assert [m.content for m in b.history()] == ["b0", "b1"]


def test_iter_history_returns_only_this_threads_messages(memory):
    a = memory.create_thread("a")
    b = memory.create_thread("b")
    _msgs(a, 3, "a")
    _msgs(b, 2, "b")

    assert [m.content for m in a.iter_history()] == ["a0", "a1", "a2"]


def test_search_returns_only_this_threads_messages(memory):
    a = memory.create_thread("a")
    b = memory.create_thread("b")
    _msgs(a, 2, "a")
    _msgs(b, 2, "b")

    hits = a.search(vector=_v(), limit=50)
    assert {h["payload"]["content"] for h in hits} == {"a0", "a1"}


def test_count_and_iter_return_only_this_threads_messages(memory):
    a = memory.create_thread("a")
    b = memory.create_thread("b")
    _msgs(a, 3, "a")
    _msgs(b, 7, "b")

    assert a.count() == 3
    assert b.count() == 7
    assert {p["payload"]["content"] for p in a.iter()} == {"a0", "a1", "a2"}


def test_retrieve_will_not_reach_into_a_sibling_thread(memory):
    a = memory.create_thread("a")
    b = memory.create_thread("b")
    (a_id,) = _msgs(a, 1, "a")
    (b_id,) = _msgs(b, 1, "b")

    assert [p["id"] for p in a.retrieve([a_id])] == [a_id]
    # b's id exists in the shared collection, but not in a.
    assert a.retrieve([b_id]) == []


def test_retrieve_without_payload_still_scopes_and_still_omits_the_payload(memory):
    a = memory.create_thread("a")
    b = memory.create_thread("b")
    (a_id,) = _msgs(a, 1, "a")
    (b_id,) = _msgs(b, 1, "b")

    got = a.retrieve([a_id, b_id], with_payload=False)
    assert [p["id"] for p in got] == [a_id]
    assert "payload" not in got[0]


def test_delete_by_id_will_not_reach_into_a_sibling_thread(memory):
    a = memory.create_thread("a")
    b = memory.create_thread("b")
    (b_id,) = _msgs(b, 1, "b")

    a.delete([b_id])
    assert b.count() == 1


def test_delete_by_id_scopes_server_side_in_one_request(memory, store):
    """The thread clause travels WITH the ids, so the engine enforces the
    boundary. A client-side check first would be a second round trip and a
    rule the next caller could step around.
    """
    a = memory.create_thread("a")
    (keep, drop) = _msgs(a, 2, "a")

    sent = []
    real_delete = store.delete
    store.delete = lambda name, sel, **kw: (
        sent.append(sel) or real_delete(name, sel, **kw)
    )
    a.delete([drop])

    (selector,) = sent
    assert selector["must"][0] == {
        "key": THREAD_ID_KEY,
        "match": {"value": "a"},
    }
    assert selector["must"][1] == {"has_id": [drop]}
    assert [p["id"] for p in a.iter()] == [keep]


def test_delete_with_an_empty_id_list_sends_no_request(memory, store):
    """A behaviour change, pinned because it is one.

    It used to send a delete carrying an empty points list. It now returns
    True without a request, and for a Thread that is a SAFETY property
    rather than a saved round trip: an id list becomes a `has_id` clause,
    and a request carrying an empty `has_id` is one engine-side semantic
    away from matching the whole thread.
    """
    a = memory.create_thread("a")
    _msgs(a, 3, "a")

    sent = []
    real_delete = store.delete
    store.delete = lambda name, sel, **kw: (
        sent.append(sel) or real_delete(name, sel, **kw)
    )

    assert a.delete([]) is True
    assert sent == []
    assert a.count() == 3


def test_namespace_delete_with_an_empty_id_list_sends_no_request(memory, store):
    ns = memory.create_namespace("kb", vector_size=DIM)
    ns.add(text="x", vector=_v())

    sent = []
    real_delete = store.delete
    store.delete = lambda name, sel, **kw: (
        sent.append(sel) or real_delete(name, sel, **kw)
    )

    assert ns.delete([]) is True
    assert sent == []
    assert ns.count() == 1


def test_metadata_writes_still_raise_rather_than_silently_no_op(memory):
    """Why the metadata writers keep their read.

    The payload endpoints accept a filter, so these could scope themselves
    the way delete() does. They do not, because a filter that matches
    nothing is a SUCCESS, and these are documented to raise
    PointNotFoundError when the point is not there. Scoping them by filter
    would turn a write to a missing id into a silent no-op reported as
    success. delete() has no such contract to lose.
    """
    a = memory.create_thread("a")
    missing = "00000000-0000-4000-8000-0000000000aa"
    with pytest.raises(PointNotFoundError):
        a.merge_metadata(missing, {"x": 1})


def test_metadata_writes_will_not_reach_into_a_sibling_thread(memory):
    a = memory.create_thread("a")
    b = memory.create_thread("b")
    (b_id,) = _msgs(b, 1, "b")

    for call in (
        lambda: a.set_metadata(b_id, {"x": 1}),
        lambda: a.merge_metadata(b_id, {"x": 1}),
        lambda: a.delete_metadata_keys(b_id, ["x"]),
    ):
        with pytest.raises(PointNotFoundError):
            call()


# ===========================================================================
# Markers never read as messages
# ===========================================================================


def _fail_open_with_a_stamped_marker(store):
    """Make every read behave as if the marker exclusion had been mistyped.

    The docs are explicit that a filter is forwarded verbatim and a key the
    engine does not recognise "is passed along and quietly does nothing", so
    the server-side exclusion is not a guarantee. Stamping a `ts` on the
    marker removes the one incidental reason it would be dropped, leaving
    only the client-side guard under test.
    """
    store.fail_open_on_must_not = True
    for p in store.collections[THREADS_COLLECTION]["points"].values():
        if p["payload"].get(THREAD_MARKER_KEY):
            p["payload"]["ts"] = 99.0


def test_marker_never_surfaces_in_history(memory, store):
    a = memory.create_thread("a")
    _msgs(a, 2, "a")
    assert [m.content for m in a.history()] == ["a0", "a1"]

    # Second guard, exercised on its own. The filter fails OPEN by design
    # (the proxy forwards it verbatim and never validates it), so make the
    # store do exactly that and give the marker a ts so nothing else would
    # drop it. history() must still refuse to call it a message.
    _fail_open_with_a_stamped_marker(store)
    assert [m.content for m in a.history()] == ["a0", "a1"]


def test_marker_never_surfaces_in_iter_history(memory, store):
    a = memory.create_thread("a")
    _msgs(a, 2, "a")
    assert [m.content for m in a.iter_history()] == ["a0", "a1"]

    _fail_open_with_a_stamped_marker(store)
    assert [m.content for m in a.iter_history()] == ["a0", "a1"]


def test_marker_never_surfaces_in_search(memory):
    a = memory.create_thread("a")
    _msgs(a, 1, "a")
    hits = a.search(vector=_v(), limit=50)
    assert len(hits) == 1
    assert hits[0]["payload"]["content"] == "a0"


def test_marker_is_not_counted_and_not_iterated(memory):
    a = memory.create_thread("a")
    _msgs(a, 3, "a")
    assert a.count() == 3
    assert len(list(a.iter())) == 3


def test_a_filtered_delete_leaves_the_thread_in_existence(memory):
    a = memory.create_thread("a")
    _msgs(a, 2, "a")
    a.delete({"must": [{"key": "role", "match": {"value": "user"}}]})
    assert a.count() == 0
    # The marker survived a message delete, so the thread still exists.
    assert memory.thread_exists("a") is True


# ===========================================================================
# A caller filter is COMBINED with the thread clause, never substituted
# ===========================================================================


def test_a_caller_filter_cannot_widen_the_scope(memory):
    a = memory.create_thread("a")
    b = memory.create_thread("b")
    _msgs(a, 1, "a")
    _msgs(b, 1, "b")

    # A filter that, on its own, would match every point of thread b.
    widen = {"must": [{"key": THREAD_ID_KEY, "match": {"value": "b"}}]}

    assert a.search(vector=_v(), limit=50, filter=widen) == []
    assert a.count(filter=widen) == 0
    assert list(a.iter(filter=widen)) == []

    a.delete(widen)
    assert b.count() == 1


def test_a_caller_should_clause_cannot_widen_the_scope(memory):
    a = memory.create_thread("a")
    b = memory.create_thread("b")
    _msgs(a, 1, "a")
    _msgs(b, 1, "b")

    widen = {"should": [{"key": THREAD_ID_KEY, "match": {"value": "b"}}]}
    assert a.count(filter=widen) == 0


def test_a_caller_filter_still_narrows(memory):
    a = memory.create_thread("a")
    a.add(role="user", content="keep", vector=_v(), ts=1.0)
    a.add(role="assistant", content="drop", vector=_v(), ts=2.0)

    narrowed = a.search(
        vector=_v(),
        limit=50,
        filter={"must": [{"key": "role", "match": {"value": "user"}}]},
    )
    assert [h["payload"]["content"] for h in narrowed] == ["keep"]


def test_a_caller_clause_typo_still_fails_loudly(memory):
    a = memory.create_thread("a")
    with pytest.raises(Exception) as excinfo:
        a.count(filter={"mustNot": [{"key": "role", "match": {"value": "user"}}]})
    assert "mustNot" in str(excinfo.value)


# ===========================================================================
# clear() and delete_thread()
# ===========================================================================


def test_clear_leaves_a_sibling_threads_points_intact(memory):
    a = memory.create_thread("a")
    b = memory.create_thread("b")
    _msgs(a, 3, "a")
    _msgs(b, 4, "b")

    a.clear()

    # The sibling is untouched — messages AND its existence.
    assert b.count() == 4
    assert [m.content for m in b.history()] == ["b0", "b1", "b2", "b3"]
    assert memory.thread_exists("b") is True
    # ...and the cleared thread is gone, the way clear() has always meant.
    assert memory.thread_exists("a") is False


def test_clear_does_not_drop_the_shared_collection(memory, store):
    a = memory.create_thread("a")
    memory.create_thread("b")
    a.clear()
    assert THREADS_COLLECTION in store.collections


def test_delete_thread_leaves_a_sibling_threads_points_intact(memory):
    a = memory.create_thread("a")
    b = memory.create_thread("b")
    _msgs(a, 3, "a")
    _msgs(b, 4, "b")

    assert memory.delete_thread("a") is True
    assert memory.thread_exists("a") is False
    assert b.count() == 4
    assert memory.list_threads() == ["b"]


def test_a_cleared_thread_can_be_created_again(memory):
    a = memory.create_thread("a")
    _msgs(a, 2, "a")
    a.clear()
    again = memory.create_thread("a")
    assert again.count() == 0


# ===========================================================================
# get_thread / namespaces / schema surface
# ===========================================================================


def test_get_thread_counts_this_thread_not_the_collection(memory):
    a = memory.create_thread("a")
    b = memory.create_thread("b")
    _msgs(a, 3, "a")
    _msgs(b, 9, "b")

    info = memory.get_thread("a")
    assert info.name == "a"
    assert info.points_count == 3
    assert info.config.size == DIM


def test_namespaces_are_unaffected_and_still_one_collection_each(memory, store):
    memory.create_namespace("kb", vector_size=DIM)
    memory.create_thread("a")
    assert memory.list_namespaces() == ["kb"]
    assert sorted(store.collections) == sorted([THREADS_COLLECTION, "kb"])


def test_namespace_clear_still_drops_its_own_collection(memory, store):
    ns = memory.create_namespace("kb", vector_size=DIM)
    ns.clear()
    assert "kb" not in store.collections


def test_a_thread_has_no_collection_level_schema_surface(memory):
    a = memory.create_thread("a")
    # A schema belongs to a collection, and a Thread no longer has one to
    # itself: `set_schema` on one thread would have imposed a schema on every
    # other thread in the workspace. Removed rather than left lying.
    for gone in (
        "get_schema",
        "set_schema",
        "delete_schema",
        "analyze_schema",
        "refresh_schema",
        "clear_schema_cache",
    ):
        assert not hasattr(a, gone), gone


def test_a_namespace_keeps_the_schema_surface(memory):
    ns = memory.create_namespace("kb", vector_size=DIM)
    for kept in (
        "get_schema",
        "set_schema",
        "delete_schema",
        "analyze_schema",
        "refresh_schema",
        "clear_schema_cache",
    ):
        assert hasattr(ns, kept), kept


def test_the_threads_collection_indexes_both_filtered_keys(memory, store):
    memory.create_thread("a")
    assert store.indexes == [
        (THREADS_COLLECTION, THREAD_ID_KEY, "keyword"),
        (THREADS_COLLECTION, THREAD_MARKER_KEY, "bool"),
    ]


def test_an_unreadable_dimension_is_not_treated_as_a_mismatch(memory, store):
    """0 means UNKNOWN, not "a zero-dimension collection".

    Collection.from_dict defaults size to 0 when the response carried no
    vectors config, so comparing it against the client's size would report a
    mismatch that is really "we could not read it". The skip is explicit in
    the code for exactly this reason; this pins that it stays a skip and not a
    silently-passing check.
    """
    from aetherfy_vectors.models import Collection, DistanceMetric, VectorConfig

    memory.create_thread("first")

    store.get_collection = lambda name, **kw: Collection(
        name=name,
        config=VectorConfig(size=0, distance=DistanceMetric.COSINE),
    )
    # No ThreadVectorSizeMismatchError: there is nothing to compare against.
    memory.create_thread("second")
    assert sorted(memory.list_threads()) == ["first", "second"]


def test_a_readable_mismatch_still_raises(memory, store):
    from aetherfy_vectors.models import Collection, DistanceMetric, VectorConfig

    memory.create_thread("first")
    store.get_collection = lambda name, **kw: Collection(
        name=name,
        config=VectorConfig(size=1536, distance=DistanceMetric.COSINE),
    )
    with pytest.raises(ThreadVectorSizeMismatchError) as excinfo:
        memory.create_thread("second")
    assert excinfo.value.existing == 1536


def test_the_client_default_dimension_is_still_384(store):
    m = MemoryClient(client=store)
    m.create_thread("a")
    assert store.collections[THREADS_COLLECTION]["config"].size == DEFAULT_VECTOR_SIZE
