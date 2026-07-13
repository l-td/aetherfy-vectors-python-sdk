"""
Tests for the Aetherfy Memory SDK (MemoryClient, Namespace, Thread).

MemoryClient delegates every operation to an underlying AetherfyVectorsClient,
so these tests mock the vectors client and verify that MemoryClient produces
the correct calls and enforces its own contracts:

- Namespace/thread lifecycle (create, list, exists, delete)
- Required-scope rule (no root-level add/search)
- Required-create rule (add/search before create → error)
- Reserved prefix rule (namespace names can't start with __thread__)
- Collection naming convention (__thread__<id> for threads)
- Forward-compat error when vector is omitted
- Operations parity with AetherfyVectorsClient
- Thread.history() ordering
"""

from unittest.mock import MagicMock

import pytest

from aetherfy_memory import (
    MemoryClient,
    Message,
    Namespace,
    Thread,
    DEFAULT_VECTOR_SIZE,
)
from aetherfy_memory.exceptions import (
    EmbeddingNotSupportedError,
    InvalidNameError,
    NamespaceAlreadyExistsError,
    NamespaceNotFoundError,
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
)
from aetherfy_vectors.models import Collection, DistanceMetric, VectorConfig
from aetherfy_vectors.exceptions import ValidationError
from aetherfy_vectors.utils import validate_point_id


def _validating_upsert(_collection, points):
    """Mock upsert side-effect that runs the real point-id validator on each
    point, mirroring what AetherfyVectorsClient.upsert does (via
    format_points_for_upsert). Lets memory tests pin that an invalid explicit
    id reaches — and is rejected by — the validator, rather than being masked
    by a fully no-op MagicMock upsert.
    """
    for p in points:
        validate_point_id(p["id"])
    return True


def _fake_collection(name: str) -> Collection:
    """Helper — Collection requires config; tests only care about the name."""
    return Collection(
        name=name,
        config=VectorConfig(size=DEFAULT_VECTOR_SIZE, distance=DistanceMetric.COSINE),
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_vectors_client():
    """A MagicMock AetherfyVectorsClient configured for MemoryClient's needs."""
    client = MagicMock()
    client.workspace = "my-bot"
    # collection_exists defaults to False — tests override as needed.
    client.collection_exists.return_value = False
    # get_collections defaults to empty.
    client.get_collections.return_value = []
    # _scope_collection passes through without workspace prefix for assertion
    # simplicity. The real client's workspace prefixing is tested separately
    # in test_workspace.py; here we verify MemoryClient's own naming layer.
    client._scope_collection.side_effect = lambda name: (
        f"my-bot/{name}" if name else name
    )
    return client


@pytest.fixture
def memory(fake_vectors_client):
    """A MemoryClient wired to the fake vectors client."""
    return MemoryClient(client=fake_vectors_client)


# =============================================================================
# Construction / introspection
# =============================================================================


class TestMemoryClientConstruction:
    def test_workspace_surfaces_from_underlying_client(
        self, memory, fake_vectors_client
    ):
        assert memory.workspace == "my-bot"

    def test_vectors_escape_hatch(self, memory, fake_vectors_client):
        assert memory.vectors is fake_vectors_client

    def test_repr(self, memory):
        assert repr(memory) == "MemoryClient(workspace='my-bot')"


# =============================================================================
# Name validation
# =============================================================================


class TestNameValidation:
    @pytest.mark.parametrize(
        "bad_name",
        [
            "",
            " ",
            "-leading-dash",
            "_leading-underscore",
            ".leading-dot",
            "has/slash",
            "has space",
            "has:colon",
            "has\\backslash",
            None,
            123,
        ],
    )
    def test_invalid_namespace_names_rejected(self, memory, bad_name):
        with pytest.raises(InvalidNameError):
            memory.create_namespace(bad_name)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "good_name",
        [
            "a",
            "customer-42",
            "customer_42",
            "customer.42",
            "CustomerNotes",
            "1facts",
            "scrape-log-v2",
        ],
    )
    def test_valid_namespace_names_accepted(
        self, memory, fake_vectors_client, good_name
    ):
        fake_vectors_client.collection_exists.return_value = False
        ns = memory.create_namespace(good_name)
        assert ns.name == good_name
        fake_vectors_client.create_collection.assert_called_once()


# =============================================================================
# Thread-prefix isolation
# =============================================================================
#
# The internal thread collection prefix starts with `__`, and the user-name
# regex forbids leading `_`. So user-facing APIs can never accept a name
# that collides with the thread prefix — it's rejected at the regex gate
# with InvalidNameError. This test pins that contract so a future relaxation
# of the regex doesn't silently open the collision.


class TestThreadPrefixIsolation:
    @pytest.mark.parametrize(
        "op", ["create_namespace", "namespace", "delete_namespace"]
    )
    def test_user_apis_reject_thread_prefix_via_regex(self, memory, op):
        with pytest.raises(InvalidNameError):
            getattr(memory, op)("__thread__foo")


# =============================================================================
# Namespace lifecycle
# =============================================================================


class TestNamespaceLifecycle:
    def test_create_namespace_creates_collection_with_defaults(
        self, memory, fake_vectors_client
    ):
        fake_vectors_client.collection_exists.return_value = False

        memory.create_namespace("customer-42")

        call = fake_vectors_client.create_collection.call_args
        assert call.args[0] == "customer-42"
        cfg = call.args[1]
        assert isinstance(cfg, VectorConfig)
        assert cfg.size == DEFAULT_VECTOR_SIZE
        assert cfg.distance == DistanceMetric.COSINE

    def test_create_namespace_accepts_custom_dimension_and_distance(
        self, memory, fake_vectors_client
    ):
        memory.create_namespace(
            "customer-42", vector_size=1536, distance=DistanceMetric.DOT
        )
        call = fake_vectors_client.create_collection.call_args
        cfg = call.args[1]
        assert cfg.size == 1536
        assert cfg.distance == DistanceMetric.DOT

    def test_create_namespace_raises_when_already_exists(
        self, memory, fake_vectors_client
    ):
        fake_vectors_client.collection_exists.return_value = True
        with pytest.raises(NamespaceAlreadyExistsError):
            memory.create_namespace("customer-42")
        fake_vectors_client.create_collection.assert_not_called()

    def test_namespace_requires_prior_create(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = False
        with pytest.raises(NamespaceNotFoundError):
            memory.namespace("customer-42")

    def test_namespace_returns_handle_when_exists(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        assert isinstance(ns, Namespace)
        assert ns.name == "customer-42"

    def test_list_namespaces_excludes_threads(self, memory, fake_vectors_client):
        fake_vectors_client.get_collections.return_value = [
            _fake_collection("customer-42"),
            _fake_collection("scrape-log"),
            _fake_collection("__thread__conv-99"),
            _fake_collection("__thread__conv-100"),
        ]
        assert memory.list_namespaces() == ["customer-42", "scrape-log"]

    def test_list_threads_returns_stripped_ids(self, memory, fake_vectors_client):
        fake_vectors_client.get_collections.return_value = [
            _fake_collection("customer-42"),
            _fake_collection("__thread__conv-99"),
            _fake_collection("__thread__conv-100"),
        ]
        assert memory.list_threads() == ["conv-99", "conv-100"]

    def test_delete_namespace_is_idempotent(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = False
        assert memory.delete_namespace("does-not-exist") is False
        fake_vectors_client.delete_collection.assert_not_called()

    def test_delete_namespace_drops_collection(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        fake_vectors_client.delete_collection.return_value = True
        assert memory.delete_namespace("customer-42") is True
        fake_vectors_client.delete_collection.assert_called_once_with("customer-42")

    def test_get_namespace_returns_metadata(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        info = _fake_collection("customer-42")
        info.points_count = 123
        fake_vectors_client.get_collection.return_value = info

        returned = memory.get_namespace("customer-42")
        assert returned.name == "customer-42"
        assert returned.points_count == 123
        fake_vectors_client.get_collection.assert_called_once_with("customer-42")

    def test_get_namespace_missing_raises(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = False
        with pytest.raises(NamespaceNotFoundError):
            memory.get_namespace("missing")


# =============================================================================
# Thread lifecycle
# =============================================================================


class TestThreadLifecycle:
    def test_create_thread_uses_reserved_prefix(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = False
        memory.create_thread("conv-99")
        call = fake_vectors_client.create_collection.call_args
        assert call.args[0] == "__thread__conv-99"

    def test_create_thread_raises_when_already_exists(
        self, memory, fake_vectors_client
    ):
        fake_vectors_client.collection_exists.return_value = True
        with pytest.raises(ThreadAlreadyExistsError):
            memory.create_thread("conv-99")

    def test_thread_requires_prior_create(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = False
        with pytest.raises(ThreadNotFoundError):
            memory.thread("conv-99")

    def test_thread_returns_handle_when_exists(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        t = memory.thread("conv-99")
        assert isinstance(t, Thread)
        assert t.id == "conv-99"

    def test_delete_thread_is_idempotent(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = False
        assert memory.delete_thread("conv-99") is False

    def test_delete_thread_drops_prefixed_collection(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        fake_vectors_client.delete_collection.return_value = True
        memory.delete_thread("conv-99")
        fake_vectors_client.delete_collection.assert_called_once_with(
            "__thread__conv-99"
        )

    def test_get_thread_strips_prefix_from_returned_name(
        self, memory, fake_vectors_client
    ):
        fake_vectors_client.collection_exists.return_value = True
        info = _fake_collection("__thread__conv-99")
        info.points_count = 42
        fake_vectors_client.get_collection.return_value = info

        returned = memory.get_thread("conv-99")
        # User-facing id, not the internal prefix
        assert returned.name == "conv-99"
        assert returned.points_count == 42
        fake_vectors_client.get_collection.assert_called_once_with("__thread__conv-99")

    def test_get_thread_missing_raises(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = False
        with pytest.raises(ThreadNotFoundError):
            memory.get_thread("missing")


# =============================================================================
# Namespace operations
# =============================================================================


class TestPointIdGeneration:
    """Pin the canonical-UUID contract for SDK-generated point IDs.

    Qdrant emits the canonical 8-4-4-4-12 hyphenated form on read. The
    SDK must emit the same so round-trip ID equality (caller tracks
    SDK-returned IDs and compares against scroll/iter output) holds.
    A regression that uses ``uuid.uuid4().hex`` (32 hex chars, no
    hyphens) silently breaks every memory-iter contract.
    """

    CANONICAL_UUID_RE = (
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )

    def test_namespace_add_returns_canonical_uuid(self, memory, fake_vectors_client):
        import re

        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        pid = ns.add(text="x", vector=[0.1])
        assert re.match(
            self.CANONICAL_UUID_RE, pid
        ), f"Expected canonical UUID, got {pid!r}"

    def test_namespace_add_many_returns_canonical_uuids(
        self, memory, fake_vectors_client
    ):
        import re

        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        ids = ns.add_many([{"text": str(i), "vector": [0.1]} for i in range(8)])
        for pid in ids:
            assert re.match(
                self.CANONICAL_UUID_RE, pid
            ), f"Expected canonical UUID, got {pid!r}"

    def test_thread_add_and_append_many_return_canonical_uuids(
        self, memory, fake_vectors_client
    ):
        import re

        fake_vectors_client.collection_exists.return_value = True
        t = memory.thread("conv-99")
        single = t.add(role="user", content="hi", vector=[0.1])
        many = t.append_many(
            [{"role": "user", "content": str(i), "vector": [0.1]} for i in range(4)]
        )
        for pid in [single, *many]:
            assert re.match(
                self.CANONICAL_UUID_RE, pid
            ), f"Expected canonical UUID, got {pid!r}"


class TestNamespaceOperations:
    def test_add_requires_vector(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        with pytest.raises(EmbeddingNotSupportedError):
            ns.add(text="Lives in NYC")
        fake_vectors_client.upsert.assert_not_called()

    def test_add_writes_point_with_vector(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        point_id = ns.add(text="Lives in NYC", vector=[0.1, 0.2, 0.3])

        fake_vectors_client.upsert.assert_called_once()
        args = fake_vectors_client.upsert.call_args
        assert args.args[0] == "customer-42"
        points = args.args[1]
        assert len(points) == 1
        assert points[0]["id"] == point_id
        assert points[0]["vector"] == [0.1, 0.2, 0.3]
        assert points[0]["payload"]["text"] == "Lives in NYC"

    def test_add_respects_custom_id(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        ns.add(id="11111111-1111-4111-8111-111111111111", text="x", vector=[0.1])
        args = fake_vectors_client.upsert.call_args
        assert args.args[1][0]["id"] == "11111111-1111-4111-8111-111111111111"

    def test_add_integer_id_reaches_upsert_as_a_number(
        self, memory, fake_vectors_client
    ):
        # An integer is a valid unsigned-integer point id. It must reach the
        # wire AS A NUMBER — a prior str() coercion turned 42 into "42", a
        # numeric string the ingress validator rejects.
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        returned = ns.add(id=42, text="x", vector=[0.1])
        sent_id = fake_vectors_client.upsert.call_args.args[1][0]["id"]
        assert sent_id == 42
        assert isinstance(sent_id, int) and not isinstance(sent_id, bool)
        # add() returns the id as-authored, not stringified.
        assert returned == 42 and isinstance(returned, int)

    def test_add_default_id_is_a_uuid_string(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        returned = ns.add(text="x", vector=[0.1])
        sent_id = fake_vectors_client.upsert.call_args.args[1][0]["id"]
        assert isinstance(sent_id, str)
        # Canonical uuid4 — must pass the point-id validator.
        validate_point_id(sent_id)
        assert returned == sent_id

    def test_add_non_uuid_string_id_is_rejected_by_the_validator(
        self, memory, fake_vectors_client
    ):
        # The (b) boundary: a semantic string key is NOT a valid point id.
        # Memory passes it through as-authored; the client's upsert validator
        # rejects it. (The mock's upsert runs the real validator here.)
        fake_vectors_client.collection_exists.return_value = True
        fake_vectors_client.upsert.side_effect = _validating_upsert
        ns = memory.namespace("customer-42")
        with pytest.raises(ValidationError):
            ns.add(id="user-42-preferences", text="x", vector=[0.1])

    def test_add_nests_user_metadata(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        ns.add(text="x", vector=[0.1], metadata={"kind": "pref"})
        payload = fake_vectors_client.upsert.call_args.args[1][0]["payload"]
        # Metadata is nested so it can't shadow reserved fields like `text`.
        assert payload["metadata"] == {"kind": "pref"}
        assert payload["text"] == "x"

    def test_search_delegates(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        ns.search(vector=[0.1, 0.2], limit=5, filter={"x": 1})
        args = fake_vectors_client.search.call_args
        assert args.args[0] == "customer-42"
        assert args.kwargs["query_vector"] == [0.1, 0.2]
        assert args.kwargs["limit"] == 5
        assert args.kwargs["query_filter"] == {"x": 1}

    def test_retrieve_delegates(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        ns.retrieve(["a", "b"], with_vectors=True)
        args = fake_vectors_client.retrieve.call_args
        assert args.args[0] == "customer-42"
        assert args.args[1] == ["a", "b"]
        assert args.kwargs["with_vectors"] is True

    def test_count_delegates(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        ns.count(filter={"k": "v"})
        fake_vectors_client.count.assert_called_once()
        args = fake_vectors_client.count.call_args
        assert args.args[0] == "customer-42"
        assert args.kwargs["count_filter"] == {"k": "v"}

    def test_delete_by_ids_delegates(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        ns.delete(["a", "b"])
        fake_vectors_client.delete.assert_called_once_with("customer-42", ["a", "b"])

    def test_clear_drops_collection(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        ns.clear()
        fake_vectors_client.delete_collection.assert_called_once_with("customer-42")

    def test_add_many_empty_returns_empty_list_no_round_trip(
        self, memory, fake_vectors_client
    ):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        assert ns.add_many([]) == []
        fake_vectors_client.upsert.assert_not_called()

    def test_add_many_single_upsert_returns_ids_in_order(
        self, memory, fake_vectors_client
    ):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        ids = ns.add_many(
            [
                {"text": "a", "vector": [0.1], "metadata": {"i": 0}},
                {
                    "text": "b",
                    "vector": [0.2],
                    "id": "22222222-2222-4222-8222-222222222222",
                },
                {"text": "c", "vector": [0.3]},
            ]
        )
        assert len(ids) == 3
        assert ids[1] == "22222222-2222-4222-8222-222222222222"
        fake_vectors_client.upsert.assert_called_once()
        coll, points = fake_vectors_client.upsert.call_args.args
        assert coll == "customer-42"
        assert [p["payload"]["text"] for p in points] == ["a", "b", "c"]
        assert points[0]["payload"]["metadata"] == {"i": 0}

    def test_add_many_integer_id_reaches_upsert_as_a_number(
        self, memory, fake_vectors_client
    ):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        ids = ns.add_many(
            [
                {"text": "a", "vector": [0.1], "id": 7},
                {"text": "b", "vector": [0.2]},
            ]
        )
        _coll, points = fake_vectors_client.upsert.call_args.args
        assert points[0]["id"] == 7 and isinstance(points[0]["id"], int)
        assert ids[0] == 7 and isinstance(ids[0], int)

    def test_add_many_pinpoints_bad_index(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        with pytest.raises(EmbeddingNotSupportedError, match=r"add_many\[1\]"):
            ns.add_many(
                [
                    {"text": "good", "vector": [0.1]},
                    {"text": "bad"},
                ]
            )
        fake_vectors_client.upsert.assert_not_called()

    def test_add_many_rejects_non_list(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        ns = memory.namespace("customer-42")
        with pytest.raises(TypeError):
            ns.add_many("not-a-list")  # type: ignore[arg-type]


# =============================================================================
# Thread operations
# =============================================================================


class TestThreadOperations:
    def _open_thread(self, memory, fake_vectors_client, thread_id="conv-99"):
        fake_vectors_client.collection_exists.return_value = True
        return memory.thread(thread_id)

    def test_thread_add_requires_vector(self, memory, fake_vectors_client):
        t = self._open_thread(memory, fake_vectors_client)
        with pytest.raises(EmbeddingNotSupportedError):
            t.add(role="user", content="hi")

    def test_thread_add_rejects_empty_role(self, memory, fake_vectors_client):
        t = self._open_thread(memory, fake_vectors_client)
        with pytest.raises(ValueError):
            t.add(role="", content="hi", vector=[0.1])

    def test_thread_add_writes_message_payload(self, memory, fake_vectors_client):
        t = self._open_thread(memory, fake_vectors_client)
        pid = t.add(role="user", content="hi", vector=[0.1, 0.2], ts=1000.0)

        fake_vectors_client.upsert.assert_called_once()
        args = fake_vectors_client.upsert.call_args
        assert args.args[0] == "__thread__conv-99"
        point = args.args[1][0]
        assert point["id"] == pid
        assert point["vector"] == [0.1, 0.2]
        payload = point["payload"]
        assert payload["role"] == "user"
        assert payload["content"] == "hi"
        assert payload["ts"] == 1000.0

    def test_thread_add_integer_id_reaches_upsert_as_a_number(
        self, memory, fake_vectors_client
    ):
        t = self._open_thread(memory, fake_vectors_client)
        pid = t.add(role="user", content="hi", vector=[0.1], id=99)
        sent_id = fake_vectors_client.upsert.call_args.args[1][0]["id"]
        assert sent_id == 99 and isinstance(sent_id, int)
        assert pid == 99 and isinstance(pid, int)

    def test_thread_add_non_uuid_string_id_rejected_by_validator(
        self, memory, fake_vectors_client
    ):
        fake_vectors_client.upsert.side_effect = _validating_upsert
        t = self._open_thread(memory, fake_vectors_client)
        with pytest.raises(ValidationError):
            t.add(role="user", content="hi", vector=[0.1], id="conv-99-msg-1")

    def test_thread_add_sets_ts_when_omitted(self, memory, fake_vectors_client):
        t = self._open_thread(memory, fake_vectors_client)
        t.add(role="user", content="hi", vector=[0.1])
        payload = fake_vectors_client.upsert.call_args.args[1][0]["payload"]
        assert payload["ts"] is not None
        assert isinstance(payload["ts"], float)

    def test_append_many_empty_returns_empty_list_no_round_trip(
        self, memory, fake_vectors_client
    ):
        t = self._open_thread(memory, fake_vectors_client)
        assert t.append_many([]) == []
        fake_vectors_client.upsert.assert_not_called()

    def test_append_many_single_upsert_with_role_content_ts_payloads(
        self, memory, fake_vectors_client
    ):
        t = self._open_thread(memory, fake_vectors_client)
        ids = t.append_many(
            [
                {"role": "user", "content": "hi", "vector": [0.1], "ts": 1000.0},
                {
                    "role": "assistant",
                    "content": "hello",
                    "vector": [0.2],
                    "ts": 1001.0,
                    "id": "33333333-3333-4333-8333-333333333333",
                },
            ]
        )
        assert ids[1] == "33333333-3333-4333-8333-333333333333"
        fake_vectors_client.upsert.assert_called_once()
        coll, points = fake_vectors_client.upsert.call_args.args
        assert coll == "__thread__conv-99"
        assert len(points) == 2
        assert points[0]["payload"]["role"] == "user"
        assert points[0]["payload"]["content"] == "hi"
        assert points[0]["payload"]["ts"] == 1000.0
        assert points[1]["payload"]["role"] == "assistant"

    def test_append_many_generates_per_message_ts_when_omitted(
        self, memory, fake_vectors_client
    ):
        t = self._open_thread(memory, fake_vectors_client)
        t.append_many(
            [
                {"role": "user", "content": "a", "vector": [0.1]},
                {"role": "user", "content": "b", "vector": [0.1]},
            ]
        )
        points = fake_vectors_client.upsert.call_args.args[1]
        assert isinstance(points[0]["payload"]["ts"], float)
        assert isinstance(points[1]["payload"]["ts"], float)

    def test_append_many_pinpoints_bad_index(self, memory, fake_vectors_client):
        t = self._open_thread(memory, fake_vectors_client)
        with pytest.raises(ValueError, match=r"append_many\[1\]"):
            t.append_many(
                [
                    {"role": "user", "content": "good", "vector": [0.1]},
                    {"role": "", "content": "bad", "vector": [0.1]},
                ]
            )
        fake_vectors_client.upsert.assert_not_called()

    def test_thread_has_no_add_many(self, memory, fake_vectors_client):
        # Thread and Namespace both extend _Scope but declare their own write
        # API; Thread does NOT inherit Namespace.add_many, so there is no
        # method to call (previously a NoReturn override raised at runtime —
        # now the method simply does not exist). Callers use append_many.
        t = self._open_thread(memory, fake_vectors_client)
        assert not hasattr(t, "add_many")
        assert hasattr(t, "append_many")

    def test_thread_history_returns_messages_in_order(
        self, memory, fake_vectors_client
    ):
        t = self._open_thread(memory, fake_vectors_client)

        # Simulate scroll response with unordered points
        fake_vectors_client.scroll.return_value = {
            "next_page_offset": None,
            "points": [
                {
                    "id": "00000000-0000-4000-8000-000000000003",
                    "payload": {"role": "user", "content": "third", "ts": 3.0},
                },
                {
                    "id": "00000000-0000-4000-8000-000000000001",
                    "payload": {"role": "user", "content": "first", "ts": 1.0},
                },
                {
                    "id": "00000000-0000-4000-8000-000000000002",
                    "payload": {
                        "role": "assistant",
                        "content": "second",
                        "ts": 2.0,
                    },
                },
            ],
        }

        history = t.history(limit=10)

        assert [m.content for m in history] == ["first", "second", "third"]
        assert all(isinstance(m, Message) for m in history)

    def test_thread_history_desc_order(self, memory, fake_vectors_client):
        t = self._open_thread(memory, fake_vectors_client)
        fake_vectors_client.scroll.return_value = {
            "next_page_offset": None,
            "points": [
                {
                    "id": "00000000-0000-4000-8000-000000000001",
                    "payload": {"role": "u", "content": "a", "ts": 1.0},
                },
                {
                    "id": "00000000-0000-4000-8000-000000000002",
                    "payload": {"role": "u", "content": "b", "ts": 2.0},
                },
            ],
        }
        history = t.history(limit=10, order="desc")
        assert [m.content for m in history] == ["b", "a"]

    # Round-trip: what add() writes is what history() reads back, id TYPE intact.
    def test_thread_history_round_trips_integer_id_as_int(
        self, memory, fake_vectors_client
    ):
        t = self._open_thread(memory, fake_vectors_client)
        t.add(role="user", content="hi", vector=[0.1], id=42, ts=5.0)
        # Feed back exactly what add wrote, as the server would return it.
        written = fake_vectors_client.upsert.call_args.args[1][0]
        fake_vectors_client.scroll.return_value = {
            "next_page_offset": None,
            "points": [written],
        }
        history = t.history(limit=10)
        assert len(history) == 1
        assert history[0].id == 42
        assert isinstance(history[0].id, int) and not isinstance(history[0].id, bool)

    def test_thread_history_round_trips_uuid_id_as_str(
        self, memory, fake_vectors_client
    ):
        t = self._open_thread(memory, fake_vectors_client)
        uuid_id = "550e8400-e29b-41d4-a716-446655440000"
        t.add(role="user", content="hi", vector=[0.1], id=uuid_id, ts=5.0)
        written = fake_vectors_client.upsert.call_args.args[1][0]
        fake_vectors_client.scroll.return_value = {
            "next_page_offset": None,
            "points": [written],
        }
        history = t.history(limit=10)
        assert history[0].id == uuid_id
        assert isinstance(history[0].id, str)

    def test_thread_history_round_trips_default_uuid_id_as_str(
        self, memory, fake_vectors_client
    ):
        t = self._open_thread(memory, fake_vectors_client)
        pid = t.add(role="user", content="hi", vector=[0.1], ts=5.0)
        assert isinstance(pid, str)
        written = fake_vectors_client.upsert.call_args.args[1][0]
        fake_vectors_client.scroll.return_value = {
            "next_page_offset": None,
            "points": [written],
        }
        history = t.history(limit=10)
        assert history[0].id == pid

    def test_thread_history_respects_limit(self, memory, fake_vectors_client):
        t = self._open_thread(memory, fake_vectors_client)
        fake_vectors_client.scroll.return_value = {
            "next_page_offset": None,
            "points": [
                {
                    "id": f"00000000-0000-4000-8000-{i:012d}",
                    "payload": {"role": "u", "content": str(i), "ts": float(i)},
                }
                for i in range(10)
            ],
        }
        history = t.history(limit=3)
        assert len(history) == 3
        # First 3 in ascending ts order
        assert [m.content for m in history] == ["0", "1", "2"]

    def test_thread_history_rejects_bad_order(self, memory, fake_vectors_client):
        t = self._open_thread(memory, fake_vectors_client)
        with pytest.raises(ValueError):
            t.history(order="sideways")

    def test_thread_history_rejects_nonpositive_limit(
        self, memory, fake_vectors_client
    ):
        t = self._open_thread(memory, fake_vectors_client)
        with pytest.raises(ValueError):
            t.history(limit=0)

    def test_thread_clear_drops_prefixed_collection(self, memory, fake_vectors_client):
        t = self._open_thread(memory, fake_vectors_client)
        t.clear()
        fake_vectors_client.delete_collection.assert_called_once_with(
            "__thread__conv-99"
        )


# =============================================================================
# Analytics parity
# =============================================================================


class TestAnalyticsParity:
    def test_global_performance_analytics(self, memory, fake_vectors_client):
        memory.get_performance_analytics(time_range="7d", region="us-east-1")
        fake_vectors_client.get_performance_analytics.assert_called_once_with(
            time_range="7d", region="us-east-1"
        )

    def test_namespace_analytics_requires_existence(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = False
        with pytest.raises(NamespaceNotFoundError):
            memory.get_namespace_analytics("nope")

    def test_namespace_analytics_delegates(self, memory, fake_vectors_client):
        fake_vectors_client.collection_exists.return_value = True
        memory.get_namespace_analytics("customer-42", time_range="1h")
        fake_vectors_client.get_collection_analytics.assert_called_once_with(
            "customer-42", time_range="1h"
        )

    def test_thread_analytics_uses_prefixed_collection(
        self, memory, fake_vectors_client
    ):
        fake_vectors_client.collection_exists.return_value = True
        memory.get_thread_analytics("conv-99")
        fake_vectors_client.get_collection_analytics.assert_called_once_with(
            "__thread__conv-99", time_range="24h"
        )

    def test_usage_stats(self, memory, fake_vectors_client):
        memory.get_usage_stats()
        fake_vectors_client.get_usage_stats.assert_called_once()

    def test_client_level_clear_schema_cache_wipes_all(
        self, memory, fake_vectors_client
    ):
        memory.clear_schema_cache()
        fake_vectors_client.clear_schema_cache.assert_called_once_with(None)


# =============================================================================
# Lifecycle
# =============================================================================


class TestClientLifecycle:
    def test_close_delegates(self, memory, fake_vectors_client):
        memory.close()
        fake_vectors_client.close.assert_called_once()

    def test_context_manager_closes(self, fake_vectors_client):
        with MemoryClient(client=fake_vectors_client) as m:
            assert m.vectors is fake_vectors_client
        fake_vectors_client.close.assert_called_once()
