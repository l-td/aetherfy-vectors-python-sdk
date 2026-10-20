"""Collection performance modes: create_collection(performance_mode=...),
update_collection, Collection.performance_mode, and the fake store the memory
layer's tests share.

The wire contract is vectordb's (backend/docs/PERFORMANCE_MODES.md): a mode is
sent only when the caller names one, so a caller who never does sends the body
this SDK always sent; the server answers 400 for an unknown mode, 422
PERFORMANCE_MODE_UNSUITABLE for compact below 512 dimensions, 409
PERFORMANCE_MODE_CHANGE_TOO_SOON inside the cooldown.
"""

from unittest.mock import Mock, patch

import pytest

from aetherfy_vectors import AetherfyVectorsClient, PerformanceMode
from aetherfy_vectors.exceptions import (
    ConflictError,
    ServiceUnavailableError,
    ValidationError,
)
from aetherfy_vectors.models import Collection, DistanceMetric, VectorConfig
from aetherfy_vectors.performance_modes import (
    DEFAULT_PERFORMANCE_MODE,
    coerce_performance_mode,
    parse_performance_mode,
)
from aetherfy_vectors.retry_policy import OPERATIONS, REPEATABLE
from tests.fake_vectors_store import FakeVectorsClient

CONFIG = VectorConfig(size=768, distance=DistanceMetric.COSINE)
LIST_PATH = "/api/v1/workspaces/w/collections"
ITEM_PATH = "/api/v1/workspaces/w/collections/docs"


def _response(status, payload):
    r = Mock()
    r.status_code = status
    r.json.return_value = payload
    r.content = True
    return r


@pytest.fixture(autouse=True)
def _no_backoff():
    with patch("time.sleep"):
        yield


@pytest.fixture
def client():
    c = AetherfyVectorsClient(
        api_key="afy_test_1234567890abcdef1234",
        endpoint="https://test-api.aetherfy.com",
        workspace="w",
    )
    c.session.request = Mock()
    return c


def queue(client, *outcomes):
    client.session.request.reset_mock()
    client.session.request.side_effect = list(outcomes)
    return client.session.request


def sent_body(request):
    return request.call_args.kwargs["json"]


def sent_request(request):
    return request.call_args.kwargs["method"], request.call_args.kwargs["url"]


class TestEnum:
    def test_the_four_modes_are_their_wire_names(self):
        assert [m.value for m in PerformanceMode] == [
            "exact",
            "balanced",
            "fast",
            "compact",
        ]
        assert DEFAULT_PERFORMANCE_MODE is PerformanceMode.EXACT

    def test_a_mode_compares_equal_to_its_name(self):
        assert PerformanceMode.BALANCED == "balanced"

    def test_coerce_accepts_the_enum_and_the_name(self):
        assert coerce_performance_mode("fast") is PerformanceMode.FAST
        assert coerce_performance_mode(PerformanceMode.FAST) is PerformanceMode.FAST

    @pytest.mark.parametrize("bad", ["turbo", "Exact", "", None, 3, {}])
    def test_coerce_names_the_accepted_values(self, bad):
        with pytest.raises(ValueError, match="exact, balanced, fast, compact"):
            coerce_performance_mode(bad)

    def test_parse_keeps_an_unknown_name_instead_of_raising(self):
        assert parse_performance_mode("balanced") is PerformanceMode.BALANCED
        assert parse_performance_mode("turbo4") == "turbo4"
        assert parse_performance_mode(None) is None
        assert parse_performance_mode(7) is None


class TestCreate:
    def test_omitting_the_mode_sends_the_body_this_sdk_always_sent(self, client):
        request = queue(client, _response(200, {"regions": ["us-east-1"]}))
        client.create_collection("docs", CONFIG)
        assert sent_request(request) == ("POST", "https://test-api.aetherfy.com" + LIST_PATH)
        assert sent_body(request) == {
            "name": "docs",
            "vectors": {"size": 768, "distance": "Cosine"},
            "description": None,
        }

    @pytest.mark.parametrize(
        "mode", [PerformanceMode.BALANCED, "balanced"], ids=["enum", "string"]
    )
    def test_a_named_mode_is_sent_as_its_wire_name(self, client, mode):
        request = queue(client, _response(200, {"performance_mode": "balanced"}))
        client.create_collection("docs", CONFIG, performance_mode=mode)
        assert sent_body(request)["performance_mode"] == "balanced"

    def test_exact_named_explicitly_is_still_sent(self, client):
        # Omitting and naming the default are different requests on the wire;
        # the SDK does not decide they mean the same.
        request = queue(client, _response(200, {"performance_mode": "exact"}))
        client.create_collection("docs", CONFIG, performance_mode="exact")
        assert sent_body(request)["performance_mode"] == "exact"

    def test_an_unknown_mode_raises_before_any_request(self, client):
        request = queue(client)
        with pytest.raises(ValueError, match="exact, balanced, fast, compact"):
            client.create_collection("docs", CONFIG, performance_mode="turbo")
        assert request.call_count == 0

    def test_the_returned_collection_carries_the_mode_the_server_stored(self, client):
        # An idempotent re-create answers with the STORED mode, not the asked one.
        queue(client, _response(200, {"performance_mode": "fast"}))
        col = client.create_collection("docs", CONFIG, performance_mode="balanced")
        assert col.performance_mode is PerformanceMode.FAST

    def test_an_older_server_that_reports_no_mode_leaves_it_none(self, client):
        queue(client, _response(200, {"regions": ["us-east-1"]}))
        assert client.create_collection("docs", CONFIG).performance_mode is None

    def test_a_422_for_compact_below_the_floor_is_a_validation_error(self, client):
        queue(
            client,
            _response(
                422,
                {
                    "error": {
                        "code": "PERFORMANCE_MODE_UNSUITABLE",
                        "message": "performance_mode 'compact' needs vectors of at least 512 dimensions",
                        "field": "performance_mode",
                        "min_vector_size": 512,
                        "suggested_mode": "balanced",
                    }
                },
            ),
        )
        with pytest.raises(ValidationError) as exc:
            client.create_collection(
                "docs",
                VectorConfig(size=384, distance=DistanceMetric.COSINE),
                performance_mode="compact",
            )
        assert exc.value.status_code == 422
        assert exc.value.error_code == "PERFORMANCE_MODE_UNSUITABLE"
        assert exc.value.details["suggested_mode"] == "balanced"

    def test_the_engine_setting_the_mode_replaces_is_still_refused(self, client):
        request = queue(client)
        with pytest.raises(TypeError, match="performance_mode"):
            client.create_collection(
                "docs", CONFIG, quantization_config={"scalar": {"type": "int8"}}
            )
        assert request.call_count == 0


class TestRead:
    def test_get_collection_reads_the_top_level_mode(self, client):
        queue(
            client,
            _response(
                200,
                {
                    "name": "docs",
                    "config": {"params": {"vectors": {"size": 768, "distance": "Cosine"}}},
                    "performance_mode": "compact",
                },
            ),
        )
        assert client.get_collection("docs").performance_mode is PerformanceMode.COMPACT

    def test_get_collections_reads_each_collections_mode(self, client):
        def row(name, mode):
            return {
                "name": name,
                "config": {"params": {"vectors": {"size": 768, "distance": "Cosine"}}},
                "performance_mode": mode,
            }

        queue(client, _response(200, {"collections": [row("a", "exact"), row("b", "turbo4")]}))
        a, b = client.get_collections()
        assert a.performance_mode is PerformanceMode.EXACT
        assert b.performance_mode == "turbo4"  # a mode this SDK has not heard of

    def test_from_dict_without_a_mode_is_none(self):
        col = Collection.from_dict(
            {"name": "docs", "config": {"params": {"vectors": {"size": 3, "distance": "Dot"}}}}
        )
        assert col.performance_mode is None


class TestUpdate:
    def test_a_mode_change_is_a_patch_of_the_collection(self, client):
        request = queue(client, _response(200, {"performance_mode": "fast"}))
        assert client.update_collection("docs", performance_mode=PerformanceMode.FAST) is True
        assert sent_request(request) == ("PATCH", "https://test-api.aetherfy.com" + ITEM_PATH)
        assert sent_body(request) == {"performance_mode": "fast"}

    def test_a_description_alone_sends_only_the_description(self, client):
        request = queue(client, _response(200, {}))
        client.update_collection("docs", description="notes")
        assert sent_body(request) == {"description": "notes"}

    def test_none_clears_the_description_and_is_not_the_same_as_omitting_it(self, client):
        request = queue(client, _response(200, {}))
        client.update_collection("docs", description=None)
        assert sent_body(request) == {"description": None}

    def test_both_go_in_one_request(self, client):
        request = queue(client, _response(200, {}))
        client.update_collection("docs", description="d", performance_mode="balanced")
        assert request.call_count == 1
        assert sent_body(request) == {"description": "d", "performance_mode": "balanced"}

    def test_nothing_to_change_is_an_error_before_any_request(self, client):
        request = queue(client)
        with pytest.raises(ValueError, match="nothing to change"):
            client.update_collection("docs")
        assert request.call_count == 0

    def test_an_unknown_mode_raises_before_any_request(self, client):
        request = queue(client)
        with pytest.raises(ValueError, match="exact, balanced, fast, compact"):
            client.update_collection("docs", performance_mode="turbo")
        assert request.call_count == 0

    def test_the_cooldown_is_a_non_retried_conflict_carrying_its_code_and_wait(self, client):
        request = queue(
            client,
            _response(
                409,
                {
                    "error": {
                        "code": "PERFORMANCE_MODE_CHANGE_TOO_SOON",
                        "message": "It can change again in 7 minutes.",
                        "field": "performance_mode",
                        "retry_after_seconds": 437,
                        "cooldown_seconds": 900,
                    }
                },
            ),
        )
        with pytest.raises(ConflictError) as exc:
            client.update_collection("docs", performance_mode="fast")
        assert request.call_count == 1
        assert exc.value.status_code == 409
        assert exc.value.error_code == "PERFORMANCE_MODE_CHANGE_TOO_SOON"
        assert exc.value.details["retry_after_seconds"] == 437

    def test_an_ambiguous_failure_is_retried_because_a_repeat_is_a_no_op(self, client):
        assert OPERATIONS["update_collection"] == REPEATABLE
        request = queue(
            client,
            _response(502, {"error": {"code": "BAD_GATEWAY", "message": "x"}}),
            _response(200, {"performance_mode": "fast"}),
        )
        assert client.update_collection("docs", performance_mode="fast") is True
        assert request.call_count == 2

    def test_a_persistent_503_raises_after_the_retries(self, client):
        queue(
            client,
            *[_response(503, {"error": {"code": "X", "message": "x"}})] * 4,
        )
        with pytest.raises(ServiceUnavailableError):
            client.update_collection("docs", performance_mode="fast")

    @pytest.mark.parametrize(
        "name",
        [
            "optimizers_config",
            "collection_params",
            "vectors_config",
            "hnsw_config",
            "quantization_config",
            "sparse_vectors_config",
            "strict_mode_config",
        ],
    )
    def test_every_engine_setting_is_refused_pointing_at_the_mode(self, client, name):
        request = queue(client)
        with pytest.raises(TypeError, match="performance_mode"):
            client.update_collection("docs", performance_mode="fast", **{name: {"x": 1}})
        assert request.call_count == 0

    def test_a_change_does_not_touch_the_schema_cache(self, client):
        # The cache holds size and distance, which a mode never changes.
        client._schema_cache["w/docs"] = {"size": 768, "distance": "Cosine", "etag": "e"}
        queue(client, _response(200, {}))
        client.update_collection("docs", performance_mode="fast")
        assert client._schema_cache["w/docs"]["etag"] == "e"


class TestFakeStore:
    """The double the memory layer's tests use behaves as the server does."""

    def test_a_collection_starts_exact_and_reads_back_its_mode(self):
        fake = FakeVectorsClient()
        fake.create_collection("c", CONFIG)
        assert fake.get_collection("c").performance_mode is PerformanceMode.EXACT

    def test_create_honours_a_named_mode(self):
        fake = FakeVectorsClient()
        col = fake.create_collection("c", CONFIG, performance_mode="balanced")
        assert col.performance_mode is PerformanceMode.BALANCED

    def test_an_unknown_mode_is_a_400_validation_error(self):
        fake = FakeVectorsClient()
        with pytest.raises(ValidationError) as exc:
            fake.create_collection("c", CONFIG, performance_mode="turbo")
        assert exc.value.status_code == 400

    def test_compact_below_the_floor_is_a_422(self):
        fake = FakeVectorsClient()
        small = VectorConfig(size=384, distance=DistanceMetric.COSINE)
        with pytest.raises(ValidationError) as exc:
            fake.create_collection("c", small, performance_mode="compact")
        assert exc.value.error_code == "PERFORMANCE_MODE_UNSUITABLE"
        fake.create_collection("ok", CONFIG, performance_mode="compact")

    def test_the_cooldown_blocks_a_second_change_until_it_has_passed(self):
        fake = FakeVectorsClient()
        fake.create_collection("c", CONFIG)
        assert fake.update_collection("c", performance_mode="fast") is True
        fake.now = 100
        with pytest.raises(ConflictError) as exc:
            fake.update_collection("c", performance_mode="balanced")
        assert exc.value.error_code == "PERFORMANCE_MODE_CHANGE_TOO_SOON"
        assert exc.value.details["retry_after_seconds"] == 800
        assert fake.get_collection("c").performance_mode is PerformanceMode.FAST
        fake.now = 900
        fake.update_collection("c", performance_mode="balanced")
        assert fake.get_collection("c").performance_mode is PerformanceMode.BALANCED

    def test_asking_for_the_current_mode_is_a_no_op_that_starts_no_cooldown(self):
        fake = FakeVectorsClient()
        fake.create_collection("c", CONFIG)
        fake.update_collection("c", performance_mode="exact")
        fake.update_collection("c", performance_mode="fast")  # not blocked
