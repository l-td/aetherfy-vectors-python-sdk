"""
Tests for body-aware timeout scaling.

Verifies that the per-request timeout grows with payload size for write
methods (POST/PUT) so a single upsert chunk can complete on residential
WAN uplinks. Without this, the default 30 s aborted ~24 MB chunks on
slow links — the SDK then retried 3 times, all timed out, and the
chunk landed in PartialUpsertError.failed even though the origin would
have accepted it given enough time.

Mirrors aetherfy-vectors-js-sdk/src/http/client.ts's body-aware timeout
so both SDKs carry the same policy.
"""

import math

import pytest

from aetherfy_vectors import AetherfyVectorsClient


@pytest.fixture
def timed_client(api_key, test_endpoint, mock_requests):
    """A client at the standard 30 s base timeout for these computations."""
    return AetherfyVectorsClient(api_key=api_key, endpoint=test_endpoint, timeout=30.0)


class TestEstimateBodyBytes:
    """The fast estimator drives the timeout. Verify each branch."""

    def test_none_returns_zero(self, timed_client):
        assert timed_client._estimate_body_bytes(None) == 0

    def test_str_returns_utf8_byte_length(self, timed_client):
        # "café" = 5 bytes in UTF-8 (4-char string, é is 2 bytes).
        assert timed_client._estimate_body_bytes("café") == 5

    def test_bytes_returns_length(self, timed_client):
        assert timed_client._estimate_body_bytes(b"\x00\x01\x02\x03") == 4

    def test_upsert_dict_uses_point_wire_bytes_fast_path(self, timed_client):
        # 100 framing + 4 floats × 18 bytes = 172 per point, two points = 344.
        # Critically: this does NOT call json.dumps on the points array.
        body = {
            "points": [
                {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]},
                {"id": 2, "vector": [0.5, 0.6, 0.7, 0.8]},
            ]
        }
        assert timed_client._estimate_body_bytes(body) == 344

    def test_non_upsert_dict_falls_back_to_json_dumps_length(self, timed_client):
        body = {"filter": {"must": [{"key": "category", "match": {"value": "a"}}]}}
        # Whatever the exact serialization, it must be > 0 and roughly
        # match a compact JSON encoding.
        bytes_est = timed_client._estimate_body_bytes(body)
        import json

        assert bytes_est == len(json.dumps(body, separators=(",", ":")))

    def test_unserializable_returns_zero(self, timed_client):
        # Circular reference — json.dumps raises ValueError. The estimator
        # swallows it and returns 0 so the caller falls back to the base
        # timeout (better than crashing on weird input).
        a = {}
        a["self"] = a
        assert timed_client._estimate_body_bytes(a) == 0


class TestComputeBodyAwareTimeout:
    """Linear scaling: base for ≤ threshold, +1 s/MB beyond."""

    def test_none_body_returns_base_timeout(self, timed_client):
        assert timed_client._compute_body_aware_timeout(None) == 30.0

    def test_under_threshold_returns_base_timeout(self, timed_client):
        # 1 MB string — well under the 5 MB threshold.
        assert timed_client._compute_body_aware_timeout("x" * (1024 * 1024)) == 30.0

    def test_exactly_at_threshold_returns_base_timeout(self, timed_client):
        # Threshold = 5 MB exactly — boundary should NOT add overhead.
        body = "x" * (5 * 1024 * 1024)
        assert timed_client._compute_body_aware_timeout(body) == 30.0

    def test_one_byte_over_threshold_adds_one_mb_worth_of_overhead(self, timed_client):
        # 5 MB + 1 byte → ceil(1/1MB) = 1 → +1.0 s.
        body = "x" * (5 * 1024 * 1024 + 1)
        assert timed_client._compute_body_aware_timeout(body) == 31.0

    def test_scaling_grows_linearly_with_megabytes_over_threshold(self, timed_client):
        # 84 MB body (chunk 1 of the billing test failure):
        # over_threshold = 84 - 5 = 79 MB → 79 s extra → 109 s total.
        eighty_four_mb = 84 * 1024 * 1024
        body = "x" * eighty_four_mb
        assert timed_client._compute_body_aware_timeout(body) == 109.0

    def test_custom_base_timeout_scales_from_that_base(
        self, api_key, test_endpoint, mock_requests
    ):
        # A client with timeout=60.0 should add the same per-MB overhead
        # on top of its own base, not reset to the class default.
        client = AetherfyVectorsClient(
            api_key=api_key, endpoint=test_endpoint, timeout=60.0
        )
        body = "x" * (10 * 1024 * 1024)  # 5 MB over threshold → +5 s
        assert client._compute_body_aware_timeout(body) == 65.0


class TestTimeoutWiringIntoRequests:
    """End-to-end: the computed timeout reaches session.request()."""

    def _last_request_timeout(self, mock_requests):
        """Pull the timeout kwarg from the most recent mocked HTTP call."""
        # mock_requests.request is the wrapped function our patched
        # Session uses (see conftest.py); its last call's kwargs hold
        # the timeout the client passed.
        return mock_requests.request.call_args[1]["timeout"]

    def test_get_uses_base_timeout(
        self, client, mock_requests, mock_successful_response
    ):
        # GETs don't have bodies → always base timeout regardless of size.
        mock_requests.request.return_value = mock_successful_response(
            {
                "result": {"collections": []},
            }
        )
        client.get_collections()
        assert self._last_request_timeout(mock_requests) == client.timeout

    def test_small_post_uses_base_timeout(
        self, client, mock_requests, mock_successful_response
    ):
        # A tiny create-collection POST stays at base timeout.
        mock_requests.request.return_value = mock_successful_response({"result": True})
        from aetherfy_vectors.models import VectorConfig, DistanceMetric

        client.create_collection(
            "c", VectorConfig(size=4, distance=DistanceMetric.COSINE)
        )
        assert self._last_request_timeout(mock_requests) == client.timeout

    def test_large_upsert_put_scales_timeout(
        self, client, mock_requests, mock_successful_response
    ):
        # Build a points array whose estimated wire bytes exceed the 5 MB
        # threshold so we can observe the +1 s/MB scaling on a real upsert
        # call. Upsert hits THREE endpoints (matching the cached schema
        # flow tested elsewhere in test_client.py): GET collection config,
        # GET payload schema (404), PUT points.
        from aetherfy_vectors.exceptions import AetherfyVectorsException

        collection_config = mock_successful_response(
            {
                "result": {
                    "config": {
                        "params": {"vectors": {"size": 384, "distance": "Cosine"}}
                    }
                },
                "schema_version": "v1",
            }
        )
        schema_404 = AetherfyVectorsException("Schema not found", status_code=404)
        upsert_ok = mock_successful_response({"result": {"operation_id": 1}})
        mock_requests.request.side_effect = [collection_config, schema_404, upsert_ok]

        # Per-point estimate via chunking.point_wire_bytes:
        # 100 framing + 384 floats × 18 = 6 992 bytes.
        # 1000 points ≈ 6.66 MB → over_threshold ≈ 1.66 MB → ceil = 2 →
        # base + 2 s = 12.0 s on this client (which uses timeout=10.0).
        big_batch = [{"id": i, "vector": [0.0] * 384} for i in range(1000)]
        client.upsert("c", big_batch)

        # The PUT (third call) is the one whose timeout matters — the
        # two GETs are body-less and use base timeout.
        put_call = mock_requests.request.call_args_list[2]
        put_timeout = put_call.kwargs["timeout"]
        assert (
            put_timeout > client.timeout
        ), f"expected PUT timeout > base ({client.timeout}), got {put_timeout}"
