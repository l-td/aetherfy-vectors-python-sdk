"""Unit tests for chunking (byte-bounded chunk splitter for upsert
payloads) and the PartialUpsertError exception used by the multi-chunk
path in AetherfyVectorsClient.upsert.

The integration path (upsert dispatching multiple HTTP requests under
the byte cap) is covered by the e2e suite; here we pin the math and
the error shape.
"""

import json
import pytest

from aetherfy_vectors.chunking import (
    point_wire_bytes,
    chunk_points_by_bytes,
    MAX_REQUEST_BYTES,
)
from aetherfy_vectors.exceptions import (
    PartialUpsertError,
    AetherfyVectorsException,
    ValidationError,
)


class TestPointWireBytes:
    """point_wire_bytes returns a deterministic upper bound on the
    JSON wire size of a single point."""

    def test_returns_zero_for_unmeasurable_inputs(self):
        # Chunker treats 0 as "send alone" so it doesn't infinite-loop
        # on adversarial input.
        assert point_wire_bytes(None) == 0
        assert point_wire_bytes(42) == 0
        assert point_wire_bytes("not a dict") == 0
        assert point_wire_bytes([1, 2, 3]) == 0

    def test_estimates_vector_only_points_as_framing_plus_dim_x18(self):
        point = {"id": "p0", "vector": [0.5] * 100}
        b = point_wire_bytes(point)
        assert b > 100 * 18
        assert b < 100 * 18 + 200  # framing allowance

    def test_adds_payload_bytes_via_json_length(self):
        vec_only = {"id": "p0", "vector": [0.5] * 100}
        with_payload = {
            "id": "p0",
            "vector": [0.5] * 100,
            "payload": {"text": "hello world"},
        }
        assert point_wire_bytes(with_payload) > point_wire_bytes(vec_only)

    def test_returns_max_for_non_serializable_payload(self):
        # A set isn't JSON-serializable in CPython's default json. The
        # chunker treats this as "send alone" by returning the max
        # threshold, isolating the offending point in its own chunk.
        point = {"id": "p0", "vector": [0.1], "payload": {"bad": {1, 2, 3}}}
        assert point_wire_bytes(point) == MAX_REQUEST_BYTES

    def test_handles_non_list_vector_gracefully(self):
        # Defensive: a malformed point with vector as dict shouldn't
        # crash the chunker. (Upsert's own validation would reject
        # such a point before chunking in practice.)
        point = {"id": "p0", "vector": {"not": "a list"}}
        assert point_wire_bytes(point) >= 0


class TestChunkPointsByBytes:
    """chunk_points_by_bytes is a generator that yields byte-bounded
    chunks of an input list, preserving order."""

    def test_yields_nothing_for_empty_or_invalid_input(self):
        assert list(chunk_points_by_bytes([], 100)) == []
        assert list(chunk_points_by_bytes(None, 100)) == []  # type: ignore

    def test_keeps_small_batch_in_one_chunk(self):
        points = [{"id": f"p{i}", "vector": [0.1, 0.2]} for i in range(10)]
        chunks = list(chunk_points_by_bytes(points, MAX_REQUEST_BYTES))
        assert len(chunks) == 1
        assert len(chunks[0]) == 10

    def test_splits_when_in_flight_chunk_would_exceed_target(self):
        # 10 points × 100-dim vector ≈ 1900 bytes each (100 framing +
        # 1800 vector). With a 5000-byte target, ~2 points per chunk.
        points = [{"id": f"p{i}", "vector": [0.5] * 100} for i in range(10)]
        chunks = list(chunk_points_by_bytes(points, 5000))
        assert len(chunks) > 1
        # No points dropped — total across all chunks equals input.
        total_forwarded = sum(len(c) for c in chunks)
        assert total_forwarded == 10

    def test_preserves_input_order(self):
        # FIFO MessageGroupId on the server depends on per-chunk
        # delivery order being the same as the caller's input order.
        points = [{"id": f"p{i}", "vector": [0.5] * 200} for i in range(50)]
        chunks = list(chunk_points_by_bytes(points, 5000))
        flattened_ids = [p["id"] for c in chunks for p in c]
        assert flattened_ids == [p["id"] for p in points]

    def test_sends_single_oversized_point_alone_no_silent_drop(self):
        # One point with a huge embedded payload that itself exceeds
        # the target. The chunker must yield it in its own chunk
        # rather than refuse to emit it. The backend (or Cloudflare)
        # may then reject it, but the SDK NEVER drops user data
        # without telling them.
        tiny = {"id": "tiny", "vector": [0.1]}
        huge = {"id": "huge", "vector": [0.1], "payload": {"text": "x" * 50000}}
        more_tiny = {"id": "moreTiny", "vector": [0.2]}
        chunks = list(chunk_points_by_bytes([tiny, huge, more_tiny], 10000))
        # Find the chunk containing the huge point — it should be alone.
        huge_chunk = next(c for c in chunks if any(p["id"] == "huge" for p in c))
        assert len(huge_chunk) == 1
        assert huge_chunk[0]["id"] == "huge"

    def test_heterogeneous_payloads_chunked_correctly(self):
        # First-point-only measurement would underestimate and overflow.
        # Per-point byte tracking flushes BEFORE adding a point that
        # would push the in-flight chunk past target.
        small = {"id": "s0", "vector": [0.1], "payload": {"tag": "x"}}
        large_text = "a" * 20000
        points = [small] + [
            {"id": f"l{i}", "vector": [0.1], "payload": {"text": large_text}}
            for i in range(10)
        ]
        chunks = list(chunk_points_by_bytes(points, 50000))

        assert len(chunks) > 1
        # All points accounted for.
        assert sum(len(c) for c in chunks) == len(points)
        # No chunk far exceeds the target (only single-oversized-point
        # case can; none here do).
        for chunk in chunks:
            chunk_bytes = sum(point_wire_bytes(p) for p in chunk)
            # 2× target as a sanity ceiling — anything beyond indicates
            # the splitter broke.
            assert chunk_bytes < 50000 * 2

    def test_max_request_bytes_is_24mb(self):
        # Binding constraint is backend processing time, not the edge body
        # cap: the server re-chunks each request into ~12 MB Qdrant
        # sub-batches committed serially with wait=true inside a 90 s
        # timeout, so 24 MB (~2 sub-batches) stays well under budget.
        # See chunking.py for the full rationale.
        assert MAX_REQUEST_BYTES == 24 * 1024 * 1024


class TestPartialUpsertError:
    """PartialUpsertError reports saved/total counts and failed chunk
    details so callers can retry just the failed point IDs."""

    def test_reports_saved_total_and_failed_chunks(self):
        failed = [
            {
                "point_ids": ["p4", "p5", "p6"],
                "error": ValidationError("bad chunk"),
            }
        ]
        err = PartialUpsertError(3, 6, failed)

        assert isinstance(err, AetherfyVectorsException)
        assert err.saved == 3
        assert err.total == 6
        assert err.failed == failed
        assert "3 of 6 points saved" in err.message
        assert "3 failed" in err.message
        assert "1 chunk" in err.message

    def test_aggregates_failed_count_across_multiple_failed_chunks(self):
        failed = [
            {"point_ids": ["a", "b"], "error": ValidationError("chunk 1")},
            {"point_ids": ["c", "d", "e"], "error": ValidationError("chunk 2")},
        ]
        err = PartialUpsertError(0, 5, failed)
        assert "0 of 5 points saved" in err.message
        assert "5 failed across 2 chunk(s)" in err.message

    def test_is_an_aetherfy_vectors_exception_subclass(self):
        # Callers catching AetherfyVectorsException should catch
        # PartialUpsertError too — keeps the broad-catch path working
        # for users who don't need partial-success granularity.
        err = PartialUpsertError(
            0, 1, [{"point_ids": ["x"], "error": ValidationError("e")}]
        )
        assert isinstance(err, PartialUpsertError)
        assert isinstance(err, AetherfyVectorsException)
        assert isinstance(err, Exception)

    def test_failed_entries_carry_concrete_error_instances(self):
        # The failed list must contain real exception instances
        # (not stringified messages) so callers can isinstance-check
        # for specific error types and branch their retry logic.
        v_err = ValidationError("bad point")
        err = PartialUpsertError(
            0, 1, [{"point_ids": ["x"], "error": v_err}]
        )
        assert err.failed[0]["error"] is v_err
        assert isinstance(err.failed[0]["error"], ValidationError)
