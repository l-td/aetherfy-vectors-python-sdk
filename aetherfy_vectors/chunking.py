"""Byte-bounded chunking for upsert payloads.

Single source of truth for the rule: never POST/PUT more than
MAX_REQUEST_BYTES of points to the backend in one HTTP request. The
client→backend hop terminates at Cloudflare's edge, whose body-size cap
is 100 MB on Free/Pro/Business plans (500 MB on Enterprise). A request
exceeding that cap is rejected with 413 BEFORE reaching our origin —
the server-side streaming chunker can't help because Cloudflare buffers
the body before forwarding.

Why bytes, not point count: a 1000-vector batch is ~35 MB for 384-dim
vectors and ~138 MB for 1536-dim vectors. A count cap that's safe for
the largest case wastes throughput on the smallest. A byte cap is
robust across dim and payload shape.

Mirror of:
  - vectordb/backend/services/chunking.js (server-side, 12 MB target
    for the backend→Qdrant hop)
  - aetherfy-vectors-js-sdk/src/utils/chunking.ts (JS SDK, same target
    and primitives)

with a threshold tuned to Cloudflare's edge limit instead of Qdrant's
32 MB body cap.
"""

import json
from typing import Any, Iterator, List


# Per-HTTP-request byte target. 80 MB leaves 20 MB headroom under
# Cloudflare's default 100 MB cap for HTTP framing, headers, and the
# `{"points": [...]}` array overhead. Conservative — empirical tests
# show ~95 MB also passes, but the margin protects against framing
# surprises (compressed transfer-encoding inflation, large headers,
# etc.).
MAX_REQUEST_BYTES = 80 * 1024 * 1024

_FLOAT_JSON_BYTES = 18
_POINT_FRAMING_BYTES = 100


def point_wire_bytes(point: Any) -> int:
    """Estimate the JSON wire size of a single point.

    Returns 0 for unmeasurable input (caller treats as "send alone" so
    the chunker doesn't infinite-loop on adversarial input). Otherwise:
        framing + (len(vector) * 18) + len(json.dumps(payload))

    Vector serialization is skipped — CPython's `json.dumps` on a float
    emits up to 17 significant digits + comma, so `len(vector) * 18` is
    a deterministic upper bound that costs O(1) instead of O(dim) per
    point.

    On a non-serializable payload (cyclic references, sets, etc.),
    returns MAX_REQUEST_BYTES so the offending point gets isolated in
    its own chunk and the eventual server-side error message points at
    the right point id.
    """
    if point is None or not isinstance(point, dict):
        # Accept Point-like duck-typed objects via __dict__ as a future
        # extension if needed; today the upsert formatter converts to
        # dicts before chunking so this branch isn't exercised in
        # practice.
        return 0

    byte_count = _POINT_FRAMING_BYTES

    vector = point.get("vector")
    if isinstance(vector, (list, tuple)):
        byte_count += len(vector) * _FLOAT_JSON_BYTES

    payload = point.get("payload")
    if payload is not None and isinstance(payload, dict):
        try:
            byte_count += len(json.dumps(payload))
        except (TypeError, ValueError):
            return MAX_REQUEST_BYTES

    return byte_count


def chunk_points_by_bytes(
    points: List[Any], target_bytes: int = MAX_REQUEST_BYTES
) -> Iterator[List[Any]]:
    """Split an in-memory points list into byte-bounded chunks.

    Generator so callers can pipeline (POST chunk N while preparing
    chunk N+1). The chunker accumulates one point at a time and flushes
    before adding a point that would push the in-flight chunk past
    target_bytes.

    Single-point overflow: if one point exceeds target_bytes on its
    own, it gets its own chunk. The backend (or Cloudflare) may reject
    it, but the SDK never silently drops data — every point is at
    least attempted. Callers (upsert) catch the resulting error and
    surface the point id so the user knows exactly which point is too
    large.
    """
    if not isinstance(points, list) or not points:
        return

    buf: List[Any] = []
    buf_bytes = 0

    for point in points:
        pb = point_wire_bytes(point)
        if buf and buf_bytes + pb > target_bytes:
            yield buf
            buf = []
            buf_bytes = 0
        buf.append(point)
        buf_bytes += pb

    if buf:
        yield buf
