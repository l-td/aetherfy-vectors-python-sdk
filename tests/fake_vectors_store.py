"""An in-memory stand-in for AetherfyVectorsClient that really applies filters.

Why this exists: threads are now rows in one shared collection, separated from
each other by a payload filter and nothing else. A MagicMock can prove which
filter the memory layer SENT, but not that the filter actually isolates one
thread from the next — and "clear() must not destroy a sibling thread" is a
claim about the second thing. This double stores points and evaluates
must / must_not / should the way the engine does, so a test can assert on
surviving DATA rather than on recorded calls.

Deliberately STRICTER than nothing and LOOSER than Qdrant: it supports exactly
the condition shapes the documented Aetherfy filter vocabulary has (match and
range, under the three clause arrays). An unknown condition shape raises here
rather than silently matching everything — the fail-open behaviour of the real
proxy is the hazard being defended against, so a stand-in that reproduced it
would hide the defect instead of catching it.
"""

from typing import Any, Dict, List, Optional, Union

from aetherfy_vectors.models import Collection, DistanceMetric, VectorConfig


def _lookup(payload: Dict[str, Any], key: str) -> Any:
    cur: Any = payload
    for part in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _match_condition(payload: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    if "key" not in cond:
        raise AssertionError(f"unsupported filter condition: {cond!r}")
    value = _lookup(payload, cond["key"])
    if "match" in cond:
        return value == cond["match"]["value"]
    if "range" in cond:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        r = cond["range"]
        if "gt" in r and not value > r["gt"]:
            return False
        if "gte" in r and not value >= r["gte"]:
            return False
        if "lt" in r and not value < r["lt"]:
            return False
        if "lte" in r and not value <= r["lte"]:
            return False
        return True
    raise AssertionError(f"unsupported filter condition: {cond!r}")


def matches(
    payload: Dict[str, Any],
    flt: Optional[Dict[str, Any]],
    *,
    fail_open_on_must_not: bool = False,
) -> bool:
    """Evaluate a filter the way Aetherfy documents it: the three clause
    arrays compose as a conjunction — everything in ``must`` holds AND at
    least one ``should`` holds AND nothing in ``must_not`` holds.

    ``fail_open_on_must_not`` reproduces the ONE failure mode the docs
    promise: the proxy forwards a filter verbatim and never validates it, so
    a mistyped clause "is passed along and quietly does nothing — a
    successful response with unfiltered results rather than a 400". Set it to
    exercise the client-side guards that exist precisely because the filter
    can fail open.
    """
    if not flt:
        return True
    unknown = set(flt) - {"must", "must_not", "should"}
    if unknown:
        raise AssertionError(f"unknown filter clause(s): {sorted(unknown)}")
    if not all(_match_condition(payload, c) for c in flt.get("must") or []):
        return False
    if not fail_open_on_must_not and any(
        _match_condition(payload, c) for c in flt.get("must_not") or []
    ):
        return False
    should = flt.get("should") or []
    if should and not any(_match_condition(payload, c) for c in should):
        return False
    return True


class FakeVectorsClient:
    """Enough of AetherfyVectorsClient for the memory layer to run for real."""

    def __init__(self, workspace: Optional[str] = "my-bot"):
        self.workspace = workspace
        self.collections: Dict[str, Dict[str, Any]] = {}
        self.indexes: List[tuple] = []
        # Flip to make every read behave as if the must_not clause had been
        # mistyped: the documented fail-open. See `matches`.
        self.fail_open_on_must_not = False

    # -- collections --------------------------------------------------------

    def create_collection(self, name: str, config: VectorConfig, **kwargs):
        assert name not in self.collections, f"{name} already exists"
        self.collections[name] = {"config": config, "points": {}}
        return Collection(name=name, config=config)

    def collection_exists(self, name: str, **kwargs) -> bool:
        return name in self.collections

    def get_collection(self, name: str, **kwargs) -> Collection:
        col = self.collections[name]
        return Collection(
            name=name,
            config=col["config"],
            points_count=len(col["points"]),
            status="green",
        )

    def get_collections(self, **kwargs) -> List[Collection]:
        return [self.get_collection(n) for n in self.collections]

    def delete_collection(self, name: str, **kwargs) -> bool:
        self.collections.pop(name, None)
        return True

    def create_field_index(self, name, field_name, field_schema="keyword") -> bool:
        self.indexes.append((name, field_name, field_schema))
        return True

    def delete_field_index(self, name, field_name) -> bool:
        self.indexes = [i for i in self.indexes if i[:2] != (name, field_name)]
        return True

    # -- points -------------------------------------------------------------

    def _points(self, name: str) -> Dict[Any, Dict[str, Any]]:
        return self.collections[name]["points"]

    def upsert(self, name: str, points: List[Dict[str, Any]], **kwargs) -> bool:
        size = self.collections[name]["config"].size
        for p in points:
            assert len(p["vector"]) == size, (
                f"vector of {len(p['vector'])} dims into a {size}-dim collection"
            )
            self._points(name)[p["id"]] = {
                "id": p["id"],
                "vector": list(p["vector"]),
                "payload": dict(p.get("payload") or {}),
            }
        return True

    def delete(self, name: str, points_selector, **kwargs) -> bool:
        store = self._points(name)
        if isinstance(points_selector, list):
            for pid in points_selector:
                store.pop(pid, None)
            return True
        doomed = [
            pid for pid, p in store.items() if matches(p["payload"], points_selector)
        ]
        for pid in doomed:
            store.pop(pid)
        return True

    def _project(self, p, with_payload, with_vectors):
        out: Dict[str, Any] = {"id": p["id"]}
        if with_payload:
            out["payload"] = dict(p["payload"])
        if with_vectors:
            out["vector"] = list(p["vector"])
        return out

    def _selected(self, name, flt):
        return [
            p
            for p in self._points(name).values()
            if matches(
                p["payload"],
                flt,
                fail_open_on_must_not=self.fail_open_on_must_not,
            )
        ]

    def scroll(
        self,
        name: str,
        limit: int = 10,
        offset=None,
        scroll_filter=None,
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        sel = self._selected(name, scroll_filter)
        start = offset or 0
        page = sel[start : start + limit]
        nxt = start + limit if start + limit < len(sel) else None
        return {
            "points": [self._project(p, with_payload, with_vectors) for p in page],
            "next_page_offset": nxt,
        }

    def scroll_iter(
        self,
        name: str,
        *,
        batch_size: int = 256,
        scroll_filter=None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ):
        for p in self._selected(name, scroll_filter):
            yield self._project(p, with_payload, with_vectors)

    def count(self, name: str, count_filter=None, exact: bool = True, **kwargs) -> int:
        return len(self._selected(name, count_filter))

    def retrieve(
        self,
        name: str,
        ids: List[Union[str, int]],
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        store = self._points(name)
        return [
            self._project(store[i], with_payload, with_vectors)
            for i in ids
            if i in store
        ]

    def search(
        self,
        name: str,
        query_vector: List[float],
        limit: int = 10,
        offset: int = 0,
        query_filter=None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold=None,
        search_params=None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        sel = self._selected(name, query_filter)
        scored = sorted(
            sel,
            key=lambda p: -sum(a * b for a, b in zip(p["vector"], query_vector)),
        )
        out = []
        for p in scored[offset : offset + limit]:
            hit = self._project(p, with_payload, with_vectors)
            hit["score"] = sum(a * b for a, b in zip(p["vector"], query_vector))
            out.append(hit)
        return out

    # -- schema / misc ------------------------------------------------------

    def clear_schema_cache(self, name=None) -> None:
        return None

    def close(self) -> None:
        return None


def unit_vector(size: int, axis: int = 0) -> List[float]:
    v = [0.0] * size
    v[axis] = 1.0
    return v


COSINE = DistanceMetric.COSINE
