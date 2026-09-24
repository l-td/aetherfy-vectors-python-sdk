"""The qdrant-client compatibility contract, per method and per argument.

The README promises that code written against qdrant-client keeps working once
the import and the constructor are swapped. That promise covers qdrant-client's
SIGNATURES, not arbitrary names, so a method here accepts exactly:

  * the parameters it names itself. Where one of those is also a qdrant-client
    parameter (``timeout``, ``scroll(order_by=...)``), it is HONOURED: it does
    what the caller asked for.
  * the qdrant-client parameters listed in ``QDRANT_COMPAT`` below, each
    classified by what ignoring it would do to the caller:

      IGNORE  our behaviour already gives the caller what they asked for, so
              the value is accepted and has no further effect.
      REFUSE  ignoring it would change what the caller gets. It raises
              TypeError naming the argument and why, unless the value passed
              is one that already describes our behaviour (qdrant's own
              default, for instance), which is accepted.

Any other keyword raises TypeError exactly as a plain Python signature would.
This is what stops a stale or misspelt argument being accepted and dropped: the
constructor's ``**kwargs`` used to swallow the pre-rename ``region=``, so a
caller asking for one region was silently routed to the default.

The constructor is NOT covered: the README's migration replaces qdrant's
constructor call wholesale, so no qdrant constructor argument is meant to reach
ours, and it takes no ``**kwargs`` at all.

The table is checked, not trusted: tests/test_qdrant_compat.py derives every
qdrant-client parameter from ``inspect.signature`` against the qdrant-client
version pinned in requirements-dev.txt (``QDRANT_CLIENT_VERSION``), and fails if
a parameter is neither named by our method nor classified here, or if an entry
here names something qdrant-client does not have. Upgrading the pin is therefore
a deliberate change that has to classify whatever the new version added.

Why 1.15.1: it is the last qdrant-client release that has ``search``, the
method the README's migration example calls (1.16 removed it in favour of
``query_points``, which this SDK does not have), and it supports Python 3.9,
the floor of this package, so every CI lane derives from the same version.
"""

from typing import Any, Dict, Mapping, NamedTuple, Tuple

# The qdrant-client release the table below was derived from. Must equal the
# pin in requirements-dev.txt and the installed version; the test checks both.
QDRANT_CLIENT_VERSION = "1.15.1"

IGNORE = "ignore"
REFUSE = "refuse"


class QdrantArgRule(NamedTuple):
    """How one qdrant-client argument is treated by one of our methods."""

    kind: str
    reason: str
    # REFUSE only: values accepted anyway, because they describe what we
    # already do. Compared after unwrapping an Enum to its ``.value``.
    accepted: Tuple[Any, ...] = ()


# Point writes: upsert, delete and the three payload writers.
_WAIT = QdrantArgRule(
    IGNORE,
    "every point write is committed before the API answers (it runs the write "
    "with wait=true), so wait=True already holds and wait=False gets a "
    "stronger guarantee than it asked for",
)
_ORDERING = QdrantArgRule(
    REFUSE,
    "writes use Qdrant's default 'weak' ordering; 'medium' and 'strong' "
    "ordering are not provided, so only ordering='weak' is accepted",
    accepted=(None, "weak"),
)
_SHARD_KEY_SELECTOR = QdrantArgRule(
    REFUSE,
    "custom shard keys are not supported; placement is per collection, set "
    "with create_collection(regions=...)",
    accepted=(None,),
)
# Reads: retrieve, scroll, search.
_CONSISTENCY = QdrantArgRule(
    REFUSE,
    "a read is answered by the region you are connected to; a multi-replica "
    "read consistency level is not provided",
    accepted=(None,),
)

_WRITE_ARGS: Dict[str, QdrantArgRule] = {
    "wait": _WAIT,
    "ordering": _ORDERING,
    "shard_key_selector": _SHARD_KEY_SELECTOR,
}

# create_collection: the service creates every collection with the same
# storage and index configuration (vectordb routes/proxy.js builds the Qdrant
# body from `vectors` alone), so none of these would be applied.
_NOT_APPLIED = (
    "collection storage and index settings are fixed by the service, so this "
    "would not be applied and the collection would behave differently from "
    "what you asked for"
)
_REPLICATION = (
    "replication across regions is managed by the service (choose regions with "
    "create_collection(regions=...)); the per-collection durability and write "
    "consistency this asks for are not provided"
)
_SHARDING = "sharding is managed by the service and cannot be configured"


def _refuse(reason: str) -> QdrantArgRule:
    return QdrantArgRule(REFUSE, reason, accepted=(None,))


QDRANT_COMPAT: Dict[str, Dict[str, QdrantArgRule]] = {
    "close": {
        "grpc_grace": QdrantArgRule(
            IGNORE,
            "this client has no gRPC channel, so close() has nothing to wait for",
        ),
    },
    "count": {
        "shard_key_selector": _SHARD_KEY_SELECTOR,
    },
    "create_collection": {
        "sparse_vectors_config": _refuse(
            "only dense vectors are supported; the collection would be "
            "created without the sparse vectors you configured"
        ),
        "shard_number": _refuse(_SHARDING),
        "sharding_method": _refuse(_SHARDING),
        "replication_factor": _refuse(_REPLICATION),
        "write_consistency_factor": _refuse(_REPLICATION),
        "on_disk_payload": _refuse(_NOT_APPLIED),
        "hnsw_config": _refuse(_NOT_APPLIED),
        "optimizers_config": _refuse(_NOT_APPLIED),
        "wal_config": _refuse(_NOT_APPLIED),
        "quantization_config": _refuse(_NOT_APPLIED),
        "strict_mode_config": _refuse(_NOT_APPLIED),
        "init_from": _refuse(
            "creating a collection from another one is not supported; it "
            "would be created empty"
        ),
    },
    "delete": dict(_WRITE_ARGS),
    "delete_payload": dict(_WRITE_ARGS),
    "overwrite_payload": dict(_WRITE_ARGS),
    "retrieve": {
        "consistency": _CONSISTENCY,
        "shard_key_selector": _SHARD_KEY_SELECTOR,
    },
    "scroll": {
        "consistency": _CONSISTENCY,
        "shard_key_selector": _SHARD_KEY_SELECTOR,
    },
    "search": {
        "append_payload": QdrantArgRule(
            REFUSE,
            "it is a deprecated alias in qdrant-client itself; pass "
            "with_payload=False instead",
            accepted=(True,),
        ),
        "consistency": _CONSISTENCY,
        "shard_key_selector": _SHARD_KEY_SELECTOR,
    },
    "set_payload": dict(_WRITE_ARGS),
    "upsert": dict(_WRITE_ARGS),
}


def check_qdrant_kwargs(method: str, kwargs: Mapping[str, Any]) -> None:
    """Enforce the contract for the extra keywords ``method`` received.

    Raises:
        TypeError: for a keyword that is not a classified qdrant-client
            parameter of ``method`` (worded like Python's own error), or for
            a REFUSE parameter passed a value we would not honour.
    """
    rules = QDRANT_COMPAT[method]
    for name, value in kwargs.items():
        rule = rules.get(name)
        if rule is None:
            raise TypeError(f"{method}() got an unexpected keyword argument {name!r}")
        if rule.kind == REFUSE and getattr(value, "value", value) not in rule.accepted:
            raise TypeError(
                f"{method}() does not support qdrant-client's {name!r} argument: "
                f"{rule.reason}."
            )
