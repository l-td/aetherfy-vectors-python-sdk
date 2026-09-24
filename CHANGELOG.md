# Changelog

All notable changes to the Aetherfy Vectors Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING: `AetherfyVectorsClient(...)` raises `TypeError` for an argument
  it does not name.** The constructor ended in a `**kwargs` that nothing read,
  so any unknown argument was accepted and dropped. `region=` was renamed to
  `api_region=` on 2026-06-29, and a caller still passing
  `region="eu-central-1"` got no error and no warning, just a client routed to
  the default endpoint instead of the region they asked for. That call now
  fails at construction. Rename `region=` to `api_region=`. There is
  deliberately no deprecated alias: it would keep alive the name that caused
  the misrouting. A typo (`api_regoin=`) fails the same way, and so does a
  `QdrantClient` constructor argument (`host=`, `port=`), which the migration
  replaces rather than carries over.
- **BREAKING: the vectors client's methods accept only their own parameters
  and the qdrant-client parameters of the same method.** `create_collection`,
  `upsert`, `delete`, `retrieve`, `scroll`, `count`, `delete_collection`,
  `get_collection`, `get_collections` and `collection_exists` also swallowed
  unknown keywords. A name that qdrant-client 1.15.1 does not have for that
  method now raises `TypeError`. A name it does have is handled as the
  README's new compatibility table says, and the table is the contract:
  - **accepted, no effect** where our behaviour already gives the caller what
    they asked for: `wait` on point writes (every write is committed before the
    call returns) and `close(grpc_grace=...)`.
  - **refused** (a `TypeError` naming the argument and the reason) where
    ignoring it would change what the caller gets: `ordering` other than
    `'weak'`, `shard_key_selector`, `consistency`, `search(append_payload=False)`,
    and every storage, index, sharding and replication setting of
    `create_collection`. Passing the value that already describes our
    behaviour (qdrant's own default) is accepted.
  - **honoured**: `timeout=` on `create_collection`, `delete_collection`,
    `retrieve`, `scroll`, `search` and `count` is a deadline for the WHOLE
    call, as qdrant-client's is: each attempt gets only the remaining budget,
    a backoff sleep that would overrun it is not taken, and no attempt starts
    after it. (Counting it per attempt would have let `timeout=5` take ~18 s
    across three retries and their backoff.) The constructor's `timeout`
    keeps its meaning, a per-attempt bound with writes retried up to 3
    times. `scroll(order_by=...)` orders by a payload key.
  `set_payload`, `overwrite_payload`, `delete_payload`, `search` and `close`,
  which had no `**kwargs`, now also accept their qdrant-client arguments on
  the same terms. `get_collections`, `get_collection`, `collection_exists` and
  `delete_collection` lost `**kwargs` entirely: qdrant-client gives them no
  argument beyond the ones they name.

  The classification lives in `aetherfy_vectors/qdrant_compat.py` and is
  derived, not trusted: a test reads every parameter from qdrant-client's real
  signatures at the version pinned in `requirements-dev.txt` and fails if one
  is neither named by our method nor classified.
- **BREAKING: parameters past the order qdrant-client shares are
  keyword-only.** Five methods took positional arguments in a different order
  from qdrant-client 1.15.1, so a migrated positional call bound its values to
  the wrong parameters without any error:
  `search` (third positional: our `limit`, qdrant's `query_filter`),
  `scroll` (second: `limit` / `scroll_filter`), `create_collection` (third:
  `distance` / `sparse_vectors_config`), `retrieve` (fifth: `timeout` /
  `consistency`) and `count` (fourth: `timeout` / `shard_key_selector`).
  Everything after the prefix the two orders share is now keyword-only, so
  such a call raises `TypeError`. Pass those arguments by keyword:
  `client.search("c", vec, limit=5)`, not `client.search("c", vec, 5)`. A test
  derives the rule from qdrant-client's real signatures: every method's
  positional parameters must be a prefix of qdrant-client's.

### Removed
- **`Thread` has no payload-schema methods.** `get_schema`, `set_schema`,
  `delete_schema`, `analyze_schema`, `refresh_schema` and `clear_schema_cache`
  moved from the shared `_Scope` base to `Namespace`. A schema belongs to a
  collection, and a thread no longer has one to itself, so
  `thread.set_schema(...)` would have imposed a schema on every other thread in
  the workspace. Removed rather than left lying; `MemoryClient.clear_schema_cache()`
  is unchanged.

### Changed
- **BREAKING: a thread is a payload scope, not a collection.** Every thread in a
  workspace now lives in ONE collection (`__threads__`) with `thread_id` as a
  payload key, and every per-thread operation is a filtered operation over it.
  The old model gave each thread its own collection, so threads counted against
  `plans.max_collections` — a Free account (limit 3) got THREE CONVERSATIONS,
  EVER, and the fourth `create_thread` raised `COLLECTION_LIMIT_EXCEEDED` and
  fired the "you hit your plan limit" email. Creating a thread now consumes no
  collection slot.

  - `create_thread(thread_id)` no longer accepts `vector_size` or `distance`.
    One collection has one of each; they come from
    `MemoryClient(thread_vector_size=384, thread_distance=DistanceMetric.COSINE)`
    and are fixed when the threads collection is first created. Passing the old
    keywords is a `TypeError`. `create_namespace` keeps both — a namespace is
    still one collection.
  - `list_threads` / `listThreads` is de-duplicated: creating a thread is a
    check-then-write, so two callers racing the exists-check can both write a
    marker. Nothing else notices, but the listing would have named the thread
    twice.
  - Creating a thread writes one MARKER point, which is what makes an EMPTY
    thread exist: `thread_exists`, `list_threads` and `ThreadAlreadyExistsError`
    keep the behaviour they had. The marker is never a message — `history`,
    `iter_history`, `search`, `count`, `iter` and a filtered `delete` all
    exclude it.
  - `Thread.clear()` is a delete-by-filter on that thread's rows, not a
    collection drop. It keeps its old meaning (the thread stops existing) and
    leaves every sibling thread intact. `Namespace.clear()` still drops its
    collection.
  - A `filter` you pass to a `Thread`'s `search` / `count` / `iter` / `delete`
    is COMBINED with the thread's own clause, never substituted for it, so it
    can narrow a thread's results but cannot reach another thread's messages.
  - `retrieve`, `delete` by id list, and the three metadata writers refuse ids
    belonging to another thread: a thread's point ids are unique within the
    shared collection, not within the thread. `delete` enforces this
    SERVER-side, addressing `{thread_id AND has_id}` in one request; the
    metadata writers read first, because a filter that matches nothing is a
    success and they are documented to raise `PointNotFoundError`.
    `retrieve` cannot be filtered at all — Qdrant's point-request takes ids
    only — so it filters what came back.
  - `delete([])` returns True without a request where it used to send one.
    For a thread that is a safety property: an id list becomes a `has_id`
    clause, and an empty one must never reach the engine.
  - `get_thread(id)` reports the shared collection's config with `name` set to
    the thread id and `points_count` set to that thread's own message count.
  - `Thread`'s reserved metadata keys gain `thread_id` and `thread_marker`.
  - New `ThreadVectorSizeMismatchError` when the threads collection already
    exists at another dimension — it names the dimension that is there instead
    of surfacing later as a bare dimension error on the first write.
- **`_Scope.count`'s `filter` accepts a `Filter` as well as a dict**, matching
  `search` and `iter`, which always did.

### Added
- **`aetherfy_vectors` exports every exception it raises.** `ValidationError`,
  `PointNotFoundError`, `RequestTimeoutError` and `NetworkError` were raised by
  the client (and `PointNotFoundError` by a memory `Thread`'s and `Namespace`'s
  metadata writers), and `CollectionNotFoundError` is built from a 404, yet none
  of the five was importable from the package root — only from
  `aetherfy_vectors.exceptions`. All five are in `aetherfy_vectors.__all__` now,
  matching the JavaScript root. `tests/test_public_surface.py` makes the class
  structural: every SDK exception raised anywhere in `aetherfy_vectors`,
  `aetherfy_memory` or `aetherfy_agent` must be exported from the root of the
  package that defines it, and so must every exception class a package defines
  under its own base — which covers the classes only `parse_error_response`
  builds, invisible to a search for `raise`.
- **`ConflictError`, for a 409 with no more specific class.** Parity with the
  JavaScript SDK, which has always had it. `COLLECTION_IN_USE` and
  `COLLECTION_EXISTS_IN_OTHER_REGION` keep `CollectionInUseError` and
  `CollectionInOtherRegionError` (neither is a `ConflictError` subclass, as in
  JavaScript); every other 409 used to arrive as the bare
  `AetherfyVectorsException` and is now a `ConflictError` carrying the backend's
  code in `error_code`. It subclasses `AetherfyVectorsException`, so
  `except AetherfyVectorsException` still catches it and it is still not
  retried. The one observable difference is to code comparing the exact type
  (`type(e) is AetherfyVectorsException`).
- **`create_field_index` / `delete_field_index` on `AetherfyVectorsClient`.**
  `PUT /collections/{name}/index` and
  `DELETE /collections/{name}/index/{field_name}` have been on the backend (and
  replicated) all along, but no SDK exposed them. A filter on an unindexed
  payload key is scanned, not looked up, so any key you filter on for every
  read wants one. `field_schema` defaults to `"keyword"` and is forwarded
  verbatim.

### Release needed
- **The pending release is no longer a patch.** The changes above remove
  `create_thread`'s `vector_size` / `distance` parameters and the `Thread`
  schema methods, and change where a thread's data lives. Under semver that is
  a MAJOR bump: publish this as **2.0.0**, not 1.1.1 or 1.2.0. Threads written
  by 1.1.0 live in per-thread collections that 2.0.0 does not read; there is no
  migration and no shim, which is fine while the SDK has no users but must be
  stated in the release notes.
- **The published 1.1.0 does not recognise `AGENT_RUN_CONCURRENCY_LIMIT_EXCEEDED`.**
  It still matches the old code, so against the current platform a full
  runs-in-flight limit reaches 1.1.0 callers as a plain `SpawnError` instead of
  `TooManyRunsInFlight`, the one spawn refusal worth retrying. Publish this
  version before anyone relies on that retry.

### Planned Features
- Additional distance metrics support
- Streaming search results
- Bulk export/import utilities
- Enhanced analytics dashboards
- Integration with popular ML frameworks
- CLI tools for management operations

## [1.1.0] - 2026-09-07

### Added
- **`aetherfy_agent` — what code running on an Aetherfy agent does.** A new
  top-level module in this same distribution, beside `aetherfy_vectors` and
  `aetherfy_memory`:

  - `payload()` reads this run's input. The file named by
    `AETHERFY_SPAWN_PAYLOAD_PATH` first, then the documented HTTP fallback
    when the machine could not write it; `{}` for a run given no input, which
    is the normal case for a scheduled fire.
  - `machine()` returns the run's `MachineShape` — `vcpus`, `memory_mb`,
    `region` — as whole numbers rather than the strings the environment
    carries.
  - `fan_out(fn, items, width=None, kind="threads")` runs an in-machine pool
    and returns results in INPUT order, re-raising the lowest-indexed failure
    rather than swallowing it. The default width follows the POOL, because
    the pool already declares the shape of the work: `kind="threads"` mostly
    waits, so it defaults to `vcpus * 8`; `kind="processes"` is only worth
    its pickling cost for CPU-bound work, so it defaults to `vcpus` — one
    worker per core. It prints one line to stdout before running, so a run's
    width is visible in its logs afterwards.
  - `spawn(child, payload=None)` runs a different task agent.
    `413 RUN_PAYLOAD_TOO_LARGE` becomes `PayloadTooLarge` (carrying
    `payload_bytes` / `max_bytes`) and
    `429 AGENT_SPAWN_CONCURRENCY_LIMIT_EXCEEDED` becomes
    `TooManyRunsInFlight`, the one refusal worth retrying. THE STATUS AND
    THE CODE TOGETHER select the type: a 413 or 429 carrying any other
    code, or none, becomes a plain `SpawnError` reporting the code and
    message that actually arrived, rather than wearing a code the platform
    never sent. Every other status becomes `SpawnError` with the
    platform's stable `error_code` too.
    `TooManyRunsInFlight` carries `in_flight_count`, `limit` (which plan
    limit was hit, `"max_in_flight_runs"` today) and `max_in_flight_runs`
    (its value, `None` on a plan that declares no cap). The cap is the
    ACCOUNT's, set by the plan — not a per-agent spawn ceiling.
  - `write_result(value)` returns this run's answer to whoever started it.
    The mirror of `payload()`: a file named by `AETHERFY_SPAWN_RESULT_PATH`,
    nothing over the network. It refuses over the machine's inline cap
    (`AETHERFY_RUN_INLINE_MAX_BYTES`, one number bounding the payload and the
    result alike) with `ResultTooLarge` carrying `result_bytes` /
    `max_bytes` — the platform would have dropped the value and recorded
    `result_error` instead, and a value discarded silently is a value the
    caller never learns to shrink. It refuses `NaN` and the infinities, which
    Python's `json` would otherwise write as tokens no other JSON reader
    accepts. When `AETHERFY_SPAWN_RESULT_PATH` is absent there is nowhere to
    put an answer, and that raises `NotRunningOnAgent` rather than no-opping:
    a library that silently discards the one value it was called to deliver
    is worse than one that says so. That one error explains the two ways a
    machine can lack the path — no inline cap, or a `service` agent, which
    has no runs to answer from — instead of the module's usual "the platform
    sets this before your entrypoint starts", which is true of every other
    variable it reads and not of this one.
  - `result(run_id)` reads one run back — `state`, `result`, `result_error`,
    `has_result`, and the rest of the object verbatim in `Run.raw`.
  - `wait(run_id, timeout_seconds=30)` is the same read with the waiting done
    server-side, holding ONE request open instead of polling. `1..60`,
    checked here before anything is sent, so a bad argument costs no round
    trip. A TIMEOUT IS NOT AN ERROR: the run comes back exactly as it stands
    and `state` is what tells the two apart. Unlike every other call in this
    module it does not retry a dropped connection — a retry would hold a
    second full timeout and hand back a run up to twice as late as the number
    the caller passed.
  - `WAIT_TIMEOUT_MIN_SECONDS`, `WAIT_TIMEOUT_MAX_SECONDS` and
    `WAIT_TIMEOUT_DEFAULT_SECONDS` are public, and are public in the
    JavaScript helper too — a caller sizing its own loop around `wait()`
    reads the bound rather than copying the numbers out of an error message.

  Reading a run maps its refusals the way `spawn()` does, on the status AND
  the code together: `404 DEPLOYMENT_NOT_FOUND` is `RunNotFound`,
  `403 DEPLOYMENT_ACCESS_DENIED` is `RunAccessDenied`,
  `422 DEPLOYMENT_WAIT_TIMEOUT_INVALID` is `WaitTimeoutInvalid`, and anything
  else — including those statuses carrying another code — is `RunReadError`
  reporting what actually arrived. The plain read and the waiting read refuse
  identically, because upstream they are one loader behind two routes.

  Every id this module puts in a URL path is percent-encoded. Left raw, an
  id carrying a slash or a `..` normalises into a request to a DIFFERENT
  route before it leaves the process, and the answer is then parsed as
  though it were the object that was asked for — a wrong object read as the
  right one, silently. Encoded, the platform answers 404. This covers the
  payload fallback's spawn id as well as the two run reads.

  Each of these was already a documented platform contract that every task
  hand-rolled; none of them is a new protocol. The module adds NO dependency:
  its HTTP is `urllib` from the standard library. Every request sets an
  explicit `User-Agent`, because urllib's default is blocked at the edge and
  produces a 403 that reads exactly like an auth failure.

  The standard runtime image preinstalls this distribution, so a plain agent
  gets the helper with nothing in its requirements, and a version the customer
  pins wins over it.

### Fixed
- `UsageStats` now describes the response `GET /api/v1/analytics/usage`
  actually serves: `storage_bytes_used`, `storage_limit_bytes` (`None` on an
  unlimited tier), `collections_count`, `collections_limit` (also `None` on an
  unlimited tier — one sentinel for both),
  `tier`, `active_regions` and `usage_percentage`. The nine fields it declared
  before (`current_collections`, `max_collections`, `current_points`,
  `max_points`, `requests_this_month`, `max_requests_per_month`,
  `storage_used_mb`, `max_storage_mb`, `plan_name`) were never served by
  anything, so `client.get_usage_stats()` raised `KeyError` on every genuine
  200 — the unit tests passed only because they mocked the invented payload.
  The derived `*_usage_percent` properties are gone with the fields they
  divided; the endpoint's own `usage_percentage` is the only percentage it
  reports. A live e2e call now pins the shape
  (aetherfy-e2e-tests `tests/sdk/test_usage_stats_sdk.py`).

## [1.0.0] - 2026-08-17

First public release on PyPI. The work that had accumulated under
`[Unreleased]` is folded in here: `1.0.0` had never been published, so there
is no earlier release for these changes to be "changes since".

### Added
- `search_params` on `client.search()` and `Namespace`/`Thread.search()` —
  engine params sent verbatim as the body's `params`, e.g.
  `search_params={"hnsw_ef": 256}` to trade latency for recall. Omitting it
  leaves the default body unchanged. Works against every deployed backend:
  the API has always forwarded the search body verbatim, so there is no
  version gate.
- `client.count()` accepts a `Filter` object, not only a plain dict —
  matching `search`, `scroll` and `delete`. A `Filter` passed to `count`
  previously reached the wire un-serialized.
- Drop-in replacement for qdrant-client with API compatibility.
- Global vector database operations with automatic replication, intelligent
  caching, and routing.
- Built-in usage statistics and limit tracking.
- Comprehensive error handling with a detailed exception hierarchy.
- Batch operations, complex filtering, context-manager support, and a
  thread-safe client.
- Type hints throughout, with `py.typed` markers on both packages.

### Changed
- `client.search()` no longer ends in `**kwargs`: unknown keyword arguments
  now raise `TypeError` instead of being silently dropped from the request
  body — the same contract `scroll_iter` already had. (`Namespace`/
  `Thread.search()` were already keyword-only.)
- `validate_point_id` now enforces the server's point-id rule client-side:
  an id must be an unsigned integer `<= 2**53 - 1` or a UUID string in any
  of the four Qdrant-accepted forms (canonical, simple 32-hex, braced,
  `urn:uuid:`). Invalid ids raise `ValidationError` with the same wording
  as the server's 400 `INVALID_POINT_ID` response. This does not change
  which ids work — ids the validator now rejects were already rejected by
  the server; the error just surfaces before the request is sent. The
  `2**53 - 1` bound mirrors the server's JSON-number parse layer
  (IEEE-754 doubles), not a Python `int` limitation.
- Filter clauses serialize in a fixed order (`must`, `must_not`, `should`)
  regardless of the order the caller wrote them. Server cache keys are
  derived from the request body bytes, so two callers expressing the same
  filter differently now share one cache entry.
- An unrecognized filter clause raises `ValidationError` instead of being
  forwarded. This closes the dict escape hatch: `Filter.to_dict()` was
  always correct, but `search`/`scroll`/`count`/`delete` also accept a plain
  dict and used to send it to the engine unexamined. A caller who wrote
  `{"mustNot": [...]}` — the JavaScript SDK's spelling — had the entire
  exclusion clause dropped with no error and no warning, and got back
  exactly the points they meant to exclude. The error names `must_not` as
  the correct key.
- Minimum supported Python is 3.9 (3.8 is end-of-life and was dropped from
  the support matrix).

### Fixed
- **`MemoryClient` ignored `AETHERFY_VECTORS_URL`.** Its `endpoint`
  parameter defaulted to the literal default URL rather than `None`, and
  `AetherfyVectorsClient` treats any explicit endpoint as
  highest-precedence — so the constructor always looked like a caller
  asking for the global endpoint. The control plane injects
  `AETHERFY_VECTORS_URL` on every agent machine, which meant a deployed
  Python agent using memory silently talked to the default endpoint instead
  of its regional one. Resolution order is now identical to
  `AetherfyVectorsClient`: explicit argument, then the environment
  variable, then the default. The JS SDK was never affected.
- Memory SDK: `Namespace.add`/`add_many` and `Thread.add`/`append_many` no
  longer `str()`-coerce an explicit `id`. An integer id (a valid
  unsigned-integer point id) now reaches the wire as an `int` instead of
  being turned into a numeric string like `"42"` — which the point-id
  validator rejects. A non-int/non-UUID explicit id is passed through and
  correctly rejected by the upsert validator. Return types widen from
  `str`/`List[str]` to `Union[str, int]` / `List[Union[str, int]]`, and
  `Message.id` accepts `Union[str, int]`.

### Packaging
- Added `aetherfy_memory/py.typed`. `setup.py` declared it in
  `package_data`, but the marker file did not exist, so type checkers
  treated the whole memory package as untyped.
- Added `MANIFEST.in`. `find_packages(exclude=["tests*"])` governs the wheel
  only; the sdist was built from setuptools' default sweep and shipped the
  entire test suite, so the two artifacts of one release disagreed about
  what the package contained.
- Corrected the repository URLs, which pointed at a
  `github.com/aetherfy/aetherfy-vectors-python` repository that does not
  exist.

### Core Features
- `AetherfyVectorsClient` - Main client class with a qdrant-client compatible API
- Collection management (create, delete, list, info)
- Point operations (upsert, retrieve, delete, count)
- Vector search with filtering and pagination
- Usage statistics and quota monitoring
- API key authentication with environment variable support
- Automatic request routing, failover and retry

### Models and Types
- `VectorConfig` - Vector configuration with size and distance metric
- `Point` - Vector point with ID, vector, and payload
- `SearchResult` - Search result with score and metadata
- `Collection` - Collection information and configuration
- `UsageStats` - Usage statistics and limits
- `Filter` - Query filter for search operations
- Comprehensive exception hierarchy for error handling

---

For upgrade instructions and breaking changes, see the documentation at
<https://docs.aetherfy.com/vectors>.
