# Changelog

All notable changes to the Aetherfy Vectors Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **`TooManyRunsInFlight` is raised on `429 AGENT_RUN_CONCURRENCY_LIMIT_EXCEEDED`.**
  The platform renamed the code (it was `AGENT_SPAWN_CONCURRENCY_LIMIT_EXCEEDED`)
  because the account's runs-in-flight limit answers a manual and a scheduled
  run too. The exported constant is renamed with it; there is no alias.
- **`spawn(child)` accepts a child of either type.** A service child's run is a
  request to its own `POST /aetherfy/run`; the parent is recorded on the run,
  never on the child.

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
