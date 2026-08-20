# Changelog

All notable changes to the Aetherfy Vectors Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned Features
- Additional distance metrics support
- Streaming search results
- Bulk export/import utilities
- Enhanced analytics dashboards
- Integration with popular ML frameworks
- CLI tools for management operations

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
- Built-in performance analytics, usage statistics and limit tracking.
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
- Global performance analytics
- Usage statistics and quota monitoring
- API key authentication with environment variable support
- Automatic request routing, failover and retry

### Models and Types
- `VectorConfig` - Vector configuration with size and distance metric
- `Point` - Vector point with ID, vector, and payload
- `SearchResult` - Search result with score and metadata
- `Collection` - Collection information and configuration
- `PerformanceAnalytics` - Global performance metrics
- `UsageStats` - Usage statistics and limits
- `Filter` - Query filter for search operations
- Comprehensive exception hierarchy for error handling

---

For upgrade instructions and breaking changes, see the documentation at
<https://docs.aetherfy.com/vectors>.
