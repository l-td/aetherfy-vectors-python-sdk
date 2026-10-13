"""
Main client implementation for Aetherfy Vectors SDK.

Compatible with qdrant-client 1.15.1's core methods, on the terms in
aetherfy_vectors.qdrant_compat, and routes requests through the global vector
database service.
"""

import json
import math
import os
import time
from typing import List, Dict, Any, Iterator, Optional, Sequence, Union
import requests
from requests.adapters import HTTPAdapter

from .auth import APIKeyManager
from .models import (
    Point,
    SearchResult,
    Collection,
    VectorConfig,
    DistanceMetric,
    Filter,
    UsageStats,
)
from .exceptions import (
    AetherfyVectorsException,
    CollectionNotFoundError,
    PointNotFoundError,
    RequestTimeoutError,
    ValidationError,
    NetworkError,
    SchemaValidationError,
    SchemaNotFoundError,
    PartialUpsertError,
)
from .qdrant_compat import check_qdrant_kwargs
from .chunking import chunk_points_by_bytes, MAX_REQUEST_BYTES, point_wire_bytes
from .schema import (
    Schema,
    FieldDefinition,
    AnalysisResult,
    validate_vectors,
)
from .utils import (
    validate_vector,
    validate_collection_name,
    validate_point_id,
    build_api_url,
    parse_error_response,
    format_points_for_upsert,
    quote_collection_name,
    serialize_filter,
)


class AetherfyVectorsClient:
    """
    Aetherfy Vectors client, compatible with qdrant-client 1.15.1's core methods.

    The exact contract (which qdrant-client arguments are accepted, refused or
    honoured, per method) is aetherfy_vectors.qdrant_compat. Requests go through
    the global vector database service for enhanced performance, automatic
    global replication, and zero DevOps complexity.
    """

    DEFAULT_ENDPOINT = "https://vectors.aetherfy.com"
    DEFAULT_TIMEOUT = 30.0
    VALID_REGIONS = ("us-east-1", "eu-central-1", "ap-southeast-1")

    # Body-aware timeout scaling. The default 30 s is fine for small
    # requests, but a single upsert chunk can be 24 MB (MAX_REQUEST_BYTES
    # in chunking.py) and that doesn't fit in 30 s on residential / WAN
    # uplinks at 25 Mbps and below. Without scaling, requests aborts mid-
    # upload, retry_with_backoff fires its 3 attempts, each timing out —
    # the chunk lands in PartialUpsertError.failed even though the origin
    # would have accepted it given enough time. Linear scaling above a
    # small floor: cheap requests stay snappy, large uploads get the
    # runway they need.
    #
    # Tuned for ~25 Mbps as the floor — at that bandwidth 1 MB takes
    # ~320 ms, so +1 s/MB gives ~3× margin for TLS + server processing.
    # Mirrors aetherfy-vectors-js-sdk/src/http/client.ts so the two SDKs
    # carry the same timeout policy.
    TIMEOUT_THRESHOLD_BYTES = 5 * 1024 * 1024
    TIMEOUT_PER_MB_OVER_THRESHOLD_S = 1.0

    # Payload-index writes (create and delete). The server holds ONE of them
    # for up to INDEX_WAIT_BUDGET_S while Qdrant applies it, then answers
    # "acknowledged" (still in progress). Mirrors vectordb
    # backend/config/timeouts.js INDEX_WAIT_BUDGET_MS. When the region that
    # answers does not host the collection it forwards the write first, which
    # vectordb allows INDEX_FORWARD_MARGIN_S for (FORWARD_MARGIN_MS in the
    # same file). Each attempt's HTTP timeout, INDEX_ATTEMPT_TIMEOUT_S (or the
    # constructor's timeout if that is longer), must outlast both, plus this
    # client's own hop, or the SDK times out before the server's answer
    # arrives. The default 30 s did not leave room for the forward. The
    # relation is pinned against vectordb's and the JS SDK's source by
    # aetherfy-e2e-tests tests/pyunit/test_index_timeouts_pair.py, and locally
    # in tests/test_field_index.py. Mirrors aetherfy-vectors-js-sdk/src/client.ts.
    INDEX_WAIT_BUDGET_S = 25.0
    INDEX_FORWARD_MARGIN_S = 5.0
    INDEX_ATTEMPT_TIMEOUT_S = 45.0
    # create_field_index is never unbounded: with no ``timeout=`` it stops at
    # INDEX_DEFAULT_DEADLINE_S with the same "still building" error. And an
    # "acknowledged" that came back faster than INDEX_WAIT_BUDGET_S means the
    # server did not hold the create (a vectordb from before db26396 answers
    # every create that way, at once), so the next create waits first:
    # INDEX_RESEND_PAUSE_FIRST_S, doubling up to INDEX_RESEND_PAUSE_MAX_S.
    # An "acknowledged" the server held for its budget is re-sent at once.
    # Mirrors aetherfy-vectors-js-sdk/src/client.ts.
    INDEX_DEFAULT_DEADLINE_S = 600.0
    INDEX_RESEND_PAUSE_FIRST_S = 1.0
    INDEX_RESEND_PAUSE_MAX_S = 10.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_region: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        workspace: Optional[str] = "auto",
    ):
        """Initialize Aetherfy Vectors client.

        Args:
            api_key: Aetherfy API key. If None, will try environment variables.
            endpoint: API endpoint URL. Resolution order:
                1. Explicit ``endpoint`` argument.
                2. ``AETHERFY_VECTORS_URL`` environment variable.
                3. ``api_region`` argument (or ``AETHERFY_VECTORS_API_REGION``
                   env var). Resolved via ``/api/v1/regions`` discovery
                   against the default global URL.
                4. Default ``https://vectors.aetherfy.com``.
            api_region: Which regional API endpoint to CONNECT to
                ('us-east-1', 'eu-central-1', or 'ap-southeast-1') — a
                transport/routing override, NOT where collections live. In
                the integrated product this is effectively ignored: the
                control plane injects ``AETHERFY_VECTORS_URL`` on every agent
                machine and an explicit URL always wins (a warning is logged
                if both are set). It is load-bearing only for standalone
                vectordb usage, local development, and debugging, where it
                selects the regional endpoint. Distinct from a collection's
                placement ``regions`` (``create_collection(regions=...)``).
            timeout: Per-ATTEMPT request timeout in seconds (default: 30.0).
                Writes (POST/PUT) scale it up with body size above 5 MB and are
                retried up to 3 times with exponential backoff, so one call
                can take several times this long. A method's own ``timeout=``
                is different: a deadline for that whole call.
            workspace: Workspace name for multi-agent coordination.
                Defaults to ``'auto'``: read ``AETHERFY_WORKSPACE`` from the
                environment, and fall back to no workspace when it is unset.
                The control plane sets that variable only on an agent that
                really has a workspace, so 'auto' can only ever resolve to a
                name the workspaces table holds.
                - Set to a string to use a specific workspace
                - Pass None to force no workspace (collections are not
                  namespaced) even inside a workspaced agent

        There is deliberately no ``**kwargs``: an argument this constructor
        does not name raises TypeError. It used to be accepted and dropped, so
        the pre-rename ``region=`` silently routed a caller to the default
        endpoint instead of the region they asked for. qdrant-client's
        constructor arguments are not accepted either: the migration replaces
        that call wholesale (see ``aetherfy_vectors.qdrant_compat``).

        Raises:
            AuthenticationError: If API key is invalid or missing.
            ValueError: If ``api_region`` is not one of us-east-1/eu-central-1/ap-southeast-1.
            AetherfyVectorsException: If region discovery fails.
        """
        # Validate the API-region override eagerly so a typo fails at
        # construction, not on the first network round trip.
        env_region = os.getenv("AETHERFY_VECTORS_API_REGION")
        chosen_region = api_region if api_region is not None else env_region
        if chosen_region is not None and chosen_region not in self.VALID_REGIONS:
            raise ValueError(
                f"api_region must be one of {self.VALID_REGIONS}, got {chosen_region!r}"
            )
        self.api_region: Optional[str] = chosen_region

        # Cached /api/v1/regions response, populated lazily on first
        # discovery call. Per-instance (NOT module-global) — multiple
        # client instances may target different default endpoints (test
        # suite vs prod) and must not share resolved state.
        self._regions_discovery_cache: Optional[Dict[str, str]] = None

        self.timeout = timeout

        # Auth must be initialized BEFORE endpoint resolution because
        # region discovery hits /api/v1/regions with the API key.
        self.auth_manager = APIKeyManager(api_key)

        # Endpoint resolution. Explicit `endpoint=` and AETHERFY_VECTORS_URL
        # both bypass discovery entirely; api_region= triggers discovery only
        # when neither of the above is set, so a deployment-time env var
        # always wins over an api_region= argument the caller may have left in
        # local-dev code.
        env_url = os.getenv("AETHERFY_VECTORS_URL")
        if endpoint is not None:
            resolved_endpoint = endpoint
        elif env_url is not None:
            if self.api_region is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "Both AETHERFY_VECTORS_URL and api_region=%s are set; using "
                    "AETHERFY_VECTORS_URL — api_region= is a standalone/local-dev "
                    "override, the injected URL wins in integrated agents",
                    self.api_region,
                )
            resolved_endpoint = env_url
        elif self.api_region is not None:
            resolved_endpoint = self._resolve_region_endpoint(self.api_region)
        else:
            resolved_endpoint = self.DEFAULT_ENDPOINT

        self.endpoint = resolved_endpoint.rstrip("/")

        # Initialize workspace (auto-detect or explicit)
        if workspace == "auto":
            self.workspace = os.getenv("AETHERFY_WORKSPACE")
        else:
            self.workspace = workspace

        # auth_manager is initialized above (before endpoint resolution).
        # Build the standard header bundle now that the endpoint is known.
        self.auth_headers = {
            **self.auth_manager.get_auth_headers(),
            "Content-Type": "application/json",
            "User-Agent": "aetherfy-vectors-python/1.0.0",
        }

        # Initialize HTTP session with connection pooling
        # This prevents TCP/TLS handshake overhead on every request
        self.session = self._create_session()

        # Initialize schema cache for ETag-based validation (vector configs)
        self._schema_cache: Dict[
            str, Dict[str, Any]
        ] = {}  # {collection_name: {schema, etag}}

        # Initialize payload schema cache for schema validation
        self._payload_schema_cache: Dict[
            str, Dict[str, Any]
        ] = {}  # {collection_name: {schema: Schema, etag: str, enforcement_mode: str}}

    def _resolve_region_endpoint(self, region: str) -> str:
        """Resolve a region code to its public URL via /api/v1/regions.

        Hits the default global endpoint with the configured API key,
        caches the discovery response on this instance, and returns
        the URL for ``region``. Raises if discovery fails or the
        region is not in the response.

        The cache is per-instance and lives for the process lifetime —
        there is no TTL.
        """
        if self._regions_discovery_cache is None:
            import json as _json

            url = build_api_url(self.DEFAULT_ENDPOINT, "regions")
            headers = {
                **self.auth_manager.get_auth_headers(),
                "Content-Type": "application/json",
                "User-Agent": "aetherfy-vectors-python/1.0.0",
            }
            try:
                resp = requests.get(url, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                raise AetherfyVectorsException(
                    f"Could not resolve region {region!r} via discovery: {exc}. "
                    "Check that the default endpoint is reachable, or pass "
                    "endpoint= directly."
                )
            if resp.status_code != 200:
                raise AetherfyVectorsException(
                    f"Region discovery returned {resp.status_code} from {url}. "
                    "Check that your API key is valid for the discovery endpoint."
                )
            try:
                self._regions_discovery_cache = _json.loads(resp.content or b"{}")
            except ValueError as exc:
                raise AetherfyVectorsException(
                    f"Region discovery returned non-JSON body: {exc}"
                )

        cache = self._regions_discovery_cache or {}
        if region not in cache:
            raise AetherfyVectorsException(
                f"Region {region!r} not configured at the discovery endpoint "
                f"(available: {sorted(cache.keys())})."
            )
        return cache[region]

    def _estimate_body_bytes(self, data: Any) -> int:
        """Fast estimate of JSON-serialized body size in bytes.

        Used by _compute_body_aware_timeout to scale per-request timeouts
        against the upload payload. Two paths:

          - Upsert fast path: data is ``{"points": [...]}``. Sum
            point_wire_bytes per point — O(1) per point, deterministic
            upper bound, no full serialization. Matches what
            chunk_points_by_bytes uses for sizing.
          - Other paths: fall back to json.dumps length. These bodies are
            typically small (search filters, scroll cursors) so the
            extra serialization is cheap.

        Returns 0 on unserializable input; caller treats that as
        "use base timeout".
        """
        if data is None:
            return 0
        if isinstance(data, (bytes, bytearray)):
            return len(data)
        if isinstance(data, str):
            return len(data.encode("utf-8"))
        if isinstance(data, dict):
            points = data.get("points")
            if isinstance(points, list):
                return sum(point_wire_bytes(p) for p in points)
        try:
            return len(json.dumps(data, separators=(",", ":")))
        except (TypeError, ValueError):
            return 0

    def _compute_body_aware_timeout(self, data: Any) -> float:
        """Compute the per-request timeout given the body's payload size.

        Bodies up to TIMEOUT_THRESHOLD_BYTES use ``self.timeout``
        unchanged; beyond that, add TIMEOUT_PER_MB_OVER_THRESHOLD_S for
        each megabyte over the threshold. See the TIMEOUT_* class
        constants for the rationale (and the JS SDK mirror in
        aetherfy-vectors-js-sdk/src/http/client.ts).
        """
        body_bytes = self._estimate_body_bytes(data)
        if body_bytes <= self.TIMEOUT_THRESHOLD_BYTES:
            return self.timeout
        mb_over = math.ceil((body_bytes - self.TIMEOUT_THRESHOLD_BYTES) / (1024 * 1024))
        return self.timeout + mb_over * self.TIMEOUT_PER_MB_OVER_THRESHOLD_S

    def _create_session(self) -> requests.Session:
        """Create a requests Session with connection pooling and retry logic.

        Returns:
            Configured requests Session object with persistent connections.
        """
        session = requests.Session()

        # Configure connection pooling via HTTPAdapter
        # This keeps connections alive and reuses them across requests
        adapter = HTTPAdapter(
            pool_connections=10,  # Number of connection pools to cache
            pool_maxsize=50,  # Max connections to keep in pool
            max_retries=0,  # No automatic retries (we handle this ourselves)
            pool_block=False,  # Don't block when pool is full
        )

        # Mount adapter for both HTTP and HTTPS
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set default headers on session
        session.headers.update(self.auth_headers)

        return session

    def _scope_collection(self, collection_name: str) -> str:
        """Local cache-key for a collection name, with workspace prefix when set.

        NOT sent on the wire anymore — the canonical vectordb wire form
        post-A/B is the nested URL `/workspaces/{ws}/collections/{name}`
        with a bare name in the URL/body (see `_build_collection_path`).
        Kept as a stable schema-cache key so same-name collections in
        different workspaces don't collide.
        """
        if self.workspace:
            return f"{self.workspace}/{collection_name}"
        return collection_name

    def _build_collection_path(self, collection_name: str, suffix: str = "") -> str:
        """Build the canonical URL path for a collection.

        Workspaced operations use the nested
        form `workspaces/{ws}/collections/{name}` instead of the old
        slash-in-name encoding. Workspaceless calls continue to use the
        flat form.

        Args:
            collection_name: BARE (unscoped) collection name.
            suffix: Optional path suffix (e.g. ``"/points/query"``).
                Must already begin with ``/`` when non-empty.
        """
        from urllib.parse import quote

        enc_name = quote(collection_name, safe="")
        if self.workspace:
            enc_ws = quote(self.workspace, safe="")
            return f"workspaces/{enc_ws}/collections/{enc_name}{suffix}"
        return f"collections/{enc_name}{suffix}"

    def _build_collections_list_path(self) -> str:
        """Workspaced list/create endpoint or workspaceless."""
        if self.workspace:
            from urllib.parse import quote

            enc_ws = quote(self.workspace, safe="")
            return f"workspaces/{enc_ws}/collections"
        return "collections"

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        enable_retry: bool = True,
        headers: Optional[Dict[str, str]] = None,
        evict_caches_on_404: Optional[str] = None,
        timeout: Optional[float] = None,
        attempt_timeout: Optional[float] = None,
    ) -> Any:
        """Make HTTP request to the API with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE).
            endpoint: API endpoint path.
            data: Request body data.
            params: Query parameters.
            enable_retry: Whether to enable retry logic (default: True).
            headers: Additional headers to include in request.
            evict_caches_on_404: If set, the scoped collection name whose
                schema and payload-schema caches should be dropped when
                the response is HTTP 404. Use for collection-scoped ops
                where 404 unambiguously means "the collection is gone"
                (cross-client delete or pre-existence check). Don't pass
                for /schema/<name> reads, where 404 also covers the
                legitimate "no payload schema set" state.
            timeout: A DEADLINE in seconds for the whole call, retries and
                backoff included, from a public method's ``timeout=``
                argument (qdrant-client's method ``timeout`` bounds the whole
                operation too). Each attempt is given only the budget that
                remains, a backoff sleep that would end past the deadline is
                not taken, and no attempt starts once it has passed. None keeps
                the client's policy: the constructor's ``timeout`` (body-aware
                for writes) bounds EACH attempt, and POST/PUT are retried up
                to 3 times with backoff on top of that.
            attempt_timeout: The per-attempt timeout of an endpoint the server
                may legitimately hold longer than the client's (the
                payload-index writes). Without ``timeout`` it replaces the
                client's per-attempt timeout; with it, each attempt gets the
                smaller of this and what remains of the deadline.

        Returns:
            Response data.

        Raises:
            AetherfyVectorsException: If request fails.
        """
        from .utils import retry_with_backoff

        # Body-aware timeout for write methods: large upserts on slow
        # uplinks need more runway than the 30 s default. Read methods
        # always use the base timeout (their bodies are tiny). See
        # _compute_body_aware_timeout / TIMEOUT_* class constants.
        deadline: Optional[float] = None
        if timeout is not None:
            deadline = time.monotonic() + timeout
            request_timeout = timeout
        elif attempt_timeout is not None:
            request_timeout = attempt_timeout
        elif method in ("POST", "PUT") and data is not None:
            request_timeout = self._compute_body_aware_timeout(data)
        else:
            request_timeout = self.timeout

        def make_single_request():
            url = build_api_url(self.endpoint, endpoint)
            this_attempt = request_timeout
            if deadline is not None:
                this_attempt = deadline - time.monotonic()
                if this_attempt <= 0:
                    raise RequestTimeoutError(
                        f"Request to {endpoint} exceeded its {timeout} s deadline"
                    )
                if attempt_timeout is not None:
                    this_attempt = min(this_attempt, attempt_timeout)

            try:
                # Use session for persistent connections instead of requests.request()
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data if data is not None else None,
                    params=params,
                    headers=headers,  # Pass additional headers if provided
                    timeout=this_attempt,
                )

                if response.status_code in [200, 201]:
                    return response.json() if response.content else None
                else:
                    error_data = response.json() if response.content else {}
                    # Self-healing: a 404 on a collection-scoped op means
                    # the collection no longer exists upstream (e.g. a
                    # cross-client delete). Drop the local caches so the
                    # next call doesn't keep believing the cached entry.
                    if evict_caches_on_404 is not None and response.status_code == 404:
                        self._schema_cache.pop(evict_caches_on_404, None)
                        self._payload_schema_cache.pop(evict_caches_on_404, None)
                    raise parse_error_response(error_data, response.status_code)

            except requests.Timeout:
                if deadline is not None and time.monotonic() >= deadline:
                    raise RequestTimeoutError(
                        f"Request to {endpoint} exceeded its {timeout} s deadline"
                    )
                raise RequestTimeoutError(
                    f"Request to {endpoint} timed out after {this_attempt} seconds"
                )
            except requests.ConnectionError as e:
                # Network connection errors should be retryable
                raise NetworkError(f"Network connection failed: {str(e)}")
            except requests.RequestException as e:
                # Other request errors - generic exception
                raise AetherfyVectorsException(f"Request failed: {str(e)}")

        # Apply retry logic only for write operations (POST, PUT)
        if enable_retry and method in ["POST", "PUT"]:
            return retry_with_backoff(
                make_single_request, max_retries=3, base_delay=1.0, deadline=deadline
            )
        else:
            return make_single_request()

    # Schema Cache Helpers

    def _get_cached_schema(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get cached schema for a collection if available."""
        return self._schema_cache.get(collection_name)

    def _fetch_and_cache_schema(self, collection_name: str) -> Dict[str, Any]:
        """Fetch collection schema from API and cache it with ETag.

        Caller passes the BARE collection name (no workspace slash).
        The wire URL uses the nested form via _build_collection_path
        when workspaced; the local _schema_cache key uses the slash-
        form scoped_name for collision-free lookups across workspaces
        with same-name collections.
        """
        scoped_name = self._scope_collection(collection_name)
        response = self._make_request(
            "GET",
            self._build_collection_path(collection_name),
            evict_caches_on_404=scoped_name,
        )

        # Extract schema info
        result = response.get("result", {})
        schema_version = response.get("schema_version")
        vector_config = result.get("config", {}).get("params", {}).get("vectors", {})

        schema = {
            "size": vector_config.get("size"),
            "distance": vector_config.get("distance"),
            "etag": schema_version,
            "full_config": result,
        }

        # Cache it under the scoped (slash-form) key — consistent with
        # _get_cached_schema lookups by scoped_name elsewhere.
        self._schema_cache[scoped_name] = schema
        return schema

    def clear_schema_cache(self, collection_name: Optional[str] = None) -> None:
        """Clear schema cache for a collection or all collections."""
        if collection_name:
            self._schema_cache.pop(collection_name, None)
        else:
            self._schema_cache.clear()

    def _normalize_distance_metric(
        self, distance: Union[str, DistanceMetric]
    ) -> DistanceMetric:
        """Normalize distance metric to proper DistanceMetric enum.

        Args:
            distance: Distance metric as string or DistanceMetric enum.

        Returns:
            Normalized DistanceMetric enum value.

        Raises:
            ValueError: If distance metric is invalid.
        """
        if isinstance(distance, DistanceMetric):
            return distance

        # Normalize string to the capitalized format the API expects
        distance_map = {
            "cosine": DistanceMetric.COSINE,
            "euclidean": DistanceMetric.EUCLIDEAN,
            "euclid": DistanceMetric.EUCLIDEAN,
            "dot": DistanceMetric.DOT,
            "manhattan": DistanceMetric.MANHATTAN,
        }

        normalized = distance_map.get(distance.lower())
        if normalized:
            return normalized

        # Try to create DistanceMetric directly (handles already-capitalized strings)
        try:
            return DistanceMetric(distance)
        except ValueError:
            raise ValueError(
                f"Invalid distance metric: {distance}. "
                f"Must be one of: {', '.join(d.value for d in DistanceMetric)}"
            )

    # Collection Management Methods

    def create_collection(
        self,
        collection_name: str,
        vectors_config: Union[VectorConfig, Dict[str, Any]],
        *,
        distance: Optional[DistanceMetric] = None,
        description: Optional[str] = None,
        regions: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> "Collection":
        """Create a new collection.

        Args:
            collection_name: Name of the collection to create.
            vectors_config: Vector configuration or dict with size/distance.
            distance: Distance metric (deprecated, use vectors_config).
            description: Optional collection description (max 500 characters).
            regions: Optional explicit placement regions for this collection.
                Omit to default to your full
                scope — the server resolves it and the returned Collection
                echoes the explicit list. Pass a subset of your scope to pin
                the collection to those regions. The list must be a subset of
                your account/workspace scope; an empty list is rejected by the
                server (422). Subset/empty validation is server-side. Distinct
                from the client constructor's ``api_region``, which selects the
                endpoint to connect to rather than where the collection lives.
            timeout: Deadline in seconds for this whole call, retries and
                backoff included, like qdrant-client's ``timeout``. Raises
                RequestTimeoutError once it passes. None keeps the client's
                timeout, which bounds each attempt rather than the call.
            **kwargs: qdrant-client ``create_collection`` arguments, checked
                against ``aetherfy_vectors.qdrant_compat``. None of its
                storage, index or replication settings can be applied, so each
                raises TypeError unless passed as None; any other name raises
                TypeError too.

        Returns:
            The created Collection, including its resolved ``regions`` list.

        Raises:
            ValidationError: If parameters are invalid.
            TypeError: For an unknown or unsupported keyword argument.
            AetherfyVectorsException: If creation fails.
        """
        check_qdrant_kwargs("create_collection", kwargs)
        validate_collection_name(collection_name)

        # Handle different input formats for compatibility
        if isinstance(vectors_config, dict):
            if "size" in vectors_config and "distance" in vectors_config:
                config = VectorConfig(
                    size=vectors_config["size"],
                    distance=self._normalize_distance_metric(
                        vectors_config["distance"]
                    ),
                )
            else:
                # Handle qdrant-client format
                size = vectors_config.get("size", vectors_config.get("vector_size"))
                if size is None:
                    raise ValueError("Vector size must be specified")
                config = VectorConfig(
                    size=int(size),
                    distance=self._normalize_distance_metric(
                        vectors_config.get("distance", "Cosine")
                    ),
                )
        elif isinstance(vectors_config, VectorConfig):
            config = vectors_config
        else:
            raise ValueError(
                "vectors_config must be VectorConfig instance or dictionary"
            )

        # Override distance if provided separately (for compatibility)
        if distance:
            config.distance = self._normalize_distance_metric(distance)

        scoped_name = self._scope_collection(collection_name)

        # Post-A/B: workspace lives in URL path, body name is bare.
        # POST /workspaces/{ws}/collections {name: bare, ...} for workspaced;
        # POST /collections {name: bare, ...} for workspaceless. vectordb
        # rejects any "/" in the body name.
        data: Dict[str, Any] = {
            "name": collection_name,
            "vectors": config.to_dict(),
            "description": description,
        }
        # §66: only send `regions` when the caller provided it. Omission
        # triggers server-side resolve-on-omit (defaults to the full scope);
        # an explicit empty list IS forwarded so the server returns its
        # 422 COLLECTION_REGIONS_EMPTY, rather than the SDK silently treating
        # [] as "all regions". Subset enforcement is the server's job.
        if regions is not None:
            data["regions"] = regions

        response = self._make_request(
            "POST", self._build_collections_list_path(), data, timeout=timeout
        )

        # Prepopulate the schema cache from the request we just authored.
        # GET /collections/<name> can be eventually consistent w.r.t. its
        # own writes — a read immediately after a successful create can
        # briefly return 4xx. Seeding the cache here makes the next
        # upsert/exists call hit local state instead of racing the
        # read-after-write window. We have ground truth (size + distance)
        # from the caller, so no extra round trip is needed. etag stays
        # None: upsert treats falsy etag as "no If-Match header," which
        # is correct until a real GET assigns a schema_version.
        distance_value = (
            config.distance.value
            if isinstance(config.distance, DistanceMetric)
            else config.distance
        )
        self._schema_cache[scoped_name] = {
            "size": config.size,
            "distance": distance_value,
            "etag": None,
            "full_config": {"config": {"params": {"vectors": config.to_dict()}}},
        }
        # §66 Option SDK-B: echo the resolved placement to the caller. The
        # create response carries the explicit `regions` the row was stored
        # with — the full scope on omit, the caller's subset otherwise (and
        # the existing row's stored list on an idempotent re-create).
        resolved_regions = (
            response.get("regions") if isinstance(response, dict) else None
        )
        return Collection(
            name=collection_name,
            config=config,
            description=description,
            regions=resolved_regions,
        )

    def delete_collection(
        self, collection_name: str, timeout: Optional[float] = None
    ) -> bool:
        """Delete a collection.

        Args:
            collection_name: Name of the collection to delete.
            timeout: Deadline in seconds for this whole call, retries and
                backoff included, like qdrant-client's ``timeout``. Raises
                RequestTimeoutError once it passes. None keeps the client's
                timeout, which bounds each attempt rather than the call.

        Returns:
            True if collection was deleted successfully.
        """
        validate_collection_name(collection_name)
        scoped_name = self._scope_collection(collection_name)
        # evict_caches_on_404 here covers the "collection already gone"
        # case (e.g. cross-client delete that beat us to it) so we don't
        # leave stale entries when our DELETE hits a 404.
        self._make_request(
            "DELETE",
            self._build_collection_path(collection_name),
            evict_caches_on_404=scoped_name,
            timeout=timeout,
        )
        # Drop both caches so a subsequent recreate-with-different-shape
        # doesn't see stale size/distance/etag/payload-schema entries.
        self._schema_cache.pop(scoped_name, None)
        self._payload_schema_cache.pop(scoped_name, None)
        return True

    def get_collections(self) -> List[Collection]:
        """Get list of all collections.

        Returns:
            List of Collection objects.
        """
        # Post-A/B: GET /workspaces/{ws}/collections returns ONLY this
        # workspace's collections with bare names; GET /collections returns
        # workspaceless collections (also bare). No client-side filtering
        # or name-unscoping needed.
        response = self._make_request("GET", self._build_collections_list_path())
        collections = response.get("collections", [])
        return [Collection.from_dict(col) for col in collections]

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection to check.

        Returns:
            True if collection exists, False if a 404 confirms it doesn't.

        Raises:
            AetherfyVectorsException: on any non-404 failure (auth errors,
                rate limits, service unavailability, network errors). A
                bare ``except`` here would silently mask "you got logged
                out" / "we're rate-limited" as "collection doesn't exist",
                producing confusing downstream behavior. Mirrors the JS
                SDK's collectionExists, which only swallows 404.
        """
        scoped_name = self._scope_collection(collection_name)
        # Fast path: if this client just created (or recently used) the
        # collection, the schema cache holds proof of existence. Skip the
        # network round trip and the read-after-write window of the
        # upstream store. delete_collection() clears the cache, so a
        # stale True after a remote delete is bounded to cross-client
        # deletes only — and any subsequent operation will surface the
        # real 404.
        if self._get_cached_schema(scoped_name) is not None:
            return True
        try:
            # evict_caches_on_404 is a no-op here (cache already empty if
            # we got past the fast-path check), but we pass it so the
            # contract "collection-scoped 404 → caches dropped" holds
            # uniformly across every collection-scoped call site.
            self._make_request(
                "GET",
                self._build_collection_path(collection_name),
                evict_caches_on_404=scoped_name,
            )
            return True
        except AetherfyVectorsException as e:
            if getattr(e, "status_code", None) == 404:
                return False
            raise

    def get_collection(self, collection_name: str) -> Collection:
        """Get collection information.

        Args:
            collection_name: Name of the collection.

        Returns:
            Collection object with details.
        """
        validate_collection_name(collection_name)
        scoped_name = self._scope_collection(collection_name)
        response = self._make_request(
            "GET",
            self._build_collection_path(collection_name),
            evict_caches_on_404=scoped_name,
        )
        # Post-A/B: vectordb returns the bare collection name (PG stores
        # `name` without workspace prefix; workspace is the workspace_id
        # join key). No client-side unscope needed.
        collection_data = response.get("result", response)
        return Collection.from_dict(collection_data)

    # Point Management Methods

    def upsert(
        self,
        collection_name: str,
        points: Sequence[Union[Point, Dict[str, Any]]],
        **kwargs,
    ) -> bool:
        """Insert or update points in a collection.

        Auto-chunks large batches into multiple HTTP requests to stay
        under the per-request byte cap (~24 MB, sized for the backend's
        90 s processing budget under wait=true). Most batches fit in one
        chunk; the chunker is transparent for small/medium upserts.

        Failure behaviour:
            - Transient errors (network blips, 5xx, 429) are auto-retried
              per chunk inside _make_request's retry budget.
            - Permanent errors on a chunk after retries: if there's only
              one chunk, the specific error (ValidationError,
              ServiceUnavailableError, etc.) is raised directly — same
              as pre-chunking behaviour.
            - Permanent errors when there are multiple chunks AND at
              least one chunk succeeded: raises PartialUpsertError
              carrying the saved count and the failed chunks' point IDs
              + errors. Callers can retry just those IDs (Qdrant upsert
              is idempotent by point ID so a retry of an already-saved
              point is also safe).
            - Permanent errors when ALL chunks fail (multi-chunk): also
              raises PartialUpsertError with saved=0 and all chunks'
              IDs in failed.

        Args:
            collection_name: Name of the target collection.
            points: List of Point objects or dictionaries.
            **kwargs: qdrant-client arguments of this method, checked against
                ``aetherfy_vectors.qdrant_compat``: ``wait`` is accepted (every
                write is already committed before the call returns),
                ``ordering`` only as 'weak', ``shard_key_selector`` only as
                None. Any other name raises TypeError.

        Returns:
            True if all points were saved.

        Raises:
            SchemaValidationError: If payloads fail schema validation.
            PartialUpsertError: When multi-chunk upsert has any failed
                chunks.
            ValidationError: Single-chunk validation / 400 errors.
            ValueError: Single-chunk 400 (re-raised for backward
                compatibility).
            TypeError: For an unknown or unsupported keyword argument.
        """
        check_qdrant_kwargs("upsert", kwargs)
        validate_collection_name(collection_name)

        scoped_name = self._scope_collection(collection_name)

        # Get vector config schema (from cache or fetch). _fetch_and_cache_schema
        # computes its own scoped_name for the cache key from the bare
        # collection_name, and builds the wire URL via _build_collection_path.
        schema = self._get_cached_schema(scoped_name)
        if not schema:
            schema = self._fetch_and_cache_schema(collection_name)

        # Validate vector dimensions
        expected_dim = schema.get("size")
        if expected_dim:
            for point in points:
                vector = (
                    point.get("vector") if isinstance(point, dict) else point.vector
                )
                if not vector or not isinstance(vector, (list, tuple)):
                    raise ValueError("Each point must have a vector array")

                if len(vector) != expected_dim:
                    raise ValueError(
                        f"Vector dimension mismatch: expected {expected_dim}, got {len(vector)}"
                    )

        # Convert Point objects to dictionaries if needed
        formatted_points = []
        for point in points:
            if isinstance(point, Point):
                formatted_points.append(point.to_dict())
            elif isinstance(point, dict):
                formatted_points.append(point)
            else:
                raise ValueError("Points must be Point objects or dictionaries")

        # Get payload schema for validation (if exists)
        payload_schema_data = self._payload_schema_cache.get(scoped_name)
        if payload_schema_data is None:  # Not cached yet (different from cached None)
            # Try to fetch schema from server
            try:
                # get_schema will handle scoping internally
                schema_result = self.get_schema(collection_name)
                if schema_result is None:
                    # Cache the fact that no schema exists to avoid repeated fetches
                    self._payload_schema_cache[scoped_name] = {
                        "schema": None,
                        "enforcement_mode": "off",
                        "etag": None,
                    }
                payload_schema_data = self._payload_schema_cache.get(scoped_name)
            except Exception:
                # Error fetching schema - cache None to avoid retrying
                self._payload_schema_cache[scoped_name] = {
                    "schema": None,
                    "enforcement_mode": "off",
                    "etag": None,
                }
                payload_schema_data = None

        # Client-side payload validation
        if payload_schema_data and payload_schema_data["schema"]:
            enforcement_mode = payload_schema_data.get("enforcement_mode", "off")

            # Only validate if enforcement is not 'off'
            if enforcement_mode != "off":
                validation_errors = validate_vectors(
                    formatted_points, payload_schema_data["schema"]
                )
                if validation_errors:
                    # Only raise error in strict mode
                    if enforcement_mode == "strict":
                        # Convert to dict format for exception
                        errors_dict = [e.to_dict() for e in validation_errors]
                        raise SchemaValidationError(errors_dict)
                    # In warn mode, just log the warnings (client-side logging would go here)
                    # For now, we allow the request to proceed

        # Validate and format points
        formatted_points = format_points_for_upsert(formatted_points)

        # Chunk by byte size. Most upserts produce a single chunk; the
        # multi-chunk path only fires for batches large enough to risk
        # the backend's per-request processing budget (>~24 MB wire size).
        chunks = list(chunk_points_by_bytes(formatted_points, MAX_REQUEST_BYTES))

        if len(chunks) == 1:
            # Single-chunk fast path: preserves pre-chunking behaviour
            # exactly — specific exceptions (ValidationError,
            # NetworkError, etc.) propagate directly without
            # PartialUpsertError wrapping.
            return self._upload_points_chunk(
                scoped_name,
                collection_name,
                chunks[0],
                schema,
                payload_schema_data,
            )

        # Multi-chunk path: per-chunk error tracking. Each chunk runs
        # through the same upload+retry+412 handling as the single-chunk
        # path; only the outer failure aggregation differs.
        saved = 0
        failed: List[Dict[str, Any]] = []

        for chunk in chunks:
            try:
                self._upload_points_chunk(
                    scoped_name,
                    collection_name,
                    chunk,
                    schema,
                    payload_schema_data,
                )
                saved += len(chunk)
            except AetherfyVectorsException as e:
                # Covers ValidationError, NetworkError, ServiceUnavailableError,
                # RequestTimeoutError, SchemaValidationError, etc. — all
                # SDK-domain errors.
                failed.append({"point_ids": [p["id"] for p in chunk], "error": e})
            except ValueError as e:
                # _upload_points_chunk raises ValueError on a 400 (kept
                # for backward compatibility with the pre-chunking
                # contract — see _upload_points_chunk's 400 branch).
                # Wrap as ValidationError so failed-list entries are
                # uniform AetherfyVectorsException instances.
                failed.append(
                    {
                        "point_ids": [p["id"] for p in chunk],
                        "error": ValidationError(str(e)),
                    }
                )
            # Programming errors (TypeError, AttributeError, etc.)
            # intentionally propagate. They indicate SDK bugs and should
            # not be silently buried in a PartialUpsertError's failed list.

        if failed:
            raise PartialUpsertError(saved, len(formatted_points), failed)
        return True

    def _upload_points_chunk(
        self,
        scoped_name: str,
        original_collection_name: str,
        chunk: List[Dict[str, Any]],
        schema: Dict[str, Any],
        payload_schema_data: Optional[Dict[str, Any]],
    ) -> bool:
        """Per-chunk upload.

        Mirrors the pre-chunking single-PUT logic exactly: If-Match
        headers from schema ETags, retry on 412 with fresh schema, per-
        status handling (400, 412), 404 cache self-heal via
        _make_request's evict_caches_on_404. Returns True on 200; raises
        on failure.

        Extracted so the multi-chunk loop can call it per-chunk and
        aggregate failures into PartialUpsertError without duplicating
        the ~80 lines of error-handling logic.
        """
        # Make request with If-Match headers (for both schemas)
        data = {"points": chunk}

        # Add If-Match headers if we have ETags
        extra_headers: Dict[str, str] = {}
        if schema.get("etag"):
            extra_headers["If-Match"] = schema["etag"]

        # Add schema ETag for payload validation
        if payload_schema_data and payload_schema_data.get("etag"):
            extra_headers["If-Match"] = payload_schema_data["etag"]

        try:
            # Pass If-Match header directly to the request
            self._make_request(
                "PUT",
                self._build_collection_path(original_collection_name, "/points"),
                data,
                headers=extra_headers if extra_headers else None,
                evict_caches_on_404=scoped_name,
            )
            return True

        except ValidationError as e:
            # Handle 412 Precondition Failed (schema changed)
            if e.status_code == 412:
                self.clear_schema_cache(scoped_name)
                self._payload_schema_cache.pop(original_collection_name, None)

                # Fetch updated schema and re-validate
                updated_schema = None
                try:
                    self.get_schema(original_collection_name)
                    updated_schema = self._payload_schema_cache.get(
                        original_collection_name
                    )
                    if updated_schema and updated_schema["schema"]:
                        enforcement_mode = updated_schema.get("enforcement_mode", "off")
                        if enforcement_mode != "off":
                            validation_errors = validate_vectors(
                                chunk, updated_schema["schema"]
                            )
                            if validation_errors and enforcement_mode == "strict":
                                errors_dict = [e.to_dict() for e in validation_errors]
                                raise SchemaValidationError(errors_dict)
                except SchemaValidationError:
                    # Re-raise schema validation errors
                    raise
                except Exception:
                    # Ignore other errors during schema refresh
                    updated_schema = self._payload_schema_cache.get(
                        original_collection_name
                    )

                # Retry the upsert with updated schema
                try:
                    extra_headers_retry: Dict[str, str] = {}
                    if schema.get("etag"):
                        extra_headers_retry["If-Match"] = schema["etag"]
                    if updated_schema and updated_schema.get("etag"):
                        extra_headers_retry["If-Match"] = updated_schema["etag"]

                    self._make_request(
                        "PUT",
                        self._build_collection_path(
                            original_collection_name, "/points"
                        ),
                        data,
                        headers=(extra_headers_retry if extra_headers_retry else None),
                        evict_caches_on_404=scoped_name,
                    )
                    return True
                except Exception:
                    # If retry also fails, raise the original 412 error
                    raise ValidationError(
                        f"Schema changed for collection '{original_collection_name}'. Please retry your request.",
                        status_code=412,
                    )

            # Handle 400 Bad Request (validation error from backend or client-side)
            # Re-raise as ValueError for backward compatibility
            raise ValueError(str(e))

        except AetherfyVectorsException:
            # Re-raise other errors
            raise

    def delete(
        self,
        collection_name: str,
        points_selector: Union[List[Union[str, int]], Dict[str, Any]],
        **kwargs,
    ) -> bool:
        """Delete points from a collection.

        Args:
            collection_name: Name of the collection.
            points_selector: List of point IDs or filter conditions.
            **kwargs: qdrant-client arguments of this method, checked against
                ``aetherfy_vectors.qdrant_compat``: ``wait`` is accepted (every
                write is already committed before the call returns),
                ``ordering`` only as 'weak', ``shard_key_selector`` only as
                None. Any other name raises TypeError.

        Returns:
            True if deletion was successful.

        Raises:
            TypeError: For an unknown or unsupported keyword argument.
        """
        check_qdrant_kwargs("delete", kwargs)
        validate_collection_name(collection_name)

        scoped_name = self._scope_collection(collection_name)

        if isinstance(points_selector, list):
            # Delete by point IDs
            for point_id in points_selector:
                validate_point_id(point_id)
            data: Dict[str, Any] = {"points": points_selector}
        else:
            # Delete by filter
            data = {"filter": serialize_filter(points_selector, "delete")}

        self._make_request(
            "POST",
            self._build_collection_path(collection_name, "/points/delete"),
            data,
            evict_caches_on_404=scoped_name,
        )
        return True

    # ------------------------------------------------------------------
    # Payload mutation
    #
    # Three helpers for the three payload-mutation endpoints. Server-side
    # cap: body.points.length <= 512.
    # ------------------------------------------------------------------

    def set_payload(
        self,
        collection_name: str,
        payload: Dict[str, Any],
        points: List[Union[str, int]],
        key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Set (additive merge) payload keys on a list of points.

        POST /collections/{name}/points/payload — keys not on the point are
        added; keys that already exist are overwritten with the new value;
        keys present on the point but not in `payload` are left untouched.

        Args:
            collection_name: Target collection.
            payload: Payload object to merge into each point's payload.
            points: Point IDs to update. Server caps at 512.
            key: Optional nested-path target. When set, the merge happens
                inside ``payload[key]`` instead of at the top level — every
                key in ``payload`` is merged into the existing nested object,
                preserving sibling keys not mentioned in the partial. Used
                by ``merge_metadata`` to get atomic per-point partial-merge
                semantics under ``payload.metadata``.
            **kwargs: qdrant-client arguments of this method, checked against
                ``aetherfy_vectors.qdrant_compat``: ``wait`` is accepted (every
                write is already committed before the call returns),
                ``ordering`` only as 'weak', ``shard_key_selector`` only as
                None. Any other name raises TypeError.

        Returns:
            Server response dict.
        """
        check_qdrant_kwargs("set_payload", kwargs)
        validate_collection_name(collection_name)
        for pid in points:
            validate_point_id(pid)
        scoped_name = self._scope_collection(collection_name)
        data: Dict[str, Any] = {"payload": payload, "points": points}
        if key is not None:
            data["key"] = key
        response = self._make_request(
            "POST",
            self._build_collection_path(collection_name, "/points/payload"),
            data,
            evict_caches_on_404=scoped_name,
        )
        return response.get("result", response) if response else {}

    def overwrite_payload(
        self,
        collection_name: str,
        payload: Dict[str, Any],
        points: List[Union[str, int]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Replace the entire payload on a list of points.

        PUT /collections/{name}/points/payload — keys present on the point
        but absent from `payload` are REMOVED. Use this when you want the
        payload to be exactly `payload` after the call.

        Args:
            collection_name: Target collection.
            payload: The complete payload each point ends up with.
            points: Point IDs to update. Server caps at 512.
            **kwargs: qdrant-client arguments of this method, checked against
                ``aetherfy_vectors.qdrant_compat``: ``wait`` is accepted (every
                write is already committed before the call returns),
                ``ordering`` only as 'weak', ``shard_key_selector`` only as
                None. Any other name raises TypeError.
        """
        check_qdrant_kwargs("overwrite_payload", kwargs)
        validate_collection_name(collection_name)
        for pid in points:
            validate_point_id(pid)
        scoped_name = self._scope_collection(collection_name)
        data = {"payload": payload, "points": points}
        response = self._make_request(
            "PUT",
            self._build_collection_path(collection_name, "/points/payload"),
            data,
            evict_caches_on_404=scoped_name,
        )
        return response.get("result", response) if response else {}

    def delete_payload(
        self,
        collection_name: str,
        keys: List[str],
        points: List[Union[str, int]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Delete specific payload keys from a list of points.

        POST /collections/{name}/points/payload/delete — only the named
        keys are removed; other keys on each point's payload are preserved.

        Args:
            collection_name: Target collection.
            keys: Payload keys to remove.
            points: Point IDs to update. Server caps at 512.
            **kwargs: qdrant-client arguments of this method, checked against
                ``aetherfy_vectors.qdrant_compat``: ``wait`` is accepted (every
                write is already committed before the call returns),
                ``ordering`` only as 'weak', ``shard_key_selector`` only as
                None. Any other name raises TypeError.
        """
        check_qdrant_kwargs("delete_payload", kwargs)
        validate_collection_name(collection_name)
        for pid in points:
            validate_point_id(pid)
        scoped_name = self._scope_collection(collection_name)
        data = {"keys": keys, "points": points}
        response = self._make_request(
            "POST",
            self._build_collection_path(collection_name, "/points/payload/delete"),
            data,
            evict_caches_on_404=scoped_name,
        )
        return response.get("result", response) if response else {}

    # ------------------------------------------------------------------
    # Payload field indexes
    #
    # An unindexed payload filter is SCANNED, not looked up. A tenant key
    # you filter on for every read (e.g. a per-conversation id) wants an
    # index the moment the collection holds more than one tenant's rows.
    # ------------------------------------------------------------------

    def create_field_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: Union[str, Dict[str, Any]] = "keyword",
        timeout: Optional[float] = None,
    ) -> bool:
        """Create a payload index on one field, and return once it is built.

        PUT /collections/{name}/index with ``{field_name, field_schema}``.
        The server waits for the build for up to 25 s. A build that takes
        longer is answered "acknowledged" (still building), and this method
        then sends the create again, which waits for the running build, until
        the answer is "completed". So when it returns, a filter or an
        ``order_by`` scroll on the key works in the region that answered.
        Re-creating an existing index with the same schema returns at once.

        There is deliberately no way to return before the build finishes: a
        caller that did would scroll into "No range index for order_by key".
        The wait is always bounded: by ``timeout``, or else by
        ``INDEX_DEFAULT_DEADLINE_S`` (10 minutes). An "acknowledged" that
        came back before the server's 25 s wait was up (so the server did not
        hold the create) is re-sent only after a pause, 1 s doubling to 10 s.

        Args:
            collection_name: Name of the collection.
            field_name: The payload key to index. Dotted paths address a
                nested key (``"metadata.tag"``).
            field_schema: Index type. A string for the simple types
                (``"keyword"``, ``"integer"``, ``"float"``, ``"bool"``,
                ``"geo"``, ``"datetime"``, ``"uuid"``, ``"text"``), or a
                dict for the parameterised forms. Forwarded verbatim.
            timeout: Deadline in seconds for this whole call, every create,
                retry and pause included: a finite number above 0. None means
                ``INDEX_DEFAULT_DEADLINE_S`` (600 s). Each create has an HTTP
                timeout of ``INDEX_ATTEMPT_TIMEOUT_S`` (45 s), or the
                constructor's timeout if that is longer, or what remains of
                the deadline if that is shorter.

        Returns:
            True, and only once the index is built. It never returns False.

        Raises:
            ValidationError: ``timeout`` is not a finite number above 0.
                Nothing is sent.
            RequestTimeoutError: The deadline (``timeout``, or the default)
                passed while the index was still building. The build carries
                on server-side, and calling this method again waits for it.
                If no answer at all came back within the deadline, the
                message says so instead: it is then not known whether the
                create was taken.
            AetherfyVectorsException: The server answered a status other than
                "completed" or "acknowledged"; the index is not confirmed.
        """
        validate_collection_name(collection_name)
        if not isinstance(field_name, str) or not field_name:
            raise ValidationError("field_name must be a non-empty string")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValidationError(
                f"timeout must be a finite number of seconds above 0, got {timeout!r}"
            )
        scoped_name = self._scope_collection(collection_name)
        if timeout is None:
            timeout = self.INDEX_DEFAULT_DEADLINE_S
        deadline = time.monotonic() + timeout
        attempt_timeout = max(self.timeout, self.INDEX_ATTEMPT_TIMEOUT_S)
        pause = self.INDEX_RESEND_PAUSE_FIRST_S
        still_building = RequestTimeoutError(
            f"The payload index on {field_name!r} in collection "
            f"{collection_name!r} is still building after the {timeout:g} s "
            "deadline. The build carries on server-side; calling "
            "create_field_index again waits for it."
        )
        no_answer = RequestTimeoutError(
            f"The payload index create on {field_name!r} in collection "
            f"{collection_name!r} got no answer within the {timeout:g} s "
            "deadline, so it is not known whether it was taken. "
            "Calling create_field_index again is safe."
        )
        acknowledged = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Only a build some answer reported is claimed.
                raise still_building if acknowledged else no_answer
            started = time.monotonic()
            try:
                response = self._make_request(
                    "PUT",
                    self._build_collection_path(collection_name, "/index"),
                    {"field_name": field_name, "field_schema": field_schema},
                    evict_caches_on_404=scoped_name,
                    timeout=remaining,
                    attempt_timeout=attempt_timeout,
                )
            except RequestTimeoutError:
                # Before the deadline, this is one create outliving its own
                # HTTP timeout, and that error stands. At the deadline, after
                # an "acknowledged" the build is known to be running; on the
                # first create it is not known the create was even taken, so
                # that is not claimed.
                if time.monotonic() < deadline:
                    raise
                raise (still_building if acknowledged else no_answer) from None
            result = response.get("result") if isinstance(response, dict) else None
            status = result.get("status") if isinstance(result, dict) else None
            if status == "completed":
                return True
            if status != "acknowledged":
                raise AetherfyVectorsException(
                    f"create_field_index got status {status!r} for the payload "
                    f"index on {field_name!r} in collection {collection_name!r}, "
                    "expected 'completed' or 'acknowledged'; the index is not "
                    "confirmed built."
                )
            acknowledged = True
            if time.monotonic() - started < self.INDEX_WAIT_BUDGET_S:
                # The server did not hold this create, so re-sending at once
                # would hammer it.
                time.sleep(min(pause, max(deadline - time.monotonic(), 0.0)))
                pause = min(pause * 2, self.INDEX_RESEND_PAUSE_MAX_S)

    def delete_field_index(self, collection_name: str, field_name: str) -> bool:
        """Drop the payload index on one field.

        DELETE /collections/{name}/index/{field_name}. Returns True when the
        collection exists, INCLUDING when that field was never indexed: the
        server answers that with 200, like a real drop. Returns False only
        when the collection itself does not exist (404).

        One request, not retried. The server may hold it as long as a create
        (up to 25 s, plus a forward), so its HTTP timeout is
        ``INDEX_ATTEMPT_TIMEOUT_S`` (45 s), or the constructor's timeout if
        that is longer.
        """
        validate_collection_name(collection_name)
        if not isinstance(field_name, str) or not field_name:
            raise ValidationError("field_name must be a non-empty string")
        from urllib.parse import quote

        scoped_name = self._scope_collection(collection_name)
        try:
            self._make_request(
                "DELETE",
                self._build_collection_path(
                    collection_name, f"/index/{quote(field_name, safe='')}"
                ),
                evict_caches_on_404=scoped_name,
                attempt_timeout=max(self.timeout, self.INDEX_ATTEMPT_TIMEOUT_S),
            )
            return True
        except AetherfyVectorsException as e:
            if getattr(e, "status_code", None) == 404:
                return False
            raise

    def merge_metadata(
        self,
        collection_name: str,
        point_id: Union[str, int],
        partial: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Additive merge into existing ``payload.metadata``.

        ``merge_metadata({tag: 'x'})`` adds/updates the listed keys and
        leaves every other key untouched. Use ``set_payload`` with the
        full metadata object if you want to fully replace the metadata
        sub-key. Concurrent patches to different keys all land
        atomically; concurrent writes to the same key resolve via
        last-writer-wins per the storage operation order. Raises
        ``PointNotFoundError`` if the point doesn't exist.
        """
        if not isinstance(partial, dict):
            raise TypeError("partial must be a dict")
        try:
            return self.set_payload(
                collection_name,
                payload=partial,
                points=[point_id],
                key="metadata",
            )
        except AetherfyVectorsException as e:
            if e.status_code == 404 and not isinstance(
                e, (PointNotFoundError, CollectionNotFoundError)
            ):
                raise PointNotFoundError(str(point_id), collection_name) from e
            raise

    def delete_metadata_keys(
        self,
        collection_name: str,
        point_id: Union[str, int],
        keys: List[str],
    ) -> Dict[str, Any]:
        """Removes the listed keys from ``payload.metadata``.

        Keys not in the list are left untouched. Raises
        ``PointNotFoundError`` if the point doesn't exist.
        """
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise TypeError("keys must be a list of strings")
        dotted = [f"metadata.{k}" for k in keys]
        try:
            return self.delete_payload(
                collection_name,
                keys=dotted,
                points=[point_id],
            )
        except AetherfyVectorsException as e:
            if e.status_code == 404 and not isinstance(
                e, (PointNotFoundError, CollectionNotFoundError)
            ):
                raise PointNotFoundError(str(point_id), collection_name) from e
            raise

    def retrieve(
        self,
        collection_name: str,
        ids: List[Union[str, int]],
        with_payload: bool = True,
        with_vectors: bool = False,
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Retrieve points by IDs.

        Args:
            collection_name: Name of the collection.
            ids: List of point IDs to retrieve.
            with_payload: Include payload in results.
            with_vectors: Include vectors in results.
            timeout: Deadline in seconds for this whole call, retries and
                backoff included, like qdrant-client's ``timeout``. Raises
                RequestTimeoutError once it passes. None keeps the client's
                timeout, which bounds each attempt rather than the call.
            **kwargs: qdrant-client arguments of this method, checked against
                ``aetherfy_vectors.qdrant_compat``: ``consistency`` and
                ``shard_key_selector`` are accepted only as None. Any other
                name raises TypeError.

        Returns:
            List of retrieved points.
        """
        check_qdrant_kwargs("retrieve", kwargs)
        validate_collection_name(collection_name)

        scoped_name = self._scope_collection(collection_name)

        for point_id in ids:
            validate_point_id(point_id)

        data = {"ids": ids, "with_payload": with_payload, "with_vector": with_vectors}

        # Dedicated retrieve URL — POST /collections/<name>/points was
        # previously dual-purpose (upsert vs retrieve, distinguished by
        # body shape). Backend now serves retrieve at /points/retrieve so
        # /points can be unambiguously upsert (and stream-parsed).
        response = self._make_request(
            "POST",
            self._build_collection_path(collection_name, "/points/retrieve"),
            data,
            evict_caches_on_404=scoped_name,
            timeout=timeout,
        )
        return response.get("result", [])

    # Search Methods

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        *,
        limit: int = 10,
        offset: int = 0,
        query_filter: Optional[Union[Filter, Dict[str, Any]]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None,
        search_params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """Search for similar vectors in a collection.

        Sent as ``POST /collections/{name}/points/query``, Qdrant's search
        route (the API refuses the retired ``/points/search`` with 410
        ROUTE_RETIRED). The call and its arguments are unchanged: the vector
        goes in the body as ``query``, the other arguments keep their wire
        names, and the matches are read from ``result.points``.

        Args:
            collection_name: Name of the collection to search in.
            query_vector: Query vector for similarity search.
            limit: Maximum number of results to return.
            offset: Number of results to skip.
            query_filter: Filter conditions for search.
            with_payload: Include payload in results.
            with_vectors: Include vectors in results.
            score_threshold: Minimum score threshold for results.
            search_params: Search-time engine parameters, sent verbatim as the
                request body's `params` field. The headline use is
                `{"hnsw_ef": 256}`: a larger ef makes the HNSW graph walk visit
                more candidates, buying recall at the cost of latency (and a
                smaller ef does the reverse). Omit it to keep the server-side
                default of hnsw_ef=100, which is the tuned default and measures
                recall@10 ≈ 0.996 on a realistic corpus. Cache note: the server
                cache key is derived from the request body bytes, so the same
                query at a different ef is a separate cache entry — a
                params-varying call can never hit an entry stored under
                different params. Contents are not validated or translated
                here; the API and Qdrant own the schema.
            timeout: Deadline in seconds for this whole call, retries and
                backoff included, like qdrant-client's ``timeout``. Raises
                RequestTimeoutError once it passes. None keeps the client's
                timeout, which bounds each attempt rather than the call.
            **kwargs: qdrant-client arguments of this method, checked against
                ``aetherfy_vectors.qdrant_compat``: ``append_payload`` only as
                True (use ``with_payload``), ``consistency`` and
                ``shard_key_selector`` only as None.

        Returns:
            List of SearchResult objects.

        Raises:
            TypeError: If an unknown or unsupported keyword argument is passed.
                ``**kwargs`` is not a sink: a misspelled or unsupported option
                fails loudly instead of being silently dropped from the
                request body.
        """
        check_qdrant_kwargs("search", kwargs)
        validate_collection_name(collection_name)
        validate_vector(query_vector)

        scoped_name = self._scope_collection(collection_name)

        data = {
            "query": query_vector,
            "limit": limit,
            "offset": offset,
            "with_payload": with_payload,
            "with_vector": with_vectors,
        }

        if query_filter:
            data["filter"] = serialize_filter(query_filter, "search")

        if score_threshold is not None:
            data["score_threshold"] = score_threshold

        # Untranslated pass-through. Enumerating or validating param names here
        # would make the SDK a compatibility treadmill behind Qdrant's own
        # schema; the backend forwards the search body verbatim, so anything
        # the engine accepts works without an SDK release. Added last and only
        # when present, so the default body is byte-for-byte what it was
        # before this option existed (server cache keys are body-derived).
        if search_params is not None:
            data["params"] = search_params

        response = self._make_request(
            "POST",
            self._build_collection_path(collection_name, "/points/query"),
            data,
            evict_caches_on_404=scoped_name,
            timeout=timeout,
        )

        results = []
        for result in (response.get("result") or {}).get("points", []):
            results.append(SearchResult.from_dict(result))

        return results

    def scroll(
        self,
        collection_name: str,
        *,
        limit: int = 10,
        offset: Optional[Union[str, int]] = None,
        scroll_filter: Optional[Union[Filter, Dict[str, Any]]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        order_by: Optional[Union[str, Dict[str, Any]]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Scroll through points in a collection (Qdrant-compatible pagination).

        Unlike `search`, scroll iterates over points without vector similarity —
        used for bulk reads, history fetches, and payload-filtered iteration.

        Args:
            collection_name: Name of the collection.
            limit: Maximum points per page.
            offset: Pagination cursor from a previous call's `next_page_offset`.
            scroll_filter: Payload filter conditions.
            with_payload: Include payload in results.
            with_vectors: Include vectors in results.
            order_by: Order the points by a payload key instead of by id, as
                qdrant-client's ``order_by``: a key name, or a dict such as
                ``{"key": "ts", "direction": "desc"}``. Sent verbatim as the
                body's ``order_by``; Qdrant owns its schema (the key needs a
                payload index, and it cannot be combined with ``offset``).
            timeout: Deadline in seconds for this whole call, retries and
                backoff included, like qdrant-client's ``timeout``. Raises
                RequestTimeoutError once it passes. None keeps the client's
                timeout, which bounds each attempt rather than the call.
            **kwargs: qdrant-client arguments of this method, checked against
                ``aetherfy_vectors.qdrant_compat``: ``consistency`` and
                ``shard_key_selector`` are accepted only as None. Any other
                name raises TypeError.

        Returns:
            Dict with `points` (list of point dicts) and `next_page_offset`
            (cursor or None if this was the last page).

        Raises:
            TypeError: For an unknown or unsupported keyword argument.
            ValidationError: If ``order_by`` is neither a str nor a dict.
        """
        check_qdrant_kwargs("scroll", kwargs)
        validate_collection_name(collection_name)

        scoped_name = self._scope_collection(collection_name)

        data: Dict[str, Any] = {
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vectors,
        }
        if offset is not None:
            data["offset"] = offset
        if scroll_filter:
            data["filter"] = serialize_filter(scroll_filter, "scroll")
        if order_by is not None:
            # A str or a plain dict, like the filters: qdrant-client's model
            # objects are not serialised here.
            if not isinstance(order_by, (str, dict)):
                raise ValidationError(
                    "scroll: order_by must be a payload key (str) or a dict "
                    "such as {'key': ..., 'direction': ...}, got "
                    f"{type(order_by).__name__}."
                )
            data["order_by"] = order_by

        response = self._make_request(
            "POST",
            self._build_collection_path(collection_name, "/points/scroll"),
            data,
            evict_caches_on_404=scoped_name,
            timeout=timeout,
        )

        # Scroll response shape: {"result": {"points": [...], "next_page_offset": ...}, ...}
        result = response.get("result") or {}
        return {
            "points": result.get("points", []),
            "next_page_offset": result.get("next_page_offset"),
        }

    def scroll_iter(
        self,
        collection_name: str,
        *,
        batch_size: int = 256,
        scroll_filter: Optional[Union[Filter, Dict[str, Any]]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """Auto-paginating scroll. Yields each point one at a time, fetches the
        next page transparently, stops when next_page_offset is None.

        Why this exists: scroll() is single-shot. Without this, callers reach
        for scroll(limit=very_large) which (a) blows past the server-side
        1000-point cap, (b) creates a 10MB+ response that hits the
        RESPONSE_TOO_LARGE 413 from the backend, and (c) loads everything
        into memory at once. The iterator gives them a paging helper that's
        correct by default.

        Args:
            collection_name: Collection to iterate.
            batch_size: Points per server round-trip. Default 256, comfortably
                under the server's 1000 cap. Pass `batch_size` if you need a
                different page size — the underlying scroll's `limit` is owned
                by the iterator and not caller-controllable.
            scroll_filter: Optional payload filter, same shape as scroll().
            with_payload: Forwarded to each scroll() call.
            with_vectors: Forwarded to each scroll() call.

        Yields:
            Each point dict from each page, in order.

        Raises:
            ValueError: if batch_size is not in 1..1000.
        """
        # The kwarg allowlist is intentional: no **kwargs, so unknown options
        # (notably `limit` and `offset`) raise TypeError automatically. The
        # iterator owns those; mid-stream offsets are a footgun and a manual
        # `limit` would defeat the size invariant the cap enforces.
        if not (1 <= batch_size <= 1000):
            raise ValueError(
                f"batch_size must be 1-1000 (server cap), got {batch_size}"
            )

        cursor: Optional[Union[str, int]] = None
        while True:
            page = self.scroll(
                collection_name,
                limit=batch_size,
                offset=cursor,
                scroll_filter=scroll_filter,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
            for point in page.get("points", []):
                yield point
            cursor = page.get("next_page_offset")
            if cursor is None:
                return

    def count(
        self,
        collection_name: str,
        count_filter: Optional[Union[Filter, Dict[str, Any]]] = None,
        exact: bool = True,
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> int:
        """Count points in collection.

        Args:
            collection_name: Name of the collection.
            count_filter: Filter conditions for counting. Accepts a
                ``Filter`` or a plain dict, matching search/scroll/delete —
                it used to take a dict only.
            exact: Whether to return exact count.
            timeout: Deadline in seconds for this whole call, retries and
                backoff included, like qdrant-client's ``timeout``. Raises
                RequestTimeoutError once it passes. None keeps the client's
                timeout, which bounds each attempt rather than the call.
            **kwargs: qdrant-client arguments of this method, checked against
                ``aetherfy_vectors.qdrant_compat``: ``shard_key_selector`` is
                accepted only as None. Any other name raises TypeError.

        Returns:
            Number of points matching the filter.
        """
        check_qdrant_kwargs("count", kwargs)
        validate_collection_name(collection_name)

        scoped_name = self._scope_collection(collection_name)

        data: Dict[str, Any] = {"exact": exact}
        if count_filter:
            data["filter"] = serialize_filter(count_filter, "count")

        response = self._make_request(
            "POST",
            self._build_collection_path(collection_name, "/points/count"),
            data,
            evict_caches_on_404=scoped_name,
            timeout=timeout,
        )
        # The wire response is `{"result": {"count": N}, "status": "ok"}`,
        # not flat — the count lives under `result`. Mirrors the JS SDK.
        return response.get("result", {}).get("count", 0)

    # Schema Management Methods

    def get_schema(self, collection_name: str) -> Optional[Schema]:
        """Get schema for a collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            The collection's Schema, or None when no schema is defined.

        Raises:
            AetherfyVectorsException: On any non-404 failure.

        A missing schema does NOT raise. The backend answers 404 for both
        "collection exists but has no schema" and "collection is gone", and
        neither is an error for a getter — the 404 is caught and None is
        returned (the COLLECTION_NOT_FOUND case additionally evicts the local
        caches). This docstring used to promise SchemaNotFoundError as well as
        None, which cannot both be true; delete_schema is the sibling that
        really does raise it, and the two are deliberately different.
        """
        validate_collection_name(collection_name)

        scoped_name = self._scope_collection(collection_name)

        try:
            response = self._make_request(
                "GET", f"schema/{quote_collection_name(scoped_name)}"
            )

            schema = Schema.from_dict(response["schema"])
            schema.description = response.get("description")
            etag = response["etag"]
            enforcement_mode = response["enforcement_mode"]

            # Cache it (use scoped name for cache)
            self._payload_schema_cache[scoped_name] = {
                "schema": schema,
                "etag": etag,
                "enforcement_mode": enforcement_mode,
            }

            return schema

        except AetherfyVectorsException as e:
            if e.status_code == 404:
                # Backend disambiguates the two 404 cases via error.code:
                #   COLLECTION_NOT_FOUND → collection is gone, evict caches
                #   SCHEMA_NOT_DEFINED → collection exists but no schema (legit)
                # Without the code (older backend or unstructured body) we
                # treat 404 as "no schema set" (return None) without
                # evicting — same shape as today's behavior. The eviction
                # path only fires when the backend explicitly tells us
                # the collection is gone.
                error_code = self._extract_error_code(e)
                if error_code == "COLLECTION_NOT_FOUND":
                    self._schema_cache.pop(scoped_name, None)
                    self._payload_schema_cache.pop(scoped_name, None)
                return None
            raise

    @staticmethod
    def _extract_error_code(exc: "AetherfyVectorsException") -> Optional[str]:
        """Pull error.code out of an exception's parsed details.

        parse_error_response stashes the backend's error.code under
        either exc.error_code (flat shape) or exc.details["code"]
        (nested {"error": {"code": ...}} shape). Both are checked so
        SDK code can read the code without knowing which body shape
        the backend used.
        """
        code = getattr(exc, "error_code", None)
        if code:
            return code
        details = getattr(exc, "details", None) or {}
        if isinstance(details, dict):
            return details.get("code")
        return None

    def set_schema(
        self,
        collection_name: str,
        schema: Schema,
        enforcement: str = "off",
        description: Optional[str] = None,
    ) -> str:
        """Set schema for a collection.

        Args:
            collection_name: Name of the collection.
            schema: Schema definition.
            enforcement: Enforcement mode - 'off', 'warn', or 'strict' (default: 'off').
            description: Optional schema description (max 500 characters).

        Returns:
            ETag of the new schema.

        Raises:
            ValidationError: If schema or enforcement mode is invalid.
        """
        validate_collection_name(collection_name)

        scoped_name = self._scope_collection(collection_name)

        if enforcement not in ["off", "warn", "strict"]:
            raise ValueError("enforcement must be 'off', 'warn', or 'strict'")

        data: Dict[str, Any] = {
            "schema": schema.to_dict(),
            "enforcement_mode": enforcement,
        }
        if description is not None:
            data["description"] = description

        response = self._make_request(
            "PUT",
            f"schema/{quote_collection_name(scoped_name)}",
            data,
            evict_caches_on_404=scoped_name,
        )
        etag = response["etag"]

        # Update cache (use scoped name)
        self._payload_schema_cache[scoped_name] = {
            "schema": schema,
            "etag": etag,
            "enforcement_mode": enforcement,
        }

        return etag

    def delete_schema(self, collection_name: str) -> bool:
        """Remove schema from a collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            True if schema was deleted successfully.

        Raises:
            SchemaNotFoundError: If no schema is defined for the collection.
        """
        validate_collection_name(collection_name)

        scoped_name = self._scope_collection(collection_name)

        try:
            self._make_request("DELETE", f"schema/{quote_collection_name(scoped_name)}")

            # Clear from cache (use scoped name)
            self._payload_schema_cache.pop(scoped_name, None)

            return True

        except AetherfyVectorsException as e:
            if e.status_code == 404:
                # Disambiguate via backend's error.code:
                #   COLLECTION_NOT_FOUND → collection is gone, evict caches.
                #   SCHEMA_NOT_DEFINED → collection exists, no schema set.
                # Both surface as SchemaNotFoundError to the caller — the
                # difference is whether we self-heal the local caches.
                error_code = self._extract_error_code(e)
                if error_code == "COLLECTION_NOT_FOUND":
                    self._schema_cache.pop(scoped_name, None)
                    self._payload_schema_cache.pop(scoped_name, None)
                raise SchemaNotFoundError(collection_name)
            raise

    def analyze_schema(
        self, collection_name: str, sample_size: int = 1000
    ) -> AnalysisResult:
        """Analyze existing data to understand payload structure.

        Args:
            collection_name: Name of the collection to analyze.
            sample_size: Number of vectors to sample (100-10000, default: 1000).

        Returns:
            Analysis result including field presence, types, and suggested schema.

        Raises:
            ValidationError: If sample_size is out of range.
            CollectionNotFoundError: If collection doesn't exist.
        """
        validate_collection_name(collection_name)

        scoped_name = self._scope_collection(collection_name)

        if sample_size < 100 or sample_size > 10000:
            raise ValueError("sample_size must be between 100 and 10000")

        data = {"sample_size": sample_size}
        response = self._make_request(
            "POST",
            f"schema/{quote_collection_name(scoped_name)}/analyze",
            data,
            evict_caches_on_404=scoped_name,
        )

        # Return with unscoped collection name
        result = AnalysisResult.from_dict(response)
        if hasattr(result, "collection"):
            result.collection = collection_name
        return result

    def refresh_schema(self, collection_name: str) -> None:
        """Force refresh of cached schema.

        Args:
            collection_name: Name of the collection.
        """
        scoped_name = self._scope_collection(collection_name)
        self._payload_schema_cache.pop(scoped_name, None)
        self.get_schema(collection_name)  # get_schema will handle scoping internally

    # Analytics Methods (SDK-specific)
    #
    # One method, implemented here rather than behind an AnalyticsClient
    # sub-client. GET /api/v1/analytics/usage is the only analytics endpoint
    # that reports measured values (it reads Postgres); every other one was
    # deleted for reporting invented or unreachable data. A sub-client holding
    # a single method would also have been this SDK's only namespace-style
    # sub-client -- auth_manager and the caches are infrastructure, and
    # Namespace/Thread are factory-returned scopes, not client attributes.

    def get_usage_stats(self) -> UsageStats:
        """Retrieve current usage statistics against customer limits.

        Returns:
            A :class:`UsageStats` carrying the endpoint's fields verbatim:
            ``storage_bytes_used``, ``storage_limit_bytes``,
            ``collections_count``, ``collections_limit``, ``tier``,
            ``active_regions`` (the union of every active collection's
            regions) and ``usage_percentage`` (``0`` when there is no storage
            limit). Both limit fields are ``None`` on an unlimited tier —
            one sentinel, the same for both.

        Raises:
            AetherfyVectorsException: If request fails.
        """
        url = build_api_url(self.endpoint, "analytics/usage")

        try:
            response = self.session.get(
                url, headers=self.auth_headers, timeout=self.timeout
            )

            if response.status_code == 200:
                return UsageStats.from_dict(response.json())

            error_data = response.json() if response.content else {}
            raise parse_error_response(error_data, response.status_code)

        except requests.RequestException as e:
            raise AetherfyVectorsException(
                f"Failed to retrieve usage statistics: {str(e)}"
            )

    # Utility Methods

    def close(self, **kwargs) -> None:
        """Close the client connection and cleanup resources.

        Args:
            **kwargs: qdrant-client's ``grpc_grace`` is accepted and has no
                effect (there is no gRPC channel to wait on). Any other name
                raises TypeError.
        """
        check_qdrant_kwargs("close", kwargs)
        if hasattr(self, "session"):
            self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation of the client."""
        masked_key = self.auth_manager.mask_api_key()
        return (
            f"AetherfyVectorsClient(endpoint='{self.endpoint}', api_key='{masked_key}')"
        )
