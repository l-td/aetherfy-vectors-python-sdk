# Aetherfy Vectors Python SDK

[![PyPI version](https://badge.fury.io/py/aetherfy-vectors.svg)](https://badge.fury.io/py/aetherfy-vectors)
[![Python Support](https://img.shields.io/pypi/pyversions/aetherfy-vectors.svg)](https://pypi.org/project/aetherfy-vectors/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/aetherfy/aetherfy-vectors-python/workflows/Tests/badge.svg)](https://github.com/aetherfy/aetherfy-vectors-python/actions)

A **drop-in replacement** for `qdrant-client` that provides **global vector database operations** with automatic replication, intelligent caching, and **sub-50ms latency worldwide**.

## 🚀 Key Features

- **🔄 Drop-in Replacement**: 100% compatible with `qdrant-client` API
- **🌍 Global Performance**: Sub-50ms latency from anywhere in the world
- **⚡ Intelligent Caching**: 85%+ cache hit rates for optimal performance
- **🛡️ Zero DevOps**: No infrastructure management or regional deployment needed
- **📊 Built-in Analytics**: Real-time performance metrics and usage insights
- **🔧 Auto-Failover**: Intelligent routing and retry mechanisms
- **🔐 Enterprise Security**: API key authentication and audit logging

## 📦 Installation

```bash
pip install aetherfy-vectors
```

## 🏃‍♂️ Quick Start

### Migration from qdrant-client

Replace your existing qdrant-client code with just **2 lines changed**:

```python
# Before (qdrant-client)
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)

# After (aetherfy-vectors) - Only 2 changes needed!
from aetherfy_vectors import AetherfyVectorsClient
client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here")

# All your existing code works unchanged! 🎉
results = client.search(collection="my_collection", vector=[0.1, 0.2, 0.3])
```

### Basic Usage Example

```python
from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import VectorConfig, DistanceMetric

# Initialize client
client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here")

# Create a collection
client.create_collection(
    "documents",
    VectorConfig(size=128, distance=DistanceMetric.COSINE)
)

# Add vectors — a point id is an unsigned integer (<= 2**53 - 1) or a UUID string
points = [
    {
        "id": 1,
        "vector": [0.1, 0.2, ...],  # 128-dimensional vector
        "payload": {"title": "Document 1", "category": "research"}
    }
]
client.upsert("documents", points)

# Search for similar vectors
results = client.search(
    collection_name="documents",
    query_vector=[0.1, 0.2, ...],
    limit=10,
    with_payload=True
)

for result in results:
    print(f"ID: {result.id}, Score: {result.score}")
    print(f"Title: {result.payload['title']}")
```

## 🌟 Unique Features

### Global Performance Analytics

Monitor your vector database performance globally:

```python
# Get global performance metrics
analytics = client.get_performance_analytics()
print(f"Cache hit rate: {analytics.cache_hit_rate:.1%}")
print(f"Average latency: {analytics.avg_latency_ms:.1f}ms")
print(f"Active regions: {len(analytics.active_regions)}")

# Collection-specific analytics
collection_stats = client.get_collection_analytics("documents")
print(f"Search requests: {collection_stats.search_requests:,}")

# Usage statistics
usage = client.get_usage_stats()
print(f"Points used: {usage.current_points:,}/{usage.max_points:,}")
```

### Intelligent Global Routing

Your requests are automatically routed to the optimal region:

```python
# No configuration needed - routing is automatic!
# Requests from US → US regions (20ms)
# Requests from EU → EU regions (15ms)  
# Requests from Asia → Asia regions (25ms)
```

## 🔧 Advanced Usage

### Batch Operations

Efficient batch processing with automatic optimization:

```python
# Bulk insert thousands of points (ids are unsigned integers or UUID strings)
large_batch = [
    {"id": i, "vector": [...], "payload": {...}}
    for i in range(10000)
]

# Automatically optimized batch size and routing
client.upsert("large_collection", large_batch)
```

### Complex Filtering

Full compatibility with qdrant-client filters:

```python
results = client.search(
    collection_name="products",
    query_vector=[...],
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "electronics"}},
            {"key": "price", "range": {"gte": 100, "lte": 1000}}
        ]
    },
    limit=20
)
```

### Context Manager Support

```python
with AetherfyVectorsClient(api_key="your_key") as client:
    results = client.search("collection", [0.1, 0.2, 0.3])
    # Automatic cleanup
```

## 🔁 Iterating Large Collections

For bulk reads, use `scroll_iter()` rather than `scroll(limit=…)`. The
iterator pages transparently and stays within the server's per-request
caps (1000 points/call, 10 MB/response):

```python
for point in client.scroll_iter("my_collection", batch_size=256):
    process(point)

# With a filter and selective payload/vector return
for point in client.scroll_iter(
    "my_collection",
    batch_size=512,
    scroll_filter={"must": [{"key": "status", "match": {"value": "active"}}]},
    with_payload=True,
    with_vectors=False,
):
    process(point)
```

`batch_size` is the page size for one HTTP round trip (max 1000 server-side).
The iterator handles cursor management, page exhaustion, and pagination
errors — no offset bookkeeping in user code.

## ✏️ Editing Payload on Existing Points

Three operations on the payload of points that already exist — no need to
re-upsert vectors:

```python
# MERGE: add or update keys, leave others alone
client.set_payload(
    "my_collection",
    {"reviewed": True, "reviewer": "alice"},
    [point_id],
)

# OVERWRITE: replace the entire payload object
client.overwrite_payload(
    "my_collection",
    {"category": "X"},   # all other keys are dropped
    [point_id],
)

# DELETE: remove specific keys, leave others alone
client.delete_payload(
    "my_collection",
    ["draft_field", "stale_score"],
    [point_id],
)
```

Each call accepts up to **512 points** in one round trip; for larger
mutations, batch on the caller side. The semantics map exactly to
qdrant's `set_payload` / `overwrite_payload` / `delete_payload` so
existing patterns transfer.

## 🧠 Memory SDK — Iter, Bulk-load, set_metadata

The Memory layer (`aetherfy_memory`) layers `Namespace` and `Thread`
abstractions on top of `AetherfyVectorsClient`. Three additions worth
knowing once you go past `add()` / `search()`:

### Iterating a namespace or a thread

```python
from aetherfy_memory import MemoryClient
memory = MemoryClient(api_key="afy_live_…", workspace="my-bot")

ns = memory.namespace("customer-42")
for point in ns.iter(batch_size=256):
    process(point)

# Threads have iter_history() — yields messages in ts order across the
# whole conversation. Distinct from history(limit=N), which caps at 5000
# in memory for the most-recent slice.
thread = memory.thread("conv-99")
for msg in thread.iter_history(order="asc"):
    print(msg.role, msg.content)
```

Use `history(limit=N)` for "show me the last N messages" (bounded, fast).
Use `iter_history()` for "walk every message in this thread" (paged,
memory-bounded by the iterator).

### Bulk-loading memories

`add_many()` and `append_many()` batch into a single `client.upsert` so
N items become 1 round trip. IDs are returned in input order; missing
IDs are auto-generated as canonical UUIDs (the same format `iter()` and
`retrieve()` yield back, so equality comparisons just work).

```python
items = [
    {"text": "first",  "vector": embed("first"),  "metadata": {"src": "a"}},
    {"text": "second", "vector": embed("second"), "metadata": {"src": "b"}},
]
ids = ns.add_many(items)            # single round trip; preserves input order

# Threads use append_many — role/content/ts payloads, ts auto-set per
# message when omitted (each message gets its own ts, not one shared).
msgs = [
    {"role": "user",      "content": "hi",    "vector": embed("hi")},
    {"role": "assistant", "content": "hello", "vector": embed("hello")},
]
ids = thread.append_many(msgs)
```

Threads have no `add_many()` — a `Thread` is not a `Namespace` subclass
(they share a scope base but declare their own write API), so there is
no `add_many` to call. `add_many` writes `text`/`metadata` payloads,
which don't fit a thread's `role`/`content`/`ts` schema. Use
`append_many()` on threads.

### set_metadata — atomic replace, explicit-compose merge

`set_metadata()` replaces the entire metadata sub-key.
`set_metadata({"tag": "x"})` nukes every other key. Use
`merge_metadata()` if you want additive updates that preserve existing
keys. Reserved fields (`text` for Namespace; `role`/`content`/`ts` for
Thread) are untouched either way.

```python
ns.set_metadata(point_id, {"reviewed": True, "score": 0.92})
```

To merge into existing metadata via the explicit-compose pattern (race
visible at the call site, no atomicity guarantee):

```python
current = ns.retrieve([point_id])[0]["payload"].get("metadata", {})
current.update({"reviewed": True})
ns.set_metadata(point_id, current)
```

If two callers run this concurrently, one update wins and the other
sees its read be stale — by design, you see that race in your own code
rather than have the SDK hide it.

### merge_metadata — atomic per-point partial merge

`merge_metadata({"tag": "x"})` adds/updates the listed keys and leaves
every other key untouched. Concurrent patches to different keys all
land atomically; concurrent writes to the same key resolve via
last-writer-wins per the storage operation order. Raises
`PointNotFoundError` if the point doesn't exist. Reserved keys (`text`
on Namespace; `role`, `content`, `ts` on Thread) cannot appear in the
partial — raises a local `ValueError` before the request is sent.

```python
ns.merge_metadata(point_id, {"reviewed": True})
ns.merge_metadata(point_id, {"score": 0.92})
# final metadata: original keys + reviewed + score
```

### delete_metadata_keys — atomic key removal

`delete_metadata_keys(point_id, ["tag", "score"])` removes the listed
keys from metadata; keys not in the list are left untouched. Raises
`PointNotFoundError` if the point doesn't exist. Reserved keys cannot
appear in the keys list (same set as `merge_metadata`).

```python
ns.delete_metadata_keys(point_id, ["draft", "stale_score"])
```

## 📐 Limits

Two axes constrain a single call: per-request size (PRS) and requests
per minute (RPM). Both axes return a 4xx with a structured `error.code`
when they fire — no surprise 5xx, no silent truncation.

| Class | Endpoint | Cap |
|-------|----------|-----|
| READS | `scroll` · `search` · `retrieve` | ≤ 1000 points/call · ≤ 10 MB/response |
| WRITES | `upsert` | ≤ 10 K vectors/call · streaming |
|        | payload edits · batch delete | ≤ 512 points/call |

> **Upserts stream** — there is no body-size cap on the public upsert
> URL. The 10 K vectors/call is a defensive request-level ceiling, not
> a body limit; one call can upload millions of bytes via byte-target
> chunking on the receiving end. For bulk reads, use `scroll_iter()` —
> it pages transparently and stays within both quotas.

`requests_per_minute` is a sliding-window minutely cap derived from your
subscription tier. When it fires, the SDK raises
`RateLimitExceededError` with a structured `retry_after` (seconds);
PRS violations raise `ValidationError` (400) or surface as 413
`RESPONSE_TOO_LARGE` for oversized response bodies.

## 🤝 Multi-Agent Workspaces

Workspaces let multiple agents share vector collections without name collisions. All collections created through a workspace-scoped client are automatically namespaced — agents in the same workspace see each other's collections; agents in different workspaces are fully isolated.

### Creating a workspace-scoped client

```python
from aetherfy_vectors import AetherfyVectorsClient

client = AetherfyVectorsClient(
    api_key="afy_live_your_api_key_here",
    workspace="invoice-pipeline",  # All operations are scoped to this workspace
)
```

### How scoping works

Collection names are automatically prefixed — you always use the short name:

```python
# Create a collection (stored as "invoice-pipeline/documents" internally)
from aetherfy_vectors.models import VectorConfig, DistanceMetric

client.create_collection("documents", VectorConfig(size=768, distance=DistanceMetric.COSINE))

# Search — no need to know the full scoped name
results = client.search("documents", query_vector=embedding, limit=10)

# List — only returns collections in your workspace
collections = client.get_collections()
# → [CollectionDescription(name='documents', ...)]  (short names, not scoped names)
```

### Multi-agent example

```python
# Agent A: extractor
extractor = AetherfyVectorsClient(api_key=api_key, workspace="invoice-pipeline")
extractor.create_collection("raw-invoices", VectorConfig(size=768, distance=DistanceMetric.COSINE))
extractor.upsert("raw-invoices", extracted_points)

# Agent B: classifier — same workspace, sees Agent A's collection
classifier = AetherfyVectorsClient(api_key=api_key, workspace="invoice-pipeline")
results = classifier.search("raw-invoices", query_vector=embedding, limit=20)
```

### Workspace auto-detection from environment

Agents deployed by Aetherfy automatically have `AETHERFY_WORKSPACE` set. The client picks this up if no `workspace` is passed explicitly:

```python
import os

# In a deployed agent, AETHERFY_WORKSPACE is injected automatically
client = AetherfyVectorsClient(api_key=os.environ["AETHERFY_API_KEY"])
# → workspace is auto-detected from AETHERFY_WORKSPACE env var
```

### No workspace (backward-compatible)

```python
# No workspace — collections are stored as-is, not scoped
client = AetherfyVectorsClient(api_key="afy_live_your_key")
client.create_collection("my-global-collection", VectorConfig(size=768, distance=DistanceMetric.COSINE))
```

> **Tip:** Create workspaces explicitly in the Aetherfy control plane before use (`afy workspaces create invoice-pipeline`). Agents deployed to a workspace automatically receive the workspace name via `AETHERFY_WORKSPACE`.

## 🧩 Payload Schemas

Collections can carry an optional payload schema that the SDK validates against **before** upsert — catching malformed payloads client-side without a round trip. Schemas are cached and automatically revalidated when they change server-side (via ETag).

```python
from aetherfy_vectors import Schema, FieldDefinition

schema = Schema(
    fields={
        "title":    FieldDefinition(type="string",  required=True),
        "price":    FieldDefinition(type="float",   required=True),
        "tags":     FieldDefinition(type="array",   required=False, element_type="string"),
        "in_stock": FieldDefinition(type="boolean", required=False),
    },
    description="Product catalog payloads",
)

# enforcement: "off" (no validation), "warn" (log warnings), "strict" (raise on violation)
etag = client.set_schema("products", schema, enforcement="strict")

# Inspect or remove the schema
current = client.get_schema("products")   # returns None if no schema is defined
client.delete_schema("products")
```

### Infer a schema from existing data

```python
analysis = client.analyze_schema("products", sample_size=1000)
print(analysis.suggested_schema)   # a Schema you can set_schema() directly
```

Schema violations raise `SchemaValidationError` (with a list of per-field errors). If the server reports a stale schema (`412 Precondition Failed`), the SDK auto-refreshes the cache and retries.

## 📊 Performance Comparison

| Feature | Local Qdrant | Aetherfy Vectors |
|---------|-------------|------------------|
| **Global Latency** | 200-2000ms | **<50ms** |
| **Cache Hit Rate** | 0% | **85%+** |
| **DevOps Required** | High | **Zero** |
| **Auto-Failover** | Manual | **Automatic** |
| **Global Replication** | Manual | **Automatic** |
| **Analytics** | Limited | **Built-in** |

## 🛠️ Environment Setup

### API Key Configuration

Set your API key using environment variables (recommended):

```bash
# Either of these is read automatically
export AETHERFY_API_KEY="afy_live_your_api_key_here"
export AETHERFY_VECTORS_API_KEY="afy_live_your_api_key_here"
```

Or pass it directly:

```python
client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here")
```

Relevant environment variables:

| Variable | Purpose |
|----------|---------|
| `AETHERFY_API_KEY` | Primary API key |
| `AETHERFY_VECTORS_API_KEY` | Alternative API key (useful when the same process talks to multiple Aetherfy services) |
| `AETHERFY_WORKSPACE` | Used when the client is constructed with `workspace="auto"` (set automatically on deployed agents) |
| `AETHERFY_VECTORS_URL` | Pin the client to a specific endpoint URL. Set automatically by the control-plane on deployed agents. Wins over `api_region=` if both are set. |
| `AETHERFY_VECTORS_API_REGION` | Equivalent to `api_region=` constructor arg. Local-dev / debugging only. |

### Local development across regions

Production agents have `AETHERFY_VECTORS_URL` injected by the control-plane —
that's the URL they reach the regional backend through, and it takes
precedence over `api_region=`. For local development (no env var injected),
you can pin a client to a specific regional endpoint:

```python
client = AetherfyVectorsClient(
    api_key="afy_test_...",
    api_region="eu-central-1",  # 'us-east-1' | 'eu-central-1' | 'ap-southeast-1'
)
```

The first call resolves `api_region` against `GET /api/v1/regions` on the
default global endpoint and caches the result on the client instance.
If both `AETHERFY_VECTORS_URL` and `api_region=` are set, the env var wins
and a warning is logged — that's the production-agent protection rule.

`api_region` selects *which regional endpoint to connect to* — it is a
transport/routing override, not where a collection's data lives. To control
collection placement, pass `regions=` to `create_collection` (see below).

### Python Version Support

- **Minimum**: Python 3.9
- **Recommended**: Python 3.10+
- **Tested**: Python 3.9, 3.10, 3.11, 3.12

## 📚 Complete API Reference

### Collection Management

```python
# Create collection — returns the created Collection (with its resolved
# `regions` echoed back by the server). Pass `regions=[...]` to pin the
# collection to a subset of your scope; omit it to default to your full
# scope. (Distinct from the constructor's `api_region`, which only picks
# the endpoint to connect to.)
collection = client.create_collection(name, vectors_config, distance=None, regions=None)

# List collections
collections = client.get_collections()

# Get collection info
info = client.get_collection(name)

# Check existence
exists = client.collection_exists(name)

# Delete collection
client.delete_collection(name)
```

### Point Operations

```python
# Insert/update points
client.upsert(collection_name, points)

# Retrieve points
points = client.retrieve(collection_name, ids, with_payload=True, with_vectors=False)

# Delete points
client.delete(collection_name, point_ids_or_filter)

# Count points
count = client.count(collection_name, count_filter=None, exact=True)
```

### Search Operations

```python
# Vector search
results = client.search(
    collection_name,
    query_vector,
    limit=10,
    offset=0,
    query_filter=None,
    with_payload=True,
    with_vectors=False,
    score_threshold=None
)
```

### Schema Management (Aetherfy-specific)

```python
# Set, fetch, and remove a payload schema
etag = client.set_schema(collection_name, schema, enforcement="strict", description=None)
schema = client.get_schema(collection_name)           # None if not set
client.delete_schema(collection_name)

# Infer a schema from existing data
analysis = client.analyze_schema(collection_name, sample_size=1000)  # 100–10000

# Cache control (rarely needed)
client.refresh_schema(collection_name)
client.clear_schema_cache(collection_name=None)       # None clears all
```

### Analytics (Aetherfy-specific)

```python
# Global performance
analytics = client.get_performance_analytics(time_range="24h", region=None)

# Collection analytics
stats = client.get_collection_analytics(collection_name, time_range="24h")

# Usage statistics
usage = client.get_usage_stats()
```

## 🚨 Error Handling

The SDK provides detailed error handling compatible with qdrant-client:

```python
from aetherfy_vectors.exceptions import (
    AetherfyVectorsException,   # base class for all SDK errors
    AuthenticationError,
    CollectionNotFoundError,
    PointNotFoundError,
    RateLimitExceededError,
    ServiceUnavailableError,
    ValidationError,
    RequestTimeoutError,
    NetworkError,
    SchemaValidationError,
    SchemaNotFoundError,
    CollectionInUseError,
    QuotaExceededError,
)

try:
    results = client.search("nonexistent", [0.1, 0.2, 0.3])
except CollectionNotFoundError as e:
    print(f"Collection not found: {e}")
except AuthenticationError as e:
    print(f"Invalid API key: {e}")
except RateLimitExceededError as e:
    print(f"Rate limit exceeded. Retry after: {e.retry_after}s")
except SchemaValidationError as e:
    for violation in e.errors:
        print(violation)
except QuotaExceededError as e:
    print(f"Quota '{e.quota_type}' exceeded: {e.current}/{e.limit}")
```

## 🔧 Configuration Options

```python
client = AetherfyVectorsClient(
    api_key="your_api_key",                   # Required (or set AETHERFY_API_KEY)
    endpoint="https://vectors.aetherfy.com",  # Optional: Custom endpoint
    timeout=30.0,                             # Optional: Request timeout (seconds)
    workspace=None,                           # Optional: None, a workspace name, or "auto"
                                              #           to read AETHERFY_WORKSPACE
)
```

## 📈 Monitoring & Observability

### Built-in Dashboard Data

```python
# Get comprehensive metrics for monitoring dashboards
perf = client.get_performance_analytics()
usage = client.get_usage_stats()

dashboard_data = {
    "latency_ms": perf.avg_latency_ms,
    "cache_hit_rate": perf.cache_hit_rate,
    "requests_per_second": perf.requests_per_second,
    "error_rate": perf.error_rate,
    "usage_percent": usage.points_usage_percent,
    "active_regions": len(perf.active_regions)
}
```

### Health Checks

```python
def health_check():
    try:
        collections = client.get_collections()
        return {"status": "healthy", "collections": len(collections)}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## 🔗 Migration Guide

### Step-by-Step Migration

1. **Install aetherfy-vectors**:
   ```bash
   pip install aetherfy-vectors
   ```

2. **Get your API key** from [Aetherfy Dashboard](https://dashboard.aetherfy.com)

3. **Update imports**:
   ```python
   # from qdrant_client import QdrantClient
   from aetherfy_vectors import AetherfyVectorsClient
   ```

4. **Update initialization**:
   ```python
   # client = QdrantClient(host="localhost", port=6333)
   client = AetherfyVectorsClient(api_key="your_api_key")
   ```

5. **Test existing functionality** (should work unchanged!)

6. **Optional**: Add analytics calls for insights

7. **Deploy** and enjoy global performance! 🚀

### Migration Compatibility

| qdrant-client Method | aetherfy-vectors | Compatible |
|---------------------|------------------|------------|
| `create_collection()` | ✅ | 100% |
| `get_collections()` | ✅ | 100% |
| `upsert()` | ✅ | 100% |
| `search()` | ✅ | 100% |
| `retrieve()` | ✅ | 100% |
| `delete()` | ✅ | 100% |
| `count()` | ✅ | 100% |

**Plus additional analytics methods unique to Aetherfy!**

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/aetherfy/aetherfy-vectors-python.git
cd aetherfy-vectors-python

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black aetherfy_vectors/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://docs.aetherfy.com/vectors](https://docs.aetherfy.com/vectors)
- **API Reference**: [https://vectors.aetherfy.com/docs](https://vectors.aetherfy.com/docs)
- **Issues**: [GitHub Issues](https://github.com/aetherfy/aetherfy-vectors-python/issues)
- **Discord**: [Aetherfy Community](https://discord.gg/aetherfy)
- **Email**: [developers@aetherfy.com](mailto:developers@aetherfy.com)

## 🌟 Why Choose Aetherfy Vectors?

### For Solo Developers
- **Zero setup time** - start building immediately
- **Predictable pricing** - no surprise infrastructure costs
- **Global reach** - your users get fast responses worldwide

### For Teams
- **No DevOps overhead** - focus on your product, not infrastructure
- **Built-in monitoring** - comprehensive analytics out of the box
- **Reliable performance** - 99.9% uptime with automatic failover

### For Enterprises
- **Global scalability** - handles millions of vectors effortlessly
- **Security first** - enterprise-grade authentication and audit logs
- **Cost effective** - pay only for what you use, no idle infrastructure

---

**Ready to experience global vector search?** [Get your API key](https://dashboard.aetherfy.com) and migrate in minutes! 🚀