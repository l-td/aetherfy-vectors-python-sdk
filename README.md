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

# Add vectors
points = [
    {
        "id": "doc_1",
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
# Bulk insert thousands of points
large_batch = [
    {"id": f"doc_{i}", "vector": [...], "payload": {...}}
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
export AETHERFY_API_KEY="afy_live_your_api_key_here"
```

Or pass it directly:

```python
client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here")
```

### Python Version Support

- **Minimum**: Python 3.8
- **Recommended**: Python 3.10+
- **Tested**: Python 3.8, 3.9, 3.10, 3.11, 3.12

## 📚 Complete API Reference

### Collection Management

```python
# Create collection
client.create_collection(name, vectors_config, distance=None)

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
    AuthenticationError,
    CollectionNotFoundError,
    RateLimitExceededError,
    ServiceUnavailableError
)

try:
    results = client.search("nonexistent", [0.1, 0.2, 0.3])
except CollectionNotFoundError as e:
    print(f"Collection not found: {e}")
except AuthenticationError as e:
    print(f"Invalid API key: {e}")
except RateLimitExceededError as e:
    print(f"Rate limit exceeded. Retry after: {e.retry_after}s")
```

## 🔧 Configuration Options

```python
client = AetherfyVectorsClient(
    api_key="your_api_key",           # Required: Your Aetherfy API key
    endpoint="https://vectors.aetherfy.com",  # Optional: Custom endpoint
    timeout=30.0,                     # Optional: Request timeout (seconds)
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