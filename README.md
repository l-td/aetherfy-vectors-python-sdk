# Aetherfy Vectors Python SDK

[![PyPI version](https://badge.fury.io/py/aetherfy-vectors.svg)](https://badge.fury.io/py/aetherfy-vectors)
[![Python Support](https://img.shields.io/pypi/pyversions/aetherfy-vectors.svg)](https://pypi.org/project/aetherfy-vectors/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/l-td/aetherfy-vectors-python-sdk/workflows/Tests/badge.svg)](https://github.com/l-td/aetherfy-vectors-python-sdk/actions)

**Memory for AI agents** — conversations, knowledge bases, scrape logs, trading
histories. Named scopes you create and search, replicated across regions, with
no infrastructure to run.

## 🚀 Key Features

- **🧠 Memory Primitives**: `Namespace` for any named scope, `Thread` for conversations — not a raw collection API
- **🌍 Global Performance**: Low latency from anywhere in the world
- **⚡ Intelligent Caching**: High cache hit rates for optimal performance
- **🛡️ Zero DevOps**: No infrastructure management or regional deployment needed
- **🤝 Workspaces**: every collection lives in one; inside a deployed agent the agent's own is used
- **📊 Built-in Analytics**: Real-time performance metrics and usage insights
- **🔧 Auto-Failover**: Intelligent routing and retry mechanisms
- **🔐 Enterprise Security**: API key authentication and audit logging
- **🪜 Escape Hatch**: `AetherfyVectorsClient` underneath, compatible with qdrant-client ≤1.15.1's core methods (see [Migration Compatibility](#migration-compatibility))

## 📦 Installation

```bash
pip install aetherfy-vectors
```

## 🏃‍♂️ Quick Start

Every memory lives in a scope you create by name. Two kinds: a **namespace** for
anything with a name, a **thread** for a conversation. Bring your own embedding
vectors — Aetherfy stores and searches them, it does not generate them.

```python
from aetherfy_memory import MemoryClient

memory = MemoryClient()  # inside an Aetherfy agent: its key and its workspace

# --- A namespace: any named scope ---------------------------------------
memory.create_namespace("customer-42")
customer = memory.namespace("customer-42")

customer.add(text="Lives in NYC, prefers email", vector=embed("Lives in NYC, prefers email"))
hits = customer.search(vector=embed("where is this customer based?"), limit=5)

# --- A thread: a conversation -------------------------------------------
memory.create_thread("conv-99")
thread = memory.thread("conv-99")

thread.add(role="user", content="hi", vector=embed("hi"))
thread.add(role="assistant", content="hello", vector=embed("hello"))
recent = thread.history(limit=20)   # in message order

# Clearing a scope is atomic. After it, the scope no longer exists.
thread.clear()
```

Scopes must be created before you write to them, so a typo raises instead of
silently creating a second store.

A **namespace** is one collection, and takes its own `vector_size` (default
384; pass 1536 for OpenAI small, 3072 for large, 1024 for Cohere v3).

**Threads do not work that way.** Every thread in a workspace lives in one
shared collection with the thread id as a payload key, so a conversation costs
no collection slot and a plan's collection limit does not cap how many
conversations you can have. One collection means one dimension for all of
them, so `create_thread` takes no `vector_size`: it comes from the client.

That collection is `__threads__`, one in each workspace (collection names are
unique per workspace). It is managed by the SDK, and Aetherfy refuses to move
or rename it: it holds the conversations of every agent in its workspace, so
moving it would hand them to another workspace's agents. An agent that moves
to another workspace leaves its conversation history behind in the old one.

```python
memory = MemoryClient(thread_vector_size=1536)   # threads, for OpenAI small
```

Deployed on Aetherfy, `MemoryClient()` takes no arguments at all: the control
plane injects `AETHERFY_API_KEY` and `AETHERFY_WORKSPACE_ID` (the agent's
workspace) at deploy time. Anywhere else, name a workspace:
`MemoryClient(workspace="default")`.

Full memory API — iteration, bulk loading, metadata — under
[Memory SDK](#-memory-sdk--iter-bulk-load-set_metadata) below.

## 🧱 The Low-Level Client

`AetherfyVectorsClient` is the layer `MemoryClient` is built on, exported and
supported. Use it when you want raw collections and points rather than named
scopes, or an operation the memory layer doesn't expose.

The snippets below build on each other; each assumes the imports and client
from the previous ones.

### Migration from qdrant-client

If your code uses qdrant-client **1.15.1 or earlier** and sticks to the methods
listed under [Migration Compatibility](#migration-compatibility), migrating
means changing the import and the constructor call:

```python
# Before (qdrant-client)
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)

# After (aetherfy-vectors): the import and the constructor change
from aetherfy_vectors import AetherfyVectorsClient
client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here", workspace="default")

# Calls to the listed methods stay as they are
results = client.search(collection_name="my_collection", query_vector=[0.1, 0.2, 0.3])
```

That holds when qdrant-client arguments are passed by keyword, and it has
limits:

- Some qdrant-client arguments are refused, because ignoring them would change
  what you get: `ordering` other than `'weak'`, `shard_key_selector`,
  `consistency`, `search(append_payload=False)`, and every sharding,
  replication, storage and index setting of `create_collection`. Each raises
  `TypeError` naming itself.
- Code using `query_points` (qdrant-client **1.16 and later**, which removed
  `search`) does not migrate. This SDK has no `query_points`.
- Methods not in the list (`recommend`, `discover`, the snapshot and alias
  APIs, and so on) do not exist here.

### Basic Usage Example

```python
from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import VectorConfig, DistanceMetric

# Initialize client
client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here", workspace="default")

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

### Usage Against Your Plan Limits

```python
usage = client.get_usage_stats()
print(f"Storage used: {usage.storage_bytes_used:,} bytes")
print(f"Collections: {usage.collections_count}/{usage.collections_limit}")
print(f"Tier: {usage.tier}")
print(f"Replicating to: {', '.join(usage.active_regions)}")
```

### Intelligent Global Routing

Your requests are automatically routed to the optimal region:

```python
# No configuration needed - routing is automatic!
# Requests from the US  → the US region
# Requests from the EU  → the EU region
# Requests from Asia    → the Asia-Pacific region
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

Full compatibility with qdrant-client filters. Three clauses, and only
three: `must`, `must_not`, `should`.

```python
results = client.search(
    collection_name="products",
    query_vector=[...],
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "electronics"}},
            {"key": "price", "range": {"gte": 100, "lte": 1000}}
        ],
        "must_not": [
            {"key": "status", "match": {"value": "discontinued"}}
        ],
    },
    limit=20
)
```

The same clauses are available as a typed object, if you prefer it to a dict:

```python
from aetherfy_vectors.models import Filter

results = client.search(
    collection_name="products",
    query_vector=[...],
    query_filter=Filter(
        must=[{"key": "category", "match": {"value": "electronics"}}],
        must_not=[{"key": "status", "match": {"value": "discontinued"}}],
    ),
    limit=20,
)
```

A clause name outside those three raises `ValidationError` before the
request leaves the process — including `mustNot`, which is the JavaScript
SDK's spelling of `must_not`. Unknown clauses are never forwarded and never
dropped: writing the wrong one is an error you see, not a filter that
quietly matches everything. The same rule applies everywhere a filter is
accepted — `search`, `scroll`, `scroll_iter`, `count`, `delete`.

### Context Manager Support

```python
with AetherfyVectorsClient(api_key="your_key", workspace="default") as client:
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

Every collection lives in exactly one workspace, and every account starts with
one named `default`. A workspace decides where a collection is reachable from:
agents in the same workspace see each other's collections, and an agent reaches
another workspace only if the account owner granted it access.

Collection names are unique **per workspace**: `invoices` in `customer-a` and
`invoices` in `customer-b` are two collections.

### Choosing the workspace

```python
from aetherfy_vectors import AetherfyVectorsClient

client = AetherfyVectorsClient(
    api_key="afy_live_your_api_key_here",
    workspace="invoice-pipeline",  # All operations are scoped to this workspace
)
```

`workspace` defaults to `"auto"`: inside an agent deployed on Aetherfy, the
agent's own workspace, by id, from `AETHERFY_WORKSPACE_ID` (the control plane
injects it on every agent). A workspace rename never breaks an agent, because
it is addressed by id.

```python
import os

# Inside a deployed agent: its own workspace, nothing to configure.
client = AetherfyVectorsClient(api_key=os.environ["AETHERFY_API_KEY"])
```

There is no "no workspace". `workspace=None`, an empty name, and `"auto"`
outside an agent (where `AETHERFY_WORKSPACE_ID` is not set) raise `ValueError`
at construction, saying to pass a workspace name or run inside an agent.
`client.workspace` is the name (None when built from the id) and
`client.workspace_id` the id (None when given a name).

`"auto"` is only the agent's home workspace. Code that works across several
workspaces (one per customer, say) passes each workspace's name explicitly,
one client per workspace.

### How scoping works

You always use the collection's own name; the client sends it on the
workspace's routes:

```python
# Create a collection in the client's workspace
from aetherfy_vectors.models import VectorConfig, DistanceMetric

client.create_collection("documents", VectorConfig(size=768, distance=DistanceMetric.COSINE))

# Search — no need to know the full scoped name
results = client.search("documents", query_vector=embedding, limit=10)

# List — only returns collections in your workspace
collections = client.get_collections()
# → [Collection(name='documents', ...)]  (short names, not scoped names)
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

### Workspace errors

```python
from aetherfy_vectors import WorkspaceNotFoundError

try:
    client.get_collections()
except WorkspaceNotFoundError as e:
    # No such workspace in this account. A typo is never read as "empty".
    print(f"Unknown workspace: {e.workspace or e.workspace_id}")
```

`collection_exists()` returns False only when the workspace exists and the
collection is not in it; an unknown workspace raises.

### When a request is retried

One rule, the same in the Python and JS SDKs (`aetherfy_vectors/retry_policy.py`). Up to 3
retries, waiting about 1 s, 2 s, then 4 s (each jittered to 50-100%), within
a call's timeout.

| Failure | Retried for |
|---|---|
| Refused before anything ran: 503 `WORKSPACE_LOOKUP_UNAVAILABLE`, `COLLECTION_LOOKUP_UNAVAILABLE`, `PLACEMENT_UNAVAILABLE`; a connection never made (DNS, refused, connect timeout); 429 with a Retry-After | every operation |
| Ambiguous, the request may have run: a read timeout, a reset after sending, any other 5xx | only operations whose repeat gives the same result (below) |
| Any other 4xx | never |

| Operations | Class |
|---|---|
| `get_collections`, `get_collection`, `collection_exists`, `get_schema`, `analyze_schema`, `search`, `retrieve`, `scroll`, `count`, `get_usage_stats` | reads: repeatable |
| `upsert`, `set_payload`, `overwrite_payload`, `delete_payload`, `create_field_index`, `set_schema` | writes keyed by id: repeatable |
| `create_collection`, `delete_collection`, `delete` (points), `delete_schema`, `delete_field_index` | once: a repeat after a hidden success reports a false error, so an ambiguous failure is raised after one attempt |

`set_schema` is repeatable because it is sent without `If-Match`: a repeat writes
the same schema again. A conditional write would get a false 412 on a repeat
after a hidden success, so it would be classed once.

When the retries run out, the last error is raised (`ServiceUnavailableError`, `RequestTimeoutError` or `NetworkError`).

The client already retries under this rule. To retry more around your own
operation, say whether repeating it gives the same result: `REPEATABLE` (a
read, a write keyed by id) or `ONCE` (a create, a delete). The same failures
are retried as above; a 4xx never is.

```python
from aetherfy_vectors import retry_with_backoff, REPEATABLE

results = retry_with_backoff(
    lambda: client.search("collection", vector),
    REPEATABLE,
    max_retries=3,
    base_delay=1.0,
)
```

Renaming a collection, and moving one to another workspace, are done from the
dashboard or the CLI (`afy collections update`), not from the SDK. Code still
using the old name gets `CollectionNotFoundError`.

> **Tip:** Create workspaces in the dashboard or with the CLI
> (`afy workspaces create invoice-pipeline`). Agents deployed to a workspace
> receive its id as `AETHERFY_WORKSPACE_ID`.

## 🤖 Running as an Agent

Code deployed to Aetherfy as an agent gets a set of helpers in the same
distribution, under `aetherfy_agent`. Nothing to add to your requirements: the
standard runtime image preinstalls `aetherfy-vectors`, and a version you pin
yourself wins over it. A custom container installs it like any other package.

Each of these is a thin wrapper over a platform contract the docs already
publish; the helper exists so the contract stops being copied into every task.

```python
from aetherfy_agent import payload, machine, fan_out, spawn

# This run's input. A scheduled fire sends none, so {} is the normal case.
data = payload()

# The machine this run is on: whole numbers, not the strings the env carries.
shape = machine()

# An in-machine pool. Results come back in INPUT order and no failure is
# swallowed: the lowest-indexed exception is re-raised once every worker has
# finished. Width defaults to vcpus x 8 for threads, the default pool.
answers = fan_out(summarise, data.get("items", []))

# CPU-bound work wants processes. The default width follows the pool:
# threads get vcpus x 8 (they mostly wait), processes get vcpus.
digests = fan_out(hash_one, data.get("files", []), kind="processes")

# Run a different task agent, on its own machine, with its own lifecycle.
run = spawn("nightly-rollup", {"date": "2026-09-07"})
print(run.spawn_id, run.region, run.status)
```

`fan_out` prints one line to the run's logs before it starts, so how wide a run
went is visible after the fact:

```text
aetherfy: fanning out 32 wide on 4 vCPU / 8192 MB (120 tasks)
```

A run over your account's in-flight limit is not refused: it waits in the
account's line, with no machine and nothing billed, and starts when a slot
frees. The spawn comes back with `status` `"waiting"` and `position`, its place
in the line (1 is next); a run that goes straight to a machine has `status`
`"queued"` and no `position`. `wait` includes the time in the line.

Spawning has three outcomes worth telling apart. The payload cap and a full line
get their own types; everything else carries the platform's stable error code,
which is the thing to branch on:

```python
from aetherfy_agent import spawn
from aetherfy_agent.exceptions import (
    PayloadTooLarge,
    RunLineFull,
    SpawnError,
)

try:
    run = spawn("nightly-rollup", {"date": "2026-09-07"})
    print(run.status, run.position)  # "waiting" and 3, or "queued" and None
except PayloadTooLarge as exc:
    # The payload is for parameters and references, not data. Write the data to
    # a collection and pass its id.
    print(exc.payload_bytes, "exceeds", exc.max_bytes)
except RunLineFull as exc:
    # The account's line of waiting runs is full: there is nowhere left to
    # wait. The limits are the ACCOUNT's and are set by your plan; `limit` names
    # which one was hit. Nothing is retried for you: spawn again when runs have
    # left the line.
    print(exc.waiting_count, "of", exc.max_waiting_runs, "are waiting")
except SpawnError as exc:
    print(exc.error_code)
```

### Returning a result, and reading one back

Output on Aetherfy mirrors input. A task writes its answer with `write_result`;
whoever started the run reads it back off the run itself, so a parent hears
from a child in another region with no side channel between them.

```python
from aetherfy_agent import write_result

# In the child task. The mirror of payload(): a file on the machine, nothing
# over the network. Returning nothing is the normal case, so most tasks never
# call this at all.
write_result({"rows": 128, "date": "2026-09-08"})
```

```python
from aetherfy_agent import spawn, wait, result

# In the parent. wait() holds one request open instead of polling, and the time
# the run spends waiting in the account's line is part of what it waits for. A
# run that has not finished in time comes back exactly as it stands, which is
# not an error — read `state` and call again.
run = spawn("nightly-rollup", {"date": "2026-09-08"})
finished = wait(run.spawn_id, timeout_seconds=45)

if finished.state == "completed":
    print(finished.result)
elif finished.state == "failed":
    # failure_code is the value to switch on; error_message says it in words.
    print("the run failed:", finished.failure_code, finished.error_message)
elif finished.state == "cancelled":
    print("the run was cancelled:", finished.error_message)
else:
    # "waiting" (in the account's line) or "active" (executing): not over yet.
    print("still", finished.state)

# While the platform retries an attempt that met a temporary unavailability,
# `retry` says which attempt comes next and when; it is None otherwise.
if finished.retry:
    print(
        "attempt", finished.retry.attempt, "of", finished.retry.max_attempts,
        "at", finished.retry.next_attempt_at,
    )

# result() is the same read without the waiting.
now = result(run.spawn_id)
print(now.state, now.has_result, now.result_error)
```

A run's `state` is `waiting` while it waits in the account's line, `active`
while it executes, and ends `completed`, `failed` or `cancelled`. While it is
`waiting`, `run.raw["position"]` is its place in the line. The states belong to
the platform, so the helper does not enumerate them.

While the platform retries an attempt that met a temporary unavailability,
`run.retry` is a `RunRetry`: its `code`, the number of the next `attempt` out of
`max_attempts`, and `next_attempt_at`, when that attempt is made (a
timezone-aware UTC `datetime`). It is `None` when nothing is being retried, and
it says nothing about how the run ended: `state` and `failure_code` do.

The result shares the payload's inline cap — one number bounding both
directions — and it is for answers and references, not data. Anything larger
belongs in a collection, with its id in the result:

```python
from aetherfy_agent import write_result
from aetherfy_agent.exceptions import ResultTooLarge

try:
    write_result(everything)
except ResultTooLarge as exc:
    print(exc.result_bytes, "exceeds", exc.max_bytes)
    write_result({"collection": "nightly-rollup", "rows": len(everything)})
```

Reading a run has the same shape as spawning it: the two refusals worth telling
apart get their own types, and everything else carries the platform's stable
error code.

```python
from aetherfy_agent import result
from aetherfy_agent.exceptions import RunAccessDenied, RunNotFound, RunReadError

try:
    result(run_id)
except RunNotFound:
    print("no run has that id")
except RunAccessDenied:
    print("that run belongs to another account")
except RunReadError as exc:
    print(exc.error_code)
```

With the agent's own `AETHERFY_API_KEY`, a read the control plane does not
allow that key ([which ones it allows](https://docs.aetherfy.com/platform/api-keys#agent-scoped-keys-issued-by-aetherfy))
is refused with `RunReadError` and `error_code` `AUTH_AGENT_KEY_OUT_OF_SCOPE`.
An agent's key reads and waits on the runs in its own run tree, the runs that
share its root run.

### Cancelling a run

`cancel` stops a run, whether it is waiting in the account's line or executing,
and answers with what it did:

```python
from aetherfy_agent import cancel
from aetherfy_agent.exceptions import (
    NotARun,
    RunAccessDenied,
    RunCancelError,
    RunNotFound,
)

try:
    cancellation = cancel(run.spawn_id)
except RunNotFound:
    print("no run has that id")
except RunAccessDenied as exc:
    # Another account's run, or a run this agent's key may not cancel.
    print(exc.error_code)
except NotARun:
    print("that id is a release, not a run")
except RunCancelError as exc:
    print(exc.error_code)
else:
    # "cancelled": it is cancelled now. "stopping": it was executing and is being
    # stopped, and it ends `cancelled`. "already_finished": it had ended before
    # the call, which is an answer and not an error.
    print(cancellation.outcome, cancellation.cancelled_count)
```

`cancelled_count` counts every run the call cancelled: the runs started below
the cancelled one are cancelled with it. A cancelled run's key stops working at
once, so nothing it was still starting gets through. An account's key may cancel
any run of the account; an agent's own key may cancel only the runs below its own
run. A release is cancelled with `afy cancel <agent>`, not with this call.
`RunNotFound` and `RunAccessDenied` are the read family's types and are not
`RunCancelError`s: catch `AgentError` to see every failure.

### Messages between running runs

A result is one answer at the end. While both runs are still going, they can
talk: every run has a mailbox, `send` puts a message in another run's mailbox,
and `receive` hands over this run's next one. A child reports progress or asks a
question; the parent answers or changes the plan.

```python
from aetherfy_agent import Ended, parent, receive, send, spawn

# In the parent: start a child, then read what it says until it ends.
child = spawn("researcher", {"topic": "battery recycling"})
while True:
    got = receive()                          # the next message, or a child's end
    if isinstance(got, Ended):
        print("the child ended", got.state)  # read its answer with result()
        break
    if got.body.get("question"):
        send(got.sender_run_id, {"answer": "Europe only"})
    else:
        print("progress", got.body)
```

```python
# In the child: tell the run that started it, and wait for its answer.
send(parent(), {"question": "which region?"})
reply = receive(from_run=parent())           # waits for that run's message only
```

`receive()` waits as long as it takes. `receive(timeout_seconds=30)` returns
`None` when nothing came in that time, and `timeout_seconds=0` only checks. A
task run that waits past a short grace is suspended and is not billed until a
message wakes it; a service run waits awake, inside its request. A service's own
code waiting outside any request does not keep its machine awake: the machine
sleeps after its idle timeout with the `receive()` still waiting, and a message
wakes it. A parent also receives the end of each run it started (an `Ended`,
once per child), unless it already saw that end through `wait` or `result`.

When the agent is paused, paused at the account's usage limit, or stopped at a
suspension, a task run waiting for a message ends `failed` with the hold's
`RUN_INTERRUPTED_*` code, as a run working when the hold comes does. A paused
service is not woken by a message: the message stays unread, and the next
`receive()` hands it over once the agent runs again.

A message is acknowledged when its run asks for the next one, never by the
answer that handed it over. If a run's machine is lost while it waits, the run
starts again on a new one: `restarted()` is then `True`, the messages it had not
acknowledged are handed over again, and `messages()` reads the run's whole
history, in order, to rebuild its state from. `parent()` is the run that started
this one, or `None`.

A message body is any JSON value, up to the same size as a payload. A refused
send raises a type of its own, all of them `MessageError`s:

```python
from aetherfy_agent import (
    MailboxFull,
    MessageError,
    MessageRateLimited,
    MessageTooLarge,
    RecipientEnded,
    send,
)

try:
    send(run_id, {"progress": 0.5})
except MessageTooLarge as error:
    print(error.message_bytes, "exceeds", error.max_bytes)
except MessageRateLimited as error:
    print("send again in", error.retry_after, "s")
except MailboxFull as error:
    # The recipient has not read what it was sent; it clears as it reads.
    print(error.unread, "unread of", error.max_unread)
except RecipientEnded as error:
    print("that run has ended:", error.recipient_state)
except MessageError as error:
    print(error.error_code)
```

A run sends up to 60 messages a minute, and a mailbox holds up to 100 unread
ones. Messages count toward the account's log storage. A mailbox takes messages
until its run ends; then it is read-only, and its history is kept as long as the
run's logs. An agent's own key sends to the runs in its own run tree and
receives only from its own mailbox; an account's key may send to any run of the
account.

### Inside a service's run handler

A service handles a run in `POST /aetherfy/run`. The platform gives that run a
key of its own and sends it, with the run's id, in the headers
`X-Aetherfy-Run-Key` and `X-Aetherfy-Run`, on that one call. `service_run` reads
them into a handle whose `spawn`, `result`, `wait`, `cancel`, `send`, `receive`,
`messages` and `parent` are the functions above, with the same signatures and
errors, authenticating as the run:

```python
import asyncio

from aetherfy_agent import service_run


@app.post("/aetherfy/run")
async def run_handler(request):
    run = service_run(request.headers)  # a dict, Starlette's or Flask's headers
    body = await request.json()
    child = await asyncio.to_thread(run.spawn, "worker", body["payload"])
    finished = await asyncio.to_thread(run.wait, child.spawn_id)
    return {"answer": finished.result}
```

What the handler starts through the handle belongs to the run: cancelling the
run cancels it. Headers without the run's id and key raise `NotAServiceRun`;
`service_run` is for the handler of `POST /aetherfy/run` only, and a public
request carries neither header.

The handle is explicit on purpose. A service answers many runs at once, and an
implicit context would be lost the moment work crosses into another thread (a
thread pool, `run_in_executor`): the call would fall back, without an error, to
the release's key, and what it started would land outside the run, where
cancelling the run does not reach. The module-level `spawn`, `result`, `wait` and
`cancel` are unchanged and read `AETHERFY_API_KEY`, which on a task machine is
the run's key and in a service is the release's: right for the service's own
traffic, wrong inside a run. Do not mix the two on one run. A run spawned through
the handle is in the run's tree, not the release's, so a module-level `wait` on
it is refused with `AUTH_AGENT_KEY_OUT_OF_SCOPE`: wait through the handle. The run's
key stops working when the run ends, so do not keep the handle past the handler.

### Finding another agent at call time

A service that talks to another service in its workspace never holds that
service's address. It asks for it by name, at the moment it needs it, so a peer
that was deleted and created again is reached at its new address without a
restart:

```python
from aetherfy_agent import agent_url, agent_urls, call_agent

url = agent_url("analyst")              # "https://analyst-k3m7x2.aetherfy.dev"
peers = agent_urls()                    # {"analyst": "https://...", ...}

reply = call_agent("analyst", "/summarise", method="POST", json={"id": 7})
print(reply.status, reply.json())
```

A peer is another agent of THIS agent's own workspace that has a public URL.
The answer is remembered in the process for `PEER_CACHE_SECONDS` (30), per name,
with no request while it is fresh; `fresh=True` skips the memory and refreshes
it. `call_agent` looks the peer up again, once, when the address it used is
gone: the name does not resolve, the connection is refused, or the edge answered that address with a
`404` whose body is `{"error": {"code": "AGENT_NOT_FOUND"}}` or
`AGENT_NOT_REACHABLE`. Any other status, a `5xx` included, is returned as it
came and never retried; when the control plane cannot answer a lookup (429, 5xx, no
connection) the last address known for that peer is returned and the control plane is
not asked again before its `Retry-After` (at least `PEER_RETRY_FLOOR_SECONDS`, at most
`PEER_RETRY_CAP_SECONDS`), and only with no known address is `PeerLookupError` raised;
a reset or a timeout raises `AgentTransportError`
and is not repeated either, since the peer may already have acted. The peer is another service, so your Aetherfy key goes
to the lookup and never to the peer.

```python
from aetherfy_agent import agent_url
from aetherfy_agent.exceptions import PeerHasNoUrl, PeerLookupError, PeerNotFound

try:
    agent_url("analyst")
except PeerNotFound:
    print("this workspace has no agent of that name")
except PeerHasNoUrl:
    print("it exists, but it is not a service: nothing listens on it")
except PeerLookupError as exc:
    print(exc.error_code)
```

Full contract, including the environment variables behind every call:
[docs.aetherfy.com/agents/task-contract](https://docs.aetherfy.com/agents/task-contract).

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
| **Global Latency** | Single region, wherever you host it | **Served from the nearest region** |
| **Caching** | None | **Built in** |
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
client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here", workspace="default")
```

Relevant environment variables:

| Variable | Purpose |
|----------|---------|
| `AETHERFY_API_KEY` | Primary API key |
| `AETHERFY_VECTORS_API_KEY` | Alternative API key (useful when the same process talks to multiple Aetherfy services) |
| `AETHERFY_WORKSPACE_ID` | The agent's workspace id, read by `workspace="auto"` (the default). Set automatically on deployed agents |
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
    workspace="default",
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

- **Minimum**: Python 3.11
- **Tested**: Python 3.11, 3.12, 3.13, 3.14, 3.15

## 📚 Complete API Reference

### Collection Management

```python
# Create collection — returns the created Collection (with its resolved
# `regions` echoed back by the server). Pass `regions=[...]` to pin the
# collection to a subset of your scope; omit it to default to your full
# scope. (Distinct from the constructor's `api_region`, which only picks
# the endpoint to connect to.)
collection = client.create_collection(
    collection_name, vectors_config, distance=None, description=None, regions=None
)

# List collections
collections = client.get_collections()

# Get collection info
info = client.get_collection(name)

# Check existence
exists = client.collection_exists(name)

# Delete collection
client.delete_collection(name)
```

### Payload Indexes

A payload filter on an UNINDEXED key is scanned, not looked up. Index any key
you filter on for every read — a tenant id, a status, a timestamp you range
over.

```python
# field_schema: "keyword" | "integer" | "float" | "bool" | "geo" |
#               "datetime" | "uuid" | "text", or a parameterised object.
# Returns once the index is built, so a filter or an order_by scroll on the
# key can follow straight away. A large collection can take longer than the
# server's 25 s wait; the call then waits on, up to timeout= (default 600 s).
# Past it, RequestTimeoutError: the build carries on, and calling again waits.
client.create_field_index(collection_name, "tenant_id", "keyword")
client.create_field_index(collection_name, "ts", "integer", timeout=120)

# True when the collection exists, even if the field was never indexed.
# False only when the collection does not exist.
client.delete_field_index(collection_name, "tenant_id")
```

### Point Operations

```python
# Insert/update points
client.upsert(collection_name, points)

# Retrieve points
points = client.retrieve(collection_name, ids, with_payload=True, with_vectors=False)

# Delete points
client.delete(collection_name, point_ids_or_filter)

# Count points — count_filter takes a Filter or a plain dict, same as
# search / scroll / delete
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
    score_threshold=None,
    search_params=None      # engine params, sent verbatim as the body's `params`
)

# Trade latency for recall: a larger hnsw_ef makes the graph walk visit more
# candidates. Omit search_params to keep the tuned server default (hnsw_ef=100).
results = client.search(collection_name, query_vector, search_params={"hnsw_ef": 256})
```

`search_params` is passed through untranslated — the API and Qdrant own the
schema, so the SDK validates nothing and needs no release to track new params.
Note that the server cache key is derived from the request body, so the same
query at a different `hnsw_ef` is a separate cache entry (never a wrong hit).

### Schema Management (Aetherfy-specific)

```python
# Set, fetch, and remove a payload schema.
# enforcement defaults to "off" — pass "warn" or "strict" to have the
# server act on violations.
etag = client.set_schema(collection_name, schema, enforcement="off", description=None)
schema = client.get_schema(collection_name)           # None if not set
client.delete_schema(collection_name)

# Infer a schema from existing data
analysis = client.analyze_schema(collection_name, sample_size=1000)  # 100–10000

# Cache control (rarely needed)
client.refresh_schema(collection_name)
client.clear_schema_cache(collection_name=None)       # None clears all
```

### Usage statistics (Aetherfy-specific)

```python
usage = client.get_usage_stats()

usage.storage_bytes_used     # int  — bytes stored across every collection
usage.storage_limit_bytes    # int | None — None on an unlimited tier
usage.collections_count      # int  — active collections
usage.collections_limit      # int | None — None on an unlimited tier
usage.tier                   # str  — the plan's tier name
usage.active_regions         # list[str] — union of your collections' regions
usage.usage_percentage       # int  — storage %, 0 when there is no limit
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
    PermissionDeniedError,
    InvalidFilterError,
    WorkspaceNotFoundError,
    NotFoundError,              # parent of every not-found error above
)

try:
    results = client.search("nonexistent", [0.1, 0.2, 0.3])
except CollectionNotFoundError as e:
    print(f"Collection not found: {e}")
except WorkspaceNotFoundError as e:
    print(f"Workspace not found: {e.workspace or e.workspace_id}")
except AuthenticationError as e:
    print(f"Invalid API key: {e}")        # 401 only: the key itself is wrong
except PermissionDeniedError as e:
    # 403: the key is valid and this request is not allowed. Subclasses say
    # why: AccessLevelInsufficientError (.workspace, .level, .required_level),
    # CollectionProtectedError (.collection_name), AgentWorkspaceForbiddenError
    # (.workspace) and AgentKeyOutOfScopeError (.agent_id).
    print(f"Not allowed: {e}")
except InvalidFilterError as e:
    # 400: a filter key the API does not know
    print(f"Bad filter key {e.key!r} at {e.field}; did you mean {e.suggestion!r}?")
except RateLimitExceededError as e:
    print(f"Rate limit exceeded. Retry after: {e.retry_after}s")
except SchemaValidationError as e:
    for violation in e.errors:
        print(violation)
except QuotaExceededError as e:
    # A plan limit: .limit names it ("max_collections"), .allowed is the plan's
    # number (False for a feature the plan lacks, None for no cap), .current the
    # usage now (None for a yes/no feature) and .upgrade_url where to raise it.
    print(f"Quota {e.limit} exceeded: {e.current}/{e.allowed}; see {e.upgrade_url}")
```

## 🔧 Configuration Options

```python
client = AetherfyVectorsClient(
    api_key="your_api_key",                   # Required (or set AETHERFY_API_KEY)
    endpoint="https://vectors.aetherfy.com",  # Optional: Custom endpoint
    timeout=30.0,                             # Optional: Request timeout (seconds)
    workspace="default",                      # A workspace name, or "auto" (the default):
                                              #   inside an agent, AETHERFY_WORKSPACE_ID
)
```

## 📈 Monitoring & Observability

### Built-in Dashboard Data

```python
# Quota consumption, for a plan-usage dashboard
usage = client.get_usage_stats()

dashboard_data = {
    "storage_bytes": usage.storage_bytes_used,
    "storage_limit_bytes": usage.storage_limit_bytes,   # None = unlimited
    "storage_percent": usage.usage_percentage,
    "collections": usage.collections_count,
    "collections_limit": usage.collections_limit,       # None = unlimited
    "tier": usage.tier,
    "regions": usage.active_regions,
}
```

The fields are the endpoint's own, verbatim — no derived percentages beyond the
one it serves, and no unit conversion. Anything else you want (a collections
percentage, megabytes) is a calculation at your call site, where the "unlimited"
sentinels are yours to handle.

Aetherfy does not expose latency or cache-hit telemetry through this SDK. Measure
request latency at your own call site — it is the only number that reflects what
your application actually experienced.

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

2. **Get your API key** from the [Aetherfy dashboard](https://app.aetherfy.com/dashboard/settings/api-keys)

3. **Update imports**:
   ```python
   # from qdrant_client import QdrantClient
   from aetherfy_vectors import AetherfyVectorsClient
   ```

4. **Update initialization**:
   ```python
   # client = QdrantClient(host="localhost", port=6333)
   client = AetherfyVectorsClient(api_key="your_api_key", workspace="default")
   ```

5. **Run your tests.** Calls to the methods in the table below, with
   qdrant-client arguments passed by keyword, need no change. A refused
   argument or a `query_points` call fails at once, naming what to change.

6. **Optional**: Add analytics calls for insights

7. **Deploy** and enjoy global performance! 🚀

### Migration Compatibility

These qdrant-client methods exist here under the same name, taking the same
keyword arguments:
`create_collection`, `delete_collection`, `get_collections`, `get_collection`,
`collection_exists`, `upsert`, `delete`, `retrieve`, `search`, `scroll`,
`count`, `set_payload`, `overwrite_payload`, `delete_payload`, `close`.

The contract is qdrant-client **1.15.1**'s signatures, the last release that
still has `search`. A keyword argument that qdrant-client doesn't have raises
`TypeError`, the same as any Python function would: a misspelt or outdated
name fails straight away instead of being dropped without a word. The
qdrant-client arguments these methods don't name themselves are handled one
by one:

| Method | qdrant-client argument | Here | Why |
|--------|------------------------|------|-----|
| `upsert`, `delete`, `set_payload`, `overwrite_payload`, `delete_payload` | `wait` | accepted, no effect | Every point write is committed before the call returns, so `wait=True` already holds and `wait=False` gets a stronger guarantee |
| `upsert`, `delete`, `set_payload`, `overwrite_payload`, `delete_payload` | `ordering` | refused unless `'weak'` | Writes use Qdrant's default weak ordering; medium and strong are not provided |
| `upsert`, `delete`, `set_payload`, `overwrite_payload`, `delete_payload`, `retrieve`, `scroll`, `search`, `count` | `shard_key_selector` | refused unless `None` | Custom shard keys are not supported; placement is per collection (`regions=`) |
| `retrieve`, `scroll`, `search` | `consistency` | refused unless `None` | A read is answered by the region you are connected to; multi-replica read consistency is not provided |
| `search` | `append_payload` | refused unless `True` | Deprecated by qdrant-client itself; use `with_payload=False` |
| `create_collection` | `sparse_vectors_config` | refused unless `None` | Only dense vectors are supported |
| `create_collection` | `shard_number`, `sharding_method` | refused unless `None` | Sharding is managed by the service |
| `create_collection` | `replication_factor`, `write_consistency_factor` | refused unless `None` | Replication across regions is managed by the service (`regions=`) |
| `create_collection` | `on_disk_payload`, `hnsw_config`, `optimizers_config`, `wal_config`, `quantization_config`, `strict_mode_config` | refused unless `None` | Storage and index settings are fixed by the service and would not be applied |
| `create_collection` | `init_from` | refused unless `None` | Creating a collection from another one is not supported |
| `close` | `grpc_grace` | accepted, no effect | There is no gRPC channel to wait on |
| `create_collection`, `delete_collection`, `retrieve`, `scroll`, `search`, `count` | `timeout` | honoured | A deadline for the whole call, retries and backoff included, as in qdrant-client. The constructor's `timeout` is different: it bounds each attempt, and writes are retried up to 3 times |
| `scroll` | `order_by` | honoured | A payload key or `{"key": ..., "direction": ...}`, sent as-is |

A refused argument raises `TypeError` naming it and the reason, so a
migration finds out on its first run instead of in production.

The constructor is the one call the migration replaces outright, so it
accepts none of `QdrantClient`'s arguments. It takes only what it documents
(`api_key`, `endpoint`, `api_region`, `timeout`, `workspace`), and anything
else raises `TypeError`. Code written before 1.2.0 that passes `region=` now
fails at construction. Rename it to `api_region=`: until 1.2.0 the old name
was accepted and ignored, which quietly routed the client to the default
endpoint.

Positional arguments are accepted only as far as a method's parameter order
matches qdrant-client's. Past that point parameters are keyword-only, so a
positional call that would have bound the wrong parameter raises `TypeError`.
For example, `search`'s third positional is qdrant-client's `query_filter`
but was this SDK's `limit`.

## 🤝 Contributing

We welcome contributions! Open an issue or a pull request on
[GitHub](https://github.com/l-td/aetherfy-vectors-python-sdk).

### Development Setup

```bash
# Clone the repository
git clone https://github.com/l-td/aetherfy-vectors-python-sdk.git
cd aetherfy-vectors-python-sdk

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
- **API Reference**: [https://docs.aetherfy.com/vectors/api](https://docs.aetherfy.com/vectors/api)
- **SDK Reference**: [https://docs.aetherfy.com/vectors/sdk](https://docs.aetherfy.com/vectors/sdk)
- **Issues**: [GitHub Issues](https://github.com/l-td/aetherfy-vectors-python-sdk/issues)
- **Email**: [developers@aetherfy.com](mailto:developers@aetherfy.com)

## 🌟 Why Choose Aetherfy Vectors?

### For Solo Developers
- **Zero setup time** - start building immediately
- **Predictable pricing** - no surprise infrastructure costs
- **Global reach** - your users get fast responses worldwide

### For Teams
- **No DevOps overhead** - focus on your product, not infrastructure
- **Built-in monitoring** - comprehensive analytics out of the box
- **Reliable performance** - automatic failover

### For Enterprises
- **Global scalability** - handles millions of vectors effortlessly
- **Security first** - enterprise-grade authentication and audit logs
- **Cost effective** - pay only for what you use, no idle infrastructure

---

**Ready to experience global vector search?** [Get your API key](https://app.aetherfy.com/dashboard/settings/api-keys) and migrate in minutes! 🚀