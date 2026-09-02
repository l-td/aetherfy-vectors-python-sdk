"""
Advanced features example for Aetherfy Vectors SDK.

Two things this SDK gives you beyond plain vector operations: your consumption
against your plan's limits, and — because Aetherfy does NOT report latency to
you — a worked example of measuring it yourself at the call site.

This file used to demonstrate global performance analytics as well. Those
methods were deleted: the backend synthesised the per-region latency and request
figures it returned rather than measuring them, so charting them told you about
a random number generator, not your database. Nothing replaced them, and a
placeholder that returns plausible numbers would be worse than their absence.
The measurement in `latency_example()` below is the honest substitute — it is
the only latency figure that reflects what your application actually
experienced, including the network between you and us.
"""

import time
from datetime import datetime
from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import VectorConfig, DistanceMetric


def usage_example():
    """Consumption against plan limits, and the warnings worth acting on."""

    client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here")

    print("=== Aetherfy Usage & Limits ===\n")

    try:
        usage = client.get_usage_stats()

        print(f"Tier: {usage.tier}")
        # Both limit fields are None on an unlimited plan — one sentinel, the
        # same for each. The SDK passes them through as served.
        if usage.collections_limit is None:
            print(f"Collections: {usage.collections_count:,} (unlimited)")
        else:
            print(f"Collections: {usage.collections_count:,}/{usage.collections_limit:,}")
        if usage.storage_limit_bytes is None:
            print(f"Storage: {usage.storage_bytes_used:,} bytes (unlimited)")
        else:
            print(f"Storage: {usage.storage_bytes_used:,}/{usage.storage_limit_bytes:,} "
                  f"bytes ({usage.usage_percentage}%)")
        print(f"Replicating to: {', '.join(usage.active_regions) or '(no collections)'}")

        # `is not None` before the arithmetic, not `> 0`: an unlimited tier
        # sends None, and comparing that to an int raises TypeError.
        if usage.collections_limit is not None and usage.collections_limit > 0:
            if (usage.collections_count / usage.collections_limit) > 0.8:
                print("⚠️  Warning: Collection usage above 80%")
        if usage.usage_percentage > 80:
            print("⚠️  Warning: Storage usage above 80%")

    except Exception as e:
        print(f"Usage stats unavailable: {e}")

    finally:
        client.close()


def latency_example():
    """Measure search latency yourself — Aetherfy does not report it to you."""

    client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here")
    collection = "latency_demo"

    print("\n=== Measured Search Latency ===\n")

    try:
        client.create_collection(
            collection,
            VectorConfig(size=4, distance=DistanceMetric.COSINE),
        )

        client.upsert(collection, [
            {"id": i, "vector": [i * 0.1, i * 0.2, i * 0.3, i * 0.4],
             "payload": {"category": f"cat_{i % 3}", "value": i}}
            for i in range(10)
        ])

        # Warm the caches so the timings below are steady-state rather than
        # dominated by the first-request path.
        for i in range(5):
            client.search(collection, [i * 0.1, i * 0.2, i * 0.3, i * 0.4], limit=3)

        search_times = []
        for i in range(5):
            started = time.time()
            results = client.search(collection, [0.1, 0.2, 0.3, 0.4], limit=3)
            elapsed_ms = (time.time() - started) * 1000
            search_times.append(elapsed_ms)
            print(f"  Search {i + 1}: {elapsed_ms:.1f}ms ({len(results)} results)")

        avg = sum(search_times) / len(search_times)
        lo, hi = min(search_times), max(search_times)

        print(f"\nMeasured at this call site, {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"  Average: {avg:.1f}ms")
        print(f"  Min: {lo:.1f}ms")
        print(f"  Max: {hi:.1f}ms")
        print(f"  Spread: {((hi - lo) / avg * 100):.1f}% of the mean")
        print("\nThis includes your network path to Aetherfy, which is the")
        print("number that matters and the one only you can observe.")

    except Exception as e:
        print(f"Latency demo skipped: {e}")

    finally:
        try:
            client.delete_collection(collection)
            print("\n✓ Cleaned up the demo collection")
        except Exception:
            pass
        client.close()


def performance_optimization_tips():
    """Display performance optimization recommendations."""

    print("\n" + "=" * 60)
    print("PERFORMANCE OPTIMIZATION TIPS")
    print("=" * 60)

    tips = [
        "🚀 Use batch operations (upsert multiple points at once)",
        "🌍 Global routing is automatic - no configuration needed",
        "⚡ Smaller payloads = faster responses (avoid large JSON objects)",
        "🔍 Use filters to reduce search scope and improve speed",
        "📈 Monitor usage stats to prevent hitting plan limits",
        "🎯 Choose appropriate vector dimensions (smaller = faster)",
        "🔄 Use appropriate distance metrics for your use case",
        "📱 Set reasonable timeouts based on your performance requirements",
        "⏱️  Time your own calls — that is the latency your users feel",
    ]

    for tip in tips:
        print(tip)

    print("=" * 60)


if __name__ == "__main__":
    usage_example()
    latency_example()
    performance_optimization_tips()
