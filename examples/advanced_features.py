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

        print(f"Plan: {usage.plan_name}")
        print(f"Collections: {usage.current_collections:,}/{usage.max_collections:,} "
              f"({usage.collections_usage_percent:.1f}%)")
        print(f"Points: {usage.current_points:,}/{usage.max_points:,} "
              f"({usage.points_usage_percent:.1f}%)")
        print(f"Requests: {usage.requests_this_month:,}/{usage.max_requests_per_month:,} "
              f"({usage.requests_usage_percent:.1f}%)")
        print(f"Storage: {usage.storage_used_mb:.1f}/{usage.max_storage_mb:.1f} MB "
              f"({usage.storage_usage_percent:.1f}%)")

        if usage.collections_usage_percent > 80:
            print("⚠️  Warning: Collection usage above 80%")
        if usage.points_usage_percent > 80:
            print("⚠️  Warning: Points usage above 80%")
        if usage.requests_usage_percent > 80:
            print("⚠️  Warning: Request usage above 80%")

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
