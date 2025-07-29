"""
Advanced features example for Aetherfy Vectors SDK.

This example demonstrates advanced features unique to Aetherfy Vectors,
including global analytics, performance monitoring, and usage tracking.
"""

import time
from datetime import datetime
from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import VectorConfig, DistanceMetric


def analytics_example():
    """Demonstrate analytics and monitoring features."""
    
    client = AetherfyVectorsClient(
        api_key="afy_live_your_api_key_here"
    )
    
    print("=== Aetherfy Advanced Features Demo ===\n")
    
    try:
        # 1. Global Performance Analytics
        print("1. Global Performance Analytics")
        print("-" * 40)
        
        perf_analytics = client.get_performance_analytics(time_range="24h")
        
        print(f"Cache Hit Rate: {perf_analytics.cache_hit_rate:.1%}")
        print(f"Average Latency: {perf_analytics.avg_latency_ms:.1f}ms")
        print(f"Requests Per Second: {perf_analytics.requests_per_second:.0f}")
        print(f"Total Requests (24h): {perf_analytics.total_requests:,}")
        print(f"Error Rate: {perf_analytics.error_rate:.3%}")
        print(f"Active Regions: {len(perf_analytics.active_regions)}")
        
        print("\nRegion Performance:")
        for region, metrics in perf_analytics.region_performance.items():
            print(f"  {region}:")
            print(f"    Latency: {metrics.get('latency_ms', 0):.1f}ms")
            print(f"    RPS: {metrics.get('requests_per_second', 0):.0f}")
        
        # 2. Regional Performance Comparison
        print(f"\n2. Regional Performance Breakdown")
        print("-" * 40)
        
        # Get performance for different time ranges
        for time_range in ["1h", "24h", "7d"]:
            analytics = client.get_performance_analytics(time_range=time_range)
            print(f"{time_range:>3}: {analytics.avg_latency_ms:>6.1f}ms avg, "
                  f"{analytics.cache_hit_rate:>5.1%} cache hit")
        
        # 3. Usage Statistics and Limits
        print(f"\n3. Usage Statistics & Limits")
        print("-" * 40)
        
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
        
        # Usage warnings
        if usage.collections_usage_percent > 80:
            print("⚠️  Warning: Collection usage above 80%")
        if usage.points_usage_percent > 80:
            print("⚠️  Warning: Points usage above 80%")
        if usage.requests_usage_percent > 80:
            print("⚠️  Warning: Request usage above 80%")
        
        # 4. Collection-Specific Analytics
        print(f"\n4. Collection Analytics")
        print("-" * 40)
        
        # First, let's create a test collection with some data
        test_collection = "analytics_demo"
        
        try:
            # Create collection
            client.create_collection(
                test_collection,
                VectorConfig(size=4, distance=DistanceMetric.COSINE)
            )
            
            # Add some test data
            test_points = [
                {"id": f"point_{i}", "vector": [i*0.1, i*0.2, i*0.3, i*0.4], 
                 "payload": {"category": f"cat_{i%3}", "value": i}}
                for i in range(10)
            ]
            client.upsert(test_collection, test_points)
            
            # Perform some searches to generate analytics data
            for i in range(5):
                client.search(test_collection, [i*0.1, i*0.2, i*0.3, i*0.4], limit=3)
            
            # Get collection analytics
            coll_analytics = client.get_collection_analytics(test_collection)
            print(f"Collection: {coll_analytics.collection_name}")
            print(f"Total Points: {coll_analytics.total_points:,}")
            print(f"Search Requests: {coll_analytics.search_requests:,}")
            print(f"Avg Search Latency: {coll_analytics.avg_search_latency_ms:.1f}ms")
            print(f"Cache Hit Rate: {coll_analytics.cache_hit_rate:.1%}")
            print(f"Top Regions: {', '.join(coll_analytics.top_regions)}")
            if coll_analytics.storage_size_mb:
                print(f"Storage Size: {coll_analytics.storage_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"Collection analytics demo skipped: {e}")
        
        # 5. Cache Performance Monitoring
        print(f"\n5. Cache Performance")
        print("-" * 40)
        
        try:
            cache_analytics = client.analytics.get_cache_analytics(time_range="24h")
            print(f"Cache Hit Rate: {cache_analytics.get('hit_rate', 0):.1%}")
            print(f"Cache Miss Rate: {cache_analytics.get('miss_rate', 0):.1%}")
            print(f"Total Cache Requests: {cache_analytics.get('total_requests', 0):,}")
            print(f"Cache Size: {cache_analytics.get('cache_size_mb', 0):.1f} MB")
        except Exception as e:
            print(f"Cache analytics: {e}")
        
        # 6. Top Collections by Activity
        print(f"\n6. Top Collections")
        print("-" * 40)
        
        try:
            top_collections = client.analytics.get_top_collections(
                metric="requests", 
                time_range="24h", 
                limit=5
            )
            
            print("Most Active Collections (by requests):")
            for i, collection in enumerate(top_collections, 1):
                print(f"  {i}. {collection['name']} - {collection['requests']:,} requests")
        except Exception as e:
            print(f"Top collections: {e}")
        
        # 7. Region-Specific Performance
        print(f"\n7. Region Performance Details")
        print("-" * 40)
        
        try:
            region_perf = client.analytics.get_region_performance(time_range="24h")
            
            print("Region Performance Rankings:")
            # Sort by latency (ascending)
            sorted_regions = sorted(
                region_perf.items(),
                key=lambda x: x[1].get('latency_ms', float('inf'))
            )
            
            for i, (region, metrics) in enumerate(sorted_regions, 1):
                latency = metrics.get('latency_ms', 0)
                rps = metrics.get('requests_per_second', 0)
                print(f"  {i}. {region}: {latency:.1f}ms latency, {rps:.0f} RPS")
        except Exception as e:
            print(f"Region performance: {e}")
        
        # 8. Real-time Performance Monitoring
        print(f"\n8. Real-time Monitoring Example")
        print("-" * 40)
        
        if test_collection and client.collection_exists(test_collection):
            print("Performing real-time performance test...")
            
            start_time = time.time()
            search_times = []
            
            # Perform multiple searches and measure latency
            for i in range(5):
                search_start = time.time()
                results = client.search(
                    test_collection, 
                    [0.1, 0.2, 0.3, 0.4], 
                    limit=3
                )
                search_end = time.time()
                
                search_latency = (search_end - search_start) * 1000  # Convert to ms
                search_times.append(search_latency)
                print(f"  Search {i+1}: {search_latency:.1f}ms ({len(results)} results)")
            
            avg_latency = sum(search_times) / len(search_times)
            min_latency = min(search_times)
            max_latency = max(search_times)
            
            print(f"\nPerformance Summary:")
            print(f"  Average: {avg_latency:.1f}ms")
            print(f"  Min: {min_latency:.1f}ms")
            print(f"  Max: {max_latency:.1f}ms")
            print(f"  Consistency: {((max_latency - min_latency) / avg_latency * 100):.1f}% variance")
        
        # Cleanup
        try:
            if test_collection:
                client.delete_collection(test_collection)
                print(f"\n✓ Cleaned up test collection")
        except:
            pass
        
    except Exception as e:
        print(f"\n✗ Analytics demo error: {e}")
        print("Note: Some features require a live API connection and data")


def performance_optimization_tips():
    """Display performance optimization recommendations."""
    
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATION TIPS")
    print("="*60)
    
    tips = [
        "🚀 Use batch operations (upsert multiple points at once)",
        "💾 Monitor cache hit rates - aim for >80% for best performance",
        "🌍 Global routing is automatic - no configuration needed",
        "📊 Use analytics to identify bottlenecks and usage patterns",
        "⚡ Smaller payloads = faster responses (avoid large JSON objects)",
        "🔍 Use filters to reduce search scope and improve speed",
        "📈 Monitor usage stats to prevent hitting rate limits",
        "🎯 Choose appropriate vector dimensions (smaller = faster)",
        "🔄 Use appropriate distance metrics for your use case",
        "📱 Set reasonable timeouts based on your performance requirements"
    ]
    
    for tip in tips:
        print(tip)
    
    print("\nFor more optimization advice, check your performance analytics!")
    print("="*60)


def monitoring_dashboard_example():
    """Example of building a simple monitoring dashboard."""
    
    print("\n" + "="*60)
    print("MONITORING DASHBOARD EXAMPLE")
    print("="*60)
    
    client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here")
    
    try:
        # Collect all metrics
        perf = client.get_performance_analytics()
        usage = client.get_usage_stats()
        
        # Simple dashboard format
        print(f"📊 AETHERFY VECTORS DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        
        # Health Status
        health = "🟢 HEALTHY"
        if perf.error_rate > 0.01:  # >1% error rate
            health = "🟡 WARNING"
        if perf.error_rate > 0.05:  # >5% error rate
            health = "🔴 CRITICAL"
        
        print(f"Status: {health}")
        print(f"Uptime Regions: {len(perf.active_regions)}/3+ regions")
        
        # Performance Metrics
        print(f"\n🚀 PERFORMANCE (24h)")
        print(f"  Latency: {perf.avg_latency_ms:.1f}ms average")
        print(f"  Cache Hit: {perf.cache_hit_rate:.1%}")
        print(f"  Throughput: {perf.requests_per_second:.0f} RPS")
        print(f"  Error Rate: {perf.error_rate:.3%}")
        
        # Usage Metrics
        print(f"\n📈 USAGE ({usage.plan_name} Plan)")
        print(f"  Collections: {usage.current_collections}/{usage.max_collections} ({usage.collections_usage_percent:.0f}%)")
        print(f"  Points: {usage.current_points:,}/{usage.max_points:,} ({usage.points_usage_percent:.0f}%)")
        print(f"  Requests: {usage.requests_this_month:,}/{usage.max_requests_per_month:,} ({usage.requests_usage_percent:.0f}%)")
        
        # Alerts
        alerts = []
        if usage.collections_usage_percent > 90:
            alerts.append("⚠️  Collection limit near maximum")
        if usage.points_usage_percent > 90:
            alerts.append("⚠️  Point limit near maximum")
        if usage.requests_usage_percent > 90:
            alerts.append("⚠️  Request limit near maximum")
        if perf.error_rate > 0.01:
            alerts.append("⚠️  Elevated error rate detected")
        
        if alerts:
            print(f"\n🚨 ALERTS")
            for alert in alerts:
                print(f"  {alert}")
        else:
            print(f"\n✅ No alerts - system operating normally")
        
        print("-" * 60)
        
    except Exception as e:
        print(f"Dashboard unavailable: {e}")


if __name__ == "__main__":
    analytics_example()
    performance_optimization_tips()
    monitoring_dashboard_example()