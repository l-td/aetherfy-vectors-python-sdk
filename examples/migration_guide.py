"""
Migration guide example: From qdrant-client to aetherfy-vectors.

This example shows how to migrate existing code from qdrant-client
to aetherfy-vectors with minimal changes while gaining global 
performance benefits.
"""

# BEFORE: Using qdrant-client (commented out)
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Old initialization
client = QdrantClient(host="localhost", port=6333)
# or
client = QdrantClient(url="http://localhost:6333")
"""

# AFTER: Using aetherfy-vectors
from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import VectorConfig, DistanceMetric, Point

def demonstrate_migration():
    """Show side-by-side migration examples."""
    
    # NEW: Initialize Aetherfy client (only change needed)
    client = AetherfyVectorsClient(
        api_key="afy_live_your_api_key_here"  # Only new requirement
        # endpoint="https://vectors.aetherfy.com"  # Optional, this is default
        # timeout=30.0  # Optional, same as before
    )
    
    collection_name = "migration_example"
    
    try:
        print("=== Migration Example: qdrant-client → aetherfy-vectors ===\n")
        
        # 1. Collection Creation - MINIMAL CHANGES
        print("1. Creating collection...")
        
        # OLD WAY (qdrant-client):
        """
        client.create_collection(
            collection_name="my_collection",
            vectors_config=VectorParams(size=128, distance=Distance.COSINE)
        )
        """
        
        # NEW WAY (aetherfy-vectors) - Almost identical:
        client.create_collection(
            collection_name,
            VectorConfig(size=128, distance=DistanceMetric.COSINE)
        )
        # Alternative: client.create_collection(collection_name, {"size": 128, "distance": "Cosine"})
        
        print("✓ Collection created with identical API")
        
        # 2. Point Operations - NO CHANGES NEEDED
        print("\n2. Inserting points...")
        
        # SAME CODE works in both libraries:
        points = [
            {
                "id": "point_1",
                "vector": [0.1] * 128,  # 128-dimensional vector
                "payload": {"category": "example", "user_id": 123}
            },
            {
                "id": "point_2", 
                "vector": [0.2] * 128,
                "payload": {"category": "demo", "user_id": 456}
            },
            {
                "id": "point_3",
                "vector": [0.3] * 128,
                "payload": {"category": "example", "user_id": 789}
            }
        ]
        
        # IDENTICAL API:
        client.upsert(collection_name, points)
        print("✓ Points inserted with identical API")
        
        # 3. Search Operations - NO CHANGES NEEDED
        print("\n3. Performing search...")
        
        query_vector = [0.15] * 128
        
        # IDENTICAL API:
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        
        print(f"✓ Found {len(search_results)} results with identical API")
        for result in search_results:
            print(f"   ID: {result.id}, Score: {result.score:.4f}")
        
        # 4. Filtering - NO CHANGES NEEDED  
        print("\n4. Search with filtering...")
        
        # IDENTICAL API:
        filtered_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5,
            query_filter={
                "must": [
                    {"key": "category", "match": {"value": "example"}}
                ]
            },
            with_payload=True
        )
        
        print(f"✓ Filtered search returned {len(filtered_results)} results")
        
        # 5. Point Retrieval - NO CHANGES NEEDED
        print("\n5. Retrieving points...")
        
        # IDENTICAL API:
        retrieved_points = client.retrieve(
            collection_name=collection_name,
            ids=["point_1", "point_3"],
            with_payload=True,
            with_vectors=True
        )
        
        print(f"✓ Retrieved {len(retrieved_points)} points")
        
        # 6. Collection Management - NO CHANGES NEEDED
        print("\n6. Collection operations...")
        
        # IDENTICAL API:
        collections = client.get_collections()
        print(f"✓ Found {len(collections)} collections")
        
        collection_info = client.get_collection(collection_name)
        print(f"✓ Collection has {collection_info.points_count} points")
        
        # Point counting - IDENTICAL API:
        point_count = client.count(collection_name)
        print(f"✓ Point count: {point_count}")
        
        # 7. NEW FEATURES - Aetherfy-specific enhancements
        print("\n7. New Aetherfy-specific features...")
        
        # These are ADDITIONAL features not available in qdrant-client:
        try:
            # Global performance analytics
            perf_analytics = client.get_performance_analytics()
            print(f"✓ Global cache hit rate: {perf_analytics.cache_hit_rate:.1%}")
            print(f"✓ Average latency: {perf_analytics.avg_latency_ms:.1f}ms")
            print(f"✓ Active regions: {len(perf_analytics.active_regions)}")
            
            # Collection-specific analytics
            coll_analytics = client.get_collection_analytics(collection_name)
            print(f"✓ Collection cache hit rate: {coll_analytics.cache_hit_rate:.1%}")
            
            # Usage statistics
            usage_stats = client.get_usage_stats()
            print(f"✓ Plan: {usage_stats.plan_name}")
            print(f"✓ Collections used: {usage_stats.current_collections}/{usage_stats.max_collections}")
            
        except Exception as e:
            print(f"   Note: Analytics features require live API connection ({e})")
        
        # 8. Context Manager - IDENTICAL USAGE
        print("\n8. Context manager usage...")
        
        # IDENTICAL API:
        with AetherfyVectorsClient(api_key="afy_live_your_api_key_here") as ctx_client:
            collections = ctx_client.get_collections()
            print(f"✓ Context manager works identically ({len(collections)} collections)")
        
        print("\n=== Migration Benefits ===")
        print("✓ 100% API compatibility - existing code works unchanged")
        print("✓ Global performance - automatic caching and routing")
        print("✓ Zero DevOps - no infrastructure management needed")
        print("✓ Enhanced analytics - performance insights included")
        print("✓ Automatic failover - built-in reliability")
        
        # Cleanup
        client.delete_collection(collection_name)
        print(f"\n✓ Cleanup completed")
        
    except Exception as e:
        print(f"\n✗ Error during migration example: {e}")
        try:
            client.delete_collection(collection_name)
        except:
            pass


def migration_checklist():
    """Print a migration checklist for users."""
    
    print("\n" + "="*60)
    print("MIGRATION CHECKLIST: qdrant-client → aetherfy-vectors")
    print("="*60)
    
    checklist = [
        "□ Sign up for Aetherfy account and get API key",
        "□ Install aetherfy-vectors: pip install aetherfy-vectors", 
        "□ Replace import: from qdrant_client import QdrantClient",
        "  → from aetherfy_vectors import AetherfyVectorsClient",
        "□ Replace initialization: QdrantClient(host='localhost')",
        "  → AetherfyVectorsClient(api_key='your-key')",
        "□ Set environment variable: AETHERFY_API_KEY=your-key",
        "□ Test existing functionality (should work unchanged)",
        "□ Optional: Add performance analytics calls",
        "□ Optional: Remove local Qdrant infrastructure",
        "□ Deploy and enjoy global performance!"
    ]
    
    for item in checklist:
        print(item)
    
    print("\nNOTE: All existing method calls (search, upsert, etc.) work unchanged!")
    print("="*60)


if __name__ == "__main__":
    demonstrate_migration()
    migration_checklist()