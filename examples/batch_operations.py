"""
Batch operations example for Aetherfy Vectors SDK.

This example demonstrates efficient batch processing techniques
for large-scale vector operations, including bulk inserts,
batch searches, and performance optimization strategies.
"""

import time
import random
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import VectorConfig, DistanceMetric


def generate_sample_data(count: int, vector_dim: int = 128) -> List[Dict[str, Any]]:
    """Generate sample vector data for testing."""
    
    categories = ["technology", "science", "business", "entertainment", "sports"]
    
    points = []
    for i in range(count):
        # Generate random vector
        vector = [random.uniform(-1.0, 1.0) for _ in range(vector_dim)]
        
        # Create payload with sample metadata
        payload = {
            "id": i,
            "category": random.choice(categories),
            "timestamp": int(time.time()) + i,
            "score": random.uniform(0.0, 1.0),
            "tags": random.sample(["tag1", "tag2", "tag3", "tag4", "tag5"], 
                                 random.randint(1, 3))
        }
        
        points.append({
            "id": f"doc_{i}",
            "vector": vector,
            "payload": payload
        })
    
    return points


def batch_upsert_example():
    """Demonstrate efficient batch upsert operations."""
    
    client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here")
    collection_name = "batch_demo"
    
    try:
        print("=== Batch Upsert Example ===\n")
        
        # Create collection
        print("Creating collection...")
        client.create_collection(
            collection_name,
            VectorConfig(size=128, distance=DistanceMetric.COSINE)
        )
        
        # Generate test data
        total_points = 10000
        batch_size = 1000
        
        print(f"Generating {total_points:,} test points...")
        all_points = generate_sample_data(total_points, vector_dim=128)
        
        # Batch upsert with timing
        print(f"Upserting in batches of {batch_size:,}...")
        
        start_time = time.time()
        batch_times = []
        
        for i in range(0, total_points, batch_size):
            batch_start = time.time()
            
            batch = all_points[i:i + batch_size]
            client.upsert(collection_name, batch)
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            print(f"  Batch {i//batch_size + 1}: {len(batch):,} points in {batch_time:.2f}s "
                  f"({len(batch)/batch_time:.0f} points/sec)")
        
        total_time = time.time() - start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        total_throughput = total_points / total_time
        
        print(f"\n📊 Batch Upsert Results:")
        print(f"  Total Points: {total_points:,}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Average Batch Time: {avg_batch_time:.2f}s")
        print(f"  Overall Throughput: {total_throughput:.0f} points/sec")
        
        # Verify point count
        final_count = client.count(collection_name)
        print(f"  Final Point Count: {final_count:,}")
        
        return collection_name
        
    except Exception as e:
        print(f"Batch upsert error: {e}")
        return None


def batch_search_example(collection_name: str):
    """Demonstrate efficient batch search operations."""
    
    client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here")
    
    try:
        print(f"\n=== Batch Search Example ===\n")
        
        # Generate multiple query vectors
        num_queries = 100
        query_vectors = []
        
        print(f"Generating {num_queries} query vectors...")
        for i in range(num_queries):
            vector = [random.uniform(-1.0, 1.0) for _ in range(128)]
            query_vectors.append(vector)
        
        # Sequential search (baseline)
        print(f"\n1. Sequential Search (baseline)")
        sequential_start = time.time()
        
        sequential_results = []
        for i, query_vector in enumerate(query_vectors[:20]):  # Limit for demo
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=10,
                with_payload=True
            )
            sequential_results.append(results)
            
            if i % 5 == 0:
                print(f"  Query {i+1}: {len(results)} results")
        
        sequential_time = time.time() - sequential_start
        sequential_qps = 20 / sequential_time
        
        print(f"  Sequential Time: {sequential_time:.2f}s ({sequential_qps:.1f} QPS)")
        
        # Parallel search with threading
        print(f"\n2. Parallel Search (threaded)")
        parallel_start = time.time()
        
        def search_query(query_vector):
            return client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=10,
                with_payload=True
            )
        
        parallel_results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(search_query, qv): i 
                for i, qv in enumerate(query_vectors[:20])
            }
            
            # Collect results
            for future in as_completed(future_to_query):
                query_idx = future_to_query[future]
                try:
                    results = future.result()
                    parallel_results.append((query_idx, results))
                    print(f"  Query {query_idx+1}: {len(results)} results")
                except Exception as e:
                    print(f"  Query {query_idx+1} failed: {e}")
        
        parallel_time = time.time() - parallel_start
        parallel_qps = 20 / parallel_time
        speedup = sequential_time / parallel_time
        
        print(f"  Parallel Time: {parallel_time:.2f}s ({parallel_qps:.1f} QPS)")
        print(f"  Speedup: {speedup:.1f}x faster")
        
        # Batch search with filters
        print(f"\n3. Filtered Batch Search")
        
        categories = ["technology", "science", "business"]
        filtered_start = time.time()
        
        for category in categories:
            filter_condition = {
                "must": [{"key": "category", "match": {"value": category}}]
            }
            
            category_results = []
            for query_vector in query_vectors[:10]:  # Limit for demo
                results = client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=5,
                    query_filter=filter_condition,
                    with_payload=True
                )
                category_results.extend(results)
            
            print(f"  {category}: {len(category_results)} total results")
        
        filtered_time = time.time() - filtered_start
        print(f"  Filtered Search Time: {filtered_time:.2f}s")
        
    except Exception as e:
        print(f"Batch search error: {e}")


def batch_delete_example(collection_name: str):
    """Demonstrate efficient batch delete operations."""
    
    client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here")
    
    try:
        print(f"\n=== Batch Delete Example ===\n")
        
        initial_count = client.count(collection_name)
        print(f"Initial point count: {initial_count:,}")
        
        # 1. Delete by ID batches
        print(f"\n1. Delete by ID batches")
        
        # Generate IDs to delete (first 1000 points)
        ids_to_delete = [f"doc_{i}" for i in range(1000)]
        batch_size = 100
        
        delete_start = time.time()
        
        for i in range(0, len(ids_to_delete), batch_size):
            batch_ids = ids_to_delete[i:i + batch_size]
            client.delete(collection_name, batch_ids)
            print(f"  Deleted batch {i//batch_size + 1}: {len(batch_ids)} points")
        
        delete_time = time.time() - delete_start
        count_after_id_delete = client.count(collection_name)
        
        print(f"  Delete Time: {delete_time:.2f}s")
        print(f"  Points Remaining: {count_after_id_delete:,}")
        
        # 2. Delete by filter conditions
        print(f"\n2. Delete by filter conditions")
        
        # Delete all points from specific categories
        categories_to_delete = ["technology", "sports"]
        
        for category in categories_to_delete:
            filter_condition = {
                "must": [{"key": "category", "match": {"value": category}}]
            }
            
            # Count points before deletion
            count_before = client.count(collection_name, count_filter=filter_condition)
            
            # Delete by filter
            client.delete(collection_name, filter_condition)
            
            print(f"  Deleted {count_before:,} points from '{category}' category")
        
        final_count = client.count(collection_name)
        total_deleted = initial_count - final_count
        
        print(f"\n📊 Batch Delete Results:")
        print(f"  Initial Count: {initial_count:,}")
        print(f"  Final Count: {final_count:,}")
        print(f"  Total Deleted: {total_deleted:,}")
        
    except Exception as e:
        print(f"Batch delete error: {e}")


def batch_retrieve_example(collection_name: str):
    """Demonstrate efficient batch retrieve operations."""
    
    client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here")
    
    try:
        print(f"\n=== Batch Retrieve Example ===\n")
        
        # Generate IDs to retrieve
        total_ids = 500
        batch_size = 50
        
        # Get available IDs (assuming some points still exist)
        all_ids = [f"doc_{i}" for i in range(2000, 2000 + total_ids)]
        
        print(f"Retrieving {total_ids} points in batches of {batch_size}...")
        
        retrieve_start = time.time()
        all_retrieved = []
        
        for i in range(0, total_ids, batch_size):
            batch_ids = all_ids[i:i + batch_size]
            
            batch_start = time.time()
            retrieved_points = client.retrieve(
                collection_name=collection_name,
                ids=batch_ids,
                with_payload=True,
                with_vectors=False  # Faster without vectors
            )
            batch_time = time.time() - batch_start
            
            all_retrieved.extend(retrieved_points)
            
            print(f"  Batch {i//batch_size + 1}: {len(retrieved_points)}/{len(batch_ids)} points "
                  f"in {batch_time:.3f}s")
        
        total_retrieve_time = time.time() - retrieve_start
        
        print(f"\n📊 Batch Retrieve Results:")
        print(f"  Total Retrieved: {len(all_retrieved):,}")
        print(f"  Total Time: {total_retrieve_time:.2f}s")
        print(f"  Throughput: {len(all_retrieved)/total_retrieve_time:.0f} points/sec")
        
        # Analyze retrieved data
        if all_retrieved:
            categories = {}
            for point in all_retrieved:
                if point.get('payload', {}).get('category'):
                    cat = point['payload']['category']
                    categories[cat] = categories.get(cat, 0) + 1
            
            print(f"\n  Retrieved by Category:")
            for cat, count in sorted(categories.items()):
                print(f"    {cat}: {count:,} points")
        
    except Exception as e:
        print(f"Batch retrieve error: {e}")


def performance_optimization_tips():
    """Print performance optimization tips for batch operations."""
    
    print(f"\n" + "="*60)
    print("BATCH OPERATIONS PERFORMANCE TIPS")
    print("="*60)
    
    tips = [
        "📦 Optimal batch size: 100-1000 points per batch",
        "⚡ Use parallel requests for independent operations",
        "🎯 Include only necessary data (avoid large payloads)",
        "🔍 Use filters to reduce data transfer",
        "📊 Monitor cache hit rates during bulk operations",
        "⏱️  Set appropriate timeouts for large batches",
        "🔄 Implement retry logic for failed batches",
        "💾 Consider memory usage with large datasets",
        "📈 Use analytics to monitor batch performance",
        "🚀 Batch similar operations together for better caching"
    ]
    
    for tip in tips:
        print(tip)
    
    print("="*60)


def main():
    """Run all batch operation examples."""
    
    print("Starting batch operations demonstration...\n")
    
    # Run batch upsert example
    collection_name = batch_upsert_example()
    
    if collection_name:
        # Run other examples using the created collection
        batch_search_example(collection_name)
        batch_retrieve_example(collection_name)
        batch_delete_example(collection_name)
        
        # Cleanup
        try:
            client = AetherfyVectorsClient(api_key="afy_live_your_api_key_here")
            client.delete_collection(collection_name)
            print(f"\n✓ Cleaned up collection '{collection_name}'")
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    # Show optimization tips
    performance_optimization_tips()
    
    print(f"\nBatch operations demonstration completed!")


if __name__ == "__main__":
    main()