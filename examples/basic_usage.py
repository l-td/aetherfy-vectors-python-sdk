"""
Basic usage examples for Aetherfy Vectors SDK.

This example demonstrates the fundamental operations you can perform
with the Aetherfy Vectors client, including creating collections,
managing points, and performing vector searches.
"""

from aetherfy_vectors import AetherfyVectorsClient
from aetherfy_vectors.models import VectorConfig, DistanceMetric, Point

def main():
    """Demonstrate basic SDK usage."""
    
    # Initialize the client with your API key
    # You can also set the AETHERFY_API_KEY environment variable
    client = AetherfyVectorsClient(
        api_key="afy_live_your_api_key_here",  # Replace with your actual API key
        timeout=30.0  # Optional: request timeout in seconds
    )
    
    # Collection name for this example
    collection_name = "basic_example"
    
    try:
        # 1. Create a collection
        print("Creating collection...")
        vector_config = VectorConfig(
            size=4,  # Vector dimension
            distance=DistanceMetric.COSINE  # Distance metric for similarity
        )
        
        client.create_collection(collection_name, vector_config)
        print(f"✓ Collection '{collection_name}' created successfully")
        
        # 2. Prepare some sample data
        sample_points = [
            {
                "id": "doc_1",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {
                    "title": "Introduction to Machine Learning",
                    "category": "education",
                    "tags": ["ml", "ai", "beginner"]
                }
            },
            {
                "id": "doc_2", 
                "vector": [0.5, 0.6, 0.7, 0.8],
                "payload": {
                    "title": "Advanced Neural Networks",
                    "category": "education", 
                    "tags": ["neural", "deep-learning", "advanced"]
                }
            },
            {
                "id": "doc_3",
                "vector": [0.9, 0.1, 0.5, 0.3],
                "payload": {
                    "title": "Vector Databases Explained",
                    "category": "technology",
                    "tags": ["vectors", "database", "search"]
                }
            }
        ]
        
        # 3. Insert points into the collection
        print("Inserting points...")
        client.upsert(collection_name, sample_points)
        print(f"✓ Inserted {len(sample_points)} points successfully")
        
        # 4. Search for similar vectors
        print("Performing vector search...")
        query_vector = [0.2, 0.3, 0.4, 0.5]  # Example query vector
        
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3,  # Return top 3 results
            with_payload=True  # Include payload in results
        )
        
        print(f"✓ Found {len(search_results)} similar vectors:")
        for i, result in enumerate(search_results, 1):
            print(f"  {i}. ID: {result.id}")
            print(f"     Score: {result.score:.4f}")
            print(f"     Title: {result.payload.get('title', 'N/A')}")
            print(f"     Category: {result.payload.get('category', 'N/A')}")
            print()
        
        # 5. Get collection information
        print("Getting collection info...")
        collection_info = client.get_collection(collection_name)
        print(f"✓ Collection info:")
        print(f"  Name: {collection_info.name}")
        print(f"  Vector size: {collection_info.config.size}")
        print(f"  Distance metric: {collection_info.config.distance.value}")
        print(f"  Points count: {collection_info.points_count}")
        print()
        
        # 6. Count points in collection
        total_points = client.count(collection_name)
        print(f"✓ Total points in collection: {total_points}")
        
        # 7. Retrieve specific points
        print("Retrieving specific points...")
        retrieved_points = client.retrieve(
            collection_name=collection_name,  
            ids=["doc_1", "doc_3"],
            with_payload=True,
            with_vectors=True
        )
        
        print(f"✓ Retrieved {len(retrieved_points)} points:")
        for point in retrieved_points:
            print(f"  ID: {point['id']}")
            print(f"  Title: {point['payload']['title']}")
            print(f"  Vector: {point['vector']}")
            print()
        
        # 8. Search with filters
        print("Searching with filters...")
        filtered_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=5,
            query_filter={
                "must": [
                    {"key": "category", "match": {"value": "education"}}
                ]
            },
            with_payload=True
        )
        
        print(f"✓ Found {len(filtered_results)} education-related documents:")
        for result in filtered_results:
            print(f"  - {result.payload['title']} (Score: {result.score:.4f})")
        print()
        
        # 9. Clean up - delete the collection
        print("Cleaning up...")
        client.delete_collection(collection_name)
        print(f"✓ Collection '{collection_name}' deleted successfully")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        # Try to clean up in case of error
        try:
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)
                print(f"✓ Cleaned up collection '{collection_name}'")
        except:
            pass
    
    print("Basic usage example completed!")


if __name__ == "__main__":
    main()