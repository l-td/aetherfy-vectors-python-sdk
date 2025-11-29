"""
Test script for Schema Visualizer feature.

This script demonstrates the complete schema workflow:
1. Create a collection and add data
2. Analyze the schema
3. Define a schema with enforcement
4. Test validation (both valid and invalid inserts)

Prerequisites:
- Backend server running at http://localhost:3000 (or set AETHERFY_ENDPOINT)
- Valid API key configured (SDK checks AETHERFY_API_KEY automatically)

This is essentially an E2E test that exercises the full stack.
"""

import os
import sys
from aetherfy_vectors import (
    AetherfyVectorsClient,
    Schema,
    FieldDefinition,
    SchemaValidationError,
    Point,
)


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    """Run complete schema feature test."""

    print_section("Schema Visualizer Feature Test")

    # Initialize client - SDK handles API key from environment automatically
    # For local testing, point to local backend
    endpoint = os.getenv('AETHERFY_ENDPOINT', 'http://localhost:3000')

    try:
        client = AetherfyVectorsClient(endpoint=endpoint)
        print(f"✓ Client initialized (endpoint: {endpoint})")
    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        print(f"\nMake sure AETHERFY_API_KEY is set in your environment")
        sys.exit(1)

    collection_name = "test_schema_collection"

    # Step 1: Create collection
    print_section("Step 1: Create Collection")
    try:
        # Delete if exists
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except:
            pass

        client.create_collection(
            collection_name,
            vectors_config={'size': 3, 'distance': 'Cosine'}
        )
        print(f"✓ Created collection: {collection_name}")
    except Exception as e:
        print(f"✗ Failed to create collection: {e}")
        return

    # Step 2: Insert sample data with varied schema
    print_section("Step 2: Insert Sample Data")
    try:
        sample_data = [
            Point(
                id="product_1",
                vector=[0.1, 0.2, 0.3],
                payload={"price": 100, "name": "Product A", "category": "electronics"}
            ),
            Point(
                id="product_2",
                vector=[0.4, 0.5, 0.6],
                payload={"price": 200, "name": "Product B", "category": "electronics"}
            ),
            Point(
                id="product_3",
                vector=[0.7, 0.8, 0.9],
                payload={"price": "300", "name": "Product C"}  # Intentional type mismatch for price
            ),
        ]

        client.upsert(collection_name, sample_data)
        print(f"✓ Inserted {len(sample_data)} sample vectors")
    except Exception as e:
        print(f"✗ Failed to insert data: {e}")
        return

    # Step 3: Analyze schema
    print_section("Step 3: Analyze Existing Data")
    try:
        analysis = client.analyze_schema(collection_name, sample_size=100)

        print(f"Collection: {analysis.collection}")
        print(f"Sample size: {analysis.sample_size}")
        print(f"Total points: {analysis.total_points}")
        print(f"\nField Analysis:")

        for field_name, field_info in analysis.fields.items():
            print(f"\n  {field_name}:")
            print(f"    Presence: {field_info['presence']*100:.1f}%")
            print(f"    Types: {field_info['types']}")
            if field_info.get('warnings'):
                print(f"    ⚠️  Warnings: {field_info['warnings']}")

        print(f"\n✓ Analysis complete")
        print(f"\nSuggested Schema:")
        for field_name, field_def in analysis.suggested_schema.fields.items():
            print(f"  {field_name}: {field_def.type} ({'required' if field_def.required else 'optional'})")

    except Exception as e:
        print(f"✗ Failed to analyze schema: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Define strict schema
    print_section("Step 4: Define Schema with Strict Enforcement")
    try:
        schema = Schema(fields={
            'price': FieldDefinition(type='integer', required=True),
            'name': FieldDefinition(type='string', required=True),
            'category': FieldDefinition(type='string', required=False),
        })

        etag = client.set_schema(collection_name, schema, enforcement='strict')
        print(f"✓ Schema defined with ETag: {etag}")
        print(f"  Enforcement mode: strict")
        print(f"\nSchema fields:")
        for field_name, field_def in schema.fields.items():
            print(f"  {field_name}: {field_def.type} ({'required' if field_def.required else 'optional'})")

    except Exception as e:
        print(f"✗ Failed to set schema: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Test valid insert
    print_section("Step 5: Test Valid Insert")
    try:
        valid_point = Point(
            id="product_4",
            vector=[0.1, 0.1, 0.1],
            payload={"price": 400, "name": "Product D", "category": "books"}
        )

        client.upsert(collection_name, [valid_point])
        print(f"✓ Valid insert succeeded")

    except SchemaValidationError as e:
        print(f"✗ Validation failed (unexpected): {e}")
    except Exception as e:
        print(f"✗ Insert failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 6: Test invalid insert (type mismatch)
    print_section("Step 6: Test Invalid Insert (Type Mismatch)")
    try:
        invalid_point = Point(
            id="product_5",
            vector=[0.2, 0.2, 0.2],
            payload={"price": "five hundred", "name": "Product E"}  # price should be integer
        )

        client.upsert(collection_name, [invalid_point])
        print(f"✗ Invalid insert succeeded (should have failed!)")

    except SchemaValidationError as e:
        print(f"✓ Validation correctly rejected invalid data:")
        for error in e.errors:
            print(f"  Vector {error['index']}:")
            for err in error['errors']:
                print(f"    - {err['message']}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    # Step 7: Test invalid insert (missing required field)
    print_section("Step 7: Test Invalid Insert (Missing Required Field)")
    try:
        invalid_point = Point(
            id="product_6",
            vector=[0.3, 0.3, 0.3],
            payload={"price": 600}  # missing required 'name' field
        )

        client.upsert(collection_name, [invalid_point])
        print(f"✗ Invalid insert succeeded (should have failed!)")

    except SchemaValidationError as e:
        print(f"✓ Validation correctly rejected invalid data:")
        for error in e.errors:
            print(f"  Vector {error['index']}:")
            for err in error['errors']:
                print(f"    - {err['message']}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    # Step 8: Get schema
    print_section("Step 8: Retrieve Schema")
    try:
        retrieved_schema = client.get_schema(collection_name)
        if retrieved_schema:
            print(f"✓ Retrieved schema:")
            for field_name, field_def in retrieved_schema.fields.items():
                print(f"  {field_name}: {field_def.type} ({'required' if field_def.required else 'optional'})")
        else:
            print(f"✗ No schema found")
    except Exception as e:
        print(f"✗ Failed to get schema: {e}")
        import traceback
        traceback.print_exc()

    # Step 9: Update schema enforcement to 'warn'
    print_section("Step 9: Change Enforcement to Warn Mode")
    try:
        etag = client.set_schema(collection_name, schema, enforcement='warn')
        print(f"✓ Schema enforcement changed to 'warn' mode")
        print(f"  New ETag: {etag}")

        # Now invalid data should be allowed
        invalid_point = Point(
            id="product_7",
            vector=[0.4, 0.4, 0.4],
            payload={"price": "seven hundred", "name": "Product G"}
        )

        client.upsert(collection_name, [invalid_point])
        print(f"✓ In warn mode, invalid data was allowed (with warnings logged)")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 10: Delete schema
    print_section("Step 10: Delete Schema")
    try:
        client.delete_schema(collection_name)
        print(f"✓ Schema deleted")

        # Verify schema is gone
        retrieved_schema = client.get_schema(collection_name)
        if retrieved_schema is None:
            print(f"✓ Confirmed: No schema defined for collection")
        else:
            print(f"✗ Schema still exists (unexpected)")

    except Exception as e:
        print(f"✗ Failed to delete schema: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print_section("Test Complete")
    print(f"All schema features have been tested!")
    print(f"\nYou can now:")
    print(f"  - Analyze any collection's data structure")
    print(f"  - Define schemas with field types and requirements")
    print(f"  - Choose enforcement modes (off/warn/strict)")
    print(f"  - Automatic client-side validation prevents bad data")
    print(f"  - Server-side validation provides defense-in-depth")


if __name__ == '__main__':
    main()
